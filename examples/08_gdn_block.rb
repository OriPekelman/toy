# examples/08_gdn_block.rb — train a Gated-DeltaNet block from scratch (v0.9.0).
#
# WHAT YOU'LL SEE: a tiny from-scratch model whose mixer is NOT attention but
# a TRAINABLE Gated-DeltaNet block (the Dragon arc's building block) —
# embed → GDNBlock → RMSNorm → tied-unembed → cross-entropy. One
# "step N: loss=…" line per step; the loss falls visibly as it overfits one
# fixed batch. This proves the GDN block is a correct, end-to-end trainable
# residual unit on the standard backward + AdamW machinery — with NO
# hand-written kernel backward (the recurrence is an unrolled autograd graph,
# "Path B"; see docs/roadmap/dragon-gdn-arch-2026-06-20.md).
#
# WHY THIS ONE IS DIFFERENT: 01–07 ride the high-level recipe API
# (RecipeOptions / FromScratch / TrainingBatch). The GDN block has no L5
# recipe yet, so this example drops one level and builds the train graph by
# hand on the FFI — which is also a good look at what a recipe does for you.
# The full interleaved attention+GDN hybrid lives in its own runner:
#   make gate-gdn-hybrid     # libexec/toy-train-hybrid (lib/toy/run/train_hybrid.rb)
#
# HOW LONG: <1 s (default 14 steps, CPU). Build: one make.
#
#   make example_08
#   ./examples/example_08_gdn_block
#
# WHAT TO TWEAK (env, no recompile):
#   STEPS=40          train longer (the curve keeps falling)
#   LR=0.05           bigger AdamW steps — converge faster, or wobble
#   SEED=1            a different random init
#
# THE PIECES (the math, named — docs/architecture.md L1/L2):
#   Toy::LLM::Blocks::GDNBlock        — rmsnorm → q/k/v/z/b/a proj → L2 →
#                                       gates → per-head recur_unrolled →
#                                       gated-out → out-proj → residual
#   Toy::LLM::Primitives::GDN         — the gated delta rule, unrolled
#   Toy::LLM::Primitives::RMSNorm     — the final pre-unembed norm
#
# Unlike 01–07 this is a teaching graph, not a recipe — so it requires the
# FFI + the two primitives + the block directly (no one-require surface).

require_relative "../lib/toy"
require_relative "../lib/toy/ffi/tinynn"
require_relative "../lib/toy/llm/primitives/rms_norm"
require_relative "../lib/toy/llm/primitives/gdn"
require_relative "../lib/toy/llm/blocks/gdn_block"

STEPS = (ENV["STEPS"] || "14").to_i
SEED  = (ENV["SEED"]  || "0").to_i
LR    = (ENV["LR"]    || "0.02").to_f

VOCAB = 16
DM    = 8
H     = 2
S_V   = 4    # H*S_V == DM (inner width == residual width)
T     = 4
EPS   = 1.0e-5

# small pseudo-random fill (deterministic; SEED shifts the stream)
def fillv(n, seed)
  a = [0.0]; a.pop
  i = 0
  while i < n
    a.push(((((i + seed) * 1103515245 + 12345) % 1000) - 500).to_f * 0.001)
    i = i + 1
  end
  a
end

def zeros(n)
  a = [0.0]; a.pop
  i = 0
  while i < n
    a.push(0.0)
    i = i + 1
  end
  a
end

sess = TinyNN.tnn_session_new(0)
TinyNN.tnn_session_set_graph_capacity(sess, 65536)

# --- params: embed + final-norm + the GDN block's weights (parallel arrays) ---
# P = leaves, PM/PV = the AdamW first/second moments, one slot each.
P  = [TinyNN.tnn_null_ptr]; P.pop
PM = [TinyNN.tnn_null_ptr]; PM.pop
PV = [TinyNN.tnn_null_ptr]; PV.pop

t_embed = TinyNN.tnn_input_2d_f32_persistent(sess, VOCAB, DM)   # ne0=DM, ne1=VOCAB
P.push(t_embed)
PM.push(TinyNN.tnn_input_2d_f32_persistent(sess, VOCAB, DM))
PV.push(TinyNN.tnn_input_2d_f32_persistent(sess, VOCAB, DM))

t_fnorm = TinyNN.tnn_input_1d_f32_persistent(sess, DM)
P.push(t_fnorm)
PM.push(TinyNN.tnn_input_1d_f32_persistent(sess, DM))
PV.push(TinyNN.tnn_input_1d_f32_persistent(sess, DM))

blk = Toy::LLM::Blocks::GDNBlock.new
blk.alloc_trainable_f32_weights!(sess, DM, S_V, H)
bi = 0
while bi < blk.ft_weights.length
  P.push(blk.ft_weights[bi]); PM.push(blk.ft_m[bi]); PV.push(blk.ft_v[bi])
  bi = bi + 1
end

# set_param BEFORE finalize (load-bearing order).
gi = 0
while gi < P.length
  TinyNN.tnn_set_param(P[gi])
  gi = gi + 1
end
TinyNN.tnn_finalize_weights(sess)
blk.zero_state!(sess)

# Initialise weights (small pseudo-random) + zero the Adam moments.
gi = 0
while gi < P.length
  n = TinyNN.tnn_tensor_nelements(P[gi])
  TinyNN.tnn_upload_from_float_array(sess, P[gi], fillv(n, gi * 7 + 1 + SEED), n)
  TinyNN.tnn_zero_tensor(sess, PM[gi])
  TinyNN.tnn_zero_tensor(sess, PV[gi])
  gi = gi + 1
end

# --- forward graph: embed → GDN mixer → final-norm → tied unembed → CE ---
t_tok = TinyNN.tnn_input_1d_i32(sess, T)
x = TinyNN.tnn_get_rows(sess, t_embed, t_tok)             # [DM, T]
x = blk.build_forward(sess, x, T, EPS)                    # [DM, T] (residual)
xf = Toy::LLM::Primitives::RMSNorm.build(sess, x, t_fnorm, EPS)
lgt = TinyNN.tnn_matmul(sess, t_embed, xf)                # [VOCAB, T] tied unembed

t_labels = TinyNN.tnn_input_2d_f32(sess, T, VOCAB)        # ne0=VOCAB, ne1=T
t_hp     = TinyNN.tnn_input_1d_f32(sess, 7)               # AdamW hyper-params
t_loss   = TinyNN.tnn_cross_entropy_loss(sess, lgt, t_labels)
TinyNN.tnn_set_output(t_loss)
TinyNN.tnn_set_loss(t_loss)

TinyNN.tnn_build_forward_only(sess, t_loss)
TinyNN.tnn_build_backward(sess)

# Wire one AdamW opt_step per parameter into the backward graph.
gj = 0
while gj < P.length
  tg = TinyNN.tnn_tensor_grad(sess, P[gj])
  to = TinyNN.tnn_opt_step_adamw(sess, P[gj], tg, PM[gj], PV[gj], t_hp)
  TinyNN.tnn_extend_backward_graph(sess, to)
  gj = gj + 1
end
TinyNN.tnn_pin_all_graph_b_nodes(sess)
TinyNN.tnn_realize_backward(sess)

# --- one fixed batch (overfit so the drop is unmistakable) ---
ids = [1, 2, 3, 4]
labels = zeros(VOCAB * T)            # one-hot next-token-ish targets
tt = 0
while tt < T
  tgt = (ids[tt] + 1) % VOCAB
  labels[tgt + VOCAB * tt] = 1.0
  tt = tt + 1
end
# hp = [lr, beta1, beta2, eps, weight_decay, beta1, beta2]
hp = [LR, 0.9, 0.95, 1.0e-8, 0.0, 0.9, 0.95]

first_loss = 0.0
last_loss  = 0.0
s = 0
while s < STEPS
  if s == 0
    TinyNN.tnn_graph_reset(sess)
  else
    TinyNN.tnn_graph_reset_grads_only(sess)
  end
  TinyNN.upload_int_array(sess, t_tok, ids)
  TinyNN.tnn_upload_from_float_array(sess, t_labels, labels, VOCAB * T)
  TinyNN.tnn_upload_from_float_array(sess, t_hp, hp, 7)
  TinyNN.tnn_compute_backward(sess)
  TinyNN.tnn_download(sess, t_loss)
  lv = TinyNN.tnn_scratch_get(sess, 0)
  if s == 0
    first_loss = lv
  end
  last_loss = lv
  puts "step " + s.to_s + ": loss=" + lv.to_s
  s = s + 1
end

puts "GDN block trained: CE loss " + first_loss.to_s + " -> " + last_loss.to_s +
     " over " + STEPS.to_s + " steps (mixer = trainable Gated-DeltaNet, Path B)."
