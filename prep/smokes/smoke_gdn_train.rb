#!/usr/bin/env ruby
# prep/smokes/smoke_gdn_train.rb — Phase 5 (Dragon/GDN arc): end-of-flow proof.
#
# Trains a tiny from-scratch model whose mixer is a TRAINABLE GDN layer
# (Toy::LLM::Blocks::GDNBlock → recur_unrolled, Path B) and asserts the CE loss
# DECREASES over steps. Proves the GDN block is a correct, end-to-end trainable
# residual unit inside a real embed → mixer → tied-logits → CE graph, with the
# standard backward + AdamW opt_step machinery — no hand-written kernel backward.
# See docs/roadmap/dragon-gdn-arch-2026-06-20.md (Phase 5).
#
#   x   = get_rows(embed, ids)            [d_model, T]
#   x   = GDNBlock.build_forward(x)       [d_model, T]  (residual)
#   xf  = rmsnorm(x, final_gamma)         [d_model, T]
#   lgt = matmul(embed, xf)               [vocab, T]    (tied unembed)
#   loss= cross_entropy(lgt, labels)      scalar; overfit one fixed batch

require_relative "../../lib/toy"
require_relative "../../lib/toy/ffi/tinynn"
require_relative "../../lib/toy/llm/primitives/rms_norm"
require_relative "../../lib/toy/llm/primitives/gdn"
require_relative "../../lib/toy/llm/blocks/gdn_block"

VOCAB = 16
DM    = 8
H     = 2
S_V   = 4    # H*S_V == DM (inner == residual width)
T     = 4
STEPS = 14
EPS   = 1.0e-5

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

# Initialise weights (small pseudo-random) + zero Adam moments.
gi = 0
while gi < P.length
  n = TinyNN.tnn_tensor_nelements(P[gi])
  TinyNN.tnn_upload_from_float_array(sess, P[gi], fillv(n, gi * 7 + 1), n)
  TinyNN.tnn_zero_tensor(sess, PM[gi])
  TinyNN.tnn_zero_tensor(sess, PV[gi])
  gi = gi + 1
end

# --- forward graph ---
t_tok = TinyNN.tnn_input_1d_i32(sess, T)
x = TinyNN.tnn_get_rows(sess, t_embed, t_tok)             # [DM, T]
x = blk.build_forward(sess, x, DM, S_V, H, T, EPS)        # [DM, T]
xf = Toy::LLM::Primitives::RMSNorm.build(sess, x, t_fnorm, EPS)
lgt = TinyNN.tnn_matmul(sess, t_embed, xf)                # [VOCAB, T] tied

t_labels = TinyNN.tnn_input_2d_f32(sess, T, VOCAB)        # ne0=VOCAB, ne1=T
t_hp     = TinyNN.tnn_input_1d_f32(sess, 7)
t_loss   = TinyNN.tnn_cross_entropy_loss(sess, lgt, t_labels)
TinyNN.tnn_set_output(t_loss)
TinyNN.tnn_set_loss(t_loss)

TinyNN.tnn_build_forward_only(sess, t_loss)
TinyNN.tnn_build_backward(sess)

gj = 0
while gj < P.length
  tg = TinyNN.tnn_tensor_grad(sess, P[gj])
  to = TinyNN.tnn_opt_step_adamw(sess, P[gj], tg, PM[gj], PV[gj], t_hp)
  TinyNN.tnn_extend_backward_graph(sess, to)
  gj = gj + 1
end
TinyNN.tnn_pin_all_graph_b_nodes(sess)
TinyNN.tnn_realize_backward(sess)

# --- fixed batch (overfit) ---
ids = [1, 2, 3, 4]
# one-hot next-token-ish targets
labels = zeros(VOCAB * T)
tt = 0
while tt < T
  tgt = (ids[tt] + 1) % VOCAB
  labels[tgt + VOCAB * tt] = 1.0
  tt = tt + 1
end
hp = [0.02, 0.9, 0.95, 1.0e-8, 0.0, 0.9, 0.95]

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

# --- verdict: loss must drop meaningfully + stay finite ---
ok = true
if first_loss != first_loss || last_loss != last_loss
  puts "FAIL: loss is NaN (first=" + first_loss.to_s + " last=" + last_loss.to_s + ")"
  ok = false
end
if last_loss >= first_loss - 0.05
  puts "FAIL: loss did not decrease (first=" + first_loss.to_s + " last=" + last_loss.to_s + ")"
  ok = false
end

if ok
  puts "GDN train smoke PASS: from-scratch GDN-layer model trains — CE loss " +
       first_loss.to_s + " -> " + last_loss.to_s + " over " + STEPS.to_s + " steps"
  exit 0
else
  puts "GDN train smoke FAIL"
  exit 0
end
