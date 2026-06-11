# P2.6 gate — GQA-divergent (w_o) parity fixture.
#
# Exercises the realize-time branch where n_heads*head_dim != d_model,
# i.e. the attention output projection w_o is allocated with shape
# [d_model, n_heads*head_dim] (transformer_block.rb:311/315) rather than
# the [d_model, d_model] square assumed by the mmap path. This is the
# "divergent" realize slice the realize-bulk workflow must NOT silently
# unify; this fixture records a reproducible loss baseline that proves
# the divergent graph genuinely allocates + runs forward+backward.
#
# Pure-Ruby CPU, Spinel-compiled. NO lib/ behavior change, NO mirror regen.
#
#   make examples/smoke_gate_gqa_divergent
#   STEPS=5 SEED=0 ./examples/smoke_gate_gqa_divergent   (run from repo root)
#
# Determinism: SEED=0 + weight_dtype=0 (F32, bit-identical) + B=1 +
# fixed first line of data/ts_seqs.txt => bit-reproducible loss curve.

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/llm/engine/llama_seq_engine"
require_relative "../lib/toy/llm/adamw"
require_relative "../lib/toy/train/toy_drift_grad"

STEPS    = (ENV["STEPS"] || "5").to_i
SEED     = (ENV["SEED"]  || "0").to_i

VOCAB    = 627
D_MODEL  = 64
N_HEADS  = 4
N_KV     = 4
D_FF     = 128
N_LAYERS = 2
CONTEXT  = 32

# head_dim = 24 forces n_heads*head_dim = 96 != d_model (64). Default would
# be d_model/n_heads = 16, giving 64 == 64 (no divergence). Both head_dim
# (24) and n_heads*head_dim (96) stay EVEN for RoPE pairing.
HEAD_DIM = 24

cfg = Toy::SmolLM2Config.new(VOCAB, D_MODEL, N_HEADS, N_KV,
                              D_FF, N_LAYERS, CONTEXT, 10000.0, 1.0e-5)
cfg.donor_d_in = 0      # disable the projection lens — this gate is w_o divergence
cfg.head_dim   = HEAD_DIM

nhd = N_HEADS * cfg.head_dim
puts "config: vocab=" + cfg.vocab.to_s +
     " d_model=" + cfg.d_model.to_s +
     " n_heads=" + cfg.n_heads.to_s +
     " head_dim=" + cfg.head_dim.to_s +
     " n_heads*head_dim=" + nhd.to_s +
     " L=" + cfg.n_layers.to_s

# ASSERT 1 (static, fail-loud): the divergence must actually be present.
if N_HEADS * cfg.head_dim == D_MODEL
  puts "ABORT: no divergence — n_heads*head_dim == d_model (" + nhd.to_s + " == " + D_MODEL.to_s + ")"
  raise "gqa_divergent gate did not construct a divergent config"
end
puts "ASSERT 1 OK: " + nhd.to_s + " != " + D_MODEL.to_s

# Realize: B=1, weight_dtype=0 (F32), untied=true, qkv_bias=false,
# init_scale=1.0. rope_scaling defaults to :none so t_seq_rope_freq_factors
# is NULL and RoPE operates self-consistently on d_head=24.
fcache = Toy::LLM::Engine::LlamaSeqEngine.new
fcache.realize_for_random_init(cfg, CONTEXT, 1, 0, true, false, SEED, 1.0)
puts "realize OK"

# ASSERT 2 (runtime shape proof): find blk.0.attn_output.weight among the
# PARAM tensors and prove its allocated shape carries the divergent
# n_heads*head_dim (96) extent alongside d_model (64), NOT a [d_model,
# d_model] square. ggml reports ne0 = the fastest-varying extent: for w_o
# realized via tnn_input_2d_f32_persistent(sess, d_model, n_heads*head_dim)
# the persistent tensor lands as ne0 = n_heads*head_dim (96), ne1 = d_model
# (64). The load-bearing proof is that one extent is 96 (divergent) and the
# square [64,64] shape never appears.
plist = ToyDriftGrad.params(fcache.sess)
puts "params (" + plist.length.to_s + "):"
i = 0
wo_tensor = TinyNN.tnn_null_ptr
found_wo  = false
while i < plist.length
  name = TinyNN.tnn_tensor_name(plist[i])
  if name == "blk.0.attn_output.weight"
    wo_tensor = plist[i]
    found_wo  = true
  end
  if i < 6
    puts "  " + name
  end
  i = i + 1
end

if !found_wo
  puts "ABORT: PARAM 'blk.0.attn_output.weight' not found (never-mask rule)"
  raise "gqa_divergent gate could not locate w_o PARAM"
end

wo_ne0 = TinyNN.tnn_tensor_ne0(wo_tensor)
wo_ne1 = TinyNN.tnn_tensor_ne1(wo_tensor)
puts "w_o shape: ne0=" + wo_ne0.to_s + " ne1=" + wo_ne1.to_s

# One extent must be d_model (64), the other the divergent n_heads*head_dim
# (96). A unified [d_model, d_model] square would report ne0==ne1==64.
shape_ok = ((wo_ne0 == nhd) && (wo_ne1 == D_MODEL)) ||
           ((wo_ne0 == D_MODEL) && (wo_ne1 == nhd))
if !shape_ok
  puts "ABORT: w_o shape [" + wo_ne0.to_s + "," + wo_ne1.to_s +
       "] is not the divergent [d_model=" + D_MODEL.to_s +
       ", n_heads*head_dim=" + nhd.to_s + "]"
  raise "gqa_divergent gate: w_o shape mismatch"
end
if wo_ne0 == wo_ne1
  puts "ABORT: w_o is SQUARE (" + wo_ne0.to_s + "x" + wo_ne1.to_s + ") — not divergent"
  raise "gqa_divergent gate: w_o not divergent"
end
if (wo_ne0 != nhd) && (wo_ne1 != nhd)
  puts "ABORT: w_o carries no n_heads*head_dim (96) extent — square unification slipped through"
  raise "gqa_divergent gate: w_o missing divergent extent"
end
puts "ASSERT 2 OK: w_o = [" + wo_ne0.to_s + ", " + wo_ne1.to_s +
     "] divergent (carries n_heads*head_dim=" + nhd.to_s +
     ", not [d_model,d_model] square)"

# Train a few steps on a fixed sequence (reuse projection-lens scaffold).
raw        = File.read("data/ts_seqs.txt")
first_line = raw.split("\n")[0]
parts      = first_line.split(" ")
seq_ids    = [0]; seq_ids.pop
k = 0
while k < parts.length && k < CONTEXT
  seq_ids.push(parts[k].to_i)
  k = k + 1
end
while seq_ids.length < CONTEXT; seq_ids.push(0); end

result   = fcache.build_training_step
t_loss   = result[0]
t_labels = result[1]
t_hp     = result[2]

positions = [0]; positions.pop
p = 0; while p < CONTEXT; positions.push(p); p = p + 1; end

# Build labels: shift-by-one one-hot.
m_labels = Mat.new(CONTEXT, VOCAB)
j = 0; while j < CONTEXT * VOCAB; m_labels.flat[j] = 0.0; j = j + 1; end
k = 0
while k < CONTEXT
  target = (k + 1 < CONTEXT) ? seq_ids[k + 1] : seq_ids[k]
  m_labels.flat[k * VOCAB + target] = 1.0
  k = k + 1
end

# NAMED AdamW (byte-identical to the old hand-filled m_hp): all defaults
# (lr=0.001, β1=0.9, β2=0.95, eps=1e-8, wd=0, bias_correct=false → slots
# 5/6 = constant betas).
m_hp = Toy::AdamW.for_from_scratch.hp(0)

losses = [0.0]; losses.pop
step = 0
while step < STEPS
  if step == 0
    TinyNN.tnn_graph_reset(fcache.sess)
  else
    TinyNN.tnn_graph_reset_grads_only(fcache.sess)
  end
  TinyNN.upload_int_array(fcache.sess, fcache.t_seq_token_ids, seq_ids)
  TinyNN.upload_int_array(fcache.sess, fcache.t_seq_positions, positions)
  TinyNN.upload_row_major(fcache.sess, t_labels, m_labels)
  TinyNN.upload_row_major(fcache.sess, t_hp,     m_hp)
  TinyNN.tnn_compute_backward(fcache.sess)
  loss_mat = TinyNN.download_row_major(fcache.sess, t_loss, 1, 1)
  losses.push(loss_mat.flat[0])
  puts "step " + (step + 1).to_s + ": loss=" + loss_mat.flat[0].to_s
  step = step + 1
end

# ASSERT 3: forward+backward both executed at the divergent shape — the
# first step's loss must be finite (NaN/Inf would mean the divergent graph
# mis-shaped). We do NOT gate on a loss-decrease ratio (not reliable at
# random-init divergent shape over a few steps).
l0 = losses[0]
finite = (l0 == l0) && (l0 < 1.0e30) && (l0 > -1.0e30)
if !finite
  puts "ABORT: step-1 loss is not finite (" + l0.to_s + ") — divergent graph mis-shaped"
  raise "gqa_divergent gate: non-finite loss"
end
puts "ASSERT 3 OK: step-1 loss finite = " + l0.to_s

puts "BASELINE losses:"
b = 0
while b < losses.length
  puts "  loss[" + b.to_s + "]=" + losses[b].to_s
  b = b + 1
end
puts "VERDICT: gqa_divergent gate PASS (w_o = [64,96], forward+backward ran)"
