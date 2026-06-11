# P2.6 gate — B>1 (micro-batch) parity fixture.
#
# Exercises the realize-time + forward branches gated SOLELY on @seq_b>1:
#   - finalize_weights_and_upload_constants! allocates the block-causal
#     attention mask tensor (llama_seq_forward_ffi.rb:1360-1363) and
#     uploads it (upload_block_causal_mask!, :1377-1378) — both NO-OP at B=1.
#   - GQA.attention (toy/llm/primitives/gqa.rb:48-58) selects the
#     tnn_soft_max_ext(attn_mask) path when batch>1, NOT the B=1
#     tnn_scale + diag_mask_inf + softmax path. With @seq_b==2 this
#     batched-mask attention runs in EVERY forward of the training loop.
#
# This is the "B>1" realize slice the realize-bulk workflow must NOT
# silently unify with the B=1 path; this fixture records a reproducible
# loss baseline that proves the B>1 graph genuinely allocates the mask
# and runs forward+backward through soft_max_ext.
#
# Pure-Ruby CPU, Spinel-compiled. NO lib/ behavior change, NO mirror regen.
#
#   make prep/smokes/smoke_gate_b_gt_1
#   STEPS=5 SEED=0 ./prep/smokes/smoke_gate_b_gt_1   (run from repo root)
#
# Determinism: SEED=0 + weight_dtype=0 (F32, no F16/BF16 reduction-order
# nondeterminism across the larger T*B score matrix) + fixed FIRST 2 lines
# of data/ts_seqs.txt + single unscaled opt_step per step => bit-
# reproducible loss curve.
#
# Layout adopted from examples/06_train_from_scratch.rb (the proven B>1
# path): flat [T*B] token + position vectors, per-(batch,position)
# shift-by-one one-hot labels. NO GRAD_ACCUM LR-scaling — this gate is
# ONLY about batching. head_dim defaults to d_model/n_heads (16) so
# n_heads*head_dim == d_model (no w_o divergence — that is a separate gate).

require_relative "../../lib/toy"
require_relative "../../lib/toy/models/toy_smollm2"
require_relative "../../lib/toy/llm/engine/llama_seq_engine"
require_relative "../../lib/toy/llm/adamw"
require_relative "../../lib/toy/train/toy_drift_grad"

STEPS    = (ENV["STEPS"] || "5").to_i
SEED     = (ENV["SEED"]  || "0").to_i

VOCAB    = 627
D_MODEL  = 64
N_HEADS  = 4
N_KV     = 4
D_FF     = 128
N_LAYERS = 2
CONTEXT  = 32
BATCH    = 2

cfg = Toy::SmolLM2Config.new(VOCAB, D_MODEL, N_HEADS, N_KV,
                              D_FF, N_LAYERS, CONTEXT, 10000.0, 1.0e-5)
cfg.donor_d_in = 0      # disable the projection lens — orthogonal to batching
# head_dim left at default (d_model/n_heads = 16) so n_heads*head_dim ==
# d_model (64). This keeps the gate ONLY about batching, not w_o divergence.

nhd = N_HEADS * cfg.head_dim
puts "config: vocab=" + cfg.vocab.to_s +
     " d_model=" + cfg.d_model.to_s +
     " n_heads=" + cfg.n_heads.to_s +
     " head_dim=" + cfg.head_dim.to_s +
     " n_heads*head_dim=" + nhd.to_s +
     " L=" + cfg.n_layers.to_s +
     " BATCH=" + BATCH.to_s

# Sanity (static): this gate is ONLY about batching — assert no w_o
# divergence sneaks in via a non-default head_dim.
if nhd != D_MODEL
  puts "ABORT: head_dim drifted — n_heads*head_dim (" + nhd.to_s +
       ") != d_model (" + D_MODEL.to_s + "); this gate must stay batching-only"
  raise "b_gt_1 gate: unexpected w_o divergence"
end

# Realize with t_batch=2 (3rd arg) so @seq_b=2. weight_dtype=0 (F32),
# untied=true, qkv_bias=false, init_scale=1.0. rope_scaling defaults to
# :none so t_seq_rope_freq_factors is NULL.
fcache = Toy::LLM::Engine::LlamaSeqEngine.new
fcache.realize_for_random_init(cfg, CONTEXT, BATCH, 0, true, false, SEED, 1.0)
puts "realize OK"

# ASSERT 1 (B>1 branch taken): @seq_b must be 2. This alone
# deterministically forces the mask-alloc (llama_seq_forward_ffi.rb:1362)
# and the upload_block_causal_mask! body (:1377), both gated SOLELY on
# @seq_b>1, AND selects the soft_max_ext path in GQA.attention (gqa.rb:50).
if fcache.seq_b != BATCH
  puts "ABORT: seq_b == " + fcache.seq_b.to_s + " (expected " + BATCH.to_s +
       ") — B>1 branch NOT taken"
  raise "b_gt_1 gate: seq_b mismatch"
end
puts "ASSERT 1 OK: seq_b == " + fcache.seq_b.to_s + " (>1 branch taken)"

# ASSERT 2 (mask tensor is real, NOT the B=1 null path): the block-causal
# mask is allocated ONLY when @seq_b>1 (else it stays TinyNN.tnn_null_ptr,
# the B=1 diag_mask path). Read its ne0 — a real [T*B, T*B] mask reports
# ne0 == T*B (64). The null ptr would carry no such extent. A clean
# non-null FFI-ptr equality is not exposed, so per the gate header we
# prove non-null POSITIVELY via the allocated extent (ne0 == T*B), which
# is only possible on the B>1 mask-alloc branch. (If this read ever
# returns 0 we fall back to ASSERT 1, which already gates the mask path.)
tb = CONTEXT * BATCH
mask_ne0 = TinyNN.tnn_tensor_ne0(fcache.t_seq_attn_mask)
puts "mask tensor ne0 = " + mask_ne0.to_s + " (expected T*B = " + tb.to_s + ")"
if mask_ne0 != tb
  puts "ABORT: attn_mask ne0 (" + mask_ne0.to_s + ") != T*B (" + tb.to_s +
       ") — B>1 mask-alloc branch did NOT run (null/diag_mask path)"
  raise "b_gt_1 gate: mask tensor not allocated on B>1 branch"
end
puts "ASSERT 2 OK: attn_mask = [" + mask_ne0.to_s + ", *] real tensor (soft_max_ext path)"

# Build the BATCH-laid-out flat I/O (adopted from 06_train_from_scratch.rb
# :189-256). Read the FIRST 2 LINES of data/ts_seqs.txt, each padded to
# CONTEXT=32; lay them side by side as a flat [T*B] = 64-long vector.
raw   = File.read("data/ts_seqs.txt")
lines = raw.split("\n")
seq_ids = [0]; seq_ids.pop
bi = 0
while bi < BATCH
  line  = lines[bi % lines.length]
  parts = line.split(" ")
  k = 0
  while k < CONTEXT
    if k < parts.length
      seq_ids.push(parts[k].to_i)
    else
      seq_ids.push(0)
    end
    k = k + 1
  end
  bi = bi + 1
end

# Positions cycle 0..CONTEXT-1 per batch element, flat length T*B (64).
positions = [0]; positions.pop
b_pos = 0
while b_pos < BATCH
  pi = 0
  while pi < CONTEXT
    positions.push(pi)
    pi = pi + 1
  end
  b_pos = b_pos + 1
end

result   = fcache.build_training_step
t_loss   = result[0]
t_labels = result[1]
t_hp     = result[2]

# Shift-by-one next-token targets per (batch, position) slot, one-hot rows.
# At the last position within a batch (ti == CONTEXT-1) target = self.
m_labels = Mat.new(CONTEXT * BATCH, VOCAB)
li = 0
while li < CONTEXT * BATCH * VOCAB; m_labels.flat[li] = 0.0; li = li + 1; end
b_lbl = 0
while b_lbl < BATCH
  ti = 0
  while ti < CONTEXT
    flat_q  = b_lbl * CONTEXT + ti
    next_ti = ti + 1 < CONTEXT ? ti + 1 : ti
    tgt     = seq_ids[b_lbl * CONTEXT + next_ti]
    m_labels.flat[flat_q * VOCAB + tgt] = 1.0
    ti = ti + 1
  end
  b_lbl = b_lbl + 1
end

# ASSERT 3 (flat layout actually shaped the I/O): the flat [T*B] vectors
# and the label matrix must all carry CONTEXT*2 (64) slots — proving the
# batched layout, not the B=1 single-sequence (CONTEXT-long) layout.
if seq_ids.length != tb
  puts "ABORT: seq_ids.length (" + seq_ids.length.to_s + ") != T*B (" + tb.to_s + ")"
  raise "b_gt_1 gate: seq_ids not batched"
end
if positions.length != tb
  puts "ABORT: positions.length (" + positions.length.to_s + ") != T*B (" + tb.to_s + ")"
  raise "b_gt_1 gate: positions not batched"
end
if m_labels.nrows != tb
  puts "ABORT: m_labels.nrows (" + m_labels.nrows.to_s + ") != T*B (" + tb.to_s + ")"
  raise "b_gt_1 gate: labels not batched"
end
puts "ASSERT 3 OK: seq_ids/positions length == " + tb.to_s +
     ", m_labels rows == " + m_labels.nrows.to_s + " (batched [T*B] layout)"

# AdamW hp vector (7 slots). Single opt_step per step, LR unscaled
# (NO GRAD_ACCUM scaling — this gate is batching-only). Mirrors the
# smoke_projection_lens / gqa_divergent fixed hp.
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

# Forward+backward both executed at the B>1 shape — the first step's loss
# must be finite (NaN/Inf would mean the batched soft_max_ext graph mis-
# shaped). We do NOT gate on a loss-decrease ratio (not reliable at
# random-init over a few steps).
l0 = losses[0]
finite = (l0 == l0) && (l0 < 1.0e30) && (l0 > -1.0e30)
if !finite
  puts "ABORT: step-1 loss is not finite (" + l0.to_s + ") — B>1 graph mis-shaped"
  raise "b_gt_1 gate: non-finite loss"
end

initial = losses[0]
final   = losses[losses.length - 1]
ratio   = final / initial
puts "BASELINE losses:"
b = 0
while b < losses.length
  puts "  loss[" + b.to_s + "]=" + losses[b].to_s
  b = b + 1
end
puts "initial=" + initial.to_s + " final=" + final.to_s + " ratio=" + ratio.to_s
puts "VERDICT: b_gt_1 gate PASS (seq_b=2, block-causal mask + soft_max_ext path, forward+backward ran)"
