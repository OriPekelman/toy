# F1.2 step 6a — multi-example SFT-shaped training on CUDA.
#
# Builds on step 5 (AdamW with m/v survival) by training over MULTIPLE
# target tokens instead of just one. Cycles through a small "dataset"
# of (prefix, target_id) pairs; each iter trains on the next pair.
# AdamW state accumulates across the whole loop, so the model has to
# learn ALL targets, not just memorize the last one.
#
# Why same prefix for all examples: re-prefilling the KV cache mid-
# training-loop would call decode_step → tnn_reset_for_rebuild →
# tnn_realize, which would tear down graph_b's sched-alloc. Real SFT
# would either (a) build N training graphs (one per prefix shape) or
# (b) wait for M3 (task #69) to make the decode graph reusable + pull
# prefix-feed into the same graph. Out of step-6a scope.
#
# Acceptance: average CE across the target set DECREASES over the
# training run. Per-target CE also tracked for observation — they
# should mostly trend down.

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv_cuda"

GGUF      = ENV["GGUF"]      || "data/smollm2-135m-native.gguf"
MAX_T     = (ENV["MAX_T"]    || "32").to_i
RANK      = (ENV["RANK"]     || "16").to_i
SEED      = (ENV["SEED"]     || "42").to_i
EPOCHS    = (ENV["EPOCHS"]   || "10").to_i
LR        = (ENV["LR"]       || "0.001").to_f
BETA1     = 0.9
BETA2     = 0.999
EPS       = 1.0e-8
WD        = 0.0

PROMPT    = [12092, 4845, 253, 1429]   # "Once upon a time"
TRAIN_POS = PROMPT.length
# Five rare target tokens. We want the model to learn to predict each
# one (in turn) at TRAIN_POS — clearly impossible to satisfy all at
# once for a single fixed input, but each individual loss should drop
# and the AVERAGE should drop monotonically over epochs.
TARGETS   = [99, 100, 1234, 5555, 31337]

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "cfg: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " n_q=" + cfg.n_heads.to_s
puts "SFT: " + TARGETS.length.to_s + " targets, " + EPOCHS.to_s +
     " epochs, LR=" + LR.to_s + " r=" + RANK.to_s

gguf = TinyNNCuda.tnn_gguf_load(GGUF)
kv = SmolLM2KVFFICacheCuda.new
kv.enable_lora_q!(RANK)
kv.realize_for_mmap(gguf, cfg, MAX_T, flags.untied, flags.qkv_bias)
kv.upload_lora_q_init!(SEED, 0.01)

# Prefill the shared prompt once. The KV cache holds positions
# 0..TRAIN_POS-1; training graph runs at TRAIN_POS.
i = 0
while i < PROMPT.length
  SmolLM2KVCuda.decode_step(kv, PROMPT[i], i)
  i = i + 1
end
last_prompt_id = PROMPT[PROMPT.length - 1]

# Adam state — see step 5 smoke for the shape derivation.
m_tensors_a = []
v_tensors_a = []
m_tensors_b = []
v_tensors_b = []
sess = kv.sess
li = 0
while li < cfg.n_layers
  hq = 0
  while hq < cfg.n_heads
    m_a = TinyNNCuda.tnn_input_2d_f32(sess, RANK, cfg.d_model)
    v_a = TinyNNCuda.tnn_input_2d_f32(sess, RANK, cfg.d_model)
    m_b = TinyNNCuda.tnn_input_2d_f32(sess, cfg.d_model / cfg.n_heads, RANK)
    v_b = TinyNNCuda.tnn_input_2d_f32(sess, cfg.d_model / cfg.n_heads, RANK)
    m_tensors_a.push(m_a); v_tensors_a.push(v_a)
    m_tensors_b.push(m_b); v_tensors_b.push(v_b)
    hq = hq + 1
  end
  li = li + 1
end

TinyNNCuda.tnn_reset_for_rebuild(sess)
step_h = kv.build_decode_step(TRAIN_POS)
t_labels = TinyNNCuda.tnn_input_2d_f32(sess, 1, cfg.vocab)
t_hp_v   = TinyNNCuda.tnn_input_1d_f32(sess, 7)
t_loss = TinyNNCuda.tnn_cross_entropy_loss(sess, step_h.kv_step_logits, t_labels)
TinyNNCuda.tnn_set_output(t_loss); TinyNNCuda.tnn_set_loss(t_loss)

TinyNNCuda.tnn_build_forward_only(sess, t_loss)
TinyNNCuda.tnn_build_backward(sess)

idx = 0
li = 0
while li < cfg.n_layers
  blk = kv.kv_blocks_ffi[li]
  hq = 0
  while hq < cfg.n_heads
    t_a = blk.t_w_lora_a_q[hq]
    t_b = blk.t_w_lora_b_q[hq]
    t_grad_a = TinyNNCuda.tnn_tensor_grad(sess, t_a)
    t_grad_b = TinyNNCuda.tnn_tensor_grad(sess, t_b)
    t_opt_a = TinyNNCuda.tnn_opt_step_adamw(sess, t_a, t_grad_a,
                                              m_tensors_a[idx], v_tensors_a[idx], t_hp_v)
    t_opt_b = TinyNNCuda.tnn_opt_step_adamw(sess, t_b, t_grad_b,
                                              m_tensors_b[idx], v_tensors_b[idx], t_hp_v)
    TinyNNCuda.tnn_extend_backward_graph(sess, t_opt_a)
    TinyNNCuda.tnn_extend_backward_graph(sess, t_opt_b)
    idx = idx + 1
    hq = hq + 1
  end
  li = li + 1
end
TinyNNCuda.tnn_realize_backward(sess)
puts "  graph realized (" + idx.to_s + " AdamW pairs)."

# Single labels Mat reused per step (Array<Mat> pre-allocation
# pattern triggers a Spinel #626-family widening cascade through
# GGUFLoad — sidestep by zeroing + setting the one-hot in place).
m_labels = Mat.new(1, cfg.vocab)
i = 0
while i < cfg.vocab; m_labels.flat[i] = 0.0; i = i + 1; end
prev_target = -1

# graph_reset ONCE before the loop; m/v are clean.
TinyNNCuda.tnn_graph_reset(sess)

# Track per-epoch averages.
epoch_avg_losses = []
step = 0   # global AdamW step counter (drives beta1h/beta2h)

e = 0
while e < EPOCHS
  total_loss = 0.0
  per_target = []
  ti = 0
  while ti < TARGETS.length
    if step > 0
      TinyNNCuda.tnn_graph_reset_grads_only(sess)
    end

    t = step + 1
    beta1h = 1.0 / (1.0 - (BETA1 ** t.to_f))
    beta2h = 1.0 / (1.0 - (BETA2 ** t.to_f))
    m_hp_v = Mat.new(1, 7)
    m_hp_v.flat[0] = LR
    m_hp_v.flat[1] = BETA1
    m_hp_v.flat[2] = BETA2
    m_hp_v.flat[3] = EPS
    m_hp_v.flat[4] = WD
    m_hp_v.flat[5] = beta1h
    m_hp_v.flat[6] = beta2h

    if prev_target >= 0; m_labels.flat[prev_target] = 0.0; end
    m_labels.flat[TARGETS[ti]] = 1.0
    prev_target = TARGETS[ti]

    TinyNNCuda.upload_int_array(sess, step_h.t_token_id, [last_prompt_id])
    TinyNNCuda.upload_int_array(sess, step_h.t_pos,      [TRAIN_POS])
    TinyNNCuda.upload_row_major(sess, t_labels, m_labels)
    TinyNNCuda.upload_row_major(sess, t_hp_v,   m_hp_v)

    rc = TinyNNCuda.tnn_compute_backward(sess)
    if rc != 0; puts "compute_backward rc=" + rc.to_s; exit 1; end

    TinyNNCuda.tnn_download(sess, t_loss)
    loss_v = TinyNNCuda.tnn_scratch_get(sess, 0)
    per_target.push(loss_v)
    total_loss = total_loss + loss_v
    step = step + 1
    ti = ti + 1
  end
  avg = total_loss / TARGETS.length.to_f
  epoch_avg_losses.push(avg)
  print "  epoch " + (e + 1).to_s + ": avg CE=" + avg.to_s + "  per-target=["
  pi = 0
  while pi < per_target.length
    print " " + per_target[pi].to_s
    pi = pi + 1
  end
  puts " ]"
  e = e + 1
end

initial = epoch_avg_losses[0]
final   = epoch_avg_losses[epoch_avg_losses.length - 1]
puts ""
puts "epoch-1 avg CE = " + initial.to_s
puts "epoch-N avg CE = " + final.to_s

# Acceptance: average CE per epoch decreases meaningfully. Since the
# 5 targets are mutually incompatible (only one token can be the
# argmax), perfect convergence isn't possible — but the average should
# drop substantially compared to one epoch's worth of training on
# a single target.
threshold = 0.7 * initial
if final >= threshold
  puts "VERDICT: FAIL (epoch-avg CE " + initial.to_s + " → " + final.to_s +
       "; needed < " + threshold.to_s + ")"
  exit 1
end
puts "VERDICT: PASS (epoch-avg CE " + initial.to_s + " → " + final.to_s + ")"
