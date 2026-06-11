# F1.2 step 5 — multi-layer SmolLM2 LoRA training with real AdamW.
#
# Builds on demos/smollm2_lora_train_ce_cuda (which used SGD) by
# switching opt_step_sgd → opt_step_adamw. The Adam optimizer needs:
#   - one (m, v) momentum pair per parameter, same shape as the param;
#   - a 7-element hyperparams tensor: [alpha, beta1, beta2, eps,
#     weight_decay, beta1h, beta2h], where beta1h = 1/(1 - beta1^t)
#     and beta2h = 1/(1 - beta2^t) are the bias-correction factors
#     (ggml's convention — they're MULTIPLIERS, not denominators).
#
# Critical: tnn_graph_reset zeros opt_step_adamw's m and v every call,
# which kills momentum across steps. F1.2 step 5 ships
# tnn_graph_reset_grads_only — same as graph_reset minus the
# m,v-zeroing arm. So between iters: zero grads, keep m / v.
#
# This is the foundation for real SFT — AdamW with a sensible LR
# (1e-4 here) converges where SGD-at-huge-LR struggles, and the
# m, v state correctly accumulates across steps.
#
# Run on CUDA (CPU is broken until ggml#1501 is fixed; see task #70):
#   GGUF=data/smollm2-135m-native.gguf ./demos/smollm2_lora_train_adamw_cuda

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../lib/toy/llm/engine/llama_kv_engine_cuda"

GGUF      = ENV["GGUF"]      || "data/smollm2-135m-native.gguf"
MAX_T     = (ENV["MAX_T"]    || "32").to_i
RANK      = (ENV["RANK"]     || "16").to_i
SEED      = (ENV["SEED"]     || "42").to_i
STEPS     = (ENV["STEPS"]    || "20").to_i
LR        = (ENV["LR"]       || "0.001").to_f
BETA1     = 0.9
BETA2     = 0.999
EPS       = 1.0e-8
WD        = 0.0
TARGET_ID = (ENV["TARGET_ID"]|| "99").to_i

PROMPT    = [12092, 4845, 253, 1429]
TRAIN_POS = PROMPT.length

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "cfg: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " n_q=" + cfg.n_heads.to_s
puts "AdamW: LR=" + LR.to_s + " STEPS=" + STEPS.to_s +
     " TARGET=" + TARGET_ID.to_s + " r=" + RANK.to_s

gguf = TinyNNCuda.tnn_gguf_load(GGUF)
kv = SmolLM2KVFFICacheCuda.new
kv.enable_lora_q!(RANK)
kv.realize_for_mmap(gguf, cfg, MAX_T, flags.untied, flags.qkv_bias)
kv.upload_lora_q_init!(SEED, 0.01)

# Allocate persistent (m, v) pair per LoRA tensor. AdamW also wants
# a single 7-element hyperparams tensor that gets re-uploaded with
# step-dependent beta1h/beta2h each iter.
puts "  allocating Adam state for " + (cfg.n_layers * cfg.n_heads * 2).to_s + " params"
sess = kv.sess
# Adam state lives in the persistent ctx_w (same buffer as the weights),
# so it has to be allocated BEFORE finalize. realize_for_mmap already
# called finalize_weights though — we need a SECOND persistent ctx for
# Adam state. For this smoke we'll instead allocate the m/v as
# compute-side tensors (non-persistent) and accept that each
# realize_backward will re-alloc them (cheap; they're per-LoRA-pair).
# Wait — tnn_graph_reset_grads_only relies on m/v surviving across
# graph_reset calls. We need PERSISTENT m/v.
#
# Workaround for this session: keep m/v compute-side, but only call
# tnn_realize_backward once (which we already do). Once a backward
# graph is realized, the compute buffers are stable across the
# loop's compute_backward calls — even though they're "compute-side"
# in ctx, they're sched-allocated once and not freed between steps.

# Allocate AFTER realize_for_mmap (in the compute ctx). They'll be
# fixed-shape per layer, head.

m_tensors_a = []
v_tensors_a = []
m_tensors_b = []
v_tensors_b = []
li = 0
while li < cfg.n_layers
  blk = kv.kv_blocks_ffi[li]
  hq = 0
  while hq < cfg.n_heads
    # Same shape as the LoRA persistent tensors declared in
    # lib/toy_smollm2_ffi_kv_cuda.rb#realize_for_mmap. The call form
    # is tnn_input_2d_f32(sess, rows, cols) → ne=[cols, rows].
    # LoRA-A persistent is tnn_input_2d_f32_persistent(sess, r, d_model);
    # LoRA-B persistent is tnn_input_2d_f32_persistent(sess, d_head, r).
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
puts "  m/v tensors allocated"

# Prefill prompt
i = 0
while i < PROMPT.length
  SmolLM2KVCuda.decode_step(kv, PROMPT[i], i)
  i = i + 1
end
last_prompt_id = PROMPT[PROMPT.length - 1]

# Build training graph
TinyNNCuda.tnn_reset_for_rebuild(sess)
step_h = kv.build_decode_step(TRAIN_POS)
t_labels = TinyNNCuda.tnn_input_2d_f32(sess, 1, cfg.vocab)
t_hp_v   = TinyNNCuda.tnn_input_1d_f32(sess, 7)   # AdamW: 7 hp slots
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
puts "  graph realized (" + idx.to_s + " AdamW opt nodes)."

m_labels = Mat.new(1, cfg.vocab)
i = 0
while i < cfg.vocab; m_labels.flat[i] = 0.0; i = i + 1; end
m_labels.flat[TARGET_ID] = 1.0

# Initial graph_reset clears EVERYTHING including m, v. After this we
# use graph_reset_grads_only to keep m, v across iters.
TinyNNCuda.tnn_graph_reset(sess)

losses = []
s = 0
while s < STEPS
  if s > 0
    TinyNNCuda.tnn_graph_reset_grads_only(sess)
  end

  # Per-step hyperparams. ggml's AdamW expects bias-correction factors
  # as multipliers: beta1h = 1/(1-beta1^t), beta2h = 1/(1-beta2^t).
  t = s + 1
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

  TinyNNCuda.upload_int_array(sess, step_h.t_token_id, [last_prompt_id])
  TinyNNCuda.upload_int_array(sess, step_h.t_pos,      [TRAIN_POS])
  TinyNNCuda.upload_row_major(sess, t_labels, m_labels)
  TinyNNCuda.upload_row_major(sess, t_hp_v,   m_hp_v)

  rc = TinyNNCuda.tnn_compute_backward(sess)
  if rc != 0; puts "compute_backward rc=" + rc.to_s; exit 1; end

  TinyNNCuda.tnn_download(sess, t_loss)
  loss_v = TinyNNCuda.tnn_scratch_get(sess, 0)
  losses.push(loss_v)
  puts "  step " + (s + 1).to_s + ": CE=" + loss_v.to_s
  s = s + 1
end

any_nan = false
s = 0
while s < losses.length
  if losses[s].to_s == "NaN"; any_nan = true; end
  s = s + 1
end
initial = losses[0]
final   = losses[losses.length - 1]
puts ""
puts "initial CE = " + initial.to_s
puts "final   CE = " + final.to_s
if any_nan
  puts "VERDICT: FAIL (NaN in loss trajectory)"; exit 1
end
threshold = 0.5 * initial
if final >= threshold
  puts "VERDICT: FAIL (CE " + initial.to_s + " → " + final.to_s +
       "; needed < " + threshold.to_s + ")"
  exit 1
end
puts "VERDICT: PASS (AdamW CE " + initial.to_s + " → " + final.to_s + ")"
