# F1.2 step 4 — multi-layer SmolLM2 LoRA training with real CE loss.
#
# Builds on step 3 (single-layer synthetic loss) by:
#   - Wiring opt_step_sgd on EVERY layer's LoRA-Q params (30 layers ×
#     9 heads × 2 params = 540 trainable tensors).
#   - Using ggml's built-in ggml_cross_entropy_loss against a one-hot
#     target token instead of sum(logits²).
#   - Running enough SGD steps (default 10) to observe a clean
#     monotonic decrease in CE loss.
#
# Setup:
#   - Load SmolLM2-135M with LoRA r=16 enabled on Q.
#   - Initialise A = small Gaussian, B = 0 (the LoRA contributes
#     exactly zero at step 0; forward parity vs baseline is the F1.2
#     step 2 gate, already green).
#   - Prefill a fixed prompt.
#   - Pick a target token id and a one-hot label tensor of shape
#     (vocab, 1) over it.
#   - Build training graph: decode at pos=N, ggml_cross_entropy_loss
#     on the resulting logits, set_loss on it.
#   - build_forward_only → build_backward → 540 opt_step_sgd nodes →
#     extend_backward_graph for each → realize_backward.
#   - Loop SGD steps: graph_reset, re-upload inputs, compute_backward,
#     download loss.
#
# Acceptance:
#   - No NaN at any step.
#   - Final loss < initial loss (any monotonic-ish decrease is fine —
#     this is one-position SFT so the model overfits very quickly
#     and may saturate).
#
# Run: GGUF=data/smollm2-135m-native.gguf ./demos/smollm2_lora_train_ce

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv"

GGUF      = ENV["GGUF"]      || "data/smollm2-135m-native.gguf"
MAX_T     = (ENV["MAX_T"]    || "32").to_i
RANK      = (ENV["RANK"]     || "16").to_i
SEED      = (ENV["SEED"]     || "42").to_i
STEPS     = (ENV["STEPS"]    || "20").to_i
# LR=0.5 is aggressive for SGD on SmolLM2 — chosen so the smoke shows
# visible movement in a few steps. Real SFT would use AdamW at lr=1e-4
# (we don't have a graph_reset that preserves Adam m/v across steps yet
# — that's a Phase F1.2-step-5 concern; see ggml-opt's two-graph
# pattern in vendor/ggml/src/ggml-opt.cpp).
LR        = (ENV["LR"]       || "0.5").to_f
# TARGET_ID=99 is a rare token in the SmolLM2 vocab — initial CE ≈
# 7.5 (model assigns ~0.05% probability). Picking a target the model
# already wants (e.g. 198 == newline-ish, initial CE ≈ 3.15) makes
# the smoke essentially flat because there's no error to drive
# training. Override via env if you want to compare paths.
TARGET_ID = (ENV["TARGET_ID"]|| "99").to_i

PROMPT    = [12092, 4845, 253, 1429]   # "Once upon a time"
TRAIN_POS = PROMPT.length

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "cfg: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " n_q=" + cfg.n_heads.to_s
puts "training: LR=" + LR.to_s + " STEPS=" + STEPS.to_s +
     " TARGET_ID=" + TARGET_ID.to_s + " r=" + RANK.to_s

gguf = TinyNN.tnn_gguf_load(GGUF)
kv = SmolLM2KVFFICache.new
kv.enable_lora_q!(RANK)
kv.realize_for_mmap(gguf, cfg, MAX_T, flags.untied, flags.qkv_bias)
kv.upload_lora_q_init!(SEED, 0.01)

# === Prefill ===
puts ""
puts "=== Prefill " + PROMPT.length.to_s + " tokens ==="
i = 0
while i < PROMPT.length
  SmolLM2KV.decode_step(kv, PROMPT[i], i)
  i = i + 1
end
last_prompt_id = PROMPT[PROMPT.length - 1]

# === Build training graph at TRAIN_POS ===
puts ""
puts "=== Build training graph (all " + cfg.n_layers.to_s + " layers, " +
     (cfg.n_layers * cfg.n_heads * 2).to_s + " LoRA params) ==="
sess = kv.sess
TinyNN.tnn_reset_for_rebuild(sess)
step_h = kv.build_decode_step(TRAIN_POS)

# Compute-side inputs: labels (one-hot over vocab) + hp ([lr, wd]).
t_labels = TinyNN.tnn_input_2d_f32(sess, 1, cfg.vocab)  # ne=[vocab, 1]
t_hp_v   = TinyNN.tnn_input_1d_f32(sess, 2)

t_loss = TinyNN.tnn_cross_entropy_loss(sess, step_h.kv_step_logits, t_labels)
TinyNN.tnn_set_output(t_loss)
TinyNN.tnn_set_loss(t_loss)

rc = TinyNN.tnn_build_forward_only(sess, t_loss)
if rc != 0; puts "build_forward_only rc=" + rc.to_s; exit 1; end
rc = TinyNN.tnn_build_backward(sess)
if rc != 0; puts "build_backward rc=" + rc.to_s; exit 1; end

# Wire opt_step_sgd for ALL layers (30 × 9 × 2 = 540 nodes).
n_wired = 0
li = 0
while li < cfg.n_layers
  blk = kv.kv_blocks_ffi[li]
  hq = 0
  while hq < cfg.n_heads
    t_a = blk.t_w_lora_a_q[hq]
    t_b = blk.t_w_lora_b_q[hq]
    t_grad_a = TinyNN.tnn_tensor_grad(sess, t_a)
    t_grad_b = TinyNN.tnn_tensor_grad(sess, t_b)
    if t_grad_a == nil; puts "FAIL: no grad for layer" + li.to_s + ".head" + hq.to_s + ".A"; exit 1; end
    if t_grad_b == nil; puts "FAIL: no grad for layer" + li.to_s + ".head" + hq.to_s + ".B"; exit 1; end
    t_opt_a = TinyNN.tnn_opt_step_sgd(sess, t_a, t_grad_a, t_hp_v)
    t_opt_b = TinyNN.tnn_opt_step_sgd(sess, t_b, t_grad_b, t_hp_v)
    TinyNN.tnn_extend_backward_graph(sess, t_opt_a)
    TinyNN.tnn_extend_backward_graph(sess, t_opt_b)
    n_wired = n_wired + 2
    hq = hq + 1
  end
  li = li + 1
end
puts "  wired " + n_wired.to_s + " opt_step nodes."

rc = TinyNN.tnn_realize_backward(sess)
if rc != 0; puts "realize_backward rc=" + rc.to_s; exit 1; end
puts "  graph realized."

# === Prepare per-step inputs ===
m_labels = Mat.new(1, cfg.vocab)
i = 0
while i < cfg.vocab
  m_labels.flat[i] = 0.0
  i = i + 1
end
m_labels.flat[TARGET_ID] = 1.0

m_hp_v = Mat.new(1, 2)
m_hp_v.flat[0] = LR
m_hp_v.flat[1] = 0.0

# === Train ===
puts ""
puts "=== Train " + STEPS.to_s + " SGD steps (target_id=" + TARGET_ID.to_s + ") ==="
losses = []
s = 0
while s < STEPS
  TinyNN.tnn_graph_reset(sess)
  TinyNN.upload_int_array(sess, step_h.t_token_id, [last_prompt_id])
  TinyNN.upload_int_array(sess, step_h.t_pos,      [TRAIN_POS])
  TinyNN.upload_row_major(sess, t_labels, m_labels)
  TinyNN.upload_row_major(sess, t_hp_v,   m_hp_v)

  rc = TinyNN.tnn_compute_backward(sess)
  if rc != 0; puts "compute_backward rc=" + rc.to_s; exit 1; end

  TinyNN.tnn_download(sess, t_loss)
  loss_v = TinyNN.tnn_scratch_get(sess, 0)
  losses.push(loss_v)
  puts "  step " + (s + 1).to_s + ": CE=" + loss_v.to_s
  s = s + 1
end

# === Acceptance ===
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
# Tightened gate (2026-05-21): CPU sched aliases intermediate grad
# tensors in the long SmolLM2 backward chain — gradients flow at
# ~1/1000 the correct magnitude. CPU "passes" a loose monotonic-
# decrease gate purely from FP accumulation drift, even though the
# model isn't learning. The strict `final < 0.5 * initial` gate
# rejects that fake pass. CUDA passes; CPU fails until the upstream
# ggml-cpu sched bug is fixed. See docs/design/task70-root-cause-2026-05-21.md.
threshold = 0.5 * initial
if final >= threshold
  puts "VERDICT: FAIL (CE " + initial.to_s + " → " + final.to_s +
       "; needed < " + threshold.to_s + " for real training signal)"
  exit 1
end
puts "VERDICT: PASS (CE " + initial.to_s + " → " + final.to_s + ")"
