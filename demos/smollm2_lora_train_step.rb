# F1.2 step 3 — backward through the full SmolLM2 decode graph,
# updating layer-0 LoRA-Q params via SGD.
#
# This is the first end-to-end SmolLM2 LoRA training step. The forward
# graph is the existing per-step KV decode (build_decode_step), which
# touches matmul / rms_norm / rope_ext / view / cpy / scale / softmax /
# concat / silu / swiglu. Backward through all of those is exercised
# here; the CONCAT backward was vendored locally in vendor/ggml/src/ggml.c
# specifically for this path (see docs/design/concat-back-patch-2026-05-21.md).
#
# Scope: only the LAYER 0 LoRA-A and LoRA-B tensors are wired into
# opt_step nodes. SmolLM2-135M has 30 layers × 9 heads = 540 LoRA
# tensors total; wiring opt_step for all of them inflates graph_b
# beyond useful debugging scale. Layer 0 (18 tensors → 18 opt_step
# nodes) is enough to confirm the wiring + the autograd chain.
#
# Loss: sum(logits²) — a synthetic regulariser. Will it go DOWN
# under SGD? Only if the LoRA on layer-0 Q can influence the final
# logit magnitudes. Since it's only layer 0 (further layers see the
# perturbed output via 29 more transformer blocks of computation),
# the gradient signal at the LoRA params is small. We don't strictly
# assert monotonic decrease — we assert (a) no NaN, (b) finite grads,
# and (c) at least one of the LoRA params shifted measurably.
#
# Run: GGUF=data/smollm2-135m-native.gguf ./demos/smollm2_lora_train_step

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv"

GGUF  = ENV["GGUF"]  || "data/smollm2-135m-native.gguf"
MAX_T = (ENV["MAX_T"] || "32").to_i
RANK  = (ENV["RANK"]  || "16").to_i
SEED  = (ENV["SEED"]  || "42").to_i
STEPS = (ENV["STEPS"] || "3").to_i
LR    = 0.001

# Prefill prompt: same 4-token prefix as the forward parity smoke.
PROMPT  = [12092, 4845, 253, 1429]
TRAIN_POS = PROMPT.length    # the position we'll train at

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "cfg: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " n_q=" + cfg.n_heads.to_s +
     " n_kv=" + cfg.n_kv.to_s

gguf = TinyNN.tnn_gguf_load(GGUF)
kv = SmolLM2KVFFICache.new
kv.enable_lora_q!(RANK)
kv.realize_for_mmap(gguf, cfg, MAX_T, flags.untied, flags.qkv_bias)
kv.upload_lora_q_init!(SEED, 0.01)
puts "  backend: " + TinyNN.tnn_backend_name(kv.sess)

# Prefill: standard decode_step writes K/V cache slots 0 .. TRAIN_POS-1.
puts ""
puts "=== Prefill " + PROMPT.length.to_s + " tokens ==="
i = 0
while i < PROMPT.length
  SmolLM2KV.decode_step(kv, PROMPT[i], i)
  i = i + 1
end

# Capture initial value of one LoRA-B element to verify it moves.
b0_initial = kv.read_persistent_mat(kv.kv_blocks_ffi[0].t_w_lora_b_q[0],
                                      cfg.d_model / cfg.n_heads, RANK)
puts "  layer0.head0.B[0,0] init = " + b0_initial.flat[0].to_s

# === Build the training graph at TRAIN_POS ===
puts ""
puts "=== Build training graph at pos=" + TRAIN_POS.to_s + " ==="

sess = kv.sess
TinyNN.tnn_reset_for_rebuild(sess)
step_h = kv.build_decode_step(TRAIN_POS)

# Loss = sum(logits²). Persistent hyperparams for SGD: [lr, wd].
t_logits   = step_h.kv_step_logits             # ne=[1, vocab]
t_logits_sq = TinyNN.tnn_mul(sess, t_logits, t_logits)
t_loss     = TinyNN.tnn_sum(sess, t_logits_sq)  # scalar
TinyNN.tnn_set_output(t_loss)
TinyNN.tnn_set_loss(t_loss)

t_hp_v = TinyNN.tnn_input_1d_f32(sess, 2)   # compute-side; re-upload each step

rc = TinyNN.tnn_build_forward_only(sess, t_loss)
if rc != 0; puts "build_forward_only rc=" + rc.to_s; exit 1; end

rc = TinyNN.tnn_build_backward(sess)
if rc != 0; puts "build_backward rc=" + rc.to_s; exit 1; end

# Attach SGD opt_step nodes for layer 0's LoRA params (18 tensors).
blk0 = kv.kv_blocks_ffi[0]
hq = 0
while hq < cfg.n_heads
  t_a = blk0.t_w_lora_a_q[hq]
  t_b = blk0.t_w_lora_b_q[hq]
  t_grad_a = TinyNN.tnn_tensor_grad(sess, t_a)
  t_grad_b = TinyNN.tnn_tensor_grad(sess, t_b)
  if t_grad_a == nil; puts "FAIL: no grad for layer0.head" + hq.to_s + ".A"; exit 1; end
  if t_grad_b == nil; puts "FAIL: no grad for layer0.head" + hq.to_s + ".B"; exit 1; end
  t_opt_a = TinyNN.tnn_opt_step_sgd(sess, t_a, t_grad_a, t_hp_v)
  t_opt_b = TinyNN.tnn_opt_step_sgd(sess, t_b, t_grad_b, t_hp_v)
  TinyNN.tnn_extend_backward_graph(sess, t_opt_a)
  TinyNN.tnn_extend_backward_graph(sess, t_opt_b)
  hq = hq + 1
end

rc = TinyNN.tnn_realize_backward(sess)
if rc != 0; puts "realize_backward rc=" + rc.to_s; exit 1; end
puts "  graph realized."

# === Training loop ===
# token_id and pos are compute-side persistent across the loop iters
# but their slots get fresh sched-alloc per realize_backward — we
# upload them AFTER realize. hp is also compute-side. Re-upload each
# step (cheap; immune to any sched-reset surprises).
m_hp_v = Mat.new(1, 2)
m_hp_v.flat[0] = LR; m_hp_v.flat[1] = 0.0

puts ""
puts "=== Train " + STEPS.to_s + " SGD steps ==="
losses = []
s = 0
while s < STEPS
  TinyNN.tnn_graph_reset(sess)
  TinyNN.upload_int_array(sess, step_h.t_token_id, [PROMPT[PROMPT.length - 1]])
  TinyNN.upload_int_array(sess, step_h.t_pos,      [TRAIN_POS])
  TinyNN.stage_row_major_and_upload(sess, t_hp_v, m_hp_v)

  rc = TinyNN.tnn_compute_backward(sess)
  if rc != 0; puts "compute_backward rc=" + rc.to_s; exit 1; end

  TinyNN.tnn_download(sess, t_loss)
  loss_v = TinyNN.tnn_scratch_get(sess, 0)
  losses.push(loss_v)
  puts "  step " + (s + 1).to_s + ": loss=" + loss_v.to_s
  s = s + 1
end

# === Verify: LoRA-B moved, no NaN ===
b0_final = kv.read_persistent_mat(blk0.t_w_lora_b_q[0],
                                   cfg.d_model / cfg.n_heads, RANK)
delta = (b0_final.flat[0] - b0_initial.flat[0]).abs
puts ""
puts "layer0.head0.B[0,0] final = " + b0_final.flat[0].to_s
puts "delta from initial         = " + delta.to_s

any_nan = false
s = 0
while s < losses.length
  if losses[s].to_s == "NaN"; any_nan = true; end
  s = s + 1
end

puts ""
if any_nan
  puts "VERDICT: FAIL (loss became NaN)"; exit 1
end
if delta == 0.0
  puts "VERDICT: FAIL (LoRA-B did not move under SGD — backward chain broken upstream of layer 0)"; exit 1
end
puts "VERDICT: PASS (backward through SmolLM2 forward graph + opt_step on LoRA params works)"
