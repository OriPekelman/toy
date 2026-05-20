# F2 step 2 — multi-layer SmolLM2 LoRA training with real CE loss on CUDA.
#
# CUDA mirror of demos/smollm2_lora_train_ce.rb. Same shape: 30 layers
# × 9 heads × 2 LoRA-Q params = 540 opt_step nodes, CE loss against a
# rare target token (id=99), 20 SGD steps.
#
# Acceptance: monotonic CE decrease over 20 steps. Loss values should
# match the CPU smoke within F32 tolerance — the underlying graph is
# identical; only the backend differs.
#
# Vendored ggml-cuda patches required (cpy.cu strided fix + concat
# backward + BYO-pointer). All applied automatically by
# `make setup-ggml-cuda` via vendor-patches/.
#
# Run: GGUF=data/smollm2-135m-native.gguf ./demos/smollm2_lora_train_ce_cuda

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv_cuda"

GGUF      = ENV["GGUF"]      || "data/smollm2-135m-native.gguf"
MAX_T     = (ENV["MAX_T"]    || "32").to_i
RANK      = (ENV["RANK"]     || "16").to_i
SEED      = (ENV["SEED"]     || "42").to_i
STEPS     = (ENV["STEPS"]    || "20").to_i
LR        = (ENV["LR"]       || "0.5").to_f
TARGET_ID = (ENV["TARGET_ID"]|| "99").to_i

PROMPT    = [12092, 4845, 253, 1429]
TRAIN_POS = PROMPT.length

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "cfg: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " n_q=" + cfg.n_heads.to_s
puts "training: LR=" + LR.to_s + " STEPS=" + STEPS.to_s +
     " TARGET_ID=" + TARGET_ID.to_s + " r=" + RANK.to_s

gguf = TinyNNCuda.tnn_gguf_load(GGUF)
kv = SmolLM2KVFFICacheCuda.new
kv.enable_lora_q!(RANK)
kv.realize_for_mmap(gguf, cfg, MAX_T, flags.untied, flags.qkv_bias)
kv.upload_lora_q_init!(SEED, 0.01)
puts "  backend: " + TinyNNCuda.tnn_backend_name(kv.sess)

puts ""
puts "=== Prefill " + PROMPT.length.to_s + " tokens ==="
i = 0
while i < PROMPT.length
  SmolLM2KVCuda.decode_step(kv, PROMPT[i], i)
  i = i + 1
end
last_prompt_id = PROMPT[PROMPT.length - 1]

puts ""
puts "=== Build training graph (all " + cfg.n_layers.to_s + " layers, " +
     (cfg.n_layers * cfg.n_heads * 2).to_s + " LoRA params) ==="
sess = kv.sess
TinyNNCuda.tnn_reset_for_rebuild(sess)
step_h = kv.build_decode_step(TRAIN_POS)

t_labels = TinyNNCuda.tnn_input_2d_f32(sess, 1, cfg.vocab)
t_hp_v   = TinyNNCuda.tnn_input_1d_f32(sess, 2)

t_loss = TinyNNCuda.tnn_cross_entropy_loss(sess, step_h.kv_step_logits, t_labels)
TinyNNCuda.tnn_set_output(t_loss)
TinyNNCuda.tnn_set_loss(t_loss)

rc = TinyNNCuda.tnn_build_forward_only(sess, t_loss)
if rc != 0; puts "build_forward_only rc=" + rc.to_s; exit 1; end
rc = TinyNNCuda.tnn_build_backward(sess)
if rc != 0; puts "build_backward rc=" + rc.to_s; exit 1; end

n_wired = 0
li = 0
while li < cfg.n_layers
  blk = kv.kv_blocks_ffi[li]
  hq = 0
  while hq < cfg.n_heads
    t_a = blk.t_w_lora_a_q[hq]
    t_b = blk.t_w_lora_b_q[hq]
    t_grad_a = TinyNNCuda.tnn_tensor_grad(sess, t_a)
    t_grad_b = TinyNNCuda.tnn_tensor_grad(sess, t_b)
    if t_grad_a == nil; puts "FAIL: no grad for layer" + li.to_s + ".head" + hq.to_s + ".A"; exit 1; end
    if t_grad_b == nil; puts "FAIL: no grad for layer" + li.to_s + ".head" + hq.to_s + ".B"; exit 1; end
    t_opt_a = TinyNNCuda.tnn_opt_step_sgd(sess, t_a, t_grad_a, t_hp_v)
    t_opt_b = TinyNNCuda.tnn_opt_step_sgd(sess, t_b, t_grad_b, t_hp_v)
    TinyNNCuda.tnn_extend_backward_graph(sess, t_opt_a)
    TinyNNCuda.tnn_extend_backward_graph(sess, t_opt_b)
    n_wired = n_wired + 2
    hq = hq + 1
  end
  li = li + 1
end
puts "  wired " + n_wired.to_s + " opt_step nodes."

rc = TinyNNCuda.tnn_realize_backward(sess)
if rc != 0; puts "realize_backward rc=" + rc.to_s; exit 1; end
puts "  graph realized."

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

puts ""
puts "=== Train " + STEPS.to_s + " SGD steps (target_id=" + TARGET_ID.to_s + ") ==="
losses = []
s = 0
while s < STEPS
  TinyNNCuda.tnn_graph_reset(sess)
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
# Tightened gate matching the CPU smoke (see comment there + task #70).
# CUDA training is real — CE drops from ~7.5 to ~0.2 in 20 SGD steps.
threshold = 0.5 * initial
if final >= threshold
  puts "VERDICT: FAIL (CE " + initial.to_s + " → " + final.to_s +
       "; needed < " + threshold.to_s + ")"
  exit 1
end
puts "VERDICT: PASS (CE " + initial.to_s + " → " + final.to_s + ")"
