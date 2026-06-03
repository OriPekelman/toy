# CPU mirror of demos/smollm2_lora_train_bench_cuda.rb. Same shape;
# swap TinyNNCuda → TinyNN, SmolLM2KVFFICacheCuda → SmolLM2KVFFICache,
# SmolLM2KVCuda → SmolLM2KV.

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/models/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv"

GGUF      = ENV["GGUF"]      || "data/smollm2-135m-native.gguf"
MAX_T     = (ENV["MAX_T"]    || "32").to_i
RANK      = (ENV["RANK"]     || "16").to_i
SEED      = (ENV["SEED"]     || "42").to_i
N_WARMUP  = (ENV["N_WARMUP"] || "3").to_i
N_STEPS   = (ENV["N_STEPS"]  || "10").to_i
LR        = (ENV["LR"]       || "0.1").to_f
TARGET_ID = (ENV["TARGET_ID"]|| "99").to_i

PROMPT    = [12092, 4845, 253, 1429]
TRAIN_POS = PROMPT.length

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "=== Training bench (CPU): " + GGUF + " ==="
puts "  cfg: L=" + cfg.n_layers.to_s + " heads=" + cfg.n_heads.to_s +
     " d=" + cfg.d_model.to_s + " r=" + RANK.to_s

t_setup_0 = Time.now
gguf = TinyNN.tnn_gguf_load(GGUF)
kv = SmolLM2KVFFICache.new
kv.enable_lora_q!(RANK)
kv.realize_for_mmap(gguf, cfg, MAX_T, flags.untied, flags.qkv_bias)
kv.upload_lora_q_init!(SEED, 0.01)
t_setup_load_ms = (Time.now - t_setup_0) * 1000.0

i = 0
while i < PROMPT.length
  SmolLM2KV.decode_step(kv, PROMPT[i], i)
  i = i + 1
end
last_prompt_id = PROMPT[PROMPT.length - 1]

sess = kv.sess
t_build_0 = Time.now
TinyNN.tnn_reset_for_rebuild(sess)
step_h = kv.build_decode_step(TRAIN_POS)

t_labels = TinyNN.tnn_input_2d_f32(sess, 1, cfg.vocab)
t_hp_v   = TinyNN.tnn_input_1d_f32(sess, 2)
t_loss = TinyNN.tnn_cross_entropy_loss(sess, step_h.kv_step_logits, t_labels)
TinyNN.tnn_set_output(t_loss)
TinyNN.tnn_set_loss(t_loss)

TinyNN.tnn_build_forward_only(sess, t_loss)
TinyNN.tnn_build_backward(sess)

li = 0
while li < cfg.n_layers
  blk = kv.kv_blocks_ffi[li]
  hq = 0
  while hq < cfg.n_heads
    t_a = blk.t_w_lora_a_q[hq]
    t_b = blk.t_w_lora_b_q[hq]
    t_grad_a = TinyNN.tnn_tensor_grad(sess, t_a)
    t_grad_b = TinyNN.tnn_tensor_grad(sess, t_b)
    t_opt_a = TinyNN.tnn_opt_step_sgd(sess, t_a, t_grad_a, t_hp_v)
    t_opt_b = TinyNN.tnn_opt_step_sgd(sess, t_b, t_grad_b, t_hp_v)
    TinyNN.tnn_extend_backward_graph(sess, t_opt_a)
    TinyNN.tnn_extend_backward_graph(sess, t_opt_b)
    hq = hq + 1
  end
  li = li + 1
end
TinyNN.tnn_realize_backward(sess)
t_build_ms = (Time.now - t_build_0) * 1000.0

puts "  setup (load + LoRA init + prefill): " + t_setup_load_ms.to_s + " ms"
puts "  build training graph (reset + decode_step + loss + bwd + 540 opt_step + realize): " + t_build_ms.to_s + " ms"

m_labels = Mat.new(1, cfg.vocab)
i = 0
while i < cfg.vocab; m_labels.flat[i] = 0.0; i = i + 1; end
m_labels.flat[TARGET_ID] = 1.0
m_hp_v = Mat.new(1, 2)
m_hp_v.flat[0] = LR
m_hp_v.flat[1] = 0.0

i = 0
while i < N_WARMUP
  TinyNN.tnn_graph_reset(sess)
  TinyNN.upload_int_array(sess, step_h.t_token_id, [last_prompt_id])
  TinyNN.upload_int_array(sess, step_h.t_pos, [TRAIN_POS])
  TinyNN.upload_row_major(sess, t_labels, m_labels)
  TinyNN.upload_row_major(sess, t_hp_v,   m_hp_v)
  TinyNN.tnn_compute_backward(sess)
  TinyNN.tnn_download(sess, t_loss)
  i = i + 1
end

reset_ms   = []
upload_ms  = []
compute_ms = []
dl_ms      = []
total_ms   = []
loss_vals  = []

s = 0
while s < N_STEPS
  t_step_0 = Time.now
  TinyNN.tnn_graph_reset(sess)
  t1 = Time.now
  TinyNN.upload_int_array(sess, step_h.t_token_id, [last_prompt_id])
  TinyNN.upload_int_array(sess, step_h.t_pos,      [TRAIN_POS])
  TinyNN.upload_row_major(sess, t_labels, m_labels)
  TinyNN.upload_row_major(sess, t_hp_v,   m_hp_v)
  t2 = Time.now
  TinyNN.tnn_compute_backward(sess)
  t3 = Time.now
  TinyNN.tnn_download(sess, t_loss)
  t4 = Time.now
  reset_ms.push((t1 - t_step_0) * 1000.0)
  upload_ms.push((t2 - t1) * 1000.0)
  compute_ms.push((t3 - t2) * 1000.0)
  dl_ms.push((t4 - t3) * 1000.0)
  total_ms.push((t4 - t_step_0) * 1000.0)
  loss_vals.push(TinyNN.tnn_scratch_get(sess, 0))
  s = s + 1
end

def mean_ms(arr)
  s = 0.0
  i = 0
  while i < arr.length
    s = s + arr[i]
    i = i + 1
  end
  s / arr.length.to_f
end

r_mean = mean_ms(reset_ms)
u_mean = mean_ms(upload_ms)
c_mean = mean_ms(compute_ms)
d_mean = mean_ms(dl_ms)
t_mean = mean_ms(total_ms)

puts ""
puts "=== Per-step phase means (CPU, " + N_STEPS.to_s + " steps after " + N_WARMUP.to_s + " warmup) ==="
puts "  graph_reset       : " + r_mean.to_s + " ms"
puts "  upload (inputs)   : " + u_mean.to_s + " ms"
puts "  compute_backward  : " + c_mean.to_s + " ms"
puts "  download (loss)   : " + d_mean.to_s + " ms"
puts "  total per step    : " + t_mean.to_s + " ms"
puts ""
puts "  steps/sec         : " + (1000.0 / t_mean).to_s
puts "  loss[0]=" + loss_vals[0].to_s + " loss[last]=" + loss_vals[loss_vals.length - 1].to_s
