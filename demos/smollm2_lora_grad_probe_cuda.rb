# CUDA mirror of demos/smollm2_lora_grad_probe.rb. Same setup, same
# probe; only the FFI module + cache class differ.

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../lib/toy/llm/engine/llama_kv_engine_cuda"

GGUF      = ENV["GGUF"]      || "data/smollm2-135m-native.gguf"
MAX_T     = (ENV["MAX_T"]    || "32").to_i
RANK      = (ENV["RANK"]     || "16").to_i
SEED      = (ENV["SEED"]     || "42").to_i
TARGET_ID = (ENV["TARGET_ID"]|| "99").to_i

PROMPT    = [12092, 4845, 253, 1429]
TRAIN_POS = PROMPT.length

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "# GradProbe CUDA: " + GGUF + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " heads=" + cfg.n_heads.to_s

gguf = TinyNNCuda.tnn_gguf_load(GGUF)
kv = SmolLM2KVFFICacheCuda.new
kv.enable_lora_q!(RANK)
kv.realize_for_mmap(gguf, cfg, MAX_T, flags.untied, flags.qkv_bias)
kv.upload_lora_q_init!(SEED, 0.01)

i = 0
while i < PROMPT.length
  SmolLM2KVCuda.decode_step(kv, PROMPT[i], i)
  i = i + 1
end

sess = kv.sess
TinyNNCuda.tnn_reset_for_rebuild(sess)
step_h = kv.build_decode_step(TRAIN_POS)
t_labels = TinyNNCuda.tnn_input_2d_f32(sess, 1, cfg.vocab)
t_hp_v   = TinyNNCuda.tnn_input_1d_f32(sess, 2)
t_loss = TinyNNCuda.tnn_cross_entropy_loss(sess, step_h.kv_step_logits, t_labels)
TinyNNCuda.tnn_set_output(t_loss); TinyNNCuda.tnn_set_loss(t_loss)
TinyNNCuda.tnn_build_forward_only(sess, t_loss)
TinyNNCuda.tnn_build_backward(sess)

grad_a = []
grad_b = []
li = 0
while li < cfg.n_layers
  blk = kv.kv_blocks_ffi[li]
  hq = 0
  while hq < cfg.n_heads
    t_a = blk.t_w_lora_a_q[hq]
    t_b = blk.t_w_lora_b_q[hq]
    t_grad_a = TinyNNCuda.tnn_tensor_grad(sess, t_a)
    t_grad_b = TinyNNCuda.tnn_tensor_grad(sess, t_b)
    if hq == 0
      grad_a.push(t_grad_a)
      grad_b.push(t_grad_b)
      TinyNNCuda.tnn_set_output(t_grad_a)
      TinyNNCuda.tnn_set_output(t_grad_b)
    end
    t_opt_a = TinyNNCuda.tnn_opt_step_sgd(sess, t_a, t_grad_a, t_hp_v)
    t_opt_b = TinyNNCuda.tnn_opt_step_sgd(sess, t_b, t_grad_b, t_hp_v)
    TinyNNCuda.tnn_extend_backward_graph(sess, t_opt_a)
    TinyNNCuda.tnn_extend_backward_graph(sess, t_opt_b)
    hq = hq + 1
  end
  li = li + 1
end

TinyNNCuda.tnn_realize_backward(sess)

m_labels = Mat.new(1, cfg.vocab)
i = 0
while i < cfg.vocab; m_labels.flat[i] = 0.0; i = i + 1; end
m_labels.flat[TARGET_ID] = 1.0
m_hp_v = Mat.new(1, 2)
m_hp_v.flat[0] = 0.0
m_hp_v.flat[1] = 0.0

TinyNNCuda.tnn_graph_reset(sess)
TinyNNCuda.upload_int_array(sess, step_h.t_token_id, [PROMPT[PROMPT.length - 1]])
TinyNNCuda.upload_int_array(sess, step_h.t_pos,      [TRAIN_POS])
TinyNNCuda.upload_row_major(sess, t_labels, m_labels)
TinyNNCuda.upload_row_major(sess, t_hp_v,   m_hp_v)
TinyNNCuda.tnn_compute_backward(sess)

def max_abs_of(sess, t, n)
  TinyNNCuda.tnn_download(sess, t)
  m = 0.0
  i = 0
  while i < n
    v = TinyNNCuda.tnn_scratch_get(sess, i)
    av = v
    if av < 0.0; av = -av; end
    if av > m; m = av; end
    i = i + 1
  end
  m
end

d_head = cfg.d_model / cfg.n_heads
n_a = RANK * cfg.d_model
n_b = d_head * RANK
puts "# layer  maxabs(gradA)  maxabs(gradB)"
li = 0
while li < cfg.n_layers
  ga = max_abs_of(sess, grad_a[li], n_a)
  gb = max_abs_of(sess, grad_b[li], n_b)
  puts li.to_s + " " + ga.to_s + " " + gb.to_s
  li = li + 1
end
