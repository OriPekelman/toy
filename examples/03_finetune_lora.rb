# Fine-tune SmolLM2-135M with LoRA on its attention Q projection.
# Adds rank-16 adapters per Q head (270 trainable matrices, 540 with
# their (m, v) Adam state); the 134M base weights stay frozen.
#
# This example trains the model to shift its logit distribution at a
# fixed position toward a chosen target token. Real SFT (varied
# prefixes + multi-position + a dataset) needs a sequence-mode
# forward graph (see docs/design/phase-f1-2-step6-status.md). The
# mechanics here are the same.
#
#   make example_finetune_cuda
#   ./examples/example_finetune_cuda
#
# CUDA only: ggml-cpu's sched aliases intermediate grad tensors in
# long backward chains (filed as ggml-org/ggml#1501; full diagnosis at
# docs/design/task70-root-cause-2026-05-21.md). CPU LoRA training
# silently underflows gradients until that lands.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv_cuda"

GGUF      = ENV["GGUF"]      || "data/smollm2-135m-native.gguf"
RANK      = (ENV["RANK"]     || "16").to_i
STEPS     = (ENV["STEPS"]    || "20").to_i
LR        = (ENV["LR"]       || "0.001").to_f
TARGET_ID = (ENV["TARGET_ID"]|| "99").to_i

PROMPT    = [12092, 4845, 253, 1429]  # "Once upon a time"
TRAIN_POS = PROMPT.length

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)

gguf = TinyNNCuda.tnn_gguf_load(GGUF)
kv = SmolLM2KVFFICacheCuda.new
kv.enable_lora_q!(RANK)
kv.realize_for_mmap(gguf, cfg, 32, flags.untied, flags.qkv_bias)
kv.upload_lora_q_init!(42, 0.01)

i = 0
while i < PROMPT.length
  SmolLM2KVCuda.decode_step(kv, PROMPT[i], i)
  i = i + 1
end

# Adam state — same shape as each LoRA pair.
m_a = []; v_a = []; m_b = []; v_b = []
sess = kv.sess
n = cfg.n_layers * cfg.n_heads; k = 0
while k < n
  m_a.push(TinyNNCuda.tnn_input_2d_f32(sess, RANK, cfg.d_model))
  v_a.push(TinyNNCuda.tnn_input_2d_f32(sess, RANK, cfg.d_model))
  m_b.push(TinyNNCuda.tnn_input_2d_f32(sess, cfg.d_model / cfg.n_heads, RANK))
  v_b.push(TinyNNCuda.tnn_input_2d_f32(sess, cfg.d_model / cfg.n_heads, RANK))
  k = k + 1
end

TinyNNCuda.tnn_reset_for_rebuild(sess)
step_h = kv.build_decode_step(TRAIN_POS)
t_labels = TinyNNCuda.tnn_input_2d_f32(sess, 1, cfg.vocab)
t_hp     = TinyNNCuda.tnn_input_1d_f32(sess, 7)
t_loss   = TinyNNCuda.tnn_cross_entropy_loss(sess, step_h.kv_step_logits, t_labels)
TinyNNCuda.tnn_set_output(t_loss); TinyNNCuda.tnn_set_loss(t_loss)
TinyNNCuda.tnn_build_forward_only(sess, t_loss)
TinyNNCuda.tnn_build_backward(sess)

# Attach AdamW opt_step per LoRA pair.
idx = 0; li = 0
while li < cfg.n_layers
  blk = kv.kv_blocks_ffi[li]; hq = 0
  while hq < cfg.n_heads
    g_a = TinyNNCuda.tnn_tensor_grad(sess, blk.t_w_lora_a_q[hq])
    g_b = TinyNNCuda.tnn_tensor_grad(sess, blk.t_w_lora_b_q[hq])
    TinyNNCuda.tnn_extend_backward_graph(sess,
      TinyNNCuda.tnn_opt_step_adamw(sess, blk.t_w_lora_a_q[hq], g_a, m_a[idx], v_a[idx], t_hp))
    TinyNNCuda.tnn_extend_backward_graph(sess,
      TinyNNCuda.tnn_opt_step_adamw(sess, blk.t_w_lora_b_q[hq], g_b, m_b[idx], v_b[idx], t_hp))
    idx = idx + 1; hq = hq + 1
  end
  li = li + 1
end
TinyNNCuda.tnn_realize_backward(sess)

m_labels = Mat.new(1, cfg.vocab)
i = 0
while i < cfg.vocab; m_labels.flat[i] = 0.0; i = i + 1; end
m_labels.flat[TARGET_ID] = 1.0
TinyNNCuda.tnn_graph_reset(sess)

last = PROMPT[PROMPT.length - 1]
s = 0
while s < STEPS
  if s > 0; TinyNNCuda.tnn_graph_reset_grads_only(sess); end
  t = s + 1
  b1h = 1.0 / (1.0 - (0.9 ** t.to_f))
  b2h = 1.0 / (1.0 - (0.999 ** t.to_f))
  hp = Mat.new(1, 7)
  hp.flat[0] = LR; hp.flat[1] = 0.9; hp.flat[2] = 0.999
  hp.flat[3] = 1.0e-8; hp.flat[4] = 0.0; hp.flat[5] = b1h; hp.flat[6] = b2h
  TinyNNCuda.upload_int_array(sess, step_h.t_token_id, [last])
  TinyNNCuda.upload_int_array(sess, step_h.t_pos,      [TRAIN_POS])
  TinyNNCuda.upload_row_major(sess, t_labels, m_labels)
  TinyNNCuda.upload_row_major(sess, t_hp,     hp)
  TinyNNCuda.tnn_compute_backward(sess)
  TinyNNCuda.tnn_download(sess, t_loss)
  puts "step " + (s + 1).to_s + ": CE=" + TinyNNCuda.tnn_scratch_get(sess, 0).to_s
  s = s + 1
end
