# CUDA mirror of examples/03_finetune_lora.rb.
# Same shape: sequence-mode forward + LoRA-Q + persistent AdamW.
#
#   make example_finetune_cuda
#   ./examples/example_finetune_cuda
#
# F32 base on CUDA — fully supported.
# Q8 base on CUDA — currently hits a BYO-pointer buffer-padding
# limitation in vendor-patches/0002 (works on CPU; CUDA path pending).
# Use the CPU example (examples/03_finetune_lora.rb) with a Q8 GGUF
# for QLoRA today.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/llama_seq_forward_ffi_cuda"

GGUF      = ENV["GGUF"]    || "data/smollm2-135m-native.gguf"
RANK      = (ENV["RANK"]   || "8").to_i
STEPS     = (ENV["STEPS"]  || "20").to_i
LR        = (ENV["LR"]     || "0.001").to_f

TOKENS    = [12092, 4845, 253, 1429]
TARGET_ID = (ENV["TARGET_ID"] || "99").to_i

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "config: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " heads=" + cfg.n_heads.to_s
puts "training (CUDA): GGUF=" + GGUF + " RANK=" + RANK.to_s + " STEPS=" + STEPS.to_s

gguf = TinyNNCuda.tnn_gguf_load(GGUF)
seq = LlamaSeqForwardFFICacheCuda.new
seq.enable_lora_q!(RANK)
seq.enable_lora_q_adamw!
seq.realize_for_mmap(gguf, cfg, TOKENS.length, flags.untied, flags.qkv_bias)
seq.upload_lora_q_init!(42, 0.01)

result   = seq.build_training_step
t_loss   = result[0]
t_labels = result[1]
t_hp     = result[2]

m_labels = Mat.new(TOKENS.length, cfg.vocab)
i = 0
while i < TOKENS.length * cfg.vocab; m_labels.flat[i] = 0.0; i = i + 1; end
ti = 0
while ti < TOKENS.length
  m_labels.flat[ti * cfg.vocab + TARGET_ID] = 1.0
  ti = ti + 1
end

m_hp = Mat.new(1, 7)
m_hp.flat[0] = LR
m_hp.flat[1] = 0.9
m_hp.flat[2] = 0.999
m_hp.flat[3] = 1.0e-8
m_hp.flat[4] = 0.0

positions = [0, 1, 2, 3]
step = 1
while step <= STEPS
  m_hp.flat[5] = 1.0 / (1.0 - (0.9   ** step.to_f))
  m_hp.flat[6] = 1.0 / (1.0 - (0.999 ** step.to_f))
  if step == 1
    TinyNNCuda.tnn_graph_reset(seq.sess)
  else
    TinyNNCuda.tnn_graph_reset_grads_only(seq.sess)
  end
  TinyNNCuda.upload_int_array(seq.sess, seq.t_seq_token_ids, TOKENS)
  TinyNNCuda.upload_int_array(seq.sess, seq.t_seq_positions, positions)
  TinyNNCuda.upload_row_major(seq.sess, t_labels, m_labels)
  TinyNNCuda.upload_row_major(seq.sess, t_hp,     m_hp)
  TinyNNCuda.tnn_compute_backward(seq.sess)
  TinyNNCuda.tnn_download(seq.sess, t_loss)
  puts "step " + step.to_s.rjust(3) + ": CE=" + TinyNNCuda.tnn_scratch_get(seq.sess, 0).to_s
  step = step + 1
end
