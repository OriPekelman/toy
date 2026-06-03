# demos/smollm2_seq_qlora_cuda.rb — F4 acceptance on CUDA.
#
# QLoRA = Q8 base + F32 LoRA adapter on the Q projection. The base
# weights stay quantized in the standard CUDA buffer (allocated via
# tnn_input_2d_persistent_typed) and are loaded verbatim — bytes from
# the GGUF straight into the buffer. The LoRA-A/B pairs and their
# AdamW (m, v) state are F32 in ctx_w next to them.
#
# Why this instead of the BYO-pointer mmap path: vendor-patches/0002's
# CUDA buffer reuse the standard cuda buffer interface, which pads
# quantized tensor allocations to MATRIX_ROW_PADDING=512 and then
# zero-fills the padding via cudaMemset. On a BYO-pointer buffer
# (host-mmap'd region pinned through cudaHostRegister) the zero-fill
# writes past the file's per-tensor bytes into adjacent tensors'
# memory — crash. The standard ctx_w buffer size is computed
# correctly via get_alloc_size so the padding is included in the
# allocation; no overflow.
#
# Trade-off: weights live in CUDA device memory (not host mmap), so
# load time is dominated by the cudaMemcpy. For Qwen2.5-0.5B Q8
# (~460 MB on disk) that's < a second on GB10.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy/llm/engine/llama_seq_engine_cuda"

GGUF      = ENV["GGUF"]    || "data/qwen25-0.5b-native-q8.gguf"
RANK      = (ENV["RANK"]   || "8").to_i
STEPS     = (ENV["STEPS"]  || "20").to_i
LR        = (ENV["LR"]     || "0.001").to_f

TOKENS    = [12092, 4845, 253, 1429]
TARGET_ID = (ENV["TARGET_ID"] || "99").to_i

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "config: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " heads=" + cfg.n_heads.to_s
puts "qlora (CUDA): GGUF=" + GGUF + " RANK=" + RANK.to_s + " STEPS=" + STEPS.to_s

gguf = TinyNNCuda.tnn_gguf_load(GGUF)
seq = Toy::LLM::Engine::LlamaSeqEngineCuda.new
seq.enable_lora_q!(RANK)
seq.enable_lora_q_adamw!
seq.realize_for_q8_copy(gguf, cfg, TOKENS.length, flags.untied, flags.qkv_bias)
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

losses = [0.0]; losses.pop
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
  loss = TinyNNCuda.tnn_scratch_get(seq.sess, 0)
  losses.push(loss)
  if (step <= 5) || (step % 5 == 0) || (step == STEPS)
    puts "  step " + step.to_s.rjust(2) + ": CE=" + loss.to_s
  end
  step = step + 1
end

init  = losses[0]
final = losses[losses.length - 1]
puts ""
puts "loss: " + init.to_s + " → " + final.to_s

if final.to_s == "NaN"
  puts "VERDICT: FAIL (NaN — QLoRA divergence)"; exit 1
end
if final >= init
  puts "VERDICT: FAIL (no convergence: " + init.to_s + " → " + final.to_s + ")"; exit 1
end
ratio = final / init
puts "ratio = " + ratio.to_s
if ratio >= 0.5
  puts "VERDICT: FAIL (slow: ratio " + ratio.to_s + ")"; exit 1
end
puts "VERDICT: PASS (CUDA QLoRA: " + init.to_s + " → " + final.to_s + ")"
