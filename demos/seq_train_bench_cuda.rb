require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/llama_seq_forward_ffi_cuda"

GGUF = ENV["GGUF"] || "data/smollm2-135m-native.gguf"
MODE = ENV["MODE"] || "lora"  # lora | ft
STEPS = (ENV["STEPS"] || "10").to_i

TOKENS = [12092, 4845, 253, 1429]
cfg = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)

gguf = TinyNNCuda.tnn_gguf_load(GGUF)
seq = LlamaSeqForwardFFICacheCuda.new
if MODE == "ft"
  seq.enable_full_finetune!
  seq.realize_for_full_finetune(gguf, cfg, TOKENS.length, flags.untied, flags.qkv_bias)
else
  seq.enable_lora_q!(8)
  seq.enable_lora_q_adamw!
  seq.realize_for_mmap(gguf, cfg, TOKENS.length, flags.untied, flags.qkv_bias)
  seq.upload_lora_q_init!(42, 0.01)
end

result = seq.build_training_step
t_loss, t_labels, t_hp = result[0], result[1], result[2]

m_labels = Mat.new(TOKENS.length, cfg.vocab)
i = 0; while i < TOKENS.length * cfg.vocab; m_labels.flat[i] = 0.0; i = i + 1; end
i = 0; while i < TOKENS.length; m_labels.flat[i * cfg.vocab + 99] = 1.0; i = i + 1; end

m_hp = Mat.new(1, 7)
m_hp.flat[0] = 0.001; m_hp.flat[1] = 0.9; m_hp.flat[2] = 0.999
m_hp.flat[3] = 1.0e-8; m_hp.flat[4] = 0.0

times = [0.0]; times.pop
positions = [0, 1, 2, 3]
step = 1
while step <= STEPS
  m_hp.flat[5] = 1.0 / (1.0 - (0.9 ** step.to_f))
  m_hp.flat[6] = 1.0 / (1.0 - (0.999 ** step.to_f))
  if step == 1
    TinyNNCuda.tnn_graph_reset(seq.sess)
  else
    TinyNNCuda.tnn_graph_reset_grads_only(seq.sess)
  end
  t0 = Time.now
  TinyNNCuda.upload_int_array(seq.sess, seq.t_seq_token_ids, TOKENS)
  TinyNNCuda.upload_int_array(seq.sess, seq.t_seq_positions, positions)
  TinyNNCuda.upload_row_major(seq.sess, t_labels, m_labels)
  TinyNNCuda.upload_row_major(seq.sess, t_hp, m_hp)
  TinyNNCuda.tnn_compute_backward(seq.sess)
  ms = (Time.now - t0) * 1000.0
  times.push(ms)
  step = step + 1
end

# Drop first step (compile warmup)
total = 0.0; i = 1; while i < times.length; total = total + times[i]; i = i + 1; end
puts MODE.upcase + " step time (excl warmup): mean=" + (total / (times.length - 1)).to_s + " ms over " + (times.length - 1).to_s + " steps"
