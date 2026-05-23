# Bench: LoRA training step wallclock (SmolLM2-135M, r=8, T=4).
# Reuses the exact realize_for_mmap + build_training_step path the
# user-facing example_finetune uses. Reports steady-state ms/step
# averaged over the last N-WARMUP steps; the first step is excluded
# because graph realize/finalize dominates and is one-shot.
#
# Output format (last lines, parsed by bench/check.rb):
#   BENCH lora_step_ms <float>
#   BENCH lora_steady_state_ms <float>

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/llama_seq_forward_ffi"

GGUF   = ENV["GGUF"]   || "data/smollm2-135m-native.gguf"
STEPS  = (ENV["STEPS"] || "10").to_i
WARMUP = 2

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
gguf  = TinyNN.tnn_gguf_load(GGUF)

TOKENS = [12092, 4845, 253, 1429]
seq = LlamaSeqForwardFFICache.new
seq.enable_lora_q!(8)
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
  m_labels.flat[ti * cfg.vocab + 99] = 1.0
  ti = ti + 1
end

m_hp = Mat.new(1, 7)
m_hp.flat[0] = 0.001; m_hp.flat[1] = 0.9; m_hp.flat[2] = 0.999
m_hp.flat[3] = 1.0e-8; m_hp.flat[4] = 0.0

positions = [0, 1, 2, 3]

# Per-step timings.
times = [0.0]; times.pop

step = 1
while step <= STEPS
  t0 = Time.now
  m_hp.flat[5] = 1.0 / (1.0 - (0.9   ** step.to_f))
  m_hp.flat[6] = 1.0 / (1.0 - (0.999 ** step.to_f))
  if step == 1
    TinyNN.tnn_graph_reset(seq.sess)
  else
    TinyNN.tnn_graph_reset_grads_only(seq.sess)
  end
  TinyNN.upload_int_array(seq.sess, seq.t_seq_token_ids, TOKENS)
  TinyNN.upload_int_array(seq.sess, seq.t_seq_positions, positions)
  TinyNN.upload_row_major(seq.sess, t_labels, m_labels)
  TinyNN.upload_row_major(seq.sess, t_hp,     m_hp)
  TinyNN.tnn_compute_backward(seq.sess)
  TinyNN.tnn_download(seq.sess, t_loss)
  times.push((Time.now - t0).to_f * 1000.0)
  step = step + 1
end

# Mean of all steps; mean of warm steps only.
sum_all = 0.0
i = 0
while i < times.length; sum_all = sum_all + times[i]; i = i + 1; end
mean_all = sum_all / times.length.to_f

sum_warm = 0.0
n_warm = 0
i = WARMUP
while i < times.length
  sum_warm = sum_warm + times[i]
  n_warm = n_warm + 1
  i = i + 1
end
mean_warm = n_warm > 0 ? sum_warm / n_warm.to_f : mean_all

puts "BENCH lora_step_ms " + mean_all.to_s
puts "BENCH lora_steady_state_ms " + mean_warm.to_s
