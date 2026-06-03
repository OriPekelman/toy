# demos/smollm2_seq_train_cuda.rb — M3 step 3 acceptance (CUDA).
#
# Mirror of demos/smollm2_seq_train.rb on the GPU. CUDA's sched is fine
# on long backward chains (no graph_b pin needed; the CPU sched-alias
# bug doesn't reproduce here per
# project_cpu_cuda_lora_train_divergence_2026_05_21).
#
# Acceptance: loss decreases monotonically, final < 0.5 × initial.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy/llm/engine/llama_seq_engine_cuda"

GGUF      = ENV["GGUF"]    || "data/smollm2-135m-native.gguf"
RANK      = (ENV["RANK"]   || "8").to_i
SEED      = (ENV["SEED"]   || "42").to_i
STEPS     = (ENV["STEPS"]  || "20").to_i
LR        = (ENV["LR"]     || "0.001").to_f

TOKENS    = [12092, 4845, 253, 1429]
TARGET_ID = 99

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "config: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " n_q=" + cfg.n_heads.to_s
puts "training: T=" + TOKENS.length.to_s + " RANK=" + RANK.to_s +
     " STEPS=" + STEPS.to_s + " LR=" + LR.to_s

gguf = TinyNNCuda.tnn_gguf_load(GGUF)
seq = Toy::LLM::Engine::LlamaSeqEngineCuda.new
seq.enable_lora_q!(RANK)
seq.enable_lora_q_adamw!
seq.realize_for_mmap(gguf, cfg, TOKENS.length, flags.untied, flags.qkv_bias)
seq.upload_lora_q_init!(SEED, 0.01)

puts ""
puts "building training graph..."
result = seq.build_training_step
if result == nil
  puts "build_training_step returned nil"; exit 1
end
t_loss   = result[0]
t_labels = result[1]
t_hp     = result[2]

positions = [0, 1, 2, 3]

m_labels = Mat.new(TOKENS.length, cfg.vocab)
i = 0
while i < TOKENS.length * cfg.vocab
  m_labels.flat[i] = 0.0
  i = i + 1
end
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

losses = [0.0]
losses.pop

puts ""
puts "training " + STEPS.to_s + " steps..."
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

init = losses[0]
final = losses[losses.length - 1]
puts ""
puts "loss: " + init.to_s + " → " + final.to_s

if final.to_s == "NaN"
  puts "VERDICT: FAIL (NaN)"; exit 1
end
if final >= init
  puts "VERDICT: FAIL (no convergence: " + init.to_s + " → " + final.to_s + ")"; exit 1
end
ratio = final / init
puts "ratio = " + ratio.to_s
if ratio >= 0.5
  puts "VERDICT: FAIL (slow convergence; ratio " + ratio.to_s + ")"; exit 1
end
puts "VERDICT: PASS (seq-mode CUDA training: " + init.to_s + " → " + final.to_s + ")"
