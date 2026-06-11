# Bench: LoRA training step wallclock (SmolLM2-135M, r=8, T=4).
# Reuses the exact realize_for_mmap + build_training_step path the
# user-facing example_finetune uses. Reports steady-state ms/step
# averaged over the last N-WARMUP steps; the first step is excluded
# because graph realize/finalize dominates and is one-shot.
#
# Knobs:
#   STEPS=N   total step count (default 10).
#   BATCH=N   GH#7 micro-batching. BATCH=1 (default) is the legacy
#             single-sequence path. BATCH>1 lays N copies of TOKENS
#             side-by-side with a block-causal mask. Reported numbers
#             are still ms/step (not ms/sample) so a B>1 column tells
#             you how step time scales with effective batch size.
#
# Output format (last lines, parsed by bench/check.rb):
#   BENCH lora_step_ms <float>
#   BENCH lora_steady_state_ms <float>

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../lib/toy/llm/engine/llama_seq_engine"

GGUF   = ENV["GGUF"]   || "data/smollm2-135m-native.gguf"
STEPS  = (ENV["STEPS"] || "10").to_i
BATCH  = (ENV["BATCH"] || "1").to_i
WARMUP = 2

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
gguf  = TinyNN.tnn_gguf_load(GGUF)

TOKENS_PER_SEQ = [12092, 4845, 253, 1429]
T_SEQ = TOKENS_PER_SEQ.length

# Build flat [T*B] token + position arrays. Each batch element holds
# the same TOKENS payload (a synthetic bench input — content doesn't
# matter, only the shape does).
tokens_flat = [0]; tokens_flat.pop
positions   = [0]; positions.pop
b_idx = 0
while b_idx < BATCH
  ki = 0
  while ki < T_SEQ
    tokens_flat.push(TOKENS_PER_SEQ[ki])
    positions.push(ki)
    ki = ki + 1
  end
  b_idx = b_idx + 1
end

seq = Toy::LLM::Engine::LlamaSeqEngine.new
seq.enable_lora_q!(8)
seq.enable_lora_q_adamw!
# GH#7 — caller opts in to micro-batching via attr_accessor BEFORE
# realize. realize_for_mmap reads @seq_b, allocates + uploads the
# block-causal mask when > 1.
seq.seq_b = BATCH
seq.realize_for_mmap(gguf, cfg, T_SEQ, flags.untied, flags.qkv_bias)
seq.upload_lora_q_init!(42, 0.01)
result   = seq.build_training_step
t_loss   = result[0]
t_labels = result[1]
t_hp     = result[2]

# Label matrix is (T*B) rows × vocab cols, one-hot target token 99
# per row (synthetic bench input).
m_labels = Mat.new(T_SEQ * BATCH, cfg.vocab)
i = 0
while i < T_SEQ * BATCH * cfg.vocab; m_labels.flat[i] = 0.0; i = i + 1; end
ti = 0
while ti < T_SEQ * BATCH
  m_labels.flat[ti * cfg.vocab + 99] = 1.0
  ti = ti + 1
end

m_hp = Mat.new(1, 7)
m_hp.flat[0] = 0.001; m_hp.flat[1] = 0.9; m_hp.flat[2] = 0.999
m_hp.flat[3] = 1.0e-8; m_hp.flat[4] = 0.0

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
  TinyNN.upload_int_array(seq.sess, seq.t_seq_token_ids, tokens_flat)
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

# Emit batch-suffixed metrics so multiple BATCH runs can coexist in
# the bench harness (bench/check.rb merges all observed metrics into
# one map — duplicate keys collide). At BATCH=1 we ALSO emit the
# legacy unsuffixed name so the pre-GH#7 baseline row keeps its anchor.
suffix = "_b" + BATCH.to_s
puts "BENCH lora_step" + suffix + "_ms " + mean_all.to_s
puts "BENCH lora_steady_state" + suffix + "_ms " + mean_warm.to_s
if BATCH == 1
  puts "BENCH lora_step_ms " + mean_all.to_s
  puts "BENCH lora_steady_state_ms " + mean_warm.to_s
end
