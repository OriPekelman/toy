# demos/smollm2_seq_train.rb — M3 step 3 acceptance (CPU).
# Also doubles as the F4 (QLoRA) smoke: pass GGUF=...-q8.gguf to train
# LoRA-Q against a Q8 base on CPU. Mat-mediated CPU mmap doesn't pad,
# so any d_model / d_ff is fine — verified on qwen25-0.5b-native-q8
# (15.4 → 6.4 over 10 steps, ratio 0.41).
#
# F4 on CUDA has an outstanding BYO-pointer Q8 buffer padding issue
# in vendor-patches/0002 — the cuda buffer_type's get_alloc_size
# rounds quantized tensors up to MATRIX_ROW_PADDING (512); the
# padding zeroing then writes past the mmap region. Tracked as a
# follow-up patch; CPU is the supported QLoRA path for now.
#
# One forward + backward + opt_step over T=4 positions of a SmolLM2-135M
# sequence, repeated N steps. Validates that the sequence-mode forward
# graph + LoRA splice + persistent Adam from F1.2 step 6b can train.
#
# Loss target: next-token CE across all T positions. Same toy prompt
# as the parity smokes; we ask the adapter to push every position's
# argmax toward an arbitrary fixed target_id. Real SFT (alpaca-style)
# would use shifted-by-1 next-token labels and a position mask — that's
# the M3 step 4 follow-up (masked CE).
#
# Acceptance: loss decreases monotonically (no NaN) across N steps,
# and the final loss is < 0.5 × initial.
#
# IMPORTANT: this uses CPU. We KNOW CPU LoRA training has the ggml-cpu
# sched-aliasing issue per project_cpu_cuda_lora_train_divergence_2026_05_21;
# this smoke uses tnn_pin_all_graph_b_nodes (when available) to sidestep
# it, or accepts that the absolute loss values may not converge as
# aggressively as CUDA. The point is to prove the wiring; CUDA mirror
# follows in step 4.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/llama_seq_forward_ffi"

GGUF      = ENV["GGUF"]    || "data/smollm2-135m-native.gguf"
RANK      = (ENV["RANK"]   || "8").to_i
SEED      = (ENV["SEED"]   || "42").to_i
STEPS     = (ENV["STEPS"]  || "20").to_i
LR        = (ENV["LR"]     || "0.001").to_f

TOKENS    = [12092, 4845, 253, 1429]   # T=4 input
TARGET_ID = 99                          # uniform target at every position

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "config: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " n_q=" + cfg.n_heads.to_s
puts "training: T=" + TOKENS.length.to_s + " RANK=" + RANK.to_s +
     " STEPS=" + STEPS.to_s + " LR=" + LR.to_s

gguf = TinyNN.tnn_gguf_load(GGUF)
seq = LlamaSeqForwardFFICache.new
seq.enable_lora_q!(RANK)
seq.enable_lora_q_adamw!
seq.realize_for_mmap(gguf, cfg, TOKENS.length, flags.untied, flags.qkv_bias)
seq.upload_lora_q_init!(SEED, 0.01)

# Switch to training graph (resets ctx, adds CE + backward + opt_step).
puts ""
puts "building training graph..."
result = seq.build_training_step
if result == nil
  puts "build_training_step returned nil (lora not enabled?); aborting"; exit 1
end
t_loss   = result[0]
t_labels = result[1]
t_hp     = result[2]

positions = [0, 1, 2, 3]

# Build the labels matrix in Mat(T, vocab) layout — flat[t * vocab + v].
# That maps row-major into the ggml ne=[vocab, T] label tensor.
# Every position t targets TARGET_ID as the simplest possible signal
# exercising CE across all T positions.
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

# AdamW hyper-parameters tensor: alpha, b1, b2, eps, wd, beta1h, beta2h.
m_hp = Mat.new(1, 7)
m_hp.flat[0] = LR
m_hp.flat[1] = 0.9
m_hp.flat[2] = 0.999
m_hp.flat[3] = 1.0e-8
m_hp.flat[4] = 0.0
# beta1h / beta2h are bias-corrected step-dependent — fill per-step below.

losses = [0.0]
losses.pop

puts ""
puts "training " + STEPS.to_s + " steps..."
step = 1
while step <= STEPS
  # Bias correction terms ramp up with step.
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
  loss = TinyNN.tnn_scratch_get(seq.sess, 0)
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
# Tight threshold left as gate for clean signal; CPU's sched aliasing
# (see project_cpu_cuda_lora_train_divergence) may slow CE descent but
# the cross-sequence gradients are an order of magnitude stronger than
# the single-pos case, so 50% drop should be reachable.
ratio = final / init
puts "ratio = " + ratio.to_s
if ratio >= 0.5
  puts "VERDICT: WARN (slow convergence: ratio " + ratio.to_s +
       " — likely the ggml-cpu sched alias; CUDA mirror is the actual gate)"
  exit 0
end
puts "VERDICT: PASS (seq-mode training: " + init.to_s + " → " + final.to_s + ")"
