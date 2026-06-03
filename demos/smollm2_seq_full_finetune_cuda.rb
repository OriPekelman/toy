# demos/smollm2_seq_full_finetune_cuda.rb — F3 acceptance.
#
# Full fine-tune on CUDA: every per-block weight tensor (norms, Q/K/V
# per-head, O, FFN gate/up/down) is trainable, paired with a
# persistent AdamW (m, v) state, and steps via opt_step_adamw in
# every training iteration.
#
# Embedding + final-norm gamma stay frozen (mmap'd) — extending those
# to trainable is a one-line set_param + opt_step per tensor; the
# MVP skips them to keep the param count down.
#
# Acceptance: monotonic CE descent over 20 steps with ratio < 0.5,
# same gate as the LoRA examples; full-FT has more capacity so
# convergence is usually faster at the same LR.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy/llm/engine/llama_seq_engine_cuda"

GGUF      = ENV["GGUF"]    || "data/smollm2-135m-native.gguf"
STEPS     = (ENV["STEPS"]  || "20").to_i
LR        = (ENV["LR"]     || "0.0005").to_f   # smaller than LoRA — more capacity

TOKENS    = [12092, 4845, 253, 1429]
TARGET_ID = (ENV["TARGET_ID"] || "99").to_i

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "config: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " heads=" + cfg.n_heads.to_s
puts "full fine-tune (CUDA): GGUF=" + GGUF + " STEPS=" + STEPS.to_s + " LR=" + LR.to_s

gguf = TinyNNCuda.tnn_gguf_load(GGUF)
seq = Toy::LLM::Engine::LlamaSeqEngineCuda.new
seq.enable_full_finetune!
# Train embeddings when EMBED=1 in env. Works for any vocab size
# after vendor-patches/0006 chunked the get_rows_back kernel launch
# (Qwen-class V=152K previously overran CUDA's gridDim.y limit).
# Off by default since training the embed adds memory pressure.
if (ENV["EMBED"] || "0").to_i == 1
  seq.enable_full_finetune_embeddings!
end
seq.realize_for_full_finetune(gguf, cfg, TOKENS.length, flags.untied, flags.qkv_bias)

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

init = losses[0]
final = losses[losses.length - 1]
puts ""
puts "loss: " + init.to_s + " → " + final.to_s

if final.to_s == "NaN"
  puts "VERDICT: FAIL (NaN — full-FT divergence)"; exit 1
end
if final >= init
  puts "VERDICT: FAIL (no convergence: " + init.to_s + " → " + final.to_s + ")"; exit 1
end
ratio = final / init
puts "ratio = " + ratio.to_s
if ratio >= 0.5
  puts "VERDICT: FAIL (slow: ratio " + ratio.to_s + ")"; exit 1
end
puts "VERDICT: PASS (full fine-tune CUDA: " + init.to_s + " → " + final.to_s + ")"
