# demos/smollm2_seq_parity_t4_cuda.rb — M3 step 2 (CUDA, T=4).
#
# Per-position seq logits from CUDA must match the CPU KV-decode
# trajectory. Looser tolerance than CPU-vs-CPU because CUDA matmul
# uses different reduction kernels.

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv"
require_relative "../lib/toy/llm/engine/llama_seq_engine_cuda"

GGUF = ENV["GGUF"] || "data/smollm2-135m-native.gguf"

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "config: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " n_q=" + cfg.n_heads.to_s

TOKENS = [12092, 4845, 253, 1429]
T_SEQ  = TOKENS.length

# CPU reference trajectory.
puts ""
puts "realizing CPU KV cache..."
kv_gguf = TinyNN.tnn_gguf_load(GGUF)
kv = SmolLM2KVFFICache.new
kv.realize_for_mmap(kv_gguf, cfg, 8, flags.untied, flags.qkv_bias)
puts "decoding " + T_SEQ.to_s + " positions on CPU..."
ref_per_pos = [Mat.new(1, cfg.vocab)]
ref_per_pos.pop
t = 0
while t < T_SEQ
  ref_per_pos.push(SmolLM2KV.decode_step(kv, TOKENS[t], t))
  t = t + 1
end

# CUDA sequence-mode forward.
puts ""
puts "realizing CUDA seq cache at T=" + T_SEQ.to_s + "..."
seq_gguf = TinyNNCuda.tnn_gguf_load(GGUF)
seq = Toy::LLM::Engine::LlamaSeqEngineCuda.new
seq.realize_for_mmap(seq_gguf, cfg, T_SEQ, flags.untied, flags.qkv_bias)
puts "CUDA forward..."
t_logits = seq.forward(TOKENS, [0, 1, 2, 3])
seq_logits_mat = TinyNNCuda.download_row_major(seq.sess, t_logits, T_SEQ, cfg.vocab)

# Per-position compare.
puts ""
puts "per-position parity:"
overall_max = 0.0
any_fail = false
t2 = 0
while t2 < T_SEQ
  ref = ref_per_pos[t2]
  max_abs = 0.0
  ref_argmax = 0
  seq_argmax = 0
  ref_max_l = ref.flat[0]
  seq_max_l = seq_logits_mat.flat[t2 * cfg.vocab]
  i = 0
  while i < cfg.vocab
    r = ref.flat[i]
    s = seq_logits_mat.flat[t2 * cfg.vocab + i]
    d = (r - s).abs
    if d > max_abs; max_abs = d; end
    if r > ref_max_l; ref_max_l = r; ref_argmax = i; end
    if s > seq_max_l; seq_max_l = s; seq_argmax = i; end
    i = i + 1
  end
  if max_abs > overall_max; overall_max = max_abs; end
  argmatch = ref_argmax == seq_argmax
  # CUDA-vs-CPU FP32 drift accumulates ~3-5e-2 over 30 layers with T×T
  # attention (different matmul reduction orders + softmax fast-math).
  # Argmax preservation is the load-bearing acceptance — magnitudes are
  # consultative.
  marker = argmatch && max_abs < 1.0e-1 ? "ok" : "FAIL"
  puts "  pos " + t2.to_s + ": max_abs_diff=" + max_abs.to_s +
       "  ref_argmax=" + ref_argmax.to_s + " seq_argmax=" + seq_argmax.to_s +
       "  [" + marker + "]"
  if !argmatch || max_abs >= 1.0e-1
    any_fail = true
  end
  t2 = t2 + 1
end

puts ""
puts "overall max_abs_diff: " + overall_max.to_s
if any_fail
  puts "VERDICT: FAIL"; exit 1
end
puts "VERDICT: PASS (CUDA seq T=" + T_SEQ.to_s + " trajectory parity: " +
     "max_abs_diff " + overall_max.to_s + ", all argmaxes match CPU)"
