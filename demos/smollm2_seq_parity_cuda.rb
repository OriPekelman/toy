# demos/smollm2_seq_parity_cuda.rb — M3 step 2 (CUDA).
#
# Mirror of demos/smollm2_seq_parity.rb but for CUDA. Validates that
# the sequence-mode forward graph on GPU matches a fresh CPU KV-decode
# at T=1 within the usual CUDA/CPU numerical tolerance (~1e-3, looser
# than CPU-vs-CPU because the kernels use slightly different reductions).

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/models/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv"
require_relative "../lib/toy/llm/engine/llama_seq_engine_cuda"

GGUF = ENV["GGUF"] || "data/smollm2-135m-native.gguf"

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "config: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s + " n_q=" + cfg.n_heads.to_s

TOKEN_ID = 12092

# CPU reference (KV-decode).
puts ""
puts "realizing CPU KV cache..."
kv_gguf = TinyNN.tnn_gguf_load(GGUF)
kv = SmolLM2KVFFICache.new
kv.realize_for_mmap(kv_gguf, cfg, 8, flags.untied, flags.qkv_bias)
puts "CPU decode_step at pos=0..."
ref_logits = SmolLM2KV.decode_step(kv, TOKEN_ID, 0)

# CUDA sequence-mode forward.
puts ""
puts "realizing CUDA seq cache at T=1..."
seq_gguf = TinyNNCuda.tnn_gguf_load(GGUF)
seq = Toy::LLM::Engine::LlamaSeqEngineCuda.new
seq.realize_for_mmap(seq_gguf, cfg, 1, flags.untied, flags.qkv_bias)
puts "CUDA forward..."
t_logits = seq.forward([TOKEN_ID], [0])
seq_logits = TinyNNCuda.download_row_major(seq.sess, t_logits, 1, cfg.vocab)

# Compare.
max_abs_diff = 0.0
ref_argmax = 0
seq_argmax = 0
ref_max = ref_logits.flat[0]
seq_max = seq_logits.flat[0]
i = 0
while i < cfg.vocab
  r = ref_logits.flat[i]
  s = seq_logits.flat[i]
  d = (r - s).abs
  if d > max_abs_diff; max_abs_diff = d; end
  if r > ref_max; ref_max = r; ref_argmax = i; end
  if s > seq_max; seq_max = s; seq_argmax = i; end
  i = i + 1
end
puts ""
puts "max_abs_diff = " + max_abs_diff.to_s
puts "ref_argmax   = " + ref_argmax.to_s + " (logit " + ref_max.to_s + ")"
puts "seq_argmax   = " + seq_argmax.to_s + " (logit " + seq_max.to_s + ")"

if ref_argmax != seq_argmax
  puts "VERDICT: FAIL (argmax mismatch)"; exit 1
end
if max_abs_diff > 1.0e-3
  puts "VERDICT: FAIL (max_abs_diff " + max_abs_diff.to_s + " > 1e-3)"; exit 1
end
puts "VERDICT: PASS (CUDA seq T=1 parity vs CPU decode: " + max_abs_diff.to_s + ")"
