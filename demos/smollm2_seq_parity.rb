# demos/smollm2_seq_parity.rb — M3 step 1 acceptance.
#
# Single-token parity: Toy::LLM::Engine::LlamaSeqEngine#forward([id], [0]) should
# produce the same logits as SmolLM2KVFFICache + SmolLM2KV.decode_step(id, 0)
# (both reading the same SmolLM2-135M weights). At T=1 the K/V cache
# stores exactly one position, so the math reduces to:
#
#   scores = matmul(k_pre_rope, q_pre_rope_pos0)  — same shape
#   attn   = softmax(scores / sqrt(d_head))        — 1×1 matrix
#   head   = matmul(V_t, attn)                     — d_head × 1
#
# The seq-mode graph uses fresh K/V (no cache); decode_step uses the
# KV-cache. With a fresh cache at pos=0 they're identical bytes.
#
# Expected: max_abs_diff < 1e-3 on f32 SmolLM2 weights at d_model=576.
# (Float32 + ggml's flash-attn-style ordering can drift up to ~1e-4
# per matmul; we propagate through 30 layers so allow a few orders of
# magnitude headroom.)
#
# Two GGUF mmaps are loaded — one per cache. Each session keeps its
# own gguf_handle alive. Same file, so no extra disk reads.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv"
require_relative "../lib/toy/llm/engine/llama_seq_engine"

GGUF = ENV["GGUF"] || "data/smollm2-135m-native.gguf"

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
puts "config: vocab=" + cfg.vocab.to_s +
     " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s +
     " n_q=" + cfg.n_heads.to_s +
     " n_kv=" + cfg.n_kv.to_s
puts ""

# === KV-cache decode (reference) =========================================
puts "realizing KV cache..."
kv_gguf = TinyNN.tnn_gguf_load(GGUF)
kv = SmolLM2KVFFICache.new
kv.realize_for_mmap(kv_gguf, cfg, 8, flags.untied, flags.qkv_bias)

# Token id at pos 0 — any in-vocab id will do; pick a small one.
TOKEN_ID = 12092

puts "decode_step at pos=0..."
ref_logits = SmolLM2KV.decode_step(kv, TOKEN_ID, 0)
puts "  ref_logits shape: 1 x " + cfg.vocab.to_s

# === Sequence-mode forward (under test) ==================================
puts ""
puts "realizing sequence-mode cache at T=1..."
seq_gguf = TinyNN.tnn_gguf_load(GGUF)
seq = Toy::LLM::Engine::LlamaSeqEngine.new
seq.realize_for_mmap(seq_gguf, cfg, 1, flags.untied, flags.qkv_bias)

puts "forward([TOKEN], [0])..."
t_logits = seq.forward([TOKEN_ID], [0])
seq_logits = TinyNN.download_row_major(seq.sess, t_logits, 1, cfg.vocab)
puts "  seq_logits shape: 1 x " + cfg.vocab.to_s

# === Compare ============================================================
puts ""
puts "comparing logits..."
max_abs_diff = 0.0
max_idx = 0
sum_sq_diff = 0.0
ref_argmax = 0
seq_argmax = 0
ref_max = ref_logits.flat[0]
seq_max = seq_logits.flat[0]
i = 0
while i < cfg.vocab
  r = ref_logits.flat[i]
  s = seq_logits.flat[i]
  d = (r - s).abs
  if d > max_abs_diff
    max_abs_diff = d
    max_idx = i
  end
  sum_sq_diff = sum_sq_diff + d * d
  if r > ref_max; ref_max = r; ref_argmax = i; end
  if s > seq_max; seq_max = s; seq_argmax = i; end
  i = i + 1
end
rms_diff = Math.sqrt(sum_sq_diff / cfg.vocab.to_f)

puts "  max_abs_diff = " + max_abs_diff.to_s + " (at idx " + max_idx.to_s + ")"
puts "  rms_diff     = " + rms_diff.to_s
puts "  ref argmax   = " + ref_argmax.to_s + " (logit " + ref_max.to_s + ")"
puts "  seq argmax   = " + seq_argmax.to_s + " (logit " + seq_max.to_s + ")"

# Acceptance — argmax must match, max_abs_diff must be small relative
# to the typical logit magnitude.
if ref_argmax != seq_argmax
  puts "VERDICT: FAIL (argmax mismatch: ref " + ref_argmax.to_s +
       " vs seq " + seq_argmax.to_s + ")"; exit 1
end
if max_abs_diff > 1.0e-3
  puts "VERDICT: FAIL (max_abs_diff " + max_abs_diff.to_s + " > 1e-3)"; exit 1
end
puts "VERDICT: PASS (T=1 parity: max_abs_diff " + max_abs_diff.to_s +
     ", argmax " + ref_argmax.to_s + ")"
