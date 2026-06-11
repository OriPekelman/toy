# demos/smollm2_seq_parity_t4.rb — M3 step 2 acceptance (CPU).
#
# Multi-token parity: Toy::LLM::Engine::LlamaSeqEngine#forward([id_0..id_3], [0..3])
# should produce a (vocab, T=4) logit matrix whose column-t logits
# match SmolLM2KV.decode_step(id_t, t) called sequentially. This proves
# the full T×T causal attention shape — per-position seq logits track
# the KV-decode trajectory exactly.
#
# Failure modes this catches:
#  - diag_mask_inf semantics: if the mask isn't strict causal (j > i),
#    column-t in seq output sees future tokens and diverges.
#  - rope_ext over multi-position: if the per-token position vector
#    isn't honored, all positions get the same rotation and seq drifts.
#  - rms_norm over (d_model, T): if the norm collapses across positions
#    instead of being per-column, every position interacts.
#
# Expected: per-column max_abs_diff < 1e-3. Float32 + matmul kernel
# ordering can drift through 30 layers; same threshold as step 1.

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../lib/toy/llm/engine/llama_kv_engine"
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

# Test prompt: 4 small in-vocab IDs. Specific values don't matter — we
# just need a consistent sequence both paths can compute over.
TOKENS = [12092, 4845, 253, 1429]
T_SEQ  = TOKENS.length

# === KV-cache decode (reference trajectory) ==============================
puts "realizing KV cache (max_T=8)..."
kv_gguf = TinyNN.tnn_gguf_load(GGUF)
kv = SmolLM2KVFFICache.new
kv.realize_for_mmap(kv_gguf, cfg, 8, flags.untied, flags.qkv_bias)

puts "decoding " + T_SEQ.to_s + " positions sequentially..."
ref_per_pos = [Mat.new(1, cfg.vocab)]
ref_per_pos.pop
t = 0
while t < T_SEQ
  logits_t = SmolLM2KV.decode_step(kv, TOKENS[t], t)
  ref_per_pos.push(logits_t)
  t = t + 1
end
puts "  reference trajectory: " + ref_per_pos.length.to_s + " positions × " +
     cfg.vocab.to_s + " vocab"

# === Sequence-mode forward (one compute) =================================
puts ""
puts "realizing seq cache at T=" + T_SEQ.to_s + "..."
seq_gguf = TinyNN.tnn_gguf_load(GGUF)
seq = Toy::LLM::Engine::LlamaSeqEngine.new
seq.realize_for_mmap(seq_gguf, cfg, T_SEQ, flags.untied, flags.qkv_bias)

positions = [0, 1, 2, 3]
puts "forward([" + TOKENS.join(", ") + "], [" + positions.join(", ") + "])..."
t_logits = seq.forward(TOKENS, positions)
# Logits ne=[vocab, T] in ggml (vocab is ne0, T is ne1). download_row_major
# reads as Mat shape (T, vocab) — row t is the logits at position t.
seq_logits_mat = TinyNN.download_row_major(seq.sess, t_logits, T_SEQ, cfg.vocab)
puts "  seq logits: " + T_SEQ.to_s + " × " + cfg.vocab.to_s

# === Per-position comparison ============================================
puts ""
puts "per-position parity:"
overall_max = 0.0
any_fail = false
t2 = 0
while t2 < T_SEQ
  ref = ref_per_pos[t2]
  max_abs = 0.0
  max_idx = 0
  ref_argmax = 0
  seq_argmax = 0
  ref_max_logit = ref.flat[0]
  seq_max_logit = seq_logits_mat.flat[t2 * cfg.vocab]
  i = 0
  while i < cfg.vocab
    r = ref.flat[i]
    s = seq_logits_mat.flat[t2 * cfg.vocab + i]
    d = (r - s).abs
    if d > max_abs
      max_abs = d
      max_idx = i
    end
    if r > ref_max_logit; ref_max_logit = r; ref_argmax = i; end
    if s > seq_max_logit; seq_max_logit = s; seq_argmax = i; end
    i = i + 1
  end
  if max_abs > overall_max; overall_max = max_abs; end
  argmatch = ref_argmax == seq_argmax
  marker = argmatch && max_abs < 1.0e-3 ? "ok" : "FAIL"
  puts "  pos " + t2.to_s + ": max_abs_diff=" + max_abs.to_s +
       " (idx " + max_idx.to_s + ")  ref_argmax=" + ref_argmax.to_s +
       " seq_argmax=" + seq_argmax.to_s + "  [" + marker + "]"
  if !argmatch || max_abs >= 1.0e-3
    any_fail = true
  end
  t2 = t2 + 1
end

puts ""
puts "overall max_abs_diff across all positions: " + overall_max.to_s
if any_fail
  puts "VERDICT: FAIL (per-position trajectory mismatch)"; exit 1
end
puts "VERDICT: PASS (T=" + T_SEQ.to_s +
     " trajectory parity: overall max_abs_diff " + overall_max.to_s + ")"
