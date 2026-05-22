# T1 experimentation smoke: does lib/tokenizer.rb round-trip realistic
# English on SmolLM2, Llama-3, and Qwen2.5 vocabs? Spinel-compiled
# (FFI/GGUF bindings).
#
# For each model:
#   - Load the *-tok.gguf (embedded vocab + merges).
#   - Encode a short prompt, decode it back, compare byte-for-byte.
#   - Encode a few representative English samples (chat-style prefix,
#     punctuation-heavy sentence) and verify they all round-trip.
#
# Bail-loud: any divergence prints first-byte diff. A perfect round-trip
# across all three families is the v1 "tokenizer works" signal — the
# next step is wiring it into example_inference + the OpenAI server.

require_relative "../lib/tokenizer"
require_relative "../lib/gguf_kv"

# Test corpus: short English samples that exercise common edge cases.
SAMPLES = ["Once upon a time",
           "The quick brown fox jumps over the lazy dog.",
           "Hello, my name is",
           "Q: What is 2 + 2?\nA: ",
           "I/O bound: read('foo.txt')"]

MODELS = ["data/smollm2-135m-tok.gguf",
          "data/llama-3.2-1b-tok.gguf",
          "data/qwen25-0.5b-tok.gguf"]

total_pass = 0
total_fail = 0

mi = 0
while mi < MODELS.length
  path = MODELS[mi]
  puts "===================================================================="
  puts "model: " + path
  tok = Tokenizer.from_gguf(path)
  if !tok.present
    puts "  SKIP: no vocab embedded"
    mi = mi + 1
    next
  end
  puts "  vocab_size=" + tok.vocab_size.to_s +
       " bos=" + tok.bos_id.to_s +
       " eos=" + tok.eos_id.to_s

  si = 0
  while si < SAMPLES.length
    text = SAMPLES[si]
    ids  = tok.encode(text)
    dec  = tok.decode(ids)
    ok   = (dec == text)
    if ok
      total_pass = total_pass + 1
      puts "  [" + si.to_s + "] PASS  ids=" + ids.length.to_s + " — " + text.inspect
    else
      total_fail = total_fail + 1
      puts "  [" + si.to_s + "] FAIL"
      puts "      source:  " + text.inspect
      puts "      decoded: " + dec.inspect
      puts "      ids:     " + ids.inspect
    end
    si = si + 1
  end

  mi = mi + 1
end

puts ""
puts "summary: " + total_pass.to_s + " pass, " + total_fail.to_s + " fail"

# SmolLM2 deep-dive: per-character encode of the failing string.
# If "?" or "\n" or "/" or "." or "'" produces unk_id, the vocab is
# missing that single-char token. Compare to Llama-3 for the same chars.
puts ""
puts "=== single-char probe ==="
probe = ["?", "\n", "/", "'", ".", "!", "@", "#",
         "?\n", "I/O", ".txt", "('", "2?", "Q:", "is 2"]

# Probe vocab around id 0 to see if SmolLM2's "UNK" is actually a
# real token. If vocab[0] is "?Ċ" then "UNK" isn't UNK at all — the
# encoder just happens to emit id 0 when lookup fails AND id 0
# corresponds to a real token in vocab.
mi2 = 0
while mi2 < MODELS.length
  t2 = Tokenizer.from_gguf(MODELS[mi2])
  if t2.present
    puts MODELS[mi2] + " vocab[0..4] = " +
         [t2.token_at(0), t2.token_at(1), t2.token_at(2), t2.token_at(3), t2.token_at(4)].inspect
  end
  mi2 = mi2 + 1
end
mi = 0
while mi < MODELS.length
  tok = Tokenizer.from_gguf(MODELS[mi])
  puts MODELS[mi]
  h = GgufKV.tnn_gguf_load(MODELS[mi])
  if h != nil
    pre = GgufKV.tnn_gguf_get_str(h, "tokenizer.ggml.pre")
    mdl = GgufKV.tnn_gguf_get_str(h, "tokenizer.ggml.model")
    puts "  pre=" + pre.to_s + " model=" + mdl.to_s
    GgufKV.tnn_gguf_free(h)
  end
  if tok.present
    pi = 0
    while pi < probe.length
      ids_p = tok.encode(probe[pi])
      dec_p = tok.decode(ids_p)
      tag = "BAD"
      if dec_p == probe[pi]
        tag = "OK "
      end
      puts "  " + tag + " " + probe[pi].inspect + " -> " + ids_p.inspect + " -> " + dec_p.inspect
      pi = pi + 1
    end
  end
  mi = mi + 1
end

exit (total_fail == 0 ? 0 : 1)
