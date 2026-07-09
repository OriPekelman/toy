# Smoke test for lib/bpe.rb. Tokenizes a small set of fixed prompts and
# prints the resulting IDs. Compare with the Python tokenizer's output
# (run prep/tokens.py encode <prompt> on the same string).
#
# Expected outputs (from HF tokenizers):
#   "Hello, my name is"              → 15496 11 616 1438 318
#   "Hello"                          → 15496
#   "The quick brown fox"            → 464 2068 7586 21831
#   "It's a test."                   → 1026 338 257 1332 13

require_relative "../lib/toy/models/transformer"   # for Mat (unused but pulls in deps)
require_relative "../lib/toy/io/bpe"

puts "loading BPE tables..."
t0 = Time.now
tables = GPT2BPE.load("data")
puts "  loaded in " + ((Time.now - t0) * 1000).to_s + " ms" +
     "  (vocab=" + tables.vocab_id.length.to_s + ", merges=" +
     tables.merge_rank.length.to_s + ")"
puts ""

# Helper: encode + decode + print.
def smoke(label, text, tables)
  ids = GPT2BPE.encode(text, tables)
  back = GPT2BPE.decode(ids, tables)
  puts label + ":"
  puts "  text:    " + text.inspect
  out = "  ids:     ["
  i = 0
  while i < ids.length
    out = out + ids[i].to_s
    if i < ids.length - 1
      out = out + ", "
    end
    i = i + 1
  end
  out = out + "]"
  puts out
  puts "  roundtrip: " + back.inspect + "  (matches: " + (back == text).to_s + ")"
end

smoke("hello", "Hello, my name is", tables)
smoke("single", "Hello", tables)
smoke("quick", "The quick brown fox", tables)
smoke("contraction", "It's a test.", tables)
smoke("number", "Year 2024 was interesting", tables)
smoke("punct", "Wait... really?", tables)
