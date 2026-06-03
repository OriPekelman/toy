# Bench: tokenizer encode throughput.
# Encodes a fixed corpus (the Sherlock Holmes paragraph below — 712
# bytes of English, no special chars) 100× and reports μs/token and
# tokens/sec. Roundtrip-checks the result so a regression that breaks
# correctness shows up as RUN_ERROR rather than a fake speedup.
#
# Output format:
#   BENCH tokenizer_encode_us_per_tok <float>
#   BENCH tokenizer_encode_toks_per_sec <float>

require_relative "../lib/toy/io/tokenizer"

GGUF   = ENV["GGUF"]  || "data/smollm2-135m-tok.gguf"
ITERS  = (ENV["ITERS"] || "2000").to_i

CORPUS = "Once upon a time there was a quick brown fox who jumped over " +
         "the lazy dog. He was a clever fox, and he knew that the dog " +
         "was watching him. So he ran fast, dodging around the apple " +
         "trees and through the bushes. The dog barked loudly but did " +
         "not give chase. Eventually the fox reached the river, where " +
         "he sat down to catch his breath. The water sparkled in the " +
         "afternoon sun, and the fox felt grateful to be alive. He " +
         "thought of his family back home, and of all the adventures " +
         "he had ahead of him."

tok = Tokenizer.from_gguf(GGUF)
if !tok.present
  puts "BENCH tokenizer_encode_us_per_tok 0"
  puts "BENCH tokenizer_encode_toks_per_sec 0"
  exit 1
end

# Warm-up + roundtrip correctness check.
ids = tok.encode(CORPUS)
dec = tok.decode(ids)
if dec != CORPUS
  puts "RUN_ERROR: encode-decode roundtrip mismatch"
  exit 1
end
n_toks_per_iter = ids.length

t0 = Time.now
i = 0
while i < ITERS
  ids = tok.encode(CORPUS)
  i = i + 1
end
elapsed = (Time.now - t0).to_f

total_toks = n_toks_per_iter * ITERS
us_per_tok = (elapsed * 1.0e6) / total_toks.to_f
toks_per_sec = total_toks.to_f / elapsed
puts "BENCH tokenizer_encode_us_per_tok " + us_per_tok.to_s
puts "BENCH tokenizer_encode_toks_per_sec " + toks_per_sec.to_s
