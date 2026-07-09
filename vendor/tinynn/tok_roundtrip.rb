require_relative "../lib/toy/io/tokenizer"
t = Tokenizer.from_gguf("data/smollm2-135m-tok.gguf")
puts "vocab=" + t.vocab_size.to_s + " merges=" + t.vocab_size.to_s

samples = [
  "Hello world.",
  "Once upon a time, there was a little girl.",
  "The quick brown fox jumps over the lazy dog.",
  "I have 3 apples and 7 oranges.",
  "It's 2026-05-21 now.",
]

i = 0
while i < samples.length
  s = samples[i]
  ids = t.encode(s)
  back = t.decode(ids)
  ok = (back == s)
  puts ""
  puts "in:    " + s.inspect
  puts "  ids: " + ids.inspect
  puts "  out: " + back.inspect
  puts "  round-trip OK: " + ok.to_s
  i = i + 1
end
