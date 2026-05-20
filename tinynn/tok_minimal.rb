require_relative "../lib/tokenizer"
puts "loading..."
$stdout.flush
t = Tokenizer.from_gguf("data/smollm2-135m-tok.gguf")
puts "vocab=" + t.vocab_size.to_s
