# Reads general.architecture + key tokenizer metadata from a GGUF.
# Path is hardcoded; flip GGUF_PATH to inspect different models.

require_relative "../lib/transformer"
require_relative "../lib/tinynn"

GGUF_PATH = ENV["GGUF"] || "data/qwen25-1.5b-native.gguf"

h = TinyNN.tnn_gguf_load(GGUF_PATH)
if h == nil
  puts "open failed: " + GGUF_PATH
  exit 1
end

puts "file: " + GGUF_PATH
puts ""

# Architecture identity
arch = TinyNN.tnn_gguf_get_str(h, "general.architecture")
name = TinyNN.tnn_gguf_get_str(h, "general.name")
puts "general.architecture = " + (arch.nil? ? "(nil)" : arch.to_s)
puts "general.name         = " + (name.nil? ? "(nil)" : name.to_s)
puts ""

# Try a few model-family-specific keys; report which exist
puts "scalar keys:"
keys = ["llama.block_count", "llama.embedding_length",
        "llama.attention.head_count", "llama.attention.head_count_kv",
        "llama.attention.layer_norm_rms_epsilon", "llama.rope.freq_base",
        "qwen2.block_count", "qwen2.embedding_length",
        "qwen2.attention.head_count", "qwen2.attention.head_count_kv",
        "qwen3.block_count", "llama.feed_forward_length"]
i = 0
while i < keys.length
  k  = keys[i]
  vu = TinyNN.tnn_gguf_get_u32(h, k)
  vf = TinyNN.tnn_gguf_get_f32(h, k)
  if vu >= 0
    puts "  " + k + " = " + vu.to_s + " (u32)"
  end
  if vf > 0.0
    puts "  " + k + " = " + vf.to_s + " (f32)"
  end
  i = i + 1
end
puts ""

# Tokenizer arrays
puts "tokenizer:"
vocab_n   = TinyNN.tnn_gguf_arr_n(h, "tokenizer.ggml.tokens")
merges_n  = TinyNN.tnn_gguf_arr_n(h, "tokenizer.ggml.merges")
bos       = TinyNN.tnn_gguf_get_u32(h, "tokenizer.ggml.bos_token_id")
eos       = TinyNN.tnn_gguf_get_u32(h, "tokenizer.ggml.eos_token_id")
pad       = TinyNN.tnn_gguf_get_u32(h, "tokenizer.ggml.padding_token_id")
puts "  tokens.length  = " + vocab_n.to_s
puts "  merges.length  = " + merges_n.to_s
puts "  bos_token_id   = " + bos.to_s
puts "  eos_token_id   = " + eos.to_s
puts "  pad_token_id   = " + pad.to_s
if vocab_n > 0
  puts "  tokens[0]      = " + TinyNN.tnn_gguf_arr_str(h, "tokenizer.ggml.tokens", 0).to_s
  puts "  tokens[1]      = " + TinyNN.tnn_gguf_arr_str(h, "tokenizer.ggml.tokens", 1).to_s
end

TinyNN.tnn_gguf_free(h)
