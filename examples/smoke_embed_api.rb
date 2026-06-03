# toy#embed-api (#145) — smoke for ToyLM#embed_lookup. Loads a model,
# fetches embedding vectors for a few token IDs, prints shapes + a
# couple of values. Tep's future /v1/embeddings consumes this primitive
# (then mean-pools or returns per-token).
#
#   make examples/smoke_embed_api
#   GGUF=data/smollm2-135m-native.gguf ./examples/smoke_embed_api
#
# Default path uses a from-scratch toy ckpt because that's the cheapest
# end-to-end smoke; vocab is small (627) and the embedding rows are f32.
# For a quantized-table test, set GGUF=data/qwen25-0.5b-native-q8.gguf.

require_relative "../lib/toy/models/arch"
require_relative "../lib/transformer_lm"

GGUF = ENV["GGUF"] || "data/llama-3.2-1b-native.gguf"

arch = Arch.from_gguf(GGUF)
if arch == nil
  puts "smoke_embed_api: could not load " + GGUF
  exit 1
end
puts arch.summary

lm = ToyLM.new(arch, :cpu)
lm.load(GGUF)

# Lookup 4 token IDs (under-min vocab to avoid OOB). Print the first
# 4 floats of each row so we can eyeball that they look different
# (i.e., not all zeros or NaN) and that the dequantize-aware path
# returned bounded values.
ids = [0, 1, 2, 5]
flat = lm.embed_lookup(ids)
d_model = arch.d_model
puts "got " + flat.length.to_s + " floats (= " + ids.length.to_s + " × " +
     d_model.to_s + ")"

i = 0
while i < ids.length
  base = i * d_model
  puts "token " + ids[i].to_s + ": [" +
       flat[base].to_s + ", " + flat[base + 1].to_s + ", " +
       flat[base + 2].to_s + ", " + flat[base + 3].to_s + ", ...]"
  i = i + 1
end
puts "embed_lookup OK"
