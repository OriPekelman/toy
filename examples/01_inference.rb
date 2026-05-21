# Run an open-weight model. Loads a Qwen2.5 GGUF, decodes 16 tokens
# greedily from the prompt, prints the token IDs. Decoding text from
# IDs is a separate concern (see examples/README — server-side
# tokenizer is opt-in via lib/tokenizer.rb).
#
#   make example_inference
#   GGUF=data/qwen25-0.5b-native.gguf ./examples/example_inference
#
# Swap GGUF to any supported model (see README). On CUDA, change
# `:cpu` → `:cuda` and build with `make example_inference_cuda` (the
# only difference is the FFI backend module).

require_relative "../lib/arch"
require_relative "../lib/transformer_lm"

GGUF  = ENV["GGUF"]  || "data/qwen25-0.5b-native.gguf"
N_NEW = (ENV["N_NEW"] || "16").to_i

arch = Arch.from_gguf(GGUF)
puts arch.summary

lm = ToyLM.new(arch, :cpu)
lm.load(GGUF)

# 5-token prompt: "Hello, my name is" in the Qwen2 vocab.
ids = lm.generate([9707, 11, 847, 829, 374], N_NEW)

print "ids:"
k = 0
while k < ids.length
  print " " + ids[k].to_s
  k = k + 1
end
puts ""
