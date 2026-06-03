require_relative "../lib/toy/models/arch"
require_relative "../lib/toy/models/transformer_lm"

GGUF = ENV["GGUF"] || "data/llama-3.2-1b-f32.gguf"

arch = Arch.from_gguf(GGUF)
puts arch.summary

lm = ToyLM.new(arch, :cpu)
lm.load(GGUF)

# Llama-3.2 tokens for "Hello, my name is"
ids = lm.generate([791, 6864, 315, 9822, 374], 12)

print "generated ids:"
k = 0
while k < ids.length
  print " " + ids[k].to_s
  k = k + 1
end
puts ""
