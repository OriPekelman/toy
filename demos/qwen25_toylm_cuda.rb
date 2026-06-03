# ToyLMCuda smoke. Mirror of demos/qwen25_transformer_lm.rb.

require_relative "../lib/toy/models/arch"
require_relative "../lib/toy/models/transformer_lm_cuda"

GGUF  = ENV["GGUF"]  || "data/qwen25-1.5b-native.gguf"
N_NEW = (ENV["N_NEW"] || "8").to_i

arch = Arch.from_gguf(GGUF)
puts arch.summary

lm = ToyLMCuda.new(arch)
lm.load(GGUF)

ids = lm.generate([9707, 11, 847, 829, 374], N_NEW)

print "generated ids:"
k = 0
while k < ids.length
  print " " + ids[k].to_s
  k = k + 1
end
puts ""
