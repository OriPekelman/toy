# Phase 0.5 smoke: Qwen2.5 inference via the generic TransformerLM
# delegation layer. Expect bit-identical output vs demos/qwen25_native_mmap.

require_relative "../lib/arch"
require_relative "../lib/transformer_lm"

GGUF  = ENV["GGUF"]  || "data/qwen25-1.5b-native.gguf"
N_NEW = (ENV["N_NEW"] || "8").to_i

arch = Arch.from_gguf(GGUF)
puts arch.summary

lm = TransformerLM.new(arch, :cpu)
lm.load(GGUF)

ids = lm.generate([9707, 11, 847, 829, 374], N_NEW)

print "generated ids:"
k = 0
while k < ids.length
  print " " + ids[k].to_s
  k = k + 1
end
puts ""
