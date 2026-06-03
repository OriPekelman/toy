require_relative "../lib/toy/models/arch"
require_relative "../lib/transformer_lm"
GGUF = "data/smollm2-135m-f32.gguf"
arch = Arch.from_gguf(GGUF)
puts arch.summary
lm = ToyLM.new(arch, :cpu)
lm.load(GGUF)
ids = lm.generate([12092, 4845, 253, 1429], 12)
print "generated ids:"
k = 0
while k < ids.length
  print " " + ids[k].to_s
  k = k + 1
end
puts ""
