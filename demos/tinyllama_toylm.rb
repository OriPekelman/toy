require_relative "../lib/arch"
require_relative "../lib/transformer_lm"

GGUF = "data/tinyllama-1.1b-f32.gguf"
arch = Arch.from_gguf(GGUF)
puts arch.summary
lm = ToyLM.new(arch, :cpu)
lm.load(GGUF)
ids = lm.generate([450, 7483, 310, 3444, 338], 8)
print "generated ids:"
k = 0
while k < ids.length
  print " " + ids[k].to_s
  k = k + 1
end
puts ""
