require_relative "../lib/toy/models/arch"
require_relative "../lib/transformer_lm"

GGUF = ENV["GGUF"] || "data/mistral-7b-instruct-v0.2.gguf"
arch = Arch.from_gguf(GGUF)
puts arch.summary
lm = ToyLM.new(arch, :cpu)
lm.load(GGUF)
# Mistral prompt IDs filled in below
ids = lm.generate([415, 5565, 302, 4843, 349], 12)
print "generated ids:"
k = 0
while k < ids.length
  print " " + ids[k].to_s
  k = k + 1
end
puts ""
