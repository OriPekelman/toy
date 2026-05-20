require_relative "../lib/arch"
require_relative "../lib/transformer_lm"
GGUF = "data/qwen25-0.5b-native.gguf"
arch = Arch.from_gguf(GGUF)
lm = ToyLM.new(arch, :cpu)
lm.load(GGUF)
ids = lm.generate([9707, 11, 847, 829, 374], 8)
