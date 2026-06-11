# demos/algorithm_cards.rb — print the Phuong–Hutter algorithm cards
# for the loaded models. Doesn't run inference; just emits the cards.
#
# Useful for seeing the "shape of the problem" without burning a forward.

require_relative "../lib/toy"
require_relative "../lib/toy/models/toy_gpt2"
require_relative "../lib/toy/models/toy_smollm2"
require_relative "../lib/toy/io/loaders/toy_gpt2_loader"
require_relative "../lib/toy/io/loaders/toy_smollm2_loader"

GPT2_GGUF    = "data/distilgpt2-f32.gguf"
SMOLLM2_GGUF = "data/smollm2-135m-f32.gguf"

puts "═══════════════════════════════════════════════════════════════════════════"
puts " Toy::GPT2 — algorithm cards"
puts "═══════════════════════════════════════════════════════════════════════════"
ext  = GPT2ConfigLoader.read(GPT2_GGUF)
gcfg = Toy::GPT2Config.new(ext.vocab_size, ext.d_model, ext.n_heads,
                           ext.d_ff, ext.n_layers, ext.context_length)
gpt = Toy::GPT2.new(gcfg)
GGUFLoad.load_toy_gpt2(gpt, GPT2_GGUF)
puts ""
puts gpt.algorithm_card_full

puts ""
puts "═══════════════════════════════════════════════════════════════════════════"
puts " Toy::SmolLM2 — algorithm cards"
puts "═══════════════════════════════════════════════════════════════════════════"
scfg = SmolLM2ConfigLoader.read(SMOLLM2_GGUF)
llm  = Toy::SmolLM2.new(scfg)
GGUFLoad.load_toy_smollm2(llm, SMOLLM2_GGUF)
puts ""
puts llm.algorithm_card_full
