# examples/smoke_compute_surface.rb — proves the toy#42 full-API require
# (lib/toy/compute.rb) compiles under Spinel and yields a working compute
# surface from ONE require, via the RECIPE path.
#
# WHY THE RECIPE PATH (not just an engine). An earlier version drove the engine
# directly (realize_for_random_init) and compiled even while a recipe-path
# consumer did NOT — Spinel poly-degradation (OriPekelman/spinel-dev#11) only
# bites once a recipe co-loads with the full surface. So this gate trains via
# Toy::LLM::Recipes::FromScratch (realize! + step!), the exact shape the
# `toy new --lib` scaffold and a real consumer use. The BUILD is the
# combined-surface gate (every file compute.rb pulls co-compiles); the RUN
# proves the surface is live.
#
# WHY LoRA IS INSTANTIATED BUT NOT TRAINED (toy#52). `recipes/lora` is the file
# whose UNCALLED LoRA#realize! historically poisoned the shared SmolLM2/RoPE
# constructor (cfg → poly via the constructor slot, spinel-dev#12; the regular-
# method facet was spinel-dev#11). Keeping realize! UNCALLED here is the point:
# the C compile of this very gate exercises exactly that shape, so a Spinel
# regression re-breaks the BUILD loudly. Calling realize! would pin cfg
# concretely and mask the bug — and would also drag a GGUF base model into a
# gate that is deliberately data-free (the byte-exact LoRA *training* gate is
# examples/smoke_recipe_lora.rb). We instantiate LoRA.new to prove the class is
# addressable from the one require.
#
#   make examples/smoke_compute_surface && ./examples/smoke_compute_surface
#
# Expected last line: "compute-surface: ok".

require_relative "../lib/toy/compute"

VOCAB   = 627
CONTEXT = 16
STEPS   = 5

cfg = Toy::SmolLM2Config.new(VOCAB, 64, 4, 4, 128, 2, CONTEXT, 10000.0, 1.0e-5)

# Engines co-compiled (instantiate all three — proves they're addressable from
# the one require), then train via the L4 recipe.
llama = Toy::LLM::Engine::LlamaSeqEngine.new
vit   = Toy::LLM::Engine::ViTTinyEngine.new
gpt2  = Toy::LLM::Engine::GPT2SeqEngine.new

# LoRA recipe: instantiated, realize! left UNCALLED on purpose (see header).
lora = Toy::LLM::Recipes::LoRA.new

recipe = Toy::LLM::Recipes::FromScratch.new
recipe.realize!(cfg, CONTEXT, 1, 0, false, false, 0, 1.0)

seq_ids = [0]
seq_ids.pop
positions = [0]
positions.pop
i = 0
while i < CONTEXT
  seq_ids.push(i % VOCAB)
  positions.push(i)
  i = i + 1
end

m_labels = Toy::Labels.next_token(seq_ids, VOCAB, CONTEXT, 1)
m_hp     = Toy::AdamW.new.hp(0)

loss = 0.0
step = 0
while step < STEPS
  loss = recipe.step!(seq_ids, positions, m_labels, m_hp, step == 0)
  step = step + 1
end

puts "compute-surface: engines=[llama,vit,gpt2] recipes=[from_scratch trained, lora addressable] final_loss=" + loss.to_s
if loss.finite?
  puts "compute-surface: ok"
else
  puts "compute-surface: FAIL (non-finite loss)"
end
