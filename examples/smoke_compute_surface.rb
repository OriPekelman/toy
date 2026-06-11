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

# Named realize options (toy#64): canonical defaults (t_batch=1,
# weight_dtype=0/f32, untied=false, qkv_bias=false, seed=0,
# init_scale=1.0); only t_seq needs setting.
opts = Toy::LLM::RecipeOptions.new
opts.t_seq = CONTEXT

recipe = Toy::LLM::Recipes::FromScratch.new
recipe.realize!(cfg, opts)

seq_ids = [0]
seq_ids.pop
i = 0
while i < CONTEXT
  seq_ids.push(i % VOCAB)
  i = i + 1
end

# The validating per-step quartet (toy#64 item 3): positions built by
# the ctor, fill! validates the sequence + rebuilds labels via
# Toy::Labels (byte-identical to the former hand-built inputs), hp is
# caller-owned.
batch = Toy::LLM::TrainingBatch.new(VOCAB, CONTEXT, 1)
batch.fill!(seq_ids)
batch.hp = Toy::AdamW.new.hp(0)

loss = 0.0
step = 0
while step < STEPS
  loss = recipe.step!(batch.seq_ids, batch.positions, batch.labels,
                      batch.hp, step == 0)
  step = step + 1
end

puts "compute-surface: engines=[llama,vit,gpt2] + FromScratch recipe trained, final_loss=" + loss.to_s
if loss.finite?
  puts "compute-surface: ok"
else
  puts "compute-surface: FAIL (non-finite loss)"
end
