# prep/smokes/smoke_compute_surface_cuda.rb — proves the CUDA compute
# entry (lib/toy/compute_cuda.rb, toy#64 item 8) compiles under Spinel
# and yields a working compute surface from ONE require, via the
# RECIPE path on the GPU.
#
# CONSUMER-ISH twin of smoke_compute_surface.rb: the BODY below is
# device-agnostic (everything constructs through Toy::Device — the
# same lines compile against compute.rb / compute_metal.rb); only this
# require line picks the device, AT COMPILE TIME. No ViT engine line:
# vit has no CUDA mirror (see compute_cuda.rb header).
#
#   make gate-compute-surface-cuda
#
# Expected last line: "compute-surface-cuda: ok".

require_relative "../../lib/toy/compute_cuda"

VOCAB   = 627
CONTEXT = 16
STEPS   = 5

cfg = Toy::SmolLM2Config.new(VOCAB, 64, 4, 4, 128, 2, CONTEXT, 10000.0, 1.0e-5)

# Engines co-compiled + addressable through the device seam.
llama = Toy::Device.llama_engine
gpt2  = Toy::Device.gpt2_engine

opts = Toy::LLM::RecipeOptions.new
opts.t_seq = CONTEXT

recipe = Toy::Device.from_scratch_recipe
recipe.realize!(cfg, opts)

seq_ids = [0]
seq_ids.pop
i = 0
while i < CONTEXT
  seq_ids.push(i % VOCAB)
  i = i + 1
end

batch = Toy::LLM::TrainingBatch.new(VOCAB, CONTEXT, 1)
batch.fill!(seq_ids)
batch.hp = Toy::AdamW.for_from_scratch.hp(0)

loss = 0.0
step = 0
while step < STEPS
  loss = recipe.step!(batch.seq_ids, batch.positions, batch.labels,
                      batch.hp, step == 0)
  step = step + 1
end

puts "compute-surface-cuda: device=" + Toy::Device.kind +
     " engines=[llama,gpt2] + FromScratch recipe trained, final_loss=" + loss.to_s
if loss.finite?
  puts "compute-surface-cuda: ok"
else
  puts "compute-surface-cuda: FAIL (non-finite loss)"
end
