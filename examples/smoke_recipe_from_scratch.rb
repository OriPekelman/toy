# P2.6 — L4 FromScratch recipe gate. Drives the SAME random-init config
# as the frozen reference smoke_projection_lens.rb (vocab=627 d=64
# donor=128 n_heads=4 d_ff=128 L=2 ctx=32, SEED=0, 5 steps) THROUGH
# Toy::LLM::Recipes::FromScratch and prints the same "step N: loss=" lines.
# Its loss curve MUST byte-equal the reference captured from
# smoke_projection_lens.
#
#   make examples/smoke_recipe_from_scratch
#   SEED=0 STEPS=5 ./examples/smoke_recipe_from_scratch | grep '^step'
#
# Reference (must reproduce byte-for-byte):
#   step 1: loss=6.440947532653809
#   step 2: loss=6.390329360961914
#   step 3: loss=6.321597576141357
#   step 4: loss=6.240766525268555
#   step 5: loss=6.151132583618164
#
# The ONLY difference from the inlined gate is that realize/step go
# THROUGH the recipe. Every numeric input (config, donor, untied, seed,
# seq_ids, positions, labels, constant hp) is identical to
# smoke_projection_lens.rb so the loss is bit-identical. The experiment
# config + Mat construction stay in this FIXTURE per lib-vs-example scope.
#
# Load order is verbatim so the backend TinyNN loads (via
# llama_seq_forward_ffi) before the recipe is required.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/llama_seq_forward_ffi"
require_relative "../lib/toy/llm/adamw"
require_relative "../lib/toy/llm/labels"
require_relative "../lib/toy/llm/recipes/from_scratch"

STEPS    = (ENV["STEPS"]    || "5").to_i
D_MODEL  = (ENV["D_MODEL"]  || "64").to_i
DONOR_D  = (ENV["DONOR_D"]  || "128").to_i
SEED     = (ENV["SEED"]     || "0").to_i
VOCAB    = 627
N_HEADS  = 4
D_FF     = 128
N_LAYERS = 2
CONTEXT  = 32

cfg = Toy::SmolLM2Config.mha(VOCAB, D_MODEL, N_HEADS,
                              D_FF, N_LAYERS, CONTEXT, 10000.0, 1.0e-5)
cfg.donor_d_in = DONOR_D
puts "config: vocab=" + cfg.vocab.to_s +
     " donor_d_in=" + cfg.donor_d_in.to_s +
     " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s

recipe = Toy::LLM::Recipes::FromScratch.new
recipe.realize!(cfg, CONTEXT, 1, 0, true, false, SEED, 1.0)
puts "realize OK"

# Train a few steps on a fixed sequence — identical input construction
# to smoke_projection_lens.rb:61-93.
raw        = File.read("data/ts_seqs.txt")
first_line = raw.split("\n")[0]
parts      = first_line.split(" ")
seq_ids    = [0]; seq_ids.pop
k = 0
while k < parts.length && k < CONTEXT
  seq_ids.push(parts[k].to_i)
  k = k + 1
end
while seq_ids.length < CONTEXT; seq_ids.push(0); end

positions = [0]; positions.pop
p = 0; while p < CONTEXT; positions.push(p); p = p + 1; end

# Build labels: shift-by-one one-hot (UNGUARDED — known-good first line).
m_labels = Toy::Labels.next_token(seq_ids, VOCAB, CONTEXT, 1)

# CONSTANT hyper-params via NAMED AdamW (NOT example 06's bias-corrected
# per-step hp; defaults beta2=0.95, bias_correct=false → slots5/6=betas).
m_hp = Toy::AdamW.new.hp(0)

losses = [0.0]; losses.pop
step = 0
while step < STEPS
  loss = recipe.step!(seq_ids, positions, m_labels, m_hp, step == 0)
  losses.push(loss)
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s
  step = step + 1
end

initial = losses[0]
final   = losses[losses.length - 1]
ratio   = final / initial
puts "initial=" + initial.to_s + " final=" + final.to_s + " ratio=" + ratio.to_s
if ratio < 0.95
  puts "VERDICT: from_scratch recipe training is learning"
else
  puts "VERDICT: training NOT learning (final/initial = " + ratio.to_s + ")"
end
