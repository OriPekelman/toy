# examples/train_from_scratch.rb — THE BLESSED from-scratch path.
#
# A tiny Llama-shape model (RMSNorm + GQA + RoPE + SwiGLU) trained
# through the L4 FromScratch recipe. This is the SHORT tutorial: a
# clean read of how the four value objects compose —
#   Toy::SmolLM2Config.mha  — the model shape (no 9-arg positional soup)
#   Toy::LLM::Recipes::FromScratch — realize! + step! (the algorithm)
#   Toy::Labels.next_token  — the shift-by-one one-hot label Mat
#   Toy::AdamW              — the named optimizer hyper-params
#
# For the instrumented / Tao-harness version with events, checkpoints,
# drift sentinels, CKA taps, BATCH/GRAD_ACCUM/WEIGHT_DTYPE knobs, see
# examples/06_train_from_scratch.rb. THIS file is the place to start.
#
#   make example_train_from_scratch
#   ./examples/example_train_from_scratch
#
# Reproduces the gate fixture curve (prep/fixtures/train_baseline.txt):
#   step 1: loss=6.440947532653809
#   step 5: loss=6.151132583618164
#
# Load order is verbatim: TinyNN (via llama_seq_forward_ffi) must load
# before the recipe + value objects are required.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy/llm/engine/llama_seq_engine"
require_relative "../lib/toy/llm/adamw"
require_relative "../lib/toy/llm/labels"
require_relative "../lib/toy/llm/recipes/from_scratch"

STEPS    = (ENV["STEPS"]   || "5").to_i
D_MODEL  = (ENV["D_MODEL"] || "64").to_i
DONOR_D  = (ENV["DONOR_D"] || "128").to_i
SEED     = (ENV["SEED"]    || "0").to_i

# Gate-fixture shape so the curve reproduces byte-for-byte.
VOCAB    = 627
N_HEADS  = 4
D_FF     = 128
N_LAYERS = 2
CONTEXT  = 32

# Model shape — named factory, n_kv == n_heads (MHA), rope_base=10000.0.
cfg = Toy::SmolLM2Config.mha(VOCAB, D_MODEL, N_HEADS,
                             D_FF, N_LAYERS, CONTEXT, 10000.0, 1.0e-5)
cfg.donor_d_in = DONOR_D
puts "config: vocab=" + cfg.vocab.to_s +
     " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s +
     " heads=" + cfg.n_heads.to_s

# Realize the random-init graph THROUGH the recipe. untied=true (arg4)
# is mandatory when donor_d_in > 0.
recipe = Toy::LLM::Recipes::FromScratch.new
recipe.realize!(cfg, CONTEXT, 1, 0, true, false, SEED, 1.0)
puts "realize OK"

# A fixed training sequence — the first corpus line, CONTEXT ints,
# zero-padded to CONTEXT.
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

# Shift-by-one one-hot labels (UNGUARDED — known-good first line).
m_labels = Toy::Labels.next_token(seq_ids, VOCAB, CONTEXT, 1)

# Named AdamW hyper-params. Defaults (lr=0.001, beta1=0.9, beta2=0.95,
# eps=1e-8, wd=0.0, bias_correct=false) → constant slots 5/6 = betas.
# from-scratch hp is CONSTANT, so build it once before the loop.
m_hp = Toy::AdamW.new.hp(0)

step = 0
while step < STEPS
  loss = recipe.step!(seq_ids, positions, m_labels, m_hp, step == 0)
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s
  step = step + 1
end
