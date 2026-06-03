# P2.6 — L4 WarmStart recipe gate. Drives the EXACT same warm-start
# config as the frozen reference examples/09_warm_start_train.rb at
# INIT=scratch (09's DEFAULT, so no donor GGUF dependency) THROUGH
# Toy::LLM::Recipes::WarmStart and prints "step N: loss=" lines whose
# loss VALUES MUST byte-equal the reference's per-step loss.
#
#   make examples/smoke_recipe_warm_start
#   SEED=0 STEPS=5 ./examples/smoke_recipe_warm_start | grep '^step'
#
# Reference (SEED=0 STEPS=5 INIT=scratch ./examples/example_warm_start_train):
#   step 1: loss=6.440947532653809   (lr=0.0002)
#   step 2: loss=6.4390740394592285  (lr=0.0004)
#   step 3: loss=6.431989669799805   (lr=0.0006000000000000001)
#   step 4: loss=6.418195724487305   (lr=0.0008)
#   step 5: loss=6.384598731994629   (lr=0.001)
#
# The ONLY difference from the inlined reference is that realize/build/
# step go THROUGH the recipe (realize_scratch! → build! → step!). Every
# numeric input (config, donor_d_in, untied, seed, the cosine LR
# schedule, the streamed corpus sequences, the shift-by-one one-hot
# labels with the in-vocab guard, the constant hp[1..6]) is identical to
# 09_warm_start_train.rb at INIT=scratch so the loss is bit-identical.
# INIT=scratch skips realize_warm! entirely (matches 09's default arm).
# The experiment config + Mat construction + LR schedule + corpus stream
# stay in this FIXTURE per lib-vs-example scope. CPU-only, like the
# FromScratch / LoRA siblings.
#
# 09 prints "step N: lr=.. loss=.." (N rjust(4)); this fixture prints
# plain "step N: loss=<num>". Both use Float#to_s so the loss NUMBERS
# are directly comparable. The gate compares loss NUMBERS only.
# Determinism is automatic: seeded realize + deterministic corpus
# streaming → byte-identical across runs.
#
# Load order is verbatim so the backend TinyNN loads (via
# llama_seq_forward_ffi) before the recipe is required.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy/llm/engine/llama_seq_engine"
require_relative "../lib/toy/io/toy_corpus_loader"
require_relative "../lib/toy/train/toy_lr_schedule"
require_relative "../lib/toy/llm/adamw"
require_relative "../lib/toy/llm/labels"
require_relative "../lib/toy/llm/recipes/warm_start"

# All 09 defaults (the gate's resolved config).
STEPS    = (ENV["STEPS"]    || "5").to_i
D_MODEL  = (ENV["D_MODEL"]  || "64").to_i
DONOR_D  = (ENV["DONOR_D"]  || "128").to_i
SEED     = (ENV["SEED"]     || "0").to_i
CONTEXT  = (ENV["CONTEXT"]  || "32").to_i
VOCAB    = (ENV["VOCAB"]    || "627").to_i
N_HEADS  = (ENV["N_HEADS"]  || "4").to_i
D_FF     = (ENV["D_FF"]     || "128").to_i
N_LAYERS = (ENV["N_LAYERS"] || "2").to_i
LR_MAX   = (ENV["LR_MAX"]   || "0.001").to_f
LR_MIN   = (ENV["LR_MIN"]   || "0.00001").to_f
WARMUP   = (ENV["WARMUP"]   || "5").to_i
CORPUS   = ENV["CORPUS"]    || "data/ts_seqs.bin"

cfg = Toy::SmolLM2Config.mha(VOCAB, D_MODEL, N_HEADS,
                              D_FF, N_LAYERS, CONTEXT, 10000.0, 1.0e-5)
cfg.donor_d_in = DONOR_D
puts "config: vocab=" + cfg.vocab.to_s +
     " donor_d_in=" + cfg.donor_d_in.to_s +
     " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s +
     " heads=" + cfg.n_heads.to_s +
     " d_ff=" + cfg.d_ff.to_s

recipe = Toy::LLM::Recipes::WarmStart.new
# untied=true is mandatory when donor_d_in > 0. Positional args identical
# to 09 L138 / FromScratch#realize!.
recipe.realize_scratch!(cfg, CONTEXT, 1, 0, true, false, SEED, 1.0)
# INIT=scratch: skip realize_warm! (matches 09's default arm — no donor
# GGUF, no PCA lens; train from the random init).
recipe.build!
puts "realize OK"

# NAMED AdamW. Defaults (beta2=0.95, bias_correct=false) → slots5/6 =
# constant betas — byte-identical to 09's inline hp. lr refreshes each
# step from the cosine schedule; the hp Mat is rebuilt per step below.
adamw = Toy::AdamW.new

# Pre-compute the position vector (shared across steps). 09 L284-285.
positions = [0]; positions.pop
p = 0; while p < CONTEXT; positions.push(p); p = p + 1; end

# Training loop — VERBATIM from 09 L290-356 (minus the events/checkpoint
# side-channels). Cosine LR + streamed corpus + shift-by-one one-hot.
losses        = [0.0]; losses.pop
byte_offset   = 0
step          = 0
while step < STEPS
  # Cosine LR for this step (09 L297-298).
  lr = ToyLR.cosine(step, STEPS, LR_MAX, LR_MIN, WARMUP)
  adamw.lr = lr
  m_hp = adamw.hp(step)   # bias_correct=false → slots5/6=betas

  # Read next sequence from the corpus (09 L301-302).
  seq_ids = ToyCorpusLoader.read_seq(CORPUS, byte_offset, CONTEXT)
  byte_offset = byte_offset + CONTEXT * 4   # i32

  # Shift-by-one one-hot labels with the in-vocab guard (09 L304-317).
  m_labels = Toy::Labels.next_token_guarded(seq_ids, VOCAB, CONTEXT, 1)

  loss = recipe.step!(seq_ids, positions, m_labels, m_hp, step == 0)
  losses.push(loss)
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s
  step = step + 1
end

initial = losses[0]
final   = losses[losses.length - 1]
ratio   = final / initial
puts "initial=" + initial.to_s + " final=" + final.to_s + " ratio=" + ratio.to_s
# Note: the 09 INIT=scratch smoke ratio is ~0.991 (> 0.95), so 09 itself
# reports gate=failed on its LOSS-RATIO heuristic at 5 steps. THIS gate
# is bit-identity to 09's loss curve, NOT the loss-ratio heuristic.
puts "VERDICT: warm_start recipe drives the warm-start path"
