# examples/02_finetune_warm_start.rb — warm-start a tiny model from a
# REAL model's embeddings, then fine-tune.
#
# WHAT YOU'LL SEE: the same 2-layer tiny Llama as 01, but instead of a
# fully random init, its token-embedding table is loaded from a real
# donor GGUF (any llama-family model) and projected into the tiny
# model's width through a learned lens. Training then proceeds with a
# cosine LR schedule. One "step N: loss=…" line per step.
#
# HOW LONG: ~3 s to run (default 20 steps, CPU) once you have a donor
# GGUF. No big downloads needed if you already have one cached —
# `toy list` shows what's around; any llama-family GGUF works.
#
#   make example_02
#   DONOR_GGUF=data/qwen25-0.5b-native.gguf ./examples/example_02_finetune_warm_start
#
# WHAT TO TWEAK (env, no recompile):
#   DONOR_GGUF=…   the donor model (its d_model is read from the GGUF)
#   STEPS=60       train longer
#   SEED=1         a different random init for the non-donated weights
#   INIT=scratch   SKIP the donor upload (same graph, random embeddings)
#                  — diff the loss curves to see what the donor buys
#
# THE API — Toy::LLM::Recipes::WarmStart splits realize into a window:
#   donor_embed_width(gguf)       read the donor's width for the cfg
#   realize_scratch!(cfg, opts)   build the random-init graph, window OPEN
#   realize_warm!(gguf, cfg)      read + dim-check + upload the donor
#                                 embeddings (fails loud on mismatch)
#   build!                        bake forward+loss+backward+AdamW, CLOSED
#   step!(…)                      same per-step contract as FromScratch

require_relative "../lib/toy/compute"

STEPS      = (ENV["STEPS"] || "20").to_i
SEED       = (ENV["SEED"]  || "0").to_i
INIT       = ENV["INIT"]       || "warm"
DONOR_GGUF = ENV["DONOR_GGUF"] || "data/qwen25-0.5b-native.gguf"
CORPUS     = ENV["CORPUS"]     || "prep/fixtures/ts_seqs_gate.bin"

# Tiny model shape (vocab matches the bundled corpus tokenization).
VOCAB   = 627
CONTEXT = 32

# Fail LOUD on missing user-suppliable paths BEFORE compute starts
# (spinel-dev#17: a silent File.read of a missing path returns "").
if !File.exist?(DONOR_GGUF)
  puts "02_finetune_warm_start: donor GGUF not found: " + DONOR_GGUF
  puts "  point DONOR_GGUF= at any llama-family GGUF. Find one with `toy list`,"
  puts "  or fetch one:  toy fetch bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf"
  exit 1
end
if !File.exist?(CORPUS)
  puts "02_finetune_warm_start: corpus not found: " + CORPUS
  puts "  run from the repo root (the bundled corpus is prep/fixtures/ts_seqs_gate.bin)."
  exit 1
end

# Read the donor's embedding width — the projection lens is sized
# donor_d_in × d_model, so the cfg must know the donor's width before
# realize. (Fails loud if the donor is not llama-family.)
donor_d = Toy::LLM::Recipes::WarmStart.donor_embed_width(DONOR_GGUF)

cfg = Toy::SmolLM2Config.tiny
cfg.donor_d_in = donor_d        # embed table becomes vocab × donor_d
puts "model: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " ctx=" + cfg.ctx.to_s + "  donor: " + DONOR_GGUF +
     " (d=" + donor_d.to_s + ")"

opts = Toy::LLM::RecipeOptions.new
opts.t_seq  = CONTEXT
opts.untied = true              # mandatory when donor_d_in > 0
opts.seed   = SEED

recipe = Toy::LLM::Recipes::WarmStart.new
recipe.realize_scratch!(cfg, opts)

# Warm the embed table from the donor while the window is open:
# realize_warm! owns the read (open, dim-check vs cfg.donor_d_in, read
# the first VOCAB rows of token_embd.weight, upload, free) and fails
# loud on any mismatch. (The donor's vocab is much larger; rows align
# when the corpus is tokenized with the donor's tokenizer — for this
# tiny demo the POINT is the mechanism, not token alignment.)
if INIT == "warm"
  recipe.realize_warm!(DONOR_GGUF, cfg)
  puts "warm: loaded " + (VOCAB * donor_d).to_s + " donor embedding floats"
else
  puts "warm: SKIPPED (INIT=" + INIT + " — random embeddings, same graph)"
end

recipe.build!                   # bake the graph; window closed

# Cosine LR schedule + the validating batch + named AdamW.
batch = Toy::LLM::TrainingBatch.new(VOCAB, CONTEXT, 1)
adamw = Toy::AdamW.for_from_scratch

first_loss  = 0.0
final_loss  = 0.0
byte_offset = 0
step = 0
while step < STEPS
  adamw.lr = ToyLR.cosine(step, STEPS, 0.001, 0.00001, 5)
  seq_ids = ToyCorpusLoader.read_seq(CORPUS, byte_offset, CONTEXT)
  byte_offset = byte_offset + CONTEXT * 4

  batch.fill!(seq_ids)
  batch.hp = adamw.hp_for_step(step)
  loss = recipe.step!(batch.seq_ids, batch.positions, batch.labels,
                      batch.hp, step == 0)
  if step == 0
    first_loss = loss
  end
  final_loss = loss
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s
  step = step + 1
end

puts ""
puts "warm-start: " + STEPS.to_s + " steps, loss " + first_loss.to_s +
     " -> " + final_loss.to_s
if final_loss < first_loss
  puts "VERDICT: learning (over donor-initialized embeddings)"
else
  puts "VERDICT: NOT learning (loss did not fall — try more STEPS)"
end
