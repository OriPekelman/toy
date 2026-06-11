# examples/01_train_tiny.rb — train a tiny Llama from scratch. START HERE.
#
# WHAT YOU'LL SEE: a 2-layer, 64-dim Llama-shape model (RMSNorm + MHA +
# RoPE + SwiGLU) trained on the bundled TinyStories token corpus. One
# "step N: loss=…" line per step — cross-entropy starts near ln(627)
# ≈ 6.44 (uniform over the 627-token vocab) and falls visibly within
# 30 steps. At the end: a run summary, and a runs/<id>/events.jsonl
# bundle you can inspect from plain Ruby (see 06_runlog_compare.rb).
#
# HOW LONG: ~2 s to run (default 30 steps, CPU). Build: one make.
#
#   make example_01
#   ./examples/example_01_train_tiny
#
# WHAT TO TWEAK (env, no recompile):
#   STEPS=100         train longer (the curve keeps falling)
#   LR=0.01           bigger steps — watch it converge faster, or wobble
#   SEED=1            a different random init
#   RUN_ID=lr-01      label the runs/<RUN_ID>/ bundle (for 06's table)
#
# THE API (the whole story is six named objects — docs/framework.md):
#   Toy::SmolLM2Config.tiny      — the model shape (no 9-arg soup)
#   Toy::LLM::RecipeOptions      — named realize-time options
#   Toy::LLM::Recipes::FromScratch — realize!(cfg, opts) once, step! per step
#   Toy::LLM::TrainingBatch      — validates each step's ids + labels
#   Toy::AdamW.for_from_scratch  — the optimizer hyper-params, named
#   Toy::RunBundle               — the runs/<id>/events.jsonl writer
#
# Everything compute rides ONE require (lib/toy/compute.rb). The corpus
# streamer is the one extra: it lives outside the compute surface.

require_relative "../lib/toy/compute"
require_relative "../lib/toy/io/toy_corpus_loader"

STEPS  = (ENV["STEPS"]  || "30").to_i
SEED   = (ENV["SEED"]   || "0").to_i
LR     = (ENV["LR"]     || "0.001").to_f
RUN_ID = ENV["RUN_ID"]  || "example-01-tiny"
CORPUS = ENV["CORPUS"]  || "prep/fixtures/ts_seqs_gate.bin"

# Fail LOUD on a missing corpus BEFORE compute starts: under Spinel,
# File.read on a missing path silently returns "" (spinel-dev#17), so
# every user-suppliable path gets an explicit existence check.
if !File.exist?(CORPUS)
  puts "01_train_tiny: corpus not found: " + CORPUS
  puts "  the bundled tiny corpus is prep/fixtures/ts_seqs_gate.bin (committed);"
  puts "  run this from the repo root, or point CORPUS= at a packed-i32 token file."
  exit 1
end

# The model shape: vocab 627, d_model 64, 4 heads, d_ff 128, 2 layers,
# context 32 — the canonical tiny experiment shape.
cfg = Toy::SmolLM2Config.tiny
puts "model: vocab=" + cfg.vocab.to_s + " d=" + cfg.d_model.to_s +
     " heads=" + cfg.n_heads.to_s + " layers=" + cfg.n_layers.to_s +
     " ctx=" + cfg.ctx.to_s

# Named realize-time options: only t_seq is required; everything else
# has sane defaults (t_batch=1, f32 weights, tied embeddings, seed 0).
opts = Toy::LLM::RecipeOptions.new
opts.t_seq = cfg.ctx
opts.seed  = SEED

# realize! builds the WHOLE graph natively — forward + CE loss +
# backward + AdamW — once. step! drives one training step through it.
recipe = Toy::LLM::Recipes::FromScratch.new
recipe.realize!(cfg, opts)

# The validating per-step quartet: positions are built by the ctor,
# fill! checks every id is in-vocab and rebuilds the shift-by-one
# one-hot labels. Garbage data fails loud here, not silently mid-graph.
batch = Toy::LLM::TrainingBatch.new(cfg.vocab, cfg.ctx, 1)

# Named optimizer hyper-params (lr, betas, eps, weight decay).
adamw = Toy::AdamW.for_from_scratch
adamw.lr = LR

# Run bundle: runs/<RUN_ID>/events.jsonl in the toy/v1 schema, the same
# stream `toy train` writes — Toy::RunLog reads it back (06_runlog_compare).
# Toy::RunBundle owns the dirs + the JSON + the sink, and TRUNCATES on
# open, so a re-run with the same RUN_ID replaces the bundle.
bundle = Toy::RunBundle.new("runs", RUN_ID)
bundle.run_start!("llama", cfg.vocab, cfg.d_model, cfg.n_layers,
                  cfg.ctx, STEPS, LR, SEED)

# Train: stream context-sized windows off the corpus, one step each.
first_loss  = 0.0
final_loss  = 0.0
byte_offset = 0
step = 0
while step < STEPS
  seq_ids = ToyCorpusLoader.read_seq(CORPUS, byte_offset, cfg.ctx)
  byte_offset = byte_offset + cfg.ctx * 4              # packed i32

  batch.fill!(seq_ids)                                  # validate + labels
  batch.hp = adamw.hp(step)

  loss = recipe.step!(batch.seq_ids, batch.positions, batch.labels,
                      batch.hp, step == 0)
  if step == 0
    first_loss = loss
  end
  final_loss = loss
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s
  bundle.step!(step + 1, loss)
  step = step + 1
end

wrote_bundle = bundle.active
bundle.run_end!(STEPS, final_loss)

# The run-log summary.
puts ""
puts "run " + RUN_ID + ": " + STEPS.to_s + " steps, loss " +
     first_loss.to_s + " -> " + final_loss.to_s
if wrote_bundle
  puts "bundle: " + bundle.events_path + "  (toy/v1 events)"
  puts "inspect from plain Ruby:  ruby examples/06_runlog_compare.rb"
end
if final_loss < first_loss
  puts "VERDICT: learning"
else
  puts "VERDICT: NOT learning (loss did not fall — try more STEPS or a lower LR)"
end
