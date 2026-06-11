# examples/06_runlog_compare.rb — compare your training runs from plain Ruby.
#
# WHAT YOU'LL SEE: every run bundle under runs/ parsed by Toy::RunLog
# and printed as a comparison table, best (lowest) final loss first:
#
#   run                    steps  first      final      config
#   ---------------------  -----  ---------  ---------  -------------------
#   example-01-tiny           30  6.440948   6.024838   d=64 L=2 lr=0.001
#   lr-01                     30  6.440948   7.415019   d=64 L=2 lr=0.01
#
# (yes — 10x the LR DIVERGES on this tiny model; the table is how you
# see it without re-reading 30 lines of step output.)
#
# HOW LONG: instant. This is PLAIN CRUBY — no Spinel, no FFI, no build:
#
#   ruby examples/06_runlog_compare.rb            # or: make example_06
#
# Make some runs to compare first (one compile, many runs — ENV only):
#
#   make example_01
#   ./examples/example_01_train_tiny                       # baseline
#   LR=0.01 RUN_ID=lr-01 ./examples/example_01_train_tiny  # 10x the LR
#   STEPS=100 RUN_ID=long ./examples/example_01_train_tiny # train longer
#
# (`toy train` writes the same toy/v1 bundles, so CLI runs land in the
# same table.)
#
# THE API — Toy::RunLog (lib/toy/core/run_log.rb):
#   RunLog.scan("runs")  -> every parseable bundle, best final loss first
#   log.config           -> the run_start event (model + config hashes)
#   log.loss_curve       -> Array<Float>, one per step
#   log.final_loss       -> run_end's final_loss (or last step's loss)
# Malformed bundles raise — a corrupt stream should be seen, not
# averaged over.

require_relative "../lib/toy/core/run_log"

RUNS_ROOT = ARGV[0] || "runs"

unless Dir.exist?(RUNS_ROOT)
  warn "06_runlog_compare: no #{RUNS_ROOT}/ directory here."
  warn "  train something first (see the header of this file), or pass a"
  warn "  runs root:  ruby examples/06_runlog_compare.rb path/to/runs"
  exit 1
end

logs = Toy::RunLog.scan(RUNS_ROOT)
if logs.empty?
  warn "06_runlog_compare: #{RUNS_ROOT}/ has no parseable run bundles"
  warn "  (a bundle is a subdir with an events.jsonl carrying a run_start)."
  exit 1
end

# One row per run: id, steps, first/final loss, and a compact config
# echo pulled out of the run_start event.
fmt = "%-22s  %5s  %-9s  %-9s  %s"
puts format(fmt, "run", "steps", "first", "final", "config")
puts format(fmt, "-" * 22, "-" * 5, "-" * 9, "-" * 9, "-" * 19)
logs.each do |log|
  curve  = log.loss_curve
  model  = log.config["model"]  || {}
  config = log.config["config"] || {}
  bits = []
  bits << "d=#{model['d_model']}"  if model["d_model"]
  bits << "L=#{model['n_layers']}" if model["n_layers"]
  bits << "lr=#{config['lr']}"     if config["lr"]
  bits << "seed=#{config['seed']}" if config["seed"]
  puts format(fmt,
              log.run_id || File.basename(log.dir),
              curve.length,
              curve.first ? format("%.6f", curve.first) : "-",
              log.final_loss ? format("%.6f", log.final_loss) : "-",
              bits.join(" "))
end

best = logs.first
if best.final_loss
  puts ""
  puts "best: #{best.run_id} (final loss #{format('%.6f', best.final_loss)})"
end
