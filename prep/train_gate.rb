#!/usr/bin/env ruby
# prep/train_gate.rb — deterministic functional gate for `toy train from-scratch`.
#
# Proves the CRuby→runner COMPUTE BRIDGE reproduces a RECORDED BASELINE loss
# curve byte-for-byte AND lays down a structurally-valid run bundle. `toy train
# from-scratch --steps 5 --seed 0` builds the lib-side Spinel runner
# libexec/toy-train (source lib/toy/run/train.rb) and shells out to it; the
# runner re-uses the FromScratch reference's exact config/seed/inputs/constant
# hp, so its "step N: loss=" curve is deterministic.
#
#   ruby prep/train_gate.rb       # exit 0 on byte-for-byte match + valid layout
#
# `toy train` builds libexec/toy-train itself (ToyRoot.ensure_built), so no
# separate `make` is needed first — only bin/toy must be present.
#
# TWO CHECKS:
#   (1) HARD PARITY — the "step N: loss=" lines on stdout MUST byte-equal
#       prep/fixtures/train_baseline.txt. Do NOT loosen (no ratio/tolerance).
#   (2) STRUCTURAL — runs/<id>/ exists; events.jsonl is valid JSONL with a
#       run_start first + run_end last; weights/step_<N>.gguf exists. Existence
#       + validity only (the GGUF uses training-graph tensor naming, not yet
#       loadable by `toy infer` — no round-trip assertion).

require "open3"
require "json"

ROOT     = File.expand_path("..", __dir__)
TOY      = File.join(ROOT, "bin", "toy")
BASELINE = File.join(ROOT, "prep", "fixtures", "train_baseline.txt")
STEPS    = 5
SEED     = 0

unless File.executable?(TOY)
  warn "train_gate: bin/toy not executable: #{TOY}"
  exit 2
end
unless File.file?(BASELINE)
  warn "train_gate: missing recorded baseline: #{BASELINE}"
  exit 2
end

# Load the recorded baseline: the bare "step N: loss=…" lines, in order.
expected = []
File.foreach(BASELINE) do |line|
  next if line.strip.empty? || line.start_with?("#")
  expected << line.chomp if line.start_with?("step ")
end
if expected.empty?
  warn "train_gate: baseline file has no `step ` records: #{BASELINE}"
  exit 2
end

# Run from ROOT so data/ts_seqs.txt (relative) resolves, and --json so we can
# capture the run_dir the CLI created. The loss lines are on stdout; with
# --json the human run/loss lines are replaced by a JSON doc, so run WITHOUT
# --json for the loss curve and parse the printed run dir.
out, status = Open3.capture2e(TOY, "train", "from-scratch",
                              "--steps", STEPS.to_s, "--seed", SEED.to_s,
                              chdir: ROOT)
unless status.success?
  warn "train_gate: `toy train` exited #{status.exitstatus}:\n#{out.lines.last(20).join}"
  exit 1
end

# (1) HARD PARITY: the step/loss curve byte-for-byte.
got = out.lines.map(&:chomp).select { |l| l.start_with?("step ") }
puts "expected curve:"
expected.each { |l| puts "  #{l}" }
puts "actual curve:"
got.each { |l| puts "  #{l}" }
if got != expected
  warn "GATE FAIL: toy train loss curve diverged from recorded baseline"
  exit 1
end

# Parse the "run <id> → <run_dir>" human line for the structural checks.
run_line = out.lines.find { |l| l.start_with?("run ") }
unless run_line && run_line =~ /\Arun (\S+) .* (\/\S+)\s*\z/
  warn "train_gate: could not parse run dir from CLI output: #{run_line.inspect}"
  exit 1
end
run_id  = $1
run_dir = $2.strip

# (2) STRUCTURAL checks.
errors = []
errors << "run dir missing: #{run_dir}" unless File.directory?(run_dir)

events = File.join(run_dir, "events.jsonl")
if File.file?(events)
  parsed = []
  File.foreach(events) do |line|
    next if line.strip.empty?
    begin
      parsed << JSON.parse(line)
    rescue JSON::ParserError => e
      errors << "events.jsonl has an unparseable line: #{e.message}"
    end
  end
  if parsed.empty?
    errors << "events.jsonl has no events"
  else
    errors << "first event is #{parsed.first['kind'].inspect}, expected run_start" unless parsed.first["kind"] == "run_start"
    errors << "last event is #{parsed.last['kind'].inspect}, expected run_end" unless parsed.last["kind"] == "run_end"
  end
else
  errors << "events.jsonl missing: #{events}"
end

ckpt = File.join(run_dir, "weights", "step_#{STEPS}.gguf")
errors << "checkpoint missing: #{ckpt}" unless File.file?(ckpt)

if errors.empty?
  puts "GATE PASS: toy train reproduces recorded baseline byte-for-byte; " \
       "runs/#{run_id}/ has valid events.jsonl + checkpoint"
  exit 0
else
  warn "GATE FAIL (structural): \n  - #{errors.join("\n  - ")}"
  exit 1
end
