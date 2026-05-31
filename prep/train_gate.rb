#!/usr/bin/env ruby
# prep/train_gate.rb — deterministic functional gate for `toy train`.
#
# Proves the CRuby→runner COMPUTE BRIDGE reproduces a RECORDED BASELINE loss
# curve byte-for-byte AND lays down a structurally-valid run bundle, for EACH
# supported recipe. `toy train <recipe> <args>` builds the lib-side Spinel
# runner (libexec/toy-train for from-scratch + warm-start, libexec/
# toy-train-lora for lora) and shells out to it; each runner re-uses the
# corresponding smoke reference's exact config/seed/inputs/hp, so its
# "step N: loss=" curve is deterministic.
#
#   ruby prep/train_gate.rb       # exit 0 on byte-for-byte match + valid layout
#                                 # for ALL recipes
#
# `toy train` builds the runner itself (ToyRoot.ensure_built), so no separate
# `make` is needed first — only bin/toy must be present.
#
# TWO CHECKS PER RECIPE:
#   (1) HARD PARITY — the "step N: loss=" lines on stdout MUST byte-equal
#       prep/fixtures/<baseline>. Do NOT loosen (no ratio/tolerance).
#   (2) STRUCTURAL — runs/<id>/ exists; events.jsonl is valid JSONL with a
#       run_start first + run_end last; weights/step_<N>.gguf exists.
#       Existence + validity only. The checkpoint is now a STANDARD
#       fused-llama GGUF (projection lens folded into token_embd.weight +
#       per-head attention fused) that `toy infer` loads directly; the
#       deterministic BYTE-IDENTICAL train→infer round-trip assertion lives
#       in the SEPARATE gate prep/ckpt_roundtrip_gate.rb. This gate stays
#       existence-only to keep its 3 recipes byte-identical and avoid the
#       load/generate cost.
#
# The from-scratch case is IDENTICAL to the historical single-recipe gate
# (args --steps 5 --seed 0, baseline train_baseline.txt) so its regression is
# preserved byte-for-byte.

require "open3"
require "json"

ROOT = File.expand_path("..", __dir__)
TOY  = File.join(ROOT, "bin", "toy")

CASES = [
  { recipe: "from-scratch", baseline: "train_baseline.txt",
    args: ["--steps", "5", "--seed", "0"], steps: 5 },
  { recipe: "lora",         baseline: "train_lora_baseline.txt",
    args: ["--steps", "5"], steps: 5 },
  { recipe: "warm-start",   baseline: "train_warm_start_baseline.txt",
    args: ["--steps", "5", "--seed", "0"], steps: 5 },
]

unless File.executable?(TOY)
  warn "train_gate: bin/toy not executable: #{TOY}"
  exit 2
end

# Run one case. Returns [ok(Boolean), messages(Array<String>)].
def run_case(c)
  msgs     = []
  baseline = File.join(ROOT, "prep", "fixtures", c[:baseline])
  unless File.file?(baseline)
    return [false, ["missing recorded baseline: #{baseline}"]]
  end

  # Load the recorded baseline: the bare "step N: loss=…" lines, in order.
  expected = []
  File.foreach(baseline) do |line|
    next if line.strip.empty? || line.start_with?("#")
    expected << line.chomp if line.start_with?("step ")
  end
  if expected.empty?
    return [false, ["baseline file has no `step ` records: #{baseline}"]]
  end

  out, status = Open3.capture2e(TOY, "train", c[:recipe], *c[:args], chdir: ROOT)
  unless status.success?
    return [false, ["`toy train #{c[:recipe]}` exited #{status.exitstatus}:\n#{out.lines.last(20).join}"]]
  end

  # (1) HARD PARITY: the step/loss curve byte-for-byte.
  got = out.lines.map(&:chomp).select { |l| l.start_with?("step ") }
  if got != expected
    msgs << "GATE FAIL [#{c[:recipe]}]: loss curve diverged from recorded baseline"
    msgs << "  expected:"
    expected.each { |l| msgs << "    #{l}" }
    msgs << "  actual:"
    got.each { |l| msgs << "    #{l}" }
    return [false, msgs]
  end

  # Parse the "run <id> → <run_dir>" human line for the structural checks.
  run_line = out.lines.find { |l| l.start_with?("run ") }
  unless run_line && run_line =~ /\Arun (\S+) .* (\/\S+)\s*\z/
    return [false, ["could not parse run dir from CLI output: #{run_line.inspect}"]]
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

  ckpt = File.join(run_dir, "weights", "step_#{c[:steps]}.gguf")
  errors << "checkpoint missing: #{ckpt}" unless File.file?(ckpt)

  if errors.empty?
    [true, ["GATE PASS [#{c[:recipe]}]: reproduces baseline byte-for-byte; " \
            "runs/#{run_id}/ has valid events.jsonl + checkpoint"]]
  else
    [false, ["GATE FAIL (structural) [#{c[:recipe]}]:"] + errors.map { |e| "  - #{e}" }]
  end
end

all_ok = true
CASES.each do |c|
  ok, msgs = run_case(c)
  msgs.each { |m| ok ? puts(m) : warn(m) }
  all_ok = false unless ok
end

exit(all_ok ? 0 : 1)
