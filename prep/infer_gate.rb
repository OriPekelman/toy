#!/usr/bin/env ruby
# prep/infer_gate.rb — deterministic functional gate for `toy infer`.
#
# Proves the CRuby→runner COMPUTE BRIDGE reproduces a RECORDED BASELINE
# byte-for-byte. `toy infer <model> --prompt X --n N` builds the lib-side
# Spinel runner libexec/toy-infer (source lib/toy/run/infer.rb) and shells
# out to it. Greedy (argmax) decode → deterministic, no seed (the runner
# passes no sampler_config → Sampler.argmax). We assert its generated line
# equals the committed baseline in prep/fixtures/infer_baseline.txt.
#
#   ruby prep/infer_gate.rb       # exit 0 on byte-for-byte match, 1 otherwise
#
# `toy infer` builds libexec/toy-infer itself (ToyRoot.ensure_built), so no
# separate `make` is needed first — only bin/toy must be present.
#
# FIXTURE: prefers data/smollm2-135m-tok.gguf (embedded tokenizer → `text:`
# path); falls back to data/smollm2-135m-f32.gguf (no tokenizer → `ids:`
# path). Both are deterministic. The baseline file is keyed by fixture
# basename; we gate whichever fixtures are present on disk, requiring at
# least one (fail loud, never silently skip).

require "open3"

ROOT     = File.expand_path("..", __dir__)
TOY      = File.join(ROOT, "bin", "toy")
BASELINE = File.join(ROOT, "prep", "fixtures", "infer_baseline.txt")
PROMPT   = "Once upon a time"
N        = 8

unless File.executable?(TOY)
  warn "infer_gate: bin/toy not executable: #{TOY}"
  exit 2
end
unless File.file?(BASELINE)
  warn "infer_gate: missing recorded baseline: #{BASELINE}"
  exit 2
end

# Load the recorded baseline: fixture-basename → expected full line.
expected = {}
File.foreach(BASELINE) do |line|
  next if line.strip.empty? || line.start_with?("#")
  key, want = line.chomp.split("\t", 2)
  expected[key] = want if key && want
end
if expected.empty?
  warn "infer_gate: baseline file has no records: #{BASELINE}"
  exit 2
end

TOK = File.join(ROOT, "data", "smollm2-135m-tok.gguf")
F32 = File.join(ROOT, "data", "smollm2-135m-f32.gguf")

# Run `toy infer` (stdout only — build chatter is on stderr). Returns the
# generated line: the `text:`-prefixed (tokenizer) or `ids:`-prefixed (no
# tokenizer) line. text path prints the bare text + a separate "ids:" never;
# ids path prints "ids: …". The `toy infer` text path prints decoded text
# only (no "text: " prefix), so we reconstruct the runner's full line shape.
def run_infer(model)
  got, st = Open3.capture2e(TOY, "infer", model, "--prompt", PROMPT, "--n", N.to_s)
  return nil unless st.success?
  lines = got.lines.map(&:chomp).reject(&:empty?)
  ids = lines.find { |l| l.start_with?("ids:") }
  return ids if ids
  # text path: the decoded continuation is the last stdout line; the recorded
  # baseline stores it as "text: <decoded>", so re-prefix for comparison.
  last = lines.last
  last ? "text: #{last}" : nil
end

ran = 0
failures = []
[TOK, F32].each do |m|
  next unless File.file?(m)
  base = File.basename(m)
  want = expected[base]
  unless want
    warn "infer_gate: fixture #{base} present but no baseline record; skipping"
    next
  end
  got = run_infer(m)
  ran += 1
  ok = got == want
  prefix = want.start_with?("ids:") ? "ids" : "text"
  puts "fixture : #{base} (#{prefix} path)"
  puts "expected: #{want.inspect}"
  puts "actual  : #{got.inspect}"
  failures << base unless ok
end

if ran == 0
  warn "infer_gate: no usable fixture (need data/smollm2-135m-{tok,f32}.gguf with a baseline record)"
  exit 2
end

if failures.empty?
  puts "GATE PASS: toy infer reproduces recorded baseline byte-for-byte (#{ran} fixture(s))"
  exit 0
else
  warn "GATE FAIL: toy infer diverged from recorded baseline on: #{failures.join(', ')}"
  exit 1
end
