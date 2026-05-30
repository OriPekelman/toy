#!/usr/bin/env ruby
# prep/eval_gate.rb — deterministic functional gate for `toy eval`.
#
# Proves the CRuby→runner COMPUTE BRIDGE reproduces a RECORDED BASELINE
# byte-for-byte. `toy eval <model> --top-k K` builds the lib-side Spinel
# runner libexec/toy-eval (source lib/toy/run/eval.rb) and shells out to it.
# The eval point is frozen (prefill ids=[1,100,200], logprobs at pos=3) and
# the compute is a pure CPU f32 forward + log_softmax + manual top-K → no
# sampler, no seed, deterministic. We assert the gathered "logprob:" block
# equals the committed baseline in prep/fixtures/eval_baseline.txt.
#
#   ruby prep/eval_gate.rb       # exit 0 on byte-for-byte match, 1 otherwise
#
# `toy eval` builds libexec/toy-eval itself (ToyRoot.ensure_built), so no
# separate `make` is needed first — only bin/toy must be present.
#
# FIXTURE: data/smollm2-135m-f32.gguf (no tokenizer needed; logprobs work on
# raw IDs). The baseline file is keyed by fixture basename; the stored value
# is the full ordered TOP_K-line block, \n-escaped. We require the fixture +
# its baseline record (fail loud, never silently skip).

require "open3"

ROOT     = File.expand_path("..", __dir__)
TOY      = File.join(ROOT, "bin", "toy")
BASELINE = File.join(ROOT, "prep", "fixtures", "eval_baseline.txt")
TOP_K    = 5

unless File.executable?(TOY)
  warn "eval_gate: bin/toy not executable: #{TOY}"
  exit 2
end
unless File.file?(BASELINE)
  warn "eval_gate: missing recorded baseline: #{BASELINE}"
  exit 2
end

# Load the recorded baseline: fixture-basename → expected block (decoded).
expected = {}
File.foreach(BASELINE) do |line|
  next if line.strip.empty? || line.start_with?("#")
  key, want = line.chomp.split("\t", 2)
  expected[key] = want.gsub("\\n", "\n") if key && want
end
if expected.empty?
  warn "eval_gate: baseline file has no records: #{BASELINE}"
  exit 2
end

F32 = File.join(ROOT, "data", "smollm2-135m-f32.gguf")

# Run `toy eval` (stdout only — build chatter is on stderr). Returns the
# ordered "logprob:" block joined with "\n", or nil on failure.
def run_eval(model)
  got, st = Open3.capture2e(TOY, "eval", model, "--top-k", TOP_K.to_s)
  return nil unless st.success?
  lines = got.lines.map(&:chomp).select { |l| l.start_with?("logprob:") }
  return nil if lines.empty?
  lines.join("\n")
end

ran = 0
failures = []
[F32].each do |m|
  next unless File.file?(m)
  base = File.basename(m)
  want = expected[base]
  unless want
    warn "eval_gate: fixture #{base} present but no baseline record; skipping"
    next
  end
  got = run_eval(m)
  ran += 1
  ok = got == want
  puts "fixture : #{base} (logprob path, top-#{TOP_K})"
  puts "expected: #{want.inspect}"
  puts "actual  : #{got.inspect}"
  failures << base unless ok
end

if ran == 0
  warn "eval_gate: no usable fixture (need data/smollm2-135m-f32.gguf with a baseline record)"
  exit 2
end

if failures.empty?
  puts "GATE PASS: toy eval reproduces recorded baseline byte-for-byte (#{ran} fixture(s))"
  exit 0
else
  warn "GATE FAIL: toy eval diverged from recorded baseline on: #{failures.join(', ')}"
  exit 1
end
