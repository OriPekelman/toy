#!/usr/bin/env ruby
# prep/lmc_gate.rb — deterministic functional gate for `toy eval lmc`.
#
# Proves the CRuby→runner COMPUTE BRIDGE reproduces a RECORDED BASELINE
# byte-for-byte. `toy eval lmc --ckpt A --other B` builds the lib-side Spinel
# runner libexec/toy-eval-lmc (source lib/toy/run/eval_lmc.rb) and shells out
# to it. The two checkpoints are PINNED fixtures (the warm-start cross-machine
# drift lesson — never regenerate in-gate); the eval sequence is the fixed
# first TinyStories line; the per-α compute is a pure forward + ggml-INTERNAL
# CE (no Ruby libm). So the α→loss curve is byte-exact reproducible on EVERY
# platform — like the train loss curves which are gated strict-everywhere.
# Hence: STRICT byte-exact, no portable/canonical arm, no float tolerance.
#
#   ruby prep/lmc_gate.rb       # exit 0 on byte-for-byte match, 1 otherwise
#
# `toy eval lmc` builds libexec/toy-eval-lmc itself (ToyRoot.ensure_built), so
# no separate `make` is needed first — only bin/toy must be present.
#
# FIXTURES (committed, ~651KB each, well under the 5MB pre-commit guard):
#   prep/fixtures/lmc_ckpt_a.gguf  — seed-0 5-step from-scratch checkpoint
#   prep/fixtures/lmc_ckpt_b.gguf  — seed-1 5-step from-scratch checkpoint
#   prep/fixtures/lmc_baseline.txt — the recorded "lmc:" curve (one line per α)
# We require all three (fail loud, never silently skip).

require "open3"

ROOT     = File.expand_path("..", __dir__)
TOY      = File.join(ROOT, "bin", "toy")
BASELINE = File.join(ROOT, "prep", "fixtures", "lmc_baseline.txt")
CKPT_A   = File.join(ROOT, "prep", "fixtures", "lmc_ckpt_a.gguf")
CKPT_B   = File.join(ROOT, "prep", "fixtures", "lmc_ckpt_b.gguf")

unless File.executable?(TOY)
  warn "lmc_gate: bin/toy not executable: #{TOY}"
  exit 2
end
unless File.file?(BASELINE)
  warn "lmc_gate: missing recorded baseline: #{BASELINE}"
  exit 2
end
unless File.file?(CKPT_A)
  warn "lmc_gate: missing pinned checkpoint A: #{CKPT_A}"
  exit 2
end
unless File.file?(CKPT_B)
  warn "lmc_gate: missing pinned checkpoint B: #{CKPT_B}"
  exit 2
end

# Expected: the baseline "lmc:" lines joined with "\n" (robust to trailing nl).
expected = File.read(BASELINE).lines.map(&:chomp).select { |l| l.start_with?("lmc:") }.join("\n")

# Run `toy eval lmc` (stdout only — build chatter is on stderr). Returns the
# ordered "lmc:" block joined with "\n", or nil on failure.
def run_lmc
  argv = [TOY, "eval", "lmc",
          "--ckpt", "prep/fixtures/lmc_ckpt_a.gguf",
          "--other", "prep/fixtures/lmc_ckpt_b.gguf"]
  got, st = Open3.capture2e(*argv, chdir: ROOT)
  return nil unless st.success?
  lines = got.lines.map(&:chomp).select { |l| l.start_with?("lmc:") }
  return nil if lines.empty?
  lines.join("\n")
end

# Run TWICE to prove determinism.
got1 = run_lmc
got2 = run_lmc

if got1.nil? || got2.nil?
  warn "GATE FAIL: toy eval lmc failed to run (or produced no `lmc:` line)"
  exit 1
end

puts "fixture : lmc_ckpt_a.gguf × lmc_ckpt_b.gguf (α→loss curve)"
puts "expected: #{expected.inspect}"
puts "actual  : #{got1.inspect}"

unless got1 == got2
  warn "GATE FAIL: non-deterministic — run1 != run2"
  warn "run1: #{got1.inspect}"
  warn "run2: #{got2.inspect}"
  exit 1
end

unless got1 == expected
  warn "GATE FAIL: diverged from recorded baseline"
  exit 1
end

puts "GATE PASS: toy eval lmc reproduces recorded baseline byte-for-byte (run-twice deterministic)"
exit 0
