#!/usr/bin/env ruby
# prep/gdn_engine_gate.rb — the GDN reintegration gate (the fold-in of
# docs/roadmap/gdn-hybrid-engine-reintegration.md, applied post-thaw).
#
#   1. ALL-ATTENTION UNCHANGED: GDN_LAYERS unset, STEPS=5 SEED=0 — the
#      stdout curve byte-equals prep/fixtures/train_baseline.txt (the
#      corruption gate: GDN code in the unit must not perturb the
#      proven path; this is the historical grads==NULL trigger).
#   2. GDN TRAINS THROUGH THE ENGINE: GDN_LAYERS=1 SEED=1 STEPS=8 —
#      loss decreases, no NaN. (SEED=1: the seed-0 stream is degenerate
#      — see the init-quality issue — and GDN fails loud on it.)
#   3. SEED GUARD: GDN_LAYERS=1 SEED=0 exits nonzero with the loud
#      message (never-mask: degeneracy must not be trainable-looking).
#   4. BYTE-REPRO: two identical GDN runs, identical stdout.

ROOT    = File.expand_path("..", __dir__)
RUNNER  = File.join(ROOT, "libexec", "toy-train")
FIXTURE = File.join(ROOT, "prep", "fixtures", "train_baseline.txt")

require "open3"

def run_train(env)
  out, st = Open3.capture2e({ "STEPS" => "8", "SEED" => "1" }.merge(env), RUNNER, chdir: ROOT)
  [out, st]
end

unless File.executable?(RUNNER)
  bo, bs = Open3.capture2e("make", "-C", ROOT, "libexec/toy-train")
  unless bs.success? && File.executable?(RUNNER)
    warn "gdn_engine_gate: build failed:\n#{bo.lines.last(15).join}"
    exit 2
  end
end

failures = []

# ---- 1. all-attention byte-exact ----
out, st = run_train({ "STEPS" => "5", "SEED" => "0" })
curve = out.lines.select { |l| l.start_with?("step ") }
expect = File.readlines(FIXTURE).reject { |l| l.start_with?("#") || l.strip.empty? }
failures << "all-attention: runner exited #{st.exitstatus}" unless st.success?
failures << "all-attention: curve != train_baseline.txt" unless curve == expect
puts failures.empty? ? "  ok: all-attention byte-equals train_baseline.txt (corruption gate)" : "  FAIL: all-attention"

# ---- 2. GDN trains ----
out, st = run_train({ "GDN_LAYERS" => "1" })
failures << "gdn: runner exited #{st.exitstatus}" unless st.success?
losses = out.lines.select { |l| l.start_with?("step ") }
                  .map { |l| l[/loss=(\S+)/, 1].to_f }
if losses.length == 8
  failures << "gdn: NaN loss" if losses.any?(&:nan?)
  failures << "gdn: loss did not decrease (#{losses.first} -> #{losses.last})" unless losses.last < losses.first - 0.05
else
  failures << "gdn: #{losses.length} step lines (want 8)"
end
puts failures.empty? ? "  ok: GDN_LAYERS=1 trains through the engine (#{losses.first&.round(3)} -> #{losses.last&.round(3)})" : "  FAIL: gdn training"

# ---- 3. seed guard ----
out, st = run_train({ "GDN_LAYERS" => "1", "SEED" => "0" })
if st.success? || !out.include?("nonzero seed")
  failures << "seed-guard: GDN+seed=0 did not fail loud (exit=#{st.exitstatus})"
  puts "  FAIL: seed guard"
else
  puts "  ok: GDN + seed=0 fails loud (degenerate-stream guard)"
end

# ---- 4. byte-repro ----
o1, = run_train({ "GDN_LAYERS" => "1" })
o2, = run_train({ "GDN_LAYERS" => "1" })
failures << "byte-repro: outputs differ" unless o1 == o2
puts o1 == o2 ? "  ok: byte-repro — two GDN runs identical" : "  FAIL: byte-repro"

if failures.empty?
  puts "GATE PASS [gdn-engine]: all-attention byte-exact + GDN-trains + seed-guard + byte-repro"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [gdn-engine]: #{f}" }
  exit 1
end
