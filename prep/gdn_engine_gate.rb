#!/usr/bin/env ruby
# prep/gdn_engine_gate.rb — the GDN reintegration gate (the fold-in of
# docs/roadmap/gdn-hybrid-engine-reintegration.md, applied post-thaw).
#
#   1. ALL-ATTENTION UNCHANGED: GDN_LAYERS unset, STEPS=5 SEED=0 — the
#      stdout curve byte-equals prep/fixtures/train_baseline.txt (the
#      corruption gate: GDN code in the unit must not perturb the
#      proven path; this is the historical grads==NULL trigger).
#   2. GDN TRAINS THROUGH THE ENGINE: GDN_LAYERS=1 SEED=1 STEPS=8 —
#      loss decreases, no NaN.
#   3. SEED=0 TRAINS TOO (post toy#114 mixer): the historical
#      degenerate-stream guard is retired — zero-seed init is healthy.
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
# LEG BOOKKEEPING: every leg records the failure count at its START in
# `n0` and summarises with `failures.length == n0`, so each leg reports
# on ITS OWN assertions. Legs used to summarise with `failures.empty?`,
# which made every later leg print FAIL once ANY earlier leg had failed
# — misleading exactly when you are debugging. `n0` is seeded at top
# level so re-assignments inside blocks mutate the outer local.
n0 = 0
n0 = failures.length

# ---- 1. all-attention byte-exact ----
out, st = run_train({ "STEPS" => "5", "SEED" => "0" })
curve = out.lines.select { |l| l.start_with?("step ") }
expect = File.readlines(FIXTURE).reject { |l| l.start_with?("#") || l.strip.empty? }
failures << "all-attention: runner exited #{st.exitstatus}" unless st.success?
failures << "all-attention: curve != train_baseline.txt" unless curve == expect
puts failures.length == n0 ? "  ok: all-attention byte-equals train_baseline.txt (corruption gate)" : "  FAIL: all-attention"
n0 = failures.length

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
puts failures.length == n0 ? "  ok: GDN_LAYERS=1 trains through the engine (#{losses.first&.round(3)} -> #{losses.last&.round(3)})" : "  FAIL: gdn training"

# ---- 3. seed=0 trains (toy#114 mixer) ----
out, st = run_train({ "GDN_LAYERS" => "1", "SEED" => "0" })
l0 = out.lines.select { |l| l.start_with?("step ") }.map { |l| l[/loss=(\S+)/, 1].to_f }
if st.success? && l0.length == 8 && l0.none?(&:nan?) && l0.last < l0.first - 0.05
  puts "  ok: GDN + seed=0 trains (mixer-seeded stream; #{l0.first.round(3)} -> #{l0.last.round(3)})"
else
  failures << "seed0: GDN at seed=0 did not train (exit=#{st.exitstatus}, #{l0.first} -> #{l0.last})"
end

# ---- 4. byte-repro ----
o1, = run_train({ "GDN_LAYERS" => "1" })
o2, = run_train({ "GDN_LAYERS" => "1" })
failures << "byte-repro: outputs differ" unless o1 == o2
puts o1 == o2 ? "  ok: byte-repro — two GDN runs identical" : "  FAIL: byte-repro"

if failures.empty?
  puts "GATE PASS [gdn-engine]: all-attention byte-exact + GDN-trains(seed 0+1) + byte-repro"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [gdn-engine]: #{f}" }
  exit 1
end
