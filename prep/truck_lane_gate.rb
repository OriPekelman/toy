#!/usr/bin/env ruby
# prep/truck_lane_gate.rb — the toy#189 truck-backer-upper CONTROL lane
# gate (libexec/toy-train-truck, five arms over the toy#188 plant).
#
# WHAT EACH LEG IS FOR. This lane's headline claim is a COMPARISON
# between credit rules, so the things that must not break silently are
# the things that make the arms different from each other. Every leg is
# two-sided: it asserts the thing that should move DID and the thing
# that should not move DID NOT, because in this lane a broken arm
# reports a plausible number rather than an error.
#
#   1  SELFTEST     the analytic BPTT gradient, finite-differenced
#                   against the plant, every parameter, in two grading
#                   modes, with the suite's own coverage asserted.
#                   `bptt` is the ceiling every other arm is read
#                   against; a sign error there flatters DFA.
#   2  ARMS RUN     all five arms complete and label themselves.
#   3  FROZEN       `frozen` leaves the hidden layer EXACTLY at init
#                   while its readout moves. This is the leg that
#                   catches the Spinel conditional-constant landmine:
#                   READOUT_ONLY assigned inside the arm branch reads
#                   back empty and `frozen` silently becomes `bptt` —
#                   which would look like a strong fixture.
#   4  B REACHES    the B SEED MOVES the DFA arms and does NOT move
#                   `bptt`. The only cheap proof that the random
#                   surrogate actually reaches the weights (toy#158);
#                   an allocated-but-unused B comes back bit-identical
#                   to its control and reads as "the control is right".
#   5  BUDGET       TRUCK_BUDGET caps plant steps, so the matched-work
#                   comparison the whole lane rests on is enforceable.
#   6  ZERO_GRAD    landmine 4's accounting: an episode whose best
#                   approach is its start contributes nothing, and the
#                   count is reported instead of absorbed.
#   7  EXPORT       the frontend's format — [layer][unit][w..., bias]
#                   with the bias LAST — and a sidecar that names the
#                   activation and the output->steering map, because
#                   the frontend's shipped nets are tanh and ours are
#                   logistic.
#   8  REPRO        two identical runs, identical stdout.
#   9  C-FIXTURE    bptt vs frozen, REPORTED not asserted — see the leg.

ROOT   = File.expand_path("..", __dir__)
RUNNER = File.join(ROOT, "libexec", "toy-train-truck")

require "open3"
require "json"

def run_truck(env)
  base = { "STEPS" => "200", "SEED" => "0", "TRUCK_EVAL_N" => "8" }
  out, st = Open3.capture2e(base.merge(env), RUNNER, chdir: ROOT)
  [out, st]
end

def prov_field(out, key)
  line = out.lines.find { |l| l.start_with?("truck: ") }
  return nil if line.nil?
  m = line.match(/(?:\A|\s)#{Regexp.escape(key)}=(\S+)/)
  m && m[1]
end

def dock5(out)
  line = out.lines.find { |l| l.start_with?("eval: set=ensemble ") }
  return nil if line.nil?
  m = line.match(/dock5=(\S+)/)
  m && m[1].to_f
end

def ensemble_mean(out)
  line = out.lines.find { |l| l.start_with?("eval: set=ensemble ") }
  return nil if line.nil?
  m = line.match(/mean_d2=(\S+)/)
  m && m[1].to_f
end

unless File.executable?(RUNNER)
  bo, bs = Open3.capture2e("make", "-C", ROOT, "libexec/toy-train-truck")
  unless bs.success? && File.executable?(RUNNER)
    warn "truck_lane_gate: build failed:\n#{bo.lines.last(15).join}"
    exit 2
  end
end

failures = []
n0 = 0

# ------------------------------------------------------------------ 1
n0 = failures.length
st_out, st_ok = Open3.capture2e({ "TRUCK_SELFTEST" => "1", "SEED" => "0" },
                                RUNNER, chdir: ROOT)
failures << "selftest exited non-zero" unless st_ok.success?
failures << "selftest did not report PASS" unless st_out.include?("selftest: PASS")
# The suite's own coverage lines must be present AND positive: the
# gradcheck passed vacuously twice while being written (a trivially
# zero gradient, then a dead clamp-row substitution), so their absence
# is a gate failure in itself.
unless st_out.include?("coverage ok — a case graded at step >= 5")
  failures << "selftest recursion coverage missing/failed"
end
unless st_out.include?("coverage ok — a clamped step at t <= grade-2")
  failures << "selftest clamp-row coverage missing/failed"
end
if failures.length == n0
  worst = st_out.scan(/worst_rel=(\S+)/).flatten.map(&:to_f).max
  puts "GATE ok [selftest]: gradcheck passes in both grading modes, " \
       "worst_rel=#{worst}, recursion + clamp row both covered"
else
  puts "GATE FAIL [selftest]:\n#{st_out}"
end

# ------------------------------------------------------------------ 2
n0 = failures.length
arm_out = {}
%w[ga bptt frozen dfa_tb dfa_rx].each do |arm|
  env = { "TRUCK_ARM" => arm, "STEPS" => "20" }
  env["TRUCK_GA_POP"] = "8" if arm == "ga"
  out, ok = run_truck(env)
  arm_out[arm] = out
  failures << "#{arm}: exited non-zero" unless ok.success?
  failures << "#{arm}: no provenance line" if prov_field(out, "arm").nil?
  failures << "#{arm}: mislabelled as #{prov_field(out, 'arm')}" if prov_field(out, "arm") != arm
  %w[ensemble point far near].each do |set|
    failures << "#{arm}: missing eval set #{set}" unless out.include?("eval: set=#{set} ")
  end
end
puts(failures.length == n0 ?
  "GATE ok [arms]: all five arms complete, self-label, and report all four eval sets separately" :
  "GATE FAIL [arms]: #{failures[n0..].join('; ')}")

# ------------------------------------------------------------------ 3
n0 = failures.length
require "tmpdir"
Dir.mktmpdir("truck-gate") do |dir|
  init = File.join(dir, "init.json")
  froz = File.join(dir, "frozen.json")
  bptt = File.join(dir, "bptt.json")
  # STEPS=1 with LR=0 is the init: an update at lr 0 is a weight no-op.
  run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "1", "TRUCK_LR" => "0",
              "TRUCK_EXPORT" => init })
  run_truck({ "TRUCK_ARM" => "frozen", "STEPS" => "50", "TRUCK_LR" => "1.0",
              "TRUCK_EXPORT" => froz })
  run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "50", "TRUCK_LR" => "1.0",
              "TRUCK_EXPORT" => bptt })
  if File.exist?(init) && File.exist?(froz) && File.exist?(bptt)
    i = JSON.parse(File.read(init))
    f = JSON.parse(File.read(froz))
    b = JSON.parse(File.read(bptt))
    # hidden layer: frozen must be EXACTLY init; bptt must differ.
    failures << "frozen: hidden layer moved" unless f[0] == i[0]
    failures << "frozen: readout did NOT move (nothing trained)" if f[1] == i[1]
    failures << "bptt: hidden layer did NOT move (frozen and bptt are the same arm)" if b[0] == i[0]
  else
    failures << "frozen leg: export files missing"
  end
end
puts(failures.length == n0 ?
  "GATE ok [frozen]: hidden layer bit-identical to init while the readout moved, and bptt's did move" :
  "GATE FAIL [frozen]: #{failures[n0..].join('; ')}")

# ------------------------------------------------------------------ 4
n0 = failures.length
%w[dfa_tb dfa_rx].each do |arm|
  a, = run_truck({ "TRUCK_ARM" => arm, "STEPS" => "50", "TRUCK_B_SEED" => "1234" })
  c, = run_truck({ "TRUCK_ARM" => arm, "STEPS" => "50", "TRUCK_B_SEED" => "99991" })
  if ensemble_mean(a) == ensemble_mean(c)
    failures << "#{arm}: B_SEED does not move the result — the surrogate never reaches the weights"
  end
end
bp1, = run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "50", "TRUCK_B_SEED" => "1234" })
bp2, = run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "50", "TRUCK_B_SEED" => "99991" })
if ensemble_mean(bp1) != ensemble_mean(bp2)
  failures << "bptt: B_SEED moved the exact-gradient arm — B is leaking into a path that has no B"
end
puts(failures.length == n0 ?
  "GATE ok [b-reaches]: B_SEED moves both DFA arms and leaves bptt bit-identical" :
  "GATE FAIL [b-reaches]: #{failures[n0..].join('; ')}")

# ------------------------------------------------------------------ 5
n0 = failures.length
bo, = run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "10000", "TRUCK_BUDGET" => "5000" })
steps = prov_field(bo, "plant_steps").to_i
ups   = prov_field(bo, "updates").to_i
failures << "budget: plant_steps #{steps} did not stop near the 5000 budget" if steps < 5000 || steps > 5000 * 3
failures << "budget: ran all 10000 updates, so the budget never bound" if ups >= 10000
unbounded, = run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "60", "TRUCK_BUDGET" => "0" })
if prov_field(unbounded, "updates").to_i != 60
  failures << "budget: BUDGET=0 should be unlimited, but the run stopped early"
end
puts(failures.length == n0 ?
  "GATE ok [budget]: the plant-step budget binds (#{steps} steps, #{ups} updates) and 0 means unlimited" :
  "GATE FAIL [budget]: #{failures[n0..].join('; ')}")

# ------------------------------------------------------------------ 6
n0 = failures.length
# The paper's single point at seed 0 is a start an untrained policy
# never improves on, so its best approach IS step 0 and the episode
# contributes no gradient. That must be COUNTED (landmine 4), not
# silently treated as a converged update.
zg, = run_truck({ "TRUCK_ARM" => "bptt", "TRUCK_START" => "point", "STEPS" => "1" })
zc = prov_field(zg, "zero_grad").to_i
failures << "zero_grad not reported" if prov_field(zg, "zero_grad").nil?
failures << "zero_grad=#{zc}: a no-contribution episode was not counted" if zc != 1
# And the other side: a healthy ensemble run must NOT be all-zero-grad,
# or the arm never trained while printing a loss curve.
he, = run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "50", "TRUCK_LR" => "1.0" })
hz = prov_field(he, "zero_grad").to_i
failures << "zero_grad=#{hz}/50 on the ensemble: no update ever contributed" if hz >= 50
puts(failures.length == n0 ?
  "GATE ok [zero_grad]: a no-contribution episode is counted (1), and the ensemble run is not all-zero (#{hz}/50)" :
  "GATE FAIL [zero_grad]: #{failures[n0..].join('; ')}")

# ------------------------------------------------------------------ 7
n0 = failures.length
Dir.mktmpdir("truck-export") do |dir|
  path = File.join(dir, "ctrl.json")
  out, = run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "5", "TRUCK_EXPORT" => path })
  if File.exist?(path)
    j = JSON.parse(File.read(path))
    failures << "export: expected 2 layers, got #{j.length}" unless j.length == 2
    if j.length == 2
      failures << "export: hidden layer has #{j[0].length} units, expected 9" unless j[0].length == 9
      failures << "export: hidden unit has #{j[0][0].length} entries, expected 4 weights + bias" unless j[0][0].length == 5
      failures << "export: output layer has #{j[1].length} units, expected 1" unless j[1].length == 1
      failures << "export: output unit has #{j[1][0].length} entries, expected 9 weights + bias" unless j[1][0].length == 10
    end
    meta_path = path + ".meta.json"
    if File.exist?(meta_path)
      m = JSON.parse(File.read(meta_path))
      failures << "sidecar: no activation named" unless m["activation"] == "logistic"
      failures << "sidecar: no output map named" unless m["output_map"].to_s.include?("2*out")
      failures << "sidecar: bias position not stated" unless m["layer_order"].to_s.include?("bias LAST")
      failures << "sidecar: fitness not labelled reconstructed" unless m["fitness"] == "reconstructed"
    else
      failures << "sidecar: #{meta_path} not written"
    end
    failures << "export: no export line on stdout" unless out.include?("export: ")
  else
    failures << "export: #{path} not written"
  end
end
puts(failures.length == n0 ?
  "GATE ok [export]: frontend shape [9][4+bias] / [1][9+bias], sidecar names activation, output map, bias position and the reconstruction" :
  "GATE FAIL [export]: #{failures[n0..].join('; ')}")

# ------------------------------------------------------------------ 8
n0 = failures.length
r1, = run_truck({ "TRUCK_ARM" => "dfa_tb", "STEPS" => "40" })
r2, = run_truck({ "TRUCK_ARM" => "dfa_tb", "STEPS" => "40" })
failures << "repro: two identical dfa_tb runs differ" unless r1 == r2
g1, = run_truck({ "TRUCK_ARM" => "ga", "STEPS" => "6", "TRUCK_GA_POP" => "8" })
g2, = run_truck({ "TRUCK_ARM" => "ga", "STEPS" => "6", "TRUCK_GA_POP" => "8" })
failures << "repro: two identical ga runs differ" unless g1 == g2
puts(failures.length == n0 ?
  "GATE ok [repro]: dfa_tb and ga are byte-reproducible at a fixed seed" :
  "GATE FAIL [repro]: #{failures[n0..].join('; ')}")

# ------------------------------------------------------------------ 9
# C-FIXTURE, REPORTED AND NOT ASSERTED. The programme's rule is that
# `bptt` must beat `frozen` with margin or no DFA reading on this
# fixture is interpretable. That is a RESEARCH outcome, not a code
# defect, so failing the battery on it would make an experimental
# result gate every unrelated commit in the tree. It is printed on
# every run instead, so the row cannot be skipped by accident.
n0 = failures.length
#
# EACH ARM AT ITS OWN CELL. Running both at one LR is how toy#160 nearly
# published "attention is DFA-hostile" from BP's learning rate. Measured
# here over lr in {0.003 .. 30} x {best, terminal} x 3 seeds at 5000
# updates: bptt peaks at 6.0 (interior — 12.0 collapses to 1513) and
# frozen peaks at 0.1. Reading the pair at a shared lr=1.0 reported
# 4999.7 vs 5399.8, a 7% margin, for a pair whose real separation is
# three orders of magnitude.
bl = []
fl = []
bd = []
fd = []
[0, 1, 2].each do |seed|
  b, = run_truck({ "TRUCK_ARM" => "bptt", "SEED" => seed.to_s,
                   "STEPS" => "2000", "TRUCK_LR" => "6.0" })
  f, = run_truck({ "TRUCK_ARM" => "frozen", "SEED" => seed.to_s,
                   "STEPS" => "2000", "TRUCK_LR" => "0.1" })
  bl << ensemble_mean(b)
  fl << ensemble_mean(f)
  bd << dock5(b)
  fd << dock5(f)
end
if bl.any?(&:nil?) || fl.any?(&:nil?)
  failures << "cfixture: could not read ensemble means"
  puts "GATE FAIL [cfixture]: eval lines unreadable"
else
  bm = bl.sum / bl.length
  fm = fl.sum / fl.length
  wins = bl.zip(fl).count { |b, f| b < f }
  bdm = bd.compact.sum / [bd.compact.length, 1].max
  fdm = fd.compact.sum / [fd.compact.length, 1].max
  puts "cfixture: bptt(lr6) mean_d2 #{bm.round(1)} dock5 #{bdm.round(3)} vs " \
       "frozen(lr0.1) mean_d2 #{fm.round(1)} dock5 #{fdm.round(3)}, " \
       "3 paired seeds (bptt lower on #{wins}/3) — " \
       "#{bm < fm ? 'bptt ahead, the fixture DISCRIMINATES' : 'FROZEN NOT BEATEN — every DFA row on this fixture is uninterpretable (C-FIXTURE)'}"
  puts "GATE ok [cfixture]: the comparison ran on 3 paired seeds; the reading " \
       "above is research output, deliberately not a pass/fail condition"
end

# ----------------------------------------------------------------------
if failures.empty?
  puts "GATE PASS [truck-lane]: 8 legs (gradcheck, arms, frozen, b-reaches, budget, zero_grad, export, repro)"
else
  puts "GATE FAIL [truck-lane]: #{failures.length} failure(s)"
  failures.each { |f| puts "  - #{f}" }
  exit 1
end
