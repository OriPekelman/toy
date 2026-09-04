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
#  10  TRACE        toy#190's tbu-traces/1 bundle: every per-run summary
#                   and every `u` recomputed FROM THE TRACE, the
#                   export->load round trip exact, a wrong-shape
#                   controller refused.
#  11  STRIDE       the two things the frontend depends on: yard starts
#                   reproducible from the seed alone, and a stride that
#                   never drops the last or best-approach row.
#  12  HALF_YARD    toy#192's C2b sign test: each half draws only its own
#                   side, the full yard spans both, unknown schemes are
#                   refused, and the scheme reaches TRAINING too.
#  13  METRICS      the four behaviour metrics are present, in range,
#                   and DISCRIMINATING — they must separate dfa_tb's
#                   bang-bang from dfa_rx's passive jack-knife, which
#                   mean_d2/dock5 cannot.
#  14  E-MODE       toy#193's TRUCK_E: xyt byte-null, every mode reaches
#                   the runner, yt resizes B, and no mode is a rename.
#  15  IMIT         toy#194's imitation leg: the expert is required, the
#                   frozen body stays at init while its readout moves,
#                   B_SEED moves imit_dfa and not imit_bp, and the
#                   imitation loss actually falls.
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
require "fileutils"
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
# TRUCK_LOSS=best is PINNED here: the vanishing gradient is a property
# of THAT loss (the graded step can be 0), and the default is now
# `terminal`, where the graded step is the episode length and can never
# be 0. Leaving this leg on the default made it assert something the
# default cannot produce.
zg, = run_truck({ "TRUCK_ARM" => "bptt", "TRUCK_START" => "point",
                  "TRUCK_LOSS" => "best", "STEPS" => "1" })
zc = prov_field(zg, "zero_grad").to_i
failures << "zero_grad not reported" if prov_field(zg, "zero_grad").nil?
failures << "zero_grad=#{zc} under loss=best: a no-contribution episode was not counted" if zc != 1
# And the other side, which is exactly what the loss choice buys: under
# `terminal` the same start always contributes.
zt, = run_truck({ "TRUCK_ARM" => "bptt", "TRUCK_START" => "point",
                  "TRUCK_LOSS" => "terminal", "STEPS" => "1" })
if prov_field(zt, "zero_grad").to_i != 0
  failures << "zero_grad under loss=terminal should be 0 (the graded step is the episode length)"
end
# And the other side: a healthy ensemble run must NOT be all-zero-grad,
# or the arm never trained while printing a loss curve.
he, = run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "50", "TRUCK_LR" => "1.0",
                  "TRUCK_LOSS" => "best" })
hz = prov_field(he, "zero_grad").to_i
failures << "zero_grad=#{hz}/50 on the ensemble: no update ever contributed" if hz >= 50
puts(failures.length == n0 ?
  "GATE ok [zero_grad]: under loss=best a no-contribution episode is counted (1) and the ensemble run is not all-zero (#{hz}/50); under loss=terminal it is always 0" :
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

# ----------------------------------------------------------------- 10
# toy#190 — the trace bundle. The frontend overlays what THIS engine
# drew, so the bundle's own numbers must be recoverable from its own
# trace: that is the property the format promises and the one nobody
# would notice breaking.
n0 = failures.length
Dir.mktmpdir("truck-trace") do |dir|
  ctrl  = File.join(dir, "ctrl.json")
  trace = File.join(dir, "traces.json")
  trained, = run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "200",
                         "TRUCK_LR" => "6.0", "TRUCK_EXPORT" => ctrl })
  out, ok = run_truck({ "TRUCK_LOAD" => ctrl, "TRUCK_TRACE" => trace,
                        "TRUCK_TRACE_SCHEME" => "ensemble" })
  failures << "trace: rollout exited non-zero" unless ok.success?
  failures << "trace: no trace line on stdout" unless out.include?("trace: ")

  # A LOADED controller must reproduce the run that exported it, exactly.
  # Anything less means the export or the loader is lossy, and a lossy
  # controller reads as a slightly worse arm rather than as an error.
  if ensemble_mean(trained) != ensemble_mean(out)
    failures << "trace: reloaded controller scores #{ensemble_mean(out)} vs " \
                "#{ensemble_mean(trained)} when exported — the round trip is lossy"
  end

  if File.exist?(trace)
    b = JSON.parse(File.read(trace))
    failures << "trace: wrong format tag #{b['format']}" unless b["format"] == "tbu-traces/1"
    cols = %w[signal u x y tc ts clamped]
    failures << "trace: columns #{b['columns']}" unless b["columns"] == cols
    %w[arm engine engine_git weights train_seed net objective].each do |k|
      failures << "trace: provenance missing #{k}" unless b.dig("provenance", k)
    end
    %w[ls lc u_max_deg r step_cap dock_ref wrap].each do |k|
      failures << "trace: plant missing #{k}" if b.dig("plant", k).nil?
    end
    failures << "trace: expected 15 ensemble runs, got #{b['runs']&.length}" unless b["runs"]&.length == 15

    # Recompute every per-run summary from the trace itself.
    umax = b.dig("plant", "u_max_deg").to_f * Math::PI / 180.0
    (b["runs"] || []).each do |r|
      t = r["trace"]
      if t.length != r["steps"]
        failures << "trace: run #{r['id']} has #{t.length} rows for #{r['steps']} steps"
        next
      end
      d2 = lambda do |row|
        x, y, ts = row[2], row[3], row[5]
        x * x + y * y + [ts**2, (ts - 2 * Math::PI)**2, (ts + 2 * Math::PI)**2].min
      end
      if r["best_step"] >= 1 && (d2.call(t[r["best_step"] - 1]) - r["best_d2"]).abs > 1e-9
        failures << "trace: run #{r['id']} best_d2 does not match its own best_step row"
      end
      if (d2.call(t[-1]) - r["terminal_d2"]).abs > 1e-9
        failures << "trace: run #{r['id']} terminal_d2 does not match its last row"
      end
      bad_u = t.find { |row| (row[1] - [[row[0], -1.0].max, 1.0].min * umax).abs > 1e-12 }
      failures << "trace: run #{r['id']} has a u that is not signal*u_max" if bad_u
      unless %w[docked wall bound cap].include?(r["end"])
        failures << "trace: run #{r['id']} end=#{r['end'].inspect}"
      end
    end
  else
    failures << "trace: #{trace} not written"
  end

  # The other side of the loader: a controller of the wrong SHAPE must be
  # REFUSED, not silently half-applied leaving the tail at init.
  _, bad_ok = run_truck({ "TRUCK_LOAD" => ctrl, "TRUCK_OBS" => "3",
                          "TRUCK_TRACE" => File.join(dir, "no.json") })
  failures << "trace: a 4-9-1 controller loaded into a 3-input net was accepted" if bad_ok.success?
end
puts(failures.length == n0 ?
  "GATE ok [trace]: tbu-traces/1 bundle parses, every per-run summary and every u recomputes from its own trace, the load round-trip is exact, and a wrong-shape controller is refused" :
  "GATE FAIL [trace]: #{failures[n0..].join('; ')}")

# ----------------------------------------------------------------- 11
# The two properties the frontend actually depends on: yard starts
# reproducible FROM THE SEED ALONE (so bundles from different engines
# overlay on identical states), and a stride that never drops the two
# rows every summary is computed from.
n0 = failures.length
Dir.mktmpdir("truck-stride") do |dir|
  ctrl = File.join(dir, "ctrl.json")
  run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "100", "TRUCK_LR" => "6.0",
              "TRUCK_EXPORT" => ctrl })
  mk = lambda do |name, seed, stride|
    path = File.join(dir, name)
    run_truck({ "TRUCK_LOAD" => ctrl, "TRUCK_TRACE" => path,
                "TRUCK_TRACE_SCHEME" => "yard", "TRUCK_TRACE_N" => "12",
                "TRUCK_TRACE_SEED" => seed.to_s, "TRUCK_STRIDE" => stride.to_s })
    File.exist?(path) ? JSON.parse(File.read(path)) : nil
  end
  a  = mk.call("a.json", 1234, 1)
  a2 = mk.call("a2.json", 1234, 1)
  c  = mk.call("c.json", 4321, 1)
  k  = mk.call("k.json", 1234, 10)
  if [a, a2, c, k].any?(&:nil?)
    failures << "stride: a bundle was not written"
  else
    starts = ->(b) { b["runs"].map { |r| r["start"].slice("x", "y", "ts", "tc") } }
    failures << "stride: same seed gave different yard starts" unless starts.call(a) == starts.call(a2)
    # Two-sided: if a DIFFERENT seed also gave the same starts, the seed
    # is not driving the draw and "reproducible from the seed" is vacuous.
    failures << "stride: a different seed gave identical starts — the seed is not wired" if starts.call(a) == starts.call(c)
    failures << "stride: yard x left [50,100]" unless a["runs"].all? { |r| r["start"]["x"].between?(50, 100) }
    a["runs"].each_with_index do |r, i|
      kr = k["runs"][i]
      failures << "stride: run #{i} kept #{kr['trace'].length} of #{r['steps']} rows (stride did nothing)" if r["steps"] > 30 && kr["trace"].length >= r["steps"]
      failures << "stride: run #{i} dropped the LAST row" unless kr["trace"].last == r["trace"][r["steps"] - 1]
      bs = r["best_step"]
      if bs >= 1 && !kr["trace"].include?(r["trace"][bs - 1])
        failures << "stride: run #{i} dropped the BEST-APPROACH row, so its own best_d2 is unrecoverable"
      end
    end
  end
end
puts(failures.length == n0 ?
  "GATE ok [stride]: yard starts reproduce from the seed alone (and a different seed moves them), and the stride keeps the last and best-approach rows" :
  "GATE FAIL [stride]: #{failures[n0..].join('; ')}")

# ----------------------------------------------------------------- 12
# toy#192 — the half-yard start schemes (C2b's sign test) and the four
# behaviour metrics on the eval line.
#
# The half-yard legs are TWO-SIDED by necessity: a scheme that silently
# fell through to a full-yard draw would still produce a plausible
# bundle labelled `half_yard`, which is what the first cut of the trace
# writer did — its dispatch chain's `else` swallowed both half modes.
n0 = failures.length
Dir.mktmpdir("truck-half") do |dir|
  ctrl = File.join(dir, "c.json")
  run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "50", "TRUCK_EXPORT" => ctrl })
  ys = {}
  %w[yard half_yard half_yard_neg].each do |sch|
    path = File.join(dir, "#{sch}.json")
    run_truck({ "TRUCK_LOAD" => ctrl, "TRUCK_TRACE" => path,
                "TRUCK_TRACE_SCHEME" => sch, "TRUCK_TRACE_N" => "120",
                "TRUCK_TRACE_SEED" => "7", "TRUCK_STRIDE" => "99" })
    ys[sch] = File.exist?(path) ? JSON.parse(File.read(path))["runs"].map { |r| r["start"]["y"] } : nil
  end
  if ys.values.any?(&:nil?)
    failures << "half_yard: a bundle was not written"
  else
    failures << "half_yard: drew y < 0" if ys["half_yard"].any?(&:negative?)
    failures << "half_yard_neg: drew y > 0" if ys["half_yard_neg"].any?(&:positive?)
    # The other side: `yard` must span BOTH signs, or "restricted to one
    # side" is not a restriction of anything.
    unless ys["yard"].any?(&:negative?) && ys["yard"].any?(&:positive?)
      failures << "half_yard: the full yard did not span both signs, so the halves restrict nothing"
    end
    # x and the angle axis are untouched by the restriction.
    unless ys["half_yard"].length == 120 && ys["half_yard_neg"].length == 120
      failures << "half_yard: expected 120 draws per half"
    end
  end
  # An unknown scheme is refused rather than silently becoming a yard draw.
  _, ok = run_truck({ "TRUCK_LOAD" => ctrl, "TRUCK_TRACE" => File.join(dir, "x.json"),
                      "TRUCK_TRACE_SCHEME" => "sideways" })
  failures << "half_yard: an unknown trace scheme was accepted" if ok.success?
  # And as a TRAINING scheme, not only a trace scheme.
  ho, hok = run_truck({ "TRUCK_ARM" => "dfa_tb", "TRUCK_START" => "half_yard", "STEPS" => "20" })
  failures << "half_yard: not accepted as a training start scheme" unless hok.success?
  failures << "half_yard: provenance does not name the scheme" unless prov_field(ho, "start") == "half_yard"
end
puts(failures.length == n0 ?
  "GATE ok [half_yard]: each half draws only its own side, the full yard spans both, an unknown scheme is refused, and the scheme reaches training as well as tracing" :
  "GATE FAIL [half_yard]: #{failures[n0..].join('; ')}")

# ----------------------------------------------------------------- 13
# The four metrics must be PRESENT, in range, and DISCRIMINATING. The
# last is the point of them: mean_d2/dock5 cannot separate bang-bang
# steering with the wrong sign from an under-actuated passive
# jack-knife, and a metric that reads the same for both would be no
# better. dfa_tb saturates (|signal| ~ 1.0) and dfa_rx barely steers
# (~0.04), so if those two come back equal the metric is not wired.
n0 = failures.length
mv = {}
%w[dfa_tb dfa_rx].each do |arm|
  out, = run_truck({ "TRUCK_ARM" => arm, "STEPS" => "400", "TRUCK_LR" => "0.2",
                     "TRUCK_EVAL_N" => "16" })
  line = out.lines.find { |l| l.start_with?("eval: set=far ") }
  if line.nil?
    failures << "metrics: no far eval line for #{arm}"
    next
  end
  h = line.scan(/(\w+)=(\S+)/).to_h
  %w[mean_abs_signal frac_sat frac_clamped_runs median_path_len].each do |k|
    failures << "metrics: #{arm} eval line missing #{k}" unless h.key?(k)
  end
  mv[arm] = h
  next unless h["frac_sat"]
  failures << "metrics: #{arm} frac_sat out of [0,1]" unless h["frac_sat"].to_f.between?(0, 1)
  failures << "metrics: #{arm} frac_clamped_runs out of [0,1]" unless h["frac_clamped_runs"].to_f.between?(0, 1)
  failures << "metrics: #{arm} mean_abs_signal out of [0,1]" unless h["mean_abs_signal"].to_f.between?(0, 1)
  failures << "metrics: #{arm} median_path_len is not positive" unless h["median_path_len"].to_f > 0.0
end
if mv["dfa_tb"] && mv["dfa_rx"]
  a = mv["dfa_tb"]["mean_abs_signal"].to_f
  b = mv["dfa_rx"]["mean_abs_signal"].to_f
  failures << "metrics: dfa_tb and dfa_rx report the same mean_abs_signal (#{a}) — not measuring the arm" if (a - b).abs < 1e-9
end
puts(failures.length == n0 ?
  "GATE ok [metrics]: all four present and in range, and they separate dfa_tb (#{mv.dig('dfa_tb', 'mean_abs_signal')}) from dfa_rx (#{mv.dig('dfa_rx', 'mean_abs_signal')})" :
  "GATE FAIL [metrics]: #{failures[n0..].join('; ')}")

# ----------------------------------------------------------------- 14
# toy#193 / C2d — TRUCK_E: what the broadcast projects.
#
# `xyt` must be BYTE-NULL (the mode existing at all must not move the
# default cell), each mode must reach the runner and name itself, and
# `yt` must actually resize B — a mode that renamed the arm without
# changing the projection would read as "the input is not the mechanism"
# while never having tested it.
n0 = failures.length
base, = run_truck({ "TRUCK_ARM" => "dfa_tb", "STEPS" => "150", "TRUCK_LR" => "0.2" })
xyt,  = run_truck({ "TRUCK_ARM" => "dfa_tb", "STEPS" => "150", "TRUCK_LR" => "0.2",
                    "TRUCK_E" => "xyt" })
strip = ->(o) { o.lines.reject { |l| l.start_with?("truck: ") }.join }
failures << "e-mode: TRUCK_E=xyt is not byte-null against the unset default" unless strip.call(base) == strip.call(xyt)
failures << "e-mode: e= missing from provenance" if prov_field(base, "e").nil?
failures << "e-mode: default is not xyt (#{prov_field(base, 'e')})" unless prov_field(base, "e") == "xyt"
failures << "e-mode: default e_dim is not 3" unless prov_field(base, "e_dim") == "3"

modes = {}
%w[yt centered signed_x].each do |m|
  out, ok = run_truck({ "TRUCK_ARM" => "dfa_tb", "STEPS" => "150",
                        "TRUCK_LR" => "0.2", "TRUCK_E" => m })
  failures << "e-mode: #{m} exited non-zero" unless ok.success?
  failures << "e-mode: #{m} not named in provenance" unless prov_field(out, "e") == m
  modes[m] = out
end
# yt drops a component, so B is 1x2 and the arm CANNOT be bit-identical
# to the 3-component default; the other two keep 3 components but change
# the values, so they must differ too.
failures << "e-mode: yt did not resize the error (e_dim #{prov_field(modes['yt'], 'e_dim')})" unless prov_field(modes["yt"], "e_dim") == "2"
modes.each do |m, out|
  failures << "e-mode: #{m} is bit-identical to xyt — the projection was not changed" if strip.call(out) == strip.call(xyt)
end
_, bad = run_truck({ "TRUCK_ARM" => "dfa_tb", "STEPS" => "5", "TRUCK_E" => "bogus" })
failures << "e-mode: an unknown TRUCK_E was accepted" if bad.success?
puts(failures.length == n0 ?
  "GATE ok [e-mode]: xyt is byte-null, all four modes reach the runner and name themselves, yt resizes B to e_dim 2, and every mode changes the result" :
  "GATE FAIL [e-mode]: #{failures[n0..].join('; ')}")

# ----------------------------------------------------------------- 15
# toy#194 — the imitation leg. The arms differ ONLY in what the hidden
# layer is told, so the legs that matter are the ones proving that: a
# frozen body that stays at init, a B that reaches the weights, and a
# BP body that is not secretly frozen.
n0 = failures.length
Dir.mktmpdir("truck-imit") do |dir|
  expert = File.join(dir, "expert.json")
  run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "3000", "TRUCK_LR" => "6.0",
              "TRUCK_EXPERT" => "", "TRUCK_EXPORT" => expert })
  # The imitation arms REQUIRE an expert and say so.
  _, noexp = run_truck({ "TRUCK_ARM" => "imit_dfa", "STEPS" => "5" })
  failures << "imit: imit_dfa ran without TRUCK_EXPERT" if noexp.success?

  base = { "TRUCK_EXPERT" => expert, "STEPS" => "600", "TRUCK_LR" => "2.0",
           "TRUCK_EVAL_N" => "8" }
  outs = {}
  %w[imit_bp imit_dfa imit_frozen].each do |arm|
    out, ok = run_truck(base.merge("TRUCK_ARM" => arm))
    failures << "imit: #{arm} exited non-zero" unless ok.success?
    failures << "imit: #{arm} printed no demos line" unless out.include?("demos: ")
    failures << "imit: #{arm} mislabelled" unless prov_field(out, "arm") == arm
    %w[demos demo_pairs wd imit_mse expert_mean_d2].each do |k|
      failures << "imit: #{arm} provenance missing #{k}" if prov_field(out, k).nil?
    end
    outs[arm] = out
  end

  # scarce and abundant must actually differ in how much data they make.
  sc, = run_truck(base.merge("TRUCK_ARM" => "imit_bp", "TRUCK_DEMOS" => "scarce"))
  ab, = run_truck(base.merge("TRUCK_ARM" => "imit_bp", "TRUCK_DEMOS" => "abundant",
                             "TRUCK_DEMO_N" => "60"))
  np_s = prov_field(sc, "demo_pairs").to_i
  np_a = prov_field(ab, "demo_pairs").to_i
  failures << "imit: scarce produced #{np_s} pairs, expected some" unless np_s > 0
  failures << "imit: abundant (#{np_a}) is not larger than scarce (#{np_s})" unless np_a > np_s

  # THE FROZEN BODY. Same leg as the closed-loop `frozen` arm and for
  # the same reason: if READOUT-only training silently trained the body
  # too, imit_frozen would be imit_bp under another name and the
  # C-FIXTURE control would be vacuous.
  init = File.join(dir, "i.json")
  fz   = File.join(dir, "f.json")
  bp   = File.join(dir, "b.json")
  run_truck(base.merge("TRUCK_ARM" => "imit_bp", "TRUCK_LR" => "0", "TRUCK_EXPORT" => init))
  run_truck(base.merge("TRUCK_ARM" => "imit_frozen", "TRUCK_EXPORT" => fz))
  run_truck(base.merge("TRUCK_ARM" => "imit_bp", "TRUCK_EXPORT" => bp))
  if [init, fz, bp].all? { |f| File.exist?(f) }
    i = JSON.parse(File.read(init)); f = JSON.parse(File.read(fz)); b = JSON.parse(File.read(bp))
    failures << "imit: imit_frozen moved its hidden layer" unless f[0] == i[0]
    failures << "imit: imit_frozen did not move its readout" if f[1] == i[1]
    failures << "imit: imit_bp did not move its hidden layer" if b[0] == i[0]
  else
    failures << "imit: export files missing"
  end

  # B REACHES THE WEIGHTS — and only the arm that uses it. toy#194 asks
  # for >= 2 draws because every C0 cell shared one.
  d1, = run_truck(base.merge("TRUCK_ARM" => "imit_dfa", "TRUCK_B_SEED" => "1234"))
  d2, = run_truck(base.merge("TRUCK_ARM" => "imit_dfa", "TRUCK_B_SEED" => "777"))
  failures << "imit: B_SEED does not move imit_dfa" if ensemble_mean(d1) == ensemble_mean(d2)
  p1, = run_truck(base.merge("TRUCK_ARM" => "imit_bp", "TRUCK_B_SEED" => "1234"))
  p2, = run_truck(base.merge("TRUCK_ARM" => "imit_bp", "TRUCK_B_SEED" => "777"))
  failures << "imit: B_SEED moved imit_bp, which has no B" if ensemble_mean(p1) != ensemble_mean(p2)

  # The student must FIT better than it started: imitation loss falls.
  first = outs["imit_bp"].lines.find { |l| l.start_with?("step 1: ") }
  fmse  = first && first[/loss=(\S+)/, 1].to_f
  lmse  = prov_field(outs["imit_bp"], "imit_mse").to_f
  failures << "imit: imit_bp's imitation loss did not fall (#{fmse} -> #{lmse})" unless fmse && lmse < fmse
end
puts(failures.length == n0 ?
  "GATE ok [imit]: expert required, demos generated, scarce < abundant, imit_frozen's body stays at init while its readout moves, B_SEED moves imit_dfa only, and the imitation loss falls" :
  "GATE FAIL [imit]: #{failures[n0..].join('; ')}")

# ----------------------------------------------------------------- 16
# toy#195 — the regime knobs: TRUCK_CAR, TRUCK_LESSON, and the `train`
# eval set that makes a regime self-scoring.
n0 = failures.length
car, ok = run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "300", "TRUCK_CAR" => "1",
                      "TRUCK_LR" => "6.0" })
failures << "regime: car mode exited non-zero" unless ok.success?
failures << "regime: provenance does not say body=car" unless car.include?("body=car")
trk, = run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "300", "TRUCK_LR" => "6.0" })
failures << "regime: truck provenance does not say body=truck" unless trk.include?("body=truck")

# A car has no hitch, so no rollout can clamp. The truck's rollouts do.
# Both sides asserted: a frac_clamped_runs of 0 on the truck would mean
# the metric stopped working rather than that the car has no hitch.
clamped = lambda do |out, set|
  line = out.lines.find { |l| l.start_with?("eval: set=#{set} ") }
  line && line[/frac_clamped_runs=(\S+)/, 1].to_f
end
failures << "regime: a car rollout clamped (#{clamped.call(car, 'far')})" unless clamped.call(car, "far") == 0.0
failures << "regime: no truck rollout clamped, so the flag is not being read" unless clamped.call(trk, "far").to_f > 0.0

# The lesson index reaches the runner and IS a difficulty axis: lesson 0
# is the near field and lesson 19 the full yard, so the same arm at the
# same budget must not score them the same.
l0, = run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "800", "TRUCK_LR" => "6.0",
                  "TRUCK_START" => "lesson", "TRUCK_LESSON" => "0", "TRUCK_EVAL_N" => "16" })
l19, = run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "800", "TRUCK_LR" => "6.0",
                   "TRUCK_START" => "lesson", "TRUCK_LESSON" => "19", "TRUCK_EVAL_N" => "16" })
failures << "regime: lesson index missing from provenance" unless prov_field(l0, "lesson") == "0"
failures << "regime: lesson 19 not named" unless prov_field(l19, "lesson") == "19"
tmean = lambda do |out|
  line = out.lines.find { |l| l.start_with?("eval: set=train ") }
  line && line[/mean_d2=(\S+)/, 1].to_f
end
if tmean.call(l0).nil? || tmean.call(l19).nil?
  failures << "regime: no train eval line"
elsif tmean.call(l0) >= tmean.call(l19)
  failures << "regime: lesson 0 (#{tmean.call(l0)}) is not easier than lesson 19 (#{tmean.call(l19)}) — the index is not a difficulty axis"
end
_, bad = run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "5", "TRUCK_LESSON" => "25" })
failures << "regime: TRUCK_LESSON=25 was accepted" if bad.success?

# The `train` set must BE the training scheme: under the default
# ensemble it is set 0 by construction, which is the honest degenerate
# case and the cheapest proof it is wired to the scheme at all.
ens, = run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "200", "TRUCK_EVAL_N" => "8" })
e0 = ens.lines.find { |l| l.start_with?("eval: set=ensemble ") }
e4 = ens.lines.find { |l| l.start_with?("eval: set=train ") }
if e0.nil? || e4.nil?
  failures << "regime: missing ensemble or train eval line"
elsif e0.sub("set=ensemble", "") != e4.sub("set=train", "")
  failures << "regime: `train` differs from `ensemble` under TRUCK_START=ensemble"
end
puts(failures.length == n0 ?
  "GATE ok [regime]: car mode never clamps while the truck does, the lesson index is a measured difficulty axis and is range-checked, and `train` is the run's own scheme" :
  "GATE FAIL [regime]: #{failures[n0..].join('; ')}")

# ----------------------------------------------------------------- 17
# toy#197 — TRUCK_EXPERT_SHARPEN. The one thing this knob must not do is
# change the STATE DISTRIBUTION: the plant is driven by the expert's
# true action and only the recorded target is transformed. If sharpening
# leaked into the driving action, the sweep would vary two things and
# every reading off it would be confounded — so that is asserted
# directly, not documented and hoped for.
n0 = failures.length
Dir.mktmpdir("truck-sharp") do |dir|
  expert = File.join(dir, "e.json")
  run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "3000", "TRUCK_LR" => "6.0",
              "TRUCK_EXPORT" => expert })
  base = { "TRUCK_ARM" => "imit_bp", "TRUCK_EXPERT" => expert, "STEPS" => "100",
           "TRUCK_EVAL_N" => "4" }
  demo_line = lambda { |o| o.lines.find { |l| l.start_with?("demos: ") } }
  sat = lambda { |o| demo_line.call(o)[/target_sat=(\S+)/, 1].to_f }

  outs = {}
  %w[0.25 0.5 1 2 4].each { |k| outs[k], = run_truck(base.merge("TRUCK_EXPERT_SHARPEN" => k)) }

  # 1. THE STATE DISTRIBUTION IS FIXED: identical pair count and an
  #    identical expert score at every k.
  pairs = outs.values.map { |o| prov_field(o, "demo_pairs") }.uniq
  d2s   = outs.values.map { |o| prov_field(o, "expert_mean_d2") }.uniq
  failures << "sharpen: demo_pairs varies with k (#{pairs.inspect}) — the driving action was sharpened too" unless pairs.length == 1
  failures << "sharpen: expert_mean_d2 varies with k (#{d2s.inspect}) — the plant saw the transformed action" unless d2s.length == 1

  # 2. IT IS MONOTONE IN k, and the ACHIEVED hardness is reported, not
  #    just the knob.
  s_lo = sat.call(outs["0.25"])
  s_1  = sat.call(outs["1"])
  s_hi = sat.call(outs["4"])
  failures << "sharpen: target_sat is not monotone (#{s_lo}, #{s_1}, #{s_hi})" unless s_lo <= s_1 && s_1 <= s_hi
  failures << "sharpen: k=4 did not harden the target (#{s_hi} vs #{s_1})" unless s_hi > s_1
  failures << "sharpen: k=0.25 did not soften the target (#{s_lo} vs #{s_1})" unless s_lo < s_1

  # 3. k = 1 IS BYTE-NULL against the unset default.
  unset, = run_truck(base)
  strip = ->(o) { o.lines.reject { |l| l.start_with?("truck: ") }.join }
  failures << "sharpen: k=1 is not byte-null against the unset default" unless strip.call(unset) == strip.call(outs["1"])
  # ...and a k that is not 1 must actually move the run, or the knob is
  # a label.
  failures << "sharpen: k=4 changed nothing" if strip.call(outs["4"]) == strip.call(outs["1"])

  _, bad = run_truck(base.merge("TRUCK_EXPERT_SHARPEN" => "0"))
  failures << "sharpen: k=0 was accepted" if bad.success?
end
puts(failures.length == n0 ?
  "GATE ok [sharpen]: k moves the target's achieved saturation monotonically while demo_pairs and expert_mean_d2 stay identical — the state distribution is held fixed; k=1 is byte-null" :
  "GATE FAIL [sharpen]: #{failures[n0..].join('; ')}")

# ----------------------------------------------------------------- 18
# toy#198 — a rollout is labelled by the CONTROLLER, not by the runner.
# A bundle naming the wrong arm is the wrong-number-that-looks-right
# toy#190's format exists to prevent, and the frontend overlays it under
# that label.
n0 = failures.length
Dir.mktmpdir("truck-label") do |dir|
  expert = File.join(dir, "e.json")
  run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "2000", "TRUCK_LR" => "6.0",
              "TRUCK_EXPORT" => expert })
  ctrl = File.join(dir, "idfa.json")
  run_truck({ "TRUCK_ARM" => "imit_dfa", "TRUCK_EXPERT" => expert,
              "TRUCK_DEMOS" => "scarce", "STEPS" => "300", "TRUCK_LR" => "8.0",
              "TRUCK_EXPORT" => ctrl })
  sidecar = ctrl + ".meta.json"
  failures << "label: no sidecar written beside the controller" unless File.exist?(sidecar)

  # 1. NO --arm: the label comes from the sidecar, not the runner default.
  trace = File.join(dir, "t.json")
  out, ok = run_truck({ "TRUCK_LOAD" => ctrl, "TRUCK_TRACE" => trace,
                        "TRUCK_TRACE_SCHEME" => "ensemble" })
  failures << "label: loaded rollout exited non-zero" unless ok.success?
  failures << "label: provenance says arm=#{prov_field(out, 'arm')}, not the controller's imit_dfa" unless prov_field(out, "arm") == "imit_dfa"
  failures << "label: arm_source is not `sidecar`" unless prov_field(out, "arm_source") == "sidecar"
  if File.exist?(trace)
    b = JSON.parse(File.read(trace))
    failures << "label: the BUNDLE says arm=#{b.dig('provenance', 'arm')}" unless b.dig("provenance", "arm") == "imit_dfa"
    failures << "label: the bundle does not record arm_source" unless b.dig("provenance", "arm_source") == "sidecar"
    # train_seed must be the seed that TRAINED the controller, not the
    # rollout's — for a pure rollout the latter means nothing.
    failures << "label: train_seed is the rollout's, not the controller's" unless b.dig("provenance", "train_seed") == 0
  else
    failures << "label: no trace written"
  end

  # 2. A DISAGREEING --arm is refused. Silently preferring either side
  #    relabels somebody's experiment.
  _, bad = run_truck({ "TRUCK_LOAD" => ctrl, "TRUCK_ARM" => "bptt",
                       "TRUCK_TRACE" => File.join(dir, "x.json") })
  failures << "label: a --arm disagreeing with the sidecar was accepted" if bad.success?

  # 3. An AGREEING --arm is fine — and this is also the leg that catches
  #    an imitation arm demanding an expert it does not need to roll out.
  ag, agok = run_truck({ "TRUCK_LOAD" => ctrl, "TRUCK_ARM" => "imit_dfa" })
  failures << "label: an agreeing --arm was refused (does the rollout demand an expert?)" unless agok.success?
  failures << "label: agreeing --arm lost the sidecar source" unless prov_field(ag, "arm_source") == "sidecar"

  # 4. NO SIDECAR: warn, and say the label is the runner's rather than
  #    assert a controller label nothing supports.
  bare = File.join(dir, "bare.json")
  FileUtils.cp(ctrl, bare)
  ns, = run_truck({ "TRUCK_LOAD" => bare })
  failures << "label: a missing sidecar did not warn" unless ns.include?("warning: no sidecar")
  failures << "label: arm_source should be `flag` without a sidecar" unless prov_field(ns, "arm_source") == "flag"
end
puts(failures.length == n0 ?
  "GATE ok [label]: a loaded rollout takes arm and train_seed from the controller's sidecar, records arm_source, refuses a disagreeing --arm, accepts an agreeing one, and warns when there is no sidecar" :
  "GATE FAIL [label]: #{failures[n0..].join('; ')}")

# ----------------------------------------------------------------- 19
# toy#199 — TRUCK_B_SIGMA, and the clip interaction that decides whether
# it is an axis at all.
n0 = failures.length
base = { "TRUCK_ARM" => "dfa_tb", "STEPS" => "200", "SEED" => "0",
         "TRUCK_EVAL_N" => "4", "TRUCK_B_SCALE" => "fixed" }
strip = ->(o) { o.lines.reject { |l| l.start_with?("truck: ") }.join }

# 1. BYTE-NULL: the knob's default is the literal the code used before it
#    existed.
un,  = run_truck(base)
one, = run_truck(base.merge("TRUCK_B_SIGMA" => "1.0"))
failures << "b_sigma: unset is not byte-identical to 1.0" unless strip.call(un) == strip.call(one)

# 2. WITH THE CLIP OFF it is a real axis...
off1, = run_truck(base.merge("TRUCK_CLIP" => "0"))
off4, = run_truck(base.merge("TRUCK_CLIP" => "0", "TRUCK_B_SIGMA" => "4.0"))
failures << "b_sigma: inert even with the clip off — the knob is not wired" if strip.call(off1) == strip.call(off4)

# 3. ...and WITH THE CLIP ON it is exactly inert, because a broadcast
#    arm's whole gradient is proportional to B and a global-norm clip
#    divides the magnitude back out. Asserted, not just documented: if
#    this ever stops holding, the reason a 600-cell search treated
#    lr x scale-rule as one axis has changed.
on1, = run_truck(base.merge("TRUCK_CLIP" => "1.0"))
on4, = run_truck(base.merge("TRUCK_CLIP" => "1.0", "TRUCK_B_SIGMA" => "4.0"))
failures << "b_sigma: NOT inert under the clip — the clip/B interaction changed" unless strip.call(on1) == strip.call(on4)
failures << "b_sigma: clip_hits missing from provenance" if prov_field(on1, "clip_hits").nil?
failures << "b_sigma: clip_hits is 0 while the clip was on" unless prov_field(on1, "clip_hits").to_i > 0
failures << "b_sigma: clip_hits is non-zero with the clip off" unless prov_field(off1, "clip_hits").to_i == 0

# 4. The field is printed ONLY where it is the sigma.
failures << "b_sigma: not on the line under fixed" if prov_field(one, "b_sigma").nil?
dim, = run_truck({ "TRUCK_ARM" => "dfa_tb", "STEPS" => "5", "TRUCK_EVAL_N" => "2" })
failures << "b_sigma: printed under a dimensional rule, where it is not the sigma" unless prov_field(dim, "b_sigma").nil?

# 5. Refused where it would have no effect, and at zero.
_, r1 = run_truck({ "TRUCK_ARM" => "dfa_tb", "STEPS" => "5", "TRUCK_B_SIGMA" => "2.0" })
failures << "b_sigma: accepted under a dimensional rule" if r1.success?
_, r2 = run_truck(base.merge("TRUCK_B_SIGMA" => "0"))
failures << "b_sigma: accepted at 0" if r2.success?
puts(failures.length == n0 ?
  "GATE ok [b-sigma]: byte-null at 1.0, a real axis with the clip off, EXACTLY inert with it on (clip_hits says which), printed only under `fixed`, refused elsewhere" :
  "GATE FAIL [b-sigma]: #{failures[n0..].join('; ')}")

# ----------------------------------------------------------------- 20
# toy#200 — TRUCK_B_SIGN and B2 on the line. The point of the knob is
# that -1 is the EXACT MIRROR of a run and not a different draw, so
# that is asserted elementwise off the provenance rather than assumed.
n0 = failures.length
base = { "TRUCK_ARM" => "dfa_tb", "TRUCK_E" => "yt", "TRUCK_DFA_SUM" => "to_best",
         "TRUCK_B_SEED" => "227935", "STEPS" => "200", "SEED" => "0",
         "TRUCK_EVAL_N" => "4" }
b2_of = lambda do |out|
  f = prov_field(out, "b2")
  f && f.gsub(/[\[\]]/, "").split(",").map(&:to_f)
end
pos, = run_truck(base)
neg, = run_truck(base.merge("TRUCK_B_SIGN" => "-1"))
p2 = b2_of.call(pos)
n2 = b2_of.call(neg)
if p2.nil? || n2.nil?
  failures << "b_sign: b2 is not on the provenance line for dfa_tb"
else
  failures << "b2 has #{p2.length} entries, expected e_dim 2 under yt" unless p2.length == 2
  unless p2.zip(n2).all? { |a, b| (a + b).abs < 1e-15 }
    failures << "b_sign: -1 is not the exact elementwise negation (#{p2.inspect} vs #{n2.inspect})"
  end
end
strip = ->(o) { o.lines.reject { |l| l.start_with?("truck: ") }.join }
one, = run_truck(base.merge("TRUCK_B_SIGN" => "1"))
failures << "b_sign: unset is not byte-identical to 1" unless strip.call(pos) == strip.call(one)
failures << "b_sign: -1 changed nothing" if strip.call(pos) == strip.call(neg)
failures << "b_sign: not reported" unless prov_field(neg, "b_sign") == "-1"

# Printed only where it acted: bptt uses no B, and dfa_rx never touches
# B2 (its readout signal is the exact gradient).
bp, = run_truck({ "TRUCK_ARM" => "bptt", "STEPS" => "5", "TRUCK_EVAL_N" => "2" })
failures << "b_sign: printed on bptt, which uses no B" unless prov_field(bp, "b_sign").nil?
failures << "b2: printed on bptt" unless prov_field(bp, "b2").nil?
rx, = run_truck(base.merge("TRUCK_ARM" => "dfa_rx"))
failures << "b_sign: missing on dfa_rx, which does use B1" if prov_field(rx, "b_sign").nil?
failures << "b2: printed on dfa_rx, which never consumes B2" unless prov_field(rx, "b2").nil?

_, bad = run_truck(base.merge("TRUCK_B_SIGN" => "0"))
failures << "b_sign: 0 was accepted" if bad.success?
puts(failures.length == n0 ?
  "GATE ok [b-sign]: -1 is the exact elementwise mirror of the draw, +1 is byte-null, and b_sign/b2 appear only on the arms that consume them" :
  "GATE FAIL [b-sign]: #{failures[n0..].join('; ')}")

# ------------------------------------------------------------------ 9
# C-FIXTURE, REPORTED AND NOT ASSERTED. The programme's rule is that
# `bptt` must beat `frozen` with margin or no DFA reading on this
# fixture is interpretable. That is a RESEARCH outcome, not a code
# defect, so failing the battery on it would make an experimental
# result gate every unrelated commit in the tree. It is printed on
# every run instead, so the row cannot be skipped by accident.
n0 = failures.length
#
# EACH ARM AT ITS OWN CELL, UNDER THE DEFAULT LOSS. Running both at one
# LR is how toy#160 nearly published "attention is DFA-hostile" from
# BP's learning rate. Measured over lr in {0.003 .. 48} x {best,
# terminal} x 3 seeds at 5000 updates: under the default loss=best, bptt
# peaks at 6.0 (interior — 12.0 collapses) and frozen at 0.1. Reading
# the pair at a shared lr=1.0 reported a 7% margin for a pair whose real
# separation is three orders of magnitude.
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
  # "bptt ahead" needs BOTH a margin and a majority of PAIRED seeds.
  # The first version said "the fixture DISCRIMINATES" off `bm < fm`
  # alone, and duly said it for a 0.3% mean difference that bptt won on
  # 1 of 3 seeds. Arms sharing a controlled factor are read paired
  # (B0a's verdict flipped on exactly that), and a margin inside seed
  # noise is not a margin.
  margin = fm > 0.0 ? (fm - bm) / fm : 0.0
  verdict = if wins >= 2 && margin > 0.1
              "bptt ahead by #{(margin * 100).round(1)}% on #{wins}/3 — the fixture DISCRIMINATES"
            elsif bm < fm
              "bptt ahead by only #{(margin * 100).round(1)}% on #{wins}/3 — INSIDE SEED NOISE at this smoke budget; " \
              "the research reading is 5000 updates at each arm's own cell, not 2000 here"
            else
              "FROZEN NOT BEATEN — if this holds at the research budget, every DFA row on this fixture is uninterpretable (C-FIXTURE)"
            end
  puts "cfixture: bptt(lr6) mean_d2 #{bm.round(1)} dock5 #{bdm.round(3)} vs " \
       "frozen(lr0.1) mean_d2 #{fm.round(1)} dock5 #{fdm.round(3)}, " \
       "3 paired seeds — #{verdict}"
  puts "GATE ok [cfixture]: the comparison ran on 3 paired seeds; the reading " \
       "above is research output, deliberately not a pass/fail condition"
end

# ----------------------------------------------------------------------
if failures.empty?
  puts "GATE PASS [truck-lane]: 19 legs (gradcheck, arms, frozen, b-reaches, budget, zero_grad, export, repro, trace, stride, half_yard, metrics, e-mode, imit, regime, sharpen, label, b-sigma, b-sign)"
else
  puts "GATE FAIL [truck-lane]: #{failures.length} failure(s)"
  failures.each { |f| puts "  - #{f}" }
  exit 1
end
