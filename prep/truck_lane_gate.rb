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
  puts "GATE PASS [truck-lane]: 12 legs (gradcheck, arms, frozen, b-reaches, budget, zero_grad, export, repro, trace, stride, half_yard, metrics)"
else
  puts "GATE FAIL [truck-lane]: #{failures.length} failure(s)"
  failures.each { |f| puts "  - #{f}" }
  exit 1
end
