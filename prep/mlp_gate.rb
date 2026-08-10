#!/usr/bin/env ruby
# prep/mlp_gate.rb — toy#152 (DFA-arch T0) gate for the MLP-classifier
# ANCHOR (libexec/toy-train-mlp, `toy train mlp`).
#
# Legs:
#   1. BYTE FIXTURE + CHAIN BYTE-NULL: the default 5-step curve equals
#      prep/fixtures/train_mlp_baseline.txt, and an EXPLICIT all-chain
#      policy is byte-identical to no policy at all (the "default that
#      changes nothing" discipline — a policy knob must not perturb the
#      arm it is supposed to reproduce).
#   2. DETERMINISM: two identical dfa runs → identical stdout.
#   3. ARM EFFECT: the dfa and frozen curves both differ from chain.
#   4. WIRING (structure, not curves): run_start.dfa reports the exact
#      dfa_wired / frozen counts the policy asked for. A silently
#      unwired dfa layer would otherwise look like "DFA did nothing".
#   5. BUNDLE: JSONL shape, align events WITH `wname` (tao#19 item 3),
#      the eval event, run_end last.
#  5b. THE ALIGNMENT PHASE: cos(g_dfa, g_bp) climbs from ~0 — the
#      mechanism ("align, then memorise") rather than the outcome.
#   6. THE SUCCESS BAR (tao#19 item 4 — MANDATORY, not optional):
#         positive = all-DFA within the stated gap of all-BP
#                    AND provably beating the frozen control,
#                    at matched init and matched seed.
#      Every arm here differs ONLY in MLP_POLICY.
#   7. OUTPUT-DIM DEGRADATION: the measured claim of the F13 lens —
#      the fraction of BP's over-frozen gain that DFA recovers must
#      FALL monotonically as #classes grows over {2, 10, 100, 1000}.
#   8. FAIL-LOUD: bad policy tokens / over-long policies / degenerate
#      shapes exit non-zero instead of quietly doing nothing.
#   9. CLI: `toy train mlp` reproduces the same curve through the
#      controlled-env bridge.
#
# WHY THIS GATE MATTERS MORE THAN ITS SIZE. F4–F14 are entirely
# NEGATIVE results for DFA. If this harness cannot reproduce a KNOWN
# DFA positive at small output dim, those negatives are not findings —
# they are potentially harness artifacts. Leg 6+7 are that control.

ROOT    = File.expand_path("..", __dir__)
RUNNER  = File.join(ROOT, "libexec", "toy-train-mlp")
TOY     = File.join(ROOT, "bin", "toy")
FIXTURE = File.join(ROOT, "prep", "fixtures", "train_mlp_baseline.txt")

require "open3"
require "json"
require "tmpdir"
require "fileutils"

# The anchor cell: 3x64 hidden, 1000 steps, lr 0.003, and a 2048-sample
# held-out set. Held in ONE place so leg 6 and leg 7 cannot drift apart.
#
# VAL_BATCHES=32 (2048 samples), NOT the runner default of 8 (512), and
# that is a MEASURED requirement, not caution: at 512 samples the
# 1000-class row's recovery is noise-dominated (its numerator and
# denominator are both differences of ~0.1-accuracy estimates), and the
# 100 -> 1000 comparison flipped sign between two honest val sets. At
# 2048 the full 2 -> 10 -> 100 -> 1000 ordering holds on every seed
# tried. A gate that reads a noisy statistic is a gate that fails for
# reasons unrelated to the change under test.
ANCHOR = { "STEPS" => "1000", "SEED" => "0", "MLP_VAL_BATCHES" => "32" }.freeze
# BP-over-DFA gap allowed on val accuracy, and the margin by which DFA
# must beat the frozen control. MEASURED at this cell, 10 classes,
# seeds 0/1/2 (the gate itself pins seed 0):
#   seed 0: chain .773 dfa .742 frozen .516   gap .031  edge .226
#   seed 1: chain .762 dfa .733 frozen .513   gap .029  edge .220
#   seed 2: chain .771 dfa .742 frozen .517   gap .029  edge .225
# The thresholds sit ~3x above the observed spread, because a gate
# tuned to one seed's luck is a gate that fails on the next honest
# change. They are still far below a REGRESSION: DFA collapsing to the
# frozen control would show a ~.26 gap.
BP_GAP       = 0.09
FROZEN_EDGE  = 0.05

def run_mlp(extra_env, run_dir)
  env = { "STEPS" => "5", "SEED" => "0" }.merge(extra_env)
  env = env.merge("TAO_RUN_DIR" => run_dir, "TOY_RUN_ID" => "mlp-gate") if run_dir
  out, st = Open3.capture2e(env, RUNNER, chdir: ROOT)
  abort "mlp_gate: runner exited #{st.exitstatus}:\n#{out.lines.last(10).join}" unless st.success?
  out
end

def curve(out)
  out.lines.select { |l| l.start_with?("step ") }
end

def val_acc(out)
  line = out.lines.find { |l| l.start_with?("val: ") }
  raise "mlp_gate: no val line in\n#{out}" unless line
  line[/acc=([0-9.eE+-]+)/, 1].to_f
end

unless File.executable?(RUNNER)
  build_out, build_st = Open3.capture2e("make", "-C", ROOT, "libexec/toy-train-mlp")
  unless build_st.success? && File.executable?(RUNNER)
    warn "mlp_gate: build failed:\n#{build_out.lines.last(15).join}"
    exit 2
  end
end

failures = []
# LEG BOOKKEEPING: every leg records the failure count at its START in
# `n0` and summarises with `failures.length == n0`, so each leg reports
# on ITS OWN assertions (d878143 — never summarise with
# `failures.empty?`, which makes every later leg print FAIL once ANY
# earlier leg failed, exactly when you are debugging).
n0 = 0

# ---- 1. byte fixture + chain byte-null ----
n0 = failures.length
base_out   = run_mlp({}, nil)
base_curve = curve(base_out)
if File.file?(FIXTURE)
  expect = File.readlines(FIXTURE).reject { |l| l.start_with?("#") || l.strip.empty? }
  failures << "fixture: default curve != train_mlp_baseline.txt\ngot:  #{base_curve.join}want: #{expect.join}" unless base_curve == expect
else
  failures << "fixture: #{FIXTURE} missing"
end
chain_out = run_mlp({ "MLP_POLICY" => "chain,chain,chain" }, nil)
failures << "chain byte-null: explicit all-chain curve != absent-policy curve" unless curve(chain_out) == base_curve
puts failures.length == n0 ? "  ok: default curve matches the fixture AND an explicit all-chain policy is byte-null" : "  FAIL: fixture / chain byte-null"

# ---- 2. determinism ----
n0 = failures.length
d1 = run_mlp({ "MLP_POLICY" => "dfa,dfa,dfa", "STEPS" => "20" }, nil)
d2 = run_mlp({ "MLP_POLICY" => "dfa,dfa,dfa", "STEPS" => "20" }, nil)
failures << "determinism: two identical dfa runs differ" unless d1 == d2
puts failures.length == n0 ? "  ok: two identical dfa runs are byte-identical (curve + val)" : "  FAIL: determinism"

# ---- 3. arm effect ----
n0 = failures.length
fz = run_mlp({ "MLP_POLICY" => "frozen,frozen,frozen", "STEPS" => "20" }, nil)
ch20 = run_mlp({ "STEPS" => "20" }, nil)
failures << "arm effect: dfa curve identical to chain" if curve(d1) == curve(ch20)
failures << "arm effect: frozen curve identical to chain" if curve(fz) == curve(ch20)
puts failures.length == n0 ? "  ok: dfa and frozen both move the curve off chain" : "  FAIL: arm effect"

# ---- 4 + 5. wiring + bundle ----
n0 = failures.length
Dir.mktmpdir("mlp_gate") do |dir|
  out = run_mlp({ "MLP_POLICY" => "dfa,dfa,frozen", "STEPS" => "10",
                  "MLP_ALIGN" => "1", "MLP_B_SEED" => "42" }, dir)
  failures << "bundle: runner printed no val line" unless out.lines.any? { |l| l.start_with?("val: ") }
  ev_path = File.join(dir, "events.jsonl")
  if File.file?(ev_path)
    events = File.readlines(ev_path).map { |l| JSON.parse(l) }
    rs = events.first || {}
    failures << "bundle: first event not run_start" unless rs["kind"] == "run_start"
    failures << "bundle: schema != toy/v1" unless rs["schema"] == "toy/v1"
    failures << "bundle: last event not run_end" unless events.last && events.last["kind"] == "run_end"
    md = rs["model"] || {}
    failures << "bundle: model.arch != mlp (#{md['arch'].inspect})" unless md["arch"] == "mlp"
    failures << "bundle: model.num_classes != 10" unless md["num_classes"] == 10
    co = rs["cost"] || {}
    failures << "bundle: cost fields not positive (#{co.inspect})" unless %w[total_params active_params flops_per_token].all? { |k| co[k].is_a?(Numeric) && co[k] > 0 }
    # --- leg 4: the WIRING, asserted on structure rather than curves.
    df = rs["dfa"]
    if df.nil?
      failures << "bundle: run_start has no dfa object (this lane is NOT franken — it must carry its own provenance)"
    else
      failures << "bundle: dfa.policy != [1,1,2] (got #{df['policy'].inspect})" unless df["policy"] == [1, 1, 2]
      failures << "bundle: dfa.b_seed != 42" unless df["b_seed"] == 42
      failures << "bundle: dfa.dfa_wired != 2 (got #{df['dfa_wired'].inspect}) — an unwired dfa layer looks exactly like 'DFA did nothing'" unless df["dfa_wired"] == 2
      failures << "bundle: dfa.frozen != 1 (got #{df['frozen'].inspect})" unless df["frozen"] == 1
    end
    steps_ev = events.count { |e| e["kind"] == "step" }
    failures << "bundle: #{steps_ev} step events (want 10)" unless steps_ev == 10
    aligns = events.select { |e| e["kind"] == "align" }
    failures << "bundle: #{aligns.length} align events (want 20 = 2 dfa layers x 10 steps; the frozen layer emits none)" unless aligns.length == 20
    bad = aligns.count do |e|
      c = e["cos"]
      !c.is_a?(Numeric) || c.to_f.nan? || c.to_f.abs > 1.0001 ||
        !e["dfa_norm"].is_a?(Numeric) || e["dfa_norm"] <= 0 ||
        !e["bp_norm"].is_a?(Numeric) || e["bp_norm"] <= 0
    end
    failures << "bundle: #{bad} malformed align events (cos out of range, or a zero norm = dead download vs dead gradient)" unless bad == 0
    # tao#19 item 3: ingest keys on `wname`, NOT on a per-lane wi table.
    names = aligns.map { |e| e["wname"] }.uniq.sort
    failures << "bundle: align wnames #{names.inspect} (want [\"w1\", \"w2\"])" unless names == %w[w1 w2]
    failures << "bundle: align events missing lane-local wi" unless aligns.all? { |e| e["wi"] == 0 }
    failures << "bundle: align li set #{aligns.map { |e| e['li'] }.uniq.sort.inspect} (want [0, 1])" unless aligns.map { |e| e["li"] }.uniq.sort == [0, 1]
    evs = events.select { |e| e["kind"] == "eval" }
    if evs.length != 1
      failures << "bundle: #{evs.length} eval events (want 1)"
    else
      a = evs.first["accuracy"]
      failures << "bundle: eval accuracy #{a.inspect} not in [0,1]" unless a.is_a?(Numeric) && a >= 0.0 && a <= 1.0
      failures << "bundle: eval n != 512 (the runner default 8 x 64)" unless evs.first["n"] == 512
    end
    re = events.last || {}
    failures << "bundle: run_end has no val_acc" unless re["val_acc"].is_a?(Numeric)
  else
    failures << "bundle: no events.jsonl"
  end
  fj = File.join(dir, "flow.json")
  if File.file?(fj)
    flow = JSON.parse(File.read(fj)) rescue nil
    failures << "bundle: flow.json invalid (format/nodes)" if flow.nil? || flow["format"] != "toy/v1" || !flow["nodes"].is_a?(Array) || flow["nodes"].empty?
  else
    failures << "bundle: no flow.json"
  end
end
puts failures.length == n0 ? "  ok: bundle structure + align events carry wname (tao#19 item 3) + the wiring counts match the policy" : "  FAIL: wiring / bundle"

# ---- 5b. THE ALIGNMENT PHASE (structure, not curves) ----
# Refinetti et al.'s "align, then memorise": under DFA the BP gradient
# ROTATES TOWARDS the fixed random feedback, so cos(g_dfa, g_bp) starts
# at ~0 and climbs. This is the mechanism the whole lens rests on, and
# it is a far sharper assertion than "the curve moved" — a broken DFA
# wiring can still move a curve, but it cannot make the shadow
# gradient align with a random matrix. Measured at seed 0, 400 steps:
# w1 cos runs -0.002 -> ~0.49, w2 -0.089 -> ~0.51.
n0 = failures.length
Dir.mktmpdir("mlp_gate_align") do |dir|
  run_mlp({ "MLP_POLICY" => "dfa,dfa,dfa", "STEPS" => "400",
            "MLP_ALIGN" => "1", "MLP_ALIGN_EVERY" => "100" }, dir)
  al = File.readlines(File.join(dir, "events.jsonl"))
         .map { |l| JSON.parse(l) }.select { |e| e["kind"] == "align" }
  %w[w1 w2].each do |wn|
    series = al.select { |e| e["wname"] == wn }.sort_by { |e| e["step"] }
    if series.length < 4
      failures << "alignment: only #{series.length} align events for #{wn}"
      next
    end
    first = series.first["cos"]
    last  = series.last["cos"]
    failures << "alignment: #{wn} cos did not rise (#{first.round(4)} -> #{last.round(4)}) — DFA is not aligning, so a near-BP curve here would not be the mechanism we think it is" unless last > first + 0.2 && last > 0.25
  end
end
puts failures.length == n0 ? "  ok: alignment phase live — cos(g_dfa, g_bp) climbs from ~0 (Refinetti's 'align, then memorise', reproduced)" : "  FAIL: alignment phase"

# ---- 6 + 7. the success bar, and the output-dim degradation ----
# One sweep serves both: leg 6 reads the C=10 row, leg 7 reads all of
# them. Every arm in a row differs ONLY in MLP_POLICY (matched init,
# matched seed, matched task).
n0 = failures.length
rows = {}
[2, 10, 100, 1000].each do |c|
  accs = {}
  %w[chain dfa frozen].each do |arm|
    pol = ([arm] * 3).join(",")
    accs[arm] = val_acc(run_mlp(ANCHOR.merge("MLP_CLASSES" => c.to_s,
                                             "MLP_POLICY"  => pol), nil))
  end
  rows[c] = accs
end

a = rows[10]
if a["dfa"] < a["chain"] - BP_GAP
  failures << "SUCCESS BAR (BP gap): dfa #{a['dfa']} is more than #{BP_GAP} below chain #{a['chain']} at 10 classes"
end
if a["dfa"] < a["frozen"] + FROZEN_EDGE
  failures << "SUCCESS BAR (frozen control): dfa #{a['dfa']} does not beat frozen #{a['frozen']} by #{FROZEN_EDGE} at 10 classes — 'near-BP' alone cannot tell 'DFA learned' from 'the task is easy'"
end
puts failures.length == n0 ?
  "  ok: SUCCESS BAR at 10 classes — dfa #{a['dfa'].round(4)} within #{BP_GAP} of chain #{a['chain'].round(4)} AND beating frozen #{a['frozen'].round(4)} (matched init + seed)" :
  "  FAIL: success bar"

n0 = failures.length
# "Recovery" = the fraction of BP's over-frozen gain that DFA recovers.
# Raw accuracy is NOT comparable across #classes (chance is 1/C), this
# is: 1.0 means DFA matched BP, 0.0 means it did no better than leaving
# the hidden layers at init.
recovery = {}
rows.each do |c, r|
  span = r["chain"] - r["frozen"]
  recovery[c] = span > 0 ? (r["dfa"] - r["frozen"]) / span : Float::NAN
end
rows.each do |c, r|
  puts format("    classes=%-4d chain=%.4f dfa=%.4f frozen=%.4f  recovery=%.3f",
              c, r["chain"], r["dfa"], r["frozen"], recovery[c])
end
if recovery.values.any? { |v| v.nan? }
  failures << "degradation: a row had chain <= frozen — BP failed to beat the frozen control, so the row says nothing about DFA"
else
  [[2, 10], [10, 100], [100, 1000]].each do |lo, hi|
    next if recovery[lo] > recovery[hi]
    failures << "degradation: recovery did not fall from #{lo} to #{hi} classes (#{recovery[lo].round(3)} -> #{recovery[hi].round(3)}) — the output-dim lens is the measured claim of this anchor"
  end
end
puts failures.length == n0 ? "  ok: DFA's recovery falls monotonically with output dim (2 -> 10 -> 100 -> 1000 classes)" : "  FAIL: output-dim degradation"

# ---- 8. fail-loud ----
n0 = failures.length
[
  [{ "MLP_POLICY" => "chain,chain,chain,chain" }, "policy longer than MLP_LAYERS"],
  [{ "MLP_POLICY" => "dfa,bogus,chain" },         "unknown policy token"],
  [{ "MLP_CLASSES" => "1" },                      "MLP_CLASSES=1"],
  [{ "MLP_LAYERS" => "0" },                       "MLP_LAYERS=0"],
  [{ "MLP_TASK" => "mnist" },                     "unknown MLP_TASK"],
].each do |env, label|
  out, st = Open3.capture2e({ "STEPS" => "2" }.merge(env), RUNNER, chdir: ROOT)
  failures << "fail-loud: #{label} exited 0 (silently did nothing):\n#{out.lines.last(2).join}" if st.success?
end
puts failures.length == n0 ? "  ok: 5 degenerate configs all fail loud instead of quietly doing nothing" : "  FAIL: fail-loud"

# ---- 9. CLI parity ----
n0 = failures.length
Dir.mktmpdir("mlp_gate_cli") do |dir|
  out, st = Open3.capture2e({ "SPINEL_SKIP_PIN_CHECK" => nil }, TOY, "train", "mlp",
                            "--steps", "5", "--seed", "0", "--out", dir, chdir: ROOT)
  if st.success?
    failures << "cli: `toy train mlp` curve != the runner's default curve" unless curve(out) == base_curve
    failures << "cli: `toy train mlp` did not echo the val line" unless out.lines.any? { |l| l.start_with?("val: ") }
  else
    failures << "cli: `toy train mlp` exited #{st.exitstatus}:\n#{out.lines.last(5).join}"
  end
  # tao#18 item 1: --policy-scope is NOT accepted on this recipe.
  sout, sst = Open3.capture2e({}, TOY, "train", "mlp", "--policy-scope", "ffn", chdir: ROOT)
  failures << "cli: mlp accepted --policy-scope (tao#18 item 1 says reject it)" unless sst.exitstatus == 2 && sout.include?("only valid with")
  # CPU-only by decision (tao#18 item 2).
  cout, cst = Open3.capture2e({}, TOY, "train", "mlp", "--device", "cuda", chdir: ROOT)
  failures << "cli: mlp accepted --device cuda (CPU-only by decision)" unless cst.exitstatus == 2
end
puts failures.length == n0 ? "  ok: CLI reproduces the curve; --policy-scope and --device cuda are rejected (tao#18)" : "  FAIL: CLI"

if failures.empty?
  puts "GATE PASS [mlp]: T0 anchor — byte fixture + chain byte-null, determinism, wiring, align/wname bundle, the MANDATORY success bar, and the output-dim degradation (toy#152)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [mlp]: #{f}" }
  exit 1
end
