#!/usr/bin/env ruby
# prep/ssm_gate.rb — toy#155 (DFA-arch T2) gate for the selective-scan /
# Mamba-lite lane (libexec/toy-train-ssm, `toy train ssm`).
#
# Legs:
#   1. BYTE FIXTURE + CHAIN BYTE-NULL.
#   2. DETERMINISM.
#   3. FORWARD-IDENTITY: step 1 under `dfa` is BYTE-IDENTICAL to `chain`
#      (tnn_detach is forward-identity), so the arms are comparable at
#      all; steps 2+ diverge, so the backward really is different.
#   4. THE B SEED MOVES THE DFA CURVE and is INERT on chain.
#   5. WIRING + BUNDLE: the tap counts the cut asks for, the provenance,
#      the eval event, run_end last.
#   6. THE MANDATORY SUCCESS BAR (tao#19 item 4).
#   7. THE TICKET'S LINEAR CONTROL — the sharpest leg here (see below).
#   8. THE DEGENERATE TASK, measured.
#   9. FAIL-LOUD.
#  10. CLI.
#
# ── WHAT THIS LANE MEASURED, AND WHY LEG 7 IS THE POINT ──
#
# toy#155's own caveat: "in the purely-LINEAR case FA collapses to plain
# gradient descent, so the interesting alignment MUST live in the
# NONLINEAR parts. The experiment must isolate that." It does, and the
# answer inverts the ticket's structural argument. Val accuracy, 600
# steps, seeds 0/1/2, delayed-cue task:
#
#   selective   BP 1.000/.992/.988   DFA(layer) .996/.988/1.000
#               DFA(step) .250/.250/.238   frozen .227/.227/.238
#   lti         BP .750/.730/.738    DFA(layer) .645/.621/.656
#               DFA(step) .645/.668/.617   frozen .305/.270/.266
#
# Read the DFA(step) column. In the LINEAR model, cutting BPTT entirely
# costs NOTHING — DFA(step) matches DFA(layer) to within noise. In the
# SELECTIVE model it costs EVERYTHING — .99 collapses to chance, and
# that is not an LR artifact (swept 3e-5 .. 1e-2; the best step-cut cell
# is .355 against a .227 frozen control and a 1.000 BP).
#
# The ticket's hypothesis was that the SSM's linear recurrence makes
# random feedback natural. The measurement says the linearity is exactly
# what makes it free — and the selection that makes the model good is
# exactly what makes it fail. The per-step cut buys its memory story
# only in the regime where the model cannot do the task.
#
# Leg 7 pins BOTH halves, because either one alone is misreadable: the
# lti equality alone looks like "the step cut works", and the selective
# collapse alone looks like a broken build. Together they are the
# finding — and the lti equality is ALSO this gate's proof that the step
# cut is correctly WIRED, since a dead cut could not learn at all.

ROOT    = File.expand_path("..", __dir__)
RUNNER  = File.join(ROOT, "libexec", "toy-train-ssm")
TOY     = File.join(ROOT, "bin", "toy")
FIXTURE = File.join(ROOT, "prep", "fixtures", "train_ssm_baseline.txt")

require "open3"
require "json"
require "tmpdir"
require "fileutils"

CELL = { "STEPS" => "600", "SEED" => "0" }.freeze
# Measured margins at that cell (seeds 0/1/2 above). BP beats the frozen
# control by ~.76 and DFA(layer) tracks BP to within .012, so these
# thresholds sit far under the observed effects rather than at seed 0's
# exact value.
BP_GAP      = 0.06
FROZEN_EDGE = 0.20
BP_PRECOND  = 0.30

def run_ssm(extra_env, run_dir)
  env = { "STEPS" => "5", "SEED" => "0" }.merge(extra_env)
  env = env.merge("TAO_RUN_DIR" => run_dir, "TOY_RUN_ID" => "ssm-gate") if run_dir
  out, st = Open3.capture2e(env, RUNNER, chdir: ROOT)
  abort "ssm_gate: runner exited #{st.exitstatus}:\n#{out.lines.last(10).join}" unless st.success?
  out
end

def curve(out)
  out.lines.select { |l| l.start_with?("step ") }
end

def val_acc(out)
  line = out.lines.find { |l| l.start_with?("val: ") }
  raise "ssm_gate: no val line in\n#{out}" unless line
  line[/acc=([0-9.eE+-]+)/, 1].to_f
end

def graph_nodes(out)
  line = out.lines.find { |l| l.start_with?("graph: ") }
  raise "ssm_gate: no graph line in\n#{out}" unless line
  line[/nodes=(\d+)/, 1].to_i
end

unless File.executable?(RUNNER)
  build_out, build_st = Open3.capture2e("make", "-C", ROOT, "libexec/toy-train-ssm")
  unless build_st.success? && File.executable?(RUNNER)
    warn "ssm_gate: build failed:\n#{build_out.lines.last(15).join}"
    exit 2
  end
end

failures = []
# d878143: every leg records the failure count at its START in `n0` and
# summarises with `failures.length == n0`, so each leg reports on ITS
# OWN assertions.
n0 = 0

# ---- 1. byte fixture + chain byte-null ----
n0 = failures.length
base_out   = run_ssm({}, nil)
base_curve = curve(base_out)
if File.file?(FIXTURE)
  expect = File.readlines(FIXTURE).reject { |l| l.start_with?("#") || l.strip.empty? }
  failures << "fixture: default curve != train_ssm_baseline.txt\ngot:  #{base_curve.join}want: #{expect.join}" unless base_curve == expect
else
  failures << "fixture: #{FIXTURE} missing"
end
chain_out = run_ssm({ "SSM_POLICY" => "chain,chain" }, nil)
failures << "chain byte-null: explicit all-chain curve != absent-policy curve" unless curve(chain_out) == base_curve
puts failures.length == n0 ? "  ok: default curve matches the fixture AND an explicit all-chain policy is byte-null" : "  FAIL: fixture / chain byte-null"

# ---- 2. determinism ----
n0 = failures.length
d1 = run_ssm({ "SSM_POLICY" => "dfa,dfa", "STEPS" => "20" }, nil)
d2 = run_ssm({ "SSM_POLICY" => "dfa,dfa", "STEPS" => "20" }, nil)
failures << "determinism: two identical dfa runs differ" unless d1 == d2
puts failures.length == n0 ? "  ok: two identical dfa runs are byte-identical" : "  FAIL: determinism"

# ---- 3. forward-identity, then divergence ----
# tnn_detach is forward-identity, so a dfa build computes the SAME
# forward pass as a chain build — which is what makes their numbers
# comparable at all. Steps 2+ must then differ, or the backward is not
# actually different and the "arm" is a relabelling.
n0 = failures.length
ch20 = run_ssm({ "STEPS" => "20" }, nil)
failures << "forward identity: step 1 differs between dfa and chain — tnn_detach is supposed to be forward-identity, so the two arms are not comparable" unless curve(d1).first == curve(ch20).first
failures << "arm effect: the whole dfa curve equals chain — the backward is not different" if curve(d1) == curve(ch20)
fz = run_ssm({ "SSM_POLICY" => "frozen,frozen", "STEPS" => "20" }, nil)
failures << "arm effect: frozen curve identical to chain" if curve(fz) == curve(ch20)
puts failures.length == n0 ? "  ok: step 1 is byte-identical to BP (detach is forward-identity) and steps 2+ diverge" : "  FAIL: forward identity / arm effect"

# ---- 4. the B seed moves the dfa curve, and is inert on chain ----
# toy#158's discipline, and the ONLY cheap proof the random feedback
# reaches the weights. This lane cannot use cos(g_dfa, g_bp): its DFA
# update arrives through autodiff from the surrogate roots, so it lands
# in the same accumulator a BP run would use and there is no second
# tensor to compare against.
n0 = failures.length
dfa_b999   = run_ssm({ "SSM_POLICY" => "dfa,dfa", "STEPS" => "20", "SSM_B_SEED" => "999" }, nil)
chain_b999 = run_ssm({ "STEPS" => "20", "SSM_B_SEED" => "999" }, nil)
failures << "b seed: --dfa-b-seed does not move the dfa curve — the feedback matrix is not reaching the weights" if curve(dfa_b999) == curve(d1)
failures << "b seed: --dfa-b-seed moved the CHAIN curve — a pure-BPTT arm must not see the feedback matrix" unless curve(chain_b999) == curve(ch20)
puts failures.length == n0 ? "  ok: the B seed moves dfa and is inert on chain (toy#158)" : "  FAIL: b seed"

# ---- 5. wiring + bundle ----
n0 = failures.length
Dir.mktmpdir("ssm_gate") do |dir|
  out = run_ssm({ "SSM_POLICY" => "dfa,frozen", "STEPS" => "10",
                  "SSM_DFA_CUT" => "step", "SSM_B_SEED" => "42" }, dir)
  failures << "bundle: runner printed no val line" unless out.lines.any? { |l| l.start_with?("val: ") }
  failures << "bundle: runner printed no graph line" unless out.lines.any? { |l| l.start_with?("graph: ") }
  # The tap counts are the structural assertion. Under the STEP cut a
  # dfa layer taps its state at every step; it taps its OUTPUT only
  # where that output can reach the readout — for a non-final layer,
  # every step. Layer 0 is dfa here, so: 64 state taps, 64 output taps.
  wire = out.lines.find { |l| l.start_with?("ssm: ") }
  if wire.nil?
    failures << "bundle: no wiring line"
  else
    failures << "wiring: dfa_wired != 1 (#{wire.strip})" unless wire.include?("dfa_wired=1")
    failures << "wiring: frozen != 1 (#{wire.strip})" unless wire.include?("frozen=1")
    failures << "wiring: cut != step (#{wire.strip})" unless wire.include?("cut=step")
    failures << "wiring: tap_h != 64 — the step cut must tap the state at EVERY step (#{wire.strip})" unless wire.include?("tap_h=64")
    failures << "wiring: tap_o != 64 (#{wire.strip})" unless wire.include?("tap_o=64")
  end
  ev_path = File.join(dir, "events.jsonl")
  if File.file?(ev_path)
    events = File.readlines(ev_path).map { |l| JSON.parse(l) }
    rs = events.first || {}
    failures << "bundle: first event not run_start" unless rs["kind"] == "run_start"
    failures << "bundle: schema != toy/v1" unless rs["schema"] == "toy/v1"
    failures << "bundle: last event not run_end" unless events.last && events.last["kind"] == "run_end"
    md = rs["model"] || {}
    failures << "bundle: model.arch != ssm (#{md['arch'].inspect})" unless md["arch"] == "ssm"
    failures << "bundle: model.selection missing" unless md["selection"] == "selective"
    failures << "bundle: model.readout != last_step — a mean-pool readout would delete the memory requirement this lane exists to test" unless md["readout"] == "last_step"
    co = rs["cost"] || {}
    failures << "bundle: cost fields not positive" unless %w[total_params active_params graph_nodes].all? { |k| co[k].is_a?(Numeric) && co[k] > 0 }
    df = rs["dfa"]
    if df.nil?
      failures << "bundle: run_start has no dfa object (this lane is NOT franken)"
    else
      failures << "bundle: dfa.policy != [1,2] (got #{df['policy'].inspect})" unless df["policy"] == [1, 2]
      failures << "bundle: dfa.cut != step" unless df["cut"] == "step"
      failures << "bundle: dfa.b_seed != 42" unless df["b_seed"] == 42
      failures << "bundle: dfa.dfa_wired != 1" unless df["dfa_wired"] == 1
      failures << "bundle: dfa.frozen != 1" unless df["frozen"] == 1
    end
    failures << "bundle: #{events.count { |e| e['kind'] == 'step' }} step events (want 10)" unless events.count { |e| e["kind"] == "step" } == 10
    evs = events.select { |e| e["kind"] == "eval" }
    if evs.length != 1
      failures << "bundle: #{evs.length} eval events (want 1)"
    else
      a = evs.first["accuracy"]
      failures << "bundle: eval accuracy #{a.inspect} not in [0,1]" unless a.is_a?(Numeric) && a >= 0.0 && a <= 1.0
    end
  else
    failures << "bundle: no events.jsonl"
  end
  fj = File.join(dir, "flow.json")
  if File.file?(fj)
    flow = JSON.parse(File.read(fj)) rescue nil
    failures << "bundle: flow.json invalid" if flow.nil? || flow["format"] != "toy/v1" || !flow["nodes"].is_a?(Array) || flow["nodes"].empty?
  else
    failures << "bundle: no flow.json"
  end
end
puts failures.length == n0 ? "  ok: bundle structure + the tap counts the cut asks for" : "  FAIL: wiring / bundle"

# ---- 6 + 7. the success bar, and the ticket's LINEAR control ----
n0 = failures.length
rows = {}
%w[selective lti].each do |sel|
  r = {}
  r["chain"]  = val_acc(run_ssm(CELL.merge("SSM_SELECTION" => sel, "SSM_POLICY" => "chain,chain"), nil))
  r["dfa"]    = val_acc(run_ssm(CELL.merge("SSM_SELECTION" => sel, "SSM_POLICY" => "dfa,dfa",
                                           "SSM_DFA_CUT" => "layer"), nil))
  r["step"]   = val_acc(run_ssm(CELL.merge("SSM_SELECTION" => sel, "SSM_POLICY" => "dfa,dfa",
                                           "SSM_DFA_CUT" => "step"), nil))
  r["frozen"] = val_acc(run_ssm(CELL.merge("SSM_SELECTION" => sel, "SSM_POLICY" => "frozen,frozen"), nil))
  rows[sel] = r
  puts format("    %-10s chain=%.4f dfa(layer)=%.4f dfa(step)=%.4f frozen=%.4f",
              sel, r["chain"], r["dfa"], r["step"], r["frozen"])
end

sv = rows["selective"]
if sv["chain"] < sv["frozen"] + BP_PRECOND
  failures << "SUCCESS BAR (precondition): BP #{sv['chain']} does not beat frozen #{sv['frozen']} by #{BP_PRECOND} — the cell cannot discriminate"
end
if sv["dfa"] < sv["chain"] - BP_GAP
  failures << "SUCCESS BAR (BP gap): dfa(layer) #{sv['dfa']} is more than #{BP_GAP} below chain #{sv['chain']}"
end
if sv["dfa"] < sv["frozen"] + FROZEN_EDGE
  failures << "SUCCESS BAR (frozen control): dfa(layer) #{sv['dfa']} does not beat frozen #{sv['frozen']} by #{FROZEN_EDGE}"
end
puts failures.length == n0 ?
  "  ok: SUCCESS BAR — DFA at the LAYER cut matches BP and crushes the frozen control (matched init + seed)" :
  "  FAIL: success bar"

n0 = failures.length
lt = rows["lti"]
# HALF ONE: in the LINEAR model, cutting BPTT is FREE. This is the
# ticket's own prediction, and it doubles as the proof that the step cut
# is correctly wired — a dead cut could not learn at all.
if (lt["step"] - lt["dfa"]).abs > 0.12
  failures << "LINEAR CONTROL: under lti, dfa(step) #{lt['step']} and dfa(layer) #{lt['dfa']} differ by more than 0.12 — the ticket's prediction is that cutting BPTT is FREE in the purely-linear case, and this lane's whole reading depends on it"
end
if lt["step"] < lt["frozen"] + 0.15
  failures << "LINEAR CONTROL: under lti, dfa(step) #{lt['step']} does not beat frozen #{lt['frozen']} — the step cut is not learning at all, so the selective-case collapse below would just be a broken build"
end
# HALF TWO: in the SELECTIVE model, cutting BPTT is CATASTROPHIC.
if sv["step"] > sv["frozen"] + 0.25
  failures << "LINEAR CONTROL: under selective, dfa(step) #{sv['step']} is doing far better than the frozen control #{sv['frozen']} — this lane's finding is that the per-step cut COLLAPSES once selection is on (measured .250/.250/.238 at seeds 0/1/2, best over an LR sweep .355). If that changed, re-read docs/roadmap/dfa-arch-program-2026-08-10.md before re-baselining."
end
puts failures.length == n0 ?
  "  ok: LINEAR CONTROL — cutting BPTT is FREE under lti and CATASTROPHIC under selection (the ticket's caveat, measured both ways)" :
  "  FAIL: linear control"

# ---- 8. the degenerate task, measured ----
# `mean` spreads the class signal over every step, so neither memory nor
# selection is needed and a frozen random recurrence plus a trained head
# already integrates it. This lane's `blobs`.
n0 = failures.length
mean_fz = val_acc(run_ssm(CELL.merge("SSM_TASK" => "mean", "SSM_POLICY" => "frozen,frozen"), nil))
cue_fz  = rows["selective"]["frozen"]
failures << "degenerate task: --task mean frozen #{mean_fz} is not far above the cue task's frozen #{cue_fz} — `mean` is documented as solvable without memory OR selection, and the default `cue` task is justified against that measurement" unless mean_fz > cue_fz + 0.25
puts failures.length == n0 ? "  ok: --task mean is measurably degenerate (a FROZEN recurrence solves it), which is why `cue` is the default" : "  FAIL: degenerate task"

# ---- 9. fail-loud ----
n0 = failures.length
[
  [{ "SSM_POLICY" => "chain,chain,chain" },       "policy longer than SSM_LAYERS"],
  [{ "SSM_POLICY" => "bogus,chain" },             "unknown policy token"],
  [{ "SSM_POLICY" => "chain,dfa" },               "a chain layer BELOW a dfa layer (silently frozen)"],
  [{ "SSM_SELECTION" => "mamba" },                "unknown SSM_SELECTION"],
  [{ "SSM_DFA_CUT" => "token" },                  "unknown SSM_DFA_CUT"],
  [{ "SSM_TASK" => "copy" },                      "unknown SSM_TASK"],
  [{ "SSM_CLASSES" => "1" },                      "SSM_CLASSES=1"],
  [{ "SSM_LAYERS" => "0" },                       "SSM_LAYERS=0"],
  [{ "SSM_CUE_SPAN" => "64" },                    "cue span == sequence length (no delay left to carry)"],
].each do |env, label|
  out, st = Open3.capture2e({ "STEPS" => "2" }.merge(env), RUNNER, chdir: ROOT)
  failures << "fail-loud: #{label} exited 0 (silently did nothing):\n#{out.lines.last(2).join}" if st.success?
end
puts failures.length == n0 ? "  ok: 9 degenerate configs all fail loud instead of quietly doing nothing" : "  FAIL: fail-loud"

# ---- 10. CLI ----
n0 = failures.length
Dir.mktmpdir("ssm_gate_cli") do |dir|
  out, st = Open3.capture2e({ "SPINEL_SKIP_PIN_CHECK" => nil }, TOY, "train", "ssm",
                            "--steps", "5", "--seed", "0", "--out", dir, chdir: ROOT)
  if st.success?
    failures << "cli: `toy train ssm` curve != the runner's default curve" unless curve(out) == base_curve
    failures << "cli: `toy train ssm` did not echo the val line" unless out.lines.any? { |l| l.start_with?("val: ") }
  else
    failures << "cli: `toy train ssm` exited #{st.exitstatus}:\n#{out.lines.last(5).join}"
  end
  [
    [%w[ssm --align-events],       "ssm accepted --align-events (its DFA update lands in the SAME accumulator a BP run uses, so a cosine would mean nothing)"],
    [%w[ssm --policy-scope ffn],   "ssm accepted --policy-scope"],
    [%w[ssm --task blobs],         "ssm accepted --task blobs"],
    [%w[mlp --dfa-cut step],       "mlp accepted --dfa-cut"],
    [%w[gnn --selection lti],      "gnn accepted --selection"],
    [%w[franken --seq 32],         "franken accepted --seq"],
  ].each do |argv, label|
    sout, sst = Open3.capture2e({}, TOY, "train", *argv, chdir: ROOT)
    failures << "cli: #{label} (exit #{sst.exitstatus}: #{sout.lines.last(1).join.strip})" unless sst.exitstatus == 2
  end
  cout, cst = Open3.capture2e({}, TOY, "train", "ssm", "--device", "cuda", chdir: ROOT)
  failures << "cli: ssm accepted --device cuda (CPU-only this slice — tao#19)" unless cst.exitstatus == 2
end
puts failures.length == n0 ? "  ok: CLI reproduces the curve; lane-foreign flags, --align-events and --device cuda are rejected" : "  FAIL: CLI"

if failures.empty?
  puts "GATE PASS [ssm]: selective scan + per-layer policy — byte fixture, forward-identity under detach, the B seed, the tap counts, the MANDATORY success bar at the layer cut, and the ticket's LINEAR control showing the per-step BPTT cut is free under lti and catastrophic under selection (toy#155)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [ssm]: #{f}" }
  exit 1
end
