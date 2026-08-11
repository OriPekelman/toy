#!/usr/bin/env ruby
# prep/gnn_gate.rb — toy#153 (DFA-arch T1) gate for the GNN
# node-classification lane (libexec/toy-train-gnn, `toy train gnn`).
#
# Legs:
#   1. BYTE FIXTURE + CHAIN BYTE-NULL: the default 5-step curve equals
#      prep/fixtures/train_gnn_baseline.txt, and an EXPLICIT all-chain
#      policy is byte-identical to no policy at all.
#   2. DETERMINISM: two identical dfa runs → identical stdout.
#   3. ARM EFFECT: dfa and frozen both differ from chain.
#   4. WIRING (structure, not curves): run_start.dfa reports the exact
#      dfa_wired / frozen / feedback_hops the policy asked for.
#   5. BUNDLE: JSONL shape, align events WITH `wname` (tao#19 item 3),
#      the eval event carrying BOTH sides of the split, run_end last.
#  5b. THE ALIGNMENT PHASE: cos(g_dfa, g_bp) climbs from ~0.
#   6. THE STRUCTURE ROUTE IS WIRED THROUGH THE ADJACENCY — the sharp
#      one, and the reason this lane exists at all (see below).
#   7. THE MANDATORY SUCCESS BAR, on CORA (see below).
#   8. THE SYNTHETIC GRAPH'S OWN LIMIT, measured not assumed.
#   9. FAIL-LOUD: bad tokens, silent-ignore traps, degenerate shapes.
#  10. CLI: `toy train gnn` reproduces the curve; lane-foreign flags and
#      --device cuda are rejected.
#
# ── WHY THE SUCCESS BAR RUNS ON CORA AND NOT ON THE SEEDED GRAPH ──
#
# tao#19 item 4 makes this bar MANDATORY in every lane:
#     positive = all-DFA within the stated gap of all-BP
#                AND provably beating the frozen control.
# Its first half presupposes that BP itself beats the frozen control —
# otherwise the cell says nothing about anybody
# ([[control-arm-must-be-able-to-lose]]). On this lane's SEEDED graph
# that presupposition is FALSE, and that is measured, not feared.
# BP minus frozen, val accuracy, at the default cell:
#
#   features 64,  300 steps: seed0 +.086  seed1 +.033  seed2 +.020
#   features 64,  600 steps: seed0 +.034  seed1 -.012  seed2 +.021
#   features 128, 300 steps: seed0 +.062  seed1 -.008  seed2 -.051
#   features 128, 600 steps: seed0 +.019  seed1 -.011  seed2 -.002
#
# It flips sign. In a GNN, neighbourhood aggregation is ARCHITECTURE,
# not learning — a frozen random hidden stack still smooths features
# over the graph — so on a synthetic graph whose labels are largely
# recoverable from that smoothing, training the hidden layers buys
# almost nothing and a gate stated there would be measuring noise.
#
# On CORA it is robustly positive (+.079 / +.071 / +.119 at seeds
# 0/1/2), because Cora's 1433-dim bag-of-words forces the hidden layer
# to COMPRESS and a random projection loses what a learned one keeps.
# So the bar is stated on the graph the ticket names, and the seeded
# graph keeps the cheap structural legs plus leg 8, which pins its own
# limitation so a future change that quietly makes it discriminate gets
# noticed rather than silently promoted.
#
# The Cora bundle is a Makefile PREREQUISITE (`ruby prep/fetch_cora.rb`,
# one 168 KB download, cached in data/). If it is missing this gate
# FAILS with the fix rather than skipping — a bar that can be skipped is
# not mandatory.

ROOT    = File.expand_path("..", __dir__)
RUNNER  = File.join(ROOT, "libexec", "toy-train-gnn")
TOY     = File.join(ROOT, "bin", "toy")
FIXTURE = File.join(ROOT, "prep", "fixtures", "train_gnn_baseline.txt")
CORA    = File.join(ROOT, "data", "gnn_cora")

require "open3"
require "json"
require "tmpdir"
require "fileutils"

# The Cora cell: the canonical 2-layer GCN (1 hidden layer + head),
# width 64, lr 0.01 — which is BP's OWN best on this graph (BP val
# .351 / .661 / .697 / .668 / .593 at lr .001 / .003 / .01 / .03 / .1),
# so "DFA beats BP" here cannot be dismissed as an under-tuned BP.
# 100 steps: both arms peak there and decay after, so a longer run
# would only measure who overfits 140 labels faster.
CORA_CELL = { "GNN_GRAPH" => CORA, "GNN_LAYERS" => "1", "GNN_HIDDEN" => "64",
              "GNN_LR" => "0.01", "STEPS" => "100", "SEED" => "0" }.freeze
# Measured at that cell, seeds 0/1/2:
#   chain  .697 / .661 / .643
#   dfa    .762 / .759 / .760
#   frozen .618 / .590 / .524
# DFA is ABOVE BP on every seed, so BP_GAP is a floor DFA clears with
# room; the thresholds sit ~3x under the observed margins rather than at
# seed 0's exact value.
BP_GAP        = 0.05   # dfa must not fall this far below chain
FROZEN_EDGE   = 0.05   # dfa must beat frozen by this much
BP_PRECOND    = 0.04   # ...and BP must beat frozen first, or the row is mute

def run_gnn(extra_env, run_dir)
  env = { "STEPS" => "5", "SEED" => "0" }.merge(extra_env)
  env = env.merge("TAO_RUN_DIR" => run_dir, "TOY_RUN_ID" => "gnn-gate") if run_dir
  out, st = Open3.capture2e(env, RUNNER, chdir: ROOT)
  abort "gnn_gate: runner exited #{st.exitstatus}:\n#{out.lines.last(10).join}" unless st.success?
  out
end

def curve(out)
  out.lines.select { |l| l.start_with?("step ") }
end

def val_acc(out)
  line = out.lines.find { |l| l.start_with?("val: ") }
  raise "gnn_gate: no val line in\n#{out}" unless line
  line[/acc=([0-9.eE+-]+)/, 1].to_f
end

unless File.executable?(RUNNER)
  build_out, build_st = Open3.capture2e("make", "-C", ROOT, "libexec/toy-train-gnn")
  unless build_st.success? && File.executable?(RUNNER)
    warn "gnn_gate: build failed:\n#{build_out.lines.last(15).join}"
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
base_out   = run_gnn({}, nil)
base_curve = curve(base_out)
if File.file?(FIXTURE)
  expect = File.readlines(FIXTURE).reject { |l| l.start_with?("#") || l.strip.empty? }
  failures << "fixture: default curve != train_gnn_baseline.txt\ngot:  #{base_curve.join}want: #{expect.join}" unless base_curve == expect
else
  failures << "fixture: #{FIXTURE} missing"
end
chain_out = run_gnn({ "GNN_POLICY" => "chain" }, nil)
failures << "chain byte-null: explicit all-chain curve != absent-policy curve" unless curve(chain_out) == base_curve
puts failures.length == n0 ? "  ok: default curve matches the fixture AND an explicit all-chain policy is byte-null" : "  FAIL: fixture / chain byte-null"

# ---- 2. determinism ----
n0 = failures.length
d1 = run_gnn({ "GNN_POLICY" => "dfa", "STEPS" => "20" }, nil)
d2 = run_gnn({ "GNN_POLICY" => "dfa", "STEPS" => "20" }, nil)
failures << "determinism: two identical dfa runs differ" unless d1 == d2
puts failures.length == n0 ? "  ok: two identical dfa runs are byte-identical (curve + train/val)" : "  FAIL: determinism"

# ---- 3. arm effect + THE B SEED MOVES THE CURVE ----
# toy#158's discipline: changing --dfa-b-seed is the cheapest proof that
# the random feedback actually reaches the weights. If B were computed
# and then dropped, the dfa arm would still differ from chain (it takes a
# different code path) and would still look like "DFA did something" —
# only the seed test separates those. It must be INERT on chain, which
# is the other half of the same assertion.
n0 = failures.length
fz   = run_gnn({ "GNN_POLICY" => "frozen", "STEPS" => "20" }, nil)
ch20 = run_gnn({ "STEPS" => "20" }, nil)
failures << "arm effect: dfa curve identical to chain" if curve(d1) == curve(ch20)
failures << "arm effect: frozen curve identical to chain" if curve(fz) == curve(ch20)
dfa_b999   = run_gnn({ "GNN_POLICY" => "dfa", "STEPS" => "20", "GNN_B_SEED" => "999" }, nil)
chain_b999 = run_gnn({ "STEPS" => "20", "GNN_B_SEED" => "999" }, nil)
failures << "b seed: --dfa-b-seed does not move the dfa curve — the feedback matrix is not reaching the weights" if curve(dfa_b999) == curve(d1)
failures << "b seed: --dfa-b-seed moved the CHAIN curve — a pure-backprop arm must not see the feedback matrix at all" unless curve(chain_b999) == curve(ch20)
puts failures.length == n0 ? "  ok: dfa and frozen move the curve off chain, and the B seed moves dfa while staying inert on chain (toy#158)" : "  FAIL: arm effect / b seed"

# ---- 4 + 5. wiring + bundle ----
# Two hidden layers here (the default is one) so a MIXED policy has
# something to mix: dfa on layer 0, frozen on layer 1.
n0 = failures.length
Dir.mktmpdir("gnn_gate") do |dir|
  out = run_gnn({ "GNN_LAYERS" => "2", "GNN_POLICY" => "dfa,frozen",
                  "STEPS" => "10", "GNN_ALIGN" => "1", "GNN_B_SEED" => "42",
                  "GNN_FEEDBACK_ROUTE" => "structure",
                  "GNN_FEEDBACK_HOPS" => "2" }, dir)
  failures << "bundle: runner printed no val line" unless out.lines.any? { |l| l.start_with?("val: ") }
  failures << "bundle: runner printed no train line" unless out.lines.any? { |l| l.start_with?("train: ") }
  ev_path = File.join(dir, "events.jsonl")
  if File.file?(ev_path)
    events = File.readlines(ev_path).map { |l| JSON.parse(l) }
    rs = events.first || {}
    failures << "bundle: first event not run_start" unless rs["kind"] == "run_start"
    failures << "bundle: schema != toy/v1" unless rs["schema"] == "toy/v1"
    failures << "bundle: last event not run_end" unless events.last && events.last["kind"] == "run_end"
    md = rs["model"] || {}
    failures << "bundle: model.arch != gnn (#{md['arch'].inspect})" unless md["arch"] == "gnn"
    failures << "bundle: model.propagation missing — a consumer cannot tell this from an MLP without it" unless md["propagation"].is_a?(String)
    co = rs["cost"] || {}
    failures << "bundle: cost fields not positive (#{co.inspect})" unless %w[total_params active_params flops_per_token].all? { |k| co[k].is_a?(Numeric) && co[k] > 0 }
    # Message passing is PARAMETER-FREE, so a params-only cost reads
    # this lane as far cheaper than it is. The separate field is what
    # stops a consumer from comparing it to a dense MLP's flops.
    failures << "bundle: cost.propagation_flops_per_step missing/zero" unless co["propagation_flops_per_step"].is_a?(Numeric) && co["propagation_flops_per_step"] > 0
    cfg = rs["config"] || {}
    failures << "bundle: config n_train + n_val != nodes (#{cfg.inspect})" unless cfg["n_train"].to_i + cfg["n_val"].to_i == cfg["nodes"].to_i
    failures << "bundle: config.graph != synthetic" unless cfg["graph"] == "synthetic"
    # --- leg 4: the WIRING, asserted on structure rather than curves.
    df = rs["dfa"]
    if df.nil?
      failures << "bundle: run_start has no dfa object (this lane is NOT franken — it must carry its own provenance)"
    else
      failures << "bundle: dfa.policy != [1,2] (got #{df['policy'].inspect})" unless df["policy"] == [1, 2]
      failures << "bundle: dfa.b_seed != 42" unless df["b_seed"] == 42
      failures << "bundle: dfa.dfa_wired != 1 (got #{df['dfa_wired'].inspect}) — an unwired dfa layer looks exactly like 'DFA did nothing'" unless df["dfa_wired"] == 1
      failures << "bundle: dfa.frozen != 1 (got #{df['frozen'].inspect})" unless df["frozen"] == 1
      failures << "bundle: dfa.feedback != structure" unless df["feedback"] == "structure"
      failures << "bundle: dfa.feedback_hops != 2 (got #{df['feedback_hops'].inspect})" unless df["feedback_hops"] == 2
    end
    steps_ev = events.count { |e| e["kind"] == "step" }
    failures << "bundle: #{steps_ev} step events (want 10)" unless steps_ev == 10
    aligns = events.select { |e| e["kind"] == "align" }
    failures << "bundle: #{aligns.length} align events (want 10 = 1 dfa layer x 10 steps; the frozen layer emits none)" unless aligns.length == 10
    bad = aligns.count do |e|
      c = e["cos"]
      !c.is_a?(Numeric) || c.to_f.nan? || c.to_f.abs > 1.0001 ||
        !e["dfa_norm"].is_a?(Numeric) || e["dfa_norm"] <= 0 ||
        !e["bp_norm"].is_a?(Numeric) || e["bp_norm"] <= 0
    end
    failures << "bundle: #{bad} malformed align events (cos out of range, or a zero norm = dead download vs dead gradient)" unless bad == 0
    # tao#19 item 3: ingest keys on `wname`, NOT on a per-lane wi table.
    names = aligns.map { |e| e["wname"] }.uniq.sort
    failures << "bundle: align wnames #{names.inspect} (want [\"w1\"])" unless names == %w[w1]
    failures << "bundle: align events missing lane-local wi" unless aligns.all? { |e| e["wi"] == 0 }
    evs = events.select { |e| e["kind"] == "eval" }
    if evs.length != 1
      failures << "bundle: #{evs.length} eval events (want 1)"
    else
      a = evs.first["accuracy"]
      failures << "bundle: eval accuracy #{a.inspect} not in [0,1]" unless a.is_a?(Numeric) && a >= 0.0 && a <= 1.0
      # BOTH sides of the split ride the eval event: on this lane the
      # arms separate as much by their train/val GAP as by val accuracy.
      failures << "bundle: eval event carries no train_accuracy" unless evs.first["train_accuracy"].is_a?(Numeric)
      failures << "bundle: eval train_n + n != nodes" unless evs.first["train_n"].to_i + evs.first["n"].to_i == cfg["nodes"].to_i
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

# ---- 5b. ALIGNMENT — AND THE FINDING THAT IT DOES NOT HAPPEN HERE ----
#
# toy#152's MLP anchor reproduced Refinetti et al.'s "align, then
# memorise": cos(g_dfa, g_bp) climbed from ~0 to 0.6 and STAYED there,
# and its gate asserts that. This lane does NOT do that, and the
# difference is the most interesting thing the lane measured — so it is
# pinned rather than papered over. w1 cos, 300 steps:
#
#   seeded graph  seed0  -.033 -> PEAK +.359 @61 -> -.124
#                 seed1  -.050 -> PEAK +.351 @51 -> -.111
#                 seed2  +.046 -> PEAK +.046 @1  -> -.111   (no phase at all)
#   cora          seed0  +.065 -> PEAK +.096 @21 -> -.057
#
# So an alignment phase is at best transient and is not even present on
# every seed, and on Cora — where DFA BEATS BP by 6.4 points — cos never
# leaves the noise and ends NEGATIVE. Whatever makes DFA work on this
# architecture, it is NOT that the DFA update approximates the BP one.
# That is the ticket's own skeptic hypothesis (an implicit regulariser
# rather than DFA-approximating-BP) surviving its first real test.
#
# The gate therefore asserts the two things that ARE stable: the series
# MOVES (a dead telemetry path would be constant), and it does NOT
# converge towards BP. If this leg ever fails because cos rose and
# stayed, that is a FINDING, not a flake — re-read
# docs/roadmap/dfa-arch-program-2026-08-10.md before touching it.
n0 = failures.length
Dir.mktmpdir("gnn_gate_align") do |dir|
  run_gnn({ "GNN_POLICY" => "dfa", "STEPS" => "300",
            "GNN_ALIGN" => "1", "GNN_ALIGN_EVERY" => "10" }, dir)
  series = File.readlines(File.join(dir, "events.jsonl"))
             .map { |l| JSON.parse(l) }
             .select { |e| e["kind"] == "align" && e["wname"] == "w1" }
             .sort_by { |e| e["step"] }
  if series.length < 8
    failures << "alignment: only #{series.length} align events for w1"
  else
    cosines = series.map { |e| e["cos"] }
    span = cosines.max - cosines.min
    failures << "alignment: cos(g_dfa, g_bp) is CONSTANT to within #{span.round(4)} over 300 steps — a live shadow gradient moves; a dead download does not" unless span > 0.1
    failures << "alignment: cos ENDED at #{cosines.last.round(4)}, i.e. DFA converged towards the BP gradient — that is toy#152's MLP behaviour and NOT what this lane measured. Read the roadmap before re-baselining: the lane's headline (DFA beats BP on Cora) currently rests on alignment NOT being the mechanism." unless cosines.last < 0.1
  end
end
puts failures.length == n0 ? "  ok: align telemetry live AND cos does not converge to BP — the mechanism here is not 'align, then memorise' (contrast toy#152)" : "  FAIL: alignment"

# ---- 6. the structure route really goes through the adjacency ----
#
# THE SHARP LEG. `--feedback-route structure` propagates the error along
# S-hat before projecting it, which is DFA-GNN's whole mechanism and the
# only place graph structure touches the credit path (the propagation
# sits BEFORE the weight in each layer, so nothing else in the DFA
# backward sees the graph). An implementation that computed the hops
# but dropped them, or that perturbed the run some other way, would
# still "move the curve" — so curve-watching cannot gate this.
#
# The identity that CAN: at --degree 0 the graph is edgeless, S-hat is
# the IDENTITY, and S-hat^k e == e for every k. So `structure` must be
# BYTE-IDENTICAL to `direct` there, and must differ once edges exist.
# One assertion pins both "the hops are applied" and "they are applied
# through the adjacency and not through something else".
n0 = failures.length
edgeless_direct = curve(run_gnn({ "GNN_DEGREE" => "0", "GNN_POLICY" => "dfa", "STEPS" => "20" }, nil))
edgeless_struct = curve(run_gnn({ "GNN_DEGREE" => "0", "GNN_POLICY" => "dfa", "STEPS" => "20",
                                  "GNN_FEEDBACK_ROUTE" => "structure" }, nil))
graph_direct    = curve(run_gnn({ "GNN_POLICY" => "dfa", "STEPS" => "20" }, nil))
graph_struct    = curve(run_gnn({ "GNN_POLICY" => "dfa", "STEPS" => "20",
                                  "GNN_FEEDBACK_ROUTE" => "structure" }, nil))
failures << "structure route: on an EDGELESS graph (S-hat = I) structure feedback differs from direct — it is not routing through the adjacency" unless edgeless_direct == edgeless_struct
failures << "structure route: on the real graph structure feedback is identical to direct — the hops are being dropped" if graph_direct == graph_struct
puts failures.length == n0 ? "  ok: structure feedback == direct on an edgeless graph and != on a real one (the hops go through S-hat)" : "  FAIL: structure route"

# ---- 7. THE MANDATORY SUCCESS BAR, on Cora ----
n0 = failures.length
unless File.file?(CORA + ".meta.i32")
  failures << "success bar: the Cora bundle is missing — run `ruby prep/fetch_cora.rb` (one 168 KB download, cached in data/). This leg is NOT skippable: tao#19 item 4 makes the bar mandatory, and the seeded graph cannot carry it (see leg 8)."
end
if File.file?(CORA + ".meta.i32")
  cora = {}
  %w[chain dfa frozen].each { |arm| cora[arm] = val_acc(run_gnn(CORA_CELL.merge("GNN_POLICY" => arm), nil)) }
  cora["structure"] = val_acc(run_gnn(CORA_CELL.merge("GNN_POLICY" => "dfa",
                                                      "GNN_FEEDBACK_ROUTE" => "structure"), nil))
  puts format("    cora  chain=%.4f dfa=%.4f dfa(structure)=%.4f frozen=%.4f",
              cora["chain"], cora["dfa"], cora["structure"], cora["frozen"])
  # The PRECONDITION first, exactly as ctr_gate does: if BP cannot beat
  # the frozen control the row says nothing about DFA, and reporting a
  # pass would be reporting a free pass.
  if cora["chain"] < cora["frozen"] + BP_PRECOND
    failures << "SUCCESS BAR (precondition): BP #{cora['chain']} does not beat frozen #{cora['frozen']} by #{BP_PRECOND} — the cell cannot discriminate, so nothing below it means anything"
  end
  if cora["dfa"] < cora["chain"] - BP_GAP
    failures << "SUCCESS BAR (BP gap): dfa #{cora['dfa']} is more than #{BP_GAP} below chain #{cora['chain']}"
  end
  if cora["dfa"] < cora["frozen"] + FROZEN_EDGE
    failures << "SUCCESS BAR (frozen control): dfa #{cora['dfa']} does not beat frozen #{cora['frozen']} by #{FROZEN_EDGE} — 'near-BP' alone cannot tell 'DFA learned' from 'this task is easy'"
  end
end
puts failures.length == n0 ?
  "  ok: SUCCESS BAR on Cora — DFA within #{BP_GAP} of BP AND beating the frozen control, with BP beating it first (matched init + seed)" :
  "  FAIL: success bar"

# ---- 8. the SEEDED graph's own limit, pinned ----
# On the synthetic graph the DFA-beats-frozen half holds robustly
# (+.106 / +.077 / +.063 at seeds 0/1/2) while the BP-beats-frozen half
# does NOT (+.086 / +.033 / +.020, and negative at other cells). Both
# facts are asserted: the first because it is the lane's own positive,
# the second because a change that quietly made this graph
# discriminating would otherwise be silently promoted into the bar.
n0 = failures.length
syn = {}
%w[chain dfa frozen].each { |arm| syn[arm] = val_acc(run_gnn({ "STEPS" => "300", "GNN_POLICY" => arm }, nil)) }
puts format("    synth chain=%.4f dfa=%.4f frozen=%.4f", syn["chain"], syn["dfa"], syn["frozen"])
if syn["dfa"] < syn["frozen"] + FROZEN_EDGE
  failures << "seeded graph: dfa #{syn['dfa']} does not beat frozen #{syn['frozen']} by #{FROZEN_EDGE}"
end
if syn["frozen"] < 0.35
  failures << "seeded graph: the frozen control collapsed to #{syn['frozen']} — this graph's whole point is that aggregation is architecture, so a WEAK frozen arm means the task changed under us"
end
# The degenerate task, MEASURED (toy#152's `--task blobs` in this
# lane's clothing): with labels == community identity, one hop of
# smoothing solves it, a FROZEN random net already scores .92, and no
# credit-assignment rule can be told from another near that ceiling.
comm = val_acc(run_gnn({ "STEPS" => "300", "GNN_TASK" => "community", "GNN_POLICY" => "frozen" }, nil))
if comm < 0.85
  failures << "degenerate task: --task community frozen #{comm} < 0.85 — it is documented as saturated-by-construction and the default `teacher` task is justified against that measurement"
end
puts failures.length == n0 ? "  ok: seeded graph — DFA beats frozen, the frozen arm is strong (aggregation is architecture), and --task community is measurably saturated" : "  FAIL: seeded graph limits"

# ---- 9. fail-loud ----
n0 = failures.length
[
  [{ "GNN_POLICY" => "chain,chain" },                     "policy longer than GNN_LAYERS"],
  [{ "GNN_POLICY" => "bogus" },                           "unknown policy token"],
  [{ "GNN_FEEDBACK_ROUTE" => "graph" },                   "unknown feedback route"],
  [{ "GNN_FEEDBACK_HOPS" => "2" },                        "hops without structure (silent-ignore trap)"],
  [{ "GNN_GRAPH" => CORA, "GNN_NODES" => "512" },         "loaded graph + a synthetic knob"],
  [{ "GNN_CLASSES" => "1" },                              "GNN_CLASSES=1"],
  [{ "GNN_LAYERS" => "0" },                               "GNN_LAYERS=0"],
  [{ "GNN_TASK" => "cora" },                              "unknown GNN_TASK"],
  [{ "GNN_TRAIN_PER_CLASS" => "100000" },                 "a split no label distribution can support"],
].each do |env, label|
  out, st = Open3.capture2e({ "STEPS" => "2" }.merge(env), RUNNER, chdir: ROOT)
  failures << "fail-loud: #{label} exited 0 (silently did nothing):\n#{out.lines.last(2).join}" if st.success?
end
puts failures.length == n0 ? "  ok: 9 degenerate configs all fail loud instead of quietly doing nothing" : "  FAIL: fail-loud"

# ---- 10. CLI parity ----
n0 = failures.length
Dir.mktmpdir("gnn_gate_cli") do |dir|
  out, st = Open3.capture2e({ "SPINEL_SKIP_PIN_CHECK" => nil }, TOY, "train", "gnn",
                            "--steps", "5", "--seed", "0", "--out", dir, chdir: ROOT)
  if st.success?
    failures << "cli: `toy train gnn` curve != the runner's default curve" unless curve(out) == base_curve
    failures << "cli: `toy train gnn` did not echo the val line" unless out.lines.any? { |l| l.start_with?("val: ") }
  else
    failures << "cli: `toy train gnn` exited #{st.exitstatus}:\n#{out.lines.last(5).join}"
  end
  # Lane-foreign flags are rejected, both directions.
  [
    [%w[gnn --policy-scope ffn],   "gnn accepted --policy-scope"],
    [%w[gnn --val-batches 4],      "gnn accepted --val-batches (transductive: there are no val BATCHES)"],
    [%w[gnn --task blobs],         "gnn accepted --task blobs (that is the mlp lane's control)"],
    [%w[mlp --feedback-route structure], "mlp accepted --feedback-route"],
    [%w[mlp --task community],     "mlp accepted --task community (that is the gnn lane's control)"],
    [%w[ctr --graph data/x],       "ctr accepted --graph"],
  ].each do |argv, label|
    sout, sst = Open3.capture2e({}, TOY, "train", *argv, chdir: ROOT)
    failures << "cli: #{label} (exit #{sst.exitstatus}: #{sout.lines.last(1).join.strip})" unless sst.exitstatus == 2
  end
  cout, cst = Open3.capture2e({}, TOY, "train", "gnn", "--device", "cuda", chdir: ROOT)
  failures << "cli: gnn accepted --device cuda (CPU-only by decision — tao#18)" unless cst.exitstatus == 2
end
puts failures.length == n0 ? "  ok: CLI reproduces the curve; lane-foreign flags and --device cuda are rejected" : "  FAIL: CLI"

if failures.empty?
  puts "GATE PASS [gnn]: message passing + per-layer policy — byte fixture + chain byte-null, determinism, wiring, align/wname bundle, the structure route pinned by the edgeless identity, the MANDATORY success bar on Cora, and the seeded graph's own measured limit (toy#153)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [gnn]: #{f}" }
  exit 1
end
