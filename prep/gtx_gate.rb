#!/usr/bin/env ruby
# prep/gtx_gate.rb — toy#160 (DFA-arch T4) gate for the GRAPH TRANSFORMER
# lane (libexec/toy-train-gtx, `toy train gtx`).
#
# Legs:
#   1. BYTE FIXTURE + CHAIN BYTE-NULL.
#   2. DETERMINISM.
#   3. FORWARD-IDENTITY + arm effect.
#   4. THE B SEED MOVES THE DFA CURVE and is INERT on chain.
#   5. WIRING + BUNDLE, including the SMALL-HEAD assertion.
#   6. THE MANDATORY SUCCESS BAR, each arm at ITS OWN best LR.
#   7. THE MIXING CUT collapses — attention has to be learned THROUGH
#      the gradient, and that is the lane's second finding.
#   8. THE TWO CONTROLS THAT MAKE THE LANE MEAN ANYTHING: frozen must
#      LOSE, and `--task local` must be measurably degenerate.
#   9. FAIL-LOUD.
#  10. CLI.
#
# ── WHAT THIS LANE ANSWERS ──
#
# The program's one unresolved negative is the transformer LM — and it
# CONFOUNDS two things, because every LM run was DFA at a ~50k-vocab
# output while F13/F18 showed DFA's alignment collapses as the output
# dimension grows. This lane keeps the attention and shrinks the head to
# 16 relation classes. Measured, 1500 steps, EACH ARM AT ITS OWN BEST LR,
# seeds 0/1/2:
#
#   chain (BP)      lr 0.003   .983 / .990 / .981     mean .985
#   dfa (layer cut) lr 0.001   .872 / .947 / .940     mean .920
#   dfa (step cut)  lr 0.001   .127 / .154 / .165     mean .149
#   frozen          lr 0.003   .090 / .138 / .105     mean .111
#
# **ATTENTION IS NOT DFA-HOSTILE.** Block-tap DFA reaches 93% of BP at a
# small output dim while beating the frozen control by .81. So the
# transformer-LM negative was about the OUTPUT DIMENSION, not about
# attention — which is what F22 needed to know before choosing between
# keeping attention and going pure message-passing.
#
# ── WHY EACH ARM GETS ITS OWN LR, AND WHY THAT IS NOT GENEROSITY ──
#
# At BP's own rate (0.01) the DFA arm reads .064 — CHANCE. At 0.001 it
# reads .920. A single-LR matrix here would have published "attention is
# DFA-hostile", the exact opposite of the truth, and it would have been
# quoted against the whole graph-LLM route. DFA tolerating less LR than
# BP is toy#152's finding restated, and this lane is where ignoring it
# would have been most expensive.
#
# ── THE CONTROLS TOOK THREE MEASUREMENTS TO GET RIGHT ──
#
# A structural attention mask aggregates each node's neighbourhood for
# FREE, so a frozen random transformer is already a neighbourhood
# averager. Three successive builds of the task were degenerate and each
# was caught by the frozen control rather than by inspection:
#   (a) verbatim key match          -> frozen 1.000 (random projections
#       preserve inner products, so retrieval needed no learning)
#   (b) permuted key, pair-level split -> frozen .957 (the head memorised
#       each entity's fingerprint; the split leaked)
#   (c) permuted key, entity-level split -> BP itself stuck at .246
#       (memorising 36 entities was cheaper than learning the rule)
# Only (d) — permuted key with the graph's CONTENT redrawn every step —
# leaves nothing to memorise and forces the retrieval. See
# lib/toy/io/toy_gtx_task.rb.

ROOT    = File.expand_path("..", __dir__)
RUNNER  = File.join(ROOT, "libexec", "toy-train-gtx")
TOY     = File.join(ROOT, "bin", "toy")
FIXTURE = File.join(ROOT, "prep", "fixtures", "train_gtx_baseline.txt")

require "open3"
require "json"
require "tmpdir"
require "fileutils"

# Each arm's OWN best cell (see the header). They differ, and that is the
# point — do not "tidy" them into one LR.
STEPS_CELL = "1500"
LR_CHAIN   = "0.003"
LR_DFA     = "0.001"
SEEDS      = %w[0 1 2].freeze
CHANCE     = 1.0 / 16.0
# Margins sit far under the measured effects: BP clears frozen by .87,
# the layer cut clears it by .81 and sits .065 under BP.
BP_PRECOND  = 0.50   # BP must beat frozen by this — the cell must discriminate
DFA_FLOOR   = 0.70   # the layer cut must SOLVE the task, not merely beat chance
BP_GAP      = 0.12   # ... and stay within this of BP
FROZEN_EDGE = 0.50

def run_gtx(extra_env, run_dir)
  env = { "STEPS" => "5", "SEED" => "0" }.merge(extra_env)
  env = env.merge("TAO_RUN_DIR" => run_dir, "TOY_RUN_ID" => "gtx-gate") if run_dir
  out, st = Open3.capture2e(env, RUNNER, chdir: ROOT)
  abort "gtx_gate: runner exited #{st.exitstatus}:\n#{out.lines.last(10).join}" unless st.success?
  out
end

def curve(out)
  out.lines.select { |l| l.start_with?("step ") }
end

def val_acc(out)
  line = out.lines.find { |l| l.start_with?("val: ") }
  raise "gtx_gate: no val line in\n#{out}" unless line
  line[/acc=([0-9.eE+-]+)/, 1].to_f
end

unless File.executable?(RUNNER)
  build_out, build_st = Open3.capture2e("make", "-C", ROOT, "libexec/toy-train-gtx")
  unless build_st.success? && File.executable?(RUNNER)
    warn "gtx_gate: build failed:\n#{build_out.lines.last(15).join}"
    exit 2
  end
end

failures = []
# d878143: every leg records the failure count at its START in `n0`.
n0 = 0

# ---- 1. byte fixture + chain byte-null ----
n0 = failures.length
base_out   = run_gtx({}, nil)
base_curve = curve(base_out)
if File.file?(FIXTURE)
  expect = File.readlines(FIXTURE).reject { |l| l.start_with?("#") || l.strip.empty? }
  failures << "fixture: default curve != train_gtx_baseline.txt\ngot:  #{base_curve.join}want: #{expect.join}" unless base_curve == expect
else
  failures << "fixture: #{FIXTURE} missing"
end
chain_out = run_gtx({ "GTX_POLICY" => "chain,chain" }, nil)
failures << "chain byte-null: explicit all-chain policy curve != absent-policy curve" unless curve(chain_out) == base_curve
puts failures.length == n0 ? "  ok: default curve matches the fixture AND an explicit all-chain policy is byte-null" : "  FAIL: fixture / chain byte-null"

# ---- 2. determinism ----
n0 = failures.length
d1 = run_gtx({ "GTX_POLICY" => "dfa,dfa", "STEPS" => "20" }, nil)
d2 = run_gtx({ "GTX_POLICY" => "dfa,dfa", "STEPS" => "20" }, nil)
failures << "determinism: two identical dfa runs differ" unless d1 == d2
puts failures.length == n0 ? "  ok: two identical dfa runs are byte-identical" : "  FAIL: determinism"

# ---- 3. forward identity, then divergence ----
n0 = failures.length
ch20 = run_gtx({ "STEPS" => "20" }, nil)
st20 = run_gtx({ "GTX_POLICY" => "dfa,dfa", "GTX_DFA_CUT" => "step", "STEPS" => "20" }, nil)
failures << "forward identity: step 1 differs between dfa and chain — tnn_detach is forward-identity, so the arms are not comparable if step 1 moved" unless curve(d1).first == curve(ch20).first
failures << "forward identity: step 1 differs between dfa(step) and chain — detaching the ATTENTION PATTERN must not change the forward pass, or the cut changes the model and not just the credit rule" unless curve(st20).first == curve(ch20).first
failures << "arm effect: the whole dfa curve equals chain — the backward is not different" if curve(d1) == curve(ch20)
failures << "arm effect: dfa(step) equals dfa(layer) — cutting the mixing did nothing" if curve(st20) == curve(d1)
fz = run_gtx({ "GTX_POLICY" => "frozen,frozen", "STEPS" => "20" }, nil)
failures << "arm effect: frozen curve identical to chain" if curve(fz) == curve(ch20)
puts failures.length == n0 ? "  ok: step 1 is byte-identical to BP under BOTH cuts and steps 2+ diverge" : "  FAIL: forward identity / arm effect"

# ---- 4. the B seed moves the dfa curve, and is inert on chain ----
n0 = failures.length
dfa_b999   = run_gtx({ "GTX_POLICY" => "dfa,dfa", "STEPS" => "20", "GTX_B_SEED" => "999" }, nil)
chain_b999 = run_gtx({ "STEPS" => "20", "GTX_B_SEED" => "999" }, nil)
failures << "b seed: --dfa-b-seed does not move the dfa curve — the feedback matrix is not reaching the weights, and every DFA number below would be measuring nothing" if curve(dfa_b999) == curve(d1)
failures << "b seed: --dfa-b-seed moved the CHAIN curve — a pure-BP arm must not see the feedback matrix" unless curve(chain_b999) == curve(ch20)
puts failures.length == n0 ? "  ok: the B seed moves dfa and is inert on chain (toy#158)" : "  FAIL: b seed"

# ---- 5. wiring + bundle, and the SMALL HEAD ----
n0 = failures.length
Dir.mktmpdir("gtx_gate") do |dir|
  out = run_gtx({ "GTX_POLICY" => "dfa,frozen", "STEPS" => "10",
                  "GTX_DFA_CUT" => "step", "GTX_B_SEED" => "42" }, dir)
  wire = out.lines.find { |l| l.start_with?("gtx: ") }
  if wire.nil?
    failures << "bundle: no wiring line"
  else
    failures << "wiring: dfa_wired != 1 (#{wire.strip})" unless wire.include?("dfa_wired=1")
    failures << "wiring: frozen != 1 (#{wire.strip})" unless wire.include?("frozen=1")
    failures << "wiring: cut != step (#{wire.strip})" unless wire.include?("cut=step")
    # The step cut taps the block output AND Q and K per head: with one
    # dfa block and 4 heads that is 1 + 2*4 = 9 taps. Without the Q/K
    # taps this arm would silently be "attention frozen", which is a
    # different and much weaker claim than the one the lane makes.
    failures << "wiring: taps != 9 — the step cut must tap the block output plus Q and K for each of the 4 heads (#{wire.strip})" unless wire.include?("taps=9")
  end
  ev_path = File.join(dir, "events.jsonl")
  if File.file?(ev_path)
    events = File.readlines(ev_path).map { |l| JSON.parse(l) }
    rs = events.first || {}
    failures << "bundle: first event not run_start" unless rs["kind"] == "run_start"
    failures << "bundle: schema != toy/v1" unless rs["schema"] == "toy/v1"
    failures << "bundle: run_start.name != gtx (#{rs['name'].inspect})" unless rs["name"] == "gtx"
    failures << "bundle: last event not run_end" unless events.last && events.last["kind"] == "run_end"
    md = rs["model"] || {}
    failures << "bundle: model.arch != gtx (#{md['arch'].inspect})" unless md["arch"] == "gtx"
    failures << "bundle: model.attn_mask != adjacency — if the mask stops being the graph, this is no longer a graph transformer" unless md["attn_mask"] == "adjacency"
    failures << "bundle: model.readout != node_pair" unless md["readout"] == "node_pair"
    # THE SMALL-HEAD ASSERTION. The entire point of the lane is to ask
    # about attention OUTSIDE the large-output regime F13/F18 already
    # explained. A big head here would answer a question we have already
    # answered, under a name suggesting otherwise.
    nc = md["num_classes"]
    failures << "bundle: num_classes #{nc.inspect} is not small — this lane exists to test attention AWAY from the large-output regime; re-entering it (>= 256) makes the result a restatement of F13/F18" unless nc.is_a?(Numeric) && nc >= 4 && nc < 256
    df = rs["dfa"]
    if df.nil?
      failures << "bundle: run_start has no dfa object"
    else
      failures << "bundle: dfa.policy != [1,2] (got #{df['policy'].inspect})" unless df["policy"] == [1, 2]
      failures << "bundle: dfa.cut != step" unless df["cut"] == "step"
      failures << "bundle: dfa.b_seed != 42" unless df["b_seed"] == 42
      failures << "bundle: dfa.route != pair_incidence — the error lives on PAIRS and the taps on NODES, so how it is routed is part of the experiment's identity" unless df["route"] == "pair_incidence"
    end
    failures << "bundle: #{events.count { |e| e['kind'] == 'step' }} step events (want 10)" unless events.count { |e| e["kind"] == "step" } == 10
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
puts failures.length == n0 ? "  ok: bundle structure, the adjacency mask, the SMALL head, and the tap count the cut asks for" : "  FAIL: wiring / bundle"

# ---- 6 + 7 + 8. the bar, the mixing cut, and the controls ----
n0 = failures.length
rows = {}
SEEDS.each do |s|
  r = {}
  r["chain"]  = val_acc(run_gtx({ "STEPS" => STEPS_CELL, "SEED" => s, "GTX_LR" => LR_CHAIN,
                                  "GTX_POLICY" => "chain,chain" }, nil))
  r["dfa"]    = val_acc(run_gtx({ "STEPS" => STEPS_CELL, "SEED" => s, "GTX_LR" => LR_DFA,
                                  "GTX_POLICY" => "dfa,dfa", "GTX_DFA_CUT" => "layer" }, nil))
  r["step"]   = val_acc(run_gtx({ "STEPS" => STEPS_CELL, "SEED" => s, "GTX_LR" => LR_DFA,
                                  "GTX_POLICY" => "dfa,dfa", "GTX_DFA_CUT" => "step" }, nil))
  r["frozen"] = val_acc(run_gtx({ "STEPS" => STEPS_CELL, "SEED" => s, "GTX_LR" => LR_CHAIN,
                                  "GTX_POLICY" => "frozen,frozen" }, nil))
  rows[s] = r
  puts format("    seed %s  chain=%.4f dfa(layer)=%.4f dfa(step)=%.4f frozen=%.4f",
              s, r["chain"], r["dfa"], r["step"], r["frozen"])
end

SEEDS.each do |s|
  r = rows[s]
  if r["chain"] < r["frozen"] + BP_PRECOND
    failures << "SUCCESS BAR (precondition, seed #{s}): BP #{r['chain']} does not beat frozen #{r['frozen']} by #{BP_PRECOND} — the cell cannot discriminate, so nothing below means anything"
  end
  if r["dfa"] < DFA_FLOOR
    failures << "SUCCESS BAR (seed #{s}): dfa(layer) #{r['dfa']} is below #{DFA_FLOOR}. This lane's headline is that ATTENTION IS NOT DFA-HOSTILE at a small output dim — if the layer cut stops solving the task, that headline is gone and F22's route-A recommendation has to be re-derived"
  end
  if r["dfa"] < r["chain"] - BP_GAP
    failures << "SUCCESS BAR (BP gap, seed #{s}): dfa(layer) #{r['dfa']} is more than #{BP_GAP} below chain #{r['chain']}"
  end
  if r["dfa"] < r["frozen"] + FROZEN_EDGE
    failures << "SUCCESS BAR (frozen control, seed #{s}): dfa(layer) #{r['dfa']} does not beat frozen #{r['frozen']} by #{FROZEN_EDGE}"
  end
  # CONTROL-CAN-LOSE, asserted per seed. A structural attention mask
  # aggregates neighbourhoods for free, so this is the leg that caught
  # three degenerate versions of the task (see the header).
  if r["frozen"] > CHANCE + 0.20
    failures << "CONTROL: frozen #{r['frozen']} at seed #{s} is far above chance #{CHANCE.round(4)} — a frozen random transformer should NOT be able to do this task. Every earlier version of the task failed exactly here, and each time the lane would have reported a meaningless comparison. Re-read toy_gtx_task.rb before touching thresholds"
  end
end
puts failures.length == n0 ?
  "  ok: SUCCESS BAR — at each arm's OWN best LR, on all three seeds, block-tap DFA solves a graph transformer and crushes a frozen control that provably cannot" :
  "  FAIL: success bar / control"

# ---- 7. the mixing cut collapses ----
n0 = failures.length
step_mean = SEEDS.map { |s| rows[s]["step"] }.sum / SEEDS.length.to_f
dfa_mean  = SEEDS.map { |s| rows[s]["dfa"] }.sum / SEEDS.length.to_f
if step_mean > dfa_mean - 0.4
  failures << "MIXING CUT: dfa(step) #{step_mean.round(4)} is not far below dfa(layer) #{dfa_mean.round(4)}. The measured finding is that cutting the gradient through the ATTENTION PATTERN collapses this lane (.149 vs .920) even though the same random feedback works at the block boundary — the pattern has to be learned through the true gradient. That collapse is also what makes this lane agree with toy#155's selective scan and disagree with toy#157's gated LSTM; if it moved, re-read all three before re-baselining"
end
puts failures.length == n0 ?
  "  ok: MIXING CUT — cutting the gradient through the attention pattern collapses the arm, as on toy#155's selective scan and unlike toy#157's gated LSTM" :
  "  FAIL: mixing cut"

# ---- 8b. the degenerate task, measured ----
# `local` puts the type in the entity's OWN features, so no retrieval and
# no attention are needed. It is also the control that proves the DFA
# path WORKS: a lane reporting "DFA fails" has to show DFA succeeding
# somewhere, or the negative is indistinguishable from a wiring bug.
n0 = failures.length
loc_frozen = val_acc(run_gtx({ "STEPS" => STEPS_CELL, "SEED" => "0", "GTX_LR" => LR_CHAIN,
                               "GTX_TASK" => "local", "GTX_POLICY" => "frozen,frozen" }, nil))
loc_dfa    = val_acc(run_gtx({ "STEPS" => STEPS_CELL, "SEED" => "0", "GTX_LR" => LR_DFA,
                               "GTX_TASK" => "local", "GTX_POLICY" => "dfa,dfa" }, nil))
puts format("    --task local: frozen=%.4f dfa=%.4f   (relational frozen=%.4f)",
            loc_frozen, loc_dfa, rows["0"]["frozen"])
if loc_frozen < rows["0"]["frozen"] + 0.4
  failures << "degenerate task: --task local frozen #{loc_frozen} is not far above the relational task's frozen #{rows['0']['frozen']} — `local` is documented as solvable with no retrieval at all, and the default task is justified against that measurement"
end
if loc_dfa < 0.4
  failures << "DFA SANITY: dfa scores #{loc_dfa} on the DEGENERATE task, where no attention has to be learned. If DFA cannot learn even that, the collapse of the step cut above is a WIRING BUG rather than a finding, and no negative from this lane may be reported"
end
puts failures.length == n0 ? "  ok: --task local is measurably degenerate, AND DFA provably learns when the task needs no retrieval (so the step-cut collapse is a finding, not a bug)" : "  FAIL: degenerate task / dfa sanity"

# ---- 9. fail-loud ----
n0 = failures.length
[
  [{ "GTX_POLICY" => "chain,chain,chain" },       "policy longer than GTX_BLOCKS"],
  [{ "GTX_POLICY" => "bogus,chain" },             "unknown policy token"],
  [{ "GTX_POLICY" => "chain,dfa" },               "a chain block BELOW a dfa block (silently frozen)"],
  [{ "GTX_DFA_CUT" => "token" },                  "unknown GTX_DFA_CUT"],
  [{ "GTX_TASK" => "cue" },                       "unknown GTX_TASK"],
  [{ "GTX_TYPES" => "1" },                        "GTX_TYPES=1"],
  [{ "GTX_BLOCKS" => "0" },                       "GTX_BLOCKS=0"],
  [{ "GTX_HEADS" => "5" },                        "d_model not divisible by heads"],
  [{ "GTX_FEATURES" => "7" },                     "odd GTX_FEATURES (key/value halves)"],
  [{ "GTX_FEATURES" => "2" },                     "GTX_FEATURES below 4"],
  [{ "GTX_ENTITIES" => "2" },                     "fewer entities than types"],
  [{ "GTX_VAL_BATCHES" => "0" },                  "GTX_VAL_BATCHES=0"],
].each do |env, label|
  out, st = Open3.capture2e({ "STEPS" => "2" }.merge(env), RUNNER, chdir: ROOT)
  failures << "fail-loud: #{label} exited 0 (silently did nothing):\n#{out.lines.last(2).join}" if st.success?
end
puts failures.length == n0 ? "  ok: 12 degenerate configs all fail loud instead of quietly doing nothing" : "  FAIL: fail-loud"

# ---- 10. CLI ----
n0 = failures.length
Dir.mktmpdir("gtx_gate_cli") do |dir|
  out, st = Open3.capture2e({ "SPINEL_SKIP_PIN_CHECK" => nil }, TOY, "train", "gtx",
                            "--steps", "5", "--seed", "0", "--out", dir, chdir: ROOT)
  if st.success?
    failures << "cli: `toy train gtx` curve != the runner's default curve" unless curve(out) == base_curve
    failures << "cli: `toy train gtx` did not echo the val line" unless out.lines.any? { |l| l.start_with?("val: ") }
  else
    failures << "cli: `toy train gtx` exited #{st.exitstatus}:\n#{out.lines.last(5).join}"
  end
  [
    [%w[gtx --align-events],    "gtx accepted --align-events"],
    [%w[gtx --selection lti],   "gtx accepted --selection"],
    [%w[gtx --seq 32],          "gtx accepted --seq"],
    [%w[gtx --dt-init -4.0],    "gtx accepted --dt-init"],
    [%w[gtx --task cue],        "gtx accepted --task cue"],
    [%w[mlp --heads 4],         "mlp accepted --heads"],
    [%w[ssm --entities 32],     "ssm accepted --entities"],
    [%w[lstm --types 4],        "lstm accepted --types"],
  ].each do |argv, label|
    sout, sst = Open3.capture2e({}, TOY, "train", *argv, chdir: ROOT)
    failures << "cli: #{label} (exit #{sst.exitstatus}: #{sout.lines.last(1).join.strip})" unless sst.exitstatus == 2
  end
  cout, cst = Open3.capture2e({}, TOY, "train", "gtx", "--device", "cuda", chdir: ROOT)
  failures << "cli: gtx accepted --device cuda (CPU-only — tao#18)" unless cst.exitstatus == 2
end
puts failures.length == n0 ? "  ok: CLI reproduces the curve; lane-foreign flags and --device cuda are rejected" : "  FAIL: CLI"

if failures.empty?
  puts "GATE PASS [gtx]: graph transformer + per-block policy — byte fixture, the B seed, the small-head assertion, the MANDATORY success bar with each arm at ITS OWN best LR showing ATTENTION IS NOT DFA-HOSTILE (dfa .920 vs BP .985 vs frozen .111), the mixing-cut collapse, and a frozen control that provably CAN lose (toy#160)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [gtx]: #{f}" }
  exit 1
end
