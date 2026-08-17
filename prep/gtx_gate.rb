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
#  11. THE RETROFIT (toy#161): the precondition, the freeze, the bar.
#  11c. THE CHECKPOINT (toy#164): bit-exact round trip, refused mismatch.
#  12. THE BYTE-LM TASK (toy#170 / capstone P3): the embedding tap, the
#      two cuts as distinct wirings, and PER-ARM EFFECT.
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
# The retrofit head is TY-wide (the modular sum), not TY*TY.
CHANCE_RETRO = 1.0 / 4.0
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

  # toy#169/#170 — THE BYTE-LM PATH THROUGH THE CLI.
  #
  # Every P3-P6 cell was driven through the GTX_* env harness because
  # `--task bytelm` did not exist here, which is what blocked the formal
  # registry from recording those runs. Two things had to land, and both
  # are asserted because either one silently guts the path:
  #
  #   * the flag itself, plus --text to name the pack (there is NO
  #     synthetic fallback corpus, by design), and --vocab for the P5
  #     nominal head width;
  #   * the stdout ALLOWLIST. `bytelm:` IS the lane's result and `gtx:`
  #     carries vocab=/b_dim=. Filtered, a CLI run trains correctly and
  #     reports neither its result nor its provenance — green, and
  #     useless to anyone reading it.
  if File.file?(File.join(ROOT, "data", "ae_shak_a65.tok.i32"))
    bout, bst = Open3.capture2e({}, TOY, "train", "gtx", "--task", "bytelm",
                                "--text", "data/ae_shak_a65", "--vocab", "65",
                                "--steps", "5", "--seed", "0", "--val-batches", "2",
                                "--policy", "dfa,dfa", "--out", dir, chdir: ROOT)
    if bst.success?
      failures << "cli bytelm: the run did not emit its RESULT line (bytelm: bpb=...) — the allowlist filters it, so every number this lane produces would be invisible to the caller" unless
        bout.lines.any? { |l| l.start_with?("bytelm: ") }
      failures << "cli bytelm: the run did not emit its WIRING line (gtx: bytelm ...) — vocab= and b_dim= are how a cell proves which head width it used" unless
        bout.lines.any? { |l| l.start_with?("gtx: bytelm ") }
      failures << "cli bytelm: --vocab 65 did not reach the runner (expected vocab=65 b_dim=65)" unless
        bout.include?("vocab=65") && bout.include?("b_dim=65")
      failures << "cli bytelm: --text did not reach the runner (no corpus: line naming the pack)" unless
        bout.include?("pack=data/ae_shak_a65 ")
    else
      failures << "cli bytelm: `toy train gtx --task bytelm` exited #{bst.exitstatus}:\n#{bout.lines.last(5).join}"
    end
    # The refusals that keep the flag honest.
    [
      [%w[gtx --task bytelm],                                   "bytelm without --text"],
      [%w[mlp --task bytelm --text data/ae_shak_a65],           "bytelm on a lane that has no such task"],
      [%w[gtx --task bytelm --text data/nope],                  "--text naming a pack that is not there"],
      [%w[gtx --task relational --vocab 512],                   "--vocab without bytelm on gtx"],
    ].each do |argv, label|
      rout, rst = Open3.capture2e({}, TOY, "train", *argv, chdir: ROOT)
      failures << "cli bytelm fail-loud: #{label} exited #{rst.exitstatus} (expected 2)" unless rst.exitstatus == 2
    end
  end
end
puts failures.length == n0 ? "  ok: CLI reproduces the curve; lane-foreign flags and --device cuda are rejected; `--task bytelm` runs end-to-end with --text/--vocab and its RESULT and WIRING lines reach the caller, and 4 degenerate forms fail loud" : "  FAIL: CLI"

# ---- 11. THE RETROFIT LANE (toy#161) ----
#
# Phase 1 BP-pretrains the backbone on the 16-class pair task; phase 2
# freezes AND DETACHES it, adds a pair-site adapter stack and a fresh
# 4-class head, and trains only the added capacity under
# --adapter-policy. Both phases run in ONE process, so every arm's
# backbone is BIT-IDENTICAL by construction.
#
# Measured, 1500 + 1500 steps, seeds 0/1/2, each arm at its own LR:
#
#   bp-adapt   (chain, lr .003)   .990 / .987 / .979   mean .985
#   dfa-adapt  (dfa,   lr .001)   .989 / .991 / .980   mean .987
#   frozen-adapt        (.003)    .313 / .314 / .295   mean .307
#
# DFA-adapt MATCHES BP-adapt and beats the frozen control by .68. Note
# also that DFA is insensitive to LR here (.985-.987 across 10x), unlike
# the from-scratch lane where it needed 3x less than BP — retrofitting a
# frozen backbone through a small adapter is an EASY regime for DFA,
# which is the ticket's thesis.
#
# WHY THE RETROFIT LABEL IS A MODULAR SUM, and why that is not a detail:
# the pretrain label is a BIJECTION from pairs to classes, and so is any
# "new relation types" relabeling — a retrained LINEAR head absorbs it
# (u_c[a]=1{a=a*}, v_c[b]=1{b=b*} scores the right class 2 and every
# other <=1), the frozen control cannot lose, and the bar is vacuous.
# The modular sum is MANY-TO-ONE and provably needs an INTERACTION
# between the endpoints, which exists only where the pair is formed.
n0 = failures.length
RETRO = { "GTX_RETROFIT" => "1", "GTX_PRETRAIN_STEPS" => "1500",
          "STEPS" => "1500", "GTX_PRETRAIN_LR" => "0.003" }.freeze
rt = {}
%w[chain dfa frozen].each do |ap|
  lr = ap == "dfa" ? "0.001" : "0.003"
  out = run_gtx(RETRO.merge("GTX_ADAPTER_POLICY" => ap, "GTX_LR" => lr,
                            "SEED" => "0"), nil)
  rt[ap] = val_acc(out)
  if ap == "dfa"
    bl = out.lines.find { |l| l.start_with?("backbone: ") }
    if bl.nil?
      failures << "retrofit: no backbone signature line"
    else
      pre = bl[/sig_pre=([0-9.eE+-]+)/, 1]
      post = bl[/sig_post=([0-9.eE+-]+)/, 1]
      # THE FREEZE, PROVED rather than promised: a skipped optimizer step
      # is not the same fact as a weight that did not move.
      failures << "retrofit: --freeze-backbone did NOT hold the backbone (#{pre} -> #{post})" unless pre == post
    end
    rl = out.lines.find { |l| l.start_with?("retrofit: ") }
    failures << "retrofit: no cost line" if rl.nil?
    failures << "retrofit: bb_grad_bytes_avoided is 0 under --freeze-backbone" if rl && rl.include?("bb_grad_bytes_avoided=0")
  end
end
puts format("    retrofit: bp-adapt=%.4f dfa-adapt=%.4f frozen-adapt=%.4f (chance %.4f)",
            rt["chain"], rt["dfa"], rt["frozen"], CHANCE_RETRO)
# PRECONDITION first: without it nothing below discriminates.
if rt["chain"] < rt["frozen"] + 0.40
  failures << "RETROFIT PRECONDITION: bp-adapt #{rt['chain']} does not beat frozen-adapt #{rt['frozen']} by 0.40 — the retrofit task must be one the frozen backbone CANNOT solve, or the comparison is vacuous. If this fires, check the label is still MANY-TO-ONE: any bijective relabeling is absorbed by the retrained head (see toy_gtx_task.rb)"
end
if rt["frozen"] > CHANCE_RETRO + 0.25
  failures << "RETROFIT CONTROL: frozen-adapt #{rt['frozen']} is far above chance #{CHANCE_RETRO} — the added capacity is supposed to be NECESSARY here"
end
if rt["dfa"] < 0.85
  failures << "RETROFIT BAR: dfa-adapt #{rt['dfa']} is below 0.85 — the lane's headline is that DFA can adapt a frozen BP-pretrained backbone"
end
if rt["dfa"] < rt["chain"] - 0.10
  failures << "RETROFIT BAR: dfa-adapt #{rt['dfa']} trails bp-adapt #{rt['chain']} by more than 0.10"
end
if rt["dfa"] < rt["frozen"] + 0.40
  failures << "RETROFIT BAR (frozen control): dfa-adapt #{rt['dfa']} does not beat frozen-adapt #{rt['frozen']} by 0.40"
end
puts failures.length == n0 ?
  "  ok: RETROFIT (toy#161) — DFA-adapt matches BP-adapt on a frozen BP-pretrained backbone and crushes a frozen-adapter control that provably cannot solve the task; the freeze is proved bit-identical" :
  "  FAIL: retrofit"

# ---- 11b. the retrofit's own fail-loud + byte-null ----
n0 = failures.length
plain = run_gtx({ "STEPS" => "5" }, nil)
failures << "retrofit: a NON-retrofit run changed — the retrofit capacity must be inert when it is not asked for" unless curve(plain) == base_curve
[
  [{ "GTX_ADAPTER_POLICY" => "dfa" },                              "GTX_ADAPTER_POLICY without GTX_RETROFIT"],
  [{ "GTX_RETROFIT" => "1", "GTX_ADAPTER_POLICY" => "bogus" },     "unknown adapter policy"],
  [{ "GTX_RETROFIT" => "1", "GTX_PRETRAIN_STEPS" => "0" },         "retrofit of an UNTRAINED backbone"],
  [{ "GTX_RETROFIT" => "1", "GTX_ADAPTER_LAYERS" => "0" },         "zero adapter layers"],
  [{ "GTX_RETROFIT" => "1", "GTX_ADAPTER_RANK" => "0" },           "zero adapter rank"],
].each do |env, label|
  out, st = Open3.capture2e({ "STEPS" => "2" }.merge(env), RUNNER, chdir: ROOT)
  failures << "fail-loud: #{label} exited 0 (silently did nothing)" if st.success?
end
puts failures.length == n0 ? "  ok: the retrofit is INERT when unasked, and 5 degenerate retrofit configs fail loud" : "  FAIL: retrofit guards"

# ---- 11c. the CHECKPOINT round trip (toy#164) ----
# One pretrain, many retrofit arms. The property that has to hold is
# BIT-EXACTNESS: the backbone a load produces must be the backbone the
# write saw, or every arm sweeping off that checkpoint is measuring a
# subtly different model and nothing downstream would show it.
n0 = failures.length
Dir.mktmpdir("gtx_ckpt") do |dir|
  w = run_gtx({ "STEPS" => "200", "SEED" => "0", "GTX_CKPT_EVERY" => "200",
                "TAO_RUN_DIR" => dir, "TOY_RUN_ID" => "gtx-ckpt" }, nil)
  ck = File.join(dir, "step_200.gguf")
  if !File.file?(ck)
    failures << "ckpt: --ckpt-every wrote no step_200.gguf"
  else
    wl = w.lines.find { |l| l.start_with?("ckpt: ") }
    wsig = wl ? wl[/bb_sig=([0-9.eE+-]+)/, 1] : nil
    r = run_gtx({ "GTX_RETROFIT" => "1", "GTX_LOAD_CKPT" => ck,
                  "STEPS" => "5", "SEED" => "0",
                  "GTX_ADAPTER_POLICY" => "dfa", "GTX_LR" => "0.001" }, nil)
    ll = r.lines.find { |l| l.start_with?("loaded: ") }
    lsig = ll ? ll[/bb_sig=([0-9.eE+-]+)/, 1] : nil
    if wsig.nil? || lsig.nil?
      failures << "ckpt: no bb_sig on the write (#{wsig.inspect}) or the load (#{lsig.inspect})"
    else
      # STRING equality, not a tolerance: a round trip that is merely
      # close is a round trip that lost bits.
      failures << "ckpt: round trip is not bit-exact — wrote bb_sig=#{wsig}, loaded bb_sig=#{lsig}" unless wsig == lsig
    end
    failures << "ckpt: a loaded run still ran the pretrain phase (it must SKIP it — redoing it is the cost this flag removes)" unless r.include?("pretrain: acc=loaded")
    # A checkpoint written at one shape must be REFUSED by another, and
    # the refusal must name both sides.
    mm, mst = Open3.capture2e({ "GTX_RETROFIT" => "1", "GTX_LOAD_CKPT" => ck,
                                "GTX_TYPES" => "5", "STEPS" => "2" }, RUNNER, chdir: ROOT)
    failures << "ckpt: a shape-MISMATCHED checkpoint loaded anyway (exit #{mst.exitstatus}) — a silently wrong backbone makes every downstream number look healthy" if mst.success?
    failures << "ckpt: the mismatch message does not name both sides" unless mm.include?("vs instrument")
  end
end
[
  [{ "GTX_LOAD_CKPT" => "/nonexistent.gguf", "GTX_RETROFIT" => "1" }, "missing checkpoint file"],
  [{ "GTX_LOAD_CKPT" => "/nonexistent.gguf" },                        "--load-ckpt without --retrofit"],
  [{ "GTX_CKPT_EVERY" => "10" },                                      "--ckpt-every with no run dir"],
].each do |env, label|
  out, st = Open3.capture2e({ "STEPS" => "2" }.merge(env), RUNNER, chdir: ROOT)
  failures << "fail-loud: #{label} exited 0 (silently did nothing)" if st.success?
end
puts failures.length == n0 ?
  "  ok: CHECKPOINT (toy#164) — the backbone round-trips BIT-EXACT, a load skips the pretrain phase, a shape mismatch is refused naming both sides, and 3 degenerate configs fail loud" :
  "  FAIL: checkpoint"

# ── LEG 12: THE BYTE-LM TASK (toy#170 / capstone P3) ──
#
# The gtx lane now has a second task: `--task bytelm`, a causal
# byte-level LM with a per-position readout. It shares the engine with
# the relational task, which is exactly why leg 1 above still asserts
# the toy#160 byte fixture — the P3 build must be forward-null on the
# path it did not touch, and leg 1 proves it.
#
# What this leg asserts is the wiring the ARMS depend on, because every
# P3 number is meaningless if any of it is wrong. Two of these fired
# during the build:
#
#   * The EMBEDDING TAP. Under DFA the byte embedding sits BELOW the
#     layer-0 detach, so without its own tap it never trains and the
#     "DFA" arm is really "DFA with a frozen embedding". This is the
#     same bug as toy#169's, and it is asserted here rather than
#     trusted, because the failure is silent — the run trains, the loss
#     falls, and only the gap is wrong.
#   * PER-ARM EFFECT. Each arm must move its OWN curve away from the
#     frozen control. An arm whose knob is a no-op looks like a clean
#     negative, which is the most expensive way to be wrong in this
#     program (toy#141, and toy#169's audit).
n0 = failures.length
BL = { "GTX_TASK" => "bytelm", "GTX_TEXT" => "data/ae_shakespeare",
       "GTX_CONTEXT" => "64", "GTX_D_MODEL" => "64", "GTX_HEADS" => "4",
       "GTX_D_FF" => "128", "GTX_BLOCKS" => "2", "GTX_VAL_BATCHES" => "2" }
if !File.file?(File.join(ROOT, "data", "ae_shakespeare.tok.i32"))
  puts "  skip: BYTE-LM (toy#170) — data/ae_shakespeare pack absent (run prep/fetch_text.rb)"
else
  def bl_run(extra)
    run_gtx(BL.merge({ "STEPS" => "40", "GTX_LR" => "0.003" }).merge(extra), nil)
  end
  def bpb(out)
    l = out.lines.find { |x| x.start_with?("bytelm: ") }
    l ? l[/bpb=([0-9.eE+-]+)/, 1].to_f : nil
  end
  def wiring(out)
    out.lines.find { |x| x.start_with?("gtx: bytelm ") } || ""
  end

  arms = {}
  [["chain,chain", "layer"], ["dfa,dfa", "layer"], ["dfa,dfa", "step"],
   ["frozen,frozen", "layer"]].each do |pol, cut|
    o = bl_run("GTX_POLICY" => pol, "GTX_DFA_CUT" => cut)
    arms["#{pol.split(",").first}/#{cut}"] = [bpb(o), wiring(o)]
  end

  # 12a. Every arm produced a number, and the readout is the one we think.
  arms.each do |k, (b, w)|
    failures << "bytelm: arm #{k} emitted no bpb= line" if b.nil?
    failures << "bytelm: arm #{k} did not report the per-position causal readout" unless
      w.include?("attn=causal") && w.include?("readout=per_position") && w.include?("vocab=256")
  end

  # 12b. THE EMBEDDING TAP. Under DFA the embedding must be tapped; under
  # chain and frozen it must NOT be (chain reaches it through the chain,
  # and a frozen body has nothing to feed back).
  failures << "bytelm: the DFA arm did NOT tap the byte embedding — it sits below the layer-0 detach, so it would never train and the arm would silently be 'DFA with a frozen embedding' (toy#169's bug)" unless
    arms["dfa/layer"][1].include?("emb_tapped=1")
  failures << "bytelm: the chain arm tapped the embedding — BP reaches it through the chain, so a tap there is a second, unaccounted path" if
    arms["chain/layer"][1].include?("emb_tapped=1")

  # 12c. The cut is wired, and the two cuts are DIFFERENT wirings.
  lt = arms["dfa/layer"][1][/taps=(\d+)/, 1].to_i
  st = arms["dfa/step"][1][/taps=(\d+)/, 1].to_i
  failures << "bytelm: --dfa-cut layer wired #{lt} taps (expected one per block)" unless lt > 0
  failures << "bytelm: --dfa-cut step wired #{st} taps, not more than layer's #{lt} — the per-step cut must tap per POSITION, so if these match the flag is a no-op and 'the cuts agree' would be an artifact" unless st > lt
  failures << "bytelm: the chain arm wired #{arms["chain/layer"][1][/taps=(\d+)/, 1]} DFA taps (must be 0)" unless
    arms["chain/layer"][1].include?("taps=0")

  # 12d. PER-ARM EFFECT — each arm's CURVE must differ from the frozen
  # control's. Deliberately NOT "each arm must win": at gate scale (40
  # steps) the frozen control is still training its head and the body
  # arms have not paid off yet, so a win assertion would be measuring
  # noise and would make the gate the experiment. What the gate owes P3
  # is that no arm's knob is INERT — an inert arm reports as a clean
  # negative, which is the most expensive way to be wrong here (toy#141,
  # toy#169's audit). Whether DFA beats frozen is the FINDING, and it is
  # allowed to come back negative.
  fzc = curve(bl_run("GTX_POLICY" => "frozen,frozen", "GTX_DFA_CUT" => "layer"))
  [["chain,chain", "layer"], ["dfa,dfa", "layer"], ["dfa,dfa", "step"]].each do |pol, cut|
    c = curve(bl_run("GTX_POLICY" => pol, "GTX_DFA_CUT" => cut))
    failures << "bytelm: arm #{pol.split(",").first}/#{cut} produced a curve BYTE-IDENTICAL to the frozen control — its knob is inert, and an inert arm reports as a clean negative" if c == fzc
  end
  # The one arm we DO require to win, because if BP cannot beat a frozen
  # body on this task then the task itself is degenerate and no negative
  # measured on it would mean anything (the control-can-lose rule, but
  # pointed the other way — here it is the CONTROL that must not win).
  if arms["chain/layer"][0] && arms["frozen/layer"][0]
    failures << "bytelm: BP (#{arms["chain/layer"][0].round(3)}) did not beat the frozen body (#{arms["frozen/layer"][0].round(3)}) — if the task is solvable by the head alone, every credit-assignment number measured on it is uninterpretable" unless
      arms["chain/layer"][0] < arms["frozen/layer"][0] - 0.01
  end

  # 12e. Determinism on the new task — same seed, same bytes.
  a = bl_run("GTX_POLICY" => "dfa,dfa", "GTX_DFA_CUT" => "layer")
  failures << "bytelm: the DFA arm is not deterministic across runs at a fixed seed" unless
    curve(a) == curve(bl_run("GTX_POLICY" => "dfa,dfa", "GTX_DFA_CUT" => "layer"))

  # 12f. FAIL-LOUD on the task's own degenerate configs.
  [[{ "GTX_TASK" => "bytelm" },                                  "--task bytelm with no --text"],
   [{ "GTX_TASK" => "bytelm", "GTX_TEXT" => "data/nope" },        "--text naming a pack that is not there"],
   [{ "GTX_TASK" => "nonsense" },                                 "an unknown --task"]].each do |env, label|
    out, st2 = Open3.capture2e({ "STEPS" => "2" }.merge(env), RUNNER, chdir: ROOT)
    failures << "bytelm fail-loud: #{label} exited 0 (silently did nothing)" if st2.success?
  end

  puts failures.length == n0 ?
    "  ok: BYTE-LM (toy#170) — 4 arms score, the readout is per-position causal at vocab 256, the DFA arm TAPS THE EMBEDDING (chain does not), the two cuts are distinct wirings, no arm's knob is inert (curve differs from frozen's) and BP beats the frozen body so the task is not head-solvable, deterministic, 3 degenerate configs fail loud" :
    "  FAIL: bytelm"
end

# ── LEG 13: THE NOMINAL HEAD WIDTH (toy#170 / spec P5) ──
#
# `--vocab k` narrows the byte-LM head, the byte embedding AND the DFA
# feedback matrix B together. That last one is the reason the knob
# exists: the output-dim law is a claim about B, whose width and whose
# 1/sqrt(fan) scale were both pinned at 256 on every byte-LM cell run on
# this lane. An alphabet sweep at a fixed head measures the EFFECTIVE
# rank of the error; this measures the width of B. They are the same
# number only when the head is sized to the corpus.
#
# What this leg owes P5 is that the knob is neither inert nor silently
# reinterpreting old configs, and that it refuses the one input that
# would corrupt a result without crashing.
n0 = failures.length
# Self-contained rather than reusing leg 12's BL/wiring: those are bound
# inside leg 12's else branch, so a skipped leg 12 would take this one
# down with a NameError instead of a skip.
HW = { "GTX_TASK" => "bytelm", "GTX_CONTEXT" => "64", "GTX_D_MODEL" => "64",
       "GTX_HEADS" => "4", "GTX_D_FF" => "128", "GTX_BLOCKS" => "2",
       "GTX_VAL_BATCHES" => "2" }
def hw_wiring(out)
  out.lines.find { |x| x.start_with?("gtx: bytelm ") } || ""
end
def hw_bpb(out)
  l = out.lines.find { |x| x.start_with?("bytelm: ") }
  l ? l[/bpb=([0-9.eE+-]+)/, 1].to_f : nil
end
# Defined at top level, NOT inside a leg's else: legs 13 and 14 both call
# it, so binding it inside one of them would turn a SKIP into a NameError
# in the other.
def hw_run(extra)
  run_gtx(HW.merge({ "STEPS" => "40", "GTX_LR" => "0.003",
                     "GTX_TEXT" => "data/ae_shak_a65",
                     "GTX_POLICY" => "dfa,dfa", "GTX_DFA_CUT" => "layer" })
            .merge(extra), nil)
end
if !File.file?(File.join(ROOT, "data", "ae_shak_a65.tok.i32")) ||
   !File.file?(File.join(ROOT, "data", "ae_shakespeare.tok.i32"))
  puts "  skip: HEAD WIDTH (toy#170/P5) — data/ae_shak_a65 or data/ae_shakespeare absent (run prep/fetch_text.rb then prep/remap_alphabet.rb)"
else

  # 13a. THE DEFAULT CHANGES NOTHING. Not "vocab 256 works" — that the
  # knob left unset and the knob set to its default are BYTE-IDENTICAL
  # runs. This lane's flag strings are experiment identity: every cell
  # measured before P5 was run without GTX_VOCAB, and they stay
  # comparable only if unset really is 256.
  base = hw_run({})
  failures << "head width: GTX_VOCAB unset is not byte-identical to GTX_VOCAB=256 — every byte-LM cell measured before P5 was run without the flag, so a default that changes anything silently re-bases the whole arc" unless
    curve(base) == curve(hw_run("GTX_VOCAB" => "256"))
  failures << "head width: the default run does not report vocab=256 b_dim=256" unless
    hw_wiring(base).include?("vocab=256") && hw_wiring(base).include?("b_dim=256")

  # 13b. The knob MOVES, and it moves B with it. A head that narrows
  # while B stays 256 wide would measure the opposite of what P5 claims
  # to measure, and would do it while producing entirely plausible
  # numbers — which is why b_dim is asserted separately from vocab.
  narrow = hw_run("GTX_VOCAB" => "65")
  failures << "head width: --vocab 65 did not report vocab=65" unless
    hw_wiring(narrow).include?("vocab=65")
  failures << "head width: --vocab 65 left the DFA feedback matrix at b_dim=256 — B is the thing the output-dim law is about, so a head that narrows without B is measuring the wrong axis and would still produce plausible numbers" unless
    hw_wiring(narrow).include?("b_dim=65")
  failures << "head width: --vocab 65 produced a curve byte-identical to --vocab 256 — the knob is inert" if
    curve(narrow) == curve(base)

  # 13c. FAIL LOUD on a pack the head cannot cover. ae_shakespeare has 65
  # symbols carried on byte ids up to 122, so `--vocab 65` there is a
  # size error, not a narrower head. Unguarded it would score tokens
  # against rows past the end of the label — silent, and biased toward
  # the null, because a class nothing can predict costs every arm the
  # same amount and that reads exactly like "the axis is flat".
  out13, st13 = Open3.capture2e(
    HW.merge({ "STEPS" => "2", "GTX_TEXT" => "data/ae_shakespeare",
               "GTX_VOCAB" => "65" }), RUNNER, chdir: ROOT)
  failures << "head width fail-loud: --vocab 65 on a pack with token ids up to 122 exited 0 — the out-of-range ids would be scored against rows that do not exist, silently and toward the null" if st13.success?
  failures << "head width fail-loud: the refusal does not name the max token id, so it cannot be acted on" unless
    out13.include?("max token id")

  puts failures.length == n0 ?
    "  ok: HEAD WIDTH (toy#170/P5) — GTX_VOCAB unset is byte-identical to 256 (old cells stay comparable), --vocab 65 narrows the head AND the DFA feedback matrix together (b_dim asserted separately from vocab) and is not inert, and a pack whose ids overflow the head is REFUSED by max id with the number named" :
    "  FAIL: head width"
end

# ── LEG 14: THE WIDENED PACK CONTRACT (toy#170 / spec P6) ──
#
# AeTask carried a hard `VOCAB = 256` refusal — "the pack is not byte
# ids" — which capped symbol-inflation at m=3 (rank 192 < 256) purely by
# luck. P6 widens the contract to token ids so the arithmetic-rank ladder
# can run, and AeTask is shared by FOUR runners (ae, difflm, gtx, ssm).
#
# The whole P3/depth/P4/P5 arc has to stay comparable across that change,
# so what this leg owes the program is that widening the contract did NOT
# move the byte path, and that the new ceiling still refuses loudly.
n0 = failures.length
WIDE = File.join(ROOT, "data", "ae_shak_a380.tok.i32")
if !File.file?(File.join(ROOT, "data", "ae_shak_a65.tok.i32"))
  puts "  skip: PACK CONTRACT (toy#170/P6) — data/ae_shak_a65 absent (run prep/remap_alphabet.rb)"
else
  # 14a. THE BYTE PATH DID NOT MOVE. A byte-id pack under the default
  # ceiling must still train the same curve it did before the contract
  # widened — this is the assertion that keeps every pre-P6 cell in the
  # arc comparable, and it is the reason the default is 256 rather than
  # VOCAB_MAX.
  bp1 = hw_run({})
  failures << "pack contract: a byte-id pack under the DEFAULT ceiling did not produce a bpb — the widened contract broke the byte path" if
    hw_bpb(bp1).nil?
  failures << "pack contract: the default run no longer reports vocab=256 b_dim=256" unless
    hw_wiring(bp1).include?("vocab=256") && hw_wiring(bp1).include?("b_dim=256")

  if !File.file?(WIDE)
    puts "  note: PACK CONTRACT — data/ae_shak_a380 absent, skipping the token-id half (run prep/remap_alphabet.rb)"
  else
    # 14b. A pack with ids past 255 is REFUSED under the default ceiling.
    # Not "handled" — refused. Silently accepting it is how a rank-380
    # corpus would get scored against a 256-wide head with 124 symbols
    # folded into nothing.
    outw, stw = Open3.capture2e(
      HW.merge({ "STEPS" => "2", "GTX_TEXT" => "data/ae_shak_a380" }), RUNNER, chdir: ROOT)
    failures << "pack contract: a token-id pack (rank 380) was accepted under the DEFAULT 256 ceiling — ids past 255 must be refused, not folded" if stw.success?

    # 14c. ...and ACCEPTED once the head asks for the width. This is the
    # pair that proves the widening is real rather than a relaxed check:
    # same pack, refused at 256, runs at 512.
    okw = hw_run("GTX_TEXT" => "data/ae_shak_a380", "GTX_VOCAB" => "512")
    failures << "pack contract: rank-380 pack did not run at --vocab 512 — the widening is not actually wired" if
      hw_bpb(okw).nil?
    failures << "pack contract: the rank-380 run did not report vocab=512 b_dim=512" unless
      hw_wiring(okw).include?("vocab=512") && hw_wiring(okw).include?("b_dim=512")
    failures << "pack contract: the rank-380 run reports an alphabet other than 380 — the pack or the counter disagrees with the file" unless
      okw.include?("alphabet=380")
  end

  # 14d. The ceiling itself fails loud. It exists because the labels are
  # a DENSE one-hot Mat, so asking for 50k would not be wrong, it would
  # be quietly enormous — the hardest kind of mistake to catch.
  outc, stc = Open3.capture2e(
    HW.merge({ "STEPS" => "2", "GTX_VOCAB" => "65536" }), RUNNER, chdir: ROOT)
  failures << "pack contract fail-loud: --vocab 65536 exited 0 — past VOCAB_MAX the dense one-hot label upload is ~25 MB/step, which is slow rather than wrong and would never announce itself" if stc.success?

  puts failures.length == n0 ?
    "  ok: PACK CONTRACT (toy#170/P6) — the byte path is UNMOVED by the widening (default ceiling still 256, still trains), a rank-380 token-id pack is REFUSED at the default and RUNS at --vocab 512 reporting its own alphabet, and a request past VOCAB_MAX fails loud" :
    "  FAIL: pack contract"
end

if failures.empty?
  puts "GATE PASS [gtx]: graph transformer + RETROFIT + per-block policy — byte fixture, the B seed, the small-head assertion, the MANDATORY success bar with each arm at ITS OWN best LR showing ATTENTION IS NOT DFA-HOSTILE (dfa .920 vs BP .985 vs frozen .111), the mixing-cut collapse, and a frozen control that provably CAN lose (toy#160)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [gtx]: #{f}" }
  exit 1
end
