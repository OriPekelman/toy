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
#  13-14. THE NOMINAL HEAD WIDTH + the widened pack contract (toy#170/P5,P6).
#  15. THE B-CONDITIONING INSTRUMENT (toy#172/E1 Phase 1.1).
#  16. nDFA, the error-side preconditioner (toy#172/E1 Phase 1.2).
#  17. LDFA, adaptive low-rank feedback (toy#172/E2): the scale match is
#      the load-bearing assertion — the arms must differ in RANK and not
#      in SCALE, or "low rank hurts" is just "the updates got smaller".
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
  # tao#24 narrowed this from "gtx is CPU-only" to "gtx is CPU-only
  # EXCEPT under --task bytelm". With no task named, the default is
  # relational, so the refusal stands — and it has to, or the twin's
  # scope would be the recipe rather than the task. The positive case
  # (`--task bytelm --device cuda` runs, on cuda) and the rest of the
  # device matrix live in prep/gtx_cuda_gate.rb, which needs a GPU.
  cout, cst = Open3.capture2e({}, TOY, "train", "gtx", "--device", "cuda", chdir: ROOT)
  failures << "cli: gtx accepted --device cuda on the RELATIONAL task (tao#24 reopened the device scope for --task bytelm ONLY)" unless cst.exitstatus == 2

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
# EACH PROBE ASSERTS ITS OWN REASON, not merely a non-zero exit. The
# third column is a fragment of the message this config MUST produce.
#
# Checking only `st.success?` is what let toy#179 rot a probe silently:
# the `--load-ckpt without --retrofit` row below in 11c kept passing after
# that guard was deliberately REMOVED, because the path it used does not
# exist and the run then died on the file check instead. A rejection for
# the wrong reason is indistinguishable from the right one under an
# exit-code-only assertion, and the label goes on claiming otherwise.
RETRO_GUARDS = [
  [{ "GTX_ADAPTER_POLICY" => "dfa" },                          "GTX_ADAPTER_POLICY without GTX_RETROFIT",
   "GTX_ADAPTER_POLICY is meaningless without GTX_RETROFIT=1"],
  [{ "GTX_RETROFIT" => "1", "GTX_ADAPTER_POLICY" => "bogus" }, "unknown adapter policy",
   "GTX_ADAPTER_POLICY bogus unsupported"],
  [{ "GTX_RETROFIT" => "1", "GTX_PRETRAIN_STEPS" => "0" },     "retrofit of an UNTRAINED backbone",
   "GTX_PRETRAIN_STEPS must be >= 1"],
  [{ "GTX_RETROFIT" => "1", "GTX_ADAPTER_LAYERS" => "0" },     "zero adapter layers",
   "GTX_ADAPTER_LAYERS must be >= 1"],
  [{ "GTX_RETROFIT" => "1", "GTX_ADAPTER_RANK" => "0" },       "zero adapter rank",
   "GTX_ADAPTER_RANK must be >= 1"],
  # toy#180. build_training_step branches to the bytelm tail BEFORE the
  # retrofit adapter graph, so this combination used to allocate adapters
  # nothing consumed and then print a run that read like a retrofit while
  # training a plain byte-LM — silent, on a lane where every other
  # configuration trap fails loud. The message must also point at the
  # route that DOES work, or the refusal just leaves the user stuck.
  [{ "GTX_RETROFIT" => "1", "GTX_TASK" => "bytelm" },          "retrofit x bytelm (toy#180)",
   "GTX_RETROFIT=1 is not supported with GTX_TASK=bytelm"],
].freeze
RETRO_GUARDS.each do |env, label, want|
  out, st = Open3.capture2e({ "STEPS" => "2" }.merge(env), RUNNER, chdir: ROOT)
  if st.success?
    failures << "fail-loud: #{label} exited 0 (silently did nothing)"
  elsif !out.include?(want)
    failures << "fail-loud: #{label} was rejected for the WRONG REASON — expected a message containing #{want.inspect}, got: #{out.lines.grep(/toy-train-gtx:/).first.to_s.strip[0, 120]}"
  end
end
failures << "fail-loud: the retrofit x bytelm refusal does not name the working alternative (GTX_CKPT_EVERY / GTX_LOAD_CKPT)" unless
  Open3.capture2e({ "STEPS" => "2", "GTX_RETROFIT" => "1", "GTX_TASK" => "bytelm" }, RUNNER, chdir: ROOT).first =~ /GTX_CKPT_EVERY.*GTX_LOAD_CKPT/m
puts failures.length == n0 ? "  ok: the retrofit is INERT when unasked, and #{RETRO_GUARDS.length} degenerate retrofit configs fail loud FOR THEIR OWN STATED REASON" : "  FAIL: retrofit guards"

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

    # toy#179: A PLAIN RUN CAN LOAD. The checkpoint was always WRITTEN by
    # plain runs but could only be READ by a retrofit, so `pretrain on
    # corpus A -> adapt on corpus B` could not be expressed as two ordinary
    # processes with the checkpoint as an auditable artifact between them.
    # This is the positive half of that change, and nothing asserted it:
    # the only probe that mentioned the pairing was a REJECTION probe for
    # the guard that was removed, which kept passing on an unrelated error.
    #
    # bb_sig is compared as a STRING against the WRITE, exactly as the
    # retrofit path above does — a load that is merely close is a load that
    # lost bits, and every downstream cell would inherit the drift.
    pl, plst = Open3.capture2e({ "GTX_LOAD_CKPT" => ck, "STEPS" => "5", "SEED" => "0" },
                               RUNNER, chdir: ROOT)
    if !plst.success?
      failures << "ckpt: a PLAIN run (no GTX_RETROFIT) could not load a checkpoint (exit #{plst.exitstatus}) — toy#179 is the two-process pretrain->adapt shape: #{pl.lines.grep(/toy-train-gtx:/).first.to_s.strip[0, 120]}"
    else
      pll = pl.lines.find { |l| l.start_with?("loaded: ") }
      plsig = pll ? pll[/bb_sig=([0-9.eE+-]+)/, 1] : nil
      if plsig.nil?
        failures << "ckpt: a plain load printed no `loaded: ... bb_sig=` line, so the load is unauditable"
      elsif plsig != wsig
        failures << "ckpt: the PLAIN-run round trip is not bit-exact — wrote bb_sig=#{wsig}, loaded bb_sig=#{plsig}"
      end
    end
  end
end
CKPT_GUARDS = [
  [{ "GTX_LOAD_CKPT" => "/nonexistent.gguf", "GTX_RETROFIT" => "1" }, "missing checkpoint file (retrofit)",
   "no such checkpoint: /nonexistent.gguf"],
  # This row USED to be labelled "--load-ckpt without --retrofit" and to
  # exist for a guard that toy#179 deliberately removed. It kept passing
  # anyway, on the missing-file check, because the harness only asked for
  # a non-zero exit — so the gate went on advertising a refusal that no
  # longer happened. The config is still worth probing; the LABEL and the
  # expected message now say what it actually tests.
  [{ "GTX_LOAD_CKPT" => "/nonexistent.gguf" },                        "missing checkpoint file (plain run)",
   "no such checkpoint: /nonexistent.gguf"],
  [{ "GTX_CKPT_EVERY" => "10" },                                      "--ckpt-every with no run dir",
   "GTX_CKPT_EVERY needs RUN_DIR"],
].freeze
CKPT_GUARDS.each do |env, label, want|
  out, st = Open3.capture2e({ "STEPS" => "2" }.merge(env), RUNNER, chdir: ROOT)
  if st.success?
    failures << "fail-loud: #{label} exited 0 (silently did nothing)"
  elsif !out.include?(want)
    failures << "fail-loud: #{label} was rejected for the WRONG REASON — expected #{want.inspect}, got: #{out.lines.grep(/toy-train-gtx:/).first.to_s.strip[0, 120]}"
  end
end
puts failures.length == n0 ?
  "  ok: CHECKPOINT (toy#164/#179) — the backbone round-trips BIT-EXACT through BOTH the retrofit path and a PLAIN load, a retrofit load skips the pretrain phase, a shape mismatch is refused naming both sides, and #{CKPT_GUARDS.length} degenerate configs fail loud FOR THEIR OWN STATED REASON" :
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

  # ── LEG 12d: THE ACTIVATION-RANK INSTRUMENT (toy#183 / N3) ──
  #
  # E1's statistics over the BODY's hidden activations instead of the error
  # covariance. N3 asks why FAITHFUL feedback hurts; the pre-registered
  # hypothesis is that it collapses representational diversity.
  #
  # ASSERTED: the instrument is byte-null when off, live when on, and — the
  # load-bearing one — DOES NOT MOVE THE bpb IT SITS BESIDE. An instrument
  # that perturbs its own cell cannot be left on, and every arm it is meant
  # to compare would then be measured on a different run than the one that
  # produced the number.
  #
  # NOT ASSERTED: any direction of rank across arms. That is the hypothesis
  # under test, and the arm it names (oracle) is not on main at all.
  n0 = failures.length
  ar_off = bl_run("GTX_POLICY" => "chain,chain", "GTX_DFA_CUT" => "layer")
  ar_on  = bl_run("GTX_POLICY" => "chain,chain", "GTX_DFA_CUT" => "layer",
                  "GTX_ACTRANK" => "1", "GTX_ACTRANK_N" => "256")
  failures << "actrank: the instrument emitted a line while OFF — it must be byte-null unless GTX_ACTRANK=1" if ar_off.include?("actrank:")
  al = ar_on.lines.find { |x| x.start_with?("actrank: ") }
  if al.nil?
    failures << "actrank: GTX_ACTRANK=1 emitted no `actrank:` line"
  else
    a = Hash[al.scan(/(\w+)=([0-9.A-Za-z-]+)/)]
    failures << "actrank: n=#{a['n']} — no activation samples were collected" if a["n"].to_i.zero?
    %w[trace lambda_max participation_ratio stable_rank].each do |k|
      failures << "actrank: #{k}=#{a[k]} — a zero statistic means the Gram matrix is empty, so the walk read nothing" if a[k].to_f <= 0.0
    end
    # rank <= min(n, d_model). At this cell the ceiling is d_model=64 and the
    # measured values are ~1-3, so `ok` is the expected reading; CEILING-CAPPED
    # here would mean the statistic is reporting the model's WIDTH rather than
    # its representation, and — because the hypothesis is about COLLAPSE — a
    # ceiling reads as high rank, i.e. it looks like the hypothesis failing.
    failures << "actrank: capped=#{a['capped']} at d_model=64 with n=256 — the statistic is bounded by the ceiling rather than measured, and a ceiling reads as HIGH rank, the direction that would look like N3's hypothesis being refuted" unless a["capped"] == "ok"
  end
  # THE ONE THAT MATTERS: the measurement must not perturb the measured cell.
  off_bpb = ar_off[/bpb=([0-9.eE+-]+)/, 1]
  on_bpb  = ar_on[/bpb=([0-9.eE+-]+)/, 1]
  failures << "actrank: turning the instrument on MOVED the bpb (#{off_bpb} -> #{on_bpb}). The instrument must be byte-null on the number it sits beside, or no arm it compares is measured on the run that produced its score" unless off_bpb == on_bpb
  puts failures.length == n0 ?
    "  ok: ACTIVATION RANK (toy#183) — byte-null when off, live and uncapped when on, and the bpb is BIT-IDENTICAL with the instrument on, so it can be left enabled without invalidating the cell it measures" :
    "  FAIL: activation rank"

  # ── LEG 12e: THE ORACLE-B PATH (toy#176 / D1 + D1b) ──
  #
  # GTX_DFA_ADAPT=oracle builds a pooling P from a map pack; GTX_LDFA_CSUP
  # fills the rank above n_base with an orthonormal complement. ~200 lines of
  # engine and runner carrying the programme's most striking result — perfect
  # routing performing exactly like never training the body — and until this
  # leg NOTHING asserted any of it.
  #
  # WHAT THIS LEG CANNOT DO, stated so nobody reads it as more than it is:
  # it does not reproduce D1's cells. Those need the pooling MAP PACK, which
  # is not committed — `data/` has token packs, and the engine correctly
  # refuses one ("map covers N codes but the head has only C classes"). So
  # the oracle's BEHAVIOUR is ungated here; only its contract is.
  # Committing a small map fixture is what would close that, and it is the
  # one thing this path still needs.
  n0 = failures.length
  # 1. INERT unless asked for. This is the whole basis on which the path is
  #    safe to carry on main, so it is asserted first and on a real curve.
  orc_off = bl_run("GTX_POLICY" => "dfa,dfa", "GTX_DFA_CUT" => "layer")
  failures << "oracle: a default run mentions the oracle path — it must be byte-null unless GTX_DFA_ADAPT=oracle" if orc_off.include?("oracle P map")
  # 2. Every guard refuses FOR ITS OWN STATED REASON, not merely non-zero.
  #    Six of them, and the last two are the interesting ones: they refuse
  #    configurations that would otherwise RUN TO COMPLETION and print a
  #    plausible number for an oracle that never adapted or never matched
  #    the head.
  [[{ "GTX_LDFA_CSUP" => "active" },
    "GTX_LDFA_CSUP is meaningless without"],
   [{ "GTX_DFA_ADAPT" => "oracle" },
    "needs GTX_LDFA_MAP"],
   [{ "GTX_DFA_ADAPT" => "oracle", "GTX_LDFA_MAP" => "data/ae_shak_a65",
      "GTX_DFA_RANK" => "65", "GTX_LDFA_CSUP" => "bogus" },
    "GTX_LDFA_CSUP bogus unsupported"],
   [{ "GTX_DFA_ADAPT" => "oracle", "GTX_LDFA_MAP" => "data/ae_shak_a65",
      "GTX_DFA_RANK" => "full" },
    "meaningless at GTX_DFA_RANK=full"],
   [{ "GTX_DFA_ADAPT" => "oracle", "GTX_LDFA_MAP" => "data/ae_shak_a65",
      "GTX_DFA_RANK" => "65", "GTX_LDFA_EVERY" => "500", "STEPS" => "20" },
    "is never reached in STEPS"]].each do |env, want|
    o, st = Open3.capture2e(BL.merge("STEPS" => "20", "GTX_POLICY" => "dfa,dfa").merge(env),
                            RUNNER, chdir: ROOT)
    if st.success? && !o.include?(want)
      failures << "oracle: #{env.inspect} was ACCEPTED — it must be refused (#{want.inspect})"
    elsif !o.include?(want)
      failures << "oracle: #{env.inspect} was refused for the WRONG REASON — expected #{want.inspect}, got: #{o.lines.grep(/toy-train-gtx:|gtx_engine:/).first.to_s.strip[0, 110]}"
    end
  end
  # 3. A mismatched map is refused NAMING BOTH SIDES, so a wrong pack cannot
  #    be diagnosed by guesswork. This is also what stops a token pack being
  #    silently accepted as a pooling map.
  # GTX_LDFA_EVERY must be reachable within STEPS or the "never adapted"
  # guard fires FIRST and preempts the map check — which is what happened
  # when this probe was written, and the leg caught it.
  mo, _ = Open3.capture2e(BL.merge("STEPS" => "40", "GTX_POLICY" => "dfa,dfa",
                                   "GTX_DFA_RANK" => "65", "GTX_DFA_ADAPT" => "oracle",
                                   "GTX_LDFA_EVERY" => "10",
                                   "GTX_LDFA_MAP" => "data/ae_shak_a65"),
                          RUNNER, chdir: ROOT)
  failures << "oracle: a map/head class mismatch is not refused naming both sides — a token pack would be accepted as a pooling map" unless
    mo =~ /map covers \d+ codes but the head has only \d+ classes/
  puts failures.length == n0 ?
    "  ok: ORACLE-B PATH (toy#176) — inert unless GTX_DFA_ADAPT=oracle, six degenerate configs refused each for its OWN stated reason, and a map/head mismatch named on both sides. NOTE: the oracle's behaviour is NOT gated — that needs a committed map pack" :
    "  FAIL: oracle-B path"

  # ── LEG 12c: THE N-COST INSTRUMENT (toy#182) ──
  #
  # `graph: bytes` is the whole session graph in one number, so an arm that
  # genuinely avoided a backward chain still reports MORE if its surrogate
  # costs more to store — B0b saw exactly that and could not interpret it.
  # The `ncost:` line splits it.
  #
  # THE LOAD-BEARING ASSERTION IS bwd_nodes > 0. graph_b is reachable only
  # through tnn_graph_b_n_nodes / tnn_graph_b_node, which did not exist until
  # toy#182 — the shim's own comment referred to them for months while they
  # were absent. The first version of this instrument snapshotted
  # tnn_graph_n_nodes (graph_a) either side of tnn_build_backward and reported
  # bwd_nodes=0 on EVERY arm. Zero is not a neutral reading here: it is
  # indistinguishable from "DFA removed the entire backward chain", which is
  # the programme's headline claim. An instrument that fails silently toward
  # the conclusion you are hoping for is worse than no instrument, so a zero
  # backward count is a FAILURE, not a datum.
  n0 = failures.length
  nc = {}
  [["chain", "chain"], ["dfa", "dfa"], ["frozen", "frozen"]].each do |k, pol|
    o = bl_run("GTX_POLICY" => ([pol] * 2).join(","), "GTX_DFA_CUT" => "layer")
    l = o.lines.find { |x| x.start_with?("ncost: ") }
    nc[k] = l ? Hash[l.scan(/(\w+)=(\d+)/).map { |a, b| [a, b.to_i] }] : nil
  end
  if nc.any? { |_, v| v.nil? }
    failures << "ncost: an arm emitted no `ncost:` line (#{nc.map { |k, v| "#{k}=#{v ? 'ok' : 'MISSING'}" }.join(' ')})"
  else
    nc.each do |k, v|
      failures << "ncost: #{k} reports fwd_nodes=0 — the forward graph cannot be empty" if v["fwd_nodes"].to_i.zero?
      failures << "ncost: #{k} reports bwd_nodes=0. The backward graph is only reachable via tnn_graph_b_n_nodes/tnn_graph_b_node (toy#182); if those are missing or graph_b is unbuilt the walk silently returns nothing, and a zero backward count reads exactly like 'DFA removed the backward chain' while having measured NOTHING" if v["bwd_nodes"].to_i.zero?
    end
    # B is materialised per tap, so it must be charged to the DFA arm and to
    # nothing else. If chain reports b_bytes > 0 the separation is broken and
    # the seed-regenerated counterfactual cannot be computed.
    failures << "ncost: the dfa arm reports b_bytes=0 — B is materialised per tap, so it must be counted" if nc["dfa"]["b_bytes"].to_i.zero?
    %w[chain frozen].each do |k|
      failures << "ncost: the #{k} arm reports b_bytes=#{nc[k]['b_bytes']}, expected 0 — only DFA allocates B" unless nc[k]["b_bytes"].to_i.zero?
    end
  end
  # act_retained: the forward activations a backward node names as a source —
  # the number the cost claim is ACTUALLY about ("a block below a detach needs
  # none of its forward activations kept alive").
  #
  # ASSERTED HERE: the instrument is live and discriminating. NOT asserted:
  # that dfa retains less than chain. That is the programme's HYPOTHESIS, and
  # as measured it is false (dfa retains slightly MORE at both depths). A gate
  # that encoded the hoped-for direction would fail on correct code and would
  # have to be "fixed" by whoever later makes the claim true — the opposite of
  # a control that can lose.
  #
  # The depth check is what separates a working walk from a plausible constant:
  # a stub returning some fixed number passes "> 0" and fails this.
  deep = bl_run("GTX_POLICY" => (["chain"] * 8).join(","), "GTX_BLOCKS" => "8",
                "GTX_DFA_CUT" => "layer")
  dl = deep.lines.find { |x| x.start_with?("ncost: ") }
  deep_ret = dl ? dl[/act_retained=(\d+)/, 1].to_i : 0
  nc.each do |k, v|
    failures << "ncost: #{k} reports act_retained=0 — the retained-activation walk returned nothing. Zero here reads as 'this arm keeps no activations alive for backward', which is the cost claim itself; an instrument that fails silently toward the hoped-for answer is worse than none (toy#182)" if v["act_retained"].to_i.zero?
  end
  shallow_ret = nc["chain"]["act_retained"].to_i
  failures << "ncost: act_retained does not grow with depth (#{shallow_ret} at 2 blocks vs #{deep_ret} at 8) — retained activations must scale with depth, so a flat value means the walk is returning a constant rather than measuring the graph" unless deep_ret > shallow_ret

  puts failures.length == n0 ?
    "  ok: N-COST (toy#182) — the `ncost:` line splits forward from backward (via the graph_b accessors this issue added) and charges the materialised B separately, so the cost claim is computable instead of netted into one number; a ZERO backward count is a failure, not a DFA win" :
    "  FAIL: n-cost"

  # ── LEG 12b: BLOCK-SITE ADAPTERS ON THE BYTELM TAIL (toy#181 / B0b) ──
  #
  # `GTX_ADAPTER_SITE=block` puts one adapter per block, so trainable capacity
  # sits BELOW the frozen top. That is the point: with adapters stacked on top
  # of a frozen backbone neither arm backprops through it and DFA's structural
  # saving vanishes (F12's conclusion — the win there was the FREEZING).
  #
  # THIS LEG EXISTS BECAUSE THE FAILURE MODE IS INVISIBLE. A zero-init adapter
  # is EXACTLY THE IDENTITY, so adapters that never step produce arms that do
  # not error, do not warn and do not look wrong — they come back BIT-IDENTICAL
  # TO THE FROZEN CONTROL. That identity is the only tell.
  #
  # The assertions are two-sided and NEITHER SIDE IS OPTIONAL:
  #   frozen adapters MUST equal the control -> the zero-init site is inert
  #                                             when asked to be
  #   chain/dfa MUST NOT equal the control   -> the adapters actually TRAIN.
  #                                             This is the side that catches
  #                                             the pinned-adapter bug.
  # Asserting only the first passes with every adapter dead.
  #
  # Deliberately stated over OBSERVABLE behaviour rather than over any one
  # gating branch: during review, the branch in build_bytelm_tail! carrying the
  # explanatory comment turned out never to execute (0 hits under all three
  # adapter policies, proved by instrumentation) while the arms still separated
  # correctly. An assertion tied to that branch would have been testing code
  # that does not run.
  n0 = failures.length
  ad = {}
  [["control", {}],
   ["frozen",  { "GTX_ADAPTER_SITE" => "block", "GTX_ADAPTER_POLICY" => "frozen" }],
   ["chain",   { "GTX_ADAPTER_SITE" => "block", "GTX_ADAPTER_POLICY" => "chain" }],
   ["dfa",     { "GTX_ADAPTER_SITE" => "block", "GTX_ADAPTER_POLICY" => "dfa" }]].each do |k, extra|
    o = bl_run({ "GTX_POLICY" => "frozen,frozen", "GTX_DFA_CUT" => "layer" }.merge(extra))
    ad[k] = [o[/bpb=([0-9.eE+-]+)/, 1], o[/taps=(\d+)/, 1].to_i]
  end
  ctl = ad["control"][0]
  if ad.any? { |_, (b, _)| b.nil? }
    failures << "adapters: an arm emitted no bpb= line (#{ad.map { |k, (b, _)| "#{k}=#{b.inspect}" }.join(' ')})"
  else
    failures << "adapters: frozen block-adapters are NOT bit-identical to the frozen control (#{ad['frozen'][0]} vs #{ctl}) — W_up is zero-init, so an untrained adapter must be exactly the identity" unless ad["frozen"][0] == ctl
    %w[chain dfa].each do |arm|
      failures << "adapters: the #{arm} adapter arm is BIT-IDENTICAL to the frozen control (#{ctl}) — the adapters never step, so this arm is not the arm under test (toy#181)" if ad[arm][0] == ctl
    end
    nb = BL["GTX_BLOCKS"].to_i
    failures << "adapters: the dfa adapter arm reports taps=#{ad['dfa'][1]}, expected one per block (#{nb})" unless ad["dfa"][1] == nb
    failures << "adapters: the frozen adapter arm pushed #{ad['frozen'][1]} taps, expected 0" unless ad["frozen"][1].zero?
  end
  po, pst = Open3.capture2e(BL.merge("STEPS" => "2", "GTX_ADAPTER_SITE" => "pair"), RUNNER, chdir: ROOT)
  if pst.success?
    failures << "adapters: GTX_ADAPTER_SITE=pair outside a retrofit exited 0"
  elsif !po.include?("GTX_ADAPTER_SITE=pair is meaningless without")
    failures << "adapters: GTX_ADAPTER_SITE=pair outside a retrofit was rejected for the WRONG REASON: #{po.lines.grep(/toy-train-gtx:/).first.to_s.strip[0, 120]}"
  end
  puts failures.length == n0 ?
    "  ok: BLOCK-SITE ADAPTERS (toy#181) — frozen adapters are BIT-IDENTICAL to the frozen control so the zero-init site is provably the identity, chain AND dfa both MOVE OFF that value so the adapters demonstrably train, dfa pushes one tap per block, and `pair` outside a retrofit is refused for its own stated reason" :
    "  FAIL: block-site adapters"
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

# ── LEG 15: THE B-CONDITIONING INSTRUMENT (toy#172 / E1 Phase 1.1) ──
#
# P6's verdict is stated on the DFA arm's excess-over-noise, NOT on any
# measured rank. This instrument is a NEW claim, so what the gate owes it
# is that it is honest about its own limits rather than that it confirms
# anything.
#
# The load-bearing assertion is the SAMPLING CEILING. rank(C_E) <= n, not
# <= V, so a naive read shows "effective rank collapses with width" purely
# because the sample count stopped keeping up — which is exactly E1's
# `rank` verdict, manufactured from an artifact. `n` and `v` must appear
# on every line and a capped reading must SAY it is capped.
n0 = failures.length
if !File.file?(File.join(ROOT, "data", "ae_shak_a65.tok.i32"))
  puts "  skip: B-CONDITIONING (toy#172/E1) — data/ae_shak_a65 absent"
else
  def bc_run(extra)
    run_gtx(HW.merge({ "STEPS" => "40", "GTX_LR" => "0.003",
                       "GTX_TEXT" => "data/ae_shak_a65",
                       "GTX_POLICY" => "dfa,dfa", "GTX_DFA_CUT" => "layer" })
              .merge(extra), nil)
  end
  def bc_line(out)
    out.lines.find { |l| l.start_with?("bcond: ") } || ""
  end

  # 15a. OFF BY DEFAULT AND BYTE-NULL. An instrument that perturbs the
  # sweep it measures is worthless; every P3-P6 cell was run without it
  # and must stay comparable.
  bc_off = bc_run({})
  failures << "b-conditioning: the instrument emitted a bcond line when NOT asked for — it must be opt-in" unless
    bc_line(bc_off).empty?
  failures << "b-conditioning: GTX_INSTRUMENT unset is not byte-identical to GTX_INSTRUMENT=0" unless
    curve(bc_off) == curve(bc_run("GTX_INSTRUMENT" => "0"))

  # 15b. ON, it emits — and it does NOT move the bpb it sits beside.
  bc_on = bc_run("GTX_INSTRUMENT" => "1", "GTX_INSTRUMENT_N" => "128")
  failures << "b-conditioning: --instrument produced no bcond line" if bc_line(bc_on).empty?
  failures << "b-conditioning: no bcond_control line (B's own stable rank is the flat control that proves the draw is sane)" unless
    bc_on.lines.any? { |l| l.start_with?("bcond_control: ") }
  off_bpb = hw_bpb(bc_off)
  on_bpb  = hw_bpb(bc_on)
  failures << "b-conditioning: the instrument MOVED the bpb (#{off_bpb} -> #{on_bpb}) — a measurement that perturbs its own subject" unless
    off_bpb && on_bpb && off_bpb == on_bpb

  # 15c. n AND v ARE ON THE LINE. A rank without its sample count cannot
  # be read at all — it is the difference between a finding and an
  # artifact, and it is the one thing E1's verdict hinges on.
  bl = bc_line(bc_on)
  failures << "b-conditioning: bcond line omits n= (a rank without its sample count is unreadable — rank(C_E) <= n)" unless bl.include?(" n=")
  failures << "b-conditioning: bcond line omits v= (the ceiling cannot be re-derived without it)" unless bl.include?(" v=")
  failures << "b-conditioning: bcond line omits capped= (a sample-capped reading must SAY so, or it reads as the E1 `rank` verdict)" unless bl.include?(" capped=")
  failures << "b-conditioning: requested n=128 but the line reports otherwise (#{bl[/ n=(\d+)/, 1]})" unless bl.include?(" n=128 ")

  # 15d. THE CEILING IS DETECTED. Asking for fewer samples than a rank
  # could occupy must produce `SAMPLE-CAPPED`, not a confident number.
  # Without this the instrument's headline failure mode is silent.
  bc_tiny = bc_run("GTX_INSTRUMENT" => "1", "GTX_INSTRUMENT_N" => "8")
  tl = bc_line(bc_tiny)
  failures << "b-conditioning: n=8 against a 256-wide error vector did not report capped=SAMPLE-CAPPED or below-V — the ceiling is undetected, which is precisely how a sampling artifact becomes the `rank` verdict" unless
    tl.include?("capped=SAMPLE-CAPPED") || tl.include?("capped=below-V")

  # 15e. The statistics are finite and ordered. stable_rank <= n and
  # participation_ratio <= n hold by construction; violating either means
  # the Gram or the power iteration is wrong.
  sr = bl[/stable_rank=([0-9.eE+-]+)/, 1].to_f
  pr = bl[/participation_ratio=([0-9.eE+-]+)/, 1].to_f
  lm = bl[/lambda_max=([0-9.eE+-]+)/, 1].to_f
  failures << "b-conditioning: stable_rank #{sr} is not in (0, n] — the Gram or the power iteration is wrong" unless sr > 0.0 && sr <= 128.0 + 1e-6
  failures << "b-conditioning: participation_ratio #{pr} is not in (0, n]" unless pr > 0.0 && pr <= 128.0 + 1e-6
  failures << "b-conditioning: lambda_max #{lm} is not positive — C_E is PSD by construction" unless lm > 0.0

  puts failures.length == n0 ?
    "  ok: B-CONDITIONING (toy#172/E1) — opt-in and byte-null when off, does NOT move the bpb it sits beside, emits n/v/capped on every line, DETECTS the sampling ceiling (the artifact that would manufacture the `rank` verdict), and its statistics are finite and bounded by n" :
    "  FAIL: b-conditioning"
end

# ── LEG 16: nDFA, THE ERROR-SIDE PRECONDITIONER (toy#172 / E1 Phase 1.2) ──
#
# nDFA left-multiplies the broadcast error by lambda(C_E + lambda I)^-1,
# folded into B host-side. What this leg owes the program is FOUR things,
# and each of them is a way the flag could look like a finding while
# being nothing of the sort:
#
#   * OFF IS THE OLD BINARY. Not "off works" — byte-identical. Every
#     P3-P6 cell was measured before this flag existed and the E1 sweep
#     REUSES those cells as its nDFA=off arm, so an off path that moved
#     by one ulp would silently re-base the comparison.
#   * LARGE LAMBDA IS THE IDENTITY, BYTE-FOR-BYTE. That is what the
#     lambda-NORMALISATION buys: P = lambda(C_E+lambda I)^-1 tends to I,
#     where the unnormalised (C_E+lambda I)^-1 tends to I/lambda and
#     would silently rescale every DFA update — an LR change wearing a
#     conditioning costume. A byte identity is the only version of this
#     check that cannot be argued with.
#   * IT IS NOT INERT. An inert knob reads as a clean negative, which is
#     this program's most expensive failure mode.
#   * IT FAILS LOUD WHERE IT MEANS NOTHING. Not bytelm, no dfa block, no
#     ridge, a cadence longer than the run: each of those would otherwise
#     produce a cell that reports itself preconditioned and is not.
n0 = failures.length
if !File.file?(File.join(ROOT, "data", "ae_shak_a65.tok.i32"))
  puts "  skip: nDFA (toy#172/E1 Phase 1.2) — data/ae_shak_a65 absent"
else
  ND_BASE = HW.merge({ "STEPS" => "60", "GTX_LR" => "0.003",
                       "GTX_TEXT" => "data/ae_shak_a65",
                       "GTX_POLICY" => "dfa,dfa", "GTX_DFA_CUT" => "layer" })
  # STEPS 60 / every 20 gives THREE refreshes, so `refreshes=` is a real
  # count rather than a boolean, and m=128 at context 64 makes the
  # collection window two steps — enough for the window arithmetic to be
  # exercised rather than degenerate at one.
  ND_ON = { "GTX_NDFA" => "1", "GTX_NDFA_EVERY" => "20", "GTX_NDFA_M" => "128" }
  def nd_run(extra)
    run_gtx(ND_BASE.merge(extra), nil)
  end
  def nd_fail(extra)
    Open3.capture2e(ND_BASE.merge(extra), RUNNER, chdir: ROOT)
  end
  def nd_line(out)
    out.lines.find { |l| l.start_with?("ndfa: ") } || ""
  end

  # 16a. OFF BY DEFAULT AND BYTE-NULL.
  nd_off = nd_run({})
  failures << "ndfa: an off run emitted an ndfa: line — the flag must be silent when it is not on, or every pre-existing cell's provenance changes" unless
    nd_line(nd_off).empty?
  failures << "ndfa: GTX_NDFA unset is not byte-identical to GTX_NDFA=0. The E1 sweep REUSES the P6 cells as its off arm; if unset and 0 differ, that reuse is invalid and so is every comparison built on it" unless
    curve(nd_off) == curve(nd_run("GTX_NDFA" => "0"))

  # 16b. LARGE LAMBDA IS A BYTE IDENTITY. The bracket
  # (lambda m I + G)^-1 goes to zero, so B' underflows back to exactly B
  # in f64 long before the f32 upload — the whole point of normalising P
  # by lambda rather than shipping (C_E + lambda I)^-1 raw.
  nd_inf = nd_run(ND_ON.merge({ "GTX_NDFA_LAMBDA" => "1e30" }))
  failures << "ndfa: lambda=1e30 is NOT byte-identical to the unpreconditioned arm — P is supposed to be the IDENTITY in that limit, and if it is not then the lambda normalisation is wrong and every finite-lambda cell carries an unstated global gain" unless
    curve(nd_inf) == curve(nd_off)
  failures << "ndfa: lambda=1e30 moved the bpb it is supposed to leave alone" unless
    hw_bpb(nd_inf) == hw_bpb(nd_off)
  failures << "ndfa: lambda=1e30 did not report b_shrink=1.0 — B' is bit-identical to B there, so the ratio is exactly 1 and anything else means the norms are not being computed on what is uploaded" unless
    nd_line(nd_inf).include?("b_shrink=1.0 ")

  # 16c. IT MOVES THE DFA CURVE AT A FINITE LAMBDA.
  nd_on = nd_run(ND_ON.merge({ "GTX_NDFA_LAMBDA" => "0.01" }))
  failures << "ndfa: lambda=0.01 produced a curve byte-identical to the unpreconditioned arm — the knob is INERT, and an inert knob reads as a clean negative" if
    curve(nd_on) == curve(nd_off)

  # 16d. THE PROVENANCE CARRIES THE CADENCE. A preconditioner refreshed
  # every 20 steps and one refreshed every 500 are different experiments;
  # without the cadence on the line they carry the same label.
  nl = nd_line(nd_on)
  failures << "ndfa: no ndfa: provenance line on an nDFA run" if nl.empty?
  %w[on= lambda= every= m= gain= refreshes= err_pre= err_post= b_shrink=].each do |k|
    failures << "ndfa: provenance line omits #{k} — the cadence, the sample count and the ridge ARE the experiment on this flag" unless nl.include?(" #{k}") || nl.start_with?("ndfa: #{k}")
  end
  failures << "ndfa: the cadence is not honoured — STEPS=60 every=20 must be 3 refreshes, line says #{nl[/refreshes=(\d+)/, 1].inspect}" unless
    nl[/refreshes=(\d+)/, 1] == "3"
  failures << "ndfa: the line reports m=#{nl[/ m=(\d+)/, 1].inspect} but 128 samples were requested — an m that silently differs from the request is the Phase 1.1 sampling-ceiling trap one level down" unless
    nl[/ m=(\d+)/, 1] == "128"

  # 16e. THE PRECONDITIONED ERROR NORM IS FINITE, AND IT ACTUALLY
  # SHRANK. Finite first: a blowup in the Cholesky solve would upload
  # inf/nan into B and the loss would simply stop being a number partway
  # through a 4000-step run, with nothing in the output saying so.
  # `num_or_null` prints non-finite as `null`, so a null here is the
  # failure the spec asked to be gated.
  ep = nl[/err_pre=([0-9.eE+-]+|null)/, 1]
  eq = nl[/err_post=([0-9.eE+-]+|null)/, 1]
  failures << "ndfa: err_post is #{eq.inspect} — the preconditioned error norm is NOT FINITE, which is silent everywhere else in the run" if eq.nil? || eq == "null"
  failures << "ndfa: err_pre is #{ep.inspect} — not finite" if ep.nil? || ep == "null"
  if ep && eq && ep != "null" && eq != "null"
    failures << "ndfa: ||B'E||_F (#{eq}) is not smaller than ||BE||_F (#{ep}) at lambda=0.01 — P's eigenvalues are lambda/(lambda+s) in (0,1], so the preconditioned error CANNOT be larger; if it is, the solve is wrong" unless
      eq.to_f < ep.to_f
  end

  # 16f. FAIL LOUD WHERE THE FLAG WOULD MEAN NOTHING. Five refusals, and
  # each is a cell that would otherwise be filed as an nDFA measurement
  # while being the plain DFA arm.
  [
    [{ "GTX_NDFA" => "1", "GTX_NDFA_LAMBDA" => "0.01", "GTX_TASK" => "relational" },
     "bytelm", "accepted nDFA on the RELATIONAL task, whose head is 16 classes wide"],
    [ND_ON.merge({ "GTX_NDFA_LAMBDA" => "0.01", "GTX_POLICY" => "chain,chain" }),
     "no `dfa` block", "accepted nDFA on a chain-only policy, where there is no B to fold into"],
    [ND_ON.merge({ "GTX_POLICY" => "dfa,dfa" }),
     "GTX_NDFA_LAMBDA", "accepted nDFA with no ridge — lambda IS the experiment"],
    [{ "GTX_NDFA" => "1", "GTX_NDFA_LAMBDA" => "0.01", "GTX_NDFA_EVERY" => "1",
       "GTX_NDFA_M" => "128" },
     "collection window", "accepted a cadence shorter than its own collection window"],
    [{ "GTX_NDFA" => "1", "GTX_NDFA_LAMBDA" => "0.01", "GTX_NDFA_EVERY" => "500",
       "GTX_NDFA_M" => "128" },
     "no refresh ever ran", "accepted a run whose step budget never reaches the first refresh — that cell IS the unpreconditioned arm wearing an nDFA label"],
  ].each do |extra, needle, label|
    o, s = nd_fail(extra)
    if s.success?
      failures << "ndfa fail-loud: #{label}"
    else
      failures << "ndfa fail-loud: the refusal for `#{label}` does not name #{needle.inspect}, so it cannot be acted on:\n#{o.lines.last(2).join}" unless o.include?(needle)
    end
  end

  # 16g. THE CLI SURFACE REFUSES THE SAME THINGS. The sweeps call
  # libexec/ directly and `toy train` is a second front door; a knob
  # guarded on only one of them is guarded on neither.
  [
    [%w[mlp --dfa-feedback-precond ndfa --ndfa-lambda 0.01], "mlp accepted --dfa-feedback-precond"],
    [%w[gtx --dfa-feedback-precond ndfa --ndfa-lambda 0.01], "gtx accepted nDFA without --task bytelm"],
    [%w[gtx --task bytelm --text data/ae_shak_a65 --dfa-feedback-precond ndfa], "gtx accepted --dfa-feedback-precond ndfa with no --ndfa-lambda"],
    [%w[gtx --task bytelm --text data/ae_shak_a65 --ndfa-lambda 0.01], "gtx accepted --ndfa-lambda with the preconditioner off"],
  ].each do |argv, label|
    cout, cst = Open3.capture2e({}, TOY, "train", *argv, chdir: ROOT)
    failures << "ndfa cli: #{label} (exit #{cst.exitstatus}: #{cout.lines.last(1).join.strip})" unless cst.exitstatus == 2
  end

  puts failures.length == n0 ?
    "  ok: nDFA (toy#172/E1 Phase 1.2) — off is BYTE-IDENTICAL to the pre-flag runner (so the P6 cells are a legitimate off arm), lambda->inf is a BYTE IDENTITY rather than an approximation, a finite lambda MOVES the curve and provably SHRINKS ||B'E||_F, the cadence/m/ridge all ride in the provenance, and five configurations where the flag would mean nothing are REFUSED on both front doors" :
    "  FAIL: ndfa"
end

# ── LEG 17: LDFA, ADAPTIVE LOW-RANK FEEDBACK (toy#172 / E2) ──
#
# LDFA factorises B as Q[dout x r].P[r x V], folded into the same uploaded
# tensor. What this leg owes the program is SIX things, and every one of
# them is a way the arms could look like a finding while being nothing of
# the sort:
#
#   * `full` IS THE OLD BINARY. Byte-identical, not "works" — every P3-P6
#     cell predates the flag and the E2 sweep states its contrast against
#     them.
#   * RANK IS NOT INERT. An inert knob reads as a clean negative, which is
#     this program's most expensive failure mode. Twice over here: the
#     rank must move the curve, and `oja` must differ from `none` at the
#     SAME r, or the adaptation is a no-op wearing an adaptive label.
#   * THE SCALE MATCHES. This is the leg's load-bearing assertion. A
#     rank-r Q.P has a different ||.||_F from the full-width B it
#     replaces, so an unnormalised arm would make "low rank hurts"
#     indistinguishable from "the updates got smaller" — a global gain on
#     B is an LR change wearing a rank costume. The arms must differ in
#     RANK and not in SCALE, and it must be checkable from the output.
#   * THE EMITTED RANK IS THE RANK ASKED FOR — and rank_eff, which is
#     min(r, dout) and is NOT r once r passes d_model.
#   * OJA'S BASIS STAYS ORTHONORMAL-ISH. Silent divergence here would
#     look exactly like a result: B would fill with garbage of the right
#     norm and the bpb would simply be worse.
#   * IT FAILS LOUD WHERE IT MEANS NOTHING. Not bytelm, no dfa block,
#     r >= V, composed with nDFA, adaptation with no factorisation: each
#     of those is a cell that reports itself low-rank and is not.
n0 = failures.length
if !File.file?(File.join(ROOT, "data", "ae_shak_a65.tok.i32"))
  puts "  skip: LDFA (toy#172/E2) — data/ae_shak_a65 absent"
else
  LD_BASE = HW.merge({ "STEPS" => "60", "GTX_LR" => "0.003",
                       "GTX_TEXT" => "data/ae_shak_a65",
                       "GTX_POLICY" => "dfa,dfa", "GTX_DFA_CUT" => "layer" })
  # STEPS 60 / every 20 gives THREE refreshes, so `refreshes=` is a real
  # count rather than a boolean, and m=128 at context 64 makes the
  # collection window two steps — the window arithmetic gets exercised
  # rather than being degenerate at one.
  LD_ON = { "GTX_LDFA_EVERY" => "20", "GTX_LDFA_M" => "128" }
  def ld_run(extra)
    run_gtx(LD_BASE.merge(extra), nil)
  end
  def ld_fail(extra)
    Open3.capture2e(LD_BASE.merge(extra), RUNNER, chdir: ROOT)
  end
  def ld_line(out)
    out.lines.find { |l| l.start_with?("ldfa: ") } || ""
  end
  def ld_num(line, key)
    line[/#{Regexp.escape(key)}=([0-9.eE+-]+)/, 1]
  end

  # 17a. `full` IS BYTE-NULL, and unset means `full`.
  ld_off = ld_run({})
  failures << "ldfa: an unconfigured run emitted an ldfa: line — the flag must be silent when it is not on, or every pre-existing cell's provenance changes" unless
    ld_line(ld_off).empty?
  failures << "ldfa: GTX_DFA_RANK unset is not byte-identical to GTX_DFA_RANK=full. The E2 sweep states its contrast against cells measured before this flag existed; if unset and `full` differ, that contrast is invalid" unless
    curve(ld_off) == curve(ld_run("GTX_DFA_RANK" => "full"))
  failures << "ldfa: GTX_DFA_RANK=full moved the bpb it is supposed to leave alone" unless
    hw_bpb(ld_off) == hw_bpb(ld_run("GTX_DFA_RANK" => "full"))

  # 17b. RANK IS NOT INERT, AND ADAPTATION IS NOT INERT. The head here is
  # 256 wide (HW leaves GTX_VOCAB at its default), so r=16 is a genuine
  # 16x compression and r=64 sits at dout=64, the rank_eff boundary.
  ld_r16 = ld_run(LD_ON.merge({ "GTX_DFA_RANK" => "16" }))
  failures << "ldfa: r=16 produced a curve byte-identical to the full-width arm — the RANK knob is INERT, and an inert knob reads as a clean negative" if
    curve(ld_r16) == curve(ld_off)
  ld_r16o = ld_run(LD_ON.merge({ "GTX_DFA_RANK" => "16",
                                 "GTX_DFA_ADAPT" => "oja",
                                 "GTX_LDFA_ETA" => "0.05" }))
  failures << "ldfa: adapt=oja produced a curve byte-identical to adapt=none at the SAME r=16 — the ADAPTATION is a no-op, and the fixed-vs-adaptive contrast IS the whole hypothesis" if
    curve(ld_r16o) == curve(ld_r16)
  # ... and eta=0 must be the frozen control it claims to be: same P as
  # `none` after the shared init orthonormalisation, so the two curves
  # must agree. If they do not, something other than the Oja update
  # differs between the arms and the contrast is not a clean B-test.
  ld_r16z = ld_run(LD_ON.merge({ "GTX_DFA_RANK" => "16",
                                 "GTX_DFA_ADAPT" => "oja",
                                 "GTX_LDFA_ETA" => "0" }))
  failures << "ldfa: adapt=oja at eta=0 is NOT byte-identical to adapt=none at the same r. Both arms share the init orthonormalisation and differ only in whether the Oja update is applied, so at eta=0 they must be the same run — if they are not, the fixed/adaptive contrast is confounded by something other than the adaptivity" unless
    curve(ld_r16z) == curve(ld_r16)
  failures << "ldfa: eta=0 did not relabel itself `adapt=oja-frozen` — an eta=0 cell that reports `adapt=oja` would be filed as an adaptive measurement" unless
    ld_line(ld_r16z).include?("adapt=oja-frozen ")

  # 17c. THE SCALE MATCHES, AND IT IS CHECKABLE FROM THE OUTPUT. Without
  # this the fixed-vs-adaptive contrast — and the whole rank ladder — is
  # confounded by a scalar rather than by rank.
  [["none", ld_r16], ["oja", ld_r16o]].each do |lbl, out|
    l = ld_line(out)
    if l.empty?
      failures << "ldfa: no ldfa: provenance line on the #{lbl} arm"
      next
    end
    eff  = ld_num(l, "b_eff_fro")
    full = ld_num(l, "b_full_fro")
    rat  = ld_num(l, "scale_ratio")
    if eff.nil? || full.nil? || rat.nil? || eff == "null" || full == "null"
      failures << "ldfa (#{lbl}): the line omits or nulls b_eff_fro/b_full_fro/scale_ratio — without both norms a reader cannot tell whether the arms differ in RANK or in SCALE, which is the one thing this experiment turns on: #{l}"
    else
      # f64 host-side arithmetic, so the tolerance is numerical rather
      # than statistical. 1e-9 relative is ~7 orders looser than the
      # 1e-14 observed and still far tighter than any gain that could
      # masquerade as an LR change.
      failures << "ldfa (#{lbl}): ||B_eff||_F=#{eff} does not match ||B_full||_F=#{full} (ratio #{rat}) — the low-rank arm is running at a different SCALE from the full-width arm it is compared against, so any difference between them is confounded" unless
        (rat.to_f - 1.0).abs < 1e-9
    end
  end

  # 17d. THE EMITTED RANK IS THE RANK ASKED FOR — and rank_eff is NOT r
  # once r passes dout. On this config dout = d_model = 64, so r=128 must
  # report rank_eff=64: reading "r=128" as "rank 128" would misstate every
  # rung of the ladder above d_model.
  l16 = ld_line(ld_r16)
  failures << "ldfa: asked for rank 16, the line says #{ld_num(l16, 'rank').inspect}" unless ld_num(l16, "rank") == "16"
  failures << "ldfa: r=16 at dout=64 must report rank_eff=16, line says #{ld_num(l16, 'rank_eff').inspect}" unless ld_num(l16, "rank_eff") == "16"
  ld_r128 = ld_run(LD_ON.merge({ "GTX_DFA_RANK" => "128" }))
  l128 = ld_line(ld_r128)
  failures << "ldfa: r=128 against dout=64 must report rank_eff=64 — rank(Q.P) <= min(dout, r), so a reader who took r at face value would believe the arm had twice the feedback rank it has. Line says rank_eff=#{ld_num(l128, 'rank_eff').inspect} dout=#{ld_num(l128, 'dout').inspect}" unless
    ld_num(l128, "rank_eff") == "64" && ld_num(l128, "dout") == "64"

  # 17e. THE PROVENANCE CARRIES THE EXPERIMENT, and the cadence is
  # honoured. Same discipline as nDFA's: a P adapted every 20 steps and
  # one adapted every 500 are different experiments that would otherwise
  # carry the same label.
  lo = ld_line(ld_r16o)
  %w[rank= rank_eff= dout= v= adapt= eta= every= m= refreshes= b_eff_fro= b_full_fro= p_row_min= p_row_max= p_offdiag_max= p_energy= p_energy_rand=].each do |k|
    failures << "ldfa: provenance line omits #{k} — the rank, the cadence, the sample count and the realised scale ARE the experiment on this flag" unless lo.include?(" #{k}") || lo.start_with?("ldfa: #{k}")
  end
  failures << "ldfa: the cadence is not honoured — STEPS=60 every=20 must be 3 refreshes, line says #{lo[/refreshes=(\d+)/, 1].inspect}" unless
    lo[/refreshes=(\d+)/, 1] == "3"
  failures << "ldfa: the line reports m=#{lo[/ m=(\d+)/, 1].inspect} but 128 samples were requested — an m that silently differs from the request is the Phase 1.1 sampling-ceiling trap one level down" unless
    lo[/ m=(\d+)/, 1] == "128"

  # 17f. OJA'S BASIS STAYS ORTHONORMAL-ISH, AND IT ACTUALLY LEARNED
  # SOMETHING. Both halves matter. A diverging basis fills B with garbage
  # of exactly the right Frobenius norm and reads as a worse bpb, i.e. as
  # a RESULT. An adaptation that ran but learned nothing sits at the
  # random baseline r/V and is indistinguishable from one that worked
  # unless the captured energy is measured against that baseline.
  rmin = ld_num(lo, "p_row_min").to_f
  rmax = ld_num(lo, "p_row_max").to_f
  offd = ld_num(lo, "p_offdiag_max").to_f
  failures << "ldfa: Oja left P's row norms at [#{rmin}, #{rmax}] before re-orthonormalisation — the basis is diverging, and a diverged basis uploads as garbage of the right norm and reads as a result" unless
    rmin > 0.5 && rmax < 2.0
  failures << "ldfa: Oja left max |P_i . P_j| = #{offd} before re-orthonormalisation — the rows are collapsing onto each other, so the effective rank is below the reported one" unless
    offd < 0.25
  # The FIXED arm is the control for both: it never steps P, so its basis
  # must still be exactly the orthonormal one MGS produced at init.
  ln = ld_line(ld_r16)
  failures << "ldfa: the FIXED arm's P is not orthonormal (row norms #{ld_num(ln, 'p_row_min')}..#{ld_num(ln, 'p_row_max')}, offdiag #{ld_num(ln, 'p_offdiag_max')}) — nothing steps P on that arm, so anything but 1/1/~0 means the init orthonormalisation is wrong and the two arms do not start from the same basis" unless
    (ld_num(ln, "p_row_min").to_f - 1.0).abs < 1e-9 &&
    (ld_num(ln, "p_row_max").to_f - 1.0).abs < 1e-9 &&
    ld_num(ln, "p_offdiag_max").to_f < 1e-9
  # ... which also makes the fixed arm the right yardstick for `learned
  # anything`: a random orthonormal P captures r/V of the error energy.
  e_rand = ld_num(ln, "p_energy_rand").to_f
  e_fix  = ld_num(ln, "p_energy").to_f
  e_oja  = ld_num(lo, "p_energy").to_f
  failures << "ldfa: the FIXED arm captured #{e_fix} of the error energy against a random-basis expectation of #{e_rand} — the two should agree, since a fixed random orthonormal P IS the random baseline" unless
    e_rand > 0.0 && (e_fix / e_rand) > 0.3 && (e_fix / e_rand) < 3.0
  failures << "ldfa: Oja captured #{e_oja} of the error energy against the fixed arm's #{e_fix} — the adaptation ran but learned nothing, which is indistinguishable from a no-op except through this number" unless
    e_oja > e_fix * 1.5

  # 17g. FAIL LOUD WHERE THE FLAG WOULD MEAN NOTHING. Six refusals, each
  # a cell that would otherwise be filed as a low-rank measurement while
  # being the full-width arm.
  [
    [{ "GTX_DFA_RANK" => "16", "GTX_TASK" => "relational" },
     "bytelm", "accepted LDFA on the RELATIONAL task, whose head is 16 classes wide"],
    [LD_ON.merge({ "GTX_DFA_RANK" => "16", "GTX_POLICY" => "chain,chain" }),
     "no `dfa` block", "accepted LDFA on a chain-only policy, where there is no B to factorise"],
    [LD_ON.merge({ "GTX_DFA_RANK" => "256" }),
     "not below the error width", "accepted r >= V, which is `full` wearing a low-rank label"],
    [LD_ON.merge({ "GTX_DFA_RANK" => "16", "GTX_NDFA" => "1",
                   "GTX_NDFA_LAMBDA" => "0.01", "GTX_NDFA_EVERY" => "20",
                   "GTX_NDFA_M" => "128" }),
     "opposite interventions", "accepted LDFA composed with nDFA — two opposite interventions on the same uploaded B"],
    [{ "GTX_DFA_ADAPT" => "oja" },
     "meaningless at GTX_DFA_RANK=full", "accepted an adaptation with no factorisation to adapt"],
    [{ "GTX_DFA_RANK" => "16", "GTX_LDFA_EVERY" => "500", "GTX_LDFA_M" => "128" },
     "never reached", "accepted a run whose step budget never reaches the first refresh — that cell IS the fixed arm wearing an adaptive label"],
  ].each do |extra, needle, label|
    o, s = ld_fail(extra)
    if s.success?
      failures << "ldfa fail-loud: #{label}"
    else
      failures << "ldfa fail-loud: the refusal for `#{label}` does not name #{needle.inspect}, so it cannot be acted on:\n#{o.lines.last(2).join}" unless o.include?(needle)
    end
  end
  # The empty-string trap, explicitly: "" must be `full`, not rank 0.
  o_es, s_es = Open3.capture2e(LD_BASE.merge({ "GTX_DFA_RANK" => "" }), RUNNER, chdir: ROOT)
  failures << "ldfa: GTX_DFA_RANK=\"\" was not treated as `full` — (ENV[x] || d).to_i is 0 for the empty string, and a knob that reads an unset value as rank 0 would silently disable every feedback matrix in the run" unless
    s_es.success? && !o_es.lines.any? { |l| l.start_with?("ldfa: ") }

  # 17h. THE CLI SURFACE REFUSES THE SAME THINGS. The sweeps call
  # libexec/ directly and `toy train` is a second front door; a knob
  # guarded on only one of them is guarded on neither.
  [
    [%w[mlp --dfa-feedback-rank 16], "mlp accepted --dfa-feedback-rank"],
    [%w[gtx --dfa-feedback-rank 16], "gtx accepted LDFA without --task bytelm"],
    [%w[gtx --task bytelm --text data/ae_shak_a65 --dfa-feedback-adapt oja], "gtx accepted --dfa-feedback-adapt at rank full"],
    [%w[gtx --task bytelm --text data/ae_shak_a65 --dfa-feedback-rank 16 --ldfa-eta 0.05], "gtx accepted --ldfa-eta without --dfa-feedback-adapt oja"],
    [%w[gtx --task bytelm --text data/ae_shak_a65 --vocab 4096 --dfa-feedback-rank 8192], "gtx accepted r >= vocab"],
    [%w[gtx --task bytelm --text data/ae_shak_a65 --dfa-feedback-rank 16 --dfa-feedback-precond ndfa --ndfa-lambda 0.01], "gtx accepted LDFA composed with nDFA"],
    [%w[gtx --task bytelm --text data/ae_shak_a65 --dfa-feedback-rank bogus], "gtx accepted a non-integer, non-`full` rank"],
  ].each do |argv, label|
    cout, cst = Open3.capture2e({}, TOY, "train", *argv, chdir: ROOT)
    failures << "ldfa cli: #{label} (exit #{cst.exitstatus}: #{cout.lines.last(1).join.strip})" unless cst.exitstatus == 2
  end

  puts failures.length == n0 ?
    "  ok: LDFA (toy#172/E2) — `full` is BYTE-IDENTICAL to the pre-flag runner, rank MOVES the curve and adaptation moves it AGAIN at the same r, eta=0 is byte-identical to the fixed arm (so the contrast is the Oja update and nothing else), ||B_eff||_F matches ||B_full||_F to 1e-9 so the arms differ in RANK and not in SCALE, rank_eff is reported as min(r, dout) rather than r, Oja's basis stays orthonormal-ish and provably CAPTURES more error energy than the random baseline, and thirteen configurations where the flag would mean nothing are REFUSED on both front doors" :
    "  FAIL: ldfa"
end

if failures.empty?
  puts "GATE PASS [gtx]: graph transformer + RETROFIT + per-block policy — byte fixture, the B seed, the small-head assertion, the MANDATORY success bar with each arm at ITS OWN best LR showing ATTENTION IS NOT DFA-HOSTILE (dfa .920 vs BP .985 vs frozen .111), the mixing-cut collapse, and a frozen control that provably CAN lose (toy#160)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [gtx]: #{f}" }
  exit 1
end
