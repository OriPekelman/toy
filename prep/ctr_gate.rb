#!/usr/bin/env ruby
# prep/ctr_gate.rb — toy#154 (DFA-arch T1) gate for the CTR tower
# (libexec/toy-train-ctr, `toy train ctr`).
#
# Legs:
#   1. BYTE FIXTURE + CHAIN BYTE-NULL — the default 5-step curve equals
#      the recorded baseline, and an explicit all-chain policy is
#      byte-identical to no policy at all.
#   2. DETERMINISM — two identical dfa runs, identical stdout.
#   3. WIRING (structure, not curves) — run_start.dfa reports the exact
#      dfa_wired / frozen counts, error_dim 1, and the scalar head.
#   4. THE SURROGATE ROOTS REACH THE TOWER — changing --dfa-b-seed
#      changes the curve. If the roots were unwired the tower would be
#      frozen and the feedback seed could not matter; the loss curve
#      alone would look entirely plausible.
#   5. THE EMBEDDINGS TRAIN UNDER A FULLY-DFA TOWER — the all-frozen
#      arm still improves its AUC over an untrained model, which is
#      only possible because the tables + head train by BP. This is the
#      property that forced the toy#158 surrogate-root construction
#      instead of toy#152's direct-gradient one.
#   6. THE TASK CAN TELL THE ARMS APART — BP must beat the frozen
#      control by a wide margin, or the success bar is unfalsifiable
#      and every number below is meaningless (the toy#152-with-blobs
#      lesson: check the control CAN lose).
#   7. DFA BEATS THE FROZEN CONTROL — the half of the bar this lane
#      does meet.
#   8. FAIL-LOUD on degenerate configs.
#
# WHAT THIS GATE DELIBERATELY DOES NOT ASSERT: that DFA lands within
# ~0.01 AUC of BP. It does not — measured 0.058-0.067 behind across
# seeds 0/1/2 — so gating on the ticket's target would gate on a
# falsehood. The gate pins the ORDERING that is real and reproducible
# (BP > DFA > frozen) and PRINTS the gap, so an improvement to DFA
# shows up as a number moving rather than a gate flipping.

ROOT    = File.expand_path("..", __dir__)
RUNNER  = File.join(ROOT, "libexec", "toy-train-ctr")
TOY     = File.join(ROOT, "bin", "toy")
FIXTURE = File.join(ROOT, "prep", "fixtures", "train_ctr_baseline.txt")

require "open3"
require "json"
require "tmpdir"
require "fileutils"

# The comparison cell. CTR_PAIRS=40 and the default CTR_LIN_SCALE=0.25
# are what make the CROSSES carry the signal; at the shipped defaults
# of 12 pairs / lin_scale 1.0 the additive part dominates, embeddings +
# head alone solve it, and all three arms tie within 0.003 AUC with the
# FROZEN control ahead. Both facts are recorded in toy_ctr_task.rb.
ARMS = { "STEPS" => "2000", "SEED" => "0", "CTR_PAIRS" => "40" }.freeze
# Measured across seeds 0/1/2 at this cell:
#   BP - frozen : .0856 / .0821 / .0807     (the task discriminates)
#   DFA - frozen: .0188 / .0243 / .0217     (DFA learns something)
#   BP - DFA    : .0668 / .0578 / .0590     (parity NOT reached)
# Thresholds sit well below the observed spread.
DISCRIMINATES = 0.05
DFA_EDGE      = 0.008

def run_ctr(extra_env, run_dir)
  env = { "STEPS" => "5", "SEED" => "0" }.merge(extra_env)
  env = env.merge("TAO_RUN_DIR" => run_dir, "TOY_RUN_ID" => "ctr-gate") if run_dir
  out, st = Open3.capture2e(env, RUNNER, chdir: ROOT)
  abort "ctr_gate: runner exited #{st.exitstatus}:\n#{out.lines.last(10).join}" unless st.success?
  out
end

def curve(out)
  out.lines.select { |l| l.start_with?("step ") }
end

def auc(out)
  line = out.lines.find { |l| l.start_with?("val: ") }
  raise "ctr_gate: no val line in\n#{out}" unless line
  line[/auc=([0-9.eE+-]+)/, 1].to_f
end

unless File.executable?(RUNNER)
  build_out, build_st = Open3.capture2e("make", "-C", ROOT, "libexec/toy-train-ctr")
  unless build_st.success? && File.executable?(RUNNER)
    warn "ctr_gate: build failed:\n#{build_out.lines.last(15).join}"
    exit 2
  end
end

failures = []
# LEG BOOKKEEPING: every leg records the failure count at its START in
# `n0` and summarises with `failures.length == n0` (d878143).
n0 = 0

# ---- 1. byte fixture + chain byte-null ----
n0 = failures.length
base_out   = run_ctr({}, nil)
base_curve = curve(base_out)
if File.file?(FIXTURE)
  expect = File.readlines(FIXTURE).reject { |l| l.start_with?("#") || l.strip.empty? }
  failures << "fixture: default curve != train_ctr_baseline.txt\ngot:  #{base_curve.join}want: #{expect.join}" unless base_curve == expect
else
  failures << "fixture: #{FIXTURE} missing"
end
chain_out = run_ctr({ "CTR_POLICY" => "chain,chain,chain" }, nil)
failures << "chain byte-null: explicit all-chain curve != absent-policy curve" unless curve(chain_out) == base_curve
puts failures.length == n0 ? "  ok: default curve matches the fixture AND an explicit all-chain policy is byte-null" : "  FAIL: fixture / chain byte-null"

# ---- 2. determinism ----
n0 = failures.length
d1 = run_ctr({ "CTR_POLICY" => "dfa,dfa,dfa", "STEPS" => "30" }, nil)
d2 = run_ctr({ "CTR_POLICY" => "dfa,dfa,dfa", "STEPS" => "30" }, nil)
failures << "determinism: two identical dfa runs differ" unless d1 == d2
puts failures.length == n0 ? "  ok: two identical dfa runs are byte-identical (curve + val)" : "  FAIL: determinism"

# ---- 3. wiring / provenance ----
n0 = failures.length
Dir.mktmpdir("ctr_gate") do |dir|
  out = run_ctr({ "CTR_POLICY" => "dfa,dfa,frozen", "STEPS" => "10",
                  "CTR_B_SEED" => "42" }, dir)
  failures << "bundle: runner printed no val line" unless out.lines.any? { |l| l.start_with?("val: ") }
  ev_path = File.join(dir, "events.jsonl")
  if File.file?(ev_path)
    events = File.readlines(ev_path).map { |l| JSON.parse(l) }
    rs = events.first || {}
    failures << "bundle: first event not run_start" unless rs["kind"] == "run_start"
    failures << "bundle: last event not run_end" unless events.last && events.last["kind"] == "run_end"
    md = rs["model"] || {}
    failures << "bundle: model.arch != ctr" unless md["arch"] == "ctr"
    # The output really is SCALAR — that is the regime under test.
    failures << "bundle: model.num_classes != 1 (this lane is the scalar-output case)" unless md["num_classes"] == 1
    failures << "bundle: model.head != sigmoid-scalar" unless md["head"] == "sigmoid-scalar"
    df = rs["dfa"]
    if df.nil?
      failures << "bundle: run_start has no dfa object"
    else
      failures << "bundle: dfa.policy != [1,1,2] (got #{df['policy'].inspect})" unless df["policy"] == [1, 1, 2]
      failures << "bundle: dfa.dfa_wired != 2 (got #{df['dfa_wired'].inspect}) — an unwired root looks exactly like 'DFA did nothing'" unless df["dfa_wired"] == 2
      failures << "bundle: dfa.frozen != 1" unless df["frozen"] == 1
      failures << "bundle: dfa.error_dim != 1" unless df["error_dim"] == 1
    end
    evs = events.select { |e| e["kind"] == "eval" }
    if evs.length != 1
      failures << "bundle: #{evs.length} eval events (want 1)"
    else
      a = evs.first["auc"]
      failures << "bundle: eval auc #{a.inspect} not in [0,1]" unless a.is_a?(Numeric) && a >= 0.0 && a <= 1.0
      failures << "bundle: eval carries no positive count" unless evs.first["pos"].is_a?(Numeric) && evs.first["pos"] > 0
    end
  else
    failures << "bundle: no events.jsonl"
  end
  fj = File.join(dir, "flow.json")
  failures << "bundle: no flow.json" unless File.file?(fj)
end
puts failures.length == n0 ? "  ok: provenance — scalar head, error_dim 1, exact dfa_wired/frozen counts" : "  FAIL: wiring / bundle"

# ---- 4. the surrogate roots reach the tower ----
n0 = failures.length
b1 = run_ctr({ "CTR_POLICY" => "dfa,dfa,dfa", "STEPS" => "30", "CTR_B_SEED" => "1" }, nil)
b2 = run_ctr({ "CTR_POLICY" => "dfa,dfa,dfa", "STEPS" => "30", "CTR_B_SEED" => "2" }, nil)
failures << "roots: the B seed does not change the curve — the surrogate roots are not reaching the tower weights" if curve(b1) == curve(b2)
puts failures.length == n0 ? "  ok: the feedback seed moves the curve — the surrogate roots really drive the tower" : "  FAIL: surrogate roots"

# ---- 5. the embeddings train under a frozen/DFA tower ----
# The all-frozen arm updates NO tower weight, so any improvement over
# an untrained model comes from the embedding tables + head training by
# backprop THROUGH the frozen tower. That is the property that forced
# the surrogate-root construction (toy#152's direct-gradient rule
# propagates nothing and would leave the tables at init).
n0 = failures.length
fz_short = auc(run_ctr(ARMS.merge("CTR_POLICY" => "frozen,frozen,frozen", "STEPS" => "20"), nil))
fz_long  = auc(run_ctr(ARMS.merge("CTR_POLICY" => "frozen,frozen,frozen"), nil))
failures << "embeddings: the frozen-tower arm did not improve with training (#{fz_short.round(4)} -> #{fz_long.round(4)}) — the tables are not training" unless fz_long > fz_short + 0.02
puts failures.length == n0 ? "  ok: embeddings train under a fully-frozen tower (AUC #{fz_short.round(4)} -> #{fz_long.round(4)}) — BP reaches the tables" : "  FAIL: embeddings"

# ---- 6 + 7. the arms ----
n0 = failures.length
a_chain  = auc(run_ctr(ARMS.merge("CTR_POLICY" => "chain,chain,chain"), nil))
a_dfa    = auc(run_ctr(ARMS.merge("CTR_POLICY" => "dfa,dfa,dfa"), nil))
a_frozen = fz_long
puts format("    chain=%.4f  dfa=%.4f  frozen=%.4f   (BP-DFA gap %.4f, DFA-frozen edge %.4f)",
            a_chain, a_dfa, a_frozen, a_chain - a_dfa, a_dfa - a_frozen)
# THE TASK MUST DISCRIMINATE, or nothing else in this gate means
# anything: if BP cannot beat the frozen control the tower is not
# load-bearing and "DFA within X of BP" is satisfied by doing nothing.
if a_chain < a_frozen + DISCRIMINATES
  failures << "DISCRIMINATION: BP #{a_chain.round(4)} does not beat frozen #{a_frozen.round(4)} by #{DISCRIMINATES} — the tower is not load-bearing on this task, so the success bar is unfalsifiable"
end
if a_dfa < a_frozen + DFA_EDGE
  failures << "CONTROL: dfa #{a_dfa.round(4)} does not beat frozen #{a_frozen.round(4)} by #{DFA_EDGE}"
end
puts failures.length == n0 ?
  "  ok: the task discriminates (BP beats frozen by #{(a_chain - a_frozen).round(4)}) AND dfa beats the frozen control" :
  "  FAIL: arms"

# ---- 8. fail-loud ----
n0 = failures.length
[
  [{ "CTR_POLICY" => "chain,chain,chain,chain" }, "policy longer than CTR_LAYERS"],
  [{ "CTR_POLICY" => "dfa,bogus,chain" },         "unknown policy token"],
  [{ "CTR_LAYERS" => "0" },                       "CTR_LAYERS=0"],
  [{ "CTR_CARD" => "1" },                         "CTR_CARD=1"],
  [{ "CTR_BATCH" => "1" },                        "CTR_BATCH=1 (AUC needs both classes)"],
  [{ "CTR_BASE_RATE" => "1.5" },                  "CTR_BASE_RATE out of (0,1)"],
].each do |env, label|
  out, st = Open3.capture2e({ "STEPS" => "2" }.merge(env), RUNNER, chdir: ROOT)
  failures << "fail-loud: #{label} exited 0 (silently did nothing):\n#{out.lines.last(2).join}" if st.success?
end
puts failures.length == n0 ? "  ok: 6 degenerate configs all fail loud" : "  FAIL: fail-loud"

# ---- 9. CLI parity ----
n0 = failures.length
Dir.mktmpdir("ctr_gate_cli") do |dir|
  out, st = Open3.capture2e({ "SPINEL_SKIP_PIN_CHECK" => nil }, TOY, "train", "ctr",
                            "--steps", "5", "--seed", "0", "--out", dir, chdir: ROOT)
  if st.success?
    failures << "cli: `toy train ctr` curve != the runner's default curve" unless curve(out) == base_curve
    failures << "cli: `toy train ctr` did not echo the val line" unless out.lines.any? { |l| l.start_with?("val: ") }
  else
    failures << "cli: `toy train ctr` exited #{st.exitstatus}:\n#{out.lines.last(5).join}"
  end
  cout, cst = Open3.capture2e({}, TOY, "train", "ctr", "--device", "cuda", chdir: ROOT)
  failures << "cli: ctr accepted --device cuda (CPU-only by decision)" unless cst.exitstatus == 2
end
puts failures.length == n0 ? "  ok: CLI reproduces the curve; --device cuda rejected (tao#18)" : "  FAIL: CLI"

if failures.empty?
  puts "GATE PASS [ctr]: T1 CTR tower — scalar-output DFA wired and driving the tower, embeddings training through it, a task that DISCRIMINATES, and dfa over the frozen control (toy#154)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [ctr]: #{f}" }
  exit 1
end
