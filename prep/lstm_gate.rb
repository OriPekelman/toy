#!/usr/bin/env ruby
# prep/lstm_gate.rb — toy#157 (DFA-arch T3) gate for the LSTM lane
# (libexec/toy-train-lstm, `toy train lstm`).
#
# Legs:
#   1. BYTE FIXTURE + CHAIN BYTE-NULL.
#   2. DETERMINISM.
#   3. FORWARD-IDENTITY: step 1 under `dfa` is BYTE-IDENTICAL to `chain`
#      (tnn_detach is forward-identity), so the arms are comparable at
#      all; steps 2+ diverge, so the backward really is different.
#   4. THE B SEED MOVES THE DFA CURVE and is INERT on chain.
#   5. WIRING + BUNDLE: the tap counts the cut asks for, provenance, the
#      eval event, run_end last, and `cost.graph_bytes`.
#   6. THE MANDATORY SUCCESS BAR (tao#19 item 4), stated over THREE
#      SEEDS — see below for why one seed will not do here.
#   7. THE CROSS-ARCHITECTURE CONTRAST: the IDENTICAL per-step cut, on
#      toy#155's selective scan, at its own cell. This lane's reason to
#      exist.
#   8. THE MEMORY MEASUREMENT, in the units the ticket asks for, and
#      8b. the ANALYTIC streaming instrument beside it (toy#159).
#   8c. --clip-grad, THE FAIR BPTT CONTROL (toy#162).
#   9. THE DEGENERATE TASK, measured.
#  10. FAIL-LOUD.
#  11. CLI.
#
# ── WHAT THIS LANE MEASURED, AND WHY THE BAR RUNS THREE SEEDS ──
#
# The point of the lane is a CROSS-ARCHITECTURE contrast. toy#155 ran the
# per-step cut (no gradient crosses a timestep at all) on a selective
# scan and it collapsed to chance. This lane runs the IDENTICAL cut, on
# the IDENTICAL task generator, on a gated LSTM — and it solves the task.
# The gates are the difference: a forget gate can hold the cue in c_t
# with no gradient crossing time, so a per-step random-feedback update
# has something to sharpen; the SSM's input-dependent decay — the part
# that makes it good — is the part that needs credit assigned ACROSS
# time. "Can you delete BPTT?" has no architecture-independent answer.
# Leg 7 runs BOTH lanes so that claim is one measurement, not two
# anecdotes.
#
# THREE SEEDS, because one would have shipped a wrong result. Every arm
# here is bimodal — it solves the task (~1.000) or sits at chance (.250
# on 4 classes) — and at 4000 steps the measured grid is:
#
#   lr     BP (seeds 0/1/2)          DFA step cut (seeds 0/1/2)
#   0.005  .504 / 1.000 / .996       1.000 / 1.000 / 1.000
#   0.010  .250 / 1.000 / .996       1.000 / 1.000 / 1.000
#   0.020  .250 /  .996 / 1.000      1.000 /  .996 / 1.000
#   0.030  1.000 / .250 /  .238      1.000 / 1.000 /  .996
#
# The cut solves 12 of 12 cells; BPTT solves 7, and WHICH rate works
# depends on the SEED. So a one-seed bar would report "DFA ties BP" at
# lr 0.03 or "DFA BEATS BP" at 0.01, both on luck. The cell below is
# BP's OWN BEST over an (lr x warmup x seed) sweep — with warmup 200 at
# lr 0.02, BP reads 1.000/.992/1.000 — and the bar is stated there, on
# all three seeds, so "matches BP" means matching a BP that trained.
#
# THE STABILITY HALF NOW HAS ITS FAIR CONTROL (toy#162). Gradient
# clipping — the standard fix for exactly this BPTT failure mode — is
# implemented (`--clip-grad`, global norm, off by default) and it does
# NOT rescue BP. Cells solved (>= .9) out of the 12 (lr x seed) at 4000
# steps: unclipped 7, clip 1.0 -> 8, clip 0.1 -> 7, while the per-step
# DFA cut stays 12/12 with clipping on. Clipping RE-ROLLS which cells
# land in the good basin (at clip 0.1, seed 0 solves everywhere and seed
# 2 mostly fails — the exact inverse of the unclipped pattern) without
# making BP reliable. So the fragility is BPTT's, not the harness's.
ROOT    = File.expand_path("..", __dir__)
RUNNER  = File.join(ROOT, "libexec", "toy-train-lstm")
SSM     = File.join(ROOT, "libexec", "toy-train-ssm")
TOY     = File.join(ROOT, "bin", "toy")
FIXTURE = File.join(ROOT, "prep", "fixtures", "train_lstm_baseline.txt")

require "open3"
require "json"
require "tmpdir"
require "fileutils"

# BP's OWN BEST CELL, found by sweeping lr x warmup x seed (see above).
# Warmup is load-bearing for the BP arm and inert-ish for the DFA ones:
# at lr 0.02 with no warmup BP reads .250 at seed 0, with warmup 200 it
# reads 1.000. Do not "simplify" it away — and note it is spelled out
# here rather than defaulted in the runner ON PURPOSE: shipping it as a
# default relabelled a consumer's experiment within hours of landing.
CELL  = { "STEPS" => "4000", "LSTM_LR" => "0.02", "LSTM_WARMUP" => "200" }.freeze
SEEDS = %w[0 1 2].freeze
# Margins sit far under the measured effects rather than at any seed's
# exact value: BP clears frozen by ~.73 at this cell, and the per-step
# cut lands within .01 of BP.
BP_PRECOND  = 0.30   # BP must beat its own frozen control — the comparison must be real
STEP_FLOOR  = 0.90   # the per-step cut must SOLVE the task, not merely track BP
BP_GAP      = 0.06   # ... and stay within this of BP
FROZEN_EDGE = 0.30

def run_lstm(extra_env, run_dir)
  env = { "STEPS" => "5", "SEED" => "0" }.merge(extra_env)
  env = env.merge("TAO_RUN_DIR" => run_dir, "TOY_RUN_ID" => "lstm-gate") if run_dir
  out, st = Open3.capture2e(env, RUNNER, chdir: ROOT)
  abort "lstm_gate: runner exited #{st.exitstatus}:\n#{out.lines.last(10).join}" unless st.success?
  out
end

def run_ssm(extra_env)
  env = { "STEPS" => "5", "SEED" => "0" }.merge(extra_env)
  out, st = Open3.capture2e(env, SSM, chdir: ROOT)
  abort "lstm_gate: ssm runner exited #{st.exitstatus}:\n#{out.lines.last(10).join}" unless st.success?
  out
end

def curve(out)
  out.lines.select { |l| l.start_with?("step ") }
end

def val_acc(out)
  line = out.lines.find { |l| l.start_with?("val: ") }
  raise "lstm_gate: no val line in\n#{out}" unless line
  line[/acc=([0-9.eE+-]+)/, 1].to_f
end

def graph_bytes(out)
  line = out.lines.find { |l| l.start_with?("graph: ") }
  raise "lstm_gate: no graph line in\n#{out}" unless line
  b = line[/bytes=(\d+)/, 1]
  raise "lstm_gate: graph line has no bytes= field: #{line}" unless b
  b.to_i
end

# toy#159 — the ANALYTIC line. Returns {bptt:, sqrt_t:, cut:}.
def stream_bytes(out)
  line = out.lines.find { |l| l.start_with?("stream: ") }
  raise "lstm_gate: no stream line in\n#{out}" unless line
  h = {}
  %w[bptt sqrt_t cut].each do |k|
    v = line[/\b#{k}=(\d+)/, 1]
    raise "lstm_gate: stream line has no #{k}= field: #{line}" unless v
    h[k] = v.to_i
  end
  h
end

unless File.executable?(RUNNER)
  build_out, build_st = Open3.capture2e("make", "-C", ROOT, "libexec/toy-train-lstm")
  unless build_st.success? && File.executable?(RUNNER)
    warn "lstm_gate: build failed:\n#{build_out.lines.last(15).join}"
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
base_out   = run_lstm({}, nil)
base_curve = curve(base_out)
if File.file?(FIXTURE)
  expect = File.readlines(FIXTURE).reject { |l| l.start_with?("#") || l.strip.empty? }
  failures << "fixture: default curve != train_lstm_baseline.txt\ngot:  #{base_curve.join}want: #{expect.join}" unless base_curve == expect
else
  failures << "fixture: #{FIXTURE} missing"
end
chain_out = run_lstm({ "LSTM_POLICY" => "chain" }, nil)
failures << "chain byte-null: explicit all-chain policy curve != absent-policy curve" unless curve(chain_out) == base_curve
puts failures.length == n0 ? "  ok: default curve matches the fixture AND an explicit all-chain policy is byte-null" : "  FAIL: fixture / chain byte-null"

# ---- 2. determinism ----
n0 = failures.length
d1 = run_lstm({ "LSTM_POLICY" => "dfa", "STEPS" => "20" }, nil)
d2 = run_lstm({ "LSTM_POLICY" => "dfa", "STEPS" => "20" }, nil)
failures << "determinism: two identical dfa runs differ" unless d1 == d2
puts failures.length == n0 ? "  ok: two identical dfa runs are byte-identical" : "  FAIL: determinism"

# ---- 3. forward-identity, then divergence ----
# tnn_detach is forward-identity, so a dfa build computes the SAME
# forward pass as a chain build — which is what makes their numbers
# comparable at all. Steps 2+ must then differ, or the backward is not
# actually different and the "arm" is a relabelling.
n0 = failures.length
ch20 = run_lstm({ "STEPS" => "20" }, nil)
st20 = run_lstm({ "LSTM_POLICY" => "dfa", "LSTM_DFA_CUT" => "step", "STEPS" => "20" }, nil)
failures << "forward identity: step 1 differs between dfa and chain — tnn_detach is supposed to be forward-identity, so the two arms are not comparable" unless curve(d1).first == curve(ch20).first
failures << "forward identity: step 1 differs between dfa(step) and chain — the per-step detach of h_{t-1}/c_{t-1} must be forward-identity too, or the cut changes the MODEL and not just the credit rule" unless curve(st20).first == curve(ch20).first
failures << "arm effect: the whole dfa curve equals chain — the backward is not different" if curve(d1) == curve(ch20)
failures << "arm effect: dfa(step) curve equals dfa(layer) — cutting the time axis did nothing" if curve(st20) == curve(d1)
fz = run_lstm({ "LSTM_POLICY" => "frozen", "STEPS" => "20" }, nil)
failures << "arm effect: frozen curve identical to chain" if curve(fz) == curve(ch20)
puts failures.length == n0 ? "  ok: step 1 is byte-identical to BP under BOTH cuts (detach is forward-identity) and steps 2+ diverge" : "  FAIL: forward identity / arm effect"

# ---- 4. the B seed moves the dfa curve, and is inert on chain ----
# toy#158's discipline, and the ONLY cheap proof the random feedback
# reaches the weights. This lane cannot use cos(g_dfa, g_bp): its DFA
# update arrives through autodiff from the surrogate roots, so it lands
# in the same accumulator a BP run would use and there is no second
# tensor to compare against. The CLI rejects --align-events for exactly
# this reason (leg 11).
n0 = failures.length
dfa_b999   = run_lstm({ "LSTM_POLICY" => "dfa", "STEPS" => "20", "LSTM_B_SEED" => "999" }, nil)
chain_b999 = run_lstm({ "STEPS" => "20", "LSTM_B_SEED" => "999" }, nil)
failures << "b seed: --dfa-b-seed does not move the dfa curve — the feedback matrix is not reaching the weights" if curve(dfa_b999) == curve(d1)
failures << "b seed: --dfa-b-seed moved the CHAIN curve — a pure-BPTT arm must not see the feedback matrix" unless curve(chain_b999) == curve(ch20)
puts failures.length == n0 ? "  ok: the B seed moves dfa and is inert on chain (toy#158)" : "  FAIL: b seed"

# ---- 5. wiring + bundle ----
n0 = failures.length
Dir.mktmpdir("lstm_gate") do |dir|
  out = run_lstm({ "LSTM_POLICY" => "dfa,frozen", "LSTM_LAYERS" => "2",
                   "LSTM_SEQ" => "16", "STEPS" => "10",
                   "LSTM_DFA_CUT" => "step", "LSTM_B_SEED" => "42" }, dir)
  failures << "bundle: runner printed no val line" unless out.lines.any? { |l| l.start_with?("val: ") }
  failures << "bundle: runner printed no graph line" unless out.lines.any? { |l| l.start_with?("graph: ") }
  # The tap count is the structural assertion. Under the STEP cut a dfa
  # layer taps h_t at EVERY step; under the layer cut it taps once, at
  # the readout step. Unlike the SSM lane there is only ONE tap family:
  # h_t IS the layer output and it reaches the prediction from every
  # step, because it is what propagates forward in time.
  wire = out.lines.find { |l| l.start_with?("lstm: ") }
  if wire.nil?
    failures << "bundle: no wiring line"
  else
    failures << "wiring: dfa_wired != 1 (#{wire.strip})" unless wire.include?("dfa_wired=1")
    failures << "wiring: frozen != 1 (#{wire.strip})" unless wire.include?("frozen=1")
    failures << "wiring: cut != step (#{wire.strip})" unless wire.include?("cut=step")
    failures << "wiring: taps != 16 — the step cut must tap h_t at EVERY step (#{wire.strip})" unless wire.include?("taps=16")
  end
  lay = run_lstm({ "LSTM_POLICY" => "dfa,frozen", "LSTM_LAYERS" => "2",
                   "LSTM_SEQ" => "16", "STEPS" => "2" }, nil)
  lwire = lay.lines.find { |l| l.start_with?("lstm: ") }
  if lwire.nil?
    failures << "wiring: no wiring line under the layer cut"
  else
    failures << "wiring: the LAYER cut must tap ONCE, at the readout step (#{lwire.strip})" unless lwire.include?("taps=1") && lwire.include?("cut=layer")
  end
  ev_path = File.join(dir, "events.jsonl")
  if File.file?(ev_path)
    events = File.readlines(ev_path).map { |l| JSON.parse(l) }
    rs = events.first || {}
    failures << "bundle: first event not run_start" unless rs["kind"] == "run_start"
    failures << "bundle: schema != toy/v1" unless rs["schema"] == "toy/v1"
    # The lane NAME, asserted: this runner was written from the ssm one
    # and shipped with run_start.name = "ssm", which is invisible in the
    # curve and mislabels every bundle a consumer reads.
    failures << "bundle: run_start.name != lstm (#{rs['name'].inspect}) — a bundle that names the sibling lane is a mislabelled experiment" unless rs["name"] == "lstm"
    failures << "bundle: last event not run_end" unless events.last && events.last["kind"] == "run_end"
    md = rs["model"] || {}
    failures << "bundle: model.arch != lstm (#{md['arch'].inspect})" unless md["arch"] == "lstm"
    failures << "bundle: model.cell != lstm" unless md["cell"] == "lstm"
    failures << "bundle: model.readout != last_step — a mean-pool readout would delete the memory requirement this lane exists to test" unless md["readout"] == "last_step"
    co = rs["cost"] || {}
    failures << "bundle: cost fields not positive" unless %w[total_params active_params graph_nodes graph_bytes].all? { |k| co[k].is_a?(Numeric) && co[k] > 0 }
    # graph_bytes is THE instrument this lane adds over toy#155's node
    # count, so assert it is really bytes and not a second node count.
    failures << "bundle: cost.graph_bytes (#{co['graph_bytes'].inspect}) is not larger than cost.graph_nodes — it is supposed to be summed ggml_nbytes, not a node count" unless co["graph_bytes"].is_a?(Numeric) && co["graph_nodes"].is_a?(Numeric) && co["graph_bytes"] > co["graph_nodes"]
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
puts failures.length == n0 ? "  ok: bundle structure, the lane's own name, graph_bytes, and the tap count each cut asks for" : "  FAIL: wiring / bundle"

# ---- 6. the MANDATORY success bar, at BP's OWN BEST CELL, three seeds ----
n0 = failures.length
# These 12 cells (3 seeds x 4 arms) at 4000 steps ARE this gate's cost —
# they made it the whole battery's critical path at ~15 min, against ~29
# min of total work across all 55 legs, so no amount of `make -j` could
# get under it.
#
# They are also completely independent: separate processes, each with its
# own SEED and policy, aggregated only afterwards. So they run in a
# bounded worker pool. Ruby threads are the right tool DESPITE the GVL
# because every one of them is blocked in Open3 waiting on a subprocess,
# which releases it.
#
# WHAT DOES NOT CHANGE: every assertion below reads `rows`, keyed by seed
# and arm, exactly as before. Nothing is sampled, dropped or reordered —
# the 12 cells are the same 12 cells. The per-seed line is printed after
# the join, in SEEDS order, so the gate's output stays deterministic even
# though completion order is not.
require "etc"
LSTM_GATE_JOBS = Integer(ENV["LSTM_GATE_JOBS"] || "4")
cells = SEEDS.flat_map do |s|
  [["chain",  s, CELL.merge("SEED" => s, "LSTM_POLICY" => "chain")],
   ["dfa",    s, CELL.merge("SEED" => s, "LSTM_POLICY" => "dfa", "LSTM_DFA_CUT" => "layer")],
   ["step",   s, CELL.merge("SEED" => s, "LSTM_POLICY" => "dfa", "LSTM_DFA_CUT" => "step")],
   ["frozen", s, CELL.merge("SEED" => s, "LSTM_POLICY" => "frozen")]]
end
queue = Queue.new
cells.each { |c| queue << c }
results = {}
mutex = Mutex.new
# run_lstm aborts on a non-zero runner exit. Inside a worker thread Ruby
# would SWALLOW that — the thread dies, the gate carries on, and the cell
# comes back missing instead of loud. This restores the serial behaviour:
# any worker failure takes the process down, the way it did before.
Thread.abort_on_exception = true
workers = Array.new([LSTM_GATE_JOBS, cells.size].min) do
  Thread.new do
    loop do
      arm, s, env = begin
                      queue.pop(true)
                    rescue ThreadError
                      break
                    end
      acc = val_acc(run_lstm(env, nil))
      mutex.synchronize { (results[s] ||= {})[arm] = acc }
    end
  end
end
workers.each(&:join)
# Belt and braces: even with abort_on_exception, a cell that never landed
# must not be read as a nil accuracy further down. The bar is 12 cells or
# it is not the bar.
missing = cells.reject { |arm, s, _| results.dig(s, arm) }
unless missing.empty?
  abort "lstm_gate: #{missing.size} of #{cells.size} success-bar cells did not " \
        "complete (#{missing.map { |a, s, _| "seed #{s}/#{a}" }.join(', ')}) — " \
        "the bar is all 12 cells or it is not the bar"
end
rows = {}
SEEDS.each do |s|
  r = results[s]
  rows[s] = r
  puts format("    seed %s  chain=%.4f dfa(layer)=%.4f dfa(step)=%.4f frozen=%.4f",
              s, r["chain"], r["dfa"], r["step"], r["frozen"])
end

SEEDS.each do |s|
  r = rows[s]
  # PRECONDITION: BP is healthy AT THIS SEED. Without it "DFA matches BP"
  # is a comparison against an arm that did not train — which is exactly
  # what this lane produced three times before the cell was swept
  # properly (a bias-free LSTM, the inherited 0.003 default, and the
  # seed-0-only 0.03/2000 cell).
  if r["chain"] < r["frozen"] + BP_PRECOND
    failures << "SUCCESS BAR (precondition, seed #{s}): BP #{r['chain']} does not beat frozen #{r['frozen']} by #{BP_PRECOND}. This cell was chosen as BP's OWN BEST over a (lr x warmup x seed) sweep; if BP no longer trains here, re-sweep BEFORE reading any DFA number — a DFA arm 'beating' an untrained BP arm is this lane's characteristic wrong result"
  end
  if r["step"] < STEP_FLOOR
    failures << "SUCCESS BAR (seed #{s}): dfa(step) #{r['step']} is below #{STEP_FLOOR}. The per-step cut solving this task on a GATED recurrence — where the identical cut collapses to chance on toy#155's selective scan — is this lane's headline"
  end
  if r["step"] < r["chain"] - BP_GAP
    failures << "SUCCESS BAR (BP gap, seed #{s}): dfa(step) #{r['step']} is more than #{BP_GAP} below chain #{r['chain']}"
  end
  if r["step"] < r["frozen"] + FROZEN_EDGE
    failures << "SUCCESS BAR (frozen control, seed #{s}): dfa(step) #{r['step']} does not beat frozen #{r['frozen']} by #{FROZEN_EDGE}"
  end
end
puts failures.length == n0 ?
  "  ok: SUCCESS BAR — at BP's own best cell, on ALL THREE SEEDS, the per-step cut matches BP and crushes the frozen control" :
  "  FAIL: success bar"

# ---- 6b. and BP is the FRAGILE arm, which is half the result ----
# The cell above is BP's best over a sweep; BP needs it. The per-step cut
# does not: measured, it solves the task at every one of 4 LRs x 3 seeds
# (12/12) while BP solves 7/12 and its working LR depends on the SEED.
# One cheap probe pins the direction — at a rate BP is known to fail at
# for this seed, the cut must still solve it. If this ever flips, the
# stability half of the lane's claim needs re-measuring, not re-baselining
# (docs/roadmap/dfa-arch-program-2026-08-10.md).
n0 = failures.length
off_lr    = "0.03"
# WARMUP=0 EXPLICITLY: the grid above was measured without the ramp, and
# the runner's default warmup is 200 (the fair cell's). Inheriting it
# here would compare against a cell that was never measured.
off_env   = { "STEPS" => "4000", "SEED" => "1", "LSTM_LR" => off_lr, "LSTM_WARMUP" => "0" }
off_chain = val_acc(run_lstm(off_env.merge("LSTM_POLICY" => "chain"), nil))
off_step  = val_acc(run_lstm(off_env.merge("LSTM_POLICY" => "dfa",
                                           "LSTM_DFA_CUT" => "step"), nil))
puts format("    off-cell (lr %s, seed 1): chain=%.4f dfa(step)=%.4f", off_lr, off_chain, off_step)
if off_step < STEP_FLOOR
  failures << "STABILITY: at lr #{off_lr} the per-step cut #{off_step} fell below #{STEP_FLOOR} too. The measured claim is that the cut is INDIFFERENT to the rate (no gradient crosses time, so none can explode) while BPTT is not — re-measure the grid before restating it"
end
if off_chain > off_step - BP_GAP
  failures << "STABILITY: at lr #{off_lr}, seed 1, BP scored #{off_chain} against the cut's #{off_step} — this cell is documented as one BP FAILS at (measured .250 at 2000 and 4000 steps). BP recovering here means the arm's fragility has changed; re-read the roadmap grid before re-baselining"
end
puts failures.length == n0 ?
  "  ok: STABILITY — off BP's best cell, BPTT falls to chance and the per-step cut does not" :
  "  FAIL: stability"

# ---- 7. THE CROSS-ARCHITECTURE CONTRAST ----
# The reason this lane exists. The IDENTICAL per-step cut, on the
# IDENTICAL task generator, at toy#155's own documented cell: on the
# selective scan it collapses to chance, here it solves the task. Run
# both, in one gate, or the claim is two separate anecdotes.
n0 = failures.length
ssm_cell   = { "STEPS" => "600", "SEED" => "0" }
ssm_step   = val_acc(run_ssm(ssm_cell.merge("SSM_POLICY" => "dfa,dfa", "SSM_DFA_CUT" => "step")))
ssm_frozen = val_acc(run_ssm(ssm_cell.merge("SSM_POLICY" => "frozen,frozen")))
lstm_step  = rows[SEEDS.first]["step"]
puts format("    per-step cut: selective scan=%.4f (its frozen=%.4f)  vs  gated LSTM=%.4f",
            ssm_step, ssm_frozen, lstm_step)
if ssm_step > ssm_frozen + 0.25
  failures << "CROSS-ARCH: on toy#155's selective scan the per-step cut scored #{ssm_step} against its frozen control #{ssm_frozen} — that lane's measured finding is a COLLAPSE to chance (.250/.250/.238 at seeds 0/1/2, best over an LR sweep .355). If it now learns, the contrast this lane is built on has moved; re-read both lanes' roadmap sections before re-baselining"
end
if lstm_step < ssm_step + 0.5
  failures << "CROSS-ARCH: the per-step cut scored #{lstm_step} on the LSTM against #{ssm_step} on the selective scan — the whole point of this lane is that the same cut, same task and same axis give OPPOSITE outcomes on the two architectures"
end
puts failures.length == n0 ?
  "  ok: CROSS-ARCHITECTURE — the identical per-step cut collapses on the selective scan and holds on the gated LSTM (toy#155 vs toy#157)" :
  "  FAIL: cross-architecture contrast"

# ---- 8. the memory measurement, in the units the ticket asks for ----
# toy#157's success target is "matches BP at k-times-less activation
# memory for sequence length L". toy#155 could only report node COUNT;
# this lane sums ggml_nbytes over every node of the realized graph. The
# assertion below is the FINDING, not a wish: the per-step cut — the arm
# whose whole value proposition is not storing the BPTT graph — builds a
# BIGGER graph than BPTT, and the gap GROWS with L. If that ever
# inverts, this harness has changed shape and the claim needs re-deriving
# rather than re-baselining (see tao#21).
n0 = failures.length
mem = {}
%w[16 64].each do |t|
  mem[t] = {
    "chain" => graph_bytes(run_lstm({ "STEPS" => "1", "LSTM_SEQ" => t }, nil)),
    "layer" => graph_bytes(run_lstm({ "STEPS" => "1", "LSTM_SEQ" => t,
                                      "LSTM_POLICY" => "dfa" }, nil)),
    "step"  => graph_bytes(run_lstm({ "STEPS" => "1", "LSTM_SEQ" => t,
                                      "LSTM_POLICY" => "dfa",
                                      "LSTM_DFA_CUT" => "step" }, nil))
  }
  puts format("    T=%-4s bytes chain=%d dfa(layer)=%d dfa(step)=%d", t,
              mem[t]["chain"], mem[t]["layer"], mem[t]["step"])
end
%w[16 64].each do |t|
  if mem[t]["step"] <= mem[t]["chain"]
    failures << "MEMORY: at T=#{t} the per-step cut (#{mem[t]['step']} bytes) is NOT larger than BPTT (#{mem[t]['chain']} bytes). Measured, it is ~17% bigger: in a graph autodiff every forward tensor is materialised whatever the credit rule, and the detach dups plus per-step surrogate terms are net ADDITIONS. An inversion here means the harness changed shape — re-derive the claim, do not re-baseline (tao#21)"
  end
end
grow_chain = mem["64"]["chain"].to_f / mem["16"]["chain"].to_f
grow_step  = mem["64"]["step"].to_f  / mem["16"]["step"].to_f
if grow_step < grow_chain
  failures << "MEMORY: the per-step cut's bytes grow SLOWER in L than BPTT's (#{grow_step.round(3)}x vs #{grow_chain.round(3)}x from T=16 to T=64) — the ticket states the memory claim as a function of L, and the measured answer is that it does not improve with L either"
end
puts failures.length == n0 ?
  "  ok: MEMORY measured in BYTES — the per-step cut costs MORE than BPTT at both lengths, and no better with L (tao#21)" :
  "  FAIL: memory"

# ---- 8b. the ANALYTIC streaming instrument (toy#159) ----
# The measured leg above answers "what does toy BUILD". This one answers
# the question the ticket actually asks — "what would a streaming
# implementation HOLD" — and the two disagreeing is the finding, not a
# discrepancy to be reconciled away. Asserted here:
#   * cut is CONSTANT in T (that is the whole claim), and
#   * bptt is LINEAR in T, and
#   * the MEASURED graph still contradicts both, so nobody quotes the
#     analytic win as if toy had implemented it.
n0 = failures.length
st = {}
%w[16 64 128].each do |t|
  st[t] = stream_bytes(run_lstm({ "STEPS" => "1", "LSTM_SEQ" => t }, nil))
end
puts format("    stream T=16/64/128: cut=%d/%d/%d  bptt=%d/%d/%d",
            st["16"]["cut"], st["64"]["cut"], st["128"]["cut"],
            st["16"]["bptt"], st["64"]["bptt"], st["128"]["bptt"])
unless st["16"]["cut"] == st["64"]["cut"] && st["64"]["cut"] == st["128"]["cut"]
  failures << "STREAM: the per-step cut's analytic bytes are NOT constant in T (#{st['16']['cut']}/#{st['64']['cut']}/#{st['128']['cut']} at T=16/64/128). O(1) in T is the entire claim — if this figure moved with T, it acquired a T term and is no longer measuring a streaming implementation"
end
# 4x the length must cost ~4x under BPTT (the head term is the only
# non-scaling part, so allow a small slack rather than demanding exact).
grow = st["64"]["bptt"].to_f / st["16"]["bptt"].to_f
if grow < 3.9 || grow > 4.1
  failures << "STREAM: analytic bptt grew #{grow.round(3)}x from T=16 to T=64, expected ~4x — BPTT holding every step's activations IS the O(T) baseline the cut is stated against"
end
if st["64"]["cut"] >= st["64"]["sqrt_t"]
  failures << "STREAM: at T=64 the cut (#{st['64']['cut']}) is not below sqrt-T checkpointed BPTT (#{st['64']['sqrt_t']}) — the instrument quotes BP's own best counter-move so the win is not measured against BP's worst case only, and the claim needs to survive that comparison"
end
# The two instruments must keep disagreeing, in the direction measured.
if mem["64"]["step"] <= st["64"]["cut"]
  failures << "STREAM: the MEASURED graph bytes for the step cut (#{mem['64']['step']}) are no longer above the ANALYTIC streaming figure (#{st['64']['cut']}). The gap between what toy builds and what a streaming implementation would hold is the whole point of carrying both numbers — if it closed, toy started streaming, which would be a much bigger change than this instrument"
end
puts failures.length == n0 ?
  "  ok: STREAM (toy#159) — the analytic cut is CONSTANT in T while bptt is linear, it beats sqrt-T checkpointing too, and it still disagrees with what toy actually builds" :
  "  FAIL: streaming instrument"

# ---- 8c. --clip-grad: the FAIR BPTT control (toy#162) ----
# Three properties, and the third is the finding.
n0 = failures.length
huge = run_lstm({ "STEPS" => "20", "LSTM_CLIP_GRAD" => "1e9" }, nil)
tiny = run_lstm({ "STEPS" => "20", "LSTM_CLIP_GRAD" => "0.01" }, nil)
clip_det1 = run_lstm({ "STEPS" => "20", "LSTM_CLIP_GRAD" => "0.05" }, nil)
clip_det2 = run_lstm({ "STEPS" => "20", "LSTM_CLIP_GRAD" => "0.05" }, nil)
failures << "clip: a 1e9 norm changed the curve — a clip that can never bind MUST be a no-op, or --clip-grad is not clipping, it is perturbing" unless curve(huge) == curve(ch20)
failures << "clip: a 0.01 norm did NOT change the curve — clipping that never binds cannot be a control for anything" if curve(tiny) == curve(ch20)
failures << "clip: two identical clipped runs differ" unless clip_det1 == clip_det2
# THE FINDING. At this cell BP fails clipped exactly as it fails
# unclipped, while the per-step cut solves it. If clipping ever DID
# rescue BP here, toy#157's stability claim would be a harness artifact
# and both the roadmap and Tao's F21 conclusion would need correcting —
# so this leg is the tripwire for that, not decoration.
clip_cell = { "STEPS" => "4000", "SEED" => "0", "LSTM_LR" => "0.01",
              "LSTM_WARMUP" => "0", "LSTM_CLIP_GRAD" => "1.0" }
clip_bp   = val_acc(run_lstm(clip_cell.merge("LSTM_POLICY" => "chain"), nil))
clip_step = val_acc(run_lstm(clip_cell.merge("LSTM_POLICY" => "dfa",
                                             "LSTM_DFA_CUT" => "step"), nil))
puts format("    clipped (norm 1.0, lr 0.01, seed 0): chain=%.4f dfa(step)=%.4f", clip_bp, clip_step)
if clip_bp > 0.5
  failures << "FAIR CONTROL: with gradient clipping at norm 1.0, BP scored #{clip_bp} at a cell it fails at unclipped (.250). If the standard BPTT stabiliser now rescues BP, then toy#157's 'BPTT is the fragile arm' is substantially OUR harness and not the method — correct the roadmap and tao#20 BEFORE re-baselining this threshold. Measured over the full grid: clipping moves BP from 7/12 to 8/12 solved cells and re-rolls WHICH ones, without making it reliable"
end
if clip_step < STEP_FLOOR
  failures << "FAIR CONTROL: the per-step cut scored #{clip_step} with clipping on — the cut is supposed to be indifferent to it (12/12 measured), and if clipping is what carries the DFA arm then the lane's comparison is not what it says"
end
puts failures.length == n0 ?
  "  ok: --clip-grad is a no-op when it cannot bind, binds when it can, is deterministic, and does NOT rescue BP (toy#162 — the stability claim survives its fair control)" :
  "  FAIL: clip-grad"

# ---- 9. the degenerate task, measured ----
# `mean` spreads the class signal over every step, so no carry across
# time is needed and a FROZEN recurrence plus a trained head already
# integrates it. This lane's `blobs` (toy#152) — asserted, not assumed,
# because it is what justifies `cue` as the default.
n0 = failures.length
mean_fz = val_acc(run_lstm(CELL.merge("LSTM_TASK" => "mean", "LSTM_POLICY" => "frozen"), nil))
cue_fz  = rows["0"]["frozen"]
puts format("    frozen: --task mean=%.4f  --task cue=%.4f", mean_fz, cue_fz)
failures << "degenerate task: --task mean frozen #{mean_fz} is not far above the cue task's frozen #{cue_fz} — `mean` is documented as solvable with no carry across time at all, and the default `cue` task is justified against that measurement" unless mean_fz > cue_fz + 0.25
puts failures.length == n0 ? "  ok: --task mean is measurably degenerate (a FROZEN recurrence solves it), which is why `cue` is the default" : "  FAIL: degenerate task"

# ---- 10. fail-loud ----
n0 = failures.length
[
  [{ "LSTM_POLICY" => "chain,chain" },            "policy longer than LSTM_LAYERS"],
  [{ "LSTM_POLICY" => "bogus" },                  "unknown policy token"],
  [{ "LSTM_POLICY" => "chain,dfa", "LSTM_LAYERS" => "2" }, "a chain layer BELOW a dfa layer (silently frozen)"],
  [{ "LSTM_DFA_CUT" => "token" },                 "unknown LSTM_DFA_CUT"],
  [{ "LSTM_TASK" => "copy" },                     "unknown LSTM_TASK"],
  [{ "LSTM_CLASSES" => "1" },                     "LSTM_CLASSES=1"],
  [{ "LSTM_LAYERS" => "0" },                      "LSTM_LAYERS=0"],
  [{ "LSTM_HIDDEN" => "0" },                      "LSTM_HIDDEN=0"],
  [{ "LSTM_SEQ" => "1" },                         "LSTM_SEQ=1"],
  [{ "LSTM_FEATURES" => "1" },                    "LSTM_FEATURES=1"],
  [{ "LSTM_VAL_BATCHES" => "0" },                 "LSTM_VAL_BATCHES=0"],
  [{ "LSTM_CUE_SPAN" => "64" },                   "cue span == sequence length (no delay left to carry)"],
  [{ "LSTM_CLIP_GRAD" => "0" },                   "LSTM_CLIP_GRAD=0 (omit it to disable, do not set a fake norm)"],
  [{ "LSTM_CLIP_GRAD" => "-1" },                  "negative LSTM_CLIP_GRAD"],
].each do |env, label|
  out, st = Open3.capture2e({ "STEPS" => "2" }.merge(env), RUNNER, chdir: ROOT)
  failures << "fail-loud: #{label} exited 0 (silently did nothing):\n#{out.lines.last(2).join}" if st.success?
end
puts failures.length == n0 ? "  ok: 14 degenerate configs all fail loud instead of quietly doing nothing" : "  FAIL: fail-loud"

# ---- 11. CLI ----
n0 = failures.length
Dir.mktmpdir("lstm_gate_cli") do |dir|
  out, st = Open3.capture2e({ "SPINEL_SKIP_PIN_CHECK" => nil }, TOY, "train", "lstm",
                            "--steps", "5", "--seed", "0", "--out", dir, chdir: ROOT)
  if st.success?
    failures << "cli: `toy train lstm` curve != the runner's default curve — the CLI and the runner must agree on every default, LR INCLUDED (0.02 here, NOT the ssm lane's 0.003), and --warmup must reach the runner as 0 when unset" unless curve(out) == base_curve
    failures << "cli: `toy train lstm` did not echo the val line" unless out.lines.any? { |l| l.start_with?("val: ") }
    failures << "cli: `toy train lstm` did not echo the graph line (the memory number rides stdout so a --seq sweep can read it without opening a bundle)" unless out.lines.any? { |l| l.start_with?("graph: ") }
    failures << "cli: `toy train lstm` did not echo the stream line (toy#159 — the ANALYTIC figure the memory claim is actually stated in; the CLI dropping it would leave callers with only the measured number, which is the one that says the opposite)" unless out.lines.any? { |l| l.start_with?("stream: ") }
  else
    failures << "cli: `toy train lstm` exited #{st.exitstatus}:\n#{out.lines.last(5).join}"
  end
  [
    [%w[lstm --align-events],      "lstm accepted --align-events (its DFA update lands in the SAME accumulator a BP run uses, so a cosine would mean nothing)"],
    [%w[lstm --policy-scope ffn],  "lstm accepted --policy-scope"],
    [%w[lstm --selection lti],     "lstm accepted --selection (an LSTM has no selective-scan branch to select)"],
    [%w[lstm --d-inner 32],        "lstm accepted --d-inner"],
    [%w[lstm --conv-k 2],          "lstm accepted --conv-k"],
    [%w[lstm --dt-init -4.0],      "lstm accepted --dt-init"],
    [%w[lstm --task blobs],        "lstm accepted --task blobs"],
    [%w[mlp --dfa-cut step],       "mlp accepted --dfa-cut"],
    [%w[ssm --clip-grad 1.0],      "ssm accepted --clip-grad (toy#162 is the lstm lane's fair control; widening it is a follow-on)"],
    [%w[gnn --seq 32],             "gnn accepted --seq"],
  ].each do |argv, label|
    sout, sst = Open3.capture2e({}, TOY, "train", *argv, chdir: ROOT)
    failures << "cli: #{label} (exit #{sst.exitstatus}: #{sout.lines.last(1).join.strip})" unless sst.exitstatus == 2
  end
  cout, cst = Open3.capture2e({}, TOY, "train", "lstm", "--device", "cuda", chdir: ROOT)
  failures << "cli: lstm accepted --device cuda (CPU-only — tao#18/#21: a device twin changes throughput, not the materialised bytes this lane measures)" unless cst.exitstatus == 2
end
puts failures.length == n0 ? "  ok: CLI reproduces the curve; lane-foreign flags, --align-events and --device cuda are rejected" : "  FAIL: CLI"

if failures.empty?
  puts "GATE PASS [lstm]: gated recurrence + per-layer policy — byte fixture, forward-identity under BOTH cuts, the B seed, the tap counts, the MANDATORY success bar over three seeds, the CROSS-ARCHITECTURE contrast (the per-step cut holds on an LSTM and collapses on the selective scan), and the activation-memory claim measured in BYTES and NOT met (toy#157/tao#21)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [lstm]: #{f}" }
  exit 1
end
