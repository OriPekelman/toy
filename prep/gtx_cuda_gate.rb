#!/usr/bin/env ruby
# prep/gtx_cuda_gate.rb — tao#24 gate for the gtx byte-LM CUDA TWIN
# (libexec/toy-train-gtx-cuda, `toy train gtx --task bytelm --device cuda`).
#
# ── WHAT THIS GATE IS FOR, AND WHY IT COMES BEFORE THE SPEEDUP ──
#
# tao#24 reopened the CPU-only scope for ONE task because toy#170/P3 grew
# it by orders of magnitude (vocab up to 4096, ctx 128, 4000 steps, ~3.2
# TFLOP/cell measured at ~32 GFLOP/s with the GB10 at 0%). A twin that is
# 5x faster and quietly computes something else is WORSE than no twin: it
# would put a wrong number into the E-series ladder with a plausible face
# on it. So cross-backend numeric agreement is the deliverable and the
# wall-clock win is the reward for passing.
#
# Legs:
#   1. THE TWIN RUNS, ON CUDA, and emits both provenance lines.
#   2. STRUCTURAL BIT-EQUALITY — the wiring line and the graph
#      nodes=/bytes= line must be BYTE-IDENTICAL to the CPU lane.
#   3. ARITHMETIC AGREEMENT at lr=0 (the hard tolerance).
#   4. TRAINED AGREEMENT at each arm's OWN lr (the loose tolerance).
#   5. DETERMINISM on CUDA.
#   6. THE SCOPE IS REALLY bytelm-ONLY — the binary refuses, the CLI
#      refuses, and the sibling lanes still refuse.
#   7. THE CPU LANE IS BYTE-IDENTICAL TO BEFORE THE TWIN.
#
# ── THE TOLERANCE, AND WHY IT IS TWO NUMBERS RATHER THAN ONE ──
#
# Train losses are byte-exact PER PLATFORM and toleranced ACROSS them,
# which is the standing rule. But "toleranced" needs a number, and a
# single number is not honest here, because two different things are
# being compared:
#
#   ARITHMETIC. Same graph, same weights, different reduction orders in
#   the CUDA kernels. MEASURED with GTX_LR=0 — which freezes every weight,
#   since the AdamW update is scaled by lr, so each step recomputes the
#   same forward and NOTHING can accumulate. Across small/big/8-block
#   shapes, chain/dfa-layer/dfa-step/frozen policies and a 2504-symbol
#   pack, the worst relative deviation measured was 1.19e-4 per step and
#   3.03e-5 on bpb. The gate asserts 1e-3 — roughly 8x headroom over the
#   worst measurement, tight enough that a real wiring difference (which
#   shows up as percent, not as 1e-4) cannot hide under it.
#
#   DYNAMICS. Once lr > 0 the optimizer is a nonlinear map and a 1-ULP
#   difference is an initial condition. It amplifies. MEASURED on the
#   dfa/layer arm at head 4096, reading the final-step deviation against
#   the horizon:
#
#     steps        1       2       4       8      16      32      64     128
#     lr 0.003  2.5e-5  1.6e-5  4.9e-5  3.0e-4  9.2e-4  2.4e-3  1.7e-1  2.0e-2
#     lr 0.0003 2.5e-5  4.3e-5  4.3e-6  1.2e-5  4.7e-5  3.9e-5  9.4e-5  2.8e-4
#
#   Both columns start at the SAME 2.5e-5 arithmetic floor. At the DFA
#   arm's own cell (0.0003) it stays there for 128 steps. At BP's cell
#   (0.003, ~10x this arm's optimum) it climbs four orders and goes
#   non-monotone — the signature of a trajectory above its stability edge,
#   not of a miswired graph. This is the program's per-arm-LR rule showing
#   up in a new place: an arm measured at another arm's LR does not even
#   REPRODUCE ACROSS DEVICES, let alone compare.
#
#   So leg 4 runs each arm at ITS OWN lr and asserts 1e-2 relative on the
#   final loss and on bpb. What leg 4 actually measures, 32 steps at head
#   4096:
#
#     chain     lr 0.003    dloss 7.0e-4   dbpb 2.9e-3
#     dfa/layer lr 0.0003   dloss 3.9e-5   dbpb 9.3e-5
#     dfa/step  lr 0.0003   dloss 4.1e-5   dbpb 2.1e-5
#
#   The headroom is 3.4x, not 35x, and it is the CHAIN arm that sets it —
#   stated plainly because a tolerance quoted against the easiest cell is
#   not a tolerance. Chain is loosest because it is the arm that actually
#   moves in 32 steps: the DFA arms at 1/10th the LR have barely left
#   their init, so their trajectories have had nothing to amplify yet.
#   3.4x is enough to separate float noise from a wiring fault (which
#   shows up in percent) and not so wide that a real regression hides;
#   if a future change pushes chain past it, leg 3 is the diagnostic —
#   lr=0 still agreeing means the arithmetic is fine and the trajectory
#   is what moved.
#
# ── WHAT THIS GATE DOES NOT CLAIM ──
#
# It does NOT claim a CUDA cell is interchangeable with a CPU cell at the
# 4000-step operating point. Measured there (head 4096, 4000 steps): BP
# bpb 3.4117 vs 3.4165 (rel 1.4e-3), DFA bpb 3.9821 vs 3.9483 (rel
# 8.5e-3, i.e. 0.034 BITS). The E-series reads absolute bits recovered at
# signals of 0.065-0.230 bits, so 0.034 bits of device is NOT noise
# against that. tao#24 pins the consequence and this gate does not soften
# it: an E-series sweep runs ENTIRELY ON ONE DEVICE, and any
# cross-reference to a CPU P3-P6 number is re-run on that device or not
# made.
#
# ── THE WALL-CLOCK, MEASURED (the reason the ticket exists) ──
#
# P6 anchor cell, head 4096 / ctx 128 / d_model 128 / 4 blocks / 4000
# steps / 8 val batches, on this box while a gate battery was running:
#
#   bp-body   (chain, lr 0.003)   CPU  82.4 s   CUDA 17.4 s   4.7x
#   dfa-layer (dfa,   lr 0.0003)  CPU 105.3 s   CUDA 18.2 s   5.8x
#
# Not the 10x a bigger model would give — at d_model 128 the GPU is
# launch-bound, not arithmetic-bound. It is still ~6 h of ladder becoming
# ~1 h, which is what the E-series needed.

ROOT    = File.expand_path("..", __dir__)
CPU_BIN = File.join(ROOT, "libexec", "toy-train-gtx")
GPU_BIN = File.join(ROOT, "libexec", "toy-train-gtx-cuda")
TOY     = File.join(ROOT, "bin", "toy")
FIXTURE = File.join(ROOT, "prep", "fixtures", "train_gtx_bytelm_baseline.txt")
PACK    = "data/ae_shak_a65"

require "open3"
require "json"
require "tmpdir"

# The two tolerances, named once. See the header for how each was measured.
TOL_ARITH = 1.0e-3   # lr=0, forward-only; worst measured 1.19e-4
TOL_TRAIN = 1.0e-2   # own-lr, short horizon; worst measured 2.8e-4

failures = []

# The lane is REAL-TEXT ONLY (no synthetic fallback by design), so without
# the pack this gate can assert nothing. Skip loudly rather than pass
# vacuously — a green light that measured nothing is the failure mode this
# whole tree is built against.
unless File.file?(File.join(ROOT, PACK + ".tok.i32"))
  puts "GATE SKIP [gtx-cuda]: #{PACK}.tok.i32 is absent — this lane has NO synthetic corpus by design, so there is nothing to measure. Run: ruby prep/fetch_text.rb --all && ruby prep/remap_alphabet.rb"
  exit 0
end
[CPU_BIN, GPU_BIN].each do |b|
  unless File.executable?(b)
    warn "GATE FAIL [gtx-cuda]: #{b} missing — build it (make #{b.sub(ROOT + "/", "")})"
    exit 1
  end
end

SMALL = { "GTX_TASK" => "bytelm", "GTX_TEXT" => PACK,
          "GTX_VOCAB" => "65", "GTX_CONTEXT" => "32", "GTX_D_MODEL" => "32",
          "GTX_HEADS" => "2", "GTX_D_FF" => "64", "GTX_BLOCKS" => "2",
          "GTX_VAL_BATCHES" => "2" }
# The shape the science actually runs (prep/research/p6_ladder.sh), at a step count
# a gate can afford. Shape is what decides which CUDA kernels are hit, so
# it must not be shrunk to the small cell.
BIG   = { "GTX_TASK" => "bytelm", "GTX_TEXT" => PACK,
          "GTX_VOCAB" => "4096", "GTX_CONTEXT" => "128", "GTX_D_MODEL" => "128",
          "GTX_HEADS" => "4", "GTX_D_FF" => "256", "GTX_BLOCKS" => "4",
          "GTX_VAL_BATCHES" => "8" }

def run(bin, env)
  out, st = Open3.capture2e(env, bin, chdir: ROOT)
  [out, st]
end
def steps_of(o) = o.lines.select { |l| l.start_with?("step ") }.map(&:chomp)
def losses(o)   = steps_of(o).map { |l| l[/loss=(\S+)/, 1].to_f }
def bpb(o)      = o[/^bytelm: bpb=(\S+)/, 1]&.to_f
def wire(o)     = o.lines.find { |l| l.start_with?("gtx: bytelm ") }&.chomp
def graphline(o) = o.lines.find { |l| l.start_with?("graph: ") }&.chomp
def rel(a, b)   = a.nil? || b.nil? || a == 0.0 ? Float::INFINITY : ((a - b) / a).abs

puts "gtx-cuda gate (tao#24): the byte-LM twin"

# ---- 1. THE TWIN RUNS, ON CUDA, AND SAYS SO ----
#
# "It ran" is not the assertion. ggml's scheduler will happily fall back
# to the CPU backend for an op with no CUDA kernel, and a twin that fell
# back for EVERYTHING would still print a correct-looking curve. The
# run_start event carries backend.kind from tnn_backend_name(sess), which
# is the session's actual backend, so that is what gets asserted.
n0 = failures.length
Dir.mktmpdir("gtx_cuda_gate") do |dir|
  env = SMALL.merge("STEPS" => "5", "SEED" => "0",
                    "TAO_RUN_DIR" => dir, "TOY_RUN_ID" => "gate-cuda")
  out, st = run(GPU_BIN, env)
  if !st.success?
    failures << "the CUDA twin exited #{st.exitstatus}:\n#{out.lines.last(6).join}"
  else
    failures << "the CUDA twin emitted no RESULT line (bytelm: bpb=...) — every number this lane produces would be invisible" unless
      out.lines.any? { |l| l.start_with?("bytelm: bpb=") }
    failures << "the CUDA twin emitted no WIRING line (gtx: bytelm ...) — vocab= and b_dim= are how a cell proves which head width it used" unless
      out.lines.any? { |l| l.start_with?("gtx: bytelm ") }
    ev = File.join(dir, "events.jsonl")
    if File.file?(ev)
      rs = JSON.parse(ev.then { |p| File.readlines(p).first.to_s })
      kind = rs.dig("backend", "kind")
      failures << "the CUDA twin ran on backend #{kind.inspect}, not cuda — ggml falls back silently, so a twin that never touched the GPU would still print a plausible curve" unless kind == "cuda"
    else
      failures << "the CUDA twin wrote no events.jsonl under TAO_RUN_DIR — the backend provenance is unverifiable"
    end
  end
end
puts failures.length == n0 ? "  ok: the twin runs, emits its RESULT and WIRING lines, and its run_start records backend=cuda" : "  FAIL: twin runs"

# ---- 2. STRUCTURAL BIT-EQUALITY ----
#
# STRONGER THAN ANY TOLERANCE, and cheap. The wiring line carries every
# policy/tap/head decision and the graph line carries the realized node
# count and byte size. Those are INTEGERS and strings: they do not depend
# on float reduction order, so they must match EXACTLY. If the mirror
# generator ever drops a tap or reorders a build, this fires before the
# numbers get a chance to look plausible.
n0 = failures.length
CELLS = [
  ["small chain",     SMALL.merge("STEPS" => "4", "SEED" => "0")],
  ["small dfa/layer", SMALL.merge("STEPS" => "4", "SEED" => "0", "GTX_POLICY" => "dfa,dfa")],
  ["small dfa/step",  SMALL.merge("STEPS" => "4", "SEED" => "0", "GTX_POLICY" => "dfa,dfa", "GTX_DFA_CUT" => "step")],
  ["small frozen",    SMALL.merge("STEPS" => "4", "SEED" => "0", "GTX_POLICY" => "frozen,frozen")],
  ["big   chain",     BIG.merge("STEPS" => "4", "SEED" => "0", "GTX_POLICY" => "chain,chain,chain,chain")],
  ["big   dfa/layer", BIG.merge("STEPS" => "4", "SEED" => "0", "GTX_POLICY" => "dfa,dfa,dfa,dfa")],
]
PAIRS = CELLS.map do |label, env|
  a, sa = run(CPU_BIN, env)
  b, sb = run(GPU_BIN, env)
  failures << "structure: #{label} — CPU exited #{sa.exitstatus}:\n#{a.lines.last(4).join}" unless sa.success?
  failures << "structure: #{label} — CUDA exited #{sb.exitstatus}:\n#{b.lines.last(4).join}" unless sb.success?
  if sa.success? && sb.success?
    failures << "structure: #{label} — the WIRING lines differ across backends, so the twin did not build the same graph:\n  cpu:  #{wire(a)}\n  cuda: #{wire(b)}" unless
      wire(a) && wire(a) == wire(b)
    failures << "structure: #{label} — graph nodes/bytes differ across backends (#{graphline(a)} vs #{graphline(b)}). These are integers; a difference is a different graph, not float noise." unless
      graphline(a) && graphline(a) == graphline(b)
    failures << "structure: #{label} — the two backends emitted a different NUMBER of step lines (#{steps_of(a).length} vs #{steps_of(b).length})" unless
      steps_of(a).length == steps_of(b).length
  end
  [label, env, a, b]
end
puts failures.length == n0 ? "  ok: 6 cells (4 policies x 2 shapes) build a BYTE-IDENTICAL graph on both backends — same wiring line, same nodes/bytes" : "  FAIL: structural bit-equality"

# ---- 3. ARITHMETIC AGREEMENT (lr = 0) ----
#
# The hard number. GTX_LR=0 freezes the weights, so this is a pure
# forward comparison repeated over fresh windows — no dynamics, nothing
# to amplify, and therefore the one place a tight tolerance is honest.
n0 = failures.length
worst_arith = 0.0
CELLS.each do |label, env|
  e = env.merge("GTX_LR" => "0", "STEPS" => "8")
  a, sa = run(CPU_BIN, e)
  b, sb = run(GPU_BIN, e)
  next failures << "arithmetic: #{label} — a run failed (cpu #{sa.exitstatus}, cuda #{sb.exitstatus})" unless sa.success? && sb.success?
  la, lb = losses(a), losses(b)
  if la.empty? || la.length != lb.length
    failures << "arithmetic: #{label} — no comparable step curve (#{la.length} vs #{lb.length})"
    next
  end
  d = la.zip(lb).map { |x, y| rel(x, y) }.max
  db = rel(bpb(a), bpb(b))
  worst_arith = [worst_arith, d, db].max
  failures << "arithmetic: #{label} — per-step loss deviates #{"%.3e" % d} at lr=0, past the #{TOL_ARITH} tolerance. At lr=0 no weight moves, so this CANNOT be trajectory divergence: it is the two backends computing a different forward." if d > TOL_ARITH
  failures << "arithmetic: #{label} — bpb deviates #{"%.3e" % db} at lr=0, past the #{TOL_ARITH} tolerance (cpu #{bpb(a)}, cuda #{bpb(b)})" if db > TOL_ARITH
end
puts failures.length == n0 ? "  ok: forward-only (lr=0) agreement across all 6 cells, worst #{"%.3e" % worst_arith} vs a #{TOL_ARITH} tolerance" : "  FAIL: arithmetic agreement"

# ---- 4. TRAINED AGREEMENT, EACH ARM AT ITS OWN LR ----
#
# Per-arm LR is not politeness here, it is the difference between a
# reproducible comparison and a meaningless one — see the header table.
# BP's arm sits at 0.003, the DFA arms at 0.0003 (measured, P5/P6).
n0 = failures.length
worst_train = 0.0
[["chain",     BIG.merge("GTX_POLICY" => "chain,chain,chain,chain"), "0.003"],
 ["dfa/layer", BIG.merge("GTX_POLICY" => "dfa,dfa,dfa,dfa"),         "0.0003"],
 ["dfa/step",  BIG.merge("GTX_POLICY" => "dfa,dfa,dfa,dfa", "GTX_DFA_CUT" => "step"), "0.0003"]].each do |label, env, lr|
  e = env.merge("STEPS" => "32", "SEED" => "0", "GTX_LR" => lr)
  a, sa = run(CPU_BIN, e)
  b, sb = run(GPU_BIN, e)
  next failures << "trained: #{label} — a run failed (cpu #{sa.exitstatus}, cuda #{sb.exitstatus})" unless sa.success? && sb.success?
  d  = rel(losses(a).last, losses(b).last)
  db = rel(bpb(a), bpb(b))
  worst_train = [worst_train, d, db].max
  failures << "trained: #{label} at its own lr #{lr} — final loss deviates #{"%.3e" % d}, past #{TOL_TRAIN}. Check leg 3 first: if the lr=0 arithmetic still agrees, this is the arm sitting above its stability edge rather than a wiring fault, and the LR is what to fix." if d > TOL_TRAIN
  failures << "trained: #{label} at its own lr #{lr} — bpb deviates #{"%.3e" % db}, past #{TOL_TRAIN} (cpu #{bpb(a)}, cuda #{bpb(b)})" if db > TOL_TRAIN
end
puts failures.length == n0 ? "  ok: 32-step trained agreement with EACH ARM AT ITS OWN LR, worst #{"%.3e" % worst_train} vs a #{TOL_TRAIN} tolerance" : "  FAIL: trained agreement"

# ---- 5. DETERMINISM ON CUDA ----
#
# Byte-exact PER PLATFORM is the standing rule, and it is what makes a
# CUDA-only sweep readable at all: if the twin were only reproducible to
# a tolerance, no two E-series cells could be compared even on one device.
n0 = failures.length
det_env = BIG.merge("STEPS" => "8", "SEED" => "0", "GTX_POLICY" => "dfa,dfa,dfa,dfa", "GTX_LR" => "0.0003")
d1, _ = run(GPU_BIN, det_env)
d2, _ = run(GPU_BIN, det_env)
failures << "determinism: two identical CUDA runs produced different curves — a CUDA-only sweep cannot compare its own cells if the device is not byte-exact with itself" unless
  steps_of(d1) == steps_of(d2) && bpb(d1) == bpb(d2)
puts failures.length == n0 ? "  ok: the CUDA twin is BYTE-EXACT with itself across runs" : "  FAIL: determinism"

# ---- 6. THE SCOPE IS REALLY bytelm-ONLY ----
#
# tao#24 reopened ONE task. If the twin quietly accepted the relational
# task, cells would appear that are comparable to nothing ever measured —
# so the refusal is part of the deliverable, and it is asserted at BOTH
# layers (the binary refuses even when driven by env directly; the CLI
# refuses before it ever selects a binary).
n0 = failures.length
[[{}, "relational (the default task)"],
 [{ "GTX_TASK" => "relational" }, "--task relational"],
 [{ "GTX_TASK" => "local" },      "--task local"]].each do |extra, label|
  out, st = run(GPU_BIN, { "STEPS" => "2", "SEED" => "0" }.merge(extra))
  failures << "scope: the CUDA binary ran #{label} (exit #{st.exitstatus}) — tao#24 scoped the twin to bytelm, and a cell from this path would be comparable to nothing" if st.success?
  failures << "scope: the CUDA binary refused #{label} without naming tao#24 or the CPU runner, so the refusal does not tell the caller what to do instead:\n#{out.lines.last(2).join}" unless
    st.success? || (out.include?("tao#24") && out.include?("toy-train-gtx"))
end
# The CLI layer. `--device cuda` is ACCEPTED for bytelm and REFUSED
# everywhere else, including the sibling lanes whose CPU-only rationale
# tao#24 explicitly did NOT lapse.
Dir.mktmpdir("gtx_cuda_cli") do |dir|
  ok_out, ok_st = Open3.capture2e({}, TOY, "train", "gtx", "--task", "bytelm",
                                  "--text", PACK, "--vocab", "65", "--device", "cuda",
                                  "--steps", "3", "--seed", "0", "--val-batches", "2",
                                  "--out", dir, chdir: ROOT)
  if ok_st.success?
    failures << "scope: `toy train gtx --task bytelm --device cuda` ran but emitted no bytelm: line — the allowlist filters the twin's result" unless
      ok_out.lines.any? { |l| l.start_with?("bytelm: ") }
    ev = File.join(dir, "events.jsonl")
    if File.file?(ev)
      kind = JSON.parse(File.readlines(ev).first.to_s).dig("backend", "kind")
      failures << "scope: `--device cuda` was accepted but the run recorded backend=#{kind.inspect} — the CLI accepted the flag and then dispatched to the CPU binary, which is the silent-no-op failure this tree refuses" unless kind == "cuda"
    else
      failures << "scope: `--device cuda` run wrote no events.jsonl, so the dispatch is unverifiable"
    end
  else
    failures << "scope: `toy train gtx --task bytelm --device cuda` exited #{ok_st.exitstatus} — the twin is not reachable through the CLI:\n#{ok_out.lines.last(5).join}"
  end
end
[[%w[gtx --device cuda],                                        "gtx --device cuda with no task"],
 [%w[gtx --task relational --device cuda],                      "gtx --task relational --device cuda"],
 [%w[gtx --task local --device cuda],                           "gtx --task local --device cuda"],
 [%w[gtx --task bytelm --text data/ae_shak_a65 --device metal], "gtx --device metal (no metal binary exists)"],
 [%w[ssm --task bytelm --text data/ae_shak_a65 --device cuda],  "ssm --device cuda (tao#24 kept it CPU-only)"],
 [%w[lstm --device cuda],                                       "lstm --device cuda (tao#21 does NOT lapse)"],
 [%w[mlp --device cuda],                                        "mlp --device cuda"],
 [%w[gnn --device cuda],                                        "gnn --device cuda"],
 [%w[diff --device cuda],                                       "diff --device cuda"],
 [%w[ctr --device cuda],                                        "ctr --device cuda"],
 [%w[ae --text data/ae_shak_a65 --device cuda],                 "ae --device cuda"]].each do |argv, label|
  out, st = Open3.capture2e({}, TOY, "train", *argv, chdir: ROOT)
  failures << "scope: the CLI accepted #{label} (exit #{st.exitstatus}) — tao#24 reopened the device scope for gtx-under-bytelm ONLY:\n#{out.lines.last(2).join}" unless st.exitstatus == 2
end
puts failures.length == n0 ? "  ok: the twin is bytelm-ONLY — the binary refuses relational/local naming tao#24, the CLI accepts `--device cuda` ONLY with `--task bytelm` (and proves it dispatched to cuda), and 10 other device forms across 8 lanes still fail loud" : "  FAIL: scope"

# ---- 7. THE CPU LANE IS UNMOVED ----
#
# A default that changes nothing. The fixture was recorded from the
# PRE-tao#24 binary (see its header), so this is a diff against what the
# lane did before the twin existed, not a re-record dressed as a proof.
n0 = failures.length
if File.file?(FIXTURE)
  want = File.readlines(FIXTURE).reject { |l| l.start_with?("#") }.join
  blocks = want.split(/^--- arm .*\n/).reject { |b| b.strip.empty? }
  got = [SMALL.merge("STEPS" => "5", "SEED" => "0"),
         SMALL.merge("STEPS" => "5", "SEED" => "0", "GTX_POLICY" => "dfa,dfa")].map do |env|
    o, _ = run(CPU_BIN, env)
    # The corpus: line names the pack's own token count, which belongs to
    # the data rather than to the code under test.
    o.lines.reject { |l| l.start_with?("corpus: ") }.join
  end
  if blocks.length != 2
    failures << "cpu byte-null: #{FIXTURE} does not parse into 2 arm blocks (got #{blocks.length}) — the fixture is malformed and this leg is measuring nothing"
  else
    %w[chain dfa].each_with_index do |arm, i|
      next if blocks[i] == got[i]
      diff = blocks[i].lines.zip(got[i].lines).reject { |x, y| x == y }.first(3)
      failures << "cpu byte-null: the #{arm} arm's CPU byte-LM curve MOVED. tao#24 was supposed to ADD a device, not change the one that was there — every P3-P6 cell is stated on these bytes. First differences (want -> got):\n" +
                  diff.map { |x, y| "    - #{x.to_s.chomp}\n    + #{y.to_s.chomp}" }.join("\n")
    end
  end
else
  failures << "cpu byte-null: #{FIXTURE} is missing — the leg that proves the CPU lane did not move cannot run"
end
puts failures.length == n0 ? "  ok: the CPU byte-LM curves (chain + dfa) are BYTE-IDENTICAL to the pre-tao#24 binary's" : "  FAIL: cpu byte-null"

if failures.empty?
  puts "GATE PASS [gtx-cuda]: the byte-LM CUDA twin — runs on cuda (asserted from run_start provenance), builds a BYTE-IDENTICAL graph on both backends, agrees to #{TOL_ARITH} on pure arithmetic (lr=0) and #{TOL_TRAIN} trained at each arm's own LR, is byte-exact with itself, is bytelm-ONLY at both the binary and the CLI, and left the CPU lane byte-identical (tao#24)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [gtx-cuda]: #{f}" }
  exit 1
end
