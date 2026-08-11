#!/usr/bin/env ruby
# prep/diff_gate.rb — toy#156 (DFA-arch T2) gate for the latent
# diffusion lane (libexec/toy-train-diff, `toy train diff`).
#
# Legs:
#   1. BYTE FIXTURE + CHAIN BYTE-NULL.
#   2. DETERMINISM.
#   3. ARM EFFECT + the B seed moves dfa and is inert on chain.
#   4. WIRING + BUNDLE (including the abar_final provenance).
#  4b. THE ALIGNMENT PHASE — cos(g_dfa, g_bp) climbs.
#   5. THE MANDATORY SUCCESS BAR, on the GENERATIVE metric at latent 4.
#   6. THE OUTPUT-DIM LENS — the boundary, at latent 16.
#   7. THE DEGENERATE TASK, measured.
#   8. THE SCHEDULE GUARD — the bug that was silent in the loss.
#   9. FAIL-LOUD.
#  10. CLI.
#
# ── THE METRIC IS LOWER-IS-BETTER, SO THE BAR INVERTS ──
#
# Every other lane in this program scores accuracy or AUC. This one
# scores an ENERGY DISTANCE between ancestrally-sampled points and
# held-out reals — a proper metric, zero iff the distributions match, so
# SMALLER IS BETTER and every comparison below runs the other way round.
# A negative value is not a bug: the estimator is unbiased (within-set
# pairs exclude the diagonal), so it goes slightly below zero when the
# two sets are genuinely indistinguishable at this sample size.
#
# ── WHAT THIS LANE MEASURED ──
#
# All three arms at ONE shared learning rate (3e-4), 10000 steps, so the
# arms differ ONLY in --policy:
#
#   latent 4   seed 0/1/2   BP .102/.184/.131  DFA .010/.135/.072  frozen .590/1.084/1.005
#   latent 16  seed 0       BP .130            DFA .979            frozen 9.020
#
# **DFA BEATS BP at latent 4, on every seed**, while both crush the
# frozen control — the ticket's success target met and exceeded. At
# latent 16 DFA is 7.5x WORSE than BP (and 15-27x at each arm's own best
# LR), while still beating frozen 9x. That is the output-dim lens
# toy#152 measured, reproduced on a GENERATIVE objective, with the
# boundary located between 4 and 16 — much lower than the classification
# lens, where 10 classes was still comfortable.
#
# Legs 5 and 6 pin both ends. Either alone is misreadable: the latent-4
# win alone reads as "DFA works on diffusion", the latent-16 loss alone
# as "it does not".

ROOT    = File.expand_path("..", __dir__)
RUNNER  = File.join(ROOT, "libexec", "toy-train-diff")
TOY     = File.join(ROOT, "bin", "toy")
FIXTURE = File.join(ROOT, "prep", "fixtures", "train_diff_baseline.txt")

require "open3"
require "json"
require "tmpdir"
require "fileutils"

# The shared-LR cell. 3e-4 and NOT this lane's default 3e-3, and that is
# the one place the gate deviates from the runner defaults on purpose:
# BP's own best is 3e-3 (energy .036 at latent 16) but DFA diverges
# there (4.6e6 at 1e-2, 5.99 at 3e-3), so a 3e-3 cell would compare a
# converged BP against a broken DFA. 3e-4 is the highest rate BOTH arms
# tolerate, which is the honest shared cell — and DFA tolerating less LR
# than BP is itself the toy#152 finding, restated here.
#
# 10000 steps is also load-bearing: at 3000 steps BP has NOT converged
# at this rate (latent 16: BP .802 vs DFA .947, which reads as parity)
# and the lane's whole result would invert. Measured, not cautious.
CELL = { "STEPS" => "10000", "SEED" => "0", "DIFF_LR" => "0.0003" }.freeze
BAR_LATENT  = "4"
LENS_LATENT = "16"
# Margins, set well inside the measured effects (see the table above).
BP_GAP      = 0.10   # dfa must not exceed chain by more than this
FROZEN_EDGE = 0.20   # ...and must beat frozen by at least this
BP_PRECOND  = 0.20   # ...and BP must beat frozen first, or the row is mute
LENS_RATIO  = 3.0    # at latent 16 dfa must be at least this many x worse

def run_diff(extra_env, run_dir)
  env = { "STEPS" => "5", "SEED" => "0" }.merge(extra_env)
  env = env.merge("TAO_RUN_DIR" => run_dir, "TOY_RUN_ID" => "diff-gate") if run_dir
  out, st = Open3.capture2e(env, RUNNER, chdir: ROOT)
  abort "diff_gate: runner exited #{st.exitstatus}:\n#{out.lines.last(10).join}" unless st.success?
  out
end

def curve(out)
  out.lines.select { |l| l.start_with?("step ") }
end

def energy(out)
  line = out.lines.find { |l| l.start_with?("gen: ") }
  raise "diff_gate: no gen line in\n#{out}" unless line
  line[/energy=([0-9.eE+-]+)/, 1].to_f
end

def arms_at(latent)
  r = {}
  %w[chain dfa frozen].each do |arm|
    r[arm] = energy(run_diff(CELL.merge("DIFF_LATENT" => latent,
                                        "DIFF_POLICY" => ([arm] * 3).join(",")), nil))
  end
  r
end

unless File.executable?(RUNNER)
  build_out, build_st = Open3.capture2e("make", "-C", ROOT, "libexec/toy-train-diff")
  unless build_st.success? && File.executable?(RUNNER)
    warn "diff_gate: build failed:\n#{build_out.lines.last(15).join}"
    exit 2
  end
end

failures = []
# d878143: each leg records the failure count at its START in `n0` and
# summarises with `failures.length == n0`, so it reports on its OWN
# assertions.
n0 = 0

# ---- 1. byte fixture + chain byte-null ----
n0 = failures.length
base_out   = run_diff({}, nil)
base_curve = curve(base_out)
if File.file?(FIXTURE)
  expect = File.readlines(FIXTURE).reject { |l| l.start_with?("#") || l.strip.empty? }
  failures << "fixture: default curve != train_diff_baseline.txt\ngot:  #{base_curve.join}want: #{expect.join}" unless base_curve == expect
else
  failures << "fixture: #{FIXTURE} missing"
end
chain_out = run_diff({ "DIFF_POLICY" => "chain,chain,chain" }, nil)
failures << "chain byte-null: explicit all-chain curve != absent-policy curve" unless curve(chain_out) == base_curve
puts failures.length == n0 ? "  ok: default curve matches the fixture AND an explicit all-chain policy is byte-null" : "  FAIL: fixture / chain byte-null"

# ---- 2. determinism ----
n0 = failures.length
d1 = run_diff({ "DIFF_POLICY" => "dfa,dfa,dfa", "STEPS" => "20" }, nil)
d2 = run_diff({ "DIFF_POLICY" => "dfa,dfa,dfa", "STEPS" => "20" }, nil)
failures << "determinism: two identical dfa runs differ" unless d1 == d2
puts failures.length == n0 ? "  ok: two identical dfa runs are byte-identical (curve + val + gen)" : "  FAIL: determinism"

# ---- 3. arm effect + the B seed ----
n0 = failures.length
ch20 = run_diff({ "STEPS" => "20" }, nil)
fz   = run_diff({ "DIFF_POLICY" => "frozen,frozen,frozen", "STEPS" => "20" }, nil)
failures << "arm effect: dfa curve identical to chain" if curve(d1) == curve(ch20)
failures << "arm effect: frozen curve identical to chain" if curve(fz) == curve(ch20)
dfa_b999   = run_diff({ "DIFF_POLICY" => "dfa,dfa,dfa", "STEPS" => "20", "DIFF_B_SEED" => "999" }, nil)
chain_b999 = run_diff({ "STEPS" => "20", "DIFF_B_SEED" => "999" }, nil)
failures << "b seed: --dfa-b-seed does not move the dfa curve — the feedback matrix is not reaching the weights" if curve(dfa_b999) == curve(d1)
failures << "b seed: --dfa-b-seed moved the CHAIN curve — a pure-backprop arm must not see the feedback matrix" unless curve(chain_b999) == curve(ch20)
puts failures.length == n0 ? "  ok: dfa and frozen move the curve off chain; the B seed moves dfa and is inert on chain" : "  FAIL: arm effect / b seed"

# ---- 4 + 4b. wiring, bundle, alignment ----
n0 = failures.length
Dir.mktmpdir("diff_gate") do |dir|
  out = run_diff({ "DIFF_POLICY" => "dfa,dfa,frozen", "STEPS" => "400",
                   "DIFF_ALIGN" => "1", "DIFF_ALIGN_EVERY" => "100",
                   "DIFF_B_SEED" => "42" }, dir)
  failures << "bundle: runner printed no gen line" unless out.lines.any? { |l| l.start_with?("gen: ") }
  ev_path = File.join(dir, "events.jsonl")
  if File.file?(ev_path)
    events = File.readlines(ev_path).map { |l| JSON.parse(l) }
    rs = events.first || {}
    failures << "bundle: first event not run_start" unless rs["kind"] == "run_start"
    failures << "bundle: last event not run_end" unless events.last && events.last["kind"] == "run_end"
    md = rs["model"] || {}
    failures << "bundle: model.arch != diff" unless md["arch"] == "diff"
    failures << "bundle: model.objective != eps_prediction" unless md["objective"] == "eps_prediction"
    # The OUTPUT DIM under test is named latent_dim, NOT num_classes:
    # this lane regresses epsilon, and a consumer reading it as a class
    # count would compare it to the wrong lanes.
    failures << "bundle: model.latent_dim missing (got #{md['latent_dim'].inspect})" unless md["latent_dim"] == 16
    cfg = rs["config"] || {}
    # The schedule provenance. abar_final is what says the forward
    # process actually reached noise — see leg 8.
    failures << "bundle: config.abar_final missing/too large (#{cfg['abar_final'].inspect})" unless cfg["abar_final"].is_a?(Numeric) && cfg["abar_final"] < 0.01
    df = rs["dfa"]
    if df.nil?
      failures << "bundle: run_start has no dfa object"
    else
      failures << "bundle: dfa.policy != [1,1,2] (got #{df['policy'].inspect})" unless df["policy"] == [1, 1, 2]
      failures << "bundle: dfa.dfa_wired != 2" unless df["dfa_wired"] == 2
      failures << "bundle: dfa.frozen != 1" unless df["frozen"] == 1
    end
    evs = events.select { |e| e["kind"] == "eval" }
    if evs.length != 1
      failures << "bundle: #{evs.length} eval events (want 1)"
    else
      failures << "bundle: eval event carries no energy_distance" unless evs.first["energy_distance"].is_a?(Numeric)
      failures << "bundle: eval event carries no val_mse" unless evs.first["val_mse"].is_a?(Numeric)
    end
    # 4b — the alignment phase. This lane uses toy#152's DIRECT rule, so
    # unlike the ssm/franken macro lanes it CAN compare the DFA update
    # against the chain shadow. cos climbing from ~0 is Refinetti's
    # "align, then memorise" on a regression objective, and it is a far
    # sharper assertion than "the curve moved": a broken wiring can move
    # a curve but cannot rotate the shadow gradient towards a fixed
    # random matrix.
    aligns = events.select { |e| e["kind"] == "align" }
    names = aligns.map { |e| e["wname"] }.uniq.sort
    failures << "bundle: align wnames #{names.inspect} (want [\"w1\", \"w2\"])" unless names == %w[w1 w2]
    bad = aligns.count do |e|
      c = e["cos"]
      !c.is_a?(Numeric) || c.to_f.nan? || c.to_f.abs > 1.0001 ||
        !e["dfa_norm"].is_a?(Numeric) || e["dfa_norm"] <= 0 ||
        !e["bp_norm"].is_a?(Numeric) || e["bp_norm"] <= 0
    end
    failures << "bundle: #{bad} malformed align events" unless bad == 0
    %w[w1 w2].each do |wn|
      series = aligns.select { |e| e["wname"] == wn }.sort_by { |e| e["step"] }
      if series.length < 3
        failures << "alignment: only #{series.length} align events for #{wn}"
        next
      end
      failures << "alignment: #{wn} cos did not rise (#{series.first['cos'].round(4)} -> #{series.last['cos'].round(4)}) — DFA is not aligning, so a near-BP number on this lane would not be the mechanism we think it is" unless series.last["cos"] > series.first["cos"] + 0.15 && series.last["cos"] > 0.2
    end
  else
    failures << "bundle: no events.jsonl"
  end
end
puts failures.length == n0 ? "  ok: bundle + wiring counts + the alignment phase (cos climbs from ~0 on a REGRESSION objective)" : "  FAIL: wiring / bundle / alignment"

# ---- 5. THE MANDATORY SUCCESS BAR, at latent 4 ----
n0 = failures.length
bar = arms_at(BAR_LATENT)
puts format("    latent=%-3s chain=%.4f dfa=%.4f frozen=%.4f   (energy: LOWER is better)",
            BAR_LATENT, bar["chain"], bar["dfa"], bar["frozen"])
if bar["chain"] > bar["frozen"] - BP_PRECOND
  failures << "SUCCESS BAR (precondition): BP #{bar['chain']} does not beat frozen #{bar['frozen']} by #{BP_PRECOND} — the cell cannot discriminate, so nothing below it means anything"
end
if bar["dfa"] > bar["chain"] + BP_GAP
  failures << "SUCCESS BAR (BP gap): dfa #{bar['dfa']} is more than #{BP_GAP} worse than chain #{bar['chain']}"
end
if bar["dfa"] > bar["frozen"] - FROZEN_EDGE
  failures << "SUCCESS BAR (frozen control): dfa #{bar['dfa']} does not beat frozen #{bar['frozen']} by #{FROZEN_EDGE} — 'near-BP' alone cannot tell 'DFA learned' from 'this task is easy'"
end
puts failures.length == n0 ?
  "  ok: SUCCESS BAR at latent 4 — DFA matches (in fact beats) BP on the generative metric AND crushes the frozen control, all three arms at ONE learning rate" :
  "  FAIL: success bar"

# ---- 6. THE OUTPUT-DIM LENS — the other end ----
n0 = failures.length
lens = arms_at(LENS_LATENT)
puts format("    latent=%-3s chain=%.4f dfa=%.4f frozen=%.4f", LENS_LATENT,
            lens["chain"], lens["dfa"], lens["frozen"])
# Half one: DFA still beats the frozen control here, so the arm is alive.
if lens["dfa"] > lens["frozen"] * 0.5
  failures << "LENS: at latent #{LENS_LATENT} dfa #{lens['dfa']} is not clearly better than frozen #{lens['frozen']} — the DFA arm has stopped learning altogether, which is a different (and worse) result than the degradation this leg is about"
end
# Half two: and it NO LONGER matches BP. Asserting the negative is
# deliberate — it is the boundary of the lens, and a change that made
# DFA match BP at latent 16 would be a finding worth stopping for.
if lens["dfa"] < lens["chain"] * LENS_RATIO
  failures << "LENS: at latent #{LENS_LATENT} dfa #{lens['dfa']} is within #{LENS_RATIO}x of chain #{lens['chain']}. This lane's measured result is that DFA matches BP at latent 4 and is 7.5x worse at latent 16 — the output-dim lens on a GENERATIVE objective. If that boundary moved, re-read docs/roadmap/dfa-arch-program-2026-08-10.md before re-baselining."
end
puts failures.length == n0 ?
  "  ok: OUTPUT-DIM LENS — at latent 16 DFA still beats frozen but no longer matches BP (the boundary sits between 4 and 16)" :
  "  FAIL: output-dim lens"

# ---- 7. the degenerate task, measured ----
# `--task single` replaces the mixture with ONE Gaussian, which a frozen
# random denoiser reproduces as well as a trained one — measured at
# chain -.0020 / dfa -.0011 / frozen -.0012, i.e. all three
# indistinguishable from the data. This lane's `blobs`.
n0 = failures.length
sing = {}
%w[chain frozen].each do |arm|
  sing[arm] = energy(run_diff(CELL.merge("DIFF_LATENT" => BAR_LATENT,
                                         "DIFF_TASK" => "single",
                                         "DIFF_POLICY" => ([arm] * 3).join(",")), nil))
end
if (sing["frozen"] - sing["chain"]).abs > 0.05
  failures << "degenerate task: --task single separates chain #{sing['chain']} from frozen #{sing['frozen']} — it is documented as solvable WITHOUT a trained denoiser, and the default `mixture` task is justified against that measurement"
end
puts failures.length == n0 ? "  ok: --task single is measurably degenerate (a FROZEN denoiser matches a trained one), which is why `mixture` is the default" : "  FAIL: degenerate task"

# ---- 8. the schedule guard ----
# THE BUG THAT WAS SILENT IN THE LOSS. With Ho et al.'s betas over
# T=100 the forward process leaves abar_T = 0.60, so the ancestral
# sampler — which starts from pure N(0,I) — starts OUT OF DISTRIBUTION.
# Measured at those defaults: BP had the BEST denoising MSE (.684) and
# the WORST energy distance (29.1, against 4.95 for an untrained net),
# with a perfectly healthy-looking training curve throughout.
n0 = failures.length
out, st = Open3.capture2e({ "STEPS" => "2", "DIFF_BETA_HI" => "0.02",
                            "DIFF_BETA_LO" => "0.0001" }, RUNNER, chdir: ROOT)
failures << "schedule guard: a schedule leaving abar_T = 0.6 was accepted — the sampler would start out of distribution and the generative metric would score that instead of the model" if st.success?
failures << "schedule guard: rejected, but the message does not name abar_T (#{out.lines.last(1).join.strip})" unless out.include?("abar_T")
puts failures.length == n0 ? "  ok: a noise schedule that never reaches pure noise is REJECTED (the failure that was silent in the loss)" : "  FAIL: schedule guard"

# ---- 9. fail-loud ----
n0 = failures.length
[
  [{ "DIFF_POLICY" => "chain,chain,chain,chain" }, "policy longer than DIFF_LAYERS"],
  [{ "DIFF_POLICY" => "dfa,bogus,chain" },         "unknown policy token"],
  [{ "DIFF_TASK" => "swiss-roll" },                "unknown DIFF_TASK"],
  [{ "DIFF_LATENT" => "0" },                       "DIFF_LATENT=0"],
  [{ "DIFF_LAYERS" => "0" },                       "DIFF_LAYERS=0"],
  [{ "DIFF_DIFF_STEPS" => "1" },                   "DIFF_DIFF_STEPS=1"],
  [{ "DIFF_EVAL_N" => "100" },                     "eval_n not a multiple of batch"],
  [{ "DIFF_BETA_LO" => "0.5", "DIFF_BETA_HI" => "0.2" }, "beta_lo >= beta_hi"],
].each do |env, label|
  o, s = Open3.capture2e({ "STEPS" => "2" }.merge(env), RUNNER, chdir: ROOT)
  failures << "fail-loud: #{label} exited 0 (silently did nothing):\n#{o.lines.last(2).join}" if s.success?
end
puts failures.length == n0 ? "  ok: 8 degenerate configs all fail loud instead of quietly doing nothing" : "  FAIL: fail-loud"

# ---- 10. CLI ----
n0 = failures.length
Dir.mktmpdir("diff_gate_cli") do |dir|
  out, st = Open3.capture2e({ "SPINEL_SKIP_PIN_CHECK" => nil }, TOY, "train", "diff",
                            "--steps", "5", "--seed", "0", "--out", dir, chdir: ROOT)
  if st.success?
    failures << "cli: `toy train diff` curve != the runner's default curve" unless curve(out) == base_curve
    failures << "cli: `toy train diff` did not echo the gen line" unless out.lines.any? { |l| l.start_with?("gen: ") }
  else
    failures << "cli: `toy train diff` exited #{st.exitstatus}:\n#{out.lines.last(5).join}"
  end
  [
    [%w[diff --policy-scope ffn],  "diff accepted --policy-scope"],
    [%w[diff --task cue],          "diff accepted --task cue (that is the ssm lane's)"],
    [%w[mlp --latent 8],           "mlp accepted --latent"],
    [%w[ssm --diff-steps 50],      "ssm accepted --diff-steps"],
    [%w[gnn --modes 4],            "gnn accepted --modes"],
  ].each do |argv, label|
    sout, sst = Open3.capture2e({}, TOY, "train", *argv, chdir: ROOT)
    failures << "cli: #{label} (exit #{sst.exitstatus}: #{sout.lines.last(1).join.strip})" unless sst.exitstatus == 2
  end
  cout, cst = Open3.capture2e({}, TOY, "train", "diff", "--device", "cuda", chdir: ROOT)
  failures << "cli: diff accepted --device cuda (CPU-only by decision — tao#18)" unless cst.exitstatus == 2
end
puts failures.length == n0 ? "  ok: CLI reproduces the curve; lane-foreign flags and --device cuda are rejected" : "  FAIL: CLI"

if failures.empty?
  puts "GATE PASS [diff]: latent diffusion — byte fixture, determinism, the B seed, the alignment phase on a REGRESSION objective, the MANDATORY success bar on the GENERATIVE metric at latent 4 (DFA beats BP), the output-dim lens boundary at latent 16, the degenerate single-Gaussian control, and the abar_T schedule guard (toy#156)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [diff]: #{f}" }
  exit 1
end
