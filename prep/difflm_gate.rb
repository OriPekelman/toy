#!/usr/bin/env ruby
# prep/difflm_gate.rb — toy#166 (capstone P1b) gate for the LATENT
# DIFFUSION BYTE-LM lane (libexec/toy-train-difflm, `toy train difflm`).
#
# Legs:
#   1. DETERMINISM, and the arm actually selects a different model.
#   2. THE SESSIONS ARE SEQUENCED — the property the runner's whole shape
#      rests on, asserted rather than assumed.
#   3. LATENT STANDARDISATION lands, and is refused loudly if it does not.
#   4. THE abar_T GUARD fires on a schedule that never reaches noise.
#   5. THE JUDGE IS NOT AN ARM (different seed) and it discriminates.
#   6. CONTROL-CAN-LOSE — prior-floor must lose decisively (tao#19).
#   7. THE RESIDUAL INSTRUMENT reports in latent-std units and grows with t.
#  7b. THE eps-SKILL PROBE reports against a TRIVIAL baseline, and an arm
#      with no denoiser survives a run dir.
#   8. FAIL-LOUD.
#   9. CLI.
#
# ── WHAT THIS LANE ANSWERS ──
#
# P1a (toy#165) showed a d=8 per-token latent carries a byte decodably
# under noise. P1b asks whether a DIFFUSION model can GENERATE coherent
# text through it, against an AR yardstick and a prior-decode floor.
# ALL BP — DFA is P1c, and it attaches to the DENOISER (output dim = the
# latent, F20's window), never to the 256-way decode head.
#
# ── THE ONE THING TO READ BEFORE CHANGING THIS RUNNER ──
#
# tnn_session_new_on() calls ggml_backend_sched_reset() on the scheduler
# EVERY session on that backend shares. So two live sessions corrupt each
# other, SILENTLY: prep/smokes/smoke_two_sessions.rb measures a first
# session returning 1.049 where it returned 5.940 once a second exists.
# No crash, no warning. That is why this lane builds its three models
# one at a time and carries everything across a boundary as plain Ruby
# arrays — the decode head included, which is why decoding is a Ruby-side
# [256, d] matmul rather than a graph.
#
# Leg 2 asserts that property directly, so a future "tidy-up" that holds
# two sessions open fails here instead of in a published number.

ROOT   = File.expand_path("..", __dir__)
RUNNER = File.join(ROOT, "libexec", "toy-train-difflm")
PROBE  = File.join(ROOT, "prep", "smokes", "smoke_two_sessions")
TOY    = File.join(ROOT, "bin", "toy")
PACK   = File.join(ROOT, "data", "ae_shakespeare")

require "open3"
require "json"
require "tmpdir"

# A cheap cell: the legs below assert STRUCTURE (determinism, controls,
# guards, units), not the verdict. The verdict comes from the 12-cell
# sweep, which is far too expensive to gate on.
CELL = {
  "DL_TEXT" => PACK, "DL_LATENT" => "8", "DL_CONTEXT" => "64",
  "DL_D_MODEL" => "64", "DL_D_FF" => "128", "DL_BLOCKS" => "1",
  "DL_HEADS" => "4", "DL_AR_D_MODEL" => "64", "DL_AR_BLOCKS" => "1",
  "DL_AE_STEPS" => "60", "DL_TSTEPS" => "100", "DL_GEN_BYTES" => "256",
  "DL_JUDGE_STEPS" => "60",
}.freeze

def run_dl(extra, run_dir = nil)
  env = { "STEPS" => "20", "SEED" => "0" }.merge(CELL).merge(extra)
  env = env.merge("TAO_RUN_DIR" => run_dir, "TOY_RUN_ID" => "difflm-gate") if run_dir
  out, st = Open3.capture2e(env, RUNNER, chdir: ROOT)
  abort "difflm_gate: runner exited #{st.exitstatus}:\n#{out.lines.last(12).join}" unless st.success?
  out
end

def line(out, prefix)
  l = out.lines.find { |x| x.start_with?(prefix) }
  raise "difflm_gate: no #{prefix.inspect} line in\n#{out.lines.last(20).join}" unless l
  l
end

def field(out, prefix, key)
  line(out, prefix)[/#{Regexp.escape(key)}=([0-9.eE+-]+)/, 1]
end

def curve(out)
  out.lines.select { |l| l.start_with?("step ") }
end

unless %w[.meta.i32 .tok.i32].all? { |s| File.file?(PACK + s) }
  warn "difflm_gate: #{PACK} missing — run: ruby prep/fetch_text.rb --all"
  exit 2
end
unless File.executable?(RUNNER)
  o, st = Open3.capture2e("make", "-C", ROOT, "libexec/toy-train-difflm")
  unless st.success? && File.executable?(RUNNER)
    warn "difflm_gate: build failed:\n#{o.lines.last(15).join}"
    exit 2
  end
end

failures = []
n0 = 0

# ---- 1. determinism + the arm selects a model ----
n0 = failures.length
a1 = run_dl({ "DL_ARM" => "diff-selfcond" })
a2 = run_dl({ "DL_ARM" => "diff-selfcond" })
failures << "determinism: two identical diff-selfcond runs differ" unless a1 == a2
plain = run_dl({ "DL_ARM" => "diff-plain" })
failures << "arm effect: diff-plain and diff-selfcond produce the same training curve — self-conditioning is not reaching the model, so the ablation compares nothing" if curve(plain) == curve(a1)
seed1 = run_dl({ "DL_ARM" => "diff-selfcond", "SEED" => "1" })
failures << "seed inert: --seed did not move the curve" if curve(seed1) == curve(a1)
puts failures.length == n0 ?
  "  ok: identical runs are byte-identical; --seed moves the curve; self-conditioning changes the model (one graph, two arms, by DATA)" :
  "  FAIL: determinism / arm effect"

# ---- 2. THE SESSIONS ARE SEQUENCED ----
#
# The runner's entire shape rests on this. If a future change holds two
# sessions open, the FIRST one silently returns garbage and every number
# downstream is confident nonsense.
n0 = failures.length
unless File.executable?(PROBE)
  o, st = Open3.capture2e("make", "-C", ROOT, "prep/smokes/smoke_two_sessions")
  failures << "sessions: could not build the two-session probe" unless st.success?
end
if File.executable?(PROBE)
  po, pst = Open3.capture2e(PROBE, chdir: ROOT)
  failures << "sessions: the probe exited #{pst.exitstatus}" unless pst.success?
  failures << "sessions: a SECOND session no longer trains correctly (#{po.lines.grep(/q1/).join.strip})" unless po.include?("PASS q1")
  # The failing half is the POINT: if this ever starts passing, ggml has
  # changed and the runner could be simplified — but until then a
  # co-resident session is a silent corruption, so the gate records the
  # constraint rather than the wish.
  failures << "sessions: the probe now says a first session SURVIVES a second (#{po.lines.grep(/q2/).join.strip}) — if that is real the runner can be simplified, but verify before trusting it" unless po.include?("FAIL q2")
end
puts failures.length == n0 ?
  "  ok: the shared-scheduler constraint still holds (a second session invalidates the first) — the runner's sequenced shape is still required" :
  "  FAIL: session sequencing"

# ---- 3. latent standardisation ----
n0 = failures.length
lat = line(a1, "latent: ")
mm = lat[/std_max_mean=([0-9.eE+-]+)/, 1].to_f
sd = lat[/std_max_sd_err=([0-9.eE+-]+)/, 1].to_f
failures << "standardisation: max|mean| #{mm} is not ~0 — the sampler starts at N(0, I), so a non-standard aggregate means it begins OFF-MANIFOLD and the samples would be garbage for a reason unrelated to the latent width" if mm > 1.0e-3
failures << "standardisation: max|sd-1| #{sd} is not ~0 (same reason)" if sd > 1.0e-3
failures << "standardisation: the raw per-dim sd is not reported — it is what makes the residual instrument's units meaningful" unless lat.include?("raw_sd=")
puts failures.length == n0 ?
  "  ok: the latent is standardised to N(0, I) per dim, ASSERTED (max|mean|=#{mm}, max|sd-1|=#{sd}) — not assumed" :
  "  FAIL: latent standardisation"

# ---- 4. the abar_T guard ----
n0 = failures.length
o4, st4 = Open3.capture2e({ "STEPS" => "5" }.merge(CELL).merge(
  "DL_ARM" => "diff-selfcond", "DL_TSTEPS" => "20", "DL_BETA_HI" => "0.02"),
  RUNNER, chdir: ROOT)
failures << "abar guard: a schedule leaving abar_T >> 0 was accepted — the sampler would start OUT OF DISTRIBUTION and the generative metric would score that mismatch instead of the model (toy#156)" if st4.success?
failures << "abar guard: the message does not report abar_T" unless o4.include?("abar_T=")
puts failures.length == n0 ?
  "  ok: a schedule that never reaches pure noise is REFUSED, naming abar_T (toy#156's landmine, carried)" :
  "  FAIL: abar_T guard"

# ---- 5. the judge is not an arm, and it discriminates ----
n0 = failures.length
jl = line(a1, "judge: ")
failures << "judge: seed is not offset from the run seed (#{jl.strip}) — scoring an arm under itself hands the ceiling an unquantifiable advantage, and the whole 'competitive' bar is stated against that anchor" unless jl.include?("seed=1000")
ar = run_dl({ "DL_ARM" => "ar-baseline" })
jl_ar = line(ar, "judge: ")
failures << "judge: the ar-baseline arm's judge shares its seed — that IS scoring a model under itself" if jl_ar.include?("seed=0 ")
bpb_real_a = field(a1, "gen: ", "bpb_real").to_f
bpb_real_b = field(ar, "gen: ", "bpb_real").to_f
failures << "judge: bpb_real differs between arms (#{bpb_real_a} vs #{bpb_real_b}) — the anchor must be a property of the judge and the corpus, not of the arm" if (bpb_real_a - bpb_real_b).abs > 1.0e-6
puts failures.length == n0 ?
  "  ok: the judge is a DIFFERENT seed from every arm and its real-text anchor is arm-independent (#{bpb_real_a})" :
  "  FAIL: judge independence"

# ---- 6. CONTROL-CAN-LOSE (tao#19) ----
n0 = failures.length
floor = run_dl({ "DL_ARM" => "prior-floor" })
bpb_floor = field(floor, "gen: ", "bpb_gen").to_f
bpb_real  = field(floor, "gen: ", "bpb_real").to_f
failures << "control: the prior-floor decode (#{bpb_floor}) does not score WORSE than real text (#{bpb_real}) under the judge — a metric where decoding random latents looks like text cannot discriminate anything (tao#19, the F18/F9e trap on a generative lane)" unless bpb_floor > bpb_real + 0.5
fl = line(floor, "arm: ")
failures << "control: prior-floor reports a denoiser (#{fl.strip}) — it must decode the PRIOR, with no denoising at all" unless fl.include?("denoiser_params=0")
puts failures.length == n0 ?
  "  ok: CONTROL CAN LOSE — prior-floor scores #{bpb_floor} bits/byte against real text at #{bpb_real}, with no denoiser at all (tao#19)" :
  "  FAIL: control-can-lose"

# ---- 7. the residual instrument ----
#
# P1a's margin is denominated in latent-std units. This is the number
# that makes "does the margin accommodate the sampler" a MEASUREMENT
# rather than an inference from sample quality.
n0 = failures.length
res = line(a1, "resid: ")
vals = res.scan(/t\d+=([0-9.eE+-]+)/).flatten.map(&:to_f)
failures << "residual: fewer than 3 t points (#{res.strip})" if vals.length < 3
failures << "residual: not reported in latent-std units" unless res.include?("latent-std")
if vals.length >= 3
  failures << "residual: not monotone in t (#{vals.inspect}) — a longer reverse chain compounds more error, so a non-monotone read means the chain or the schedule is wrong" unless vals.each_cons(2).all? { |a, b| b >= a }
end
failures << "residual: the ar-baseline arm reports a sampler residual, which it has no sampler for" if ar.lines.any? { |l| l.start_with?("resid:") }
puts failures.length == n0 ?
  "  ok: the sampler residual is reported in LATENT-STD units at 3 chain lengths and grows with t (#{vals.inspect}) — P1a's margin axis, measured" :
  "  FAIL: residual instrument"

# ---- 7b. the eps-SKILL probe, and it must survive EVERY arm ----
#
# The probe only FILLS on the diffusion arms, but the events block reads
# its arrays for every arm. Declaring them where they were filled left
# prior-floor referencing unassigned locals, and the runner SEGV'd only
# on the path that ALSO had a run dir — i.e. only under `toy train`,
# never under a direct runner call. So this leg exercises a run dir on
# the arm that does NOT fill the probe.
n0 = failures.length
Dir.mktmpdir("difflm_probe") do |dir|
  fl2 = run_dl({ "DL_ARM" => "prior-floor" }, dir)
  failures << "eps probe: prior-floor emitted epsmse lines, which it has no denoiser for" if fl2.lines.any? { |l| l.start_with?("epsmse:") }
end
eps = a1.lines.select { |l| l.start_with?("epsmse:") }
failures << "eps probe: no epsmse lines on a diffusion arm" if eps.empty?
if eps.length >= 3
  sk = eps.map { |l| l[/skill=([0-9.eE+-]+)/, 1].to_f }
  ab = eps.map { |l| l[/abar=([0-9.eE+-]+)/, 1].to_f }
  failures << "eps probe: abar is not decreasing across the reported t grid — the schedule or the grid is wrong" unless ab.each_cons(2).all? { |x, y| y <= x }
  # The probe's WHOLE point is that raw eps-MSE is inverted at high t
  # (the trivial predictor eps_hat = x_t scores ~0 as abar -> 0), so the
  # baseline must be reported and must shrink with abar. Without that the
  # skill number is not interpretable.
  tv = eps.map { |l| l[/trivial=([0-9.eE+-]+)/, 1].to_f }
  failures << "eps probe: the trivial baseline is not reported shrinking with abar — skill is uninterpretable without it" unless tv.last < tv.first
  failures << "eps probe: skill is identical at every t (#{sk.uniq.length} distinct) — the probe is not varying with t" if sk.uniq.length < 3
end
puts failures.length == n0 ?
  "  ok: the eps-skill probe reports model vs the TRIVIAL baseline across a decreasing-abar grid, and an arm with no denoiser survives a run dir (the SEGV that only appeared under `toy train`)" :
  "  FAIL: eps probe"

# ---- 8. fail-loud ----
n0 = failures.length
[
  [{ "DL_TEXT" => "" },                       "no corpus"],
  [{ "DL_TEXT" => "/nonexistent" },           "nonexistent pack"],
  [{ "DL_ARM" => "nope" },                    "unknown arm"],
  [{ "DL_LATENT" => "64", "DL_D_MODEL" => "64" }, "latent >= d_model"],
  [{ "DL_TSTEPS" => "1" },                    "tsteps 1"],
  [{ "DL_GEN_BYTES" => "8" },                 "gen-bytes < context"],
  [{ "DL_JUDGE_STEPS" => "0" },               "judge steps 0 (every arm scored by an untrained judge)"],
].each do |env, label|
  _o, st = Open3.capture2e({ "STEPS" => "2" }.merge(CELL).merge(env), RUNNER, chdir: ROOT)
  failures << "fail-loud: #{label} exited 0" if st.success?
end
puts failures.length == n0 ? "  ok: 7 degenerate configs fail loud" : "  FAIL: fail-loud"

# ---- 9. CLI ----
n0 = failures.length
Dir.mktmpdir("difflm_cli") do |dir|
  out, st = Open3.capture2e({ "SPINEL_DIR" => ENV["SPINEL_DIR"].to_s },
    TOY, "train", "difflm", "--text", PACK, "--arm", "prior-floor",
    "--latent", "8", "--context", "64", "--d-model", "64", "--d-ff", "128",
    "--layers", "1", "--ae-steps", "40", "--gen-bytes", "256",
    "--judge-steps", "40", "--steps", "10", "--out", dir, chdir: ROOT)
  if st.success?
    %w[stage1: arm: gen: judge:].each do |k|
      failures << "cli: output lacks #{k.inspect}" unless out.include?(k)
    end
    %w[gen.bytes real.bytes sample.txt].each do |f|
      failures << "cli: #{f} not written — the n-gram metrics are computed OUT of the runner (prep/difflm_report.rb), so the byte streams have to reach disk" unless File.file?(File.join(dir, f))
    end
  else
    failures << "cli: `toy train difflm` exited #{st.exitstatus}:\n#{out.lines.last(6).join}"
  end
end
[
  [%w[difflm --arm diff-plain],                     "--text omitted"],
  [%w[difflm --text data/ae_shakespeare --policy dfa], "--policy on an all-BP lane"],
  [%w[difflm --text data/ae_shakespeare --device cuda], "--device cuda"],
  [%w[difflm --text data/ae_shakespeare --arm nope],   "unknown arm"],
  [%w[ae --arm diff-plain],                         "--arm on the ae lane"],
].each do |args, label|
  out, st = Open3.capture2e(TOY, "train", *args, "--steps", "2", chdir: ROOT)
  failures << "cli: #{label} exited 0" if st.success?
end
puts failures.length == n0 ?
  "  ok: `toy train difflm` runs, writes the byte streams the metrics read, and 5 misuses are refused — including --policy, because P1b is all-BP by DESIGN and DFA belongs on the denoiser in P1c" :
  "  FAIL: cli"

if failures.empty?
  puts "GATE PASS [difflm]: latent-diffusion byte-LM (capstone P1b, all BP) — three models built SEQUENTIALLY because two live sessions silently corrupt each other, a latent standardised to N(0,I) and ASSERTED so the sampler cannot start off-manifold, toy#156's abar_T guard carried, a judge that is never an arm, a prior-floor control that provably CAN lose, and a sampler residual measured in P1a's own latent-std units (toy#166) + the eps-SKILL probe scored against a trivial baseline, because raw eps-MSE is INVERTED at high t (toy#167)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [difflm]: #{f}" }
  exit 1
end
