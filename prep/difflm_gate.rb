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
#  7c. THE OBJECTIVE AXIS (toy#168) — four objectives, all distinct,
#      eps-uniform byte-null, v-param scored on the eps axis.
#  7d. THE GAUSSIANITY PROBE — reported against a same-n N(0,I) reference
#      that itself sits at the noise floor.
#  7f. THE JOINT PROBE reports its shuffled floor (kept as a NEGATIVE
#      result: linear correlation is the wrong lens for a categorical code).
#  7e. THE STAGE-1 KL ARMS — beta=0 byte-null, learned sigma distinct from
#      fixed, and the decode head survives a weight being appended.
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

# ---- 7c. the OBJECTIVE axis (toy#168) ----
#
# The weights are applied as a t-SAMPLING distribution, never as a loss
# multiplier: this lane draws ONE t per step and Adam is scale-invariant
# per parameter, so a multiplier would be inert and the arm would report
# a silent null (toy#152's B-scale landmine in a new place). The legs
# below assert the axis is real and that eps-uniform is byte-null.
n0 = failures.length
epsu = run_dl({ "DL_ARM" => "diff-plain", "DL_LOSS_WEIGHT" => "eps-uniform" })
failures << "objective: an explicit eps-uniform is NOT byte-null against the default — the baseline arm must be exactly the pre-toy#168 behaviour or nothing compares across the axis" unless curve(epsu) == curve(plain)
vp  = run_dl({ "DL_ARM" => "diff-plain", "DL_LOSS_WEIGHT" => "v-param" })
ms  = run_dl({ "DL_ARM" => "diff-plain", "DL_LOSS_WEIGHT" => "min-snr-gamma" })
nu  = run_dl({ "DL_ARM" => "diff-plain", "DL_LOSS_WEIGHT" => "nonuniform-t" })
failures << "objective: v-param did not move the curve — the target is still eps" if curve(vp) == curve(plain)
failures << "objective: min-snr-gamma did not move the curve — the weight is not reaching the t draw (a LOSS multiplier here would be inert under Adam; it must be a sampling distribution)" if curve(ms) == curve(plain)
failures << "objective: nonuniform-t did not move the curve" if curve(nu) == curve(plain)
failures << "objective: min-snr-gamma and nonuniform-t are identical — two different t distributions produced one curve" if curve(ms) == curve(nu)
[[epsu, "eps-uniform", "uniform", "eps"], [vp, "v-param", "uniform", "v"],
 [ms, "min-snr-gamma", "min-snr-gamma", "eps"], [nu, "nonuniform-t", "ramp", "eps"]].each do |out, name, dist, tgt|
  ol = line(out, "objective: ")
  failures << "objective: #{name} does not report its t distribution (#{ol.strip})" unless ol.include?("t_dist=" + dist)
  failures << "objective: #{name} does not report target=#{tgt} (#{ol.strip})" unless ol.include?("target=" + tgt)
end
# v-param's prediction is v, not eps — the probe must undo the
# parameterisation BEFORE the metric or the arms are not on one axis.
vk = vp.lines.select { |l| l.start_with?("epsmse:") }
failures << "objective: v-param emits no epsmse lines — every arm must be scored on the eps axis" if vk.empty?
if !vk.empty?
  tv = vk.map { |l| l[/trivial=([0-9.eE+-]+)/, 1].to_f }
  ek = vk.map { |l| l[/skill=([0-9.eE+-]+)/, 1].to_f }
  failures << "objective: v-param's trivial baseline differs from the eps arms' — the baseline is a property of the SCHEDULE, so a differing one means the probe is scoring v against an eps baseline" unless (tv.last - epsu.lines.select { |l| l.start_with?("epsmse:") }.map { |l| l[/trivial=([0-9.eE+-]+)/, 1].to_f }.last).abs < 1e-9
  failures << "objective: v-param skill is constant across t (#{ek.uniq.length} distinct)" if ek.uniq.length < 3
end
puts failures.length == n0 ?
  "  ok: all four objectives move the curve and are distinct, eps-uniform is BYTE-NULL against the default, and v-param is scored on the eps axis against the same schedule-derived baseline" :
  "  FAIL: objective axis"

# ---- 7d. the GAUSSIANITY probe (toy#168 followup) ----
#
# Standardisation makes the aggregate zero-mean/unit-variance PER DIM. It
# does NOT make it Gaussian and does NOT make the dims uncorrelated —
# measured here at max|corr| 0.44-0.63 and Var||z||^2 ~4x below 2d, i.e.
# a correlated thin shell where the sampler starts from an isotropic
# ball. Asserting the standardisation moments (leg 3) was asserting the
# wrong property; this leg records the RIGHT one so the gap stays visible
# rather than being re-forgotten.
#
# It does NOT gate on the aggregate BEING Gaussian — it is not, and
# failing the battery on a known open finding would be noise. It gates
# that the number is REPORTED against a same-n reference, because a bare
# kurtosis is uninterpretable without its sampling-noise floor.
n0 = failures.length
g  = line(a1, "gauss: ")
gr = line(a1, "gauss_ref: ")
grad = line(a1, "gauss_radial: ")
%w[skew_max kurt_max corr_max corr_mean].each do |k|
  failures << "gaussianity: the measured line lacks #{k}" unless g.include?(k + "=")
  failures << "gaussianity: the REFERENCE line lacks #{k} — a bare statistic has no interpretable scale without the same-n N(0,I) floor" unless gr.include?(k + "=")
end
%w[r2_mean r2_var expect_var ref_var].each do |k|
  failures << "gaussianity: the radial line lacks #{k}" unless grad.include?(k + "=")
end
ref_k = gr[/kurt_max=([0-9.eE+-]+)/, 1].to_f
ref_c = gr[/corr_max=([0-9.eE+-]+)/, 1].to_f
failures << "gaussianity: the N(0,I) reference itself reports kurt_max=#{ref_k} — the reference should sit at the sampling-noise floor, so a large value means the reference draw is not standard normal and every comparison against it is void" if ref_k > 0.2
failures << "gaussianity: the N(0,I) reference reports corr_max=#{ref_c} — same problem" if ref_c > 0.2
rm = grad[/ref_mean=([0-9.eE+-]+)/, 1].to_f
rv = grad[/ref_var=([0-9.eE+-]+)/, 1].to_f
d_lat = 8
failures << "gaussianity: the reference radial mean #{rm} is not ~d (#{d_lat}) — the chi-square identity the radial read rests on does not hold for the reference" if (rm - d_lat).abs > 0.5
failures << "gaussianity: the reference radial variance #{rv} is not ~2d (#{2 * d_lat}) — same" if (rv - 2 * d_lat).abs > 1.5
puts failures.length == n0 ?
  "  ok: the Gaussianity read is reported against a same-n N(0,I) reference, and that reference sits at its own noise floor (kurt #{ref_k}, radial mean #{rm} ~ d, var #{rv} ~ 2d)" :
  "  FAIL: gaussianity probe"

# ---- 7e. the STAGE-1 KL arms (toy#168 followup) ----
#
# beta 0 must build NO extra graph — the whole comparison against every
# pre-KL number depends on it. And the learned-sigma arm appends a weight
# to the engine, which is why the decode head is fetched by NAMED index
# now: `nw - 2` would still have been a matrix of the right shape while
# being the wrong matrix (toy#160's suffix-matched-init class of bug).
n0 = failures.length
kl0 = run_dl({ "DL_ARM" => "diff-plain", "DL_STAGE1_KL" => "0.0" })
failures << "stage1 kl: an explicit beta=0 is NOT byte-null against the default — every pre-KL number stops comparing" unless curve(kl0) == curve(plain)
klf = run_dl({ "DL_ARM" => "diff-plain", "DL_STAGE1_KL" => "0.05" })
kll = run_dl({ "DL_ARM" => "diff-plain", "DL_STAGE1_KL" => "0.05", "DL_STAGE1_KL_LEARNED" => "1" })
failures << "stage1 kl: beta>0 did not change stage 1 (the latent line is identical)" if line(klf, "latent: ") == line(plain, "latent: ")
failures << "stage1 kl: learned sigma is identical to fixed sigma at the same beta — the logvar head is not reaching the objective" if line(kll, "latent: ") == line(klf, "latent: ")
failures << "stage1 kl: the fixed arm does not report kl_sigma as a number" unless line(klf, "stage1: ").include?("kl_sigma=0.1") || line(klf, "stage1: ") =~ /kl_sigma=[0-9]/
failures << "stage1 kl: the learned arm does not report kl_sigma=learned" unless line(kll, "stage1: ").include?("kl_sigma=learned")
# The decode head must still be the DECODE head once a weight is appended.
# If it were not, prior-floor (which decodes the prior through that head)
# would stop landing near the unigram floor.
fl3 = run_dl({ "DL_ARM" => "prior-floor", "DL_STAGE1_KL" => "0.05", "DL_STAGE1_KL_LEARNED" => "1" })
bfl = field(fl3, "gen: ", "bpb_gen").to_f
brl = field(fl3, "gen: ", "bpb_real").to_f
failures << "stage1 kl: with the learned-sigma head appended, prior-floor scores #{bfl} against real #{brl} — the decode head fetched by named index is not the decode head" unless bfl > brl + 0.5
puts failures.length == n0 ?
  "  ok: beta=0 is BYTE-NULL, beta>0 moves stage 1, learned sigma differs from fixed at the same beta, and the decode head survives a weight being appended (named indices, not nw-N)" :
  "  FAIL: stage-1 KL"

# ---- 7f. the JOINT probe, and its FLOOR ----
#
# This probe was a MISS in its chosen lens and the gate says so rather
# than quietly keeping a number nobody should read: the real pool's
# lag-1 correlation is ~0.035 against a shuffled floor of ~0.002, i.e.
# the latent sequence carries almost no LINEAR position-to-position
# structure, because the code is CATEGORICAL (the codes for `q` and `u`
# are arbitrary points, so their covariance says nothing about `qu`).
#
# It is kept because the FLOOR is the useful part — it is what makes the
# absence of dynamic range visible instead of inviting someone to read a
# 5x ratio between two numbers that are both nearly zero. So the gate
# asserts the floor is REPORTED and is genuinely near zero; it does not
# assert anything about the ratio.
n0 = failures.length
jl = a1.lines.select { |l| l.start_with?("joint: ") }
jf = a1.lines.find { |l| l.start_with?("joint_floor:") }
failures << "joint: no joint lines on a diffusion arm" if jl.empty?
failures << "joint: no joint_floor line — without the shuffled floor the correlations have no scale and a ratio between two near-zero numbers reads as signal" if jf.nil?
if jf
  fr = jf[/real=([0-9.eE+-]+)/, 1].to_f
  failures << "joint: the shuffled REAL floor is #{fr}, not near zero — pairing random positions must destroy the structure, or the floor is not a floor" if fr > 0.02
end
failures << "joint: the ar-baseline arm emits joint lines, which it has no latents for" if ar.lines.any? { |l| l.start_with?("joint: ") }
puts failures.length == n0 ?
  "  ok: the joint probe reports its position-shuffled FLOOR (the part that makes its own lack of dynamic range legible), and only on arms that have latents" :
  "  FAIL: joint probe"

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
  [{ "DL_LOSS_WEIGHT" => "nope" },            "unknown loss weight"],
  [{ "DL_MINSNR_GAMMA" => "0" },              "min-snr gamma 0"],
  [{ "DL_STAGE1_KL" => "-1" },                "negative KL beta"],
  [{ "DL_STAGE1_KL_LEARNED" => "1" },         "learned sigma with no KL term"],
].each do |env, label|
  _o, st = Open3.capture2e({ "STEPS" => "2" }.merge(CELL).merge(env), RUNNER, chdir: ROOT)
  failures << "fail-loud: #{label} exited 0" if st.success?
end
puts failures.length == n0 ? "  ok: 11 degenerate configs fail loud" : "  FAIL: fail-loud"

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
