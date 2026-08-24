# lib/toy/run/train_diff.rb — Spinel-compiled latent-diffusion training
# runner (-> libexec/toy-train-diff), the toy#156 / DFA-arch T2 lane.
#
# WHAT THIS LANE IS FOR. DiffusionBlocks (Shibata et al. 2025) already
# showed that block-local, gradient-isolated diffusion training matches
# end-to-end BP; DFA specifically on the denoiser is unclaimed. And the
# ticket's HARD CONSTRAINT is the reason it is plausible at all: the
# eps-prediction target has the INPUT's dimensionality, which is small
# for latent / tabular / time-series diffusion and ~2e5 for pixels. So
# this lane is scoped strictly to a low-dim latent, and it is a direct
# test of the output-dim lens toy#152 measured — the feedback matrices
# are [latent_dim, d_hidden] with latent_dim ~ 16.
#
# CPU-ONLY (tao#18). Own compilation unit (landmine #16).
#
# ENV CONTRACT:
#   STEPS / SEED / RUN_DIR / TOY_RUN_ID   — as every other runner
#   DIFF_POLICY      — per-HIDDEN-layer tokens: chain | dfa | frozen
#   DIFF_LAYERS      — hidden layers (default 3)
#   DIFF_HIDDEN      — hidden width (default 128)
#   DIFF_LATENT      — latent dim = THE OUTPUT DIM under test (default 16)
#   DIFF_TIME_FEAT   — Fourier time features (default 8)
#   DIFF_TASK        — mixture (default) | single  [toy_diff_task.rb]
#   DIFF_MODES       — mixture components (default 8)
#   DIFF_SPREAD      — mode separation, in units of the component sd (4.0)
#   DIFF_SCALE       — per-component sd (default 0.5)
#   DIFF_DIFF_STEPS  — DDPM steps T (default 100)
#   DIFF_BETA_LO / DIFF_BETA_HI — linear schedule ends (1e-3 / 0.15,
#                    NOT Ho et al.'s 1e-4/0.02 — those assume T=1000; see
#                    the note at the constants, and the abar_T guard)
#   DIFF_TASK_SEED   — task seed, SEPARATE from SEED (default 7)
#   DIFF_BATCH       — samples per step (default 128)
#   DIFF_EVAL_N      — samples generated + held-out reals scored (512)
#   DIFF_LR / DIFF_WARMUP
#   DIFF_B_SEED / DIFF_B_DIST / DIFF_B_SCALE — the DfaB feedback axes
#   DIFF_ALIGN / DIFF_ALIGN_EVERY — per-step align events
#
# STDOUT (byte-gated): "step <N>: loss=<float>" per step, then
# "val: mse=<m> n=<n>" and "gen: energy=<e> n=<n>".
#
# ── THE METRIC IS GENERATIVE, NOT DENOISING MSE ──
#
# The ticket asks to "compare DFA-denoiser vs BP-denoiser on the
# GENERATIVE metric". Held-out eps-MSE is not that: a denoiser can have
# a respectable MSE and still generate nothing like the data, because
# MSE is dominated by the high-noise timesteps where the Bayes-optimal
# prediction is nearly the input. So the headline number here is the
# ENERGY DISTANCE between `eval_n` ancestrally-sampled points and
# `eval_n` held-out real ones:
#
#     E = 2 mean|X - Y| - mean|X - X'| - mean|Y - Y'|
#
# a proper metric (zero iff the distributions match), cheap in low
# dimension, and deterministic given the seeds. LOWER IS BETTER, which
# inverts the direction of the success bar — see prep/diff_gate.rb.
# val mse rides stdout too, as the secondary that shows the two can
# disagree.
#
# THE SUCCESS BAR (tao#19 item 4) is MANDATORY here as in every lane:
#     positive = all-DFA within the stated gap of all-BP
#                AND provably beating the frozen control,
#                at matched init and matched seed.
#
# Spinel hygiene: hand-built String-concat JSON (no #{}), ENV reads as
# TOP-LEVEL constants, no Struct, while loops.

require_relative "../io/json_builder"
require_relative "../io/json"
require_relative "../io/toy_events"
require_relative "../io/toy_diff_task"
require_relative "../llm/engine/diff_engine"
require_relative "../llm/recipes/diff_denoiser"
require_relative "../llm/adamw"
require_relative "../train/dfa_b"
require_relative "../dev/toy_describe_flow"

STEPS       = (ENV["STEPS"] || "5").to_i
SEED        = (ENV["SEED"]  || "0").to_i
# The run-directory contract. TOY_RUN_DIR is canonical; RUN_DIR is
# the compatibility fallback — the framework's own contract should not
# be named after a client repo. Length-checked, not truthiness-checked:
# "" is truthy in Ruby.
RUN_DIR_NEW = ENV["TOY_RUN_DIR"] || ""
RUN_DIR     = RUN_DIR_NEW.length > 0 ? RUN_DIR_NEW : (ENV["TAO_RUN_DIR"] || "")
RUN_ID      = ENV["TOY_RUN_ID"]  || ""
EVENTS      = RUN_DIR.length > 0 ? (RUN_DIR + "/events.jsonl") : ""

POLICY_S    = ENV["DIFF_POLICY"] || ""
N_LAYERS    = (ENV["DIFF_LAYERS"] || "3").to_i
D_HIDDEN    = (ENV["DIFF_HIDDEN"] || "128").to_i
LATENT      = (ENV["DIFF_LATENT"] || "16").to_i
N_TIME      = (ENV["DIFF_TIME_FEAT"] || "8").to_i
TASK_S      = ENV["DIFF_TASK"] || ""
N_MODES     = (ENV["DIFF_MODES"] || "8").to_i
SPREAD_S    = ENV["DIFF_SPREAD"] || ""
SCALE_S     = ENV["DIFF_SCALE"] || ""
DIFF_STEPS  = (ENV["DIFF_DIFF_STEPS"] || "100").to_i
BETA_LO_S   = ENV["DIFF_BETA_LO"] || ""
BETA_HI_S   = ENV["DIFF_BETA_HI"] || ""
TASK_SEED   = (ENV["DIFF_TASK_SEED"] || "7").to_i
BATCH       = (ENV["DIFF_BATCH"] || "128").to_i
EVAL_N      = (ENV["DIFF_EVAL_N"] || "512").to_i
LR_S        = ENV["DIFF_LR"] || ""
WARMUP      = (ENV["DIFF_WARMUP"] || "0").to_i
B_SEED      = (ENV["DIFF_B_SEED"] || "1234").to_i
B_DIST_S    = ENV["DIFF_B_DIST"]  || ""
B_SCALE_S   = ENV["DIFF_B_SCALE"] || ""
ALIGN_ON    = (ENV["DIFF_ALIGN"] || "") == "1"
ae_raw      = (ENV["DIFF_ALIGN_EVERY"] || "1").to_i
ALIGN_EVERY = ae_raw < 1 ? 1 : ae_raw

SPREAD  = SPREAD_S.length  > 0 ? SPREAD_S.to_f  : 4.0
SCALE   = SCALE_S.length   > 0 ? SCALE_S.to_f   : 0.5
# The schedule ends are NOT Ho et al.'s (1e-4 .. 0.02): those assume
# T = 1000. At T = 100 they leave abar_T = 0.60, i.e. the forward
# process never destroys the signal — and then the ancestral sampler,
# which starts from pure N(0, I), starts OUT OF DISTRIBUTION and the
# generative metric measures that mismatch rather than the model.
# MEASURED, not guessed: at the old defaults BP scored the BEST
# denoising MSE (.684) and the WORST energy distance (29.1, against
# 4.95 for an untrained net). 1e-3 .. 0.15 over 100 steps gives
# abar_T = 3.5e-4, and the guard below refuses to run if a future edit
# breaks that again.
BETA_LO = BETA_LO_S.length > 0 ? BETA_LO_S.to_f : 0.001
BETA_HI = BETA_HI_S.length > 0 ? BETA_HI_S.to_f : 0.15
LR      = LR_S.length      > 0 ? LR_S.to_f      : 0.003

# ---- fail loud on every out-of-range shape (never-mask). ----
if STEPS < 1
  puts "toy-train-diff: STEPS must be >= 1, got " + STEPS.to_s
  exit 1
end
if N_LAYERS < 1
  puts "toy-train-diff: DIFF_LAYERS must be >= 1, got " + N_LAYERS.to_s
  exit 1
end
if D_HIDDEN < 1
  puts "toy-train-diff: DIFF_HIDDEN must be >= 1, got " + D_HIDDEN.to_s
  exit 1
end
if LATENT < 1
  puts "toy-train-diff: DIFF_LATENT must be >= 1, got " + LATENT.to_s
  exit 1
end
if N_TIME < 1
  puts "toy-train-diff: DIFF_TIME_FEAT must be >= 1, got " + N_TIME.to_s
  exit 1
end
if DIFF_STEPS < 2
  puts "toy-train-diff: DIFF_DIFF_STEPS must be >= 2, got " + DIFF_STEPS.to_s
  exit 1
end
if BETA_LO <= 0.0 || BETA_HI <= BETA_LO || BETA_HI >= 1.0
  puts "toy-train-diff: need 0 < DIFF_BETA_LO < DIFF_BETA_HI < 1, got " +
       BETA_LO.to_s + " / " + BETA_HI.to_s
  exit 1
end
if BATCH < 1
  puts "toy-train-diff: DIFF_BATCH must be >= 1, got " + BATCH.to_s
  exit 1
end
if EVAL_N < 8
  puts "toy-train-diff: DIFF_EVAL_N must be >= 8, got " + EVAL_N.to_s
  exit 1
end
if TASK_S.length > 0 && TASK_S != "mixture" && TASK_S != "single"
  puts "toy-train-diff: DIFF_TASK " + TASK_S + " unsupported (mixture|single)"
  exit 1
end
if N_MODES < 1
  puts "toy-train-diff: DIFF_MODES must be >= 1, got " + N_MODES.to_s
  exit 1
end
# The sampler runs the SAME graph as training, so its batch is the
# training batch. Rather than silently generating a different number of
# points than asked for, require the two to line up.
if EVAL_N % BATCH != 0
  puts "toy-train-diff: DIFF_EVAL_N (" + EVAL_N.to_s + ") must be a multiple" +
       " of DIFF_BATCH (" + BATCH.to_s + ") — the ancestral sampler reuses the" +
       " training graph, so it generates whole batches at a time"
  exit 1
end

TASK_KIND = TASK_S == "single" ? DiffTask::KIND_SINGLE : DiffTask::KIND_MIXTURE
D_IN      = LATENT + N_TIME

def parse_diff_policy(pol_s, n_layers)
  policy = [0]; policy.pop
  parts = pol_s.split(",")
  if parts.length > n_layers
    puts "toy-train-diff: DIFF_POLICY names " + parts.length.to_s +
         " layers but DIFF_LAYERS=" + n_layers.to_s +
         " — a policy token for a layer that does not exist would silently do nothing"
    exit 1
  end
  i = 0
  while i < n_layers
    m = 0
    if i < parts.length
      tk = parts[i]
      if tk == "dfa"
        m = 1
      elsif tk == "frozen"
        m = 2
      elsif tk != "chain" && tk.length > 0
        puts "toy-train-diff: unknown DIFF_POLICY token " + tk + " (chain|dfa|frozen)"
        exit 1
      end
    end
    policy.push(m)
    i = i + 1
  end
  policy
end

def dist_code(s)
  if s == "uniform"
    return Toy::Train::DfaB::DIST_UNIFORM
  end
  if s == "rademacher"
    return Toy::Train::DfaB::DIST_RADEMACHER
  end
  Toy::Train::DfaB::DIST_GAUSSIAN
end

def scale_code(s)
  if s == "glorot"
    return Toy::Train::DfaB::SCALE_GLOROT
  end
  if s.length >= 6 && s[0, 6] == "fixed:"
    return Toy::Train::DfaB::SCALE_FIXED
  end
  Toy::Train::DfaB::SCALE_INV_SQRT_FAN
end

def scale_sigma(s)
  if s.length >= 6 && s[0, 6] == "fixed:"
    return s[6, s.length - 6].to_f
  end
  0.0
end

def num_or_null(x)
  d = x - x
  if d == 0.0
    x.to_s
  else
    "null"
  end
end

def cosv3(a, b, n)
  dot = 0.0; na = 0.0; nb = 0.0
  i = 0
  while i < n
    dot = dot + a[i] * b[i]
    na = na + a[i] * a[i]
    nb = nb + b[i] * b[i]
    i = i + 1
  end
  sa = Math.sqrt(na)
  sb = Math.sqrt(nb)
  d = sa * sb
  c = 0.0
  if d > 0.0
    c = dot / d
  end
  [c, sa, sb]
end

# Mean pairwise Euclidean distance between two flat sample sets.
def mean_pair_dist(a, na, b, nb, dim, same)
  tot = 0.0
  cnt = 0
  i = 0
  while i < na
    j = same ? i + 1 : 0
    while j < nb
      s = 0.0
      ai = i * dim
      bj = j * dim
      k = 0
      while k < dim
        d = a[ai + k] - b[bj + k]
        s = s + d * d
        k = k + 1
      end
      tot = tot + Math.sqrt(s)
      cnt = cnt + 1
      j = j + 1
    end
    i = i + 1
  end
  if cnt == 0
    return 0.0
  end
  tot / cnt.to_f
end

# The ENERGY DISTANCE. Zero iff the two sample sets come from the same
# distribution; strictly positive otherwise. Lower is better.
def energy_distance(gen, real, n, dim)
  cross = mean_pair_dist(gen, n, real, n, dim, false)
  gg    = mean_pair_dist(gen, n, gen, n, dim, true)
  rr    = mean_pair_dist(real, n, real, n, dim, true)
  2.0 * cross - gg - rr
end

POLICY = parse_diff_policy(POLICY_S, N_LAYERS)

recipe = Toy::LLM::Recipes::DiffDenoiser.new
recipe.realize!(D_IN, LATENT, D_HIDDEN, N_LAYERS, BATCH, SEED, 1.0,
                POLICY, B_SEED, dist_code(B_DIST_S),
                scale_code(B_SCALE_S), scale_sigma(B_SCALE_S))
ToyDescribeFlow.emit_flow_json(RUN_DIR, recipe.dn_cache.sess)

task = DiffTask.new(TASK_KIND, LATENT, N_MODES, SPREAD, SCALE,
                    DIFF_STEPS, BETA_LO, BETA_HI, TASK_SEED)
# THE SCHEDULE HAS TO ACTUALLY DESTROY THE SIGNAL. The sampler starts
# from pure N(0, I); if abar_T is not small then x_T under the forward
# process still carries the data and the sampler begins out of
# distribution, so the generative metric scores that mismatch instead of
# the model. This bit once already (see the BETA_LO/BETA_HI note above),
# and it is silent: the training loss looks fine throughout.
ABAR_T = task.dt_abar[DIFF_STEPS - 1]
if ABAR_T > 0.01
  puts "toy-train-diff: the noise schedule leaves abar_T=" + ABAR_T.to_s +
       " (> 0.01), so the forward process never reaches pure noise and the" +
       " ancestral sampler would start OUT OF DISTRIBUTION. Raise" +
       " DIFF_BETA_HI or DIFF_DIFF_STEPS. (Ho et al.'s 1e-4..0.02 assumes" +
       " T=1000; this lane runs T=" + DIFF_STEPS.to_s + ".)"
  exit 1
end
task.reset_stream!(TASK_SEED + 1)

x_flat = Array.new(BATCH * D_IN, 0.0)
x0     = Array.new(BATCH * LATENT, 0.0)
xt     = Array.new(BATCH * LATENT, 0.0)
epsb   = Array.new(BATCH * LATENT, 0.0)
ts     = Array.new(BATCH, 0)
m_eps  = Mat.new(BATCH, LATENT)
pred   = Array.new(BATCH * LATENT, 0.0)

adamw = Toy::AdamW.for_from_scratch
adamw.lr = LR

# ---- the held-out real sample set, MATERIALISED FIRST ----
# Same discipline as toy#152: drawn from the HEAD of the stream, then
# training continues from where it stopped, so train/eval disjointness
# holds by construction rather than by two seeds happening to miss each
# other in the same LCG cycle.
real = Array.new(EVAL_N * LATENT, 0.0)
rb = 0
while rb < EVAL_N / BATCH
  task.sample_x0!(BATCH, x0)
  ri = 0
  while ri < BATCH * LATENT
    real[rb * BATCH * LATENT + ri] = x0[ri]
    ri = ri + 1
  end
  rb = rb + 1
end

# ---- run_start (FILE only). ----
if EVENTS.length > 0
  rc = TinyNN.tnn_events_open(EVENTS)
  if rc == 0
    rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
    rs = Toy::Json::Builder.new
    rs.add_str("kind", "run_start")
    rs.add_str("schema", "toy/v1")
    rs.add_num("t", TinyNN.tnn_events_now_seconds)
    rs.add_str("started_at", TinyNN.tnn_events_iso8601_now)
    rs.add_str("run_id", rid)
    rs.add_str("phase", "train")
    rs.add_str("name", "diff")
    Toy::Events.add_provenance(rs,
      TinyNN.tnn_provenance_host_name, TinyNN.tnn_provenance_host_os,
      TinyNN.tnn_provenance_host_arch,
      TinyNN.tnn_backend_name(recipe.dn_cache.sess))
    model = Toy::Json::Builder.new
    model.add_str("arch", "diff")
    model.add_str("name", "latent-denoiser")
    model.add_num("d_in",       D_IN)
    model.add_num("d_hidden",   D_HIDDEN)
    model.add_num("n_layers",   N_LAYERS)
    # The OUTPUT DIM under test. Named `latent_dim` and not
    # `num_classes`: this lane regresses epsilon, it does not classify,
    # and a consumer that read it as a class count would compare it to
    # the wrong lanes.
    model.add_num("latent_dim", LATENT)
    model.add_num("time_feat",  N_TIME)
    model.add_str("objective",  "eps_prediction")
    model.add_str("act",        "silu")
    rs.add_obj("model", model)
    cost = Toy::Json::Builder.new
    cost.add_num("total_params",  recipe.dn_cache.param_count)
    cost.add_num("active_params", recipe.dn_cache.param_count)
    cost.add_num("flops_per_token", 2 * recipe.dn_cache.param_count)
    # Sampling is T forward passes per generated batch — invisible in a
    # params-only cost and the dominant cost of the generative metric.
    cost.add_num("sampling_forwards", DIFF_STEPS * (EVAL_N / BATCH))
    rs.add_obj("cost", cost)
    config = Toy::Json::Builder.new
    config.add_num("steps",       STEPS)
    config.add_num("seed",        SEED)
    config.add_num("batch",       BATCH)
    config.add_num("eval_n",      EVAL_N)
    config.add_num("diff_steps",  DIFF_STEPS)
    config.add_str("task",        TASK_KIND == DiffTask::KIND_SINGLE ? "single" : "mixture")
    config.add_num("modes",       task.dt_modes)
    config.add_raw("spread",      SPREAD.to_s)
    config.add_raw("scale",       SCALE.to_s)
    config.add_raw("beta_lo",     BETA_LO.to_s)
    config.add_raw("beta_hi",     BETA_HI.to_s)
    config.add_raw("abar_final",  ABAR_T.to_s)
    config.add_num("task_seed",   TASK_SEED)
    config.add_raw("lr",          LR.to_s)
    config.add_num("warmup",      WARMUP)
    rs.add_obj("config", config)
    dfa = Toy::Json::Builder.new
    dfa.add_raw("policy",  Toy::Json.from_int_array(POLICY))
    dfa.add_num("b_seed",  B_SEED)
    dfa.add_str("b_dist",  B_DIST_S.length > 0 ? B_DIST_S : "gaussian")
    dfa.add_str("b_scale", B_SCALE_S.length > 0 ? B_SCALE_S : "inv_sqrt_fan")
    dfa.add_raw("b_sigma", scale_sigma(B_SCALE_S).to_s)
    dfa.add_num("dfa_wired", recipe.dn_cache.df_dfa_wired)
    dfa.add_num("frozen",    recipe.dn_cache.df_frozen_count)
    rs.add_obj("dfa", dfa)
    TinyNN.tnn_events_emit(rs.dump)
  else
    puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
  end
end

# ---- align telemetry buffers ----
n_align = recipe.dn_cache.df_align_grads.length
abuf = [0.0]; abuf.pop
gbuf = [0.0]; gbuf.pop
if ALIGN_ON && n_align > 0
  nmax = 0
  ai0 = 0
  while ai0 < n_align
    nw = TinyNN.tnn_tensor_nelements(recipe.dn_cache.df_align_grads[ai0])
    if nw > nmax; nmax = nw; end
    ai0 = ai0 + 1
  end
  z = 0
  while z < nmax
    abuf.push(0.0); gbuf.push(0.0)
    z = z + 1
  end
end

# ---- training loop. ----
final_loss = 0.0
step = 0
while step < STEPS
  step_wall_start = TinyNN.tnn_events_now_seconds
  if WARMUP > 0 && step < WARMUP
    adamw.lr = LR * ((step + 1).to_f / WARMUP.to_f)
  else
    adamw.lr = LR
  end
  m_hp = adamw.hp(step)

  task.corrupt_batch!(BATCH, x0, xt, epsb, ts)
  bi = 0
  while bi < BATCH
    xb = bi * LATENT
    ib = bi * D_IN
    j = 0
    while j < LATENT
      x_flat[ib + j] = xt[xb + j]
      m_eps.flat[xb + j] = epsb[xb + j]
      j = j + 1
    end
    task.time_features(ts[bi], N_TIME, x_flat, ib + LATENT)
    bi = bi + 1
  end

  loss = recipe.step!(x_flat, m_eps, m_hp, step == 0)
  final_loss = loss
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s

  if EVENTS.length > 0
    step_wall_us = ((TinyNN.tnn_events_now_seconds - step_wall_start) * 1.0e6).to_i
    es = Toy::Json::Builder.new
    es.add_str("kind",  "step")
    es.add_str("phase", "train")
    es.add_num("t",       TinyNN.tnn_events_now_seconds)
    es.add_num("step",    step + 1)
    es.add_raw("loss",    num_or_null(loss))
    es.add_raw("lr",      adamw.lr.to_s)
    es.add_num("samples", BATCH)
    es.add_num("wall_us", step_wall_us)
    TinyNN.tnn_events_emit(es.dump)
  end

  if ALIGN_ON && EVENTS.length > 0 && n_align > 0 && (step % ALIGN_EVERY) == 0
    ai = 0
    while ai < n_align
      nw = TinyNN.tnn_tensor_nelements(recipe.dn_cache.df_align_grads[ai])
      rc_g = TinyNN.tnn_download_to_f64_array(recipe.dn_cache.sess,
        recipe.dn_cache.df_align_grads[ai], gbuf, nw)
      rc_a = TinyNN.tnn_download_to_f64_array(recipe.dn_cache.sess,
        recipe.dn_cache.df_align_accs[ai], abuf, nw)
      if rc_g != 0 || rc_a != 0
        puts "align download failed: step=" + (step + 1).to_s +
             " wname=" + recipe.dn_cache.df_align_wnames[ai] +
             " rc_g=" + rc_g.to_s + " rc_a=" + rc_a.to_s
      end
      ae = Toy::Json::Builder.new
      ae.add_str("kind",  "align")
      ae.add_str("phase", "train")
      ae.add_num("t",     TinyNN.tnn_events_now_seconds)
      ae.add_num("step",  step + 1)
      ae.add_num("li",    recipe.dn_cache.df_align_lis[ai])
      ae.add_num("wi",    0)
      ae.add_str("wname", recipe.dn_cache.df_align_wnames[ai])
      cnn = cosv3(gbuf, abuf, nw)
      ae.add_raw("cos",      num_or_null(cnn[0]))
      ae.add_raw("dfa_norm", num_or_null(cnn[1]))
      ae.add_raw("bp_norm",  num_or_null(cnn[2]))
      TinyNN.tnn_events_emit(ae.dump)
      ai = ai + 1
    end
  end
  step = step + 1
end

# ---- evaluation. ----
#
# MIRROR RULE (the toy#139/#146 class of bug, in its generative form):
# every hp vector the graph can reach must be zeroed, or the eval and
# the sampler silently keep training. This lane has exactly ONE.
val_hp = Mat.new(1, 7)
val_hp.flat[0] = 0.0
val_hp.flat[1] = adamw.beta1
val_hp.flat[2] = adamw.beta2
val_hp.flat[3] = adamw.eps
val_hp.flat[4] = 0.0
val_hp.flat[5] = adamw.beta1
val_hp.flat[6] = adamw.beta2

# (a) held-out denoising MSE, over freshly drawn corruptions.
val_batches = EVAL_N / BATCH
mse_sum = 0.0
vb = 0
while vb < val_batches
  task.corrupt_batch!(BATCH, x0, xt, epsb, ts)
  bi2 = 0
  while bi2 < BATCH
    xb = bi2 * LATENT
    ib = bi2 * D_IN
    j2 = 0
    while j2 < LATENT
      x_flat[ib + j2] = xt[xb + j2]
      m_eps.flat[xb + j2] = epsb[xb + j2]
      j2 = j2 + 1
    end
    task.time_features(ts[bi2], N_TIME, x_flat, ib + LATENT)
    bi2 = bi2 + 1
  end
  mse_sum = mse_sum + recipe.step!(x_flat, m_eps, val_hp, false)
  vb = vb + 1
end
val_mse = mse_sum / val_batches.to_f
puts "val: mse=" + val_mse.to_s + " n=" + (val_batches * BATCH).to_s

# (b) THE GENERATIVE METRIC. Ancestral DDPM sampling from pure noise,
# then the energy distance against the held-out reals.
gen = Array.new(EVAL_N * LATENT, 0.0)
gb = 0
while gb < val_batches
  # x_T ~ N(0, I), from the SAME deterministic stream as everything else.
  gi = 0
  while gi < BATCH * LATENT
    xt[gi] = task.gauss
    gi = gi + 1
  end
  s = DIFF_STEPS - 1
  while s >= 0
    bi3 = 0
    while bi3 < BATCH
      xb = bi3 * LATENT
      ib = bi3 * D_IN
      j3 = 0
      while j3 < LATENT
        x_flat[ib + j3] = xt[xb + j3]
        j3 = j3 + 1
      end
      task.time_features(s, N_TIME, x_flat, ib + LATENT)
      bi3 = bi3 + 1
    end
    rcp = recipe.denoise!(x_flat, val_hp, pred)
    if rcp != 0
      puts "toy-train-diff: denoise download failed: rc=" + rcp.to_s
      exit 1
    end
    # DDPM ancestral step:
    #   mu   = (x_t - beta_t/sqrt(1-abar_t) * eps^) / sqrt(alpha_t)
    #   x_{t-1} = mu + sqrt(beta_t) z,   z ~ N(0,I) for t > 0
    beta  = task.dt_beta[s]
    alpha = task.dt_alpha[s]
    sqa   = Math.sqrt(alpha)
    sqb   = Math.sqrt(1.0 - task.dt_abar[s])
    sig   = s > 0 ? Math.sqrt(beta) : 0.0
    k4 = 0
    while k4 < BATCH * LATENT
      mu = (xt[k4] - (beta / sqb) * pred[k4]) / sqa
      if sig > 0.0
        mu = mu + sig * task.gauss
      end
      xt[k4] = mu
      k4 = k4 + 1
    end
    s = s - 1
  end
  gc = 0
  while gc < BATCH * LATENT
    gen[gb * BATCH * LATENT + gc] = xt[gc]
    gc = gc + 1
  end
  gb = gb + 1
end
energy = energy_distance(gen, real, EVAL_N, LATENT)
puts "gen: energy=" + energy.to_s + " n=" + EVAL_N.to_s

if EVENTS.length > 0 && TinyNN.tnn_events_active == 1
  ev = Toy::Json::Builder.new
  ev.add_str("kind",  "eval")
  ev.add_str("phase", "eval")
  ev.add_num("t",     TinyNN.tnn_events_now_seconds)
  ev.add_str("name",  "gen")
  ev.add_num("n",     EVAL_N)
  # `energy` is the HEADLINE and LOWER IS BETTER — the opposite
  # direction from every accuracy-scored lane in this program. Named so
  # a consumer cannot mistake it for one.
  ev.add_raw("energy_distance", num_or_null(energy))
  ev.add_raw("val_mse",         num_or_null(val_mse))
  TinyNN.tnn_events_emit(ev.dump)

  re = Toy::Json::Builder.new
  re.add_str("kind", "run_end")
  re.add_num("t",          TinyNN.tnn_events_now_seconds)
  re.add_str("ended_at",   TinyNN.tnn_events_iso8601_now)
  re.add_str("reason",     "completed")
  re.add_num("final_step", STEPS)
  re.add_raw("final_loss", num_or_null(final_loss))
  re.add_raw("energy_distance", num_or_null(energy))
  re.add_str("checkpoint", "none")
  re.add_raw("exit_code",  "0")
  TinyNN.tnn_events_emit(re.dump)
  TinyNN.tnn_events_close
end
