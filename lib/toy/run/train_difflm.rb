# lib/toy/run/train_difflm.rb — Spinel-compiled latent-diffusion byte-LM
# runner (-> libexec/toy-train-difflm), the toy#166 / capstone P1b lane.
#
# THE QUESTION. P1a (toy#165) showed a small per-token latent can carry a
# byte DECODABLY UNDER NOISE, and pinned the operating point at d=8.
# P1b asks whether a DIFFUSION model can GENERATE coherent text through
# that latent — clearing a prior-decode floor and approaching a plain
# autoregressive byte-LM. If it cannot, the capstone is capped by
# GENERATION quality and no credit rule rescues it. ALL BP; DFA is P1c.
#
# ── THE RUNNER IS SEQUENCED, AND THAT IS NOT A STYLE CHOICE ──
#
# The ticket describes P1b as composing three models in one process. It
# cannot: tnn_session_new_on() calls ggml_backend_sched_reset() on the
# scheduler EVERY session on that backend shares. Measured
# (prep/smokes/smoke_two_sessions.rb): after a second session is created,
# recomputing the first returns 1.049 where it returned 5.940 — no
# crash, no warning, just a wrong number. Exactly the failure mode this
# program keeps meeting (toy#133's silently-zero inputs, toy#160's
# suffix-matched init): confident garbage.
#
# So models are built ONE AT A TIME, and anything needed across a
# boundary leaves as a plain Ruby array before the next session exists:
#
#   SESSION A  the autoencoder (toy#165's AeEngine, unchanged).
#              Train -> encode the train span into a LATENT POOL ->
#              encode the held-out windows -> compute the standardisation
#              stats -> download the decode head (W, b). Then done.
#   SESSION B  the arm. ar-baseline: an AR byte-LM, sampled. diff-*: the
#              denoiser, trained on the pooled latents, sampled, and
#              measured for sampler residual. prior-floor: no session at
#              all — its latents come straight from the prior.
#              Decoding uses the Ruby-side (W, b): the decode head is one
#              [256, d] matmul per position, so it does not need a graph.
#   SESSION C  the JUDGE, an AR byte-LM on a DIFFERENT SEED. Every arm's
#              samples are scored under it, including the ar-baseline's
#              — a model scores its own samples as unusually likely, so
#              scoring the ceiling under itself would hand it an
#              unquantifiable advantage and the whole "competitive" bar
#              is stated against that anchor.
#
# ── THE LATENT IS STANDARDISED, AND THAT IS THE CRUX ──
#
# The P1a autoencoder's latent is UNREGULARISED — its geometry is
# whatever cross-entropy produced, and its per-dim std moves by an order
# of magnitude with training alone. A diffusion sampler starts at
# N(0, I). If the aggregate posterior is not close to that, the sampler
# begins OFF-MANIFOLD and the samples are garbage for a reason that has
# nothing to do with d — P1b would report a representation verdict that
# was really a normalisation bug. That is toy#156's landmine in a new
# place (Ho et al.'s betas at T=100 left abar_T=0.60, the sampler started
# out of distribution, and it was SILENT in the loss).
#
# So the pooled latents are standardised per-dim (the LDM/SD scaling
# trick): deterministic, no new loss term, no hyper-parameter to tune,
# and CHECKABLE — the standardised moments are asserted, not assumed. A
# KL term is the escalation if this is not enough, not the default.
#
# ENV CONTRACT:
#   STEPS / SEED / TAO_RUN_DIR / TOY_RUN_ID
#   DL_ARM        — ar-baseline | diff-selfcond | diff-plain | prior-floor
#   DL_TEXT       — byte pack prefix (prep/fetch_text.rb)
#   DL_LATENT     — the operating d (default 8, P1a's pinned point)
#   DL_CONTEXT    — window length T (default 256)
#   DL_AE_STEPS   — stage-1 steps (default 2000)
#   DL_D_MODEL / DL_BLOCKS / DL_HEADS / DL_D_FF   — encoder + denoiser
#   DL_AR_D_MODEL / DL_AR_BLOCKS                  — the AR arm and judge
#   DL_TSTEPS     — diffusion steps (default 100)
#   DL_BETA_LO / DL_BETA_HI                       — the schedule
#   DL_GEN_BYTES  — bytes generated per arm (default 16384)
#   DL_JUDGE_STEPS— judge training steps (default 3000)
#   DL_LR / DL_WARMUP / DL_TASK_SEED / DL_NOISE_SEED
#
# STDOUT (byte-gated): "step <N>: loss=<f>" for the arm's own training,
# then "stage1:", "latent:", "arm:", "gen:", "judge:", "ngram:",
# "resid:" (diffusion arms) and "graph:".
#
# CPU-only (tao#18). Spinel hygiene: hand-built JSON, ENV as top-level
# constants, no Struct, while loops, no #{}.

require_relative "../io/json_builder"
require_relative "../io/json"
require_relative "../io/toy_events"
require_relative "../io/toy_ae_task"
require_relative "../llm/engine/ae_engine"
require_relative "../llm/engine/ar_engine"
require_relative "../llm/engine/difflm_engine"
require_relative "../llm/recipes/ae_auto"
require_relative "../llm/adamw"
require_relative "../dev/toy_describe_flow"

STEPS       = (ENV["STEPS"] || "5").to_i
SEED        = (ENV["SEED"]  || "0").to_i
TAO_RUN_DIR = ENV["TAO_RUN_DIR"] || ""
RUN_ID      = ENV["TOY_RUN_ID"]  || ""
EVENTS      = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/events.jsonl") : ""

ARM         = ENV["DL_ARM"]  || "diff-selfcond"
TEXT        = ENV["DL_TEXT"] || ""
D_LATENT    = (ENV["DL_LATENT"]  || "8").to_i
CONTEXT     = (ENV["DL_CONTEXT"] || "256").to_i
AE_STEPS    = (ENV["DL_AE_STEPS"] || "2000").to_i
D_MODEL     = (ENV["DL_D_MODEL"] || "128").to_i
N_BLOCKS    = (ENV["DL_BLOCKS"]  || "2").to_i
N_HEADS     = (ENV["DL_HEADS"]   || "4").to_i
D_FF        = (ENV["DL_D_FF"]    || "256").to_i
AR_D_MODEL  = (ENV["DL_AR_D_MODEL"] || "128").to_i
AR_BLOCKS   = (ENV["DL_AR_BLOCKS"]  || "2").to_i
TSTEPS      = (ENV["DL_TSTEPS"] || "100").to_i
BETA_LO_S   = ENV["DL_BETA_LO"] || ""
BETA_HI_S   = ENV["DL_BETA_HI"] || ""
GEN_BYTES   = (ENV["DL_GEN_BYTES"] || "16384").to_i
JUDGE_STEPS = (ENV["DL_JUDGE_STEPS"] || "3000").to_i
LR_S        = ENV["DL_LR"] || ""
WARMUP_S    = ENV["DL_WARMUP"] || ""
TASK_SEED   = (ENV["DL_TASK_SEED"] || "7").to_i
NOISE_SEED  = (ENV["DL_NOISE_SEED"] || "4242").to_i
VAL_WINDOWS = (ENV["DL_VAL_WINDOWS"] || "16").to_i

LR      = LR_S.length > 0 ? LR_S.to_f : 0.001
WARMUP  = WARMUP_S.length > 0 ? WARMUP_S.to_i : 0
BETA_LO = BETA_LO_S.length > 0 ? BETA_LO_S.to_f : 0.001
BETA_HI = BETA_HI_S.length > 0 ? BETA_HI_S.to_f : 0.20
VOCAB   = 256
VAL_FRAC_PCT = 10

ARM_AR    = 0
ARM_SELF  = 1
ARM_PLAIN = 2
ARM_FLOOR = 3

def arm_code(s)
  if s == "ar-baseline"
    return 0
  end
  if s == "diff-selfcond"
    return 1
  end
  if s == "diff-plain"
    return 2
  end
  if s == "prior-floor"
    return 3
  end
  -1
end
ARM_C = arm_code(ARM)

if ARM_C < 0
  puts "toy-train-difflm: DL_ARM " + ARM +
       " unsupported (ar-baseline|diff-selfcond|diff-plain|prior-floor)"
  exit 1
end
if STEPS < 1
  puts "toy-train-difflm: STEPS must be >= 1, got " + STEPS.to_s
  exit 1
end
if TEXT.length == 0
  puts "toy-train-difflm: DL_TEXT is required (prep/fetch_text.rb)"
  exit 1
end
if D_LATENT < 1 || D_LATENT >= D_MODEL
  puts "toy-train-difflm: DL_LATENT must be >= 1 and < DL_D_MODEL, got " +
       D_LATENT.to_s + " with d_model " + D_MODEL.to_s
  exit 1
end
if CONTEXT < 8
  puts "toy-train-difflm: DL_CONTEXT must be >= 8, got " + CONTEXT.to_s
  exit 1
end
if D_MODEL % N_HEADS != 0 || AR_D_MODEL % N_HEADS != 0
  puts "toy-train-difflm: d_model must be a multiple of DL_HEADS"
  exit 1
end
if TSTEPS < 2
  puts "toy-train-difflm: DL_TSTEPS must be >= 2, got " + TSTEPS.to_s
  exit 1
end
if GEN_BYTES < CONTEXT
  puts "toy-train-difflm: DL_GEN_BYTES must be >= DL_CONTEXT, got " +
       GEN_BYTES.to_s
  exit 1
end
if JUDGE_STEPS < 1
  puts "toy-train-difflm: DL_JUDGE_STEPS must be >= 1 — every arm is" +
       " scored under the judge, so a judge that never trained would" +
       " make every arm look identical"
  exit 1
end

# ---- small helpers (top-level defs must precede use) ----

def num_or_null(x)
  d = x - x
  if d == 0.0
    x.to_s
  else
    "null"
  end
end

def float_array_json(a)
  out = "["
  i = 0
  while i < a.length
    if i > 0
      out = out + ","
    end
    out = out + a[i].to_s
    i = i + 1
  end
  out + "]"
end

def lcg_state(seed)
  s = ((seed + 104729) * 2654435761) % 2147483647
  if s <= 0
    s = seed + 104729
  end
  w = 0
  while w < 8
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    w = w + 1
  end
  [s]
end

def lcg_u01(st)
  s = st[0]
  s = (s * 1103515245 + 12345) & 0x7FFFFFFF
  st[0] = s
  (s.to_f + 1.0) / 2147483648.0
end

def lcg_gauss(st)
  u1 = lcg_u01(st)
  u2 = lcg_u01(st)
  if u1 < 1.0e-12
    u1 = 1.0e-12
  end
  Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
end

# argmax over a [VOCAB] logit slice starting at base
def argmax_at(buf, base, n)
  best = 0
  bv = buf[base]
  c = 1
  while c < n
    if buf[base + c] > bv
      bv = buf[base + c]
      best = c
    end
    c = c + 1
  end
  best
end

# Sample from a logit slice at temperature 1.0. The AR arm MUST sample
# rather than argmax: greedy decoding from an AR LM degenerates into
# repetition, which would make the yardstick arm look worse than it is
# and inflate every "closer to AR than to the floor" comparison.
def sample_at(buf, base, n, st)
  mx = buf[base]
  c = 1
  while c < n
    if buf[base + c] > mx
      mx = buf[base + c]
    end
    c = c + 1
  end
  tot = 0.0
  probs = Array.new(n, 0.0)
  c2 = 0
  while c2 < n
    e = Math.exp(buf[base + c2] - mx)
    probs[c2] = e
    tot = tot + e
    c2 = c2 + 1
  end
  r = lcg_u01(st) * tot
  acc = 0.0
  c3 = 0
  while c3 < n
    acc = acc + probs[c3]
    if acc >= r
      return c3
    end
    c3 = c3 + 1
  end
  n - 1
end

task = AeTask.new(CONTEXT, TASK_SEED)
if task.load_pack!(TEXT) != 0
  exit 1
end
N_TOKENS = task.at_n_tokens
SPLIT_AT = N_TOKENS - (N_TOKENS * VAL_FRAC_PCT) / 100
TRAIN_HI = SPLIT_AT - CONTEXT
VAL_HI   = N_TOKENS - CONTEXT
if TRAIN_HI < 1 || VAL_HI < SPLIT_AT
  puts "toy-train-difflm: corpus too short to split"
  exit 1
end

# The held-out REAL bytes: the anchor every metric is read against. A
# contiguous slice of the val span, so the n-gram statistics are those of
# real text and not of a shuffle.
REAL_N = GEN_BYTES < (N_TOKENS - SPLIT_AT) ? GEN_BYTES : (N_TOKENS - SPLIT_AT)
real_bytes = Array.new(REAL_N, 0)
rb = 0
while rb < REAL_N
  real_bytes[rb] = task.at_tokens[SPLIT_AT + rb]
  rb = rb + 1
end

adamw = Toy::AdamW.for_from_scratch
adamw.lr = LR

# The NO-OP hp (toy#165): lr=0 freezes the weights but ggml's adamw still
# updates m and v unless the betas are 1.0. Used for every
# prediction-only forward — the self-conditioning pre-pass and the
# sampler — so none of them perturbs training.
noop_hp = Mat.new(1, 7)
noop_hp.flat[0] = 0.0
noop_hp.flat[1] = 1.0
noop_hp.flat[2] = 1.0
noop_hp.flat[3] = adamw.eps
noop_hp.flat[4] = 0.0
noop_hp.flat[5] = 1.0
noop_hp.flat[6] = 1.0

gen_bytes = Array.new(GEN_BYTES, 0)
head_w = [0.0]; head_w.pop
head_b = [0.0]; head_b.pop
lat_mu = Array.new(D_LATENT, 0.0)
lat_sd = Array.new(D_LATENT, 1.0)
POOL_N = TRAIN_HI / CONTEXT
ae_params = 0
arm_params = 0
resid_t = [0]; resid_t.pop
resid_v = [0.0]; resid_v.pop
# Declared at TOP LEVEL, not inside the diff-arm branch, because the
# events block below reads them for EVERY arm. Declaring them where they
# are filled left prior-floor referencing unassigned locals and the
# runner SEGV'd only on the path that also had TAO_RUN_DIR set — i.e.
# only under `toy train`, not under a direct runner call. resid_t/resid_v
# are up here for exactly this reason; the probe should have followed.
eps_t = [0]; eps_t.pop
eps_m = [0.0]; eps_m.pop
eps_b = [0.0]; eps_b.pop
eps_k = [0.0]; eps_k.pop
std_mean_max = 0.0
std_sd_err   = 0.0
final_loss = 0.0

# ============================================================
# SESSION A — the autoencoder (every arm except ar-baseline)
# ============================================================
if ARM_C != ARM_AR
  ae = Toy::LLM::Recipes::AeAuto.new
  ae.realize!(D_MODEL, N_HEADS, D_FF, N_BLOCKS, CONTEXT, D_LATENT, SEED, 1.0)
  ae_params = ae.ae_cache.param_count
  perm_id = Array.new(CONTEXT, 0)
  pi = 0
  while pi < CONTEXT
    perm_id[pi] = pi
    pi = pi + 1
  end
  gain_one  = Array.new(CONTEXT * D_LATENT, 1.0)
  noise_zero = Array.new(CONTEXT * D_LATENT, 0.0)
  ae.upload_probe!(perm_id, gain_one, noise_zero)

  toks   = Array.new(CONTEXT, 0)
  m_lab  = Mat.new(CONTEXT, VOCAB)
  ae_st  = lcg_state(TASK_SEED + 11)
  s1 = 0
  while s1 < AE_STEPS
    if WARMUP > 0 && s1 < WARMUP
      adamw.lr = LR * ((s1 + 1).to_f / WARMUP.to_f)
    else
      adamw.lr = LR
    end
    start = (lcg_u01(ae_st) * TRAIN_HI.to_f).to_i % TRAIN_HI
    task.fill_window!(toks, start)
    k = 0
    while k < CONTEXT * VOCAB
      m_lab.flat[k] = 0.0
      k = k + 1
    end
    b = 0
    while b < CONTEXT
      m_lab.flat[b * VOCAB + toks[b]] = 1.0
      b = b + 1
    end
    l = ae.step!(toks, m_lab, adamw.hp(s1), s1 == 0)
    if (s1 + 1) % 200 == 0 || s1 == AE_STEPS - 1
      puts "stage1 step " + (s1 + 1).to_s + ": loss=" + l.to_s
    end
    s1 = s1 + 1
  end

  # THE LATENT POOL: the train span tiled into non-overlapping windows
  # and encoded once. Precomputing is not an optimisation here — the
  # encoder's session cannot coexist with the denoiser's, so the latents
  # have to leave as plain arrays before session B exists.
  pool = Array.new(POOL_N * CONTEXT * D_LATENT, 0.0)
  lat_buf = Array.new(CONTEXT * D_LATENT, 0.0)
  pj = 0
  while pj < POOL_N
    task.fill_window!(toks, pj * CONTEXT)
    kk = 0
    while kk < CONTEXT * VOCAB
      m_lab.flat[kk] = 0.0
      kk = kk + 1
    end
    bb = 0
    while bb < CONTEXT
      m_lab.flat[bb * VOCAB + toks[bb]] = 1.0
      bb = bb + 1
    end
    ae.step!(toks, m_lab, noop_hp, false)
    TinyNN.tnn_download_to_f64_array(ae.ae_cache.sess, ae.ae_cache.t_latent,
                                     lat_buf, CONTEXT * D_LATENT)
    zi = 0
    while zi < CONTEXT * D_LATENT
      pool[pj * CONTEXT * D_LATENT + zi] = lat_buf[zi]
      zi = zi + 1
    end
    pj = pj + 1
  end

  # HELD-OUT latents, for the eps-skill probe. Same encoder, windows drawn
  # from the VAL span, standardised with the TRAIN statistics (using val
  # statistics would leak the held-out set into its own normalisation).
  vpool = Array.new(VAL_WINDOWS * CONTEXT * D_LATENT, 0.0)
  vj = 0
  while vj < VAL_WINDOWS
    vstart = SPLIT_AT + vj * CONTEXT
    if vstart > VAL_HI
      vstart = VAL_HI
    end
    task.fill_window!(toks, vstart)
    vk = 0
    while vk < CONTEXT * VOCAB
      m_lab.flat[vk] = 0.0
      vk = vk + 1
    end
    vb = 0
    while vb < CONTEXT
      m_lab.flat[vb * VOCAB + toks[vb]] = 1.0
      vb = vb + 1
    end
    ae.step!(toks, m_lab, noop_hp, false)
    TinyNN.tnn_download_to_f64_array(ae.ae_cache.sess, ae.ae_cache.t_latent,
                                     lat_buf, CONTEXT * D_LATENT)
    vz = 0
    while vz < CONTEXT * D_LATENT
      vpool[vj * CONTEXT * D_LATENT + vz] = lat_buf[vz]
      vz = vz + 1
    end
    vj = vj + 1
  end

  # STANDARDISE (the LDM scaling trick). Deterministic, no new loss term,
  # and the moments are asserted below rather than assumed.
  npos = POOL_N * CONTEXT
  sj = 0
  while sj < D_LATENT
    su = 0.0
    sq = 0.0
    q = 0
    while q < npos
      v = pool[q * D_LATENT + sj]
      su = su + v
      sq = sq + v * v
      q = q + 1
    end
    mu = su / npos.to_f
    vr = sq / npos.to_f - mu * mu
    if vr < 1.0e-12
      vr = 1.0e-12
    end
    lat_mu[sj] = mu
    lat_sd[sj] = Math.sqrt(vr)
    sj = sj + 1
  end
  pz = 0
  while pz < npos
    dj = 0
    while dj < D_LATENT
      pool[pz * D_LATENT + dj] = (pool[pz * D_LATENT + dj] - lat_mu[dj]) / lat_sd[dj]
      dj = dj + 1
    end
    pz = pz + 1
  end
  vq = 0
  while vq < VAL_WINDOWS * CONTEXT
    vd = 0
    while vd < D_LATENT
      vpool[vq * D_LATENT + vd] = (vpool[vq * D_LATENT + vd] - lat_mu[vd]) / lat_sd[vd]
      vd = vd + 1
    end
    vq = vq + 1
  end

  # ASSERT the standardisation actually landed. If the aggregate is not
  # ~N(0, I) the sampler starts off-manifold and every downstream number
  # is about a normalisation bug rather than about d.
  cj = 0
  while cj < D_LATENT
    su2 = 0.0
    sq2 = 0.0
    q2 = 0
    while q2 < npos
      v2 = pool[q2 * D_LATENT + cj]
      su2 = su2 + v2
      sq2 = sq2 + v2 * v2
      q2 = q2 + 1
    end
    m2 = su2 / npos.to_f
    if m2 < 0.0
      m2 = -m2
    end
    if m2 > std_mean_max
      std_mean_max = m2
    end
    sd2 = Math.sqrt(sq2 / npos.to_f - (su2 / npos.to_f) * (su2 / npos.to_f))
    e2 = sd2 - 1.0
    if e2 < 0.0
      e2 = -e2
    end
    if e2 > std_sd_err
      std_sd_err = e2
    end
    cj = cj + 1
  end
  if std_mean_max > 1.0e-3 || std_sd_err > 1.0e-3
    puts "toy-train-difflm: latent standardisation FAILED — max|mean|=" +
         std_mean_max.to_s + " max|sd-1|=" + std_sd_err.to_s +
         ". The diffusion sampler starts at N(0, I); an aggregate that is" +
         " not standard means it begins OFF-MANIFOLD and the samples would" +
         " be garbage for a reason unrelated to the latent width."
    exit 1
  end

  # THE DECODE HEAD LEAVES THE SESSION. One [VOCAB, d] matmul plus a bias
  # per position — no graph needed, and it has to be Ruby-side anyway
  # because session A stops being valid the moment session B is created.
  nw = ae.ae_cache.ft_weights.length
  head_w = Array.new(VOCAB * D_LATENT, 0.0)
  head_b = Array.new(VOCAB, 0.0)
  TinyNN.tnn_download_to_f64_array(ae.ae_cache.sess,
    ae.ae_cache.ft_weights[nw - 2], head_w, VOCAB * D_LATENT)
  TinyNN.tnn_download_to_f64_array(ae.ae_cache.sess,
    ae.ae_cache.ft_weights[nw - 1], head_b, VOCAB)

  puts "stage1: ae_steps=" + AE_STEPS.to_s +
       " pool_windows=" + POOL_N.to_s +
       " params=" + ae_params.to_s
  lstr = ""
  li = 0
  while li < D_LATENT
    if li > 0
      lstr = lstr + ","
    end
    lstr = lstr + lat_sd[li].to_s
    li = li + 1
  end
  puts "latent: d=" + D_LATENT.to_s +
       " raw_sd=" + lstr +
       " std_max_mean=" + std_mean_max.to_s +
       " std_max_sd_err=" + std_sd_err.to_s
end

# The diffusion schedule, shared by stage 2 and the sampler.
betas = Array.new(TSTEPS, 0.0)
alphas = Array.new(TSTEPS, 0.0)
abar = Array.new(TSTEPS, 0.0)
bi2 = 0
while bi2 < TSTEPS
  betas[bi2] = BETA_LO + (BETA_HI - BETA_LO) * (bi2.to_f / (TSTEPS - 1).to_f)
  alphas[bi2] = 1.0 - betas[bi2]
  if bi2 == 0
    abar[bi2] = alphas[bi2]
  else
    abar[bi2] = abar[bi2 - 1] * alphas[bi2]
  end
  bi2 = bi2 + 1
end
# toy#156's guard, carried verbatim in intent: if the forward process
# never reaches pure noise, the ancestral sampler starts OUT OF
# DISTRIBUTION and the generative metric scores that mismatch instead of
# the model. It is silent in the training loss.
if (ARM_C == ARM_SELF || ARM_C == ARM_PLAIN) && abar[TSTEPS - 1] > 0.01
  puts "toy-train-difflm: the schedule leaves abar_T=" + abar[TSTEPS - 1].to_s +
       " (> 0.01), so the forward process never reaches pure noise and the" +
       " sampler would start OUT OF DISTRIBUTION. Raise DL_BETA_HI or" +
       " DL_TSTEPS."
  exit 1
end

# ============================================================
# SESSION B — the arm
# ============================================================
gen_st = lcg_state(NOISE_SEED)

if ARM_C == ARM_AR
  arlm = Toy::LLM::Engine::ArEngine.new
  arlm.realize_for_random_init(AR_D_MODEL, N_HEADS, D_FF, AR_BLOCKS,
                               CONTEXT, SEED, 1.0)
  r = arlm.build_training_step
  ar_loss = r[0]; ar_lab = r[1]; ar_hp = r[2]
  arm_params = arlm.param_count
  cmask = Array.new(CONTEXT * CONTEXT, 0.0)
  arlm.fill_causal_mask!(cmask)
  TinyNN.tnn_upload_from_float_array(arlm.sess, arlm.t_mask, cmask,
                                     CONTEXT * CONTEXT)
  toks2 = Array.new(CONTEXT, 0)
  m_lab2 = Mat.new(CONTEXT, VOCAB)
  lbuf = Array.new(CONTEXT * VOCAB, 0.0)
  ar_st = lcg_state(TASK_SEED + 21)
  s2 = 0
  while s2 < STEPS
    if WARMUP > 0 && s2 < WARMUP
      adamw.lr = LR * ((s2 + 1).to_f / WARMUP.to_f)
    else
      adamw.lr = LR
    end
    st0 = (lcg_u01(ar_st) * (TRAIN_HI - 1).to_f).to_i % (TRAIN_HI - 1)
    task.fill_window!(toks2, st0)
    k2 = 0
    while k2 < CONTEXT * VOCAB
      m_lab2.flat[k2] = 0.0
      k2 = k2 + 1
    end
    # Position i predicts token i+1 — INCLUDING the last position, whose
    # target is the byte just past the window. Leaving that row all-zero
    # (the obvious reading of "the last position has no target") trains
    # every position EXCEPT the one generation samples from, and the
    # arm then emits noise: measured, the AR baseline scored 9.59
    # bits/byte on its own samples against 3.16 for real text — worse
    # than uniform over the observed alphabet. The window therefore
    # reads one byte beyond CONTEXT for its final label.
    b2 = 0
    while b2 < CONTEXT - 1
      m_lab2.flat[b2 * VOCAB + toks2[b2 + 1]] = 1.0
      b2 = b2 + 1
    end
    m_lab2.flat[(CONTEXT - 1) * VOCAB + task.at_tokens[st0 + CONTEXT]] = 1.0
    if s2 == 0
      TinyNN.tnn_graph_reset(arlm.sess)
    else
      TinyNN.tnn_graph_reset_grads_only(arlm.sess)
    end
    TinyNN.tnn_upload_from_int_array(arlm.sess, arlm.t_tokens, toks2, CONTEXT)
    TinyNN.upload_row_major(arlm.sess, ar_lab, m_lab2)
    TinyNN.upload_row_major(arlm.sess, ar_hp, adamw.hp(s2))
    TinyNN.tnn_compute_backward(arlm.sess)
    lm = TinyNN.download_row_major(arlm.sess, ar_loss, 1, 1)
    final_loss = lm.flat[0]
    puts "step " + (s2 + 1).to_s + ": loss=" + final_loss.to_s
    s2 = s2 + 1
  end
  # GENERATE by sampling, one byte at a time, seeded from real text.
  ctxw = Array.new(CONTEXT, 0)
  ci = 0
  while ci < CONTEXT
    ctxw[ci] = real_bytes[ci % REAL_N]
    ci = ci + 1
  end
  g = 0
  while g < GEN_BYTES
    TinyNN.tnn_graph_reset_grads_only(arlm.sess)
    TinyNN.tnn_upload_from_int_array(arlm.sess, arlm.t_tokens, ctxw, CONTEXT)
    TinyNN.upload_row_major(arlm.sess, ar_lab, m_lab2)
    TinyNN.upload_row_major(arlm.sess, ar_hp, noop_hp)
    TinyNN.tnn_compute_backward(arlm.sess)
    TinyNN.tnn_download_to_f64_array(arlm.sess, arlm.t_logits, lbuf,
                                     CONTEXT * VOCAB)
    nb = sample_at(lbuf, (CONTEXT - 1) * VOCAB, VOCAB, gen_st)
    gen_bytes[g] = nb
    sh = 0
    while sh < CONTEXT - 1
      ctxw[sh] = ctxw[sh + 1]
      sh = sh + 1
    end
    ctxw[CONTEXT - 1] = nb
    g = g + 1
  end
elsif ARM_C == ARM_FLOOR
  # THE MANDATORY CONTROL (tao#19). Latents straight from the prior, no
  # denoising at all, decoded by the same frozen head. If this does not
  # lose decisively the metric cannot discriminate and the lane measures
  # nothing.
  gz = 0
  while gz < GEN_BYTES
    zj = 0
    zvec = Array.new(D_LATENT, 0.0)
    while zj < D_LATENT
      zvec[zj] = lcg_gauss(gen_st) * lat_sd[zj] + lat_mu[zj]
      zj = zj + 1
    end
    best = 0
    bv = 0.0
    c = 0
    while c < VOCAB
      acc = head_b[c]
      d2 = 0
      while d2 < D_LATENT
        acc = acc + head_w[c * D_LATENT + d2] * zvec[d2]
        d2 = d2 + 1
      end
      if c == 0 || acc > bv
        bv = acc
        best = c
      end
      c = c + 1
    end
    gen_bytes[gz] = best
    gz = gz + 1
  end
else
  # ---- stage 2: the denoiser ----
  dn = Toy::LLM::Engine::DifflmEngine.new
  dn.realize_for_random_init(D_MODEL, N_HEADS, D_FF, N_BLOCKS, CONTEXT,
                             D_LATENT, TSTEPS, SEED, 1.0)
  rd = dn.build_training_step
  dn_loss = rd[0]; dn_eps = rd[1]; dn_hp = rd[2]
  arm_params = dn.param_count
  SELFCOND = ARM_C == ARM_SELF

  xin  = Array.new(CONTEXT * 2 * D_LATENT, 0.0)
  temb = Array.new(CONTEXT * D_MODEL, 0.0)
  m_eps = Mat.new(CONTEXT, D_LATENT)
  pred = Array.new(CONTEXT * D_LATENT, 0.0)
  z0   = Array.new(CONTEXT * D_LATENT, 0.0)
  dn_st = lcg_state(TASK_SEED + 31)

  # Sinusoidal time features, one d_model vector per timestep, broadcast
  # over positions (every position of a diffusion step shares its t).
  def fill_temb!(buf, t, tsteps, d_model, context)
    j = 0
    while j < d_model
      half = d_model / 2
      if j < half
        freq = Math.exp(-9.21034037 * (j.to_f / half.to_f))
        v = Math.sin(t.to_f * freq)
      else
        freq = Math.exp(-9.21034037 * ((j - half).to_f / half.to_f))
        v = Math.cos(t.to_f * freq)
      end
      p = 0
      while p < context
        buf[p * d_model + j] = v
        p = p + 1
      end
      j = j + 1
    end
    nil
  end

  def dn_forward!(dn, xin, temb, m_eps, hp, context, latent, d_model, pred, first)
    if first
      TinyNN.tnn_graph_reset(dn.sess)
    else
      TinyNN.tnn_graph_reset_grads_only(dn.sess)
    end
    TinyNN.tnn_upload_from_float_array(dn.sess, dn.t_xin, xin,
                                       context * 2 * latent)
    TinyNN.tnn_upload_from_float_array(dn.sess, dn.t_temb, temb,
                                       context * d_model)
    TinyNN.upload_row_major(dn.sess, dn.t_eps, m_eps)
    TinyNN.upload_row_major(dn.sess, dn.t_hp, hp)
    TinyNN.tnn_compute_backward(dn.sess)
    TinyNN.tnn_download_to_f64_array(dn.sess, dn.t_pred, pred, context * latent)
    lm = TinyNN.download_row_major(dn.sess, dn.t_loss, 1, 1)
    lm.flat[0]
  end

  s3 = 0
  while s3 < STEPS
    if WARMUP > 0 && s3 < WARMUP
      adamw.lr = LR * ((s3 + 1).to_f / WARMUP.to_f)
    else
      adamw.lr = LR
    end
    w = (lcg_u01(dn_st) * POOL_N.to_f).to_i % POOL_N
    t = (lcg_u01(dn_st) * TSTEPS.to_f).to_i % TSTEPS
    sa = Math.sqrt(abar[t])
    sb = Math.sqrt(1.0 - abar[t])
    zi2 = 0
    while zi2 < CONTEXT * D_LATENT
      z0[zi2] = pool[w * CONTEXT * D_LATENT + zi2]
      e = lcg_gauss(dn_st)
      m_eps.flat[zi2] = e
      xin[(zi2 / D_LATENT) * 2 * D_LATENT + (zi2 % D_LATENT)] = sa * z0[zi2] + sb * e
      zi2 = zi2 + 1
    end
    # self-conditioning half the time: a no-op-hp pre-pass supplies the
    # model's own x0 estimate, then the real step conditions on it.
    sc = 0
    while sc < CONTEXT * D_LATENT
      xin[(sc / D_LATENT) * 2 * D_LATENT + D_LATENT + (sc % D_LATENT)] = 0.0
      sc = sc + 1
    end
    if SELFCOND && lcg_u01(dn_st) < 0.5
      fill_temb!(temb, t, TSTEPS, D_MODEL, CONTEXT)
      dn_forward!(dn, xin, temb, m_eps, noop_hp, CONTEXT, D_LATENT, D_MODEL,
                  pred, s3 == 0)
      p2 = 0
      while p2 < CONTEXT * D_LATENT
        xt = xin[(p2 / D_LATENT) * 2 * D_LATENT + (p2 % D_LATENT)]
        x0h = (xt - sb * pred[p2]) / sa
        xin[(p2 / D_LATENT) * 2 * D_LATENT + D_LATENT + (p2 % D_LATENT)] = x0h
        p2 = p2 + 1
      end
    end
    fill_temb!(temb, t, TSTEPS, D_MODEL, CONTEXT)
    l3 = dn_forward!(dn, xin, temb, m_eps, adamw.hp(s3), CONTEXT, D_LATENT,
                     D_MODEL, pred, s3 == 0)
    final_loss = l3
    puts "step " + (s3 + 1).to_s + ": loss=" + l3.to_s
    s3 = s3 + 1
  end

  # ---- sample: full reverse chain, decode via the Ruby-side head ----
  nwin = GEN_BYTES / CONTEXT
  gi = 0
  while gi < nwin
    xcur = Array.new(CONTEXT * D_LATENT, 0.0)
    x0p  = Array.new(CONTEXT * D_LATENT, 0.0)
    ii = 0
    while ii < CONTEXT * D_LATENT
      xcur[ii] = lcg_gauss(gen_st)
      ii = ii + 1
    end
    tt = TSTEPS - 1
    while tt >= 0
      sa2 = Math.sqrt(abar[tt])
      sb2 = Math.sqrt(1.0 - abar[tt])
      q3 = 0
      while q3 < CONTEXT * D_LATENT
        xin[(q3 / D_LATENT) * 2 * D_LATENT + (q3 % D_LATENT)] = xcur[q3]
        xin[(q3 / D_LATENT) * 2 * D_LATENT + D_LATENT + (q3 % D_LATENT)] =
          SELFCOND ? x0p[q3] : 0.0
        m_eps.flat[q3] = 0.0
        q3 = q3 + 1
      end
      fill_temb!(temb, tt, TSTEPS, D_MODEL, CONTEXT)
      dn_forward!(dn, xin, temb, m_eps, noop_hp, CONTEXT, D_LATENT, D_MODEL,
                  pred, false)
      q4 = 0
      while q4 < CONTEXT * D_LATENT
        x0h = (xcur[q4] - sb2 * pred[q4]) / sa2
        x0p[q4] = x0h
        if tt > 0
          mean = (xcur[q4] - (betas[tt] / sb2) * pred[q4]) / Math.sqrt(alphas[tt])
          xcur[q4] = mean + Math.sqrt(betas[tt]) * lcg_gauss(gen_st)
        else
          xcur[q4] = x0h
        end
        q4 = q4 + 1
      end
      tt = tt - 1
    end
    # unstandardise, then decode
    pp = 0
    while pp < CONTEXT
      best = 0
      bv = 0.0
      c = 0
      while c < VOCAB
        acc = head_b[c]
        d3 = 0
        while d3 < D_LATENT
          acc = acc + head_w[c * D_LATENT + d3] *
                (xcur[pp * D_LATENT + d3] * lat_sd[d3] + lat_mu[d3])
          d3 = d3 + 1
        end
        if c == 0 || acc > bv
          bv = acc
          best = c
        end
        c = c + 1
      end
      if gi * CONTEXT + pp < GEN_BYTES
        gen_bytes[gi * CONTEXT + pp] = best
      end
      pp = pp + 1
    end
    gi = gi + 1
  end

  # ---- THE eps-SKILL PROBE (toy#167) ----
  #
  # WHY NOT RAW eps-MSE. As abar_t -> 0, x_t -> eps, so the TRIVIAL
  # predictor eps_hat = x_t scores near-zero error at high t: at t=99 its
  # MSE is 0.0000. A denoiser whose score is worthless far from the data
  # manifold would therefore look BEST exactly where generation starts.
  # Raw eps-MSE vs t is not just uninformative there, it is inverted.
  #
  # So the probe reports the model against the two trivial baselines on
  # the same axis and scores SKILL:
  #
  #     skill(t) = 1 - mse_model(t) / min( mse[eps_hat = x_t], 1.0 )
  #
  # skill > 0 means the denoiser carries information the trivial
  # predictors do not. skill ~ 0 at high t is the diagnosis toy#167 is
  # after: the score is uninformative where the reverse chain begins, and
  # no sampler can rescue that.
  #
  # Measured on HELD-OUT latents, and it is forward-only (noop hp), so it
  # perturbs neither the weights nor Adam's moments.
  pstate = lcg_state(NOISE_SEED + 555)
  gi3 = 0
  while gi3 < 10
    tp = (gi3 * TSTEPS) / 10
    if tp > TSTEPS - 1
      tp = TSTEPS - 1
    end
    sap = Math.sqrt(abar[tp])
    sbp = Math.sqrt(1.0 - abar[tp])
    acc_m = 0.0
    acc_b = 0.0
    nseen = 0
    vw = 0
    while vw < VAL_WINDOWS
      q8 = 0
      while q8 < CONTEXT * D_LATENT
        zt2 = vpool[vw * CONTEXT * D_LATENT + q8]
        ev = lcg_gauss(pstate)
        xtv = sap * zt2 + sbp * ev
        xin[(q8 / D_LATENT) * 2 * D_LATENT + (q8 % D_LATENT)] = xtv
        xin[(q8 / D_LATENT) * 2 * D_LATENT + D_LATENT + (q8 % D_LATENT)] = 0.0
        m_eps.flat[q8] = ev
        acc_b = acc_b + (xtv - ev) * (xtv - ev)
        q8 = q8 + 1
      end
      fill_temb!(temb, tp, TSTEPS, D_MODEL, CONTEXT)
      dn_forward!(dn, xin, temb, m_eps, noop_hp, CONTEXT, D_LATENT, D_MODEL,
                  pred, false)
      q9 = 0
      while q9 < CONTEXT * D_LATENT
        d9 = pred[q9] - m_eps.flat[q9]
        acc_m = acc_m + d9 * d9
        q9 = q9 + 1
      end
      nseen = nseen + CONTEXT * D_LATENT
      vw = vw + 1
    end
    mm2 = acc_m / nseen.to_f
    bb2 = acc_b / nseen.to_f
    best = bb2 < 1.0 ? bb2 : 1.0
    sk = best > 0.0 ? (1.0 - mm2 / best) : 0.0
    eps_t.push(tp)
    eps_m.push(mm2)
    eps_b.push(best)
    eps_k.push(sk)
    gi3 = gi3 + 1
  end
  ei = 0
  while ei < eps_t.length
    puts "epsmse: t=" + eps_t[ei].to_s +
         " abar=" + abar[eps_t[ei]].to_s +
         " model=" + eps_m[ei].to_s +
         " trivial=" + eps_b[ei].to_s +
         " skill=" + eps_k[ei].to_s
    ei = ei + 1
  end

  # ---- THE SAMPLER-RESIDUAL INSTRUMENT ----
  #
  # P1a's margin is denominated in latent-std units (d=8 holds ~.87 at
  # sigma 0.5). The question "does that margin accommodate the sampler's
  # residual" is usually answered by squinting at sample quality. It can
  # be MEASURED: take a REAL latent, forward-noise it to t, run the
  # reverse chain from t down to 0, and report ||x_hat - z|| per dim.
  # Because the latents are standardised, that number is already in
  # sigma units and drops straight onto P1a's axis.
  rts = [0]; rts.pop
  rts.push(TSTEPS / 10)
  rts.push(TSTEPS / 4)
  rts.push(TSTEPS / 2)
  ri2 = 0
  while ri2 < rts.length
    t0 = rts[ri2]
    if t0 < 1
      t0 = 1
    end
    acc_sq = 0.0
    nres = 4
    rw = 0
    while rw < nres
      wsel = (POOL_N - 1 - rw) % POOL_N
      zt = Array.new(CONTEXT * D_LATENT, 0.0)
      zz = 0
      while zz < CONTEXT * D_LATENT
        z0[zz] = pool[wsel * CONTEXT * D_LATENT + zz]
        zz = zz + 1
      end
      sa3 = Math.sqrt(abar[t0])
      sb3 = Math.sqrt(1.0 - abar[t0])
      zi3 = 0
      while zi3 < CONTEXT * D_LATENT
        zt[zi3] = sa3 * z0[zi3] + sb3 * lcg_gauss(gen_st)
        zi3 = zi3 + 1
      end
      x0p2 = Array.new(CONTEXT * D_LATENT, 0.0)
      t2 = t0
      while t2 >= 0
        sa4 = Math.sqrt(abar[t2])
        sb4 = Math.sqrt(1.0 - abar[t2])
        q5 = 0
        while q5 < CONTEXT * D_LATENT
          xin[(q5 / D_LATENT) * 2 * D_LATENT + (q5 % D_LATENT)] = zt[q5]
          xin[(q5 / D_LATENT) * 2 * D_LATENT + D_LATENT + (q5 % D_LATENT)] =
            SELFCOND ? x0p2[q5] : 0.0
          m_eps.flat[q5] = 0.0
          q5 = q5 + 1
        end
        fill_temb!(temb, t2, TSTEPS, D_MODEL, CONTEXT)
        dn_forward!(dn, xin, temb, m_eps, noop_hp, CONTEXT, D_LATENT, D_MODEL,
                    pred, false)
        q6 = 0
        while q6 < CONTEXT * D_LATENT
          x0h2 = (zt[q6] - sb4 * pred[q6]) / sa4
          x0p2[q6] = x0h2
          if t2 > 0
            mn = (zt[q6] - (betas[t2] / sb4) * pred[q6]) / Math.sqrt(alphas[t2])
            zt[q6] = mn + Math.sqrt(betas[t2]) * lcg_gauss(gen_st)
          else
            zt[q6] = x0h2
          end
          q6 = q6 + 1
        end
        t2 = t2 - 1
      end
      q7 = 0
      while q7 < CONTEXT * D_LATENT
        dd = zt[q7] - z0[q7]
        acc_sq = acc_sq + dd * dd
        q7 = q7 + 1
      end
      rw = rw + 1
    end
    resid_t.push(t0)
    resid_v.push(Math.sqrt(acc_sq / (nres * CONTEXT * D_LATENT).to_f))
    ri2 = ri2 + 1
  end
end

if ARM_C != ARM_AR
  puts "arm: " + ARM + " denoiser_params=" + arm_params.to_s +
       " ae_params=" + ae_params.to_s +
       " decode_head_params=" + (VOCAB * D_LATENT + VOCAB).to_s
else
  puts "arm: " + ARM + " ar_params=" + arm_params.to_s
end
if resid_t.length > 0
  rstr = ""
  rj = 0
  while rj < resid_t.length
    if rj > 0
      rstr = rstr + " "
    end
    rstr = rstr + "t" + resid_t[rj].to_s + "=" + resid_v[rj].to_s
    rj = rj + 1
  end
  puts "resid: " + rstr + " (rmse in latent-std units; P1a's margin axis)"
end

# ============================================================
# SESSION C — the JUDGE (a DIFFERENT seed; never an arm)
# ============================================================
judge = Toy::LLM::Engine::ArEngine.new
judge.realize_for_random_init(AR_D_MODEL, N_HEADS, D_FF, AR_BLOCKS,
                              CONTEXT, SEED + 1000, 1.0)
rj2 = judge.build_training_step
j_loss = rj2[0]; j_lab = rj2[1]; j_hp = rj2[2]
jmask = Array.new(CONTEXT * CONTEXT, 0.0)
judge.fill_causal_mask!(jmask)
TinyNN.tnn_upload_from_float_array(judge.sess, judge.t_mask, jmask,
                                   CONTEXT * CONTEXT)
jt = Array.new(CONTEXT, 0)
j_m = Mat.new(CONTEXT, VOCAB)
jbuf = Array.new(CONTEXT * VOCAB, 0.0)
j_st = lcg_state(TASK_SEED + 41)
sj2 = 0
while sj2 < JUDGE_STEPS
  adamw.lr = LR
  st1 = (lcg_u01(j_st) * (TRAIN_HI - 1).to_f).to_i % (TRAIN_HI - 1)
  task.fill_window!(jt, st1)
  k3 = 0
  while k3 < CONTEXT * VOCAB
    j_m.flat[k3] = 0.0
    k3 = k3 + 1
  end
  b3 = 0
  while b3 < CONTEXT - 1
    j_m.flat[b3 * VOCAB + jt[b3 + 1]] = 1.0
    b3 = b3 + 1
  end
  j_m.flat[(CONTEXT - 1) * VOCAB + task.at_tokens[st1 + CONTEXT]] = 1.0
  if sj2 == 0
    TinyNN.tnn_graph_reset(judge.sess)
  else
    TinyNN.tnn_graph_reset_grads_only(judge.sess)
  end
  TinyNN.tnn_upload_from_int_array(judge.sess, judge.t_tokens, jt, CONTEXT)
  TinyNN.upload_row_major(judge.sess, j_lab, j_m)
  TinyNN.upload_row_major(judge.sess, j_hp, adamw.hp(sj2))
  TinyNN.tnn_compute_backward(judge.sess)
  if (sj2 + 1) % 500 == 0
    lmj = TinyNN.download_row_major(judge.sess, j_loss, 1, 1)
    puts "judge step " + (sj2 + 1).to_s + ": loss=" + lmj.flat[0].to_s
  end
  sj2 = sj2 + 1
end

# bits/byte of a byte array under the judge. Computed in RUBY from the
# logits over positions 0..T-2 rather than from the training CE, so the
# padded last row carries no weight and the number is exact.
def judge_bpb(judge, j_lab, j_hp, j_m, jbuf, noop_hp, bytes, n, context, vocab)
  nll = 0.0
  cnt = 0
  w = 0
  while (w + 1) * context <= n
    i = 0
    while i < context
      # reuse jt via direct upload below
      i = i + 1
    end
    toks = Array.new(context, 0)
    z = 0
    while z < context
      toks[z] = bytes[w * context + z]
      z = z + 1
    end
    TinyNN.tnn_graph_reset_grads_only(judge.sess)
    TinyNN.tnn_upload_from_int_array(judge.sess, judge.t_tokens, toks, context)
    TinyNN.upload_row_major(judge.sess, j_lab, j_m)
    TinyNN.upload_row_major(judge.sess, j_hp, noop_hp)
    TinyNN.tnn_compute_backward(judge.sess)
    TinyNN.tnn_download_to_f64_array(judge.sess, judge.t_logits, jbuf,
                                     context * vocab)
    p = 0
    while p < context - 1
      base = p * vocab
      mx = jbuf[base]
      c = 1
      while c < vocab
        if jbuf[base + c] > mx
          mx = jbuf[base + c]
        end
        c = c + 1
      end
      tot = 0.0
      c2 = 0
      while c2 < vocab
        tot = tot + Math.exp(jbuf[base + c2] - mx)
        c2 = c2 + 1
      end
      tgt = toks[p + 1]
      lp = (jbuf[base + tgt] - mx) - Math.log(tot)
      nll = nll - lp
      cnt = cnt + 1
      p = p + 1
    end
    w = w + 1
  end
  if cnt == 0
    return -1.0
  end
  nll / (cnt.to_f * Math.log(2.0))
end

bpb_gen  = judge_bpb(judge, j_lab, j_hp, j_m, jbuf, noop_hp, gen_bytes,
                     GEN_BYTES, CONTEXT, VOCAB)
bpb_real = judge_bpb(judge, j_lab, j_hp, j_m, jbuf, noop_hp, real_bytes,
                     REAL_N, CONTEXT, VOCAB)

puts "judge: seed=" + (SEED + 1000).to_s + " steps=" + JUDGE_STEPS.to_s +
     " params=" + judge.param_count.to_s
puts "gen: arm=" + ARM + " bytes=" + GEN_BYTES.to_s +
     " bpb_gen=" + bpb_gen.to_s + " bpb_real=" + bpb_real.to_s +
     " excess=" + (bpb_gen - bpb_real).to_s

# ---- the generated bytes leave as DATA ----
#
# The n-gram JS divergence and the degeneracy read are PURE POST-
# PROCESSING over a byte array: no model, no FFI, no session. They were
# written here first and had to move out — a dense count table of 2500
# cells reproducibly took the process to 117 GB RSS and SIGKILL under
# Spinel, which is a codegen problem and not a real memory need.
#
# Emitting the bytes and scoring them in plain MRI (prep/difflm_report.rb,
# and the gate) is better regardless of that bug: the metric becomes
# independently checkable, re-scorable without re-running a cell, and
# reviewable in a language where a JS divergence is four obvious lines.
# The runner keeps only what genuinely needs the judge MODEL — bits/byte.
if TAO_RUN_DIR.length > 0
  fg = File.open(TAO_RUN_DIR + "/gen.bytes", "w")
  gi2 = 0
  while gi2 < GEN_BYTES
    fg.write(gen_bytes[gi2].to_s)
    fg.write("\n")
    gi2 = gi2 + 1
  end
  fg.close
  fr = File.open(TAO_RUN_DIR + "/real.bytes", "w")
  ri4 = 0
  while ri4 < REAL_N
    fr.write(real_bytes[ri4].to_s)
    fr.write("\n")
    ri4 = ri4 + 1
  end
  fr.close
end
puts "graph: nodes=" + judge.ar_graph_nodes.to_s +
     " bytes=" + judge.ar_graph_bytes.to_s

# ---- the qualitative dump ----
if TAO_RUN_DIR.length > 0
  smp = ""
  sn = GEN_BYTES < 1024 ? GEN_BYTES : 1024
  si = 0
  while si < sn
    smp = smp + gen_bytes[si].chr
    si = si + 1
  end
  f = File.open(TAO_RUN_DIR + "/sample.txt", "w")
  f.write(smp)
  f.close
end

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
    rs.add_str("name", "difflm")
    Toy::Events.add_provenance(rs,
      TinyNN.tnn_provenance_host_name, TinyNN.tnn_provenance_host_os,
      TinyNN.tnn_provenance_host_arch,
      TinyNN.tnn_backend_name(judge.sess))
    model = Toy::Json::Builder.new
    model.add_str("arch", "difflm")
    model.add_str("arm", ARM)
    model.add_num("latent_dim", D_LATENT)
    model.add_num("context", CONTEXT)
    model.add_num("tsteps", TSTEPS)
    model.add_num("ae_params", ae_params)
    model.add_num("arm_params", arm_params)
    model.add_num("judge_params", judge.param_count)
    rs.add_obj("model", model)
    rs2 = Toy::Json::Builder.new
    rs2.add_str("pack", TEXT)
    rs2.add_num("alphabet", task.at_alphabet)
    rs2.add_num("n_tokens", N_TOKENS)
    rs.add_obj("corpus", rs2)
    TinyNN.tnn_events_emit(rs.dump)

    ev = Toy::Json::Builder.new
    ev.add_str("kind", "eval")
    ev.add_str("phase", "eval")
    ev.add_num("t", TinyNN.tnn_events_now_seconds)
    ev.add_str("name", "generation")
    ev.add_str("arm", ARM)
    ev.add_num("gen_bytes", GEN_BYTES)
    ev.add_raw("bpb_gen", num_or_null(bpb_gen))
    ev.add_raw("bpb_real", num_or_null(bpb_real))
    ev.add_raw("eps_t",     Toy::Json.from_int_array(eps_t))
    ev.add_raw("eps_model", float_array_json(eps_m))
    ev.add_raw("eps_skill", float_array_json(eps_k))
    ev.add_raw("resid_t", Toy::Json.from_int_array(resid_t))
    ev.add_raw("resid_rmse", float_array_json(resid_v))
    TinyNN.tnn_events_emit(ev.dump)

    re = Toy::Json::Builder.new
    re.add_str("kind", "run_end")
    re.add_num("t", TinyNN.tnn_events_now_seconds)
    re.add_str("ended_at", TinyNN.tnn_events_iso8601_now)
    re.add_str("reason", "completed")
    re.add_num("final_step", STEPS)
    re.add_raw("final_loss", num_or_null(final_loss))
    re.add_raw("val_acc", "null")
    re.add_str("checkpoint", "none")
    re.add_raw("exit_code", "0")
    TinyNN.tnn_events_emit(re.dump)
    TinyNN.tnn_events_close
  end
end
