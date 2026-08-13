# lib/toy/run/train_ae.rb — Spinel-compiled latent-autoencoder training
# runner (-> libexec/toy-train-ae), the toy#165 / capstone P1a lane.
#
# WHAT THIS LANE IS FOR. The diffusion-text-LM capstone needs a per-token
# continuous latent of 4-8 dims, because F20/toy#156 put DFA's advantage
# in exactly that output-dim window. Whether TEXT survives such a latent
# is unrun in the literature. P1a is the cheapest decisive form of the
# question, and it is ALL BP — no DFA, no diffusion. If a tiny per-token
# latent cannot be decoded back to its token UNDER NOISE, no downstream
# denoiser and no credit rule can rescue the path.
#
# THE METRIC IS THE NOISE MARGIN, NOT CLEAN RECONSTRUCTION. Packing a few
# hundred codepoints into 4 continuous dims is ample analog capacity, so
# clean accuracy is near-vacuous at small d and is reported as a sanity
# check only. What decides the verdict is how much perturbation the
# latent tolerates before the decode head stops recovering the token,
# because a diffusion sampler hands the decoder an ESTIMATE with residual
# error and never the exact latent.
#
# THE ALPHABET IS THE SECOND AXIS (tao#22). The margin is packing-limited
# and packing depends on how many distinct symbols the head must
# separate, so the corpus is not a detail: at N=27 the d=4 problem is
# about as hard as N=256 at d=8, and a `go` could be manufactured by the
# alphabet alone. The lane ships three pinned corpora at measured N =
# 27 / 65 / 201 (prep/fetch_text.rb) and reports the alphabet it actually
# observed, over the pack AND over the scored windows.
#
# CPU-ONLY (tao#18). Own compilation unit (landmine #16).
#
# ENV CONTRACT:
#   STEPS / SEED / TAO_RUN_DIR / TOY_RUN_ID   — as every other runner
#   AE_TEXT         — pack prefix (prep/fetch_text.rb), e.g. data/ae_names
#   AE_LATENT       — the bottleneck width d, the swept axis
#   AE_CONTEXT      — window length T; also the batch, see below (default 256)
#   AE_BLOCKS       — encoder blocks (default 2)
#   AE_D_MODEL      — residual width (default 128)
#   AE_HEADS        — attention heads (default 4)
#   AE_D_FF         — FFN width (default 256)
#   AE_NOISE_EVAL   — comma list of latent SNR sigmas (default 0,0.25,0.5,1,2)
#   AE_NOISE_SEED   — the perturbation stream (default 4242)
#   AE_VAL_BATCHES  — held-out windows scored at the end (default 16)
#   AE_VAL_FRAC_PCT — percent of the corpus RESERVED for val (default 10)
#   AE_TASK_SEED    — window-draw seed, SEPARATE from SEED (default 7)
#   AE_LR / AE_WARMUP
#
# ONE WINDOW IS ONE STEP. The encoder attends within a window, so a
# "batch" of several windows would either mix them under one attention or
# need a third tensor axis. T positions already make a batch of T
# reconstruction targets, so `--context` is the batch and there is no
# separate `--batch` on this lane.
#
# NO --vocab. Byte-level is this lane's only tokenization and the head is
# 256-wide on every corpus (see toy_ae_task.rb on why it is not sized to
# the observed alphabet), so a `--vocab` knob would have exactly one
# legal value. If a BPE arm is ever added it gets its own flag then,
# rather than a flag that means an integer on one recipe and a string on
# another.
#
# STDOUT (byte-gated): "step <N>: loss=<float>" per step, then the
# alphabet line, "val: clean_acc=...", one "noise:" line per sigma, the
# half-accuracy SNR, the two control lines, and "graph: nodes=... bytes=...".
#
# Spinel hygiene: hand-built String-concat JSON (no #{}), ENV reads as
# TOP-LEVEL constants, no Struct, while loops.

require_relative "../io/json_builder"
require_relative "../io/json"
require_relative "../io/toy_events"
require_relative "../io/toy_ae_task"
require_relative "../llm/engine/ae_engine"
require_relative "../llm/recipes/ae_auto"
require_relative "../llm/adamw"
require_relative "../dev/toy_describe_flow"

STEPS       = (ENV["STEPS"] || "5").to_i
SEED        = (ENV["SEED"]  || "0").to_i
TAO_RUN_DIR = ENV["TAO_RUN_DIR"] || ""
RUN_ID      = ENV["TOY_RUN_ID"]  || ""
EVENTS      = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/events.jsonl") : ""

TEXT        = ENV["AE_TEXT"] || ""
D_LATENT    = (ENV["AE_LATENT"]   || "8").to_i
CONTEXT     = (ENV["AE_CONTEXT"]  || "256").to_i
N_BLOCKS    = (ENV["AE_BLOCKS"]   || "2").to_i
D_MODEL     = (ENV["AE_D_MODEL"]  || "128").to_i
N_HEADS     = (ENV["AE_HEADS"]    || "4").to_i
D_FF        = (ENV["AE_D_FF"]     || "256").to_i
NOISE_EVAL_S = ENV["AE_NOISE_EVAL"] || ""
NOISE_SEED  = (ENV["AE_NOISE_SEED"] || "4242").to_i
VAL_BATCHES = (ENV["AE_VAL_BATCHES"] || "16").to_i
VAL_FRAC_PCT = (ENV["AE_VAL_FRAC_PCT"] || "10").to_i
TASK_SEED   = (ENV["AE_TASK_SEED"] || "7").to_i
LR_S        = ENV["AE_LR"] || ""
WARMUP_S    = ENV["AE_WARMUP"] || ""

LR     = LR_S.length     > 0 ? LR_S.to_f     : 0.001
WARMUP = WARMUP_S.length > 0 ? WARMUP_S.to_i : 0
VOCAB  = 256

# ---- fail loud on every out-of-range shape (never-mask). ----
if STEPS < 1
  puts "toy-train-ae: STEPS must be >= 1, got " + STEPS.to_s
  exit 1
end
if TEXT.length == 0
  puts "toy-train-ae: AE_TEXT is required — this lane measures a REAL text" +
       " corpus and has no synthetic fallback, because a low-entropy" +
       " synthetic byte stream would INFLATE the noise margin at exactly" +
       " the small latent where the verdict is decided (tao#22)." +
       " Run: ruby prep/fetch_text.rb --all"
  exit 1
end
if D_LATENT < 1
  puts "toy-train-ae: AE_LATENT must be >= 1, got " + D_LATENT.to_s
  exit 1
end
if CONTEXT < 2
  puts "toy-train-ae: AE_CONTEXT must be >= 2, got " + CONTEXT.to_s
  exit 1
end
if N_BLOCKS < 1
  puts "toy-train-ae: AE_BLOCKS must be >= 1, got " + N_BLOCKS.to_s
  exit 1
end
if N_HEADS < 1
  puts "toy-train-ae: AE_HEADS must be >= 1, got " + N_HEADS.to_s
  exit 1
end
if D_MODEL < N_HEADS || D_MODEL % N_HEADS != 0
  puts "toy-train-ae: AE_D_MODEL must be a positive multiple of AE_HEADS" +
       " (a head slice of width 0 would make attention a no-op), got " +
       D_MODEL.to_s + " with " + N_HEADS.to_s + " heads"
  exit 1
end
if D_FF < 1
  puts "toy-train-ae: AE_D_FF must be >= 1, got " + D_FF.to_s
  exit 1
end
if VAL_BATCHES < 1
  puts "toy-train-ae: AE_VAL_BATCHES must be >= 1, got " + VAL_BATCHES.to_s
  exit 1
end
if VAL_FRAC_PCT < 1 || VAL_FRAC_PCT > 50
  puts "toy-train-ae: AE_VAL_FRAC_PCT must be in 1..50, got " + VAL_FRAC_PCT.to_s
  exit 1
end
if D_LATENT >= D_MODEL
  puts "toy-train-ae: AE_LATENT (" + D_LATENT.to_s + ") must be < AE_D_MODEL (" +
       D_MODEL.to_s + ") — a bottleneck at least as wide as the residual" +
       " stream is not a bottleneck, and the reference ceiling would then" +
       " be measuring the encoder rather than the latent"
  exit 1
end

# The noise grid. sigma is in units of the latent's OWN per-dim standard
# deviation, so the curve is comparable across d — a wider latent must
# not look more robust merely for carrying larger activations.
def parse_sigmas(s)
  out = [0.0]; out.pop
  if s.length == 0
    out.push(0.0)
    out.push(0.25)
    out.push(0.5)
    out.push(1.0)
    out.push(2.0)
    return out
  end
  parts = s.split(",")
  i = 0
  while i < parts.length
    tk = parts[i]
    if tk.length > 0
      v = tk.to_f
      if v < 0.0
        puts "toy-train-ae: AE_NOISE_EVAL sigma must be >= 0, got " + tk
        exit 1
      end
      out.push(v)
    end
    i = i + 1
  end
  if out.length == 0
    puts "toy-train-ae: AE_NOISE_EVAL named no sigmas"
    exit 1
  end
  out
end
SIGMAS = parse_sigmas(NOISE_EVAL_S)

# A JSON array of floats. Local rather than a Toy::Json helper on
# purpose: json.rb compiles into every runner, and a method only this
# unit ever calls is a type-inference hazard there for no gain here.
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

# toy#118 — non-finite floats serialize as JSON null (finite iff x-x==0).
def num_or_null(x)
  d = x - x
  if d == 0.0
    x.to_s
  else
    "null"
  end
end

def hits_in(buf, labels, n, classes)
  hit = 0
  i = 0
  while i < n
    base = i * classes
    best = 0
    bv   = buf[base]
    c = 1
    while c < classes
      if buf[base + c] > bv
        bv   = buf[base + c]
        best = c
      end
      c = c + 1
    end
    if best == labels[i]
      hit = hit + 1
    end
    i = i + 1
  end
  hit
end

# The perturbation stream. Reset before EVERY sigma pass, so each sigma
# sees the SAME standard normal draws scaled differently. That makes the
# margin curve a paired comparison down the grid instead of a fresh
# sample per point, which is where most of its variance would otherwise
# come from — and it is why a non-monotone step in the curve is worth
# looking at rather than shrugging off.
def noise_state(seed)
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

def noise_u01(state)
  s = state[0]
  s = (s * 1103515245 + 12345) & 0x7FFFFFFF
  state[0] = s
  (s.to_f + 1.0) / 2147483648.0
end

def noise_gauss(state)
  u1 = noise_u01(state)
  u2 = noise_u01(state)
  if u1 < 1.0e-12
    u1 = 1.0e-12
  end
  Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
end

# ---- the corpus. ----
task = AeTask.new(CONTEXT, TASK_SEED)
rc_pack = task.load_pack!(TEXT)
if rc_pack != 0
  exit 1
end
N_TOKENS = task.at_n_tokens

# TRAIN and VAL come from DISJOINT SPANS of the corpus. Same-span draws
# could overlap, and an overlapping window lets a memorising encoder
# inflate clean reconstruction — the very number every noise point is
# normalised against.
SPLIT_AT   = N_TOKENS - (N_TOKENS * VAL_FRAC_PCT) / 100
TRAIN_HI   = SPLIT_AT - CONTEXT
VAL_HI     = N_TOKENS - CONTEXT
if TRAIN_HI < 0 || VAL_HI < SPLIT_AT
  puts "toy-train-ae: corpus too short to split — n_tokens=" + N_TOKENS.to_s +
       " context=" + CONTEXT.to_s + " val_frac_pct=" + VAL_FRAC_PCT.to_s +
       " leaves train span " + TRAIN_HI.to_s + " and val span " +
       (VAL_HI - SPLIT_AT).to_s
  exit 1
end

recipe = Toy::LLM::Recipes::AeAuto.new
recipe.realize!(D_MODEL, N_HEADS, D_FF, N_BLOCKS, CONTEXT, D_LATENT,
                SEED, 1.0)
ToyDescribeFlow.emit_flow_json(TAO_RUN_DIR, recipe.ae_cache.sess)

# ---- the probe buffers. ----
PROBE_N = CONTEXT * D_LATENT
perm_id  = Array.new(CONTEXT, 0)
pi = 0
while pi < CONTEXT
  perm_id[pi] = pi
  pi = pi + 1
end
gain_one  = Array.new(PROBE_N, 1.0)
gain_zero = Array.new(PROBE_N, 0.0)
noise_zero = Array.new(PROBE_N, 0.0)
noise_buf  = Array.new(PROBE_N, 0.0)

# Training runs the probe at identity/ones/zeros, so the training graph
# IS the measurement graph (see ae_engine.rb).
recipe.upload_probe!(perm_id, gain_one, noise_zero)

tokens    = Array.new(CONTEXT, 0)
m_labels  = Mat.new(CONTEXT, VOCAB)
logit_buf = Array.new(CONTEXT * VOCAB, 0.0)
lat_buf   = Array.new(PROBE_N, 0.0)

# ---- the held-out windows, materialised FIRST. ----
task.reset_stream!(TASK_SEED + 1)
val_tok = Array.new(VAL_BATCHES * CONTEXT, 0)
vfill = 0
while vfill < VAL_BATCHES
  vs = task.next_start_in(SPLIT_AT, VAL_HI)
  task.fill_window!(tokens, vs)
  vi = 0
  while vi < CONTEXT
    val_tok[vfill * CONTEXT + vi] = tokens[vi]
    vi = vi + 1
  end
  vfill = vfill + 1
end

# The alphabet the SCORED windows actually contain, and the unigram floor
# over them. tao#22 asks for the effective alphabet next to latent_dim,
# and this is the honest place to take it: the head had to separate these
# ids, on these windows, not whatever the whole pack contains.
val_counts = Array.new(VOCAB, 0)
vc = 0
while vc < VAL_BATCHES * CONTEXT
  val_counts[val_tok[vc]] = val_counts[val_tok[vc]] + 1
  vc = vc + 1
end
VAL_N_TOK = VAL_BATCHES * CONTEXT
val_distinct = 0
val_best_c   = -1
val_best_id  = 0
val_ent      = 0.0
vk = 0
while vk < VOCAB
  kc = val_counts[vk]
  if kc > 0
    val_distinct = val_distinct + 1
    pv = kc.to_f / VAL_N_TOK.to_f
    val_ent = val_ent - pv * Math.log(pv) / Math.log(2.0)
  end
  if kc > val_best_c
    val_best_c  = kc
    val_best_id = vk
  end
  vk = vk + 1
end
VAL_FLOOR = val_best_c.to_f / VAL_N_TOK.to_f

adamw = Toy::AdamW.for_from_scratch
adamw.lr = LR

# lr = 0 on EVERY hp slot the graph can reach. This lane has exactly one
# hp vector; the mirror rule (toy#139/#146) says a second one added later
# must be zeroed here too, or the held-out set silently becomes training
# data.
val_hp = Mat.new(1, 7)
val_hp.flat[0] = 0.0
val_hp.flat[1] = adamw.beta1
val_hp.flat[2] = adamw.beta2
val_hp.flat[3] = adamw.eps
val_hp.flat[4] = 0.0
val_hp.flat[5] = adamw.beta1
val_hp.flat[6] = adamw.beta2

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
    rs.add_str("name", "ae")
    Toy::Events.add_provenance(rs,
      TinyNN.tnn_provenance_host_name, TinyNN.tnn_provenance_host_os,
      TinyNN.tnn_provenance_host_arch,
      TinyNN.tnn_backend_name(recipe.ae_cache.sess))
    model = Toy::Json::Builder.new
    model.add_str("arch", "ae")
    model.add_str("name", "latent-autoencoder")
    model.add_num("d_model",     D_MODEL)
    model.add_num("n_heads",     N_HEADS)
    model.add_num("d_ff",        D_FF)
    model.add_num("n_blocks",    N_BLOCKS)
    model.add_num("context",     CONTEXT)
    model.add_num("latent_dim",  D_LATENT)
    model.add_num("vocab",       VOCAB)
    model.add_str("encoder",     "bidirectional")
    model.add_str("decoder",     "per_position")
    rs.add_obj("model", model)
    cost = Toy::Json::Builder.new
    cost.add_num("total_params",  recipe.ae_cache.param_count)
    cost.add_num("active_params", recipe.ae_cache.param_count)
    cost.add_num("flops_per_token", 2 * recipe.ae_cache.param_count)
    cost.add_num("graph_nodes", recipe.ae_cache.ae_graph_nodes)
    cost.add_num("graph_bytes", recipe.ae_cache.ae_graph_bytes)
    rs.add_obj("cost", cost)
    config = Toy::Json::Builder.new
    config.add_num("steps",       STEPS)
    config.add_num("seed",        SEED)
    config.add_num("task_seed",   TASK_SEED)
    config.add_num("val_batches", VAL_BATCHES)
    config.add_num("val_frac_pct", VAL_FRAC_PCT)
    config.add_num("noise_seed",  NOISE_SEED)
    config.add_raw("lr",          LR.to_s)
    config.add_num("warmup",      WARMUP)
    rs.add_obj("config", config)
    # THE CORPUS, in provenance — the alphabet is the second axis, so a
    # margin curve without it is unscoped (tao#22).
    corpus = Toy::Json::Builder.new
    corpus.add_str("pack",         TEXT)
    corpus.add_num("n_tokens",     N_TOKENS)
    corpus.add_num("alphabet",     task.at_alphabet)
    corpus.add_raw("entropy_bits", task.at_entropy.to_s)
    corpus.add_num("val_alphabet", val_distinct)
    corpus.add_raw("val_entropy_bits", val_ent.to_s)
    corpus.add_raw("unigram_floor",    VAL_FLOOR.to_s)
    corpus.add_num("floor_id",     val_best_id)
    corpus.add_num("split_at",     SPLIT_AT)
    rs.add_obj("corpus", corpus)
    TinyNN.tnn_events_emit(rs.dump)
  else
    puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
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

  ts = task.next_start_in(0, TRAIN_HI)
  task.fill_window!(tokens, ts)
  k = 0
  while k < CONTEXT * VOCAB
    m_labels.flat[k] = 0.0
    k = k + 1
  end
  b = 0
  while b < CONTEXT
    # RECONSTRUCTION, not prediction: the target at position i is the
    # token AT position i. No shift anywhere in this lane.
    m_labels.flat[b * VOCAB + tokens[b]] = 1.0
    b = b + 1
  end

  loss = recipe.step!(tokens, m_labels, m_hp, step == 0)
  final_loss = loss
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s

  if EVENTS.length > 0
    rc_l = TinyNN.tnn_download_to_f64_array(recipe.ae_cache.sess,
             recipe.ae_cache.t_logits, logit_buf, CONTEXT * VOCAB)
    tr_acc = -1.0
    if rc_l != 0
      puts "logits download failed: step=" + (step + 1).to_s + " rc=" + rc_l.to_s
    else
      tr_acc = hits_in(logit_buf, tokens, CONTEXT, VOCAB).to_f / CONTEXT.to_f
    end
    step_wall_us = ((TinyNN.tnn_events_now_seconds - step_wall_start) * 1.0e6).to_i
    es = Toy::Json::Builder.new
    es.add_str("kind",  "step")
    es.add_str("phase", "train")
    es.add_num("t",       TinyNN.tnn_events_now_seconds)
    es.add_num("step",    step + 1)
    es.add_raw("loss",    num_or_null(loss))
    es.add_raw("train_acc", tr_acc >= 0.0 ? num_or_null(tr_acc) : "null")
    es.add_raw("lr",      adamw.lr.to_s)
    es.add_num("samples", CONTEXT)
    es.add_num("wall_us", step_wall_us)
    TinyNN.tnn_events_emit(es.dump)
  end
  step = step + 1
end

# ---- ONE held-out pass, parameterised by the probe. ----
#
# Every arm below — clean, each noise sigma, and both controls — is THIS
# function over THE SAME realized graph, differing only in the three
# uploaded probe tensors. There is no second eval path that could drift
# from the trained model.
def probe_pass!(recipe, val_tok, tokens, m_labels, val_hp, logit_buf,
                lat_buf, n_batches, context, vocab, d_latent,
                perm, gain, noise, per_window_noise, sigma, stds, nstate,
                lat_sum, lat_sq, collect_stats)
  hits = 0
  seen = 0
  loss_sum = 0.0
  vb = 0
  while vb < n_batches
    vi = 0
    while vi < context
      tokens[vi] = val_tok[vb * context + vi]
      vi = vi + 1
    end
    if per_window_noise
      # Fresh draws per window, but from a stream reset once per sigma —
      # so sigma is the only thing that changes down the grid.
      nb = 0
      while nb < context * d_latent
        noise[nb] = sigma * stds[nb % d_latent] * noise_gauss(nstate)
        nb = nb + 1
      end
    end
    recipe.upload_probe!(perm, gain, noise)
    k2 = 0
    while k2 < context * vocab
      m_labels.flat[k2] = 0.0
      k2 = k2 + 1
    end
    b2 = 0
    while b2 < context
      m_labels.flat[b2 * vocab + tokens[b2]] = 1.0
      b2 = b2 + 1
    end
    vloss = recipe.step!(tokens, m_labels, val_hp, false)
    loss_sum = loss_sum + vloss
    rc_v = TinyNN.tnn_download_to_f64_array(recipe.ae_cache.sess,
             recipe.ae_cache.t_logits, logit_buf, context * vocab)
    if rc_v != 0
      puts "toy-train-ae: val logits download failed: rc=" + rc_v.to_s
      exit 1
    end
    hits = hits + hits_in(logit_buf, tokens, context, vocab)
    seen = seen + context
    if collect_stats
      rc_z = TinyNN.tnn_download_to_f64_array(recipe.ae_cache.sess,
               recipe.ae_cache.t_latent, lat_buf, context * d_latent)
      if rc_z != 0
        puts "toy-train-ae: latent download failed: rc=" + rc_z.to_s
        exit 1
      end
      zi = 0
      while zi < context * d_latent
        j = zi % d_latent
        lat_sum[j] = lat_sum[j] + lat_buf[zi]
        lat_sq[j]  = lat_sq[j] + lat_buf[zi] * lat_buf[zi]
        zi = zi + 1
      end
    end
    vb = vb + 1
  end
  out = [0.0]
  out[0] = hits.to_f / seen.to_f
  out.push(loss_sum / n_batches.to_f)
  out
end

lat_sum = Array.new(D_LATENT, 0.0)
lat_sq  = Array.new(D_LATENT, 0.0)
stds    = Array.new(D_LATENT, 0.0)
dummy_state = noise_state(NOISE_SEED)

# 1. CLEAN — and the calibration pass that measures the latent's own
#    per-dim spread, which every sigma below is expressed in units of.
clean = probe_pass!(recipe, val_tok, tokens, m_labels, val_hp, logit_buf,
                    lat_buf, VAL_BATCHES, CONTEXT, VOCAB, D_LATENT,
                    perm_id, gain_one, noise_zero, false, 0.0, stds,
                    dummy_state, lat_sum, lat_sq, true)
clean_acc  = clean[0]
clean_loss = clean[1]
NLAT = VAL_BATCHES * CONTEXT
sj = 0
while sj < D_LATENT
  mu = lat_sum[sj] / NLAT.to_f
  vr = lat_sq[sj] / NLAT.to_f - mu * mu
  if vr < 0.0
    vr = 0.0
  end
  stds[sj] = Math.sqrt(vr)
  sj = sj + 1
end

puts "corpus: pack=" + TEXT + " n_tokens=" + N_TOKENS.to_s +
     " alphabet=" + task.at_alphabet.to_s +
     " val_alphabet=" + val_distinct.to_s +
     " val_entropy_bits=" + val_ent.to_s +
     " unigram_floor=" + VAL_FLOOR.to_s
puts "val: clean_acc=" + clean_acc.to_s + " loss=" + clean_loss.to_s +
     " n=" + NLAT.to_s + " latent=" + D_LATENT.to_s

# 2. THE NOISE-MARGIN CURVE — the read this lane exists for.
accs = [0.0]; accs.pop
si = 0
while si < SIGMAS.length
  sg = SIGMAS[si]
  nstate = noise_state(NOISE_SEED)
  r = probe_pass!(recipe, val_tok, tokens, m_labels, val_hp, logit_buf,
                  lat_buf, VAL_BATCHES, CONTEXT, VOCAB, D_LATENT,
                  perm_id, gain_one, noise_buf, true, sg, stds, nstate,
                  lat_sum, lat_sq, false)
  accs.push(r[0])
  puts "noise: sigma=" + sg.to_s + " acc=" + r[0].to_s + " loss=" + r[1].to_s
  si = si + 1
end

# 3. THE HALF-ACCURACY SNR — one scalar per cell. Linearly interpolated
#    between the two grid points that bracket the crossing.
#
#    If the curve never falls to half of clean within the grid the scalar
#    is UNDEFINED, and it is reported as ">=<max sigma>" rather than
#    clamped to the last grid point: a clamped value reads as a
#    measurement and would make a d=32 cell look like it collapsed
#    exactly where the grid happened to stop (never-mask).
half = 0.5 * clean_acc
h_sigma = -1.0
hi = 1
while hi < accs.length
  if h_sigma < 0.0 && accs[hi] < half && accs[hi - 1] >= half
    a0 = accs[hi - 1]
    a1 = accs[hi]
    s0 = SIGMAS[hi - 1]
    s1 = SIGMAS[hi]
    den = a0 - a1
    if den > 0.0
      h_sigma = s0 + (a0 - half) * (s1 - s0) / den
    else
      h_sigma = s1
    end
  end
  hi = hi + 1
end
if h_sigma < 0.0
  puts "half_snr: >=" + SIGMAS[SIGMAS.length - 1].to_s +
       " (accuracy never fell to half of clean inside the grid)"
else
  puts "half_snr: " + h_sigma.to_s
end

# 4. THE CONTROLS (tao#19 — a control that cannot lose measures nothing).
#
#    ZEROED is reported but NOT the gated leg: with a per-position decoder
#    and no context, a zeroed latent emits a constant logit vector by
#    construction, so it sits at the unigram floor whether or not training
#    worked. It is an identity, not an assertion.
#
#    SHUFFLED is the one with teeth. The latents are permuted ACROSS
#    POSITIONS, so each position's decode reads a real latent drawn from
#    the same distribution — just the wrong one. It can score above the
#    floor if the head learned a positional or frequency prior that
#    survives the permutation, so it CAN lose, and it is what proves the
#    decode reads the latent rather than a prior.
zero_r = probe_pass!(recipe, val_tok, tokens, m_labels, val_hp, logit_buf,
                     lat_buf, VAL_BATCHES, CONTEXT, VOCAB, D_LATENT,
                     perm_id, gain_zero, noise_zero, false, 0.0, stds,
                     dummy_state, lat_sum, lat_sq, false)

perm_sh = Array.new(CONTEXT, 0)
ps = 0
while ps < CONTEXT
  perm_sh[ps] = ps
  ps = ps + 1
end
sstate = noise_state(NOISE_SEED + 991)
pj = CONTEXT - 1
while pj > 0
  pr = (noise_u01(sstate) * (pj + 1).to_f).to_i % (pj + 1)
  ptmp = perm_sh[pj]
  perm_sh[pj] = perm_sh[pr]
  perm_sh[pr] = ptmp
  pj = pj - 1
end
shuf_r = probe_pass!(recipe, val_tok, tokens, m_labels, val_hp, logit_buf,
                     lat_buf, VAL_BATCHES, CONTEXT, VOCAB, D_LATENT,
                     perm_sh, gain_one, noise_zero, false, 0.0, stds,
                     dummy_state, lat_sum, lat_sq, false)

puts "control: zero_acc=" + zero_r[0].to_s +
     " shuffle_acc=" + shuf_r[0].to_s +
     " unigram_floor=" + VAL_FLOOR.to_s
lstr = ""
li = 0
while li < D_LATENT
  if li > 0
    lstr = lstr + ","
  end
  lstr = lstr + stds[li].to_s
  li = li + 1
end
puts "latent_std: " + lstr
puts "graph: nodes=" + recipe.ae_cache.ae_graph_nodes.to_s +
     " bytes=" + recipe.ae_cache.ae_graph_bytes.to_s

if EVENTS.length > 0 && TinyNN.tnn_events_active == 1
  ev = Toy::Json::Builder.new
  ev.add_str("kind",  "eval")
  ev.add_str("phase", "eval")
  ev.add_num("t",     TinyNN.tnn_events_now_seconds)
  ev.add_str("name",  "val")
  ev.add_num("n",     NLAT)
  ev.add_raw("accuracy", num_or_null(clean_acc))
  ev.add_raw("loss",     num_or_null(clean_loss))
  TinyNN.tnn_events_emit(ev.dump)

  # The margin curve as data, not only as stdout — this is what a Tao
  # report reads to build the margin(d, N) surface.
  mg = Toy::Json::Builder.new
  mg.add_str("kind",  "eval")
  mg.add_str("phase", "eval")
  mg.add_num("t",     TinyNN.tnn_events_now_seconds)
  mg.add_str("name",  "noise_margin")
  mg.add_num("latent_dim",   D_LATENT)
  mg.add_num("val_alphabet", val_distinct)
  mg.add_raw("sigmas",   float_array_json(SIGMAS))
  mg.add_raw("accs",     float_array_json(accs))
  mg.add_raw("clean_acc", num_or_null(clean_acc))
  mg.add_raw("half_snr",  h_sigma < 0.0 ? "null" : num_or_null(h_sigma))
  mg.add_raw("zero_acc",    num_or_null(zero_r[0]))
  mg.add_raw("shuffle_acc", num_or_null(shuf_r[0]))
  mg.add_raw("unigram_floor", num_or_null(VAL_FLOOR))
  TinyNN.tnn_events_emit(mg.dump)

  re = Toy::Json::Builder.new
  re.add_str("kind", "run_end")
  re.add_num("t",          TinyNN.tnn_events_now_seconds)
  re.add_str("ended_at",   TinyNN.tnn_events_iso8601_now)
  re.add_str("reason",     "completed")
  re.add_num("final_step", STEPS)
  re.add_raw("final_loss", num_or_null(final_loss))
  re.add_raw("val_acc",    num_or_null(clean_acc))
  re.add_str("checkpoint", "none")
  re.add_raw("exit_code",  "0")
  TinyNN.tnn_events_emit(re.dump)
  TinyNN.tnn_events_close
end
