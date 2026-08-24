# lib/toy/run/train_ctr.rb — Spinel-compiled CTR-tower training runner
# (→ libexec/toy-train-ctr), the toy#154 / DFA-arch T1 lane.
#
# WHAT THIS IS FOR. LightOn 2020 (arXiv:2006.12878) put DFA within
# ~0.01 AUC of BP on Criteo — their most convincing non-vision result,
# untouched since. It is the second-safest positive in the program and
# it probes a regime the anchor cannot: a SCALAR output (the extreme of
# the small-output-dim lens) with large sparse embedding tables under
# it. toy#152 measured DFA's recovery FALLING with output dim
# (1.01 -> 0.42 across 2 -> 1000 classes); a scalar output sits at the
# far good end of that curve, so this lane is where the lens predicts
# DFA should do best.
#
# CPU-ONLY (tao#18). Own compilation unit (landmine #16).
#
# ENV CONTRACT:
#   STEPS / SEED / RUN_DIR / TOY_RUN_ID   — as every other runner
#   CTR_POLICY      — per-TOWER-layer tokens: chain | dfa | frozen
#   CTR_FIELDS      — categorical fields (default 8)
#   CTR_CARD        — cardinality per field (default 64)
#   CTR_NUMERIC     — numeric features (default 4)
#   CTR_EMB         — embedding width (default 8)
#   CTR_HIDDEN      — tower width (default 64)
#   CTR_LAYERS      — tower depth (default 3)
#   CTR_PAIRS       — teacher interaction pairs (default 12)
#   CTR_BASE_RATE   — target positive rate (default 0.25)
#   CTR_LIN_SCALE   — weight of the ADDITIVE teacher part vs the
#                     pairwise crosses (default 0.25). At 1.0 the
#                     additive part dominates and embeddings+head alone
#                     solve the task — every arm ties and the success
#                     bar becomes unfalsifiable. See toy_ctr_task.rb.
#   CTR_TASK_SEED   — task seed, SEPARATE from SEED (default 7)
#   CTR_BATCH       — samples per step (default 128)
#   CTR_VAL_BATCHES — held-out batches (default 16)
#   CTR_WIDE        — 1 = add DeepFM's wide/FM branch (default off)
#   CTR_LR / CTR_WARMUP
#   CTR_B_SEED / CTR_B_DIST / CTR_B_SCALE
#
# STDOUT (byte-gated): "step <N>: loss=<float>" per step, then one
# "val: auc=<a> logloss=<l> n=<n> pos=<p>" line.
#
# THE METRIC IS AUC, and it is computed EXACTLY: every positive is
# compared against every negative (ties counted as 0.5), which is the
# definition. No sorting, no binning, no approximation to argue about
# when a result lands within 0.01 of another.
#
# Spinel hygiene: hand-built JSON (no #{}), ENV reads as TOP-LEVEL
# constants, no Struct, while loops.

require_relative "../io/json_builder"
require_relative "../io/json"
require_relative "../io/toy_events"
require_relative "../io/toy_ctr_task"
require_relative "../llm/engine/ctr_engine"
require_relative "../llm/recipes/ctr_tower"
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

POLICY_S    = ENV["CTR_POLICY"] || ""
N_FIELDS    = (ENV["CTR_FIELDS"]  || "8").to_i
CARD        = (ENV["CTR_CARD"]    || "64").to_i
N_NUMERIC   = (ENV["CTR_NUMERIC"] || "4").to_i
D_EMB       = (ENV["CTR_EMB"]     || "8").to_i
D_HIDDEN    = (ENV["CTR_HIDDEN"]  || "64").to_i
N_LAYERS    = (ENV["CTR_LAYERS"]  || "3").to_i
N_PAIRS     = (ENV["CTR_PAIRS"]   || "12").to_i
br_s        = ENV["CTR_BASE_RATE"] || ""
BASE_RATE   = br_s.length > 0 ? br_s.to_f : 0.25
ls_s        = ENV["CTR_LIN_SCALE"] || ""
LIN_SCALE   = ls_s.length > 0 ? ls_s.to_f : 0.25
TASK_SEED   = (ENV["CTR_TASK_SEED"] || "7").to_i
BATCH       = (ENV["CTR_BATCH"] || "128").to_i
VAL_BATCHES = (ENV["CTR_VAL_BATCHES"] || "16").to_i
LR_S        = ENV["CTR_LR"] || ""
LR          = LR_S.length > 0 ? LR_S.to_f : 0.003
WARMUP      = (ENV["CTR_WARMUP"] || "0").to_i
B_SEED      = (ENV["CTR_B_SEED"]  || "1234").to_i
B_DIST_S    = ENV["CTR_B_DIST"]   || ""
B_SCALE_S   = ENV["CTR_B_SCALE"]  || ""
# toy#154: CTR_WIDE=1 adds DeepFM's wide/FM branch — a linear map from
# the embeddings straight to the logit, AROUND the tower, always
# trained by BP. Default OFF: with it on, the tower stops being the
# only path to the label and every arm (including frozen) converges,
# which is the unfalsifiable-bar trap again. It exists to EXPLAIN the
# literature result, not to produce it — see the gate's two regimes.
WIDE_ON     = (ENV["CTR_WIDE"] || "") == "1"

if STEPS < 1
  puts "toy-train-ctr: STEPS must be >= 1, got " + STEPS.to_s
  exit 1
end
if N_FIELDS < 1 || CARD < 2 || D_EMB < 1 || D_HIDDEN < 1 || N_LAYERS < 1
  puts "toy-train-ctr: CTR_FIELDS/CARD/EMB/HIDDEN/LAYERS must be >= 1 (CARD >= 2)"
  exit 1
end
if N_NUMERIC < 0
  puts "toy-train-ctr: CTR_NUMERIC must be >= 0, got " + N_NUMERIC.to_s
  exit 1
end
if BATCH < 2
  puts "toy-train-ctr: CTR_BATCH must be >= 2 (AUC needs both classes present), got " + BATCH.to_s
  exit 1
end
if VAL_BATCHES < 1
  puts "toy-train-ctr: CTR_VAL_BATCHES must be >= 1, got " + VAL_BATCHES.to_s
  exit 1
end
if BASE_RATE <= 0.0 || BASE_RATE >= 1.0
  puts "toy-train-ctr: CTR_BASE_RATE must be in (0,1), got " + BASE_RATE.to_s
  exit 1
end

# Tokens are LANE-LOCAL: chain | dfa | frozen, one per TOWER layer.
# The embedding tables and the head are never policied — the tables
# stay chain by construction (the ticket), and at the output layer DFA
# and BP coincide.
def parse_ctr_policy(pol_s, n_layers)
  policy = [0]; policy.pop
  parts = pol_s.split(",")
  if parts.length > n_layers
    puts "toy-train-ctr: CTR_POLICY names " + parts.length.to_s +
         " layers but CTR_LAYERS=" + n_layers.to_s +
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
        puts "toy-train-ctr: unknown CTR_POLICY token " + tk + " (chain|dfa|frozen)"
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

# EXACT AUC: the Mann-Whitney statistic. Every positive against every
# negative, ties at 0.5 — the definition, not an approximation. Returns
# -1.0 when a class is absent (AUC is undefined then, and reporting 0.5
# would be a silent lie).
def auc_of(scores, labels, n)
  pos = 0
  i = 0
  while i < n
    if labels[i] == 1
      pos = pos + 1
    end
    i = i + 1
  end
  neg = n - pos
  if pos == 0 || neg == 0
    return -1.0
  end
  acc = 0.0
  p = 0
  while p < n
    if labels[p] == 1
      sp = scores[p]
      q = 0
      while q < n
        if labels[q] == 0
          if sp > scores[q]
            acc = acc + 1.0
          elsif sp == scores[q]
            acc = acc + 0.5
          end
        end
        q = q + 1
      end
    end
    p = p + 1
  end
  acc / (pos.to_f * neg.to_f)
end

POLICY = parse_ctr_policy(POLICY_S, N_LAYERS)

recipe = Toy::LLM::Recipes::CtrTower.new
recipe.realize!(N_FIELDS, CARD, N_NUMERIC, D_EMB, D_HIDDEN, N_LAYERS,
                BATCH, SEED, 1.0, POLICY, B_SEED, dist_code(B_DIST_S),
                scale_code(B_SCALE_S), scale_sigma(B_SCALE_S),
                WIDE_ON ? 1 : 0)
ToyDescribeFlow.emit_flow_json(RUN_DIR, recipe.ct_cache.sess)

task = CtrTask.new(N_FIELDS, CARD, N_NUMERIC, 8, N_PAIRS, TASK_SEED,
                   BASE_RATE, LIN_SCALE)
task.reset_stream!(TASK_SEED + 1)

idx      = [0]; idx.pop
ii = 0
while ii < N_FIELDS * BATCH
  idx.push(0)
  ii = ii + 1
end
m_num    = Mat.new(BATCH, N_NUMERIC > 0 ? N_NUMERIC : 1)
m_labels = Mat.new(BATCH, 2)
m_y      = Mat.new(BATCH, 1)
labels   = [0]; labels.pop
li0 = 0
while li0 < BATCH
  labels.push(0)
  li0 = li0 + 1
end
logit_buf = [0.0]; logit_buf.pop
lb0 = 0
while lb0 < BATCH
  logit_buf.push(0.0)
  lb0 = lb0 + 1
end

adamw = Toy::AdamW.for_from_scratch
adamw.lr = LR

# ---- the held-out set, MATERIALISED FIRST (toy#152's discipline).
# Drawn from the head of the sampling stream; training continues from
# what follows and cannot revisit it. Two differently-seeded streams
# would be two offsets into the SAME LCG cycle and could silently
# overlap — and a CTR metric measured on training rows is exactly the
# kind of quiet inflation this program exists to avoid. ----
VAL_N = VAL_BATCHES * BATCH
val_idx = [0]; val_idx.pop
vi0 = 0
while vi0 < N_FIELDS * VAL_N
  val_idx.push(0)
  vi0 = vi0 + 1
end
val_num = [0.0]; val_num.pop
vn0 = 0
while vn0 < VAL_N * (N_NUMERIC > 0 ? N_NUMERIC : 1)
  val_num.push(0.0)
  vn0 = vn0 + 1
end
val_y = [0]; val_y.pop
vy0 = 0
while vy0 < VAL_N
  val_y.push(0)
  vy0 = vy0 + 1
end
vfill = 0
while vfill < VAL_BATCHES
  task.fill_batch!(BATCH, idx, m_num, labels)
  vk = 0
  while vk < BATCH
    dst = vfill * BATCH + vk
    fj = 0
    while fj < N_FIELDS
      val_idx[fj * VAL_N + dst] = idx[fj * BATCH + vk]
      fj = fj + 1
    end
    nj = 0
    while nj < N_NUMERIC
      val_num[dst * N_NUMERIC + nj] = m_num.flat[vk * N_NUMERIC + nj]
      nj = nj + 1
    end
    val_y[dst] = labels[vk]
    vk = vk + 1
  end
  vfill = vfill + 1
end
val_pos = 0
vp = 0
while vp < VAL_N
  if val_y[vp] == 1
    val_pos = val_pos + 1
  end
  vp = vp + 1
end

# ---- run_start ----
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
    rs.add_str("name", "ctr")
    Toy::Events.add_provenance(rs,
      TinyNN.tnn_provenance_host_name, TinyNN.tnn_provenance_host_os,
      TinyNN.tnn_provenance_host_arch,
      TinyNN.tnn_backend_name(recipe.ct_cache.sess))
    model = Toy::Json::Builder.new
    model.add_str("arch", "ctr")
    model.add_str("name", "ctr-tower")
    model.add_num("fields",      N_FIELDS)
    model.add_num("cardinality", CARD)
    model.add_num("numeric",     N_NUMERIC)
    model.add_num("d_emb",       D_EMB)
    model.add_num("d_in",        recipe.ct_cache.ctr_d_in)
    model.add_num("d_hidden",    D_HIDDEN)
    model.add_num("n_layers",    N_LAYERS)
    model.add_num("num_classes", 1)
    model.add_str("act",         "silu")
    model.add_str("head",        "sigmoid-scalar")
    model.add_bool("wide_path",  WIDE_ON)
    rs.add_obj("model", model)
    cost = Toy::Json::Builder.new
    cost.add_num("total_params",  recipe.ct_cache.param_count)
    cost.add_num("active_params", recipe.ct_cache.param_count)
    cost.add_num("tower_params",  recipe.ct_cache.tower_param_count)
    cost.add_num("flops_per_token", 2 * recipe.ct_cache.param_count)
    rs.add_obj("cost", cost)
    config = Toy::Json::Builder.new
    config.add_num("steps",       STEPS)
    config.add_num("seed",        SEED)
    config.add_num("batch",       BATCH)
    config.add_num("val_batches", VAL_BATCHES)
    config.add_num("val_n",       VAL_N)
    config.add_num("val_pos",     val_pos)
    config.add_num("task_seed",   TASK_SEED)
    config.add_num("pairs",       N_PAIRS)
    config.add_raw("base_rate",   BASE_RATE.to_s)
    config.add_raw("lin_scale",   LIN_SCALE.to_s)
    config.add_raw("lr",          LR.to_s)
    config.add_num("warmup",      WARMUP)
    rs.add_obj("config", config)
    # NOT the `franken` key: this is not a transformer run (toy#152's
    # precedent). Ingest keys on align `wname`, per tao#19 item 3.
    dfa = Toy::Json::Builder.new
    dfa.add_raw("policy",  Toy::Json.from_int_array(POLICY))
    dfa.add_str("policied_tensors", "tower_layer_output")
    dfa.add_str("dfa_granularity",  "block")
    dfa.add_num("b_seed",  B_SEED)
    dfa.add_str("b_dist",  B_DIST_S.length > 0 ? B_DIST_S : "gaussian")
    dfa.add_str("b_scale", B_SCALE_S.length > 0 ? B_SCALE_S : "inv_sqrt_fan")
    dfa.add_num("error_dim", 1)
    dfa.add_num("dfa_wired", recipe.ct_cache.ctr_dfa_wired)
    dfa.add_num("frozen",    recipe.ct_cache.ctr_frozen_count)
    rs.add_obj("dfa", dfa)
    TinyNN.tnn_events_emit(rs.dump)
  else
    puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
  end
end

# ---- training loop ----
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

  task.fill_batch!(BATCH, idx, m_num, labels)
  b = 0
  while b < BATCH
    y = labels[b]
    m_labels.flat[b * 2]     = y == 1 ? 0.0 : 1.0
    m_labels.flat[b * 2 + 1] = y == 1 ? 1.0 : 0.0
    m_y.flat[b] = y.to_f
    b = b + 1
  end

  loss = recipe.step!(idx, m_num, m_labels, m_y, m_hp, step == 0)
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
  step = step + 1
end

# ---- held-out AUC. lr=0 makes the optimizer step a weight no-op.
# MIRROR RULE: "lr=0" must mean EVERY hp vector the graph can reach —
# this lane has exactly ONE; if a per-layer vector is ever added here
# it must be zeroed in this loop too, or the metric trains on its own
# test split (the toy#139/#146 class of bug). ----
val_hp = Mat.new(1, 7)
val_hp.flat[0] = 0.0
val_hp.flat[1] = adamw.beta1
val_hp.flat[2] = adamw.beta2
val_hp.flat[3] = adamw.eps
val_hp.flat[4] = 0.0
val_hp.flat[5] = adamw.beta1
val_hp.flat[6] = adamw.beta2

val_scores = [0.0]; val_scores.pop
vs0 = 0
while vs0 < VAL_N
  val_scores.push(0.0)
  vs0 = vs0 + 1
end
val_loss_sum = 0.0
vb = 0
while vb < VAL_BATCHES
  vr = 0
  while vr < BATCH
    src = vb * BATCH + vr
    fj = 0
    while fj < N_FIELDS
      idx[fj * BATCH + vr] = val_idx[fj * VAL_N + src]
      fj = fj + 1
    end
    nj = 0
    while nj < N_NUMERIC
      m_num.flat[vr * N_NUMERIC + nj] = val_num[src * N_NUMERIC + nj]
      nj = nj + 1
    end
    y = val_y[src]
    labels[vr] = y
    m_labels.flat[vr * 2]     = y == 1 ? 0.0 : 1.0
    m_labels.flat[vr * 2 + 1] = y == 1 ? 1.0 : 0.0
    m_y.flat[vr] = y.to_f
    vr = vr + 1
  end
  vloss = recipe.step!(idx, m_num, m_labels, m_y, val_hp, false)
  val_loss_sum = val_loss_sum + vloss
  rc_v = TinyNN.tnn_download_to_f64_array(recipe.ct_cache.sess,
           recipe.ct_cache.t_logit, logit_buf, BATCH)
  if rc_v != 0
    puts "toy-train-ctr: val logit download failed: rc=" + rc_v.to_s
    exit 1
  end
  vk2 = 0
  while vk2 < BATCH
    # The logit is a monotone function of the probability, so ranking
    # by it gives the same AUC — no sigmoid needed here.
    val_scores[vb * BATCH + vk2] = logit_buf[vk2]
    vk2 = vk2 + 1
  end
  vb = vb + 1
end
val_auc  = auc_of(val_scores, val_y, VAL_N)
val_loss = val_loss_sum / VAL_BATCHES.to_f
if val_auc < 0.0
  puts "toy-train-ctr: held-out set has only one class (" + val_pos.to_s + "/" + VAL_N.to_s +
       " positive) — AUC is undefined. Raise CTR_VAL_BATCHES or CTR_BASE_RATE."
  exit 1
end
puts "val: auc=" + val_auc.to_s + " logloss=" + val_loss.to_s +
     " n=" + VAL_N.to_s + " pos=" + val_pos.to_s

if EVENTS.length > 0 && TinyNN.tnn_events_active == 1
  ev = Toy::Json::Builder.new
  ev.add_str("kind",  "eval")
  ev.add_str("phase", "eval")
  ev.add_num("t",     TinyNN.tnn_events_now_seconds)
  ev.add_str("name",  "val")
  ev.add_num("n",     VAL_N)
  ev.add_num("pos",   val_pos)
  ev.add_raw("auc",     num_or_null(val_auc))
  ev.add_raw("logloss", num_or_null(val_loss))
  TinyNN.tnn_events_emit(ev.dump)

  re = Toy::Json::Builder.new
  re.add_str("kind", "run_end")
  re.add_num("t",          TinyNN.tnn_events_now_seconds)
  re.add_str("ended_at",   TinyNN.tnn_events_iso8601_now)
  re.add_str("reason",     "completed")
  re.add_num("final_step", STEPS)
  re.add_raw("final_loss", num_or_null(final_loss))
  re.add_raw("val_auc",    num_or_null(val_auc))
  re.add_str("checkpoint", "none")
  re.add_raw("exit_code",  "0")
  TinyNN.tnn_events_emit(re.dump)
  TinyNN.tnn_events_close
end
