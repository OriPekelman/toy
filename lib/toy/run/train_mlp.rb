# lib/toy/run/train_mlp.rb — Spinel-compiled MLP-classifier training
# runner (→ libexec/toy-train-mlp), the toy#152 / DFA-arch T0 ANCHOR.
#
# WHAT THIS IS FOR. Every DFA result we have (F4–F14) is NEGATIVE, and
# all of it is on transformer LMs at vocab 50257. Our lens (F13/F13b)
# and the theory (Refinetti et al. 2021, "Align, then memorise";
# Nøkland 2016) say DFA works when the OUTPUT DIMENSION is small. If
# this harness cannot reproduce a KNOWN DFA positive at small output
# dim, those negatives are not findings — they are potentially harness
# artifacts. So this lane is the control for everything already
# shipped, and it is deliberately the cheapest one: an N-layer MLP, a
# small-#classes head, cross-entropy, synthetic data.
#
# CPU-ONLY, no CUDA twin (tao#18). Own compilation unit (landmine #16).
#
# ENV CONTRACT:
#   STEPS / SEED / RUN_DIR / TOY_RUN_ID   — as every other runner
#   MLP_POLICY      — per-HIDDEN-layer tokens: chain | dfa | frozen
#                     (default: all chain, i.e. plain backprop)
#   MLP_LAYERS      — hidden layers (default 3)
#   MLP_HIDDEN      — hidden width (default 64)
#   MLP_FEATURES    — input dim (default 32)
#   MLP_CLASSES     — output dim; the axis under test (default 10)
#   MLP_TASK        — teacher (default) | blobs   [see toy_mlp_task.rb]
#   MLP_TEACHER_DIM — teacher hidden width (default 32)
#   MLP_TASK_SEED   — task seed; SEPARATE from SEED so the same task can
#                     be trained from different inits (default 7)
#   MLP_BATCH       — samples per step (default 64)
#   MLP_VAL_BATCHES — held-out batches evaluated at the end (default 8)
#   MLP_LR          — AdamW lr (default 0.01)
#   MLP_WARMUP      — linear lr ramp over N steps (default 0)
#   MLP_B_SEED / MLP_B_DIST / MLP_B_SCALE — the DfaB feedback axes
#   MLP_ALIGN       — "1" => per-step align events (cos∠(g_dfa, g_bp))
#   MLP_ALIGN_EVERY — thin align emissions to every Nth step
#
# STDOUT (byte-gated): "step <N>: loss=<float>" per step, then one
# "val: acc=<a> loss=<l> n=<n>" line. events.jsonl goes to the run dir.
#
# THE THREE ARMS, and why the frozen one is not optional: tao#19 made
# the success bar
#     positive = all-DFA within the stated gap of all-BP
#                AND provably beating the frozen control,
#                at matched init and matched seed
# MANDATORY, because "near-BP" on its own cannot distinguish "DFA
# taught the hidden layers something" from "this task is trivially
# easy" — the trap that inflated our own early MoE numbers, and the
# shape of toy#141 where the frozen arm BEAT both dfa and chain. The
# frozen arm here is FROZEN WEIGHTS (hidden layers stay at init, only
# the head trains), which is the reading that does the job; see
# docs/roadmap/dfa-arch-program-2026-08-10.md for why the literal
# "random B, never updated" reading would make the gate vacuous.
#
# Spinel hygiene: hand-built String-concat JSON (no #{}), ENV reads as
# TOP-LEVEL constants (a constant assigned inside a conditional arm
# reads back empty at runtime), no Struct, while loops.

require_relative "../io/json_builder"
require_relative "../io/json"
require_relative "../io/toy_events"
require_relative "../io/toy_mlp_task"
require_relative "../llm/engine/mlp_engine"
require_relative "../llm/recipes/mlp_classifier"
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

POLICY_S    = ENV["MLP_POLICY"] || ""
N_LAYERS    = (ENV["MLP_LAYERS"]   || "3").to_i
D_HIDDEN    = (ENV["MLP_HIDDEN"]   || "64").to_i
D_IN        = (ENV["MLP_FEATURES"] || "32").to_i
N_CLASSES   = (ENV["MLP_CLASSES"]  || "10").to_i
TASK_S      = ENV["MLP_TASK"] || ""
D_TEACH     = (ENV["MLP_TEACHER_DIM"] || "32").to_i
TASK_SEED   = (ENV["MLP_TASK_SEED"]   || "7").to_i
BATCH       = (ENV["MLP_BATCH"] || "64").to_i
VAL_BATCHES = (ENV["MLP_VAL_BATCHES"] || "8").to_i
LR_S        = ENV["MLP_LR"] || ""
# 0.003, not the tree-wide 0.001: measured on this lane's own arms
# (3x64 hidden, 10 classes, 1000 steps, seed 0) —
#   lr 0.001: BP .701 DFA .656 frozen .469
#   lr 0.003: BP .746 DFA .715 frozen .516   <- the anchor
#   lr 0.010: BP .764 DFA .689 frozen .545
#   lr 0.030: BP .762 DFA .662 (DFA val loss blows up to 78)
# DFA tolerates less LR than BP — the alignment phase is what a large
# step destroys — which is the literature's own finding and the reason
# this lane's default is NOT inherited from the llama lanes.
LR          = LR_S.length > 0 ? LR_S.to_f : 0.003
WARMUP      = (ENV["MLP_WARMUP"] || "0").to_i
B_SEED      = (ENV["MLP_B_SEED"]  || "1234").to_i
B_DIST_S    = ENV["MLP_B_DIST"]   || ""
# MEASURED, so nobody re-runs the sweep: under AdamW the B SCALE barely
# moves this lane (inv_sqrt_fan / glorot / fixed:1.0 all land within
# 1 point of each other at 1000 steps) — Adam normalises per-parameter
# magnitude, so only B's DIRECTION carries information. The scale axis
# matters for SGD lanes, not here.
B_SCALE_S   = ENV["MLP_B_SCALE"]  || ""
ALIGN_ON    = (ENV["MLP_ALIGN"] || "") == "1"
ae_raw      = (ENV["MLP_ALIGN_EVERY"] || "1").to_i
ALIGN_EVERY = ae_raw < 1 ? 1 : ae_raw

# ---- fail loud on every out-of-range shape (never-mask). ----
if STEPS < 1
  puts "toy-train-mlp: STEPS must be >= 1, got " + STEPS.to_s
  exit 1
end
if N_LAYERS < 1
  puts "toy-train-mlp: MLP_LAYERS must be >= 1, got " + N_LAYERS.to_s
  exit 1
end
if D_HIDDEN < 1 || D_IN < 1 || D_TEACH < 1
  puts "toy-train-mlp: MLP_HIDDEN / MLP_FEATURES / MLP_TEACHER_DIM must be >= 1"
  exit 1
end
if N_CLASSES < 2
  puts "toy-train-mlp: MLP_CLASSES must be >= 2, got " + N_CLASSES.to_s
  exit 1
end
if BATCH < 1
  puts "toy-train-mlp: MLP_BATCH must be >= 1, got " + BATCH.to_s
  exit 1
end
if VAL_BATCHES < 1
  puts "toy-train-mlp: MLP_VAL_BATCHES must be >= 1, got " + VAL_BATCHES.to_s
  exit 1
end
if TASK_S.length > 0 && TASK_S != "teacher" && TASK_S != "blobs"
  puts "toy-train-mlp: MLP_TASK " + TASK_S + " unsupported (teacher|blobs)"
  exit 1
end
TASK_KIND = TASK_S == "blobs" ? MlpTask::KIND_BLOBS : MlpTask::KIND_TEACHER

# ---- policy parsing. Tokens are LANE-LOCAL: chain | dfa | frozen.
# `frozen` has no franken counterpart (franken's mode 2 is mix:) —
# these two lanes share the DfaB machinery, NOT the token table. ----
def parse_mlp_policy(pol_s, n_layers)
  policy = [0]; policy.pop
  parts = pol_s.split(",")
  if parts.length > n_layers
    puts "toy-train-mlp: MLP_POLICY names " + parts.length.to_s +
         " layers but MLP_LAYERS=" + n_layers.to_s +
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
        puts "toy-train-mlp: unknown MLP_POLICY token " + tk + " (chain|dfa|frozen)"
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

# toy#118 — non-finite floats serialize as JSON null (finite iff x-x==0).
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

# argmax over the class axis of a downloaded logits buffer. ggml lays
# [n_classes, B] out with ne0 fastest, so sample b starts at b*classes.
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

POLICY = parse_mlp_policy(POLICY_S, N_LAYERS)

recipe = Toy::LLM::Recipes::MlpClassifier.new
recipe.realize!(D_IN, D_HIDDEN, N_LAYERS, N_CLASSES, BATCH, SEED, 1.0,
                POLICY, B_SEED, dist_code(B_DIST_S),
                scale_code(B_SCALE_S), scale_sigma(B_SCALE_S))
# tao#flow-json-emit (#25): self-describing run bundle.
ToyDescribeFlow.emit_flow_json(RUN_DIR, recipe.mc_cache.sess)

task = MlpTask.new(TASK_KIND, D_IN, N_CLASSES, D_TEACH, TASK_SEED, 1.0)
task.reset_stream!(TASK_SEED + 1)

m_x      = Mat.new(BATCH, D_IN)
m_labels = Mat.new(BATCH, N_CLASSES)
labels   = [0]; labels.pop
li0 = 0
while li0 < BATCH
  labels.push(0)
  li0 = li0 + 1
end
logit_buf = [0.0]; logit_buf.pop
lb0 = 0
while lb0 < BATCH * N_CLASSES
  logit_buf.push(0.0)
  lb0 = lb0 + 1
end

adamw = Toy::AdamW.for_from_scratch
adamw.lr = LR

# ---- the held-out set, MATERIALISED FIRST (train/val disjointness). ----
#
# The val set is drawn from the head of the sampling stream and stored;
# training then continues from where val stopped and never revisits it
# (the stream would have to wrap its whole 2^31 period first, which a
# run of this size cannot reach). Two DIFFERENTLY-SEEDED streams would
# NOT have given that: they are two offsets into the SAME LCG cycle, so
# a val stream can land inside the span the training stream later walks
# through, and then the "held-out" accuracy is measured on samples the
# model trained on. Small odds, silent when it happens, and this lane's
# entire job is to be the methodologically clean control — so it is
# disjoint by construction instead of by luck.
VAL_N = VAL_BATCHES * BATCH
val_x = [0.0]; val_x.pop
vx0 = 0
while vx0 < VAL_N * D_IN
  val_x.push(0.0)
  vx0 = vx0 + 1
end
val_y = [0]; val_y.pop
vy0 = 0
while vy0 < VAL_N
  val_y.push(0)
  vy0 = vy0 + 1
end
vfill = 0
while vfill < VAL_BATCHES
  task.fill_batch!(BATCH, m_x, labels)
  vi = 0
  while vi < BATCH
    src = vi * D_IN
    dst = (vfill * BATCH + vi) * D_IN
    vk = 0
    while vk < D_IN
      val_x[dst + vk] = m_x.flat[src + vk]
      vk = vk + 1
    end
    val_y[vfill * BATCH + vi] = labels[vi]
    vi = vi + 1
  end
  vfill = vfill + 1
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
    rs.add_str("name", "mlp")
    Toy::Events.add_provenance(rs,
      TinyNN.tnn_provenance_host_name, TinyNN.tnn_provenance_host_os,
      TinyNN.tnn_provenance_host_arch,
      TinyNN.tnn_backend_name(recipe.mc_cache.sess))
    model = Toy::Json::Builder.new
    model.add_str("arch", "mlp")
    model.add_str("name", "mlp-classifier")
    model.add_num("d_in",        D_IN)
    model.add_num("d_hidden",    D_HIDDEN)
    model.add_num("n_layers",    N_LAYERS)
    model.add_num("num_classes", N_CLASSES)
    model.add_str("act",         "silu")
    rs.add_obj("model", model)
    cost = Toy::Json::Builder.new
    cost.add_num("total_params",  recipe.mc_cache.param_count)
    cost.add_num("active_params", recipe.mc_cache.param_count)
    # Dense MLP: 2 MACs per weight per sample, forward only.
    cost.add_num("flops_per_token", 2 * recipe.mc_cache.param_count)
    rs.add_obj("cost", cost)
    config = Toy::Json::Builder.new
    config.add_num("steps",       STEPS)
    config.add_num("seed",        SEED)
    config.add_num("batch",       BATCH)
    config.add_num("val_batches", VAL_BATCHES)
    config.add_str("task",        TASK_KIND == MlpTask::KIND_BLOBS ? "blobs" : "teacher")
    config.add_num("task_seed",   TASK_SEED)
    config.add_num("teacher_dim", D_TEACH)
    config.add_raw("lr",          LR.to_s)
    config.add_num("warmup",      WARMUP)
    rs.add_obj("config", config)
    # The credit-assignment provenance. This lane is NOT franken, so it
    # does NOT reuse the "franken" key: a consumer keying on `franken`
    # would silently read an MLP run as a transformer one. Per tao#19
    # item 3, ingest keys on the align events' `wname`, not on a
    # per-lane weight-index table.
    dfa = Toy::Json::Builder.new
    dfa.add_raw("policy",  Toy::Json.from_int_array(POLICY))
    dfa.add_num("b_seed",  B_SEED)
    dfa.add_str("b_dist",  B_DIST_S.length > 0 ? B_DIST_S : "gaussian")
    dfa.add_str("b_scale", B_SCALE_S.length > 0 ? B_SCALE_S : "inv_sqrt_fan")
    dfa.add_raw("b_sigma", scale_sigma(B_SCALE_S).to_s)
    dfa.add_num("dfa_wired",  recipe.mc_cache.mlp_dfa_wired)
    dfa.add_num("frozen",     recipe.mc_cache.mlp_frozen_count)
    rs.add_obj("dfa", dfa)
    TinyNN.tnn_events_emit(rs.dump)
  else
    puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
  end
end

# ---- align telemetry buffers (sized once, like the franken lane). ----
n_align = recipe.mc_cache.mlp_align_grads.length
abuf = [0.0]; abuf.pop
gbuf = [0.0]; gbuf.pop
if ALIGN_ON && n_align > 0
  nmax = 0
  ai0 = 0
  while ai0 < n_align
    nw = TinyNN.tnn_tensor_nelements(recipe.mc_cache.mlp_align_grads[ai0])
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

  task.fill_batch!(BATCH, m_x, labels)
  k = 0
  while k < BATCH * N_CLASSES
    m_labels.flat[k] = 0.0
    k = k + 1
  end
  b = 0
  while b < BATCH
    m_labels.flat[b * N_CLASSES + labels[b]] = 1.0
    b = b + 1
  end

  loss = recipe.step!(m_x, m_labels, m_hp, step == 0)
  final_loss = loss
  # The byte-gated line — to STDOUT, no decoration.
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s

  if EVENTS.length > 0
    # Train accuracy rides the logits we already computed — one bulk
    # download, no extra forward pass.
    rc_l = TinyNN.tnn_download_to_f64_array(recipe.mc_cache.sess,
             recipe.mc_cache.t_logits, logit_buf, BATCH * N_CLASSES)
    tr_acc = -1.0
    if rc_l != 0
      puts "logits download failed: step=" + (step + 1).to_s + " rc=" + rc_l.to_s
    else
      tr_acc = hits_in(logit_buf, labels, BATCH, N_CLASSES).to_f / BATCH.to_f
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
    es.add_num("samples", BATCH)
    es.add_num("wall_us", step_wall_us)
    TinyNN.tnn_events_emit(es.dump)
  end

  if ALIGN_ON && EVENTS.length > 0 && n_align > 0 && (step % ALIGN_EVERY) == 0
    ai = 0
    while ai < n_align
      nw = TinyNN.tnn_tensor_nelements(recipe.mc_cache.mlp_align_grads[ai])
      rc_g = TinyNN.tnn_download_to_f64_array(recipe.mc_cache.sess,
        recipe.mc_cache.mlp_align_grads[ai], gbuf, nw)
      rc_a = TinyNN.tnn_download_to_f64_array(recipe.mc_cache.sess,
        recipe.mc_cache.mlp_align_accs[ai], abuf, nw)
      if rc_g != 0 || rc_a != 0
        # never-mask: a failed shadow download must not masquerade as a
        # zero/stale gradient in the telemetry.
        puts "align download failed: step=" + (step + 1).to_s +
             " wname=" + recipe.mc_cache.mlp_align_wnames[ai] +
             " rc_g=" + rc_g.to_s + " rc_a=" + rc_a.to_s
      end
      ae = Toy::Json::Builder.new
      ae.add_str("kind",  "align")
      ae.add_str("phase", "train")
      ae.add_num("t",     TinyNN.tnn_events_now_seconds)
      ae.add_num("step",  step + 1)
      ae.add_num("li",    recipe.mc_cache.mlp_align_lis[ai])
      # tao#19 item 3: `li` stays the layer index, `wi` is LANE-LOCAL
      # (this lane has exactly one weight per layer, so it is 0), and
      # `wname` is the string Tao's ingest keys on.
      ae.add_num("wi",    0)
      ae.add_str("wname", recipe.mc_cache.mlp_align_wnames[ai])
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

# ---- held-out val (the metric the success bar is stated in). ----
#
# The eval runs the SAME graph at lr=0: an optimizer step at lr=0 is a
# weight no-op, so the measured weights are exactly the trained ones.
# MIRROR RULE (the toy#139/#146 class of bug): "lr=0" has to mean EVERY
# hp vector the graph can reach. This lane has exactly ONE — if a
# per-layer or per-optimizer vector is ever added here, it must be
# zeroed in this loop too, or the val split silently becomes training
# data. The val stream is re-seeded so every arm sees the SAME val set.
val_hp = Mat.new(1, 7)
val_hp.flat[0] = 0.0
val_hp.flat[1] = adamw.beta1
val_hp.flat[2] = adamw.beta2
val_hp.flat[3] = adamw.eps
val_hp.flat[4] = 0.0
val_hp.flat[5] = adamw.beta1
val_hp.flat[6] = adamw.beta2
val_hits = 0
val_seen = 0
val_loss_sum = 0.0
vb = 0
while vb < VAL_BATCHES
  # Replay the materialised held-out set (never re-drawn: it must be
  # the SAME samples for every arm and every step it is measured at).
  vr = 0
  while vr < BATCH
    src = (vb * BATCH + vr) * D_IN
    dst = vr * D_IN
    vc = 0
    while vc < D_IN
      m_x.flat[dst + vc] = val_x[src + vc]
      vc = vc + 1
    end
    labels[vr] = val_y[vb * BATCH + vr]
    vr = vr + 1
  end
  k2 = 0
  while k2 < BATCH * N_CLASSES
    m_labels.flat[k2] = 0.0
    k2 = k2 + 1
  end
  b2 = 0
  while b2 < BATCH
    m_labels.flat[b2 * N_CLASSES + labels[b2]] = 1.0
    b2 = b2 + 1
  end
  vloss = recipe.step!(m_x, m_labels, val_hp, false)
  val_loss_sum = val_loss_sum + vloss
  rc_v = TinyNN.tnn_download_to_f64_array(recipe.mc_cache.sess,
           recipe.mc_cache.t_logits, logit_buf, BATCH * N_CLASSES)
  if rc_v != 0
    puts "toy-train-mlp: val logits download failed: rc=" + rc_v.to_s
    exit 1
  end
  val_hits = val_hits + hits_in(logit_buf, labels, BATCH, N_CLASSES)
  val_seen = val_seen + BATCH
  vb = vb + 1
end
val_acc  = val_hits.to_f / val_seen.to_f
val_loss = val_loss_sum / VAL_BATCHES.to_f
puts "val: acc=" + val_acc.to_s + " loss=" + val_loss.to_s + " n=" + val_seen.to_s

if EVENTS.length > 0 && TinyNN.tnn_events_active == 1
  ev = Toy::Json::Builder.new
  ev.add_str("kind",  "eval")
  ev.add_str("phase", "eval")
  ev.add_num("t",     TinyNN.tnn_events_now_seconds)
  ev.add_str("name",  "val")
  ev.add_num("n",     val_seen)
  ev.add_raw("accuracy", num_or_null(val_acc))
  ev.add_raw("loss",     num_or_null(val_loss))
  TinyNN.tnn_events_emit(ev.dump)

  re = Toy::Json::Builder.new
  re.add_str("kind", "run_end")
  re.add_num("t",          TinyNN.tnn_events_now_seconds)
  re.add_str("ended_at",   TinyNN.tnn_events_iso8601_now)
  re.add_str("reason",     "completed")
  re.add_num("final_step", STEPS)
  re.add_raw("final_loss", num_or_null(final_loss))
  re.add_raw("val_acc",    num_or_null(val_acc))
  # No checkpoint: there is no MLP GGUF writer and no infer consumer for
  # this arch — the anchor's product is the metric, not the weights.
  re.add_str("checkpoint", "none")
  re.add_raw("exit_code",  "0")
  TinyNN.tnn_events_emit(re.dump)
  TinyNN.tnn_events_close
end
