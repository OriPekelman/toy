# lib/toy/run/train_gtx.rb — Spinel-compiled graph-transformer training
# runner (-> libexec/toy-train-gtx), the toy#160 / DFA-arch T4 lane.
#
# WHAT THIS LANE IS FOR. The DFA program has exactly one unresolved
# negative left — the transformer LM — and it CONFOUNDS two things.
# Every LM run was DFA at a ~50k-vocab output, and F13/F18 established
# that DFA's alignment degrades as the OUTPUT DIMENSION grows. So the
# negative could be about attention, or about the vocab, and nobody has
# separated them. This lane removes the output-dim half: a transformer
# whose attention mask IS an adjacency and whose head is a 16-class
# relation classifier, asking whether ATTENTION ITSELF is DFA-hostile.
#
# Modelled on Graph Language Models (Plenz & Frank, arXiv:2401.07105),
# whose gGLM is a graph transformer with a 17-relation head.
#
# CPU-ONLY (tao#18). Own compilation unit (landmine #16).
#
# ENV CONTRACT:
#   STEPS / SEED / TAO_RUN_DIR / TOY_RUN_ID   — as every other runner
#   GTX_POLICY      — per-BLOCK tokens: chain | dfa | frozen
#   GTX_DFA_CUT     — layer (default) | step. `layer` cuts the block
#                     boundary and taps the block output, BP intact
#                     INSIDE the block (attention included). `step` also
#                     detaches the attention PROBABILITIES, so no
#                     gradient crosses the token-mixing, and taps Q and K
#                     with random feedback so they still learn.
#   GTX_BLOCKS      — transformer blocks (default 2)
#   GTX_D_MODEL     — residual width (default 64)
#   GTX_HEADS       — attention heads (default 4)
#   GTX_D_FF        — FFN width (default 128)
#   GTX_ENTITIES    — entity nodes (default 48)
#   GTX_TYPES       — latent types TY; classes = TY*TY (default 4 -> 16)
#   GTX_FEATURES    — node feature dim, half key / half value (default 16)
#   GTX_NOISE       — feature noise scale (default 0.3)
#   GTX_TASK        — relational (default) | local  [toy_gtx_task.rb]
#   GTX_PAIRS       — labelled pairs per step (default 128)
#   GTX_VAL_BATCHES — held-out INSTANCES evaluated at the end (default 8).
#                     The graph TOPOLOGY is fixed but its CONTENT is
#                     redrawn every step, so held-out means fresh
#                     instances, not held-out pairs over one graph — see
#                     toy_gtx_task.rb on the memorisation shortcut a
#                     fixed graph leaves open.
#   GTX_TASK_SEED   — task seed, SEPARATE from SEED (default 7)
#   GTX_LR / GTX_WARMUP
#   GTX_B_SEED / GTX_B_DIST / GTX_B_SCALE — the DfaB feedback axes
#
# STDOUT (byte-gated): "step <N>: loss=<float>" per step, then
# "val: acc=... loss=... n=..." and "graph: nodes=<n> bytes=<b>".
#
# THE SUCCESS BAR (tao#19 item 4) is MANDATORY, and on this lane the
# FROZEN control is load-bearing in a way it is not elsewhere: the
# attention mask is structural, so a frozen random transformer is still
# a perfectly good neighbourhood averager. The task is built so that
# averaging is PROVABLY uninformative (every neighbourhood holds exactly
# one attribute of each type) — see toy_gtx_task.rb. Measured, not
# assumed: prep/gtx_gate.rb asserts frozen sits at chance.
#
# Spinel hygiene: hand-built String-concat JSON (no #{}), ENV reads as
# TOP-LEVEL constants, no Struct, while loops.

require_relative "../io/json_builder"
require_relative "../io/json"
require_relative "../io/toy_events"
require_relative "../io/toy_gtx_task"
require_relative "../llm/engine/gtx_engine"
require_relative "../llm/recipes/gtx_graph"
require_relative "../llm/adamw"
require_relative "../train/dfa_b"
require_relative "../dev/toy_describe_flow"

STEPS       = (ENV["STEPS"] || "5").to_i
SEED        = (ENV["SEED"]  || "0").to_i
TAO_RUN_DIR = ENV["TAO_RUN_DIR"] || ""
RUN_ID      = ENV["TOY_RUN_ID"]  || ""
EVENTS      = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/events.jsonl") : ""

POLICY_S    = ENV["GTX_POLICY"] || ""
CUT_S       = ENV["GTX_DFA_CUT"] || ""
N_BLOCKS    = (ENV["GTX_BLOCKS"]   || "2").to_i
D_MODEL     = (ENV["GTX_D_MODEL"]  || "64").to_i
N_HEADS     = (ENV["GTX_HEADS"]    || "4").to_i
D_FF        = (ENV["GTX_D_FF"]     || "128").to_i
N_ENTITIES  = (ENV["GTX_ENTITIES"] || "48").to_i
N_TYPES     = (ENV["GTX_TYPES"]    || "4").to_i
D_FEAT      = (ENV["GTX_FEATURES"] || "16").to_i
NOISE_S     = ENV["GTX_NOISE"] || ""
TASK_S      = ENV["GTX_TASK"] || ""
N_PAIRS     = (ENV["GTX_PAIRS"]     || "128").to_i
VAL_BATCHES = (ENV["GTX_VAL_BATCHES"] || "8").to_i
TASK_SEED   = (ENV["GTX_TASK_SEED"] || "7").to_i
LR_S        = ENV["GTX_LR"] || ""
WARMUP_S    = ENV["GTX_WARMUP"] || ""
B_SEED      = (ENV["GTX_B_SEED"] || "1234").to_i
B_DIST_S    = ENV["GTX_B_DIST"]  || ""
B_SCALE_S   = ENV["GTX_B_SCALE"] || ""

NOISE  = NOISE_S.length  > 0 ? NOISE_S.to_f  : 0.3
LR     = LR_S.length     > 0 ? LR_S.to_f     : 0.003
WARMUP = WARMUP_S.length > 0 ? WARMUP_S.to_i : 0

# ---- fail loud on every out-of-range shape (never-mask). ----
if STEPS < 1
  puts "toy-train-gtx: STEPS must be >= 1, got " + STEPS.to_s
  exit 1
end
if N_BLOCKS < 1
  puts "toy-train-gtx: GTX_BLOCKS must be >= 1, got " + N_BLOCKS.to_s
  exit 1
end
if N_HEADS < 1
  puts "toy-train-gtx: GTX_HEADS must be >= 1, got " + N_HEADS.to_s
  exit 1
end
if D_MODEL < N_HEADS || D_MODEL % N_HEADS != 0
  puts "toy-train-gtx: GTX_D_MODEL must be a positive multiple of GTX_HEADS" +
       " (a head slice of width 0 would make attention a no-op), got " +
       D_MODEL.to_s + " with " + N_HEADS.to_s + " heads"
  exit 1
end
if D_FF < 1
  puts "toy-train-gtx: GTX_D_FF must be >= 1, got " + D_FF.to_s
  exit 1
end
if N_TYPES < 2
  puts "toy-train-gtx: GTX_TYPES must be >= 2 (one type means one class" +
       " and there is nothing to classify), got " + N_TYPES.to_s
  exit 1
end
if N_ENTITIES < N_TYPES
  puts "toy-train-gtx: GTX_ENTITIES must be >= GTX_TYPES, got " +
       N_ENTITIES.to_s + " with " + N_TYPES.to_s + " types"
  exit 1
end
if D_FEAT < 4 || D_FEAT % 2 != 0
  puts "toy-train-gtx: GTX_FEATURES must be an EVEN integer >= 4 — the" +
       " first half is the retrieval KEY and the second half the VALUE," +
       " got " + D_FEAT.to_s
  exit 1
end
if N_PAIRS < 1
  puts "toy-train-gtx: GTX_PAIRS must be >= 1, got " + N_PAIRS.to_s
  exit 1
end
if VAL_BATCHES < 1
  puts "toy-train-gtx: GTX_VAL_BATCHES must be >= 1, got " + VAL_BATCHES.to_s
  exit 1
end
if CUT_S.length > 0 && CUT_S != "layer" && CUT_S != "step"
  puts "toy-train-gtx: GTX_DFA_CUT " + CUT_S + " unsupported (layer|step)"
  exit 1
end
if TASK_S.length > 0 && TASK_S != "relational" && TASK_S != "local"
  puts "toy-train-gtx: GTX_TASK " + TASK_S + " unsupported (relational|local)"
  exit 1
end

TASK_KIND = TASK_S == "local" ? GtxTask::KIND_LOCAL : GtxTask::KIND_RELATIONAL
DFA_CUT   = CUT_S == "step" ? Toy::LLM::Engine::GtxEngine::CUT_STEP :
                              Toy::LLM::Engine::GtxEngine::CUT_LAYER
N_CLASSES = N_TYPES * N_TYPES
# degree == TY is not a tunable: one attribute of EACH type per
# neighbourhood is what makes mean-pooling provably uninformative, which
# is what lets the frozen control lose. See toy_gtx_task.rb.
DEGREE    = N_TYPES

def parse_gtx_policy(pol_s, n_blocks)
  policy = [0]; policy.pop
  parts = pol_s.split(",")
  if parts.length > n_blocks
    puts "toy-train-gtx: GTX_POLICY names " + parts.length.to_s +
         " blocks but GTX_BLOCKS=" + n_blocks.to_s +
         " — a policy token for a block that does not exist would silently do nothing"
    exit 1
  end
  i = 0
  while i < n_blocks
    m = 0
    if i < parts.length
      tk = parts[i]
      if tk == "dfa"
        m = 1
      elsif tk == "frozen"
        m = 2
      elsif tk != "chain" && tk.length > 0
        puts "toy-train-gtx: unknown GTX_POLICY token " + tk + " (chain|dfa|frozen)"
        exit 1
      end
    end
    policy.push(m)
    i = i + 1
  end
  # A `dfa` block detaches its INPUT, so nothing below it receives a
  # gradient from above. A `chain` block underneath one would be
  # silently frozen while reporting itself as trained.
  seen_dfa = false
  j = n_blocks - 1
  while j >= 0
    if policy[j] == 1
      seen_dfa = true
    elsif policy[j] == 0 && seen_dfa
      puts "toy-train-gtx: GTX_POLICY has a `chain` block BELOW a `dfa` block" +
           " (block " + j.to_s + "). A dfa block detaches its input, so that" +
           " chain block would receive no gradient at all and be silently" +
           " frozen while reporting itself as trained. Use `frozen` if that is" +
           " what you meant, or make the lower blocks dfa too."
      exit 1
    end
    j = j - 1
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

POLICY = parse_gtx_policy(POLICY_S, N_BLOCKS)

task = GtxTask.new(TASK_KIND, D_FEAT, N_ENTITIES, N_TYPES, DEGREE,
                   TASK_SEED, NOISE)
N_NODES = task.gt_nodes

recipe = Toy::LLM::Recipes::GtxGraph.new
recipe.realize!(D_FEAT, D_MODEL, N_HEADS, D_FF, N_BLOCKS, N_NODES,
                N_PAIRS, N_CLASSES, SEED, 1.0, POLICY, DFA_CUT, B_SEED,
                dist_code(B_DIST_S), scale_code(B_SCALE_S),
                scale_sigma(B_SCALE_S))
ToyDescribeFlow.emit_flow_json(TAO_RUN_DIR, recipe.gr_cache.sess)

# ---- the TOPOLOGY is constant; the CONTENT is not. ----
# The adjacency mask is uploaded once. Node features are redrawn every
# step, because a fixed instance is memorisable and a memorised instance
# does not need the retrieval this lane exists to measure.
x_flat = Array.new(N_NODES * D_FEAT, 0.0)
mask_flat = Array.new(N_NODES * N_NODES, 0.0)
task.fill_mask!(mask_flat)
recipe.upload_mask!(mask_flat)

# The pair stream. Val pairs are materialised FIRST and training
# continues from where val stopped (toy#152's discipline): two
# differently-seeded streams are two offsets into the same LCG cycle and
# a val stream can land inside the span training later walks.
idx_a  = Array.new(N_PAIRS, 0)
idx_b  = Array.new(N_PAIRS, 0)
labels = Array.new(N_PAIRS, 0)
m_labels  = Mat.new(N_PAIRS, N_CLASSES)
logit_buf = Array.new(N_PAIRS * N_CLASSES, 0.0)

# The held-out set is MATERIALISED FIRST — whole instances, features
# included — and training then continues from where val stopped. Same
# discipline as toy#152: two differently-seeded streams are two offsets
# into one LCG cycle, so a val stream can land inside the span training
# later walks.
task.reset_stream!(TASK_SEED + 1)
VAL_N  = VAL_BATCHES * N_PAIRS
val_x  = Array.new(VAL_BATCHES * N_NODES * D_FEAT, 0.0)
val_a  = Array.new(VAL_N, 0)
val_b  = Array.new(VAL_N, 0)
val_y  = Array.new(VAL_N, 0)
vfill = 0
while vfill < VAL_BATCHES
  task.resample!
  task.fill_features!(x_flat)
  vbase = vfill * N_NODES * D_FEAT
  vi = 0
  while vi < N_NODES * D_FEAT
    val_x[vbase + vi] = x_flat[vi]
    vi = vi + 1
  end
  task.fill_pairs!(N_PAIRS, idx_a, idx_b, labels)
  vp = 0
  while vp < N_PAIRS
    val_a[vfill * N_PAIRS + vp] = idx_a[vp]
    val_b[vfill * N_PAIRS + vp] = idx_b[vp]
    val_y[vfill * N_PAIRS + vp] = labels[vp]
    vp = vp + 1
  end
  vfill = vfill + 1
end

# The pair->node incidence the DFA route reads. Kept allocated and
# patched in place: only 2 * N_PAIRS entries are ever non-zero, so
# rewriting the whole [P, N] array every step would cost more Ruby than
# the step itself.
inc_flat = Array.new(N_PAIRS * N_NODES, 0.0)
def set_incidence!(inc, prev_a, prev_b, idx_a, idx_b, n_pairs, n_nodes, first)
  p = 0
  while p < n_pairs
    if !first
      inc[p * n_nodes + prev_a[p]] = 0.0
      inc[p * n_nodes + prev_b[p]] = 0.0
    end
    inc[p * n_nodes + idx_a[p]] = 1.0
    inc[p * n_nodes + idx_b[p]] = 1.0
    prev_a[p] = idx_a[p]
    prev_b[p] = idx_b[p]
    p = p + 1
  end
  nil
end
prev_a = Array.new(N_PAIRS, 0)
prev_b = Array.new(N_PAIRS, 0)

adamw = Toy::AdamW.for_from_scratch
adamw.lr = LR

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
    rs.add_str("name", "gtx")
    Toy::Events.add_provenance(rs,
      TinyNN.tnn_provenance_host_name, TinyNN.tnn_provenance_host_os,
      TinyNN.tnn_provenance_host_arch,
      TinyNN.tnn_backend_name(recipe.gr_cache.sess))
    model = Toy::Json::Builder.new
    model.add_str("arch", "gtx")
    model.add_str("name", "graph-transformer")
    model.add_num("d_in",        D_FEAT)
    model.add_num("d_model",     D_MODEL)
    model.add_num("n_heads",     N_HEADS)
    model.add_num("d_ff",        D_FF)
    model.add_num("n_blocks",    N_BLOCKS)
    model.add_num("n_nodes",     N_NODES)
    model.add_num("num_classes", N_CLASSES)
    model.add_str("attn_mask",   "adjacency")
    model.add_str("readout",     "node_pair")
    rs.add_obj("model", model)
    cost = Toy::Json::Builder.new
    cost.add_num("total_params",  recipe.gr_cache.param_count)
    cost.add_num("active_params", recipe.gr_cache.param_count)
    cost.add_num("flops_per_token", 2 * recipe.gr_cache.param_count)
    cost.add_num("graph_nodes", recipe.gr_cache.gx_graph_nodes)
    cost.add_num("graph_bytes", recipe.gr_cache.gx_graph_bytes)
    rs.add_obj("cost", cost)
    config = Toy::Json::Builder.new
    config.add_num("steps",      STEPS)
    config.add_num("seed",       SEED)
    config.add_num("pairs",      N_PAIRS)

    config.add_str("task",       TASK_KIND == GtxTask::KIND_LOCAL ? "local" : "relational")
    config.add_num("task_seed",  TASK_SEED)
    config.add_num("entities",    N_ENTITIES)
    config.add_num("val_batches", VAL_BATCHES)
    config.add_num("types",      N_TYPES)
    config.add_num("degree",     DEGREE)
    config.add_raw("noise",      NOISE.to_s)
    config.add_raw("lr",         LR.to_s)
    config.add_num("warmup",     WARMUP)
    rs.add_obj("config", config)
    dfa = Toy::Json::Builder.new
    dfa.add_raw("policy",  Toy::Json.from_int_array(POLICY))
    dfa.add_str("cut",     CUT_S == "step" ? "step" : "layer")
    dfa.add_num("b_seed",  B_SEED)
    dfa.add_str("b_dist",  B_DIST_S.length > 0 ? B_DIST_S : "gaussian")
    dfa.add_str("b_scale", B_SCALE_S.length > 0 ? B_SCALE_S : "inv_sqrt_fan")
    dfa.add_raw("b_sigma", scale_sigma(B_SCALE_S).to_s)
    dfa.add_num("dfa_wired", recipe.gr_cache.gx_dfa_wired)
    dfa.add_num("frozen",    recipe.gr_cache.gx_frozen_count)
    dfa.add_num("taps",      recipe.gr_cache.gx_taps)
    dfa.add_str("route",     "pair_incidence")
    rs.add_obj("dfa", dfa)
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

  task.resample!
  task.fill_features!(x_flat)
  recipe.upload_features!(x_flat)
  task.fill_pairs!(N_PAIRS, idx_a, idx_b, labels)
  set_incidence!(inc_flat, prev_a, prev_b, idx_a, idx_b, N_PAIRS, N_NODES, step == 0)
  k = 0
  while k < N_PAIRS * N_CLASSES
    m_labels.flat[k] = 0.0
    k = k + 1
  end
  b = 0
  while b < N_PAIRS
    m_labels.flat[b * N_CLASSES + labels[b]] = 1.0
    b = b + 1
  end

  loss = recipe.step!(idx_a, idx_b, inc_flat, m_labels, m_hp, step == 0)
  final_loss = loss
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s

  if EVENTS.length > 0
    rc_l = TinyNN.tnn_download_to_f64_array(recipe.gr_cache.sess,
             recipe.gr_cache.t_logits, logit_buf, N_PAIRS * N_CLASSES)
    tr_acc = -1.0
    if rc_l != 0
      puts "logits download failed: step=" + (step + 1).to_s + " rc=" + rc_l.to_s
    else
      tr_acc = hits_in(logit_buf, labels, N_PAIRS, N_CLASSES).to_f / N_PAIRS.to_f
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
    es.add_num("samples", N_PAIRS)
    es.add_num("wall_us", step_wall_us)
    TinyNN.tnn_events_emit(es.dump)
  end
  step = step + 1
end

# ---- held-out val. ----
# MIRROR RULE (toy#139/#146): "lr=0" has to mean EVERY hp vector the
# graph can reach. This lane has exactly ONE — if a per-block or
# per-optimizer vector is ever added it must be zeroed here too, or the
# val split silently becomes training data.
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
  vsrc = vb * N_NODES * D_FEAT
  vx = 0
  while vx < N_NODES * D_FEAT
    x_flat[vx] = val_x[vsrc + vx]
    vx = vx + 1
  end
  recipe.upload_features!(x_flat)
  vi = 0
  while vi < N_PAIRS
    idx_a[vi]  = val_a[vb * N_PAIRS + vi]
    idx_b[vi]  = val_b[vb * N_PAIRS + vi]
    labels[vi] = val_y[vb * N_PAIRS + vi]
    vi = vi + 1
  end
  set_incidence!(inc_flat, prev_a, prev_b, idx_a, idx_b, N_PAIRS, N_NODES, false)
  k2 = 0
  while k2 < N_PAIRS * N_CLASSES
    m_labels.flat[k2] = 0.0
    k2 = k2 + 1
  end
  b2 = 0
  while b2 < N_PAIRS
    m_labels.flat[b2 * N_CLASSES + labels[b2]] = 1.0
    b2 = b2 + 1
  end
  vloss = recipe.step!(idx_a, idx_b, inc_flat, m_labels, val_hp, false)
  val_loss_sum = val_loss_sum + vloss
  rc_v = TinyNN.tnn_download_to_f64_array(recipe.gr_cache.sess,
           recipe.gr_cache.t_logits, logit_buf, N_PAIRS * N_CLASSES)
  if rc_v != 0
    puts "toy-train-gtx: val logits download failed: rc=" + rc_v.to_s
    exit 1
  end
  val_hits = val_hits + hits_in(logit_buf, labels, N_PAIRS, N_CLASSES)
  val_seen = val_seen + N_PAIRS
  vb = vb + 1
end
val_acc  = val_hits.to_f / val_seen.to_f
val_loss = val_loss_sum / VAL_BATCHES.to_f
puts "val: acc=" + val_acc.to_s + " loss=" + val_loss.to_s + " n=" + val_seen.to_s
puts "graph: nodes=" + recipe.gr_cache.gx_graph_nodes.to_s +
     " bytes=" + recipe.gr_cache.gx_graph_bytes.to_s

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
  re.add_str("checkpoint", "none")
  re.add_raw("exit_code",  "0")
  TinyNN.tnn_events_emit(re.dump)
  TinyNN.tnn_events_close
end
