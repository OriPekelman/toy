# lib/toy/run/train_ssm.rb — Spinel-compiled selective-scan (Mamba-lite)
# training runner (-> libexec/toy-train-ssm), the toy#155 / DFA-arch T2
# lane, and the program's TOP NOVELTY pick: there is no DFA /
# forward-forward / DRTP precedent on Mamba/S4/S5 at all.
#
# WHAT MAKES IT THE STRUCTURAL FIT. The SSM core is a LINEAR RECURRENCE,
# so a random-feedback error injected per step composes through the same
# linear operator — random feedback is mathematically natural here in a
# way it is not for attention. And the value prop is concrete: BPTT's
# backward memory scales with sequence length, while a per-step DFA
# update needs only the error and that step's own activations.
#
# CPU-ONLY this slice (tao#18). The deferred exception in tao#19 is
# exactly this lane: the small-output POSITIVE is CPU-fine, and a CUDA
# twin arrives only if/when the long-sequence memory measurement does.
# Own compilation unit (landmine #16).
#
# ENV CONTRACT:
#   STEPS / SEED / TAO_RUN_DIR / TOY_RUN_ID   — as every other runner
#   SSM_POLICY     — per-LAYER tokens: chain | dfa | frozen
#                    (default: all chain, i.e. full BPTT)
#   SSM_SELECTION  — selective (default) | lti   [the ticket's control]
#   SSM_DFA_CUT    — layer (default) | step. `layer` cuts only the layer
#                    boundary and injects once at the readout step, with
#                    BPTT intact inside the layer (toy#158's macro-DFA,
#                    transposed to a recurrent block). `step` also cuts
#                    every timestep, which is what the ticket's
#                    kill-BPTT-memory value prop requires.
#   SSM_LAYERS     — recurrence layers (default 2)
#   SSM_D_MODEL    — residual width (default 24)
#   SSM_D_INNER    — recurrence width (default 48)
#   SSM_SEQ        — sequence length T (default 64)
#   SSM_CONV_K     — causal depthwise conv width (default 4)
#   SSM_CLASSES    — output dim (default 4) — small, by design
#   SSM_TASK       — cue (default) | mean   [see toy_ssm_task.rb]
#   SSM_CUE_SPAN   — cue drawn from the first N steps (default T/4)
#   SSM_NOISE      — distractor scale (default 1.0)
#   SSM_TASK_SEED  — task seed, SEPARATE from SEED (default 7)
#   SSM_BATCH      — sequences per step (default 32)
#   SSM_VAL_BATCHES— held-out batches evaluated at the end (default 8)
#   SSM_DT_INIT    — dt_bias init; sets the initial per-step decay
#                    (default -5.0 => decay 0.993). See ssm_engine.rb.
#   SSM_LR / SSM_WARMUP
#   SSM_B_SEED / SSM_B_DIST / SSM_B_SCALE — the DfaB feedback axes
#
# STDOUT (byte-gated): "step <N>: loss=<float>" per step, then one
# "val: acc=<a> loss=<l> n=<n>" line, then one "graph: nodes=<n>" line.
#
# NO --align-events ON THIS LANE, and that is structural rather than an
# omission: the DFA update arrives through autodiff from the surrogate
# roots, so it lands in the SAME accumulator a BP run would use and
# there is no second tensor to take a cosine against. toy#158 hit this
# first on macro-DFA; the answer there and here is to gate on "the B
# seed moves the curve", which is the assertion that actually catches a
# silently-unwired build.
#
# THE SUCCESS BAR (tao#19 item 4) is MANDATORY here as in every lane:
#     positive = all-DFA within the stated gap of all-BP
#                AND provably beating the frozen control,
#                at matched init and matched seed.
#
# Spinel hygiene: hand-built String-concat JSON (no #{}), ENV reads as
# TOP-LEVEL constants (a constant assigned inside a conditional arm
# reads back empty at runtime), no Struct, while loops.

require_relative "../io/json_builder"
require_relative "../io/json"
require_relative "../io/toy_events"
require_relative "../io/toy_ssm_task"
require_relative "../llm/engine/ssm_engine"
require_relative "../llm/recipes/ssm_seq"
require_relative "../llm/adamw"
require_relative "../train/dfa_b"
require_relative "../dev/toy_describe_flow"

STEPS       = (ENV["STEPS"] || "5").to_i
SEED        = (ENV["SEED"]  || "0").to_i
TAO_RUN_DIR = ENV["TAO_RUN_DIR"] || ""
RUN_ID      = ENV["TOY_RUN_ID"]  || ""
EVENTS      = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/events.jsonl") : ""

POLICY_S    = ENV["SSM_POLICY"] || ""
SELECT_S    = ENV["SSM_SELECTION"] || ""
CUT_S       = ENV["SSM_DFA_CUT"] || ""
N_LAYERS    = (ENV["SSM_LAYERS"]  || "2").to_i
D_MODEL     = (ENV["SSM_D_MODEL"] || "24").to_i
D_INNER     = (ENV["SSM_D_INNER"] || "48").to_i
SEQ_T       = (ENV["SSM_SEQ"]     || "64").to_i
CONV_K      = (ENV["SSM_CONV_K"]  || "4").to_i
N_CLASSES   = (ENV["SSM_CLASSES"] || "4").to_i
TASK_S      = ENV["SSM_TASK"] || ""
CUE_SPAN_S  = ENV["SSM_CUE_SPAN"] || ""
NOISE_S     = ENV["SSM_NOISE"] || ""
TASK_SEED   = (ENV["SSM_TASK_SEED"] || "7").to_i
BATCH       = (ENV["SSM_BATCH"] || "32").to_i
VAL_BATCHES = (ENV["SSM_VAL_BATCHES"] || "8").to_i
DT_INIT_S   = ENV["SSM_DT_INIT"] || ""
LR_S        = ENV["SSM_LR"] || ""
WARMUP      = (ENV["SSM_WARMUP"] || "0").to_i
B_SEED      = (ENV["SSM_B_SEED"] || "1234").to_i
B_DIST_S    = ENV["SSM_B_DIST"]  || ""
B_SCALE_S   = ENV["SSM_B_SCALE"] || ""

NOISE   = NOISE_S.length   > 0 ? NOISE_S.to_f   : 1.0
DT_INIT = DT_INIT_S.length > 0 ? DT_INIT_S.to_f : -5.0
# 0.003. Measured on this lane's own arms (default cell, 600 steps,
# seed 0, val accuracy, selective + the layer cut):
#   lr 0.001: BP .930  DFA .922
#   lr 0.003: BP 1.000 DFA .996    <- the default
#   lr 0.010: BP 1.000 DFA .996
# The STEP cut was swept separately and much wider (3e-5 .. 1e-2): its
# best cell is .355, against a .227 frozen control. That is a property
# of the cut, not of the learning rate, and the docs say so.
LR      = LR_S.length      > 0 ? LR_S.to_f      : 0.003

# ---- fail loud on every out-of-range shape (never-mask). ----
if STEPS < 1
  puts "toy-train-ssm: STEPS must be >= 1, got " + STEPS.to_s
  exit 1
end
if N_LAYERS < 1
  puts "toy-train-ssm: SSM_LAYERS must be >= 1, got " + N_LAYERS.to_s
  exit 1
end
if D_MODEL < 2 || D_INNER < 1
  puts "toy-train-ssm: SSM_D_MODEL must be >= 2 and SSM_D_INNER >= 1"
  exit 1
end
if SEQ_T < 2
  puts "toy-train-ssm: SSM_SEQ must be >= 2, got " + SEQ_T.to_s
  exit 1
end
if CONV_K < 1 || CONV_K > 16
  puts "toy-train-ssm: SSM_CONV_K must be in 1..16, got " + CONV_K.to_s
  exit 1
end
if N_CLASSES < 2
  puts "toy-train-ssm: SSM_CLASSES must be >= 2, got " + N_CLASSES.to_s
  exit 1
end
if BATCH < 1
  puts "toy-train-ssm: SSM_BATCH must be >= 1, got " + BATCH.to_s
  exit 1
end
if VAL_BATCHES < 1
  puts "toy-train-ssm: SSM_VAL_BATCHES must be >= 1, got " + VAL_BATCHES.to_s
  exit 1
end
if SELECT_S.length > 0 && SELECT_S != "selective" && SELECT_S != "lti"
  puts "toy-train-ssm: SSM_SELECTION " + SELECT_S + " unsupported (selective|lti)"
  exit 1
end
if CUT_S.length > 0 && CUT_S != "layer" && CUT_S != "step"
  puts "toy-train-ssm: SSM_DFA_CUT " + CUT_S + " unsupported (layer|step)"
  exit 1
end
if TASK_S.length > 0 && TASK_S != "cue" && TASK_S != "mean"
  puts "toy-train-ssm: SSM_TASK " + TASK_S + " unsupported (cue|mean)"
  exit 1
end

TASK_KIND = TASK_S == "mean" ? SsmTask::KIND_MEAN : SsmTask::KIND_CUE
SELECTION = SELECT_S == "lti" ? Toy::LLM::Engine::SsmEngine::SELECT_LTI :
                                Toy::LLM::Engine::SsmEngine::SELECT_SELECTIVE
DFA_CUT   = CUT_S == "step" ? Toy::LLM::Engine::SsmEngine::CUT_STEP :
                              Toy::LLM::Engine::SsmEngine::CUT_LAYER
cs_raw    = CUE_SPAN_S.length > 0 ? CUE_SPAN_S.to_i : SEQ_T / 4
CUE_SPAN  = cs_raw < 1 ? 1 : cs_raw
if CUE_SPAN >= SEQ_T
  puts "toy-train-ssm: SSM_CUE_SPAN must be < SSM_SEQ (the cue has to be" +
       " strictly before the last-step readout or there is no delay to" +
       " carry), got " + CUE_SPAN.to_s + " with SSM_SEQ=" + SEQ_T.to_s
  exit 1
end

# ---- policy parsing. Tokens are LANE-LOCAL: chain | dfa | frozen. ----
def parse_ssm_policy(pol_s, n_layers)
  policy = [0]; policy.pop
  parts = pol_s.split(",")
  if parts.length > n_layers
    puts "toy-train-ssm: SSM_POLICY names " + parts.length.to_s +
         " layers but SSM_LAYERS=" + n_layers.to_s +
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
        puts "toy-train-ssm: unknown SSM_POLICY token " + tk + " (chain|dfa|frozen)"
        exit 1
      end
    end
    policy.push(m)
    i = i + 1
  end
  # A `dfa` layer detaches its INPUT, so nothing below it receives a
  # gradient from above. A `chain` layer underneath one would therefore
  # be silently frozen while still reporting itself as trained — which
  # is exactly the class of bug that produces a confident wrong finding.
  # Reject it instead of quietly running it.
  seen_dfa = false
  j = n_layers - 1
  while j >= 0
    if policy[j] == 1
      seen_dfa = true
    elsif policy[j] == 0 && seen_dfa
      puts "toy-train-ssm: SSM_POLICY has a `chain` layer BELOW a `dfa` layer" +
           " (layer " + j.to_s + "). A dfa layer detaches its input, so that" +
           " chain layer would receive no gradient at all and be silently" +
           " frozen while reporting itself as trained. Use `frozen` if that is" +
           " what you meant, or make the lower layers dfa too."
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

# argmax over the class axis. ggml lays [n_classes, B] out with ne0
# fastest, so sample b starts at b*classes.
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

POLICY = parse_ssm_policy(POLICY_S, N_LAYERS)

recipe = Toy::LLM::Recipes::SsmSeq.new
recipe.realize!(D_MODEL, D_INNER, SEQ_T, BATCH, N_CLASSES, N_LAYERS,
                CONV_K, SELECTION, SEED, 1.0, DT_INIT,
                POLICY, DFA_CUT, B_SEED, dist_code(B_DIST_S),
                scale_code(B_SCALE_S), scale_sigma(B_SCALE_S))
# tao#flow-json-emit (#25): self-describing run bundle.
ToyDescribeFlow.emit_flow_json(TAO_RUN_DIR, recipe.sq_cache.sess)

task = SsmTask.new(TASK_KIND, D_MODEL, SEQ_T, N_CLASSES, CUE_SPAN,
                   TASK_SEED, NOISE)
task.reset_stream!(TASK_SEED + 1)

SEQ_FLOATS = SEQ_T * BATCH * D_MODEL
x_flat = Array.new(SEQ_FLOATS, 0.0)
m_labels = Mat.new(BATCH, N_CLASSES)
labels = Array.new(BATCH, 0)
logit_buf = Array.new(BATCH * N_CLASSES, 0.0)

adamw = Toy::AdamW.for_from_scratch
adamw.lr = LR

# ---- the held-out set, MATERIALISED FIRST (train/val disjointness). ----
# Same discipline as toy#152: the val set is drawn from the HEAD of the
# sampling stream and stored, and training continues from where val
# stopped. Two differently-seeded streams would NOT give that — they are
# two offsets into the same LCG cycle, so a val stream can land inside
# the span the training stream later walks.
VAL_N = VAL_BATCHES * BATCH
val_x = Array.new(VAL_BATCHES * SEQ_FLOATS, 0.0)
val_y = Array.new(VAL_N, 0)
vfill = 0
while vfill < VAL_BATCHES
  task.fill_batch!(BATCH, x_flat, labels)
  vb0 = vfill * SEQ_FLOATS
  vi = 0
  while vi < SEQ_FLOATS
    val_x[vb0 + vi] = x_flat[vi]
    vi = vi + 1
  end
  vl = 0
  while vl < BATCH
    val_y[vfill * BATCH + vl] = labels[vl]
    vl = vl + 1
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
    rs.add_str("name", "ssm")
    Toy::Events.add_provenance(rs,
      TinyNN.tnn_provenance_host_name, TinyNN.tnn_provenance_host_os,
      TinyNN.tnn_provenance_host_arch,
      TinyNN.tnn_backend_name(recipe.sq_cache.sess))
    model = Toy::Json::Builder.new
    model.add_str("arch", "ssm")
    model.add_str("name", "mamba-lite")
    model.add_num("d_model",     D_MODEL)
    model.add_num("d_inner",     D_INNER)
    model.add_num("n_layers",    N_LAYERS)
    model.add_num("seq_len",     SEQ_T)
    model.add_num("conv_k",      CONV_K)
    model.add_num("num_classes", N_CLASSES)
    model.add_str("selection",   SELECT_S == "lti" ? "lti" : "selective")
    model.add_str("readout",     "last_step")
    rs.add_obj("model", model)
    cost = Toy::Json::Builder.new
    cost.add_num("total_params",  recipe.sq_cache.param_count)
    cost.add_num("active_params", recipe.sq_cache.param_count)
    cost.add_num("flops_per_token", 2 * recipe.sq_cache.param_count)
    # The activation-memory proxy this lane can measure HONESTLY. It is
    # the realized graph's node count, not bytes and not peak RSS: in a
    # graph autodiff every forward tensor is materialised whatever the
    # credit rule, so the streaming memory win DFA promises is NOT
    # visible here and this number must not be read as if it were. What
    # it does show is how the two arms' graphs scale with SSM_SEQ.
    cost.add_num("graph_nodes", recipe.sq_cache.ssm_graph_nodes)
    rs.add_obj("cost", cost)
    config = Toy::Json::Builder.new
    config.add_num("steps",       STEPS)
    config.add_num("seed",        SEED)
    config.add_num("batch",       BATCH)
    config.add_num("val_batches", VAL_BATCHES)
    config.add_str("task",        TASK_KIND == SsmTask::KIND_MEAN ? "mean" : "cue")
    config.add_num("task_seed",   TASK_SEED)
    config.add_num("cue_span",    CUE_SPAN)
    config.add_raw("noise",       NOISE.to_s)
    config.add_raw("dt_init",     DT_INIT.to_s)
    config.add_raw("lr",          LR.to_s)
    config.add_num("warmup",      WARMUP)
    rs.add_obj("config", config)
    # NOT the "franken" key — a consumer keying on `franken` would read
    # an SSM run as a transformer one (tao#19 item 3).
    dfa = Toy::Json::Builder.new
    dfa.add_raw("policy",  Toy::Json.from_int_array(POLICY))
    dfa.add_str("cut",     CUT_S == "step" ? "step" : "layer")
    dfa.add_num("b_seed",  B_SEED)
    dfa.add_str("b_dist",  B_DIST_S.length > 0 ? B_DIST_S : "gaussian")
    dfa.add_str("b_scale", B_SCALE_S.length > 0 ? B_SCALE_S : "inv_sqrt_fan")
    dfa.add_raw("b_sigma", scale_sigma(B_SCALE_S).to_s)
    dfa.add_num("dfa_wired", recipe.sq_cache.ssm_dfa_wired)
    dfa.add_num("frozen",    recipe.sq_cache.ssm_frozen_count)
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

  task.fill_batch!(BATCH, x_flat, labels)
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

  loss = recipe.step!(x_flat, m_labels, m_hp, step == 0)
  final_loss = loss
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s

  if EVENTS.length > 0
    rc_l = TinyNN.tnn_download_to_f64_array(recipe.sq_cache.sess,
             recipe.sq_cache.t_logits, logit_buf, BATCH * N_CLASSES)
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
  step = step + 1
end

# ---- held-out val (the metric the success bar is stated in). ----
#
# The eval replays the materialised val set through the SAME graph at
# lr = 0 (an optimizer step at lr 0 is a weight no-op).
# MIRROR RULE (the toy#139/#146 class of bug): "lr=0" has to mean EVERY
# hp vector the graph can reach. This lane has exactly ONE — if a
# per-layer or per-optimizer vector is ever added here it must be zeroed
# in this loop too, or the val split silently becomes training data.
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
  src0 = vb * SEQ_FLOATS
  vi2 = 0
  while vi2 < SEQ_FLOATS
    x_flat[vi2] = val_x[src0 + vi2]
    vi2 = vi2 + 1
  end
  vr = 0
  while vr < BATCH
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
  vloss = recipe.step!(x_flat, m_labels, val_hp, false)
  val_loss_sum = val_loss_sum + vloss
  rc_v = TinyNN.tnn_download_to_f64_array(recipe.sq_cache.sess,
           recipe.sq_cache.t_logits, logit_buf, BATCH * N_CLASSES)
  if rc_v != 0
    puts "toy-train-ssm: val logits download failed: rc=" + rc_v.to_s
    exit 1
  end
  val_hits = val_hits + hits_in(logit_buf, labels, BATCH, N_CLASSES)
  val_seen = val_seen + BATCH
  vb = vb + 1
end
val_acc  = val_hits.to_f / val_seen.to_f
val_loss = val_loss_sum / VAL_BATCHES.to_f
puts "val: acc=" + val_acc.to_s + " loss=" + val_loss.to_s + " n=" + val_seen.to_s
# The graph size rides stdout so a sweep over SSM_SEQ can read the
# arms' scaling without opening a bundle. Read the caveat on
# cost.graph_nodes above before drawing a memory conclusion from it.
puts "graph: nodes=" + recipe.sq_cache.ssm_graph_nodes.to_s

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
  # No checkpoint: there is no SSM GGUF writer and no infer consumer for
  # this arch — the lane's product is the metric, not the weights.
  re.add_str("checkpoint", "none")
  re.add_raw("exit_code",  "0")
  TinyNN.tnn_events_emit(re.dump)
  TinyNN.tnn_events_close
end
