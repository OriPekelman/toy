# lib/toy/run/train_lstm.rb — Spinel-compiled LSTM training runner
# (-> libexec/toy-train-lstm), the toy#157 / DFA-arch T3 lane.
#
# WHAT THIS LANE IS FOR. toy#157 calls it the companion / SSM rehearsal.
# DFA on RNNs is not novel in existence (RFLO 2019; Folchini et al.
# ISC-HPC 2025 show DFA updates apply in PARALLEL across time, removing
# sequential BPTT — with BP still ahead on accuracy). The under-measured
# claim is the TRADE: "DFA matches BP accuracy at k-times-less memory
# for sequence length L". So this lane's job is to measure that trade on
# a GATED recurrence, using the same --dfa-cut axis toy#155 used, which
# makes the two lanes directly comparable.
#
# It also reuses toy#155's delayed-cue task generator UNCHANGED
# (lib/toy/io/toy_ssm_task.rb). That is deliberate: holding the task
# fixed is what lets the LSTM result be read against the selective-scan
# one as an architecture comparison rather than two separate anecdotes.
#
# CPU-ONLY (tao#18). Own compilation unit (landmine #16).
#
# ENV CONTRACT:
#   STEPS / SEED / TAO_RUN_DIR / TOY_RUN_ID   — as every other runner
#   LSTM_POLICY     — per-LAYER tokens: chain | dfa | frozen
#   LSTM_DFA_CUT    — layer (default) | step. `layer` cuts only the layer
#                     boundary and injects once at the readout step, BPTT
#                     intact inside. `step` also detaches h_{t-1} AND
#                     c_{t-1}, which is the "parallel across time" form
#                     the ticket cites and the one whose memory story is
#                     its success target.
#   LSTM_LAYERS     — stacked cells (default 1)
#   LSTM_HIDDEN     — cell width (default 64)
#   LSTM_FEATURES   — input feature dim (default 24)
#   LSTM_SEQ        — sequence length T (default 64)
#   LSTM_CLASSES    — output dim (default 4) — small, by design
#   LSTM_TASK       — cue (default) | mean   [toy_ssm_task.rb]
#   LSTM_CUE_SPAN   — cue drawn from the first N steps (default T/4)
#   LSTM_NOISE      — distractor scale (default 1.0)
#   LSTM_TASK_SEED  — task seed, SEPARATE from SEED (default 7)
#   LSTM_BATCH      — sequences per step (default 32)
#   LSTM_VAL_BATCHES— held-out batches evaluated at the end (default 8)
#   LSTM_LR         — default 0.02 (see the grid at the LR constant)
#   LSTM_WARMUP     — default 0. The lane's fair cell wants 200, and says
#                     so explicitly rather than defaulting to it; see the
#                     LR constant below for why that is not negotiable.
#   LSTM_B_SEED / LSTM_B_DIST / LSTM_B_SCALE — the DfaB feedback axes
#   LSTM_CLIP_GRAD  — global-norm gradient clipping (toy#162). OFF by
#                     default and byte-null when absent: every cell this
#                     lane has ever published was measured without it.
#
# STDOUT (byte-gated): "step <N>: loss=<float>" per step, then
# "val: acc=... loss=... n=..." and "graph: nodes=<n> bytes=<b>".
#
# THE MEMORY NUMBER IS BYTES, and it is the ticket's success target.
# toy#155 could only report node COUNT; this lane sums ggml_nbytes over
# every node of the realized graph, which is the actual materialised
# activation footprint. The caveat travels WITH the number: in a graph
# autodiff every forward tensor is materialised whatever the credit
# rule, so this measures what the harness BUILDS, not what a streaming
# implementation could get away with.
#
# NO --align-events here, for the same structural reason as toy#155: the
# DFA update arrives through autodiff from the surrogate roots, so it
# lands in the SAME accumulator a BP run would use and there is no
# second tensor to take a cosine against. Gate on the B seed instead.
#
# THE SUCCESS BAR (tao#19 item 4) is MANDATORY here as in every lane.
#
# Spinel hygiene: hand-built String-concat JSON (no #{}), ENV reads as
# TOP-LEVEL constants, no Struct, while loops.

require_relative "../io/json_builder"
require_relative "../io/json"
require_relative "../io/toy_events"
require_relative "../io/toy_ssm_task"
require_relative "../llm/engine/lstm_engine"
require_relative "../llm/recipes/lstm_seq"
require_relative "../llm/adamw"
require_relative "../train/dfa_b"
require_relative "../dev/toy_describe_flow"

STEPS       = (ENV["STEPS"] || "5").to_i
SEED        = (ENV["SEED"]  || "0").to_i
TAO_RUN_DIR = ENV["TAO_RUN_DIR"] || ""
RUN_ID      = ENV["TOY_RUN_ID"]  || ""
EVENTS      = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/events.jsonl") : ""

POLICY_S    = ENV["LSTM_POLICY"] || ""
CUT_S       = ENV["LSTM_DFA_CUT"] || ""
N_LAYERS    = (ENV["LSTM_LAYERS"]   || "1").to_i
D_HIDDEN    = (ENV["LSTM_HIDDEN"]   || "64").to_i
D_MODEL     = (ENV["LSTM_FEATURES"] || "24").to_i
SEQ_T       = (ENV["LSTM_SEQ"]      || "64").to_i
N_CLASSES   = (ENV["LSTM_CLASSES"]  || "4").to_i
TASK_S      = ENV["LSTM_TASK"] || ""
CUE_SPAN_S  = ENV["LSTM_CUE_SPAN"] || ""
NOISE_S     = ENV["LSTM_NOISE"] || ""
TASK_SEED   = (ENV["LSTM_TASK_SEED"] || "7").to_i
BATCH       = (ENV["LSTM_BATCH"] || "32").to_i
VAL_BATCHES = (ENV["LSTM_VAL_BATCHES"] || "8").to_i
LR_S        = ENV["LSTM_LR"] || ""
WARMUP_S    = ENV["LSTM_WARMUP"] || ""
B_SEED      = (ENV["LSTM_B_SEED"] || "1234").to_i
B_DIST_S    = ENV["LSTM_B_DIST"]  || ""
B_SCALE_S   = ENV["LSTM_B_SCALE"] || ""
CLIP_S      = ENV["LSTM_CLIP_GRAD"] || ""

NOISE   = NOISE_S.length   > 0 ? NOISE_S.to_f   : 1.0
# THE LANE'S FAIR CELL IS `--lr 0.02 --warmup 200 --steps 4000`, AND IT
# IS NOT THE DEFAULT — it is written out, in full, everywhere the lane's
# numbers are stated (the gate's CELL, the roadmap's For-Tao block,
# docs/cli.md). Val accuracy at 4000 steps, seeds 0/1/2, no warmup:
#
#   lr 0.005   BP .504 / 1.000 /  .996     DFA(step) 1.000 / 1.000 / 1.000
#   lr 0.010   BP .250 / 1.000 /  .996     DFA(step) 1.000 / 1.000 / 1.000
#   lr 0.020   BP .250 /  .996 / 1.000     DFA(step) 1.000 /  .996 / 1.000
#   lr 0.030   BP 1.000 / .250 /  .238     DFA(step) 1.000 / 1.000 /  .996
#
# Every arm is bimodal (solves it, or sits at chance .250 on 4 classes).
# The per-step DFA cut solves 12 of 12 cells; BP solves 7, and WHICH rate
# works depends on the SEED — so no learning rate alone gives a BP arm
# that trains at every seed. A 200-step warmup is what does: at lr 0.02
# it turns BP's .250/.996/1.000 into 1.000/.992/1.000.
#
# THIS LANE SHIPPED THAT WARMUP AS A DEFAULT ONCE, AND IT WAS WRONG.
# The reasoning was that a default which cannot train the BP arm is how
# this lane kept nearly reporting "DFA beats BP". The reasoning was right
# about the trap and wrong about the fix: a default that reinterprets an
# existing config silently relabels other people's experiments. Within
# hours, Tao's `--lr 0.03 --steps 2000` inherited the ramp and produced a
# 3-seed matrix under a cell name that was not the cell it ran (its
# frozen row matched byte-for-byte — frozen takes no optimizer step and
# is the one arm invariant to the schedule — while every trained arm
# moved). Defaults here change NOTHING; the cell is spelled out instead.
#
# The LR default of 0.02 stands: it is the better rate at either warmup
# (2 of 3 seeds bare, 3 of 3 with the ramp). The 0.03 it replaced was
# seed-0 luck — the ONE rate seed 0 trains at and seeds 1 and 2 fail at.
LR      = LR_S.length      > 0 ? LR_S.to_f      : 0.02
# 0.0 means OFF, and off is the default: toy#157's whole fragility grid
# was measured without clipping, and a default that quietly re-defines
# those cells would relabel every number the lane has published.
CLIP    = CLIP_S.length    > 0 ? CLIP_S.to_f    : 0.0
WARMUP  = WARMUP_S.length  > 0 ? WARMUP_S.to_i  : 0

# ---- fail loud on every out-of-range shape (never-mask). ----
if STEPS < 1
  puts "toy-train-lstm: STEPS must be >= 1, got " + STEPS.to_s
  exit 1
end
if N_LAYERS < 1
  puts "toy-train-lstm: LSTM_LAYERS must be >= 1, got " + N_LAYERS.to_s
  exit 1
end
if D_MODEL < 2
  puts "toy-train-lstm: LSTM_FEATURES must be >= 2, got " + D_MODEL.to_s
  exit 1
end
if D_HIDDEN < 1
  puts "toy-train-lstm: LSTM_HIDDEN must be >= 1, got " + D_HIDDEN.to_s
  exit 1
end
if SEQ_T < 2
  puts "toy-train-lstm: LSTM_SEQ must be >= 2, got " + SEQ_T.to_s
  exit 1
end
if N_CLASSES < 2
  puts "toy-train-lstm: LSTM_CLASSES must be >= 2, got " + N_CLASSES.to_s
  exit 1
end
if BATCH < 1
  puts "toy-train-lstm: LSTM_BATCH must be >= 1, got " + BATCH.to_s
  exit 1
end
if VAL_BATCHES < 1
  puts "toy-train-lstm: LSTM_VAL_BATCHES must be >= 1, got " + VAL_BATCHES.to_s
  exit 1
end
if CLIP_S.length > 0 && CLIP <= 0.0
  puts "toy-train-lstm: LSTM_CLIP_GRAD must be a POSITIVE norm (omit it to" +
       " disable clipping; 0 or negative would silently mean off while" +
       " looking like a setting), got " + CLIP_S
  exit 1
end
if CUT_S.length > 0 && CUT_S != "layer" && CUT_S != "step"
  puts "toy-train-lstm: LSTM_DFA_CUT " + CUT_S + " unsupported (layer|step)"
  exit 1
end
if TASK_S.length > 0 && TASK_S != "cue" && TASK_S != "mean"
  puts "toy-train-lstm: LSTM_TASK " + TASK_S + " unsupported (cue|mean)"
  exit 1
end

TASK_KIND = TASK_S == "mean" ? SsmTask::KIND_MEAN : SsmTask::KIND_CUE
DFA_CUT   = CUT_S == "step" ? Toy::LLM::Engine::LstmEngine::CUT_STEP :
                              Toy::LLM::Engine::LstmEngine::CUT_LAYER
cs_raw    = CUE_SPAN_S.length > 0 ? CUE_SPAN_S.to_i : SEQ_T / 4
CUE_SPAN  = cs_raw < 1 ? 1 : cs_raw
if CUE_SPAN >= SEQ_T
  puts "toy-train-lstm: LSTM_CUE_SPAN must be < LSTM_SEQ (the cue has to be" +
       " strictly before the last-step readout or there is no delay to" +
       " carry), got " + CUE_SPAN.to_s + " with LSTM_SEQ=" + SEQ_T.to_s
  exit 1
end

# ---- policy parsing. Tokens are LANE-LOCAL: chain | dfa | frozen. ----
def parse_lstm_policy(pol_s, n_layers)
  policy = [0]; policy.pop
  parts = pol_s.split(",")
  if parts.length > n_layers
    puts "toy-train-lstm: LSTM_POLICY names " + parts.length.to_s +
         " layers but LSTM_LAYERS=" + n_layers.to_s +
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
        puts "toy-train-lstm: unknown LSTM_POLICY token " + tk + " (chain|dfa|frozen)"
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
      puts "toy-train-lstm: LSTM_POLICY has a `chain` layer BELOW a `dfa` layer" +
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

POLICY = parse_lstm_policy(POLICY_S, N_LAYERS)

recipe = Toy::LLM::Recipes::LstmSeq.new
recipe.realize!(D_MODEL, D_HIDDEN, SEQ_T, BATCH, N_CLASSES, N_LAYERS,
                SEED, 1.0, POLICY, DFA_CUT, B_SEED, dist_code(B_DIST_S),
                scale_code(B_SCALE_S), scale_sigma(B_SCALE_S), CLIP)
# tao#flow-json-emit (#25): self-describing run bundle.
ToyDescribeFlow.emit_flow_json(TAO_RUN_DIR, recipe.lr_cache.sess)

# The task generator is toy#155's, UNCHANGED. Holding it fixed is what
# lets this lane's numbers be read against the selective-scan lane's as
# an architecture comparison instead of two separate anecdotes.
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
    rs.add_str("name", "lstm")
    Toy::Events.add_provenance(rs,
      TinyNN.tnn_provenance_host_name, TinyNN.tnn_provenance_host_os,
      TinyNN.tnn_provenance_host_arch,
      TinyNN.tnn_backend_name(recipe.lr_cache.sess))
    model = Toy::Json::Builder.new
    model.add_str("arch", "lstm")
    model.add_str("name", "lstm-seq")
    model.add_num("d_in",        D_MODEL)
    model.add_num("d_hidden",    D_HIDDEN)
    model.add_num("n_layers",    N_LAYERS)
    model.add_num("seq_len",     SEQ_T)
    model.add_num("num_classes", N_CLASSES)
    model.add_str("cell",        "lstm")
    model.add_str("readout",     "last_step")
    rs.add_obj("model", model)
    cost = Toy::Json::Builder.new
    cost.add_num("total_params",  recipe.lr_cache.param_count)
    cost.add_num("active_params", recipe.lr_cache.param_count)
    cost.add_num("flops_per_token", 2 * recipe.lr_cache.param_count)
    # THE MEMORY NUMBER — this lane's success target is stated in it:
    # the summed ggml_nbytes of every node of the realized graph, i.e.
    # the actual materialised activation footprint. toy#155 could only
    # report node COUNT. The caveat travels WITH the number: in a graph
    # autodiff every forward tensor is materialised whatever the credit
    # rule, so this is what the harness BUILDS, not what a streaming
    # implementation could get away with.
    cost.add_num("graph_nodes", recipe.lr_cache.lstm_graph_nodes)
    cost.add_num("graph_bytes", recipe.lr_cache.lstm_graph_bytes)
    # toy#159 — ANALYTIC, and named so a consumer cannot confuse them
    # with the measured pair above. stream_cut_bytes is O(1) in T and
    # assumes a 2x-forward replay; stream_sqrt_t_bytes is BPTT's own best
    # counter-move, quoted so the cut is not compared only to BP's worst.
    cost.add_num("stream_bptt_bytes",   recipe.lr_cache.lstm_stream_bptt)
    cost.add_num("stream_sqrt_t_bytes", recipe.lr_cache.lstm_stream_sqrt)
    cost.add_num("stream_cut_bytes",    recipe.lr_cache.lstm_stream_cut)
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
    config.add_raw("lr",          LR.to_s)
    config.add_num("warmup",      WARMUP)
    # toy#162: 0 means the run had NO clipping — the state every cell
    # this lane published before #162 was measured in.
    config.add_raw("clip_grad",   CLIP.to_s)
    rs.add_obj("config", config)
    # NOT the "franken" key — a consumer keying on `franken` would read
    # an LSTM run as a transformer one (tao#19 item 3).
    dfa = Toy::Json::Builder.new
    dfa.add_raw("policy",  Toy::Json.from_int_array(POLICY))
    dfa.add_str("cut",     CUT_S == "step" ? "step" : "layer")
    dfa.add_num("b_seed",  B_SEED)
    dfa.add_str("b_dist",  B_DIST_S.length > 0 ? B_DIST_S : "gaussian")
    dfa.add_str("b_scale", B_SCALE_S.length > 0 ? B_SCALE_S : "inv_sqrt_fan")
    dfa.add_raw("b_sigma", scale_sigma(B_SCALE_S).to_s)
    dfa.add_num("dfa_wired", recipe.lr_cache.lstm_dfa_wired)
    dfa.add_num("frozen",    recipe.lr_cache.lstm_frozen_count)
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
    rc_l = TinyNN.tnn_download_to_f64_array(recipe.lr_cache.sess,
             recipe.lr_cache.t_logits, logit_buf, BATCH * N_CLASSES)
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
  rc_v = TinyNN.tnn_download_to_f64_array(recipe.lr_cache.sess,
           recipe.lr_cache.t_logits, logit_buf, BATCH * N_CLASSES)
  if rc_v != 0
    puts "toy-train-lstm: val logits download failed: rc=" + rc_v.to_s
    exit 1
  end
  val_hits = val_hits + hits_in(logit_buf, labels, BATCH, N_CLASSES)
  val_seen = val_seen + BATCH
  vb = vb + 1
end
val_acc  = val_hits.to_f / val_seen.to_f
val_loss = val_loss_sum / VAL_BATCHES.to_f
puts "val: acc=" + val_acc.to_s + " loss=" + val_loss.to_s + " n=" + val_seen.to_s
# The graph size rides stdout so a sweep over LSTM_SEQ can read the arms'
# scaling without opening a bundle. `bytes` is the ticket's success
# metric; read the caveat on cost.graph_bytes above first.
puts "graph: nodes=" + recipe.lr_cache.lstm_graph_nodes.to_s +
     " bytes=" + recipe.lr_cache.lstm_graph_bytes.to_s
# toy#159 — the ANALYTIC line, printed next to the measured one and never
# instead of it. `graph:` is what toy BUILDS; `stream:` is what a
# streaming implementation would HOLD, which is the quantity the ticket's
# memory target is actually about and the one no graph measurement here
# can exhibit. The cut's figure is O(1) in T and costs a 2x forward
# replay, which rides the line so the number cannot be quoted as free.
puts Toy::Train::StreamBytes.line(recipe.lr_cache.lstm_stream_bptt,
                                  recipe.lr_cache.lstm_stream_sqrt,
                                  recipe.lr_cache.lstm_stream_cut)

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
  # No checkpoint: there is no LSTM GGUF writer and no infer consumer for
  # this arch — the lane's product is the metric, not the weights.
  re.add_str("checkpoint", "none")
  re.add_raw("exit_code",  "0")
  TinyNN.tnn_events_emit(re.dump)
  TinyNN.tnn_events_close
end
