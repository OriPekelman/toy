# lib/toy/run/train_gnn.rb — Spinel-compiled GNN node-classification
# training runner (-> libexec/toy-train-gnn), the toy#153 / DFA-arch T1
# lane.
#
# WHAT THIS LANE IS FOR. toy#153 calls GNN node classification the
# highest-confidence real positive in the whole cross-architecture
# survey: two papers (DFA-GNN, Zhao et al. NeurIPS 2024; LightOn 2020)
# report DFA at or near BP on citation graphs, the output dim is tiny
# (Cora 7, PubMed 3) which is the regime toy#152 found the program's
# FIRST positive in, and full-graph backprop is genuinely expensive —
# it stores every node's activations and is update-locked across layers,
# while a DFA layer needs only the error and its own activations.
#
# It also carries the ticket's own SKEPTIC CONTROL: DFA-GNN's "beats BP"
# is on tiny high-variance graphs, and its graph-aware feedback may act
# as an implicit regulariser rather than as DFA working. So the frozen
# control here is not a formality — and in a GNN it is an unusually
# strong control, because NEIGHBOURHOOD AGGREGATION IS ARCHITECTURE, NOT
# LEARNING: a frozen random hidden stack still smooths features over the
# graph. See lib/toy/io/toy_gnn_task.rb for why the default task is a
# random GNN teacher and not a plain block model.
#
# CPU-ONLY, no CUDA twin (tao#18). Own compilation unit (landmine #16).
#
# ENV CONTRACT:
#   STEPS / SEED / TAO_RUN_DIR / TOY_RUN_ID   — as every other runner
#   GNN_POLICY       — per-HIDDEN-layer tokens: chain | dfa | frozen
#                      (default: all chain, i.e. plain backprop)
#   GNN_LAYERS       — HIDDEN layers (default 1). With the head that is
#                      TWO propagations, i.e. the canonical 2-layer GCN
#                      the literature reports Cora numbers for; L hidden
#                      layers means L+1 hops of receptive field.
#   GNN_HIDDEN       — hidden width (default 32)
#   GNN_FEEDBACK_ROUTE — direct | structure  (default direct). NOT
#                      named GNN_DFA_FEEDBACK: franken-moe already has a
#                      --dfa-feedback (fixed|kolen-pollack) and that is a
#                      different axis — how B is UPDATED, not how the
#                      error is ROUTED. Same discipline as tao#18 on
#                      --policy-scope: a different meaning gets a
#                      different name, never an overloaded one.
#   GNN_FEEDBACK_HOPS— S-hat powers applied to the error (default 1)
#   GNN_GRAPH        — bundle prefix (prep/fetch_cora.rb writes one);
#                      when set, the synthetic-graph knobs are REJECTED
#                      rather than silently ignored
#   GNN_NODES / GNN_FEATURES / GNN_CLASSES        — synthetic shape
#   GNN_DEGREE / GNN_HOMOPHILY / GNN_FEAT_SIGNAL  — synthetic graph
#   GNN_TASK         — teacher (default) | community
#   GNN_TEACHER_DIM  — teacher hidden width (default 32)
#   GNN_TASK_SEED    — task seed; SEPARATE from SEED so the same graph
#                      can be trained from different inits (default 7)
#   GNN_TRAIN_PER_CLASS — labelled nodes per class (default 50)
#   GNN_LR / GNN_WD / GNN_WARMUP
#   GNN_B_SEED / GNN_B_DIST / GNN_B_SCALE — the DfaB feedback axes
#   GNN_ALIGN        — "1" => per-step align events (cos angle g_dfa,g_bp)
#   GNN_ALIGN_EVERY  — thin align emissions to every Nth step
#
# STDOUT (byte-gated): "step <N>: loss=<float>" per step, then one
# "val: acc=<a> loss=<l> n=<n>" line. events.jsonl goes to the run dir.
#
# THE SUCCESS BAR (tao#19 item 4) is MANDATORY here as in every lane:
#     positive = all-DFA within the stated gap of all-BP
#                AND provably beating the frozen control,
#                at matched init and matched seed.
# The frozen arm is FROZEN WEIGHTS (hidden layers stay at init, only the
# head trains) — see docs/roadmap/dfa-arch-program-2026-08-10.md.
#
# Spinel hygiene: hand-built String-concat JSON (no #{}), ENV reads as
# TOP-LEVEL constants (a constant assigned inside a conditional arm
# reads back empty at runtime), no Struct, while loops.

require_relative "../io/json_builder"
require_relative "../io/json"
require_relative "../io/toy_events"
require_relative "../io/toy_gnn_task"
require_relative "../llm/engine/gnn_engine"
require_relative "../llm/recipes/gnn_node"
require_relative "../llm/adamw"
require_relative "../train/dfa_b"
require_relative "../dev/toy_describe_flow"

STEPS       = (ENV["STEPS"] || "5").to_i
SEED        = (ENV["SEED"]  || "0").to_i
TAO_RUN_DIR = ENV["TAO_RUN_DIR"] || ""
RUN_ID      = ENV["TOY_RUN_ID"]  || ""
EVENTS      = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/events.jsonl") : ""

POLICY_S    = ENV["GNN_POLICY"] || ""
N_LAYERS    = (ENV["GNN_LAYERS"] || "1").to_i
D_HIDDEN    = (ENV["GNN_HIDDEN"] || "32").to_i
FEEDBACK_S  = ENV["GNN_FEEDBACK_ROUTE"] || ""
HOPS_S      = ENV["GNN_FEEDBACK_HOPS"] || ""
GRAPH_S     = ENV["GNN_GRAPH"] || ""
NODES_S     = ENV["GNN_NODES"] || ""
FEATURES_S  = ENV["GNN_FEATURES"] || ""
CLASSES_S   = ENV["GNN_CLASSES"] || ""
DEGREE_S    = ENV["GNN_DEGREE"] || ""
HOMOPHILY_S = ENV["GNN_HOMOPHILY"] || ""
SIGNAL_S    = ENV["GNN_FEAT_SIGNAL"] || ""
TASK_S      = ENV["GNN_TASK"] || ""
TEACHER_S   = ENV["GNN_TEACHER_DIM"] || ""
TASK_SEED   = (ENV["GNN_TASK_SEED"] || "7").to_i
TPC_S       = ENV["GNN_TRAIN_PER_CLASS"] || ""
LR_S        = ENV["GNN_LR"] || ""
WD_S        = ENV["GNN_WD"] || ""
WARMUP      = (ENV["GNN_WARMUP"] || "0").to_i
B_SEED      = (ENV["GNN_B_SEED"] || "1234").to_i
B_DIST_S    = ENV["GNN_B_DIST"]  || ""
B_SCALE_S   = ENV["GNN_B_SCALE"] || ""
ALIGN_ON    = (ENV["GNN_ALIGN"] || "") == "1"
ae_raw      = (ENV["GNN_ALIGN_EVERY"] || "1").to_i
ALIGN_EVERY = ae_raw < 1 ? 1 : ae_raw

# The synthetic defaults are MEASURED, not guessed, and every one of
# them is there to keep the frozen control able to LOSE
# ([[control-arm-must-be-able-to-lose]]). The two that matter:
#
#  - FEATURES 64 with HIDDEN 32. This is the single property that makes
#    the frozen arm a real control, and Cora is where we learned it: a
#    frozen random hidden layer only loses when it has to COMPRESS. At
#    features 32 / hidden 64 the random layer is information-preserving,
#    the trained head recovers nearly everything, and all three arms
#    land within .06 of each other with the frozen arm often AHEAD —
#    measured, on a full sweep of (features, hidden, feat_signal, lr).
#  - TRAIN_PER_CLASS 50, not Planetoid's 20. At 20 labels per class on
#    this graph nobody learns enough for the arms to separate; the
#    literature's 20 is about matching a published protocol, and this
#    lane's job is to separate credit-assignment rules.
N_NODES   = NODES_S.length    > 0 ? NODES_S.to_i    : 2048
N_FEAT    = FEATURES_S.length > 0 ? FEATURES_S.to_i : 64
N_CLASSES = CLASSES_S.length  > 0 ? CLASSES_S.to_i  : 7
DEGREE    = DEGREE_S.length   > 0 ? DEGREE_S.to_i   : 8
HOMOPHILY = HOMOPHILY_S.length > 0 ? HOMOPHILY_S.to_f : 0.8
SIGNAL    = SIGNAL_S.length   > 0 ? SIGNAL_S.to_f   : 0.25
D_TEACH   = TEACHER_S.length  > 0 ? TEACHER_S.to_i  : 32
TRAIN_PC  = TPC_S.length      > 0 ? TPC_S.to_i      : 50
# 0.003. Measured on this lane's own arms (the default synthetic cell,
# 300 steps, seed 0, val accuracy):
#   lr 0.001: BP .476 DFA .519 frozen .359
#   lr 0.003: BP .516 DFA .536 frozen .430   <- the default
#   lr 0.010: BP .495 DFA .462 frozen .502
# On Cora the same sweep puts BP's own best at 0.01 (BP .351 / .661 /
# .697 / .668 / .593 at lr .001 / .003 / .01 / .03 / .1), which is why
# the Cora cells in the docs pass GNN_LR explicitly rather than
# inheriting this default: a lane default is a synthetic-cell default.
LR        = LR_S.length       > 0 ? LR_S.to_f       : 0.003
HOPS      = HOPS_S.length     > 0 ? HOPS_S.to_i     : 1
# AdamW weight decay, default 0 — and it exists for ONE reason, which is
# the skeptic control toy#153 asks for by name. On Cora, DFA BEATS BP at
# this lane's defaults; the ticket's own warning is that DFA-GNN's win
# is probably an implicit REGULARISER rather than DFA working. A decay
# knob is the cheapest way to test that directly: if BP with decay
# catches DFA, "DFA beats BP" was a statement about our unregularised BP
# and not about credit assignment. See docs/roadmap/.
WD        = WD_S.length       > 0 ? WD_S.to_f       : 0.0

# ---- fail loud on every out-of-range shape (never-mask). ----
if STEPS < 1
  puts "toy-train-gnn: STEPS must be >= 1, got " + STEPS.to_s
  exit 1
end
if N_LAYERS < 1
  puts "toy-train-gnn: GNN_LAYERS must be >= 1, got " + N_LAYERS.to_s
  exit 1
end
if D_HIDDEN < 1
  puts "toy-train-gnn: GNN_HIDDEN must be >= 1, got " + D_HIDDEN.to_s
  exit 1
end
if FEEDBACK_S.length > 0 && FEEDBACK_S != "direct" && FEEDBACK_S != "structure"
  puts "toy-train-gnn: GNN_FEEDBACK_ROUTE " + FEEDBACK_S + " unsupported (direct|structure)"
  exit 1
end
if HOPS < 1 || HOPS > 8
  puts "toy-train-gnn: GNN_FEEDBACK_HOPS must be in 1..8, got " + HOPS.to_s
  exit 1
end
if HOPS_S.length > 0 && FEEDBACK_S != "structure"
  # A hop count under `direct` feedback would be read, recorded, and
  # then do nothing — the silent-ignore trap toy#158 closed on the CUDA
  # side. Say so instead.
  puts "toy-train-gnn: GNN_FEEDBACK_HOPS is only meaningful with GNN_FEEDBACK_ROUTE=structure"
  exit 1
end
if TASK_S.length > 0 && TASK_S != "teacher" && TASK_S != "community"
  puts "toy-train-gnn: GNN_TASK " + TASK_S + " unsupported (teacher|community)"
  exit 1
end
if TRAIN_PC < 1
  puts "toy-train-gnn: GNN_TRAIN_PER_CLASS must be >= 1, got " + TRAIN_PC.to_s
  exit 1
end
# A loaded graph carries its own shape and labels. Silently ignoring the
# synthetic knobs would let a swept command line record a cell it never
# ran (the [[zsh-env-var-wordsplit]] failure mode, from the other end).
if GRAPH_S.length > 0
  bad = ""
  if NODES_S.length > 0;     bad = bad + " GNN_NODES"; end
  if FEATURES_S.length > 0;  bad = bad + " GNN_FEATURES"; end
  if CLASSES_S.length > 0;   bad = bad + " GNN_CLASSES"; end
  if DEGREE_S.length > 0;    bad = bad + " GNN_DEGREE"; end
  if HOMOPHILY_S.length > 0; bad = bad + " GNN_HOMOPHILY"; end
  if SIGNAL_S.length > 0;    bad = bad + " GNN_FEAT_SIGNAL"; end
  if TASK_S.length > 0;      bad = bad + " GNN_TASK"; end
  if TEACHER_S.length > 0;   bad = bad + " GNN_TEACHER_DIM"; end
  if bad.length > 0
    puts "toy-train-gnn: GNN_GRAPH supplies the graph, so these are ignored and must not be set:" + bad
    exit 1
  end
end
if GRAPH_S.length == 0
  if N_NODES < 8 || N_FEAT < 1
    puts "toy-train-gnn: GNN_NODES must be >= 8 and GNN_FEATURES >= 1"
    exit 1
  end
  if N_CLASSES < 2
    puts "toy-train-gnn: GNN_CLASSES must be >= 2, got " + N_CLASSES.to_s
    exit 1
  end
  # 0 is legal and useful: an EDGELESS graph makes S-hat the identity,
  # so the model degenerates exactly to toy#152's MLP on the raw
  # features. That is the control that says whether the graph is
  # load-bearing at all, and it costs nothing to allow.
  if DEGREE < 0
    puts "toy-train-gnn: GNN_DEGREE must be >= 0, got " + DEGREE.to_s
    exit 1
  end
  if HOMOPHILY < 0.0 || HOMOPHILY > 1.0
    puts "toy-train-gnn: GNN_HOMOPHILY must be in [0, 1], got " + HOMOPHILY.to_s
    exit 1
  end
  if D_TEACH < 1
    puts "toy-train-gnn: GNN_TEACHER_DIM must be >= 1, got " + D_TEACH.to_s
    exit 1
  end
end

TASK_KIND = TASK_S == "community" ? GnnTask::KIND_COMMUNITY : GnnTask::KIND_TEACHER
FEEDBACK  = FEEDBACK_S == "structure" ? Toy::LLM::Engine::GnnEngine::FEEDBACK_STRUCTURE :
                                        Toy::LLM::Engine::GnnEngine::FEEDBACK_DIRECT

# ---- policy parsing. Tokens are LANE-LOCAL: chain | dfa | frozen. ----
def parse_gnn_policy(pol_s, n_layers)
  policy = [0]; policy.pop
  parts = pol_s.split(",")
  if parts.length > n_layers
    puts "toy-train-gnn: GNN_POLICY names " + parts.length.to_s +
         " layers but GNN_LAYERS=" + n_layers.to_s +
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
        puts "toy-train-gnn: unknown GNN_POLICY token " + tk + " (chain|dfa|frozen)"
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

# Accuracy + mean CE over the nodes on ONE side of the split. ggml lays
# logits [n_classes, N] out with ne0 fastest, so node i starts at
# i*classes. The CE is recomputed here rather than read off the loss
# node because the loss node is masked to the TRAINING nodes.
def score_split(buf, labels, split, want, n, classes)
  hit = 0
  seen = 0
  ce = 0.0
  i = 0
  while i < n
    if split[i] == want
      base = i * classes
      best = 0
      bv = buf[base]
      mx = buf[base]
      c = 1
      while c < classes
        v = buf[base + c]
        if v > bv
          bv = v
          best = c
        end
        if v > mx
          mx = v
        end
        c = c + 1
      end
      z = 0.0
      c2 = 0
      while c2 < classes
        z = z + Math.exp(buf[base + c2] - mx)
        c2 = c2 + 1
      end
      ce = ce - (buf[base + labels[i]] - mx - Math.log(z))
      if best == labels[i]
        hit = hit + 1
      end
      seen = seen + 1
    end
    i = i + 1
  end
  [hit, seen, ce]
end

POLICY = parse_gnn_policy(POLICY_S, N_LAYERS)

# ---- the graph ----
task = GnnTask.new
if GRAPH_S.length > 0
  rcg = task.load_bundle!(GRAPH_S)
  if rcg != 0
    exit 1
  end
else
  task.build_synthetic!(TASK_KIND, N_NODES, N_FEAT, N_CLASSES, DEGREE,
                        HOMOPHILY, SIGNAL, D_TEACH, TASK_SEED)
  rcs = task.build_split!(TRAIN_PC)
  if rcs != 0
    exit 1
  end
end

NODES   = task.gt_nodes
FEATS   = task.gt_feat_dim
CLASSES = task.gt_classes
N_TRAIN = task.gt_n_train
N_VAL   = task.gt_n_val
if N_TRAIN < 1
  puts "toy-train-gnn: the graph has no training nodes"
  exit 1
end
if N_VAL < 1
  puts "toy-train-gnn: the graph has no validation nodes"
  exit 1
end
puts "gnn graph: nodes=" + NODES.to_s + " feat=" + FEATS.to_s +
     " classes=" + CLASSES.to_s + " edges=" + task.gt_edge_a.length.to_s +
     " train=" + N_TRAIN.to_s + " val=" + N_VAL.to_s

train_idx = task.train_indices
y_flat    = Array.new(NODES * CLASSES, 0.0)
mask_flat = Array.new(NODES * CLASSES, 0.0)
ni = 0
while ni < NODES
  if task.gt_split[ni] == GnnTask::SPLIT_TRAIN
    y_flat[ni * CLASSES + task.gt_label[ni]] = 1.0
    cj = 0
    while cj < CLASSES
      mask_flat[ni * CLASSES + cj] = 1.0
      cj = cj + 1
    end
  end
  ni = ni + 1
end
lab_flat = Array.new(N_TRAIN * CLASSES, 0.0)
ti = 0
while ti < N_TRAIN
  lab_flat[ti * CLASSES + task.gt_label[train_idx[ti]]] = 1.0
  ti = ti + 1
end

recipe = Toy::LLM::Recipes::GnnNode.new
recipe.realize!(NODES, FEATS, D_HIDDEN, N_LAYERS, CLASSES, N_TRAIN,
                SEED, 1.0,
                task.propagated_features, task.adj_dense, train_idx,
                y_flat, mask_flat, lab_flat,
                POLICY, FEEDBACK, HOPS, B_SEED, dist_code(B_DIST_S),
                scale_code(B_SCALE_S), scale_sigma(B_SCALE_S))
# tao#flow-json-emit (#25): self-describing run bundle.
ToyDescribeFlow.emit_flow_json(TAO_RUN_DIR, recipe.gn_cache.sess)

adamw = Toy::AdamW.for_from_scratch
adamw.lr = LR
adamw.weight_decay = WD

logit_buf = Array.new(NODES * CLASSES, 0.0)

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
    rs.add_str("name", "gnn")
    Toy::Events.add_provenance(rs,
      TinyNN.tnn_provenance_host_name, TinyNN.tnn_provenance_host_os,
      TinyNN.tnn_provenance_host_arch,
      TinyNN.tnn_backend_name(recipe.gn_cache.sess))
    model = Toy::Json::Builder.new
    model.add_str("arch", "gnn")
    model.add_str("name", "gnn-node")
    model.add_num("d_in",        FEATS)
    model.add_num("d_hidden",    D_HIDDEN)
    model.add_num("n_layers",    N_LAYERS)
    model.add_num("num_classes", CLASSES)
    model.add_str("act",         "silu")
    model.add_str("propagation", "sym_norm_adj_self_loops")
    rs.add_obj("model", model)
    cost = Toy::Json::Builder.new
    cost.add_num("total_params",  recipe.gn_cache.param_count)
    cost.add_num("active_params", recipe.gn_cache.param_count)
    # Per NODE: 2 MACs per weight, plus the message passing (which is
    # parameter-free and therefore invisible to a params-only count —
    # recorded separately so a consumer does not read this lane's cost
    # as if it were a dense MLP's).
    cost.add_num("flops_per_token", 2 * recipe.gn_cache.param_count)
    cost.add_num("propagation_flops_per_step",
                 2 * N_LAYERS * NODES * NODES * D_HIDDEN)
    rs.add_obj("cost", cost)
    config = Toy::Json::Builder.new
    config.add_num("steps",     STEPS)
    config.add_num("seed",      SEED)
    config.add_num("nodes",     NODES)
    config.add_num("edges",     task.gt_edge_a.length)
    config.add_num("n_train",   N_TRAIN)
    config.add_num("n_val",     N_VAL)
    config.add_str("graph",     GRAPH_S.length > 0 ? GRAPH_S : "synthetic")
    config.add_str("task",      TASK_KIND == GnnTask::KIND_COMMUNITY ? "community" : "teacher")
    config.add_num("task_seed", TASK_SEED)
    config.add_num("train_per_class", TRAIN_PC)
    config.add_raw("homophily", HOMOPHILY.to_s)
    config.add_raw("feat_signal", SIGNAL.to_s)
    config.add_raw("lr",        LR.to_s)
    config.add_raw("weight_decay", WD.to_s)
    config.add_num("warmup",    WARMUP)
    rs.add_obj("config", config)
    # The credit-assignment provenance. This lane is NOT franken, so it
    # does NOT reuse the "franken" key (tao#19 item 3: ingest keys on the
    # align events' `wname`, not on a per-lane weight-index table).
    dfa = Toy::Json::Builder.new
    dfa.add_raw("policy",  Toy::Json.from_int_array(POLICY))
    dfa.add_str("feedback", FEEDBACK_S.length > 0 ? FEEDBACK_S : "direct")
    dfa.add_num("feedback_hops", recipe.gn_cache.gnn_feedback_hops)
    dfa.add_num("b_seed",  B_SEED)
    dfa.add_str("b_dist",  B_DIST_S.length > 0 ? B_DIST_S : "gaussian")
    dfa.add_str("b_scale", B_SCALE_S.length > 0 ? B_SCALE_S : "inv_sqrt_fan")
    dfa.add_raw("b_sigma", scale_sigma(B_SCALE_S).to_s)
    dfa.add_num("dfa_wired", recipe.gn_cache.gnn_dfa_wired)
    dfa.add_num("frozen",    recipe.gn_cache.gnn_frozen_count)
    rs.add_obj("dfa", dfa)
    TinyNN.tnn_events_emit(rs.dump)
  else
    puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
  end
end

# ---- align telemetry buffers (sized once, like the franken lane). ----
n_align = recipe.gn_cache.gnn_align_grads.length
abuf = [0.0]; abuf.pop
gbuf = [0.0]; gbuf.pop
if ALIGN_ON && n_align > 0
  nmax = 0
  ai0 = 0
  while ai0 < n_align
    nw = TinyNN.tnn_tensor_nelements(recipe.gn_cache.gnn_align_grads[ai0])
    if nw > nmax; nmax = nw; end
    ai0 = ai0 + 1
  end
  z = 0
  while z < nmax
    abuf.push(0.0); gbuf.push(0.0)
    z = z + 1
  end
end

# ---- training loop. ONE step == one full-graph pass. ----
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

  loss = recipe.step!(m_hp, step == 0)
  final_loss = loss
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s

  if EVENTS.length > 0
    rc_l = TinyNN.tnn_download_to_f64_array(recipe.gn_cache.sess,
             recipe.gn_cache.t_logits, logit_buf, NODES * CLASSES)
    tr_acc = -1.0
    if rc_l != 0
      puts "logits download failed: step=" + (step + 1).to_s + " rc=" + rc_l.to_s
    else
      sc = score_split(logit_buf, task.gt_label, task.gt_split,
                       GnnTask::SPLIT_TRAIN, NODES, CLASSES)
      tr_acc = sc[0].to_f / sc[1].to_f
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
    es.add_num("samples", N_TRAIN)
    es.add_num("wall_us", step_wall_us)
    TinyNN.tnn_events_emit(es.dump)
  end

  if ALIGN_ON && EVENTS.length > 0 && n_align > 0 && (step % ALIGN_EVERY) == 0
    ai = 0
    while ai < n_align
      nw = TinyNN.tnn_tensor_nelements(recipe.gn_cache.gnn_align_grads[ai])
      rc_g = TinyNN.tnn_download_to_f64_array(recipe.gn_cache.sess,
        recipe.gn_cache.gnn_align_grads[ai], gbuf, nw)
      rc_a = TinyNN.tnn_download_to_f64_array(recipe.gn_cache.sess,
        recipe.gn_cache.gnn_align_accs[ai], abuf, nw)
      if rc_g != 0 || rc_a != 0
        # never-mask: a failed shadow download must not masquerade as a
        # zero/stale gradient in the telemetry.
        puts "align download failed: step=" + (step + 1).to_s +
             " wname=" + recipe.gn_cache.gnn_align_wnames[ai] +
             " rc_g=" + rc_g.to_s + " rc_a=" + rc_a.to_s
      end
      ae = Toy::Json::Builder.new
      ae.add_str("kind",  "align")
      ae.add_str("phase", "train")
      ae.add_num("t",     TinyNN.tnn_events_now_seconds)
      ae.add_num("step",  step + 1)
      ae.add_num("li",    recipe.gn_cache.gnn_align_lis[ai])
      # tao#19 item 3: `li` stays the layer index, `wi` is LANE-LOCAL
      # (one weight per layer here, so 0), and `wname` is the string
      # Tao's ingest keys on.
      ae.add_num("wi",    0)
      ae.add_str("wname", recipe.gn_cache.gnn_align_wnames[ai])
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
# TRANSDUCTIVE: the val nodes were in every forward pass all along —
# what was held out is their LABELS, which only ever entered through the
# masked CE and the masked DFA error. One more step at lr = 0 gives the
# logits AT the trained weights (an optimizer step at lr 0 is a weight
# no-op).
# MIRROR RULE (the toy#139/#146 class of bug): "lr=0" has to mean EVERY
# hp vector the graph can reach. This lane has exactly ONE — if a
# per-layer or per-optimizer vector is ever added here, it must be
# zeroed in this pass too, or the held-out set silently becomes training
# data.
val_hp = Mat.new(1, 7)
val_hp.flat[0] = 0.0
val_hp.flat[1] = adamw.beta1
val_hp.flat[2] = adamw.beta2
val_hp.flat[3] = adamw.eps
val_hp.flat[4] = 0.0
val_hp.flat[5] = adamw.beta1
val_hp.flat[6] = adamw.beta2
recipe.step!(val_hp, false)
rc_v = TinyNN.tnn_download_to_f64_array(recipe.gn_cache.sess,
         recipe.gn_cache.t_logits, logit_buf, NODES * CLASSES)
if rc_v != 0
  puts "toy-train-gnn: val logits download failed: rc=" + rc_v.to_s
  exit 1
end
ts = score_split(logit_buf, task.gt_label, task.gt_split,
                 GnnTask::SPLIT_TRAIN, NODES, CLASSES)
train_acc_final = ts[0].to_f / ts[1].to_f
train_ce_final  = ts[2] / ts[1].to_f
vs = score_split(logit_buf, task.gt_label, task.gt_split,
                 GnnTask::SPLIT_VAL, NODES, CLASSES)
val_acc  = vs[0].to_f / vs[1].to_f
val_loss = vs[2] / vs[1].to_f
# The TRAIN side rides stdout too, and it is not decoration: on this
# lane the arms separate as much by their train/val GAP as by val
# accuracy (DFA drives the 140-label training set to ~0 CE and still
# generalises, BP underfits it at a stable LR), and a reader who only
# sees val accuracy cannot tell those two stories apart.
puts "train: acc=" + train_acc_final.to_s + " loss=" + train_ce_final.to_s +
     " n=" + ts[1].to_s
puts "val: acc=" + val_acc.to_s + " loss=" + val_loss.to_s + " n=" + vs[1].to_s

if EVENTS.length > 0 && TinyNN.tnn_events_active == 1
  ev = Toy::Json::Builder.new
  ev.add_str("kind",  "eval")
  ev.add_str("phase", "eval")
  ev.add_num("t",     TinyNN.tnn_events_now_seconds)
  ev.add_str("name",  "val")
  ev.add_num("n",     vs[1])
  ev.add_raw("accuracy", num_or_null(val_acc))
  ev.add_raw("loss",     num_or_null(val_loss))
  ev.add_num("train_n",  ts[1])
  ev.add_raw("train_accuracy", num_or_null(train_acc_final))
  ev.add_raw("train_loss",     num_or_null(train_ce_final))
  TinyNN.tnn_events_emit(ev.dump)

  re = Toy::Json::Builder.new
  re.add_str("kind", "run_end")
  re.add_num("t",          TinyNN.tnn_events_now_seconds)
  re.add_str("ended_at",   TinyNN.tnn_events_iso8601_now)
  re.add_str("reason",     "completed")
  re.add_num("final_step", STEPS)
  re.add_raw("final_loss", num_or_null(final_loss))
  re.add_raw("val_acc",    num_or_null(val_acc))
  # No checkpoint: there is no GNN GGUF writer and no infer consumer for
  # this arch — the lane's product is the metric, not the weights.
  re.add_str("checkpoint", "none")
  re.add_raw("exit_code",  "0")
  TinyNN.tnn_events_emit(re.dump)
  TinyNN.tnn_events_close
end
