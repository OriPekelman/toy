# lib/toy/run/train_truck.rb — Spinel-compiled truck-backer-upper
# CONTROL runner (-> libexec/toy-train-truck), the toy#189 lane of the
# `dfa-for-dynamic-control` arc.
#
# WHAT THIS LANE IS FOR. Reproduce Schoenauer & Ronald's ICEC'94 result
# — a 4-9-1 net that docks a truck-and-trailer from their 15-start
# ensemble and generalises across the yard — with a DFA-trained network
# in place of their genetic algorithm.
#
# It is the FIRST CLOSED-LOOP fixture in the programme. Every dfa-vs-bp
# fixture was a static dataset, so nothing measured there says whether
# DFA's noisier updates drift and compound in a loop or act as
# exploration that makes a policy robust to its own mistakes.
#
# WHY IT IS AN EXPERIMENT AND NOT A RE-RUN. The GA has NO ERROR SIGNAL
# — it sees one number per episode. So "DFA instead of GA" cannot mean
# BPTT-through-the-plant with W2' swapped for a random matrix: that
# keeps the whole gradient path the GA never had. It means an EPISODIC
# STATE ERROR, randomly projected onto the weights, with NO PLANT
# JACOBIAN ANYWHERE (`dfa_tb`). That is a stronger departure from
# textbook DFA than anything dfa-vs-bp ran — textbook DFA still hands
# the readout its true dL/dy, and here dL/du_t does not exist without
# the plant.
#
# NO tinynn GRAPH, AND THAT IS DELIBERATE. This is 55 weights and
# 300-step episodes: a ggml graph per step would be dominated by
# per-op FFI (~800us/op, measured — see the FFI per-op memory), and the
# GA needs pop x gens x 15 x 300 forward passes. Spinel compiles the
# scalar loops below to C, so the whole lane is arithmetic in one
# compilation unit. tinynn is linked ONLY for the events writer.
# CPU-only, no CUDA twin (the T0-T3 rule; 55 weights).
#
# ── THE FIVE ARMS, ALL PRE-REGISTERED, ALL RUN ALONGSIDE ──
#
#   arm       hidden W1 by          readout W2 by     role
#   ga        evolution             evolution         the paper's pole
#   bptt      exact plant Jacobian  exact             the CEILING
#   frozen    fixed at init         exact             THE FIXTURE GATE
#   dfa_tb    B1.e                  B2.e              the HEADLINE
#   dfa_rx    B1.e                  exact             disambiguation
#
# A 2x2 on (hidden signal) x (readout signal), with `ga` outside it as
# the published reference.
#
# `frozen` IS NOT A FORMALITY. Four inputs into 9 random sigmoids is a
# good random-feature basis, and a frozen body with a trained linear
# readout is a classic controller. If `bptt` does not beat `frozen`
# with margin at matched seeds, THE FIXTURE CANNOT DISCRIMINATE and
# every DFA reading on it is void (C-FIXTURE). That is exactly how G1
# died in dfa-vs-bp. Read that row before any other.
#
# ── THE REPRODUCTION BAR (the paper's own numbers) ──
#
#   net           4-9-1 logistic sigmoid, 55 weights incl. biases
#                 (3-7-1 without the cab angle; 8-17-1 duplicated at 10x)
#   plant         r = 3, 300-step episodes, trailer rear at (0,0), Os = 0
#   score         NEAREST APPROACH over the episode, not the terminal
#                 state — the GA scored the best point on the trajectory
#   single-point  from (20, 10, -2): d^2 < 5 in 10/10 seeds
#   multi-point   the 15-start ensemble, ~1500 generations
#   generalise    trained on those 15, docks from anywhere in
#                 [50,100] x [-50,50] x [-180,180] deg
#
# And the paper's own split is a FREE PRE-REGISTERED HYPOTHESIS: the
# gradient-free method owned the FAR field ("our results apply to
# travel starting far from the goal") and lost the near field
# (Nguyen-Widrow's BPTT "yielded complex close-up manoeuvering
# behaviour which we have yet failed to reproduce"). Which side does a
# randomly projected error land on? So `far` and `near` are evaluated
# SEPARATELY, always, on every arm — never pooled into one number.
#
# ── WHAT IS MATCHED, AND WHY IT IS PLANT STEPS ──
#
# The GA burns pop x generations episodes and a gradient arm one
# episode per update, so `updates` is not a comparable budget and
# wall-clock is not a claim about a credit rule. TRUCK_BUDGET caps
# PLANT STEPS and every arm reports `plant_steps`, so the comparison is
# at matched simulator work. Init (uniform [-1,1], the paper's) and the
# eval start streams are matched by construction — see the mirror rule
# note on EV_SEED.
#
# ── LANDMINES ──
#
#  1. THE GRADIENT IS SELF-CHECKED, and it has to be. `bptt` is the
#     ceiling every other row is read against, and a sign error in it
#     would make DFA look good. TRUCK_SELFTEST=1 finite-differences the
#     analytic episode gradient against the plant itself, all P
#     parameters, and `gate-truck-lane` runs it. It also probes a
#     CLAMPED configuration, because of landmine 2.
#  2. THE JACK-KNIFE CLAMP CHANGES THE JACOBIAN. On a clamped step
#     Oc' = Os' -+ pi/2, so d(Oc')/d(anything) EQUALS d(Os')/d(anything)
#     — not the free row #188 reports. The backward pass substitutes
#     row Oc := row Os on those steps. Skipping this is not a small
#     error: the clamp fires on most steps of a hard-lock rollout, and
#     the FD check catches it.
#  3. THE ARGMIN IS A DISCONTINUITY. The loss is d^2 at the
#     BEST-APPROACH step, so a weight perturbation can move which step
#     that is, and an episode can also change LENGTH (early termination
#     on dock/invalid). The selftest asserts both are stable across
#     each probe and says so when they are not, rather than reporting a
#     large FD deviation as a gradient bug.
#  4. A ZERO GRADIENT IS NOT NEUTRAL. If the best approach is step 0
#     (the start state was never improved on) no step contributed and
#     the episode's gradient is exactly zero. That is a plausible
#     no-op, so it is COUNTED (`zero_grad`) and printed. An arm whose
#     zero_grad equals its update count did not train at all while
#     reporting a loss curve.
#  5. e IS AN ERROR, SO IT IS NORMALISED AS ONE. The observation's x is
#     (x-50)/50, which is -1 at the dock; using that for an error
#     component would make the error non-zero AT THE GOAL. e uses
#     (dx/50, dy/50, dtheta/pi).
#  6. THE FITNESS IS RECONSTRUCTED. `ga` scores on
#     1/(eps+d^2) * 1/(1+gamma*l), which satisfies everything the
#     paper's text states and reduces correctly at gamma = 0, but the
#     figure did not survive OCR. The provenance line carries
#     `fitness=reconstructed` and it must be confirmed with Edmund
#     Ronald before a `ga` number is published. d^2 itself is cited and
#     unaffected, and every gradient arm is unaffected.
#
# ENV CONTRACT:
#   STEPS / SEED / RUN_DIR / TOY_RUN_ID   — as every other runner.
#                     STEPS is the UPDATE cap (generations, for `ga`).
#   TRUCK_ARM       — ga | bptt | frozen | dfa_tb | dfa_rx (default bptt)
#   TRUCK_CAR       — 1 => the CAR regime (toy#195): no trailer, no
#                     jack-knife, one heading. A docking policy on a car
#                     is near-linear in (y, theta) — the regime a fixed
#                     linear feedback law could work in. Set on BOTH
#                     plant instances so bptt's Jacobian models the body
#                     the rollout uses.
#   TRUCK_LESSON    — 0..19, the curriculum index for
#                     TRUCK_START=lesson (default 19, the full yard).
#                     Lesson 0 is x in (4,10) m, y in +/-2 m, straight
#                     trailer: the NEAR field where a proportional law
#                     docks. The index is a DIFFICULTY AXIS.
#   TRUCK_START     — ensemble (default) | point | yard | lesson
#   TRUCK_OBS       — 4 (default) | 3 | 8   [see toy_truck_task.rb]
#   TRUCK_HIDDEN    — hidden width (default 9, the paper's)
#   TRUCK_ACT       — sigmoid (default, the paper's) | tanh (frontend
#                     parity). sigmoid maps out -> 2*sigma-1; tanh is
#                     already in [-1,1] and maps through. BOTH then
#                     saturate at +/-1 and scale by u_max = 70 deg.
#   TRUCK_R         — plant step length (default 3.0, the paper's)
#   TRUCK_STEP_CAP  — steps per episode (default 300, the paper's)
#   TRUCK_LR        — gradient-arm learning rate (default 0.02). PER-ARM
#                     BY NECESSITY: an arm measured at another arm's LR
#                     is not a negative (the rule toy#160 nearly
#                     published a wrong conclusion from). Every arm's
#                     cell must be found on its own grid, and an optimum
#                     on a grid EDGE is not an optimum.
#   TRUCK_CLIP      — global-norm gradient clip; 0 disables (default 1.0).
#                     NOT optional in practice: the objective is a
#                     PHYSICAL d^2 of order 10^4 at the paper's starts,
#                     chained through ~30 plant steps, so one unclipped
#                     update saturates the steering and every subsequent
#                     episode ends at step 0 while the loss curve sits
#                     flat at the START states' d^2 — a plausible number,
#                     not an error. Clipping keeps the DESCENDED objective
#                     identical to the REPORTED one, which rescaling the
#                     loss would not (toy#188's landmine 2). It is also
#                     the fair BPTT control toy#162 established, and its
#                     known cost applies here too: clipping RE-ROLLS
#                     which basin a seed lands in, so an arm can be
#                     bimodal rather than converged.
#   TRUCK_BUDGET    — PLANT-STEP budget; 0 = unlimited (default 0)
#   TRUCK_LOSS      — best (default) | terminal. WHICH STEP THE ARMS
#                     DESCEND, as distinct from the step they are
#                     SCORED at, which is always the nearest approach.
#
#                     `best` is the DEFAULT and descends exactly what it
#                     scores. IT CANNOT TRAIN THE PAPER'S SINGLE-POINT
#                     START AT ALL, and the reason is structural rather
#                     than a tuning problem: from (20, 10, -2) backing
#                     up INCREASES x, so an untrained policy never gets
#                     closer to the dock than its start, the nearest
#                     approach IS the start, and d^2 at the start is a
#                     CONSTANT (504.0) that no weight can move. The
#                     exact gradient is then identically zero and every
#                     arm sits on a plateau forever, reporting a flat
#                     loss curve at exactly 504.0 for every seed —
#                     measured, 0/10 seeds, all four arms.
#
#                     `terminal` grades where the episode ENDED, which
#                     always has a gradient. It removes the plateau but
#                     DOES NOT RECOVER THE SINGLE POINT (measured: still
#                     exactly 504.0, 0/10 seeds, all four arms) and costs
#                     the ensemble (bptt dock5 0.911 -> 0.311), which is
#                     why it is not the default. It descends something
#                     other than the score — toy#188's landmine 2 — so
#                     whenever it is selected the difference is VISIBLE:
#                     the provenance carries `loss=`, and every step line
#                     prints the descended `loss=` beside the scored
#                     `score_d2=`.
#   TRUCK_DFA_SUM   — episode (default) | to_best. Which steps the DFA
#                     broadcast accumulates over. toy#189 writes
#                     "Sum_t ... during the episode" while the gradient
#                     arms' credit path stops at the best-approach
#                     step, so the two arms see different spans of one
#                     episode. `episode` is the issue's reading and the
#                     default; `to_best` makes the asymmetry testable
#                     instead of only noted.
#   TRUCK_B_SEED / _DIST / _SCALE — the DfaB feedback axes
#   TRUCK_GA_POP / _GENS / _AGG / _MUT / _ELITE — the GA's own knobs.
#                     _AGG = mean (default) | min, both of which the
#                     paper reports as acceptable aggregations over the
#                     15 starts. _GENS defaults to STEPS.
#   TRUCK_EVAL_N    — episodes per eval set (default 64)
#   TRUCK_EVAL_SEED — eval start stream (default 4242). SHARED by every
#                     arm: the mirror rule. If this ever varies per arm
#                     the arms are scored on different yards.
#   TRUCK_EXPORT    — write the controller in the FRONTEND's format
#                     ([layer][unit][w..., bias], bias LAST) plus a
#                     <path>.meta.json sidecar naming the activation and
#                     the output->steering map, so the viewer cannot
#                     silently replay the wrong function.
#   TRUCK_FIT_EPS / TRUCK_FIT_GAMMA — the reconstructed fitness's
#                     constants (defaults 0.1 / 0.001, the paper's).
#   TRUCK_SELFTEST  — 1 => finite-difference the analytic gradient and
#                     exit. Landmine 1.
#
#   toy#190 — HEADLESS ROLLOUT to a `tbu-traces/1` bundle:
#   TRUCK_LOAD      — a controller in the TRUCK_EXPORT format. Present =>
#                     TRAINING IS SKIPPED and the loaded weights are
#                     rolled out as-is.
#   TRUCK_TRACE     — write the trace bundle here. Also works WITHOUT
#                     TRUCK_LOAD, in which case the just-trained
#                     controller is rolled out with no round trip.
#   TRUCK_TRACE_SCHEME — point | ensemble | yard (default: TRUCK_START)
#   TRUCK_TRACE_N   — yard draws (default 64)
#   TRUCK_TRACE_SEED — the yard stream (default TRUCK_EVAL_SEED). Yard
#                     starts are reproducible FROM THE SEED ALONE, via
#                     the plant's own sample_yard!, so a bundle from
#                     this engine and one from the frontend's CLI
#                     overlay on identical states.
#   TRUCK_STRIDE    — keep every k-th trace row. The LAST row and the
#                     BEST-APPROACH row are ALWAYS kept whatever the
#                     stride: they are the two rows every summary is
#                     computed from, and a stride that dropped them
#                     would leave a bundle whose own numbers cannot be
#                     recomputed from its own trace.
#
#   THE ROLLOUT SHARES tk_rollout WITH TRAINING. It is not a second
#   implementation to keep in parity — that is the whole point of the
#   frontend consuming traces rather than running its own forward pass,
#   and it applies just as much inside this file.
#
# STDOUT (byte-gated): "step <N>: loss=<float>" per update, then
# "eval: set=<s> ..." per start set, "truck: <provenance>", and
# "selftest: ..." under TRUCK_SELFTEST.
#
# Spinel hygiene: hand-built String-concat JSON (no #{}), ENV reads as
# TOP-LEVEL constants, no Struct, while loops, typed-empty array seeds,
# top-level defs with prefixed params (landmines #12/#16).

require_relative "../io/json_builder"
require_relative "../io/json"
require_relative "../io/toy_events"
require_relative "../io/toy_truck_task"
require_relative "../train/dfa_b"
# Mat is not used by this lane — it is required because tinynn.rb's own
# staging helpers are typed against it, and without it Spinel infers
# `m.nrows * m.ncols` as poly and refuses the arithmetic. Every runner
# in the tree carries this require for the same reason.
require_relative "../models/transformer"
require_relative "../ffi/tinynn"

# EVERY numeric knob reads its STRING first and length-checks it. The
# CLI marshals "absent" as the EMPTY STRING, and "".to_i is 0, not the
# default — so `(ENV[x] || "4").to_i` silently yields 0 for every value
# the CLI does not set. That is the empty-env-string trap this tree has
# already been bitten by (the K-series .to_f case); here it turned
# `toy research train truck` into "TRUCK_OBS must be 3|4|8, got 0".
STEPS_S     = ENV["STEPS"] || ""
STEPS       = STEPS_S.length > 0 ? STEPS_S.to_i : 200
SEED_S      = ENV["SEED"] || ""
SEED        = SEED_S.length > 0 ? SEED_S.to_i : 0
RUN_DIR_NEW = ENV["TOY_RUN_DIR"] || ""
RUN_DIR     = RUN_DIR_NEW.length > 0 ? RUN_DIR_NEW : (ENV["TAO_RUN_DIR"] || "")
RUN_ID      = ENV["TOY_RUN_ID"]  || ""
EVENTS      = RUN_DIR.length > 0 ? (RUN_DIR + "/events.jsonl") : ""

ARM_S       = ENV["TRUCK_ARM"]   || ""
START_S     = ENV["TRUCK_START"] || ""
OBS_N_S     = ENV["TRUCK_OBS"] || ""
OBS_N       = OBS_N_S.length > 0 ? OBS_N_S.to_i : 4
HIDDEN_S    = ENV["TRUCK_HIDDEN"] || ""
HIDDEN      = HIDDEN_S.length > 0 ? HIDDEN_S.to_i : 9
ACT_S       = ENV["TRUCK_ACT"]   || ""
R_S         = ENV["TRUCK_R"]     || ""
PLANT_R     = R_S.length > 0 ? R_S.to_f : 3.0
STEP_CAP_S  = ENV["TRUCK_STEP_CAP"] || ""
STEP_CAP    = STEP_CAP_S.length > 0 ? STEP_CAP_S.to_i : 300
LR_S        = ENV["TRUCK_LR"]    || ""
LR          = LR_S.length > 0 ? LR_S.to_f : 0.02
CLIP_S      = ENV["TRUCK_CLIP"] || ""
CLIP        = CLIP_S.length > 0 ? CLIP_S.to_f : 1.0
BUDGET_S    = ENV["TRUCK_BUDGET"] || ""
BUDGET      = BUDGET_S.length > 0 ? BUDGET_S.to_i : 0
E_MODE_S    = ENV["TRUCK_E"] || ""
E_DECAY_S   = ENV["TRUCK_E_DECAY"] || ""
E_DECAY     = E_DECAY_S.length > 0 ? E_DECAY_S.to_f : 0.99
E_NORM_S    = ENV["TRUCK_E_NORM"] || ""
LOSS_S      = ENV["TRUCK_LOSS"] || ""
DFA_SUM_S   = ENV["TRUCK_DFA_SUM"] || ""
B_SEED_S    = ENV["TRUCK_B_SEED"] || ""
B_SEED      = B_SEED_S.length > 0 ? B_SEED_S.to_i : 1234
B_DIST_S    = ENV["TRUCK_B_DIST"]    || ""
B_SCALE_S   = ENV["TRUCK_B_SCALE"]   || ""
GA_POP_S    = ENV["TRUCK_GA_POP"] || ""
GA_POP      = GA_POP_S.length > 0 ? GA_POP_S.to_i : 50
GA_GENS_S   = ENV["TRUCK_GA_GENS"] || ""
GA_GENS     = GA_GENS_S.length > 0 ? GA_GENS_S.to_i : STEPS
GA_AGG_S    = ENV["TRUCK_GA_AGG"]  || ""
GA_MUT_S    = ENV["TRUCK_GA_MUT"]  || ""
GA_MUT      = GA_MUT_S.length > 0 ? GA_MUT_S.to_f : 0.1
GA_ELITE_S  = ENV["TRUCK_GA_ELITE"] || ""
GA_ELITE    = GA_ELITE_S.length > 0 ? GA_ELITE_S.to_i : 2
EVAL_N_S    = ENV["TRUCK_EVAL_N"] || ""
EVAL_N      = EVAL_N_S.length > 0 ? EVAL_N_S.to_i : 64
EVAL_SEED_S = ENV["TRUCK_EVAL_SEED"] || ""
EVAL_SEED   = EVAL_SEED_S.length > 0 ? EVAL_SEED_S.to_i : 4242
EXPORT      = ENV["TRUCK_EXPORT"] || ""
LOAD_PATH   = ENV["TRUCK_LOAD"] || ""
EXPERT      = ENV["TRUCK_EXPERT"] || ""
DEMOS_S     = ENV["TRUCK_DEMOS"] || ""
DEMO_N_S    = ENV["TRUCK_DEMO_N"] || ""
DEMO_N      = DEMO_N_S.length > 0 ? DEMO_N_S.to_i : 500
DEMO_B_S    = ENV["TRUCK_DEMO_BATCH"] || ""
DEMO_BATCH  = DEMO_B_S.length > 0 ? DEMO_B_S.to_i : 64
WD_S        = ENV["TRUCK_WD"] || ""
WD          = WD_S.length > 0 ? WD_S.to_f : 0.0
CAR         = (ENV["TRUCK_CAR"] || "") == "1"
LESSON_S    = ENV["TRUCK_LESSON"] || ""
LESSON      = LESSON_S.length > 0 ? LESSON_S.to_i : 19
TRACE       = ENV["TRUCK_TRACE"] || ""
TRACE_SCH_S = ENV["TRUCK_TRACE_SCHEME"] || ""
TRACE_N_S   = ENV["TRUCK_TRACE_N"] || ""
TRACE_N     = TRACE_N_S.length > 0 ? TRACE_N_S.to_i : 64
TRACE_SD_S  = ENV["TRUCK_TRACE_SEED"] || ""
STRIDE_S    = ENV["TRUCK_STRIDE"] || ""
STRIDE      = STRIDE_S.length > 0 ? STRIDE_S.to_i : 1
FIT_EPS_S   = ENV["TRUCK_FIT_EPS"] || ""
FIT_EPS     = FIT_EPS_S.length > 0 ? FIT_EPS_S.to_f : 0.1
FIT_GAM_S   = ENV["TRUCK_FIT_GAMMA"] || ""
FIT_GAMMA   = FIT_GAM_S.length > 0 ? FIT_GAM_S.to_f : 0.001
SELFTEST    = (ENV["TRUCK_SELFTEST"] || "") == "1"

# Declared WITH the arm codes, not down in the provenance section: the
# toy#190 rollout block reads it and runs earlier in the file, where a
# constant defined later is simply nil.
ARM_NAMES  = ["ga", "bptt", "frozen", "dfa_tb", "dfa_rx",
              "imit_bp", "imit_dfa", "imit_frozen"]
ARM_GA     = 0
ARM_BPTT   = 1
ARM_FROZEN = 2
ARM_DFA_TB = 3
ARM_DFA_RX = 4
# toy#194 — THE IMITATION LEG. A manufactured expert (a `ga` or `bptt`
# controller that docks) is rolled out, and a fresh net is trained on
# its PER-STEP action with a per-step, state-local error
# e_t = u*_t - u_t in SIGNAL units.
#
# WHY THIS IS A DIFFERENT EXPERIMENT FROM dfa_tb, and why the rank-1
# objection does not apply: C-SIGNAL-c objected that a scalar error
# gives B1 in R^{9x1} and therefore a rank-1 update. That objection was
# about BROADCASTING one scalar over a whole episode, where every step
# gets the same direction. Here the error is PER STEP and STATE-LOCAL:
# e_t differs at every step, so the accumulated update is a sum of
# rank-1 terms with different coefficients, which is not rank-1. This
# is also the regime every inherited DFA positive lives in (F16, F17).
ARM_IMIT_BP     = 5
ARM_IMIT_DFA    = 6
ARM_IMIT_FROZEN = 7

ARM = ARM_S == "ga"     ? ARM_GA :
      ARM_S == "frozen" ? ARM_FROZEN :
      ARM_S == "dfa_tb" ? ARM_DFA_TB :
      ARM_S == "dfa_rx" ? ARM_DFA_RX :
      ARM_S == "imit_bp" ? ARM_IMIT_BP :
      ARM_S == "imit_dfa" ? ARM_IMIT_DFA :
      ARM_S == "imit_frozen" ? ARM_IMIT_FROZEN : ARM_BPTT
IS_IMIT = ARM == ARM_IMIT_BP || ARM == ARM_IMIT_DFA || ARM == ARM_IMIT_FROZEN
DEMOS_SCARCE = DEMOS_S == "abundant" ? false : true

ACT_SIGMOID = 0
ACT_TANH    = 1
ACT = ACT_S == "tanh" ? ACT_TANH : ACT_SIGMOID

SUM_TO_BEST = DFA_SUM_S == "to_best"
# DEFAULT `best` — it descends exactly what it scores.
#
# `terminal` was briefly the default (toy#189) on the reasoning that
# `best` cannot train the paper's single-point start at all. MEASUREMENT
# WITHDREW THAT: `terminal` removes the plateau but does NOT recover the
# single point (still exactly 504.0, 0/10 seeds, all four arms, at every
# LR in {3,6,12,24}), while costing the ceiling arm two thirds of its
# docking rate (bptt ensemble dock5 0.911 -> 0.311). So the default is
# the one that keeps the descended and reported objectives IDENTICAL,
# which is also toy#188's landmine 2 not repeated.
LOSS_TERMINAL = LOSS_S == "terminal"

# ---- fail loud on every out-of-range knob (never-mask). ----
if ARM_S.length > 0 && ARM_S != "ga" && ARM_S != "bptt" && ARM_S != "frozen" &&
   ARM_S != "dfa_tb" && ARM_S != "dfa_rx" && ARM_S != "imit_bp" &&
   ARM_S != "imit_dfa" && ARM_S != "imit_frozen"
  puts "toy-train-truck: TRUCK_ARM must be " +
       "ga|bptt|frozen|dfa_tb|dfa_rx|imit_bp|imit_dfa|imit_frozen, got " + ARM_S
  exit 1
end
if DEMOS_S.length > 0 && DEMOS_S != "scarce" && DEMOS_S != "abundant"
  puts "toy-train-truck: TRUCK_DEMOS must be scarce|abundant, got " + DEMOS_S
  exit 1
end
if ACT_S.length > 0 && ACT_S != "sigmoid" && ACT_S != "tanh"
  puts "toy-train-truck: TRUCK_ACT must be sigmoid|tanh, got " + ACT_S
  exit 1
end
if E_NORM_S.length > 0 && E_NORM_S != "none" && E_NORM_S != "unit"
  puts "toy-train-truck: TRUCK_E_NORM must be none|unit, got " + E_NORM_S
  exit 1
end
if E_MODE_S.length > 0 && E_MODE_S != "xyt" && E_MODE_S != "yt" &&
   E_MODE_S != "centered" && E_MODE_S != "signed_x"
  puts "toy-train-truck: TRUCK_E must be xyt|yt|centered|signed_x, got " + E_MODE_S
  exit 1
end
if LOSS_S.length > 0 && LOSS_S != "best" && LOSS_S != "terminal"
  puts "toy-train-truck: TRUCK_LOSS must be best|terminal, got " + LOSS_S
  exit 1
end
if DFA_SUM_S.length > 0 && DFA_SUM_S != "episode" && DFA_SUM_S != "to_best"
  puts "toy-train-truck: TRUCK_DFA_SUM must be episode|to_best, got " + DFA_SUM_S
  exit 1
end
if GA_AGG_S.length > 0 && GA_AGG_S != "mean" && GA_AGG_S != "min"
  puts "toy-train-truck: TRUCK_GA_AGG must be mean|min, got " + GA_AGG_S
  exit 1
end
if START_S.length > 0 && START_S != "ensemble" && START_S != "point" &&
   START_S != "yard" && START_S != "lesson" &&
   START_S != "half_yard" && START_S != "half_yard_neg"
  puts "toy-train-truck: TRUCK_START must be " +
       "ensemble|point|yard|lesson|half_yard|half_yard_neg, got " + START_S
  exit 1
end
if LESSON < 0 || LESSON > 19
  puts "toy-train-truck: TRUCK_LESSON must be 0..19, got " + LESSON.to_s
  exit 1
end
if OBS_N != 3 && OBS_N != 4 && OBS_N != 8
  puts "toy-train-truck: TRUCK_OBS must be 3|4|8, got " + OBS_N.to_s
  exit 1
end
if HIDDEN < 1
  puts "toy-train-truck: TRUCK_HIDDEN must be >= 1, got " + HIDDEN.to_s
  exit 1
end
if STEP_CAP < 2
  puts "toy-train-truck: TRUCK_STEP_CAP must be >= 2, got " + STEP_CAP.to_s
  exit 1
end
if STEPS < 1
  puts "toy-train-truck: STEPS must be >= 1, got " + STEPS.to_s
  exit 1
end
if ARM == ARM_GA && GA_POP < 4
  puts "toy-train-truck: TRUCK_GA_POP must be >= 4, got " + GA_POP.to_s
  exit 1
end

# ---- the parameter vector, one flat array ----
#
# [ W1 (H x n_in) | b1 (H) | W2 (H) | b2 (1) ], row-major W1.
# 4-9-1 => 36 + 9 + 9 + 1 = 55, the paper's count.
N_IN   = OBS_N
OFF_W1 = 0
OFF_B1 = HIDDEN * N_IN
OFF_W2 = OFF_B1 + HIDDEN
OFF_B2 = OFF_W2 + HIDDEN
N_PARAM = OFF_B2 + 1

# ---- toy#193 / C2d: WHAT THE BROADCAST PROJECTS ----
#
# THE MEASUREMENT THIS EXISTS FOR. Every start in the yard and in the
# paper's ensemble is EAST of the dock, so the error's dx component is
# always positive — mean |dx| 1.449 against 0.479 for dy and 0.444 for
# dtheta, and its minimum over 200 episodes is +0.797, never negative.
# A fixed B2 applied to that vector therefore has a dominant term whose
# SIGN NEVER CHANGES: over 12 B draws the sign of B2.e falls on one side
# in a median of 99.0% of episodes (min 67%, max 100%). The readout gets
# the same push every update and integrates it into the constant-sign
# saturated policy toy#192 measured. dy flipping sign was the wrong
# suspect; dx NOT flipping is the right one.
E_XYT      = 0   # (dx, dy, dtheta) — current behaviour, byte-null
E_YT       = 1   # drop dx entirely: (dy, dtheta). B becomes 9x2 / 1x2
E_CENTERED = 2   # (dx, dy, dtheta) minus a running mean, so an
                 # always-positive component averages to zero
E_SIGNED_X = 3   # replace dx by the SIGNED distance along the trailer's
                 # heading to the dock, which does flip
E_MODE = E_MODE_S == "yt" ? E_YT :
         E_MODE_S == "centered" ? E_CENTERED :
         E_MODE_S == "signed_x" ? E_SIGNED_X : E_XYT
E_DIM = E_MODE == E_YT ? 2 : 3

# toy#193 follow-up — DIRECTION-ONLY BROADCAST. Orthogonal to E_MODE on
# purpose: E_MODE picks WHICH components the error carries, this picks
# whether its MAGNITUDE reaches the weights at all. Composed, they
# separate "the sign is locked" from "the size is uninformative".
#
# What it tests: the per-episode update scales with ||e|| at the graded
# step. For an arm that never improves, ||e|| stays large and roughly
# constant, so the readout integrates a same-size push every episode.
# `unit` removes the size and leaves only the direction.
#
# NOT a claim that this shrinks the signal as the policy improves —
# e already would, if the policy ever improved. It removes magnitude
# INFORMATION, which is a different and cleanly testable thing.
E_NORM = E_NORM_S == "unit" ? 1 : 0

def tk_zeros(z_n)
  z_a = [0.0]; z_a.pop
  z_i = 0
  while z_i < z_n
    z_a.push(0.0)
    z_i = z_i + 1
  end
  z_a
end

def tk_zeros_i(zi_n)
  zi_a = [0]; zi_a.pop
  zi_i = 0
  while zi_i < zi_n
    zi_a.push(0)
    zi_i = zi_i + 1
  end
  zi_a
end

# ---- the tree-wide 31-bit LCG, one stream per purpose ----

def tk_seed_state(ss_seed)
  ss_s = ((ss_seed + 104729) * 2654435761) % 2147483647
  if ss_s <= 0
    ss_s = ss_seed + 104729
  end
  ss_w = 0
  while ss_w < 8
    ss_s = (ss_s * 1103515245 + 12345) & 0x7FFFFFFF
    ss_w = ss_w + 1
  end
  ss_s
end

def tk_next_u(nu_st)
  nu_s = nu_st[0]
  nu_s = (nu_s * 1103515245 + 12345) & 0x7FFFFFFF
  nu_st[0] = nu_s
  (nu_s.to_f + 1.0) / 2147483648.0
end

def tk_next_int(ni_st, ni_n)
  ni_s = ni_st[0]
  ni_s = (ni_s * 1103515245 + 12345) & 0x7FFFFFFF
  ni_st[0] = ni_s
  (ni_s >> 8) % ni_n
end

def tk_gauss(gs_st)
  gs_u1 = tk_next_u(gs_st)
  gs_u2 = tk_next_u(gs_st)
  if gs_u1 < 1.0e-12; gs_u1 = 1.0e-12; end
  Math.sqrt(-2.0 * Math.log(gs_u1)) * Math.cos(2.0 * Math::PI * gs_u2)
end

# ---- activation ----
#
# Clamped at |z| > 40 so MRI and the Spinel-compiled binary agree
# EXACTLY rather than both drifting through exp() overflow in their own
# way. At |z| = 40 the sigmoid is 1 - 4e-18: the clamp is below double
# resolution, not an approximation.
def tk_act(ac_z, ac_kind)
  if ac_kind == 1
    if ac_z > 40.0;  return 1.0; end
    if ac_z < -40.0; return -1.0; end
    return Math.tanh(ac_z)
  end
  if ac_z > 40.0;  return 1.0; end
  if ac_z < -40.0; return 0.0; end
  1.0 / (1.0 + Math.exp(-ac_z))
end

# d(act)/dz expressed in the ACTIVATION's own value.
def tk_dact(da_v, da_kind)
  if da_kind == 1
    return 1.0 - da_v * da_v
  end
  da_v * (1.0 - da_v)
end

# The output map. sigmoid: [0,1] -> 2*sigma-1. tanh: already [-1,1].
# Both then saturate in the plant and scale by u_max.
def tk_signal_of(so_y, so_kind)
  if so_kind == 1
    return so_y
  end
  2.0 * so_y - 1.0
end

def tk_dsignal_dout(ds_y, ds_kind)
  if ds_kind == 1
    return 1.0 - ds_y * ds_y
  end
  2.0 * (ds_y * (1.0 - ds_y))
end

# ---- forward, one step ----
#
# Writes a[] and h[] into the caller's per-step slices and returns the
# steering SIGNAL. fw_out carries [o, y] back for the backward pass.
def tk_forward(fw_w, fw_obs, fw_ob, fw_a, fw_h, fw_ab, fw_nin, fw_hid,
               fw_ow1, fw_ob1, fw_ow2, fw_ob2, fw_kind, fw_out)
  fw_j = 0
  while fw_j < fw_hid
    fw_acc = fw_w[fw_ob1 + fw_j]
    fw_k = 0
    while fw_k < fw_nin
      fw_acc = fw_acc + fw_w[fw_ow1 + fw_j * fw_nin + fw_k] * fw_obs[fw_ob + fw_k]
      fw_k = fw_k + 1
    end
    fw_hv = tk_act(fw_acc, fw_kind)
    fw_a[fw_ab + fw_j] = fw_acc
    fw_h[fw_ab + fw_j] = fw_hv
    fw_j = fw_j + 1
  end
  fw_o = fw_w[fw_ob2]
  fw_j2 = 0
  while fw_j2 < fw_hid
    fw_o = fw_o + fw_w[fw_ow2 + fw_j2] * fw_h[fw_ab + fw_j2]
    fw_j2 = fw_j2 + 1
  end
  fw_y = tk_act(fw_o, fw_kind)
  fw_out[0] = fw_o
  fw_out[1] = fw_y
  tk_signal_of(fw_y, fw_kind)
end

# ---- loading a controller (toy#190) ----
#
# The export is a nested JSON array of numbers whose SHAPE is fully
# determined by (hidden, n_in): hidden rows of (n_in weights + bias),
# then one row of (hidden weights + bias). So the loader scans the
# numbers IN DOCUMENT ORDER and checks the count, rather than carrying a
# nested-array JSON parser for one known shape. A count mismatch is
# fatal and names both sides — silently accepting a short file would
# leave the tail of the net at whatever init produced, which reads as a
# slightly worse arm rather than as a broken load.
def tk_scan_floats(sf_str, sf_out, sf_max)
  sf_n = 0
  sf_i = 0
  sf_len = sf_str.length
  while sf_i < sf_len && sf_n < sf_max
    sf_c = sf_str[sf_i]
    if sf_c == "-" || sf_c == "+" || (sf_c >= "0" && sf_c <= "9")
      sf_j = sf_i
      sf_tok = ""
      while sf_j < sf_len
        sf_d = sf_str[sf_j]
        if sf_d == "-" || sf_d == "+" || sf_d == "." || sf_d == "e" ||
           sf_d == "E" || (sf_d >= "0" && sf_d <= "9")
          sf_tok = sf_tok + sf_d
          sf_j = sf_j + 1
        else
          sf_j = sf_len
        end
      end
      sf_out[sf_n] = sf_tok.to_f
      sf_n = sf_n + 1
      # Resume past the token. Recomputing its length from the string is
      # cheaper than tracking two cursors through the branch above.
      sf_i = sf_i + sf_tok.length
    else
      sf_i = sf_i + 1
    end
  end
  sf_n
end

# ---- the plant, and the second instance used for Jacobians ----

plant = TruckTask.new
plant.paper_defaults!
plant.tt_r = PLANT_R
plant.tt_max_steps = STEP_CAP
plant.tt_obs = OBS_N == 3 ? TruckTask::OBS3 : (OBS_N == 8 ? TruckTask::OBS8 : TruckTask::OBS4)
# toy#195: the CAR regime, set on BOTH instances — the Jacobian plant
# must model the same body the rollout plant does, or bptt descends a
# truck's derivatives along a car's trajectory.
if CAR
  plant.tt_car = 1
end

jplant = TruckTask.new
jplant.paper_defaults!
jplant.tt_r = PLANT_R
jplant.tt_max_steps = STEP_CAP
jplant.tt_obs = plant.tt_obs
if CAR
  jplant.tt_car = 1
end

# ---- rollout storage, allocated once ----

rs_x     = tk_zeros(STEP_CAP + 1)
rs_y     = tk_zeros(STEP_CAP + 1)
rs_oc    = tk_zeros(STEP_CAP + 1)
rs_os    = tk_zeros(STEP_CAP + 1)
rs_clamp = tk_zeros_i(STEP_CAP + 1)
rs_sig   = tk_zeros(STEP_CAP + 1)
rs_o     = tk_zeros(STEP_CAP + 1)
rs_yout  = tk_zeros(STEP_CAP + 1)
rs_a     = tk_zeros((STEP_CAP + 1) * HIDDEN)
rs_h     = tk_zeros((STEP_CAP + 1) * HIDDEN)
rs_obs   = tk_zeros((STEP_CAP + 1) * N_IN)
fw_out   = tk_zeros(2)

# Rollout results, in an array because Spinel has no tuples:
# [0] T (steps taken), [1] best_d2, [2] best_step,
# [3] best_x, [4] best_y, [5] best_angle_err,
# [6] terminal_d2, [7] term_x, [8] term_y, [9] term_angle_err
#
# The TERMINAL block exists for the selftest. The argmin loss can only
# ever walk back from t*, and a clamped step at t*-1 multiplies the
# substituted Oc row by a gradient that is still exactly zero there —
# so grading the terminal state is the only way to put clamped steps
# DEEP inside a backward window and actually gate landmine 2.
ro = tk_zeros(10)

# ---- one episode ----
#
# Drives the plant from its CURRENT state under the policy `ep_w`,
# recording everything the backward passes need. The plant's own
# nearest-approach tracker is the score (toy#188), so this function
# never re-derives it.
def tk_rollout(ep_plant, ep_w, ep_cap, ep_nin, ep_hid,
               ep_ow1, ep_ob1, ep_ow2, ep_ob2, ep_kind,
               ep_x, ep_y, ep_oc, ep_os, ep_clamp, ep_sig, ep_o, ep_yout,
               ep_a, ep_h, ep_obs, ep_fw, ep_ro)
  ep_t = 0
  while ep_t < ep_cap && ep_plant.continue?
    ep_x[ep_t]  = ep_plant.tt_x
    ep_y[ep_t]  = ep_plant.tt_y
    ep_oc[ep_t] = ep_plant.tt_oc
    ep_os[ep_t] = ep_plant.tt_os
    ep_plant.obs!
    ep_kk = 0
    while ep_kk < ep_nin
      ep_obs[ep_t * ep_nin + ep_kk] = ep_plant.tt_obs_buf[ep_kk]
      ep_kk = ep_kk + 1
    end
    ep_s = tk_forward(ep_w, ep_obs, ep_t * ep_nin, ep_a, ep_h, ep_t * ep_hid,
                      ep_nin, ep_hid, ep_ow1, ep_ob1, ep_ow2, ep_ob2,
                      ep_kind, ep_fw)
    ep_sig[ep_t]  = ep_s
    ep_o[ep_t]    = ep_fw[0]
    ep_yout[ep_t] = ep_fw[1]
    ep_plant.step!(ep_s)
    ep_clamp[ep_t] = ep_plant.tt_clamped
    ep_t = ep_t + 1
  end
  ep_ro[0] = ep_t.to_f
  ep_ro[1] = ep_plant.tt_best_d2
  ep_ro[2] = ep_plant.tt_best_step.to_f
  ep_ro[3] = ep_plant.tt_best_x
  ep_ro[4] = ep_plant.tt_best_y
  # The SIGNED branch the min selected — landmine 5 of toy#188's plant.
  ep_sv = ep_plant.tt_best_os
  ep_a1 = ep_sv - 2.0 * Math::PI
  ep_a2 = ep_sv + 2.0 * Math::PI
  ep_bb = ep_sv
  ep_tt = ep_sv * ep_sv
  if ep_a1 * ep_a1 < ep_tt
    ep_tt = ep_a1 * ep_a1
    ep_bb = ep_a1
  end
  if ep_a2 * ep_a2 < ep_tt
    ep_bb = ep_a2
  end
  ep_ro[5] = ep_bb
  ep_ro[6] = ep_plant.sr_d2
  ep_ro[7] = ep_plant.tt_x
  ep_ro[8] = ep_plant.tt_y
  ep_ro[9] = ep_plant.sr_angle_err
  nil
end

# ---- the exact gradient: BPTT through the plant Jacobian ----
#
# L = d^2 at the BEST-APPROACH step t* (the quantity the paper scores,
# so the gradient arms descend what they are graded on — not the
# terminal state, which would be toy#188's landmine 2 all over again).
#
# Backward from t*, accumulating into gr[]. `bp_readout_only` trains
# only W2/b2 (the `frozen` arm). Returns 1 if any step contributed.
def tk_bptt!(bp_jp, bp_w, bp_gr, bp_tstar, bp_ex, bp_ey, bp_eang,
             bp_nin, bp_hid,
             bp_ow1, bp_ob1, bp_ow2, bp_ob2, bp_kind,
             bp_x, bp_y, bp_oc, bp_os, bp_clamp, bp_sig, bp_o, bp_yout,
             bp_a, bp_h, bp_obs, bp_gs, bp_gsn, bp_dobs, bp_readout_only)
  if bp_tstar < 1
    return 0
  end
  # dL/ds at the graded step, slot order (x, y, Oc, Os). The cab angle
  # does not enter the paper's distance, so slot 2 starts at exactly
  # zero and only becomes non-zero one step further back.
  bp_gs[0] = 2.0 * bp_ex
  bp_gs[1] = 2.0 * bp_ey
  bp_gs[2] = 0.0
  bp_gs[3] = 2.0 * bp_eang

  bp_t = bp_tstar - 1
  while bp_t >= 0
    # Jacobian at the state the step STARTED from, under the u it used.
    bp_jp.reset_state!(bp_x[bp_t], bp_y[bp_t], bp_oc[bp_t], bp_os[bp_t])
    bp_u = bp_jp.steer_angle(bp_sig[bp_t])
    bp_jp.jacobian!(bp_u)

    # LANDMINE 2: on a clamped step Oc' = Os' -+ pi/2, so its row of
    # the Jacobian IS the trailer row. The free row the plant reports
    # describes a step the plant did not take.
    if bp_clamp[bp_t] == 1
      bp_c = 0
      while bp_c < 5
        bp_jp.tt_jac[10 + bp_c] = bp_jp.tt_jac[15 + bp_c]
        bp_c = bp_c + 1
      end
    end

    # g_u = J[:,4]^T g_s ; g_s' = J[:,0:4]^T g_s
    bp_gu = 0.0
    bp_r = 0
    while bp_r < 4
      bp_gu = bp_gu + bp_gs[bp_r] * bp_jp.tt_jac[bp_r * 5 + 4]
      bp_r = bp_r + 1
    end
    bp_c2 = 0
    while bp_c2 < 4
      bp_acc = 0.0
      bp_r2 = 0
      while bp_r2 < 4
        bp_acc = bp_acc + bp_gs[bp_r2] * bp_jp.tt_jac[bp_r2 * 5 + bp_c2]
        bp_r2 = bp_r2 + 1
      end
      bp_gsn[bp_c2] = bp_acc
      bp_c2 = bp_c2 + 1
    end

    # Through the saturating steering map, then the net.
    bp_dsig = bp_jp.dsteer_dsignal(bp_sig[bp_t])
    bp_do = bp_gu * bp_dsig * tk_dsignal_dout(bp_yout[bp_t], bp_kind)

    bp_k = 0
    while bp_k < bp_nin
      bp_dobs[bp_k] = 0.0
      bp_k = bp_k + 1
    end

    bp_j = 0
    while bp_j < bp_hid
      bp_hv = bp_h[bp_t * bp_hid + bp_j]
      bp_gr[bp_ow2 + bp_j] = bp_gr[bp_ow2 + bp_j] + bp_do * bp_hv
      bp_dh = bp_do * bp_w[bp_ow2 + bp_j]
      bp_da = bp_dh * tk_dact(bp_hv, bp_kind)
      if bp_readout_only == 0
        bp_gr[bp_ob1 + bp_j] = bp_gr[bp_ob1 + bp_j] + bp_da
      end
      bp_k2 = 0
      while bp_k2 < bp_nin
        if bp_readout_only == 0
          bp_gr[bp_ow1 + bp_j * bp_nin + bp_k2] =
            bp_gr[bp_ow1 + bp_j * bp_nin + bp_k2] +
            bp_da * bp_obs[bp_t * bp_nin + bp_k2]
        end
        # The state feeds the POLICY as well as the plant, so the
        # observation's own gradient rejoins g_s. Dropping this term
        # gives a gradient that is right for a one-step problem and
        # quietly wrong for a closed loop.
        bp_dobs[bp_k2] = bp_dobs[bp_k2] + bp_da * bp_w[bp_ow1 + bp_j * bp_nin + bp_k2]
        bp_k2 = bp_k2 + 1
      end
      bp_j = bp_j + 1
    end
    bp_gr[bp_ob2] = bp_gr[bp_ob2] + bp_do

    # d(obs)/d(state): x -> /50, y -> /50, angle -> /pi; obs8's second
    # half is the same four at 10x and contributes 10x.
    bp_ip = 1.0 / Math::PI
    bp_gsn[0] = bp_gsn[0] + bp_dobs[0] / 50.0
    bp_gsn[1] = bp_gsn[1] + bp_dobs[1] / 50.0
    if bp_nin == 3
      bp_gsn[3] = bp_gsn[3] + bp_dobs[2] * bp_ip
    else
      bp_gsn[2] = bp_gsn[2] + bp_dobs[2] * bp_ip
      bp_gsn[3] = bp_gsn[3] + bp_dobs[3] * bp_ip
      if bp_nin == 8
        bp_gsn[0] = bp_gsn[0] + bp_dobs[4] * 10.0 / 50.0
        bp_gsn[1] = bp_gsn[1] + bp_dobs[5] * 10.0 / 50.0
        bp_gsn[2] = bp_gsn[2] + bp_dobs[6] * 10.0 * bp_ip
        bp_gsn[3] = bp_gsn[3] + bp_dobs[7] * 10.0 * bp_ip
      end
    end

    bp_m = 0
    while bp_m < 4
      bp_gs[bp_m] = bp_gsn[bp_m]
      bp_m = bp_m + 1
    end
    bp_t = bp_t - 1
  end
  1
end

# ---- the DFA broadcast update ----
#
# e = the normalised state error at the best-approach step; d1 = B1.e
# and d2 = B2.e are FIXED for the whole episode — that is the point of
# a broadcast. No Jacobian is read anywhere in here, which is what
# makes `dfa_tb` a fair stand-in for the signal the GA had.
def tk_dfa!(df_w, df_gr, df_ex, df_ey, df_eang, df_gstep, df_b1, df_b2,
            df_nin, df_hid,
            df_ow1, df_ob1, df_ow2, df_ob2, df_kind,
            df_a, df_h, df_yout, df_obs, df_e, df_d1,
            df_hidden_only, df_to_best, df_T,
            df_emode, df_edim, df_mean, df_decay, df_enorm)
  # toy#193: the error the broadcast projects. LANDMINE 5 of toy#189
  # still holds in every mode — e is an ERROR, so it is normalised as
  # one (dx/50), never through the observation's (x-50)/50, which would
  # be non-zero AT THE GOAL.
  if df_emode == 1
    df_e[0] = df_ey / 50.0
    df_e[1] = df_eang / Math::PI
  elsif df_emode == 3
    # The signed distance along the trailer's heading to the dock: the
    # dock is at the origin, so the vector to it is (-x, -y) and the
    # heading is (cos, sin) of the graded trailer angle. Positive when
    # the dock is AHEAD of the trailer, negative when behind — which is
    # the sign change dx does not have.
    df_e[0] = (0.0 - df_ex * Math.cos(df_eang) - df_ey * Math.sin(df_eang)) / 50.0
    df_e[1] = df_ey / 50.0
    df_e[2] = df_eang / Math::PI
  else
    df_e[0] = df_ex / 50.0
    df_e[1] = df_ey / 50.0
    df_e[2] = df_eang / Math::PI
  end

  # (centering first, then normalisation: normalising a vector whose
  # mean has not been removed would just rescale the same locked
  # direction.)
  # `centered` subtracts a running mean, so a component that is always
  # positive averages to zero. The mean is an EMA updated AFTER this
  # episode's projection: subtracting a mean that already contains this
  # episode would shrink the very signal under test.
  if df_emode == 2
    df_ci = 0
    while df_ci < df_edim
      df_raw = df_e[df_ci]
      df_e[df_ci] = df_raw - df_mean[df_ci]
      df_mean[df_ci] = df_decay * df_mean[df_ci] + (1.0 - df_decay) * df_raw
      df_ci = df_ci + 1
    end
  end

  # Direction only: divide by ||e||. A near-zero error is left alone —
  # dividing it would AMPLIFY the one episode that went well into the
  # largest update of the run, which is the opposite of what this is
  # testing.
  if df_enorm == 1
    df_nn = 0.0
    df_ni = 0
    while df_ni < df_edim
      df_nn = df_nn + df_e[df_ni] * df_e[df_ni]
      df_ni = df_ni + 1
    end
    df_nn = Math.sqrt(df_nn)
    if df_nn > 1.0e-9
      df_nj = 0
      while df_nj < df_edim
        df_e[df_nj] = df_e[df_nj] / df_nn
        df_nj = df_nj + 1
      end
    end
  end

  df_j = 0
  while df_j < df_hid
    df_acc = 0.0
    df_k = 0
    while df_k < df_edim
      df_acc = df_acc + df_b1[df_j * df_edim + df_k] * df_e[df_k]
      df_k = df_k + 1
    end
    df_d1[df_j] = df_acc
    df_j = df_j + 1
  end
  df_d2 = 0.0
  df_k2 = 0
  while df_k2 < df_edim
    df_d2 = df_d2 + df_b2[df_k2] * df_e[df_k2]
    df_k2 = df_k2 + 1
  end

  df_n = df_to_best == 1 ? df_gstep : df_T
  df_t = 0
  while df_t < df_n
    df_dj = 0
    while df_dj < df_hid
      df_hv = df_h[df_t * df_hid + df_dj]
      df_da = df_d1[df_dj] * tk_dact(df_hv, df_kind)
      df_gr[df_ob1 + df_dj] = df_gr[df_ob1 + df_dj] + df_da
      df_dk = 0
      while df_dk < df_nin
        df_gr[df_ow1 + df_dj * df_nin + df_dk] =
          df_gr[df_ow1 + df_dj * df_nin + df_dk] +
          df_da * df_obs[df_t * df_nin + df_dk]
        df_dk = df_dk + 1
      end
      if df_hidden_only == 0
        df_gr[df_ow2 + df_dj] = df_gr[df_ow2 + df_dj] +
          df_d2 * tk_dsignal_dout(df_yout[df_t], df_kind) * df_hv
      end
      df_dj = df_dj + 1
    end
    if df_hidden_only == 0
      df_gr[df_ob2] = df_gr[df_ob2] +
        df_d2 * tk_dsignal_dout(df_yout[df_t], df_kind)
    end
    df_t = df_t + 1
  end
  nil
end

# ---- start-state selection ----

ST_ENSEMBLE = 0
ST_POINT    = 1
ST_YARD     = 2
ST_LESSON   = 3
# toy#192 / C2b — THE SIGN TEST. The yard draw restricted to ONE SIDE,
# plus its mirror. The candidate mechanism is that a FIXED B cannot
# carry a STATE-DEPENDENT SIGN: from y > 0 the correcting steering has
# one sign and from y < 0 the other, the error's dy component flips with
# it, so a fixed random map from error to weight direction is right on
# one half of the yard and wrong on the other — and over a SYMMETRIC
# start distribution the two halves cancel into a constant push.
#
# Same seeded LCG and the same bounds as `yard` on the x and angle axes,
# so a half-yard bundle is reproducible from its seed exactly as the
# full yard is and the frontend can overlay both.
ST_HALF_YARD     = 4
ST_HALF_YARD_NEG = 5
START_MODE = START_S == "point" ? ST_POINT :
             START_S == "yard"  ? ST_YARD :
             START_S == "lesson" ? ST_LESSON :
             START_S == "half_yard" ? ST_HALF_YARD :
             START_S == "half_yard_neg" ? ST_HALF_YARD_NEG : ST_ENSEMBLE

# Path length over the TRAILER point. ONE implementation, called by both
# the eval line and the trace bundle: two copies of this would be two
# numbers to keep in parity for no reason, and the bundle's `path_len`
# and the eval line's `median_path_len` are meant to be the same
# quantity. rs_x/rs_y hold the state BEFORE each step, so the final
# state comes from the plant.
def tk_path_len(pl_plant, pl_x, pl_y, pl_T, pl_sx, pl_sy)
  pl_acc = 0.0
  pl_px = pl_sx
  pl_py = pl_sy
  pl_k = 1
  while pl_k <= pl_T
    pl_nx = pl_k < pl_T ? pl_x[pl_k] : pl_plant.tt_x
    pl_ny = pl_k < pl_T ? pl_y[pl_k] : pl_plant.tt_y
    pl_dx = pl_nx - pl_px
    pl_dy = pl_ny - pl_py
    pl_acc = pl_acc + Math.sqrt(pl_dx * pl_dx + pl_dy * pl_dy)
    pl_px = pl_nx
    pl_py = pl_ny
    pl_k = pl_k + 1
  end
  pl_acc
end

# Median of the first `md_n` entries of `md_a`. Insertion sort: n is the
# eval-set size (<= a few hundred), and a partial sort would be more
# code than the whole function.
def tk_median(md_a, md_n)
  if md_n < 1; return 0.0; end
  md_i = 1
  while md_i < md_n
    md_v = md_a[md_i]
    md_j = md_i - 1
    while md_j >= 0 && md_a[md_j] > md_v
      md_a[md_j + 1] = md_a[md_j]
      md_j = md_j - 1
    end
    md_a[md_j + 1] = md_v
    md_i = md_i + 1
  end
  if md_n % 2 == 1
    return md_a[md_n / 2]
  end
  0.5 * (md_a[md_n / 2 - 1] + md_a[md_n / 2])
end

# Name -> mode. The inverse of tk_scheme_name, so the rollout can reuse
# tk_set_start! instead of dispatching on the scheme STRING itself.
# The trace writer used to carry its own if/elsif chain and its `else`
# swallowed `half_yard` and `half_yard_neg` into a full-yard draw — the
# bundle then said `scheme: "half_yard"` over starts spanning the whole
# yard, which is a wrong number that looks right. Returns -1 for an
# unknown name so the caller can refuse it.
def tk_scheme_mode(sm_name)
  if sm_name == "point";         return 1; end
  if sm_name == "yard";          return 2; end
  if sm_name == "lesson";        return 3; end
  if sm_name == "half_yard";     return 4; end
  if sm_name == "half_yard_neg"; return 5; end
  if sm_name == "ensemble";      return 0; end
  -1
end

def tk_scheme_name(sn_mode)
  if sn_mode == 1; return "point"; end
  if sn_mode == 2; return "yard"; end
  if sn_mode == 3; return "lesson"; end
  if sn_mode == 4; return "half_yard"; end
  if sn_mode == 5; return "half_yard_neg"; end
  "ensemble"
end

def tk_set_start!(sp_plant, sp_mode, sp_k, sp_lesson)
  if sp_mode == 1
    sp_plant.point_start!
    return nil
  end
  if sp_mode == 2
    sp_plant.sample_yard!(64)
    return nil
  end
  if sp_mode == 3
    sp_plant.sample_lesson!(sp_lesson, 64)
    return nil
  end
  if sp_mode == 4 || sp_mode == 5
    # Half-yard: the y bounds are narrowed AROUND the draw and put back,
    # so the caller's bounds (which the eval sets also move) survive and
    # only this one draw is restricted.
    sp_lo = sp_plant.tt_y_min
    sp_hi = sp_plant.tt_y_max
    if sp_mode == 4
      sp_plant.tt_y_min = 0.0
    else
      sp_plant.tt_y_max = 0.0
    end
    sp_plant.sample_yard!(64)
    sp_plant.tt_y_min = sp_lo
    sp_plant.tt_y_max = sp_hi
    return nil
  end
  sp_plant.ensemble_start!(sp_k % 15)
  nil
end

# How many starts one "update" aggregates over.
N_START = START_MODE == ST_ENSEMBLE ? 15 : 1

# ---- init: uniform [-1,1], the paper's ----

w  = tk_zeros(N_PARAM)
gr = tk_zeros(N_PARAM)
w_state = [tk_seed_state(SEED)]
iw = 0
while iw < N_PARAM
  w[iw] = tk_next_u(w_state) * 2.0 - 1.0
  iw = iw + 1
end

# ---- the feedback matrices, from DfaB ----

B_DIST = B_DIST_S == "uniform" ? Toy::Train::DfaB::DIST_UNIFORM :
         B_DIST_S == "rademacher" ? Toy::Train::DfaB::DIST_RADEMACHER :
         Toy::Train::DfaB::DIST_GAUSSIAN
B_SCALE = B_SCALE_S == "glorot" ? Toy::Train::DfaB::SCALE_GLOROT :
          B_SCALE_S == "fixed" ? Toy::Train::DfaB::SCALE_FIXED :
          Toy::Train::DfaB::SCALE_INV_SQRT_FAN
# fan_in = the ERROR dim (3, not the vocab this helper was written for),
# fan_out = the projected-into dim.
B_SIGMA1 = Toy::Train::DfaB.sigma_for(B_SCALE, E_DIM, HIDDEN, 1.0)
B_SIGMA2 = Toy::Train::DfaB.sigma_for(B_SCALE, E_DIM, 1, 1.0)
b1m = Toy::Train::DfaB.fill(HIDDEN * E_DIM, B_SEED, B_DIST, B_SIGMA1)
b2m = Toy::Train::DfaB.fill(E_DIM, B_SEED + 7919, B_DIST, B_SIGMA2)

# toy#190: a loaded controller REPLACES the init and training is
# skipped. Done after the init draw rather than instead of it so the
# init RNG stream is identical either way — an arm that loads and one
# that trains must not diverge in what else the seed drives.
# Loads a controller in the TRUCK_EXPORT format into `lc_w`. Returns 1
# on success, 0 on a count mismatch (which the caller reports and dies
# on). ONE implementation, used by both TRUCK_LOAD (replay a trained
# controller) and TRUCK_EXPERT (toy#194's teacher) — a second copy is a
# second thing to keep in parity with the exporter.
def tk_load_ctrl!(lc_path, lc_w, lc_nin, lc_hid, lc_ow1, lc_ob1, lc_ow2,
                  lc_ob2, lc_nparam, lc_buf)
  lc_raw = File.read(lc_path)
  lc_n = tk_scan_floats(lc_raw, lc_buf, lc_nparam + 8)
  if lc_n != lc_nparam
    puts "toy-train-truck: " + lc_path + " holds " + lc_n.to_s +
         " numbers, expected " + lc_nparam.to_s + " for a " + lc_nin.to_s +
         "-" + lc_hid.to_s + "-1 net (check --obs and --hidden against the sidecar)"
    return 0
  end
  lc_k = 0
  lc_j = 0
  while lc_j < lc_hid
    lc_c = 0
    while lc_c < lc_nin
      lc_w[lc_ow1 + lc_j * lc_nin + lc_c] = lc_buf[lc_k]
      lc_k = lc_k + 1
      lc_c = lc_c + 1
    end
    lc_w[lc_ob1 + lc_j] = lc_buf[lc_k]
    lc_k = lc_k + 1
    lc_j = lc_j + 1
  end
  lc_j2 = 0
  while lc_j2 < lc_hid
    lc_w[lc_ow2 + lc_j2] = lc_buf[lc_k]
    lc_k = lc_k + 1
    lc_j2 = lc_j2 + 1
  end
  lc_w[lc_ob2] = lc_buf[lc_k]
  1
end

ld_buf = tk_zeros(N_PARAM + 8)

# toy#190: a loaded controller REPLACES the init and training is
# skipped. Done after the init draw rather than instead of it so the
# init RNG stream is identical either way.
SKIP_TRAIN = LOAD_PATH.length > 0
if SKIP_TRAIN
  if tk_load_ctrl!(LOAD_PATH, w, N_IN, HIDDEN, OFF_W1, OFF_B1, OFF_W2,
                   OFF_B2, N_PARAM, ld_buf) == 0
    exit 1
  end
end

gs   = tk_zeros(4)
gsn  = tk_zeros(4)
dobs = tk_zeros(N_IN)
e3   = tk_zeros(E_DIM)
e_mean = tk_zeros(E_DIM)
d1v  = tk_zeros(HIDDEN)

# ======================================================================
# SELFTEST — landmine 1. The ceiling arm's gradient, finite-differenced
# against the plant itself, every parameter, including a configuration
# where the jack-knife clamp fires.
# ======================================================================

ST_CFG_X  = [60.0, 60.0, 20.0, 70.0]
ST_CFG_Y  = [0.0,  0.0,  10.0, 30.0]
ST_CFG_OS = [0.0,  0.0,  -2.0, 0.2]
ST_CFG_OC = [0.0,  1.5,  -2.0, 1.7]
ST_NCFG   = 4

if SELFTEST
  # FOUR configurations x TWO grading modes, with the suite's coverage
  # ACCOUNTED FOR rather than assumed. Two vacuous passes were caught
  # writing this test, and both are now assertions:
  #
  #  - the first version graded the paper's single point and reported
  #    worst_rel = 0.0, a PERFECT gradcheck, because an untrained random
  #    policy drives away immediately: t* was 0 and the gradient was
  #    trivially zero (landmine 4).
  #  - the second asserted "a clamped step occurred", which was true
  #    episode-wide while the clamp row substitution was DEAD CODE —
  #    disabling it left every number BIT-IDENTICAL. Only a clamped step
  #    at t <= t*-2 can matter: the cab-angle gradient is exactly zero
  #    at the graded step and becomes non-zero one step further back.
  #
  # So mode 0 grades the best-approach step (what the arms descend) and
  # mode 1 grades the TERMINAL state, which puts the whole episode in
  # the backward window and is the only mode that can gate landmine 2.
  st_fails = 0
  st_deep = 0
  st_clamp_eff = 0
  st_mode = 0
  while st_mode < 2
    st_cfg = 0
    while st_cfg < ST_NCFG
      st_x  = ST_CFG_X[st_cfg]
      st_y  = ST_CFG_Y[st_cfg]
      st_os = ST_CFG_OS[st_cfg]
      st_oc = ST_CFG_OC[st_cfg]
      st_cap = 40

      plant.tt_max_steps = st_cap
      jplant.tt_max_steps = st_cap

      plant.reset_state!(st_x, st_y, st_oc, st_os)
      tk_rollout(plant, w, st_cap, N_IN, HIDDEN, OFF_W1, OFF_B1, OFF_W2, OFF_B2,
                 ACT, rs_x, rs_y, rs_oc, rs_os, rs_clamp, rs_sig, rs_o, rs_yout,
                 rs_a, rs_h, rs_obs, fw_out, ro)
      st_T = ro[0].to_i
      st_grade = st_mode == 0 ? ro[2].to_i : st_T
      st_gx = st_mode == 0 ? ro[3] : ro[7]
      st_gy = st_mode == 0 ? ro[4] : ro[8]
      st_ga = st_mode == 0 ? ro[5] : ro[9]

      # Clamped steps that can actually change the result: t <= grade-2.
      st_eff = 0
      st_wc = 0
      while st_wc < st_grade - 1
        if rs_clamp[st_wc] == 1
          st_eff = st_eff + 1
        end
        st_wc = st_wc + 1
      end

      st_i = 0
      while st_i < N_PARAM
        gr[st_i] = 0.0
        st_i = st_i + 1
      end
      tk_bptt!(jplant, w, gr, st_grade, st_gx, st_gy, st_ga,
               N_IN, HIDDEN, OFF_W1, OFF_B1, OFF_W2, OFF_B2,
               ACT, rs_x, rs_y, rs_oc, rs_os, rs_clamp, rs_sig, rs_o, rs_yout,
               rs_a, rs_h, rs_obs, gs, gsn, dobs, 0)

      st_h = 1.0e-5
      st_worst = 0.0
      st_worst_i = 0
      st_moved = 0
      st_p = 0
      while st_p < N_PARAM
        st_save = w[st_p]

        w[st_p] = st_save + st_h
        plant.reset_state!(st_x, st_y, st_oc, st_os)
        tk_rollout(plant, w, st_cap, N_IN, HIDDEN, OFF_W1, OFF_B1, OFF_W2,
                   OFF_B2, ACT, rs_x, rs_y, rs_oc, rs_os, rs_clamp, rs_sig,
                   rs_o, rs_yout, rs_a, rs_h, rs_obs, fw_out, ro)
        st_lhi = st_mode == 0 ? ro[1] : ro[6]
        st_thi = st_mode == 0 ? ro[2].to_i : ro[0].to_i
        st_Thi = ro[0].to_i

        w[st_p] = st_save - st_h
        plant.reset_state!(st_x, st_y, st_oc, st_os)
        tk_rollout(plant, w, st_cap, N_IN, HIDDEN, OFF_W1, OFF_B1, OFF_W2,
                   OFF_B2, ACT, rs_x, rs_y, rs_oc, rs_os, rs_clamp, rs_sig,
                   rs_o, rs_yout, rs_a, rs_h, rs_obs, fw_out, ro)
        st_llo = st_mode == 0 ? ro[1] : ro[6]
        st_tlo = st_mode == 0 ? ro[2].to_i : ro[0].to_i
        st_Tlo = ro[0].to_i

        w[st_p] = st_save

        # LANDMINE 3: if the graded step or the episode length moved,
        # the difference is between two different functions. Count it;
        # do not report it as a gradient error.
        if st_thi != st_grade || st_tlo != st_grade || st_Thi != st_T || st_Tlo != st_T
          st_moved = st_moved + 1
        else
          st_fd = (st_lhi - st_llo) / (2.0 * st_h)
          st_d = st_fd - gr[st_p]
          if st_d < 0.0; st_d = -st_d; end
          st_scale = gr[st_p] < 0.0 ? -gr[st_p] : gr[st_p]
          if st_scale < 1.0; st_scale = 1.0; end
          st_rel = st_d / st_scale
          if st_rel > st_worst
            st_worst = st_rel
            st_worst_i = st_p
          end
        end
        st_p = st_p + 1
      end

      if st_grade >= 5;  st_deep = 1;      end
      if st_eff > 0;     st_clamp_eff = 1; end

      st_tag = "mode=" + (st_mode == 0 ? "best" : "terminal") +
               " cfg=" + st_cfg.to_s
      if st_worst < 1.0e-5
        puts "selftest: gradcheck " + st_tag + " ok n=" + N_PARAM.to_s +
             " T=" + st_T.to_s + " grade=" + st_grade.to_s +
             " eff_clamps=" + st_eff.to_s +
             " worst_rel=" + st_worst.to_s + " moved=" + st_moved.to_s
      else
        puts "selftest: gradcheck " + st_tag + " FAIL worst_rel=" +
             st_worst.to_s + " at param " + st_worst_i.to_s +
             " (T=" + st_T.to_s + " grade=" + st_grade.to_s +
             " eff_clamps=" + st_eff.to_s + " moved=" + st_moved.to_s + ")"
        st_fails = st_fails + 1
      end
      st_cfg = st_cfg + 1
    end
    st_mode = st_mode + 1
  end

  # SUITE COVERAGE — both of these can pass vacuously, so both are
  # asserted rather than trusted.
  if st_deep == 1
    puts "selftest: coverage ok — a case graded at step >= 5, so the BPTT recursion ran"
  else
    puts "selftest: coverage FAIL — nothing graded deeper than step 5; a zero " +
         "gradient would read as a perfect check"
    st_fails = st_fails + 1
  end
  if st_clamp_eff == 1
    puts "selftest: coverage ok — a clamped step at t <= grade-2, so the Oc row " +
         "substitution is load-bearing here"
  else
    puts "selftest: coverage FAIL — no clamped step early enough to matter; the " +
         "Oc row substitution is dead code and landmine 2 is ungated"
    st_fails = st_fails + 1
  end

  if st_fails == 0
    puts "selftest: PASS"
    exit 0
  end
  puts "selftest: FAIL (" + st_fails.to_s + ")"
  exit 1
end

# ======================================================================
# toy#194 — DEMONSTRATIONS, and the imitation training loop.
# ======================================================================

wx      = tk_zeros(N_PARAM)      # the expert
dm_obs  = [0.0]; dm_obs.pop      # (state, action) pairs
dm_u    = [0.0]; dm_u.pop
dm_n    = 0
expert_d2 = 0.0

# B1 for the imitation arms is HIDDEN x 1: the error is a SCALAR here,
# not a 3-vector, because it is per-step rather than a broadcast.
B_SIGMA_I = Toy::Train::DfaB.sigma_for(B_SCALE, 1, HIDDEN, 1.0)
b1i = Toy::Train::DfaB.fill(HIDDEN, B_SEED + 104729, B_DIST, B_SIGMA_I)

if IS_IMIT
  if EXPERT.length == 0
    puts "toy-train-truck: TRUCK_ARM=" + ARM_S + " requires TRUCK_EXPERT=<controller.json>"
    exit 1
  end
  if tk_load_ctrl!(EXPERT, wx, N_IN, HIDDEN, OFF_W1, OFF_B1, OFF_W2,
                   OFF_B2, N_PARAM, ld_buf) == 0
    exit 1
  end

  # THE TWO REGIMES.
  #   scarce   — the paper's 15 ensemble starts and nothing else. Highly
  #              correlated, ~15 x cap pairs. This is the regime the arc
  #              exists to find (F17/D3b), and it is the paper's own
  #              training set, so "the GA was provably efficient here"
  #              holds by construction.
  #   abundant — TRUCK_DEMO_N yard starts: tens of thousands of pairs.
  dm_starts = DEMOS_SCARCE ? 15 : DEMO_N
  plant.seed!(EVAL_SEED + 77)
  dm_i = 0
  while dm_i < dm_starts
    if DEMOS_SCARCE
      plant.ensemble_start!(dm_i)
    else
      plant.sample_yard!(64)
    end
    tk_rollout(plant, wx, STEP_CAP, N_IN, HIDDEN, OFF_W1, OFF_B1, OFF_W2,
               OFF_B2, ACT, rs_x, rs_y, rs_oc, rs_os, rs_clamp, rs_sig,
               rs_o, rs_yout, rs_a, rs_h, rs_obs, fw_out, ro)
    expert_d2 = expert_d2 + ro[1]
    dm_t = 0
    while dm_t < ro[0].to_i
      dm_k = 0
      while dm_k < N_IN
        dm_obs.push(rs_obs[dm_t * N_IN + dm_k])
        dm_k = dm_k + 1
      end
      dm_u.push(rs_sig[dm_t])
      dm_n = dm_n + 1
      dm_t = dm_t + 1
    end
    dm_i = dm_i + 1
  end
  expert_d2 = expert_d2 / dm_starts.to_f
  puts "demos: n_pairs=" + dm_n.to_s + " starts=" + dm_starts.to_s +
       " regime=" + (DEMOS_SCARCE ? "scarce" : "abundant") +
       " expert=" + EXPERT + " expert_mean_d2=" + expert_d2.to_s

  # THE EXPERT MUST BE ABLE TO DRIVE. A student cannot be better than
  # its teacher's data, so an expert that does not dock makes every
  # imitation number uninterpretable — said loudly rather than left in
  # the provenance for a reader to notice.
  if dm_n < 1
    puts "toy-train-truck: the expert produced no demonstration pairs"
    exit 1
  end
  if expert_d2 > 5.0
    puts "warning: the expert's own mean d2 is " + expert_d2.to_s +
         " (> 5); it does not dock its own start set, so the imitation" +
         " arms are fitting a policy that cannot drive"
  end
end

# ======================================================================
# TRAINING
# ======================================================================

plant.tt_max_steps = STEP_CAP
jplant.tt_max_steps = STEP_CAP

# GA state, allocated only for that arm (an empty typed seed otherwise,
# so the ivar's type is not conditional).
pop  = [0.0]; pop.pop
fit  = [0.0]; fit.pop
kid  = [0.0]; kid.pop
ga_state = [tk_seed_state(SEED + 31)]
if ARM == ARM_GA
  gi = 0
  while gi < GA_POP * N_PARAM
    pop.push(tk_next_u(ga_state) * 2.0 - 1.0)
    gi = gi + 1
  end
  gj = 0
  while gj < GA_POP
    fit.push(0.0)
    gj = gj + 1
  end
  gk = 0
  while gk < GA_POP * N_PARAM
    kid.push(0.0)
    gk = gk + 1
  end
  # The paper's GA is seeded the same way its net is: uniform [-1,1].
  # Member 0 is the SHARED init, so every arm starts from one point.
  gz = 0
  while gz < N_PARAM
    pop[gz] = w[gz]
    gz = gz + 1
  end
end

train_state = [tk_seed_state(SEED + 101)]
plant_steps = 0
updates     = 0
zero_grad   = 0
final_loss  = 0.0
imit_mse    = -1.0
ga_best_fit = 0.0
ga_best_d2  = 0.0

# Scores one parameter vector over the start set: mean d^2, and the
# fitness aggregation the GA selects on. Returns [mean_d2, agg_fit,
# steps_used, min_d2].
sc = tk_zeros(4)

def tk_score(scp_plant, scp_w, scp_state, scp_mode, scp_nstart, scp_cap, scp_lesson,
             scp_nin, scp_hid, scp_ow1, scp_ob1, scp_ow2, scp_ob2, scp_kind,
             scp_x, scp_y, scp_oc, scp_os, scp_clamp, scp_sig, scp_o, scp_yout,
             scp_a, scp_h, scp_obs, scp_fw, scp_ro, scp_agg_min,
             scp_eps, scp_gamma, scp_out)
  scp_sum = 0.0
  scp_min = 1.0e18
  scp_fsum = 0.0
  scp_fmin = 1.0e18
  scp_steps = 0
  scp_k = 0
  while scp_k < scp_nstart
    tk_set_start!(scp_plant, scp_mode, scp_k, scp_lesson)
    tk_rollout(scp_plant, scp_w, scp_cap, scp_nin, scp_hid,
               scp_ow1, scp_ob1, scp_ow2, scp_ob2, scp_kind,
               scp_x, scp_y, scp_oc, scp_os, scp_clamp, scp_sig, scp_o,
               scp_yout, scp_a, scp_h, scp_obs, scp_fw, scp_ro)
    scp_d2 = scp_ro[1]
    scp_l  = scp_ro[0].to_i
    scp_steps = scp_steps + scp_l
    scp_sum = scp_sum + scp_d2
    if scp_d2 < scp_min; scp_min = scp_d2; end
    scp_f = scp_plant.sr_fitness(scp_d2, scp_l, scp_eps, scp_gamma)
    scp_fsum = scp_fsum + scp_f
    if scp_f < scp_fmin; scp_fmin = scp_f; end
    scp_k = scp_k + 1
  end
  scp_out[0] = scp_sum / scp_nstart.to_f
  scp_out[1] = scp_agg_min == 1 ? scp_fmin : (scp_fsum / scp_nstart.to_f)
  scp_out[2] = scp_steps.to_f
  scp_out[3] = scp_min
  nil
end

GA_AGG_MIN = GA_AGG_S == "min" ? 1 : 0

# TOP-LEVEL, never inside the arm branch: a constant assigned inside a
# conditional arm reads back EMPTY at runtime under Spinel (the runner
# landmine the mlp lane's header names). `frozen` is the arm that would
# have silently become `bptt`.
READOUT_ONLY = ARM == ARM_FROZEN ? 1 : 0

if IS_IMIT && !SKIP_TRAIN
  # ---- imitation: per-step supervised regression on the expert's
  # action, with NO PLANT IN THE LOOP during training. The closed-loop
  # score afterwards is what says whether the student can DRIVE, which
  # is a different question from whether it FITS (compounding error).
  im_state = [tk_seed_state(SEED + 991)]
  im_gr = tk_zeros(N_PARAM)
  im_fw = tk_zeros(2)
  im_a = tk_zeros(HIDDEN)
  im_h = tk_zeros(HIDDEN)
  im_last = 0.0
  im_upd = 0
  while im_upd < STEPS
    gi3 = 0
    while gi3 < N_PARAM
      im_gr[gi3] = 0.0
      gi3 = gi3 + 1
    end
    im_sse = 0.0
    im_b = 0
    while im_b < DEMO_BATCH
      im_p = tk_next_int(im_state, dm_n)
      im_sig = tk_forward(w, dm_obs, im_p * N_IN, im_a, im_h, 0,
                          N_IN, HIDDEN, OFF_W1, OFF_B1, OFF_W2, OFF_B2,
                          ACT, im_fw)
      im_e = dm_u[im_p] - im_sig
      im_sse = im_sse + im_e * im_e

      # d(0.5 e^2)/d(out) — the readout is EXACT on all three arms; they
      # differ only in what the HIDDEN layer is told.
      im_do = (0.0 - im_e) * tk_dsignal_dout(im_fw[1], ACT)
      im_j = 0
      while im_j < HIDDEN
        im_hv = im_h[im_j]
        im_gr[OFF_W2 + im_j] = im_gr[OFF_W2 + im_j] + im_do * im_hv
        if ARM == ARM_IMIT_BP
          im_da = im_do * w[OFF_W2 + im_j] * tk_dact(im_hv, ACT)
        elsif ARM == ARM_IMIT_DFA
          # Textbook DFA: the hidden layer sees a FIXED RANDOM
          # projection of the same per-step error the readout sees.
          im_da = (0.0 - b1i[im_j] * im_e) * tk_dact(im_hv, ACT)
        else
          im_da = 0.0
        end
        if ARM != ARM_IMIT_FROZEN
          im_gr[OFF_B1 + im_j] = im_gr[OFF_B1 + im_j] + im_da
          im_k3 = 0
          while im_k3 < N_IN
            im_gr[OFF_W1 + im_j * N_IN + im_k3] =
              im_gr[OFF_W1 + im_j * N_IN + im_k3] +
              im_da * dm_obs[im_p * N_IN + im_k3]
            im_k3 = im_k3 + 1
          end
        end
        im_j = im_j + 1
      end
      im_gr[OFF_B2] = im_gr[OFF_B2] + im_do
      im_b = im_b + 1
    end

    im_sc = LR / DEMO_BATCH.to_f
    im_z = 0
    while im_z < N_PARAM
      # WEIGHT DECAY applies to every imitation arm, not just imit_bp:
      # a DFA "win" on scarce data against an unregularised BP would be
      # a win against a strawman (F17's comparison was against a fully
      # regularised BP), and giving it to only one arm would be the
      # same asymmetry pointing the other way.
      im_gr[im_z] = im_gr[im_z] + WD * w[im_z]
      w[im_z] = w[im_z] - im_sc * im_gr[im_z]
      im_z = im_z + 1
    end
    im_last = im_sse / DEMO_BATCH.to_f
    im_upd = im_upd + 1
    updates = updates + 1
    # `loss=` only, like every other arm. The step line used to repeat
    # the same number as `imit_mse=`, which made a reader (and a regex)
    # find step 1's value where they wanted the final one.
    puts "step " + updates.to_s + ": loss=" + im_last.to_s
  end
  final_loss = im_last
  imit_mse = im_last
elsif SKIP_TRAIN
  # A loaded controller is rolled out as-is. `updates` stays 0 and the
  # provenance says so, so a trace bundle can never be mistaken for one
  # produced by a run that also trained.
  final_loss = 0.0
elsif ARM == ARM_GA
  # ---- the GA: tournament selection, BLX-0.5 crossover, gaussian
  # mutation, elitism. THE OPERATORS ARE OURS AND SAID TO BE OURS: the
  # paper specifies a real-valued chromosome and "hybridised" and
  # nothing else, so matching operators nobody can see is not a check.
  # HITTING ITS NUMBERS is the check.
  gen = 0
  while gen < GA_GENS
    # Evaluate. The start stream is re-seeded per generation so every
    # member of one generation faces the SAME starts (an unpaired
    # comparison here would select on start luck — B0a's lesson).
    plant.seed!(EVAL_SEED + gen)
    best_i = 0
    best_f = -1.0e18
    pi2 = 0
    while pi2 < GA_POP
      pw = [0.0]; pw.pop
      pz = 0
      while pz < N_PARAM
        pw.push(pop[pi2 * N_PARAM + pz])
        pz = pz + 1
      end
      tk_score(plant, pw, train_state, START_MODE, N_START, STEP_CAP, LESSON,
               N_IN, HIDDEN, OFF_W1, OFF_B1, OFF_W2, OFF_B2, ACT,
               rs_x, rs_y, rs_oc, rs_os, rs_clamp, rs_sig, rs_o, rs_yout,
               rs_a, rs_h, rs_obs, fw_out, ro, GA_AGG_MIN,
               FIT_EPS, FIT_GAMMA, sc)
      fit[pi2] = sc[1]
      plant_steps = plant_steps + sc[2].to_i
      if sc[1] > best_f
        best_f = sc[1]
        best_i = pi2
        ga_best_d2 = sc[0]
      end
      pi2 = pi2 + 1
    end
    ga_best_fit = best_f
    final_loss = ga_best_d2

    # The elite carry over verbatim, then tournament + BLX + mutate.
    ez = 0
    while ez < N_PARAM
      w[ez] = pop[best_i * N_PARAM + ez]
      ez = ez + 1
    end
    ec = 0
    while ec < GA_ELITE && ec < GA_POP
      # elite 0 is the generation's best; the rest are filled by
      # repeated tournaments rather than a sort (one pass, no
      # allocation, and ties break on the stream not on index order).
      src = ec == 0 ? best_i : tk_next_int(ga_state, GA_POP)
      ez2 = 0
      while ez2 < N_PARAM
        kid[ec * N_PARAM + ez2] = pop[src * N_PARAM + ez2]
        ez2 = ez2 + 1
      end
      ec = ec + 1
    end
    ki = GA_ELITE
    while ki < GA_POP
      a1 = tk_next_int(ga_state, GA_POP)
      a2 = tk_next_int(ga_state, GA_POP)
      a3 = tk_next_int(ga_state, GA_POP)
      pa = fit[a1] >= fit[a2] ? a1 : a2
      pa = fit[pa] >= fit[a3] ? pa : a3
      b1i = tk_next_int(ga_state, GA_POP)
      b2i = tk_next_int(ga_state, GA_POP)
      b3i = tk_next_int(ga_state, GA_POP)
      pb = fit[b1i] >= fit[b2i] ? b1i : b2i
      pb = fit[pb] >= fit[b3i] ? pb : b3i
      kz = 0
      while kz < N_PARAM
        va = pop[pa * N_PARAM + kz]
        vb = pop[pb * N_PARAM + kz]
        lo = va < vb ? va : vb
        hi = va < vb ? vb : va
        d = hi - lo
        v = lo - 0.5 * d + tk_next_u(ga_state) * (2.0 * d + 1.0e-12)
        v = v + tk_gauss(ga_state) * GA_MUT
        if v > 1.0e6;  v = 1.0e6;  end
        if v < -1.0e6; v = -1.0e6; end
        kid[ki * N_PARAM + kz] = v
        kz = kz + 1
      end
      ki = ki + 1
    end
    sw = 0
    while sw < GA_POP * N_PARAM
      pop[sw] = kid[sw]
      sw = sw + 1
    end

    updates = updates + 1
    puts "step " + updates.to_s + ": loss=" + ga_best_d2.to_s
    if BUDGET > 0 && plant_steps >= BUDGET
      gen = GA_GENS
    end
    gen = gen + 1
  end
else
  # ---- the gradient / broadcast arms: one update per pass over the
  # start set, at matched plant steps.
  upd = 0
  while upd < STEPS
    plant.seed!(EVAL_SEED + upd)
    gi2 = 0
    while gi2 < N_PARAM
      gr[gi2] = 0.0
      gi2 = gi2 + 1
    end
    loss_sum = 0.0
    score_sum = 0.0
    contributed = 0
    ks = 0
    while ks < N_START
      tk_set_start!(plant, START_MODE, ks, LESSON)
      tk_rollout(plant, w, STEP_CAP, N_IN, HIDDEN, OFF_W1, OFF_B1, OFF_W2,
                 OFF_B2, ACT, rs_x, rs_y, rs_oc, rs_os, rs_clamp, rs_sig,
                 rs_o, rs_yout, rs_a, rs_h, rs_obs, fw_out, ro)
      # The SCORE is always the nearest approach (the paper's). The
      # GRADED step is TRUCK_LOSS's — kept apart on purpose.
      score_sum = score_sum + ro[1]
      g_step = LOSS_TERMINAL ? ro[0].to_i : ro[2].to_i
      g_x    = LOSS_TERMINAL ? ro[7] : ro[3]
      g_y    = LOSS_TERMINAL ? ro[8] : ro[4]
      g_ang  = LOSS_TERMINAL ? ro[9] : ro[5]
      loss_sum = loss_sum + (LOSS_TERMINAL ? ro[6] : ro[1])
      plant_steps = plant_steps + ro[0].to_i

      if ARM == ARM_DFA_TB
        tk_dfa!(w, gr, g_x, g_y, g_ang, g_step, b1m, b2m, N_IN, HIDDEN,
                OFF_W1, OFF_B1, OFF_W2, OFF_B2, ACT, rs_a, rs_h, rs_yout,
                rs_obs, e3, d1v, 0, SUM_TO_BEST ? 1 : 0, ro[0].to_i,
                E_MODE, E_DIM, e_mean, E_DECAY, E_NORM)
        # A BROADCAST arm's update is non-zero whenever the episode took
        # a step: its error is the graded STATE, not a difference, so it
        # does not vanish when the graded step is 0. Counting this arm by
        # the gradient arms' rule (t* >= 1) reported zero_grad=200/200
        # for a run whose weights were moving the whole time — a false
        # zero, which is the one direction this programme has been
        # burned by repeatedly.
        if ro[0].to_i >= 1; contributed = contributed + 1; end
      elsif ARM == ARM_DFA_RX
        # hidden from B1.e, readout from the exact signal. The Jacobian
        # is back HERE AND ONLY HERE, and the arm is labelled for it.
        tk_dfa!(w, gr, g_x, g_y, g_ang, g_step, b1m, b2m, N_IN, HIDDEN,
                OFF_W1, OFF_B1, OFF_W2, OFF_B2, ACT, rs_a, rs_h, rs_yout,
                rs_obs, e3, d1v, 1, SUM_TO_BEST ? 1 : 0, ro[0].to_i,
                E_MODE, E_DIM, e_mean, E_DECAY, E_NORM)
        tk_bptt!(jplant, w, gr, g_step, g_x, g_y, g_ang,
                 N_IN, HIDDEN, OFF_W1, OFF_B1, OFF_W2,
                 OFF_B2, ACT, rs_x, rs_y, rs_oc, rs_os, rs_clamp, rs_sig,
                 rs_o, rs_yout, rs_a, rs_h, rs_obs, gs, gsn, dobs, 1)
        # dfa_rx's HIDDEN half is a broadcast and always contributes;
        # only its readout half can vanish. So the arm contributed
        # whenever the episode ran, same as dfa_tb.
        if ro[0].to_i >= 1; contributed = contributed + 1; end
      else
        c = tk_bptt!(jplant, w, gr, g_step, g_x, g_y, g_ang,
                     N_IN, HIDDEN, OFF_W1, OFF_B1, OFF_W2,
                     OFF_B2, ACT, rs_x, rs_y, rs_oc, rs_os, rs_clamp, rs_sig,
                     rs_o, rs_yout, rs_a, rs_h, rs_obs, gs, gsn, dobs,
                     READOUT_ONLY)
        contributed = contributed + c
      end
      ks = ks + 1
    end
    # LANDMINE 4: an episode whose best approach is step 0 contributed
    # nothing. Counted, never absorbed.
    if contributed == 0
      zero_grad = zero_grad + 1
    end

    # Descent. `frozen` leaves W1/b1 at init by never accumulating into
    # them (tk_bptt! with readout_only), so no masking is needed here —
    # the gradient it never computed cannot leak through the optimizer.
    sc2 = LR / N_START.to_f
    # Global-norm clip. Applied to the SUMMED gradient before the
    # per-start average, so the clip threshold means the same thing at
    # every N_START.
    if CLIP > 0.0
      gn = 0.0
      cz = 0
      while cz < N_PARAM
        gn = gn + gr[cz] * gr[cz]
        cz = cz + 1
      end
      gn = Math.sqrt(gn)
      if gn > CLIP
        cf = CLIP / gn
        cz2 = 0
        while cz2 < N_PARAM
          gr[cz2] = gr[cz2] * cf
          cz2 = cz2 + 1
        end
      end
    end
    uz = 0
    while uz < N_PARAM
      w[uz] = w[uz] - sc2 * gr[uz]
      uz = uz + 1
    end

    final_loss = loss_sum / N_START.to_f
    updates = updates + 1
    # `loss` is what was DESCENDED; `score_d2` is the paper's nearest
    # approach. Under TRUCK_LOSS=best they are the same number, and
    # printing both anyway is what keeps them from being conflated.
    puts "step " + updates.to_s + ": loss=" + final_loss.to_s +
         " score_d2=" + (score_sum / N_START.to_f).to_s
    if BUDGET > 0 && plant_steps >= BUDGET
      upd = STEPS
    end
    upd = upd + 1
  end
end

# ======================================================================
# EVALUATION — every set separately, never pooled.
# ======================================================================
#
# `far` is the paper's generalisation region and `near` is the close-up
# field its own text says the gradient-free method did NOT own. They
# are reported apart because the pre-registered hypothesis is about
# WHICH SIDE a randomly projected error lands on; one pooled number
# cannot answer it.
#
# d^2 < 5 is the paper's own single-point criterion (~6 deg angular,
# sqrt(5) m linear), so `dock5` is its success rate, not a threshold of
# ours.

ev = tk_zeros(8)
# Sized for the largest eval set: the yard/near sets draw TRUCK_EVAL_N,
# the ensemble 15.
ev_paths = tk_zeros(EVAL_N > 15 ? EVAL_N : 15)

def tk_eval(evp_plant, evp_w, evp_set, evp_n, evp_cap, evp_nin, evp_hid,
            evp_ow1, evp_ob1, evp_ow2, evp_ob2, evp_kind,
            evp_x, evp_y, evp_oc, evp_os, evp_clamp, evp_sig, evp_o, evp_yout,
            evp_a, evp_h, evp_obs, evp_fw, evp_ro, evp_seed, evp_paths,
            evp_mode, evp_lesson, evp_out)
  # evp_set: 0 ensemble (15 fixed), 1 point, 2 far yard, 3 near field,
  #          4 TRAIN — the run's OWN start scheme (toy#195).
  #
  # Set 4 exists because the regime search asks "at what lesson does
  # this arm stop docking", and none of the four fixed sets IS a lesson.
  # Scoring a lesson-0 policy on the far yard answers a generalisation
  # question instead of the regime question. With TRUCK_START=ensemble
  # it duplicates set 0, which is the honest degenerate case.
  evp_plant.seed!(evp_seed)
  evp_xmin = evp_plant.tt_x_min
  evp_xmax = evp_plant.tt_x_max
  evp_ymin = evp_plant.tt_y_min
  evp_ymax = evp_plant.tt_y_max
  if evp_set == 3
    # NEAR is OURS, not the paper's: the paper says "close-up" and
    # gives no box. Stated here so it is a choice on the record.
    evp_plant.tt_x_min = 10.0
    evp_plant.tt_x_max = 50.0
    evp_plant.tt_y_min = -25.0
    evp_plant.tt_y_max = 25.0
  end
  evp_count = evp_set == 0 ? 15 : (evp_set == 1 ? 1 : evp_n)
  if evp_set == 4
    evp_count = evp_mode == 0 ? 15 : (evp_mode == 1 ? 1 : evp_n)
  end
  evp_sum = 0.0
  evp_min = 1.0e18
  evp_dock = 0
  # toy#192: mean_d2 and dock5 cannot tell two different routes into the
  # same absorbing state apart — bang-bang steering with the wrong sign,
  # and an under-actuated passive jack-knife, both end ~5 m from where
  # they started with the clamp touched in nearly every rollout. These
  # four separate them, and they are on the eval LINE so a reader of
  # stdout.log sees them without opening the traces.
  evp_sig_sum = 0.0
  evp_sig_n = 0
  evp_sat_n = 0
  evp_clamped_runs = 0
  evp_i = 0
  while evp_i < evp_count
    if evp_set == 4
      tk_set_start!(evp_plant, evp_mode, evp_i, evp_lesson)
    elsif evp_set == 0
      evp_plant.ensemble_start!(evp_i)
    elsif evp_set == 1
      evp_plant.point_start!
    else
      evp_plant.sample_yard!(64)
    end
    evp_sx0 = evp_plant.tt_x
    evp_sy0 = evp_plant.tt_y
    tk_rollout(evp_plant, evp_w, evp_cap, evp_nin, evp_hid,
               evp_ow1, evp_ob1, evp_ow2, evp_ob2, evp_kind,
               evp_x, evp_y, evp_oc, evp_os, evp_clamp, evp_sig, evp_o,
               evp_yout, evp_a, evp_h, evp_obs, evp_fw, evp_ro)
    evp_d2 = evp_ro[1]
    evp_sum = evp_sum + evp_d2
    if evp_d2 < evp_min; evp_min = evp_d2; end
    if evp_d2 < 5.0; evp_dock = evp_dock + 1; end

    evp_T = evp_ro[0].to_i
    evp_st = 0
    while evp_st < evp_T
      evp_as = evp_sig[evp_st]
      if evp_as < 0.0; evp_as = -evp_as; end
      evp_sig_sum = evp_sig_sum + evp_as
      if evp_as > 0.99; evp_sat_n = evp_sat_n + 1; end
      evp_sig_n = evp_sig_n + 1
      evp_st = evp_st + 1
    end
    if evp_plant.tt_clamp_count > 0
      evp_clamped_runs = evp_clamped_runs + 1
    end
    evp_paths[evp_i] = tk_path_len(evp_plant, evp_x, evp_y, evp_T,
                                   evp_sx0, evp_sy0)
    evp_i = evp_i + 1
  end
  evp_plant.tt_x_min = evp_xmin
  evp_plant.tt_x_max = evp_xmax
  evp_plant.tt_y_min = evp_ymin
  evp_plant.tt_y_max = evp_ymax
  evp_out[0] = evp_sum / evp_count.to_f
  evp_out[1] = evp_min
  evp_out[2] = evp_dock.to_f / evp_count.to_f
  evp_out[3] = evp_count.to_f
  evp_out[4] = evp_sig_n > 0 ? evp_sig_sum / evp_sig_n.to_f : 0.0
  evp_out[5] = evp_sig_n > 0 ? evp_sat_n.to_f / evp_sig_n.to_f : 0.0
  evp_out[6] = evp_clamped_runs.to_f / evp_count.to_f
  evp_out[7] = tk_median(evp_paths, evp_count)
  nil
end

EV_NAMES = ["ensemble", "point", "far", "near", "train"]
ev_mean = tk_zeros(5)
ev_dock = tk_zeros(5)
es = 0
while es < 5
  tk_eval(plant, w, es, EVAL_N, STEP_CAP, N_IN, HIDDEN, OFF_W1, OFF_B1,
          OFF_W2, OFF_B2, ACT, rs_x, rs_y, rs_oc, rs_os, rs_clamp, rs_sig,
          rs_o, rs_yout, rs_a, rs_h, rs_obs, fw_out, ro, EVAL_SEED,
          ev_paths, START_MODE, LESSON, ev)
  ev_mean[es] = ev[0]
  ev_dock[es] = ev[2]
  puts "eval: set=" + EV_NAMES[es] + " n=" + ev[3].to_i.to_s +
       " mean_d2=" + ev[0].to_s + " best_d2=" + ev[1].to_s +
       " dock5=" + ev[2].to_s +
       " mean_abs_signal=" + ev[4].to_s + " frac_sat=" + ev[5].to_s +
       " frac_clamped_runs=" + ev[6].to_s + " median_path_len=" + ev[7].to_s
  es = es + 1
end

# ======================================================================
# toy#190 — the headless rollout, written as a `tbu-traces/1` bundle.
# ======================================================================
#
# The frontend overlays what THE ENGINE drew, so it never runs a forward
# pass and there is no second controller to keep in parity. That
# argument applies inside this file too, which is why the rollout below
# calls the SAME tk_rollout the training loop calls rather than
# re-deriving a trajectory.
#
# Per-run summaries are carried AND recomputable from the trace, so a
# reader can cross-check this engine's arithmetic against its own. That
# is also why the stride never drops the last row or the best-approach
# row: a bundle whose own summary cannot be recovered from its own trace
# would be self-inconsistent in exactly the direction nobody checks.

if TRACE.length > 0
  tr_scheme = TRACE_SCH_S.length > 0 ? TRACE_SCH_S : tk_scheme_name(START_MODE)
  tr_seed = TRACE_SD_S.length > 0 ? TRACE_SD_S.to_i : EVAL_SEED
  tr_mode = tk_scheme_mode(tr_scheme)
  if tr_mode < 0
    puts "toy-train-truck: TRUCK_TRACE_SCHEME must be " +
         "ensemble|point|yard|lesson|half_yard|half_yard_neg, got " + tr_scheme
    exit 1
  end
  tr_n = tr_mode == 0 ? 15 : (tr_mode == 1 ? 1 : TRACE_N)
  tr_stride = STRIDE < 1 ? 1 : STRIDE

  tf = File.open(TRACE, "w")
  tf.write("{\"format\":\"tbu-traces/1\",\"provenance\":")
  pb = Toy::Json::Builder.new
  pb.add_str("arm",    ARM_NAMES[ARM])
  pb.add_str("engine", "toy")
  tr_gp = SpinelKit::Git.read
  pb.add_str("engine_git", tr_gp.sha)
  pb.add_str("weights", LOAD_PATH.length > 0 ? LOAD_PATH :
                        (EXPORT.length > 0 ? EXPORT : "in-process"))
  pb.add_num("train_seed", SEED)
  nb = Toy::Json::Builder.new
  nb.add_raw("shape", "[" + N_IN.to_s + "," + HIDDEN.to_s + ",1]")
  nb.add_str("activation", ACT == ACT_TANH ? "tanh" : "logistic")
  nb.add_str("output_map", ACT == ACT_TANH ? "identity" : "2s-1")
  nb.add_num("obs", N_IN)
  pb.add_obj("net", nb)
  pb.add_str("objective", "d2_best_approach")
  # Not decoration: a bundle from a run that only LOADED weights and one
  # from a run that trained are otherwise indistinguishable.
  pb.add_num("updates", updates)
  pb.add_num("stride", tr_stride)
  tf.write(pb.dump)

  tf.write(",\"plant\":")
  qb = Toy::Json::Builder.new
  qb.add_raw("ls", plant.tt_ls.to_s)
  qb.add_raw("lc", plant.tt_lc.to_s)
  qb.add_raw("u_max_deg", (plant.tt_umax * 180.0 / Math::PI).to_s)
  qb.add_raw("r", plant.tt_r.to_s)
  qb.add_num("step_cap", STEP_CAP)
  qb.add_str("dock_ref", plant.tt_dock == TruckTask::DOCK_MIDRIG ? "midrig" : "trailer")
  qb.add_str("wrap", plant.tt_wrap == TruckTask::WRAP_NONE ? "none" : "pi")
  tf.write(qb.dump)

  tf.write(",\"columns\":[\"signal\",\"u\",\"x\",\"y\",\"tc\",\"ts\",\"clamped\"]")
  tf.write(",\"runs\":[")

  # The yard stream is seeded ONCE and drawn in sequence, so run `idx` is
  # the draw index and the whole set is reproducible from the seed alone.
  plant.seed!(tr_seed)
  tr_i = 0
  while tr_i < tr_n
    tk_set_start!(plant, tr_mode, tr_i, LESSON)
    tr_sx = plant.tt_x
    tr_sy = plant.tt_y
    tr_sc = plant.tt_oc
    tr_ss = plant.tt_os

    tk_rollout(plant, w, STEP_CAP, N_IN, HIDDEN, OFF_W1, OFF_B1, OFF_W2,
               OFF_B2, ACT, rs_x, rs_y, rs_oc, rs_os, rs_clamp, rs_sig,
               rs_o, rs_yout, rs_a, rs_h, rs_obs, fw_out, ro)
    tr_T = ro[0].to_i
    tr_best_step = ro[2].to_i

    # How the episode ENDED, distinguished rather than lumped into
    # "stopped": a policy that drives through the dock wall and one that
    # runs out of steps are different failures, and the overlay colours
    # them differently.
    tr_end = "cap"
    if plant.at_dock?
      tr_end = "docked"
    elsif tr_T < STEP_CAP
      tr_end = "bound"
      if !plant.above_dock?(plant.tt_x, plant.tt_y) ||
         !plant.above_dock?(plant.coupling_x, plant.coupling_y) ||
         !plant.above_dock?(plant.cab_front_x, plant.cab_front_y)
        tr_end = "wall"
      end
    end

    tr_plen = tk_path_len(plant, rs_x, rs_y, tr_T, tr_sx, tr_sy)

    # The run object is emitted FIELD BY FIELD rather than built with
    # Json::Builder and then reopened. The first version did
    # `rb.dump.sub("}", ",\"trace\":[")`, which splices at the FIRST
    # closing brace — the nested `start` object's, not the run's — and
    # produced a bundle that parsed as far as run 0 before dying.
    if tr_i > 0; tf.write(","); end
    sb = Toy::Json::Builder.new
    sb.add_raw("x",  tr_sx.to_s)
    sb.add_raw("y",  tr_sy.to_s)
    sb.add_raw("ts", tr_ss.to_s)
    sb.add_raw("tc", tr_sc.to_s)
    sb.add_str("scheme", tr_scheme)
    sb.add_num("idx", tr_i)
    sb.add_num("seed", tr_seed)
    tf.write("{\"id\":" + tr_i.to_s)
    tf.write(",\"start\":" + sb.dump)
    tf.write(",\"end\":\"" + tr_end + "\"")
    tf.write(",\"steps\":" + tr_T.to_s)
    tf.write(",\"best_d2\":" + ro[1].to_s)
    tf.write(",\"best_step\":" + tr_best_step.to_s)
    tf.write(",\"terminal_d2\":" + ro[6].to_s)
    tf.write(",\"clamp_count\":" + plant.tt_clamp_count.to_s)
    tf.write(",\"path_len\":" + tr_plen.to_s)
    tf.write(",\"trace\":[")

    tr_wrote = 0
    tr_t = 0
    while tr_t < tr_T
      # ALWAYS the last row and the row that produced the best approach,
      # whatever the stride.
      tr_keep = (tr_t % tr_stride) == 0 || tr_t == tr_T - 1 ||
                (tr_best_step >= 1 && tr_t == tr_best_step - 1)
      if tr_keep
        tr_ax = tr_t + 1 < tr_T ? rs_x[tr_t + 1] : plant.tt_x
        tr_ay = tr_t + 1 < tr_T ? rs_y[tr_t + 1] : plant.tt_y
        tr_ac = tr_t + 1 < tr_T ? rs_oc[tr_t + 1] : plant.tt_oc
        tr_as = tr_t + 1 < tr_T ? rs_os[tr_t + 1] : plant.tt_os
        if tr_wrote > 0; tf.write(","); end
        tf.write("[" + rs_sig[tr_t].to_s + "," +
                 plant.steer_angle(rs_sig[tr_t]).to_s + "," +
                 tr_ax.to_s + "," + tr_ay.to_s + "," +
                 tr_ac.to_s + "," + tr_as.to_s + "," +
                 rs_clamp[tr_t].to_s + "]")
        tr_wrote = tr_wrote + 1
      end
      tr_t = tr_t + 1
    end
    tf.write("]}")
    tr_i = tr_i + 1
  end
  tf.write("]}")
  tf.close
  puts "trace: " + TRACE + " runs=" + tr_n.to_s + " scheme=" + tr_scheme +
       " seed=" + tr_seed.to_s + " stride=" + tr_stride.to_s
end

# ---- provenance ----

prov = "truck: arm=" + ARM_NAMES[ARM]
prov = prov + " obs=" + N_IN.to_s
prov = prov + " hidden=" + HIDDEN.to_s
prov = prov + " params=" + N_PARAM.to_s
prov = prov + " act=" + (ACT == ACT_TANH ? "tanh" : "sigmoid")
prov = prov + " outmap=" + (ACT == ACT_TANH ? "identity" : "2sigma-1")
prov = prov + " r=" + PLANT_R.to_s
prov = prov + " cap=" + STEP_CAP.to_s
prov = prov + " start=" + tk_scheme_name(START_MODE)
if START_MODE == ST_LESSON
  prov = prov + " lesson=" + LESSON.to_s
end
prov = prov + " seed=" + SEED.to_s
prov = prov + " lr=" + LR.to_s
prov = prov + " clip=" + CLIP.to_s
prov = prov + " b_dist=" + (B_DIST_S.length > 0 ? B_DIST_S : "gaussian")
prov = prov + " b_scale=" + (B_SCALE_S.length > 0 ? B_SCALE_S : "inv_sqrt_fan")
prov = prov + " b_seed=" + B_SEED.to_s
prov = prov + " dfa_sum=" + (SUM_TO_BEST ? "to_best" : "episode")
prov = prov + " e=" + (E_MODE == E_YT ? "yt" :
                       E_MODE == E_CENTERED ? "centered" :
                       E_MODE == E_SIGNED_X ? "signed_x" : "xyt")
prov = prov + " e_dim=" + E_DIM.to_s
prov = prov + " e_norm=" + (E_NORM == 1 ? "unit" : "none")
prov = prov + " objective=nearest_approach"
prov = prov + " loss=" + (LOSS_TERMINAL ? "terminal" : "best")
prov = prov + " fitness=reconstructed"
if ARM == ARM_GA
  prov = prov + " ga_pop=" + GA_POP.to_s
  prov = prov + " ga_gens=" + GA_GENS.to_s
  prov = prov + " ga_agg=" + (GA_AGG_MIN == 1 ? "min" : "mean")
  prov = prov + " ga_mut=" + GA_MUT.to_s
  prov = prov + " ga_elite=" + GA_ELITE.to_s
end
prov = prov + " budget=" + BUDGET.to_s
prov = prov + " plant_steps=" + plant_steps.to_s
prov = prov + " updates=" + updates.to_s
prov = prov + " zero_grad=" + zero_grad.to_s
if IS_IMIT
  prov = prov + " demos=" + (DEMOS_SCARCE ? "scarce" : "abundant")
  prov = prov + " demo_pairs=" + dm_n.to_s
  prov = prov + " demo_batch=" + DEMO_BATCH.to_s
  prov = prov + " wd=" + WD.to_s
  prov = prov + " imit_mse=" + imit_mse.to_s
  # The teacher's own number, beside the student's: a student cannot be
  # better than its teacher's data and the reader needs both.
  prov = prov + " expert=" + EXPERT
  prov = prov + " expert_mean_d2=" + expert_d2.to_s
end
prov = prov + " eval_seed=" + EVAL_SEED.to_s
prov = prov + " " + plant.provenance
puts prov

# ---- export in the FRONTEND's format ----
#
# [layer][unit][w..., bias] with the BIAS LAST, which is what
# AdalineUnit.forward expects. The sidecar names the activation and the
# output->steering map because the frontend's shipped nets are TANH and
# ours are LOGISTIC: replayed under the wrong function the trajectory
# is plausible and wrong.
if EXPORT.length > 0
  js = "[["
  xj = 0
  while xj < HIDDEN
    if xj > 0; js = js + ","; end
    js = js + "["
    xk = 0
    while xk < N_IN
      if xk > 0; js = js + ","; end
      js = js + w[OFF_W1 + xj * N_IN + xk].to_s
      xk = xk + 1
    end
    js = js + "," + w[OFF_B1 + xj].to_s + "]"
    xj = xj + 1
  end
  js = js + "],[["
  xj2 = 0
  while xj2 < HIDDEN
    if xj2 > 0; js = js + ","; end
    js = js + w[OFF_W2 + xj2].to_s
    xj2 = xj2 + 1
  end
  js = js + "," + w[OFF_B2].to_s + "]]]"
  fw = File.open(EXPORT, "w")
  fw.write(js)
  fw.close

  mb = Toy::Json::Builder.new
  mb.add_str("schema",     "toy-truck-controller/v1")
  mb.add_str("arm",        ARM_NAMES[ARM])
  mb.add_str("activation", ACT == ACT_TANH ? "tanh" : "logistic")
  mb.add_str("output_map", ACT == ACT_TANH ? "signal = out" : "signal = 2*out - 1")
  mb.add_str("steering",   "u = signal * 70deg, saturating at |signal| = 1")
  mb.add_str("layer_order", "[layer][unit][w..., bias] with bias LAST")
  mb.add_num("obs",        N_IN)
  mb.add_str("obs_slots",  N_IN == 3 ? "x,y,Os" : (N_IN == 8 ?
             "x,y,Oc,Os,10x,10y,10Oc,10Os" : "x,y,Oc,Os"))
  mb.add_str("obs_norm",   "x=(x-50)/50 y=y/50 angle=angle/pi")
  mb.add_num("hidden",     HIDDEN)
  mb.add_num("params",     N_PARAM)
  mb.add_num("seed",       SEED)
  mb.add_raw("r",          PLANT_R.to_s)
  mb.add_num("step_cap",   STEP_CAP)
  mb.add_str("objective",  "nearest_approach_d2")
  mb.add_str("fitness",    "reconstructed")
  mb.add_num("plant_steps", plant_steps)
  mb.add_num("updates",    updates)
  mb.add_raw("eval_ensemble_mean_d2", ev_mean[0].to_s)
  mb.add_raw("eval_far_dock5",  ev_dock[2].to_s)
  mb.add_raw("eval_near_dock5", ev_dock[3].to_s)
  fm = File.open(EXPORT + ".meta.json", "w")
  fm.write(mb.dump)
  fm.close
  puts "export: " + EXPORT + " (+ .meta.json)"
end

# ---- events ----

if EVENTS.length > 0
  rc = TinyNN.tnn_events_open(EVENTS)
  if rc == 0
    rs = Toy::Json::Builder.new
    rs.add_str("kind",   "run_start")
    rs.add_str("schema", "toy/v1")
    rs.add_num("t",      TinyNN.tnn_events_now_seconds)
    rs.add_str("started_at", TinyNN.tnn_events_iso8601_now)
    rs.add_str("run_id", RUN_ID)
    rs.add_str("phase",  "train")
    Toy::Events.add_provenance(rs, TinyNN.tnn_provenance_host_name,
                               TinyNN.tnn_provenance_host_os,
                               TinyNN.tnn_provenance_host_arch,
                               # Literal, not tnn_backend_name(sess): this
                               # lane opens no session (no graph at all), and
                               # it is CPU-only by design, not by default.
                               "cpu")
    cf = Toy::Json::Builder.new
    cf.add_str("lane",   "truck")
    cf.add_str("arm",    ARM_NAMES[ARM])
    cf.add_num("obs",    N_IN)
    cf.add_num("hidden", HIDDEN)
    cf.add_num("params", N_PARAM)
    cf.add_str("act",    ACT == ACT_TANH ? "tanh" : "sigmoid")
    cf.add_raw("r",      PLANT_R.to_s)
    cf.add_num("step_cap", STEP_CAP)
    cf.add_num("seed",   SEED)
    cf.add_raw("lr",     LR.to_s)
    cf.add_raw("clip",   CLIP.to_s)
    cf.add_num("b_seed", B_SEED)
    cf.add_str("b_dist", B_DIST_S.length > 0 ? B_DIST_S : "gaussian")
    cf.add_str("b_scale", B_SCALE_S.length > 0 ? B_SCALE_S : "inv_sqrt_fan")
    cf.add_str("e_mode", E_MODE_S.length > 0 ? E_MODE_S : "xyt")
    cf.add_num("e_dim", E_DIM)
    cf.add_str("e_norm", E_NORM == 1 ? "unit" : "none")
    cf.add_str("objective", "nearest_approach_d2")
    cf.add_str("loss", LOSS_TERMINAL ? "terminal" : "best")
    cf.add_str("fitness", "reconstructed")
    cf.add_num("plant_steps", plant_steps)
    cf.add_num("updates", updates)
    cf.add_num("zero_grad", zero_grad)
    rs.add_obj("config", cf)
    TinyNN.tnn_events_emit(rs.dump)

    ei = 0
    while ei < 5
      eb = Toy::Json::Builder.new
      eb.add_str("kind",  "eval")
      eb.add_str("phase", "eval")
      eb.add_num("t",     TinyNN.tnn_events_now_seconds)
      eb.add_str("name",  EV_NAMES[ei])
      eb.add_raw("mean_d2", ev_mean[ei].to_s)
      eb.add_raw("dock5",   ev_dock[ei].to_s)
      TinyNN.tnn_events_emit(eb.dump)
      ei = ei + 1
    end

    re = Toy::Json::Builder.new
    re.add_str("kind", "run_end")
    re.add_num("t",          TinyNN.tnn_events_now_seconds)
    re.add_str("ended_at",   TinyNN.tnn_events_iso8601_now)
    re.add_str("reason",     "completed")
    re.add_num("final_step", updates)
    re.add_raw("final_loss", final_loss.to_s)
    re.add_str("checkpoint", EXPORT.length > 0 ? EXPORT : "none")
    re.add_raw("exit_code",  "0")
    TinyNN.tnn_events_emit(re.dump)
    TinyNN.tnn_events_close
  else
    puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
  end
end
