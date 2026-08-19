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
# DEVICE SCOPE (tao#24). This runner is the CPU lane and serves all three
# tasks. `--task bytelm` ALSO has a CUDA twin — libexec/toy-train-gtx-cuda,
# mechanically mirrored from this file by prep/gen_cuda_mirror.rb — because
# toy#170/P3 changed that task's workload by orders of magnitude (vocab
# 4096, ctx 128, ~3.2 TFLOP/cell at ~32 GFLOP/s with the GB10 at 0%). The
# relational/local tasks stay CPU-only: the rationale there did NOT lapse.
# Cross-device cells are NOT numerically comparable — a sweep runs entirely
# on one device or it re-runs the reference.
# Own compilation unit (landmine #16).
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
#   GTX_TASK        — relational (default) | local | bytelm
#   GTX_TEXT        — byte-LM pack prefix (bytelm only)
#   GTX_VOCAB       — NOMINAL head width under bytelm (default 256). Sets
#                     the lm_head, the byte embedding AND the DFA
#                     feedback matrix B together — B is what the
#                     output-dim law is a claim about, and it was pinned
#                     at 256 on every byte-LM cell before toy#170/P5.
#                     Requires a DENSELY coded pack (prep/remap_alphabet.rb);
#                     refused loudly against the pack's max token id.
#   GTX_PAIRS       — labelled pairs per step (default 128)
#   GTX_VAL_BATCHES — held-out INSTANCES evaluated at the end (default 8).
#                     The graph TOPOLOGY is fixed but its CONTENT is
#                     redrawn every step, so held-out means fresh
#                     instances, not held-out pairs over one graph — see
#                     toy_gtx_task.rb on the memorisation shortcut a
#                     fixed graph leaves open.
#   GTX_TASK_SEED   — task seed, SEPARATE from SEED (default 7)
#   GTX_LR / GTX_WARMUP
#
#   toy#172 / E1 THE B-CONDITIONING INSTRUMENT + nDFA:
#   GTX_INSTRUMENT / GTX_INSTRUMENT_N — Phase 1.1, opt-in, byte-null off
#   GTX_NDFA        — 0 (default) | 1. Phase 1.2's error-side
#                     preconditioner, FOLDED INTO B host-side.
#                     `--task bytelm` + a `dfa` policy only.
#   GTX_NDFA_LAMBDA — the ridge. REQUIRED when GTX_NDFA=1; no default.
#   GTX_NDFA_EVERY  — steps between refreshes (default 500)
#   GTX_NDFA_M      — error samples per refresh (default 256)
#   GTX_NDFA_GAIN   — preserve (default) | raw. `preserve` renormalises
#                     B' back to ||B||_F so only the DIRECTION
#                     reweighting is under test.
#
#   toy#172 / E2 ADAPTIVE LOW-RANK FEEDBACK (LDFA):
#   GTX_DFA_RANK    — full (default) | <r>. Factorises B as Q[dout x r] .
#                     P[r x V]. `full` is BYTE-IDENTICAL to the pre-E2
#                     binary. `--task bytelm` + a `dfa` policy only.
#   GTX_DFA_ADAPT   — none (default) | oja. Whether P tracks the top-r
#                     subspace of the ERROR stream by Oja's rule. Requires
#                     GTX_DFA_RANK.
#   GTX_LDFA_ETA    — Oja step size (default 0.05). 0 is legal and is the
#                     FROZEN-OJA control; it relabels itself in provenance
#                     so an eta=0 cell can never be filed as adaptive.
#   GTX_LDFA_EVERY  — steps between refreshes (default 500)
#   GTX_LDFA_M      — error samples per refresh (default 128)
#
#   toy#161 RETROFIT MODE (GTX_RETROFIT=1):
#   GTX_PRETRAIN_STEPS — BP steps on the PRETRAIN task (default 1500)
#   GTX_PRETRAIN_LR    — BP's own cell for phase 1 (default 0.003)
#   GTX_ADAPTER_POLICY — chain | dfa | frozen  (the arm under test)
#   GTX_ADAPTER_LAYERS — stacked pair-site adapters (default 2)
#   GTX_ADAPTER_RANK   — bottleneck width (default 16)
#   GTX_FREEZE_BACKBONE— 1 (default) freezes AND DETACHES the backbone
#   STEPS / GTX_LR then apply to the RETROFIT phase.
#   GTX_CKPT_EVERY     — write the BACKBONE as GGUF every K steps and at
#                        the final step, into TAO_RUN_DIR (toy#164)
#   GTX_LOAD_CKPT      — load a backbone and SKIP the pretrain phase;
#                        requires GTX_RETROFIT=1
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
require_relative "../io/toy_ae_task"
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
# toy#170 (P3) — the AR byte-LM task on a CAUSAL attention body.
TEXT        = ENV["GTX_TEXT"] || ""
CONTEXT     = (ENV["GTX_CONTEXT"] || "128").to_i
# toy#170 (P5) — the NOMINAL head width, and it is a different axis from
# the corpus alphabet.
#
# P4 and P5's remap both vary how many symbols the text ACTUALLY uses
# while this stays 256. But the output-dim law is a claim about the
# FEEDBACK MATRIX: B is [d_model, vocab], its inv_sqrt_fan scale is
# 1/sqrt(vocab), and both were pinned at 256 on every byte-LM cell ever
# run on this lane. So an alphabet sweep at a fixed head measures the
# effective rank of the error, not the width of B — the two are the same
# number only when the head is sized to the corpus, which is what this
# knob makes possible.
#
# Defaults to 256, which is what every existing byte-LM config already
# gets, so no run changes meaning (see the durable rule: a knob that
# reinterprets existing configs defaults to the OLD behaviour).
VOCAB       = (ENV["GTX_VOCAB"] || "256").to_i
# toy#172 (E1) — the B-conditioning instrument. OPT-IN and byte-null when
# off: the Gram is O(V n^2), which at n=4096 is ~137s and would cost more
# than the cell it instruments (a CUDA byte-LM cell is ~18s). So a
# measurement pass asks for it; ordinary sweep cells never pay it.
#
# GTX_INSTRUMENT_N is the sample count, and it is the axis that matters:
# rank(C_E) <= n, so a rank measured at its own ceiling is not a
# measurement. Matched n ACROSS RUNGS keeps the ceiling constant instead
# of letting it move with V.
INSTRUMENT   = (ENV["GTX_INSTRUMENT"] || "0").to_i
INSTRUMENT_N = (ENV["GTX_INSTRUMENT_N"] || "1024").to_i
# toy#172 (E1 Phase 1.2) — the nDFA ERROR-SIDE PRECONDITIONER.
#
# Left-multiplies the broadcast error by lambda(C_E + lambda I)^-1, which
# is FOLDED INTO B host-side (see GtxEngine#precondition_b! for the
# Woodbury identity and why nothing [V x V] is ever formed). OFF by
# default and byte-null when off: every P3-P6 cell was measured without
# it and stays comparable only if 0 really is the old binary.
#
# THE CADENCE IS A FLAG, NOT A CONSTANT. A per-step refresh would cost
# more than the step it preconditions (the Gram alone is O(V m^2)), so
# the refresh runs every GTX_NDFA_EVERY steps and the cadence rides in
# the provenance line — a preconditioner whose staleness is invisible is
# a different experiment at every step budget.
#
# THE SAMPLES COME FROM THE TRAINING STEPS THEMSELVES, deliberately. The
# obvious implementation — run extra forward passes at the refresh point
# — would consume the corpus RNG stream and update the Adam moments
# through opt_step_adamw (lr=0 does NOT freeze m and v), so the
# lambda -> inf arm would diverge from the unpreconditioned one for
# reasons having nothing to do with B, and the byte gate below could
# never pass. Reading the logits the step already computed costs a
# download and changes no number.
NDFA         = (ENV["GTX_NDFA"] || "0").to_i
NDFA_LAM_S   = ENV["GTX_NDFA_LAMBDA"] || ""
NDFA_EVERY   = (ENV["GTX_NDFA_EVERY"] || "500").to_i
NDFA_M       = (ENV["GTX_NDFA_M"] || "256").to_i
NDFA_GAIN_S  = ENV["GTX_NDFA_GAIN"] || ""
NDFA_LAM     = NDFA_LAM_S.length > 0 ? NDFA_LAM_S.to_f : -1.0
# toy#172 (E2) — LDFA, the ADAPTIVE LOW-RANK feedback path.
#
# B is factorised as Q[dout x r] . P[r x V] and folded into the SAME
# uploaded B tensor, for the same reason nDFA was: the error reaches every
# tap through exactly one product, B . e, so a factorisation is a change
# to what refresh_b! uploads and to nothing else. No new graph node, no
# new readout path — the standing prior on the lane where the ||g||
# instrument realized as ZEROS through three separate readout paths.
#
# THE EMPTY-STRING TRAP is why these are parsed rather than `.to_i`d:
# (ENV["X"] || "d").to_i is 0 for "", so an unset knob and an explicit
# zero would be the same value and `full` would silently become rank 0.
# The string is inspected first and `full`/unset map to 0 deliberately.
LDFA_RANK_S  = ENV["GTX_DFA_RANK"]  || ""
LDFA_ADAPT_S = ENV["GTX_DFA_ADAPT"] || ""
LDFA_ETA_S   = ENV["GTX_LDFA_ETA"]  || ""
LDFA_EVERY   = (ENV["GTX_LDFA_EVERY"] || "500").to_i
LDFA_M       = (ENV["GTX_LDFA_M"] || "128").to_i
LDFA_RANK    = (LDFA_RANK_S.length == 0 || LDFA_RANK_S == "full") ?
                 0 : LDFA_RANK_S.to_i
LDFA_ADAPT   = LDFA_ADAPT_S == "oja" ? 1 : 0
LDFA_ETA     = LDFA_ETA_S.length > 0 ? LDFA_ETA_S.to_f : 0.05
# NOTE: no ||g|| instrument on this lane. The ssm-lane version (toy#169)
# works; the gtx port built the node and realized it as ZEROS, and a
# metric that silently reads 0.0 is worse than no metric — it is the
# exact failure class this arc has been chasing. Deferred rather than
# shipped broken; P3's expected failure mode is not instability anyway
# (attention has no recurrence), so it is not on the critical path.
N_PAIRS     = (ENV["GTX_PAIRS"]     || "128").to_i
VAL_BATCHES = (ENV["GTX_VAL_BATCHES"] || "8").to_i
TASK_SEED   = (ENV["GTX_TASK_SEED"] || "7").to_i
GTX_TASK_SEED_BL = TASK_SEED + 1
LR_S        = ENV["GTX_LR"] || ""
WARMUP_S    = ENV["GTX_WARMUP"] || ""
B_SEED      = (ENV["GTX_B_SEED"] || "1234").to_i
B_DIST_S    = ENV["GTX_B_DIST"]  || ""
B_SCALE_S   = ENV["GTX_B_SCALE"] || ""
RETROFIT    = (ENV["GTX_RETROFIT"] || "") == "1"
PRE_STEPS   = (ENV["GTX_PRETRAIN_STEPS"] || "1500").to_i
PRE_LR_S    = ENV["GTX_PRETRAIN_LR"] || ""
AD_POLICY_S = ENV["GTX_ADAPTER_POLICY"] || ""
AD_LAYERS   = (ENV["GTX_ADAPTER_LAYERS"] || "2").to_i
AD_RANK     = (ENV["GTX_ADAPTER_RANK"] || "16").to_i
FREEZE_BB   = (ENV["GTX_FREEZE_BACKBONE"] || "1") == "1"
CKPT_EVERY  = (ENV["GTX_CKPT_EVERY"] || "0").to_i
LOAD_CKPT   = ENV["GTX_LOAD_CKPT"] || ""

NOISE  = NOISE_S.length  > 0 ? NOISE_S.to_f  : 0.3
LR     = LR_S.length     > 0 ? LR_S.to_f     : 0.003
PRE_LR = PRE_LR_S.length > 0 ? PRE_LR_S.to_f : 0.003
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
if CKPT_EVERY > 0 && TAO_RUN_DIR.length == 0
  puts "toy-train-gtx: GTX_CKPT_EVERY needs TAO_RUN_DIR — there is nowhere" +
       " to write the checkpoint, and writing nowhere is not a silent no-op"
  exit 1
end
if LOAD_CKPT.length > 0 && !RETROFIT
  puts "toy-train-gtx: GTX_LOAD_CKPT requires GTX_RETROFIT=1 — a loaded" +
       " backbone on this lane exists to be retrofitted, and inventing a" +
       " second meaning for it would make the flag ambiguous"
  exit 1
end
if LOAD_CKPT.length > 0 && !File.exist?(LOAD_CKPT)
  puts "toy-train-gtx: no such checkpoint: " + LOAD_CKPT
  exit 1
end
if RETROFIT && LOAD_CKPT.length == 0 && PRE_STEPS < 1
  puts "toy-train-gtx: GTX_PRETRAIN_STEPS must be >= 1 in retrofit mode" +
       " (a retrofit of an UNTRAINED backbone measures nothing), got " + PRE_STEPS.to_s
  exit 1
end
if RETROFIT && AD_LAYERS < 1
  puts "toy-train-gtx: GTX_ADAPTER_LAYERS must be >= 1, got " + AD_LAYERS.to_s
  exit 1
end
if RETROFIT && AD_RANK < 1
  puts "toy-train-gtx: GTX_ADAPTER_RANK must be >= 1, got " + AD_RANK.to_s
  exit 1
end
if AD_POLICY_S.length > 0 && AD_POLICY_S != "chain" && AD_POLICY_S != "dfa" && AD_POLICY_S != "frozen"
  puts "toy-train-gtx: GTX_ADAPTER_POLICY " + AD_POLICY_S + " unsupported (chain|dfa|frozen)"
  exit 1
end
if AD_POLICY_S.length > 0 && !RETROFIT
  puts "toy-train-gtx: GTX_ADAPTER_POLICY is meaningless without GTX_RETROFIT=1" +
       " — there are no adapters outside a retrofit, so the token would silently do nothing"
  exit 1
end
if CUT_S.length > 0 && CUT_S != "layer" && CUT_S != "step"
  puts "toy-train-gtx: GTX_DFA_CUT " + CUT_S + " unsupported (layer|step)"
  exit 1
end
if TASK_S == "bytelm" && TEXT.length == 0
  puts "toy-train-gtx: GTX_TASK bytelm needs GTX_TEXT <pack-prefix> —" +
       " autoregressive next-byte over REAL text, no synthetic fallback" +
       " (prep/fetch_text.rb)"
  exit 1
end
if TASK_S.length > 0 && TASK_S != "relational" && TASK_S != "local" && TASK_S != "bytelm"
  puts "toy-train-gtx: GTX_TASK " + TASK_S + " unsupported (relational|local|bytelm)"
  exit 1
end

# ---- toy#172 (E1 Phase 1.2): the nDFA knobs, validated BEFORE the graph
# is realized. ggml aborts during graph build (exit 134 with a
# GGML_ASSERT line), so anything checked after recipe.realize! is dead
# code that never gets to speak.
if NDFA != 0 && NDFA != 1
  puts "toy-train-gtx: GTX_NDFA must be 0 or 1, got " + NDFA.to_s
  exit 1
end
if NDFA == 1 && TASK_S != "bytelm"
  puts "toy-train-gtx: GTX_NDFA=1 needs GTX_TASK bytelm. The nDFA" +
       " preconditioner inverts the ERROR COVARIANCE C_E, and the whole" +
       " question (toy#172/E1) is about a WIDE error vector: the" +
       " relational task's error is 16-wide with the head 16 classes, so" +
       " there is no anisotropy of the kind Phase 1.1 measured and the" +
       " sample count would exceed the width on the first step. Run it on" +
       " --task bytelm, where the head is up to 4096."
  exit 1
end
if NDFA == 1 && RETROFIT
  puts "toy-train-gtx: GTX_NDFA=1 is not supported with GTX_RETROFIT=1 —" +
       " a retrofit rebuilds the graph and its B mid-run, so a" +
       " preconditioner refreshed on the pretrain phase's error would be" +
       " silently carried into a different head. Refused rather than" +
       " given an unstated meaning."
  exit 1
end
if NDFA == 1 && NDFA_LAM <= 0.0
  puts "toy-train-gtx: GTX_NDFA=1 requires GTX_NDFA_LAMBDA > 0 and there" +
       " is NO defensible default. P = lambda(C_E + lambda I)^-1 is the" +
       " IDENTITY as lambda -> inf and a PROJECTOR as lambda -> 0, so the" +
       " ridge is not a tuning detail, it IS the experiment. Pick it" +
       " against the lambda_max the instrument reports (GTX_INSTRUMENT=1)."
  exit 1
end
if NDFA == 1 && NDFA_M < 1
  puts "toy-train-gtx: GTX_NDFA_M must be >= 1, got " + NDFA_M.to_s
  exit 1
end
if NDFA == 1 && NDFA_EVERY < 1
  puts "toy-train-gtx: GTX_NDFA_EVERY must be >= 1, got " + NDFA_EVERY.to_s
  exit 1
end
if NDFA == 1 && NDFA_GAIN_S.length > 0 && NDFA_GAIN_S != "preserve" && NDFA_GAIN_S != "raw"
  puts "toy-train-gtx: GTX_NDFA_GAIN " + NDFA_GAIN_S + " unsupported (preserve|raw)"
  exit 1
end
NDFA_PRESERVE = NDFA_GAIN_S == "raw" ? 0 : 1

# ---- toy#172 (E2): the LDFA knobs, validated BEFORE the graph is
# realized, for the same reason nDFA's are — ggml aborts during graph
# build (exit 134 + a GGML_ASSERT line) and anything checked after
# recipe.realize! is dead code that never gets to speak.
if LDFA_RANK_S.length > 0 && LDFA_RANK_S != "full" && LDFA_RANK < 1
  puts "toy-train-gtx: GTX_DFA_RANK " + LDFA_RANK_S + " unsupported —" +
       " it is `full` (the default, byte-identical to the pre-E2 binary)" +
       " or a POSITIVE integer rank. Note that the empty string is NOT" +
       " rank 0: it is `full`."
  exit 1
end
if LDFA_ADAPT_S.length > 0 && LDFA_ADAPT_S != "none" && LDFA_ADAPT_S != "oja"
  puts "toy-train-gtx: GTX_DFA_ADAPT " + LDFA_ADAPT_S + " unsupported (none|oja)"
  exit 1
end
if LDFA_ADAPT_S.length > 0 && LDFA_RANK == 0
  puts "toy-train-gtx: GTX_DFA_ADAPT is meaningless at GTX_DFA_RANK=full" +
       " — the adaptation acts on P, and there is no P without a" +
       " factorisation. The knob would be parsed, reported and never" +
       " read, which is an inert flag that still labels the cell."
  exit 1
end
if LDFA_ETA_S.length > 0 && LDFA_ADAPT != 1
  puts "toy-train-gtx: GTX_LDFA_ETA is meaningless without GTX_DFA_ADAPT=oja" +
       " — nothing steps P on the fixed arm."
  exit 1
end
if LDFA_ADAPT == 1 && LDFA_ETA < 0.0
  puts "toy-train-gtx: GTX_LDFA_ETA must be >= 0, got " + LDFA_ETA.to_s +
       " (0 is legal and is the FROZEN-OJA control; it relabels itself" +
       " adapt=oja-frozen in provenance)"
  exit 1
end
if LDFA_RANK > 0 && TASK_S != "bytelm"
  puts "toy-train-gtx: GTX_DFA_RANK needs GTX_TASK bytelm. LDFA compresses" +
       " the ERROR to r dimensions, and the whole question (toy#172/E2) is" +
       " about a WIDE error vector: the relational task's head is 16" +
       " classes, where r < V is not a compression worth the name. Run it" +
       " on --task bytelm, where the head is up to 4096."
  exit 1
end
if LDFA_RANK > 0 && RETROFIT
  puts "toy-train-gtx: GTX_DFA_RANK is not supported with GTX_RETROFIT=1 —" +
       " a retrofit rebuilds the graph and its B mid-run, so a P adapted to" +
       " the pretrain phase's error would be silently carried into a" +
       " different head. Refused rather than given an unstated meaning."
  exit 1
end
if LDFA_RANK > 0 && NDFA == 1
  puts "toy-train-gtx: GTX_DFA_RANK and GTX_NDFA=1 both fold into the SAME" +
       " uploaded B and are opposite interventions (nDFA WHITENS the" +
       " dominant error directions; LDFA COMPRESSES TO them). Composing" +
       " them silently would produce a cell that is neither arm. Refused."
  exit 1
end
if LDFA_RANK > 0 && LDFA_M < 1
  puts "toy-train-gtx: GTX_LDFA_M must be >= 1, got " + LDFA_M.to_s
  exit 1
end
if LDFA_RANK > 0 && LDFA_EVERY < 1
  puts "toy-train-gtx: GTX_LDFA_EVERY must be >= 1, got " + LDFA_EVERY.to_s
  exit 1
end

TASK_KIND = TASK_S == "local" ? GtxTask::KIND_LOCAL : GtxTask::KIND_RELATIONAL
DFA_CUT   = CUT_S == "step" ? Toy::LLM::Engine::GtxEngine::CUT_STEP :
                              Toy::LLM::Engine::GtxEngine::CUT_LAYER
IS_BYTELM = TASK_S == "bytelm"
N_CLASSES = IS_BYTELM ? VOCAB : (N_TYPES * N_TYPES)
# The retrofit label is the MODULAR SUM, so it has TY classes, not TY*TY.
# That is not a detail: a same-cardinality relabeling is a BIJECTION and a
# retrained linear head absorbs it, which would leave the frozen control
# unable to lose. See toy_gtx_task.rb.
RETRO_CLASSES = N_TYPES
AD_POLICY = AD_POLICY_S == "dfa" ? Toy::LLM::Engine::GtxEngine::POLICY_DFA :
            (AD_POLICY_S == "frozen" ? Toy::LLM::Engine::GtxEngine::POLICY_FROZEN :
                                       Toy::LLM::Engine::GtxEngine::POLICY_CHAIN)
N_ADAPTERS = RETROFIT ? AD_LAYERS : 0
# degree == TY is not a tunable: one attribute of EACH type per
# neighbourhood is what makes mean-pooling provably uninformative, which
# is what lets the frozen control lose. See toy_gtx_task.rb.
DEGREE    = N_TYPES

# tao#24 — THE GPU TWIN'S SCOPE. One flag, flipped by the mirror.
#
# GPU_TWIN is 0 in the CPU source and 1 in every generated backend mirror
# (the SKIP/STUB sentinel below is the only thing the generator changes).
# So `libexec/toy-train-gtx` keeps all three tasks and
# `libexec/toy-train-gtx-cuda` refuses everything but `bytelm`.
#
# Why the flag and not a stubbed-out `if` block: the generator emits STUB
# lines VERBATIM — they are collected before the substitution table runs,
# so a message written inside a stub would keep whatever binary name it
# was typed with and the metal mirror would announce itself as `-cuda`.
# Keeping the refusal as ORDINARY source puts it back under the same
# per-backend rename as every other message in this file.
#
# Why a refusal and not a silent fallback: tao#24 reopened the CPU-only
# decision for the byte-LM lane ONLY. The relational task is d_model 64 /
# 16 classes / 1500 steps — a GPU buys nothing there, and a twin that
# quietly accepted it would invite cells comparable to no P1-P6 number
# ever measured. It must fail loud.
#
# The guard sits here, BEFORE recipe.realize!, on purpose: ggml aborts
# during graph build (exit 134 + a GGML_ASSERT line), so a validation
# placed after realize! is dead code that never gets to speak.
# CUDA-MIRROR-SKIP-BEGIN: the bytelm-only scope is a GPU-twin property (tao#24)
# CUDA-MIRROR-STUB: GPU_TWIN = 1
GPU_TWIN = 0
# CUDA-MIRROR-SKIP-END
if GPU_TWIN == 1 && !IS_BYTELM
  puts "toy-train-gtx: GTX_TASK " +
       (TASK_S.length > 0 ? TASK_S : "relational") +
       " has no GPU twin (tao#24 reopened the CPU-only scope for `bytelm`" +
       " ONLY: that task is vocab up to 4096 / ctx 128 / ~3.2 TFLOP per" +
       " cell, the relational one is d_model 64 / 16 classes and a GPU" +
       " buys it nothing). Run it on libexec/toy-train-gtx."
  exit 1
end

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

def count_dfa_blocks(policy)
  n = 0
  i = 0
  while i < policy.length
    if policy[i] == 1
      n = n + 1
    end
    i = i + 1
  end
  n
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

# One window of CONTEXT bytes; position i predicts byte i+1, so the
# window reads CONTEXT + 1 bytes.
def bl_fill!(bl_task, tokens, m_labels, start, ctx, classes)
  k = 0
  while k < ctx * classes
    m_labels.flat[k] = 0.0
    k = k + 1
  end
  t = 0
  while t < ctx
    tokens[t] = bl_task.at_tokens[start + t]
    m_labels.flat[t * classes + bl_task.at_tokens[start + t + 1]] = 1.0
    t = t + 1
  end
  nil
end

# EXACT held-out bits/byte from the logits. An AR LM scores itself — no
# judge model, no sampler.
# ---- toy#172 / E1: the B-CONDITIONING INSTRUMENT ----
#
# P6 stated its verdict on the DFA arm's excess-over-noise, never on any
# measured rank. This is a NEW claim: it measures the error covariance
# C_E = E[e e'] with e = softmax(logits) - y, whose conditioning is what
# "B is poorly conditioned under a wide error vector" would be about.
#
# WHY IT COSTS NOTHING EXTRA TO OBSERVE: e is fully reconstructible from
# the logits and labels the eval loop ALREADY downloads for bpb. No new
# graph node and no new readout path — which is the right prior on this
# lane, where the ||g|| instrument realized as ZEROS through three
# separate readout paths and had to be removed.
#
# THE SAMPLING CEILING, which is the whole design constraint:
#   rank(C_E) <= n_samples, NOT <= V.
# One eval batch gives N=context error vectors; V is up to 4096. So a
# naive read would show "effective rank collapses with width" purely
# because the sample count stopped keeping up — which is exactly the E1
# `rank` verdict, manufactured from a sampling artifact. n is therefore
# emitted beside every rank, and a rank at its own ceiling is flagged
# rather than reported as a measurement.
#
# THE CHEAP IDENTITY: the nonzero spectrum of C_E [V x V] equals that of
# the Gram G = E' E [n x n], so nothing V x V is ever formed.
#   tr C  = ||E||_F^2                      (O(V n) — free)
#   tr C2 = ||G||_F^2                      (needs G: O(V n^2))
#   participation ratio = (tr C)^2 / tr C2
#   stable rank         = tr C / lambda_max(G)
# Only tr C2 and lambda_max need G, which is why n is a knob: V*n^2 is
# ~4.3e9 MACs at n=1024 and ~6.9e10 at n=4096.
def bc_fill_err!(ebuf, slot, buf, m_labels, n_pos, classes, want)
  # Append this batch's per-position error vectors to ebuf, stopping at
  # `want` samples. Returns the new slot count.
  s = slot
  i = 0
  while i < n_pos && s < want
    base = i * classes
    mx = buf[base]
    c = 1
    while c < classes
      if buf[base + c] > mx
        mx = buf[base + c]
      end
      c = c + 1
    end
    tot = 0.0
    c2 = 0
    while c2 < classes
      tot = tot + Math.exp(buf[base + c2] - mx)
      c2 = c2 + 1
    end
    dst = s * classes
    c3 = 0
    while c3 < classes
      ebuf[dst + c3] = (Math.exp(buf[base + c3] - mx) / tot) - m_labels.flat[base + c3]
      c3 = c3 + 1
    end
    s = s + 1
    i = i + 1
  end
  s
end

def bc_trace(ebuf, n, classes)
  t = 0.0
  k = 0
  lim = n * classes
  while k < lim
    t = t + ebuf[k] * ebuf[k]
    k = k + 1
  end
  t
end

def bc_gram!(g, ebuf, n, classes)
  # G[i][j] = e_i . e_j, symmetric, so only the upper triangle is
  # computed and mirrored.
  i = 0
  while i < n
    j = i
    while j < n
      s = 0.0
      ai = i * classes
      aj = j * classes
      c = 0
      while c < classes
        s = s + ebuf[ai + c] * ebuf[aj + c]
        c = c + 1
      end
      g[i * n + j] = s
      g[j * n + i] = s
      j = j + 1
    end
    i = i + 1
  end
  nil
end

def bc_fro2(g, n)
  s = 0.0
  k = 0
  lim = n * n
  while k < lim
    s = s + g[k] * g[k]
    k = k + 1
  end
  s
end

def bc_lambda_max(g, n, iters)
  # Power iteration. G is PSD by construction, so no shift is needed and
  # the dominant eigenvalue is what stable rank divides by.
  v = Array.new(n, 1.0 / Math.sqrt(n.to_f))
  w = Array.new(n, 0.0)
  lam = 0.0
  it = 0
  while it < iters
    i = 0
    while i < n
      s = 0.0
      j = 0
      while j < n
        s = s + g[i * n + j] * v[j]
        j = j + 1
      end
      w[i] = s
      i = i + 1
    end
    nrm = 0.0
    i2 = 0
    while i2 < n
      nrm = nrm + w[i2] * w[i2]
      i2 = i2 + 1
    end
    nrm = Math.sqrt(nrm)
    if nrm <= 0.0
      return 0.0
    end
    lam = nrm
    i3 = 0
    while i3 < n
      v[i3] = w[i3] / nrm
      i3 = i3 + 1
    end
    it = it + 1
  end
  lam
end

def bl_bpb(buf, m_labels, n_pos, classes)
  nll = 0.0
  i = 0
  while i < n_pos
    base = i * classes
    mx = buf[base]
    c = 1
    while c < classes
      if buf[base + c] > mx
        mx = buf[base + c]
      end
      c = c + 1
    end
    tot = 0.0
    c2 = 0
    while c2 < classes
      tot = tot + Math.exp(buf[base + c2] - mx)
      c2 = c2 + 1
    end
    tgt = 0
    c3 = 0
    while c3 < classes
      if m_labels.flat[base + c3] > 0.5
        tgt = c3
      end
      c3 = c3 + 1
    end
    nll = nll - ((buf[base + tgt] - mx) - Math.log(tot))
    i = i + 1
  end
  nll / (n_pos.to_f * Math.log(2.0))
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

# One held-out pass over the materialised instances, at lr = 0.
#
# MIRROR RULE (toy#139/#146): "lr=0" must mean EVERY hp vector the graph
# can reach. This lane has exactly ONE — if a per-block or per-adapter
# vector is ever added it must be zeroed here too, or the val split
# silently becomes training data.
def val_pass!(recipe, val_x, va, vb, vy, x_flat, idx_a, idx_b, labels,
              inc_flat, prev_a, prev_b, m_cur, val_hp, logit_buf,
              n_out, n_batches, n_pairs, n_nodes, d_feat)
  hits = 0
  seen = 0
  loss_sum = 0.0
  vb2 = 0
  while vb2 < n_batches
    vsrc = vb2 * n_nodes * d_feat
    vx = 0
    while vx < n_nodes * d_feat
      x_flat[vx] = val_x[vsrc + vx]
      vx = vx + 1
    end
    recipe.upload_features!(x_flat)
    vi = 0
    while vi < n_pairs
      idx_a[vi]  = va[vb2 * n_pairs + vi]
      idx_b[vi]  = vb[vb2 * n_pairs + vi]
      labels[vi] = vy[vb2 * n_pairs + vi]
      vi = vi + 1
    end
    set_incidence!(inc_flat, prev_a, prev_b, idx_a, idx_b, n_pairs, n_nodes, false)
    k2 = 0
    while k2 < n_pairs * n_out
      m_cur.flat[k2] = 0.0
      k2 = k2 + 1
    end
    b2 = 0
    while b2 < n_pairs
      m_cur.flat[b2 * n_out + labels[b2]] = 1.0
      b2 = b2 + 1
    end
    vloss = recipe.step!(idx_a, idx_b, inc_flat, m_cur, val_hp, false)
    loss_sum = loss_sum + vloss
    rc_v = TinyNN.tnn_download_to_f64_array(recipe.gr_cache.sess,
             recipe.gr_cache.t_logits, logit_buf, n_pairs * n_out)
    if rc_v != 0
      puts "toy-train-gtx: val logits download failed: rc=" + rc_v.to_s
      exit 1
    end
    hits = hits + hits_in(logit_buf, labels, n_pairs, n_out)
    seen = seen + n_pairs
    vb2 = vb2 + 1
  end
  # Two FLOATS, and nothing else: an array whose elements Spinel cannot
  # type uniformly comes back as `unknown` and the first .to_i on it
  # fails at runtime. The held-out count is n_batches * n_pairs by
  # construction, so it does not need to ride back in here.
  out = [0.0]
  out[0] = hits.to_f / seen.to_f
  out.push(loss_sum / n_batches.to_f)
  out
end

POLICY = parse_gtx_policy(POLICY_S, N_BLOCKS)

task = GtxTask.new(TASK_KIND, D_FEAT, N_ENTITIES, N_TYPES, DEGREE,
                   TASK_SEED, NOISE)
N_NODES = IS_BYTELM ? CONTEXT : task.gt_nodes

# toy#172 (E1 Phase 1.2) — the two nDFA checks that need the parsed
# policy and the node count.
#
# nDFA folds into B, and B exists only where a `dfa` block wired a tap.
# On a chain- or frozen-only policy the flag would therefore be a SILENT
# no-op that still reports itself as on — which is precisely how an inert
# knob gets published as a clean negative.
NDFA_DFA_BLOCKS = count_dfa_blocks(POLICY)
if NDFA == 1 && NDFA_DFA_BLOCKS == 0
  puts "toy-train-gtx: GTX_NDFA=1 with no `dfa` block in GTX_POLICY —" +
       " nDFA is folded INTO the feedback matrix B, and B is built only" +
       " where a dfa block taps. On this policy there is nothing to" +
       " precondition, so the flag would run, report itself on, and" +
       " change nothing."
  exit 1
end
# Each refresh needs GTX_NDFA_M error vectors and one step yields
# GTX_CONTEXT of them, so the collection window is ceil(m / context)
# steps. A cadence shorter than its own window could never fill the
# buffer and would refresh on a short sample without saying so.
NDFA_NEED = NDFA == 1 ? ((NDFA_M + N_NODES - 1) / N_NODES) : 0
if NDFA == 1 && NDFA_EVERY < NDFA_NEED
  puts "toy-train-gtx: GTX_NDFA_EVERY " + NDFA_EVERY.to_s +
       " is shorter than the collection window it needs (" +
       NDFA_NEED.to_s + " steps to gather GTX_NDFA_M=" + NDFA_M.to_s +
       " error vectors at GTX_CONTEXT=" + N_NODES.to_s +
       " per step). Raise the cadence or lower GTX_NDFA_M."
  exit 1
end

# toy#172 (E2) — the same three checks for LDFA, and for the same
# reasons. B_eff exists only where a `dfa` block wired a tap; the
# collection window must fit inside the cadence; and a run whose budget
# never reaches the first refresh is the FIXED arm wearing an adaptive
# label, which is the failure mode this program has paid for most.
#
# The r-vs-V check lives HERE and not up with the other LDFA parsing,
# and that is not tidiness: N_CLASSES is assigned further down the file,
# and a top-level constant read before its assignment is 0 rather than an
# error — so the check ran against "the error width 0" and refused every
# rank. Same family as the empty-env-string trap, one scope out.
if LDFA_RANK > 0 && LDFA_RANK >= N_CLASSES
  puts "toy-train-gtx: GTX_DFA_RANK " + LDFA_RANK.to_s + " is not below the" +
       " error width " + N_CLASSES.to_s + ". At r >= V the factorisation" +
       " spans the whole error space and the arm is `full` wearing a" +
       " low-rank label — which is exactly how an inert knob gets" +
       " published as a clean negative."
  exit 1
end
if LDFA_RANK > 0 && NDFA_DFA_BLOCKS == 0
  puts "toy-train-gtx: GTX_DFA_RANK with no `dfa` block in GTX_POLICY —" +
       " LDFA factorises the feedback matrix B, and B is built only where" +
       " a dfa block taps. On this policy there is nothing to factorise," +
       " so the flag would run, report itself on, and change nothing."
  exit 1
end
LDFA_NEED = LDFA_RANK > 0 ? ((LDFA_M + N_NODES - 1) / N_NODES) : 0
if LDFA_RANK > 0 && LDFA_EVERY < LDFA_NEED
  puts "toy-train-gtx: GTX_LDFA_EVERY " + LDFA_EVERY.to_s +
       " is shorter than the collection window it needs (" +
       LDFA_NEED.to_s + " steps to gather GTX_LDFA_M=" + LDFA_M.to_s +
       " error vectors at GTX_CONTEXT=" + N_NODES.to_s +
       " per step). Raise the cadence or lower GTX_LDFA_M."
  exit 1
end
if LDFA_RANK > 0 && STEPS < LDFA_EVERY
  puts "toy-train-gtx: GTX_LDFA_EVERY " + LDFA_EVERY.to_s + " is never" +
       " reached in STEPS=" + STEPS.to_s + ", so P would never be adapted" +
       " and never measured. Under GTX_DFA_ADAPT=oja that cell IS the" +
       " fixed low-rank arm wearing an adaptive label. Lower the cadence" +
       " or raise STEPS."
  exit 1
end

# The corpus is loaded BEFORE the graph is realized, and that ordering is
# load-bearing rather than tidy. Realizing the graph sizes the lm_head
# and the label tensor from N_CLASSES; if the pack's ids do not fit, ggml
# aborts inside cross_entropy_loss during the build and the process dies
# at exit 134 with a backtrace, which is not a usable refusal and would
# preempt any check placed after. Checking first is the only way the
# runner gets to say what is actually wrong.
bl_task = AeTask.new(CONTEXT + 1, GTX_TASK_SEED_BL)
if IS_BYTELM
  # toy#170 (P6) — widen the pack contract only when the head is actually
  # wider. `max(256, N_CLASSES)`, never `N_CLASSES` alone: under a NARROW
  # head the 256 byte contract must stay in force, so that a raw byte pack
  # run at `--vocab 65` is refused by the GTX_VOCAB guard below — which
  # names the flag, the max id and the remap script — rather than by
  # load_pack!'s generic range message. Same reason the arc's older cells
  # stay comparable: at the default this is exactly 256.
  bl_task.at_max_vocab = N_CLASSES > AeTask::VOCAB ? N_CLASSES : AeTask::VOCAB
  if bl_task.load_pack!(TEXT) != 0
    exit 1
  end
  # toy#170 (P5) — a token id at or above the head width has NO ROW in
  # the one-hot label or the lm_head, so it cannot be represented at all.
  #
  # Gate on the MAX ID, not the alphabet: shakespeare has 65 symbols
  # carried on byte ids up to 122, so `--vocab 65` on the raw pack is a
  # size error, not a narrower head. Use prep/remap_alphabet.rb's dense
  # packs for that.
  #
  # What this does NOT refuse, deliberately: ids that fit but leave GAPS
  # below the head width. Those are dead classes — never a label, but the
  # softmax still puts mass on them and the feedback matrix still routes
  # through them. That configuration is the default byte-LM setup (a
  # 65-symbol corpus under a 256-wide head) and it is what the head-width
  # sweep varies on purpose, so refusing it would refuse the experiment.
  # It is a real limit on what a wide head measures, and it belongs in
  # the write-up rather than in an exit code.
  if bl_task.at_max_id >= N_CLASSES
    puts "toy-train-gtx: GTX_VOCAB " + N_CLASSES.to_s + " is too small for " +
         TEXT + ": max token id " + bl_task.at_max_id.to_s +
         " (alphabet " + bl_task.at_alphabet.to_s + ")." +
         " The pack must be DENSELY coded 0..vocab-1 — see prep/remap_alphabet.rb"
    exit 1
  end
end

recipe = Toy::LLM::Recipes::GtxGraph.new
# toy#172 (E2) — set BEFORE realize!, because refresh_b! runs at the end
# of the graph build and is where B_eff is first assembled. Rank 0 is
# what the engine already initialises to, so an unconfigured run does not
# touch a single branch of the new path.
recipe.gr_cache.gx_b_rank  = LDFA_RANK
recipe.gr_cache.gx_b_adapt = LDFA_ADAPT
# P's seed is SEPARATE from every tap's B seed: P is shared across taps
# (the compression is a property of the error stream, not of the tap), so
# reusing a tap's seed would make P a copy of that tap's Q rows.
recipe.gr_cache.gx_b_pseed = B_SEED + 991
recipe.realize!(D_FEAT, D_MODEL, N_HEADS, D_FF, N_BLOCKS, N_NODES,
                N_PAIRS, N_CLASSES, SEED, 1.0, POLICY, DFA_CUT, B_SEED,
                dist_code(B_DIST_S), scale_code(B_SCALE_S),
                scale_sigma(B_SCALE_S), RETRO_CLASSES, N_ADAPTERS, AD_RANK,
                IS_BYTELM ? 1 : 0, 0)
ToyDescribeFlow.emit_flow_json(TAO_RUN_DIR, recipe.gr_cache.sess)

# toy#164 — a loaded backbone REPLACES the pretrain phase. The weights
# are persistent, so this overwrites the random init in place and the
# retrofit graph is then built over exactly the checkpointed function.
if LOAD_CKPT.length > 0
  rc_ck = recipe.gr_cache.load_backbone_ckpt(LOAD_CKPT)
  if rc_ck != 0
    exit 1
  end
  puts "loaded: " + LOAD_CKPT + " bb_sig=" + recipe.gr_cache.backbone_sig.to_s
end

# ---- the TOPOLOGY is constant; the CONTENT is not. ----
# The adjacency mask is uploaded once. Node features are redrawn every
# step, because a fixed instance is memorisable and a memorised instance
# does not need the retrieval this lane exists to measure.
x_flat = Array.new(N_NODES * D_FEAT, 0.0)
mask_flat = Array.new(N_NODES * N_NODES, 0.0)

# toy#170 — the byte corpus, and a CAUSAL mask. Causality is a DATA
# change here, not a graph change: gtx's mask is a persistent [N, N]
# additive input applied pre-softmax, so the relational and causal arms
# share one realized graph. -30 rather than -inf for the same reason the
# adjacency uses it (a fully-masked softmax row would go NaN).
BL_SPLIT = 0
BL_TRAIN_HI = 0
BL_VAL_HI = 0
if IS_BYTELM
  # The pack is already loaded and its ids already gated, above the
  # realize! call — see the comment there.
  ntok = bl_task.at_n_tokens
  BL_SPLIT = ntok - ntok / 10
  BL_TRAIN_HI = BL_SPLIT - CONTEXT - 1
  BL_VAL_HI = ntok - CONTEXT - 1
  if BL_TRAIN_HI < 1 || BL_VAL_HI < BL_SPLIT
    puts "toy-train-gtx: corpus too short for GTX_CONTEXT " + CONTEXT.to_s
    exit 1
  end
  qi = 0
  while qi < CONTEXT
    kj = 0
    while kj < CONTEXT
      mask_flat[qi * CONTEXT + kj] = kj <= qi ? 0.0 : GtxTask::MASK_NEG
      kj = kj + 1
    end
    qi = qi + 1
  end
  puts "corpus: pack=" + TEXT + " n_tokens=" + ntok.to_s +
       " alphabet=" + bl_task.at_alphabet.to_s +
       " context=" + CONTEXT.to_s + " split_at=" + BL_SPLIT.to_s
else
  task.fill_mask!(mask_flat)
end
recipe.upload_mask!(mask_flat)

# The pair stream. Val pairs are materialised FIRST and training
# continues from where val stopped (toy#152's discipline): two
# differently-seeded streams are two offsets into the same LCG cycle and
# a val stream can land inside the span training later walks.
idx_a  = Array.new(N_PAIRS, 0)
idx_b  = Array.new(N_PAIRS, 0)
labels = Array.new(N_PAIRS, 0)
LAB_ROWS  = IS_BYTELM ? N_NODES : N_PAIRS
m_labels  = Mat.new(LAB_ROWS, N_CLASSES)
bl_tokens = Array.new(N_NODES, 0)
bl_starts = Array.new(VAL_BATCHES, 0)
m_rlabels = Mat.new(N_PAIRS, RETRO_CLASSES)
logit_buf = Array.new(LAB_ROWS * N_CLASSES, 0.0)

# The held-out set is MATERIALISED FIRST — whole instances, features
# included — and training then continues from where val stopped. Same
# discipline as toy#152: two differently-seeded streams are two offsets
# into one LCG cycle, so a val stream can land inside the span training
# later walks.
task.reset_stream!(TASK_SEED + 1)
if IS_BYTELM
  bl_task.reset_stream!(TASK_SEED + 1)
  bs = 0
  while bs < VAL_BATCHES
    bl_starts[bs] = bl_task.next_start_in(BL_SPLIT, BL_VAL_HI)
    bs = bs + 1
  end
end
VAL_N  = VAL_BATCHES * N_PAIRS
val_x  = Array.new(VAL_BATCHES * N_NODES * D_FEAT, 0.0)
val_a  = Array.new(VAL_N, 0)
val_b  = Array.new(VAL_N, 0)
val_y  = Array.new(VAL_N, 0)
val_r  = Array.new(VAL_N, 0)
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
  task.fill_pairs!(N_PAIRS, idx_a, idx_b, labels, GtxTask::LABEL_PAIR)
  vp = 0
  while vp < N_PAIRS
    val_a[vfill * N_PAIRS + vp] = idx_a[vp]
    val_b[vfill * N_PAIRS + vp] = idx_b[vp]
    val_y[vfill * N_PAIRS + vp] = labels[vp]
    vp = vp + 1
  end
  # The SAME held-out pairs under the retrofit label, so the two phases
  # are scored on one set of instances rather than two.
  task.relabel!(N_PAIRS, idx_a, idx_b, labels, GtxTask::LABEL_MODSUM)
  vp2 = 0
  while vp2 < N_PAIRS
    val_r[vfill * N_PAIRS + vp2] = labels[vp2]
    vp2 = vp2 + 1
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

val_hp = Mat.new(1, 7)
val_hp.flat[0] = 0.0
val_hp.flat[1] = adamw.beta1
val_hp.flat[2] = adamw.beta2
val_hp.flat[3] = adamw.eps
val_hp.flat[4] = 0.0
val_hp.flat[5] = adamw.beta1
val_hp.flat[6] = adamw.beta2
pre_acc = -1.0

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
#
# In RETROFIT mode this loop runs BOTH phases: PRE_STEPS of BP on the
# pretrain task, then a graph REBUILD (frozen+detached backbone, adapter
# stack, fresh head), then STEPS on the retrofit task. One process, so
# every arm's backbone is BIT-IDENTICAL by construction rather than by a
# checkpoint round-trip that would have to be trusted.
# A loaded checkpoint skips phase 1 outright — redoing it is the cost
# this flag exists to remove.
PRE_STEPS_EFF = (RETROFIT && LOAD_CKPT.length == 0) ? PRE_STEPS : 0
TOTAL_STEPS = RETROFIT ? PRE_STEPS_EFF + STEPS : STEPS
bb_sig_pre  = 0.0
final_loss = 0.0
# toy#172 (E1 Phase 1.2) — the nDFA ring. Allocated at size 0 when the
# flag is off, so the off path allocates nothing and does nothing.
nd_buf = Array.new(NDFA == 1 ? NDFA_M * N_CLASSES : 0, 0.0)
nd_n   = 0
nd_refreshes  = 0
nd_wall_us    = 0
nd_err_pre    = 0.0
nd_err_post   = 0.0
nd_shrink     = 1.0
nd_first_step = 0
nd_m_used     = 0
# toy#172 (E2) — the LDFA ring. Same shape as nDFA's, and allocated at
# size 0 when the flag is off.
ld_buf = Array.new(LDFA_RANK > 0 ? LDFA_M * N_CLASSES : 0, 0.0)
ld_n   = 0
ld_refreshes  = 0
ld_wall_us    = 0
ld_first_step = 0
ld_m_used     = 0
ld_row_min    = 1.0
ld_row_max    = 1.0
ld_off_max    = 0.0
ld_energy     = 0.0
step = 0
while step < TOTAL_STEPS
  step_wall_start = TinyNN.tnn_events_now_seconds
  in_pretrain = RETROFIT && step < PRE_STEPS_EFF
  if RETROFIT && step == PRE_STEPS_EFF
    # THE HANDOVER. Record what the backbone weighs before the retrofit
    # touches it (or does not), then rebuild the graph over the same
    # persistent weights.
    if LOAD_CKPT.length == 0
      pre_r = val_pass!(recipe, val_x, val_a, val_b, val_y, x_flat, idx_a,
                        idx_b, labels, inc_flat, prev_a, prev_b, m_labels,
                        val_hp, logit_buf, N_CLASSES, VAL_BATCHES, N_PAIRS,
                        N_NODES, D_FEAT)
      pre_acc = pre_r[0]
    end
    bb_sig_pre = recipe.gr_cache.backbone_sig
    # A LOADED backbone was never scored here, so say so rather than
    # printing a sentinel: "-1.0" reads like a measurement.
    puts "pretrain: acc=" + (LOAD_CKPT.length > 0 ? "loaded" : pre_acc.to_s) +
         " bb_sig=" + bb_sig_pre.to_s
    recipe.rebuild_retrofit!(POLICY, DFA_CUT, B_SEED, dist_code(B_DIST_S),
                             scale_code(B_SCALE_S), scale_sigma(B_SCALE_S),
                             AD_POLICY, FREEZE_BB ? 1 : 0)
    adamw2 = Toy::AdamW.for_from_scratch
    adamw = adamw2
  end
  phase_lr = in_pretrain ? PRE_LR : LR
  phase_step = in_pretrain ? step : step - (RETROFIT ? PRE_STEPS : 0)
  if WARMUP > 0 && phase_step < WARMUP
    adamw.lr = phase_lr * ((phase_step + 1).to_f / WARMUP.to_f)
  else
    adamw.lr = phase_lr
  end
  m_hp = adamw.hp(phase_step)

  if IS_BYTELM
    bl_fill!(bl_task, bl_tokens, m_labels,
             bl_task.next_start_in(0, BL_TRAIN_HI), N_NODES, N_CLASSES)
  else
  task.resample!
  task.fill_features!(x_flat)
  recipe.upload_features!(x_flat)
  end
  # The retrofit label applies ONLY in the retrofit phase. `in_pretrain`
  # is false in a plain (non-retrofit) run too, so keying off it alone
  # silently fed mod-sum labels to the 16-class head — caught by toy#160's
  # byte fixture, invisible in every other signal.
  in_retro_phase = RETROFIT && !in_pretrain
  n_out = IS_BYTELM ? N_CLASSES : (in_retro_phase ? RETRO_CLASSES : N_CLASSES)
  m_cur = in_retro_phase ? m_rlabels : m_labels
  if !IS_BYTELM
  task.fill_pairs!(N_PAIRS, idx_a, idx_b, labels,
                   in_retro_phase ? GtxTask::LABEL_MODSUM : GtxTask::LABEL_PAIR)
  set_incidence!(inc_flat, prev_a, prev_b, idx_a, idx_b, N_PAIRS, N_NODES, step == 0)
  k = 0
  while k < N_PAIRS * n_out
    m_cur.flat[k] = 0.0
    k = k + 1
  end
  end
  b = 0
  while b < N_PAIRS && !IS_BYTELM
    m_cur.flat[b * n_out + labels[b]] = 1.0
    b = b + 1
  end

  is_first = step == 0 || (RETROFIT && step == PRE_STEPS_EFF)
  loss = IS_BYTELM ?
           recipe.step_bytelm!(bl_tokens, m_labels, m_hp, is_first) :
           recipe.step!(idx_a, idx_b, inc_flat, m_cur, m_hp, is_first)
  final_loss = loss
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s

  if EVENTS.length > 0
    rc_l = TinyNN.tnn_download_to_f64_array(recipe.gr_cache.sess,
             recipe.gr_cache.t_logits, logit_buf, N_PAIRS * n_out)
    tr_acc = -1.0
    if rc_l != 0
      puts "logits download failed: step=" + (step + 1).to_s + " rc=" + rc_l.to_s
    else
      tr_acc = hits_in(logit_buf, labels, N_PAIRS, n_out).to_f / N_PAIRS.to_f
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
  # ---- toy#172 (E1 Phase 1.2): COLLECT, then REFRESH B'. ----
  #
  # Collection rides the training step's OWN logits — a download, no
  # second forward pass — so it consumes no corpus RNG and steps no
  # optimizer state. That is what lets `lambda -> inf` be byte-identical
  # to the unpreconditioned arm rather than merely close to it.
  #
  # The window is the LAST ceil(m / context) steps of each cadence
  # period, so the covariance describes the error the network is making
  # NOW rather than an average over the whole period, and the download
  # cost is paid on those steps only.
  if NDFA == 1
    nd_pos = step % NDFA_EVERY
    if nd_pos >= NDFA_EVERY - NDFA_NEED
      rc_nd = TinyNN.tnn_download_to_f64_array(recipe.gr_cache.sess,
                recipe.gr_cache.t_logits, logit_buf, N_NODES * N_CLASSES)
      if rc_nd != 0
        puts "toy-train-gtx: nDFA logits download failed: step=" +
             (step + 1).to_s + " rc=" + rc_nd.to_s
        exit 1
      end
      nd_n = bc_fill_err!(nd_buf, nd_n, logit_buf, m_labels, N_NODES,
                          N_CLASSES, NDFA_M)
    end
    if nd_pos == NDFA_EVERY - 1
      nd_t0 = TinyNN.tnn_events_now_seconds
      nd_r = recipe.gr_cache.precondition_b!(nd_buf, nd_n, NDFA_LAM,
                                             NDFA_PRESERVE)
      nd_wall_us = nd_wall_us +
                   ((TinyNN.tnn_events_now_seconds - nd_t0) * 1.0e6).to_i
      if nd_r[0] == 1.0
        puts "toy-train-gtx: nDFA Cholesky failed at step " + (step + 1).to_s +
             " — (lambda*m I + G) went non-positive in f64, which means" +
             " GTX_NDFA_LAMBDA=" + NDFA_LAM.to_s + " is too small for this" +
             " width. Raise it; do not fall back."
        exit 1
      end
      if nd_r[0] == 2.0
        puts "toy-train-gtx: nDFA refresh found no feedback matrices to" +
             " precondition at step " + (step + 1).to_s
        exit 1
      end
      # FINITE OR NOTHING. f32/f64 blowup in the solve would otherwise be
      # completely silent — B' would upload as inf/nan, every DFA update
      # would go to nan, and the loss would simply stop being a number
      # partway through a 4000-step run.
      nd_chk = nd_r[2] - nd_r[2]
      if nd_chk != 0.0 || nd_r[2] < 0.0
        puts "toy-train-gtx: nDFA preconditioned error norm is NOT FINITE" +
             " at step " + (step + 1).to_s + " (||B'E||_F=" + nd_r[2].to_s +
             ", ||BE||_F=" + nd_r[1].to_s + ", lambda=" + NDFA_LAM.to_s +
             ", m=" + nd_n.to_s + ")"
        exit 1
      end
      nd_err_pre  = nd_r[1]
      nd_err_post = nd_r[2]
      nd_shrink   = nd_r[3]
      nd_m_used   = nd_n
      if nd_refreshes == 0
        nd_first_step = step + 1
      end
      nd_refreshes = nd_refreshes + 1
      nd_n = 0
    end
  end
  # ---- toy#172 (E2): COLLECT, then ADAPT P and REBUILD B_eff. ----
  #
  # Structurally identical to the nDFA ring above, and deliberately so:
  # the samples ride the training step's OWN logits (a download, no second
  # forward pass), so nothing consumes corpus RNG or steps an optimizer
  # moment, and the FIXED arm runs exactly the same collection and
  # measurement as the ADAPTIVE one. The only difference between the two
  # arms is whether the Oja update is applied — which is what makes this
  # contrast a clean B-test rather than nDFA's by-construction one.
  if LDFA_RANK > 0
    ld_pos = step % LDFA_EVERY
    if ld_pos >= LDFA_EVERY - LDFA_NEED
      rc_ld = TinyNN.tnn_download_to_f64_array(recipe.gr_cache.sess,
                recipe.gr_cache.t_logits, logit_buf, N_NODES * N_CLASSES)
      if rc_ld != 0
        puts "toy-train-gtx: LDFA logits download failed: step=" +
             (step + 1).to_s + " rc=" + rc_ld.to_s
        exit 1
      end
      ld_n = bc_fill_err!(ld_buf, ld_n, logit_buf, m_labels, N_NODES,
                          N_CLASSES, LDFA_M)
    end
    if ld_pos == LDFA_EVERY - 1
      ld_t0 = TinyNN.tnn_events_now_seconds
      ld_r = recipe.gr_cache.adapt_p!(ld_buf, ld_n, LDFA_ETA, LDFA_ADAPT)
      ld_wall_us = ld_wall_us +
                   ((TinyNN.tnn_events_now_seconds - ld_t0) * 1.0e6).to_i
      if ld_r[0] == 1.0
        puts "toy-train-gtx: LDFA Oja diverged at step " + (step + 1).to_s +
             " — P's rows went non-finite or its basis collapsed at" +
             " GTX_LDFA_ETA=" + LDFA_ETA.to_s + ". Lower eta; do not fall" +
             " back to a renormalised basis, which would look like a" +
             " healthy adaptation."
        exit 1
      end
      if ld_r[0] == 2.0
        puts "toy-train-gtx: LDFA refresh found no factorisation to adapt" +
             " at step " + (step + 1).to_s
        exit 1
      end
      ld_row_min = ld_r[1]
      ld_row_max = ld_r[2]
      ld_off_max = ld_r[3]
      ld_energy  = ld_r[4]
      ld_m_used  = ld_n
      if ld_refreshes == 0
        ld_first_step = step + 1
      end
      ld_refreshes = ld_refreshes + 1
      ld_n = 0
    end
  end
  # toy#164 — the BACKBONE checkpoint. Written only during the pretrain
  # phase (or a plain run): a retrofit's backbone is frozen, so writing
  # it again would just duplicate the file it was loaded from.
  if CKPT_EVERY > 0 && (in_pretrain || !RETROFIT)
    done = step + 1
    if done % CKPT_EVERY == 0 || done == TOTAL_STEPS
      ck_rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
      rc_w = recipe.gr_cache.write_backbone_ckpt(
               TAO_RUN_DIR + "/step_" + done.to_s + ".gguf", ck_rid, done)
      if rc_w != 0
        puts "toy-train-gtx: checkpoint write failed rc=" + rc_w.to_s
        exit 1
      end
      puts "ckpt: step_" + done.to_s + ".gguf bb_sig=" +
           recipe.gr_cache.backbone_sig.to_s
    end
  end
  step = step + 1
end

# ---- held-out val. ----
FINAL_OUT = RETROFIT ? RETRO_CLASSES : N_CLASSES
if IS_BYTELM
  bpb_sum = 0.0
  # toy#172 (E1): the error buffer is filled from the SAME download the
  # bpb uses. Sized to the requested sample count, so an instrument pass
  # simply runs more val batches rather than needing a second eval.
  bc_want = INSTRUMENT == 1 ? INSTRUMENT_N : 0
  bc_buf  = Array.new(bc_want * N_CLASSES, 0.0)
  bc_n    = 0
  bc_batches = 0
  vb = 0
  while vb < VAL_BATCHES || (INSTRUMENT == 1 && bc_n < bc_want)
    bi = vb % VAL_BATCHES
    bl_fill!(bl_task, bl_tokens, m_labels, bl_starts[bi], N_NODES, N_CLASSES)
    recipe.step_bytelm!(bl_tokens, m_labels, val_hp, false)
    rcb = TinyNN.tnn_download_to_f64_array(recipe.gr_cache.sess,
            recipe.gr_cache.t_logits, logit_buf, N_NODES * N_CLASSES)
    if rcb != 0
      puts "toy-train-gtx: bytelm logits download failed: rc=" + rcb.to_s
      exit 1
    end
    # bpb is scored over the FIRST VAL_BATCHES only, so an instrument
    # pass reports exactly the bpb a non-instrument run would: the
    # measurement must not move the number it sits beside.
    if vb < VAL_BATCHES
      bpb_sum = bpb_sum + bl_bpb(logit_buf, m_labels, N_NODES, N_CLASSES)
    end
    if INSTRUMENT == 1 && bc_n < bc_want
      bc_n = bc_fill_err!(bc_buf, bc_n, logit_buf, m_labels, N_NODES, N_CLASSES, bc_want)
      bc_batches = bc_batches + 1
    end
    vb = vb + 1
    if vb > 4096
      puts "toy-train-gtx: instrument could not reach " + bc_want.to_s +
           " samples (corpus too short for the requested GTX_INSTRUMENT_N)"
      exit 1
    end
  end
  puts "bytelm: bpb=" + (bpb_sum / VAL_BATCHES.to_f).to_s +
       " n=" + (VAL_BATCHES * N_NODES).to_s +
       " routing=position_t head=bp vocab=" + N_CLASSES.to_s + " attn=causal"
  # toy#172 (E1 Phase 1.2) — the nDFA provenance. Emitted ONLY when the
  # flag is on, so an off run is byte-identical to the pre-change binary.
  #
  # The CADENCE is on the line because a preconditioner refreshed every
  # 500 steps and one refreshed every 50 are different experiments that
  # would otherwise carry the same label. `m` is the ACTUAL sample count
  # of the last refresh, not the request — the same discipline the
  # Phase 1.1 instrument's `n` follows, and for the same reason.
  #
  # err_pre / err_post are ||B E||_F and ||B' E||_F over the taps: the
  # broadcast error as the taps ACTUALLY receive it, before and after.
  # That is what the finite-norm assertion is stated on, and it is
  # asserted at every refresh, not just printed here.
  #
  # b_shrink is ||B'||_F / ||B||_F BEFORE the gain rule. Under
  # gain=preserve it is the size of what the preconditioner removed and
  # was then given back; under gain=raw it is a live multiplier on every
  # DFA update. Either way it must be visible, because a global gain on
  # B is indistinguishable from an LR change on this lane.
  if NDFA == 1
    # A run whose step budget never reaches the first refresh IS the
    # unpreconditioned arm, and would be filed as an nDFA cell. Fail
    # loud: an inert knob that reports itself on is the most expensive
    # failure mode this program has.
    if nd_refreshes == 0
      puts "toy-train-gtx: GTX_NDFA=1 but no refresh ever ran — STEPS=" +
           STEPS.to_s + " never reaches GTX_NDFA_EVERY=" + NDFA_EVERY.to_s +
           ", so this cell is the UNPRECONDITIONED arm wearing an nDFA" +
           " label. Lower the cadence or raise STEPS."
      exit 1
    end
    puts "ndfa: on=1 lambda=" + NDFA_LAM.to_s +
         " every=" + NDFA_EVERY.to_s +
         " m=" + nd_m_used.to_s + " m_req=" + NDFA_M.to_s +
         " gain=" + (NDFA_PRESERVE == 1 ? "preserve" : "raw") +
         " refreshes=" + nd_refreshes.to_s +
         " first_refresh_step=" + nd_first_step.to_s +
         " err_pre=" + num_or_null(nd_err_pre) +
         " err_post=" + num_or_null(nd_err_post) +
         " b_shrink=" + num_or_null(nd_shrink) +
         # THE CLOCK IS NOT ALWAYS RUNNING. tnn_events_now_seconds
         # returns 0.0 when the event file is not open, so without a
         # TAO_RUN_DIR every interval measures as exactly zero — which is
         # the ||g||-instrument failure mode in miniature: a metric that
         # silently reads 0.0. Say `unmeasured` instead of printing it.
         " refresh_ms=" +
           (EVENTS.length > 0 ? (nd_wall_us / nd_refreshes / 1000).to_s :
                                "unmeasured-no-run-dir")
  end
  # toy#172 (E2) — the LDFA provenance. Emitted ONLY when the flag is on,
  # so `full` is byte-identical to the pre-E2 binary.
  #
  # THE SCALE PAIR IS THE POINT OF THIS LINE. A rank-r Q.P has a different
  # Frobenius norm from the full-width B it replaces, so without the
  # rescale "low rank hurts" would be indistinguishable from "the updates
  # got smaller" and the entire fixed-vs-adaptive contrast would be
  # confounded by a scalar. b_eff_fro and b_full_fro are both printed so a
  # reader can verify the arms differ in RANK and not in SCALE instead of
  # taking the rescale on faith.
  #
  # rank_eff IS NOT rank. rank(Q.P) <= min(dout, r), and dout is d_model
  # here — so on this fixture r=256 buys the SAME matrix rank as full
  # width and only confines the row space. Printed beside dout for exactly
  # that reason.
  #
  # p_energy is the fraction of the error's energy P captures, against
  # p_energy_rand = r/V for a random orthonormal P. It is the CONVERGENCE
  # CHECK on Oja: an adaptation that ran but learned nothing sits at the
  # random baseline, and would otherwise be indistinguishable from one
  # that worked.
  if LDFA_RANK > 0
    if ld_refreshes == 0
      puts "toy-train-gtx: GTX_DFA_RANK set but no refresh ever ran —" +
           " STEPS=" + STEPS.to_s + " never reaches GTX_LDFA_EVERY=" +
           LDFA_EVERY.to_s + ", so P was never adapted and never measured."
      exit 1
    end
    puts "ldfa: rank=" + LDFA_RANK.to_s +
         " rank_eff=" + recipe.gr_cache.gx_ld_rank_eff.to_s +
         " dout=" + recipe.gr_cache.gx_ld_dout.to_s +
         " v=" + N_CLASSES.to_s +
         " adapt=" + (LDFA_ADAPT == 1 ?
                        (LDFA_ETA > 0.0 ? "oja" : "oja-frozen") : "none") +
         " eta=" + (LDFA_ADAPT == 1 ? LDFA_ETA.to_s : "n/a") +
         " every=" + LDFA_EVERY.to_s +
         " m=" + ld_m_used.to_s + " m_req=" + LDFA_M.to_s +
         " refreshes=" + ld_refreshes.to_s +
         " first_refresh_step=" + ld_first_step.to_s +
         " p_ortho=init" +
         " b_eff_fro=" + num_or_null(recipe.gr_cache.gx_ld_fro_eff) +
         " b_full_fro=" + num_or_null(recipe.gr_cache.gx_ld_fro_full) +
         " scale_ratio=" +
           num_or_null(recipe.gr_cache.gx_ld_fro_full > 0.0 ?
             (recipe.gr_cache.gx_ld_fro_eff /
              recipe.gr_cache.gx_ld_fro_full) : 0.0) +
         " p_row_min=" + num_or_null(ld_row_min) +
         " p_row_max=" + num_or_null(ld_row_max) +
         " p_offdiag_max=" + num_or_null(ld_off_max) +
         " p_energy=" + num_or_null(ld_energy) +
         " p_energy_rand=" +
           num_or_null(LDFA_RANK.to_f / N_CLASSES.to_f) +
         # Same clock caveat as nDFA's: tnn_events_now_seconds returns 0.0
         # with no event file open, so without a TAO_RUN_DIR every interval
         # measures as exactly zero. Say `unmeasured` rather than ship a
         # metric that silently reads 0.0.
         " refresh_ms=" +
           (EVENTS.length > 0 ? (ld_wall_us / ld_refreshes / 1000).to_s :
                                "unmeasured-no-run-dir")
  end
  if INSTRUMENT == 1
    bc_g   = Array.new(bc_n * bc_n, 0.0)
    bc_gram!(bc_g, bc_buf, bc_n, N_CLASSES)
    bc_tr  = bc_trace(bc_buf, bc_n, N_CLASSES)
    bc_tr2 = bc_fro2(bc_g, bc_n)
    bc_lm  = bc_lambda_max(bc_g, bc_n, 64)
    bc_pr  = bc_tr2 > 0.0 ? (bc_tr * bc_tr / bc_tr2) : 0.0
    bc_sr  = bc_lm  > 0.0 ? (bc_tr / bc_lm) : 0.0
    # THE CEILING IS PART OF THE READING, not a footnote. rank(C_E) <= n,
    # so a participation ratio or stable rank sitting near n is capped by
    # the sample count rather than measured, and the E1 verdict must not
    # be read off it. `capped` states which side of that line this cell
    # is on, and `n`/`v` let a reader re-derive it.
    bc_cap = (bc_pr > 0.9 * bc_n.to_f || bc_sr > 0.9 * bc_n.to_f) ? "SAMPLE-CAPPED" :
             (bc_n < N_CLASSES ? "below-V" : "ok")
    puts "bcond: n=" + bc_n.to_s + " v=" + N_CLASSES.to_s +
         " batches=" + bc_batches.to_s +
         " trace=" + bc_tr.to_s +
         " lambda_max=" + bc_lm.to_s +
         " participation_ratio=" + bc_pr.to_s +
         " stable_rank=" + bc_sr.to_s +
         " capped=" + bc_cap
    # B's OWN stable rank is a flat CONTROL, never the diagnostic: B is a
    # fixed i.i.d. draw that does not move during training, so this
    # should be constant across rungs. If it is not, the draw or the
    # scale rule is wrong and every conditioning number above is suspect.
    puts "bcond_control: " + recipe.gr_cache.b_stable_rank_report
  end
  puts "graph: nodes=" + recipe.gr_cache.gx_graph_nodes.to_s +
       " bytes=" + recipe.gr_cache.gx_graph_bytes.to_s
  exit 0
end
fin = val_pass!(recipe, val_x, val_a, val_b, RETROFIT ? val_r : val_y,
                x_flat, idx_a, idx_b, labels, inc_flat, prev_a, prev_b,
                RETROFIT ? m_rlabels : m_labels, val_hp, logit_buf,
                FINAL_OUT, VAL_BATCHES, N_PAIRS, N_NODES, D_FEAT)
val_acc  = fin[0]
val_loss = fin[1]
val_seen = VAL_BATCHES * N_PAIRS
puts "val: acc=" + val_acc.to_s + " loss=" + val_loss.to_s + " n=" + val_seen.to_s
if RETROFIT
  # THE COST, ANALYTIC. What the FREEZE buys (identical for both credit
  # rules), stated apart from what DFA buys, because conflating them is
  # how this lane would overclaim.
  puts "retrofit: adapters=" + AD_LAYERS.to_s +
       " rank=" + AD_RANK.to_s +
       " credit=" + (AD_POLICY_S.length > 0 ? AD_POLICY_S : "chain") +
       " bb_grad_bytes_avoided=" +
         (FREEZE_BB ? recipe.gr_cache.backbone_grad_bytes : 0).to_s +
       " adapter_grad_bytes=" + recipe.gr_cache.adapter_grad_bytes.to_s
  # THE FREEZE, PROVED. Under --freeze-backbone the two signatures must
  # be BIT-IDENTICAL; a promise that nothing was stepped is not the same
  # as a measurement that nothing moved.
  puts "backbone: sig_pre=" + bb_sig_pre.to_s +
       " sig_post=" + recipe.gr_cache.backbone_sig.to_s +
       " frozen=" + (FREEZE_BB ? "1" : "0")
end
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
