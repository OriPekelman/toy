# lib/toy/run/train_franken_llama.rb — Spinel-compiled FRANKEN training
# runner (toy#112: the spec-callable surface for the Tao F-series).
#
# The from-scratch branch of lib/toy/run/train.rb, VERBATIM, driven through
# Recipes::FrankenFromScratch — so the EMPTY-POLICY run reproduces
# prep/fixtures/train_baseline.txt byte-for-byte (the Tao F0 check), and the
# policy/dfa knobs ride on top. Own compilation unit (landmine #16), exactly
# like every other trainer.
#
# ENV CONTRACT (superset of train.rb's):
#   STEPS / SEED / TAO_RUN_DIR / TOY_RUN_ID    — as train.rb
#   FRANKEN_POLICY   — per-layer tokens: chain | dfa | mix:<a> |
#                      maskdfa:<t> | maskbp:<t>  (default: all chain)
#   FRANKEN_B_SEED   — B-matrix seed (default 0)
#   FRANKEN_B_DIST   — gaussian | uniform | rademacher
#   FRANKEN_B_SCALE  — inv_sqrt_fan | glorot | fixed:<sigma>
#   FRANKEN_ALIGN    — "1" => emit per-step per-weight align events
#                      (kind:"align": cos∠(g_dfa, g_chain); opt-in — the
#                      shadow download costs at real sizes)
#   FRANKEN_DFA_GRANULARITY — matmul (default, per-weight MICRO DFA) |
#                      block (toy#158/F15: LightOn MACRO DFA — one tap
#                      per block output, full BP inside the block, no
#                      backward chain across blocks). `block` needs an
#                      ALL-dfa policy and is CPU-only (tao#18).
#   FRANKEN_OPTIMIZER — adamw (default) | muon | radam (toy#158: the
#                      RAdam rectification as a per-step LR scalar;
#                      carries beta2=0.999 and takes NO step while
#                      rho_t <= 4 — the runner announces that at start)
#
# EVENTS (toy/v1, additive per #112): run_start carries a "franken" object
# {policy:[...], b_seed, b_dist, b_scale, b_sigma, mix_alpha, mask_tau,
# policy_scope, policied_tensors, dfa_granularity, macro_taps}; align
# events are {"kind":"align","phase":"train","t":…,"step":N,"li":L,
# "wi":W,"wname":"attn_q.h0","cos":C,"dfa_norm":…,"bp_norm":…}.
# `wi` is LANE-LOCAL; `wname` is the cross-lane key (tao#19 item 3).
# step/run_end/checkpoint identical to train.rb.
#
# Spinel hygiene: as train.rb (hand-built JSON, no #{}, top-level consts,
# typed-empty seeds, while loops, monomorphic drive).

require_relative "../../toy"
require_relative "../io/json_builder"
require_relative "../io/json"
require_relative "../io/toy_events"
require_relative "../models/toy_smollm2"
require_relative "../llm/engine/llama_seq_engine"
require_relative "../llm/recipes/franken_from_scratch"
require_relative "../llm/adamw"
require_relative "../llm/labels"
require_relative "../train/toy_gguf_writer"
require_relative "../train/toy_gguf_fuse"
require_relative "../train/dfa_b"
require_relative "../train/toy_lr_schedule"
require_relative "../io/toy_corpus_loader"
require_relative "../dev/toy_describe_flow"

STEPS       = (ENV["STEPS"] || "5").to_i
SEED        = (ENV["SEED"]  || "0").to_i
TAO_RUN_DIR = ENV["TAO_RUN_DIR"] || ""
RUN_ID      = ENV["TOY_RUN_ID"] || ""
POLICY_S    = ENV["FRANKEN_POLICY"] || ""
# toy#151: which tensor class a :dfa layer policies. Until this ticket
# the per-layer policy reached attention qkv ONLY, which is exactly the
# caveat that made Tao's F1 wash out ("only attention qkv is policied").
# The FFN is the bulk of both params and compute, and the literature
# puts DFA nearest to BP in MLP-shaped blocks — so it is the placement
# worth testing.
#
# DEFAULT IS `attn`, deliberately: it makes every pre-toy#151 run
# reproduce BYTE-IDENTICALLY without a new flag, so F1/F3/F6/F7b/F0
# keep meaning what they meant. The cost is that whole-layer placement
# must be asked for — `--policy-scope all` — which is the right way
# round, because a flag string that silently changes meaning across a
# commit is how a re-run gets compared to a result it does not match
# (tao#17).
SCOPE_S = ENV["FRANKEN_POLICY_SCOPE"] || ""
if SCOPE_S.length > 0 && SCOPE_S != "attn" && SCOPE_S != "ffn" && SCOPE_S != "all"
  puts "toy-train-franken: FRANKEN_POLICY_SCOPE " + SCOPE_S + " unsupported (attn|ffn|all)"
  exit 1
end
SCOPE_CODE = SCOPE_S == "all" ? 0 : (SCOPE_S == "ffn" ? 2 : 1)
B_SEED      = (ENV["FRANKEN_B_SEED"] || "0").to_i
B_DIST_S    = ENV["FRANKEN_B_DIST"] || ""
B_SCALE_S   = ENV["FRANKEN_B_SCALE"] || ""
ALIGN_ON    = (ENV["FRANKEN_ALIGN"] || "") == "1"
# toy#122 (the F6 long-horizon instrument): CORPUS switches the feed to
# the streamed multi-sequence corpus (rotating windows, warm-start's
# reader); ALIGN_EVERY thins the shadow-telemetry emissions (align AND
# mask events + their downloads) to every Nth step.
CORPUS      = ENV["CORPUS"] || ""
ae_raw = (ENV["FRANKEN_ALIGN_EVERY"] || "1").to_i
ALIGN_EVERY = ae_raw < 1 ? 1 : ae_raw
# toy#126 (the F7b LR-sweep surface): FRANKEN_LR overrides the fixed
# from-scratch lr (hp slot 0; default 0.001, byte-null without the
# flag). FRANKEN_WARMUP=N ramps lr linearly over the first N steps
# (lr_t = LR*(t+1)/N, reaching LR exactly at step N) — a pure
# Ruby-side per-step hp rebuild: the hp Mat is re-fed to step! every
# step anyway, so warmup needs NO engine work.
LR_S = ENV["FRANKEN_LR"] || ""
LR   = LR_S.length > 0 ? LR_S.to_f : 0.001
wu_raw = (ENV["FRANKEN_WARMUP"] || "0").to_i
WARMUP = wu_raw < 0 ? 0 : wu_raw
# toy#129 item 2: FRANKEN_NO_SHADOW=1 drops dfa-policied qkv weights
# from the autodiff param set (no chain grad-acc, no backward
# expansion — the production cost shape). Applied updates are
# byte-identical to the shadow build (the gate pins it); align
# telemetry has nothing to compare against, so ALIGN + NO_SHADOW
# fails loud below. Modes 2/3/4 fail loud engine-side.
NO_SHADOW = (ENV["FRANKEN_NO_SHADOW"] || "") == "1"
# toy#129 item 3 (the enabling seam): FRANKEN_CKPT_EVERY=N writes the
# standard fused-llama checkpoint (lens folded, heads fused — the same
# writer as the final one) every N steps to weights/step_<s>.gguf.
# Checkpoint-boundary held-out evals ride these OFFLINE (the in-process
# eval seam collides with the global sched / F1.1 one-alloc landmines);
# 0 = off, byte-null. The write is downloads + a fresh plain-storage
# session (freed after) — no sched compute, gated byte-null vs no-ckpt.
ck_raw = (ENV["FRANKEN_CKPT_EVERY"] || "0").to_i
CKPT_EVERY = ck_raw < 0 ? 0 : ck_raw
# toy#133: FRANKEN_BATCH=B — B corpus windows per step, laid flat
# [T*B] with the GH#7 block-causal attention mask (window-isolated;
# smoke_gate_b_gt_1 pins the engine path) and WINDOW-LOCAL labels.
# The fixed-seq feed is the byte-gated single-window contract, so
# B>1 requires CORPUS. B=1 is byte-null (the mask stays NULL, the
# legacy diag_mask path). NOTE the mask approach computes (T*B)^2
# attention, not B*T^2 — the B-fold redundancy is real compute,
# reflected in run_start.cost.
# toy#136 (K1): FRANKEN_ACT ""|swiglu (default) | situ-glu (K3 M6,
# softcap both GLU factors); FRANKEN_NOPE=1 skips rope entirely (K3
# M8 — position from data order; the ablation axis until KDA lands);
# FRANKEN_SCHEDULE ""|const (default) | cosine (K3 M9 finding —
# cosine decays LR -> 0.1*LR over the post-warmup steps; NOTE libm
# cos is not cross-platform byte-stable, so cosine curves are
# platform-scoped — the const default stays byte-exact).
ACT_S = ENV["FRANKEN_ACT"] || ""
if ACT_S.length > 0 && ACT_S != "swiglu" && ACT_S != "situ-glu"
  puts "toy-train-franken: unknown FRANKEN_ACT " + ACT_S + " (swiglu|situ-glu)"
  exit 1
end
ACT_CODE = ACT_S == "situ-glu" ? 1 : 0
NOPE_ON = (ENV["FRANKEN_NOPE"] || "") == "1"
SCHED_S = ENV["FRANKEN_SCHEDULE"] || ""
if SCHED_S.length > 0 && SCHED_S != "const" && SCHED_S != "cosine"
  puts "toy-train-franken: unknown FRANKEN_SCHEDULE " + SCHED_S + " (const|cosine)"
  exit 1
end
COSINE_ON = SCHED_S == "cosine"
# toy#158 (F15) — RECIPE HYGIENE. Every transformer-LM negative we have
# (F4-F14) used MICRO DFA: per-weight `--policy dfa`, at lr 1e-3, under
# AdamW. LightOn's WORKING recipe (arXiv:2006.12878) is MACRO DFA —
# random feedback injected ONLY at block outputs, full BP inside the
# block — plus lr 5e-5 and a RAdam-class optimizer. So we may have been
# testing the wrong MODE at 20x the LR with the wrong optimizer, and
# until that is closed our negatives are not clean.
#
# `block` == LightOn's `--dfa simple` (1 tap per block at its output).
# `matmul` (default, byte-null) == today's per-weight rule. Their
# `full` (3 taps per block) is NOT our `matmul` — ours is finer still,
# per weight matrix.
#
# CALIBRATE EXPECTATIONS: LightOn's own LM number is ppl 52 (DFA) vs
# 34 (BP). F15 exists to CLOSE A DOUBT, not to find a positive; a
# negative here is the expected result and still the useful one.
GRAN_S = ENV["FRANKEN_DFA_GRANULARITY"] || ""
if GRAN_S.length > 0 && GRAN_S != "matmul" && GRAN_S != "block"
  puts "toy-train-franken: unknown FRANKEN_DFA_GRANULARITY " + GRAN_S + " (matmul|block)"
  exit 1
end
MACRO_ON = GRAN_S == "block"
# toy#137 K2b: KDA_LAYERS="0,2" builds those layers as Kimi Delta
# Attention blocks (K3's M1). Unset = all-attention, byte-null. The
# indices reach the engine via runner-direct INT-arg calls below (the
# GDN_LAYERS precedent: the opts->recipe array hop reads EMPTY here).
# A dfa/mask POLICY on a KDA layer fails loud engine-side (the
# credit-assignment wiring is attention-shaped) — KDA layers train
# chain; "KDA under DFA" is its own K-series question.
KDA_S = ENV["KDA_LAYERS"] || ""
# toy#137 K2c: KDA_CONV=0 disables the ShortConv on KDA q/k/v (default
# on = the faithful K3 form; identity-inited so step 1 is a forward
# no-op either way).
KDA_CONV_OFF = (ENV["KDA_CONV"] || "") == "0"
# toy#138 K3b: ATTNRES=1 replaces residual accumulation with K3's
# depth-attention (each layer input = learned softmax mixture over the
# embedding + every preceding layer output). Unset = byte-null.
ATTNRES_ON = (ENV["ATTNRES"] || "") == "1"
# K-series M10 (MTP): one extra block predicting t+2 through the tied
# head, with a second CE root weighted by lambda. Default off = byte-null.
MTP_ON = (ENV["FRANKEN_MTP"] || "") == "1"
# LAMBDA IS A COUPLING DIAL, NOT A LOSS WEIGHT. The second CE root is
# unscaled; lambda is applied in the arch as a gradient scale on every
# quantity the MTP branch borrows from the backbone. So MTP's own
# weights train at FULL strength for any lambda, and lambda controls
# only how hard the second root pulls on the backbone:
#   0 = the module exists and trains, backbone byte-identical to --mtp
#       off (the clean "second root exists but does not perturb" control)
#   1 = full coupling
# Scaling the loss root instead made lambda=0 freeze the module
# outright, which mtp_sig caught. See
# docs/roadmap/m10-mtp-status-2026-08-04.md.
# NOT `(ENV[...] || "0.3").to_f` — the CLI always SETS this key, to ""
# when --mtp-lambda is absent, and "".to_f is 0.0, not the default. That
# made a bare `--mtp` silently run at zero coupling, i.e. byte-identical
# to --mtp off. Length-check the string, the LR_S pattern.
mtpl_s = ENV["FRANKEN_MTP_LAMBDA"] || ""
MTP_LAMBDA = mtpl_s.length > 0 ? mtpl_s.to_f : 0.3
if MTP_LAMBDA < 0.0 || MTP_LAMBDA > 1.0
  puts "toy-train-franken: FRANKEN_MTP_LAMBDA must be in [0, 1] — it is a coupling fraction, not a loss weight"
  exit 1
end
if !MTP_ON && (ENV["FRANKEN_MTP_LAMBDA"] || "").length > 0
  puts "toy-train-franken: FRANKEN_MTP_LAMBDA needs FRANKEN_MTP=1"
  exit 1
end
# toy#139 / K-series K5: FRANKEN_OPTIMIZER adamw (default, byte-null)
# | muon. Muon rides the standard recipe — orthogonalized steps on the
# 2D hidden matrices, AdamW on norms/embeddings/head — and because
# toy's random-init layout stores q/k/v PER HEAD, that is K3's
# per-head Muon by construction (see engine.optimizer_init).
# sgd is NOT offered here: with no adamw node anywhere the shared
# recipe's unconditional hp upload would abort; the sgd control arm
# lives on franken-moe, which owns its upload.
OPT_S = ENV["FRANKEN_OPTIMIZER"] || ""
if OPT_S.length > 0 && OPT_S != "adamw" && OPT_S != "muon" && OPT_S != "radam"
  puts "toy-train-franken: FRANKEN_OPTIMIZER " + OPT_S + " unsupported here (adamw|muon|radam; sgd is franken-moe-only)"
  exit 1
end
# toy#158 (F15) ask 2: RAdam rides the SAME AdamW graph node — its
# rectification term is a per-step scalar on the LR (ToyLR.radam_rect),
# so `radam` selects optimizer code 0 and shapes lr Ruby-side.
#
# It ALSO carries beta2 = 0.999, which is part of the optimizer's
# identity (Liu et al.'s default, and LightOn's F15 recipe), NOT a
# separate knob: rho_inf is a function of beta2, so "RAdam at
# beta2=0.95" would rectify against a horizon nobody means. This is a
# REAL numerics change vs `adamw` — which is the point of running the
# arm — and it is recorded in run_start so a curve cannot be mistaken
# for an AdamW one.
RADAM_ON = OPT_S == "radam"
# toy#138 (K3a): FRANKEN_LAYER_PATTERN=hybrid — K3's layerwise hybrid
# (§2.1): THREE KDA layers then ONE global-attention layer, repeated,
# with the FINAL layer always global ("An additional Gated MLA layer
# is placed at the end of the backbone, ensuring that the final layer
# always performs global attention").
#
# TWO pattern values, and the difference is the "1" in 3:1:
#   hybrid  the global slots are toy's STANDARD attention block. This is
#           what toy#138 shipped, when no trainable MLA existed. Kept
#           bit-for-bit so its gates stay byte-null.
#   k3      the global slots are GATED MLA (K-series M2) — the faithful
#           K3 contract, 3 KDA : 1 Gated MLA with an MLA final layer.
# At N_LAYERS=6 both put KDA at 0,1,2,4; hybrid leaves 3,5 attention,
# k3 makes 3,5 MLA.
LAYER_PATTERN = ENV["FRANKEN_LAYER_PATTERN"] || ""
if LAYER_PATTERN.length > 0 && LAYER_PATTERN != "hybrid" && LAYER_PATTERN != "k3"
  puts "toy-train-franken: unknown FRANKEN_LAYER_PATTERN " + LAYER_PATTERN + " (hybrid|k3)"
  exit 1
end
MLA_S = ENV["MLA_LAYERS"] || ""
MLA_RANK = (ENV["FRANKEN_MLA_RANK"] || "0").to_i
if MLA_RANK < 0
  puts "toy-train-franken: FRANKEN_MLA_RANK must be >= 0 (0 = derived, inner/2)"
  exit 1
end
if LAYER_PATTERN.length > 0 && MLA_S.length > 0
  puts "toy-train-franken: --layer-pattern and --mla-layers both set — pick one (the pattern COMPUTES the mla layer list)"
  exit 1
end
if LAYER_PATTERN.length > 0 && KDA_S.length > 0
  puts "toy-train-franken: --layer-pattern and --kda-layers both set — pick one (the pattern COMPUTES the kda layer list)"
  exit 1
end
MLA_LIST = [0]; MLA_LIST.pop
if MLA_S.length > 0
  ml_parts = MLA_S.split(",")
  mlp = 0
  while mlp < ml_parts.length
    MLA_LIST.push(ml_parts[mlp].to_i)
    mlp = mlp + 1
  end
end
KDA_LIST = [0]; KDA_LIST.pop
if KDA_S.length > 0
  kl_parts = KDA_S.split(",")
  klp = 0
  while klp < kl_parts.length
    KDA_LIST.push(kl_parts[klp].to_i)
    klp = klp + 1
  end
end

b_raw = (ENV["FRANKEN_BATCH"] || "0").to_i
BATCH = b_raw > 0 ? b_raw : 1
if KDA_LIST.length > 0 && BATCH > 1
  puts "toy-train-franken: KDA_LAYERS with FRANKEN_BATCH > 1 is unsupported — the KDA block reshapes on the window length (B=1, like GDN); run KDA cells at --batch 1"
  exit 1
end
if BATCH > 1 && CORPUS.length == 0
  puts "toy-train-franken: FRANKEN_BATCH > 1 needs CORPUS (the fixed-seq feed is the byte-gated single-window contract)"
  exit 1
end
if NO_SHADOW && ALIGN_ON
  puts "toy-train-franken: FRANKEN_ALIGN=1 + FRANKEN_NO_SHADOW=1 — align telemetry compares the DFA grad against the chain shadow acc, which a no-shadow build does not create. Drop one."
  exit 1
end
# ---- toy#158: what macro-DFA cannot be combined with, and why. ----
# Every one of these would leave a BACKWARD PATH ACROSS BLOCKS alive,
# which is the single property that makes a run macro-DFA. A silent
# combination would produce a curve labelled `block` that is really a
# hybrid — the most expensive kind of wrong result, because it looks
# like a finding.
if MACRO_ON && ALIGN_ON
  puts "toy-train-franken: FRANKEN_DFA_GRANULARITY=block + FRANKEN_ALIGN=1 — align telemetry compares a DFA grad against the chain shadow, and macro-DFA CUTS the chain: there is no shadow to compare against (the weight's only gradient IS the DFA-derived one). Drop the align flag."
  exit 1
end
if MACRO_ON && ATTNRES_ON
  puts "toy-train-franken: FRANKEN_DFA_GRANULARITY=block + ATTNRES=1 — AttnRes mixes EVERY preceding block output into each layer's input, which re-opens the cross-block backward path the macro cut exists to remove."
  exit 1
end
if MACRO_ON && MTP_ON
  puts "toy-train-franken: FRANKEN_DFA_GRANULARITY=block + FRANKEN_MTP=1 — the MTP root couples back into backbone activations across the block boundary, so the cut would not hold."
  exit 1
end
if MACRO_ON && NO_SHADOW
  puts "toy-train-franken: FRANKEN_DFA_GRANULARITY=block + FRANKEN_NO_SHADOW=1 — no_shadow drops the per-WEIGHT chain grad-acc (the micro axis); macro has no per-weight DFA rule for it to pair with."
  exit 1
end

# Model shape — toy#124 presets (F7/F8 escalation). base = the gate
# shape, identical to train.rb (the F0 contract, byte-null); wide/deep
# scale width (and depth) at the PINNED vocab 627 + context 32 (the
# frozen-vocab corpus contract, toy#123). DONOR_D keeps the base 2x
# ratio to d_model (the projection lens is random-init here; the ratio
# keeps the architecture family consistent across presets).
SHAPE_S = ENV["FRANKEN_SHAPE"] || "base"
if SHAPE_S != "base" && SHAPE_S != "wide" && SHAPE_S != "deep"
  puts "unknown FRANKEN_SHAPE: " + SHAPE_S + " (want base|wide|deep)"
  exit 1
end
BIG      = SHAPE_S != "base"
D_MODEL  = BIG ? 256 : 64
DONOR_D  = BIG ? 512 : 128
N_HEADS  = BIG ? 8 : 4
D_FF     = BIG ? 512 : 128
N_LAYERS = SHAPE_S == "deep" ? 6 : 2

# toy#129 item 1: context + vocab join the runtime surface — the F9
# capacity fixture needs ctx >= 256 and a real-tokenizer vocab.
# Context: FRANKEN_CONTEXT (default 32, the byte-gated contract).
# Vocab: a TOYC pack header is authoritative (prep/pretokenize_pack.py;
# a conflicting FRANKEN_VOCAB fails loud); headerless packs keep the
# frozen-vocab contract (627, toy#123) unless FRANKEN_VOCAB overrides.
# No corpus -> 627, byte-null.
ctx_raw = (ENV["FRANKEN_CONTEXT"] || "0").to_i
if ctx_raw != 0 && ctx_raw < 2
  puts "toy-train-franken: FRANKEN_CONTEXT must be >= 2, got " + ctx_raw.to_s
  exit 1
end
CONTEXT = ctx_raw > 0 ? ctx_raw : 32
# toy#138 K3a: expand the hybrid pattern now that N_LAYERS is known.
if LAYER_PATTERN.length > 0
  lpi = 0
  while lpi < N_LAYERS
    is_global = ((lpi + 1) % 4) == 0 || lpi == N_LAYERS - 1
    if is_global
      # k3 fills the global slots with Gated MLA; hybrid leaves them as
      # the standard attention block (which needs no list).
      if LAYER_PATTERN == "k3"
        MLA_LIST.push(lpi)
      end
    else
      KDA_LIST.push(lpi)
    end
    lpi = lpi + 1
  end
  if KDA_LIST.length == 0
    puts "toy-train-franken: --layer-pattern " + LAYER_PATTERN + " at N_LAYERS=" + N_LAYERS.to_s + " yields no KDA layers (needs >= 2 layers)"
    exit 1
  end
  if BATCH > 1
    puts "toy-train-franken: --layer-pattern " + LAYER_PATTERN + " builds KDA layers, which are B=1 (see KDA_LAYERS)"
    exit 1
  end
end
# FAIL LOUD on out-of-range layer indices. build_gdn_flags! silently
# skips any index outside [0, N_LAYERS), so `--kda-layers 3,5` at the
# default N_LAYERS=2 marks NOTHING and produces a curve byte-identical
# to the all-attention default — indistinguishable from "the flag did
# nothing because the mechanism is a no-op". That cost real debugging
# time on this very change. Both lists are checked here, where
# N_LAYERS is finally known.
oob_i = 0
while oob_i < KDA_LIST.length
  if KDA_LIST[oob_i] < 0 || KDA_LIST[oob_i] >= N_LAYERS
    puts "toy-train-franken: KDA_LAYERS index " + KDA_LIST[oob_i].to_s +
         " is out of range for N_LAYERS=" + N_LAYERS.to_s + " (0.." + (N_LAYERS - 1).to_s + ")"
    exit 1
  end
  oob_i = oob_i + 1
end
oob_m = 0
while oob_m < MLA_LIST.length
  if MLA_LIST[oob_m] < 0 || MLA_LIST[oob_m] >= N_LAYERS
    puts "toy-train-franken: MLA_LAYERS index " + MLA_LIST[oob_m].to_s +
         " is out of range for N_LAYERS=" + N_LAYERS.to_s + " (0.." + (N_LAYERS - 1).to_s + ")"
    exit 1
  end
  oob_m = oob_m + 1
end
# MLA is B=1 for the same reason KDA is: the head slicing reshapes on
# seq_t, so one window per step.
hdr_v = CORPUS.length > 0 ? ToyCorpusLoader.probe_vocab(CORPUS) : 0
env_v = (ENV["FRANKEN_VOCAB"] || "0").to_i
if hdr_v > 0 && env_v > 0 && hdr_v != env_v
  puts "toy-train-franken: FRANKEN_VOCAB " + env_v.to_s + " conflicts with the pack header (TOYC vocab " + hdr_v.to_s + ")"
  exit 1
end
vsel0 = 627
if hdr_v > 0
  vsel0 = hdr_v
end
if hdr_v == 0 && env_v > 0
  vsel0 = env_v
end
VOCAB = vsel0

EVENTS = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/events.jsonl") : ""

# ---- policy / axes parsing (the twin-runner token grammar) ----
def parse_policy(pol_s, n_layers, p_alpha, p_tau)
  policy = [0]; policy.pop
  parts = pol_s.split(",")
  i = 0
  while i < n_layers
    m = 0
    al = 0.0
    ta = 0.0
    if i < parts.length
      tk = parts[i]
      if tk == "dfa"
        m = 1
      elsif tk.length > 4 && tk[0, 4] == "mix:"
        m = 2
        al = tk[4, tk.length - 4].to_f
      elsif tk.length > 8 && tk[0, 8] == "maskdfa:"
        m = 3
        ta = tk[8, tk.length - 8].to_f
      elsif tk.length > 7 && tk[0, 7] == "maskbp:"
        m = 4
        ta = tk[7, tk.length - 7].to_f
      end
    end
    policy.push(m)
    p_alpha.push(al)
    p_tau.push(ta)
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

# The LR SCHEDULE at `step` (0-indexed), before any RAdam rectification.
# Extracted from the training loop by toy#158 so the rectification can
# multiply it without a reassigned local (see the call site). Returns
# Float on every path — that monomorphism is the point.
def franken_lr_at(step, base, warmup, cosine_on, n_steps)
  if warmup > 0 && step < warmup
    # linear ramp; at step warmup-1 the factor is exactly 1.0 -> base
    return base * ((step + 1).to_f / warmup.to_f)
  end
  if cosine_on
    # toy#136: cosine decay base -> 0.1*base over the post-warmup span.
    span = n_steps - warmup
    prog = span > 0 ? ((step - warmup).to_f / span.to_f) : 1.0
    min_lr = base * 0.1
    return min_lr + 0.5 * (base - min_lr) * (1.0 + Math.cos(3.141592653589793 * prog))
  end
  base
end

# returns [cos, |a|, |b|] — norms ride into the align event (they matter:
# zero-norm sides distinguish dead-download from dead-gradient, and the
# |g_dfa|/|g_bp| ratio informs mix-alpha tuning).
# toy#118 — non-finite floats serialize as JSON null (the toy#106
# treatment, this runner's copy): finite iff x - x == 0.0.
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

p_alpha = [0.0]; p_alpha.pop
p_tau   = [0.0]; p_tau.pop
policy  = parse_policy(POLICY_S, N_LAYERS, p_alpha, p_tau)
# toy#158: macro-DFA is a WHOLE-STACK recipe. The cut is placed at every
# block boundary by the arch, so a per-layer policy cannot select which
# blocks are macro'd — accepting `chain,dfa` here would silently cut
# block 0's boundary too and report a mixed run as a per-layer result.
# Per-layer PLACEMENT is the micro axis (toy#151 / F14); this is the
# LightOn recipe, which is uniform by construction.
if MACRO_ON
  mp = 0
  while mp < policy.length
    if policy[mp] != 1
      puts "toy-train-franken: FRANKEN_DFA_GRANULARITY=block requires an ALL-dfa policy (layer " + mp.to_s +
           " is not `dfa`). The macro cut is placed at EVERY block boundary — a per-layer macro policy is not expressible, and per-layer placement is the MICRO axis (--policy-scope, toy#151)."
      exit 1
    end
    mp = mp + 1
  end
  if policy.length == 0
    puts "toy-train-franken: FRANKEN_DFA_GRANULARITY=block with an empty policy — pass FRANKEN_POLICY=dfa,... (block granularity is a DFA recipe; chain is already full BP)"
    exit 1
  end
end
# v1: one global alpha/tau (RecipeOptions carries scalars) — first
# non-chain layer's value wins; per-layer tables are a later slice.
mix_alpha = 0.5
mask_tau  = 0.0
pi = 0
while pi < policy.length
  if policy[pi] == 2 && mix_alpha == 0.5
    mix_alpha = p_alpha[pi]
  end
  if (policy[pi] == 3 || policy[pi] == 4) && mask_tau == 0.0
    mask_tau = p_tau[pi]
  end
  pi = pi + 1
end

cfg = Toy::SmolLM2Config.mha(VOCAB, D_MODEL, N_HEADS,
                             D_FF, N_LAYERS, CONTEXT, 10000.0, 1.0e-5)
cfg.donor_d_in = DONOR_D

opts = Toy::LLM::RecipeOptions.new
opts.t_seq  = CONTEXT
opts.t_batch = BATCH
opts.untied = true
opts.seed   = SEED
opts.no_shadow = NO_SHADOW ? 1 : 0
opts.act       = ACT_CODE
opts.rope_nope = NOPE_ON ? 1 : 0
opts.kda_conv  = KDA_CONV_OFF ? 0 : 1
opts.attnres   = ATTNRES_ON ? 1 : 0
opts.macro_dfa = MACRO_ON ? 1 : 0
opts.optimizer = OPT_S == "muon" ? 1 : 0
pi = 0
while pi < policy.length
  opts.credit_assignment.push(policy[pi])
  pi = pi + 1
end
opts.dfa_b_seed    = B_SEED
opts.dfa_b_dist    = dist_code(B_DIST_S)
opts.dfa_b_scale   = scale_code(B_SCALE_S)
opts.dfa_b_sigma   = scale_sigma(B_SCALE_S)
opts.dfa_mix_alpha = mix_alpha
opts.dfa_mask_tau  = mask_tau

kp = 0
while kp < KDA_LIST.length
  opts.kda_layers.push(KDA_LIST[kp])
  kp = kp + 1
end
mp_i = 0
while mp_i < MLA_LIST.length
  opts.mla_layers.push(MLA_LIST[mp_i])
  mp_i = mp_i + 1
end
opts.mla_rank = MLA_RANK

recipe = Toy::LLM::Recipes::FrankenFromScratch.new
kd = 0
while kd < opts.kda_layers.length
  recipe.ff_cache.add_kda_layer!(opts.kda_layers[kd])
  kd = kd + 1
end
# Driven from MLA_LIST directly rather than through opts.mla_layers —
# INT-arg calls off the local list, the standing workaround for the
# #688 array-param family. (opts.mla_layers is still populated above so
# the RecipeOptions surface stays at parity with kda_layers.)
# toy#151: the scope has to reach the cache BEFORE realize — the alloc
# reads it to decide which param flags to skip under --no-shadow, and
# the training-step builder reads it to decide which weights get the
# DFA rule. Setting it later would let those two disagree, leaving a
# live shadow acc on a weight the builder then DFA'd.
recipe.ff_cache.franken_policy_scope_init(SCOPE_CODE)
md = 0
while md < MLA_LIST.length
  recipe.ff_cache.add_mla_layer!(MLA_LIST[md])
  md = md + 1
end
recipe.ff_cache.mla_rank_init(opts.mla_rank)
recipe.ff_cache.mtp_init(MTP_ON ? 1 : 0)
recipe.ff_cache.mtp_lambda_init(MTP_LAMBDA)
recipe.realize!(cfg, opts)

# tao#flow-json-emit (#25): self-describing run bundle, parallel to
# events.jsonl — the toy#112 gap: F-series ran capture_flow:false and
# `tao report --html` lost the architecture DAG for franken runs.
ToyDescribeFlow.emit_flow_json(TAO_RUN_DIR, recipe.ff_cache.sess)

# Per-step inputs — VERBATIM train.rb (fail-loud corpus guard included).
# toy#122: with CORPUS set, sequences stream per step inside the loop
# (rotating windows over the packed-i32 file); labels are rebuilt per
# step. Without it: the byte-gated single fixed sequence, unchanged.
if CORPUS.length > 0 && !File.exist?(CORPUS)
  puts "toy-train-franken: corpus not found: " + CORPUS
  puts "  --corpus streams packed-i32 tokens (the warm-start reader)."
  exit 1
end
if CORPUS.length == 0 && !File.exist?("data/ts_seqs.txt")
  puts "toy-train-franken: corpus not found: data/ts_seqs.txt (cwd-relative)"
  puts "  franken reads the first line of data/ts_seqs.txt (train.rb contract)."
  exit 1
end
raw        = CORPUS.length == 0 ? File.read("data/ts_seqs.txt") : "0"
first_line = raw.split("\n")[0]
parts      = first_line.split(" ")
seq_ids    = [0]; seq_ids.pop
k = 0
while k < parts.length && k < CONTEXT
  seq_ids.push(parts[k].to_i)
  k = k + 1
end
while seq_ids.length < CONTEXT; seq_ids.push(0); end

positions = [0]; positions.pop
b_pos = 0
while b_pos < BATCH
  p = 0; while p < CONTEXT; positions.push(p); p = p + 1; end
  b_pos = b_pos + 1
end

m_labels = Toy::Labels.next_token(seq_ids, VOCAB, CONTEXT, 1)
adamw = Toy::AdamW.for_from_scratch
adamw.lr = LR
# toy#158: beta2 is part of `radam`'s identity (rho_inf = 2/(1-b2)-1),
# not a free knob — see the FRANKEN_OPTIMIZER block above.
if RADAM_ON
  adamw.beta2 = 0.999
  # LOUD, because the alternative is a flat curve that reads as a
  # broken arm: RAdam takes NO step while rho_t <= 4 (we return r_t=0
  # there instead of Liu et al.'s non-adaptive momentum step — see
  # ToyLR.radam_rect), and r_t then ramps in SLOWLY. At beta2=0.999
  # that is ~4 dead steps and r_t still ~0.06 at step 12, so a 5-step
  # smoke under radam is EXPECTED to look frozen. Say so at startup
  # rather than let someone debug it.
  rad_t = 1
  while rad_t < 1000 && ToyLR.radam_rect(rad_t, adamw.beta2) <= 0.0
    rad_t = rad_t + 1
  end
  puts "franken: optimizer=radam beta2=" + adamw.beta2.to_s +
       " first_stepping_step=" + rad_t.to_s +
       " r_t@" + rad_t.to_s + "=" + ToyLR.radam_rect(rad_t, adamw.beta2).to_s +
       " (steps before it take NO update by construction)"
end
m_hp = adamw.hp(0)

# ---- Events: run_start with the franken provenance object (#112). ----
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
    Toy::Events.add_provenance(rs,
      TinyNN.tnn_provenance_host_name, TinyNN.tnn_provenance_host_os,
      TinyNN.tnn_provenance_host_arch,
      TinyNN.tnn_backend_name(recipe.ff_cache.sess))
    model = Toy::Json::Builder.new
    model.add_str("arch", "llama")
    model.add_str("name", "franken-tinystories")
    model.add_str("shape", SHAPE_S)
    model.add_num("vocab",    cfg.vocab)
    model.add_num("d_model",  cfg.d_model)
    model.add_num("n_layers", cfg.n_layers)
    model.add_num("n_heads",  cfg.n_heads)
    model.add_num("n_kv",     cfg.n_kv)
    model.add_num("d_head",   cfg.head_dim)
    model.add_num("d_ff",     cfg.d_ff)
    rs.add_obj("model", model)
    config = Toy::Json::Builder.new
    config.add_num("context", CONTEXT)
    config.add_num("batch",   BATCH)
    config.add_num("steps",   STEPS)
    config.add_raw("lr",      LR.to_s)
    config.add_num("warmup",  WARMUP)
    config.add_str("schedule", COSINE_ON ? "cosine" : "const")
    config.add_str("act",      ACT_CODE == 1 ? "situ-glu" : "swiglu")
    config.add_str("rope",     NOPE_ON ? "nope" : "rope")
    config.add_str("kda_layers", KDA_S)
    config.add_bool("kda_conv", !KDA_CONV_OFF)
    config.add_str("layer_pattern", LAYER_PATTERN)
    config.add_bool("mtp", MTP_ON)
    config.add_num("mtp_lambda", MTP_ON ? MTP_LAMBDA : 0.0)
    # the clamp's cost, named rather than left to be rediscovered from a
    # suspiciously low second loss: the last 2 columns have no genuine
    # t+2 target and repeat the final in-window token.
    config.add_num("mtp_clamped_cols", MTP_ON ? 2.0 : 0.0)
    # post-init signature over every MTP weight; paired with the one in
    # run_end this is what PROVES the module trains rather than merely
    # building and reporting a loss (the K4b/experts_sig lesson).
    config.add_raw("mtp_sig", num_or_null(recipe.ff_cache.mtp_sig))
    config.add_num("mla_layers", MLA_LIST.length.to_f)
    config.add_num("mla_rank", MLA_RANK.to_f)
    config.add_bool("attnres", ATTNRES_ON)
    config.add_str("optimizer", OPT_S.length > 0 ? OPT_S : "adamw")
    # toy#158: beta2 travels with the run because `radam` changes it —
    # two curves under the same lr differ for this reason alone.
    config.add_raw("beta2", adamw.beta2.to_s)
    config.add_num("seed",    SEED)
    rs.add_obj("config", config)
    # toy#129 item 4: derived cost accounting — auditable FLOP-matching
    # from the bundle. params = embed(vocab*donor) + lens(donor*d) +
    # L*(4d^2 attn + 3*d*dff swiglu + 2d norms) + final norm d + untied
    # head d*vocab. flops_per_token = FORWARD, 2 flops/MAC: lens +
    # per-layer matmuls + attention scores/combine at full context +
    # head (embed row lookup ~0). Every param is trainable under every
    # policy arm (DFA changes the update rule, not the active set), so
    # active == total here — the MoE runner is where they diverge.
    # toy#137 K2c: a KDA layer's parameter set is NOT an attention
    # layer's — count each kind. attn layer = 4d² (qkvo) + 3·d·dff
    # (swiglu) + 2d (norms). KDA layer = rn_gamma d + 4·d·inner
    # (q/k/v/gate) + d·H (write) + d·r + r·inner (low-rank decay pair)
    # + b_α inner + A_h H + go_gamma inner + inner·d (out) + the conv's
    # 12·inner taps when enabled.
    n_kda_p = 0
    kp_i = 0
    while kp_i < KDA_LIST.length
      if KDA_LIST[kp_i] >= 0 && KDA_LIST[kp_i] < N_LAYERS
        n_kda_p = n_kda_p + 1
      end
      kp_i = kp_i + 1
    end
    n_attn_p = N_LAYERS - n_kda_p
    dh_p     = D_MODEL / N_HEADS
    inner_p  = N_HEADS * dh_p
    conv_p   = KDA_CONV_OFF ? 0 : 12 * inner_p
    kda_params  = D_MODEL + 4 * D_MODEL * inner_p + D_MODEL * N_HEADS +
                  D_MODEL * dh_p + dh_p * inner_p + inner_p + N_HEADS +
                  inner_p + inner_p * D_MODEL + conv_p
    attn_params = 4 * D_MODEL * D_MODEL + 3 * D_MODEL * D_FF + 2 * D_MODEL
    # toy#138 K3b: AttnRes adds one [d] pseudo-query per layer plus one
    # for the final aggregation (the ones-gamma is a constant, not a
    # param).
    attnres_params = ATTNRES_ON ? (N_LAYERS + 1) * D_MODEL : 0
    # K-series M10: the MTP module's OWN weights — the [d, 2d] input
    # projection plus one full block (it mirrors a backbone attention
    # layer). The head is TIED, so it adds no parameters; that tying is
    # exactly what makes MTP one extra block rather than a second model.
    # Counted here because a module that grows the model while the cost
    # block says otherwise is the silent-under-count class that already
    # bit K4b and toy#145.
    mtp_params = MTP_ON ? (2 * D_MODEL * D_MODEL + attn_params) : 0
    cost_params = VOCAB * DONOR_D + DONOR_D * D_MODEL +
                  n_attn_p * attn_params + n_kda_p * kda_params +
                  attnres_params + D_MODEL + D_MODEL * VOCAB + mtp_params
    # toy#137 K2c: KDA layers are LINEAR-attention — O(T·d²) per
    # sequence, i.e. NO T-proportional term per token (the recurrent
    # state is d_head×d_head per head, updated in O(d²) per token),
    # whereas a full-attention layer carries the 4·d·(T·B) scores +
    # combine term. Per KDA layer, per token: 5 projections
    # (q/k/v/gate/write) + the low-rank decay pair + the state ops
    # (2 matmuls of d_head² per head for u and o, plus the k⊗d outer)
    # + the out projection. Counted at 2 flops/MAC like the rest.
    n_kda = 0
    kl_i = 0
    while kl_i < KDA_LIST.length
      if KDA_LIST[kl_i] >= 0 && KDA_LIST[kl_i] < N_LAYERS
        n_kda = n_kda + 1
      end
      kl_i = kl_i + 1
    end
    n_attn = N_LAYERS - n_kda
    d_head_c = D_MODEL / N_HEADS
    kda_inner = N_HEADS * d_head_c
    kda_flops = 2 * (5 * D_MODEL * kda_inner) +
                2 * (D_MODEL * d_head_c + kda_inner * d_head_c) +
                2 * (3 * N_HEADS * d_head_c * d_head_c) +
                2 * (kda_inner * D_MODEL) +
                (KDA_CONV_OFF ? 0 : 21 * kda_inner)   # 3 streams x (4 mul + 3 add)
    # M10 forward cost, per token: the next-token embedding goes through
    # the SAME projection lens the main path uses (a second
    # DONOR_D x D_MODEL matmul), then the [2d -> d] input projection,
    # then one full block, then the tied head a second time. The head
    # costs flops even though it costs no params.
    mtp_flops = MTP_ON ? (2 * DONOR_D * D_MODEL +
                          2 * (2 * D_MODEL) * D_MODEL +
                          2 * (4 * D_MODEL * D_MODEL + 3 * D_MODEL * D_FF) +
                          4 * D_MODEL * CONTEXT * BATCH +
                          2 * D_MODEL * VOCAB) : 0
    cost_flops  = 2 * DONOR_D * D_MODEL +
                  n_attn * (2 * (4 * D_MODEL * D_MODEL + 3 * D_MODEL * D_FF) + 4 * D_MODEL * CONTEXT * BATCH) +
                  n_kda * kda_flops +
                  2 * D_MODEL * VOCAB + mtp_flops
    cost = Toy::Json::Builder.new
    cost.add_num("total_params",    cost_params)
    cost.add_num("active_params",   cost_params)
    cost.add_num("flops_per_token", cost_flops)
    rs.add_obj("cost", cost)
    fr = Toy::Json::Builder.new
    fr.add_raw("policy",    Toy::Json.from_int_array(policy))
    # toy#151: the policied SET, named. A bundle from before this ticket
    # is attention-only; without this field an old and a new run of the
    # same FRANKEN_POLICY are indistinguishable in the record.
    fr.add_str("policy_scope", SCOPE_S.length > 0 ? SCOPE_S : "attn")
    fr.add_str("policied_tensors",
               MACRO_ON ? "block_output" :
               (SCOPE_CODE == 1 ? "attn:q,k,v" :
                (SCOPE_CODE == 2 ? "ffn:gate,up,down" : "attn:q,k,v+ffn:gate,up,down")))
    # toy#158: WHICH DFA. Without this a macro run and a micro run of
    # the same FRANKEN_POLICY are indistinguishable in the record —
    # the same trap toy#151 closed for scope. `policy_scope` above is
    # a MICRO-only axis and is reported as-is; under macro the
    # policied set is the block output, which policied_tensors says.
    fr.add_str("dfa_granularity", MACRO_ON ? "block" : "matmul")
    fr.add_num("macro_taps", recipe.ff_cache.macro_b_handles.length)
    fr.add_num("b_seed",    B_SEED)
    fr.add_num("b_dist",    opts.dfa_b_dist)
    fr.add_num("b_scale",   opts.dfa_b_scale)
    fr.add_num("b_sigma",   opts.dfa_b_sigma)
    fr.add_num("mix_alpha", mix_alpha)
    fr.add_num("mask_tau",  mask_tau)
    fr.add_bool("shadow",   !NO_SHADOW)
    rs.add_obj("franken", fr)
    TinyNN.tnn_events_emit(rs.dump)
  else
    puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
  end
end

# ---- Training loop (train.rb VERBATIM + opt-in align events). ----
n_align = recipe.ff_cache.franken_align_grads.length
abuf = [0.0]; abuf.pop
gbuf = [0.0]; gbuf.pop
if ALIGN_ON && n_align > 0
  nmax = 0
  ai = 0
  while ai < n_align
    nw = TinyNN.tnn_tensor_nelements(recipe.ff_cache.franken_align_grads[ai])
    if nw > nmax; nmax = nw; end
    ai = ai + 1
  end
  z = 0
  while z < nmax
    abuf.push(0.0); gbuf.push(0.0)
    z = z + 1
  end
end

final_loss = 0.0
# toy#133: incremental batched one-hot (the eval_ce trick) — the Mat
# is allocated once; per step only the previous CONTEXT*BATCH scatter
# positions are cleared, never a ctx*batch*vocab refill (at B=32/
# ctx256/vocab50257 the refill was 412M writes per step — it dominated
# the CUDA step time). Values are byte-identical to
# next_token_guarded_batched; the batch gate's isolation null and the
# corpus byte-legs pin it.
lab_inc = CORPUS.length > 0 ? Mat.new(CONTEXT * BATCH, VOCAB) : Mat.new(1, 1)
lab_prev = [0]; lab_prev.pop
lp = 0; while lp < CONTEXT * BATCH; lab_prev.push(-1); lp = lp + 1; end
corpus_base  = CORPUS.length > 0 ? ToyCorpusLoader.data_offset(CORPUS) : 0
corpus_off   = corpus_base
corpus_bytes = CORPUS.length > 0 ? File.size(CORPUS) : 0
step = 0
while step < STEPS
  step_wall_start = TinyNN.tnn_events_now_seconds
  if CORPUS.length > 0
    # rotating-window stream: restart at 0 BEFORE the window would run
    # past EOF (read_seq's own EOF-wrap otherwise pins every later step
    # to the first window — the stuck-window failure mode).
    seq_ids = [0]; seq_ids.pop
    bw = 0
    while bw < BATCH
      if corpus_off + CONTEXT * 4 > corpus_bytes
        corpus_off = corpus_base
      end
      wnd = ToyCorpusLoader.read_seq(CORPUS, corpus_off, CONTEXT)
      corpus_off = corpus_off + CONTEXT * 4
      wk = 0
      while wk < CONTEXT
        seq_ids.push(wnd[wk])
        wk = wk + 1
      end
      bw = bw + 1
    end
    k2 = 0
    while k2 < CONTEXT * BATCH
      if lab_prev[k2] >= 0
        lab_inc.flat[k2 * VOCAB + lab_prev[k2]] = 0.0
      end
      wpos = k2 % CONTEXT
      tgt = (wpos + 1 < CONTEXT) ? seq_ids[k2 + 1] : seq_ids[k2]
      if tgt >= 0 && tgt < VOCAB
        lab_inc.flat[k2 * VOCAB + tgt] = 1.0
        lab_prev[k2] = tgt
      else
        lab_prev[k2] = -1
      end
      k2 = k2 + 1
    end
    m_labels = lab_inc
  end
  # The per-step LR is recomputed from the BASE `LR` every step, never
  # by mutating last step's value: toy#158's rectification MULTIPLIES
  # the schedule, and `adamw.lr = adamw.lr * r_t` would COMPOUND across
  # steps in the no-warmup/no-cosine case (where lr was previously set
  # once, before the loop) — an LR decaying to zero, which reads as a
  # converged run rather than a bug.
  #
  # Written as ONE expression per branch with no reassigned local:
  # `x = LR; x = <expr>` widened the local to RbVal and the compile
  # failed on `adamw.lr = x` (the Spinel poly-degradation family).
  # Byte-null guard: with none of the three shaping flags, adamw.lr and
  # m_hp stay exactly as they were built before the loop — the
  # pre-toy#158 behaviour, which the F0 fixture pins.
  if WARMUP > 0 || COSINE_ON || RADAM_ON
    adamw.lr = RADAM_ON ?
      franken_lr_at(step, LR, WARMUP, COSINE_ON, STEPS) * ToyLR.radam_rect(step + 1, adamw.beta2) :
      franken_lr_at(step, LR, WARMUP, COSINE_ON, STEPS)
    m_hp = adamw.hp(0)
  end
  recipe.ff_cache.franken_refresh_b!   # toy#117: B leaves are per-step uploads
  if MACRO_ON
    # toy#158: the macro B set is the same class of graph-input leaf as
    # the micro one (toy#117) — re-uploaded per step, not once.
    recipe.ff_cache.macro_refresh_b!
  end
  # K-series M10: the MTP inputs for this step — ids shifted by one (the
  # token the module embeds) and the t+2 one-hot. Both CLAMP at the
  # window edge, matching next_token's t+1 behaviour.
  if MTP_ON
    recipe.ff_cache.mtp_upload!(Toy::Labels.shift_ids(seq_ids, CONTEXT, BATCH, 1),
                                Toy::Labels.next_token_k(seq_ids, VOCAB, CONTEXT, BATCH, 2))
  end
  loss = recipe.step!(seq_ids, positions, m_labels, m_hp, step == 0)
  mtp_loss = MTP_ON ? recipe.ff_cache.mtp_loss_value : 0.0
  final_loss = loss
  final_mtp_loss = mtp_loss
  # The byte-gated line — to STDOUT (train.rb contract).
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s

  if EVENTS.length > 0
    step_wall_us = ((TinyNN.tnn_events_now_seconds - step_wall_start) * 1.0e6).to_i
    es = Toy::Json::Builder.new
    es.add_str("kind",  "step")
    es.add_str("phase", "train")
    es.add_num("t",       TinyNN.tnn_events_now_seconds)
    es.add_num("step",    step + 1)
    es.add_raw("loss",    num_or_null(loss))
    # K-series M10: the t+2 CE, reported ALONGSIDE the main loss rather
    # than folded into it — `loss` stays the main CE so a --mtp curve is
    # directly comparable to one without it. This is the t+2 read for
    # this lane: franken-llama has no in-runner eval_ce (it checkpoints
    # and is evaluated offline by `toy eval ce`), so the second head's
    # quality is carried per-step here and summarised in run_end.
    es.add_raw("mtp_loss", MTP_ON ? num_or_null(mtp_loss) : "null")
    es.add_raw("lr",      adamw.lr.to_s)
    es.add_num("tokens",  CONTEXT * BATCH)
    es.add_num("wall_us", step_wall_us)
    TinyNN.tnn_events_emit(es.dump)
  end

  mask_ts  = recipe.ff_cache.franken_mask_tensors
  mask_lis = recipe.ff_cache.franken_mask_lis
  mask_wis = recipe.ff_cache.franken_mask_wis
  align_step = (step % ALIGN_EVERY) == 0
  if ALIGN_ON && align_step && EVENTS.length > 0 && mask_ts.length > 0
    mi = 0
    while mi < mask_ts.length
      mt = mask_ts[mi]
      nw = TinyNN.tnn_tensor_nelements(mt)
      while gbuf.length < nw; gbuf.push(0.0); end
      rc_m = TinyNN.tnn_download_to_f64_array(recipe.ff_cache.sess, mt, gbuf, nw)
      if rc_m != 0
        puts "mask download failed: step=" + (step + 1).to_s + " rc=" + rc_m.to_s
      else
        msum = 0.0
        ii = 0
        while ii < nw
          msum = msum + gbuf[ii]
          ii = ii + 1
        end
        me = Toy::Json::Builder.new
        me.add_str("kind",  "mask")
        me.add_str("phase", "train")
        me.add_num("t",     TinyNN.tnn_events_now_seconds)
        me.add_num("step",  step + 1)
        me.add_num("li",    mask_lis[mi])
        me.add_num("wi",    mask_wis[mi])
        me.add_raw("density", num_or_null(msum / nw.to_f))
        TinyNN.tnn_events_emit(me.dump)
      end
      mi = mi + 1
    end
  end

  if ALIGN_ON && align_step && EVENTS.length > 0 && n_align > 0
    ai = 0
    while ai < n_align
      nw = TinyNN.tnn_tensor_nelements(recipe.ff_cache.franken_align_grads[ai])
      rc_g = TinyNN.tnn_download_to_f64_array(recipe.ff_cache.sess,
        recipe.ff_cache.franken_align_grads[ai], gbuf, nw)
      rc_a = TinyNN.tnn_download_to_f64_array(recipe.ff_cache.sess,
        recipe.ff_cache.franken_align_accs[ai], abuf, nw)
      if rc_g != 0 || rc_a != 0
        # never-mask: a failed shadow download must not masquerade as
        # a zero/stale gradient in the telemetry.
        puts "align download failed: step=" + (step + 1).to_s +
             " wi=" + recipe.ff_cache.franken_align_wis[ai].to_s +
             " rc_g=" + rc_g.to_s + " rc_a=" + rc_a.to_s
      end
      ae = Toy::Json::Builder.new
      ae.add_str("kind",  "align")
      ae.add_str("phase", "train")
      ae.add_num("t",     TinyNN.tnn_events_now_seconds)
      ae.add_num("step",  step + 1)
      ae.add_num("li",    recipe.ff_cache.franken_align_lis[ai])
      ae.add_num("wi",    recipe.ff_cache.franken_align_wis[ai])
      # tao#19 item 3: `wi` is LANE-LOCAL (an index into this lane's ft
      # layout); `wname` is what Tao's ingest keys on across lanes.
      ae.add_str("wname", recipe.ff_cache.franken_align_wnames[ai])
      cnn = cosv3(gbuf, abuf, nw)
      ae.add_raw("cos",      num_or_null(cnn[0]))
      ae.add_raw("dfa_norm", num_or_null(cnn[1]))
      ae.add_raw("bp_norm",  num_or_null(cnn[2]))
      TinyNN.tnn_events_emit(ae.dump)
      ai = ai + 1
    end
  end
  if CKPT_EVERY > 0 && TAO_RUN_DIR.length > 0 &&
     ((step + 1) % CKPT_EVERY) == 0 && (step + 1) < STEPS
    ck_rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
    ck_sess = TinyNN.tnn_session_new(0)
    ck_plist = ToyGGUFFuser.build_lens_folded_into_write_session(recipe.ff_cache, ck_sess, true)
    ck_rc = ToyGGUFWriter.write_step(cfg, ck_plist, TAO_RUN_DIR + "/weights", ck_rid, step + 1)
    if ck_rc != 0
      puts "checkpoint write failed: step=" + (step + 1).to_s + " rc=" + ck_rc.to_s
    end
    TinyNN.tnn_session_free(ck_sess)
  end
  step = step + 1
end

# ---- Final checkpoint + run_end (train.rb VERBATIM). ----
if EVENTS.length > 0 && TinyNN.tnn_events_active == 1
  rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
  write_sess = TinyNN.tnn_session_new(0)
  plist = ToyGGUFFuser.build_lens_folded_into_write_session(recipe.ff_cache, write_sess, true)
  rc = ToyGGUFWriter.write_step(cfg, plist, TAO_RUN_DIR + "/weights", rid, STEPS)
  if rc != 0
    puts "checkpoint write failed: rc=" + rc.to_s
  end

  re = Toy::Json::Builder.new
  re.add_str("kind", "run_end")
  re.add_num("t",          TinyNN.tnn_events_now_seconds)
  re.add_str("ended_at",   TinyNN.tnn_events_iso8601_now)
  re.add_str("reason",     "completed")
  re.add_num("final_step", STEPS)
  re.add_raw("final_loss", num_or_null(final_loss))
  re.add_raw("final_mtp_loss", MTP_ON ? num_or_null(final_mtp_loss) : "null")
  re.add_raw("mtp_sig",    num_or_null(recipe.ff_cache.mtp_sig))
  re.add_raw("exit_code",  "0")
  TinyNN.tnn_events_emit(re.dump)
  TinyNN.tnn_events_close
end
