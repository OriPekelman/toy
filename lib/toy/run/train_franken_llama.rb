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
#
# EVENTS (toy/v1, additive per #112): run_start carries a "franken" object
# {policy:[...], b_seed, b_dist, b_scale, b_sigma, mix_alpha, mask_tau};
# align events are {"kind":"align","phase":"train","t":…,"step":N,
# "li":L,"wi":W,"cos":C}. step/run_end/checkpoint identical to train.rb.
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
require_relative "../io/toy_corpus_loader"
require_relative "../dev/toy_describe_flow"

STEPS       = (ENV["STEPS"] || "5").to_i
SEED        = (ENV["SEED"]  || "0").to_i
TAO_RUN_DIR = ENV["TAO_RUN_DIR"] || ""
RUN_ID      = ENV["TOY_RUN_ID"] || ""
POLICY_S    = ENV["FRANKEN_POLICY"] || ""
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
VOCAB    = 627
D_MODEL  = BIG ? 256 : 64
DONOR_D  = BIG ? 512 : 128
N_HEADS  = BIG ? 8 : 4
D_FF     = BIG ? 512 : 128
N_LAYERS = SHAPE_S == "deep" ? 6 : 2
CONTEXT  = 32

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
opts.untied = true
opts.seed   = SEED
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

recipe = Toy::LLM::Recipes::FrankenFromScratch.new
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
p = 0; while p < CONTEXT; positions.push(p); p = p + 1; end

m_labels = Toy::Labels.next_token(seq_ids, VOCAB, CONTEXT, 1)
adamw = Toy::AdamW.for_from_scratch
adamw.lr = LR
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
    config.add_num("steps",   STEPS)
    config.add_raw("lr",      LR.to_s)
    config.add_num("warmup",  WARMUP)
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
    cost_params = VOCAB * DONOR_D + DONOR_D * D_MODEL +
                  N_LAYERS * (4 * D_MODEL * D_MODEL + 3 * D_MODEL * D_FF + 2 * D_MODEL) +
                  D_MODEL + D_MODEL * VOCAB
    cost_flops  = 2 * DONOR_D * D_MODEL +
                  N_LAYERS * (2 * (4 * D_MODEL * D_MODEL + 3 * D_MODEL * D_FF) + 4 * D_MODEL * CONTEXT) +
                  2 * D_MODEL * VOCAB
    cost = Toy::Json::Builder.new
    cost.add_num("total_params",    cost_params)
    cost.add_num("active_params",   cost_params)
    cost.add_num("flops_per_token", cost_flops)
    rs.add_obj("cost", cost)
    fr = Toy::Json::Builder.new
    fr.add_raw("policy",    Toy::Json.from_int_array(policy))
    fr.add_num("b_seed",    B_SEED)
    fr.add_num("b_dist",    opts.dfa_b_dist)
    fr.add_num("b_scale",   opts.dfa_b_scale)
    fr.add_num("b_sigma",   opts.dfa_b_sigma)
    fr.add_num("mix_alpha", mix_alpha)
    fr.add_num("mask_tau",  mask_tau)
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
corpus_off   = 0
corpus_bytes = CORPUS.length > 0 ? File.size(CORPUS) : 0
step = 0
while step < STEPS
  step_wall_start = TinyNN.tnn_events_now_seconds
  if CORPUS.length > 0
    # rotating-window stream: restart at 0 BEFORE the window would run
    # past EOF (read_seq's own EOF-wrap otherwise pins every later step
    # to the first window — the stuck-window failure mode).
    if corpus_off + CONTEXT * 4 > corpus_bytes
      corpus_off = 0
    end
    seq_ids  = ToyCorpusLoader.read_seq(CORPUS, corpus_off, CONTEXT)
    corpus_off = corpus_off + CONTEXT * 4
    m_labels = Toy::Labels.next_token_guarded(seq_ids, VOCAB, CONTEXT, 1)
  end
  if WARMUP > 0 && step < WARMUP
    # linear ramp; at step WARMUP-1 the factor is exactly 1.0 -> LR
    adamw.lr = LR * ((step + 1).to_f / WARMUP.to_f)
    m_hp = adamw.hp(0)
  end
  recipe.ff_cache.franken_refresh_b!   # toy#117: B leaves are per-step uploads
  loss = recipe.step!(seq_ids, positions, m_labels, m_hp, step == 0)
  final_loss = loss
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
    es.add_raw("lr",      adamw.lr.to_s)
    es.add_num("tokens",  CONTEXT)
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
  re.add_raw("exit_code",  "0")
  TinyNN.tnn_events_emit(re.dump)
  TinyNN.tnn_events_close
end
