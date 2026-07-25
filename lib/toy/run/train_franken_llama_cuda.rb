# lib/toy/run/train_franken_llama_cuda.rb — CUDA twin of
# train_franken_llama.rb (toy#109 CUDA franken leg). Hand-written like
# train_cuda.rb (recipe inlines backend-coupled TinyNN.* calls; the
# checkpoint write seam uses a CPU write-session inside the CUDA unit).
# Own compilation unit (landmine #16). The empty-policy contract on THIS
# backend: byte-equals libexec/toy-train-cuda (dynamic gate leg — CUDA
# curves are platform-scoped, never compared to CPU fixtures).
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
require_relative "../llm/engine/llama_seq_engine_cuda"
require_relative "../llm/recipes/franken_from_scratch_cuda"
require_relative "../llm/adamw"
require_relative "../llm/labels"
require_relative "../train/toy_gguf_writer"
require_relative "../train/toy_gguf_fuse"
require_relative "../train/dfa_b"

STEPS       = (ENV["STEPS"] || "5").to_i
SEED        = (ENV["SEED"]  || "0").to_i
TAO_RUN_DIR = ENV["TAO_RUN_DIR"] || ""
RUN_ID      = ENV["TOY_RUN_ID"] || ""
POLICY_S    = ENV["FRANKEN_POLICY"] || ""
B_SEED      = (ENV["FRANKEN_B_SEED"] || "0").to_i
B_DIST_S    = ENV["FRANKEN_B_DIST"] || ""
B_SCALE_S   = ENV["FRANKEN_B_SCALE"] || ""
ALIGN_ON    = (ENV["FRANKEN_ALIGN"] || "") == "1"

# Gate-fixed model SHAPE — identical to train.rb (the F0 contract).
VOCAB    = 627
D_MODEL  = 64
DONOR_D  = 128
N_HEADS  = 4
D_FF     = 128
N_LAYERS = 2
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

recipe = Toy::LLM::Recipes::FrankenFromScratchCuda.new
recipe.realize!(cfg, opts)

# Per-step inputs — VERBATIM train.rb (fail-loud corpus guard included).
if !File.exist?("data/ts_seqs.txt")
  puts "toy-train-franken: corpus not found: data/ts_seqs.txt (cwd-relative)"
  puts "  franken reads the first line of data/ts_seqs.txt (train.rb contract)."
  exit 1
end
raw        = File.read("data/ts_seqs.txt")
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
m_hp = Toy::AdamW.for_from_scratch.hp(0)

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
    config.add_raw("lr",      "0.001")
    config.add_num("seed",    SEED)
    rs.add_obj("config", config)
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
step = 0
while step < STEPS
  step_wall_start = TinyNN.tnn_events_now_seconds
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
    es.add_raw("lr",      "0.001")
    es.add_num("tokens",  CONTEXT)
    es.add_num("wall_us", step_wall_us)
    TinyNN.tnn_events_emit(es.dump)
  end

  mask_ts  = recipe.ff_cache.franken_mask_tensors
  mask_lis = recipe.ff_cache.franken_mask_lis
  mask_wis = recipe.ff_cache.franken_mask_wis
  if ALIGN_ON && EVENTS.length > 0 && mask_ts.length > 0
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

  if ALIGN_ON && EVENTS.length > 0 && n_align > 0
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
