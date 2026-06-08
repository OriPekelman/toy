# lib/toy/run/train_cuda.rb — Spinel-compiled from-scratch CUDA TRAINING runner.
#
# CUDA twin of lib/toy/run/train.rb, from-scratch ONLY. Hand-written (NOT
# mechanically mirrored): the recipe inlines backend-coupled TinyNN.* calls,
# and the checkpoint write seam deliberately straddles two backends, so this
# runner is maintained by hand. It is ABSENT from MIRRORABLE in
# prep/gen_cuda_mirror.rb (exactly like train.rb / infer.rb) — `make
# verify-mirrors` stays green, no mirror pair is introduced.
#
# This is the lib-side home of `toy train from-scratch --device cuda`'s
# compute. The CRuby CLI shell (lib/toy/core/cli/train.rb) cannot compute
# in-process — every ffi_lib-bearing lib crashes under MRI — so it locates the
# toy root, builds this runner (`make libexec/toy-train-cuda`), creates
# runs/<id>/, and shells out to it via Open3 with a CONTROLLED ENV. Same
# CRuby->runner COMPUTE BRIDGE as infer_cuda.rb.
#
# SINGLE-TYPE BINARY (landmine #16): TinyNNCuda is the COMPUTE path. The CPU
# TinyNN module is also defined (the CUDA monolith transitively requires
# transformer -> tinynn) and is used ONLY for the checkpoint write/fuse/drift
# seam — NOT for compute. The runner references ONLY the *Cuda recipe/cache
# types for the forward/backward graph.
#
# CONTRACT (read from ENV only):
#   STEPS        — number of training steps (default "5")
#   SEED         — random-init seed (default "0")
#   TAO_RUN_DIR  — when set, emit events.jsonl + a final checkpoint HERE.
#                  When empty, compute-only.
#   TOY_RUN_ID   — the resolved run id string (run_start/checkpoint metadata)
#
# DETERMINISM: the printed "step N: loss=" curve is EMPIRICALLY byte-
# deterministic run-to-run on THIS GB10 (sm_121, CUDA 13.0). This is NOT
# contractual — ggml-cuda float atomicAdd accumulation order is not fixed
# across GPUs / drivers / CUDA-toolkit versions / after a backend rebuild.
# If the gate is run on different hardware OR after rebuilding
# tinynn/libtinynn_ggml_cuda.a, the baseline MUST be re-pinned. The CUDA
# curve also differs from the CPU curve by ~0.3% (F32 vs f64 accumulation,
# docs/archive/p1-grad-bisection-2026-05-22.md) — that is EXPECTED, so the
# gate compares against its OWN fixture (prep/fixtures/train_cuda_baseline.txt),
# NOT the CPU train_baseline.txt.
#
# OUTPUT (byte-exact line the CLI + gate parse):
#   "step <N>: loss=<float>"   one per step, to STDOUT.
# Events go to events.jsonl and the checkpoint to weights/ — NEVER to stdout.
#
# Spinel hygiene (landmine #16): hand-built String-concat JSON (no #{}
# interpolation, no Math.exp); no Struct.new; VOCAB is a hardcoded int literal.

require_relative "../../toy"
require_relative "../../../vendor/spinel/spinel_kit/lib/spinel_kit/json_builder"
require_relative "../dev/toy_describe_flow"
require_relative "../io/toy_events"
require_relative "../models/toy_smollm2"
require_relative "../llm/engine/llama_seq_engine_cuda"
require_relative "../llm/recipes/from_scratch_cuda"
require_relative "../llm/recipes/warm_start_cuda"
require_relative "../llm/adamw"
require_relative "../llm/labels"
require_relative "../io/toy_corpus_loader"
require_relative "../train/toy_lr_schedule"
require_relative "../train/toy_gguf_writer"
require_relative "../train/toy_drift_grad"
require_relative "../train/toy_gguf_fuse"

# NOTE: this CUDA runner hosts the two RANDOM-INIT recipes (from-scratch +
# warm-start, both Toy::LLM::Engine::LlamaSeqEngineCuda#realize_for_random_init),
# selected by RECIPE. The LoRA recipe lives in a SEPARATE binary
# (lib/toy/run/train_lora_cuda.rb -> libexec/toy-train-lora-cuda): its
# #realize_for_mmap path cannot share a Spinel compilation unit with the
# random-init path without a cfg type-merge miscompile (landmine #16, same
# rationale as the CPU split train_lora.rb vs train.rb).

RECIPE      = ENV["RECIPE"] || "from-scratch"
STEPS       = (ENV["STEPS"] || "5").to_i
SEED        = (ENV["SEED"]  || "0").to_i
TAO_RUN_DIR = ENV["TAO_RUN_DIR"] || ""
RUN_ID      = ENV["TOY_RUN_ID"] || ""

# Gate-fixed model SHAPE — hardcoded (NOT env/flags), the smoke config.
# Hoisted to TOP-LEVEL (Spinel does not initialize a top-level CONSTANT
# assigned inside a conditional arm at runtime — "uninitialized constant"
# abort). vocab=627 d=64 donor=128 heads=4 n_kv=4 ff=128 L=2 ctx=32.
VOCAB    = 627
D_MODEL  = 64
DONOR_D  = 128
N_HEADS  = 4
D_FF     = 128
N_LAYERS = 2
CONTEXT  = 32

# Events sink — TOP-LEVEL (same constant-in-conditional Spinel caveat). FILE only.
EVENTS = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/events.jsonl") : ""

if RECIPE == "warm-start"
  # ==================================================================
  # WARM-START BRANCH (CUDA) — verbatim twin of the CPU warm-start branch
  # (train.rb:87-263) with TinyNN. -> TinyNNCuda. on every compute/event
  # call. INIT=scratch skips realize_warm! (no donor GGUF). The checkpoint
  # write seam stays CPU TinyNN (lens-fold path, donor_d_in>0) — the fuser
  # downloads from the CUDA session via CPU TinyNN.tnn_download_* on GB10 UVA,
  # identical to the from-scratch-cuda seam.
  # ==================================================================
  lr_max = 0.001
  lr_min = 0.00001
  warmup = 5
  corpus = ENV["CORPUS"] || "data/ts_seqs.bin"

  cfg_ws = Toy::SmolLM2Config.mha(VOCAB, D_MODEL, N_HEADS,
                                  D_FF, N_LAYERS, CONTEXT, 10000.0, 1.0e-5)
  cfg_ws.donor_d_in = DONOR_D

  recipe_ws = Toy::LLM::Recipes::WarmStartCuda.new
  # untied=true is mandatory when donor_d_in > 0.
  recipe_ws.realize_scratch!(cfg_ws, CONTEXT, 1, 0, true, false, SEED, 1.0)
  # INIT=scratch: skip realize_warm! (no donor GGUF, train from random init).
  recipe_ws.build!
  ToyDescribeFlow.emit_flow_json(TAO_RUN_DIR, recipe_ws.ws_cache.sess)

  # NAMED AdamW hp (byte-identical twin of the CPU runner). Defaults
  # (beta2=0.95, bias_correct=false) → slots5/6=constant betas. lr
  # refreshes per step from the cosine schedule; hp Mat rebuilt per step
  # (NOT the hot path) — byte-identical to the historical inline hp.
  adamw_ws = Toy::AdamW.new

  positions = [0]; positions.pop
  p = 0; while p < CONTEXT; positions.push(p); p = p + 1; end

  # --- Events (EVENTS hoisted to top-level; FILE only). ---

  if EVENTS.length > 0
    rc = TinyNNCuda.tnn_events_open(EVENTS)
    if rc == 0
      rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
      rs = SpinelKit::Json::Builder.new
      rs.add_str("kind", "run_start")
      rs.add_str("schema", "toy/v1")
      rs.add_num("t", TinyNNCuda.tnn_events_now_seconds)
      rs.add_str("started_at", TinyNNCuda.tnn_events_iso8601_now)
      rs.add_str("run_id", rid)
      rs.add_str("phase", "train")
      Toy::Events.add_provenance(rs,
        TinyNNCuda.tnn_provenance_host_name, TinyNNCuda.tnn_provenance_host_os,
        TinyNNCuda.tnn_provenance_host_arch,
        TinyNNCuda.tnn_backend_name(recipe_ws.ws_cache.sess))
      model = SpinelKit::Json::Builder.new
      model.add_str("arch", "llama")
      model.add_str("name", "warm-start-scratch-tinystories")
      model.add_num("vocab",    cfg_ws.vocab)
      model.add_num("d_model",  cfg_ws.d_model)
      model.add_num("n_layers", cfg_ws.n_layers)
      model.add_num("n_heads",  cfg_ws.n_heads)
      model.add_num("n_kv",     cfg_ws.n_kv)
      model.add_num("d_head",   cfg_ws.head_dim)
      model.add_num("d_ff",     cfg_ws.d_ff)
      rs.add_obj("model", model)
      config = SpinelKit::Json::Builder.new
      config.add_num("context", CONTEXT)
      config.add_num("steps",   STEPS)
      config.add_raw("lr",      "0.001")
      config.add_num("seed",    SEED)
      rs.add_obj("config", config)
      TinyNNCuda.tnn_events_emit(rs.dump)
    else
      puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
    end
  end

  # --- Training loop (0-indexed; cosine LR + streamed corpus). ---
  final_loss  = 0.0
  byte_offset = 0
  step        = 0
  while step < STEPS
    step_wall_start = TinyNNCuda.tnn_events_now_seconds
    lr = ToyLR.cosine(step, STEPS, lr_max, lr_min, warmup)
    adamw_ws.lr = lr
    m_hp = adamw_ws.hp(step)   # bias_correct=false → slots5/6=betas

    seq_ids = ToyCorpusLoader.read_seq(corpus, byte_offset, CONTEXT)
    byte_offset = byte_offset + CONTEXT * 4   # i32

    # In-vocab-guarded shift-by-one one-hot (warm-start streams corpus).
    m_labels = Toy::Labels.next_token_guarded(seq_ids, VOCAB, CONTEXT, 1)

    loss = recipe_ws.step!(seq_ids, positions, m_labels, m_hp, step == 0)
    final_loss = loss
    # The byte-gated line — to STDOUT.
    puts "step " + (step + 1).to_s + ": loss=" + loss.to_s

    if EVENTS.length > 0
      step_wall_us = ((TinyNNCuda.tnn_events_now_seconds - step_wall_start) * 1.0e6).to_i
      es = SpinelKit::Json::Builder.new
      es.add_str("kind",  "step")
      es.add_str("phase", "train")
      es.add_num("t",       TinyNNCuda.tnn_events_now_seconds)
      es.add_num("step",    step + 1)
      es.add_num("loss",    loss)
      es.add_num("lr",      lr)
      es.add_num("tokens",  CONTEXT)
      es.add_num("wall_us", step_wall_us)
      TinyNNCuda.tnn_events_emit(es.dump)
    end
    step = step + 1
  end

  # --- Final checkpoint + run_end (FILE only). CPU-TinyNN write seam. ---
  if EVENTS.length > 0 && TinyNNCuda.tnn_events_active == 1
    rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
    # FOLD the projection lens into the embedding so the checkpoint is a
    # STANDARD fused-llama GGUF that `toy infer`'s realize_for_mmap loads
    # unchanged (warm-start uses donor_d_in > 0 + untied=true — identical to
    # the from-scratch arm). The write session is a CPU TinyNN session; the
    # fuser downloads weights from the CUDA training session via CPU
    # TinyNN.tnn_download_* (works on GB10 UVA). write_sess_ws +
    # ws_cache.sess must stay alive until write_step returns.
    write_sess_ws = TinyNN.tnn_session_new(0)
    plist = ToyGGUFFuser.build_lens_folded_into_write_session(recipe_ws.ws_cache, write_sess_ws, true)
    rc = ToyGGUFWriter.write_step(cfg_ws, plist, TAO_RUN_DIR + "/weights", rid, STEPS)
    if rc != 0
      puts "checkpoint write failed: rc=" + rc.to_s
    end

    re = SpinelKit::Json::Builder.new
    re.add_str("kind", "run_end")
    re.add_num("t",          TinyNNCuda.tnn_events_now_seconds)
    re.add_str("ended_at",   TinyNNCuda.tnn_events_iso8601_now)
    re.add_str("reason",     "completed")
    re.add_num("final_step", STEPS)
    re.add_num("final_loss", final_loss)
    re.add_raw("exit_code",  "0")
    TinyNNCuda.tnn_events_emit(re.dump)
    TinyNNCuda.tnn_events_close
  end

else
# from-scratch — the existing body, BYTE-UNCHANGED.
cfg = Toy::SmolLM2Config.mha(VOCAB, D_MODEL, N_HEADS,
                             D_FF, N_LAYERS, CONTEXT, 10000.0, 1.0e-5)
cfg.donor_d_in = DONOR_D

# Realize the random-init graph THROUGH the CUDA recipe. CRITICAL: untied=TRUE
# (arg4), weight_dtype=0, qkv_bias=false, init_scale=1.0 — the SMOKE config,
# NOT 06's untied=false. realize_for_random_init self-enables full_finetune +
# train_embeddings, so no extra enable_* call.
recipe = Toy::LLM::Recipes::FromScratchCuda.new
recipe.realize!(cfg, CONTEXT, 1, 0, true, false, SEED, 1.0)
ToyDescribeFlow.emit_flow_json(TAO_RUN_DIR, recipe.fs_cache.sess)

# Per-step inputs built IN THE RUNNER, byte-identical to the CPU runner.
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

# Labels: shift-by-one one-hot (target = next token, or self at last pos).
# UNGUARDED (from-scratch seq_ids come from a known-good first line).
m_labels = Toy::Labels.next_token(seq_ids, VOCAB, CONTEXT, 1)

# CONSTANT hyper-params via NAMED AdamW (NOT 06's per-step bias-corrected
# hp; beta2=0.95 here, NOT 0.999; bias_correct=false → slots5/6=betas).
# Using 06's lora-style hp breaks the byte gate. Built ONCE (constant).
m_hp = Toy::AdamW.new.hp(0)

# --- Events (EVENTS hoisted to top-level; cheap-when-off; FILE only). ---

# git provenance read pure-Ruby from .git/HEAD.

if EVENTS.length > 0
  rc = TinyNNCuda.tnn_events_open(EVENTS)
  if rc == 0
    rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
    rs = SpinelKit::Json::Builder.new
    rs.add_str("kind", "run_start")
    rs.add_str("schema", "toy/v1")
    rs.add_num("t", TinyNNCuda.tnn_events_now_seconds)
    rs.add_str("started_at", TinyNNCuda.tnn_events_iso8601_now)
    rs.add_str("run_id", rid)
    rs.add_str("phase", "train")
    Toy::Events.add_provenance(rs,
      TinyNNCuda.tnn_provenance_host_name, TinyNNCuda.tnn_provenance_host_os,
      TinyNNCuda.tnn_provenance_host_arch,
      TinyNNCuda.tnn_backend_name(recipe.fs_cache.sess))
    model = SpinelKit::Json::Builder.new
    model.add_str("arch", "llama")
    model.add_str("name", "from-scratch-tinystories")
    model.add_num("vocab",    cfg.vocab)
    model.add_num("d_model",  cfg.d_model)
    model.add_num("n_layers", cfg.n_layers)
    model.add_num("n_heads",  cfg.n_heads)
    model.add_num("n_kv",     cfg.n_kv)
    model.add_num("d_head",   cfg.head_dim)
    model.add_num("d_ff",     cfg.d_ff)
    rs.add_obj("model", model)
    config = SpinelKit::Json::Builder.new
    config.add_num("context", CONTEXT)
    config.add_num("steps",   STEPS)
    config.add_raw("lr",      "0.001")
    config.add_num("seed",    SEED)
    rs.add_obj("config", config)
    TinyNNCuda.tnn_events_emit(rs.dump)
  else
    puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
  end
end

# --- Training loop: COMPOSE the recipe (drive step!, do NOT reimplement). ---
final_loss = 0.0
step = 0
while step < STEPS
  step_wall_start = TinyNNCuda.tnn_events_now_seconds
  loss = recipe.step!(seq_ids, positions, m_labels, m_hp, step == 0)
  final_loss = loss
  # The byte-gated line — to STDOUT.
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s

  if EVENTS.length > 0
    step_wall_us = ((TinyNNCuda.tnn_events_now_seconds - step_wall_start) * 1.0e6).to_i
    es = SpinelKit::Json::Builder.new
    es.add_str("kind",  "step")
    es.add_str("phase", "train")
    es.add_num("t",       TinyNNCuda.tnn_events_now_seconds)
    es.add_num("step",    step + 1)
    es.add_num("loss",    loss)
    es.add_raw("lr",      "0.001")
    es.add_num("tokens",  CONTEXT)
    es.add_num("wall_us", step_wall_us)
    TinyNNCuda.tnn_events_emit(es.dump)
  end
  step = step + 1
end

# --- Final checkpoint + run_end (only when TAO_RUN_DIR set). ---
# THE CROSS-BACKEND SEAM. The write session is a CPU TinyNN session; the fuser
# downloads weights from the CUDA training session via TinyNN.tnn_download_*
# (works on GB10 UVA) and writes them into the CPU write session. Compute above
# stayed on CUDA; only the write/fuse path uses CPU TinyNN.
if EVENTS.length > 0 && TinyNNCuda.tnn_events_active == 1
  rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
  # FOLD the projection lens into the embedding and FUSE per-head attention
  # so the checkpoint is a STANDARD fused-llama GGUF that `toy infer`'s
  # realize_for_mmap loads unchanged. from-scratch realizes with untied=true
  # and donor_d_in=DONOR_D>0, so the fold collapses the donor table + lens
  # into a standard [vocab, d_model] token_embd.weight. write_sess +
  # fs_cache.sess must stay alive until write_step returns.
  write_sess = TinyNN.tnn_session_new(0)
  plist = ToyGGUFFuser.build_lens_folded_into_write_session(recipe.fs_cache, write_sess, true)
  rc = ToyGGUFWriter.write_step(cfg, plist, TAO_RUN_DIR + "/weights", rid, STEPS)
  if rc != 0
    puts "checkpoint write failed: rc=" + rc.to_s
  end

  re = SpinelKit::Json::Builder.new
  re.add_str("kind", "run_end")
  re.add_num("t",          TinyNNCuda.tnn_events_now_seconds)
  re.add_str("ended_at",   TinyNNCuda.tnn_events_iso8601_now)
  re.add_str("reason",     "completed")
  re.add_num("final_step", STEPS)
  re.add_num("final_loss", final_loss)
  re.add_raw("exit_code",  "0")
  TinyNNCuda.tnn_events_emit(re.dump)
  TinyNNCuda.tnn_events_close
end
end
