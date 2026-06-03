# lib/toy/run/train_metal.rb — Spinel-compiled from-scratch Metal TRAINING runner.
#
# Metal twin of lib/toy/run/train.rb, from-scratch ONLY. Hand-written (NOT
# mechanically mirrored): the recipe inlines backend-coupled TinyNN.* calls,
# and the checkpoint write seam deliberately straddles two backends, so this
# runner is maintained by hand. It is ABSENT from MIRRORABLE in
# prep/gen_cuda_mirror.rb (exactly like train.rb / infer.rb) — `make
# verify-mirrors` stays green, no mirror pair is introduced.
#
# This is the lib-side home of `toy train from-scratch --device metal`'s
# compute. The CRuby CLI shell (lib/toy/core/cli/train.rb) cannot compute
# in-process — every ffi_lib-bearing lib crashes under MRI — so it locates the
# toy root, builds this runner (`make libexec/toy-train-metal`), creates
# runs/<id>/, and shells out to it via Open3 with a CONTROLLED ENV. Same
# CRuby->runner COMPUTE BRIDGE as infer_metal.rb.
#
# SINGLE-TYPE BINARY (landmine #16): TinyNNMetal is the COMPUTE path. The CPU
# TinyNN module is also defined (the Metal monolith transitively requires
# transformer -> tinynn) and is used ONLY for the checkpoint write/fuse/drift
# seam — NOT for compute. The runner references ONLY the *Metal recipe/cache
# types for the forward/backward graph.
#
# CONTRACT (read from ENV only):
#   STEPS        — number of training steps (default "5")
#   SEED         — random-init seed (default "0")
#   TAO_RUN_DIR  — when set, emit events.jsonl + a final checkpoint HERE.
#                  When empty, compute-only.
#   TOY_RUN_ID   — the resolved run id string (run_start/checkpoint metadata)
#
# DETERMINISM: the printed "step N: loss=" curve is expected to be byte-
# deterministic run-to-run on a GIVEN Mac (Apple GPU + a fixed Metal driver).
# This is NOT contractual — ggml-metal float accumulation order is not fixed
# across Apple GPUs / macOS versions / Metal-driver updates / after a backend
# rebuild. The baseline MUST be re-pinned on the Mac whenever the hardware,
# OS/driver, or tinynn/libtinynn_ggml_metal.a changes. The Metal curve also
# differs from the CPU curve (F32 vs f64 accumulation) — that is EXPECTED, so
# prep/metal_gate.rb compares the metal run against itself run-to-run (or a
# Mac-pinned fixture), NOT the CPU train_baseline.txt. RUNTIME-UNVERIFIED on
# gx10 (Linux, no Apple frameworks) — pin + gate on the Mac.
#
# OUTPUT (byte-exact line the CLI + gate parse):
#   "step <N>: loss=<float>"   one per step, to STDOUT.
# Events go to events.jsonl and the checkpoint to weights/ — NEVER to stdout.
#
# Spinel hygiene (landmine #16): hand-built String-concat JSON (no #{}
# interpolation, no Math.exp); no Struct.new; VOCAB is a hardcoded int literal.

require_relative "../../toy"
require_relative "../models/toy_smollm2"
require_relative "../llm/engine/llama_seq_engine_metal"
require_relative "../llm/recipes/from_scratch_metal"
require_relative "../llm/adamw"
require_relative "../llm/labels"
require_relative "../train/toy_gguf_writer"
require_relative "../train/toy_drift_grad"
require_relative "../train/toy_gguf_fuse"

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

cfg = Toy::SmolLM2Config.mha(VOCAB, D_MODEL, N_HEADS,
                             D_FF, N_LAYERS, CONTEXT, 10000.0, 1.0e-5)
cfg.donor_d_in = DONOR_D

# Realize the random-init graph THROUGH the Metal recipe. CRITICAL: untied=TRUE
# (arg4), weight_dtype=0, qkv_bias=false, init_scale=1.0 — the SMOKE config,
# NOT 06's untied=false. realize_for_random_init self-enables full_finetune +
# train_embeddings, so no extra enable_* call.
recipe = Toy::LLM::Recipes::FromScratchMetal.new
recipe.realize!(cfg, CONTEXT, 1, 0, true, false, SEED, 1.0)

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
git_sha    = "unknown"
git_branch = "unknown"
if File.exist?(".git/HEAD")
  head = File.read(".git/HEAD")
  if head.length > 0 && head[head.length - 1...head.length] == "\n"
    head = head[0...head.length - 1]
  end
  if head.length > 5 && head[0...5] == "ref: "
    ref_rel = head[5...head.length]
    pp = ref_rel.split("/")
    if pp.length >= 3
      git_branch = pp[pp.length - 1]
    end
    ref_path = ".git/" + ref_rel
    if File.exist?(ref_path)
      sha = File.read(ref_path)
      if sha.length >= 40
        git_sha = sha[0...40]
      end
    end
  else
    if head.length >= 40
      git_sha    = head[0...40]
      git_branch = "HEAD"
    end
  end
end

if EVENTS.length > 0
  rc = TinyNNMetal.tnn_events_open(EVENTS)
  if rc == 0
    rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
    rs  = "{\"kind\":\"run_start\",\"schema\":\"toy/v1\""
    rs = rs + ",\"t\":" + TinyNNMetal.tnn_events_now_seconds.to_s
    rs = rs + ",\"started_at\":\"" + TinyNNMetal.tnn_events_iso8601_now + "\""
    rs = rs + ",\"run_id\":\"" + rid + "\""
    rs = rs + ",\"phase\":\"train\""
    rs = rs + ",\"host\":{\"name\":\""   + TinyNNMetal.tnn_provenance_host_name + "\""
    rs = rs + ",\"os\":\""               + TinyNNMetal.tnn_provenance_host_os   + "\""
    rs = rs + ",\"arch\":\""             + TinyNNMetal.tnn_provenance_host_arch + "\"}"
    rs = rs + ",\"backend\":{\"kind\":\"" + TinyNNMetal.tnn_backend_name(recipe.fs_cache.sess) + "\"}"
    rs = rs + ",\"git\":{\"sha\":\""     + git_sha    + "\""
    rs = rs + ",\"branch\":\""           + git_branch + "\"}"
    rs = rs + ",\"model\":{\"arch\":\"llama\""
    rs = rs + ",\"name\":\"from-scratch-tinystories\""
    rs = rs + ",\"vocab\":"    + cfg.vocab.to_s
    rs = rs + ",\"d_model\":"  + cfg.d_model.to_s
    rs = rs + ",\"n_layers\":" + cfg.n_layers.to_s
    rs = rs + ",\"n_heads\":"  + cfg.n_heads.to_s
    rs = rs + ",\"n_kv\":"     + cfg.n_kv.to_s
    rs = rs + ",\"d_head\":"   + cfg.head_dim.to_s
    rs = rs + ",\"d_ff\":"     + cfg.d_ff.to_s
    rs = rs + "}"
    rs = rs + ",\"config\":{\"context\":" + CONTEXT.to_s
    rs = rs + ",\"steps\":" + STEPS.to_s
    rs = rs + ",\"lr\":0.001"
    rs = rs + ",\"seed\":"  + SEED.to_s
    rs = rs + "}"
    rs = rs + "}"
    TinyNNMetal.tnn_events_emit(rs)
  else
    puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
  end
end

# --- Training loop: COMPOSE the recipe (drive step!, do NOT reimplement). ---
final_loss = 0.0
step = 0
while step < STEPS
  step_wall_start = TinyNNMetal.tnn_events_now_seconds
  loss = recipe.step!(seq_ids, positions, m_labels, m_hp, step == 0)
  final_loss = loss
  # The byte-gated line — to STDOUT.
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s

  if EVENTS.length > 0
    step_wall_us = ((TinyNNMetal.tnn_events_now_seconds - step_wall_start) * 1.0e6).to_i
    es  = "{\"kind\":\"step\",\"phase\":\"train\""
    es = es + ",\"t\":"        + TinyNNMetal.tnn_events_now_seconds.to_s
    es = es + ",\"step\":"     + (step + 1).to_s
    es = es + ",\"loss\":"     + loss.to_s
    es = es + ",\"lr\":0.001"
    es = es + ",\"tokens\":"   + CONTEXT.to_s
    es = es + ",\"wall_us\":"  + step_wall_us.to_s
    es = es + "}"
    TinyNNMetal.tnn_events_emit(es)
  end
  step = step + 1
end

# --- Final checkpoint + run_end (only when TAO_RUN_DIR set). ---
# THE CROSS-BACKEND SEAM. The write session is a CPU TinyNN session; the fuser
# downloads weights from the Metal training session via TinyNN.tnn_download_*
# and writes them into the CPU write session. Compute above stayed on Metal;
# only the write/fuse path uses CPU TinyNN.
if EVENTS.length > 0 && TinyNNMetal.tnn_events_active == 1
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

  re  = "{\"kind\":\"run_end\""
  re = re + ",\"t\":"           + TinyNNMetal.tnn_events_now_seconds.to_s
  re = re + ",\"ended_at\":\""  + TinyNNMetal.tnn_events_iso8601_now + "\""
  re = re + ",\"reason\":\"completed\""
  re = re + ",\"final_step\":"  + STEPS.to_s
  re = re + ",\"final_loss\":"  + final_loss.to_s
  re = re + ",\"exit_code\":0"
  re = re + "}"
  TinyNNMetal.tnn_events_emit(re)
  TinyNNMetal.tnn_events_close
end
