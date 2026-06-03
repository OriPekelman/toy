# lib/toy/run/train.rb — Spinel-compiled from-scratch TRAINING compute runner.
#
# This is the lib-side home of `toy train from-scratch`'s compute. The CRuby
# CLI shell (lib/toy/core/cli/train.rb) cannot compute in-process — every
# ffi_lib-bearing lib crashes under MRI — so it locates the toy root, builds
# this runner (`make libexec/toy-train`), creates runs/<id>/, and shells out
# to it via Open3 with a CONTROLLED ENV. This is the same CRuby→runner COMPUTE
# BRIDGE that lib/toy/run/infer.rb established; train follows it verbatim.
#
# CONTRACT (read from ENV only — lib-vs-example scope, NO experiment config
# baked as flags; the gate-fixed model SHAPE is hardcoded below, exactly as
# infer.rb hardcodes its fallback prompt IDs):
#   STEPS        — number of training steps (default "5")
#   SEED         — random-init seed (default "0")
#   TAO_RUN_DIR  — when set, emit events.jsonl + a final checkpoint HERE
#                  (06/TAO_RUN_DIR convention). When empty, compute-only.
#   TOY_RUN_ID   — the resolved run id string (run_start/checkpoint metadata)
#
# Backend: CPU only. The recipe inlines TinyNN.* (backend-coupled) and the
# CUDA twin (from_scratch_cuda.rb) is DEFERRED alongside the GPU deferral, so
# this runner is NOT mechanically mirrorable; it is deliberately ABSENT from
# MIRRORABLE in prep/gen_cuda_mirror.rb (exactly like infer.rb). A --device
# runner is a later slice. `make verify-mirrors` stays green: no mirror pair
# is introduced.
#
# DETERMINISM: the runner re-uses the SMOKE reference's EXACT config / seed /
# per-step inputs / CONSTANT hyper-params (NOT example 06's untied=false +
# beta2=0.999 + bias-corrected per-step hp). The recipe's step! op-order is
# frozen (from_scratch.rb:83-97), so the printed "step N: loss=" lines are
# byte-for-byte reproducible — this is what prep/train_gate.rb gates against
# prep/fixtures/train_baseline.txt.
#
# OUTPUT (byte-exact line the CLI + gate parse):
#   "step <N>: loss=<float>"   one per step, to STDOUT (smoke L91 verbatim).
# Events go to events.jsonl and the checkpoint to weights/ — NEVER to stdout —
# so the structural additions cannot perturb the byte-gated stdout.
#
# Spinel hygiene (landmine #16): hand-built String-concat JSON (no #{}
# interpolation, no Math.exp); no Struct.new; VOCAB is a hardcoded int literal
# (never read ts_vocab.txt strings — poly-dispatch landmine, 06:21).

require_relative "../../toy"
require_relative "../models/toy_smollm2"
require_relative "../io/toy_corpus_loader"
require_relative "../train/toy_lr_schedule"
require_relative "../llm/engine/llama_seq_engine"
require_relative "../llm/recipes/from_scratch"
require_relative "../llm/recipes/warm_start"
require_relative "../llm/adamw"
require_relative "../llm/labels"
require_relative "../train/toy_gguf_writer"
require_relative "../train/toy_drift_grad"
require_relative "../train/toy_gguf_fuse"

# NOTE: this runner hosts the two RANDOM-INIT recipes (from-scratch +
# warm-start, both Toy::LLM::Engine::LlamaSeqEngine#realize_for_random_init). The
# LoRA recipe lives in a SEPARATE binary (lib/toy/run/train_lora.rb →
# libexec/toy-train-lora): its #realize_for_mmap path, compiled alongside
# the random-init path here, makes Spinel merge the `cfg` receiver type and
# miscompile the (dead) Toy::RMSNorm#forward / Toy::SmolLM2 model methods
# (landmine #16, polymorphic-merge family). Splitting lora out keeps this
# binary's realize path monomorphic and the from-scratch arm byte-identical.

RECIPE      = ENV["RECIPE"] || "from-scratch"
STEPS       = (ENV["STEPS"] || "5").to_i
SEED        = (ENV["SEED"]  || "0").to_i
TAO_RUN_DIR = ENV["TAO_RUN_DIR"] || ""
RUN_ID      = ENV["TOY_RUN_ID"] || ""

# Gate-fixed model SHAPE — hardcoded (NOT env/flags), the smoke config.
# Hoisted to TOP-LEVEL (NOT inside the RECIPE branches): Spinel does not
# initialize top-level CONSTANTS assigned inside a conditional arm at
# runtime ("uninitialized constant" abort), so the shared shape lives here.
# from-scratch + warm-start use the SAME shape (both llama-scratch).
# vocab=627 d=64 donor=128 heads=4 n_kv=4 ff=128 L=2 ctx=32 rope=1e4 eps=1e-5.
VOCAB    = 627
D_MODEL  = 64
DONOR_D  = 128
N_HEADS  = 4
D_FF     = 128
N_LAYERS = 2
CONTEXT  = 32

# Events sink — TOP-LEVEL (same constant-in-conditional Spinel caveat as the
# shape consts; an EVENTS constant assigned inside a branch reads back empty
# at runtime, silently skipping all event/checkpoint writes). FILE only.
EVENTS = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/events.jsonl") : ""

if RECIPE == "warm-start"
  # ==================================================================
  # WARM-START BRANCH — fully self-contained (landmine #16). Mirrors
  # examples/smoke_recipe_warm_start.rb at INIT=scratch (the gate ground
  # truth). INIT=scratch skips realize_warm! (no donor GGUF).
  # ==================================================================
  lr_max = 0.001
  lr_min = 0.00001
  warmup = 5
  corpus = ENV["CORPUS"] || "data/ts_seqs.bin"

  cfg_ws = Toy::SmolLM2Config.mha(VOCAB, D_MODEL, N_HEADS,
                                  D_FF, N_LAYERS, CONTEXT, 10000.0, 1.0e-5)
  cfg_ws.donor_d_in = DONOR_D

  recipe_ws = Toy::LLM::Recipes::WarmStart.new
  # untied=true is mandatory when donor_d_in > 0.
  recipe_ws.realize_scratch!(cfg_ws, CONTEXT, 1, 0, true, false, SEED, 1.0)
  # INIT=scratch: skip realize_warm! (no donor GGUF, train from random init).
  recipe_ws.build!

  # NAMED AdamW hp. Defaults (beta2=0.95, bias_correct=false) → slots
  # 5/6 = constant betas, byte-identical to the historical inline hp.
  # adamw.lr refreshes each step from the cosine schedule; we rebuild a
  # fresh 7-float Mat per step (NOT the hot path) instead of mutating
  # flat[0] in place — byte-identical, and faithful to the named object.
  adamw_ws = Toy::AdamW.new

  positions = [0]; positions.pop
  p = 0; while p < CONTEXT; positions.push(p); p = p + 1; end

  # --- Events (EVENTS hoisted to top-level; FILE only). ---
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
    rc = TinyNN.tnn_events_open(EVENTS)
    if rc == 0
      rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
      rs  = "{\"kind\":\"run_start\",\"schema\":\"toy/v1\""
      rs = rs + ",\"t\":" + TinyNN.tnn_events_now_seconds.to_s
      rs = rs + ",\"started_at\":\"" + TinyNN.tnn_events_iso8601_now + "\""
      rs = rs + ",\"run_id\":\"" + rid + "\""
      rs = rs + ",\"phase\":\"train\""
      rs = rs + ",\"host\":{\"name\":\""   + TinyNN.tnn_provenance_host_name + "\""
      rs = rs + ",\"os\":\""               + TinyNN.tnn_provenance_host_os   + "\""
      rs = rs + ",\"arch\":\""             + TinyNN.tnn_provenance_host_arch + "\"}"
      rs = rs + ",\"backend\":{\"kind\":\"" + TinyNN.tnn_backend_name(recipe_ws.ws_cache.sess) + "\"}"
      rs = rs + ",\"git\":{\"sha\":\""     + git_sha    + "\""
      rs = rs + ",\"branch\":\""           + git_branch + "\"}"
      rs = rs + ",\"model\":{\"arch\":\"llama\""
      rs = rs + ",\"name\":\"warm-start-scratch-tinystories\""
      rs = rs + ",\"vocab\":"    + cfg_ws.vocab.to_s
      rs = rs + ",\"d_model\":"  + cfg_ws.d_model.to_s
      rs = rs + ",\"n_layers\":" + cfg_ws.n_layers.to_s
      rs = rs + ",\"n_heads\":"  + cfg_ws.n_heads.to_s
      rs = rs + ",\"n_kv\":"     + cfg_ws.n_kv.to_s
      rs = rs + ",\"d_head\":"   + cfg_ws.head_dim.to_s
      rs = rs + ",\"d_ff\":"     + cfg_ws.d_ff.to_s
      rs = rs + "}"
      rs = rs + ",\"config\":{\"context\":" + CONTEXT.to_s
      rs = rs + ",\"steps\":" + STEPS.to_s
      rs = rs + ",\"lr\":0.001"
      rs = rs + ",\"seed\":"  + SEED.to_s
      rs = rs + "}"
      rs = rs + "}"
      TinyNN.tnn_events_emit(rs)
    else
      puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
    end
  end

  # --- Training loop (0-indexed; cosine LR + streamed corpus). ---
  final_loss  = 0.0
  byte_offset = 0
  step        = 0
  while step < STEPS
    step_wall_start = TinyNN.tnn_events_now_seconds
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
      step_wall_us = ((TinyNN.tnn_events_now_seconds - step_wall_start) * 1.0e6).to_i
      es  = "{\"kind\":\"step\",\"phase\":\"train\""
      es = es + ",\"t\":"        + TinyNN.tnn_events_now_seconds.to_s
      es = es + ",\"step\":"     + (step + 1).to_s
      es = es + ",\"loss\":"     + loss.to_s
      es = es + ",\"lr\":"       + lr.to_s
      es = es + ",\"tokens\":"   + CONTEXT.to_s
      es = es + ",\"wall_us\":"  + step_wall_us.to_s
      es = es + "}"
      TinyNN.tnn_events_emit(es)
    end
    step = step + 1
  end

  # --- Final checkpoint + run_end (FILE only). ---
  if EVENTS.length > 0 && TinyNN.tnn_events_active == 1
    rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
    # FOLD the projection lens into the embedding and FUSE per-head
    # attention so the checkpoint is a STANDARD fused-llama GGUF that
    # `toy infer`'s realize_for_mmap loads unchanged (warm-start uses
    # donor_d_in > 0 + untied=true, train.rb:99/103 — identical to the
    # from-scratch arm). write_sess + ws_cache.sess must stay alive
    # until write_step returns (gguf_add_tensor reads host ptrs at
    # finalize); both are local/ivar-held to end of this block.
    write_sess_ws = TinyNN.tnn_session_new(0)
    plist = ToyGGUFFuser.build_lens_folded_into_write_session(recipe_ws.ws_cache, write_sess_ws, true)
    rc = ToyGGUFWriter.write_step(cfg_ws, plist, TAO_RUN_DIR + "/weights", rid, STEPS)
    if rc != 0
      puts "checkpoint write failed: rc=" + rc.to_s
    end

    re  = "{\"kind\":\"run_end\""
    re = re + ",\"t\":"           + TinyNN.tnn_events_now_seconds.to_s
    re = re + ",\"ended_at\":\""  + TinyNN.tnn_events_iso8601_now + "\""
    re = re + ",\"reason\":\"completed\""
    re = re + ",\"final_step\":"  + STEPS.to_s
    re = re + ",\"final_loss\":"  + final_loss.to_s
    re = re + ",\"exit_code\":0"
    re = re + "}"
    TinyNN.tnn_events_emit(re)
    TinyNN.tnn_events_close
  end

else
# from-scratch — the existing body. Shape constants (VOCAB/D_MODEL/…)
# are now hoisted to top-level (see note above); the compute below is
# byte-identical to the historical from-scratch runner.
cfg = Toy::SmolLM2Config.mha(VOCAB, D_MODEL, N_HEADS,
                             D_FF, N_LAYERS, CONTEXT, 10000.0, 1.0e-5)
cfg.donor_d_in = DONOR_D

# Realize the random-init graph THROUGH the recipe. CRITICAL: untied=TRUE
# (arg4), weight_dtype=0, qkv_bias=false, init_scale=1.0 — the SMOKE config,
# NOT 06's untied=false. realize_for_random_init self-enables full_finetune +
# train_embeddings, so no extra enable_* call.
recipe = Toy::LLM::Recipes::FromScratch.new
recipe.realize!(cfg, CONTEXT, 1, 0, true, false, SEED, 1.0)

# Per-step inputs built IN THE RUNNER (the from-scratch entrypoint, the
# fixture's analog under lib-vs-example), byte-identical to smoke L56-84.
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

# git provenance read pure-Ruby from .git/HEAD (06:264-292).
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
  rc = TinyNN.tnn_events_open(EVENTS)
  if rc == 0
    rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
    rs  = "{\"kind\":\"run_start\",\"schema\":\"toy/v1\""
    rs = rs + ",\"t\":" + TinyNN.tnn_events_now_seconds.to_s
    rs = rs + ",\"started_at\":\"" + TinyNN.tnn_events_iso8601_now + "\""
    rs = rs + ",\"run_id\":\"" + rid + "\""
    rs = rs + ",\"phase\":\"train\""
    rs = rs + ",\"host\":{\"name\":\""   + TinyNN.tnn_provenance_host_name + "\""
    rs = rs + ",\"os\":\""               + TinyNN.tnn_provenance_host_os   + "\""
    rs = rs + ",\"arch\":\""             + TinyNN.tnn_provenance_host_arch + "\"}"
    rs = rs + ",\"backend\":{\"kind\":\"" + TinyNN.tnn_backend_name(recipe.fs_cache.sess) + "\"}"
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
    TinyNN.tnn_events_emit(rs)
  else
    puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
  end
end

# --- Training loop: COMPOSE the recipe (drive step!, do NOT reimplement). ---
final_loss = 0.0
step = 0
while step < STEPS
  step_wall_start = TinyNN.tnn_events_now_seconds
  loss = recipe.step!(seq_ids, positions, m_labels, m_hp, step == 0)
  final_loss = loss
  # The byte-gated line (smoke L91 verbatim) — to STDOUT.
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s

  if EVENTS.length > 0
    step_wall_us = ((TinyNN.tnn_events_now_seconds - step_wall_start) * 1.0e6).to_i
    es  = "{\"kind\":\"step\",\"phase\":\"train\""
    es = es + ",\"t\":"        + TinyNN.tnn_events_now_seconds.to_s
    es = es + ",\"step\":"     + (step + 1).to_s
    es = es + ",\"loss\":"     + loss.to_s
    es = es + ",\"lr\":0.001"
    es = es + ",\"tokens\":"   + CONTEXT.to_s
    es = es + ",\"wall_us\":"  + step_wall_us.to_s
    es = es + "}"
    TinyNN.tnn_events_emit(es)
  end
  step = step + 1
end

# --- Final checkpoint + run_end (only when TAO_RUN_DIR set). ---
if EVENTS.length > 0 && TinyNN.tnn_events_active == 1
  rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
  # FOLD the projection lens into the embedding and FUSE per-head
  # attention so the checkpoint is a STANDARD fused-llama GGUF that
  # `toy infer`'s realize_for_mmap loads unchanged. from-scratch realizes
  # with untied=true (train.rb:269) and donor_d_in=DONOR_D>0, so the embed
  # on-session is a [vocab, donor] donor table + trainable lens.proj; the
  # fold collapses them to a standard [vocab, d_model] token_embd.weight.
  # write_sess + fs_cache.sess must stay alive until write_step returns
  # (gguf_add_tensor reads host ptrs at finalize); both are local/ivar-held
  # to end of this block.
  write_sess = TinyNN.tnn_session_new(0)
  plist = ToyGGUFFuser.build_lens_folded_into_write_session(recipe.fs_cache, write_sess, true)
  rc = ToyGGUFWriter.write_step(cfg, plist, TAO_RUN_DIR + "/weights", rid, STEPS)
  if rc != 0
    puts "checkpoint write failed: rc=" + rc.to_s
  end

  re  = "{\"kind\":\"run_end\""
  re = re + ",\"t\":"           + TinyNN.tnn_events_now_seconds.to_s
  re = re + ",\"ended_at\":\""  + TinyNN.tnn_events_iso8601_now + "\""
  re = re + ",\"reason\":\"completed\""
  re = re + ",\"final_step\":"  + STEPS.to_s
  re = re + ",\"final_loss\":"  + final_loss.to_s
  re = re + ",\"exit_code\":0"
  re = re + "}"
  TinyNN.tnn_events_emit(re)
  TinyNN.tnn_events_close
end
end
