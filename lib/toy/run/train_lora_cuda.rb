# lib/toy/run/train_lora_cuda.rb — Spinel-compiled LoRA CUDA TRAINING runner.
#
# CUDA twin of lib/toy/run/train_lora.rb. Hand-written (NOT mechanically
# mirrored): the recipe inlines backend-coupled TinyNNCuda.* calls, and the
# checkpoint write seam deliberately straddles two backends, so this runner is
# maintained by hand. It is ABSENT from MIRRORABLE in prep/gen_cuda_mirror.rb
# (exactly like train_cuda.rb / train_lora.rb) — `make verify-mirrors` stays
# green, no mirror pair is introduced.
#
# DEDICATED lora-cuda binary (libexec/toy-train-lora-cuda), SEPARATE from
# libexec/toy-train-cuda for a hard Spinel reason, NOT a stylistic one: the
# LoRA recipe drives Toy::LLM::Engine::LlamaSeqEngineCuda#realize_for_mmap (frozen GGUF
# base, mmap'd in place); the from-scratch + warm-start recipes drive
# #realize_for_random_init. When BOTH realize paths are compiled in the same
# Spinel unit, whole-program type inference MERGES the `cfg` receiver type and
# miscompiles the (dead) model methods (landmine #16, the polymorphic-merge
# family — same reason train_lora.rb is split from train.rb). Splitting lora
# into its own binary keeps each binary's realize path monomorphic.
#
# SINGLE-TYPE BINARY (landmine #16): TinyNNCuda is the COMPUTE path. The CPU
# TinyNN module is also defined (the CUDA monolith transitively requires
# transformer -> tinynn) and is used ONLY for the checkpoint write seam
# (ToyDriftGrad.params downloads trainable params via CPU TinyNN — works on
# GB10 UVA) — NOT for compute.
#
# CONTRACT (ENV only): STEPS (default "5"), RANK (default "8"), GGUF
# (default the 135m native gguf), RUN_DIR (events + checkpoint sink),
# TOY_RUN_ID (run metadata). The gate-fixed lora knobs (LR=0.001,
# TARGET_ID=99, TOKENS, seed=42, init_scale=0.01, hp) are HARDCODED here,
# byte-mirroring train_lora.rb (the CPU gate ground truth). NO SEED env.
#
# CONFIG NOTE: we deliberately do NOT `require toy_smollm2_loader` (it
# transitively pulls gpt2.rb, whose ivars share an `iv_cfg` with
# Toy::SmolLM2 — Spinel then merges those receiver types; the same
# landmine-#16 family). Instead the smollm2-135m config + flags are HARDCODED
# literals, bit-identical to train_lora.rb: vocab=49152 d=576 heads=9 n_kv=3
# d_ff=1536 L=30 ctx=8192 rope_base=1e5 rms_eps=1e-5, untied=false
# qkv_bias=false.
#
# DETERMINISM: the printed "step N: loss=" curve is EMPIRICALLY byte-
# deterministic run-to-run on THIS GB10 (sm_121, CUDA 13.0). NOT contractual —
# ggml-cuda float atomicAdd accumulation order is not fixed across GPUs /
# drivers / CUDA-toolkit versions / after a backend rebuild. If run on
# different hardware OR after rebuilding tinynn/libtinynn_ggml_cuda.a, the
# baseline (prep/fixtures/train_lora_cuda_baseline.txt) MUST be re-pinned. The
# CUDA curve also differs from the CPU curve (F32 vs f64 accumulation) — the
# gate compares against its OWN cuda fixture, NOT train_lora_baseline.txt.
#
# Spinel hygiene: hand-built String-concat JSON (no #{} interpolation, no
# Math.exp); no Struct.new. Float#** in the per-step bias correction is the
# CPU runner's exact operator, reproduced verbatim for bit-equality. Events +
# checkpoint go to FILE only — the ONLY stdout is the byte-gated
# "step N: loss=" line.

require_relative "../../toy"
require_relative "../io/json_builder"
require_relative "../dev/toy_describe_flow"
require_relative "../io/toy_events"
require_relative "../models/toy_smollm2"
require_relative "../llm/engine/llama_seq_engine_cuda"
require_relative "../llm/recipes/lora_cuda"
require_relative "../llm/adamw"
require_relative "../train/toy_gguf_writer"
require_relative "../train/toy_drift_grad"

STEPS       = (ENV["STEPS"] || "5").to_i
# The run-directory contract. TOY_RUN_DIR is canonical; TAO_RUN_DIR is
# the compatibility fallback — the framework's own contract should not
# be named after a client repo. Length-checked, not truthiness-checked:
# "" is truthy in Ruby.
RUN_DIR_NEW = ENV["TOY_RUN_DIR"] || ""
RUN_DIR     = RUN_DIR_NEW.length > 0 ? RUN_DIR_NEW : (ENV["TAO_RUN_DIR"] || "")
RUN_ID      = ENV["TOY_RUN_ID"] || ""

GGUF      = ENV["GGUF"] || "data/smollm2-135m-native.gguf"
RANK_LORA = (ENV["RANK"] || "8").to_i
TARGET_ID = 99
TOKENS    = [12092, 4845, 253, 1429]

if !File.exist?(GGUF)
  puts "train_lora_cuda: cannot find " + GGUF
  exit 1
end

# Hardcoded smollm2-135m config (see CONFIG NOTE): bit-identical to
# train_lora.rb, but with NO gpt2 require (avoids the iv_cfg type-merge
# miscompile).
cfg_lora = Toy::SmolLM2Config.gqa(49152, 576, 9, 3, 1536, 30,
                                  8192, 100000.0, 1.0e-5)
lora_untied   = false
lora_qkv_bias = false

gguf_h      = TinyNNCuda.tnn_gguf_load(GGUF)
recipe_lora = Toy::LLM::Recipes::LoRACuda.new
# Named realize options (toy#64): same values as the historical
# positional call (TOKENS.length, untied, qkv_bias, 42, 0.01); RANK
# stays a leading positional (lora-specific).
opts_lora = Toy::LLM::RecipeOptions.new
opts_lora.t_seq      = TOKENS.length
opts_lora.untied     = lora_untied
opts_lora.qkv_bias   = lora_qkv_bias
opts_lora.seed       = 42
opts_lora.init_scale = 0.01
recipe_lora.realize!(gguf_h, cfg_lora, RANK_LORA, opts_lora)
ToyDescribeFlow.emit_flow_json(RUN_DIR, recipe_lora.lora_cache.sess)

# Vocab × T one-hot labels: every position targets TARGET_ID.
m_labels = Mat.new(TOKENS.length, cfg_lora.vocab)
i = 0
while i < TOKENS.length * cfg_lora.vocab; m_labels.flat[i] = 0.0; i = i + 1; end
ti = 0
while ti < TOKENS.length
  m_labels.flat[ti * cfg_lora.vocab + TARGET_ID] = 1.0
  ti = ti + 1
end

# NAMED AdamW (byte-identical twin of the CPU lora runner). lora differs
# from the from-scratch defaults: beta2=0.999 and bias_correct=true, so
# slots 5/6 carry the PER-STEP bias-correction denominators 1/(1-beta^t)
# — NOT constant betas (see the loud finding in lib/toy/llm/adamw.rb: the
# lora FFI graph interprets slots 5/6 DIFFERENTLY from from-scratch/warm/
# vit). m_hp is rebuilt per step below.
adamw_lora = Toy::AdamW.for_lora   # beta2=0.999 + per-step bias correction

positions = [0, 1, 2, 3]

# --- Events (FILE only when RUN_DIR set). ---
EVENTS = RUN_DIR.length > 0 ? (RUN_DIR + "/events.jsonl") : ""


if EVENTS.length > 0
  rc = TinyNNCuda.tnn_events_open(EVENTS)
  if rc == 0
    rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
    rs = Toy::Json::Builder.new
    rs.add_str("kind", "run_start")
    rs.add_str("schema", "toy/v1")
    rs.add_num("t", TinyNNCuda.tnn_events_now_seconds)
    rs.add_str("started_at", TinyNNCuda.tnn_events_iso8601_now)
    rs.add_str("run_id", rid)
    rs.add_str("phase", "train")
    Toy::Events.add_provenance(rs,
      TinyNNCuda.tnn_provenance_host_name, TinyNNCuda.tnn_provenance_host_os,
      TinyNNCuda.tnn_provenance_host_arch,
      TinyNNCuda.tnn_backend_name(recipe_lora.lora_cache.sess))
    model = Toy::Json::Builder.new
    model.add_str("arch", "llama")
    model.add_str("name", "smollm2-135m")
    model.add_num("vocab",    cfg_lora.vocab)
    model.add_num("d_model",  cfg_lora.d_model)
    model.add_num("n_layers", cfg_lora.n_layers)
    model.add_num("n_heads",  cfg_lora.n_heads)
    model.add_num("n_kv",     cfg_lora.n_kv)
    model.add_num("d_head",   cfg_lora.head_dim)
    model.add_num("d_ff",     cfg_lora.d_ff)
    rs.add_obj("model", model)
    config = Toy::Json::Builder.new
    config.add_num("rank",    RANK_LORA)
    config.add_num("steps",   STEPS)
    config.add_raw("lr",      "0.001")
    config.add_raw("seed",    "42")
    config.add_num("context", TOKENS.length)
    rs.add_obj("config", config)
    TinyNNCuda.tnn_events_emit(rs.dump)
  else
    puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
  end
end

# --- Training loop (0-indexed like every other recipe loop; hp_for_step
# applies the lora 1-indexed bias-correction convention internally —
# hp_for_step(k) == hp(k + 1) byte-for-byte, so the gated stdout is
# unchanged; toy#73 A.3 seed d retired the last raw-hp(step) callers). ---
final_loss = 0.0
step = 0
while step < STEPS
  step_wall_start = TinyNNCuda.tnn_events_now_seconds
  # bias_correct=true → slots 5/6 = 1/(1-0.9^t), 1/(1-0.999^t) at
  # t = step + 1. Byte-identical to the historical inline `** t.to_f`.
  m_hp = adamw_lora.hp_for_step(step)
  loss = recipe_lora.step!(TOKENS, positions, m_labels, m_hp, step == 0)
  final_loss = loss
  # The byte-gated line — to STDOUT (1-indexed, as recorded).
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s

  if EVENTS.length > 0
    step_wall_us = ((TinyNNCuda.tnn_events_now_seconds - step_wall_start) * 1.0e6).to_i
    es = Toy::Json::Builder.new
    es.add_str("kind",  "step")
    es.add_str("phase", "train")
    es.add_num("t",       TinyNNCuda.tnn_events_now_seconds)
    es.add_num("step",    step + 1)
    es.add_num("loss",    loss)
    es.add_raw("lr",      "0.001")
    es.add_num("tokens",  TOKENS.length)
    es.add_num("wall_us", step_wall_us)
    TinyNNCuda.tnn_events_emit(es.dump)
  end
  step = step + 1
end

# --- Final checkpoint + run_end (FILE only). THE CROSS-BACKEND SEAM: params
# are downloaded from the CUDA training session via CPU TinyNN (ToyDriftGrad.
# params, works on GB10 UVA) and written by the CPU writer. Compute above
# stayed on CUDA. NO lens-fold/fuser (lora uses ToyDriftGrad.params). ---
if EVENTS.length > 0 && TinyNNCuda.tnn_events_active == 1
  rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
  plist = ToyDriftGrad.params(recipe_lora.lora_cache.sess)
  rc = ToyGGUFWriter.write_step(cfg_lora, plist, RUN_DIR + "/weights", rid, STEPS)
  if rc != 0
    puts "checkpoint write failed: rc=" + rc.to_s
  end

  re = Toy::Json::Builder.new
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
