# lib/toy/run/train_lora.rb — Spinel-compiled LoRA TRAINING compute runner.
#
# DEDICATED lora runner, SIBLING of lib/toy/run/train.rb. It exists as a
# SEPARATE binary (libexec/toy-train-lora) for a hard Spinel reason, NOT a
# stylistic one:
#
#   The LoRA recipe drives Toy::LLM::Engine::LlamaSeqEngine#realize_for_mmap (frozen
#   GGUF base, mmap'd in place); the from-scratch + warm-start recipes drive
#   #realize_for_random_init. When BOTH realize paths are CALLED in the same
#   Spinel compilation unit, whole-program type inference MERGES the `cfg`
#   receiver type across the two flows and miscompiles the (otherwise dead)
#   Toy::RMSNorm#forward / Toy::SmolLM2#new/#initialize model methods —
#   `iv_eps`/`iv_cfg` come out typed as sp_RbVal and the C compile fails
#   (landmine #16, the polymorphic-merge family). from-scratch + warm-start
#   coexist cleanly (both random-init); adding the lora mmap path is what
#   breaks it. Splitting lora into its own binary keeps each binary's
#   realize path monomorphic and keeps the from-scratch runner BYTE-IDENTICAL
#   (so its existing gate holds trivially). The CLI dispatches `toy train
#   lora` to this binary and the other two recipes to libexec/toy-train.
#
# Like libexec/toy-train this is CPU only and deliberately ABSENT from
# MIRRORABLE in prep/gen_cuda_mirror.rb (no mirror pair introduced).
#
# CONTRACT (ENV only): STEPS (default "5"), RANK (default "8"), GGUF
# (default the 135m native gguf), TAO_RUN_DIR (events + checkpoint sink),
# TOY_RUN_ID (run metadata). The gate-fixed lora knobs (LR=0.001,
# TARGET_ID=99, TOKENS, seed=42, init_scale=0.01, hp) are HARDCODED here,
# byte-mirroring examples/smoke_recipe_lora.rb (the gate ground truth).
#
# CONFIG NOTE: we deliberately do NOT `require toy_smollm2_loader` (it
# transitively pulls gpt2.rb, whose CausalSelfAttention/FFN share an
# `iv_cfg` ivar with Toy::SmolLM2 — Spinel then merges those receiver types
# and miscompiles Toy::SmolLM2#algorithm; the same landmine-#16 family).
# Instead the smollm2-135m config + flags are HARDCODED literals, verified
# bit-identical to SmolLM2ConfigLoader.read / detect_smollm2_flags for
# data/smollm2-135m-native.gguf: vocab=49152 d=576 heads=9 n_kv=3 d_ff=1536
# L=30 ctx=8192 rope_base=1e5 rms_eps=1e-5 (head_dim defaults to 576/9=64,
# rope_scaling defaults to none), untied=false qkv_bias=false.
#
# Spinel hygiene: hand-built String-concat JSON (no #{} interpolation, no
# Math.exp); no Struct.new. Float#** in the per-step bias correction is the
# fixture's exact operator, reproduced verbatim for bit-equality. Events +
# checkpoint go to FILE only — the ONLY stdout is the byte-gated
# "step N: loss=" line.

require_relative "../../toy"
require_relative "../../../vendor/spinel/spinel_kit/lib/spinel_kit/json_builder"
require_relative "../dev/toy_describe_flow"
require_relative "../io/toy_events"
require_relative "../models/toy_smollm2"
require_relative "../llm/engine/llama_seq_engine"
require_relative "../llm/recipes/lora"
require_relative "../llm/adamw"
require_relative "../train/toy_gguf_writer"
require_relative "../train/toy_drift_grad"

STEPS       = (ENV["STEPS"] || "5").to_i
TAO_RUN_DIR = ENV["TAO_RUN_DIR"] || ""
RUN_ID      = ENV["TOY_RUN_ID"] || ""

GGUF      = ENV["GGUF"] || "data/smollm2-135m-native.gguf"
RANK_LORA = (ENV["RANK"] || "8").to_i
TARGET_ID = 99
TOKENS    = [12092, 4845, 253, 1429]

if !File.exist?(GGUF)
  puts "train_lora: cannot find " + GGUF
  exit 1
end

# Hardcoded smollm2-135m config (see CONFIG NOTE): bit-identical to
# SmolLM2ConfigLoader.read for data/smollm2-135m-native.gguf, but with NO
# gpt2 require (avoids the iv_cfg type-merge miscompile).
cfg_lora = Toy::SmolLM2Config.gqa(49152, 576, 9, 3, 1536, 30,
                                  8192, 100000.0, 1.0e-5)
lora_untied   = false
lora_qkv_bias = false

gguf_h      = TinyNN.tnn_gguf_load(GGUF)
recipe_lora = Toy::LLM::Recipes::LoRA.new
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
# tao#flow-json-emit (#25): self-describing run bundle, parallel to events.jsonl.
ToyDescribeFlow.emit_flow_json(TAO_RUN_DIR, recipe_lora.lora_cache.sess)

# Vocab × T one-hot labels: every position targets TARGET_ID.
m_labels = Mat.new(TOKENS.length, cfg_lora.vocab)
i = 0
while i < TOKENS.length * cfg_lora.vocab; m_labels.flat[i] = 0.0; i = i + 1; end
ti = 0
while ti < TOKENS.length
  m_labels.flat[ti * cfg_lora.vocab + TARGET_ID] = 1.0
  ti = ti + 1
end

# NAMED AdamW. lora differs from the from-scratch defaults: beta2=0.999
# and bias_correct=true, so slots 5/6 carry the PER-STEP bias-correction
# denominators 1/(1-beta^t) — NOT constant betas (see the loud finding in
# lib/toy/llm/adamw.rb: the lora FFI graph interprets slots 5/6 DIFFERENTLY
# from the from-scratch/warm/vit graphs). m_hp is rebuilt per step below.
adamw_lora = Toy::AdamW.for_lora   # beta2=0.999 + per-step bias correction

positions = [0, 1, 2, 3]

# --- Events (FILE only when TAO_RUN_DIR set). ---
EVENTS = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/events.jsonl") : ""


if EVENTS.length > 0
  rc = TinyNN.tnn_events_open(EVENTS)
  if rc == 0
    rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
    rs = SpinelKit::Json::Builder.new
    rs.add_str("kind", "run_start")
    rs.add_str("schema", "toy/v1")
    rs.add_num("t", TinyNN.tnn_events_now_seconds)
    rs.add_str("started_at", TinyNN.tnn_events_iso8601_now)
    rs.add_str("run_id", rid)
    rs.add_str("phase", "train")
    Toy::Events.add_provenance(rs,
      TinyNN.tnn_provenance_host_name, TinyNN.tnn_provenance_host_os,
      TinyNN.tnn_provenance_host_arch,
      TinyNN.tnn_backend_name(recipe_lora.lora_cache.sess))
    model = SpinelKit::Json::Builder.new
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
    config = SpinelKit::Json::Builder.new
    config.add_num("rank",    RANK_LORA)
    config.add_num("steps",   STEPS)
    config.add_raw("lr",      "0.001")
    config.add_raw("seed",    "42")
    config.add_num("context", TOKENS.length)
    rs.add_obj("config", config)
    TinyNN.tnn_events_emit(rs.dump)
  else
    puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
  end
end

# --- Training loop (1-indexed; per-step bias-corrected hp). ---
final_loss = 0.0
step = 1
while step <= STEPS
  step_wall_start = TinyNN.tnn_events_now_seconds
  # 1-indexed step; bias_correct=true → slots 5/6 = 1/(1-0.9^t),
  # 1/(1-0.999^t). Byte-identical to the historical inline `** step.to_f`.
  m_hp = adamw_lora.hp(step)
  loss = recipe_lora.step!(TOKENS, positions, m_labels, m_hp, step == 1)
  final_loss = loss
  # The byte-gated line — to STDOUT.
  puts "step " + step.to_s + ": loss=" + loss.to_s

  if EVENTS.length > 0
    step_wall_us = ((TinyNN.tnn_events_now_seconds - step_wall_start) * 1.0e6).to_i
    es = SpinelKit::Json::Builder.new
    es.add_str("kind",  "step")
    es.add_str("phase", "train")
    es.add_num("t",       TinyNN.tnn_events_now_seconds)
    es.add_num("step",    step)
    es.add_num("loss",    loss)
    es.add_raw("lr",      "0.001")
    es.add_num("tokens",  TOKENS.length)
    es.add_num("wall_us", step_wall_us)
    TinyNN.tnn_events_emit(es.dump)
  end
  step = step + 1
end

# --- Final checkpoint + run_end (FILE only). ---
if EVENTS.length > 0 && TinyNN.tnn_events_active == 1
  rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
  plist = ToyDriftGrad.params(recipe_lora.lora_cache.sess)
  rc = ToyGGUFWriter.write_step(cfg_lora, plist, TAO_RUN_DIR + "/weights", rid, STEPS)
  if rc != 0
    puts "checkpoint write failed: rc=" + rc.to_s
  end

  re = SpinelKit::Json::Builder.new
  re.add_str("kind", "run_end")
  re.add_num("t",          TinyNN.tnn_events_now_seconds)
  re.add_str("ended_at",   TinyNN.tnn_events_iso8601_now)
  re.add_str("reason",     "completed")
  re.add_num("final_step", STEPS)
  re.add_num("final_loss", final_loss)
  re.add_raw("exit_code",  "0")
  TinyNN.tnn_events_emit(re.dump)
  TinyNN.tnn_events_close
end
