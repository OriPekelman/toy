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
require_relative "../../toy_smollm2"
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
recipe_lora.realize!(gguf_h, cfg_lora, TOKENS.length, lora_untied,
                     lora_qkv_bias, RANK_LORA, 42, 0.01)

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
adamw_lora = Toy::AdamW.new
adamw_lora.beta2 = 0.999
adamw_lora.bias_correct = true

positions = [0, 1, 2, 3]

# --- Events (FILE only when TAO_RUN_DIR set). ---
EVENTS = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/events.jsonl") : ""

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
    rs = rs + ",\"backend\":{\"kind\":\"" + TinyNN.tnn_backend_name(recipe_lora.lora_cache.sess) + "\"}"
    rs = rs + ",\"git\":{\"sha\":\""     + git_sha    + "\""
    rs = rs + ",\"branch\":\""           + git_branch + "\"}"
    rs = rs + ",\"model\":{\"arch\":\"llama\""
    rs = rs + ",\"name\":\"smollm2-135m\""
    rs = rs + ",\"vocab\":"    + cfg_lora.vocab.to_s
    rs = rs + ",\"d_model\":"  + cfg_lora.d_model.to_s
    rs = rs + ",\"n_layers\":" + cfg_lora.n_layers.to_s
    rs = rs + ",\"n_heads\":"  + cfg_lora.n_heads.to_s
    rs = rs + ",\"n_kv\":"     + cfg_lora.n_kv.to_s
    rs = rs + ",\"d_head\":"   + cfg_lora.head_dim.to_s
    rs = rs + ",\"d_ff\":"     + cfg_lora.d_ff.to_s
    rs = rs + "}"
    rs = rs + ",\"config\":{\"rank\":" + RANK_LORA.to_s
    rs = rs + ",\"steps\":" + STEPS.to_s
    rs = rs + ",\"lr\":0.001"
    rs = rs + ",\"seed\":42"
    rs = rs + ",\"context\":" + TOKENS.length.to_s
    rs = rs + "}"
    rs = rs + "}"
    TinyNN.tnn_events_emit(rs)
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
    es  = "{\"kind\":\"step\",\"phase\":\"train\""
    es = es + ",\"t\":"        + TinyNN.tnn_events_now_seconds.to_s
    es = es + ",\"step\":"     + step.to_s
    es = es + ",\"loss\":"     + loss.to_s
    es = es + ",\"lr\":0.001"
    es = es + ",\"tokens\":"   + TOKENS.length.to_s
    es = es + ",\"wall_us\":"  + step_wall_us.to_s
    es = es + "}"
    TinyNN.tnn_events_emit(es)
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
