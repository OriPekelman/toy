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
require_relative "../../toy_smollm2"
require_relative "../../llama_seq_forward_ffi"
require_relative "../llm/recipes/from_scratch"
require_relative "../../toy_gguf_writer"
require_relative "../../toy_drift_grad"

STEPS       = (ENV["STEPS"] || "5").to_i
SEED        = (ENV["SEED"]  || "0").to_i
TAO_RUN_DIR = ENV["TAO_RUN_DIR"] || ""
RUN_ID      = ENV["TOY_RUN_ID"] || ""

# Gate-fixed model SHAPE — hardcoded (NOT env/flags), the smoke config.
# vocab=627 d=64 donor=128 heads=4 n_kv=4 ff=128 L=2 ctx=32 rope=1e4 eps=1e-5.
VOCAB    = 627
D_MODEL  = 64
DONOR_D  = 128
N_HEADS  = 4
D_FF     = 128
N_LAYERS = 2
CONTEXT  = 32

cfg = Toy::SmolLM2Config.new(VOCAB, D_MODEL, N_HEADS, N_HEADS,
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
m_labels = Mat.new(CONTEXT, VOCAB)
j = 0; while j < CONTEXT * VOCAB; m_labels.flat[j] = 0.0; j = j + 1; end
k = 0
while k < CONTEXT
  target = (k + 1 < CONTEXT) ? seq_ids[k + 1] : seq_ids[k]
  m_labels.flat[k * VOCAB + target] = 1.0
  k = k + 1
end

# CONSTANT hyper-params (NOT 06's per-step bias-corrected hp; b2=0.95 here,
# NOT 0.999). Using 06's hp breaks the byte gate.
m_hp = Mat.new(1, 7)
m_hp.flat[0] = 0.001; m_hp.flat[1] = 0.9; m_hp.flat[2] = 0.95
m_hp.flat[3] = 1.0e-8; m_hp.flat[4] = 0.0
m_hp.flat[5] = 0.9; m_hp.flat[6] = 0.95

# --- Events (only when TAO_RUN_DIR set; cheap-when-off; FILE only). ---
EVENTS = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/events.jsonl") : ""

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
  plist = ToyDriftGrad.params(recipe.fs_cache.sess)
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
