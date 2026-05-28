# Train a tiny Llama-shape model from scratch on TinyStories via the
# FFI training graph (forward + backward + AdamW in one ggml call).
# Llama-arch (RMSNorm + GQA + RoPE + SwiGLU); the GPT-2-style
# example_train (lib/transformer.rb) stays as the pure-Ruby teaching
# path.
#
#   make example_train_from_scratch
#   ./examples/example_train_from_scratch
#   D_MODEL=128 N_LAYERS=4 STEPS=200 ./examples/example_train_from_scratch
#
# Implementation notes:
#   - Reads sequences via File.read + split. The historical Spinel
#     d59926a/568cf0d crash where File.open-with-block segfaulted
#     during realize_for_random_init is FIXED as of Spinel a03bb49
#     (probe at tinynn/probe_file_block_ffi.rb). File.read stays
#     because it's the natural primitive for "the whole file as
#     one string", not because the block form is broken.
#   - VOCAB_SIZE is hardcoded at 627 (TinyStories). Reading
#     data/ts_vocab.txt would pull Array<String> into main scope and
#     poison Spinel's polymorphic Array<T> dispatch table inside
#     Toy_Embedding#lookup / RMSNorm#forward, breaking C codegen.
#     Acceptance: loss decreases monotonically; we don't decode
#     generated tokens.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/llama_seq_forward_ffi_metal"
require_relative "../lib/toy_describe_flow"
require_relative "../lib/toy_drift_grad"
require_relative "../lib/toy_gguf_writer"
require_relative "../lib/toy_tap"
require_relative "../lib/toy_sample"
require_relative "../lib/toy_token_drift"

VOCAB_SIZE = 627
D_MODEL    = (ENV["D_MODEL"]  || "64").to_i
D_FF       = (ENV["D_FF"]     || "128").to_i
N_HEADS    = (ENV["N_HEADS"]  || "4").to_i
N_LAYERS   = (ENV["N_LAYERS"] || "2").to_i
CONTEXT    = (ENV["CONTEXT"]  || "32").to_i
STEPS      = (ENV["STEPS"]    || "60").to_i
LR         = (ENV["LR"]       || "0.001").to_f
SEED       = (ENV["SEED"]     || "42").to_i
# GH#7 — micro-batching. BATCH=1 is the default and keeps the graph
# bit-identical to pre-GH#7 (diag_mask_inf + softmax path; flat T-long
# token + position arrays). BATCH>1 lays B sequences side-by-side as
# a flat [T*B] vector with a block-causal mask uploaded at realize.
BATCH      = (ENV["BATCH"]    || "1").to_i
# GH#9 — mixed-precision compute. 0=F32 (default; bit-identical to
# pre-GH#9), 1=F16, 30=BF16. When non-zero, weight matmuls inside the
# transformer block route through mp_matmul which casts the F32
# master to the chosen dtype inline in the forward graph (Tensor
# Core path on supporting hardware). opt_step still operates on the
# F32 master — the cast is recomputed each forward, no sidecar.
WEIGHT_DTYPE = (ENV["WEIGHT_DTYPE"] || "0").to_i
# GH#8 — gradient accumulation. Effective batch = BATCH × GRAD_ACCUM
# without the memory cost of a single big batch. Implementation note:
# ggml's opt_step_adamw is baked into the backward graph and runs
# every compute_backward call; the graph has no "skip this op"
# primitive. So instead of literally accumulating grads over N-1
# passes and firing opt_step on the Nth, we LR-scale: opt_step fires
# every micro-batch with lr = LR / GRAD_ACCUM. The cumulative weight
# movement over N micro-batches matches a single full-lr opt_step on
# mean(g_1..g_N) — but Adam's m/v state evolves per-micro-step rather
# than once per cycle. For typical (beta1=0.9, GRAD_ACCUM<=8) this is
# a small numerical divergence from "true" gradient accumulation.
# Filed as a known approximation; a true-accum follow-up would
# require either a vendor patch to opt_step_adamw (8th hp slot
# gate) or a two-graph approach.
GRAD_ACCUM = (ENV["GRAD_ACCUM"] || "1").to_i

# Events stream (v1, docs/events-schema.md). Two env knobs:
#   TAO_RUN_DIR — preferred. Tao's per-run directory; we append
#                 /events.jsonl. Used by tao watch / tao report.
#   TOY_EVENTS  — legacy/raw. Full path to the events.jsonl file.
# When neither is set, no events are emitted (cheap-when-off).
TAO_RUN_DIR = ENV["TAO_RUN_DIR"] || ""
TOY_EVENTS  = ENV["TOY_EVENTS"]  || ""
EVENTS      = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/events.jsonl") : TOY_EVENTS
RUN_ID      = ENV["TOY_RUN_ID"] || ""

# tao#drift-grad-sentinels — opt-in observability for tao compare.
# When TOY_DRIFT_EVERY=N (N>0), emit a drift event per param every N
# steps (snapshot taken at step 0). When TOY_GRAD_SENTINELS=1, emit a
# grad event per param per step. Both are cheap-when-off.
DRIFT_EVERY = (ENV["TOY_DRIFT_EVERY"] || "0").to_i
GRAD_SENT   = (ENV["TOY_GRAD_SENTINELS"] || "0") == "1"
# toy#20 — per-vocab-row drift on the embedding table. TOY_TOKEN_DRIFT=N
# emits one `drift` event per token per N macro-steps with cos_to_init,
# l2_to_init, freq. Tao renders this as the freq↔drift figure (the
# granite_transfer Pearson r = -0.835 headline). At vocab=627 each tick
# emits 627 events (fine). Snapshot taken at step 0; corpus frequency
# is a one-time histogram. Cheap-when-off.
TOKEN_DRIFT_EVERY = (ENV["TOY_TOKEN_DRIFT"] || "0").to_i

# tao#gguf-checkpoint-writer. CHECKPOINT_EVERY=N → write a snapshot
# at step N, 2N, 3N, …, plus a final snapshot at run_end. Lands in
# $TAO_RUN_DIR/weights/step_<N>.gguf with a `latest` symlink to the
# newest. Cheap-when-off (no schedule unless N > 0 AND a run-dir is
# configured).
CHECKPOINT_EVERY = (ENV["CHECKPOINT_EVERY"] || "0").to_i
WEIGHTS_DIR      = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/weights") : ""

# toy#21 — sample event emission at run end. TOY_SAMPLES=N decodes N
# completions from the first-K tokens of training sequences as prompts
# and emits one `sample` event per generation (toy/v1 schema). Greedy
# decode via tnn_compute on the trained forward graph (which produces
# logits without firing opt_step — that lives in graph_b). Cheap-when-
# off (N=0 disables; the toy_sample require itself is free).
TOY_SAMPLES = (ENV["TOY_SAMPLES"] || "0").to_i
# Sample prompts: each prompt is the first SAMPLE_PROMPT_LEN tokens of
# training sequence i (cycled if fewer lines than samples). N_NEW is
# how many tokens to generate per sample.
SAMPLE_PROMPT_LEN = (ENV["SAMPLE_PROMPT_LEN"] || "4").to_i
SAMPLE_N_NEW      = (ENV["SAMPLE_N_NEW"]      || "12").to_i

# tao#kv-tap-surface — TOY_TAP=1 enables a demonstration tap on the
# loss scalar after each compute_backward. Real production taps belong
# inside the cache's graph builder (e.g. tapping t_q_rotated inside
# build_seq_qhead with region="attn_q_post_rope"), but the loss tap is
# enough to demonstrate the wire and validate the event shape against
# the schema.
TAP_ENABLED = (ENV["TOY_TAP"] || "0") == "1"

# GH#15 — TOY_CKA=N emits Activation-Gram taps (T×T Gram per region,
# per layer) every N steps. Three regions: attn_norm, ffn_out,
# resid_post. Cheap-when-off; the graph builder always set_outputs the
# tap tensors, but the emit_cka call is the costly part (T×T×d ops +
# JSON serialization). N=0 disables; N=1 emits every step.
CKA_EVERY = (ENV["TOY_CKA"] || "0").to_i

# tao#describe-flow-clean-stdout — when TOY_DESCRIBE=json the stdout
# must be a single valid JSON document so `… | jq .` works without a
# preamble strip. Suppress the human-facing config/realize lines for
# the JSON case. text/mermaid keep their preamble (human-facing).
# Spinel has no STDERR — there's nothing to redirect *to* — so we
# skip the prints entirely.
DESCRIBE = ENV["TOY_DESCRIBE"] || ""
DESCRIBE_QUIET = (DESCRIBE == "json")

cfg = Toy::SmolLM2Config.new(VOCAB_SIZE, D_MODEL, N_HEADS, N_HEADS,
                              D_FF, N_LAYERS, CONTEXT, 10000.0, 1.0e-5)
if !DESCRIBE_QUIET
  puts "config: vocab=" + cfg.vocab.to_s +
       " d=" + cfg.d_model.to_s +
       " heads=" + cfg.n_heads.to_s +
       " d_ff=" + cfg.d_ff.to_s +
       " L=" + cfg.n_layers.to_s +
       " head_dim=" + cfg.head_dim.to_s
end

t_realize = Time.now
fcache = LlamaSeqForwardFFICacheMetal.new
fcache.realize_for_random_init(cfg, CONTEXT, BATCH, WEIGHT_DTYPE, false, false, SEED, 1.0)
if !DESCRIBE_QUIET
  puts "realize_for_random_init: " + ((Time.now - t_realize).to_f * 1000.0).to_s + " ms"
end

# tao#kv-describe-flow: opt-in DAG dump. Three formats; consumer
# picks via env (text | json | mermaid). Exits after printing so
# downstream tools can capture the structured form alone.
if DESCRIBE.length > 0
  if DESCRIBE == "json"
    puts ToyDescribeFlow.json(fcache.sess)
  elsif DESCRIBE == "mermaid"
    puts ToyDescribeFlow.mermaid(fcache.sess)
  else
    puts ToyDescribeFlow.text(fcache.sess)
  end
  exit 0
end

# tao#flow-json-emit — when TAO_RUN_DIR is set, drop a flow.json
# alongside events.jsonl so the same run bundle is self-describing.
# Cheap: one graph walk at startup; Tao's report/describe pipeline
# reads `<run_dir>/flow.json` directly. Removes the need for a
# separate TOY_DESCRIBE pre-pass.
if TAO_RUN_DIR.length > 0
  flow_path = TAO_RUN_DIR + "/flow.json"
  File.open(flow_path, "w") do |f|
    f.write(ToyDescribeFlow.json(fcache.sess))
  end
end

# Read BATCH TinyStories sequences (flat Array<Integer> of length
# CONTEXT*BATCH). BATCH=1 is the legacy single-line path; BATCH>1
# pulls the first BATCH lines (cycling if the file is shorter than
# BATCH). Each line is padded with 0 to CONTEXT.
raw      = File.read("data/ts_seqs.txt")
lines    = raw.split("\n")
seq_ids  = [0]; seq_ids.pop
bi = 0
while bi < BATCH
  line  = lines[bi % lines.length]
  parts = line.split(" ")
  k = 0
  while k < CONTEXT
    if k < parts.length
      seq_ids.push(parts[k].to_i)
    else
      seq_ids.push(0)
    end
    k = k + 1
  end
  bi = bi + 1
end

result   = fcache.build_training_step
t_loss   = result[0]
t_labels = result[1]
t_hp     = result[2]

# Shift-by-one next-token targets per batch element, one-hot rows.
# At the last position within a batch (ti == CONTEXT-1) target = self,
# same fall-back the legacy single-sequence path used.
m_labels = Mat.new(CONTEXT * BATCH, VOCAB_SIZE)
li = 0
while li < CONTEXT * BATCH * VOCAB_SIZE; m_labels.flat[li] = 0.0; li = li + 1; end
b_lbl = 0
while b_lbl < BATCH
  ti = 0
  while ti < CONTEXT
    flat_q = b_lbl * CONTEXT + ti
    next_ti = ti + 1 < CONTEXT ? ti + 1 : ti
    tgt = seq_ids[b_lbl * CONTEXT + next_ti]
    m_labels.flat[flat_q * VOCAB_SIZE + tgt] = 1.0
    ti = ti + 1
  end
  b_lbl = b_lbl + 1
end

# GH#8 — LR-scaled mini-batch: per-micro-step lr is base LR divided
# by GRAD_ACCUM so the cumulative weight movement over N micro-steps
# approximates one full-lr opt_step on the mean grad. At GRAD_ACCUM=1
# this is identity (lr_per_step == LR), preserving pre-GH#8 behaviour.
lr_per_step = LR / GRAD_ACCUM.to_f
m_hp = Mat.new(1, 7)
m_hp.flat[0] = lr_per_step
m_hp.flat[1] = 0.9
m_hp.flat[2] = 0.999
m_hp.flat[3] = 1.0e-8
m_hp.flat[4] = 0.0

# Positions cycle 0..CONTEXT-1 per batch element. RoPE applies the
# correct per-batch positional encoding because rope_ext reads
# positions[k] for each ne[2] slot.
positions = [0]; positions.pop
b_pos = 0
while b_pos < BATCH
  pi = 0
  while pi < CONTEXT
    positions.push(pi)
    pi = pi + 1
  end
  b_pos = b_pos + 1
end

losses = [0.0]; losses.pop

# Read git state for run_start.provenance.git. Pure Ruby; reads from
# the cwd's .git/. If we're run from outside the toy repo, falls back
# to "unknown" without aborting (the field is for diagnostics, not
# load-bearing). Tao's acceptance only requires git.sha to be present.
git_sha    = "unknown"
git_branch = "unknown"
if File.exist?(".git/HEAD")
  head = File.read(".git/HEAD")
  # Strip trailing newline manually — Spinel-friendly.
  if head.length > 0 && head[head.length - 1...head.length] == "\n"
    head = head[0...head.length - 1]
  end
  if head.length > 5 && head[0...5] == "ref: "
    ref_rel = head[5...head.length]              # e.g. "refs/heads/main"
    parts   = ref_rel.split("/")
    if parts.length >= 3
      git_branch = parts[parts.length - 1]
    end
    ref_path = ".git/" + ref_rel
    if File.exist?(ref_path)
      sha = File.read(ref_path)
      if sha.length >= 40
        git_sha = sha[0...40]
      end
    end
  else
    # Detached HEAD: contents IS the SHA.
    if head.length >= 40
      git_sha    = head[0...40]
      git_branch = "HEAD"
    end
  end
end

# Open the events stream (docs/events-schema.md v1). Emit run_start
# with full provenance per tao#run-start-provenance: started_at,
# host, backend, git in addition to model+config. Cheap-when-off:
# the file open and JSON formatting only happen when an env knob asks.
if EVENTS.length > 0
  rc = TinyNNMetal.tnn_events_open(EVENTS)
  if rc != 0
    puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
  else
    puts "events → " + EVENTS
    rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
    rs  = "{\"kind\":\"run_start\",\"schema\":\"toy/v1\""
    rs = rs + ",\"t\":" + TinyNNMetal.tnn_events_now_seconds.to_s
    rs = rs + ",\"started_at\":\"" + TinyNNMetal.tnn_events_iso8601_now + "\""
    rs = rs + ",\"run_id\":\"" + rid + "\""
    rs = rs + ",\"phase\":\"train\""
    rs = rs + ",\"host\":{\"name\":\""   + TinyNNMetal.tnn_provenance_host_name + "\""
    rs = rs + ",\"os\":\""               + TinyNNMetal.tnn_provenance_host_os   + "\""
    rs = rs + ",\"arch\":\""             + TinyNNMetal.tnn_provenance_host_arch + "\"}"
    rs = rs + ",\"backend\":{\"kind\":\"" + TinyNNMetal.tnn_backend_name(fcache.sess) + "\"}"
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
    rs = rs + ",\"batch\":"      + BATCH.to_s
    rs = rs + ",\"grad_accum\":" + GRAD_ACCUM.to_s
    rs = rs + ",\"steps\":" + STEPS.to_s
    rs = rs + ",\"lr\":"    + LR.to_s
    rs = rs + ",\"seed\":"  + SEED.to_s
    rs = rs + "}"
    rs = rs + "}"
    TinyNNMetal.tnn_events_emit(rs)
  end
end

# Enumerate PARAM tensors once if ANY observability or checkpoint
# feature is active. Reused by drift/grad emitters AND the checkpoint
# writer. Mat snapshots (drift_snaps) live in main scope — Spinel
# doesn't generate sp_Mat_ptr_array, so cross-function passes of
# Array<Mat> are off-limits.
drift_params = [TinyNNMetal.tnn_null_ptr]; drift_params.pop
drift_snaps  = [Mat.new(1, 1)]; drift_snaps.pop
ckpt_enabled = WEIGHTS_DIR.length > 0 && CHECKPOINT_EVERY > 0
if (EVENTS.length > 0 && (DRIFT_EVERY > 0 || GRAD_SENT)) || ckpt_enabled
  drift_params = ToyDriftGrad.params(fcache.sess)
  puts "params tracked: " + drift_params.length.to_s
  if DRIFT_EVERY > 0
    di = 0
    while di < drift_params.length
      drift_snaps.push(ToyDriftGrad.snapshot_one(fcache.sess, drift_params[di]))
      di = di + 1
    end
    puts "drift: snapshot at step 0, emitting every " + DRIFT_EVERY.to_s + " steps"
  end
  if GRAD_SENT
    puts "grad: per-step sentinels enabled"
  end
  if ckpt_enabled
    TinyNNMetal.tnn_filesystem_mkdir(WEIGHTS_DIR)
    puts "checkpoints: " + WEIGHTS_DIR + "/step_<N>.gguf every " +
         CHECKPOINT_EVERY.to_s + " steps"
  end
end

# toy#20 — per-token embedding drift + corpus freq. Independent of
# the TOY_DRIFT_EVERY whole-tensor drift schedule (both can run; the
# whole-tensor drift on token_embd is a coarse mean, this is the per-
# row distribution).
token_drift_snap = Mat.new(1, 1)
token_freqs      = [0]; token_freqs.pop
if EVENTS.length > 0 && TOKEN_DRIFT_EVERY > 0
  token_drift_snap = ToyTokenDrift.snapshot(fcache.sess, fcache.t_seq_token_embed)
  token_freqs      = ToyTokenDrift.corpus_freq("data/ts_seqs.txt", VOCAB_SIZE)
  puts "token-drift: snapshot at step 0, emitting per-vocab-row every " +
       TOKEN_DRIFT_EVERY.to_s + " macro-steps (vocab=" + VOCAB_SIZE.to_s + ")"
end

t_start = Time.now
step = 1
final_loss = 0.0
micro_count = 0       # cumulative micro-batch count across all macro-steps
while step <= STEPS
  step_wall_start = TinyNNMetal.tnn_events_now_seconds
  # GH#8 — inner GRAD_ACCUM loop. At GRAD_ACCUM=1 this runs once
  # per macro-step (identical to pre-GH#8). At GA>1, GA micro-
  # batches per macro-step, each firing opt_step with lr/GA. The
  # micro-batch step count drives Adam's bias correction so beta1h/
  # beta2h reflect the actual number of opt_step invocations.
  micro = 1
  while micro <= GRAD_ACCUM
    micro_count = micro_count + 1
    m_hp.flat[5] = 1.0 / (1.0 - (0.9   ** micro_count.to_f))
    m_hp.flat[6] = 1.0 / (1.0 - (0.999 ** micro_count.to_f))
    if micro_count == 1
      TinyNNMetal.tnn_graph_reset(fcache.sess)
    else
      TinyNNMetal.tnn_graph_reset_grads_only(fcache.sess)
    end
    TinyNNMetal.upload_int_array(fcache.sess, fcache.t_seq_token_ids, seq_ids)
    TinyNNMetal.upload_int_array(fcache.sess, fcache.t_seq_positions, positions)
    TinyNNMetal.upload_row_major(fcache.sess, t_labels, m_labels)
    TinyNNMetal.upload_row_major(fcache.sess, t_hp,     m_hp)
    TinyNNMetal.tnn_compute_backward(fcache.sess)
    micro = micro + 1
  end
  # Loss reported per macro-step uses the last micro-batch's loss
  # (model has been updated GRAD_ACCUM times since macro start;
  # this is the "current model loss" on the most recent data).
  TinyNNMetal.tnn_download(fcache.sess, t_loss)
  loss = TinyNNMetal.tnn_scratch_get(fcache.sess, 0)
  losses.push(loss)
  final_loss = loss
  if step <= 5 || step % 10 == 0 || step == STEPS
    puts "step " + step.to_s.rjust(4) + ": CE=" + loss.to_s
  end

  # Optional drift/grad emission. Both run AFTER compute_backward so
  # the grad tensors are live; both before the next graph_reset clears
  # them. Each is opt-in via env (cheap-when-off). The per-param loop
  # lives here (not in the module) so the Mat snapshots stay in main
  # scope — see lib/toy_drift_grad.rb header for the Spinel reason.
  if EVENTS.length > 0
    t_now = TinyNNMetal.tnn_events_now_seconds
    if GRAD_SENT && drift_params.length > 0
      pi = 0
      while pi < drift_params.length
        ToyDriftGrad.emit_grad_event(fcache.sess, drift_params[pi], step, t_now)
        pi = pi + 1
      end
    end
    if DRIFT_EVERY > 0 && drift_params.length > 0 && (step % DRIFT_EVERY == 0)
      pi = 0
      while pi < drift_params.length
        ToyDriftGrad.emit_drift_event(fcache.sess, drift_params[pi],
                                        drift_snaps[pi], step, t_now)
        pi = pi + 1
      end
    end
    # toy#20 — per-vocab-row drift on the embedding table.
    if TOKEN_DRIFT_EVERY > 0 && (step % TOKEN_DRIFT_EVERY == 0)
      ToyTokenDrift.emit_per_token(fcache.sess, fcache.t_seq_token_embed,
                                     token_drift_snap, token_freqs,
                                     VOCAB_SIZE, D_MODEL, step, t_now)
    end
    if TAP_ENABLED
      # Demonstration tap. Real region names come from inside the
      # graph builder; "loss_post_compute" is just the wire test.
      # layer=-1 / head=-1 → null in JSON; n_heads=0 → no per_head_l2.
      ToyTap.emit(fcache.sess, "loss_post_compute", t_loss,
                    -1, -1, step, t_now, 0)
    end
    # GH#15 — per-layer Gram taps. T×T fits in the tap event; Tao's
    # compare CKA consumer is unit-tested and ready.
    if CKA_EVERY > 0 && (step % CKA_EVERY) == 0
      li_tap = 0
      while li_tap < cfg.n_layers
        blk_tap = fcache.seq_blocks_ffi[li_tap]
        ToyTap.emit_cka(fcache.sess, "attn_norm",      blk_tap.tap_attn_norm,
                          li_tap, -1, step, t_now)
        ToyTap.emit_cka(fcache.sess, "ffn_out",        blk_tap.tap_ffn_out,
                          li_tap, -1, step, t_now)
        ToyTap.emit_cka(fcache.sess, "resid_post_block", blk_tap.tap_resid_post,
                          li_tap, -1, step, t_now)
        li_tap = li_tap + 1
      end
    end
  end

  # tao#gguf-checkpoint-writer — write on schedule. Lands AFTER the
  # opt_step has mutated weights for this step. The latest symlink
  # updates atomically (unlink + create — see tnn_filesystem_symlink).
  if ckpt_enabled && (step % CHECKPOINT_EVERY == 0)
    rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
    rc = ToyGGUFWriter.write_step(cfg, drift_params, WEIGHTS_DIR, rid, step)
    if rc != 0
      puts "checkpoint write failed: rc=" + rc.to_s + " (step=" + step.to_s + ")"
    end
  end

  # Per-step event (v1 schema). ppl=exp(loss) deferred to consumer side
  # (Math.exp under Spinel risks poly-dispatch landmines per
  # feedback_spinel_type_inference_landmines).
  if EVENTS.length > 0
    step_wall_us = ((TinyNNMetal.tnn_events_now_seconds - step_wall_start) * 1.0e6).to_i
    es  = "{\"kind\":\"step\",\"phase\":\"train\""
    es = es + ",\"t\":"        + TinyNNMetal.tnn_events_now_seconds.to_s
    es = es + ",\"step\":"     + step.to_s
    es = es + ",\"loss\":"     + loss.to_s
    es = es + ",\"lr\":"       + LR.to_s
    es = es + ",\"tokens\":"   + (CONTEXT * BATCH * GRAD_ACCUM).to_s
    es = es + ",\"wall_us\":"  + step_wall_us.to_s
    es = es + "}"
    TinyNNMetal.tnn_events_emit(es)
  end

  step = step + 1
end
total = (Time.now - t_start).to_f
puts "trained " + STEPS.to_s + " steps in " + total.to_s + "s " +
     "(" + (total / STEPS.to_f * 1000.0).to_s + " ms/step)"
puts "initial loss = " + losses[0].to_s
puts "final   loss = " + losses[losses.length - 1].to_s
ratio = losses[losses.length - 1] / losses[0]
not_learning = ratio >= 0.9
if not_learning
  puts "VERDICT: training NOT learning (final/initial = " + ratio.to_s + ")"
else
  puts "VERDICT: training is learning (final/initial = " + ratio.to_s + ")"
end

# Final checkpoint at run_end (idempotent — if STEPS is a multiple of
# CHECKPOINT_EVERY the in-loop emit already covered it; if not, this
# captures the final state and updates `latest`).
if ckpt_enabled && (STEPS % CHECKPOINT_EVERY != 0)
  rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
  rc = ToyGGUFWriter.write_step(cfg, drift_params, WEIGHTS_DIR, rid, STEPS)
  if rc != 0
    puts "final checkpoint write failed: rc=" + rc.to_s
  end
end

# toy#21 — emit sample generations. tnn_compute runs only graph
# (forward), not graph_b (backward + opt_step), so weights are not
# mutated by these compute calls. Guard on B=1 — the trainer's flat
# [T*B] layout means batched generation would interleave samples; a
# proper B>1 sample path is a follow-up.
if EVENTS.length > 0 && TOY_SAMPLES > 0
  if fcache.seq_b > 1
    puts "TOY_SAMPLES: skipping — only supported at BATCH=1 (current B=" +
         fcache.seq_b.to_s + ")"
  else
    puts "TOY_SAMPLES: decoding " + TOY_SAMPLES.to_s +
         " samples (prompt_len=" + SAMPLE_PROMPT_LEN.to_s +
         ", n_new=" + SAMPLE_N_NEW.to_s + ")"
    n_sample = 0
    while n_sample < TOY_SAMPLES
      prompt_ids = [0]; prompt_ids.pop
      src_line = lines[n_sample % lines.length]
      src_parts = src_line.split(" ")
      pi = 0
      while pi < SAMPLE_PROMPT_LEN && pi < src_parts.length
        prompt_ids.push(src_parts[pi].to_i)
        pi = pi + 1
      end
      decoded = ToySample.greedy_decode(fcache.sess, fcache.t_seq_logits,
                                          fcache.t_seq_token_ids,
                                          fcache.t_seq_positions,
                                          prompt_ids, SAMPLE_N_NEW,
                                          CONTEXT, VOCAB_SIZE)
      prompt_text  = ToySample.detokenize(prompt_ids, "data/ts_vocab.txt")
      decoded_text = ToySample.detokenize(decoded,    "data/ts_vocab.txt")
      ToySample.emit_event(prompt_text, decoded_text, STEPS,
                             TinyNNMetal.tnn_events_now_seconds)
      puts "  sample " + n_sample.to_s + ": " + decoded_text
      n_sample = n_sample + 1
    end
  end
end

# Emit run_end. The run *executed to completion* — we always reach
# this point without an exception — so reason is always "completed".
# The learning-gate verdict is a separate, structured signal (the
# `quality_gate` field). Consumers (e.g. Tao) key on reason for crash
# detection; they key on quality_gate.passed for "is this run useful?".
# CI exit-1-on-failure contract preserved separately below.
# Closes toy#run-end-reason-semantics.
exit_code = not_learning ? 1 : 0
if EVENTS.length > 0 && TinyNNMetal.tnn_events_active == 1
  re  = "{\"kind\":\"run_end\""
  re = re + ",\"t\":"           + TinyNNMetal.tnn_events_now_seconds.to_s
  re = re + ",\"ended_at\":\""  + TinyNNMetal.tnn_events_iso8601_now + "\""
  re = re + ",\"reason\":\"completed\""
  re = re + ",\"final_step\":"  + STEPS.to_s
  re = re + ",\"final_loss\":"  + final_loss.to_s
  re = re + ",\"quality_gate\":{\"passed\":" + (not_learning ? "false" : "true") + ""
  re = re + ",\"metric\":\"loss_ratio\""
  re = re + ",\"value\":" + ratio.to_s
  re = re + ",\"threshold\":0.9}"
  re = re + ",\"exit_code\":"   + exit_code.to_s
  re = re + "}"
  TinyNNMetal.tnn_events_emit(re)
  TinyNNMetal.tnn_events_close
  puts "events closed: " + EVENTS
end

if not_learning
  exit 1
end
