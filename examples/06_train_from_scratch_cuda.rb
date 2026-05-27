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
#   - Reads sequences via File.read + split. The File.open-with-block
#     form interacts badly with FFI session-init under Spinel d59926a
#     on gx10 (segfaults during realize_for_random_init). File.read
#     sidesteps it.
#   - VOCAB_SIZE is hardcoded at 627 (TinyStories). Reading
#     data/ts_vocab.txt would pull Array<String> into main scope and
#     poison Spinel's polymorphic Array<T> dispatch table inside
#     Toy_Embedding#lookup / RMSNorm#forward, breaking C codegen.
#     Acceptance: loss decreases monotonically; we don't decode
#     generated tokens.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/llama_seq_forward_ffi_cuda"
require_relative "../lib/toy_describe_flow"
require_relative "../lib/toy_drift_grad"
require_relative "../lib/toy_gguf_writer"
require_relative "../lib/toy_tap"

VOCAB_SIZE = 627
D_MODEL    = (ENV["D_MODEL"]  || "64").to_i
D_FF       = (ENV["D_FF"]     || "128").to_i
N_HEADS    = (ENV["N_HEADS"]  || "4").to_i
N_LAYERS   = (ENV["N_LAYERS"] || "2").to_i
CONTEXT    = (ENV["CONTEXT"]  || "32").to_i
STEPS      = (ENV["STEPS"]    || "60").to_i
LR         = (ENV["LR"]       || "0.001").to_f
SEED       = (ENV["SEED"]     || "42").to_i

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

# tao#gguf-checkpoint-writer. CHECKPOINT_EVERY=N → write a snapshot
# at step N, 2N, 3N, …, plus a final snapshot at run_end. Lands in
# $TAO_RUN_DIR/weights/step_<N>.gguf with a `latest` symlink to the
# newest. Cheap-when-off (no schedule unless N > 0 AND a run-dir is
# configured).
CHECKPOINT_EVERY = (ENV["CHECKPOINT_EVERY"] || "0").to_i
WEIGHTS_DIR      = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/weights") : ""

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

cfg = Toy::SmolLM2Config.new(VOCAB_SIZE, D_MODEL, N_HEADS, N_HEADS,
                              D_FF, N_LAYERS, CONTEXT, 10000.0, 1.0e-5)
puts "config: vocab=" + cfg.vocab.to_s +
     " d=" + cfg.d_model.to_s +
     " heads=" + cfg.n_heads.to_s +
     " d_ff=" + cfg.d_ff.to_s +
     " L=" + cfg.n_layers.to_s +
     " head_dim=" + cfg.head_dim.to_s

t_realize = Time.now
fcache = LlamaSeqForwardFFICacheCuda.new
fcache.realize_for_random_init(cfg, CONTEXT, false, false, SEED, 1.0)
puts "realize_for_random_init: " + ((Time.now - t_realize).to_f * 1000.0).to_s + " ms"

# tao#kv-describe-flow: opt-in DAG dump. Three formats; consumer
# picks via env (text | json | mermaid). Exits after printing so
# downstream tools can capture the structured form alone.
DESCRIBE = ENV["TOY_DESCRIBE"] || ""
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

# Read first TinyStories sequence (Array<Integer>).
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

result   = fcache.build_training_step
t_loss   = result[0]
t_labels = result[1]
t_hp     = result[2]

# Shift-by-one next-token targets, one-hot rows.
m_labels = Mat.new(CONTEXT, VOCAB_SIZE)
li = 0
while li < CONTEXT * VOCAB_SIZE; m_labels.flat[li] = 0.0; li = li + 1; end
ti = 0
while ti < CONTEXT
  tgt = ti + 1 < CONTEXT ? seq_ids[ti + 1] : seq_ids[ti]
  m_labels.flat[ti * VOCAB_SIZE + tgt] = 1.0
  ti = ti + 1
end

m_hp = Mat.new(1, 7)
m_hp.flat[0] = LR
m_hp.flat[1] = 0.9
m_hp.flat[2] = 0.999
m_hp.flat[3] = 1.0e-8
m_hp.flat[4] = 0.0

positions = [0]; positions.pop
pi = 0
while pi < CONTEXT; positions.push(pi); pi = pi + 1; end

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
  rc = TinyNNCuda.tnn_events_open(EVENTS)
  if rc != 0
    puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
  else
    puts "events → " + EVENTS
    rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
    rs  = "{\"kind\":\"run_start\",\"schema\":\"toy/v1\""
    rs = rs + ",\"t\":" + TinyNNCuda.tnn_events_now_seconds.to_s
    rs = rs + ",\"started_at\":\"" + TinyNNCuda.tnn_events_iso8601_now + "\""
    rs = rs + ",\"run_id\":\"" + rid + "\""
    rs = rs + ",\"phase\":\"train\""
    rs = rs + ",\"host\":{\"name\":\""   + TinyNNCuda.tnn_provenance_host_name + "\""
    rs = rs + ",\"os\":\""               + TinyNNCuda.tnn_provenance_host_os   + "\""
    rs = rs + ",\"arch\":\""             + TinyNNCuda.tnn_provenance_host_arch + "\"}"
    rs = rs + ",\"backend\":{\"kind\":\"" + TinyNNCuda.tnn_backend_name(fcache.sess) + "\"}"
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
    rs = rs + ",\"lr\":"    + LR.to_s
    rs = rs + ",\"seed\":"  + SEED.to_s
    rs = rs + "}"
    rs = rs + "}"
    TinyNNCuda.tnn_events_emit(rs)
  end
end

# Enumerate PARAM tensors once if ANY observability or checkpoint
# feature is active. Reused by drift/grad emitters AND the checkpoint
# writer. Mat snapshots (drift_snaps) live in main scope — Spinel
# doesn't generate sp_Mat_ptr_array, so cross-function passes of
# Array<Mat> are off-limits.
drift_params = [TinyNNCuda.tnn_null_ptr]; drift_params.pop
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
    TinyNNCuda.tnn_filesystem_mkdir(WEIGHTS_DIR)
    puts "checkpoints: " + WEIGHTS_DIR + "/step_<N>.gguf every " +
         CHECKPOINT_EVERY.to_s + " steps"
  end
end

t_start = Time.now
step = 1
final_loss = 0.0
while step <= STEPS
  step_wall_start = TinyNNCuda.tnn_events_now_seconds
  m_hp.flat[5] = 1.0 / (1.0 - (0.9   ** step.to_f))
  m_hp.flat[6] = 1.0 / (1.0 - (0.999 ** step.to_f))
  if step == 1
    TinyNNCuda.tnn_graph_reset(fcache.sess)
  else
    TinyNNCuda.tnn_graph_reset_grads_only(fcache.sess)
  end
  TinyNNCuda.upload_int_array(fcache.sess, fcache.t_seq_token_ids, seq_ids)
  TinyNNCuda.upload_int_array(fcache.sess, fcache.t_seq_positions, positions)
  TinyNNCuda.upload_row_major(fcache.sess, t_labels, m_labels)
  TinyNNCuda.upload_row_major(fcache.sess, t_hp,     m_hp)
  TinyNNCuda.tnn_compute_backward(fcache.sess)
  TinyNNCuda.tnn_download(fcache.sess, t_loss)
  loss = TinyNNCuda.tnn_scratch_get(fcache.sess, 0)
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
    t_now = TinyNNCuda.tnn_events_now_seconds
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
    step_wall_us = ((TinyNNCuda.tnn_events_now_seconds - step_wall_start) * 1.0e6).to_i
    es  = "{\"kind\":\"step\",\"phase\":\"train\""
    es = es + ",\"t\":"        + TinyNNCuda.tnn_events_now_seconds.to_s
    es = es + ",\"step\":"     + step.to_s
    es = es + ",\"loss\":"     + loss.to_s
    es = es + ",\"lr\":"       + LR.to_s
    es = es + ",\"tokens\":"   + CONTEXT.to_s
    es = es + ",\"wall_us\":"  + step_wall_us.to_s
    es = es + "}"
    TinyNNCuda.tnn_events_emit(es)
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

# Emit run_end. The run *executed to completion* — we always reach
# this point without an exception — so reason is always "completed".
# The learning-gate verdict is a separate, structured signal (the
# `quality_gate` field). Consumers (e.g. Tao) key on reason for crash
# detection; they key on quality_gate.passed for "is this run useful?".
# CI exit-1-on-failure contract preserved separately below.
# Closes toy#run-end-reason-semantics.
exit_code = not_learning ? 1 : 0
if EVENTS.length > 0 && TinyNNCuda.tnn_events_active == 1
  re  = "{\"kind\":\"run_end\""
  re = re + ",\"t\":"           + TinyNNCuda.tnn_events_now_seconds.to_s
  re = re + ",\"ended_at\":\""  + TinyNNCuda.tnn_events_iso8601_now + "\""
  re = re + ",\"reason\":\"completed\""
  re = re + ",\"final_step\":"  + STEPS.to_s
  re = re + ",\"final_loss\":"  + final_loss.to_s
  re = re + ",\"quality_gate\":{\"passed\":" + (not_learning ? "false" : "true") + ""
  re = re + ",\"metric\":\"loss_ratio\""
  re = re + ",\"value\":" + ratio.to_s
  re = re + ",\"threshold\":0.9}"
  re = re + ",\"exit_code\":"   + exit_code.to_s
  re = re + "}"
  TinyNNCuda.tnn_events_emit(re)
  TinyNNCuda.tnn_events_close
  puts "events closed: " + EVENTS
end

if not_learning
  exit 1
end
