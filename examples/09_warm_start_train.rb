# E2.5 / GH#14 — warm-start training driver.
#
# Composes the projection-lens realize path (#161/E2.3), the streaming
# corpus loader (#163/E2.4), and the cosine LR schedule into a single
# training entry point. This is the "Tao acceptance shape" for E2
# minus the real donor load — for the SMOKE we random-init the donor
# embed alongside the rest. Plugging a real Qwen-2.5-1.5B donor in is
# an upload step the caller does between realize and the first step.
#
#   uv run prep/pretokenize_corpus.py      # one-time, ts_seqs.bin
#   make example_warm_start_train
#   STEPS=20 ./examples/example_warm_start_train
#
# Env knobs:
#   D_MODEL   target hidden dim (default 64)
#   DONOR_D   donor embedding width (default 128)
#   N_LAYERS  target depth (default 2)
#   N_HEADS   target attention heads (default 4)
#   D_FF      target FFN width (default 128)
#   CONTEXT   sequence length / T (default 32)
#   STEPS     number of training steps (default 20)
#   LR_MAX    peak LR (default 1e-3)
#   LR_MIN    final LR (default 1e-5)
#   WARMUP    linear warmup steps (default 5)
#   SEED      RNG seed (default 0)
#   CORPUS    path to packed i32 binary (default data/ts_seqs.bin)
#   VOCAB     vocab size (default 627 = TinyStories)
#   TAO_RUN_DIR  emit events.jsonl + checkpoints if set
#   CHECKPOINT_EVERY  weights/step_N.gguf every N steps (default 0 = off)
#
# Acceptance: a 20-step run produces a monotonically-decreasing loss
# trajectory and emits a well-formed events.jsonl. The Tao-side
# `quality_gate` lands as part of run_end.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/llama_seq_forward_ffi"
require_relative "../lib/toy_drift_grad"
require_relative "../lib/toy_gguf_writer"
require_relative "../lib/toy_corpus_loader"
require_relative "../lib/toy_lr_schedule"

D_MODEL  = (ENV["D_MODEL"]  || "64").to_i
DONOR_D  = (ENV["DONOR_D"]  || "128").to_i
N_LAYERS = (ENV["N_LAYERS"] || "2").to_i
N_HEADS  = (ENV["N_HEADS"]  || "4").to_i
D_FF     = (ENV["D_FF"]     || "128").to_i
CONTEXT  = (ENV["CONTEXT"]  || "32").to_i
STEPS    = (ENV["STEPS"]    || "20").to_i
LR_MAX   = (ENV["LR_MAX"]   || "0.001").to_f
LR_MIN   = (ENV["LR_MIN"]   || "0.00001").to_f
WARMUP   = (ENV["WARMUP"]   || "5").to_i
SEED     = (ENV["SEED"]     || "0").to_i
CORPUS   = ENV["CORPUS"]    || "data/ts_seqs.bin"
VOCAB    = (ENV["VOCAB"]    || "627").to_i

TAO_RUN_DIR = ENV["TAO_RUN_DIR"] || ""
EVENTS      = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/events.jsonl") : ""
RUN_ID      = ENV["TOY_RUN_ID"] || "warm-start"
CHECKPOINT_EVERY = (ENV["CHECKPOINT_EVERY"] || "0").to_i
WEIGHTS_DIR      = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/weights") : ""

cfg = Toy::SmolLM2Config.new(VOCAB, D_MODEL, N_HEADS, N_HEADS,
                              D_FF, N_LAYERS, CONTEXT, 10000.0, 1.0e-5)
cfg.donor_d_in = DONOR_D
puts "config: vocab=" + cfg.vocab.to_s +
     " donor_d_in=" + cfg.donor_d_in.to_s +
     " d=" + cfg.d_model.to_s +
     " L=" + cfg.n_layers.to_s +
     " heads=" + cfg.n_heads.to_s +
     " d_ff=" + cfg.d_ff.to_s

fcache = LlamaSeqForwardFFICache.new
# untied=true is mandatory when donor_d_in > 0 — see E2.3 commit.
fcache.realize_for_random_init(cfg, CONTEXT, true, false, SEED, 1.0)
puts "realize OK"

result   = fcache.build_training_step
t_loss   = result[0]
t_labels = result[1]
t_hp     = result[2]

# Events stream.
if EVENTS.length > 0
  TinyNN.tnn_events_open(EVENTS)
  t_open = TinyNN.tnn_events_now_seconds
  rs  = "{\"kind\":\"run_start\",\"phase\":\"train\""
  rs  = rs + ",\"t\":"          + t_open.to_s
  rs  = rs + ",\"started_at\":\"" + TinyNN.tnn_events_iso8601_now + "\""
  rs  = rs + ",\"run_id\":\""   + RUN_ID + "\""
  rs  = rs + ",\"name\":\"warm-start\""
  rs  = rs + ",\"model\":{\"d_model\":" + D_MODEL.to_s +
              ",\"donor_d_in\":" + DONOR_D.to_s +
              ",\"n_layers\":" + N_LAYERS.to_s +
              ",\"n_heads\":" + N_HEADS.to_s +
              ",\"vocab\":" + VOCAB.to_s + "}"
  rs  = rs + ",\"schedule\":{\"lr_max\":" + LR_MAX.to_s +
              ",\"lr_min\":" + LR_MIN.to_s +
              ",\"warmup\":" + WARMUP.to_s +
              ",\"n_steps\":" + STEPS.to_s + "}"
  rs  = rs + ",\"backend\":{\"kind\":\"" + TinyNN.tnn_backend_name(fcache.sess) + "\"}"
  rs  = rs + "}"
  TinyNN.tnn_events_emit(rs)
end

# Optional: track PARAM list (semantic-named per #11) for checkpoint
# writes. plist drives the writer.
plist = ToyDriftGrad.params(fcache.sess)
puts "params tracked: " + plist.length.to_s
if CHECKPOINT_EVERY > 0 && WEIGHTS_DIR.length > 0
  TinyNN.tnn_filesystem_mkdir(WEIGHTS_DIR)
  puts "checkpoints: " + WEIGHTS_DIR + "/step_<N>.gguf every " +
       CHECKPOINT_EVERY.to_s + " steps"
end

# AdamW hp[1..6] are constants across steps; hp[0] (lr) refreshes
# each step from the cosine schedule.
m_hp = Mat.new(1, 7)
m_hp.flat[0] = LR_MAX
m_hp.flat[1] = 0.9
m_hp.flat[2] = 0.95
m_hp.flat[3] = 1.0e-8
m_hp.flat[4] = 0.0
m_hp.flat[5] = 0.9
m_hp.flat[6] = 0.95

# Pre-compute the position vector (shared across steps).
positions = [0]; positions.pop
p = 0; while p < CONTEXT; positions.push(p); p = p + 1; end

# Pre-allocate label buffer (we overwrite each step).
m_labels = Mat.new(CONTEXT, VOCAB)

# Training loop.
byte_offset   = 0
initial_loss  = 0.0
final_loss    = 0.0
step          = 0
while step < STEPS
  # Cosine LR for this step.
  lr = ToyLR.cosine(step, STEPS, LR_MAX, LR_MIN, WARMUP)
  m_hp.flat[0] = lr

  # Read next sequence from the corpus, wrap on EOF.
  seq_ids = ToyCorpusLoader.read_seq(CORPUS, byte_offset, CONTEXT)
  byte_offset = byte_offset + CONTEXT * 4   # i32

  # Build shift-by-one one-hot labels.
  j = 0
  while j < CONTEXT * VOCAB
    m_labels.flat[j] = 0.0
    j = j + 1
  end
  k = 0
  while k < CONTEXT
    target = (k + 1 < CONTEXT) ? seq_ids[k + 1] : seq_ids[k]
    if target >= 0 && target < VOCAB
      m_labels.flat[k * VOCAB + target] = 1.0
    end
    k = k + 1
  end

  if step == 0
    TinyNN.tnn_graph_reset(fcache.sess)
  else
    TinyNN.tnn_graph_reset_grads_only(fcache.sess)
  end
  TinyNN.upload_int_array(fcache.sess, fcache.t_seq_token_ids, seq_ids)
  TinyNN.upload_int_array(fcache.sess, fcache.t_seq_positions, positions)
  TinyNN.upload_row_major(fcache.sess, t_labels, m_labels)
  TinyNN.upload_row_major(fcache.sess, t_hp,     m_hp)

  TinyNN.tnn_compute_backward(fcache.sess)

  loss_mat = TinyNN.download_row_major(fcache.sess, t_loss, 1, 1)
  loss = loss_mat.flat[0]
  if step == 0
    initial_loss = loss
  end
  final_loss = loss
  puts "step " + (step + 1).to_s.rjust(4) + ": lr=" + lr.to_s + " loss=" + loss.to_s

  # Emit a step event.
  if EVENTS.length > 0
    t_now = TinyNN.tnn_events_now_seconds
    ev = "{\"kind\":\"step\",\"phase\":\"train\""
    ev = ev + ",\"t\":"    + t_now.to_s
    ev = ev + ",\"step\":" + (step + 1).to_s
    ev = ev + ",\"loss\":" + loss.to_s
    ev = ev + ",\"lr\":"   + lr.to_s
    ev = ev + "}"
    TinyNN.tnn_events_emit(ev)
  end

  if CHECKPOINT_EVERY > 0 && WEIGHTS_DIR.length > 0 && ((step + 1) % CHECKPOINT_EVERY) == 0
    ToyGGUFWriter.write_step(cfg, plist, WEIGHTS_DIR, RUN_ID, step + 1)
  end

  step = step + 1
end

# Run-end.
ratio = initial_loss > 0.0 ? final_loss / initial_loss : 1.0
quality_gate = "passed"
reason = "completed"
if ratio >= 0.95
  quality_gate = "failed"
  # Not "errored" — we intentionally distinguish run errors from
  # quality misses per #147 (toy#run-end-reason-semantics).
end
puts "initial=" + initial_loss.to_s + " final=" + final_loss.to_s + " ratio=" + ratio.to_s + " gate=" + quality_gate

if EVENTS.length > 0
  t_close = TinyNN.tnn_events_now_seconds
  re = "{\"kind\":\"run_end\",\"phase\":\"train\""
  re = re + ",\"t\":"      + t_close.to_s
  re = re + ",\"reason\":\"" + reason + "\""
  re = re + ",\"quality_gate\":{\"name\":\"loss_ratio\""
  re = re + ",\"value\":"  + ratio.to_s
  re = re + ",\"status\":\"" + quality_gate + "\"}"
  re = re + "}"
  TinyNN.tnn_events_emit(re)
  TinyNN.tnn_events_close
end

puts "warm-start driver done"
