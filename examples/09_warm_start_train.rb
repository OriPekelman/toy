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
#
# GH#14 — Qwen-2.5-1.5B → 410M-shape transfer invocation:
#
#   mkdir -p /tmp/qwen410
#   # 1. pretokenize FineWeb-Edu (toy#22 / prep/pretokenize_fineweb_edu.py)
#   uv run prep/pretokenize_fineweb_edu.py --tokens 10_000_000 \
#     --out data/fineweb_edu_10m.bin
#
#   # 2. PCA-init the projection lens (GH#14 / prep/pca_init_qwen_lens.py)
#   uv run prep/pca_init_qwen_lens.py --donor data/qwen25-1.5b-f32.gguf \
#     --d-model 1024 --out data/qwen_pca_lens.gguf
#
#   # 3. train (TOKENS knob derives STEPS = ceil(TOKENS / CONTEXT))
#   DONOR_GGUF=data/qwen25-1.5b-f32.gguf \
#     PCA_GGUF=data/qwen_pca_lens.gguf \
#     VOCAB=151936 DONOR_D=1536 D_MODEL=1024 \
#     N_LAYERS=24 N_HEADS=8 D_FF=4096 \
#     CONTEXT=256 TOKENS=10_000_000 SEED=0 INIT=warm \
#     CORPUS=data/fineweb_edu_10m.bin \
#     TAO_RUN_DIR=/tmp/qwen410 \
#     ./examples/example_warm_start_train
#
# The Qwen-2.5-1.5B GGUF supplies token_embd.weight directly (no
# separate extractor needed — 09 reads only token_embd from the
# donor). The 410M target shape is (1024, 24L, 8H, GQA-2-KV, 4096
# FFN) ≈ 380-400M params with tied embeddings (the projection
# lens adds another ~1.5M). For the FineWeb-Edu corpus loader
# (see issue #14 acceptance), see follow-up issue.

require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy/llm/engine/llama_seq_engine"
require_relative "../lib/toy_drift_grad"
require_relative "../lib/toy_gguf_writer"
require_relative "../lib/toy/io/toy_corpus_loader"
require_relative "../lib/toy_lr_schedule"

D_MODEL  = (ENV["D_MODEL"]  || "64").to_i
DONOR_D  = (ENV["DONOR_D"]  || "128").to_i
N_LAYERS = (ENV["N_LAYERS"] || "2").to_i
N_HEADS  = (ENV["N_HEADS"]  || "4").to_i
D_FF     = (ENV["D_FF"]     || "128").to_i
CONTEXT  = (ENV["CONTEXT"]  || "32").to_i
# GH#14 — TOKENS=N derives STEPS = ceil(N / CONTEXT). At B=1 each
# step processes CONTEXT tokens; TOKENS is the natural acceptance
# unit ("train on 10M tokens" — what the issue uses). STEPS takes
# precedence when both are set so existing STEPS-only runs are
# unchanged. Computed via a single ternary to keep Spinel's
# constant-resolution happy (the conditional-assign-inside-if form
# emits an "uninitialized constant" runtime error).
STEPS_RAW  = (ENV["STEPS"]  || "0").to_i
TOKENS_RAW = (ENV["TOKENS"] || "0").to_i
STEPS = STEPS_RAW > 0 ? STEPS_RAW : (TOKENS_RAW > 0 ? (TOKENS_RAW + CONTEXT - 1) / CONTEXT : 20)
LR_MAX   = (ENV["LR_MAX"]   || "0.001").to_f
LR_MIN   = (ENV["LR_MIN"]   || "0.00001").to_f
WARMUP   = (ENV["WARMUP"]   || "5").to_i
SEED     = (ENV["SEED"]     || "0").to_i
CORPUS   = ENV["CORPUS"]    || "data/ts_seqs.bin"
VOCAB    = (ENV["VOCAB"]    || "627").to_i

# INIT={scratch,warm}. scratch (default) = random init for both
# token_embd and W_proj (current behaviour). warm = load the donor
# embedding from a Qwen-class GGUF; W_proj stays random-init.
# Matches granite_transfer #38's warm-vs-scratch arms.
#
# Cfg-vs-donor dim check: donor's d_model (llama.embedding_length in
# the GGUF) must equal cfg.donor_d_in. We don't enforce vocab
# alignment — we read the first VOCAB rows of the donor's token_embd
# and load them as the donor side, which is correct when the toy
# corpus is *also* tokenized against the donor's tokenizer (the real
# E2 protocol uses Qwen-tokenized FineWeb-Edu). For the smoke against
# TinyStories vocab=627, those first 627 Qwen rows are essentially
# "structured noise" — exercises the warm-init machinery without
# claiming scientific alignment.
INIT        = ENV["INIT"]        || "scratch"
DONOR_GGUF  = ENV["DONOR_GGUF"]  || "data/qwen25-1.5b-f32.gguf"
# GH#14 — PCA-init the projection lens W_proj. Independent of INIT
# (the donor-embed-load knob); set PCA_GGUF= to a GGUF written by
# prep/pca_init_qwen_lens.py and the lens.proj.weight tensor is
# uploaded to fcache.t_seq_w_proj before the first step. Empty
# string = random-init (current behaviour). Matches the issue's
# "PCA-initialised against donor activations" line — we treat the
# donor embed rows AS the donor activations (they are the
# activations immediately post-embedding-lookup).
PCA_GGUF    = ENV["PCA_GGUF"]    || ""

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

fcache = Toy::LLM::Engine::LlamaSeqEngine.new
# untied=true is mandatory when donor_d_in > 0 — see E2.3 commit.
fcache.realize_for_random_init(cfg, CONTEXT, 1, 0, true, false, SEED, 1.0)
puts "realize OK"

# INIT=warm: load donor token_embd.weight into the realize'd embed
# table (first VOCAB rows; donor's vocab is normally much larger).
# Must come AFTER realize (the tensor exists) and BEFORE the first
# step (else we train through random initial values).
if INIT == "warm"
  unless File.exist?(DONOR_GGUF)
    puts "INIT=warm: donor GGUF not found at " + DONOR_GGUF
    puts "  set DONOR_GGUF= to a Qwen-shaped GGUF that has token_embd.weight"
    exit 1
  end
  ggh = TinyNN.tnn_gguf_load(DONOR_GGUF)
  if ggh == nil || ggh == TinyNN.tnn_null_ptr
    puts "INIT=warm: failed to open " + DONOR_GGUF
    exit 1
  end
  donor_d = TinyNN.tnn_gguf_get_u32(ggh, "llama.embedding_length")
  if donor_d != DONOR_D
    puts "INIT=warm: donor d_model=" + donor_d.to_s +
         " but cfg.donor_d_in=" + DONOR_D.to_s + " (DONOR_D mismatch)"
    puts "  set DONOR_D=" + donor_d.to_s + " to match the donor"
    exit 1
  end
  te_idx = TinyNN.tnn_gguf_find_index(ggh, "token_embd.weight")
  if te_idx < 0
    puts "INIT=warm: donor has no token_embd.weight tensor"
    exit 1
  end
  n_to_read = VOCAB * DONOR_D
  te_buf = Mat.new(1, n_to_read)
  # tnn_gguf_read_f32_to_doubles returns 0 on success, negative on
  # error (different convention from tnn_read_f32_file which returns
  # element count). The function caps n at available internally, so a
  # short read returns 0 too — best diagnostic is verifying values
  # downstream.
  rc = TinyNN.tnn_gguf_read_f32_to_doubles(ggh, te_idx, te_buf.flat, n_to_read)
  if rc != 0
    puts "INIT=warm: read failed rc=" + rc.to_s + " on token_embd.weight"
    exit 1
  end
  TinyNN.tnn_upload_from_float_array(fcache.sess, fcache.t_seq_token_embed, te_buf.flat, n_to_read)
  TinyNN.tnn_gguf_free(ggh)
  puts "INIT=warm: loaded " + n_to_read.to_s + " floats (" + VOCAB.to_s +
       " × " + DONOR_D.to_s + ") from " + DONOR_GGUF + " token_embd.weight"
end

# GH#14 — PCA-init the projection lens W_proj. Independent of INIT;
# can be combined with INIT=warm for the full Tao-acceptance shape.
if PCA_GGUF.length > 0
  unless File.exist?(PCA_GGUF)
    puts "PCA_GGUF: file not found at " + PCA_GGUF
    puts "  generate via: uv run prep/pca_init_qwen_lens.py --donor " +
         DONOR_GGUF + " --d-model " + D_MODEL.to_s + " --out " + PCA_GGUF
    exit 1
  end
  pgh = TinyNN.tnn_gguf_load(PCA_GGUF)
  if pgh == nil || pgh == TinyNN.tnn_null_ptr
    puts "PCA_GGUF: failed to open " + PCA_GGUF
    exit 1
  end
  pca_donor_d = TinyNN.tnn_gguf_get_u32(pgh, "lens.donor_d_in")
  pca_d_model = TinyNN.tnn_gguf_get_u32(pgh, "lens.d_model")
  if pca_donor_d != DONOR_D
    puts "PCA_GGUF: lens.donor_d_in=" + pca_donor_d.to_s +
         " but cfg.donor_d_in=" + DONOR_D.to_s + " — re-run the PCA " +
         "script with --d-model " + D_MODEL.to_s + " against the matching donor"
    exit 1
  end
  if pca_d_model != D_MODEL
    puts "PCA_GGUF: lens.d_model=" + pca_d_model.to_s +
         " but D_MODEL=" + D_MODEL.to_s + " — re-run the PCA script"
    exit 1
  end
  pw_idx = TinyNN.tnn_gguf_find_index(pgh, "lens.proj.weight")
  if pw_idx < 0
    puts "PCA_GGUF: missing lens.proj.weight tensor"
    exit 1
  end
  n_pca = D_MODEL * DONOR_D
  pw_buf = Mat.new(1, n_pca)
  rc = TinyNN.tnn_gguf_read_f32_to_doubles(pgh, pw_idx, pw_buf.flat, n_pca)
  if rc != 0
    puts "PCA_GGUF: read failed rc=" + rc.to_s + " on lens.proj.weight"
    exit 1
  end
  TinyNN.tnn_upload_from_float_array(fcache.sess, fcache.t_seq_w_proj, pw_buf.flat, n_pca)
  TinyNN.tnn_gguf_free(pgh)
  puts "PCA_GGUF: loaded " + n_pca.to_s + " floats (" + D_MODEL.to_s +
       " × " + DONOR_D.to_s + ") from " + PCA_GGUF + " lens.proj.weight"
end

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
  rs  = rs + ",\"config\":{\"init\":\"" + INIT + "\""
  rs  = rs + ",\"donor\":\"" + (INIT == "warm" ? DONOR_GGUF : "") + "\""
  rs  = rs + ",\"seed\":" + SEED.to_s + "}"
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
  # toy/v1 eval event — final loss snapshot, name="final".
  # GH#14 acceptance: Tao's report consumes this as the "validation"
  # eval at run end. (For a real held-out eval, point a separate
  # corpus at this through a new --eval-corpus knob; the current
  # snapshot is the training-set loss at the last step, which is
  # adequate for the wire-format acceptance.)
  t_eval = TinyNN.tnn_events_now_seconds
  ev = "{\"kind\":\"eval\",\"phase\":\"train\""
  ev = ev + ",\"t\":"     + t_eval.to_s
  ev = ev + ",\"step\":"  + STEPS.to_s
  ev = ev + ",\"name\":\"final\""
  ev = ev + ",\"loss\":"  + final_loss.to_s
  ev = ev + "}"
  TinyNN.tnn_events_emit(ev)

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
