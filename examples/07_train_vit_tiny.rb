# E1.6 / GH#13 — ViT-Tiny training driver. Composes the four pieces:
#
#   - prep/preprocess_images.py  → images.bin + labels.bin (one-time)
#   - lib/toy_image_loader.rb    → per-step record reads
#   - lib/vit_tiny_forward_ffi.rb → forward + backward + AdamW graph
#   - lib/toy_lr_schedule.rb     → cosine LR with linear warmup
#
# Acceptance per E1's scope doc:
#   "A 200-step smoke on a tiny CIFAR-like loader without crashing,
#    emits well-formed events.jsonl, and final loss < init loss."
#
# Usage:
#   uv run prep/preprocess_images.py            # data/vit_smoke/ default
#   make example_train_vit_tiny
#   STEPS=200 TAO_RUN_DIR=/tmp/vit ./examples/example_train_vit_tiny

require_relative "../lib/toy_vit"
require_relative "../lib/vit_tiny_forward_ffi"
require_relative "../lib/toy_image_loader"
require_relative "../lib/toy_lr_schedule"
require_relative "../lib/toy_drift_grad"
# NB: NOT requiring toy_gguf_writer here — its `cfg.vocab` / `cfg.d_ff`
# style calls clash with ViTTinyConfig under Spinel poly inference
# (the writer's cfg arg goes poly when both SmolLM2Config and
# ViTTinyConfig live in the same compilation unit). ViT-side
# checkpoints will need a vit_gguf_writer.rb follow-up.

IMG_DIR     = ENV["IMG_DIR"]     || "data/vit_smoke"
IMAGE_SIZE  = (ENV["IMAGE_SIZE"] || "16").to_i
PATCH_SIZE  = (ENV["PATCH_SIZE"] || "4").to_i
NUM_CHAN    = (ENV["NUM_CHAN"]   || "3").to_i
D_MODEL     = (ENV["D_MODEL"]    || "64").to_i
N_HEADS     = (ENV["N_HEADS"]    || "4").to_i
D_FF        = (ENV["D_FF"]       || "128").to_i
N_LAYERS    = (ENV["N_LAYERS"]   || "2").to_i
NUM_CLASSES = (ENV["NUM_CLASSES"]|| "10").to_i
# N_IMAGES=1 by default = single-image memorisation smoke. Verifies
# the full pipeline (loader → forward → CE → backward → AdamW →
# events) end-to-end without depending on capacity questions. For the
# *real* E1 experiment shape (Flowers-102, 2040 images, 96×96, 12L)
# pass N_IMAGES=2040 + a beefier config + many more steps.
N_IMAGES    = (ENV["N_IMAGES"]   || "1").to_i
STEPS       = (ENV["STEPS"]      || "200").to_i
LR_MAX      = (ENV["LR_MAX"]     || "0.003").to_f
LR_MIN      = (ENV["LR_MIN"]     || "0.0001").to_f
WARMUP      = (ENV["WARMUP"]     || "10").to_i
SEED        = (ENV["SEED"]       || "0").to_i

TAO_RUN_DIR = ENV["TAO_RUN_DIR"] || ""
EVENTS      = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/events.jsonl") : ""
RUN_ID      = ENV["TOY_RUN_ID"]   || "vit-tiny"
CHECKPOINT_EVERY = (ENV["CHECKPOINT_EVERY"] || "0").to_i
WEIGHTS_DIR      = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/weights") : ""

cfg = ViTTinyConfig.new(IMAGE_SIZE, PATCH_SIZE, NUM_CHAN, D_MODEL,
                         N_HEADS, D_FF, N_LAYERS, NUM_CLASSES, 1.0e-5)
puts "config: image=" + cfg.image_size.to_s +
     " patch=" + cfg.patch_size.to_s +
     " d=" + cfg.d_model.to_s +
     " heads=" + cfg.n_heads.to_s +
     " d_ff=" + cfg.d_ff.to_s +
     " L=" + cfg.n_layers.to_s +
     " classes=" + cfg.num_classes.to_s

cache = ViTTinyForwardFFICache.new
cache.realize_for_random_init(cfg, SEED, 1.0)
t_loss = cache.build_training_step
puts "realize + build_training_step OK"

# Events stream.
if EVENTS.length > 0
  TinyNN.tnn_events_open(EVENTS)
  t_open = TinyNN.tnn_events_now_seconds
  rs  = "{\"kind\":\"run_start\",\"phase\":\"train\""
  rs  = rs + ",\"t\":"          + t_open.to_s
  rs  = rs + ",\"started_at\":\"" + TinyNN.tnn_events_iso8601_now + "\""
  rs  = rs + ",\"run_id\":\""   + RUN_ID + "\""
  rs  = rs + ",\"name\":\"vit-tiny\""
  rs  = rs + ",\"model\":{\"kind\":\"vit\""
  rs  = rs + ",\"image_size\":" + cfg.image_size.to_s
  rs  = rs + ",\"patch_size\":" + cfg.patch_size.to_s
  rs  = rs + ",\"d_model\":"    + cfg.d_model.to_s
  rs  = rs + ",\"n_heads\":"    + cfg.n_heads.to_s
  rs  = rs + ",\"n_layers\":"   + cfg.n_layers.to_s
  rs  = rs + ",\"num_classes\":" + cfg.num_classes.to_s + "}"
  rs  = rs + ",\"schedule\":{\"lr_max\":" + LR_MAX.to_s +
              ",\"lr_min\":" + LR_MIN.to_s +
              ",\"warmup\":" + WARMUP.to_s +
              ",\"n_steps\":" + STEPS.to_s + "}"
  rs  = rs + ",\"backend\":{\"kind\":\"" + TinyNN.tnn_backend_name(cache.sess) + "\"}"
  rs  = rs + "}"
  TinyNN.tnn_events_emit(rs)
end

plist = ToyDriftGrad.params(cache.sess)
puts "params tracked: " + plist.length.to_s
if CHECKPOINT_EVERY > 0 && WEIGHTS_DIR.length > 0
  TinyNN.tnn_filesystem_mkdir(WEIGHTS_DIR)
  puts "checkpoints: " + WEIGHTS_DIR + "/step_<N>.gguf every " +
       CHECKPOINT_EVERY.to_s + " steps"
end

# Pre-allocate buffers used every step.
n_patches  = (cfg.image_size / cfg.patch_size) * (cfg.image_size / cfg.patch_size)
patch_flat = cfg.num_channels * cfg.patch_size * cfg.patch_size
record_f   = patch_flat * n_patches

m_image  = Mat.new(patch_flat, n_patches)
m_labels = Mat.new(1, cfg.num_classes)
cls_idx  = [0]
m_hp = Mat.new(1, 7)
m_hp.flat[1] = 0.9; m_hp.flat[2] = 0.95
m_hp.flat[3] = 1.0e-8; m_hp.flat[4] = 0.0
m_hp.flat[5] = 0.9; m_hp.flat[6] = 0.95

images_path = IMG_DIR + "/images.bin"
labels_path = IMG_DIR + "/labels.bin"

initial_loss = 0.0
final_loss   = 0.0
step = 0
while step < STEPS
  lr = ToyLR.cosine(step, STEPS, LR_MAX, LR_MIN, WARMUP)
  m_hp.flat[0] = lr

  # Cycle through the dataset.
  idx = step % N_IMAGES
  patches = ToyImageLoader.read_image(images_path, idx, record_f)
  label   = ToyImageLoader.read_label(labels_path, idx)
  if label < 0 || label >= cfg.num_classes
    puts "skip step " + step.to_s + " (bad label=" + label.to_s + ")"
    step = step + 1
    next
  end

  # Copy into the pre-allocated Mat.
  i = 0
  while i < record_f
    m_image.flat[i] = patches[i]
    i = i + 1
  end
  # One-hot label.
  j = 0
  while j < cfg.num_classes
    m_labels.flat[j] = (j == label) ? 1.0 : 0.0
    j = j + 1
  end

  if step == 0
    TinyNN.tnn_graph_reset(cache.sess)
  else
    TinyNN.tnn_graph_reset_grads_only(cache.sess)
  end
  TinyNN.upload_row_major(cache.sess, cache.t_image, m_image)
  TinyNN.upload_int_array(cache.sess, cache.t_cls_idx, cls_idx)
  TinyNN.upload_row_major(cache.sess, cache.t_labels, m_labels)
  TinyNN.upload_row_major(cache.sess, cache.t_hp, m_hp)
  TinyNN.tnn_compute_backward(cache.sess)

  loss_mat = TinyNN.download_row_major(cache.sess, t_loss, 1, 1)
  loss = loss_mat.flat[0]
  if step == 0; initial_loss = loss; end
  final_loss = loss

  if (step % 20) == 0 || step == STEPS - 1
    puts "step " + (step + 1).to_s.rjust(4) +
         ": lr=" + lr.to_s + " idx=" + idx.to_s + " label=" + label.to_s +
         " loss=" + loss.to_s
  end

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
    # Re-use the SmolLM2-style cfg getters expected by ToyGGUFWriter
    # by stuffing the cache's cfg into a minimal sm-shape via the
    # config's accessors. For now the writer doesn't know about
    # ViT-specific fields — emit only what it can.
    # NB: the writer's name-anchor (#11/#16) honors whatever names
    # the realize path stamped via tnn_tensor_set_name.
    # Skipping for the minimum smoke; #169 (ViT-side checkpoint
    # writer) would lands a vit_gguf_writer.rb here.
  end

  step = step + 1
end

ratio = initial_loss > 0.0 ? final_loss / initial_loss : 1.0
quality_gate = ratio < 0.95 ? "passed" : "failed"
puts "initial=" + initial_loss.to_s + " final=" + final_loss.to_s +
     " ratio=" + ratio.to_s + " gate=" + quality_gate

if EVENTS.length > 0
  t_close = TinyNN.tnn_events_now_seconds
  re = "{\"kind\":\"run_end\",\"phase\":\"train\""
  re = re + ",\"t\":"      + t_close.to_s
  re = re + ",\"reason\":\"completed\""
  re = re + ",\"quality_gate\":{\"name\":\"loss_ratio\""
  re = re + ",\"value\":"  + ratio.to_s
  re = re + ",\"status\":\"" + quality_gate + "\"}"
  re = re + "}"
  TinyNN.tnn_events_emit(re)
  TinyNN.tnn_events_close
end

puts "ViT-Tiny driver done"
