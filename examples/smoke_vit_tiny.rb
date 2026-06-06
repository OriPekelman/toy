# E1.3 / GH#13 — ViT-Tiny forward + training smoke.
#
#   make examples/smoke_vit_tiny
#   STEPS=5 ./examples/smoke_vit_tiny
#
# Runs a tiny ViT (image 16×16, patch 4, d_model 64, 2 layers, 4
# heads, 10 classes). Trains 5 steps on the same random image →
# random class label; verifies forward + backward + AdamW step
# without crashing and that loss decreases.

require_relative "../lib/toy/models/toy_vit"
require_relative "../lib/toy/llm/engine/vit_tiny_engine"
require_relative "../lib/toy/llm/adamw"

IMAGE_SIZE  = (ENV["IMAGE_SIZE"]  || "16").to_i
PATCH_SIZE  = (ENV["PATCH_SIZE"]  || "4").to_i
NUM_CHAN    = (ENV["NUM_CHAN"]    || "3").to_i
D_MODEL     = (ENV["D_MODEL"]     || "64").to_i
N_HEADS     = (ENV["N_HEADS"]     || "4").to_i
D_FF        = (ENV["D_FF"]        || "128").to_i
N_LAYERS    = (ENV["N_LAYERS"]    || "2").to_i
NUM_CLASSES = (ENV["NUM_CLASSES"] || "10").to_i
STEPS       = (ENV["STEPS"]       || "5").to_i
SEED        = (ENV["SEED"]        || "0").to_i

cfg = ViTTinyConfig.new(IMAGE_SIZE, PATCH_SIZE, NUM_CHAN, D_MODEL,
                         N_HEADS, D_FF, N_LAYERS, NUM_CLASSES, 1.0e-5)
puts "config: image=" + cfg.image_size.to_s +
     " patch=" + cfg.patch_size.to_s +
     " d=" + cfg.d_model.to_s +
     " heads=" + cfg.n_heads.to_s +
     " d_ff=" + cfg.d_ff.to_s +
     " L=" + cfg.n_layers.to_s +
     " classes=" + cfg.num_classes.to_s

cache = Toy::LLM::Engine::ViTTinyEngine.new
cache.realize_for_random_init(cfg, SEED, 1.0)
puts "realize OK"

t_loss = cache.build_training_step
puts "build_training_step OK"

# Random image. Pre-flatten patches into a [patch_flat_dim, N_patches]
# matrix host-side — this is the "patchify" step normally folded into
# conv2d(stride=patch). For a simple smoke the deterministic-fill
# pattern is fine; a real loader (E1.5) does the same thing with real
# pixels.
patch        = cfg.patch_size
img_side     = cfg.image_size
patches_side = img_side / patch
n_patches    = patches_side * patches_side
patch_flat   = cfg.num_channels * patch * patch
m_image = Mat.new(patch_flat, n_patches)

# Sequential fill — every patch ends up with a distinct first row of
# values. Detail of the fill doesn't matter for the memorisation smoke.
total = patch_flat * n_patches
i = 0
while i < total
  m_image.flat[i] = ((i % 7) - 3) * 0.1
  i = i + 1
end
target_class = 3   # arbitrary

m_labels = Mat.new(1, cfg.num_classes)
j = 0
while j < cfg.num_classes
  m_labels.flat[j] = (j == target_class) ? 1.0 : 0.0
  j = j + 1
end

cls_idx = [0]   # always take cls token at position 0

# NAMED AdamW (byte-identical to the old hand-filled m_hp): all defaults
# (lr=0.001, β1=0.9, β2=0.95, eps=1e-8, wd=0, bias_correct=false → slots
# 5/6 = constant betas).
m_hp = Toy::AdamW.new.hp(0)

losses = [0.0]; losses.pop
step = 0
while step < STEPS
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
  losses.push(loss)
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s
  step = step + 1
end

initial = losses[0]
final   = losses[losses.length - 1]
ratio   = initial > 0.0 ? final / initial : 1.0
puts "initial=" + initial.to_s + " final=" + final.to_s + " ratio=" + ratio.to_s
if ratio < 0.95
  puts "VERDICT: ViT training is learning"
else
  puts "VERDICT: training NOT learning (final/initial = " + ratio.to_s + ")"
end
