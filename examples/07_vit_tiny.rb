# examples/07_vit_tiny.rb — the same recipe shape, but for IMAGES.
#
# WHAT YOU'LL SEE: a ViT-Tiny image classifier (224x224 image, 16x16
# patches, 12 transformer blocks, class token, 10 classes) trained on
# the bundled smoke image. Cross-entropy starts near ln(10) ≈ 2.30 and
# falls as the model memorizes the label — proof that the L5 recipe
# contract (realize! once, step! per step) is arch-agnostic: this is
# the SAME shape as 01, with an image where the token sequence was.
#
# HOW LONG: ~20 s for 20 steps (CPU; 12 blocks at d=192 is the largest
# curated example). The pinned smoke corpus is COMMITTED at
# data/vit_smoke/ — nothing to download.
#
#   make example_07
#   ./examples/example_07_vit_tiny
#
# WHAT TO TWEAK (env, no recompile):
#   STEPS=50      watch it actually memorize (loss -> ~0)
#   SEED=1        a different random init
#   IMG_DIR=…     your own corpus (prep/preprocess_images.py writes one)
#
# THE API — Toy::LLM::Recipes::VitTiny:
#   realize!(cfg, opts)                — random-init ViT + training graph
#   step!(m_image, cls_idx, m_labels, m_hp, is_first)
# The per-step inputs are an image Mat ([patch_flat x n_patches]) and a
# one-hot class label — where the LLM recipes take token ids and
# shift-by-one labels. No CLI surface for ViT yet; this file is it.

require_relative "../lib/toy/compute"

STEPS   = (ENV["STEPS"] || "20").to_i
SEED    = (ENV["SEED"]  || "0").to_i
IMG_DIR = ENV["IMG_DIR"] || "data/vit_smoke"

# The timm ViT-Tiny shape data/vit_smoke matches (224/16 -> 196 patches).
IMAGE_SIZE  = 224
PATCH_SIZE  = 16
NUM_CHAN    = 3
D_MODEL     = 192
N_HEADS     = 3
D_FF        = 768
N_LAYERS    = 12
NUM_CLASSES = 10

cfg = ViTTinyConfig.new(IMAGE_SIZE, PATCH_SIZE, NUM_CHAN, D_MODEL,
                        N_HEADS, D_FF, N_LAYERS, NUM_CLASSES, 1.0e-5)
puts "model: image=" + IMAGE_SIZE.to_s + " patch=" + PATCH_SIZE.to_s +
     " d=" + D_MODEL.to_s + " heads=" + N_HEADS.to_s +
     " L=" + N_LAYERS.to_s + " classes=" + NUM_CLASSES.to_s

opts = Toy::LLM::RecipeOptions.new
opts.seed = SEED          # the ViT path consumes seed + init_scale only

recipe = Toy::LLM::Recipes::VitTiny.new
recipe.realize!(cfg, opts)

# The bundled corpus: images.bin (pre-patchified f32 records) +
# labels.bin. Fail LOUD if missing or short — the loader's zero-fill on
# a short read would otherwise silently train on zeros (never-mask).
images_path = IMG_DIR + "/images.bin"
labels_path = IMG_DIR + "/labels.bin"
n_patches   = (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE)
patch_flat  = NUM_CHAN * PATCH_SIZE * PATCH_SIZE
record_f    = patch_flat * n_patches
if !File.exist?(images_path) || !File.exist?(labels_path)
  puts "07_vit_tiny: corpus missing under " + IMG_DIR +
       " (need images.bin + labels.bin)."
  puts "  the pinned smoke corpus is committed at data/vit_smoke/ — run from"
  puts "  the repo root, or write your own: uv run prep/preprocess_images.py"
  exit 1
end
probe = Array.new(record_f, 0.0)
got = TinyNN.tnn_read_f32_file(images_path, 0, record_f, probe)
if got != record_f
  puts "07_vit_tiny: short read " + got.to_s + "/" + record_f.to_s +
       " from " + images_path + " — refusing to train on zero-fill."
  exit 1
end

m_image  = Mat.new(patch_flat, n_patches)
m_labels = Mat.new(1, NUM_CLASSES)
cls_idx  = [0]                       # read the class token at position 0
adamw    = Toy::AdamW.for_from_scratch

first_loss = 0.0
final_loss = 0.0
step = 0
while step < STEPS
  adamw.lr = ToyLR.cosine(step, STEPS, 0.003, 0.0001, 10)

  # One image per step (the smoke corpus has a single record; a real
  # corpus would advance the index).
  patches = ToyImageLoader.read_image(images_path, 0, record_f)
  label   = ToyImageLoader.read_label(labels_path, 0)
  i = 0
  while i < record_f
    m_image.flat[i] = patches[i]
    i = i + 1
  end
  j = 0
  while j < NUM_CLASSES
    m_labels.flat[j] = j == label ? 1.0 : 0.0
    j = j + 1
  end

  loss = recipe.step!(m_image, cls_idx, m_labels, adamw.hp(step), step == 0)
  if step == 0
    first_loss = loss
  end
  final_loss = loss
  puts "step " + (step + 1).to_s + ": loss=" + loss.to_s
  step = step + 1
end

puts ""
puts "vit-tiny: " + STEPS.to_s + " steps, loss " + first_loss.to_s +
     " -> " + final_loss.to_s
if final_loss < first_loss
  puts "VERDICT: learning"
else
  puts "VERDICT: NOT learning (loss did not fall)"
end
