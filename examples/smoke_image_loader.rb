# E1.5 / GH#13 — image loader smoke. Reads the first 3 images +
# labels written by prep/preprocess_images.py and prints summary
# stats to verify the round-trip.
#
#   uv run prep/preprocess_images.py            # produces data/vit_smoke/
#   make examples/smoke_image_loader
#   ./examples/smoke_image_loader

require_relative "../lib/transformer"
require_relative "../lib/tinynn"
require_relative "../lib/toy_image_loader"

DIR        = ENV["IMG_DIR"]      || "data/vit_smoke"
N          = (ENV["N"]           || "3").to_i
PATCH_FLAT = (ENV["PATCH_FLAT"]  || "48").to_i    # 3*4*4 for the synth shape
N_PATCHES  = (ENV["N_PATCHES"]   || "16").to_i    # (16/4)^2 for the synth shape

images_path = DIR + "/images.bin"
labels_path = DIR + "/labels.bin"
record_floats = PATCH_FLAT * N_PATCHES
puts "loading " + images_path + " (record=" + record_floats.to_s + " floats)"

i = 0
while i < N
  patches = ToyImageLoader.read_image(images_path, i, record_floats)
  label   = ToyImageLoader.read_label(labels_path, i)
  # Summary stats: min, max, mean over the patch vector.
  mn = patches[0]; mx = patches[0]; s = 0.0
  j = 0
  while j < record_floats
    v = patches[j]
    if v < mn; mn = v; end
    if v > mx; mx = v; end
    s = s + v
    j = j + 1
  end
  mean = s / record_floats.to_f
  puts "image " + i.to_s + ": label=" + label.to_s +
       " min=" + mn.to_s + " max=" + mx.to_s + " mean=" + mean.to_s
  i = i + 1
end
puts "image_loader smoke OK"
