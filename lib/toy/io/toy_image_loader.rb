# E1.5 / GH#13 — streaming image-dataset loader for ViT training.
#
# Reads pre-flattened patches + labels written by
# prep/preprocess_images.py:
#
#   images.bin   raw f32, n_images × (patch_flat × n_patches)
#   labels.bin   raw i32, n_images
#
# One record per call: read_image returns the patches as Array<Float>
# (patch_flat × n_patches), and read_label returns the class index.
# Caller maintains its own image-index cursor (wrap with modulo).
#
# Cheap: one syscall per training step at smoke shape. For real
# Flowers-102 / Cars / CIFAR scale (~2k–8k images × 96×96 × 3) the
# per-image record is ~30 KB; throughput is way more than enough for
# CPU training. If we ever need batch>1 throughput, the right move is
# mmap the file once + slice — a small follow-up.

module ToyImageLoader
  F32_BYTES = 4
  I32_BYTES = 4

  # Read the patches for image at `image_idx`. `record_floats` =
  # patch_flat * n_patches (the number of f32s per image, written by
  # the preprocessor).
  def self.read_image(images_path, image_idx, record_floats)
    buf = Array.new(record_floats, 0.0)
    byte_offset = image_idx * record_floats * F32_BYTES
    got = TinyNN.tnn_read_f32_file(images_path, byte_offset, record_floats, buf)
    if got != record_floats
      puts "WARN: ToyImageLoader.read_image short read " + got.to_s +
           "/" + record_floats.to_s + " at idx=" + image_idx.to_s
    end
    buf
  end

  # Read the label (class index) for image `image_idx`. Returns Integer.
  def self.read_label(labels_path, image_idx)
    buf = [0]
    byte_offset = image_idx * I32_BYTES
    got = TinyNN.tnn_read_i32_file(labels_path, byte_offset, 1, buf)
    if got != 1
      puts "WARN: ToyImageLoader.read_label short read at idx=" + image_idx.to_s
      return -1
    end
    buf[0]
  end
end
