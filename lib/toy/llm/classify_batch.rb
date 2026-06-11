# lib/toy/llm/classify_batch.rb — Toy::LLM::ClassifyBatch: the
# image-classification SIBLING of TrainingBatch (toy#73 item 3).
#
# The ViT recipe's step! quartet is (m_image, cls_idx, m_labels, m_hp)
# where the LLM recipes take (seq_ids, positions, labels, hp). Before
# this class the ViT example hand-filled the image Mat and the class
# one-hot with raw loops and NO validation — a short patches array
# silently trained on stale/zero pixels, an out-of-range label
# silently scattered off the row. ClassifyBatch fails LOUD instead:
#
#   batch = Toy::LLM::ClassifyBatch.new(NUM_CLASSES, patch_flat, n_patches)
#   batch.fill!(patches, label)        # validates + rebuilds both
#   batch.hp = adamw.hp_for_step(step) # caller-owned (optimizer knob)
#   recipe.step!(batch.image, batch.cls_idx, batch.labels,
#                batch.hp, step == 0)
#
# The ctor pins the record shape (patch_flat x n_patches — what
# ViTTinyEngine's t_image expects) and builds the cls_idx vector ([0]:
# the class token lives at position 0 in every current ViT graph).
# fill! validates the patches length and the label range, copies the
# patches into the image Mat, and rebuilds the one-hot labels via
# Toy::Labels.one_hot_class — byte-identical to the hand loops it
# replaces. ToyImageLoader.read_label returns -1 on a short read, so a
# torn labels.bin fails loud in fill!, not as a silent all-zero row.
#
# Pure-Ruby: NO TinyNN / FFI calls — only Mat (via Toy::Labels) — so it
# is SHARED (not mirrored) across CPU / CUDA / Metal, like
# TrainingBatch / AdamW / Labels. (ViT training itself is CPU-only
# this arc; the batch object is backend-free.)
#
# Spinel hygiene: PLAIN class (NO Struct.new — landmine #16), explicit
# ctor with NO default args (landmine #4), while-loops, NO #{}
# interpolation, uniquely cb_-prefixed scalar ivars.

require_relative "labels"

module Toy; module LLM
  class ClassifyBatch
    attr_accessor :cb_classes, :cb_patch_flat, :cb_n_patches,
                  :image, :cls_idx, :labels, :hp

    def initialize(num_classes, patch_flat, n_patches)
      if num_classes <= 0
        raise "ClassifyBatch: num_classes must be positive, got " +
              num_classes.to_s
      end
      if patch_flat <= 0 || n_patches <= 0
        raise "ClassifyBatch: patch_flat/n_patches must be positive, got " +
              patch_flat.to_s + "/" + n_patches.to_s
      end
      @cb_classes    = num_classes
      @cb_patch_flat = patch_flat
      @cb_n_patches  = n_patches

      # The image Mat at the engine's expected shape; zero until fill!.
      @image = Mat.new(patch_flat, n_patches)

      # The class-token read position: 0 in every current ViT graph.
      @cls_idx = [0]

      # Zero labels at the final shape until fill!; hp is caller-owned
      # (Mat(1,7) from Toy::AdamW) — zero placeholder pins the type.
      @labels = Mat.new(1, num_classes)
      @hp     = Mat.new(1, 7)
    end

    # Validate + load one (image, label) record. `patches` is the flat
    # f32 record from ToyImageLoader.read_image (length must be exactly
    # patch_flat * n_patches); `label` is the class index from
    # read_label (0...num_classes — read_label's -1 short-read sentinel
    # fails the range check loud). Returns nil.
    def fill!(patches, label)
      record_f = @cb_patch_flat * @cb_n_patches
      if patches.length != record_f
        raise "ClassifyBatch#fill!: patches length " +
              patches.length.to_s + " != patch_flat*n_patches " +
              record_f.to_s
      end
      i = 0
      while i < record_f
        @image.flat[i] = patches[i]
        i = i + 1
      end
      @labels = Toy::Labels.one_hot_class(@cb_classes, label)
      nil
    end
  end
end; end
