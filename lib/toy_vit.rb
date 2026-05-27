# E1 / GH#13 — ViT helpers, currently just the patch_embed composite.
#
# patch_embed: image [W, H, IC, N] → tokens [d_model, N_patches]
#
# Composition:
#   1. conv2d(kernel ne=[KW,KH,IC,OC], image ne=[W,H,IC,N], stride=patch)
#        → out ne=[OW, OH, OC, N]  where OW = (W - KW)/S + 1
#   2. permute(2, 0, 1, 3): bring OC to axis 0 → ne=[OC, OW, OH, N] (view)
#   3. cont_2d(OC, OW*OH): make contiguous + flatten spatial → ne=[OC, OW*OH]
#        (N=1 case; for batch the spatial flatten is the same but N stays).
#
# This produces the [d_model, N_patches] expected by the transformer
# block (same shape as the [d_model, T] sequence input on the LM side).

# ViTTinyConfig — hyperparams for the ViT-Tiny variant.
# Reference: vit_tiny_patch16_224 (timm) — image 224×224, patch 16,
# d_model 192, n_heads 3 (d_head 64), d_ff 768, n_layers 12.
# For the smoke we run at smaller dims (image 16×16, patch 4,
# d_model 64, n_heads 4, d_ff 128, n_layers 2).
class ViTTinyConfig
  attr_accessor :image_size, :patch_size, :num_channels,
                :d_model, :n_heads, :d_head, :d_ff, :n_layers,
                :num_classes, :ln_eps

  def initialize(image_size, patch_size, num_channels, d_model, n_heads,
                 d_ff, n_layers, num_classes, ln_eps)
    @image_size   = image_size
    @patch_size   = patch_size
    @num_channels = num_channels
    @d_model      = d_model
    @n_heads      = n_heads
    @d_head       = n_heads > 0 ? d_model / n_heads : 0
    @d_ff         = d_ff
    @n_layers     = n_layers
    @num_classes  = num_classes
    @ln_eps       = ln_eps
  end
end

module ToyVit
  # Returns the patch-embedding tensor handle. Caller is responsible
  # for set_param-ing the kernel (training) + downstream graph build.
  #
  # kernel: tnn 4D F32 persistent, ne=[KW, KH, IC, d_model]
  # image:  tnn 3D F32 (or 4D if N>1), ne=[W, H, IC, (N)]
  # patch:  kernel size = stride (no overlap, no padding).
  #
  # Currently assumes N=1 (single image). For batch>1 the cont_2d
  # would need to be cont_3d(OC, OW*OH, N); follow-up.
  def self.patch_embed(sess, kernel, image, patch_size)
    conv = TinyNN.tnn_conv_2d(sess, kernel, image,
                                patch_size, patch_size,   # stride
                                0, 0,                       # padding
                                1, 1)                       # dilation
    # ne=[OW, OH, OC, N] → ggml_permute moves source-axis-i to
    # result-axis-axis_i. To get result ne=[OC, OW, OH, N] (OC at 0,
    # OW at 1, OH at 2, N at 3) we map:
    #   source axis 0 (OW) → result axis 1
    #   source axis 1 (OH) → result axis 2
    #   source axis 2 (OC) → result axis 0
    #   source axis 3 (N)  → result axis 3
    # → permute(1, 2, 0, 3).
    perm = TinyNN.tnn_permute(sess, conv, 1, 2, 0, 3)
    # Flatten the OW*OH spatial axis into a single "patch" axis.
    ow = TinyNN.tnn_tensor_ne0(conv)
    oh = TinyNN.tnn_tensor_ne1(conv)
    oc = TinyNN.tnn_tensor_ne2(conv)
    TinyNN.tnn_cont_2d(sess, perm, oc, ow * oh)
  end
end
