#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "timm",
#   "torch",
#   "gguf>=0.10",
#   "numpy",
# ]
# ///
"""
E1.4 / GH#13 — extract vit_tiny_patch16_224 (IN-21k AugReg) into a
GGUF that toy's ViTTinyForwardFFICache can load.

Per E1's pure-embedding spec, only THREE tensors need to round-trip
for the warm-start arms:

  - patch_embed.proj.weight     [d_model, IC*KH*KW]  flat-linear form
  - patch_embed.proj.bias       [d_model]
  - pos_embed                   [N_patches+1, d_model]
  - cls_token                   [1, d_model]

(Block weights / FFN / final norm / head are randomly initialised in
every arm — scratch, warm-frozen, warm-trainable — so we don't emit
them. timm's stride=16 conv → our flat-linear linear is a single
reshape: timm.weight is (OC, IC, KH, KW); flatten to (OC, IC*KH*KW).)

Usage:
  uv run prep/extract_vit_tiny.py
  uv run prep/extract_vit_tiny.py data/vit_tiny_donor.gguf
"""

import sys

import numpy as np
import timm
import torch
import gguf


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else "data/vit_tiny_donor.gguf"

    model = timm.create_model("vit_tiny_patch16_224.augreg_in21k", pretrained=True)
    sd = model.state_dict()

    image_size  = 224
    patch_size  = 16
    num_channels = 3
    d_model     = 192
    n_heads     = 3
    n_layers    = 12
    d_ff        = 768
    n_patches_side = image_size // patch_size
    n_patches      = n_patches_side * n_patches_side
    seq_t          = n_patches + 1  # 196 + 1 cls

    # Reshape conv kernel (OC, IC, KH, KW) → (d_model, IC*KH*KW).
    patch_w_4d = sd["patch_embed.proj.weight"].numpy()
    assert patch_w_4d.shape == (d_model, num_channels, patch_size, patch_size), \
        f"unexpected patch_embed shape {patch_w_4d.shape}"
    patch_w = patch_w_4d.reshape(d_model, -1).astype(np.float32)
    assert patch_w.shape == (d_model, num_channels * patch_size * patch_size)

    patch_b = sd["patch_embed.proj.bias"].numpy().astype(np.float32)

    # pos_embed in timm: (1, 197, 192) → squeeze → (197, 192)
    pos_embed = sd["pos_embed"].numpy().squeeze(0).astype(np.float32)
    assert pos_embed.shape == (seq_t, d_model), f"pos_embed {pos_embed.shape}"

    # cls_token in timm: (1, 1, 192) → squeeze → (192,) — but our cache
    # allocates [1, d_model] via tnn_input_2d_f32_persistent, so emit
    # as (1, d_model).
    cls_token = sd["cls_token"].numpy().reshape(1, d_model).astype(np.float32)

    writer = gguf.GGUFWriter(out_path, "vit-tiny")

    writer.add_uint32("vit.image_size",        image_size)
    writer.add_uint32("vit.patch_size",        patch_size)
    writer.add_uint32("vit.num_channels",      num_channels)
    writer.add_uint32("vit.d_model",           d_model)
    writer.add_uint32("vit.n_heads",           n_heads)
    writer.add_uint32("vit.d_ff",              d_ff)
    writer.add_uint32("vit.n_layers",          n_layers)
    writer.add_uint32("vit.num_patches",       n_patches)
    writer.add_uint32("vit.seq_t",             seq_t)
    writer.add_string("vit.timm_source",
                       "vit_tiny_patch16_224.augreg_in21k")

    writer.add_tensor("patch_embed.proj.weight", patch_w)
    writer.add_tensor("patch_embed.proj.bias",   patch_b)
    writer.add_tensor("pos_embed",               pos_embed)
    writer.add_tensor("cls_token",               cls_token)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"wrote {out_path}")
    print(f"  patch_embed.proj.weight  shape={patch_w.shape}  ({patch_w.size * 4} bytes)")
    print(f"  patch_embed.proj.bias    shape={patch_b.shape}  ({patch_b.size * 4} bytes)")
    print(f"  pos_embed                shape={pos_embed.shape}  ({pos_embed.size * 4} bytes)")
    print(f"  cls_token                shape={cls_token.shape}  ({cls_token.size * 4} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
