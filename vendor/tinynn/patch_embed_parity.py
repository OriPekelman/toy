#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "torch",
# ]
# ///
"""
E1.2 / GH#13 — PyTorch parity check for ToyVit.patch_embed.

Reproduces ViT-Tiny's first slice: conv2d with stride=patch_size +
flatten spatial dims → [d_model, N_patches]. Compares against ggml's
output produced by tinynn/ab_smoke_patch_embed.
"""

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F


def main(path: str) -> int:
    with open(path) as f:
        d = json.load(f)

    w, h, ic, n = d["image_ne"]
    kw, kh, ic2, oc = d["kernel_ne"]
    out_oc, out_np = d["output_ne"]
    patch = d["patch_size"]

    assert ic == ic2
    assert oc == out_oc
    # Expected N_patches = (W/patch) * (H/patch)
    ow = (w + 0 - kw) // patch + 1
    oh = (h + 0 - kh) // patch + 1
    expected_np = ow * oh
    assert out_np == expected_np, f"out N_patches {out_np} vs expected {expected_np}"

    k_ggml = np.array(d["kernel"], dtype=np.float64).reshape((kw, kh, ic, oc), order="F")
    x_ggml = np.array(d["image"],  dtype=np.float64).reshape((w, h, ic, n),   order="F")
    y_ggml = np.array(d["output"], dtype=np.float64).reshape((oc, out_np),    order="F")

    # PyTorch axis order: (OC, IC, KH, KW) and (N, IC, H, W).
    kernel_pt = torch.from_numpy(k_ggml.transpose(3, 2, 1, 0).copy()).float()
    image_pt  = torch.from_numpy(x_ggml.transpose(3, 2, 1, 0).copy()).float()

    conv = F.conv2d(image_pt, kernel_pt, stride=patch, padding=0, dilation=1)
    # conv shape (N, OC, OH, OW) → flatten spatial → (N, OC, N_patches)
    pt_out = conv.reshape(n, oc, -1)
    # Take N=0 → (OC, N_patches).
    pt_out_first = pt_out[0]   # (OC, N_patches)
    y_ggml_pt = torch.from_numpy(y_ggml.copy()).float()  # (OC, N_patches)

    diff = (pt_out_first - y_ggml_pt).abs()
    max_abs = float(diff.max())
    rel = float(diff.max() / (y_ggml_pt.abs().max() + 1e-9))

    print(f"shape ggml={tuple(y_ggml_pt.shape)} pytorch={tuple(pt_out_first.shape)}")
    print(f"max abs diff = {max_abs:.3e}")
    print(f"max rel diff = {rel:.3e}")
    print(f"ggml  y[0, :8] = {y_ggml_pt[0, :8]}")
    print(f"torch y[0, :8] = {pt_out_first[0, :8]}")

    if max_abs > 1e-5:
        print("FAIL: max abs diff exceeds 1e-5")
        return 1
    print("PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/patch_embed_ref.json"))
