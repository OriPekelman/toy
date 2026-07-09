#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "torch",
# ]
# ///
"""
E1.1 / GH#13 — PyTorch parity check for toy's Conv2D FFI.

Reads the JSON dump from tinynn/ab_smoke_conv2d, reshapes the inputs +
expected output, runs PyTorch's nn.functional.conv2d on the same inputs,
and compares element-wise.

ggml's storage convention is column-major:
  kernel ne=[KW, KH, IC, OC] → numpy reshape with order="F" of shape (KW,KH,IC,OC)
  data   ne=[W,  H,  IC, N ] → reshape order="F" of shape (W,H,IC,N)
  output ne=[OW, OH, OC, N ] → reshape order="F" of shape (OW,OH,OC,N)

PyTorch's nn.functional.conv2d expects:
  input  (N, IC, H, W)
  weight (OC, IC, KH, KW)
  output (N, OC, OH, OW)

Usage:
  ./tinynn/ab_smoke_conv2d
  uv run tinynn/conv2d_parity.py /tmp/conv2d_ref.json
"""

import json
import sys

import numpy as np
import torch
import torch.nn.functional as F


def main(path: str) -> int:
    with open(path) as f:
        d = json.load(f)

    kw, kh, ic, oc = d["kernel_ne"]
    w, h, ic2, n = d["input_ne"]
    ow, oh, oc2, n2 = d["output_ne"]
    sH, sW = d["stride"]
    pH, pW = d["padding"]
    dH, dW = d["dilation"]

    assert ic == ic2, f"kernel IC {ic} vs input IC {ic2} mismatch"
    assert oc == oc2, f"kernel OC {oc} vs output OC {oc2} mismatch"
    assert n == n2, f"input N {n} vs output N {n2} mismatch"

    # Reshape ggml column-major flat into the named axes order, then
    # transpose to PyTorch's NCHW / OC,IC,KH,KW.
    k_ggml = np.array(d["kernel"], dtype=np.float64).reshape((kw, kh, ic, oc), order="F")
    x_ggml = np.array(d["input"],  dtype=np.float64).reshape((w, h, ic, n),   order="F")
    y_ggml = np.array(d["output"], dtype=np.float64).reshape((ow, oh, oc, n), order="F")

    # PyTorch axis order.
    kernel_pt = torch.from_numpy(k_ggml.transpose(3, 2, 1, 0).copy()).float()  # (OC, IC, KH, KW)
    input_pt  = torch.from_numpy(x_ggml.transpose(3, 2, 1, 0).copy()).float()  # (N, IC, H, W)
    output_ggml_pt = torch.from_numpy(y_ggml.transpose(3, 2, 1, 0).copy()).float()  # (N, OC, OH, OW)

    pt_out = F.conv2d(input_pt, kernel_pt,
                      stride=(sH, sW), padding=(pH, pW), dilation=(dH, dW))

    expected_shape = (n, oc, oh, ow)
    assert tuple(pt_out.shape) == expected_shape, \
        f"PyTorch output {tuple(pt_out.shape)} vs expected {expected_shape}"

    diff = (pt_out - output_ggml_pt).abs()
    max_abs = float(diff.max())
    rel = float(diff.max() / (output_ggml_pt.abs().max() + 1e-9))

    print(f"shape ggml={tuple(output_ggml_pt.shape)} pytorch={tuple(pt_out.shape)}")
    print(f"max abs diff = {max_abs:.3e}")
    print(f"max rel diff = {rel:.3e}")
    print(f"ggml  y[0,0,:,:] = {output_ggml_pt[0, 0]}")
    print(f"torch y[0,0,:,:] = {pt_out[0, 0]}")

    # f32 round-trip tolerance.
    if max_abs > 1e-5:
        print("FAIL: max abs diff exceeds 1e-5")
        return 1
    print("PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/conv2d_ref.json"))
