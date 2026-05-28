#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "gguf>=0.10",
# ]
# ///
"""
GH#14 — PCA-initialize the projection lens W_proj for the Qwen
warm-start transfer.

Per the issue spec: "Adds a trainable projection Linear(1536, 1024)
after embedding lookup, PCA-initialised against donor activations."

We take the donor's `token_embd.weight` rows AS the donor activations
(they ARE the activations immediately after the embedding lookup —
no model-forward needed, no pytorch dependency). Compute PCA on the
centred rows, take the top d_model principal axes, and write a small
GGUF with one tensor: `lens.proj.weight` of shape
[d_model, donor_d_in].

The cache reads `lens.proj.weight` matching this shape (see
lib/llama_seq_forward_ffi.rb realize_for_random_init's E2.3 path).
At forward time: `matmul(W_proj, x_embed)` contracts donor_d_in →
d_model. With PCA init, the random target-shape model sees a
linear projection of the donor's most-informative dimensions
from step 0, instead of starting from random noise.

Usage:
  uv run prep/pca_init_qwen_lens.py \\
    --donor data/qwen25-1.5b-f32.gguf \\
    --d-model 1024 \\
    --out data/qwen_pca_lens.gguf

Output GGUF:
  - architecture: "lens"
  - keys:
      lens.donor_d_in     uint32   (= 1536 for Qwen-2.5-1.5B)
      lens.d_model        uint32   (= 1024 by default)
      lens.donor_source   string   (path to the donor GGUF)
  - tensor:
      lens.proj.weight    f32      shape [d_model, donor_d_in]

Memory: at d_model=1024, donor_d_in=1536, output = 6 MB. PCA
itself: centring the 151936×1536 embed table needs ~0.9 GB RAM;
the SVD of the centred matrix peaks at ~2 GB. Fits comfortably
on the GB10's 121 GB unified memory; on a 16 GB box you'd want
the truncated-SVD path (small follow-up if needed).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import gguf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", required=True,
                     help="path to donor GGUF (must contain token_embd.weight)")
    ap.add_argument("--d-model", type=int, required=True,
                     help="target d_model (= number of top PCA components to keep)")
    ap.add_argument("--out", required=True, help="output GGUF path")
    args = ap.parse_args()

    print(f"reading donor {args.donor} ...", flush=True)
    reader = gguf.GGUFReader(args.donor)

    embed = None
    for t in reader.tensors:
        if t.name == "token_embd.weight":
            embed = t
            break
    if embed is None:
        print(f"error: donor GGUF has no token_embd.weight tensor", file=sys.stderr)
        return 1

    # GGUF tensor data is exposed as a numpy view. Force to f32 +
    # contiguous; some quantized donors will need a dequant first
    # (not handled here — issue acceptance specifies the F32 donor
    # data/qwen25-1.5b-f32.gguf).
    X = np.array(embed.data, dtype=np.float32, copy=True)
    # GGUF convention: token_embd ne=[d_model, vocab] in ggml's
    # column-major layout, which numpy sees as shape (vocab, d_model)
    # row-major. Don't reshape — that IS the shape we want for PCA.
    if X.ndim != 2:
        print(f"error: token_embd.weight has shape {X.shape}, expected 2D",
              file=sys.stderr)
        return 1
    vocab, donor_d_in = X.shape
    print(f"  donor token_embd shape = ({vocab}, {donor_d_in})  "
          f"({X.nbytes / 1e6:.1f} MB)", flush=True)

    if args.d_model > donor_d_in:
        print(f"error: --d-model {args.d_model} > donor_d_in {donor_d_in}; "
              f"can't take {args.d_model} principal components from a "
              f"{donor_d_in}-dim space", file=sys.stderr)
        return 1

    # PCA via covariance-matrix eigendecomposition (rather than full
    # SVD of the centred data matrix). At 151936×1536 the data
    # matrix is 933 MB and full SVD needs several GB of work memory
    # plus minutes of wall time; the cov matrix is only 1536×1536
    # (9 MB) and eigh on it is ~1s. Mathematically equivalent: the
    # principal axes are the eigenvectors of cov = Xc.T Xc / (n-1),
    # sorted by descending eigenvalue.
    print(f"centring ({vocab}×{donor_d_in} = {X.nbytes / 1e6:.0f} MB) ...",
          flush=True)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    print(f"covariance matrix ({donor_d_in}×{donor_d_in}) ...", flush=True)
    # 64-bit accumulate to keep numerical noise down; cast back to f32
    # for storage. (Xc.T @ Xc) / (n-1) is the unbiased cov estimator.
    cov = (Xc.astype(np.float64).T @ Xc.astype(np.float64)) / (vocab - 1)
    print(f"eigendecomposition (top {args.d_model} of {donor_d_in}) ...",
          flush=True)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigvals
    # Top-k principal axes = eigenvectors with largest eigenvalues.
    # eigh sorts ascending, so flip.
    order = np.argsort(eigvals)[::-1]
    top_vals = eigvals[order[:args.d_model]]
    top_vecs = eigvecs[:, order[:args.d_model]]  # shape (donor_d_in, d_model)
    print(f"  eigvals: top={top_vals[0]:.4f}  "
          f"k={args.d_model}={top_vals[-1]:.4f}  "
          f"explained_var_ratio={top_vals.sum() / eigvals.sum():.4f}",
          flush=True)

    # W_proj has ggml shape (d_model, donor_d_in); top_vecs is
    # (donor_d_in, d_model) so transpose. Cast back to f32 for storage.
    W = top_vecs.T.astype(np.float32, copy=True)
    print(f"  W_proj shape = {W.shape}  ({W.nbytes / 1e6:.1f} MB)", flush=True)

    print(f"writing {args.out} ...", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(args.out, "lens")
    w.add_uint32("lens.donor_d_in", donor_d_in)
    w.add_uint32("lens.d_model",    args.d_model)
    w.add_string("lens.donor_source", args.donor)
    w.add_tensor("lens.proj.weight", W)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(f"wrote {args.out}", flush=True)
    print(f"  lens.proj.weight {W.shape}  ({W.nbytes:,} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
