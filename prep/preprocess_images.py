#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pillow",
# ]
# ///
"""
E1.5 / GH#13 — image-dataset preprocessor for the ViT-Tiny training
path. Produces two parallel binary files:

  images.bin   raw f32, n_images × patch_flat × n_patches
  labels.bin   raw i32, n_images
  meta.json    n_images, image_size, patch_size, num_classes, n_patches, patch_flat

The Ruby loader (lib/toy_image_loader.rb) reads single records by
index via tnn_read_f32_file / tnn_read_i32_file. The patch-flatten
math matches lib/vit_tiny_forward_ffi.rb's flat-linear patch_embed:
each image is pre-flattened to [IC*P*P, N_patches] before disk.

Two modes:

  MODE=synthetic (default) — generates N random images / random labels.
                              Useful for the smoke and CI.
  MODE=real DATA_DIR=/path  — walks DATA_DIR/<class_name>/*.{jpg,png},
                              decodes via PIL, resizes to (image_size,
                              image_size), normalises, and writes
                              packed records. Class names are sorted
                              alphabetically; that's the label index.

Usage:
  uv run prep/preprocess_images.py                                  # synth → data/vit_smoke/
  IMAGE_SIZE=96 PATCH_SIZE=16 N_CLASSES=102 \\
    uv run prep/preprocess_images.py                                # Flowers-shape synth
  MODE=real DATA_DIR=/srv/flowers/train OUT_DIR=data/flowers_train \\
    uv run prep/preprocess_images.py                                # real Flowers-102
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def normalize(arr: np.ndarray) -> np.ndarray:
    """[0,255] → centered [-1, 1]-ish. Cheap; real recipe (timm AugReg)
    uses imagenet-mean/std but for a smoke this is fine."""
    return (arr.astype(np.float32) / 127.5) - 1.0


def patchify(img: np.ndarray, patch_size: int) -> np.ndarray:
    """img: (H, W, IC); returns (IC*P*P, N_patches) in *ggml column-major*
    flat order matching the patch_embed expectation."""
    h, w, ic = img.shape
    ps = patch_size
    assert h % ps == 0 and w % ps == 0
    nh, nw = h // ps, w // ps
    # Slice into patches. Order: row-major over (nh, nw) - matches
    # tnn_cont_2d's flatten direction for ViT seq.
    # img[i*P:(i+1)*P, j*P:(j+1)*P, :].flatten(order="C") gives a
    # patch_flat vector. Stack into a (n_patches, patch_flat) array
    # then transpose+ravel for our (patch_flat, n_patches) flat layout.
    patches = np.zeros((nh * nw, ic * ps * ps), dtype=np.float32)
    for i in range(nh):
        for j in range(nw):
            patches[i * nw + j] = img[i*ps:(i+1)*ps, j*ps:(j+1)*ps, :].reshape(-1)
    # (n_patches, patch_flat) → (patch_flat, n_patches) in C order
    # AND we want it laid out as flat[k + p*patch_flat] (which is what
    # ggml ne=[patch_flat, n_patches] column-major wants — patch_flat
    # is fastest). transpose + ravel("F") gives exactly that.
    return patches.T.ravel(order="F")


def synth_one(seed: int, image_size: int, num_channels: int) -> np.ndarray:
    """Deterministic synthetic image for the smoke."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(image_size, image_size, num_channels), dtype=np.uint8)


def main() -> int:
    mode        = os.environ.get("MODE", "synthetic")
    image_size  = int(os.environ.get("IMAGE_SIZE",  "16"))
    patch_size  = int(os.environ.get("PATCH_SIZE",  "4"))
    num_channels = int(os.environ.get("NUM_CHAN",   "3"))
    num_classes = int(os.environ.get("N_CLASSES",   "10"))
    n_images    = int(os.environ.get("N_IMAGES",    "100"))
    out_dir     = Path(os.environ.get("OUT_DIR",    "data/vit_smoke"))
    out_dir.mkdir(parents=True, exist_ok=True)

    assert image_size % patch_size == 0
    n_patches  = (image_size // patch_size) ** 2
    patch_flat = num_channels * patch_size * patch_size

    print(f"mode={mode} image={image_size} patch={patch_size} "
          f"chan={num_channels} classes={num_classes} n={n_images}")
    print(f"  per-image record: {patch_flat} × {n_patches} × 4 = "
          f"{patch_flat * n_patches * 4} bytes")
    print(f"  out_dir={out_dir}")

    images_path = out_dir / "images.bin"
    labels_path = out_dir / "labels.bin"
    meta_path   = out_dir / "meta.json"

    with open(images_path, "wb") as fimg, open(labels_path, "wb") as flab:
        if mode == "synthetic":
            for i in range(n_images):
                img = normalize(synth_one(i, image_size, num_channels))
                patches = patchify(img, patch_size)
                fimg.write(patches.astype(np.float32).tobytes())
                label = i % num_classes
                flab.write(np.array([label], dtype=np.int32).tobytes())
        elif mode == "real":
            data_dir = Path(os.environ.get("DATA_DIR", ""))
            if not data_dir.exists():
                print(f"DATA_DIR={data_dir} doesn't exist", file=sys.stderr)
                return 2
            classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
            class_to_idx = {c: i for i, c in enumerate(classes)}
            count = 0
            for cls in classes:
                for img_path in sorted((data_dir / cls).iterdir()):
                    if not img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        continue
                    img = Image.open(img_path).convert("RGB").resize((image_size, image_size))
                    arr = normalize(np.asarray(img))
                    patches = patchify(arr, patch_size)
                    fimg.write(patches.astype(np.float32).tobytes())
                    flab.write(np.array([class_to_idx[cls]], dtype=np.int32).tobytes())
                    count += 1
                    if count >= n_images:
                        break
                if count >= n_images:
                    break
            n_images = count   # actual count written
        else:
            print(f"unknown MODE={mode}", file=sys.stderr)
            return 1

    meta = {
        "n_images":     n_images,
        "image_size":   image_size,
        "patch_size":   patch_size,
        "num_channels": num_channels,
        "num_classes":  num_classes,
        "n_patches":    n_patches,
        "patch_flat":   patch_flat,
        "mode":         mode,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {images_path}")
    print(f"wrote {labels_path}")
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
