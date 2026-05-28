#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers",
#   "huggingface_hub",
#   "datasets",
#   "tiktoken",
#   "blobfile",
#   "hf_xet",
# ]
# ///
"""
toy#22 / GH#14 — FineWeb-Edu pretokenizer for the Qwen warm-start
transfer (#14 acceptance loop).

Streams the FineWeb-Edu dataset from HuggingFace, tokenizes each
document via the Qwen-2.5 tokenizer, and writes a packed i32
binary in the same wire format `lib/toy_corpus_loader.rb` reads
(matches `prep/pretokenize_corpus.py`'s output shape — raw
little-endian i32, no header).

Defaults:
  - Tokenizer: Qwen/Qwen2.5-0.5B  (same BPE vocab as 1.5B)
  - Dataset:   HuggingFaceFW/fineweb-edu  (default config)
  - Split:     train
  - Tokens:    10_000_000  (the GH#14 acceptance smoke target)

Usage:
  uv run prep/pretokenize_fineweb_edu.py --out data/fineweb_edu_10m.bin
  uv run prep/pretokenize_fineweb_edu.py --tokens 100_000 \\
    --out data/fineweb_edu_smoke.bin

Notes:
  - In-memory buffer until output flush. At 10M tokens × 4 bytes =
    40 MB; fits easily. For 1B+ token runs, switch to chunked
    writes (separate follow-up — current scope is the GH#14
    acceptance smoke).
  - EOS tokens are appended between documents so the trainer's
    next-token prediction doesn't bridge unrelated docs.
  - First-run cache populates ~/.cache/huggingface/; subsequent
    runs are fast.
"""

import argparse
import array
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output i32 binary path")
    ap.add_argument("--tokens", type=int, default=10_000_000,
                     help="number of tokens to write (default 10M)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B",
                     help="HF tokenizer id (Qwen2.5 family shares vocab)")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu",
                     help="HF dataset id")
    ap.add_argument("--config", default=None,
                     help="dataset config (e.g. 'sample-10BT' for the smaller subset; "
                          "default = default config of the dataset)")
    ap.add_argument("--split", default="train", help="dataset split")
    args = ap.parse_args()

    from transformers import AutoTokenizer  # noqa: WPS433
    from datasets import load_dataset       # noqa: WPS433

    print(f"loading tokenizer {args.tokenizer} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tok.eos_token_id
    if eos_id is None:
        print("warning: tokenizer has no eos_token_id; documents will not be separated",
              file=sys.stderr)

    print(f"streaming {args.dataset} (config={args.config}, split={args.split}) ...",
          flush=True)
    ds_kwargs = {"split": args.split, "streaming": True}
    if args.config:
        ds_kwargs["name"] = args.config
    ds = load_dataset(args.dataset, **ds_kwargs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    buf = array.array("i")
    n_docs = 0
    last_print = 0
    target = args.tokens

    for ex in ds:
        text = ex.get("text") or ex.get("content") or ""
        if not text:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        buf.extend(ids)
        if eos_id is not None:
            buf.append(eos_id)
        n_docs += 1
        if len(buf) >= target:
            break
        if len(buf) - last_print >= 100_000:
            print(f"  {len(buf):>11,} tokens / {n_docs:>7,} docs", flush=True)
            last_print = len(buf)

    # Truncate exactly to target if we overshot.
    if len(buf) > target:
        head = buf[:target]
        n_out = target
    else:
        head = buf
        n_out = len(buf)

    with open(args.out, "wb") as f:
        head.tofile(f)

    print(f"wrote {n_out:,} tokens ({n_docs:,} docs) → {args.out}", flush=True)
    print(f"  i32 packed = {n_out * 4:,} bytes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
