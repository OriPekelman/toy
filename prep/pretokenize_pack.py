#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers",
#   "huggingface_hub",
#   "datasets",
#   "hf_xet",
# ]
# ///
"""
toy#129 item 1 — the F9 capacity fixture: real-tokenizer corpus pack.

Streams a HuggingFace dataset, tokenizes with a REAL tokenizer (GPT-2
BPE by default — vocab 50257, with in-repo Ruby runtime parity via
lib/toy/io/bpe.rb), and writes a SELF-DESCRIBING pack:

  bytes  0-3   "TOYC" magic
  bytes  4-7   u32 LE version (1)
  bytes  8-11  u32 LE vocab   (the tokenizer's vocab size)
  bytes 12-15  u32 LE reserved (0)
  bytes 16..   packed little-endian i32 tokens

The header is the vocab-unpinning contract (toy#129): runners read
vocab from the pack instead of trusting a flag, so a mismatched
--vocab can fail loud instead of reading past the embed table.
Headerless packs (ts_seqs*.bin, the toy#123 fixture) remain readable
everywhere — ToyCorpusLoader.probe_header distinguishes them.

EOS is appended between documents (next-token prediction must not
bridge unrelated docs). Writes are CHUNKED (10M-token flushes), so
50M+ packs stream without a giant resident buffer.

Usage:
  uv run prep/pretokenize_pack.py --tokens 200_000 \
    --out data/fineweb_gpt2_smoke.bin              # the committed gate smoke
  uv run prep/pretokenize_pack.py --tokens 50_000_000 \
    --out /srv/data/scratch/toy-corpora/fineweb_edu_gpt2_50m.bin

Defaults: tokenizer gpt2, dataset HuggingFaceFW/fineweb-edu
(sample-10BT config for cheap streaming), split train.
"""

import argparse
import array
import struct
import sys
from pathlib import Path

FLUSH_TOKENS = 10_000_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output pack path")
    ap.add_argument("--tokens", type=int, default=50_000_000,
                    help="number of tokens to write (default 50M)")
    ap.add_argument("--tokenizer", default="gpt2",
                    help="HF tokenizer id (default gpt2, vocab 50257)")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu",
                    help="HF dataset id")
    ap.add_argument("--config", default="sample-10BT",
                    help="dataset config (default sample-10BT)")
    ap.add_argument("--split", default="train", help="dataset split")
    ap.add_argument("--text-key", default="text", help="document text field")
    args = ap.parse_args()

    from transformers import AutoTokenizer  # noqa: WPS433
    from datasets import load_dataset       # noqa: WPS433

    print(f"loading tokenizer {args.tokenizer} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab = tok.vocab_size
    eos = tok.eos_token_id
    if eos is None:
        print("tokenizer has no EOS id — refusing (docs would bridge)")
        return 1
    print(f"vocab={vocab} eos={eos}", flush=True)

    print(f"streaming {args.dataset}"
          f"{'/' + args.config if args.config else ''} [{args.split}] ...",
          flush=True)
    ds = load_dataset(args.dataset, args.config, split=args.split,
                      streaming=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_tok = 0
    n_doc = 0
    buf = array.array("i")
    with open(out, "wb") as f:
        f.write(b"TOYC")
        f.write(struct.pack("<III", 1, vocab, 0))
        for doc in ds:
            ids = tok(doc[args.text_key])["input_ids"]
            buf.extend(ids)
            buf.append(eos)
            n_tok += len(ids) + 1
            n_doc += 1
            if len(buf) >= FLUSH_TOKENS:
                buf.tofile(f)
                buf = array.array("i")
                print(f"  ... {n_tok:,} tokens ({n_doc:,} docs)", flush=True)
            if n_tok >= args.tokens:
                break
        if len(buf) > 0:
            # trim to the exact target so the pack size is deterministic
            excess = n_tok - args.tokens
            if excess > 0:
                buf = buf[: len(buf) - excess]
                n_tok = args.tokens
            buf.tofile(f)

    hi = max(buf[-min(len(buf), 1000):]) if len(buf) else 0
    print(f"wrote {n_tok:,} tokens ({n_doc:,} docs) → {out}")
    print(f"  header: TOYC v1 vocab={vocab}; payload {n_tok * 4:,} bytes"
          f" (+16 header)")
    if hi >= vocab:
        print(f"  ERROR: token id {hi} >= vocab {vocab}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
