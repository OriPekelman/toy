#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
E2.4 / GH#14 — pretokenize a text corpus of integer token sequences
into a packed i32 binary file for streaming during training.

Input format: one line per sequence, space-separated decimal token
IDs. Default IN = data/ts_seqs.txt (TinyStories pretokenized by
prep/prep_tinystories.rb).

Output format: raw little-endian i32 tokens, no header. The Ruby
loader (lib/toy_corpus_loader.rb) reads N tokens from a byte offset
and the caller tracks position.

Usage:
  uv run prep/pretokenize_corpus.py                          # ts_seqs.txt → ts_seqs.bin
  uv run prep/pretokenize_corpus.py in.txt out.bin
"""

import array
import sys


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "data/ts_seqs.txt"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/ts_seqs.bin"

    arr = array.array("i")
    n_seq = 0
    n_tok = 0
    with open(src) as f:
        for line in f:
            for tok in line.split():
                arr.append(int(tok))
                n_tok += 1
            n_seq += 1

    with open(dst, "wb") as f:
        arr.tofile(f)

    print(f"wrote {n_tok} tokens ({n_seq} sequences) → {dst}")
    print(f"  i32 packed = {n_tok * 4} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
