# Phase 0.7 — Qwen2.5 acceptance gates

**Status:** shipped 2026-05-21
**Predecessor:** Phase 0.5 (`lib/transformer_lm.rb` delegating wrapper)

## What it is

`demos/qwen25_acceptance.rb` — a Spinel-compiled gate that:

1. Loads each Qwen2.5 GGUF size via `ToyLM.new(arch, :cpu).load(gguf)`.
2. Runs greedy decode on the canonical prompt `[9707, 11, 847, 829, 374]`
   ("Hello, my name is" in Qwen2 tokens).
3. Compares generated token IDs to a golden array recorded in the source.
4. Exits non-zero (raises) on any mismatch.

Run before tagging a release / merging anything that touches:

- `lib/transformer_lm.rb`
- `lib/toy_smollm2_ffi_kv.rb` (the underlying graph builder)
- `lib/toy_smollm2_loader.rb` (Phase 1/2/3 weight load paths)
- `lib/tinynn.rb` or `tinynn/tinynn_ggml.c` (the FFI shim)
- `lib/arch.rb` or `lib/arch/qwen2.rb`

## What's gated, and why

| Gate | Size | Path | Why |
|---|---|---|---|
| `qwen25-0.5b-native`    | 0.5 B | F32 mmap | Smallest f32 — fast feedback (~2 s) |
| `qwen25-0.5b-native-q8` | 0.5 B | Q8 mmap  | Proves Q8-stays-Q8 dequant path matches F32 byte-for-byte |
| `qwen25-1.5b-native`    | 1.5 B | F32 mmap | Mid-size; catches GQA n_kv=2 quirks at d=1536 |
| `qwen25-3b-native`      | 3 B   | F32 mmap | Largest f32 we gate; runtime budget ~30 s total |

**Not gated:**

- **7 B** — at ~1 s/token on CPU, gating 8 tokens adds 15 s for minimal
  extra signal. The 0.5B f32-vs-Q8 byte-for-byte match already proves
  the dequant path; 3B exercises the largest f32 graph. Run it manually
  before a release: `GGUF=data/qwen25-7b-q8_0.gguf N_NEW=8 ./demos/qwen25_transformer_lm`.
- **CUDA paths** — same graph as CPU on this hardware, separately covered
  by the `*_cuda` demos. Adding to the gate doubles wall time without
  changing what it detects.
- **Sampling-mode generation** — non-determinism makes IDs hash-unstable.
  Greedy only.

## The locked goldens

Captured 2026-05-21 on gx10 CPU with Spinel @ 7beeb54 + tinynn @
commit `472cf86` (F1.1 fix). All four passed at runtime.

```
qwen25-0.5b-native:    9707 11 847 829 374 264 220 16 15 1042 2310 8171 13 358 614 264 3405
qwen25-0.5b-native-q8: 9707 11 847 829 374 264 220 16 15 1042 2310 8171 13 358 614 264 3405
qwen25-1.5b-native:    9707 11 847 829 374 71 6255 323 358 1079 264 220 16 17 339 11972 5458
qwen25-3b-native:      9707 11 847 829 374 323 358 1079 264 220 16 15 339
```

Decoded text for the curious (cross-checked against the model
tokenizer; not asserted by the gate):

- 0.5B: "Hello, my name is a 10 year old boy. I have a question"
- 1.5B: "Hello, my name is hammad and I am a 12th grade student"
- 3B:   "Hello, my name is and I am a 10th gr"

The 0.5B output reproduces the handoff doc's recorded sample
("Hello, my name is a 10 year old boy. I have a question about my
hair"). The 1.5B sample matches too. The 3B sample diverges from the
handoff's "Hello, my name is and I am a 10th grader. I am currently
taking AP" at token 13 (`339` = " gr" vs continuing) — but the gate
captures the current state, not a stale reading. This means **at the
gate's N_NEW the first 8 generated tokens match** what the handoff
shows. If the gate's golden ever needs updating because an
intentional code change moved the output, follow the recipe below.

## How to refresh goldens

When you intentionally change graph code in a way that legitimately
moves the output (a Q-bias placement fix, an RMS-eps adjustment, a
matmul order swap), the gate will fail. To re-baseline:

1. Build the new code: `make tinynn/libtinynn_ggml.a && make qwen25_acceptance`.
2. Run each gated GGUF manually:
   ```
   GGUF=data/qwen25-0.5b-native.gguf N_NEW=12 ./demos/qwen25_transformer_lm
   GGUF=data/qwen25-0.5b-native-q8.gguf N_NEW=12 ./demos/qwen25_transformer_lm
   GGUF=data/qwen25-1.5b-native.gguf N_NEW=12 ./demos/qwen25_transformer_lm
   GGUF=data/qwen25-3b-native.gguf N_NEW=8 ./demos/qwen25_transformer_lm
   ```
3. Cross-check that the new outputs make qualitative sense (don't just
   blindly accept anything that doesn't crash).
4. Paste the new arrays into `demos/qwen25_acceptance.rb`.
5. Commit with a clear "**baseline refresh**" tag in the message
   explaining what moved and why.

**Don't refresh after a non-graph change.** If the gate fails on a
docs PR, you have a bug, not a stale baseline.

## Why token-ID gates rather than text

The project's `TransformerLM.generate` returns token IDs by design
(server-side tokenizer is opt-in via the Phase D2/D3 BPE encoder, not
the default). IDs are bit-stable across runs and Spinel rebuilds; text
comparison would require a working encoder for every model family
across every gate run. Keep the gate orthogonal to the tokenizer.

## Why we kept this lightweight

The full project has ~20 demos; gating each would take 5+ minutes per
run and turn into a chore nobody runs. Four gates × ~30 s total fits
into the "always run before pushing" budget. The cost of an undetected
inference regression is high (silently bad outputs are the
worst-of-both); the cost of running this gate is ~30 s. Easy trade.

## What this does NOT replace

- **Bench numbers.** Acceptance is correctness, not perf. Use the
  `persistent_bench*` and the not-yet-built deeper CUDA bench
  (task #67) for that.
- **Smoke tests.** `tinynn/ab_smoke_*` cover the FFI primitives.
  This gate covers the integrated graph.
- **Llama / Mistral acceptance.** Different model family, different
  arch path. If `lib/arch/llama.rb` or the Llama tokenizer changes,
  add a parallel `demos/llama_acceptance.rb` following the same
  pattern. Not in scope here.
