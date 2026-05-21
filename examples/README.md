# examples/

Short, focused entry points. Each file is one runnable Ruby program that
Spinel compiles to a native binary. No Python, no PyTorch, no runtime
to install — `make`, run.

## What's here

| File | What it does | Runtime |
|---|---|---|
| `01_inference.rb`       | Load a GGUF, generate 16 tokens. Swap GGUF env var for any supported model. | seconds |
| `02_train_custom_gpt.rb`| Train a tiny GPT from scratch on TinyStories. Bumping EPOCHS makes a model that writes English. | seconds for the smoke; minutes for real |
| `03_finetune_lora.rb`   | Fine-tune SmolLM2-135M on CUDA with rank-16 LoRA on Q. CE drops 7.5 → 0.09 in 20 AdamW steps. | ~30 s on GB10 |
| `04_serve_http.rb`      | HTTP API: `POST /generate` with `{prompt:[ids], n:int}`, get JSON `{ids:[...]}`. | server |

## First run

```sh
# One-time vendor setup — clones ggml + applies our local patches.
make setup-ggml

# Pull a small model. Pre-converted GGUFs are in data/ already for the
# main sizes; if you want a fresh one:
./prep/convert_smollm2_to_gguf.py        # → data/smollm2-135m-*.gguf

# Build + run the smallest example.
make example_inference
GGUF=data/qwen25-0.5b-native.gguf ./examples/example_inference
# → Arch(qwen2, vocab=151936, d=896, ...)
# → ids: 9707 11 847 829 374 264 220 16 15 ...
```

That's it — open weights running on CPU through a self-contained
native binary. No virtualenv to remember to source.

## Training your own

```sh
./prep/prep_tinystories.rb               # one-time: builds vocab + sequences
make example_train
./examples/example_train
```

The defaults are tiny so it finishes in seconds. To train a model
that actually writes English-shaped stories, bump `EPOCHS=200` and
`D_MODEL=128`, then go make coffee. Toy::Trainer (`lib/toy_trainer.rb`)
absorbs the per-step boilerplate so what you read in `02_train_*.rb`
**is** the algorithm.

## Fine-tuning

```sh
# Build the CUDA stack (~few minutes the first time).
make setup-ggml-cuda

# Build + run the LoRA example.
make example_finetune_cuda
./examples/example_finetune_cuda
# → step 1: CE=7.519
# → step 20: CE=0.090
```

The example fine-tunes 540 LoRA-Q matrices (270 per-head pairs × 2)
while leaving the 134 M base weights frozen. Real SFT (varied prefixes
+ multi-position + a dataset) needs a sequence-mode forward graph —
see `docs/design/phase-f1-2-step6-status.md` for the roadmap; the
mechanics shown here are the same.

CPU LoRA training is currently miscalibrated due to a ggml-cpu sched
bug ([`ggml-org/ggml#1501`](https://github.com/ggml-org/ggml/issues/1501));
stick to the `_cuda` example until that lands upstream.

## Serving

```sh
make example_serve
./examples/example_serve &
curl -s localhost:4567/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":[12092,4845,253,1429],"n":16}'
```

Token-IDs in, token-IDs out — tokenizer work belongs client-side (or
wire `lib/tokenizer.rb` in if needed). One binary, no runtime.

**Heads-up (2026-05-21):** Tep (the HTTP layer) currently has a
compatibility regression with the latest Spinel where handler bodies
render as a placeholder instead of the value the handler returned.
The binary builds and starts; responses are wrong. Tracked in the
Tep repo; the example's shape is preserved for when it's fixed.

## Where to go after the examples

- `docs/design/arch-struct.md` — design for the generic `TransformerLM`
- `demos/` — exhaustive per-model and per-feature drivers
- `tinynn/` — C+CUDA shim over ggml (FFI bridge, KV cache, mmap loader)
- `tep_demo/` — full OpenAI-compatible HTTP API (the `04_serve` example is its lite cousin)

Issues + design notes live in `docs/design/`. Upstream contributions
(ggml patches, Spinel issues) live in `docs/upstream/` and
`docs/spinel-issues/`.

## Two-line takeaway

Each example is < 80 lines of Ruby. The build is one `make` target
each. The output is a single binary with the model, the math, and (in
`04`) the HTTP server linked in. That's the deal.
