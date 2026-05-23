# examples/

Short, focused entry points. Each file is one runnable Ruby program
that Spinel compiles to a native binary. No Python, no PyTorch, no
runtime to install — `make`, run.

## What's here

| File | What it does | Runtime |
|---|---|---|
| `01_inference.rb`        | Load a GGUF, generate 16 tokens. Default points at `data/smollm2-135m-f32.gguf`. Swap `GGUF=` for any supported model. | seconds |
| `02_train_custom_gpt.rb` | Train a tiny GPT from scratch on TinyStories. Bumping EPOCHS makes a model that writes English. | ~3 min default |
| `03_finetune_lora.rb`    | LoRA / **QLoRA** fine-tune via the sequence-mode forward graph (CPU). Needs a native-layout GGUF — see prep step below. | ~30 s |
| `03_finetune_lora_cuda.rb` | CUDA mirror of the above. F32 by default; pass `Q8=1` for QLoRA (Q8-stays-Q8 path via `realize_for_q8_copy`). | ~10 s on GB10 |
| `04_serve_http.rb`       | HTTP API reference shape. **Currently segfaults at startup** — Spinel codegen bug (matz/spinel#647), see file header. | — |
| `05_list_models.rb`      | Walk HF / Ollama / LM Studio / `./data` / `$TOY_MODEL_DIR` caches; print every GGUF with family + params + size. | < 1 s |

## First run — find or fetch a model

```sh
make setup-ggml                # one-time: clones ggml + applies local patches
make example_list_models       # build the discovery binary
./examples/example_list_models # see what's already cached locally
```

If nothing's cached yet, three easy ways to populate one:

```sh
# 1. Use the wrapper (defers to huggingface-cli if installed, else curl).
prep/fetch_model.sh bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf

# 2. huggingface-cli directly (any GGUF repo).
huggingface-cli download bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf

# 3. Ollama / LM Studio also work — toy reads their caches too.
ollama pull llama3.2:1b
```

Re-run `example_list_models` and the new model appears. Set
`TOY_MODEL_DIR=/srv/models` to add another search path.

If you'd rather convert from HF format yourself:

```sh
./prep/convert_smollm2_to_gguf.py        # → data/smollm2-135m-f32.gguf  (legacy layout)
./prep/convert_smollm2_to_gguf.py --ggml-native \
    --out data/smollm2-135m-native.gguf  # mmap-ready layout for LoRA/serve
```

The native build is what `03_finetune_lora.rb` and `04_serve_http.rb`
load — the LoRA / serve paths mmap base weights in place, which
needs the HF `[out, in]` layout (`toy.ggml_native=true`). The legacy
build is what `01_inference.rb` and `02_train_custom_gpt.rb` consume.

## Inference

```sh
make example_inference
./examples/example_inference                      # uses data/smollm2-135m-f32.gguf
GGUF=data/smollm2-135m-q8_0.gguf ./examples/example_inference
# → ids: 6403 1980 253 655 28 665 436 253 1838 ...
```

One binary, model bytes loaded from disk, KV-cache decode.

## Training your own

```sh
./prep/prep_tinystories.rb               # one-time: builds vocab + sequences
make example_train
./examples/example_train
```

The defaults are tiny so it finishes in seconds. To train a model
that actually writes English-shaped stories, bump `EPOCHS=200` and
`D_MODEL=128`, then go make coffee. `Toy::Trainer`
(`lib/toy_trainer.rb`) absorbs the per-step boilerplate so what you
read in `02_train_*.rb` **is** the algorithm.

## Fine-tuning (LoRA + QLoRA)

```sh
# CPU path — fastest to iterate; QLoRA-capable.
make example_finetune
./examples/example_finetune
# F32 base: SmolLM2-135M, ~9.24 → 3.60 CE over 20 steps.

GGUF=data/qwen25-0.5b-native-q8.gguf ./examples/example_finetune
# Q8 base (QLoRA): Qwen2.5-0.5B Q8 + F32 LoRA, ~15.4 → 6.4 CE.
```

```sh
# CUDA path — same loss curve as CPU on F32 bases.
make setup-ggml-cuda
make example_finetune_cuda
./examples/example_finetune_cuda
```

Both build on the same sequence-mode forward graph
(`lib/llama_seq_forward_ffi*.rb`): T positions per forward, masked CE
in one compute, AdamW state in persistent memory. The LoRA-Q rank-8
adapters add ~5 MB on top of an mmap'd base.

CUDA QLoRA (Q8 base on GPU) goes through `realize_for_q8_copy` —
weights live in the standard CUDA buffer (correct padding) instead
of the BYO-pointer mmap region. One cudaMemcpy at load; full GPU
training afterwards. Pass `Q8=1` to the CUDA example to switch paths.

## Serving

> **Status:** working on Spinel master (commit `0adca86` and later)
> after [matz/spinel#647](https://github.com/matz/spinel/issues/647)
> landed. Token-IDs in / token-IDs out by design; for text I/O wire
> in `lib/tokenizer.rb` server-side (same path
> `examples/01_inference.rb` uses, or see `tep_demo/openai_api_*`).

The intended shape:

```sh
make example_serve
./examples/example_serve &
curl -s localhost:4567/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":[12092,4845,253,1429],"n":16}'
```

Token-IDs in, token-IDs out — tokenizer work belongs client-side
(or wire `lib/tokenizer.rb` in if needed). One binary, no runtime.

For the full OpenAI-compatible API (chat completions / streaming /
multi-model registry) see [`tep_demo/`](../tep_demo/README.md).

## Where to go after the examples

- `docs/INDEX.md` — full tour of the documentation.
- `docs/architecture.md` — generic `TransformerLM` + `Arch` struct.
- `demos/` — exhaustive per-model and per-feature drivers.
- `tinynn/` — C+CUDA shim over ggml (FFI bridge, KV cache, mmap loader).
- `tep_demo/` — full OpenAI-compatible HTTP API (the `04_serve`
  example is its lite cousin).

Roadmap + deferred design notes live in `docs/roadmap/`; issues we've
filed upstream (ggml / Spinel / Tep) live in `docs/archive/upstream/`.

## Two-line takeaway

Each example is one Ruby file under ~100 lines. The build is one
`make` target each. The output is a single binary with the model,
the math, and (where relevant) the HTTP server linked in.
