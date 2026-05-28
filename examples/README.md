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
| `04_serve_http.rb`       | HTTP API reference shape. Boots cleanly but `Tep.run!` exits immediately (uses old vendored Tep); for a working serving binary see [`tep_demo/openai_api_llama`](../tep_demo/README.md). | — |
| `05_list_models.rb`      | Walk HF / Ollama / LM Studio / `./data` / `$TOY_MODEL_DIR` caches; print every GGUF with family + params + size. | < 1 s |
| `06_train_from_scratch.rb` | **Modern from-scratch trainer.** Llama-shape (RMSNorm + GQA + RoPE + SwiGLU); CPU + CUDA (DEVICE=cuda). Emits toy/v1 events. BATCH + GRAD_ACCUM + WEIGHT_DTYPE knobs. Tao's experiment harness rides this. | seconds–minutes |
| `07_train_vit_tiny.rb`   | ViT-Tiny image classifier with timm IN-21k AugReg donor warm-start. Patch embed + class token + 12 blocks + MLP head; CIFAR-shape smoke. | ~30 s for 200 steps |
| `08_lmc.rb`              | Linear Mode Connectivity blend of two from-scratch checkpoints over an α grid; emits one `eval` per α. Tao's `Analyze.lmc` consumes these. | seconds |
| `09_warm_start_train.rb` | Warm-start trainer w/ donor `token_embd` + optional PCA-init projection lens (Qwen-2.5-1.5B → 410M-shape transfer; see file header for the full invocation). | seconds–hours |
| `smoke_*.rb`             | Single-purpose wire smokes: corpus loader, decode logprobs, embed API, image loader, projection lens, ckpt reload, ViT-Tiny. Each verifies one primitive end-to-end. | < 5 s each |

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

### From-scratch — Llama arch + Tao instrumentation (`06_train_from_scratch.rb`)

The modern from-scratch path. Llama-shape (RMSNorm + GQA + RoPE +
SwiGLU), full f32 weights + AdamW state in persistent memory, one
forward + backward + opt_step graph per training step. Drives Tao's
end-to-end experiment harness.

```sh
make example_train_from_scratch
./examples/example_train_from_scratch                 # CPU, SmolLM2-shape default
DEVICE=cuda ./examples/example_train_from_scratch     # ~10× faster on GB10
```

Knobs (env, all optional):

| Env | Purpose |
| --- | --- |
| `D_MODEL=64 D_FF=128 N_HEADS=4 N_LAYERS=2 CONTEXT=32` | Model shape |
| `STEPS=60 LR=0.001 SEED=42` | Training schedule |
| `TAO_RUN_DIR=/tmp/r` | Emit toy/v1 events stream to `/tmp/r/events.jsonl` |
| `TOY_GRAD_SENTINELS=1` | Per-PARAM `grad` event every step (drift detection) |
| `TOY_DRIFT_EVERY=N` | Per-PARAM `drift` event every N steps |
| `TOY_CKA=N` | Activation-Gram tap event every N steps (per block × 3 regions) |
| `CHECKPOINT_EVERY=N` | Write `weights/step_<N>.gguf` every N steps (+ `latest` symlink) |
| `TOY_DESCRIBE=json\|mermaid\|text` | Dump the compute graph and exit (no training) |

Qwen-shape sanity (24L × 16-head × 1024) trains end-to-end via
`D_MODEL=1024 D_FF=2816 N_HEADS=16 N_LAYERS=24 DEVICE=cuda` (~314
ms/step on GB10). The graph node budget auto-sizes from cfg
(`n_layers × n_heads × 1000 + 65536`).

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

The working HTTP serving path is **`tep_demo/openai_api_llama`** —
one consolidated env-driven binary that serves any llama-family
GGUF (SmolLM2, Qwen2.5, TinyLlama, Llama-3.x …) with an
OpenAI-compatible surface (`/v1/models`, `/v1/completions`,
`/v1/embeddings`). See [`tep_demo/`](../tep_demo/README.md).

```sh
make tep_demo/openai_api_llama
./tep_demo/openai_api_llama -p 4567                    # SmolLM2-135M default
MODEL_PATH=data/qwen25-1.5b-native-q8.gguf \
  ./tep_demo/openai_api_llama -p 4567 -w 1             # any llama-family GGUF

curl -X POST http://127.0.0.1:4567/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":[12092,4845,253,1429],"max_tokens":16}'

curl -X POST http://127.0.0.1:4567/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":[1,2,3]}'
```

Token-IDs in, token-IDs out — tokenizer work belongs client-side
(`prep/qwen25_tokens.py encode "..."`). One binary, no runtime.

`04_serve_http.rb` is the older minimal-shape demo; it builds
clean but `Tep.run!` returns immediately because it uses the
pre-spinelgems vendored Tep. Wiring it to the modern vendor
path is a small follow-up; `openai_api_llama` is the canonical
serving binary today.

## Inspecting + analysing trained models

Once you have one (or two) checkpoints from `06_train_from_scratch`,
several runners read them back.

```sh
# Load a from-scratch toy GGUF and generate from it (smoke; the
# checkpoint per-head naming is detected automatically).
make examples/smoke_toy_ckpt_reload
GGUF=/tmp/run_a/weights/latest ./examples/smoke_toy_ckpt_reload

# Linear Mode Connectivity: blend two ckpts along α and eval CE.
# Emits one eval(name="lmc", extra.alpha) per α — Tao's Analyze.lmc
# consumes these to plot the α→loss curve and call same-basin /
# disconnected.
make example_lmc
LMC_A=/tmp/run_a/weights/latest LMC_B=/tmp/run_b/weights/latest \
  LMC_ALPHAS=0,0.25,0.5,0.75,1.0 TAO_RUN_DIR=/tmp/lmc \
  ./examples/example_lmc

# Embedding-table lookup (one row per token id; dequant-aware,
# F32/Q4/Q5/Q6/Q8/F16). Building block for Tep's /v1/embeddings.
make examples/smoke_embed_api
GGUF=data/llama-3.2-1b-native.gguf ./examples/smoke_embed_api

# Per-step logprobs + top-K (building block for Tep's
# /v1/chat?logprobs=true).
make examples/smoke_decode_logprobs
GGUF=data/smollm2-135m-native.gguf TOP_K=5 ./examples/smoke_decode_logprobs
```

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
