# examples/

Short, focused entry points. Each file is one runnable Ruby program
that Spinel compiles to a native binary. No Python, no PyTorch, no
runtime to install — `make`, run.

## What's here

| File | What it does | Runtime |
|---|---|---|
| `train_from_scratch.rb` | **BLESSED from-scratch path. Start here.** Tiny Llama-shape model trained through the L4 FromScratch recipe + `Toy::AdamW` / `Toy::Labels` / `Toy::SmolLM2Config.mha`. The short tutorial. | seconds |
| `02_train_custom_gpt.rb` | Train a tiny GPT from scratch on TinyStories. Bumping EPOCHS makes a model that writes English. | ~3 min default |
| `03_finetune_lora.rb`    | LoRA / **QLoRA** fine-tune via the sequence-mode forward graph (CPU). Needs a native-layout GGUF — see prep step below. | ~30 s |
| `03_finetune_lora_cuda.rb` | CUDA mirror of the above. F32 by default; pass `Q8=1` for QLoRA (Q8-stays-Q8 path via `realize_for_q8_copy`). | ~10 s on GB10 |
| `06_train_from_scratch.rb` | Instrumentation reference for the recipe path: events, checkpoints, drift sentinels, CKA taps, Tao harness. Read `train_from_scratch.rb` first; this is the full-knobs version. | seconds–minutes |
| `07_train_vit_tiny.rb`   | ViT-Tiny image classifier with timm IN-21k AugReg donor warm-start. Patch embed + class token + 12 blocks + MLP head; CIFAR-shape smoke. | ~30 s for 200 steps |
| `08_lmc.rb`              | Linear Mode Connectivity blend of two from-scratch checkpoints over an α grid; emits one `eval` per α. Tao's `Analyze.lmc` consumes these. | seconds |
| `09_warm_start_train.rb` | Warm-start trainer w/ donor `token_embd` + optional PCA-init projection lens (Qwen-2.5-1.5B → 410M-shape transfer; see file header for the full invocation). | seconds–hours |
| `smoke_recipe_{from_scratch,lora,warm_start}.rb` | The byte-gated recipe exemplars users read — each drives one L4 recipe through `realize!`/`step!` with `Toy::AdamW` + `Toy::Labels`. | < 5 s each |
| `smoke_*.rb`             | Single-purpose wire smokes: corpus loader, decode logprobs, embed API, image loader, projection lens, ckpt reload, ViT-Tiny. Each verifies one primitive end-to-end. | < 5 s each |

## First run — find or fetch a model

```sh
make setup-ggml                # one-time: clones ggml + applies local patches
toy list                       # see what's already cached locally
```

If nothing's cached yet, three easy ways to populate one:

```sh
# 1. Use the toy CLI (defers to `hf` / huggingface-cli if installed, else curl).
toy fetch bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf

# 2. huggingface-cli / hf directly (any GGUF repo).
hf download bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf

# 3. Ollama / LM Studio also work — toy reads their caches too.
ollama pull llama3.2:1b
```

Re-run `toy list` and the new model appears. Set
`TOY_MODEL_DIR=/srv/models` to add another search path.

If you'd rather convert from HF format yourself:

```sh
./prep/convert_smollm2_to_gguf.py        # → data/smollm2-135m-f32.gguf  (legacy layout)
./prep/convert_smollm2_to_gguf.py --ggml-native \
    --out data/smollm2-135m-native.gguf  # mmap-ready layout for LoRA/serve
```

The native build is what `03_finetune_lora.rb` and the `tep_demo/`
serving binaries load — the LoRA / serve paths mmap base weights in
place, which needs the HF `[out, in]` layout (`toy.ggml_native=true`).
The legacy build is what `02_train_custom_gpt.rb` consumes.

## Inference

Inference now lives in the lib-side runner (`lib/toy/run/infer.rb` →
`libexec/toy-infer`), driven by the CLI. `toy infer` builds the runner on
demand and shells to it:

```sh
toy infer data/smollm2-135m-f32.gguf              # uses the given GGUF
toy infer data/smollm2-135m-q8_0.gguf --prompt "Once upon a time" --n 16
# → ids: 6403 1980 253 655 28 665 436 253 1838 ...
```

One binary, model bytes loaded from disk, KV-cache decode. On macOS the
Metal-accelerated GPU path is still `examples/01_inference_metal.rb`
(`make example_inference_metal`) until the runner grows a `--device` flag.

## Training your own

### Blessed path — `train_from_scratch.rb` (start here)

The short tutorial. A tiny Llama-shape model (RMSNorm + GQA + RoPE +
SwiGLU) trained through the L4 `FromScratch` recipe, composed from the
three pure-Ruby value objects:

- `Toy::SmolLM2Config.mha` — the model shape (no 9-arg positional soup).
- `Toy::Labels.next_token` — the shift-by-one one-hot label Mat.
- `Toy::AdamW` — the named optimizer hyper-params (`adamw.hp(step)`).

```sh
make example_train_from_scratch_blessed
./examples/example_train_from_scratch_blessed     # reproduces the gate curve
```

The four recipes (`from_scratch`, `lora`, `warm_start`, `vit_tiny`,
under `lib/toy/llm/recipes/`) all share the same `realize!` / `step!`
shape; the `smoke_recipe_*.rb` exemplars are the byte-gated reads.

### From-scratch — full knobs + Tao instrumentation (`06_train_from_scratch.rb`)

The instrumentation reference for the recipe path. Same Llama-shape
model as the blessed tutorial, but with the full knob surface: events,
checkpoints, drift sentinels, CKA taps, BATCH / GRAD_ACCUM /
WEIGHT_DTYPE, and Tao's end-to-end experiment harness. Read
`train_from_scratch.rb` first; reach for this when you need the
instrumentation.

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

The working HTTP serving path is the toy CLI's **`toy serve`**,
backed by `lib/toy/serve/openai/`. It serves any llama-family GGUF
(SmolLM2, Qwen2.5, TinyLlama, Llama-3.x …) with an OpenAI-compatible
surface (`/v1/models`, `/v1/completions`, `/v1/embeddings`).

```sh
toy serve data/SmolLM2-135M-Instruct-Q8_0.gguf --port 4567 --name smol

curl -X POST http://127.0.0.1:4567/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt":[12092,4845,253,1429],"max_tokens":16}'

curl -X POST http://127.0.0.1:4567/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":[1,2,3]}'
```

Token-IDs in, token-IDs out — tokenizer work belongs client-side
(`prep/qwen25_tokens.py encode "..."`).

`toy serve` replaced the standalone `tep_demo/openai_api_llama`
binary; the remaining Tep+Spinel demo servers live in
[`tep_demo/`](../tep_demo/README.md).

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

- [`../README.md`](../README.md) — project overview + CLI quickstart.
- [`../docs/architecture.md`](../docs/architecture.md) — generic `TransformerLM` + `Arch` struct.
- `demos/` — exhaustive per-model and per-feature drivers.
- `tinynn/` — C+CUDA shim over ggml (FFI bridge, KV cache, mmap loader).
- `tep_demo/` — full OpenAI-compatible HTTP API (the canonical
  serving path; the old lite `04_serve` example was removed).

Roadmap + deferred design notes live in `docs/roadmap/`; issues we've
filed upstream (ggml / Spinel / Tep) live in `docs/archive/upstream/`.

## Two-line takeaway

Each example is one Ruby file under ~100 lines. The build is one
`make` target each. The output is a single binary with the model,
the math, and (where relevant) the HTTP server linked in.
