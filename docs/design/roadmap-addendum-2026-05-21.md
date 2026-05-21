# Roadmap addendum — coexistence + model discovery (2026-05-21)

Follow-up to `docs/design/roadmap-2026-05-21.md` after the user's
clarifications:

- Tep#14 is merged → the HTTP layer fix is upstream.
- Per `docs/loader-api.md`, the two weight-loading paths (Mat-mediated
  vs direct/mmap) coexist by design — they're peers, not "old vs new".
- We want both efficient inference AND training-on-CPU as supported
  paths. The roadmap should not collapse to inference-only.
- Model registry / download: prefer to reuse existing caches (HF hub,
  Ollama) rather than force users to re-download GGUFs they already
  have on disk. "Drop the binary on the system, it just works."

## Coexistence: two paths, not a winner

`docs/loader-api.md` already nails this: the Mat-mediated path
(`GGUFLoad.load_toy_smollm2`) is for inspection / fine-tuning /
parity, the direct path (`kv.load_weights` or `realize_for_mmap`) is
for serving at scale, and `read_persistent_mat` bridges them.

Practically, every supported model has both paths exercised. The
sweet-spot inference daemon (per the original roadmap) lives in the
direct/mmap path; the per-tenant fine-tuning story lives across both:

| Surface | Loader path | Why this one |
|---|---|---|
| Inference serving (1+ user, low latency)      | direct + Phase 2 mmap                | Zero-copy from page cache; instant model swap. |
| Inference debugging / acceptance              | Mat-mediated                         | Tensors as Ruby Mats; surgery + diff vs reference. |
| LoRA fine-tuning (current F1.2 work)          | direct + persistent LoRA in ctx_w     | Base weights mmap'd, adapter weights trainable. |
| Full fine-tune (F3, future)                   | Mat-mediated (or hybrid)             | Per-weight grads + Adam state in Ruby Mats for inspection. |
| From-scratch training (TinyStories etc.)      | Mat-mediated (`Toy::Trainer`)        | No GGUF yet; we're writing the weights. |

The user's "CPU training even when CUDA is unavailable" requirement
maps cleanly: Mat-mediated training works on CPU today
(`demos/train`, `examples/02_train_custom_gpt.rb`); the in-graph FFI
training path has the ggml#1501 sched-aliasing limitation until
upstream lands.

**What this changes about the original roadmap:** the multi-model
daemon's job is inference. Training stays on the Mat-mediated path
(read: lib/toy_trainer.rb shape), independent of the daemon's
runtime. The daemon binary CAN be the trainer too — but training
requests are heavyweight and probably belong in a separate "train
job" CLI invocation rather than HTTP requests against a running
server.

Proposed shape:

```
toy serve   …   # the multi-model inference daemon (HTTP)
toy train   …   # the trainer (CLI, blocking; uses Mat path)
toy finetune …  # LoRA on an existing GGUF (CUDA today)
toy chat    …   # one-shot stdin/stdout inference
```

All four are the SAME binary with a subcommand router at the top.
One ELF, one config, multiple entry points.

## Model discovery — read HF and Ollama caches, no extra downloads

Users with PyTorch/Transformers, llama.cpp, Ollama, or LM Studio
already have multi-gigabyte model caches. Forcing them to
re-download GGUFs into our `data/` is rude. Better: scan the standard
locations + present what's there.

### Where canonical caches live

| Cache              | Path                                                  | Format         | Naming                                                          |
|--------------------|-------------------------------------------------------|----------------|-----------------------------------------------------------------|
| HuggingFace hub    | `~/.cache/huggingface/hub/`                           | safetensors + (sometimes) GGUF | `models--{org}--{repo}/snapshots/{rev}/{file}`           |
| HF GGUF repos      | (above, but file ends in `.gguf`)                     | GGUF           | e.g. `models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/{rev}/qwen2.5-0.5b-instruct-q8_0.gguf` |
| Ollama             | `~/.ollama/models/`                                   | GGUF (content-addressed) | `blobs/sha256-{hash}` for files + `manifests/{registry}/{repo}/{tag}` for friendly names |
| LM Studio          | `~/.lmstudio/models/{org}/{repo}/{file}.gguf`         | GGUF           | Flat directory layout. |
| llama.cpp / manual | wherever the user puts them                           | GGUF           | Often `~/models/`, `~/llama-models/`, `./models/`. |

### Discovery algorithm

```
search_paths = [
  ENV["TOY_MODEL_DIR"],               # explicit user override
  "./models",                          # project-local
  "~/.cache/huggingface/hub",          # HF hub
  "~/.ollama/models/blobs",            # Ollama (with manifest parsing for friendly names)
  "~/.lmstudio/models",                # LM Studio
  "~/models",                          # llama.cpp convention
]

for each path in search_paths:
  walk for files matching:
    - *.gguf                              direct discovery
    - sha256-* (in ollama blobs)          via Ollama manifest crawl
  for each found file:
    head = read GGUF header (4 KB, free)
    if valid GGUF:
      name = derive friendly name from path  (see below)
      registry.add(name, file_path, head.arch, head.params...)
```

For each file we get its **arch** (from `general.architecture`), and
its file size. From that we know what `Arch.from_gguf` would say
without paying load cost. Total cost of a full scan: O(N) file
header reads ≈ a few seconds even for many GBs of models.

### Friendly names

Each cache has its own naming convention. Normalisation rules:

| Source | File path | Friendly name |
|---|---|---|
| HF hub (GGUF repo) | `models--Qwen--Qwen2.5-0.5B-Instruct-GGUF/snapshots/.../qwen2.5-0.5b-instruct-q8_0.gguf` | `qwen2.5-0.5b-instruct-q8_0` |
| HF hub (safetensors) | `models--Qwen--Qwen2.5-0.5B/snapshots/.../*.safetensors` | (not loadable — skip; we don't convert at runtime) |
| Ollama | manifests/.../qwen2.5/0.5b → blobs/sha256-abc... | `qwen2.5:0.5b` (preserves Ollama's tag form) |
| LM Studio | `~/.lmstudio/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf` | `qwen2.5-0.5b-instruct-q8_0` |
| Local dir | `./models/my-finetune.gguf` | `my-finetune` |
| `TOY_MODEL_DIR` override | as-named | same |

When two paths resolve to the same content (e.g., HF GGUF + Ollama
blob of the same file), we de-dupe by file hash and prefer the
first-found.

### What we can NOT load directly

- **safetensors / .bin / .pt**: we only consume GGUF. No runtime
  conversion. If a user has a safetensors-only model, point them at
  `prep/convert_*` (or the existing `llama.cpp/convert_*.py` they
  may already have). Document this on the daemon's startup banner:
  "found 3 safetensors models in HF cache; convert with
  `./prep/convert_to_gguf.py ...` to load".

- **Quantized formats we don't support yet (Q4_K_M, Q5_K, etc.)**:
  read the GGUF header, skip with a "unsupported tensor type" note.
  Currently we support F32 + Q8_0; expanding is a separate task.

### What this gives the deployment story

```
$ scp toy-daemon user@box:/usr/local/bin/
$ ssh user@box
user@box$ toy serve
[toy] scanning model caches...
[toy] found 7 GGUFs:
        qwen2.5:0.5b              (Q8_0, 540 MB)
        qwen2.5:1.5b              (F32,  6.0 GB)
        smollm2-135m-instruct-q8  (Q8_0, 80 MB)
        ...
[toy] OpenAI-compatible API on :8080
```

No `pip install`, no `docker pull`, no model download. The binary
introspects what's already there.

## Updated critical path

Re-ordered with these additions:

| # | What | Effort | Notes |
|---|---|---|---|
| 1 | **Tep _tep_lib integration story** | small-med | NEW. Tep#14 merged but our local vendored copy doesn't absorb cleanly because of the BATTERIES-DESIGN module restructure. Pin to a specific Tep commit OR adopt `bin/tep build`. |
| 2 | **Model discovery** (HF + Ollama + LM Studio cache walk + friendly names + dedup) | small-med | The "drop binary, just works" UX. Library code only — no servers yet. |
| 3 | **Task #69 (M3)** — reusable decode graph | medium | Biggest perf unlock. |
| 4 | **ggml#1501 fix** | upstream | Restores CPU LoRA training. |
| 5 | **Multi-model registry daemon** (HTTP, `model:` field, lazy activation, streaming) | medium | The sweet-spot product. Depends on 1 + 2. |
| 6 | **Persistent Adam state in ctx_w** | small-med | Unblocks F1.2 6b. |
| 7 | **Sequence-mode forward** | medium-large | Unblocks F1.2 6c/6d. |
| 8 | **Single-binary `toy` subcommand router** (`serve`/`train`/`finetune`/`chat`) | small | Ties it together for end users. |
| 9 | **Hot LoRA adapter loading per-request** | medium | The per-tenant fine-tuning pitch. |

Items 1–3 are the immediate blockers for the inference-daemon UX.
4 keeps CPU training in scope. 5+8 are the user-facing product.
6–7 + 9 are the longer training arc.

## Co-existence sanity checks for the daemon design

- The daemon's request handler can ONLY use the direct/mmap path
  (we won't load 7B's Mat-mediated form during a request — too slow,
  too much memory).
- The trainer (`toy train`, `toy finetune`) can use either path
  depending on what it's doing; small custom models use
  Mat-mediated, LoRA on an existing GGUF uses direct.
- The same model file can be served (mmap'd, read-only) and
  fine-tuned (mmap'd base + LoRA adapter writes) **simultaneously**
  by two processes. The base file's page cache is shared; only the
  trainee has writable LoRA state.

## What the daemon banner should say

When `toy serve` starts:

```
toy 0.x.y — multi-model LLM daemon
  scanning model caches…
    ./models                       2 GGUFs
    ~/.cache/huggingface/hub        4 GGUFs  (+ 12 safetensors, skipped)
    ~/.ollama/models/blobs          8 GGUFs
  21 unique models registered (after dedup by hash)
  3 quantizations supported: F32, Q8_0   (Q4_K_M, Q5_K coming)
  serving on 0.0.0.0:8080
  POST /v1/completions     model:"qwen2.5:1.5b"
  GET  /v1/models          list available models
```

Honest about what's present, what's skipped, and why. No magic.

## What's deliberately left out of this addendum

- **Model DOWNLOAD support.** Initially the daemon reads what's
  already there; if a user names a model the daemon doesn't have,
  return 404 with a hint message ("download via huggingface-cli or
  ollama pull, then restart"). Building a downloader is its own
  rabbit hole (auth tokens, resumable downloads, mirrors); not on
  critical path.
- **Hash-content cache for cross-tool dedup.** Could do something
  fancy with sha256 file identity. Defer; just dedup by absolute
  path normalization.
- **Per-cache eviction policy.** Daemon's job is to find models, not
  manage disk space.

## Decision points for the user, restated

1. **Subcommand-router shape** — is the `toy <serve|train|...>` form
   acceptable, or do you want separate binaries? Single binary keeps
   the deployment story tighter; separate binaries are smaller per
   binary and let users skip what they don't want.
2. **Tep migration timing** — do we pin to the last working Tep commit
   and incrementally absorb BATTERIES-DESIGN, or fork the integration
   approach (use `bin/tep build` instead of `spinel` directly for the
   serve path)?
3. **HF safetensors handling** — bail loudly + point at the
   converter, OR auto-convert in the background (with disk cost +
   surprise factor) when first requested? My read: bail loudly.

I'm happy to start on (1) and (2) as the immediate path forward
once you've called those.
