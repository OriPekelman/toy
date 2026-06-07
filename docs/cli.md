# The `toy` CLI

`toy` is a CRuby command-line front end for the Spinel-compiled toy LLM
framework. It scaffolds projects, finds and describes GGUF models, builds the
compute backend, and drives inference / training / eval / serving.

## Dispatch model

`bin/toy` loads `lib/toy/core/cli.rb`. The `COMMANDS` registry in that file is
the **single source of truth**: it drives `--help`, `--manifest`, *and* command
dispatch — there is no drift between the three.

Everything under `lib/toy/core/` is plain MRI Ruby. These files must **never**
`require` the Spinel-compiled libraries (`tinynn.rb`, `arch.rb`,
`transformer.rb`, `toy_describe_flow.rb`, …): those carry `ffi_lib` directives
that fail to load under CRuby. When a command needs to *compute*, it shells out
to a Spinel-built runner under `libexec/` (see [Build path](#build-path)).

### Global flags

| Flag | Effect |
|---|---|
| `--manifest` | Emit the machine-readable command manifest as JSON (`format: "toy/manifest-v1"`) and exit. Works with no subcommand. |
| `--help`, `-h` | Print usage and exit. |
| `--version` | Print the toy version and exit. |

`toy --manifest` reflects the live registry, so it is the authoritative
description of the surface — cross-check anything in this doc against it.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | OK |
| `2` | Bad input (unknown command, missing required arg, bad flag value) |
| `1` | Execution failure (GGUF unreadable, scaffold target exists, build/runner failed) |

## Command reference

The nine commands below are reproduced verbatim from the registry; args and
flags are authoritative.

### `new <path> [--force] [--json]`
Scaffold a conventional toy project tree at `<path>`. `--force` overwrites a
non-empty directory.

### `install [--json]`
Build / verify the CPU backend for this project (the `libexec/` runners).

### `list [--json]`
Find GGUF models in the local caches (HuggingFace / Ollama / LM Studio) and the
project's `data/`.

### `describe <model> [--json]`
Read a GGUF file's metadata and render the arch-derived **Card** (the runtime
graph-walk that infers the model's shape and capabilities).

### `fetch <hf-repo> [file.gguf] [--json]`
Download a GGUF from a HuggingFace repo into the cache and add a `data/`
symlink. The second positional selects a specific `.gguf` file within the repo.

### `infer <model> [--prompt TEXT] [--prompt-ids "ID ID …"] [--n N] [--device cpu|cuda|metal] [--json]`
Greedy decode from a GGUF model. `--prompt-ids` drives tokenizer-less models with
explicit numeric token IDs (overrides `--prompt`; the runners fail loud on a
tokenizer-less model with neither). `--device cuda|metal` selects the per-device
runner (`metal` is macOS-only). Defaults: `--prompt "Once upon a time"`, `--n 16`,
`--device cpu`.

### `train <recipe> [--steps N] [--seed S] [--arch llama|gpt2] [--device cpu|cuda|metal] [--out DIR] [--json]`
Train a model from scratch and record a run bundle.
`<recipe>` is required; `from-scratch` is the accepted recipe (other recipes are
rejected with a bad-input error). `--arch gpt2` trains the from-scratch GPT-2
arch (CPU); `--device cuda|metal` selects the per-device runner (`metal` is
macOS-only; GPT-2 is CPU-only).
Defaults: `--steps 5`, `--seed 0`, `--arch llama`, `--device cpu`, `--out runs/<id>`.
Writes the run bundle (see [Run bundle layout](#run-bundle-layout)) and echoes
the byte-deterministic loss curve to stdout.

### `eval <model> [--top-k K] [--device cpu|cuda|metal] [--json]`
Score a GGUF model: per-token logprobs. Default `--top-k 5`, `--device cpu`.
Two-checkpoint Linear Mode Connectivity is a subcommand:
`eval lmc --ckpt A --other B [--alphas "0,0.25,…"]`.

### `serve <model> [--port PORT] [--name NAME]`
Serve a GGUF model over an OpenAI-compatible HTTP API (CPU). Default
`--port 4567`; `--name` sets the model label in `/v1/models` (default: the GGUF
basename). The serve stack lives at `lib/toy/serve/openai/` and is driven by
`lib/toy/run/serve.rb` → `libexec/toy-serve`.

> Known issue: `libexec/toy-serve` currently fails to compile in tep's scheduler
> (`Tep::Scheduler.spawn_fiber` → `FiberSlot.new` incompatible-pointer under
> Spinel) — a tep-side codegen bug, tracked as
> [OriPekelman/tep#198](https://github.com/OriPekelman/tep/issues/198) (gates
> toy#30). `infer`/`train`/`eval` are tep-free and unaffected.

## Project config — `toy.yml`

A project's `toy.yml` (loaded by `lib/toy/core/config.rb`) has exactly two keys.
An absent or empty file is valid and yields all defaults. Backend is
deliberately **not** a key (it is runtime-autodetected for cross-device parity).

| Key | Default | Purpose |
|---|---|---|
| `run_id_template` | `{arch}-{date}-{seq}` | Template for `runs/<id>/` dir names. |
| `algos_path` | `algos` | Relative dir walked for user-supplied L1–L4 algorithms. |

`run_id_template` brace tokens (expanded by `toy train`):
`{arch}` (e.g. `llama`), `{date}` (`YYYYMMDD`), `{time}` (`HHMMSS`), and
`{seq}` (3-digit zero-padded daily counter). Unknown keys warn to stderr but
never abort.

## Run bundle layout

`toy train` resolves a run id from `run_id_template`, creates the directory in
the project cwd, and shells the runner with a controlled env. The bundle:

```
runs/<id>/
  events.jsonl            # toy/v1 event stream (see events.md)
  flow.json               # the realized compute-graph DAG (toy/v1) — written
                          #   once at startup so the bundle is self-describing
  weights/
    step_<N>.gguf         # checkpoints
    latest                # symlink to the most recent checkpoint
```

Events, `flow.json`, and weights are written file-side only — never to stdout.
For the event schema see [events.md](events.md).

## Build path

The CLI cannot compute under MRI, so each compute command shells out to a
Spinel-compiled runner. The mapping:

| Command | Runner source | Binary |
|---|---|---|
| `infer` | `lib/toy/run/infer.rb` | `libexec/toy-infer` |
| `train` | `lib/toy/run/train.rb` | `libexec/toy-train` |
| `eval`  | `lib/toy/run/eval.rb`  | `libexec/toy-eval` |
| `serve` | `lib/toy/serve/openai/` (via `lib/toy/run/serve.rb`) | `libexec/toy-serve` |

A command builds its runner on demand (`make <target>`, e.g.
`make libexec/toy-train`) or you can build them all up front with `toy install`.
Runners read config from a controlled env hash (e.g. `STEPS`, `SEED`,
`TAO_RUN_DIR`, `TOY_RUN_ID`) passed via `Open3`, so a stale caller environment
cannot leak in.

`infer` / `train` / `eval` accept `--device cuda|metal`, which selects a separate
per-device runner binary (e.g. `libexec/toy-infer-cuda`, `libexec/toy-train-cuda`;
`metal` is macOS-only). They are single-type binaries by Spinel necessity (one
backend module per binary), built on demand the same way. `serve` is CPU-only.

## Examples

The CLI is the supported entry point; `examples/*.rb` are focused, single-file
programs Spinel compiles to native binaries (no Python, no runtime). They cover
paths the CLI does not yet expose (GPU, ViT, LMC, warm-start) and serve as the
gate fixtures.

| Example | What it does |
|---|---|
| `examples/02_train_custom_gpt.rb` | Train a tiny GPT from scratch on TinyStories. |
| `examples/legacy/03_finetune_lora.rb` (or `toy train lora`) | LoRA / QLoRA fine-tune via the sequence-mode forward graph (CPU). |
| `examples/06_train_from_scratch.rb` | Modern Llama-shape from-scratch trainer (RMSNorm + GQA + RoPE + SwiGLU); CPU + CUDA. Emits `toy/v1` events. |
| `examples/07_train_vit_tiny.rb` | ViT-Tiny image classifier with timm AugReg donor warm-start. |
| `examples/legacy/08_lmc.rb` (or `toy eval lmc`) | Linear Mode Connectivity blend of two from-scratch checkpoints over an α grid. |
| `examples/legacy/09_warm_start_train.rb` (or `toy train warm-start`) | Warm-start trainer with donor `token_embd` + optional PCA-init projection lens. |
| `examples/smoke_*.rb` | Single-purpose wire smokes — also the gate fixtures (see [gating.md](gating.md)). |

CUDA / Metal mirrors exist for several (`*_cuda.rb`, `*_metal.rb`).

### Example env knobs

These are read by the example binaries (not the CLI; the CLI passes its own
controlled env). Most relevant to `06_train_from_scratch.rb`:

| Env | Purpose |
|---|---|
| `D_MODEL` `D_FF` `N_HEADS` `N_LAYERS` `CONTEXT` | Model shape |
| `STEPS` `LR` `SEED` | Training schedule |
| `DEVICE=cuda` | Use the CUDA mirror path |
| `TAO_RUN_DIR=<dir>` | Emit the `events.jsonl` stream (and a final checkpoint) here |
| `TOY_EVENTS=<path>` | Raw alternative to `TAO_RUN_DIR`: full path to the events file |
| `CHECKPOINT_EVERY=N` | Write `weights/step_<N>.gguf` every N steps (+ `latest` symlink) |
| `TOY_DESCRIBE=json\|mermaid\|text` | Dump the compute graph and exit (no training) |
| `TOY_GRAD_SENTINELS=1` `TOY_DRIFT_EVERY=N` `TOY_CKA=N` | Instrumentation events |

See [examples/README.md](../examples/README.md) for the full roster and
invocations.
