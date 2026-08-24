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

Train a model and record a run bundle. `<recipe>` is required.

| recipe | what it does | recipe-specific flags |
| --- | --- | --- |
| `from-scratch` | train a tiny Llama-shape model from random init (the default) | — |
| `lora` | LoRA adapters over a frozen base GGUF | `--model <gguf>` (base, default `data/smollm2-135m-native.gguf`), `--rank N` (default 8) |
| `warm-start` | fine-tune from donor embeddings over a streamed corpus | `--corpus <path>` (default `data/ts_seqs.bin`), `--init <mode>` (default `scratch`) |
| `vit-tiny` | ViT-Tiny image classifier on the committed smoke corpus (CPU-only) | — |

Defaults: `--steps 5`, `--seed 0`, `--arch llama`, `--device cpu`,
`--out runs/<id>`. Writes the run bundle (see
[Run bundle layout](#run-bundle-layout)) and echoes the byte-deterministic
loss curve to stdout.

`--arch gpt2` trains the from-scratch GPT-2 arch (`from-scratch` only).
`--device cuda|metal` selects the per-device runner; `metal` is macOS-only.
ViT-Tiny is CPU-only; GPT-2 has cuda + metal twins.

#### Research fixtures

toy also ships eleven **fixture** recipes — synthetic task lanes that exist so
toy can test its credit-assignment capabilities independently of any one
research question:

```
mlp  ctr  gnn  ssm  lstm  gtx  diff  difflm  ae  franken  franken-moe
```

They are supported, gated and reproducible, but they are not the product path.
Their reference — the flags, the device scopes, and the reasoning behind each
knob — lives in **[docs/research/lanes.md](research/lanes.md)**.

The *capabilities* those lanes test (`--policy chain|dfa|frozen`, the `--dfa-*`
feedback-matrix family, `--optimizer`, `--lr-schedule`, checkpointing, the
conditioning instrument) are framework features, not fixture-local ones.

#### Flag x recipe is a MATRIX, and it is gated

Every recipe-specific flag is rejected under the wrong recipe (exit 2,
`only valid with recipe …`), *before* any build or runner spawn. The
table lives in one place (`lib/toy/core/cli/train.rb`'s `flag_matrix`)
and `prep/train_cli_matrix_gate.rb` asserts every row — recipe parity is
a cell flip, not a scattered set of one-off checks.

### `eval <model> [--top-k K] [--device cpu|cuda|metal] [--json]`
Score a GGUF model: per-token logprobs. Default `--top-k 5`, `--device cpu`.
Two-checkpoint Linear Mode Connectivity is a subcommand:
`eval lmc --ckpt A --other B [--alphas "0,0.25,…"]`.

### `serve <model> [--port PORT] [--name NAME]`
Serve a GGUF model over an OpenAI-compatible HTTP API (CPU). Default
`--port 4567`; `--name` sets the model label in `/v1/models` (default: the GGUF
basename). The serve stack lives at `lib/toy/serve/openai/` and is driven by
`lib/toy/run/serve.rb` → `libexec/toy-serve`.

> Known issue: `libexec/toy-serve` fails to compile at toy's current Spinel pin
> (`Tep::Scheduler.spawn_fiber` → `FiberSlot.new` incompatible-pointer; plus a
> JSON-key monomorphization). This is a **Spinel regression**, not a tep bug —
> both are clean at tep's Spinel pin `f6d5eef`. The fix is to align toy's Spinel
> build to that pin (no tep release needed); tracked as
> [OriPekelman/tep#198](https://github.com/OriPekelman/tep/issues/198) (the
> pin-bump tracker; gates toy#30). `infer`/`train`/`eval` don't link tep and are
> unaffected.

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
`TOY_RUN_DIR`, `TOY_RUN_ID`) passed via `Open3`, so a stale caller environment
cannot leak in.

`infer` / `train` / `eval` accept `--device cuda|metal`, which selects a separate
per-device runner binary (e.g. `libexec/toy-infer-cuda`, `libexec/toy-train-cuda`;
`metal` is macOS-only). They are single-type binaries by Spinel necessity (one
backend module per binary), built on demand the same way. `serve` is CPU-only.

## Examples

The CLI is the supported entry point; the curated `examples/` set (toy#60)
is the narrated library-API tour — eight single-file programs Spinel
compiles to native binaries (06 is plain CRuby). Headers carry the
what/how-long/what-to-tweak narration; every knob is ENV (compile once,
run many).

| Example | What it does |
|---|---|
| `examples/01_train_tiny.rb` | From-scratch tiny Llama on the bundled corpus; recipe + value objects; writes a `runs/` bundle. |
| `examples/02_finetune_warm_start.rb` (or `toy train warm-start`) | Warm-start: donor embeddings from a real GGUF, then fine-tune. |
| `examples/03_lora.rb` (or `toy train lora`) | LoRA / QLoRA adapters over a frozen mmap'd base. |
| `examples/04_generate.rb` (or `toy infer`) | GGUF → KV-cache decode → text. |
| `examples/05_eval_logprobs.rb` (or `toy eval`) | Per-token top-K logprobs at a decode position. |
| `examples/06_runlog_compare.rb` | CRuby: `Toy::RunLog.scan` comparison table over `runs/`. |
| `examples/07_vit_tiny.rb` | ViT-Tiny classifier on the committed smoke corpus (or `toy train vit-tiny`). |
| `examples/08_gdn_block.rb` | Gated-DeltaNet block: the recurrent-memory primitive, standalone. |
| `examples/legacy/*` | Superseded tutorials (instrumented trainer, pure-Ruby GPT, …) — still build; see [examples/legacy/README.md](../examples/legacy/README.md). |
| `prep/smokes/smoke_*.rb` | Single-purpose wire smokes — the gate fixtures, not tutorials (see [gating.md](gating.md)). |

### Example env knobs

These are read by the example binaries (not the CLI; the CLI passes its own
controlled env). Each example's header lists its own; the instrumented legacy
trainer (`examples/legacy/06_train_from_scratch.rb`, built by
`make example_train_from_scratch`) carries the full set:

| Env | Purpose |
|---|---|
| `D_MODEL` `D_FF` `N_HEADS` `N_LAYERS` `CONTEXT` | Model shape |
| `STEPS` `LR` `SEED` | Training schedule |
| `DEVICE=cuda` | Use the CUDA mirror path |
| `TOY_RUN_DIR=<dir>` | Emit the `events.jsonl` stream (and a final checkpoint) here |
| `TAO_RUN_DIR=<dir>` | Compatibility fallback for `TOY_RUN_DIR`. Used only when `TOY_RUN_DIR` is unset **or empty**; `TOY_RUN_DIR` wins when both are set. Retained indefinitely — see tao#27 |
| `TOY_EVENTS=<path>` | Raw alternative to `TOY_RUN_DIR`: full path to the events file |
| `CHECKPOINT_EVERY=N` | Write `weights/step_<N>.gguf` every N steps (+ `latest` symlink) |
| `TOY_DESCRIBE=json\|mermaid\|text` | Dump the compute graph and exit (no training) |
| `TOY_GRAD_SENTINELS=1` `TOY_DRIFT_EVERY=N` `TOY_CKA=N` | Instrumentation events |

See [examples/README.md](../examples/README.md) for the full roster and
invocations.

