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

### `infer <model> [--prompt TEXT] [--prompt-ids "ID ID …"] [--n N] [--ctx N] [--device cpu|cuda|metal] [--json]`
Greedy decode from a GGUF model. `--prompt-ids` drives tokenizer-less models with
explicit numeric token IDs (overrides `--prompt`; the runners fail loud on a
tokenizer-less model with neither). `--device cuda|metal` selects the per-device
runner (`metal` is macOS-only). Defaults: `--prompt "Once upon a time"`, `--n 16`,
`--device cpu`.

`--ctx N` sets the context (the number of positions the KV cache is allocated
for). Left out, it is the model's own `context_length` clamped to [256, 4096]:
the floor is the old fixed value, the cap is there because the cache is
allocated up front. A value above the trained context is accepted and noted.
A prompt plus `--n` that does not fit is refused before any decode:

```
toy-infer: prompt 362 tokens + n 80 exceeds context 256 (raise it with --ctx / CTX=)
```

`toy eval` takes no `--ctx`: it scores a fixed 3-token prefill and cannot
overflow.

**Special tokens in `--prompt`.** A special token written out in the prompt —
`<|im_start|>`, `<|im_end|>`, `<tool_call>` — is encoded as its single id, so a
chat-formatted prompt can be passed as text:

```
toy infer model.gguf --prompt $'<|im_start|>user\nBonjour<|im_end|>\n<|im_start|>assistant\n'
```

Which strings count comes from the GGUF: the entries `tokenizer.ggml.token_type`
marks control (3) or user-defined (4), which is what both the Toy converter and
llama.cpp's write. The ids match HuggingFace's tokenizer (`make
gate-special-tokens`). A GGUF without that array has no special tokens and
tokenizes as before. In Ruby, `Tokenizer#encode` parses them and
`Tokenizer#encode_plain` does not — use the latter for text that must not be
able to spell a control token.

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

#### Shape and optimization flags

These are ordinary training knobs and work on `from-scratch` and
`warm-start`. Every default is the historical gate-fixed value, so **an
invocation that sets none of them is byte-identical to before they
existed** and every recorded run stays reproducible.

| flag | meaning | default |
| --- | --- | --- |
| `--lr F` | learning rate | `0.001` |
| `--d-model N` | model width | `64` |
| `--layers N` | layer count | `2` |
| `--heads N` | attention heads | `4` |
| `--d-ff N` | feed-forward width | `128` |
| `--context N` | sequence length | `32` |
| `--vocab N` | vocabulary size | `627` |
| `--warmup N` | cosine warmup steps — **`warm-start` only** | `5` |

`--warmup` is rejected on `from-scratch` rather than ignored: that recipe
trains with a constant hyper-parameter and has no schedule to warm up, so
accepting the flag would silently do nothing.

`--vocab` must cover the corpus. Setting it below the highest token id in
the data fails with the offending id and the minimum that would work; it is
never clamped and never allowed to index out of range.

Two knobs that the fixture lanes have and the framework recipes do **not**:
`--batch` and `--val-batches`. `from-scratch` trains on the first line of
`data/ts_seqs.txt` repeated `--steps` times — there is no batch dimension to
widen and no evaluation loop to size, so exposing them would be a lie. They
need a real data pipeline, tracked as [tao#28](https://github.com/OriPekelman/tao/issues/28).

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

Two subcommands, on opposite sides of the capability/fixture seam:

| | | |
|---|---|---|
| `eval ce --ckpt C --corpus P` | cross-entropy of a checkpoint over a corpus | capability |
| `toy research eval lmc --ckpt A --other B [--alphas "0,0.25,…"]` | two-checkpoint linear mode connectivity | research instrument |

`toy eval lmc` still works and prints one stderr line naming the new form.
No deadline on the alias (tao#25).

Both were dispatched but named in no registry, no help and no manifest
before this — discoverable only by reading `lib/toy/core/cli/eval.rb`.
They now appear in `toy eval --help` and under a `subcommands` key in
`toy --manifest`, which is emitted only for commands that have them, so
existing manifest consumers see no new key.

### `serve <model> [--port PORT] [--name NAME] [--ctx N] [--device cpu|cuda] [--stop-ids N,M] [--prefix-file F] [--prefix-cache F]`
Serve a GGUF model over an OpenAI-compatible HTTP API. Default `--port 4567`;
`--name` sets the model label in `/v1/models` (default: the GGUF basename).
`--ctx N` sets the context exactly as for `infer` (same default). `--device
cuda` runs `libexec/toy-serve-cuda`, a generated mirror of the CPU runner.

What the server offers depends on whether the GGUF embeds its tokenizer
(`prep/convert_smollm2_to_gguf.py --with-tokenizer`, or any llama.cpp file):

| | with a tokenizer | without |
| --- | --- | --- |
| `POST /v1/chat/completions` | messages in, text out; `"stream": true` supported | 501 |
| `POST /v1/completions` | token ids in; `ids` and decoded `text` out | token ids in; `ids` out, `text` is `""` |
| end of generation | stops at the model's end token, `finish_reason: "stop"` | runs to `max_tokens`, `"length"` |

- **Chat template.** There is no Jinja evaluator; the template is one of four
  built-in families (ChatML, Llama-3, Gemma 2, Mistral), chosen from the
  special tokens in the vocabulary. Messages are rendered exactly as sent: a
  default system message that a model's own template would add (Qwen's "You
  are Qwen…") is not added.
- **Stop tokens.** The GGUF's `eos_token_id`, plus `<|im_end|>`,
  `<|endoftext|>`, `<|eot_id|>`, `<|end_of_text|>` and `<end_of_turn>` where
  the vocabulary has them. `--stop-ids N,M` adds token ids. The stop token is
  not part of the reply. The OpenAI `stop` request field is not read.
- **`max_tokens`.** Limited by the context only (it was capped at 256).
  Absent, `/v1/completions` generates 16 tokens and chat generates until a
  stop token or the end of the context.
- **Sampling.** Greedy only: `temperature`, `top_p` and `logprobs` are not
  honoured.
- **Context.** A request whose prompt does not fit comes back with no tokens
  and `finish_reason: "context_length_exceeded"` on `/v1/completions` — tep's
  backend API has no error channel yet, so it is not an HTTP 400 — and one
  whose prompt fits but whose `max_tokens` does not is cut to what fits.
  Neither takes the server down.

**Prompt prefix reuse (toy#210).** The K/V rows of whatever prefix a
request shares with the previous one are kept, and only the rest of the
prompt is evaluated. Replies are the same as without reuse (`KV_REUSE=0`
in the runner's environment turns it off, for measurement). For a system
message that every request starts with:

```
toy serve model.gguf --device cuda --prefix-file system.txt --prefix-cache system.kv
```

- `--prefix-file` is the text of the system message (one trailing newline
  is dropped). It is rendered as the chat template's system turn and
  evaluated once at boot. Requests must send the same system message as
  their first message to benefit; a request that differs is still answered
  correctly, it just reuses less.
- `--prefix-cache` is a KV prefix file. If it exists and matches, it is
  restored instead of evaluating the prefix; otherwise it is built from
  `--prefix-file` and saved. Given alone, the file must exist and match, or
  the server refuses to start.
- The file is refused when it was made from a different model file, on a
  different device, with a different KV layout, or holds a different prefix
  than `--prefix-file`, and when it is damaged or truncated. The model
  check is a sampled fingerprint (file size plus 1024 chunks of 16 KiB),
  not a full hash: a change confined to a few MiB between two samples is
  not seen. The tensor data carries no checksum.
- Not supported for MLA attention models (DeepSeek-V2).
- Each request's event carries `cached_tokens`, the number of prompt tokens
  not re-evaluated.

Measured on the gx10 (CUDA, a 3.98 B Qwen2.5-shaped model in Q8, 753-token
system turn, 776-token prompt, 24 new tokens, GPU shared with another job):
about 0.79 s per request without reuse and 0.64 s with the prefix cached;
llama-server takes 0.57 s and 0.45 s on the same requests. Evaluating the
prefix at boot takes 0.34 s, saving the 68 MB file 0.22 s and restoring it
14 ms.

**Decode speed (toy#206).** Prompts are evaluated in batches of up to 512
tokens, and each generated token is one small graph covering every
attention head of a layer. On the model above that is about 40 tokens/s
generated on CUDA (llama.cpp: 55–58 under the same load) and a 776-token
prompt in about 0.2 s. Models and options the fused step does not cover
run on the older per-head graph, which is several times slower on a GPU;
see "Decode graphs" in [`models.md`](models.md).

Three things only tep can change (tep#276), so they are as tep has them:
`/v1/completions` reads `prompt` as a token-id array only (a string prompt is
an empty request); the chat reply always says `finish_reason: "stop"`, also
when it ended on `max_tokens`; and the chat reply `id` is the constant
`chatcmpl-tep`.

The serve stack lives at `lib/toy/serve/openai/` and is driven by
`lib/toy/run/serve.rb` → `libexec/toy-serve`. `make gate-serve`,
`gate-serve-chat`, `gate-serve-chat-cuda`, `gate-kv-prefix` and
`gate-kv-prefix-cuda` cover it.

> **The 0.11.x tep line needs fixes that are only on tep `main`**
> ([OriPekelman/tep#276](https://github.com/OriPekelman/tep/issues/276)).
> `make vendor-tep` applies them (`prep/patch_vendored_tep.rb`), so nothing
> has to be done by hand:
>
> - a request body with a non-ASCII byte is never read to the end, so the
>   request stalls (0.11.7 and 0.11.8);
> - a reply with non-ASCII text is cut short, because `Content-Length`
>   counted characters (0.11.7 and 0.11.8);
> - the stock-Ruby guard fires inside every compiled app at Spinel
>   `2026.10.25`, which defines `RUBY_ENGINE`, and the server exits at boot
>   (0.11.7 only; fixed in 0.11.8,
>   [OriPekelman/tep#272](https://github.com/OriPekelman/tep/issues/272)).
>
> `infer`/`train`/`eval` don't link tep and are unaffected.

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
| `serve` | `lib/toy/serve/openai/` (via `lib/toy/run/serve.rb`) | `libexec/toy-serve`, `libexec/toy-serve-cuda` |

A command builds its runner on demand (`make <target>`, e.g.
`make libexec/toy-train`) or you can build them all up front with `toy install`.
Runners read config from a controlled env hash (e.g. `STEPS`, `SEED`,
`TOY_RUN_DIR`, `TOY_RUN_ID`) passed via `Open3`, so a stale caller environment
cannot leak in.

`infer` / `train` / `eval` accept `--device cuda|metal`, which selects a separate
per-device runner binary (e.g. `libexec/toy-infer-cuda`, `libexec/toy-train-cuda`;
`metal` is macOS-only). They are single-type binaries by Spinel necessity (one
backend module per binary), built on demand the same way. `serve` accepts
`--device cuda` (no Metal runner).

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

