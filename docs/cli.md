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

### `serve <model> [--port PORT] [--name NAME] [--ctx N] [--device cpu|cuda] [--stop-ids N,M] [--prefix-file F] [--prefix-cache F] [--slots N] [--adapter F]`
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
  not part of the reply.
- **`stop` strings.** `"stop": "text"` or an array of texts ends the reply
  where the first of them begins; the text itself is not returned and
  `finish_reason` is `"stop"`. Streamed replies hold back the few bytes
  that could still be the start of a stop string. `completion_tokens`
  counts every token generated, including the ones that spelled it.
- **`max_tokens`.** Limited by the context only (it was capped at 256).
  Absent, `/v1/completions` generates 16 tokens and chat generates until a
  stop token or the end of the context.
- **Sampling.** Greedy only: `temperature` and `top_p` are not honoured.
  `response_format` is accepted and not enforced.
- **Log-probabilities.** `"logprobs": true` (with `"top_logprobs": n`, up
  to 20) on chat, or `"logprobs": n` on `/v1/completions`, adds
  `choices[0].logprobs.content`: one entry per reply token with `token`,
  `id`, `logprob`, `bytes` and the `n` most likely alternatives, best
  first. `token` is `""` when the token's bytes are not whole UTF-8
  characters; `bytes` always has them. Not available on streamed replies.
- **Forced answer prefix.** Two ways to make the model continue a text
  you wrote: a chat request whose last message has role `assistant` (its
  content is continued and the reply is only what follows), or a text
  `"prompt"` on `/v1/completions`, in which special tokens written out
  (`<|im_start|>`) are their single ids. With prefix reuse (below), a
  request that extends the previous one evaluates only the new tokens.
- **Errors.** No messages, an empty prompt, or a prompt that does not fit
  the context are HTTP 400 with `{"error": {"message", "type"}}`; an
  exception while generating is HTTP 500 and is logged. The server keeps
  serving. Streamed requests end their stream instead.
- **Concurrent requests.** By default requests are answered in turn.
  `--slots N` decodes up to N of them as one batch; see "Concurrent
  requests" below.
- **Context.** A request whose prompt does not fit is an HTTP 400 of type
  `context_length_exceeded`; one whose prompt fits but whose `max_tokens`
  does not is cut to what fits. Neither takes the server down.

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

**Concurrent requests (toy#217).** `--slots N` (1 to 64, default 1) lets N
requests decode at the same time. Each decode step carries one token of
every active request through a single graph, so N requests cost one graph
evaluation per token, not N. Requests beyond N wait for a slot.

```
toy serve model.gguf --device cuda --ctx 1024 --slots 32 --prefix-file system.txt
```

- **One cache, shared.** The K/V rows of all slots live in one pool of
  `ctx × slots` cells (plus `ctx` more with a boot prefix). The boot
  prefix is stored once and read by every request that starts with it. A
  slot keeps its rows when its request ends, and a new request takes the
  idle slot whose cached ids share the longest prefix with its prompt, so
  prefix reuse works per slot. Memory for the pool is allocated at boot:
  `2 × layers × kv_heads × head_dim × 4` bytes per cell (24 KiB on the model
  below, so 1024 × 32 cells are about 0.8 GiB).
- **Same replies?** On F32 weights, requests sent at once return the ids
  they return alone (CPU and CUDA; `gate-serve-slots` asserts it). On
  quantized weights under CUDA they can differ at near-ties, because ggml
  picks a different matmul kernel for a different batch size; llama.cpp
  behaves the same way. On the shrink student (Q8, CUDA,
  200 held-out events) 164 and 173 replies of 200 were the same text at 8
  and 32 slots as at one, and the task scores were identical at all three
  (type agreement with the teacher 87.0 %).
- **Everything else is per request**: `stop`, `logprobs`, `max_tokens`,
  streaming, errors. A request that fails does not affect the others,
  unless the decode step itself fails, which fails every request in it.
- **Which server.** With one slot the HTTP server is tep's blocking one, as
  before. With more it is tep's fiber-per-connection server, still one
  process; handlers yield to each other after every decode step.
- **Needs the fused decode graph.** Models and options that run on the
  per-head graph (MoE, MLA, a quantized KV cache, `KV_FUSED=0`; see
  [`models.md`](models.md)) refuse `--slots` above 1 at boot.
- Each request's event carries `slot` and `shared_steps`, the number of
  decode steps it shared with another request.
- Cost grows faster than linearly past a point: every token's attention
  runs over all cells in use, its own slot's or not (the others are masked
  out, not skipped). Measured on the gx10 (same model as
  above, CUDA, ~360-token prompts sharing a 257-token boot prefix, ~58 new
  tokens, GPU otherwise idle): 0.67 requests/s one at a time, 4.0 at 8
  slots, 7.0–7.2 at 32, 5.9 at 64. A decode step takes 26 ms for one
  request, 30 ms for 8 and about 56 ms for 32. `llama-server --parallel 32`
  on the same file and requests gave 6.4 requests/s (one run; the GPU is
  shared, and runs disturbed by another job came out near 3–4 for both).

**Weight edits at inference and hidden states (toy#218).** Two things for
experiments that change a model's weights without training and test the
change with forward passes:

- **`--adapter F`** loads a LoRA adapter in llama.cpp's GGUF format (the
  file `llama-server --lora` takes): per adapted matrix
  `blk.N.<name>.weight.lora_a` (`[r, d_in]`) and `.lora_b` (`[d_out, r]`),
  `<name>` one of `attn_q`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate`,
  `ffn_up`, `ffn_down`, scaled by `adapter.lora.alpha / r`. The base
  weights are not touched, quantized or not: the delta is computed beside
  them, `W x + (alpha/r) B (A x)`. A rank-one edit `W <- W + c w uᵀ W` is
  the adapter `B = c w`, `A = uᵀ W`.
- **`"lora_scale": s`** on a request (`/v1/completions`,
  `/v1/chat/completions`, `/v1/hidden`) scales the adapter for that
  request: 1 is the default, 0 is the unedited model. Both weight sets
  therefore answer from one process and one copy of the model, and with
  `--slots` base and edited requests are decoded in the same batch. That,
  with a forced answer prefix and `logprobs` (toy#216), is a paired test.
- **`POST /v1/hidden`** returns the residual stream at the last token of a
  prompt and generates nothing. Body: `messages` (the chat template is
  applied, as in chat) or `prompt` (text or token ids); `layers`, a list
  where 0 is the embedding output and `i` the stream after block `i`
  (the numbering of Hugging Face `output_hidden_states`; default every
  block, 1 to L); `position`, a token index to read instead of the last
  one (negative counts from the end); `lora_scale`. Reply:
  `{"layers":[...],"position":p,"prompt_tokens":n,"d_model":d,"lora_scale":s,"data":[[d floats],...]}`,
  one row per layer in ascending order. A batch of prompts is a batch of
  requests (`--slots`).
- **Every reply names its weights (toy#221).** Non-streamed
  `/v1/completions`, `/v1/chat/completions` and `/v1/hidden` replies carry
  `"toy":{"model_sha256":"...","adapter_sha256":"...","lora_scale":s}`:
  the sha256 of the model file, of the adapter file (`""` without one) and
  the scale the request ran at, so a paired trace shows which weight state
  produced each result. `toy serve` hashes the two files at start and
  `toy pack` stores the model's hash in the artifact; a runner started
  without either reports `""`. The `run_start` event records both hashes
  and each request event the scale. Streamed replies are not labelled.

What it costs and what it does not do:

- An edited request reads neither the boot prefix nor rows cached under
  another scale (they were computed by other weights), so it evaluates its
  whole prompt; a slot keeps the scale of what it cached.
- The adapter adds two small matrix products per adapted matrix to a
  decode step that contains an edited request, and nothing otherwise.
- One adapter per server, chosen at start. The file must be F32 or F16,
  and every tensor in it must be one of the pairs above (anything else is
  refused at boot rather than skipped).
- Fused decode graph only (as `--slots`); `toy pack` does not embed an
  adapter.
- Measured: on the F32 fixture the adapter and a model with the delta
  written into its weights give the same ids, log-probabilities within
  5e-6 and hidden states within 5e-7 on CPU (5e-3 and 1e-3 on CUDA, where
  results already vary that much with batch composition). On the Q8
  student on CUDA, an adapter on block 10's two output matrices moves the
  log-probabilities of the next token by up to 1.1, and
  `llama-server --lora` with the same file moves them by the same amounts
  to within 0.04 (the two engines differ by 0.03 without any adapter).

**Decode speed (toy#206).** Prompts are evaluated in batches of up to 512
tokens, and each generated token is one small graph covering every
attention head of a layer. On the model above that is about 40 tokens/s
generated on CUDA (llama.cpp: 55–58 under the same load) and a 776-token
prompt in about 0.2 s. Models and options the fused step does not cover
run on the older per-head graph, which is several times slower on a GPU;
see "Decode graphs" in [`models.md`](models.md).

Streamed replies still go through tep's handlers (tep#276), so three
things are as tep has them there: a streamed `/v1/completions` reads
`prompt` as a token-id array only; a streamed chat reply always ends with
`finish_reason: "stop"`, also when it ended on `max_tokens`; and its `id`
is the constant `chatcmpl-tep`. Non-streamed replies are toy's own
(toy#216) and have none of the three.

The serve stack lives at `lib/toy/serve/openai/` and is driven by
`lib/toy/run/serve.rb` → `libexec/toy-serve`. `make gate-serve`,
`gate-serve-chat`, `gate-serve-chat-cuda`, `gate-serve-slots`,
`gate-serve-slots-cuda`, `gate-serve-adapter`, `gate-serve-adapter-cuda`,
`gate-kv-prefix` and `gate-kv-prefix-cuda` cover it.

> **One fix is carried as a vendored patch**, applied by
> `make vendor-tep` (`prep/patch_vendored_tep.rb`), so nothing has to be
> done by hand: a `\uXXXX` escape above ASCII in a request body was
> decoded to `?` (toy#215); it is now decoded to UTF-8.
>
> The byte-length, boot-guard, stdin-seed and scheduler-fairness fixes that
> used to be patched in are part of tep 0.11.12, which the Gemfile requires.
> `--slots` needs the last one
> ([OriPekelman/tep#291](https://github.com/OriPekelman/tep/issues/291)):
> before it, one request ran alone while the others waited.
>
> `infer`/`train`/`eval` don't link tep and are unaffected.

### `pack <model> -o <artifact> [--prefix F] [--name NAME] [--device cpu|cuda] [--ctx N] [--port PORT] [--stop-ids N,M] [--slots N]`
Build one executable that carries the model and serves it with no arguments
(toy#211).

```
toy pack student.gguf --prefix system.txt --name support-desk --device cuda -o support-desk
./support-desk                # serves on the packed port (default 4567)
./support-desk --port 8080    # or PORT=8080
./support-desk --slots 16     # or SLOTS=16; default: the packed --slots, else 1
./support-desk --describe     # card and provenance; loads nothing
```

What is inside:

- **The model**, as the GGUF file byte for byte. Its tokenizer is the one
  embedded in the GGUF, so pack a file that has one if you want chat.
- **The KV prefix** for `--prefix F`: the system message's attention state,
  computed once at pack time on `--device`. The artifact restores it at boot
  instead of evaluating it. Requests that start with that system message
  skip it, as with `toy serve --prefix-cache`.
- **A fixed config**: name, context, stop ids, default port, default
  number of slots.
- **A description**: the `toy describe` card, the model's sha256, the toy,
  spinel and ggml commits, the runner's hash, the prefix's hash and token
  count, and the shared libraries the artifact needs.

How it behaves:

- A pack reads its model, name, context, stop ids and KV prefix from itself.
  `MODEL_PATH`, `MODEL_NAME`, `MAX_T`, `STOP_IDS`, `PREFIX_FILE` and
  `PREFIX_CACHE` in the environment are ignored. Only the port can be moved.
- The weights are mapped from the executable in place, as `toy serve` maps
  them from the GGUF file. Starting a pack does not copy or unpack the model.
- The artifact is the serve runner with the files appended, each on a 64 KiB
  boundary, then an index and a 24-byte trailer (layout in
  `tinynn/tinynn_gguf.h`). The runner looks for the trailer in its own
  executable; a runner with nothing appended is the plain runner.

Limits:

- **Not statically linked.** The artifact needs the shared libraries
  `--describe` lists: libc, libstdc++, libm, OpenSSL and SQLite (through
  tep), and for CUDA the driver's `libcuda.so.1`. The CUDA runtime itself is
  linked in, which is why a CUDA artifact is 0.7 GB larger than its model.
- **Tied to the machine class it was packed on**: same OS and CPU
  architecture, and for CUDA a GPU the linked kernels support.
- **The KV prefix matches one system message exactly.** A request with a
  different system message is still answered correctly; it just evaluates
  its whole prompt.
- **No integrity check of the payload.** A truncated file or a damaged
  trailer is refused at start, and so is a KV prefix from another model,
  but a flipped byte inside the weights is not detected. Compare the file's
  own sha256 if that matters.
- `strip` and code signing rewrite an executable's tail; do not run them on
  an artifact. macOS is not tested.

`make gate-pack` and `gate-pack-cuda` cover it.

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
| `pack`  | none of its own: appends files to the `serve` runner | the artifact named by `-o` |

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

