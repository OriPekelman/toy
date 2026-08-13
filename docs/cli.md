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
Train a model and record a run bundle. `<recipe>` is required; the accepted
recipes are:

| recipe | what it does | recipe-specific flags |
| --- | --- | --- |
| `from-scratch` | train a tiny Llama-shape model from random init (the default) | — |
| `lora` | LoRA adapters over a frozen base GGUF | `--model <gguf>` (base, default `data/smollm2-135m-native.gguf`), `--rank N` (default 8) |
| `warm-start` | fine-tune from donor embeddings over a streamed corpus | `--corpus <path>` (default `data/ts_seqs.bin`), `--init <mode>` (default `scratch`) |
| `vit-tiny` | ViT-Tiny image classifier on the committed smoke corpus (CPU-only) | — |
| `franken` | dense Llama-shape model with a **per-layer credit-assignment policy** (the DFA research lane) | `--policy`, `--policy-scope`, `--dfa-granularity`, `--dfa-b-*`, `--align-events`, `--optimizer`, `--shape`, `--corpus`, … |
| `franken-moe` | the same, MoE-shaped (routed experts, dense or top-1) | as `franken` plus `--experts`, `--routing`, `--moe-policy`, `--dfa-feedback`, `--lr-schedule`, `--lr-control`, … |
| `mlp` | MLP classifier on a seeded synthetic task — the DFA **anchor** lane (CPU-only) | `--policy`, `--classes`, `--hidden`, `--features`, `--layers`, `--task`, `--val-batches`, `--dfa-b-*`, `--align-events` |
| `ctr` | CTR/recommender tower: embedding tables + MLP + **scalar** sigmoid head, scored by AUC (CPU-only) | `--policy`, `--fields`, `--cardinality`, `--emb`, `--numeric`, `--hidden`, `--layers`, `--pairs`, `--lin-scale`, `--base-rate`, `--fm-branch`, `--val-batches` |
| `gnn` | GCN-class node classification over a normalised adjacency, transductive semi-supervised (CPU-only). Runs on a seeded graph or on real Cora (`ruby prep/fetch_cora.rb`) | `--policy`, `--graph`, `--layers`, `--hidden`, `--classes`, `--features`, `--nodes`, `--degree`, `--homophily`, `--feat-signal`, `--train-per-class`, `--task`, `--feedback-route`, `--feedback-hops`, `--weight-decay` |
| `ssm` | selective-scan / Mamba-lite sequence classifier — an UNROLLED input-dependent linear recurrence + conv branch + gating, last-step readout (CPU-only) | `--policy`, `--dfa-cut`, `--selection`, `--layers`, `--seq`, `--d-model`, `--d-inner`, `--conv-k`, `--classes`, `--task`, `--cue-span`, `--noise`, `--dt-init` |
| `lstm` | gated recurrence: a stack of textbook LSTM cells UNROLLED over T, last-step readout, on the same task + cut axis as `ssm` (CPU-only) | `--policy`, `--dfa-cut`, `--layers`, `--hidden`, `--features`, `--seq`, `--classes`, `--task`, `--cue-span`, `--noise`, `--batch`, `--val-batches`, `--task-seed`, `--lr`, `--dfa-b-*` |
| `gtx` | graph transformer — masked self-attention whose mask **is an adjacency**, small relation head (16 classes), inductive relational retrieval (CPU-only). `--retrofit` adds the DFA-adapter mode | `--policy`, `--dfa-cut`, `--layers`, `--d-model`, `--heads`, `--d-ff`, `--entities`, `--types`, `--features`, `--task`, `--noise`, `--batch`, `--val-batches`, `--task-seed`, `--dfa-b-*`, `--retrofit`, `--adapter-policy`, `--adapter-layers`, `--adapter-rank`, `--pretrain-steps`, `--pretrain-lr`, `--no-freeze-backbone` |
| `diff` | latent diffusion denoiser — time-conditioned eps-prediction over a LOW-DIM latent, scored by a generative metric (CPU-only) | `--policy`, `--latent`, `--time-feat`, `--layers`, `--hidden`, `--task`, `--modes`, `--spread`, `--mode-scale`, `--diff-steps`, `--beta-lo`, `--beta-hi`, `--eval-n`, `--align-events` |
| `ae` | per-token latent AUTOENCODER (capstone P1a) — bidirectional encoder -> **d-dim per-position bottleneck** -> **per-position** decode head back to the byte. **All BP: no `--policy`, no DFA, no diffusion.** Scored by the NOISE MARGIN, not clean reconstruction (CPU-only) | `--text`, `--latent`, `--context`, `--layers`, `--d-model`, `--heads`, `--d-ff`, `--noise-eval`, `--noise-seed`, `--val-batches`, `--val-frac-pct`, `--task-seed`, `--lr`, `--warmup`, `--target-ce`, `--eval-every`, `--probe-batches` |

An unknown recipe is rejected loud (`supported: 'from-scratch', 'lora',
'warm-start', 'vit-tiny', 'franken', 'franken-moe', 'mlp', 'ctr', 'gnn', 'ssm', 'lstm', 'gtx', 'diff', 'ae'`). `--arch gpt2`
trains the from-scratch GPT-2 arch (`from-scratch` only, CPU-only);
`--device cuda|metal` selects the per-device runner (`metal` is macOS-only;
GPT-2, ViT-Tiny, `mlp`, `ctr`, `gnn`, `ssm`, `lstm`, `gtx`, `diff` and `ae` are CPU-only, and `franken`'s macro-DFA mode is
CPU-only by decision).
Defaults: `--steps 5`, `--seed 0`, `--arch llama`, `--device cpu`, `--out runs/<id>`.
Writes the run bundle (see [Run bundle layout](#run-bundle-layout)) and echoes
the byte-deterministic loss curve to stdout.

#### Flag x recipe is a MATRIX, and it is gated

Every recipe-specific flag is rejected under the wrong recipe (exit 2,
`only valid with recipe …`), *before* any build or runner spawn. The
table lives in one place (`lib/toy/core/cli/train.rb`'s `flag_matrix`)
and `prep/train_cli_matrix_gate.rb` asserts every row — recipe parity is
a cell flip, not a scattered set of one-off checks.

#### The DFA lanes in one paragraph

`--policy chain,dfa,…` sets credit assignment **per layer**:
`chain` = backprop, `dfa` = fixed random feedback, and on `mlp`, `ctr`
and `gnn` also `frozen` (the control arm — the layer stays at init while
the head still trains). `--policy-scope attn|ffn|all` selects which tensors a `:dfa`
layer policies (dense lanes; default `attn`).
`--dfa-granularity matmul|block` selects **micro** (per-weight) vs
**macro** (one tap per block output, full BP inside the block) DFA.
`--align-events` emits the `align` telemetry described in
[events.md](events.md#align-credit-assignment-telemetry--the-dfa-lanes).
`--optimizer adamw|muon|radam` (`radam` is `franken`-only; `sgd` is
`franken-moe`-only).

`gnn` adds `--feedback-route direct|structure` (+ `--feedback-hops N`):
`structure` spreads the error along the graph before the random
projection, so an UNLABELLED node — which has exactly zero direct error
in this semi-supervised setting — receives a pseudo-error from its
labelled neighbourhood. This is deliberately **not** spelled
`--dfa-feedback`: `franken-moe`'s flag of that name selects how the
feedback matrix B is *updated* (`fixed|kolen-pollack`), a different axis.
A different meaning gets a different name (the tao#18 `--policy-scope`
discipline).

`ssm` and `lstm` add `--dfa-cut layer|step`; `--selection selective|lti`
is `ssm`-only. `layer` cuts only the layer boundary and injects the
random feedback once, at the readout step, with BPTT intact inside the
layer; `step` additionally detaches the state at every timestep (on
`lstm`, both `h_{t-1}` and `c_{t-1}`), so no gradient crosses time at
all. It is **not** spelled `--dfa-granularity`: that one
picks matmul-vs-block in *depth*, this picks layer-vs-step in *time*.
`--align-events` is REJECTED on `ssm` and `lstm` — their DFA update
arrives through autodiff from the surrogate roots, so it lands in the
same accumulator a BP run uses and a cosine against it would mean
nothing.

The two recurrent lanes share `--seq`, `--classes`, `--task cue|mean`,
`--cue-span` and `--noise` because `lstm` reuses `ssm`'s delayed-cue
generator **unchanged** — that is what makes the pair an architecture
comparison rather than two anecdotes. They do *not* share
`--d-inner`/`--conv-k`/`--selection`/`--dt-init`, which name
selective-scan parts an LSTM has no counterpart for; those are rejected
on `lstm` by the same matrix.

`lstm` also carries `--clip-grad NORM` (toy#162): global-norm gradient
clipping, applied in-graph before the AdamW moments, **off by default
and byte-null when absent**. It exists as the fair control for the
lane's stability claim rather than as a tuning knob — measured, it does
*not* make the BP arm reliable (7/12 solved cells unclipped, 8/12 at
norm 1.0, 7/12 at norm 0.1), it re-rolls which cells work. It is
rejected on every other recipe.

`lstm`'s BP arm is **bimodal** on this task — it solves it (~1.000) or
sits at chance (.250), and which learning rate works depends on the
*seed*. Its fair cell is therefore
`--lr 0.02 --warmup 200 --steps 4000`, the BP arm's own best over a
swept (lr x warmup x seed) grid, and that cell is always written out in
full. It is deliberately **not** folded into the defaults: toy#157
shipped the 200-step ramp as a default for exactly one commit, and a
consumer's `--lr 0.03 --steps 2000` silently inherited it and relabelled
a 3-seed matrix. Defaults on this lane change nothing (`--warmup 0`);
the cell is spelled out instead.

The `mlp`, `ctr` and `gnn` lanes print extra stdout lines after the
curve — `val: acc=… loss=… n=…`, `val: auc=… logloss=… n=… pos=…`, and
for `gnn` a `train:` line as well — because held-out accuracy / AUC, not
loss, is the metric their success criteria are stated in. `gnn` reports
both sides of the split because its arms separate as much by their
train/val GAP as by val accuracy. `ssm` also prints
`graph: nodes=…`, the realized graph size — how a sweep over `--seq`
reads the arms' scaling. Read it as node count, never as bytes: in a
graph autodiff every forward tensor is materialised whatever the credit
rule. For what a *streaming* implementation would hold instead, read the
`stream:` line beneath it (toy#159, below).

`lstm` prints the same line with the missing half — `graph: nodes=…
bytes=…`, where `bytes` is `ggml_nbytes` summed over every node of the
realized graph, i.e. the materialised activation footprint its ticket's
memory target is stated in. The caveat above still travels with it and
is the finding: cutting BPTT does **not** shrink the graph. Measured at
T=64, the per-step cut costs 17% MORE bytes than BPTT (15 578 112 vs
13 289 988), and the gap grows with `--seq`.

#### The `stream:` line — ANALYTIC, not measured (toy#159)

Both recurrent lanes print a second memory line, and the distinction
between the two is the whole point:

```
graph:  nodes=2116 bytes=15578112                       # MEASURED — what toy builds
stream: bptt=3868160 sqrt_t=984576 cut=78336 cut_vs_bptt=49.37x cut_vs_sqrt_t=12.56x replay=2x_fwd
```

`stream:` is computed from the cell's shapes, **not** by walking a
graph: it is what a streaming implementation would have to hold, which
is the quantity the F19/F21 success target is actually about and the one
no graph measurement here can exhibit.

- `bptt` — every step's activations live for the ordered backward: O(T).
- `cut` — the per-step cut, replaying the forward once the error is
  known and updating each step as it passes: **O(1) in T**. Identical
  across `--seq 16/64/128`, and that invariance is gated.
- `sqrt_t` — BPTT's own best counter-move, sqrt-T checkpointing at
  ~1.5x forward compute, quoted so the cut is not compared only against
  BP's worst case.
- `replay=2x_fwd` — the price of `cut`, stated inline. It is not free.

The win is real but it is not a property of the credit rule alone: the
surrogate's error is known only after the last-step readout, so an O(1)
implementation must replay the forward. **What cutting the time axis
buys is the LEGALITY of that replay** — steps become independent, so
they can be revisited in any order — which is precisely what BPTT
cannot do, and precisely why the *measured* graph comes out bigger
rather than smaller. Weights, optimizer moments and gradient buffers are
excluded from both sides: they are equal across arms and independent of
T, so they would only dilute the ratio.

`gtx` puts **attention** on the same `--dfa-cut layer|step` axis: `layer`
taps each block's output with BP intact inside the block, `step`
additionally detaches the attention **probabilities** so no gradient
crosses the token-mixing (Q and K then get their own random-feedback
taps, or the arm would just be "attention frozen"). Its `--task
relational|local` mirrors the other lanes' degenerate control: under
`local` a node's type is readable from its own features, so no retrieval
is needed at all.

**The `gtx` arms do not share a learning rate, and that is a result
rather than an inconvenience.** BP's cell is `--lr 0.003`; the DFA arms
want ~3x less, and at BP's rate DFA reads chance. A single-LR matrix on
this lane would report "attention is DFA-hostile" — the opposite of what
it measures. Always state the LR with the arm.

#### `gtx --retrofit` — adapting a frozen pretrained backbone (toy#161)

One process, two phases: `--pretrain-steps` of BP on the 16-class
relation task, then the backbone is **frozen and detached** and only
added capacity trains on a *different* task — a 4-class **modular sum**
over the same graph — under `--adapter-policy chain|dfa|frozen`.

```
toy train gtx --retrofit --pretrain-steps 1500 --steps 1500   --adapter-policy dfa --pretrain-lr 0.003 --lr 0.001
```

Three things about it are load-bearing:

- **The retrofit label is many-to-one on purpose.** Any *bijective*
  relabeling — including "new relation types" — is absorbed by a
  retrained linear head, which would leave the frozen control unable to
  lose. The modular sum provably needs an **interaction** between the
  two endpoint representations, which is why the adapters sit at the
  **pair site** rather than inside the backbone.
- **Both phases run in one process**, so every arm's backbone is
  bit-identical by construction. The runner prints `backbone: sig_pre=…
  sig_post=…`; under a freeze the two are identical, which is a
  measurement rather than a promise that nothing was stepped.
**The two-command workflow (toy#164).** `--ckpt-every K` writes the
backbone as a `toy-gtx/v1` GGUF into the run dir, and `--load-ckpt PATH`
loads it and **skips the pretrain phase**, so one pretrain can serve many
retrofit arms:

```
toy train gtx --steps 1500 --ckpt-every 1500 --out $BB
toy train gtx --retrofit --load-ckpt $BB/step_1500.gguf --adapter-policy dfa --lr 0.001
toy train gtx --retrofit --load-ckpt $BB/step_1500.gguf --adapter-policy chain --lr 0.003
```

Only the BACKBONE is written — never the adapters, which must start at
identity. A shape mismatch is refused naming both sides, and
`--load-ckpt` requires `--retrofit` on this lane. The gate asserts the
round trip is **bit-exact** (`bb_sig` string-equal across write and
load), because a round trip that is merely close is one that lost bits
and every arm sweeping off that checkpoint would inherit them.

For the *bar*, prefer the one-process form (no `--load-ckpt`): it makes
every arm's backbone bit-identical **by construction** rather than by a
round trip that itself has to be trusted.

- **The cost line is analytic, and it credits the freeze, not DFA.**
  `retrofit: … bb_grad_bytes_avoided=…` is what *freezing* saves, and it
  is the same for both credit rules at this adapter site; DFA's own
  saving is the backward through the adapter stack only. The realized
  graph counters cannot show this (they exclude the backward
  extension — freezing even reads +1 node, the detach), which is the
  mirror of tao#21's caveat.

`diff` prints `gen: energy=…` — the ENERGY DISTANCE between generated
and held-out real samples. It is the lane's headline metric and **lower
is better**, the opposite direction from every accuracy-scored lane
here; a slightly negative value means the two sets are indistinguishable
at that sample size, not that something broke.

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
`TAO_RUN_DIR`, `TOY_RUN_ID`) passed via `Open3`, so a stale caller environment
cannot leak in.

`infer` / `train` / `eval` accept `--device cuda|metal`, which selects a separate
per-device runner binary (e.g. `libexec/toy-infer-cuda`, `libexec/toy-train-cuda`;
`metal` is macOS-only). They are single-type binaries by Spinel necessity (one
backend module per binary), built on demand the same way. `serve` is CPU-only.

## Examples

The CLI is the supported entry point; the curated `examples/` set (toy#60)
is the narrated library-API tour — seven single-file programs Spinel
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
| `TAO_RUN_DIR=<dir>` | Emit the `events.jsonl` stream (and a final checkpoint) here |
| `TOY_EVENTS=<path>` | Raw alternative to `TAO_RUN_DIR`: full path to the events file |
| `CHECKPOINT_EVERY=N` | Write `weights/step_<N>.gguf` every N steps (+ `latest` symlink) |
| `TOY_DESCRIBE=json\|mermaid\|text` | Dump the compute graph and exit (no training) |
| `TOY_GRAD_SENTINELS=1` `TOY_DRIFT_EVERY=N` `TOY_CKA=N` | Instrumentation events |

See [examples/README.md](../examples/README.md) for the full roster and
invocations.

## Example env knobs

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

### `ae` — the per-token latent autoencoder (toy#165, capstone P1a)

The diffusion-text-LM capstone needs a per-token continuous latent of 4–8
dims (F20/toy#156's DFA-favourable window). Whether **text** survives such
a latent is unrun in the literature, and P1a is the cheapest decisive
test. It is **all BP** — no `--policy`, no DFA, no diffusion; those are
P1c and P1b.

```
ruby prep/fetch_text.rb --all          # once: three pinned byte corpora
toy train ae --text data/ae_names --latent 8 --context 256 --steps 4000 \
             --noise-eval 0,0.25,0.5,1,2 --seed 0
```

**Clean reconstruction is vacuous and is not the headline.** Packing a few
dozen codepoints into 4 continuous dims is ample analog capacity, so clean
accuracy sits at ~1.0 at *every* latent and cannot tell 4 from 32. The
read is the **noise margin**: reconstruction accuracy as the latent is
perturbed by Gaussian noise scaled to the latent's own per-dim std (so it
is comparable across `d`), summarised by the **half-accuracy SNR** — the
sigma at which accuracy crosses half of that cell's clean value. When the
curve never crosses inside the grid it is reported as `>=<max sigma>`,
never clamped.

**The alphabet is the second axis** (tao#22). The margin is packing-limited,
so a curve is unscoped without the number of symbols the head actually had
to separate: at N=27 the `d=4` problem is about as hard as N=256 at `d=8`,
and the alphabet alone can manufacture a `go`. Three pinned corpora ship:

| pack | source | bytes | distinct |
|---|---|---|---|
| `data/ae_names` | makemore names list | 228 K | 27 |
| `data/ae_shakespeare` | tinyshakespeare | 1.1 M | 65 |
| `data/ae_udhr` | UDHR, 388 languages, UTF-8 | 5.7 M | 201 |

Every run reports both the pack alphabet and the `val_alphabet` actually
observed in the scored windows. **Preliminary** surface (seed 0, 800 steps,
context 128, `d_model` 128 — Tao's cells are 4000 steps, so read these as
shape, not as the result):

| corpus (val alphabet) | d=4 | d=8 | d=32 |
|---|---|---|---|
| names (27) | 0.82 | 1.25 | ≥2.0 |
| shakespeare (53) | 0.66 | 0.98 | 1.93 |
| udhr (104) | 0.70 | 0.93 | 1.62 |

There is **no `--vocab`** on this lane. `--vocab` already means an integer
pack width on `franken`/`franken-moe`, and the `ae` head is byte-wide (256)
on every corpus by construction — sizing it to the observed alphabet would
confound the alphabet axis with head capacity. `--latent` is deliberately
the **same flag** the `diff` lane uses: it is the same quantity, and the
capstone compares those two numbers directly.

`--context` is also the **batch** — the encoder attends within a window, so
its T positions are the T reconstruction targets and there is no separate
`--batch`.

The controls: a **zeroed** latent leaves the head with only its bias, so it
lands at the unigram floor by construction — reported, but not gated, since
it is an identity. The **shuffled** latent (permuted across positions) is
the one with teeth and the one `prep/ae_gate.rb` gates: each position
decodes a real latent from the same distribution, just the wrong one, so it
*can* score above the floor if the head learned a prior.

#### `--target-ce` — comparing cells at matched FIDELITY, not matched steps

A noise margin is **not invariant to convergence**, and a sweep run at a
fixed step budget measures both. Measured on this lane: with clean accuracy
pinned at 1.000 throughout, names d=32 goes half-SNR **1.92 → 2.44** as CE
falls 1.9e-3 → 1.5e-9 (cross-entropy keeps rewarding wider logit separations
long after accuracy stops moving); and udhr d=4 goes **0.599 → 0.479** while
accuracy *rises* 0.951 → 0.993 (newly-learned rare bytes have to be packed
into the same `d` dims). Two opposite-signed biases, so the confound distorts
the shape of a surface, not just its scale.

The first 36-cell P1a surface was run at matched steps and its clean CE
spanned **seven orders of magnitude**. Re-run at matched CE it spans a factor
of 5.4, and the fitted law changes qualitatively.

```
toy train ae --text data/ae_udhr --latent 8 --context 256 \
             --steps 12000 --target-ce 0.05 --eval-every 10
```

Each cell trains until held-out CE crosses the target, then reports
`converged: target_ce=… achieved_ce=… steps=… matched=1`. A cell that never
reaches it reports `matched=0 … NOT REACHED` — an unmatched cell must not
join a matched surface silently.

**The probe must not perturb training, and that is not free.** ggml's AdamW
kernel is `m = m*b1 + g*(1-b1); v = v*b2 + g²*(1-b2); w = w*(1-lr*wd) - lr*mh/vh`,
so the `lr=0` eval hp every lane uses freezes the **weights** but still lets
the moments absorb the probe's gradients. Harmless at end-of-run; corrupting
for a periodic probe. The probe hp sets `b1 = b2 = 1.0`, making both moment
updates the identity. `prep/ae_gate.rb` asserts the consequence: a probed
run's training curve is byte-identical to an unprobed one.

`--probe-batches K` (default 4) runs the stopping check on a subset so the
interval can be fine. A coarse interval lets fast cells overshoot the target
by an order of magnitude — the exact mismatch the flag exists to remove. The
full held-out CE on the `val:` line stays the authoritative matched quantity;
both are printed.

Off by default and byte-null when off.
