# The toy/v1 event stream

Every toy run — training, evaluation, or serving — can emit a structured
event stream as JSON Lines (one JSON object per line) to a single file:

```
runs/<run_id>/events.jsonl
```

This file is the contract between toy (the **producer**) and any
**consumer** — live TUIs, post-run analysis, adapters to W&B / MLflow /
tensorboard, report generators, and the Tao introspection sibling. The
schema is versioned; the first event of every file carries
`"schema":"toy/v1"`.

The stream is **append-only, line-delimited, monotonic in time**. A crash
leaves a truncated last line; consumers skip lines that fail to parse
rather than aborting.

## Run directory layout

A run lives under `runs/<run_id>/` in the project (the directory holding
`toy.yml`). The CLI creates it before shelling out to the compute runner
(`lib/toy/core/cli/train.rb` → `lib/toy/run/train.rb`).

```
runs/
  <run_id>/
    events.jsonl        # the structured event stream (this doc)
    flow.json           # the realized compute-graph DAG (toy/v1, ToyDescribeFlow)
    weights/            # checkpoint snapshot(s), GGUF
```

`flow.json` is written once at startup (right after `realize!`) by every training
runner, so the run bundle is self-describing — a consumer reads the arch DAG
without a separate `TOY_DESCRIBE` pre-pass. Emitted via
`ToyDescribeFlow.emit_flow_json(run_dir, sess)`.

`run_id` is generated from `run_id_template` in `toy.yml`
(`lib/toy/core/config.rb`; default `{arch}-{date}-{seq}`, e.g.
`llama-20260531-001`). The placeholder tokens are `{arch}`, `{date}`
(YYYYMMDD), `{time}` (HHMMSS), and `{seq}` (zero-padded daily counter).

The stream and the checkpoint are file-side only — never stdout — so the
byte-gated `step <N>: loss=<float>` stdout line stays clean (the train
gate, `prep/train_gate.rb`, parses stdout against
`prep/fixtures/train_baseline.txt`).

## Design principles

1. **Cheap when off, cheap when on.** Events emit via the trace primitive
   (`tnn_events_*`). The off-path is a single length check; on-path is a
   buffered append. No serialization in the hot loop unless active.
2. **Schema is open.** Unknown keys in an event MUST be ignored by
   consumers. New event kinds and new optional keys can be added in v1
   minor revisions without breaking existing consumers. Removing a key
   requires a schema bump.
3. **One run = one file.** No multi-file fragmentation. A run's
   `events.jsonl` is everything needed to reconstruct or compare it.
   Weights are separate (`runs/<run_id>/weights/`).
4. **Aggregation lives outside.** toy emits raw events. Per-step
   summaries and derived statistics (drift, CKA, Pareto crossover) are
   computed downstream from the raw stream. The hot path stays simple.
5. **Human-readable.** Pretty-print with `jq` and it should be legible.
   Field names are full words; units are explicit (`_us` microseconds,
   `_bytes` sizes).

## Event envelope

Every event has at minimum:

```json
{ "kind": "<event_kind>", "t": <seconds_since_run_start, float> }
```

Optional but conventional:

- `step` — integer training/decode step the event belongs to. Omit for
  events that aren't step-scoped (`run_start`, `run_end`).
- `phase` — `"train"` | `"eval"` | `"decode"` | `"serve"`. Distinguishes
  which loop the event belongs to (a single run may have several phases).

## Event kinds

### `run_start` (exactly one per file, first)

Static metadata about the run, flushed before any compute begins so a
crashed run still leaves identifying metadata. The from-scratch train
runner (`lib/toy/run/train.rb`) emits, at minimum, `schema`, `t`,
`started_at`, `run_id`, `phase`, `host`, `backend.kind`, `git`, a `model`
block (`arch`, `name`, `vocab`, `d_model`, `n_layers`, `n_heads`, `n_kv`,
`d_head`, `d_ff`), and a `config` block.

```json
{
  "kind": "run_start",
  "schema": "toy/v1",
  "t": 0.0,
  "started_at": "2026-05-31T09:30:00Z",
  "run_id": "llama-20260531-001",
  "phase": "train",
  "host": { "name": "gx10", "os": "linux", "arch": "aarch64" },
  "backend": { "kind": "cpu" },
  "git": { "sha": "4bf48d3", "branch": "main" },
  "model": {
    "arch": "llama", "name": "from-scratch-tinystories",
    "vocab": 49152, "d_model": 576, "n_layers": 30,
    "n_heads": 9, "n_kv": 3, "d_head": 64, "d_ff": 1536,
    "weight_type": "f32"
  },
  "config": { "context": 32, "steps": 5, "lr": 0.001, "seed": 0 }
}
```

`config` is an open dictionary of experiment key/values. A `compare_to`
field (a baseline `run_id`) is optional; if set, consumers should expect
`compare` events later in the stream.

`model.weight_type` (optional) is the weight storage/compute dtype —
`"f32"` (default), `"f16"`, or `"bf16"` — surfaced by the from-scratch
trainer for the mixed-precision path (GH#9). Consumers asserting numerical
parity should branch on it (f16/bf16 runs are tolerance-compared, not
byte-exact, against the f32 baseline).

### `step`

A single training step OR decode step (`phase` disambiguates). Emitted
once per step with canonical metrics. The train runner emits `loss`,
`lr`, `tokens`, and `wall_us`.

```json
{ "kind": "step", "phase": "train", "t": 0.418, "step": 1,
  "loss": 9.2312, "lr": 0.001, "tokens": 32, "wall_us": 71200 }
```

Conventional fields:

- `loss` — the primary objective.
- `ppl` — `exp(loss)` for LM training; omit if not LM.
- `grad_norm` — global L2 of all PARAM-flagged grads; may be omitted if
  expensive.
- `tokens` — new tokens processed this step (throughput downstream).
- `compute_us` — the `tnn_compute*` duration. `wall_us` — full step
  including uploads/downloads.

For `phase: "decode"`:

```json
{ "kind": "step", "phase": "decode", "t": 0.084, "step": 5,
  "token_id": 12345, "logprob": -2.314, "wall_us": 84210 }
```

### `tap`

A named tap point downloaded a tensor and computed summary stats. One
event per (step, region, layer). Stats vocabulary:

| Stat          | Meaning                                              |
|---------------|------------------------------------------------------|
| `l2`          | √(sum of squares) over the full tensor               |
| `min`, `max`  | element-wise extrema                                 |
| `mean`        | arithmetic mean                                      |
| `abs_mean`    | mean of absolute values                              |
| `sparsity`    | fraction of elements with `|x| < eps` (eps=1e-6)     |
| `nan_count`   | number of NaNs                                        |
| `per_head_l2` | array, one L2 per head (multi-head tensors)          |
| `histogram`   | array of 32 bin counts spanning min..max             |
| `gram`        | T×T Gram matrix `G = Aᵀ·A` for activation `A` of shape `[d, T]` — feeds linear CKA |

`region` is the only convention; it is expected to be stable across runs
so comparisons work. Suggested naming `<sublayer>_<phase>`, e.g.
`attn_q_post_rope`, `ffn_gate_pre_silu`, `embed_post_lookup`.

```json
{ "kind": "tap", "phase": "train", "t": 0.420, "step": 1,
  "region": "attn_q_post_rope", "layer": 0, "head": null,
  "shape": [64, 1], "dtype": "f32",
  "l2": 0.0042, "abs_mean": 0.00052, "nan_count": 0 }
```

When a tap carries a `gram` field (a T×T symmetric PSD matrix
`G = Aᵀ·A`), consumers compute linear CKA from gram pairs. Suggested CKA
regions: `attn_norm`, `ffn_out`, `resid_post_block`.

### `grad`

Per-parameter gradient statistics, captured on the backward pass. Same
stats vocabulary as `tap`; one event per (step, param). `param` matches
the GGUF tensor name where possible. LoRA adapters use a documented
suffix: `blk.0.attn_q.weight#lora_a` / `#lora_b`.

```json
{ "kind": "grad", "phase": "train", "t": 0.420, "step": 1,
  "param": "blk.0.attn_q.weight", "head": 0, "shape": [64, 576],
  "l2": 0.0312, "abs_mean": 0.00041, "nan_count": 0 }
```

### `op_timing`

Aggregate of per-op timings, one event per step, bucketed by ggml op
kind. The cheap always-on summary; the full per-op Chrome trace is opt-in
(`TRACE_OPS=1`).

```json
{ "kind": "op_timing", "t": 0.420, "step": 1, "compute_us": 70510,
  "buckets": { "MUL_MAT": 12340, "OUT_PROD": 18820, "ADD": 6910,
               "ROPE": 2150, "SOFT_MAX": 1240, "OPT_STEP_ADAMW": 5430,
               "other": 23620 } }
```

### `drift`

Distance of a parameter from its init value, emitted on a schedule (not
every step). `cos_to_init` is cosine similarity to step-0 weights (1.0 =
no drift); `l2_to_init` is the L2 of the delta.

```json
{ "kind": "drift", "t": 41.8, "step": 100,
  "param": "blk.0.attn_q.weight", "head": 0,
  "cos_to_init": 0.9912, "l2_to_init": 0.0421 }
```

A per-token variant adds `token_id` (the embedding row index) and `freq`
(training-corpus occurrence count), one event per vocab row per tick:

```json
{ "kind": "drift", "phase": "train", "t": 12.4, "step": 100,
  "param": "token_embd.weight", "token_id": 7,
  "cos_to_init": 0.94, "l2_to_init": 0.080, "freq": 153 }
```

### `sample`

A decoded completion from a fixed prompt — gives report consumers a sense
of *what the model produces* alongside loss. `prompt` and `text` are
already-detokenized strings.

```json
{ "kind": "sample", "phase": "decode", "t": 506.1, "step": 200,
  "prompt": "Once upon", "text": "Once upon a time there was a cat." }
```

### `eval`

A held-out evaluation result. `phase` becomes `"eval"`. May appear
several times in a run (mid-training + final). `extra` is an open
dictionary — any benchmark suite's outputs can nest there.

```json
{ "kind": "eval", "phase": "eval", "t": 123.4, "step": 1000,
  "name": "validation", "loss": 3.421, "ppl": 30.6, "samples": 512,
  "extra": { "hellaswag_acc": 0.31, "arc_easy_acc": 0.42 } }
```

### `compare`

Live comparison against the `run_start.compare_to` baseline, one per step
where the baseline has a matching step. `verdict` is `"ahead"`,
`"behind"`, `"par"` (within 1%), or `"abort"` (a signal, not a
directive — auto-abort policy lives in the consumer).

```json
{ "kind": "compare", "t": 0.420, "step": 1,
  "baseline_run_id": "llama-20260530-001",
  "loss_delta": -0.013, "ppl_ratio": 0.987, "verdict": "ahead" }
```

### `region_enter` / `region_exit`

Marks a logical region — typically the start/end of an experimental
variant. Consumers partition events by region for "compare new bit vs
baseline bit." Regions can nest; consumers handle this with a stack.

```json
{ "kind": "region_enter", "t": 0.0,   "step": 0,   "region": "warmup" }
{ "kind": "region_exit",  "t": 100.0, "step": 100, "region": "warmup" }
```

### `note`

Free-form human or AI annotation.

```json
{ "kind": "note", "t": 100.4, "step": 100,
  "text": "switched LR schedule from cosine to linear" }
```

### `run_end` (exactly one per file, last)

```json
{ "kind": "run_end", "t": 1234.5, "ended_at": "2026-05-31T09:50:34Z",
  "reason": "completed", "final_step": 1000, "final_loss": 2.31,
  "quality_gate": { "passed": true, "metric": "loss_ratio",
                    "value": 0.71, "threshold": 0.9 },
  "exit_code": 0 }
```

`reason` is **execution status** — `"completed"`, `"interrupted"`,
`"aborted"`, or `"errored"`. A run that ran to completion without an
exception is always `"completed"` regardless of quality; an
interrupted/aborted/errored run conventionally adds an `error` field.

An optional `quality_gate` carries the **research verdict**, distinct
from `reason`: a run that completed cleanly but missed a quality bar gets
`reason:"completed"` + `quality_gate.passed:false`. Fields: `passed`
(bool, required if the block is present), `metric` (string), `value`
(number), `threshold` (number).

`exit_code` mirrors the OS exit code (0 success, 1 quality-gate fail, 2+
usage errors), independent of `reason` so CI/make contracts stay sharp.

## Serving telemetry

The serving path (`toy serve` → `lib/toy/run/serve.rb` →
`lib/toy/serve/openai/`) uses the same envelope so the consumer ingest
path is uniform across training and serving. A serving run is defined to
emit `run_start` at server boot (with `backend.kind`, `host`, `git`,
`model`) and `run_end` at shutdown, plus one per-request `eval` event
with `phase:"serve"` and `name:"request"`:

```json
{ "kind": "eval", "phase": "serve", "t": 87.42, "name": "request",
  "extra": {
    "model": "smollm2-135m",
    "prompt_tokens": 12, "completion_tokens": 8, "latency_us": 87000,
    "sampling": { "temperature": 0.7, "max_tokens": 256 },
    "request_id": "chatcmpl-abc"
  } }
```

`phase:"serve"` distinguishes inference telemetry from `phase:"eval"`
(held-out evaluation during training). The `extra` open-bag carries
caller-specific fields without bloating the schema; consumers route on
`extra.request_id` for request-keyed aggregation.

## Emitting events (producer side)

Runners build event JSON with the tep-free `Toy::Json` ordered-object builder
(`lib/toy/io/toy_json.rb`) and emit it via `TinyNN.tnn_events_emit(j.j_dump)` —
**not** hand-concatenated strings (which were unescaped and comma-fragile). The
shared `run_start` provenance — `host{}` + `backend{}` + `git{}` — is one call,
`Toy::Events.add_provenance(rs, host_name, host_os, host_arch, backend_kind)`
(`lib/toy/io/toy_events.rb`), which reads git via `Toy::Git`
(`lib/toy/io/toy_git.rb`). Hyper-parameter vectors use the `Toy::AdamW` value
object (`adamw.hp(step)`), not a hand-filled `Mat(1,7)`. All four are pure-Ruby
(no FFI), Spinel-compiled, and shared across the CPU/CUDA/Metal runners.

## Producer guarantees

1. `run_start` is written before any other event and flushed before
   compute begins.
2. The file is opened `O_APPEND`. One run, one writer — concurrent
   writers to the same file are not supported.
3. Each event is a complete line ending in `\n`; newlines inside strings
   are JSON-escaped.
4. Time-since-start (`t`) is monotonic; on clock skew, toy clamps.
5. `step` numbers are dense within a phase unless explicitly sub-sampled.

## Consumer guarantees

1. Skip lines that fail JSON parse (likely a torn final line from a
   crashed writer) — log, don't abort.
2. Treat unknown `kind` values as informational; don't error.
3. Treat unknown keys within a known event as informational; pass them
   through if forwarding to another sink.
4. Don't assume the file is closed — tail-follow semantics work for live
   consumers.

## Versioning

The `"schema":"toy/v1"` field on `run_start` is the version.

- **v1 minor revisions** (new event kinds, new optional keys): consumers
  written for an earlier minor version still work.
- **v1 → v2** (breaking change): only if we get the schema wrong. The
  intent is never to bump it.

## Deliberately NOT in the schema

- **Full tensor values** — too large; stats only. For full capture, save
  a checkpoint to `weights/`.
- **Logits** — the decode loop emits the top-1 token + its logprob; full
  logit vectors go to a run's artifacts if the experiment chose.
- **Activations** — stats via taps; full values via explicit dumps.
- **Loss landscape** — LMC and similar need two endpoints + an
  interpolation runner; done by analysis tools on saved checkpoints, not
  in-stream.

## L5: the introspection surface

`events.jsonl` is the L5 introspection surface above the five-layer algos
stack (L1 primitives → L4 recipes; see
[architecture.md](architecture.md)). It is the stable boundary the Tao
sibling consumes: raw events in, derived analysis (drift correlations,
linear CKA, Pareto crossover, HTML reports) out. toy never computes those
aggregates itself.

## Minimal complete run

```jsonl
{"kind":"run_start","schema":"toy/v1","t":0.0,"run_id":"llama-20260531-001","phase":"train","git":{"sha":"4bf48d3"},"backend":{"kind":"cpu"},"model":{"arch":"llama","name":"from-scratch-tinystories"},"config":{"steps":3,"lr":0.001}}
{"kind":"step","phase":"train","t":0.05,"step":1,"loss":9.23,"lr":0.001,"tokens":32,"wall_us":50000}
{"kind":"step","phase":"train","t":0.10,"step":2,"loss":9.21,"lr":0.001,"tokens":32,"wall_us":48000}
{"kind":"step","phase":"train","t":0.15,"step":3,"loss":9.15,"lr":0.001,"tokens":32,"wall_us":47500}
{"kind":"run_end","t":0.16,"ended_at":"2026-05-31T09:30:00Z","reason":"completed","final_step":3,"final_loss":9.15,"exit_code":0}
```

Everything beyond `run_start` / `step` / `run_end` is opt-in based on
what the run enables (taps, gradient capture, op timing, drift, samples).
</content>
</invoke>
