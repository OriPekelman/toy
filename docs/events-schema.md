# Toy events schema v1

Every Toy training or inference run emits a structured event stream as
JSON Lines (one JSON object per line) to a single file:

```
.runs/<run_id>/events.jsonl
```

This file is the contract between Toy (the producer) and any consumer
— live TUIs, post-run analysis, adapters to W&B / MLflow / tensorboard,
markdown report generators, Tao. The schema is versioned; Toy emits
`{"schema":"toy/v1"}` in the first event of every file.

The file is **append-only, line-delimited, monotonic in time**. Crashes
leave a truncated last line; consumers should skip lines that fail to
parse rather than aborting. Files older than v1 (if any) won't have the
`schema` field — treat them as v0 (best-effort).

## Design principles

1. **Cheap when off, cheap when on.** Events are emitted via the
   existing trace primitive (`tnn_trace_*`). The off-path is a single
   load + compare. The on-path is a buffered `fprintf`. No JSON
   serialization in the hot loop unless the trace is active.

2. **Schema is open.** Unknown keys in an event MUST be ignored by
   consumers. New event kinds and new keys can be added in v1
   minor revisions without breaking existing consumers. Removing a
   key requires a schema bump.

3. **One run = one file.** No multi-file fragmentation. A run's
   `events.jsonl` is everything you need to reconstruct or compare
   it. Weights are separate (`.runs/<run_id>/weights/`).

4. **Aggregation lives outside.** Toy emits raw events. Per-step
   summaries, derived statistics (drift, CKA, Pareto crossover) are
   computed by downstream consumers from the raw stream. This keeps
   Toy's hot path simple and gives consumers full control over
   what's interesting.

5. **Human-readable.** Pretty-print a JSONL with `jq` and it should
   be legible. Field names are full words, not abbreviations. Units
   are explicit (`_us` for microseconds, `_bytes` for sizes, `_f32`
   for tensor element dtype where ambiguous).

## File layout convention

```
.runs/
  <run_id>/
    events.jsonl        # the structured event stream (this doc)
    run.json            # static metadata (also in events as run_start)
    weights/            # checkpoint snapshots, *.gguf (optional)
    traces/             # P6 per-op Chrome traces (optional)
    artifacts/          # whatever the experiment chose to save
    report.md           # human-written or AI-generated narrative
```

`run_id` is conventionally ISO-8601 + a short hash:
`2026-05-24T09-30-00-4bf48d3-001`. Sortable, unique, traceable to a
git commit. Generation is a Toy responsibility but the convention is
documented here so external tools can match.

## Event kinds

Every event has at minimum:

```json
{ "kind": "<event_kind>", "t": <seconds_since_run_start, float> }
```

Optional but conventional:
- `step`: integer training/decode step the event belongs to. Omit for
  events that aren't step-scoped (run_start, run_end).
- `phase`: `"train"` | `"eval"` | `"decode"`. Distinguishes which loop
  the event belongs to (a single run may have multiple phases).

The currently-defined event kinds, alphabetical:

### `run_start` (exactly one per file, must be first)

Static metadata about the run. Equivalent to a copy of `run.json`
inlined into the event stream so a consumer reading only the stream
has everything.

```json
{
  "kind": "run_start",
  "schema": "toy/v1",
  "t": 0.0,
  "run_id": "2026-05-24T09-30-00-4bf48d3-001",
  "started_at": "2026-05-24T09:30:00Z",
  "git": { "sha": "4bf48d3", "branch": "main", "dirty": false },
  "host": { "name": "gx10", "os": "linux", "arch": "aarch64" },
  "backend": { "kind": "cuda", "device": "NVIDIA GB10", "compute": "12.1" },
  "model": {
    "arch": "llama",
    "name": "smollm2-135m",
    "vocab": 49152,
    "d_model": 576,
    "n_layers": 30,
    "n_heads": 9,
    "n_kv": 3,
    "d_head": 64,
    "d_ff": 1536,
    "rope_base": 100000.0,
    "weight_type": "f32",
    "is_moe": false,
    "n_experts": 0,
    "n_experts_used": 0,
    "swa_window": 0
  },
  "config": { "<arbitrary key/values from the experiment script>": "..." },
  "compare_to": "2026-05-23T17-15-00-5b86331-001"
}
```

`compare_to` is optional — if set, consumers should expect baseline-
comparison events later in the stream (see `compare_*` kinds).

### `step`

A single training step OR decode step (`phase` field disambiguates).
Emitted once per step, with the canonical metrics.

```json
{
  "kind": "step",
  "phase": "train",
  "t": 0.418,
  "step": 1,
  "loss": 9.2312,
  "ppl": 10254.7,
  "lr": 0.001,
  "grad_norm": 2.341,
  "tokens": 32,
  "compute_us": 70510,
  "wall_us": 71200
}
```

- `loss` is the primary objective (whatever you set via `tnn_set_loss`).
- `ppl` is `exp(loss)` for LM training. Omit if not LM.
- `grad_norm` is the global L2 of all PARAM-flagged tensor grads,
  computed once per step. May be omitted if expensive.
- `tokens` is the number of new tokens this step processed (for
  throughput computation downstream).
- `compute_us` is the `tnn_compute*` call duration. `wall_us` is the
  full step including uploads/downloads.

For `phase: "decode"`:

```json
{
  "kind": "step",
  "phase": "decode",
  "t": 0.084,
  "step": 5,
  "token_id": 12345,
  "logprob": -2.314,
  "wall_us": 84210
}
```

### `tap`

A named tap point downloaded a tensor and computed summary stats. One
event per (step, region, layer). The available stats are:

| Stat name        | Meaning                                              |
|------------------|------------------------------------------------------|
| `l2`             | √(sum of squares) over the full tensor               |
| `min`, `max`     | element-wise extrema                                 |
| `mean`           | arithmetic mean                                      |
| `abs_mean`       | mean of absolute values                              |
| `sparsity`       | fraction of elements with `|x| < eps` (eps=1e-6)     |
| `nan_count`      | number of NaNs                                       |
| `per_head_l2`    | array, one L2 per head (for multi-head tensors)      |
| `histogram`      | array of 32 bin counts spanning min..max             |
| `gram`           | T×T Gram matrix `G = Aᵀ·A` for activation `A` of shape `[d, T]` — feeds linear CKA (GH#15) |

```json
{
  "kind": "tap",
  "phase": "train",
  "t": 0.420,
  "step": 1,
  "region": "attn_q_post_rope",
  "layer": 0,
  "head": null,
  "shape": [64, 1],
  "dtype": "f32",
  "l2": 0.0042,
  "abs_mean": 0.00052,
  "nan_count": 0,
  "per_head_l2": [0.0041, 0.0044, 0.0040, 0.0042, 0.0043, 0.0041, 0.0042, 0.0042, 0.0043]
}
```

#### `tap` with `gram` (CKA, GH#15)

When emitted via `ToyTap.emit_cka`, the event additionally carries a
`gram` field: a T×T symmetric PSD matrix `G = Aᵀ·A` over an activation
`A` of shape `[d, T]`. Consumers compute linear CKA from gram pairs
(`Analyze.linear_cka` in the Tao sibling). Suggested region names:
`attn_norm`, `ffn_out`, `resid_post_block` — all stable across runs.

```json
{
  "kind": "tap",
  "phase": "train",
  "t": 0.420,
  "step": 1,
  "region": "resid_post_block",
  "layer": 0,
  "head": null,
  "shape": [64, 32],
  "dtype": "f32",
  "l2": 6318.97,
  "abs_mean": 4.21,
  "nan_count": 0,
  "gram": [[…32 floats…], [...], ..., [...]]
}
```

Tap points are user-named via `kv.tap(region, tensor, stats)`. The
`region` name is the only convention; it's expected to be stable
across runs so comparisons work. Suggested naming: `<sublayer>_<phase>`
e.g. `attn_q_post_rope`, `ffn_gate_pre_silu`, `embed_post_lookup`.

### `grad`

Per-parameter gradient statistics, captured on the backward pass.
Same stats vocabulary as `tap`. One event per (step, param_name).

```json
{
  "kind": "grad",
  "phase": "train",
  "t": 0.420,
  "step": 1,
  "param": "blk.0.attn_q.weight",
  "head": 0,
  "shape": [64, 576],
  "l2": 0.0312,
  "abs_mean": 0.00041,
  "nan_count": 0
}
```

`param` matches the GGUF tensor name where possible (so cross-run
comparison is unambiguous). LoRA adapters use a documented suffix
convention: `blk.0.attn_q.weight#lora_a` / `#lora_b`.

### `op_timing`

Aggregate of P6 per-op timings, one event per step. The per-op
breakdown is bucketed by ggml op kind.

```json
{
  "kind": "op_timing",
  "t": 0.420,
  "step": 1,
  "compute_us": 70510,
  "buckets": {
    "MUL_MAT":   12340,
    "OUT_PROD":  18820,
    "ADD":        6910,
    "ROPE":       2150,
    "SOFT_MAX":   1240,
    "OPT_STEP_ADAMW": 5430,
    "other":     23620
  }
}
```

For the full per-op trace, see `traces/step_<N>.json` (Chrome Trace
format, opt-in via `TRACE_OPS=1`). The `op_timing` event is the cheap
always-on summary.

### `drift`

Distance of a parameter from its init value. Emitted on a schedule
(e.g. every 100 steps), not every step.

```json
{
  "kind": "drift",
  "t": 41.8,
  "step": 100,
  "param": "blk.0.attn_q.weight",
  "head": 0,
  "cos_to_init": 0.9912,
  "l2_to_init": 0.0421
}
```

`cos_to_init` is cosine similarity to step-0 weights (1.0 = no drift,
0.0 = orthogonal). `l2_to_init` is the L2 of the delta. Together they
give "how far has this weight moved, and in what direction."

### `eval`

A held-out evaluation result. Phase changes to `"eval"`. May appear
multiple times in a run (mid-training evals + final).

```json
{
  "kind": "eval",
  "phase": "eval",
  "t": 123.4,
  "step": 1000,
  "name": "validation",
  "loss": 3.421,
  "ppl": 30.6,
  "samples": 512,
  "extra": {
    "hellaswag_acc": 0.31,
    "arc_easy_acc": 0.42,
    "lambada_acc": 0.18
  }
}
```

`extra` is an open dictionary — any benchmark suite's outputs (e.g.
lm-eval-harness JSON) can be nested here.

### `compare`

Live comparison against `run_start.compare_to` baseline. Emitted by
the consumer or by Toy itself if running in compare mode. One per
step where baseline has the matching step.

```json
{
  "kind": "compare",
  "t": 0.420,
  "step": 1,
  "baseline_run_id": "2026-05-23T17-15-00-5b86331-001",
  "loss_delta": -0.013,
  "ppl_ratio": 0.987,
  "verdict": "ahead"
}
```

`verdict` is one of `"ahead"` (current is better), `"behind"`, `"par"`
(within 1%), `"abort"` (current is so far behind we should stop —
emitted as a signal, not a directive). Auto-abort policies live in the
consumer, not in Toy.

### `region_enter` / `region_exit`

Marks a logical region in the event stream — typically the start
and end of an experimental variant being tested. Consumers can
partition events by region for "compare new bit vs baseline bit."

```json
{ "kind": "region_enter", "t": 0.0,   "step": 0,   "region": "warmup" }
{ "kind": "region_exit",  "t": 100.0, "step": 100, "region": "warmup" }
{ "kind": "region_enter", "t": 100.0, "step": 100, "region": "experimental_lr_schedule" }
```

Regions can nest. Consumers handle this with a stack.

### `note`

Free-form human or AI annotation. Useful for inline commentary
("disabled qk_norm here", "switched to Q8K at step 500").

```json
{ "kind": "note", "t": 100.4, "step": 100, "text": "switched LR schedule from cosine to linear" }
```

### `run_end` (exactly one per file, must be last)

```json
{
  "kind": "run_end",
  "t": 1234.5,
  "ended_at": "2026-05-24T09:50:34Z",
  "reason": "completed",
  "final_step": 1000,
  "final_loss": 2.31,
  "quality_gate": {
    "passed": true,
    "metric": "loss_ratio",
    "value": 0.68,
    "threshold": 0.9
  },
  "exit_code": 0
}
```

`reason` is reserved for **execution status** — one of `"completed"`,
`"interrupted"`, `"aborted"`, `"errored"`. A run that executed to
completion without an exception is always `"completed"` regardless of
quality. If interrupted/aborted/errored, an `error` field with the
message is conventional.

`quality_gate` (optional) carries the **research verdict** —
distinct from `reason`. A run that completed cleanly but failed a
quality bar (e.g. "loss didn't decrease enough", "perplexity above
threshold") gets `reason: "completed"` + `quality_gate.passed: false`.
This separation lets consumers distinguish "did this run crash?"
(`reason`) from "is this run useful for the experiment?"
(`quality_gate.passed`). Fields:

- `passed` (bool, required if `quality_gate` is emitted)
- `metric` (string) — the name of the gate metric, e.g. `"loss_ratio"`,
  `"perplexity"`, `"top1"`
- `value` (number) — the metric's measured value
- `threshold` (number) — the gate boundary

`exit_code` mirrors the OS exit code (0 = success, 1 = quality-gate
fail, 2+ = command/usage errors). Independent of `reason` so CI/make
contracts can stay sharp.

### `eval` with `phase: "serve"` — inference-time telemetry

Inference servers (e.g. `Tep::Llm::OpenAI::Server`, the toy
serving path) emit `eval` events with `phase: "serve"` and
`name: "request"` for per-completion telemetry. The shape:

```json
{
  "kind": "eval",
  "phase": "serve",
  "t": 87.42,
  "name": "request",
  "extra": {
    "model": "smollm2-135m",
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "latency_us": 87000,
    "sampling": { "temperature": 0.7, "max_tokens": 256 },
    "request_id": "chatcmpl-abc",
    "principal_id": "user:42"
  }
}
```

The `phase: "serve"` value distinguishes from `phase: "eval"` (which is
held-out evaluation during training). The `extra` open-bag carries
caller-specific fields without bloating the schema; consumers route on
`extra.request_id` for request-keyed aggregation.

A serving run additionally emits `run_start` at server-boot
(with `backend.kind`, `host`, `git`, `model`) and `run_end` at
shutdown — same envelope as a training run, so the consumer
ingest path is uniform across training and serving telemetry.

## Producer guarantees

When Toy emits this stream:

1. `run_start` is always written before any other event, and is
   flushed to disk before any compute begins. (So a crashed run still
   leaves identifying metadata.)
2. The file is `O_APPEND` opened. Multiple writers to the same file
   are NOT supported — one run, one writer.
3. Each event is a complete line ending in `\n`. Newlines inside
   string values are escaped per JSON rules.
4. Time-since-start (`t`) is monotonic; if a clock skews, Toy clamps.
5. `step` numbers are dense within a phase (no gaps) unless explicitly
   sub-sampled (a `subsample: N` field on `run_start.config`).

## Consumer guarantees

When you read this stream:

1. Skip lines that fail JSON parse — likely a torn final line from a
   crashed writer. Log but don't abort.
2. Treat unknown `kind` values as informational; don't error.
3. Treat unknown keys within a known event as informational; pass
   through if forwarding to another sink.
4. Don't assume the file is closed — tail-follow semantics are
   expected to work for live consumers.

## Versioning

The `"schema": "toy/v1"` field on `run_start` is the version.

- **v1 minor revisions** (adding event kinds, adding optional keys):
  consumers written for an earlier minor version will still work.
- **v1 → v2** (breaking change): only if we get the schema wrong.
  Toy will emit a v2 stream and consumers will need adapters. The
  intent is to never bump this.

## What's deliberately NOT in the schema

- **Full tensor values.** Too large. Stats only. If you want full
  tensor capture, save a checkpoint (`weights/step_<N>.gguf`).
- **Logits.** Decode loop emits the top-1 token + its logprob; full
  logit vectors go to `artifacts/` if the experiment chose.
- **Activations.** Same reasoning — stats via taps, full values via
  explicit dumps to `artifacts/`.
- **Loss landscape.** LMC and similar require two endpoints + an
  interpolation runner; out of stream scope, done by analysis tools
  on saved checkpoints.

## Example: minimal complete run

```jsonl
{"kind":"run_start","schema":"toy/v1","t":0.0,"run_id":"2026-05-24T09-30-00-4bf48d3-001","git":{"sha":"4bf48d3"},"backend":{"kind":"cuda"},"model":{"arch":"llama","name":"smollm2-135m"},"config":{"steps":3,"lr":0.001}}
{"kind":"step","phase":"train","t":0.05,"step":1,"loss":9.23,"ppl":10254,"lr":0.001,"compute_us":50000}
{"kind":"step","phase":"train","t":0.10,"step":2,"loss":9.21,"ppl":9923,"lr":0.001,"compute_us":48000}
{"kind":"step","phase":"train","t":0.15,"step":3,"loss":9.15,"ppl":9421,"lr":0.001,"compute_us":47500}
{"kind":"run_end","t":0.16,"ended_at":"2026-05-24T09:30:00Z","reason":"completed","final_step":3,"final_loss":9.15,"exit_code":0}
```

That's the minimum a Toy producer must emit. Everything else is opt-in
based on what the user enables (`TRACE`, `TRACE_OPS`, tap points,
gradient capture, etc.).
