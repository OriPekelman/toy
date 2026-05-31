# Comments on Tep Battery 7 (OpenAI Server) from toy + Tao perspective

**Date:** 2026-05-25. **Source docs:**
`../tep/docs/OPENAI-SERVER-BATTERY.md` (early draft) and
`../tao/ISSUES-FOR-TOY.md` (next-requests section). Tep explicitly
solicits sibling-project input before freezing the backend interface;
this writes our reactions down so the dialogue is in markdown not
chat memory.

## TL;DR

Battery 7's shape is broadly compatible with toy and matches Tao's
serve-from-run-dir use case. **Two contract changes need to happen
before freezing:**

1. The backend interface assumes chat-template-aware
   `generate(model, messages, sampling, &on_token)`. Toy today only
   speaks token IDs (intentionally — chat templating is per-model and
   non-trivial). Either (a) the interface needs a token-level twin,
   (b) tep ships a default templater the backend can override, or
   (c) we accept that "first concrete backend" implies "toy adds chat
   templating first." Recommendation: (a) — narrow + composable.

2. Tep's `events.jsonl` schema sketch is **not** toy/v1-compatible.
   Tep should adopt toy/v1's envelope (`kind, t, ...`) and route
   serving telemetry through the existing `eval` event with an
   `extra` open-bag — that's exactly what `extra` is for. Concrete
   re-mapping in §"Events schema" below.

Everything else (streaming, capability gating, checkpoint discovery,
hot reload) is fine as drafted. Two deferred-but-related issues from
the Tao triage have implications for Battery 7 that should be folded
in (`toy#run-end-reason-semantics`, `toy#train-device-select`).

## On the open questions

### Q1: Backend interface fit

Tep proposes:

```ruby
def generate(model_name, messages, sampling, &on_token)
```

**Toy's current state.** `tep_demo/openai_api_smollm2.rb` (353 LOC,
the de facto first backend) speaks **token IDs only** — it serves
`POST /v1/completions` with `prompt: [int, int, ...]`, returns 501
on `/v1/chat/completions`. The reason is in
`docs/notes/spinel-type-inference-landmines.md` and the
`feedback_spinel_type_inference_landmines` memory: chat templating
is per-model (Llama 3 / Qwen 2 / SmolLM2 / Gemma 2 all differ), and
the tokenizer surface in toy is intentionally minimal under Spinel
codegen constraints. Adding `messages` → tokens at the toy layer is
~200-400 LOC of per-arch chat-template logic.

**The disconnect.** Tep wants chat-shape in; toy currently emits
token-shape only. Three resolutions:

- **(a) Narrow + composable.** Tep's Backend exposes TWO methods:
  ```ruby
  def generate_from_messages(model, messages, sampling, &on_token)
  def generate_from_tokens(model, token_ids, sampling, &on_token)
  ```
  Each backend implements whichever it natively supports. Tep's
  `/v1/chat/completions` calls `generate_from_messages`; for
  backends that don't support it, tep can either return 501 (current
  toy behaviour) or fall back to a generic templater if one is
  configured. `/v1/completions` calls `generate_from_tokens`.
  **Recommended.** Keeps each backend honest about its surface.

- **(b) Tep ships a default chat templater.** Tep includes a generic
  template library; backends can override per-model. Risk: tep
  starts carrying ML-shape concerns (per-model templates) which §"What's
  deliberately NOT in this battery" disclaims.

- **(c) Toy adds chat templating first.** Toy ships a `ChatTemplate`
  primitive per arch before being able to implement the Battery 7
  backend. Reasonable but couples Tep's timeline to toy's roadmap.

Recommendation: **(a)**. It's the least invasive for both projects
and makes the API honest about what each backend speaks.

### Q2: Events schema

Tep's draft (verbatim from the doc):

```json
{"ts": 1716615000.42, "kind": "eval", "model": "smollm2-135m",
 "prompt_tokens": 12, "completion_tokens": 8, "latency_ms": 87,
 "sampling": {"temperature": 0.7, "max_tokens": 256},
 "request_id": "chatcmpl-abc",
 "principal_id": "user:42"}
```

**Problems with this against toy/v1** (`docs/events-schema.md`):

- Field name: `ts` vs toy/v1's `t`. Just rename.
- Unit: `latency_ms` vs toy/v1's `_us` convention.
- Missing envelope fields: toy/v1 mandates `phase` on every event,
  and `kind` here would conflict with toy/v1's existing `eval`
  event (which is for held-out evals during training, not per-request
  serving telemetry).
- No `run_start`. Toy/v1 mandates a one-time `run_start` as the first
  event. For tep this maps naturally to server-boot (one-time, before
  any request).
- Top-level fields (`model`, `prompt_tokens`, etc.) bypass the
  designed open-bag pattern. Toy/v1 events use `extra: {...}` for
  caller-specific fields exactly so the schema stays stable.

**Proposed re-mapping.** Tep adopts toy/v1's envelope; serving
telemetry rides on the existing `eval` event with `phase: "serve"`
(new — small extension):

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
    "sampling": {"temperature": 0.7, "max_tokens": 256},
    "request_id": "chatcmpl-abc",
    "principal_id": "user:42"
  }
}
```

Plus a one-time `run_start` at server-boot with the same shape we
already emit from training (host, backend, git, model{}, config{}).
This makes a serving run *structurally indistinguishable* from a
training run in the stream — same envelope, same provenance, same
consumers. Tao already groks the toy/v1 envelope for E0; serving
telemetry would Just Work.

For per-token timings during streaming, the existing toy/v1 `step`
event with `phase: "decode"` already documents this shape — Tep can
use it directly:

```json
{ "kind": "step", "phase": "decode", "t": 0.084, "step": 5,
  "token_id": 12345, "logprob": -2.314, "wall_us": 84210 }
```

### Q3: Checkpoint discovery

**TAO_RUN_DIR + `latest` symlink: fits.** Tao already passes
TAO_RUN_DIR; the convention is established. One gap:

**Toy doesn't currently write checkpoints during training.**
`examples/06_train_from_scratch.rb` emits `events.jsonl` but never
saves weights to disk. The Tao serve-from-run-dir use case requires
training to ALSO write `$TAO_RUN_DIR/weights/<step>.gguf` (or
similar) — that's a separate issue (`toy#checkpoint-write`, not yet
filed). Until that lands, the Battery 7 backend reads from a fixed
GGUF path passed at boot (which is what `tep_demo/openai_api_smollm2`
already does).

This is fine to defer: Battery 7 ships, serves from a fixed GGUF, and
gains run-dir auto-discovery once toy can write checkpoints. The
backend's `list_models` API supports both modes cleanly.

### Q4: Streaming semantics

Per-token. Matches toy's KV decode primitive (one `decode_step` call
per output token). No issue.

The only nuance: toy's decode loop currently doesn't yield logprobs.
If tep wants to thread logprob/top-k metadata into the SSE stream,
that needs a small extension to toy's decode primitive (record top-k
softmax outputs alongside the chosen token). Cheap-when-off, but a
real addition. Suggestion: defer to a separate issue if/when a Tao
spec needs it.

### Q5: Capability set

`:infer` is fine as a single cap for MVP. Splitting on model size or
billing class is a real future need but not v1.

### Q6: Hot reload

Defer. Backends that need it can implement file-watching internally;
tep doesn't need a route for it until a concrete use case files.

## Cross-cutting: Tao's recent issues that affect Battery 7

Tao's `ISSUES-FOR-TOY.md` triaged four "next requests" on 2026-05-25.
Two have direct implications for Tep Battery 7:

### `toy#run-end-reason-semantics` (medium, correctness)

The lesson — *don't overload `reason:"errored"` for a quality gate*
— applies symmetrically to a serving run. Tep's events.jsonl emitter
should reserve `reason:"errored"` for actual server crashes /
uncaught exceptions, not "the run had zero traffic" or "throughput
fell below threshold". Quality verdicts on serving go in a separate
field, same as Tao asks for on training.

This is already the right default — tep would surface server errors
via uncaught exception → run_end on process exit — but worth
documenting in the Battery 7 spec so it doesn't drift.

### `toy#train-device-select` (medium)

A `DEVICE=cuda|cpu|metal` env that propagates into
`run_start.backend.kind`. Same shape applies to serving: a server
booting at `DEVICE=cuda TAO_RUN_DIR=/srv/runs/...` should reflect the
actual choice. Backend interface should accept device hints at
`Backend.new` (or via env), and tep's `run_start` reads from
`backend.kind` after init (matches what toy training does already —
we just shipped this in `62265b9`).

## Chunking comments

Tep proposes 7.1–7.5+ in order. Two suggestions:

- **Move events.jsonl emission (7.3) to 7.1's scope.** It's tiny —
  the toy/v1 producer pattern is a 30-LOC port from
  `examples/06_train_from_scratch.rb`'s emit blocks. Doing it at 7.1
  means the first end-to-end smoke is already observable, which is
  exactly the design principle Tao calls out for itself.

- **`/v1/embeddings` (7.4) blocks on a toy issue.** Toy doesn't
  currently expose embedding lookup as an isolated FFI call from the
  HTTP layer. It can, but it's a small wiring task (`toy#embed-api`,
  not filed). Worth flagging so 7.4's scope is "Tep + a separate
  toy-side enable" rather than "Tep alone".

## What Battery 7 unlocks for toy (worth noting)

- A clean removal path for `tep_demo/openai_api_*.rb` — those five
  near-duplicate files (one per model size) become *one* Backend
  implementation that reads `--model <path>` from boot and serves
  whichever GGUF it was pointed at. Maintenance-burden drop.
- A natural home for the inference-side `events.jsonl` (parallel to
  the training-side stream we just shipped). Tao's compare/report
  consumers can then ingest BOTH training and serving runs of the
  same model with one ingest pipeline.
- A path to Tao's "fine-tune-then-serve" loop without a per-project
  glue script.

## Concrete asks for the Tep author

Filing these as inline replies / GitHub-issue comments would be
fine; they're small and won't drift before chunk 7.1 starts.

1. **Backend interface**: adopt the two-method shape
   (`generate_from_messages` + `generate_from_tokens`) per Q1
   recommendation (a). Toy implements the latter immediately
   (from the existing demo); the former lands once toy ships chat
   templating, with no Tep churn.
2. **Events schema**: adopt toy/v1's envelope per Q2. Re-mapping
   is mechanical; the design becomes "tep emits toy/v1 with
   `phase: "serve"`".
3. **Chunk 7.1 scope**: include the events emitter at MVP, even
   if minimal (`run_start` at boot + `eval` per request). Cheap to
   add, big to leave out.

The rest of the doc — particularly the §"What's deliberately NOT
in this battery" disclaimers and the proxy/server split rationale
— is sharp. Recommend freezing once 1–3 are resolved.
