# Documentation tour

A toy you can read top-to-bottom that runs real models. The code is
the canonical source; docs explain decisions and history. Three
buckets:

| Where | What | Read when |
| --- | --- | --- |
| `docs/` (this directory, current files) | How things work **now** — architecture, loaders, memory model. | You want to understand the running system or extend it. |
| [`docs/roadmap/`](roadmap/) | Designs for things **not yet built**, or work explicitly deferred. | You want to know what's next, or pick something up. |
| [`docs/archive/`](archive/) | Point-in-time snapshots: bench results, root-cause investigations, completed-phase status, filed upstream issues. | You're tracing the history of a decision or a fix. |

Project-level READMEs sit next to their code: [`README.md`](../README.md),
[`examples/README.md`](../examples/README.md),
[`demos/README.md`](../demos/README.md),
[`tinynn/README.md`](../tinynn/README.md),
[`tep_demo/README.md`](../tep_demo/README.md).

## Current reference

- [`architecture.md`](architecture.md) — the generic `TransformerLM`
  plus per-model `Arch` struct. Recipes for adding a new arch.
- [`loader-api.md`](loader-api.md) — Mat-mediated loader vs the
  direct GGUF→FFI loader. They're peers; pick by use case.
- [`memory-design.md`](memory-design.md) — why a 7B-Q8 model fits in
  ~7 GB of RSS (Phase 2 BYO-pointer, Phase 3 Q8-stays-Q8).
- [`cuda-byo-pointer-design.md`](cuda-byo-pointer-design.md) — CUDA
  side of the BYO-pointer mmap (UVA, the vendored ggml-cuda patch).

## Roadmap (future work + deferred)

- [`roadmap/finetuning.md`](roadmap/finetuning.md) — F1/F2/F3/F4 spec.
  F1 (LoRA + AdamW) is shipped (CPU + CUDA via the seq-forward graph).
  F3 (full fine-tune on CUDA) and F4 (QLoRA: Q8 base + F32 LoRA) are
  the next two.
- [`roadmap/phase-06-in-flight.md`](roadmap/phase-06-in-flight.md) —
  CPU/CUDA mirror deduplication. Un-deferred 2026-05-21; the seq-mode
  pair is now generated from the CPU source via
  `prep/gen_cuda_mirror.rb`; older KV-cache pair migration pending.
- [`roadmap/scout-small-models.md`](roadmap/scout-small-models.md) —
  next-target survey across modern small models.
- [`roadmap/lowerer-design.md`](roadmap/lowerer-design.md) — a
  Roundhouse-style preprocessor for toy. Not built; recorded because
  there's a concrete external proposal.

## Archive (where to find old context)

Most archived docs are dated (`*-2026-05-XX.md`) so file order tracks
the timeline. Highlights:

- [`archive/bench-gx10-2026-05-22.md`](archive/bench-gx10-2026-05-22.md) —
  perf snapshot before the v0.1-pre-alpha tag: inference tok/s across
  model sizes, sequence-mode training step time, memory footprint.

- [`archive/arc-close-2026-05-21.md`](archive/arc-close-2026-05-21.md) —
  closing memo for the F1.1 / Phase 0.6 / Phase 0.7 arc. Good starting
  point if you're picking up where 2026-05-21 left off.
- [`archive/roadmap-2026-05-21.md`](archive/roadmap-2026-05-21.md) +
  [`archive/roadmap-addendum-2026-05-21.md`](archive/roadmap-addendum-2026-05-21.md) —
  strategic snapshot of "where we want toy to live" (multi-model
  inference daemon positioning).
- [`archive/m3-seq-forward-2026-05-21.md`](archive/m3-seq-forward-2026-05-21.md) —
  M3 design doc (sequence-mode forward); the implementation that
  landed matches the spec.
- [`archive/bench-*.md`](archive/) — per-date perf numbers.
- [`archive/phase-f0-status.md`](archive/phase-f0-status.md),
  [`archive/phase-f1-status.md`](archive/phase-f1-status.md),
  [`archive/phase-f1-2-step6-status.md`](archive/phase-f1-2-step6-status.md),
  [`archive/phase-07-acceptance.md`](archive/phase-07-acceptance.md) —
  completed-phase status reports.
- [`archive/task70-*.md`](archive/) — the ggml-cpu sched-aliasing
  diagnosis; workaround lives in `tnn_pin_all_graph_b_nodes`.
- [`archive/qwen25-known-issue.md`](archive/qwen25-known-issue.md),
  [`archive/tinyllama-known-issue.md`](archive/tinyllama-known-issue.md) —
  fixed. Kept as canonical scratch-overflow / silent-truncation stories.
- [`archive/handoff.md`](archive/handoff.md) — gx10 ↔ Mac handoff
  notes from when we set the GB10 box up.
- [`archive/hf-gpt2-feature-branch.md`](archive/hf-gpt2-feature-branch.md) —
  long-form story of getting GPT-2 to run identical-to-PyTorch.
- [`archive/upstream-contrib-draft.md`](archive/upstream-contrib-draft.md) —
  unsent doc on HF→GGUF byte-equivalence (intended for ggml-org).

## Archive — upstream issues we've filed

[`archive/upstream/`](archive/upstream/) has the local copies of
upstream-issue write-ups (the GitHub issues are the canonical source):

- `issues-vendor/01-04` — ggml + llama.cpp (byte-equivalence,
  CUDA buffer_from_ptr, cpy-into-strided, Spinel-tokenizer blockers).
- `issues-spinel/01-05` — Spinel codegen/inference bugs.
- `issues-tep/01` — Tep tokenizer-import warning.
- `ggml-cpu-sched-grad-aliasing.md` — the sched-aliasing reproduction.
- `tep-issue-13-not-fully-fixed.md`,
  `tep-run-arity-silent-miscompile.md` — Tep follow-ups.

(Two more recent Spinel issues — `matz/spinel#644` and `#645` — were
filed directly on GitHub and don't have local mirrors.)
