# Session resume — 2026-05-29 (h): L1–L4 done (3/4 recipes); realize-bulk at gate ceiling; P3 next

> Supersedes (g). The five-layer refactor is structurally COMPLETE:
> L1 primitives, L2 block, L3 arch, L4 recipes (FromScratch + LoRA +
> WarmStart; Curriculum deferred). Two remaining P2 items before P3:
> the (optional, large) realize-bulk fixture-cascade.

## Where we are

| Phase | Status | Commit |
| --- | --- | --- |
| P0 / P1 | done | bc1b40e / 07aa547 |
| **P2 — Five-layer refactor** | **L1+L2+L3 done; L4 3/4 recipes; realize-bulk at gate ceiling** | be79fa4 |
| P3–P6 | pending | — |

All on `main`, pushed to both remotes (0/0).

## Gates (all deterministic, self-consistent before/after)

| Gate | Command | Baseline |
| --- | --- | --- |
| CPU random_init | `SEED=0 STEPS=5 ./examples/smoke_projection_lens` (loss lines) | `4eaf019b…` |
| CUDA random_init | `…_cuda` | `bac45f88…` |
| GGUF F32 mmap | `SEED=0 ./examples/smoke_gguf_roundtrip` | `c89fd3eb…` |
| L4 FromScratch | `…/smoke_recipe_from_scratch` | == `4eaf019b…` |
| L4 LoRA | `STEPS=5 RANK=8 ./examples/smoke_recipe_lora` | == inlined `03_finetune_lora` (9.2362…→8.9046…) |
| L4 WarmStart | `SEED=0 STEPS=5 ./examples/smoke_recipe_warm_start` | == inlined `09` INIT=scratch (6.4409…→6.3845…) |

## L4 recipes — done to the gateable extent

| Recipe | Commit | Gate |
| --- | --- | --- |
| FromScratch | 74c5afd | bit-identical loss curve |
| LoRA | 2e2ee90 | bit-identical vs inlined 03 |
| WarmStart | be79fa4 | bit-identical vs inlined 09 (INIT=scratch) |
| Curriculum | — | **DEFERRED** |

All recipes follow the user-confirmed MINIMAL FLAT shape: hand-written
class, `realize!`/`step!` (+ per-stage `realize_*!` for WarmStart), NO
Struct.new (#16), NO Stage/Trainer/DataSpec/Eval taxonomy (AdamW is in
the ggml graph). Experiment config lives in the FIXTURE (lib-vs-example).
CPU-only orchestration (no mirror).

**Curriculum deferred — why (honest):** greenfield, no inlined reference.
The only honest shape is an array-of-FromScratch (a FRESH cache per stage
— `realize_*` calls `tnn_session_new` once per cache). But only stage-1
is gateable (== FromScratch); the multi-stage trajectory has no reference,
so shipping it would be speculative illustrative code — which violates
the bit-identical discipline. Revisit if/when a concrete curriculum need
gives it a real reference. (User chose "minimal, stay consistent".)

## Remaining P2 before P3: realize-bulk fixture-cascade (LARGE, OPTIONAL)

The realize bulk is decomposed as far as current gates allow (5 steps
landed). Each remaining deferred slice needs ITS OWN new gate first — a
cascade, each its own workflow + a realize-bulk re-run:

| Deferred realize slice | Gate to build first | Effort |
| --- | --- | --- |
| `realize_for_q8_copy` | Q8 round-trip gate (quantize-on-write; ToyGGUFWriter is F32-only) | high |
| GQA-divergent w_o | fixture with n_heads*d_head ≠ d_model | med |
| per-block mmap LOAD | CUDA mmap gate + GQA fixture | med-high |
| llama3 rope / qkv_bias / B>1 | one fixture each | med |
| ft GGUF-load half | (covered once mmap-load lands) | — |

This is internal decomposition of already-working, already-tri-gated
code. Marginal value is lower than the layers/recipes (which created the
public structure). DECISION PENDING with user: full cascade vs a bounded
subset (e.g. just GQA fixture) vs declare P2 "done to the gated extent"
and start P3.

## Spinel / refs

`Struct.new` accessor names = global type-merge keys → matz/spinel#1043
(probe: tinynn/probe_struct_accessor_collision.rb; memory #16). Value
objects: hand-written class, positional ctor, no defaults, prefixed
members. ggml#1506: mul_mat_id broken for K-quants (Q8_0 experts) —
memory/reference_ggml_mul_mat_id_kquant.md.

NOTE: the recipe-building workflows twice hit a benign harness error
(build agent commits, then fails to emit StructuredOutput after nudges).
The COMMIT is good each time; verify by hand (loss-curve A/B). Not a
correctness issue.

## Workflows (all under prep/, committed)

`wf_finish_{primitives,l2_block,l3_arch,realize_bulk,l4_recipes,l4_lora,l4_multistage}.js`,
`wf_build_gguf_roundtrip_gate.js`. realize_bulk is triple-gated +
re-runnable as new gates land.

## Reference paths

- Monolith: `lib/llama_seq_forward_ffi.rb`
- Layers: `lib/toy/llm/{primitives,blocks,archs,recipes}/`
- Recipe fixtures: `examples/smoke_recipe_{from_scratch,lora,warm_start}.rb`
- Gates/fuser: `examples/smoke_{projection_lens[,_cuda],gguf_roundtrip}`, `lib/toy_gguf_fuse.rb`
- Mirror gen: `prep/gen_cuda_mirror.rb`
- Landmines: `memory/feedback_spinel_type_inference_landmines.md` (#16)
