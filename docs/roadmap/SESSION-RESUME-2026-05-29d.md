# Session resume — 2026-05-29 (d): L1+L2+L3 DONE — five-layer skeleton complete

> **Read this first.** Supersedes SESSION-RESUME-2026-05-29c.md (now
> deleted). The L1 primitives, L2 TransformerBlock, AND L3 LlamaArch
> are all landed and gated bit-identical on `main`. The structural
> refactor of the forward path is complete; what remains is the
> realize-path bulk (P2.6/P2.7) and L4 recipes.

## Where we are

| Phase | Status | Commit |
| --- | --- | --- |
| P0 — Design lock + Tao coord | done | bc1b40e |
| P1 — Card derivation refactor | done | 07aa547 |
| **P2 — Five-layer refactor** | **L1+L2+L3 done; realize bulk (P2.6/7) + L4 next** | f381031 |
| P3 — Core + CLI MVP | pending | — |
| P4 — CLI complete | pending | — |
| P5 — Generators | pending | — |
| P6 — Prism lowerer | pending (optional) | — |

## What landed (all gated bit-identical, one commit each)

| Layer | Commit | File | Class/Module |
| --- | --- | --- | --- |
| L1 RoPE | 347d452 | `primitives/rope.rb` | `Toy::LLM::Primitives::RoPE` |
| L1 SwiGLU | 8efb16e | `primitives/swiglu.rb` | `…::SwiGLU` |
| L1 RMSNorm | cdb021d | `primitives/rms_norm.rb` | `…::RMSNorm` |
| L1 GQA | d3d124e | `primitives/gqa.rb` | `…::GQA` |
| L2 Block | f29fc03 | `blocks/transformer_block.rb` | `…::Blocks::TransformerBlock` |
| L3 Arch | f381031 | `archs/llama_arch.rb` | `…::Archs::LlamaArch` |

**L3 (P2.5):** `build_forward_in_current_ctx` lifted verbatim into
`LlamaArch#build_forward` (op order + 4 `set_output` taps unchanged).
The arch OWNS token_embed / final_norm_gamma / output / w_proj + the
blocks array; it returns a `LlamaArchForwardOut` (hand-written class,
NOT `Struct.new` — landmine #16). The cache keeps the realize
allocation bulk and exposes the arch's handles via ~delegators
(reader+writer), retargeting ~50 `@t_seq_*` ivar sites; the
orchestration-written ivars (x_embed/x_final/logits) are re-assigned
onto cache ivars from the return value, so taps / `build_training_step`
CE / `examples/06` consumers are untouched. **scope_held: the
~1750-line realize allocation bulk is unchanged** (its restructuring
is deliberately P2.6/P2.7).

**Independently verified at HEAD:** A/B fixture at `ef68017` vs
`f381031` → byte-identical losses; `make verify-mirrors` clean; tree
clean on `main`.

## Caveats (unchanged from L2)

- **Nothing pushed.** All commits local on `main`.
- **CUDA is mirror-source-verified, not run for L3.** The checked-in
  `tinynn/libtinynn_ggml_cuda.a` is the 4 KB AB-smoke stub (only 7
  link-shim symbols; no `tnn_session_new`/`tnn_matmul`/`tnn_get_rows`).
  A real CUDA forward needs `make setup-ggml-cuda` (heavy one-time
  clone + CUDA build). `verify-mirrors` guarantees the `_cuda`/`_metal`
  arch + monolith mirror SOURCES are consistent; `ruby -c` clean.
  **If you want a true CUDA gate for the remaining P2 work, do the
  `setup-ggml-cuda` build once and add a CUDA `smoke_projection_lens`.**
- **B>1 batch path** still not separately smoke-tested (no B>1 CPU
  fixture); threaded through unchanged.

## Spinel landmine (now upstream)

`Struct.new` accessor names act as global type-merge keys → unrelated
code mis-typed. Filed **matz/spinel#1043** with minimal repro +
generated-C evidence. Regression probe:
`tinynn/probe_struct_accessor_collision.rb`. Memory landmine #16.
**Rule for all remaining Spinel value objects: hand-written class,
positional ctor, no default args, uniquely-prefixed members.**

## The next concrete move

Two independent fronts, pick per priority:

1. **P2.6/P2.7 — realize-path bulk.** The four `realize_for_*` methods
   (~1750 lines) are now the only large undecomposed surface. With the
   block as the allocation unit and the arch owning whole-graph
   handles, the realize logic can move onto the arch/block (weight
   loading per block, whole-graph ctx alloc on the arch). This is the
   biggest remaining lift; gate the same way (bit-identical). Consider
   doing the `setup-ggml-cuda` CUDA gate FIRST so realize changes are
   GPU-checked too.
2. **L4 recipes** (`lib/toy/llm/recipes/`) — training loop / optimizer /
   schedule extraction. Lower risk; the design doc §6 + recipes/README
   sketch the contract.

Same gate every step: bit-identical losses + clean `verify-mirrors` +
no Spinel warnings; one unit, one commit; `git reset` + re-plan on
failure.

## Reusing the workflows

Three gated-extraction harnesses now exist, increasing in scope:
- `prep/wf_finish_primitives.js` — multi-item serial loop (L1).
- `prep/wf_finish_l2_block.js` — single-unit recon→plan→gated (L2).
- `prep/wf_finish_l3_arch.js` — single-unit + explicit SCOPE-HOLD
  framing (L3). **Closest template for P2.6/P2.7**: the realize-bulk
  work has the same "lift X, don't restructure Y, keep behavior
  bit-identical, hold scope" shape — adapt the L3 script's facets to
  the realize paths and weight-loading.

## Reference paths

- Monolith (now realize-bulk + delegators): `lib/llama_seq_forward_ffi.rb`
- Primitives: `lib/toy/llm/primitives/{rope,swiglu,rms_norm,gqa}.rb`
- Block: `lib/toy/llm/blocks/transformer_block.rb`
- Arch: `lib/toy/llm/archs/llama_arch.rb`
- Recipes (next, L4): `lib/toy/llm/recipes/` (README only)
- Mirror generator: `prep/gen_cuda_mirror.rb` (regex accepts
  primitives/ + blocks/ + archs/)
- Gate fixture: `SEED=0 STEPS=5 ./examples/smoke_projection_lens`
- Workflows: `prep/wf_finish_{primitives,l2_block,l3_arch}.js`
- Landmines: `memory/feedback_spinel_type_inference_landmines.md`
  (#16 = Struct.new / matz/spinel#1043),
  `memory/project_step_bind_landmine_2026_05_28.md`
