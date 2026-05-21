# Phase 0.6 scope — graph inlining + legacy deletion

**Status:** **DEFERRED 2026-05-21** — not blocked, not scheduled.
                  See "Re-trigger conditions" below.
**Date:** 2026-05-20 (initial scoping); deferral note added 2026-05-21

## What's been working

Since Phase 0.5, **delegating wrappers** (`ToyLM`, `ToyLMCuda`) own the
caller-facing API; the actual graph-builder lives in
`lib/toy_smollm2_ffi_kv.rb` (CPU) and `lib/toy_smollm2_ffi_kv_cuda.rb`
(CUDA). Both files have the "RESOLVED 2026-05-20" header confirming
they're maintained as mirrored siblings.

That pattern WORKS:
- All 12 supported models route through `ToyLM.from_gguf(...) → load → generate`
- F32 + Q8 + native-mmap + legacy-Mat-mediated paths all dispatch
  correctly
- Bench numbers match canonical demos bit-for-bit

So nothing in Phase 0.6 is **forced** by a broken state. It's a
consolidation opportunity, not a blocker.

## What Phase 0.6 actually buys

1. **One graph builder, not two**. Currently `build_decode_step` lives in
   `SmolLM2KVFFICache#build_decode_step` AND
   `SmolLM2KVFFICacheCuda#build_decode_step`. Bug fixes have to be made
   twice (the V-matmul flip was the canonical example — CPU got Phase
   3, CUDA didn't, until 2026-05-20). Inlining unifies them.

2. **`@arch` becomes the source of truth**. Currently the cache classes
   take individual `cfg` fields and flags through `realize_for`. After
   0.6, they'd read everything from `@arch`. New architectures (Qwen3
   dense, Qwen3 MoE) become "write a new `Arch.qwen3moe` factory" + zero
   graph-builder changes if the conditionals are in place.

3. **`lib/toy_smollm2_*.rb` deletes**. ~1500 lines down to ~500 in a
   generic `lib/transformer_lm_graph.rb`. Cleaner repo.

## Why it's a multi-day job

The cache classes are NOT just data holders — they own:

- Two ctxs: `ctx_w` (persistent weights), `ctx` (compute graph)
- ~30 tensor allocation calls per layer × 28 layers = ~1000 tensor
  handles
- `realize_for_mmap` (Phase 2 BYO-pointer) vs `realize_for` (Mat-mediated)
  paths
- Per-layer FFN cache structures (`SmolLM2KVBlockFFI`)
- Trace-tap infrastructure for debugging
- KV cache slots for K and V
- Type-dispatch logic (F32 vs Q8 persistent allocator paths)

Moving all of this into `ToyLM` while keeping every test green is
~1 week. Risks: CUDA divergences re-emerge (we just resolved one);
the CPU/CUDA mirror pattern was specifically designed to make
Spinel happy (per `lib/toy_smollm2_ffi_kv_cuda.rb`'s header, classes
must be named differently to avoid Spinel class-collapse).

## Recommended scope when we get to it

1. Move tensor allocation logic to a `TransformerLMGraph` module
   parameterized by the FFI module (`TinyNN` vs `TinyNNCuda`).
2. Keep `SmolLM2KVFFICache(Cuda)` as **shim** classes that
   `include` the module. Spinel sees them as distinct classes;
   the code is shared.
3. Migrate one method at a time — `realize_for` first, then
   `build_decode_step`, then helpers.
4. Bit-identical bench parity required after each step.
5. **Only then** delete the original files.

## What replaces it for now

- The `aesthetics` pass (commit `e7332b8`) already documents the
  mirror relationship in headers, with the "any change to CPU file
  must be mirrored" rule. Drift detection is partial but reasonable.
- New architectures (Qwen3 dense, MoE) won't NEED 0.6 to land first
  — they can be added by extending the existing cache classes with
  the new arch's quirks under a flag. Less clean, still works.

Phase 0.6 stays queued as a "do when the maintenance cost crosses the
refactor cost" item. Currently it hasn't.

## Re-trigger conditions

Pick this up when any **one** of the following becomes true:

1. **A new architecture lands and needs > ~50 LOC of CPU/CUDA mirror
   duplication.** Qwen3 dense would likely fit existing classes. Qwen3
   MoE almost certainly won't — the expert routing inside FFN is the
   trip wire. When the MoE branch starts mutating both
   `SmolLM2KVFFICache` and `SmolLM2KVFFICacheCuda` in parallel, do 0.6
   first.
2. **A CPU/CUDA divergence ships to production.** We caught one
   (Phase 3 V-matmul flip) and fixed both sides in the same week. A
   second occurrence — especially one that bites a user — flips the
   maintenance-cost ledger.
3. **The Spinel ivar-collision (#626 issue #1) is upstream-fixed AND
   we want to delete `lib/gguf_kv.rb`** as the decoupling shim. Phase
   0.6 cleans up the workaround surface.

Until then: keep the CPU/CUDA mirror discipline documented in commit
`e7332b8` (each file's header has the "mirror to/from the other side"
rule). Phase 0.7 acceptance gates (docs/design/phase-07-acceptance.md)
catch CPU-side regressions; the `*_cuda` smoke demos catch GPU-side.

## What still needs Phase 0.6 eventually

Even with the gates and mirror discipline, the ~1000-line CPU/CUDA
twin will keep generating low-grade tax: every bug fix is two reviews,
every new arch feature is two PRs, every benchmark needs two
invocations. Phase 0.6 is the right end-state. It's just not the right
**this-month** state.

Re-read this doc before opening the Phase 0.6 PR — the "Recommended
scope" section is still the plan.
