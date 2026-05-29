# Session resume — 2026-05-29 (g): L1–L4(FromScratch) done; realize-bulk at gate ceiling

> Supersedes (f). The five layers now all have a landed, gated unit:
> L1 primitives, L2 block, L3 arch, L4 FromScratch recipe. Realize-bulk
> is decomposed as far as current gates allow (fixture-cascade ceiling).

## Where we are

| Phase | Status | Commit |
| --- | --- | --- |
| P0 / P1 | done | bc1b40e / 07aa547 |
| **P2 — Five-layer refactor** | **L1+L2+L3 done; realize-bulk at gate ceiling; L4 FromScratch done; other recipes deferred** | 74c5afd |
| P3–P6 | pending | — |

All on `main`, pushed to both remotes (origin local-bare + github).

## Gates (3) — each deterministic, self-consistent before/after

| Gate | Command | Baseline | Exercises |
| --- | --- | --- | --- |
| CPU | `SEED=0 STEPS=5 ./examples/smoke_projection_lens` (loss lines) | `4eaf019b…` | random_init train, CPU |
| CUDA | `…_cuda` | `bac45f88…` | same, GB10 |
| GGUF | `SEED=0 ./examples/smoke_gguf_roundtrip` (stdout) | `c89fd3eb…` | realize_for_mmap F32 reload |
| (L4) | `SEED=0 STEPS=5 ./examples/smoke_recipe_from_scratch` (loss lines) | `4eaf019b…` | FromScratch recipe == inlined loop |

## What landed (full P2 arc)

- L1 RoPE/SwiGLU/RMSNorm/GQA (347d452…d3d124e); L2 TransformerBlock
  (f29fc03); L3 LlamaArch (f381031).
- CUDA gate (1f8c27e); GGUF round-trip gate + ToyGGUFFuser (4f551bf).
- Realize-bulk pass 1: 4 dual-gated steps (…29ab9f1).
- Realize-bulk pass 2: LlamaArch#load_globals_from_gguf_mmap! (50cb79d).
- **L4 FromScratch recipe (74c5afd):** `lib/toy/llm/recipes/from_scratch.rb`
  — minimal `realize!`/`step!` surface wrapping the random_init training
  loop; recipe-driven fixture `examples/smoke_recipe_from_scratch.rb`
  reproduces the loss curve BIT-IDENTICALLY (`4eaf019b…`). NO speculative
  Trainer/Stage/DataSpec/Eval (AdamW is in the ggml graph, so a Ruby
  Trainer would be empty). CPU-only (CUDA mirror deferred with the GPU
  recipe work). Reference smoke_projection_lens untouched.

All independently re-verified (HEAD reproduces all baselines; mirrors
clean). NOTE: the L4 build agent committed correctly but the workflow
then errored emitting its final structured report — the COMMIT is good;
verified by hand.

## Deferred (each needs its own gate / design decision)

**Realize-bulk fixture-cascade** (unchanged from (f)): q8_copy (Q8
round-trip gate), GQA-divergent w_o (n_heads*d_head≠d_model fixture),
per-block mmap LOAD (CUDA mmap gate + above), llama3 rope, qkv_bias,
B>1 mask body, ft-GGUF-load half.

**Other L4 recipes:** LoRA, WarmStart, Curriculum + the DataSpecs/Evals/
Stage/Trainer taxonomy — deferred (the FromScratch pass intentionally
didn't build the taxonomy speculatively). Each is its own pass; LoRA has
a natural gate (03_finetune_lora loss curve) if pursued.

## The next move (user decision)

1. **More L4 recipes** (LoRA next — has a gate via 03_finetune_lora).
2. **Realize-bulk fixture-cascade** — build Q8/GQA/qkv_bias/B>1 gates to
   unlock the rest of the realize decomposition (big, optional).
3. **P3 — core + CLI MVP** — the next phase proper.

The recipe API (`FromScratch#realize!/step!`, no Stage/Trainer) is worth
a user sanity-check before more recipes commit to that shape.

## Spinel landmine / refs

`Struct.new` accessor names = global type-merge keys → matz/spinel#1043
(probe: tinynn/probe_struct_accessor_collision.rb; memory #16). Value
objects: hand-written class, positional ctor, no defaults, prefixed
members. ggml#1506: mul_mat_id broken for K-quants (use Q8_0 experts) —
memory/reference_ggml_mul_mat_id_kquant.md; relevant for MoE.

## Workflows (all committed under prep/)

`wf_finish_{primitives,l2_block,l3_arch,realize_bulk,l4_recipes}.js`,
`wf_build_gguf_roundtrip_gate.js`. realize_bulk is triple-gated +
re-runnable as new gates land.

## Reference paths

- Monolith: `lib/llama_seq_forward_ffi.rb`
- Layers: `lib/toy/llm/{primitives,blocks,archs,recipes}/`
- Gates/fuser: `examples/smoke_{projection_lens[,_cuda],gguf_roundtrip,recipe_from_scratch}`, `lib/toy_gguf_fuse.rb`
- Mirror gen: `prep/gen_cuda_mirror.rb`
- Landmines: `memory/feedback_spinel_type_inference_landmines.md` (#16)
