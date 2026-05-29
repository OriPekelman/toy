# Session resume — 2026-05-29 (f): L1–L3 done; realize-bulk at gate ceiling; CUDA+GGUF gates live; L4 next

> Supersedes SESSION-RESUME-2026-05-29e.md (deleted). The forward
> five-layer refactor (L1–L3) is done. The realize bulk is decomposed
> as far as the current gates safely allow. A CUDA gate and a GGUF F32
> mmap round-trip gate now exist. L4 recipes is the next move — but it
> is greenfield API design, not mechanical extraction (see below).

## Where we are

| Phase | Status | Commit |
| --- | --- | --- |
| P0 / P1 | done | bc1b40e / 07aa547 |
| **P2 — Five-layer refactor** | **L1+L2+L3 done; realize-bulk at gate ceiling; L4 next** | 50cb79d |
| P3–P6 | pending | — |

Everything is on `main` and pushed to BOTH remotes (origin = local bare,
github = OriPekelman/toy). `git cherry`-verified the earlier origin
divergence was 63 patch-equivalent commits (no lost work); reconciled
via `-s ours` merge (5022fdb).

## Gates (the spine of this whole effort)

| Gate | Command | Baseline | Exercises |
| --- | --- | --- | --- |
| CPU | `SEED=0 STEPS=5 ./examples/smoke_projection_lens` (loss lines) | `4eaf019b…` | realize_for_random_init forward+train, CPU |
| CUDA | `…_cuda` | `bac45f88…` | same, GB10 GPU (self-consistent; GPU floats ≠ CPU) |
| GGUF | `SEED=0 ./examples/smoke_gguf_roundtrip` (full stdout) | `c89fd3eb…` | realize_for_mmap F32 reload (head-fused round-trip) |

All three are deterministic and gated against their OWN baseline.

## What landed since (e)

- **CUDA gate** (1f8c27e): `examples/smoke_projection_lens_cuda` via the
  mirror generator. Real GB10 build (712 MB static link), deterministic.
- **GGUF round-trip gate** (4f551bf): `lib/toy_gguf_fuse.rb` (ToyGGUFFuser
  — per-head download → head-major concat → fused-name write) +
  `examples/smoke_gguf_roundtrip.rb`. Reload reproduces the in-memory
  forward bit-for-bit (all 20064 logits ==). Pure Ruby, no mirror change.
  Insight: head-major concat IS the identity byte layout mmap re-slices.
- **Realize-bulk pass 1** (0b3e9eb…29ab9f1): 4 dual-gated steps
  (apply_seq_cfg!, arch.seed_blocks!, finalize_and_realize!,
  block.alloc_trainable_f32_weights!).
- **Realize-bulk pass 2** (50cb79d): 1 triple-gated step —
  `LlamaArch#load_globals_from_gguf_mmap!` (mmap global-tensor alloc),
  unlocked by the GGUF gate.

All independently re-verified (A/B vs pre-refactor; HEAD reproduces all
three baselines; verify-mirrors clean).

## Realize bulk: at the GATE CEILING (a decision point)

Further safe extraction is blocked by missing gates. Everything still
deferred needs ITS OWN new fixture before it's safe to touch:

| Deferred | Needs |
| --- | --- |
| `realize_for_q8_copy` (whole path) | a Q8 round-trip gate (quantize-on-write — ToyGGUFWriter is F32-only) |
| GQA-divergent `attn_output` (w_o `[d_model,d_model]` vs `[n_heads*d_head]`) | a fixture where n_heads*d_head ≠ d_model (current gates PIN them equal → blind) |
| per-block mmap LOAD loop | a CUDA mmap gate (round-trip is CPU-class) + the above; the loop interleaves un-gated LoRA-Q + qkv_bias branches |
| llama3 rope_freq_factors branches | a llama3 fixture (gates are non-llama3) |
| qkv_bias branches | a qkv_bias=true fixture |
| B>1 attn-mask body | a B>1 fixture (gates are B=1) |
| `realize_for_full_finetune` GGUF-load half | gates mmap reload, not ft-load (dequant-slice path) |

**This is a fixture-cascade.** Each unlock is its own engineering task.
Worth deciding with the user whether to invest in it vs. move on. The
re-runnable workflow `prep/wf_finish_realize_bulk.js` will pick up new
gates automatically (add the gate's build+fp+baseline to its triple-gate
block and re-run).

## The next move: L4 recipes — NOTE it's greenfield design

`lib/toy/llm/recipes/README.md` rosters FromScratch / LoRA / WarmStart /
Curriculum recipes composing Arch + Trainer + DataSpec + Eval + Stage.
**None of Trainer/DataSpec/Eval/Stage/Recipe exist yet** — only the
sketch. The training loops are inlined in the example drivers
(06_train_from_scratch, 03_finetune_lora, 09_warm_start_train) and
`lib/toy_trainer.rb` / `lib/training.rb` / `lib/toy_lr_schedule.rb`.

So L4 is API design + extraction, unlike the mechanical L1–L3. Behavior
is still gateable: `smoke_projection_lens` IS a from_scratch training
loop (its loss curve is the reference). Recommended gate for the
FromScratch recipe: a NEW fixture that drives the SAME random_init+config
THROUGH the recipe and must reproduce the `4eaf019b…` loss curve
bit-identically (don't rewire the gate fixture itself — that's circular).

Conservative scope for pass 1: extract ONLY FromScratch with the
smallest viable Recipe/Trainer surface; DEFER LoRA/WarmStart/Curriculum
and the full DataSpec/Eval taxonomy; surface the API shape for review.

## Spinel landmine (upstream)

`Struct.new` accessor names = global type-merge keys → matz/spinel#1043.
Value objects: hand-written class, positional ctor, no default args,
uniquely-prefixed members. Probe: `tinynn/probe_struct_accessor_collision.rb`.
Also noted: ggml#1506 (mul_mat_id broken for K-quants; use Q8_0 experts)
— see `memory/reference_ggml_mul_mat_id_kquant.md`; relevant for MoE.

## Workflows

`prep/wf_finish_{primitives,l2_block,l3_arch,realize_bulk}.js` +
`wf_build_gguf_roundtrip_gate.js`. The realize_bulk one is triple-gated
and re-runnable as new gates land.

## Reference paths

- Monolith (realize bulk + delegators): `lib/llama_seq_forward_ffi.rb`
- Layers: `lib/toy/llm/{primitives,blocks,archs}/`; recipes (next): `recipes/`
- Mirror generator: `prep/gen_cuda_mirror.rb`
- Gate fixtures + fuser: `examples/smoke_{projection_lens,projection_lens_cuda,gguf_roundtrip}`, `lib/toy_gguf_fuse.rb`
- Landmines: `memory/feedback_spinel_type_inference_landmines.md` (#16)
