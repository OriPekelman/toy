# Session resume — 2026-05-29 (b): L1 primitive layer DONE

> **Read this first.** Supersedes SESSION-RESUME-2026-05-29.md (now
> deleted). The entire L1 primitive extraction is landed and gated.
> The next concrete move is L2 — the TransformerBlock.

## Where we are

| Phase | Status | Commit |
| --- | --- | --- |
| P0 — Design lock + Tao coord | done | bc1b40e |
| P1 — Card derivation refactor | done | 07aa547 |
| **P2 — Five-layer refactor** | **L1 done (all 4 primitives); L2 next** | d3d124e |
| P3 — Core + CLI MVP | pending | — |
| P4 — CLI complete | pending | — |
| P5 — Generators | pending | — |
| P6 — Prism lowerer | pending (optional) | — |

## What landed this session (one commit per primitive, gated)

A workflow (`prep/wf_finish_primitives.js`) extracted the four L1
primitives serially, each behind a **bit-identical-logits gate** +
`make verify-mirrors`, one commit each, in order:

| Commit | Primitive | File |
| --- | --- | --- |
| 347d452 | RoPE | `lib/toy/llm/primitives/rope.rb` (+ `_cuda`/`_metal`) |
| 8efb16e | SwiGLU | `lib/toy/llm/primitives/swiglu.rb` |
| cdb021d | RMSNorm | `lib/toy/llm/primitives/rms_norm.rb` |
| d3d124e | GQA | `lib/toy/llm/primitives/gqa.rb` |

**All four are pure modules** — `self.` methods only, no module ivars
(Spinel ivar-layout safety), no default-arg Cfg ctors (landmine #4).
Weight/KV-cache ownership stays on the caller (the monolith block).

**Independently verified at HEAD:**

- No bare `tnn_rope_ext` / `tnn_silu` / `tnn_rms_norm` left in
  `lib/llama_seq_forward_ffi.rb` (grep clean).
- `make verify-mirrors` passes on all 18 mirror files.
- A/B fixture (`SEED=0 STEPS=5 ./examples/smoke_projection_lens`,
  built at `bd74cc8` vs `d3d124e`) produces **byte-identical** losses.

GQA — flagged in the prior doc as "too thick" — extracted cleanly by
encapsulating only the attention math (`scores → scaled+masked
softmax → weighted V`, `attention(sess, t_k, t_q, t_vt, attn_mask,
scale, batch)`). The two softmax branches (B>1 fused `soft_max_ext`
vs B=1 `scale`+`diag_mask_inf`+`softmax`) are preserved exactly — do
NOT unify them; they are documented bit-identical to pre-GH#7 at B=1.

## Not yet done

- **Nothing pushed.** All 4 commits are local on `main`. Push to
  origin when ready (per the durable "land on main, push when ready"
  posture).
- **CUDA runtime parity is mirror-source-verified, not bit-checked.**
  `make verify-mirrors` guarantees the `_cuda`/`_metal` sources match
  the generator, and each primitive's CUDA path was smoke-run on the
  GB10 (it executes), but there is no CUDA build target for
  `smoke_projection_lens`, so no byte-for-byte CPU/CUDA parity run.
  A full CUDA fixture needs `make setup-ggml-cuda` (the checked-in
  `libtinynn_ggml_cuda.a` is the 4 KB AB-smoke stub). If L2 work wants
  a true CUDA gate, build that fixture first.

## The next concrete move: P2.4 — L2 TransformerBlock

The block is the unit that composes the four L1 primitives +
owns the weights. In `lib/llama_seq_forward_ffi.rb` the per-layer
body is `build_seq_block` + `build_seq_qhead` — these now call
`RoPE.apply_2d`, `SwiGLU.gate`, `RMSNorm.build`, `GQA.attention`.
The L2 task is to lift that body into
`lib/toy/llm/blocks/transformer_block.rb` (`Toy::LLM::Blocks::...`),
with the block owning its param tensors + KV cache, and the monolith
realize paths handing the block its weights.

Same gate every step: bit-identical losses on the A/B fixture +
clean `make verify-mirrors` + no Spinel warnings. Do the block as ONE
unit (not field-by-field) but still ONE commit, and if the gate
fails, `git reset` and re-plan rather than patch around.

After L2: L3 `LlamaArch` (P2.5), then the realize-path bulk
(P2.6/P2.7) — which becomes tractable once the block is the
allocation unit.

## Reusing the workflow

`prep/wf_finish_primitives.js` is the gated-extraction harness:
read-only recon fan-out → serial gated extraction (one commit each,
stop on first gate failure). The L2/L3 work can adapt the same
script — the gate fixture (`smoke_projection_lens`, deterministic at
`SEED=0 STEPS=5`) and the `git reset`-on-failure discipline carry
over unchanged.

## Reference paths

- Monolith: `lib/llama_seq_forward_ffi.rb`
- Primitives (done): `lib/toy/llm/primitives/{rope,swiglu,rms_norm,gqa}.rb`
- Blocks (next): `lib/toy/llm/blocks/` (currently README only)
- Mirror generator: `prep/gen_cuda_mirror.rb` (regex already accepts
  `lib/toy/llm/primitives/*.rb`; extend for `blocks/` in L2)
- Gate fixture: `SEED=0 STEPS=5 ./examples/smoke_projection_lens`
- Extraction workflow: `prep/wf_finish_primitives.js`
- Landmines: `memory/feedback_spinel_type_inference_landmines.md`,
  `memory/project_step_bind_landmine_2026_05_28.md`
