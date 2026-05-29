# Session resume — 2026-05-29 (c): L2 TransformerBlock DONE

> **Read this first.** Supersedes SESSION-RESUME-2026-05-29b.md (now
> deleted). L1 (all 4 primitives) AND L2 (TransformerBlock) are landed
> and gated on `main`. The next concrete move is L3 — the LlamaArch.

## Where we are

| Phase | Status | Commit |
| --- | --- | --- |
| P0 — Design lock + Tao coord | done | bc1b40e |
| P1 — Card derivation refactor | done | 07aa547 |
| **P2 — Five-layer refactor** | **L1 done; L2 done; L3 next** | f29fc03 |
| P3 — Core + CLI MVP | pending | — |
| P4 — CLI complete | pending | — |
| P5 — Generators | pending | — |
| P6 — Prism lowerer | pending (optional) | — |

## What landed

**L1 — primitives (P2.3/P2.4):** RoPE 347d452, SwiGLU 8efb16e,
RMSNorm cdb021d, GQA d3d124e. Pure `self.`-method modules in
`lib/toy/llm/primitives/`.

**L2 — TransformerBlock (P2.4):** f29fc03 (fast-forwarded onto main
from a branch). `lib/toy/llm/blocks/transformer_block.rb` —
`Toy::LLM::Blocks::TransformerBlock`. The former `LlamaSeqBlockFFI`
class + `build_seq_block`/`build_seq_qhead`/`mp_matmul` moved verbatim
(op order unchanged) into the block, which now OWNS its weight handles
(`t_seq_*`) and exposes `build_forward(sess, t_x, ctx)`. The cache
(`LlamaSeqForwardFFICache`) hands each block a per-forward context
object and keeps the block-stacking loop, realize allocation, final
norm, LM head, embeddings. Net −240 lines per monolith mirror.

**Independently verified at HEAD (not just agent self-report):**
- A/B fixture (`SEED=0 STEPS=5 ./examples/smoke_projection_lens`,
  built at `8a29855` vs `f29fc03`) → byte-identical losses.
- `make verify-mirrors` clean on all mirror files.
- No live `LlamaSeqBlockFFI` / `build_seq_block` / `build_seq_qhead`
  code remains in the monolith (only 3 stale comments at lines
  79/118/729 — harmless, could be tidied in L3).

## Spinel landmine hit during L2 (now in memory)

The per-forward context object could NOT be a `Struct.new(...)`: the
synthesized accessors (`#n_heads`, `#t`, `#b`) unified with same-named
methods/fields elsewhere (`SmolLM2Config#n_heads`, `Toy::Linear#b`)
and mis-compiled unrelated code. Fix: a plain class
`TransformerBlockCtx` with an explicit positional `initialize` (no
kwargs/defaults) and uniquely-prefixed members (`seq_scale`, `seq_eps`,
`t_seq_positions`, `seq_t`, `seq_b`, `t_seq_attn_mask`). See landmine
#16 in `memory/feedback_spinel_type_inference_landmines.md`. **Reuse
this Ctx pattern for L3** — same hazard applies to any new value object.

## Not yet done / caveats

- **Nothing pushed.** All commits local on `main`.
- **CUDA is mirror-source-verified + runs**, not bit-checked vs CPU.
  The CUDA block path builds and trains on the GB10 (random-init
  fixture, CE 6.48→3.23/60 steps), and `verify-mirrors` guarantees
  source consistency — but there is no CUDA `smoke_projection_lens`,
  so no byte-for-byte CPU/CUDA forward parity. Build one
  (`make setup-ggml-cuda` for the full backend) if L3 wants a true
  CUDA gate.
- **B>1 batch path not separately smoke-tested** (no B>1 CPU fixture);
  `attn_mask` + `seq_b` threaded through unchanged.

## The next concrete move: P2.5 — L3 LlamaArch

The arch is the stacking unit. Per `lib/toy/llm/archs/README.md` and
the design doc §6, L3 owns: token embedding, (learned) position
embedding, final norm, LM head, block stacking + per-layer override
resolution, and whole-graph allocation. In the monolith that's the
`LlamaSeqForwardFFICache` realize paths + the block-stacking loop +
the final-norm/LM-head tail. The L3 task is to lift the
arch-level orchestration into `lib/toy/llm/archs/llama_arch.rb`
(`Toy::LLM::Archs::...`), leaving the cache as a thin FFI/realize
shell — or folding it in, per the design.

This is the bigger lift (it touches all 4 realize paths, which are the
1750-line bulk). The roadmap explicitly defers the realize-path
*restructuring* to P2.6/P2.7; P2.5 should be the minimal faithful lift
of the arch *orchestration*, same bit-identical gate.

Same gate every step: bit-identical losses on the A/B fixture + clean
`make verify-mirrors` + no Spinel warnings. One unit, one commit;
`git reset` + re-plan on gate failure.

## Reusing the workflows

- `prep/wf_finish_primitives.js` — L1 multi-primitive gated loop.
- `prep/wf_finish_l2_block.js` — L2 single-unit harness: baseline →
  parallel facet recon → architect plan → single gated extraction.
  The L3 work should adapt the L2 script (it's the closer template:
  recon facets become embedding/posemb/final-norm/lm-head/stacking/
  realize-paths/mirror-gen; the plan step settles the arch API). Keep
  the proven gate fixture and the reset-on-failure discipline.

## Reference paths

- Monolith: `lib/llama_seq_forward_ffi.rb`
- Primitives (done): `lib/toy/llm/primitives/{rope,swiglu,rms_norm,gqa}.rb`
- Block (done): `lib/toy/llm/blocks/transformer_block.rb`
- Arch (next): `lib/toy/llm/archs/` (currently README only)
- Mirror generator: `prep/gen_cuda_mirror.rb` (regex accepts
  `lib/toy/llm/{primitives,blocks}/*.rb`; extend for `archs/` in L3)
- Gate fixture: `SEED=0 STEPS=5 ./examples/smoke_projection_lens`
- Extraction workflows: `prep/wf_finish_l2_block.js` (L3 template),
  `prep/wf_finish_primitives.js`
- Landmines: `memory/feedback_spinel_type_inference_landmines.md`
  (now incl. #16 Struct.new), `memory/project_step_bind_landmine_2026_05_28.md`
