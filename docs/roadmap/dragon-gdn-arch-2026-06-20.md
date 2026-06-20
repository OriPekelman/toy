# Dragon / Gated-DeltaNet — trainable hybrid arch

**Date:** 2026-06-20. **Status:** design + build-order for an agreed target.
**Decisions locked (user, 2026-06-20):** (1) target = **full *trainable* Dragon
arch**, not inference-only; (2) heterogeneous layers handled by a **per-layer
layer-type descriptor** (the reusable "first form" seam), not a Dragon-only
hardcode.
**Context:** [Dragon-3B-Base-alpha](https://huggingface.co/DragonLLM/Dragon-3B-Base-alpha)
+ [arch writeup](https://dragonllm.substack.com/p/inside-dragons-architecture).
Builds on the L1→L5 stack (`docs/architecture.md`) and is the first arch that
forces per-layer heterogeneity into the *trainable* seq path.

## What Dragon is

A **hybrid Gated-DeltaNet (GDN) + selective-attention** model (the
Qwen3-Next/GDN research lineage) — **not** a Llama variant. The layer stack is
**heterogeneous**: most layers are GDN linear-attention recurrent mixers; a few
are attention. Components:

- **Gated DeltaNet** linear-attention mixer (the bulk of layers), with an
  internal **short causal conv1d**.
- **Attention layers**: mostly **local windowed** (last ~1024 tokens); **3
  global** layers at thirds of the depth.
- **Differential attention** (MS DIFF: two softmax maps, λ-weighted subtract)
  on the global heads.
- **Head-wise normalization** (per-head norm / QK-norm).
- **NoPE for global** heads; **small-base RoPE for local** heads.
- **Scalable Softmax** (SSMax: length-dependent logit scaling, anti-fading).
- **Depth-dependent LayerNorm scaling**.
- SwiGLU FFN, RMSNorm, tied/untied embed (the Llama-standard parts).

Exact dims unknown (`config.json` is gated/401 on the alpha repo); dims don't
change *which* primitives are needed.

## Capability gap (toy today → Dragon)

The decisive find: **the vendored ggml (`41e7949`) already ships
`ggml_gated_delta_net(q,k,v,g,beta,state)` and `ggml_conv_1d`** — both
CPU-implemented (`ggml-cpu/ops.cpp:10789` `..._gated_delta_net_f32`). So the
hardest kernel is in-tree; the gap is FFI wiring + Ruby primitives + the
per-layer seam + (for training) **backward**.

| Dragon component | toy status |
|---|---|
| GDN mixer | ggml has the **forward** op → FFI-wire; **no ggml backward** → hand-write (training gate) |
| short conv1d | ggml has `conv_1d` / `ssm_conv` → FFI-wire |
| per-layer hybrid routing | **missing** — one shared block-ctx across all layers (`llama_arch.rb:214-218`) |
| differential attention | compose from matmul/softmax/scale → new L1 primitive (no kernel) |
| scalable softmax | compose from `tnn_scale`+softmax → new L1 primitive |
| head-wise / QK-norm | exists inference-only (`llama_kv_engine`) → lift into trainable block |
| local windowed attn | SWA exists inference-only → lift the mask into seq block |
| NoPE / small-base RoPE per layer | **missing** — one shared `RoPE::Cfg`; needs per-layer rope incl. skip |
| depth LayerNorm scaling | trivial — per-layer `tnn_scale` constant |
| SwiGLU / RMSNorm / GQA / RoPE / embed | **already have** |

## GDN op contract (for the forward primitive)

`ggml_gated_delta_net(q, k, v, g, beta, state)`, all F32:
- `v`: `[S_v, H, n_tokens, n_seqs]` — S_v = value head dim, H = heads, T, B.
- `q`,`k`: contiguous-rows (key head dim S_k along ne0).
- `g` (gate): `[1, H, T, B]` scalar, or `[S_v, H, T, B]` (KDA vector gate).
- `beta`: `ne[0]==1`, `[1, H, T, B]`.
- `state`: `[S_v*S_v*H, K, n_seqs, 1]`, K = snapshot-slot count.
- output: `[S_v*H, n_tokens*n_seqs + K*S_v*n_seqs, 1, 1]` — packs token outputs
  then the trailing state snapshots.

The q/k/v/g/beta projections (and the conv) are the surrounding graph the GDN
*primitive* owns; `ggml_gated_delta_net` is just the recurrence core.

## The per-layer layer-type descriptor (the chosen seam)

Replace `llama_arch`'s single shared `TransformerBlockCtx` with an **array of
per-layer `LayerSpec`**. `LlamaArch`/`HybridArch#build_forward` loops layers and
dispatches per `spec.kind`. A `LayerSpec` declares everything that varies by
layer:

- `kind` — `:attention` | `:gdn` (extensible: `:mamba`, …)
- attention-only: `attn_scope` (`:full` | local window int | `:global`),
  `rope` (`:rope` with a cfg | `:nope`), `diff_attn` (bool),
  `head_norm` (bool), `softmax` (`:standard` | `:scalable`)
- gdn-only: conv kernel size, S_k/S_v/H, gate kind (scalar/vector)
- shared: `norm_scale` (depth constant), `ffn` (`:swiglu` | `:moe` spec)

**Spinel hygiene is a hard design constraint here.** Per-layer-kind dispatch is
exactly the polymorphic-receiver shape that has repeatedly miscompiled on Spinel
(the #11/#12 landmine family, and the live #1449 GC bug). So `LayerSpec` must be
a **flat value struct with a `kind` integer and all fields always present**
(no subtype polymorphism), and the forward must dispatch with **explicit
`if kind == … elsif …` branches calling monomorphic builders** — never a
poly-array of heterogeneous spec objects with virtual dispatch. This keeps every
call site monomorphic. Unlocks Gemma-2 alternating-SWA, per-layer MoE, and
Dragon uniformly — the reusable payoff that justifies "first form" over a
Dragon-only arch.

## Progress

- **Phase 1 — DONE** (branch `feat/dragon-gdn`): `tnn_gated_delta_net` +
  `tnn_conv_1d` wired; `smoke_gdn_forward` + `gate-gdn-forward` green on the
  union pin; train gate 4/4.
- **Phase 2 — DONE**: 8 elementwise ggml ops wired (`sigmoid, exp, log, neg,
  sub, l2_norm, softplus, scale_bias`); L1 primitives `GDN` (l2/decay_gate/
  update_gate/recur/gated_out), `DiffAttention` (lambda_scalar/combine/subln),
  `ScalableSoftmax`, `DepthScale`. `gate-gdn-primitive` + `gate-dragon-attn-prims`
  green; train gate 4/4. (Detour lesson, captured in the smoke:
  `tnn_input_2d_f32_persistent(rows, cols)` → `ne0=cols, ne1=rows`, and
  `rms_norm` is over `ne0` — a per-head norm needs `rows=T, cols=F, gamma=[F]`.)
- **Phase 3+ — not started.** GDN/diff-attn primitives are CPU-only (not yet in
  `MIRRORABLE`; CUDA/Metal mirrors are a later pass).

## Build order (phased; each phase independently verifiable)

**Phase 1 — GDN forward through toy (foundational, de-risks the kernel).**
FFI-wire `tnn_gated_delta_net` + `tnn_conv_1d` in `tinynn_ggml.c` +
`lib/toy/ffi/tinynn.rb`; build on the **union pin**; a minimal forward smoke
(random q/k/v/g/beta/state → assert output shape + finite). Proves the in-tree
kernel computes through toy's FFI before any refactor. No Ruby arch yet.

**Phase 2 — composable L1 primitives (no kernels).** `Toy::GDN` (owns the
q/k/v/g/beta/conv projections around the core op), `Toy::DiffAttention`,
`Toy::ScalableSoftmax`, head-wise norm, depth-norm-scale. Each with a forward
smoke. Pure compositions of existing FFI ops.

**Phase 3 — the per-layer descriptor seam.** `LayerSpec` + refactor
`llama_arch` forward to loop per-layer specs with monomorphic kind-dispatch.
First validate it reproduces the *existing* homogeneous Llama byte-exact (all
layers = same attention spec) — a pure refactor gate before adding GDN layers.

**Phase 4 — GDN backward (the hard training gate).** ggml has no GDN backward;
hand-write it (toy's vendored-backward pattern: concat-back, gelu-back,
rms-norm-back precedent). Finite-difference validate against the forward. This
is the largest single risk and what makes "trainable" expensive.

**Phase 5 — hybrid forward + train.** Assemble Dragon's layer pattern as a
`LayerSpec` array; from-scratch train smoke (tiny shape) → byte-exact gate.
Lift SWA + QK-norm into the trainable block along the way.

**Phase 6 — converter + real weights.** Dragon GGUF → toy. Gated by: a converter
exists at all (Dragon ships custom Triton kernels + bespoke modeling code — no
llama.cpp path is known) AND ggml's `gated_delta_net` variant matches Dragon's
GDN (broadcast layout / gate form may differ from the Qwen3-Next target it was
written for). Until then, validate on synthetic/from-scratch shapes.

## Hard gates / risks (honest)

1. **GDN backward (Phase 4)** — no ggml backward exists; the recurrence makes it
   the hardest kernel toy would have hand-written. This is the cost of
   "trainable" vs "inference-only."
2. **No known converter** — Dragon's custom kernels/modeling mean Phase 6 may
   need a from-scratch converter, and the ggml GDN op variant must be confirmed
   to match Dragon's (else even forward is wrong on real weights).
3. **Spinel codegen** — the per-layer dispatch must stay monomorphic (above);
   and all of this must build on the **union pin** while master/#1449 is
   unresolved (the new FFI ops are simple decls — fine on legacy — but any
   poly-dispatch in the seam risks the #11/#12 family).
4. **Op-variant drift** — `ggml_gated_delta_net` is recent and flagged with TODOs
   (`[TAG_GGML_GDN_BCAST]`); a ggml bump could change its contract.

## What this is NOT

- Not inference-only (decision: full trainable).
- Not a Dragon-specific hardcoded arch (decision: reusable per-layer descriptor).
- Not started past Phase 1.
