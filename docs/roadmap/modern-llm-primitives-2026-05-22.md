# Modern LLM primitives — what we need to add

**Date:** 2026-05-22
**Predecessor:** v0.1.0-pre-alpha (CHANGELOG.md).
**Status:** survey + prioritization. No implementations queued yet.

## Goal

Catalog what's missing to run / fine-tune the current generation of
SOTA open-weight LLMs (Qwen3 dense + MoE, DeepSeek V3 / R1,
Llama 3.3 / 4.x, Gemma 2/3, Mistral / Mixtral, Phi-3, GLM-4.x). The
question isn't "how do we implement each" — it's "what's the
shortest path that unlocks the most families?"

## What we have today (post v0.1.0-pre-alpha)

**Inference primitives (CPU + CUDA):**
RMSNorm, LayerNorm, SwiGLU FFN (SiLU + mul + matmul), RoPE-NEOX
(`tnn_rope_ext`, basic; no scaling args bound), GQA via the
broadcast pattern, KV cache (`tnn_set_2d` / `tnn_view_2d` / `tnn_cpy`),
full-sequence forward (M3), `tnn_soft_max_ext` with mask + scale,
Q8_0 mmap'd weights, `tnn_get_rows` for embed.

**Training primitives:**
`tnn_rms_norm_back`, `tnn_silu_back`, `tnn_rope_ext_back`,
`tnn_get_rows_back` (chunked for vocab > 65535 — patch 0006),
`tnn_softmax_back`, `tnn_cross_entropy_loss`, `tnn_opt_step_adamw`,
`tnn_opt_step_sgd`, dual-cgraph training driver, vendored `concat`
backward (patch 0005).

**ggml upstream already has (no binding yet):**
`FLASH_ATTN_EXT`, `FLASH_ATTN_BACK`, `MUL_MAT_ID` (sparse MoE expert
dispatch), `ADD_ID`, `GLU`, `TOP_K`, `ARGSORT`, `GROUP_NORM`,
`L2_NORM`, `GATED_DELTA_NET`, `GATED_LINEAR_ATTN`, `RWKV_WKV6/7`,
`SSM_CONV`, `SSM_SCAN`, `CUMSUM`, `TRI`, `SOLVE_TRI`. RoPE scaling
args (freq_scale, ext_factor, attn_factor, beta_fast, beta_slow,
optional freq_factors tensor) live inside `ggml_rope_ext` but our
`tnn_rope_ext(sess, a, pos, n_dims, freq_base)` binding drops all
of them — so YaRN / NTK / LongRoPE are unreachable until we widen
the binding.

## Per-family deltas

### Qwen3 dense (4B–32B)

| Delta | Blocks | Difficulty |
| --- | --- | --- |
| QK-norm (RMSNorm Q and K per-head before RoPE) | both | small — extra `tnn_rms_norm` per head; `qk_norm` flag already in `Arch` |
| QKV bias dropped (Qwen3 removed it) | both | trivial — `arch.qkv_bias = false` |
| Longer rope_base (1e6) + 32k–128k context — needs YaRN | inference | small once RoPE bindings widen |

**Smallest delta on the list.** Qwen3 dense ≈ Qwen2.5 minus bias plus QK-norm.

### Qwen3 MoE (30B-A3B, 235B-A22B)

| Delta | Blocks | Difficulty |
| --- | --- | --- |
| Router (linear → top-K softmax over n_experts) | both | small — matmul + `TOP_K` + `soft_max` |
| Sparse expert dispatch (n_experts_used-of-N) | both | medium — bind `MUL_MAT_ID` + `ADD_ID`; weights become 3-D `[d_in, d_out, n_experts]` |
| GGUF expert tensor layout | both | small loader change |
| Load-balancing aux loss | training | medium — router logits as side output + KL/entropy loss outside CE graph |

**Almost entirely gated by `MUL_MAT_ID` binding.**

### DeepSeek V3 / R1

| Delta | Blocks | Difficulty |
| --- | --- | --- |
| MLA — multi-head latent attention (compressed KV) | both | medium — new attention block shape; ops exist, graph builder rewrite |
| Decoupled RoPE (RoPE on a `qk_rope_head_dim` slice, NoPE on the rest) | both | small once MLA scaffolding lands |
| Fine-grained MoE (256+ experts, top-8) | both | same primitives as Qwen3 MoE |
| Shared experts (always-on) | both | small — add a non-routed branch |
| Sigmoid gating + per-group top-K | both | small — different reduction in router |
| Multi-token prediction (auxiliary heads predicting t+1..t+k) | training; nice-to-have for inference (speculative) | medium |
| FP8 weights (native DeepSeek-V3 format) | inference | large — loader work; dequant to F16/F32 at load is acceptable interim |

**MLA is the single biggest delta.** MoE bits reuse Qwen3 MoE.

### Llama 3.3 / 4.x

| Delta | Blocks | Difficulty |
| --- | --- | --- |
| 128k context + Llama-3 RoPE freq scaling | inference | small once `tnn_rope_ext` widens |
| Llama-4 MoE (Scout/Maverick) with interleaved chunked attention | both | medium — reuses MoE primitives; chunked attention = mask shape |
| Llama-4 iRoPE (interleaved RoPE/NoPE layers) | both | small — per-layer flag on `Arch` |

3.3 ≈ free after RoPE-binding widening. 4.x = "MoE + mask variations."

### Gemma 2 / 3

| Delta | Blocks | Difficulty |
| --- | --- | --- |
| Interleaved local-global attention (alternating SWA + full) | both | medium — per-layer mask + per-layer KV-cache size |
| Sliding-window mask (4k window on local layers) | both | small — mask in Ruby; KV cache eviction policy |
| Logit soft-cap (`tanh(logits/cap)*cap`) | inference | small — `tnn_scale` + a `tnn_tanh` unary (binding for existing `GGML_UNARY_OP_TANH`) |
| Pre+post-norm sandwich | both | trivial — extra `tnn_rms_norm` in graph builder |
| 256k vocab + large embed | inference | medium — memory + sampler top_k/top_p perf check over 256k |
| Embedding scale (`embed * sqrt(d_model)`) | both | trivial — `arch.embed_scale` exists |
| Gemma 3 vision tower | inference | large — out of scope |

**No new ggml ops needed.** Almost all the work is mask construction + per-layer plumbing.

### Mixtral 8x22B

Same primitives as Qwen3 MoE (2-of-8 routing). If we bind `MUL_MAT_ID`, Mixtral runs.

### Phi-3

| Delta | Blocks | Difficulty |
| --- | --- | --- |
| LongRoPE (per-dim freq_factors tensor — NOT a scalar) | inference | medium — `ggml_rope_ext` accepts a `freq_factors` tensor arg; add overload `tnn_rope_ext_freq_factors` |
| SWA-4k (Phi-3-mini only; medium uses full attention) | both | small (mask) or none |
| Fused QKV weight (`Wqkv`) | both | trivial — split at load |

### GLM-4.x

| Delta | Blocks | Difficulty |
| --- | --- | --- |
| Partial RoPE (rotate 50% of head dim, NoPE on the rest) | both | small — `tnn_rope_ext` already accepts `n_dims < head_dim`; `Arch.rope_partial` already exists |
| Post-norm sandwich | both | trivial |
| GeGLU instead of SwiGLU (some variants) | both | small — `tnn_gelu` + `tnn_mul` |
| GLM-4.5/4.6 MoE | both | reuses MoE primitives |

## Shared infrastructure (not model-specific)

- **MoE routing (`MUL_MAT_ID` + `ADD_ID` + `TOP_K` + `ARGSORT`).**
  All exist in ggml. We have zero bindings. Medium effort; unblocks
  every MoE family on the list.
- **Sliding-window attention.** No new op needed; SWA is a different
  mask tensor fed to `soft_max_ext`. Small effort: mask construction
  + per-layer KV size policy.
- **YaRN / NTK / LongRoPE.** `ggml_rope_ext` already takes
  `freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, freq_factors`.
  Our binding drops them. Small effort: widen the signature.
- **Flash attention.** `FLASH_ATTN_EXT` + `FLASH_ATTN_BACK` exist in
  ggml. Bind + integrate = medium. Validate backward parity = large.
  Performance only — current path is correct.
- **Speculative decoding.** No new ops; harness-level (draft model
  + verify via existing forward graph). Medium-to-large. Model-agnostic.
- **Long context (paged / chunked KV).** No new op. Graph
  construction concern. Medium effort if we need >128k.

## Priority recommendation — three to pick first

Each of these is one focused piece of work that unblocks multiple
families. They form a natural sequence:

1. **Bind `MUL_MAT_ID` + `ADD_ID` + `TOP_K` + `ARGSORT`.**
   Unblocks Mixtral 8x22B, Qwen3-MoE, DeepSeek-V3 FFN, Llama-4,
   GLM-MoE. **Five families** behind one binding cluster.

2. **Widen `tnn_rope_ext` to expose `freq_scale, ext_factor,
   attn_factor, beta_fast, beta_slow, freq_factors`.** Unlocks YaRN
   (Qwen3 long-context), Llama-3.3 128k, Phi-3 LongRoPE, GLM partial
   RoPE. **Four families** behind one signature change.

3. **Sliding-window mask + per-layer KV sizing.** No new op needed.
   Gates Gemma 2/3 (interleaved local/global) and Phi-3-mini-4k;
   also feeds Llama-4 chunked attention. Pure Ruby + graph-builder.

After these three:
- **Qwen3 dense** is small additional work (QK-norm + bias-off + the
  RoPE binding from #2).
- **Gemma** needs soft-cap (bind `GGML_UNARY_OP_TANH`).
- **DeepSeek MLA** is the biggest single-family delta remaining and
  the natural next milestone after the priority three.
- **Flash attention** should follow as a perf upgrade once correctness
  paths are in place.

## Key insight

Most "missing" primitives aren't missing in ggml — they're missing
in our FFI surface. The cheapest quarters are the ones where we
widen bindings rather than touch CUDA kernels.

## What's NOT on this list

- Mamba / SSM models (Mamba-2, Jamba). ggml has `SSM_CONV` / `SSM_SCAN`
  but the architecture is enough of a departure that it deserves its
  own scoping doc.
- RWKV. Similarly, ggml has `RWKV_WKV6/7`; separate scoping.
- Diffusion / image generation. Not on the roadmap.
- Encoder-decoder models (T5 / Flan). Decoder-only universe.
