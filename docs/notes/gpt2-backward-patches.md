# GPT-2 training — the two ggml backward patches (foundation for #12)

GPT-2 train-from-scratch hits two of the three ggml backward gaps in
[ggml-org/ggml#1514](https://github.com/ggml-org/ggml/issues/1514): `GGML_OP_NORM`
(LayerNorm) and `GGML_UNARY_OP_GELU`. (The third, conv_2d/CONT, is ViT-only.)
Decision (2026-06-03): vendor both as fused backwards, modeled on the existing
`ggml_rms_norm_back` / `ggml_silu_back` siblings. CPU first (the gated reference).

## Key simplification

`ggml_norm` is the **pure normalize** `y = (x - mean)/sqrt(var + eps)` — it has
NO affine. GPT-2's LayerNorm γ/β are separate `ggml_mul` + `ggml_add` in the
graph, and those already have backward. So we only need the **normalize**
backward, not a γ/β-aware one. That makes it a near-twin of `ggml_rms_norm_back`
(which is the RMS-normalize backward) — just with mean-centering.

## Patch 1 — `ggml_norm_back(ctx, a /*x*/, b /*dy*/, eps)`

Template: `ggml_rms_norm_back` everywhere.
- **ggml.h**: declare next to `ggml_rms_norm_back` (`:1408`).
- **ggml.c**: constructor + a `GGML_OP_NORM_BACK` enum (mirror `GGML_OP_RMS_NORM_BACK`
  at `:3157`); add `"NORM_BACK"` to the op-name table (mirror `:1003`).
- **ggml-cpu/ops.cpp**: `ggml_compute_forward_norm_back_f32`, modeled on
  `ggml_compute_forward_rms_norm_back_f32` (`:3831`) and the forward
  `ggml_compute_forward_norm_f32` (`:3696`) for the mean/var convention.
- **ggml.c `ggml_compute_backward`**: add `case GGML_OP_NORM:` (currently absent
  → aborts at `:6874`) that emits `ggml_norm_back(ctx, src0, grad, eps)` into src0.

Math (per row, D = ne0; mean/var over D):
```
mu   = mean(x);  var = mean((x-mu)^2);  rstd = 1/sqrt(var+eps)
xhat = (x - mu) * rstd
dx_i = rstd * ( dy_i - mean(dy) - xhat_i * mean(dy .* xhat) )
```
(rms_norm_back is the same minus the `- mean(dy)` mean-centering term and with
`mean(x^2)` instead of `var`. Read its f32 body and add the centering.)

## Patch 2 — `ggml_gelu_back(ctx, grad, x)`

Template: `ggml_silu_back` (`ggml.h:1194`, `ggml.c:2820`); dispatch at
`ggml.c:6815-6841` already does `case GGML_UNARY_OP_SILU: … ggml_silu_back`.
- Add `case GGML_UNARY_OP_GELU:` (and optionally `GELU_ERF`, `GELU_QUICK`) →
  `ggml_gelu_back(ctx, grad, src0)`.
- `ggml_compute_forward_gelu_back_f32`: `dx = grad * gelu'(x)`. ggml's gelu is the
  tanh approximation: `gelu(x) = 0.5 x (1 + tanh(g))`, `g = √(2/π)(x + 0.044715 x³)`.
  Derivative: `0.5(1+tanh g) + 0.5 x (1 - tanh² g) · g'`, `g' = √(2/π)(1 + 3·0.044715 x²)`.

## tinynn bindings + probes

- `tinynn/tinynn_ggml.{h,c}`: `tnn_layer_norm_back(sess, x, dy, eps)` and
  `tnn_gelu_back(sess, grad, x)` (thin wrappers, like `tnn_rms_norm_back` at
  `tinynn_ggml.c:1215`). Bind in `lib/tinynn.rb` next to `tnn_rms_norm_back`.
- Standalone C probes modeled on `tinynn/rms_norm_back_probe.c`: call the op
  directly + finite-difference check the gradient. **#1491 caution:** validate
  on the *backend-sched* compute path (the one training actually uses), not only
  `compute_with_ctx` — ggml#1491 shows `rms_norm_back` disagrees between the two.

## Then (part b, the arch)

A GPT-2 training arch on the engine: learned positional embeddings (added to
token embed), GELU FFN, LayerNorm (norm + mul γ + add β), tied output embedding.
A `prep/gpt2_train_gate.rb` byte-exact gate (record-from-inline-first, like
`full_finetune_gate.rb`). CPU gated reference; CUDA/Metal mirror after.

## Confirmed implementation detail (2026-06-04, from reading the vendored ggml)

**Exact autograd dispatch points** (`vendor/ggml/src/ggml.c`, in `ggml_compute_backward`):
- GELU: the unary switch handles SILU at `:6820` and aborts at the `default` `:6843`.
  Add `case GGML_UNARY_OP_GELU:` right after SILU.
- NORM: there is no `case GGML_OP_NORM:` (the RMS_NORM case is at `:6539`), so it
  falls through to the op-level `default` abort at `:6874`. Add a `case GGML_OP_NORM:`
  modeled on `GGML_OP_RMS_NORM` (which reads `eps` from `tensor->op_params`).

**GELU forward (must match for the derivative)** — `vec.h:986`:
`gelu(x) = 0.5·x·(1 + tanh(g))`, `g = SQRT_2_OVER_PI·x·(1 + GELU_COEF_A·x²)`,
`SQRT_2_OVER_PI = 0.79788456080286535588`, `GELU_COEF_A = 0.044715`.
Derivative: `gelu'(x) = 0.5(1+t) + 0.5·x·(1−t²)·g'`, `t = tanh(g)`,
`g' = SQRT_2_OVER_PI·(1 + 3·GELU_COEF_A·x²)`.

**Approach decision — DEDICATED kernels, not synthesized subgraphs.** The
synthesized route (build the gradient from existing ggml ops in the autograd case)
avoids new `GGML_OP` enums but is fiddly: it needs scalar-add (`ggml_add1` is
deprecated) and per-row broadcast (for NORM's `mean`), which is error-prone. The
dedicated kernels have far simpler, directly-verifiable math:
- `ggml_gelu_back(ctx, grad, x)` → `GGML_OP_GELU_BACK`, elementwise `grad·gelu'(x)`.
  Plumbing copies `ggml_silu_back` (`ggml.c:2820`) + `GGML_OP_SILU_BACK` (enum,
  op-name table, ops.cpp compute dispatch).
- `ggml_norm_back(ctx, x, dy, eps)` → `GGML_OP_NORM_BACK`, kernel is a near-copy of
  `ggml_compute_forward_rms_norm_back_f32` (`ops.cpp:3831`) plus the mean-centering
  term (`− mean(dy)`) and `var` instead of `mean(x²)`. Mean/var convention from
  `ggml_compute_forward_norm_f32` (`ops.cpp:3696`).

**Fast validation loop (no full ggml rebuild):** model a standalone probe on
`tinynn/rms_norm_back_probe.c` — it compiles the ggml sources directly. Build
`x → gelu(x) → sum` (and `x → norm(x) → sum`), `ggml_build_backward_expand`,
compute, then finite-difference each `x_i` and compare to the autograd grad.
**#1491 caution:** run the probe on the **backend-sched** compute path (the one
training uses), not only `compute_with_ctx`.

## Status

Foundation fully scoped + implementation-ready (this doc): every patch point at
file:line, the GELU derivative confirmed against ggml's forward, the dedicated-kernel
decision, and the probe-based validation loop. Implementation = mechanical ggml C
(2 new ops, copying the silu_back / rms_norm_back templates) + finite-difference
validation + a ggml rebuild — the next focused pass. NOT started in-tree: invasive
ggml-core edits + unvalidated numerics should not be rushed; the kernels must pass
finite-difference on the real compute path before any GPT-2 training is trusted.
