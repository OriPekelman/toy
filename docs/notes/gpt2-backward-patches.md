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

**IMPLEMENTED + VALIDATED (2026-06-04).** Both kernels are vendored and pass a
finite-difference gradient check on the real backend-sched compute path.

Patch points landed (vendor/ggml):
- **GGML_OP_GELU_BACK** and **GGML_OP_NORM_BACK** appended to the `ggml_op` enum
  (`include/ggml.h`) → `GGML_OP_COUNT` 96 → 98; both name + symbol tables and both
  `static_assert`s updated.
- `ggml.h` decls `ggml_gelu_back(ctx, a=dy, b=x)` / `ggml_norm_back(ctx, a=x, b=dy, eps)`.
- `ggml.c` constructors + autograd dispatch: `case GGML_UNARY_OP_GELU` →
  `ggml_gelu_back(ctx, grad, src0)`; `case GGML_OP_NORM` →
  `ggml_norm_back(ctx, src0, grad, eps)`.
- CPU kernels in `ggml-cpu/ops.cpp` (`..._gelu_back_f32`, `..._norm_back_f32`),
  declared in `ops.h`, dispatched in both `ggml-cpu.c` switches (compute + n_tasks).
  GELU derivative helper `ggml_vec_gelu_backward_f32` in `ggml-cpu/vec.h`.
- `ggml-alloc.c` (can-inplace) + `ggml-backend-meta.cpp` (split-state: GELU_BACK
  generic, NORM_BACK per-row) updated. Other backends' `supports_op` default to
  false for the two new ops, so they correctly fall back to CPU.

Validation: `tinynn/gpt2_backward_probe.c` builds `loss = sum(op(x) .* r)` with a
non-trivial upstream grad, runs `ggml_build_backward_expand` + the CPU
`ggml_backend_sched` compute path (the #1491-sensitive path, NOT
`compute_with_ctx`), and compares the autograd `grad_x` to a **pure-C** central
finite difference of the *exact* functions (numpy-allclose, atol 1e-3 / rtol 2e-2).
Both PASS at the f32 fd rounding floor (gelu max_abs 5.2e-5, norm 2.4e-4).
NB: ggml's gelu *forward* uses an f16 lookup table (`GGML_GELU_FP16`), so the
reference must difference the exact tanh gelu — not the graph forward — or the
f16 buckets dominate. The kernel is the analytic derivative of the exact tanh gelu.

No explicit tinynn binding is needed for training: `build_backward_expand` emits
the kernels automatically for any graph containing `gelu`/`norm`. (Thin
`tnn_gelu_back`/`tnn_layer_norm_back` wrappers would only be for manual calls.)

## Part b (the arch) — IN PROGRESS

**Step 1 DONE (2026-06-04): minimal inline GPT-2 trainer — kernels train end-to-end.**
`prep/gpt2_train_min.rb` (`make gate-gpt2-min`) is a self-contained
forward+CE+backward+AdamW loop over the GPT-2-distinctive structure (wte+wpe
learned embeddings, composite `tnn_layer_norm` = norm+mul γ+add β, GELU FFN, tied
output) — attention OMITTED in this first proof. CE drops 3.47 → 0.007 on a
memorizable synthetic sequence. This is the "record-from-inline-first" reference
and the proof that `gelu_back` (GELU FFN) + `norm_back` (LayerNorm) train through
the real ggml/FFI stack, not just the finite-diff probe.

**Key integration finding:** the engine's `build_training_step` is
**forward-agnostic** — it consumes `@t_seq_logits` + the registered
`@ft_globals_{weights,m,v}` triples and emits CE + backward + `opt_step_adamw`
over them. So a GPT-2 forward only needs to (1) register its weights as globals
and (2) set the logits; the backward (incl. `gelu_back`/`norm_back`) is automatic
via `tnn_build_backward`. The inline trainer replicates this directly.

**Realize ordering (load-bearing):** alloc all ctx_w weights → `tnn_set_param`
each → `tnn_finalize_weights` (allocates the weight buffer) → upload weight inits
+ `tnn_zero_tensor` the Adam m/v → build forward+CE+backward+`opt_step_adamw` →
`tnn_realize_backward` → train loop. Uploading a persistent weight BEFORE
`tnn_finalize_weights` aborts with "tensor buffer not set".

**Step 2 DONE (2026-06-04): full minimal GPT-2 block trains.** Added single-head
causal self-attention (qkv biases, `tnn_diag_mask_inf` causal mask, `tnn_softmax`,
transpose/`cont_2d` for V); the pre-LN block is ln1→attn→res→ln2→GELU-FFN→res. CE
drops 3.46 → 0.0077 — every GPT-2-distinctive op now trains end-to-end.

**ggml training gotcha (carry to the full arch):** `transpose(v)`'s backward yields
a NON-contiguous gradient that `repeat_back` (the bias-broadcast backward) rejects
(`ops.cpp` `GGML_ASSERT(nb00 == sizeof(float))`). Since softmax rows sum to 1,
`Σ_k probs·(v+b_v) = Σ_k probs·v + b_v`, so the V bias is added to the attention
OUTPUT, not before the transpose — exact, and keeps the grad contiguous. Q/K
biases are fine (their grads come from matmul, already contiguous). When porting
to multi-head, the per-head reshape/permute will need the same care.

**Step 3 DONE: byte-exact gate.** `prep/gpt2_train_gate.rb` (`make gate-gpt2`)
pins the CE curve to `prep/fixtures/gpt2_train_baseline.txt`; deterministic
(seeded LCG + fixed data), ggml-internal CE byte-exact on aarch64.

**Step 4 DONE: multi-head.** Per-head weights + `tnn_concat` (the engine's
per-head-loop pattern — NOT reshape/permute, so no segfault juggling). `N_HEADS`
default 4; `N_HEADS=1` reproduces the single-head curve byte-for-byte (correctness
check). Gate re-recorded at the multi-head default.

**Step 5 DONE (2026-06-04): `toy train --arch gpt2` ships.** `GPT2SeqEngine` +
`lib/toy/run/train_gpt2.rb` → `libexec/toy-train-gpt2`, wired into the CLI
(`toy train from-scratch --arch gpt2`, CPU/from-scratch this slice), gated
byte-exact + decreasing by `make gate-gpt2-train` (loss 6.44 → 5.46). The
multi-hour "Spinel poly-degradation blocker" turned out to be a **require-path
bug** (`"../toy"` vs `"../../toy"` → `TinyNN`/`Mat` never loaded → emit-0 cascade
→ CE=0); `spinel --emit-types` surfaced the ignored require. Full post-mortem:
[`gpt2-engine-spinel-blocker.md`](gpt2-engine-spinel-blocker.md). The original
WIP commit notes below are kept for history but the blocker is RESOLVED:
- `lib/toy/llm/engine/gpt2_seq_engine.rb` — `GPT2SeqEngine` (SEPARATE from
  `LlamaSeqEngine` → protects the Llama gates; a separate binary anyway per
  landmine #16). `realize!` builds the proven forward+CE+backward+AdamW; `step!`
  drives one step.
- `lib/toy/run/train_gpt2.rb` → `libexec/toy-train-gpt2` (Makefile target added).

**BLOCKER — Spinel poly-degradation (the toy#32 class).** The engine COMPILES and
TRAINS at small dims (`prep/gpt2_engine_smoke.rb`: VOCAB=32 → 72 weights, CE
3.46→0.96), but at realistic dims (VOCAB=627) the engine's `@g_weights`
Array<:ptr> and the label Mat degrade to EMPTY/ZERO in this compilation unit →
`compute_backward` runs but CE=0, no training. **SIZE-dependent, not
env-vs-literal** (literal VOCAB=627 fails identically; `realize_backward`/
`compute_backward` both return success — the arrays are silently emptied). The
llama engine works at 627 because its unit constrains Mat/Array types differently
(it uploads init via flat `tnn_upload_from_float_array`, not many diverse-dim Ruby
Mats). **This is a concrete real-world instance of exactly what toy#32's
spinel-doctor gate is meant to catch** — a numerical path silently emitting 0.

**Investigated with `spinel doctor` (2026-06-04) — full diagnosis + spinel-dev
tool proposals in [`gpt2-engine-spinel-blocker.md`](gpt2-engine-spinel-blocker.md).**
Findings: the degradation is **restructure-resistant** — fix hypothesis (a)
(flat `Array<Float>` + `tnn_upload_from_float_array`, no `Mat`) *fixes* the
`@g_weights` array at VOCAB=627 but then the `tnn_upload_from_float_array` FFI
calls themselves degrade to emit-0 at ALL dims (logits=0, CE=0). Each
restructure trades one poly-degradation for another. The doctor localizes the
symptom (`X on int (emitting 0)`) but not the root, and `--emit-rbs` is
misleadingly clean. **Most promising next step (sidestep, not fight):** a C-side
`tnn_fill_uniform(tensor, n, scale, seed)` so the engine unit never builds large
Ruby arrays. The byte-exact INLINE trainer (`make gate-gpt2`) stays the working
reference + the curve the engine path must reproduce once unblocked.

**Then:** CUDA/Metal mirrors after the CPU engine path is gated.
