# Vendored ggml patch — `GGML_OP_CONCAT` backward

**Date:** 2026-05-21
**File:** `vendor/ggml/src/ggml.c` (`ggml_compute_backward`)
**Status:** local-only; not yet upstreamed.

## What

Added a `case GGML_OP_CONCAT` arm to `ggml_compute_backward`. The
backward of `concat(a, b, dim)` is two views of the upstream gradient
sliced along `dim`:

```c
grad_a = ggml_view_4d(ctx, grad,
                      a->ne[0], a->ne[1], a->ne[2], a->ne[3],
                      grad->nb[1], grad->nb[2], grad->nb[3],
                      /*offset=*/ 0)
grad_b = ggml_view_4d(ctx, grad,
                      b->ne[0], b->ne[1], b->ne[2], b->ne[3],
                      grad->nb[1], grad->nb[2], grad->nb[3],
                      /*offset=*/ a->ne[dim] * grad->nb[dim])
```

Each view is fed through `ggml_add_or_set` so it composes with any
other grad path that already touches the source.

## Why

`SmolLM2KVFFICache#build_decode_step` concats per-Q-head attention
outputs along dim 0 before the O projection:

```ruby
t_concat = t_head_outs[0]
hq = 1
while hq < @n_heads
  t_concat = TinyNN.tnn_concat(@sess, t_concat, t_head_outs[hq], 0)
  hq = hq + 1
end
t_out_proj = TinyNN.tnn_matmul(@sess, blk.t_w_o, t_concat)
```

LoRA on Q (F1.2) puts the trainable adapter in `t_head_outs[hq]`'s
upstream. Backward must therefore reach back through the CONCAT to
the LoRA params. Without this case, `ggml_compute_backward` hits the
`default: GGML_ABORT(...)` arm and the program aborts during
`ggml_build_backward_expand`.

## Validation

Standalone POC at `~/tmp/f1_2_concat_bw_poc.c` — A=(2,1) and B=(2,1)
both params, `C = concat(A, B, dim=0)`, `loss = sum(C * C)`. Expected
hand-computed values:

| Item | Expected | Observed |
|---|---|---|
| loss   | 30          | 30 ✓ |
| C      | [1, 2, 3, 4]| [1, 2, 3, 4] ✓ |
| ∂L/∂A  | [2, 4]      | [2, 4] ✓ |
| ∂L/∂B  | [6, 8]      | [6, 8] ✓ |

End-to-end validation in `demos/smollm2_lora_train_step` — backward
through the full SmolLM2-135M decode graph reaches layer-0 LoRA params
without NaN, and the params actually move under SGD.

## Upstream-ability

The patch is small, mechanical, and follows existing patterns in the
file (compare with `GGML_OP_GET_ROWS` and `GGML_OP_VIEW`). When the
F1 / F2 / F3 work stabilises we should propose this to ggml upstream
as a PR; until then we maintain it as a vendored delta. If `make
setup-ggml` ever re-fetches the vendor dir, re-apply this patch.

For multi-dim concat the `offset = a->ne[dim] * grad->nb[dim]`
formula generalises to all four dims correctly because nb[d] is the
parent's byte stride along d (concat output is contiguous, so nb[d]
of the view equals nb[d] of the parent). For dim != 0 the view's
nb1/nb2/nb3 still come from the parent — they describe the parent's
strides, which the view shares.

## Sibling patches in this vendor dir

- `project_ggml_cpy_patch_2026_05_18` — 6-line `cpy.cu` fix for the
  strided-dst path on CUDA.
- `project_phase2_cuda_byo_2026_05_18` — `ggml_backend_cuda_buffer_from_ptr`
  for BYO-pointer mmap.

Both also live as local vendor diffs awaiting an upstream conversation.
