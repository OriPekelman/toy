# Phase F1 status — backward-through-matmul resolved

**Date:** 2026-05-21

## Resolution

The `loss=0, grad=0, y=garbage` failure on micro2/3/5 was a sched
allocation issue: `tnn_realize` did `sched_alloc_graph(graph_a)` for
the forward pass; `tnn_realize_backward` then did
`sched_reset + sched_alloc_graph(graph_b)`. The reset released the
first alloc's buffer slots but left tensor `buffer` pointers stale —
the second alloc landed tensors on freed-pool memory.

Validated 2026-05-20 with a standalone C POC that links straight against
`vendor/ggml/build/src/libggml*.a` (no tinynn wrapper). Three flows side
by side:

| Flow | y | loss | grad(W) |
|---|---|---|---|
| `ggml_backend_alloc_ctx_tensors` (test-backend-ops / ggml-opt pattern) | 11 | 121 | [66, 88] |
| dual `sched_alloc_graph(gf)` then `(gb)` (our broken pattern) | 9 | 0 | [0, 0] |
| single `sched_alloc_graph(gb)` only (the fix) + `set_output(y)` | 11 | 121 | [66, 88] |

The middle row reproduces micro5's failure byte-for-byte; the bottom
row matches the canonical result.

## Fix

`tnn_build_forward_only(sess, result)` — same as `tnn_realize` minus the
sched_alloc. Training callers use this in place of `tnn_realize`; the
follow-up `tnn_realize_backward` then does the single sched_alloc on the
combined `graph_b` (forward + backward + optional opt_step). Inference
callers keep `tnn_realize` unchanged.

Why the canonical ggml code didn't trip this:

- `test-backend-ops.cpp` uses `ggml_backend_alloc_ctx_tensors` — one shot
  for everything, no sched, no slot reuse.
- `ggml-opt.cpp` pre-allocates grad_accs in a separate static ctx with
  its own `alloc_ctx_tensors`, then sched-allocs only the compute ctx.
  Two contexts → two independent buffers, no aliasing.

Our wrapper used a single sched with two sequential `alloc_graph` calls.
Neither canonical example takes that path, so the breakage was silent
(no asserts, just stale data).

## Micro-smoke results (post-fix)

| Smoke | Setup | Result |
|---|---|---|
| micro1 | scalar `W; loss = W²` (no matmul); SGD step | W 1.0 → 0.98, loss=1, grad=2 |
| micro2 | scalar W,x; `y = matmul(W, x); loss = y²`; SGD | W 1.0 → 0.92, loss=4, grad=8 |
| micro3 | K=2 W,x; `y = matmul(W, x); loss = y²`; SGD | W'=[0.934, 1.912], grad=[66, 88] |
| micro4 | Forward-only matmul, no backward | y=11, loss=121 |
| micro5 | Same shape as micro3, backward only (no opt) | y=11, loss=121, grad=[66, 88] |

All five pass.

## What unblocks F2 / F3 / F4

- F2 (LoRA on CUDA): same graph, swap `TinyNN` → `TinyNNCuda`. The fix
  applies identically — `tnn_build_forward_only` is backend-agnostic.
- F3 (full fine-tune): same pattern, more params.
- F4 (QLoRA): same pattern, Q8 base + F32 LoRA.

The in-graph optimizer infrastructure
(`build_backward + extend_backward_graph + realize_backward + graph_reset
+ compute_backward`) is now load-bearing and validated end-to-end on CPU.

## Footgun for future callers

When you want to read an intermediate forward tensor (e.g. `y`) after
`compute_backward`, mark it with `tnn_set_output` BEFORE
`tnn_build_forward_only`. Without it, the sched is free to alias `y`'s
slot with a backward intermediate, and the readback returns garbage.
`set_output(loss)` is already needed for the loss tensor for the same
reason.
