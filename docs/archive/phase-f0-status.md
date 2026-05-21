# Phase F0 status — fine-tuning groundwork

**Date:** 2026-05-20

## Done

| Sub-task | Status | Where |
| --- | --- | --- |
| F0.1 verify ggml backward ops + write C wrappers | ✅ | `tinynn/tinynn_ggml.c` |
| F0.2 bind backward ops through FFI | ✅ | `lib/tinynn.rb` + `lib/tinynn_cuda.rb` |
| F0.3 parity smoke tests | ✅ | `tinynn/ab_smoke_silu_back.rb`, `_rope_back.rb` |
| F0.4 ggml_build_backward_expand autograd smoke | ✅ pipeline; ⚠ readout | `tinynn/ab_smoke_autograd.rb` |

### What landed (F0.1-F0.3)

- **`tnn_silu_back(sess, x, dy)`** → ggml's `ggml_silu_back`. Header arg
  order is swapped vs implementation; we pass `(dy, x)` to match
  `ggml_vec_silu_backward_f32(n, dx, x, dy)` semantics in
  `src/ggml-cpu/ops.cpp`. Verified vs analytical `dy * sigmoid(x) *
  (1 + x * (1 - sigmoid(x)))` — max-abs-diff 5e-8.

- **`tnn_rope_ext_back(sess, dy, pos, n_dims, freq_base)`** → ggml's
  `ggml_rope_ext_back`. Smoke-tested: runs without abort; pos=0 case
  correctly returns dy unchanged.

- Existing bindings already covered: `tnn_rms_norm_back`, `tnn_softmax_back`.

### What landed (F0.4)

Five new C primitives wire up ggml's autograd:

| Function | Purpose |
| --- | --- |
| `tnn_sum(sess, a)` | Reduce all elements to a scalar — loss building block. |
| `tnn_set_loss(t)` | Mark a tensor as the training loss. |
| `tnn_build_backward(sess)` | Dup the forward graph with `force_grads=true` and call `ggml_build_backward_expand`. |
| `tnn_compute_backward(sess)` | Run the forward+backward graph as one scheduler compute call. |
| `tnn_tensor_grad(sess, t)` | Retrieve the gradient tensor for a param via `ggml_graph_get_grad`. |

Smoke (`tinynn/ab_smoke_autograd.rb`): toy graph
  `y = matmul(W, x); loss = sum(y * y)`
with W marked param, loss marked loss. Build + compute returns rc=0
for both `tnn_build_backward` and `tnn_compute_backward`. **Loss value
is exact** (12.2 = 1.4² + 3.2² for the chosen W and x). `tnn_tensor_grad`
returns a non-null tensor; ggml's autograd machinery is in motion.

### Outstanding (F0.4 follow-up)

After investigation: **leaves read back correctly** (W and x preserved
post-backward); **scalar loss reads back correctly** (12.2 — exact);
**multi-element compute nodes do not** (t_y reads back as [59.03,
59.03] instead of [1.4, 3.2]; W's gradient as [59.03, 59.03, 23.05,
59.03, 59.03, 23.05] instead of [2.8, 5.6, 8.4, 6.4, 12.8, 19.2]).

The pattern (repeated values, only 2 unique floats in the gradient)
suggests buffer overlap: ggml's scheduler reuses compute-node
backing storage aggressively, and `tnn_download(t_y)` after
`compute_backward` reads from whatever the sched left there.
`ggml_set_output(t_y)` flag (which we set) doesn't prevent reuse for
this graph shape.

**Tried + didn't help:**
- `ggml_backend_sched_alloc_graph` instead of `_reserve` for graph_b
- Reading W (a leaf — does work)
- Reading t_y after various invalidations

**What still needs investigation (≥1 day):**
- Either: copy each output-of-interest to a persistent tensor at end
  of graph (forces backing into ctx_w's stable buffer)
- Or: walk the scheduler's per-tensor backend pointer + read directly
  from there
- Or: build forward+backward+optimizer-step as one in-graph compute
  (mnist-example pattern) so we never need to read intermediates

For F1 work, option 3 is the right shape: keep gradients on-graph,
feed straight into `tnn_opt_step_adamw`. We never download
intermediates; only the scalar loss for logging. That sidesteps the
issue entirely.

## Gap analysis vs the fine-tuning design

`docs/design/finetuning.md`'s F0 acceptance was "5 bindings + autograd
smoke". We have **4 of 5 bindings** + the autograd infrastructure
landed. The 5th binding (`ggml_gelu_back`) doesn't exist upstream —
CPU-scratch fallback stays.

## Memory: ggml docstring discrepancy

`include/ggml.h`'s `ggml_silu_back` docstring claims `// a - x; b - dy`
but the implementation does the opposite: `src[0]=dy, src[1]=x` (see
`src/ggml-cpu/ops.cpp:2740-2745` and the `ggml_vec_silu_backward_f32(n,
dx, x, dy)` call at 2763). Worth filing upstream as a docs fix.
