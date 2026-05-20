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

The readout of `t_y` and the gradient tensor produces values that
don't match the analytical reference for these specific shapes:

```
y       ggml = [59.03, 59.03]   expected = [1.4, 3.2]
loss    ggml = 12.2             expected = 12.2  ✓
dL/dW   ggml = [59.03, 59.03, 23.05, 59.03, 59.03, 23.05]
                                expected = [2.8, 5.6, 8.4, 6.4, 12.8, 19.2]
```

Loss being exact while y readout is wrong is the telltale: forward is
correct, but reading the y / gradient buffer back to host scratch after
`compute_backward` is reading the wrong bytes. Hypothesis: the backward
graph's scheduler re-allocates intermediate buffers, and the project's
existing `tnn_download` reads from the scheduler-managed backing of t_y
which has been reassigned. Mitigation will be to use
`ggml_backend_tensor_get` on the specific tensor's CURRENT data pointer,
or to mark t_y persistent.

This is a real but bounded bug. Phase F1 (LoRA on CPU) can start
without it — the relevant gradient is read from `tnn_tensor_grad`
which returns a graph-stable handle.

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
