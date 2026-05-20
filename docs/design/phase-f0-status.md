# Phase F0 status — fine-tuning groundwork

**Date:** 2026-05-20

## Done

| Sub-task | Status | Where |
| --- | --- | --- |
| F0.1 verify ggml backward ops + write C wrappers | ✅ | `tinynn/tinynn_ggml.c` |
| F0.2 bind backward ops through FFI | ✅ | `lib/tinynn.rb` + `lib/tinynn_cuda.rb` |
| F0.3 parity smoke tests | ✅ | `tinynn/ab_smoke_silu_back.rb`, `_rope_back.rb` |
| F0.4 ggml_build_backward_expand autograd smoke | ⏸ deferred | next session |

### What landed

- **`tnn_silu_back(sess, x, dy)`** → ggml's `ggml_silu_back`. Header arg
  order is swapped vs implementation; we pass `(dy, x)` to match
  `ggml_vec_silu_backward_f32(n, dx, x, dy)` semantics in
  `src/ggml-cpu/ops.cpp`. Verified vs analytical `dy * sigmoid(x) *
  (1 + x * (1 - sigmoid(x)))` — max-abs-diff 5e-8 on 8-element test.

- **`tnn_rope_ext_back(sess, dy, pos, n_dims, freq_base)`** → ggml's
  `ggml_rope_ext_back`. Same YaRN defaults as `tnn_rope_ext` for
  consistency. Smoke-tested: produces finite output for sane inputs;
  pos=0 case correctly returns dy unchanged (no rotation at first
  position).

- Existing bindings already covered:
  - `tnn_rms_norm_back` — `ggml_rms_norm_back`
  - `tnn_softmax_back` — `ggml_soft_max_ext_back`
  - `tnn_gelu_back_scratch` — CPU-only scratch implementation (no
    matching CUDA kernel; ggml doesn't ship `ggml_gelu_back` either).

### Gap analysis vs the fine-tuning design

`docs/design/finetuning.md`'s F0 acceptance was "5 bindings + autograd
smoke". We have 4 of the 5 bindings (silu, rope, rms_norm, softmax)
in place; the 5th was `ggml_gelu_back` which doesn't exist upstream.
GeLU backward stays on the CPU-scratch fallback.

## Why F0.4 is deferred

`ggml_build_backward_expand(ctx, cgraph, grad_accs)` needs:

- A `ggml_cgraph *` handle exposed to Ruby. Today `tnn_session` hides
  the cgraph behind `tnn_realize` + `tnn_compute`. Need new primitives:
  `tnn_session_get_cgraph(sess)` and a helper that walks the graph
  marking params + grad accumulators.
- A way to pass `grad_accs` as an array of tensor handles — Ruby-FFI
  array-of-pointer marshaling. Spinel supports `:int_array` but
  `Array<:ptr>` is `matz/spinel#492` (open) — workaround possible
  via fixed-size struct.
- A backward-graph compute path. The forward+backward graph runs as
  one `ggml_graph_compute` call after expand. The session's existing
  realize+compute is forward-only; we'd extend or add a parallel
  path.

Each of these is bounded but non-trivial; together they're a 2-3 day
focused session. Cleanest scope: add a small `lib/lora_trainer.rb` (or
`tnn_autograd.c` helper) that owns the cgraph for a single linear
layer + loss, drives backward through it, reports gradient via a
known-good analytical reference.

Once F0.4 lands, F1 (LoRA on CPU) and F2-F4 are mechanical.

## Memory: ggml header docstring discrepancy

`include/ggml.h`'s `ggml_silu_back` docstring claims `// a - x; b - dy`
but the implementation does the opposite: `src[0]=dy, src[1]=x` (see
`src/ggml-cpu/ops.cpp:2740-2745` and the `ggml_vec_silu_backward_f32(n,
dx, x, dy)` call at 2763). Worth filing upstream as a docs fix — minor
but anyone trusting the comment ships swapped gradients.
