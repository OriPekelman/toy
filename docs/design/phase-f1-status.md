# Phase F1 status — in-graph optimizer landed; values need investigation

**Date:** 2026-05-20

## Done in this session

### Architecture (no tech debt)

Refactored autograd flow to clean phases so caller can insert
optimizer nodes between graph build and graph alloc:

```
tnn_realize(sess, loss)             # forward graph
tnn_build_backward(sess)            # adds backward nodes; NO alloc
                                    # → caller extends here ↓
for each param:
  opt = tnn_opt_step_adamw(sess, p, grad, m, v, hp)   # or _sgd
  tnn_extend_backward_graph(sess, opt)
tnn_realize_backward(sess)          # sched-alloc the FINAL graph
tnn_graph_reset(sess)               # zero grads + momenta; loss_grad = 1
loop:
  tnn_compute_backward(sess)        # fwd + bwd + opt in one call
  read scalar loss; repeat
```

Five new C primitives + FFI bindings:
- `tnn_sum(sess, a)` — scalar reduce (loss building block).
- `tnn_set_loss(t)` — flags a tensor as the training loss.
- `tnn_build_backward(sess)` — dups graph + `ggml_build_backward_expand`.
- `tnn_extend_backward_graph(sess, node)` — `ggml_build_forward_expand` for opt-step nodes.
- `tnn_realize_backward(sess)` — sched alloc.
- `tnn_graph_reset(sess)` — `ggml_graph_reset` (zeros grads, momenta).
- `tnn_tensor_grad(sess, t)` — `ggml_graph_get_grad`.
- `tnn_opt_step_adamw(...)` — already had this; bound the AdamW arg layout.
- `tnn_opt_step_sgd(...)` — added for sanity-checking gradient direction.

This is the **right shape** for fine-tuning: no host readback of
intermediates, no per-step graph rebuild, no LD/ST overhead between
forward, backward, and optimizer.

### Smokes

- `tinynn/ab_smoke_train_step.rb` — Adam in-graph training step.
- `tinynn/ab_smoke_train_sgd.rb` — SGD variant for gradient-direction
  sanity check.

Both compile + run + show W getting updated by the optimizer.

## What needs investigation

Loss + gradient *magnitudes* from the in-graph optimizer are off vs
analytical expectation by ~100×. Concrete observation:

```
Toy: y = W @ x; loss = sum(y²); W=[-0.2, 0.3, 0.5, -0.1, 0.4, 0.1]; x=[1,2,3]
Hand:    y = [1.9, 1.0]; loss = 4.61
         ∂L/∂W[0,0] = 2 * y[0] * x[0] = 3.8

ggml SGD step 1 (lr=0.01): W[0] went from -0.2 to -6.99.
  Δ = -6.79  ⇒  lr * grad = 6.79  ⇒  grad ~ 679 (not 3.8).
```

Two threads interact:

1. **F0.4 readback issue** (still open) — `tnn_download` of compute-node
   outputs returns wrong bytes. Loss reads back as 679.34, gradient also
   reads back as 679.34. Suspicious that they're equal — sched may be
   pointing both downloads at the same buffer.

2. **Possible ggml autograd shape misunderstanding** — even if readback
   is broken, the W change of -6.79 per step is REAL (we observe W's
   internal value moving), so the internal gradient really is ~679,
   not 3.8. This suggests autograd is computing the gradient of
   something other than what I marked as `loss`, OR the `tnn_sum`
   I'm using doesn't reduce all the way to a scalar.

## Next concrete debugging step

Build a fully analytical micro-smoke:

- Single param: 1-element W = 1.0
- Loss = W² (no matmul, no sum — just a square)
- Mark W param, mark loss = loss tensor
- Run one SGD step with lr=0.01
- Hand: ∂(W²)/∂W = 2W = 2. SGD: W' = 1 - 0.01*2 = 0.98.

If we see W' = 0.98 → ggml autograd works correctly at the simplest
shape; the bug is in our sum/matmul gradient handling. If we see
something else → ggml or our wrapper has a shape-handling bug at the
absolute basement.

This bisects perfectly. ~1 day to get the right answer.

## What unblocks F2 / F3 / F4

Once the F1 micro-smoke is correct, F2 (LoRA on CUDA) is the same
graph + `TinyNNCuda` swap. F3 (full FT) adds Adam state for all
weights. F4 (QLoRA) layers Q8 base.

The in-graph optimizer architecture from today is the **load-bearing**
piece — it removes the per-step graph rebuild that would have made
training at scale prohibitive. The remaining work is calibration +
real-model integration.

## Honest assessment

F1.1 is "infrastructure shipped, calibration TODO". Not the
"loss-decreasing on a real model layer" we'd hoped for, but the
right architecture is in place — once the value-mismatch is found,
everything downstream of it works.
