# Phase F1 status — backward-through-matmul resolved

**Date:** 2026-05-21
**F1.1 status:** plumbing **complete**; the "real-model LoRA layer"
                work moves to F1.2 (see "Next concrete step" below).

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

## What this phase actually shipped

| Item | Status |
|---|---|
| ggml backward op bindings (silu_back, rms_norm_back, rope_ext_back, gelu_back, soft_max_back) | F0 — shipped |
| `ggml_build_backward_expand` autograd | F0.4 — shipped |
| In-graph optimizer C primitives (build_backward / extend_backward_graph / realize_backward / graph_reset / opt_step_sgd / opt_step_adamw / sum / set_loss) | F1.0 + F1.1 — shipped |
| Minimal LoRA-shaped training step (forward + backward + opt_step in one compute) | F1.0 — shipped |
| Matmul-backward bisection (5 micros) + root-cause C POC | F1.1 — shipped |
| `tnn_build_forward_only` + acceptance: all 5 micros green | F1.1 — shipped |

## What F1.1 explicitly did NOT cover

The original F1.1 line was "LoRA on a real model layer". That phrasing
collapsed two pieces:

1. **Make the in-graph optimizer correct.** ← this is what F1.1 shipped.
2. **Actually inject LoRA adapters on (say) SmolLM2-135M's attention
   projections and train them on a tiny dataset.** ← this is real work
   (1-2 weeks per `docs/design/finetuning.md`), and properly lives in
   F1.2 (SFT loop driver + tiny dataset) now that F1.1's plumbing is
   green.

The split is honest: the optimizer mechanics and the SFT-loop ergonomics
are independent failure modes, and shipping them as one phase made the
bisection harder than it needed to be.

## Next concrete step (F1.2)

Per `docs/design/finetuning.md` Phase F1 ("LoRA on CPU first"):

- Pick the smallest live model: SmolLM2-135M (`data/smollm2-135m-*`).
- Add LoRA adapter matrices on the attention `q_proj` and `v_proj` of
  one layer first (typical LoRA scope). r=16 is the doc's default.
- Wire `ctx_w_lora` as a third persistent ctx (alongside `ctx_w`,
  `ctx`). Adapters + their Adam state live there.
- Build forward + backward + adamw_step in one `graph_b` via the
  primitives F1.1 just validated.
- Driver: `lib/lora_trainer.rb` (parallel to `lib/toy_trainer.rb`).
- Demo: `demos/sft_lora.rb` — fine-tune on a 100-example alpaca
  subset.
- Acceptance: loss decreases monotonically over 10 epochs.

CUDA is F2 (same graph, swap backend). Don't try them together.
