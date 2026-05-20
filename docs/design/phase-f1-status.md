# Phase F1 status — autograd bisected to backward-through-matmul

**Date:** 2026-05-21

## Bisection results

Five micro-smokes (`tinynn/ab_smoke_train_micro{1..5}.rb`) bracket the
issue:

| Smoke | Setup | Result |
|---|---|---|
| micro1 | scalar W; loss = W² (no matmul); SGD step | ✅ exact: W 1.0 → 0.98, loss=1, grad=2 |
| micro2 | W,x both 1-element; y = matmul(W, x); loss = y²; SGD | ✗ W unchanged; loss=0; grad=0 |
| micro3 | W,x both 2-element; y = matmul(W, x); loss = y²; SGD | ✗ W unchanged; loss=0; grad=0 (readback shows y=162382, garbage) |
| micro4 | Forward-only matmul; no backward | ✅ exact: y=11, loss=121 |
| micro5 | Same as micro3 but no optimizer (backward only) | ✗ loss=0, grad=0, y garbage |

The pattern:

- **Forward-only with matmul works** (micro4).
- **Backward without matmul works** (micro1).
- **Backward with matmul does NOT work** (micro2/3/5).

So:
- ggml autograd is functional at the basement.
- ggml matmul forward is functional.
- The **chained backward through matmul → mul** is the broken thing.

This is independent of:
- The optimizer (broken with or without opt_step_sgd).
- Output flags on intermediate tensors.
- Order of upload vs graph_reset.
- Shape (broken at K=1, K=2, K=3).

## What we know NOT to be the cause

- Not the optimizer node setup (micro5 has no optimizer; still broken).
- Not the readback path (micro1 reads everything correctly).
- Not graph_reset timing (tried before AND after uploads).

## What's most likely

- Some interaction between `ggml_build_backward_expand` and matmul
  nodes in particular. Maybe ggml requires `ggml_set_input` on `t_x`
  or a particular flag setup the autograd needs.
- Or my `tnn_realize_backward`'s sched setup doesn't allocate enough
  for the backward-through-matmul intermediate gradient tensors.
- Or the sched needs `ggml_backend_sched_reserve` first, then a real
  alloc — the order matters.

## Recommended next concrete step

Look at `vendor/ggml/tests/test-backend-ops.cpp` for the existing test
of `ggml_mul_mat` backward. They exercise the exact path. Compare:
- Their cgraph build sequence
- Their flag setup (set_input/set_output/set_param)
- Their sched setup
- Their compute path

If our setup diverges, copy theirs. If our setup matches but values
diverge, file as a ggml issue.

This is **bounded ~half a day** of focused work.

## What unblocks F2 / F3 / F4

Once F1's matmul-backward chain is fixed:
- F2 (LoRA on CUDA): same graph, swap `TinyNN` → `TinyNNCuda`
- F3 (full FT): same pattern, more params
- F4 (QLoRA): same pattern, Q8 base

The in-graph optimizer infrastructure from earlier today is
load-bearing and validated. The blocker is purely the autograd-
through-matmul behavior, not the architecture.

## What landed this session

Five micro-smokes preserved as `tinynn/ab_smoke_train_micro{1..5}.rb`
— each reproduces a specific point in the bisection. They:
- Are self-contained (no project deps beyond lib/tinynn).
- Compile cleanly under fresh Spinel (no pollution from the F1 work).
- Run in <1s each.
- Make the bisection trivially repeatable.

The next-session opener has the file paths + test inputs ready.
