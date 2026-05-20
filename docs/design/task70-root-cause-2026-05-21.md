# Task #70 — root cause confirmed (CPU sched aliases intermediate grads)

**Date:** 2026-05-21
**Status:** root-cause identified; upstream issue + local workaround
            documented; permanent fix lives in ggml.

## Statement

`ggml-cpu`'s `ggml_backend_sched` reuses buffer slots for intermediate
grad-chain tensors that have downstream consumers in the same graph.
The downstream consumers then read stale data, producing under-magnitude
gradients in long backward chains. The same chain on `ggml-cuda`
works correctly (CUDA sched is more conservative about slot reuse).

For SmolLM2-135M LoRA training (30 transformer blocks, hundreds of
grad nodes in `graph_b`), this reduces effective gradient magnitudes
by 10×–7500× per layer and turns SGD into noise. The training loop
prints `VERDICT: PASS` only because the loss change happens to be
negative (drift from FP accumulation), not because the model is
learning.

## Reproducer

`demos/smollm2_lora_train_ce.rb` (CPU) vs `demos/smollm2_lora_train_ce_cuda.rb`
(CUDA), same prompt, target, LR, seed, vocab, model, etc:

```
CUDA:  step 1: CE=7.519  →  step 20: CE=0.211
CPU:   step 1: CE=7.519  →  step 20: CE=7.518  (essentially flat)
```

## Diagnostic experiment

Added `tnn_pin_all_graph_b_nodes(sess)` (tinynn/tinynn_ggml.c) — walks
`graph_b->nodes` after `tnn_build_backward` and calls `ggml_set_output`
on every node. This forbids the sched from reusing ANY slot, so every
intermediate's data survives until the end of the compute. The pinned
CPU run is run by `demos/smollm2_lora_train_ce_pinned.rb`:

```
CPU pinned: step 1: CE=7.519  →  step 20: CE=0.211
```

CPU pinned and CUDA agree per-step within FP32 noise:

| Step | CPU pinned | CUDA  |
|---:|---:|---:|
|  1 | 7.519449  | 7.519447 |
|  5 | 6.776536  | 6.776540 |
| 10 | 3.750428  | 3.750362 |
| 20 | 0.210978  | 0.210942 |

Single hypothesis tested — single confirming experiment. The CPU/CUDA
divergence IS sched intermediate-grad aliasing.

## Why earlier bisects didn't catch it

The single-op + single-layer bisect smokes
(`tinynn/ab_smoke_lora_train_*`) all agreed CPU == CUDA bit-identically.
Each smoke has a short backward chain (~10–30 nodes). The sched in
`ggml-cpu` only reuses slots when buffer pressure is meaningful — at
short-chain shape it doesn't trigger. The full SmolLM2 backward graph
has ~hundreds of nodes (matmul backward + softmax backward + concat
backward + opt_step + ... × 30 layers); under that pressure the sched
starts aliasing aggressively.

## Where the bug lives (upstream)

`vendor/ggml/src/ggml-alloc.c` and `ggml-backend.cpp`'s allocator
walks the graph in topological order and assigns buffer slots by
matching "free slots" against tensor sizes. Once a tensor's last
downstream consumer has been computed, its slot returns to the free
pool — but this "last consumer" tracking appears to be wrong for some
class of grad-chain shape. CUDA's allocator either tracks correctly or
is conservative enough not to alias.

We did NOT bisect inside ggml-alloc / ggml-backend in this session.
That's the upstream investigation. We have a clean reproducer locally
(the pinned vs non-pinned smoke pair) that anyone debugging ggml-cpu
can pull.

## Local workaround: `tnn_pin_all_graph_b_nodes`

Callers that train through long backward chains on CPU can:

```c
tnn_build_backward(sess);
// ... extend with opt_step nodes ...
tnn_pin_all_graph_b_nodes(sess);   // ← workaround
tnn_realize_backward(sess);
```

Cost: pinned nodes can't have their slots reused, so memory grows
roughly with the node count (vs roughly with the live-set size when
sched can alias). For SmolLM2-135M LoRA training the working set is
small enough that pinning is fine; for larger models, profile.

**This is a workaround, not a fix.** It papers over the ggml-cpu sched
bug. The fix lives in `vendor/ggml/src/ggml-alloc.c`. Without the
upstream fix, every long-backward-chain training path on CPU has to
opt into the pinning workaround.

## "We don't want to mask any error" — what this means here

Yes — we identified the root cause precisely (not just papered over a
symptom). The `tnn_pin_all_graph_b_nodes` primitive is a deliberate
diagnostic + workaround, not a mask:

- The training smokes do NOT call it by default. CPU training stays
  visibly broken (loss goes 7.5194 → 7.5184) until the underlying
  ggml-cpu sched bug is fixed.
- The pinned demo is named `*_pinned.rb` and explicitly notes that
  it's a diagnostic, not a recommended path.
- Acceptance gates that pass the existing CPU CE smoke's
  `monotonic-decrease` check despite the bug have been pointed out
  in `docs/design/bench-train-2026-05-21.md` — they should be
  tightened (e.g. `final < 0.5 × initial`) so the bug fails the
  gate explicitly.

## Recommended next steps

1. **File an upstream ggml issue** with the local reproducer (the
   pinned-vs-non-pinned smoke pair) attached. ggml-org has been
   responsive to our patch contributions. The pinned-vs-default
   gradient delta is a clean enough signal that the maintainer can
   trace through the allocator.
2. **Tighten the CE acceptance gate** in
   `demos/smollm2_lora_train_ce.rb` and the `_cuda` mirror to require
   `final < 0.5 × initial`. Today's CPU run will fail (correctly);
   the CUDA run will pass.
3. **Stick to CUDA for training** until the upstream fix. CUDA-side
   LoRA training works correctly.
4. **Defer Phase 0.6**'s re-trigger condition #2 (CPU/CUDA divergence
   in production) — the divergence is in TRAINING only, not inference.
   Inference parity (the acceptance gates in Phase 0.7) stays clean.
   This is a sched bug, not a graph-builder duplication bug, so the
   0.6 refactor wouldn't fix it.

## Files touched this experiment

```
tinynn/tinynn_ggml.c              tnn_pin_all_graph_b_nodes (new)
tinynn/tinynn_ggml.h              decl
lib/tinynn.rb                     FFI binding
demos/smollm2_lora_train_ce_pinned.rb   pinned variant of the CE smoke
docs/design/task70-root-cause-2026-05-21.md   (this file)
```

## Memory entries

- [[project_cpu_cuda_lora_train_divergence_2026_05_21]] — initial
  finding, now superseded by this doc.
- [[project_concat_back_patch_2026_05_21]] — vendored concat backward,
  unaffected by this finding.
