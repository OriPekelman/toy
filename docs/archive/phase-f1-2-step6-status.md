# F1.2 step 6 status — multi-example SFT bounded by architectural needs

**Date:** 2026-05-21
**Predecessor:** F1.2 step 5 (`demos/smollm2_lora_train_adamw_cuda`)

## Sub-step status

| Sub-step | Description | Status |
|---|---|---|
| 6a | Multi-target AdamW at same prefix (cycle N targets, one prefix) | ✅ DONE (`smollm2_lora_sft_multi_cuda`, 10.82 → 3.56 over 10 epochs) |
| 6b | Multi-position training (cycle N positions, separate training graphs) | ⚠️ Architectural prerequisite — see below |
| 6c | Varied-prefix examples (different input contexts) | ⚠️ Same prerequisite as 6b |
| 6d | Real alpaca jsonl + masked CE + held-out eval | Blocked by 6b/6c |

## What blocks 6b/6c/6d

Two architectural pieces are missing for real SFT:

### 1. Persistent Adam state

`demos/smollm2_lora_sft_multipos_cuda` (this commit) demonstrates the
mechanical need for it. The smoke rebuilds the training graph at each
position-switch (via `tnn_reset_for_rebuild` + `build_decode_step` +
`tnn_build_backward` + ...). The current cache class allocates Adam
`m`, `v` tensors in `ctx` (compute-side, non-persistent). When ctx
is freed during rebuild, m/v go with it. Cycle 1 trains correctly;
cycle 2's Adam state is gone → loss diverges to NaN immediately:

```
cycle 1: pos4 CE=7.519  pos5 CE=10.896
cycle 2: pos4 CE=NaN    pos5 CE=NaN
VERDICT: FAIL (NaN — Adam m/v lost across position-switch rebuild)
```

The smoke is preserved with a failing gate as a regression marker.

The fix is for `SmolLM2KVFFICache(Cuda)#realize_for_mmap` to grow an
`enable_lora_q_adam!(r)` variant that allocates per-LoRA-pair m/v
**in ctx_w** (persistent), parallel to how the LoRA-A/B weights are
already allocated. ~50 LOC per backend.

### 2. Sequence-mode forward graph (M3-shaped, task #69)

Even with persistent Adam, the per-step rebuild cost dominates
multi-position training. From the training bench
(`docs/design/bench-train-2026-05-21.md`), the per-step training
graph build + sched-alloc is ~6 ms (cheap) but does NOT amortize when
you switch positions every step — you pay the full cost every cycle.

Real SFT processes ALL positions of a sequence in ONE forward +
backward + opt_step. That requires a forward graph that:
- Takes `T` token IDs and `T` positions as inputs (no KV-cache views;
  full attention).
- Outputs `T` logit vectors.
- Loss aggregates CE across all `T` positions (masked for prompt
  positions in SFT).

This is essentially `FullForwardFFICache` (currently GPT-2 shape) but
with the llama-family ops (RMSNorm + SwiGLU + RoPE). Estimated 1-2
sessions to write + parity-check.

### Why neither is done in this session

The user's clear direction: "we want to optimize for time to joy"
(per the next-step plan). Both the persistent-Adam refactor and the
sequence-mode forward are 1+ session efforts each. Shipping ad-hoc
workarounds now creates churn against the eventual clean
implementation. Better to:

1. Ship **6a** (done, working) as the visible F1.2 outcome.
2. Mark **6b/6c/6d** as bounded follow-ups with clear preconditions.
3. Pivot to DevEx — quick-start examples, polish, and getting users
   running. Real SFT can come back when the architectural pieces are
   in place.

## Files this session

```
demos/smollm2_lora_sft_multi_cuda.rb         step 6a — PASSES
demos/smollm2_lora_sft_multipos_cuda.rb      step 6b — FAILS by design
                                              (regression marker)
docs/design/phase-f1-2-step6-status.md       this doc
```

## What's open after this commit

- Task #69 (M3 reusable decode graph) — different motivation
  originally; we now see it's load-bearing for full SFT too.
- New: persistent Adam state in `SmolLM2KVFFICache(Cuda)`.
- Task #72 stays open as the umbrella for 6b/c/d completion.
