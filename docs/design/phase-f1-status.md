# Phase F1-F4 status — fine-tuning rollout

**Date:** 2026-05-20

## Where we are

| Sub-task | Status | Where |
| --- | --- | --- |
| F0 (groundwork) | ✅ shipped | `docs/design/phase-f0-status.md` |
| F1.0 (LoRA algorithm smoke) | ✅ shipped | `demos/lora_smoke.rb` |
| F1.1 (LoRA via FFI, scale) | next | |
| F1.2 (SFT loop + real dataset) | queued | |
| F2 (LoRA on CUDA) | queued | |
| F3 (full fine-tune) | queued | |
| F4 (QLoRA) | queued | |

## F1.0 shipped

`demos/lora_smoke.rb` — toy regression to zero output via LoRA on a
single Linear layer. W_base is frozen; A (R×K) and B (OUT×R) are
trainable. Hand-coded backward (sidesteps F0.4's tensor-readback
TODO):

  ∂L/∂y = 2y
  ∂L/∂A = Bᵀ · (∂L/∂y) · xᵀ
  ∂L/∂B = (∂L/∂y) · (A · x)ᵀ

Result: loss **1.124 → 4.3e-8** in 30 SGD steps. Eight orders of
magnitude. Algorithm proven; shape proven; the rest is plumbing.

### Spinel discoveries during F1.0

- `Mat` as multi-return value (`return [a, b]; x, y = fn()`) segvs.
  Worked around by exposing each return as a separate function.
  Worth filing alongside the other Spinel issues we've collected.

## F1.1 — LoRA via FFI ops, real model layer

Scope: instead of Mat math, run the LoRA forward + backward through
ggml ops via the existing tinynn FFI. Required for performance at
real model scale (matmul of ~1500×1500 weights, not 4×3).

Two flavors possible:

1. **Per-step session rebuild** — build a fresh forward+backward
   graph each step, stage A/B values into it. Simple but each step
   pays graph-build cost (~ms). Fine for small models (SmolLM2-135M).
2. **Persistent training graph** — build forward+backward+adam_step
   once (mnist-example pattern), run compute repeatedly. Required for
   the 7B case where graph-build amortizes badly.

F1.1 ships flavor 1 first (proves the FFI path); flavor 2 lands as
F1.1b / F1.2.

### F1.1 work items (estimated)

| Item | Effort |
| --- | --- |
| `LoraAdapter` Ruby class holding A, B, gradient buffers, Adam state | 1d |
| FFI graph builder: forward through one Linear with adapter injection | 1d |
| Hand-coded backward via FFI ops (matmul, scale) | 1d |
| Adam optimizer step (we have `tnn_opt_step_adamw` already) | 0.5d |
| Smoke on a single Linear from SmolLM2-135M's first attn layer | 0.5d |

**Estimate: ~4 days dedicated work** to land F1.1.

## F1.2 — SFT loop driver + tiny dataset

After F1.1 lands, the SFT loop is mechanical:

- Read prep'd `data/sft_train.bin` (format defined in finetuning.md)
- Per batch: tokenize → forward → masked CE loss → backward → adam → repeat
- Eval every N steps + checkpoint LoRA weights

The HARDEST part is loss masking — cross_entropy_grad already supports
`target=-100` skip; we just need to plumb it through.

**Estimate: 1-2 weeks** (per finetuning.md design).

## F2 — LoRA on CUDA

Mirror F1's CPU driver to TinyNNCuda. Mostly mechanical given:
- `lib/transformer_lm_cuda.rb` pattern already shows how (per
  `docs/design/arch-struct.md`'s "two delegating wrappers" notes).
- All ops we use have CUDA bindings.
- The toy `lora_smoke` could be ported as `lora_smoke_cuda` once
  the math is FFI-side.

**Estimate: 3-5 days after F1.1 ships.**

## F3 — full fine-tune on CUDA

All weights trainable. Adam state context. Memory math (from
`docs/design/finetuning.md`):

| Model | Weights | + grad | + Adam | Total | GB10 fit |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | 2 GB | +2 | +4 | ~8 GB | ✓ |
| Qwen2.5-1.5B | 6 GB | +6 | +12 | ~24 GB | ✓ |
| Qwen2.5-3B   | 12 GB | +12 | +24 | ~48 GB | ✓ |
| Qwen2.5-7B   | 30 GB | +30 | +60 | ~120 GB | tight |

Full-FT ceiling ~3B on GB10 with comfortable headroom. 7B needs
gradient checkpointing OR optimizer-state offload OR LoRA.

**Estimate: 1-2 weeks after F2.**

## F4 — QLoRA

Base in Q8 (existing Phase 3 path) + LoRA in F32. Mathematically:
QLoRA = LoRA + the base happens to be quantized. Memory at 7B-Q8:

- Base: 7.4 GB (Q8 mmap'd, frozen)
- LoRA A + B + Adam: ~70-100 MB
- Activations: a few hundred MB at inference shape

Total ~8 GB. Fits comfortably on any modern GPU.

**Estimate: 3-5 days after F3.** Mostly memory bookkeeping + a
mixed-precision matmul code path.

## Honest summary

F0 + F1.0 land in this session (today). F1.1 onward is real
engineering work — each phase 3-10 days. Realistic timeline:

- F1.1: ~4 days (next session focus)
- F1.2: ~1 week (data + loss masking + checkpointing)
- F2: ~3 days after F1.1+F1.2 (CUDA mirror)
- F3: ~1-2 weeks (different scope — full Adam state)
- F4: ~3-5 days after F3 (mixed precision)

Total to QLoRA-working: ~4-6 weeks of focused work. Realistic.

## What's UNBLOCKED by F0+F1.0

- The math is proven. We know LoRA works algorithm-wise.
- ggml's autograd is wired (F0.4 has the pipeline; readback fix is
  surgical when needed).
- Forward + backward + optimizer ops are all bound through FFI.
- The next session can start at F1.1 step 1 (`LoraAdapter` class)
  without any prerequisite research.
