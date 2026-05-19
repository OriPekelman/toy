# Fine-tuning design

**Status:** draft, planning only
**Authors:** Ori Pekelman + Claude
**Date:** 2026-05-19
**Predecessor:** `lib/toy_trainer.rb` + `demos/train.rb` (CPU from-scratch, working);
               `FullForwardFFICache(Cuda)` (forward-only inference graph, shipped)

## Goal

Make this codebase fine-tune the architectures it can already run.

Target: take a pretrained checkpoint (SmolLM2-135M → Qwen2.5-7B
today; Llama-3.2, Qwen3 dense/MoE later) and adapt it on a
labeled dataset — instruction-following SFT, domain adaptation,
style transfer — without rebuilding the whole training stack.

Match the rest of the project: **same code runs on CPU + CUDA**
(PyTorch-style backend agnosticism via FFI module choice); algorithms
fit on one screen; introspection via `algorithm_card`.

## Two regimes

### Full fine-tune

Every parameter receives a gradient and Adam optimizer state.

Memory cost per parameter: 4 B (weight, mmap'd or f32) +
4 B (gradient) + 8 B (Adam m + v) = **16 B/param if f32 throughout**.

| Model       | F32 weights | + grad | + Adam | Total |
| ----------- | ----------- | ------ | ------ | ----- |
| SmolLM2-135M | 0.5 GB      | +0.5   | +1.1   | ~2.1 GB |
| Qwen2.5-0.5B | 2.0 GB      | +2.0   | +4.0   | ~8.0 GB |
| Qwen2.5-1.5B | 6.0 GB      | +6.0   | +12    | ~24 GB |
| Qwen2.5-3B   | 12 GB       | +12    | +24    | ~48 GB |
| Qwen2.5-7B   | 30 GB       | +30    | +60    | ~120 GB |

On the gx10 (121 GB unified) full-fine-tune fits up to 7B by the
skin of our teeth, but we lose all activation memory headroom.
Practical full-FT ceiling: ~3B.

### LoRA / QLoRA

Adapter matrices `B(r×d_in) · A(d_out×r)` injected on each
adapted linear; base weights frozen. Gradient + Adam state only
on `A`, `B`.

Standard adapters: Q, K, V, O attention projections. Often also
the FFN gate/up/down. r=16 is a sane default.

For Qwen2.5-1.5B with r=16 LoRA on attention only:
- Q (1536→1536): r*(1536+1536) = 49 K params/layer
- K (1536→256):  r*(1536+256)  = 29 K params/layer
- V (1536→256):  r*(1536+256)  = 29 K params/layer
- O (1536→1536): r*(1536+1536) = 49 K params/layer
- ≈156 K params/layer × 28 layers ≈ 4.4 M params total
- × 16 B/param ≈ **70 MB** total optimizer footprint

For 7B-Q8 base + LoRA: 7.4 GB (Q8 mmap'd) + 70 MB (LoRA Adam) =
**~7.5 GB**. QLoRA is the obvious default for anything ≥ 1.5B.

## Forward + backward graph

### Forward path is shipped

`FullForwardFFICache(Cuda)` (`lib/tinynn.rb`, `lib/tinynn_cuda.rb`)
already builds the full transformer forward as one persistent
ggml graph: embed → N×(pre-RMS → attention → residual → pre-RMS
→ FFN → residual) → final RMS → unembed. 33–38× faster than
the naive Ruby Mat path on LLM-ish shapes.

For inference we use `SmolLM2KVFFICache(Cuda)` (per-step KV decode);
for training we want the full-sequence forward graph because
backward through KV-cached decode is awkward.

### Backward path: ggml has the ops

Key discovery 2026-05-19: ggml upstream already has every backward
op we need (`include/ggml.h`):

- `ggml_silu_back(ctx, a, b)` — for SwiGLU FFN
- `ggml_rms_norm_back(ctx, a, b, eps)` — for RMS-norm pre/final
- `ggml_rope_ext_back(ctx, dy, positions, freq_factors, n_dims, ...)`
- `ggml_gelu_back` (for the GPT-2 GeLU path)
- `ggml_build_backward_expand(ctx, cgraph, grad_accs)` — full
  graph autograd. Extends an existing forward cgraph with
  gradient nodes.

CUDA kernels for these ops exist in `vendor/ggml/src/ggml-cuda/`
(haven't verified each one runs but they ship in libggml-cuda.a).

**This rewrites the cost analysis.** The gap isn't "implement
backward CUDA kernels"; it's "bind them through the tinynn FFI
shim and drive ggml's autograd correctly".

### Two implementation strategies

**(A) Hand-built backward graph.** Mirror the forward graph
explicitly; for each forward op call the corresponding `_back`
op with the upstream gradient. More code, more control, more
debuggable, no surprises.

**(B) `ggml_build_backward_expand`.** Mark which forward tensors
are parameters (grad accumulators) and which is the loss; ggml
walks the forward graph and emits backward nodes automatically.
Less code, fewer surprises if it works, harder to debug if it
doesn't.

**Recommendation: start with (B) on a tiny model end-to-end (the
M1 toy transformer used for the FullForward bench: vocab=4096,
d_model=384, 6 layers).** If it produces correct gradients
vs a Mat-based reference, we keep it. Fall back to (A) on
specific ops where ggml's autograd disagrees with our reference.

## Op coverage — corrected gap analysis

| Forward op       | Forward CUDA | Backward present (ggml) | Bound through tinynn |
| ---------------- | ------------ | ----------------------- | -------------------- |
| matmul           | ✓            | ✓ (`mul_mat_back`)      | ✓ (as `t_matmul`/`matmul_t`) |
| add              | ✓            | identity copy           | trivial |
| scale            | ✓            | scalar passthrough      | trivial |
| silu             | ✓            | `ggml_silu_back`        | **NEED** binding |
| gelu             | ✓            | `ggml_gelu_back`        | partial (CPU scratch only); **NEED** real CUDA path |
| rms_norm         | ✓            | `ggml_rms_norm_back`    | **NEED** binding (probe started: `tinynn/rms_norm_back_probe.c`) |
| layer_norm       | ✓            | `ggml_norm_back`        | **NEED** binding (GPT-2 path) |
| softmax          | ✓            | `ggml_soft_max_back`    | ✓ (`tnn_softmax_back`) |
| rope_ext         | ✓            | `ggml_rope_ext_back`    | **NEED** binding |
| view_2d          | ✓            | scatter (autograd handles) | OK via autograd |
| cpy              | ✓ (patched!) | inverse cpy (autograd) | OK via autograd |
| cross_entropy    | ✓            | combined fwd+grad       | ✓ (`tnn_ce_grad`) |
| embed_lookup     | ✓            | scatter-add             | ✓ (`tnn_embed_back`) |
| adam_step        | ✓            | optimizer-side          | ✓ |

**True gap (5 FFI bindings + 1 driver):**

1. `tnn_silu_back(sess, dy, x)` — SwiGLU FFN backward
2. `tnn_gelu_back_cuda(sess, dy, x)` — replace the scratch impl with a real kernel; ggml has it
3. `tnn_rms_norm_back(sess, dy, x, eps)` — finish the probe
4. `tnn_norm_back(sess, dy, x, eps)` — LayerNorm backward (GPT-2)
5. `tnn_rope_ext_back(sess, dy, positions, freq_factors, n_dims, mode)` — RoPE backward
6. **Backward-graph driver** in `TransformerLM` — builds the backward cgraph (autograd or hand-built), runs it, applies optimizer

Items 1-5 are ~10-20 LOC each in C; ~5 LOC each in Ruby FFI bindings.
Item 6 is the real work — ~300-500 LOC in Ruby.

## Optimizer state placement

Three contexts on the FFI side:

- **`ctx_w` (persistent)**: model weights. mmap'd from GGUF
  (Phase 2). Read-only for full-FT-not-on-LoRA; written by Adam
  step for full-FT.
- **`ctx_w_lora` (persistent, new)**: LoRA adapter weights. Tiny.
  Adam state for these lives alongside.
- **`ctx_compute` (rebuilt per step)**: activations, gradients,
  loss. Wiped between steps.

For full-FT: gradient + Adam-m + Adam-v tensors live in their own
persistent context. Need 3× the weight footprint, which is the
budget that puts 7B out of reach on single-GPU.

For LoRA: only `ctx_w_lora` has grad + Adam state. Base `ctx_w`
stays read-only.

## Instruction-tuning loop

### Data format

Standard jsonl, one example per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

or simpler `{"prompt": "...", "response": "..."}`. Pick one,
stick to it. The chat-message format is more common in 2026; SFT
datasets on HF are mostly in that shape.

### Preprocessing (`prep/`)

New script `prep/prep_sft.py` (Python — already where we tokenize):

- Reads jsonl
- Applies the model's chat template (Qwen2/3 has one; Llama-3
  has one; GGUF embeds `tokenizer.chat_template`)
- Tokenizes prompt + response separately
- Writes one binary file per split: `[u32 n_examples] [(u32 prompt_len, u32 response_len, prompt_ids..., response_ids...)...]`
- Output: `data/sft_<dataset>_train.bin`, `data/sft_<dataset>_val.bin`

The Ruby side reads this binary; no jsonl-in-Spinel parsing.

### Loss masking

Standard SFT recipe: cross-entropy on response tokens only;
prompt tokens are masked out.

Implementation: `cross_entropy_grad` already supports masking via
the targets tensor (set target=-100 to skip a position, standard
HF convention).

### Eval

Held-out validation set:
- Compute average loss per epoch on val set
- Optionally: greedy generation on 5–10 fixed eval prompts,
  qualitative read-through (no automated metric — they don't
  correlate with usefulness at this scale)

### Checkpointing

For LoRA: save `ctx_w_lora` tensors to a small GGUF every N
steps. Resume = load the most recent. Trivial.

For full-FT: save the entire updated `ctx_w` to a new GGUF.
Big files but checkpointing is rare.

### Outer loop shape

```
load model (mmap base)
build forward graph + backward graph + adam state
load train.bin
for epoch in epochs
  shuffle examples
  for batch in batches
    fill input_ids, target_ids
    compute_forward
    compute_loss (masked CE)
    compute_backward (autograd)
    adam_step
    zero_grads
  every M epochs: eval_loss + sample_generation + checkpoint
save final adapter / weights
```

## Phased rollout

### Phase F0 — verify, bind, infrastructure

- Verify `ggml_silu_back`, `ggml_rms_norm_back`, `ggml_rope_ext_back`,
  `ggml_gelu_back` produce correct output on CUDA via parity smokes
  (mirror the existing `ab_smoke_*_cuda` pattern: random input, run
  CPU, run CUDA, compare).
- Add tinynn FFI bindings (items 1-5 in the gap table).
- Smoke-test `ggml_build_backward_expand` on a 3-op toy graph
  (matmul → gelu → matmul → loss; gradients vs Mat reference).
- Estimate: 3-5 days.

### Phase F1 — LoRA on CPU first

Why CPU first: avoids any CUDA-kernel surprises; isolates the
fine-tuning logic from accelerator quirks.

- Forward: existing CPU `FullForwardFFICache` + injected LoRA
  matmuls on attention projections.
- Backward: ggml autograd over the same graph; grad accumulators
  only on LoRA tensors.
- Driver: new `lib/lora_trainer.rb` (parallel to `lib/toy_trainer.rb`).
- Demo: `demos/sft_lora.rb` — fine-tune SmolLM2-135M on a tiny SFT
  dataset (alpaca-style, 100 examples).
- Acceptance: loss decreases monotonically over 10 epochs;
  generation post-training reflects the instruction style.
- Estimate: 1-2 weeks.

### Phase F2 — LoRA on CUDA

- Same driver, swap CPU FFI for CUDA FFI.
- Acceptance: bit-identical-to-CPU loss curve (within numerical
  tolerance, F32 throughout).
- Estimate: 3-5 days after F1.

### Phase F3 — full fine-tune on CUDA

- All weights trainable.
- Adam state context. Memory accounting: 1.5B fits comfortably
  on GB10; 3B fits; 7B requires us to make hard choices (drop
  Adam to SGD-with-momentum? gradient checkpointing? offload Adam
  to host RAM?).
- Demo: `demos/sft_full.rb`.
- Estimate: 1-2 weeks.

### Phase F4 — QLoRA

- Base weights stay Q8 (load via existing Phase 3 path).
- LoRA in F32.
- Mixed-precision matmul: existing `qwen25_native_mmap_q8` already
  does Q8 weight × F32 activation. Backward through quantized
  weight = no gradient flows back (base is frozen), so this is
  mathematically just LoRA-on-quantized-base. No new math needed.
- Acceptance: 7B-Q8 fine-tuned via QLoRA fits in 8-10 GB total;
  loss curve comparable to F32 LoRA on a smaller model.
- Estimate: 3-5 days after F3.

## Risks

| Risk | Mitigation |
| --- | --- |
| `ggml_build_backward_expand` doesn't handle our specific op composition correctly | Fall back to hand-built backward in `lib/lora_trainer.rb`. Phase F0 smoke-test catches this. |
| Spinel ivar-collision blocker (the M1 work hit this when integrating `FullForwardFFICache` into `TransformerLM`) | Retry on fresh Spinel (we just bumped to 7beeb54, and the May 13 regression is gone — probably the ivar issue is too). If still present, isolate to a Spinel issue with minimal repro. |
| Adam state pressure for full-FT on 7B | Use LoRA for ≥3B. Document the "full-FT ceiling: 3B on this hardware" honestly. |
| Backward-through-cpy after our recent CUDA cpy patch — could the autograd cpy backward also be buggy? | Add a parity smoke as part of Phase F0 (matmul → cpy → matmul backward). |
| Tokenizer encode side still external (Python) — limits SFT-in-Ruby story | Live with it. Phase D (Ruby tokenizer encode) is a separate roadmap item; not blocking. |

## Out of scope (record so we don't drift)

- Distributed / multi-GPU training (single GB10).
- Mixed-precision FP16 / BF16 (we're F32 throughout; ggml supports
  these but we don't have an FP16 inference path either).
- Gradient checkpointing (memory optimization; defer until full
  FT works on a model that needs it).
- RLHF, DPO, PPO, KTO — alignment recipes that need a reward
  model. Way past scope.
- Continued pretraining (training from scratch — different shape;
  the existing `demos/train` does this on a toy and is back to
  working).
- Encoder-decoder models (decoder-only universe only).

## Acceptance for this design doc

When a reader can:

- Predict the FFI surface that needs binding (the 5 items in the
  gap table).
- Predict the memory pressure for LoRA vs full-FT at each model
  size and pick the right regime.
- Predict the data format and how SFT loss masking works.
- Predict the rollout order and what Phase F0 ships.

— we're ready to start Phase F0.
