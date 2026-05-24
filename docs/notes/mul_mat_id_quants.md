# `mul_mat_id` × K-quantized source weights — known broken

**Status**: bug in upstream ggml. Workaround: use Q8_0 for MoE
expert weights.

**Affected**: MoE inference via `tnn_mul_mat_id` (used in OLMoE,
Mixtral, Qwen-MoE, Granite-MoE, and any other arch with routed
experts) when the expert weight tensors are stored as K-quants
(Q4_K, Q5_K, Q6_K — the "K-family" quantization formats).

## What goes wrong

M2.3 (commit 42b8609) integrated MoE inference end-to-end. Two
GGUFs of OLMoE-1B-7B-Instruct were tested:

|              | per-expert dtype | output                                           |
|--------------|------------------|--------------------------------------------------|
| Q4_K_M GGUF  | Q4_K (some Q6_K) | "The capital of France is **Dub Dub Dub Dub…**"  |
| Q8_0 GGUF    | Q8_0             | "The capital of France is **called Paris.**"    |

Same model, same inference code, same router weights (F16 in both).
Only the expert tensor dtype changes. Q4_K produces degenerate
repeating output; Q8_0 produces coherent factual text.

The model graph itself (router → softmax → top_k → 3× mul_mat_id →
silu·up → weighted sum) is identical across both runs. The only
difference is which kernel `ggml_mul_mat_id` dispatches to.

## Why we suspect a kernel gap

`vendor/ggml/tests/test-backend-ops.cpp::test_mul_mat_id` only
exercises `ggml_mul_mat_id` with F32, F16, and Q8_0 source weights
(per the test registrations around lines 8264–8439 of that file).
K-quants are not in the test matrix. Our M2.3 finding suggests
they don't actually work for mul_mat_id even though the kernel
dispatch presumably accepts them — output is garbage rather than
a hard error.

This is consistent with:
- ggml's mul_mat (non-id) DOES support K-quant sources on the
  same code paths we hit (regular per-head matmul, K-cache reads).
  OLMoE's attention weights are also Q4_K and they work fine.
- mul_mat_id's expert-indexing logic appears to mishandle the
  K-quant block layout. The legacy quants (Q4_0 / Q5_0 / Q8_0)
  have a simpler per-block scale; K-quants have a hierarchical
  scale + per-sub-block offset that requires different per-block
  unpacking.

## Workaround (current)

1. **Convert MoE models at Q8_0 or higher**. Avoid Q4_K_M, Q5_K_M,
   Q6_K for MoE expert weights specifically. Non-expert weights
   (embeddings, attention, norms) can stay at K-quants — only the
   MoE expert stacks need Q8_0.

2. **Runtime warning**: `realize_for_mmap` emits a warning when
   it detects MoE expert tensors with type ∈ {Q4_K, Q5_K, Q6_K,
   ...K_M / K_S / K_XS variants}. The model will still load and
   run, but output will be wrong. This makes the failure mode
   loud rather than silent.

3. **Choose your GGUF**: For OLMoE-1B-7B, `Meshwa/OLMoE-1b-7b-0924-
   Instruct-gguf` ships both Q4_K_M and Q8_0 variants. Use the
   Q8_0 (~7 GB, ~13 tok/s on gx10 CPU).

## Reproducing

Pure-Ruby repro is impractical because we'd need to generate Q4_K
data offline (the block layout has a hierarchical scale + per-sub-
block offset structure that's not trivial to synthesize). The
real-model repro:

```bash
# Q4_K_M — broken output
GGUF=data/OLMoE-1b-7b-0924-Instruct-Q4_K_M.gguf \
  PROMPT='The capital of France is' \
  N_NEW=8 ./examples/example_inference
# → "The capital of France is Dub Dub Dub Dub Dub Dub Dub Dub"

# Q8_0 — coherent output
GGUF=data/OLMoE-1b-7b-0924-Instruct-q8_0.gguf \
  PROMPT='The capital of France is' \
  N_NEW=8 ./examples/example_inference
# → "The capital of France is called Paris."
```

## Upstream

Reported to ggml-org/ggml:
<https://github.com/ggml-org/ggml/issues/1506> (filed 2026-05-24).

When upstream lands a fix:
- Remove the runtime warning from realize_for_mmap.
- Document that all K-quant expert types now work via a smoke that
  loads OLMoE Q4_K_M and confirms coherent output.

## Coverage-doc cross-reference

`docs/coverage.md` shows `GGML_OP_MUL_MAT_ID` as `bound: yes` on all
three backends (CPU/CUDA/Metal). That's accurate at the *binding*
level. The kernel-quality story is the (op, src_type) axis we
explicitly noted in the strategic reframe — covered by THIS doc.
