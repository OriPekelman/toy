# Perf sweep — inference + fine-tune optimization candidates (2026-05-23)

Used the Chrome Trace observability primitive (P1) and the bench
harness to profile real workloads at meaningful scale, then identify
where the bench-day budget would actually move the needle.

## Headline finding

**Both inference and LoRA fine-tune are bottlenecked by a single
ggml call: `tnn_compute_backward` (training) or `tnn_compute`
(inference).** That call accounts for 99.5%+ of step wallclock.
Marginal optimizations have to land *inside* the ggml graph or
*alongside* it (different graph shape, different kernel choice),
not in our Ruby wrapper.

This means the most productive next-step work is **binding ggml
features we haven't yet bound** — flash attention, quantized KV
cache, alternative attention shapes — rather than tweaking the
graph builder.

## Inference scaling on gx10 CPU

`example_inference` wallclock for `PROMPT="The capital of France is"
N_NEW=16` (21 total tokens):

| model           | wall (s) | tok/s |
|-----------------|---------:|------:|
| SmolLM2-135M    | 0.46     | ~46   |
| Qwen3-0.6B      | 1.05     | ~20   |
| Llama-3.2-1B    | 1.74     | ~12   |
| Qwen3-1.7B      | 1.96     | ~11   |
| Qwen3-4B        | 4.72     | ~4.5  |

Per-token cost scales roughly with parameter count, as expected.
Load time (~100ms) eats into the smallest model's headline number
but doesn't matter at 7B+.

Trace shape (from `TRACE=… example_finetune` on SmolLM2-135M LoRA):

```
step (70.7 ms):
  upload_int_array     (< 1 us)
  upload_float_array   (~100 us)    ← labels, hyperparams
  compute_backward     (~70.5 ms)   ← 99.7 %
  download             (~1 us)
```

The inference trace is the same shape: realize-once + per-step
upload, compute, download. Compute always dominates.

## What's actually slow inside compute_backward

**Updated with P6 empirical data.** As of 2026-05-23 the trace
primitive supports per-ggml-op duration events via
`TRACE_OPS=1`. Measured on SmolLM2-135M LoRA, T=4, 30 layers,
2 training steps:

```
total compute_backward: ~913 ms (2 steps × ~457 ms)
events: 21,562

OP                  count     tot_us    pct   avg_us
OUT_PROD             5378     318148   11.6%   59.2   ← matmul-back grad-W
MUL_MAT              3302     197692    7.2%   59.9   ← forward + back grad-X
ADD                  2868     123118    4.5%   42.9
OPT_STEP_ADAMW       1080      48949    1.8%   45.3
ROPE / ROPE_BACK     1434      62541    2.3%   ~44
DIAG_MASK_INF / ZERO 1080      45683    1.7%   ~42
SOFT_MAX / _BACK     1080      42629    1.6%   ~39
MUL                   422      20682    0.8%   49.0
CONCAT                480      18999    0.7%   39.6
... (≈70% accounted; rest is RMS_NORM, VIEW, RESHAPE, etc.)
```

Confirms the architectural prediction: **OUT_PROD + MUL_MAT
together account for ~19% of measured per-op wall time**, the
largest single category. That's the matmul tile in
`grad_W = grad_y · xᵀ` (OUT_PROD) and `grad_X = grad_y · Wᵀ`
(MUL_MAT). The 5,378 OUT_PROD count is roughly 2× the 3,302
MUL_MAT — consistent with backward needing one OUT_PROD per
weight matrix plus the forward MUL_MATs being replayed for
some grad chains.

**Caveat**: P6 instrumentation costs ~5× slowdown when
`TRACE_OPS=1` (the eval callback disables some kernel fusion).
Per-op `avg_us` values above are *inflated* relative to the
no-trace run. Use the relative proportions — not absolute
microseconds — to rank ops.

The original architectural sketch of what's in the graph (for
reference):

For **LoRA Q training** on SmolLM2-135M (T=4, 30 layers):
- 30 × (per-block forward: Q/K/V matmul × n_heads, RoPE, scaled-dot
  softmax, V matmul, O matmul, RMSNorm, SwiGLU FFN with 3 matmuls,
  add+norm)
- Cross-entropy loss on logits
- 30 × backward of the same (matmul backward = 2 matmuls each)
- AdamW step per param × ~9 heads × 2 (A+B) × 30 layers = 540 opt_step calls
- = ~hundreds of ggml ops per step, dominated by matmul

For **inference decoding** one token from a KV cache:
- 30 × (Q/K/V matmul per head, RoPE, attention softmax over hist,
  V matmul, O matmul, FFN matmuls)
- + final RMSNorm + tied LM-head matmul

Matmul dominates both. Per the ggml CPU kernel:
- F32 matmul: AVX2 + OpenMP threads
- Q8_0 matmul: dequant-on-the-fly into AVX2 lanes
- Cache locality matters (large K/V history)

## Ranked optimization candidates

In rough order of impact per unit effort:

### 1. Flash attention (`tnn_flash_attn_ext`) — **high impact, medium effort**

ggml has `FLASH_ATTN_EXT` + `FLASH_ATTN_BACK`. Both unbound in
our FFI. Fuses `softmax(Q · Kᵀ / √d) · V` into one kernel,
streaming over K/V tiles. Memory bandwidth dominates this step
at long context; flash attention is 2–4× on the CUDA backend
for `T ≥ 1024`. On CPU it's a smaller win but still nonzero.

Effort: 1–2 days. Bind C wrapper, integrate at the graph-build
site (replace the current scaled-softmax-matmul triplet), validate
against the existing path (bit-identical at small shapes; tolerance
at large).

Estimated: 2–4× decoder throughput at `ctx ≥ 4k`. Smaller at
short contexts.

### 2. Q8 KV cache — **medium impact, small effort**

K and V are currently allocated as F32 (`tnn_input_2d_f32_persistent`).
Allocating them as Q8_0 halves the memory bandwidth on every
attention step. ggml supports Q8 KV natively; we just pass the
right type at allocation time.

Effort: ~half day. Add an opt-in flag on `SmolLM2KVFFICache`;
validate output parity within numerical tolerance against the F32
KV baseline.

Estimated: 1.5–2× on long-context decode (where KV bandwidth is
the wall). At short contexts the win is smaller because K/V
fits in cache anyway.

### 3. Multi-token decode / speculative decoding — **high impact, large effort**

Currently `decode_step` processes one token at a time. The matmul
overhead is per-token even though the weight matrix is the same.
Multi-token decode (process N tokens in one forward) amortises
this. Speculative decoding stacks a small draft model + verify
loop on top.

Effort: 3–7 days. Requires a "decode N tokens" graph shape (we
have the sequence-mode forward but it's training-flavoured).

Estimated: 2–5× decoder throughput for batches of 4–8 tokens.
Speculative on top: 1.5–2× more (acceptance-rate dependent).

### 4. ggml CPU sched aliasing fix (upstream) — **medium impact, zero effort
   for us, depends on Matz / ggml community**

`ggml-org/ggml#1501` (the LoRA-training divergence) is masked by
our `tnn_pin_all_graph_b_nodes` workaround, which disables
buffer-slot reuse. That probably costs ~10% in cache-friendliness
on CPU. If upstream fixes the bug and we can drop the pin, that's
"free" perf.

Effort: zero (just wait + rebase). Optimization on a stick.

Estimated: ~5–10% on CPU LoRA training.

### 5. AdamW batching — **low impact, medium effort**

Per-step we run `opt_step_adamw` ~540 times for SmolLM2-135M
LoRA (per Q head per layer × A+B × momenta). Each is a small
elementwise op. Batching them into a single multi-tensor opt
step would reduce graph-traversal overhead. ggml has no direct
batched op, so this would need a custom kernel.

Effort: 2–3 days (C kernel + binding).

Estimated: 5–10% on training step.

### 6. Activation checkpointing — **specialized, only when memory-bound**

Trades memory for compute. Worth it only when fitting a bigger
model would unblock something; on gx10 (121 GB unified memory)
we have plenty of headroom for 7B-class training. Defer until
we want to train 13B+ from scratch.

Estimated: 0% on speed but unblocks new model sizes.

## Recommended next batch

If we have a focused week:

1. **Flash attention binding** — biggest decoder win, unblocks
   long-context use cases for every model on the list.
2. **Q8 KV cache** — easy companion win.
3. (Wait on the upstream ggml fix; revisit when it lands.)

Multi-token / speculative decode is the next tier — better tooling
needed before we commit to it.

**Update 2026-05-23**: items 1 + 2 + the per-op tracing (P6) all
landed in this sweep.

  - **P4 (flash_attn_ext binding)**: tnn_flash_attn_ext + Ruby/CUDA
    FFI bind. ab_smoke_flash_attn validates parity vs the
    scale→softmax→matmul triplet (max |Δ| = 3.8e-6). Backward
    blocked on upstream ggml (the impl aborts). Integration into
    SmolLM2KVFFICache deferred to P4.1.
  - **P5 (Q8_0 KV binding)**: tnn_input_2d_persistent_typed already
    supported Q8_0 for weights; ab_smoke_q8_kv exercises it for
    the KV-cache shape (max |Δ| = 2.1e-3 on varied input).
    Integration deferred to P5.1 (K layout flip needed for
    block-aligned writes).
  - **P6 (per-op timing)**: TRACE_OPS=1 emits one Chrome-Trace
    event per ggml node. Confirmed OUT_PROD + MUL_MAT dominate
    backward; MUL_MAT dominates CUDA forward.

The integration tasks (#107, #108) are the actual perf-unlock; the
binding work landed first so the integration has a known-good
primitive to call into.

**Update 2026-05-24** (P5.2 V layout flip): The headline flash
attention win finally materializes. The previous measurement showing
"flash = baseline" on Qwen3-1.7B was bottlenecked by a transpose-cont
of V per Q-head per layer per step (V cache was laid out positions-
on-ne0, flash wants positions-on-ne1). P5.2 flipped V to mirror K's
`ne=[d_head, max_T]`. Result on Qwen3-1.7B, N_NEW=32, CPU:

|                | wall  | rel    | notes |
|----------------|------:|-------:|-------|
| baseline       | 3.54s | 1.00×  | F32 KV, no flash |
| FLASH only     | 3.09s | 0.87×  | F32 KV, flash — was a wash before P5.2 |
| KV_Q8 (K+V Q8) | 3.07s | 0.87×  | Q8 KV + flash (auto-required) |

Token streams bit-identical across all three. The previous KV_Q8 row
in this doc reported Q8-K-only + non-flash; we now have full Q8 KV +
flash with comparable throughput AND ~3.75× smaller KV cache memory
(V was the bigger chunk before — it stayed F32 until P5.2). At
longer contexts where memory bandwidth dominates, the KV_Q8 win
should grow further; needs a bench with N_NEW >> 100 to confirm.

Side effect: Q8 V structurally requires flash (transposing Q8 with
non-block-aligned `hist_count` is impossible). `enable_kv_q8!` now
auto-enables flash. Documented in the cache class comment.

## Out-of-scope for this sweep

- **GPU kernel hand-tuning**: ggml-cuda's kernels are good; we
  don't have headroom to write better ones from scratch.
- **AVX-512 / SVE on CPU**: ggml's CPU kernels auto-dispatch.
- **Distributed multi-GPU**: gx10 is single-GPU.
- **Mixed precision (BF16/FP16)**: ggml-cpu doesn't fully support
  BF16 compute; ggml-cuda does, but we'd need a separate weight
  path. Worth investigating once the simpler wins are in.

## What the tracing primitive itself enabled

The Chrome Trace observability primitive (P1) didn't surface new
hot spots that we didn't already know about, because the FFI
boundary is too coarse — everything bottoms out in
`tnn_compute_backward` or `tnn_compute`. The next iteration of the
primitive would need either:

- ggml-level per-op timing instrumentation (cheap when `GGML_PERF`
  is enabled, but it changes the build), or
- Sampling inside the C side at fixed intervals (statistical
  profile of which ggml op is currently executing).

Without one of those, our tracing is great for diagnostic /
correctness work (confirming workflow shape, catching shape
regressions, debugging) but not for picking optimizations.

**Update 2026-05-23**: ggml per-op timing is now shipped (P6 /
task #106). The sched eval-callback emits one Chrome-Trace
duration event per ggml node when `TRACE_OPS=1` is set
alongside `TRACE=…`. Opt-in via env var; off-path overhead is
within timing noise. See `tinynn_trace.h` for caveats (5×
slowdown when on; CUDA timings reflect enqueue latency rather
than kernel duration).
