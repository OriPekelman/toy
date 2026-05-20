# Deep CUDA inference bench — 2026-05-21

Investigating the "current numbers seem somewhat low" suspicion from the
F1.1 close-out. Ships `demos/qwen25_bench_cuda` and `demos/qwen25_bench_cpu`
with proper warmup, long prefill, long decode, and per-token statistics.

## TL;DR

CUDA is **1.0×–1.4× faster than CPU** on GB10 across all model sizes
we tested. The handoff doc's "expect 5–10× speedup once GPU is free"
overshoots reality by 3–10×. **The bottleneck is not compute — it's
per-decode-step graph rebuild + sched_alloc**, which we do every call
in `SmolLM2KVCuda.decode_step` (and its CPU mirror).

## Sweep results

Bench config: 8-step warmup, 64-128 prefill tokens, 16-64 decode tokens.
Per-token mean / median in ms; median is more robust for the
graph-rebuild dominance pattern (no Pause-the-World outliers).

| Model           | Params | CPU mean | CUDA mean | CUDA / CPU | CUDA tok/s |
|---              |---:|---:|---:|---:|---:|
| smollm2-135m    | 135M | 9.4 ms   | 12.1 ms  | **0.78×** | 82 |
| smollm2-360m    | 360M | 20.9 ms  | 23.6 ms  | **0.89×** | 42 |
| qwen25-0.5b     | 500M | 26.4 ms  | 23.4 ms  | 1.13×     | 43 |
| tinyllama-1.1b  | 1.1B | 48.5 ms  | 44.2 ms  | 1.10×     | 23 |
| qwen25-1.5b     | 1.5B | 68.7 ms  | 49.3 ms  | 1.39×     | 20 |
| qwen25-3b       | 3.1B | 131 ms   | 92.5 ms  | 1.42×     | 11 |

**CUDA is actually slower than CPU on models ≤ 360M.** Kernel-launch
latency × ~1000 ops per layer × tens of layers eats the matmul win.
The crossover is around 500M params; even at 3B the speedup is
only 1.4×.

For reference, llama.cpp on similar-class hardware reports:
- TinyLlama-1.1B: ~80 tok/s (~12.5 ms/tok)
- Qwen2.5-3B: ~50 tok/s (~20 ms/tok)
- Qwen2.5-7B Q4: ~30-40 tok/s (~25-33 ms/tok)

So we're 4-5× slower than llama.cpp on the same hardware tier. The
gap is consistent with "per-step graph rebuild eats the kernel-launch
budget."

## Q8 also benched (CPU only)

| Model         | F32 mean | Q8 mean | Q8 / F32 |
|---            |---:|---:|---:|
| qwen25-1.5b   | 68.7 ms | 46.8 ms | 0.68× |
| qwen25-3b     | 131 ms  | 88.1 ms | 0.67× |
| qwen25-7b     | n/a     | 176 ms  | — |

Q8 saves ~30% on decode latency vs F32. The savings come from the
matmul bandwidth (Q8 weights = 1/4 the bytes). The fact that it's only
1.5× and not 4× is consistent with the graph-rebuild floor dominating.

## Prefill ≈ decode per-token cost

The most damning data point: at 1.5B F32 CUDA, prefill 128 tokens =
49.1 ms/tok mean, decode 32 tokens = 49.3 ms/tok mean. **Identical.**
A batched prefill would be 3-5× faster per token (one matmul over T
tokens, not T matmuls over 1 token). Ours isn't batched because each
step rebuilds the graph for `pos=N` and computes one decode step.

llama.cpp uses a different graph for prefill (multi-token) vs decode
(single-token) and reuses each across N invocations. That's the
unlocked speedup.

## Root cause: `decode_step` rebuilds the graph per call

`lib/toy_smollm2_ffi_kv_cuda.rb:594-602`:

```ruby
def self.decode_step(kv_cache, token_id, pos)
  TinyNNCuda.tnn_reset_for_rebuild(kv_cache.sess)   # free + reinit ctx
  step = kv_cache.build_decode_step(pos)            # ~1000+ FFI calls
  TinyNNCuda.tnn_realize(kv_cache.sess, step.kv_step_logits) # sched_alloc
  TinyNNCuda.upload_int_array(kv_cache.sess, step.t_token_id, [token_id])
  TinyNNCuda.upload_int_array(kv_cache.sess, step.t_pos,      [pos])
  TinyNNCuda.tnn_compute(kv_cache.sess)
  TinyNNCuda.download_row_major(kv_cache.sess, step.kv_step_logits, 1, kv_cache.vocab_size)
end
```

The `build_decode_step(pos)` builds the entire forward graph from
scratch — embed → N×(RMS-norm → QKV proj → RoPE → attention → output
proj → residual → RMS-norm → FFN → residual) → final norm → unembed.
~1000-3000 ggml_tensor creations per call depending on model size.

`tnn_reset_for_rebuild` (tinynn_ggml.c:1071-1100):

```c
size_t used = ggml_used_mem(s->ctx);
if (used > s->ctx_buf_size / 2) {
    ggml_free(s->ctx);           // entire compute ctx
    s->ctx        = ggml_init(...);
    s->graph_b    = ggml_new_graph_custom(s->ctx, 16384, false);
    s->realized_b = 0;
}
s->realized = 0;
s->graph    = ggml_new_graph_custom(s->ctx, 16384, false);  // every call
```

Every decode step recreates `s->graph`. Then `build_decode_step`
populates it. Then `tnn_realize` does `sched_reset` +
`sched_alloc_graph(s->graph)` (this is the path the F1.1 fix avoided
on the training side via `tnn_build_forward_only` — but inference still
sched-allocs every step, by design).

`sched_alloc_graph` on a multi-thousand-node graph in a CUDA backend
takes meaningful time per node: each tensor needs a CUDA-side buffer
slot decision, possibly a `cudaMalloc` (amortized through the buffer
pool), and the sched maintains internal split-state. Multiply that by
~2000 nodes for a 1.5B decode step, and you get the floor we observe.

## Why we built it this way

Per `project_persistent_dual_cgraph_2026_05_14.md`:

> Two contexts (ctx_w + ctx) + graph/graph_b verified on CPU + CUDA

The persistent-weights context is good — weights only allocate once.
But the compute graph was kept per-step because `pos` (used in RoPE
frequency lookup, KV-cache view offsets, attention masking) is baked
into tensor-builder C calls. Making the graph reusable across `pos`
values requires:

- `pos` as a graph input tensor (currently a Ruby int passed to
  `build_decode_step`).
- RoPE positions table as a graph input tensor (currently a slice
  built from `pos`).
- KV-cache views with parameterised offsets (currently `ggml_view_2d`
  with literal byte offsets).

ggml supports all of this — llama.cpp does it. Our wrapper just
doesn't.

## What "fix this" looks like (M3 from the original blueprint)

Per `project_full_accel_inference_blueprint_2026_05_14.md` M1-M4:
M1 (full forward) and M2 (KV cache) shipped; M3 was "pos as graph
input + reusable cgraph" and remains unbuilt.

Rough scope:

1. Add `tnn_input_1d_i32_persistent(sess, n)` if missing — a stable
   tensor in `ctx_w` for `pos` (1-element initially; T-element for
   batched prefill).
2. Rewrite `build_decode_step` to take NO `pos` argument; instead
   reference the persistent pos-tensor for:
   - RoPE positions input (already a tensor — just wire it to the
     persistent one)
   - KV-cache `ggml_view_2d` offsets (currently use `pos *
     sizeof(float)` at build time — convert to `ggml_view_2d` with a
     base + `ggml_set_2d_offset` op, or use `ggml_set_rows` which
     takes index tensors).
3. Build the decode graph ONCE in `realize_for_mmap`.
4. `decode_step` becomes: upload token_id + pos to the persistent
   inputs; `tnn_compute`; download logits.
5. (Stretch) Build a separate prefill graph (T-token batched matmul)
   and reuse it for all prefill calls.

Estimated cost: 3-5 days for inference path; the trickiest piece is
KV-cache view offsets (set_2d's `nb1 * pos` pattern in current
code). Acceptance: bench numbers move into the 3-10× CPU range for
≥ 1B models, and prefill drops to ~1/3 of decode per-token.

This is **not blocking F1.2** (LoRA on SmolLM2-135M). The fine-tuning
path uses a different forward graph (full-sequence, not per-step
decode), and small-model training cost is dominated by the backward
pass, not by the per-step inference rebuild.

## What we shipped this session

- `demos/qwen25_bench_cuda.rb` + `demos/qwen25_bench_cpu.rb` — per-token
  statistics (mean, median, min, p95, max) with proper warmup, long
  prefill, long decode. Spinel-compile and run via `make qwen25_bench_cpu`
  / `make qwen25_bench_cuda` (added).
- This document, locking in the numbers and root-cause analysis.

## What we didn't do (and why)

- Build the M3 persistent-decode-graph fix. Out of scope for "do a
  bench"; needs its own session.
- Compare to llama.cpp directly — not installed here. Quoted llama.cpp
  numbers are from public benchmarks on similar GB10 / Mac M4 / equivalent
  hardware tiers. We're consistent with each other on which factor of
  performance we're losing.
- Q8 CUDA bench — Q8 V-matmul fix landed for CPU (task #52) but the
  doc on `demos/qwen25_native_mmap_cuda.rb:10` still says "F32-only on
  CUDA today". Confirming Q8 works on the CUDA decode path is its own
  task; not gated by the M3 work.

## How to run the bench

```
make qwen25_bench_cpu qwen25_bench_cuda
GGUF=data/qwen25-1.5b-native.gguf PREFILL_T=128 N_NEW=64 ./demos/qwen25_bench_cuda
GGUF=data/qwen25-1.5b-native.gguf PREFILL_T=128 N_NEW=64 ./demos/qwen25_bench_cpu
```

Env: `MAX_T`, `N_WARMUP`, `PREFILL_T`, `N_NEW` per the script header.
Total runtime for the sweep above: ~5 min CPU + ~5 min CUDA.
