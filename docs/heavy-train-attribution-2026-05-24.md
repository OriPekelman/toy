# Heavy LoRA train: where the 2× gap lives (op-mix attribution)

**Date:** 2026-05-24. **Workload:** LoRA-Q step on Qwen2.5-1.5B,
seq=256, AdamW. **Hardware:** GB10 (`gx10`). **Question:** the
`bench-vs-pytorch-heavy` ratio sits at **2.01×**. Where is that
coming from?

## The headline

Toy's matmul-class op count is **~5× PyTorch's** at this shape. Most
of that comes from the **per-head** LoRA-Q decomposition (one A/B
adapter per Q head × 12 heads × 28 layers) where PT/PEFT runs one
full-dim adapter per layer. Each toy op is fast (~30-50 µs); the
launch-overhead cost stacks.

| Side                | Matmul-class ops / step | Kernel-time / step |
| ---                 | ---: | ---: |
| **Toy** (OUT_PROD + MUL_MAT) | **3 503** | ~154 ms |
| **PyTorch** (mm + addmm + bmm) |   **725** | ~83 ms (cuBLAS/CUTLASS) |
| Ratio               | **4.8×**                | ~1.85×           |

The per-op count is the dominant signal. Toy's mean dispatch time is
not far from PT's; we just dispatch 5× as many.

## How to reproduce

Trace:

```sh
TRACE=$HOME/tmp/toy_traces/heavy_lora_1p5b.json TRACE_OPS=1 \
  BENCH_TAG=heavy_train_lora_1p5b MODE=lora STEPS=5 SEQ_LEN=256 \
  GGUF=data/qwen25-1.5b-native.gguf \
  ./demos/seq_train_bench_cuda

ruby bench/aggregate_trace.rb $HOME/tmp/toy_traces/heavy_lora_1p5b.json
```

PyTorch op-mix:

```sh
docker run --rm --gpus all --ipc=host -v "$PWD":/w -w /w \
  gx10/dev-pytorch:latest \
  python3 bench/ref_pytorch.py --workload profile_train --device cuda \
  --arch qwen25_1p5b --lora --train_t 256 --warmup 2 \
  --profile_steps 3 --profile_top 25
```

## Toy op-mix (top by total time)

`compute_backward` is the parent FFI call (whole-graph eval) and is
double-counted with its children — excluded from the table below.

| op                | calls/step | total ms | mean µs | % step |
| ---               | ---:       | ---:     | ---:    | ---:   |
| **OUT_PROD**      | 1 598      | 85       | 26.5    | **30.8 %** |
| **MUL_MAT**       | 1 905      | 69       | 36.2    | **25.1 %** |
| ADD               | 2 192      | 24       | 10.8    | 8.6 %  |
| DIAG_MASK_ZERO    |   336      | 21       | 63.2    | 7.7 %  |
| MUL               |   197      | 10       | 51.8    | 3.7 %  |
| SCALE             |   672      |  5.4     |  8.0    | 1.9 %  |
| OPT_STEP_ADAMW    |   672      |  4.8     |  7.2    | 1.8 %  |
| CONCAT            |   308      |  3.1     | 10.1    | 1.1 %  |
| ROPE_BACK / ROPE  |   390+392  |  5.3     |  6.8    | 1.9 % combined |

Top four ops account for **72 %** of step time.

`upload_from_float_array` (label + hyperparam upload) shows up at
**5.5 %** of step time, ~7.6 ms/call. Worth investigating later — it
should be a thin host→device copy; that's too slow for what it is.
Probably non-pinned host memory + a type conversion in the copy path.

## PyTorch op-mix (top by total time)

| op                              | calls/step | total ms | % step |
| ---                             | ---:       | ---:     | ---:   |
| aten::mm                        | 474        | 78       | 33.4 % |
| (cuBLAS/CUTLASS GEMM kernels, multiple shapes) | ~446 launches | ~85 | 37 % combined |
| aten::mul                       | 953        | 11       | 4.8 %  |
| aten::bmm                       | 167        | 2.8      | 1.2 %  |
| aten::addmm                     | 84         | 2.4      | 1.0 %  |
| log_softmax / its backward      | 1+1        | 5.4      | 2.3 % combined |
| silu_backward                   | 28         | 2.4      | 1.0 %  |

PT step wallclock is **120 ms** while its cumulative kernel time is
**~234 ms** — i.e. ~50 % of kernel time happens in parallel with host
orchestration via multiple CUDA streams. **Toy's per-op sum ≈ wallclock
(270 vs 275 ms)**: ggml-cuda dispatches sequentially, no overlap.

## Where the 2× actually comes from

Three compounding effects, ranked by likely return on optimization:

1. **Per-head LoRA-Q backward emits 12 small OUT_PROD/MUL_MAT pairs per
   layer where PT emits 2**. With 28 layers × 12 heads × ~4 backward
   ops per adapter = ~1 300 extra matmul-class ops/step. At a mean
   ~30 µs of (mostly launch) overhead, that's ~40 ms/step — about
   **a third of the gap**. Toy's per-head decomposition is a Spinel-
   shape design choice; fusing the heads at LoRA-backward time
   (concat the 12 adapters into one [d_model, r·n_heads] tensor before
   the backward matmul) would eliminate most of these.

2. **No CUDA stream overlap**. Toy's per-op sum ≈ wallclock; PT's
   wallclock is ~50 % of its kernel-time sum because cuBLAS uses
   multiple streams. Even with the per-head fix, PT will continue to
   beat us on this axis until ggml-cuda overlaps host orchestration
   with kernel exec. Harder to fix; depends on ggml-cuda design.

3. **`upload_from_float_array` 5.5 % cost**. Labels + hyperparams
   upload per step at ~7.6 ms each is more than it should be (the
   payload is small). Either type conversion or non-pinned host
   memory in the copy path. Quick win to investigate.

The user's framing — *"whenever we are 'out of cuda' we should be
competitive with Python"* — partially holds: of the 275 ms step, only
~15 ms is Ruby/FFI orchestration (`compute_backward` covers 260 ms ≈
95 % of the step). The remaining gap is **inside** the CUDA dispatch
loop, not in the Ruby wrapper. Specifically, we dispatch more
operations than PT and don't overlap host scheduling with GPU exec.

## Next experiments

Cheapest-first ordering:

- **Fuse the per-head LoRA-Q adapters at graph-build time**. One
  `[d_model, r·n_heads]` adapter per layer instead of `n_heads`
  separate `[d_model, r]` adapters. Both forward and backward become
  single matmuls per layer. **Expected: ~30-40 ms saved per step**
  (closes ~half the gap).
- **Pin the host buffers used for upload_int_array / upload_row_major
  / upload_from_float_array**. Use `cudaHostAlloc`-backed memory via
  ggml-cuda. **Expected: ~10 ms saved per step**.
- **Coalesce the gradient accumulator ADDs**. 2192 ADD ops/step is
  high; many are gradient accumulation into per-layer slots. Some can
  be folded into the matmul that produced the contribution.

Re-record the heavy baselines after each successful experiment:
`make bench-heavy-update && make bench-vs-pytorch-heavy-update`.

## Methodology caveats

- `TRACE_OPS=1` adds ~7 % to step wallclock (256 ms traced vs 240 ms
  untraced). The op-count and relative-time attribution is unaffected.
- ggml-cuda per-op times include enqueue + any implicit syncs at op
  boundaries. They're not pure kernel time but they're consistent with
  observed step wallclock (sum of children ≈ step), so attribution is
  trustworthy.
- PT `self_device_time_total` is pure kernel time. Comparing toy's
  enqueue-plus-sync time to PT's pure kernel time slightly favours
  PT in absolute terms; ratios across op categories still hold.
- LoRA semantics differ between sides. Toy applies LoRA per-head
  AFTER the Q-projection split; PT/PEFT applies it to the full
  d_model × d_q linear before splitting. The output dimensionality
  is the same, the rank is the same; the wallclock difference is
  almost entirely op-count, not work done.
