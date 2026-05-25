# Issue draft: ggml-cuda single-stream serialization vs PyTorch multi-stream

**Target repo:** `ggml-org/ggml`

**Type:** Performance feature request (with side-by-side profile data)

---

## Title

ggml-cuda: graph compute serializes host orchestration with kernel exec (single-stream); PyTorch multi-stream achieves ~50 % overlap on the same workload

## Body

### Summary

When running a training-step compute graph through
`ggml_backend_graph_compute` on the CUDA backend, the per-op wallclock
sum is approximately equal to the total `compute_backward` wallclock.
This suggests ggml-cuda dispatches sequentially on a single stream and
host orchestration (sched node walk, allocator bookkeeping, ggml-cuda
op preparation) is not overlapped with the kernel execution of
preceding ops.

The same workload (logically: Qwen2.5-1.5B LoRA-Q fine-tune step at
seq=256) takes:

- **PyTorch**: ~120 ms wallclock, ~234 ms cumulative kernel time
  (measured via `torch.profiler.self_device_time_total`). I.e. PT
  achieves **~49 % overlap** between host orchestration and kernel
  execution by issuing multi-stream cuBLAS / CUTLASS launches.
- **ggml-cuda (toy framework, single-machine same-GPU)**: ~240 ms
  wallclock, per-op trace sum ≈ 270 ms (measured via per-op timing
  hooks instrumented around `ggml_backend_sched_compute_node`).
  I.e. **~0 % overlap** — the per-op sum is the wallclock.

This roughly doubles the wallclock cost of training-step compute
even when the underlying kernels are competitive.

### Why this matters

For frameworks built on ggml-cuda that target single-GPU training or
batched inference, the gap to PyTorch is largely attributable to this
serialization rather than to kernel efficiency. We isolated it by:

1. **A/B-smoking suspected per-op overheads.** Per-head LoRA-Q
   decomposition (12 small mul_mat vs 1 batched mul_mat at the same
   logical shape) shows ~1.00× speedup from batching — i.e. ggml-cuda
   already handles many small ops with low per-launch overhead, so
   batching them buys nothing.
2. **OUT_PROD vs MUL_MAT smoke** at four shapes (LoRA-A grad, LoRA-B
   grad, square d_model, lm-head-ish) shows 0.98×-1.02× ratio. So
   OUT_PROD per-op cost is normal; backward-graph gradient ops are
   not specifically expensive.
3. After ruling out per-op causes, the remaining slack is in the
   per-graph orchestration — which on a single stream cannot overlap
   with kernel exec.

### Reproducer (toy-side; can be reduced to a pure ggml C test)

The toy framework (https://github.com/OriPekelman/toy) has the full
pipeline:

```sh
# Toy side: traces a training step and emits per-op timings.
TRACE=trace.json TRACE_OPS=1 BENCH_TAG=heavy_train_lora_1p5b \
  MODE=lora STEPS=5 SEQ_LEN=256 \
  GGUF=data/qwen25-1.5b-native.gguf \
  ./demos/seq_train_bench_cuda

ruby bench/aggregate_trace.rb trace.json
# > step span: 2 steps, total = 550.94 ms, mean = 275.47 ms
# > per-op sum: 270 ms / step (≈ wallclock)

# PyTorch reference at the same logical shape.
docker run --rm --gpus all --ipc=host -v "$PWD":/w -w /w \
  gx10/dev-pytorch:latest \
  python3 bench/ref_pytorch.py --workload profile_train --device cuda \
  --arch qwen25_1p5b --lora --train_t 256 --warmup 2 \
  --profile_steps 3 --profile_top 25
# > step time ≈ 120 ms; sum(self_device_time_total) ≈ 234 ms
```

A minimal pure-ggml reproducer would build a 28-layer transformer-ish
graph (rms_norm → mul_mat → silu → mul_mat → add chain) and time both
graph_compute wallclock and the sum of per-node timings via a callback
on `ggml_backend_sched_compute_node`. Happy to provide one if useful.

### Hardware

- NVIDIA GB10 (Grace Blackwell, 121 GiB unified memory, compute
  capability 12.1, VMM enabled)
- Ubuntu 24.04, CUDA 13.0
- ggml HEAD as of 2026-05-25

### Concrete ask

A path toward overlapping host orchestration with kernel exec on
ggml-cuda. Options as we see them (not prescriptive — happy to help
land whichever is preferred):

1. **Multi-stream dispatch in ggml-cuda's scheduler.** Round-robin
   independent nodes across N CUDA streams, sync at synchronization
   points (output of the graph; nodes whose result is read host-side).
   This matches what cuBLAS does under PyTorch.
2. **Asynchronous host-side node prep.** Pipeline the next node's
   allocator bookkeeping + cuLaunchKernel arg setup against the
   previous node's kernel exec, even on a single stream.
3. **Per-graph CUDA graph capture** (`cudaGraphInstantiate`) when the
   compute graph is reused across steps with the same shape. Skip
   per-step orchestration entirely after the first capture.

We'd find (3) most impactful for our training-loop use case (same
graph shape every step) and least invasive to the rest of the
backend.

### Related context (not blocking this issue)

- Toy's framework writeup of the attribution that led here:
  `docs/heavy-train-attribution-2026-05-24.md` in
  https://github.com/OriPekelman/toy. Has the trace JSON aggregator
  output, op-mix breakdowns for both sides, and the two A/B smokes
  that refuted alternative hypotheses.
- The toy `make bench-vs-pytorch-heavy` gate runs the same workload
  and reports the toy/PT ratio for regression detection.
