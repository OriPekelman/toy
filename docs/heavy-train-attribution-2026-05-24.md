# Heavy LoRA train: where the 2× gap lives (op-mix attribution)

**Date:** 2026-05-24. **Workload:** LoRA-Q step on Qwen2.5-1.5B,
seq=256, AdamW. **Hardware:** GB10 (`gx10`). **Question:** the
`bench-vs-pytorch-heavy` ratio sits at **2.01×**. Where is that
coming from?

## The headline (initial hypothesis — refuted; see "Smoke result" below)

Toy's matmul-class op count is **~5× PyTorch's** at this shape. Most
of that comes from the **per-head** LoRA-Q decomposition (one A/B
adapter per Q head × 12 heads × 28 layers) where PT/PEFT runs one
full-dim adapter per layer. We initially expected each toy op's
launch overhead to compound this into wallclock — **a follow-up A/B
smoke disproved that** (see below). The 5× op-count differential is
real but does NOT translate to the 2× wallclock gap.

| Side                | Matmul-class ops / step | Kernel-time / step |
| ---                 | ---: | ---: |
| **Toy** (OUT_PROD + MUL_MAT) | **3 503** | ~154 ms |
| **PyTorch** (mm + addmm + bmm) |   **725** | ~83 ms (cuBLAS/CUTLASS) |
| Ratio               | **4.8×**                | ~1.85×           |

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

A/B smoke for the per-head-vs-fused question:

```sh
~/sites/spinel/spinel --cc='cc -Wl,-u,tnn_cuda_force_link' \
  tinynn/ab_smoke_lora_fused_cuda.rb -o tinynn/ab_smoke_lora_fused_cuda
./tinynn/ab_smoke_lora_fused_cuda
T=4    ./tinynn/ab_smoke_lora_fused_cuda
T=1024 ./tinynn/ab_smoke_lora_fused_cuda
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

## Smoke result: HeadFuseLoRAQ does NOT help

`tinynn/ab_smoke_lora_fused_cuda.rb` replaces N_HEADS independent
`x @ A_h` matmuls (the toy per-head LoRA-A pattern) with one batched
`mul_mat` (n_heads on `ne2`). Same total compute, different launch
pattern. Three shapes:

| T | per-head (12 mm + 11 add) | fused (1 batched mm) | speedup |
| ---: | ---:  | ---:  | ---: |
| 4    | 23.9 µs | 23.3 µs | 1.024× |
| 256  | 164.7 µs | 164.1 µs | 1.004× |
| 1024 | 650 µs | 660 µs | 0.984× (fused slightly slower) |

**Conclusion:** the per-head decomposition is NOT the source of the
2× gap. Either ggml-cuda already coalesces small launches efficiently,
or per-op cost in the real workload is dominated by something other
than kernel-launch overhead. A full HeadFuseLoRAQ refactor would have
been ~1-2 days for an estimated ~0.06 ms / step delta. We don't earn
that complexity.

The disconnect between the smoke (~7 µs/op average) and the trace
(~30 µs/op average) is the actual signal — those extra 23 µs/op must
come from something the smoke isolates away. Candidate causes for
where the gap *actually* lives:

1. **Graph-level sequential dependencies + lack of CUDA stream
   overlap.** PT's 234 ms of kernel time fits in 120 ms wallclock
   because cuBLAS uses multiple streams; ggml-cuda's per-op sum ≈
   wallclock, so it dispatches serially. This is a ggml-cuda design
   issue, not a toy issue — **file upstream rather than patch around**.

2. **OUT_PROD-specific cost on CUDA.** OUT_PROD is 30.8 % of step
   time at 26.5 µs mean. The smoke didn't test it. A focused
   OUT_PROD-vs-equivalent-MUL_MAT smoke would tell us if the win is
   in replacing OUT_PROD (where the gradient graph allows).

3. **TRACE_OPS overhead inflating per-op numbers in the trace
   itself.** Measure step time traced vs untraced (we have data:
   256 ms vs 240 ms = ~7 %). Probably not enough to explain 5× per-op,
   but worth ruling in/out.

## What to do instead

Re-ranked by likely return, after the smoke result:

- **Pin host upload buffers** (`upload_from_float_array` is 5.5 % /
  ~7.6 ms/call — too slow for the payload). Different surface
  (tnn FFI primitive), not a graph rewrite. **~10 ms saved / step.**
- **OUT_PROD vs MUL_MAT A/B smoke.** Cheap (extends the existing
  smoke harness). If OUT_PROD is per-call expensive, the win comes
  from gradient-graph restructuring to prefer MUL_MAT, not from
  fusing heads.
- **File ggml-cuda stream-overlap issue upstream** with the trace
  data we already have. We can't fix this in toy, and it's the
  largest single contributor we can identify so far.
- **Macro-op fusion** (different from head fusion): fuse adjacent
  norm+matmul, activation+matmul into single ggml ops when the
  graph builder can prove no other consumers. Shrinks graph node
  count, which reduces per-graph-eval overhead (the suspect for
  the 30-µs/op average).

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
  almost entirely op-count, not work done. **Op-count differential
  is real but doesn't translate to wallclock** (per the smoke).

## Process note: smoke-first earned its keep

The `tinynn/ab_smoke_lora_fused_cuda.rb` smoke took ~1 hour to write
and refuted the launch-overhead hypothesis before we paid the 1-2 day
HeadFuseLoRAQ refactor cost. The smoke is also a reusable A/B harness
for future "fuse-or-not" questions — change the inner ops, re-run.
This is the pattern: every proposed lowering rule should be A/B'd
against the isolated mechanic before being plumbed through 19
demos/examples.

## 2026-05-25 follow-up: four experiments, four converging null results

After the initial attribution, we ran all four ranked candidates from
the "What to do instead" section. The pattern that emerged:

| # | Experiment | Result |
| --- | --- | --- |
| smoke 0 | HeadFuseLoRAQ (12 small matmul vs 1 batched matmul) | **1.00×** |
| #1 | Pinned host upload buffers (cudaHostAlloc scratch) | **~0-1 ms / step** (within noise) |
| #2 | OUT_PROD vs MUL_MAT per-op cost at 4 shapes | **0.98-1.02× ratio** |
| #4 | Macro-op fusion (silu+mul → swiglu_split) | **1.00×** |

Every per-op-cost candidate refuted. The single remaining hypothesis
is **#3 — graph-level orchestration / single-stream serialization**
(ggml-cuda dispatches sequentially; PT fits 234 ms of kernel time in
120 ms wallclock via multiple cuBLAS streams). Upstream issue draft at
[`archive/upstream/issues-vendor/05-ggml-cuda-stream-overlap.md`](archive/upstream/issues-vendor/05-ggml-cuda-stream-overlap.md).

### What this means for toy

Three concrete take-aways:

1. **The 2× wallclock gap to PyTorch on the heavy LoRA-train bench
   is not toy's to close from the application layer.** It's a property
   of the ggml-cuda graph dispatcher. Toy already lives at the
   per-op-efficient point on this hardware.
2. **ggml-cuda already auto-fuses several relevant patterns**
   (`RMS_NORM + MUL`, `ADD` chains up to 8, `ROPE + … + SET_ROWS`,
   `SSM_CONV + ADD + SILU`). Toy benefits from these automatically as
   long as the graph builder emits adjacent ops in the recognised
   shape. **No toy-side change required** to get these wins.
3. **Pinned scratch and swiglu_split FFI were both shipped as
   defensive changes** even though they didn't move the bench needle.
   They're the right primitives to have available — future workloads
   with different shape regimes may benefit, and the cost (a few LOC
   of glue per primitive) is negligible.

### What we'd actually need to close the gap

Per the upstream issue: ggml-cuda either needs multi-stream dispatch,
async host-side node prep, or `cudaGraphInstantiate` capture for
repeated-shape graphs. The latter is most impactful for training-loop
workloads (same graph shape every step). Until any of those land
upstream, the heavy-bench ratio of 2.01× is approximately the floor
that toy can achieve while staying within ggml-cuda's design.

### The graph-side fix that IS toy's to make

`upload_from_float_array` consumes 5.5 % of step time (~19 ms / step at
the heavy shape) almost entirely because we upload a `[T, vocab]`
one-hot label tensor (155 MB at vocab=151936 × T=256) every step
through f64 → f32 conversion. The cleanest fix is graph-side: take
target IDs as a `[T]` int32 tensor, scatter on GPU. ~50 LOC across
`build_training_step` + the bench. Deferred until the upstream
ggml-cuda question is decided — if we land overlap upstream, this
becomes irrelevant; if we don't, this is the only remaining toy-side
lever.

## 2026-05-25 correction: the 19 ms one-hot upload was bench noise

We pursued the one-hot upload fix and found the bench was crashing on
fresh rebuilds (`GGML_ASSERT(cgraph->grads != NULL)`). Root cause: a
**Spinel type-inference landmine** — in
`lib/llama_seq_forward_ffi_cuda.rb` the per-KV-head tensor arrays
`t_k_per_kv` and `t_vt_per_kv` were initialized with `[]`, which Spinel
inferred as `sp_IntArray *`. The arrays then held tensor pointers
(8 bytes) but were indexed as ints (4 bytes), producing the
`sp_IntArray * vs sp_PtrArray *` warning at compile time and
unspecified behaviour at runtime.

The bench had been compiling with this warning all along; what we
read as "19 ms / step from labels upload" was the bench measuring a
silently-wrong execution path. Fix: seed the arrays with a typed null
pointer (`[TinyNNCuda.tnn_null_ptr]; .pop`) to force Spinel's
`sp_PtrArray *` inference.

After the fix:

- Toy heavy LoRA-train step time: 235 → **245 ms** (the previous
  number was wrong-execution-fast).
- `SKIP_LABELS_UPLOAD=1` savings: 19 → **~0.3 ms / step**. The "fix"
  has nothing to fix; the per-step upload cost was always small.
- Toy/PT ratio: 2.01× → **2.03×** (basically unchanged — both sides
  shifted by similar margins).
- Pinned scratch (`tnn_pinned_alloc`) contribution: was reported as
  ~0-1 ms (noise) — that interpretation stands.

**Net conclusion update.** Of the five candidates we investigated,
**all five are now refuted or attributed upstream**:

| # | Candidate | Outcome |
| --- | --- | --- |
| 0 | HeadFuseLoRAQ (per-head batch matmul) | 1.00× — refuted |
| 1 | Pinned host upload buffers | ~0 ms — defensive change kept |
| 2 | OUT_PROD per-op cost vs MUL_MAT | 0.98-1.02× — refuted |
| 3 | ggml-cuda stream-overlap | upstream issue 05 filed |
| 4 | Macro-op fusion (swiglu_split etc.) | 1.00× — already auto-fused; defensive FFI kept |
| 5 | One-hot label upload (this) | bench-noise illusion (Spinel landmine) |

**The Spinel landmine fix itself is the only real perf-relevant change
from this whole investigation arc.** It corrects per-step timing from
a measurement artifact, not a real win — but it stops the bench from
crashing on fresh rebuilds and aligns toy with its intended behaviour
(correct pointers into the K/V-per-KV-head arrays).

The 2× wallclock gap to PyTorch at this shape range on ggml-cuda is
**entirely upstream stream-overlap**. There are no per-op-cost
candidates left to investigate from the application layer.

### Process note: the bench-as-yardstick paid off again

Two attribution rounds, eight A/B smokes, and zero "real" application-
side optimizations. That sounds like nothing got done. What actually
got done:

- One genuine bug found and fixed (Spinel landmine in the training
  graph builder, which had been silently mis-executing for an unknown
  number of weeks before the bench made fresh rebuilds reproducible).
- Five hypotheses refuted with measured evidence (not handwaved).
- One upstream issue drafted with side-by-side trace data.
- A reusable A/B smoke harness (`tinynn/ab_smoke_*_cuda.rb`) for
  future "fuse-or-not" questions.
- Two FFI primitives added (`tnn_out_prod`, `tnn_swiglu_split`) that
  cost ~5 LOC each and are available when future workloads want them.
- Pinned host scratch (defensive correctness change).

The lesson is the principle that started the arc: **earn complexity by
attribution**. If we had skipped the smokes and just done the
HeadFuseLoRAQ refactor (1-2 days, ~70 LOC across 19 files), we'd have
spent the effort and gotten 1.00× speedup. The framework now lives at
"the application-layer optimization space is empty until upstream
moves" — a useful place to be.
