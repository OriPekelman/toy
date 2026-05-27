# Training backends beyond ggml-cuda

**Date:** 2026-05-27. **Status:** thinking-out-loud, not a commitment.
**Context:** companion to `backends-and-scale-2026-05-27.md` (which
covers DP/multi-GPU at high level) and the `decoupled-diloco-research`
doc (heterogeneous training). This one drills into a specific
question: **what's the path to a training backend that can beat
PyTorch on toy's workloads?**

## The honest current state

Today toy has three backends, all via ggml:

| Backend | Inference | Training |
| --- | --- | --- |
| CPU (ggml-cpu) | ✅ | ✅ |
| CUDA (ggml-cuda) | ✅ | ✅ at "works" — not "competitive" |
| Metal (ggml-metal) | ✅ | ✅ |

The bench-vs-pytorch numbers (`docs/bench-vs-pytorch.md`, May 2026)
say toy is roughly at parity for training and ~1.38× slower for
decode on GB10 at 135M. Attribution
(`heavy-train-attribution-2026-05-24.md`) says the gap is upstream
ggml-cuda single-stream dispatch — toy-side application overhead is
near zero.

So when the user says "beat PyTorch on training," they're asking:
**can we close the 1.38× decode gap and the training-side wash to
something better than parity, on at least some shapes?** Realistic
answer: yes, for specific shape regimes, with significant work.

## Where ggml-cuda is hitting its training limits

Three concrete problems:

### 1. Backward gaps in vendored ops (already filed)

[ggml-org/ggml#1514](https://github.com/ggml-org/ggml/issues/1514):
- `GGML_OP_NORM` (LayerNorm) — no backward
- `GGML_UNARY_OP_GELU` — no backward
- `ggml_conv_2d` — ends in `cont(permute)` whose backward asserts
  contiguous gradient

We work around with RMSNorm + SiLU + flat-linear patch_embed. Real
arches that ship LayerNorm + GELU (GPT-2, BERT, ViT-Tiny vanilla)
can't train through ggml-cuda's auto-backward today.

### 2. Per-op kernel launch dominates at small shapes

Memory note `project_ffi_per_op_cost_2026_05_23`: a single
session-per-op FFI matmul is ~800µs regardless of shape. Break-even
with pure Ruby is at `m*k*n ≈ 100k`. Toy's per-head-decomposed
attention emits ~50 ops per layer per training step. At 12 layers,
24 heads, that's 14,400 ops per step. Even with ggml's persistent
session dropping the FFI overhead, kernel launch + sched overhead
is real.

### 3. Single-stream dispatch

ggml-cuda's graph compute is single-stream. Modern training stacks
overlap memcpy + compute + alternate-stream prefetch. Filed as
`docs/archive/upstream/issues-vendor/05-ggml-cuda-stream-overlap.md`
upstream. Nothing toy can close from this side.

These three together cap toy's training perf well short of PyTorch's
on the "everything compiles" PyTorch-eager-with-compile shapes
(SmolLM2 / GPT-2 / Llama-3 at >1B params with batch >8).

## Where toy *could* beat PyTorch (be honest)

PyTorch's overheads that toy can avoid:
- **Python interpreter loop.** Each forward pass goes through Python.
  `torch.compile`/`Inductor` fuses, but the boundary work isn't free.
  Toy is AOT'd to native via Spinel — zero Python at runtime.
- **Autograd graph allocation.** PyTorch builds a tape per forward
  pass. Toy's training graph is realized once and reused across
  steps (persistent session pattern from May 2026).
- **Generic kernel dispatch.** PyTorch's ATen picks a kernel from a
  large set per dtype × layout × backend. Toy can pick once at
  realize time and bake in.
- **Variable batch / sequence length.** PyTorch has to handle
  arbitrary shapes per call; toy realizes a fixed shape per session.

PyTorch's wins toy probably can't beat:
- **FlashAttention-2/3.** Bespoke CUDA kernels with online softmax,
  fused QKV, recomputation. Years of NVIDIA + research effort.
- **NCCL distributed primitives.** Battle-tested all-reduce / DP / TP.
- **Tensor-parallel + pipeline-parallel.** The whole megatron-style
  stack.
- **Mixed-precision training infrastructure.** Loss scaling, dynamic
  cast, master-weights — PyTorch's AMP is mature.

**Realistic claim:** for **single-GPU training of small-to-medium
fixed-shape models** (toy's actual workloads), toy can plausibly
beat eager-PyTorch on per-step time by 1.2-2×, and roughly match
`torch.compile` once it's warmed up. Beating compiled PyTorch
requires either (a) better fused kernels than Inductor for our
specific shapes, or (b) fundamentally less work per step (which
means giving up some PyTorch generality — fine for toy).

## The backend landscape

Four backend families worth considering for the training path:

### A. ggml-cuda (current)

- **What it is:** vendored, in-tree, well-known.
- **Pros:** working, our chosen primitive surface, inference is
  great.
- **Cons:** training-backward gaps; single-stream dispatch; the
  perf attribution says we're capped by the dispatch model.

### B. Hand-rolled CUDA + cuBLASLt

- **What it is:** for the hot path (matmul / attention / norm /
  activation), call cuBLASLt directly for matmuls and ship custom
  CUDA kernels for everything else.
- **Pros:** ceiling is the hardware; per-shape optimal.
- **Cons:** **enormous** engineering surface. Estimate 6-12 months
  of full-time work to be competitive. Maintenance burden grows
  with every new op. cuBLASLt API is also non-trivial.

### C. Triton kernels (ahead-of-time compiled)

- **What it is:** OpenAI's Triton, a Python DSL that compiles GPU
  kernels to PTX. Used internally by `torch.compile`. We could
  pre-compile our hot kernels (matmul, fused softmax, AdamW step,
  layer-norm, GELU, etc.) and link the resulting `.cubin` files.
- **Pros:** much higher productivity than raw CUDA; competitive
  perf on common shapes (Triton's matmul ≈ cuBLAS within 5-10%);
  reusable kernels from the FlashAttention reference impls.
- **Cons:** Spinel can't run Python — Triton would have to be a
  **build-time** step. Each kernel needs an `.so` artifact toy
  links against. Bulky toolchain; CI/dev-env complexity.

### D. libtorch (PyTorch's C++ runtime, no Python)

- **What it is:** PyTorch ships a `libtorch.so` with ATen ops as
  C functions. Link it like any other C library; call ops directly
  from Spinel-compiled Ruby via FFI.
- **Pros:** **inherits PyTorch's entire kernel ecosystem** —
  FlashAttention, fused AdamW, cuBLASLt routing, all of it. By
  bypassing Python you avoid the interpreter overhead but keep the
  fast kernels.
- **Cons:** large dependency (~3 GB of `.so` files); tied to one
  vendor's release cycle; ATen's training graph machinery (autograd)
  doesn't trivially compose with our static-realize model — would
  need to use ATen as a kernel library only, not as an autograd
  engine. The autograd part toy already owns.

### E. Apple MLX (already in `backends-and-scale-2026-05-27.md`)

- **What it is:** Apple's tensor library for M-series, similar shape
  to ggml but with a tighter compiler. Not relevant for the CUDA
  question but worth noting in any multi-backend design.

## The "multi-backend abstraction" question

Pick-a-backend isn't really the question. The real question is:
**can we add a backend without rewriting the training graph?**

Today our training graph is built directly against `tinynn` (a thin
wrapper over ggml). To swap in libtorch or hand-rolled kernels for
specific ops, we'd either:
1. Replace `tinynn.tnn_matmul` with a polymorphic dispatch that
   picks ggml vs cuBLASLt vs Triton-cubin based on a backend
   selector.
2. Build a parallel `tinynn_torch` (etc.) and pick at session-new
   time which one to use. Composition stays the same; only the
   leaf-op implementations differ.

Option 2 is cleaner — it's the same shape as the existing
`tinynn` / `tinynn_cuda` / `tinynn_metal` split. Each backend ships
its own version of the FFI primitive table; the training graph code
calls into one of them per session.

**The hard part** isn't picking a backend — it's that the *training
graph itself* embeds backend assumptions: persistent-session
patterns, sched_alloc lifetimes, per-head decomposition. A "use
libtorch instead" backend has different lifetime semantics; the
graph code would need a thin abstraction over the persistent-tensor
allocation model.

## Concrete proposal (multi-month, staged)

### Stage 1 — measure what hurts (2-3 days)

Before any new backend, profile toy's current CUDA training step
at toy's actual workloads (SmolLM2-135M, Qwen-0.5B, ViT-Tiny). For
each, attribute time per ggml op. The result tells us *which* ops
need bespoke handling and which are fine on ggml. We have the
attribution framework already (`heavy-train-attribution-*.md`); just
needs new runs on the current code.

**Acceptance:** an attribution table that says "in SmolLM2-135M
training, X% of time is in matmul, Y% in softmax, Z% in AdamW step,
W% in scheduling overhead." Anything <5% is not worth optimizing.

### Stage 2 — libtorch matmul behind the same `tnn_matmul` (1-2 weeks)

Pick the highest-ROI op from Stage 1 (almost certainly matmul) and
build an alternate `libtorch_ffi.c` that exposes `tnn_matmul_libtorch`
backed by `at::matmul`. Compare against ggml-cuda's `tnn_matmul` on
the same shapes via the existing bench harness. If libtorch wins
≥20% on the canonical shape, the abstraction's worth the
infrastructure cost. If not, stop here.

**Acceptance:** A/B bench on 1024×1024×1024 matmul; libtorch
delta reported in BENCH metric.

### Stage 3 — backend selector + full session (3-4 weeks)

If Stage 2 proves the value, generalize: introduce a backend ID on
`tnn_session_new(backend_kind)` and route every hot op through the
chosen backend. Existing ggml backends become `backend_kind=0/1/2`
(CPU/CUDA/Metal). libtorch becomes `backend_kind=3`.

**Acceptance:** `DEVICE=cuda BACKEND=libtorch ./examples/example_train_from_scratch`
runs the training graph through libtorch and matches the ggml-cuda
numerical output within float-roundoff.

### Stage 4 — Triton-AOT for the rest of the hot path (2-3 months)

Per-op Triton kernels for fused operations ggml can't do well
(QKV-fused attention, fused AdamW with master copy, GELU+matmul
chains). Compiled at build time; toy links the `.cubin`s. This is
where we plausibly beat PyTorch — fused kernels for *our specific
shapes* better than Inductor's generic ones.

**Acceptance:** SmolLM2-135M training step time ≤ 0.8× the PyTorch
`torch.compile`-warmed-up baseline on GB10. (Aspirational; could
fall short. Worst case we hit parity and the kernel collection is
its own value.)

### Stage 5 — the "beat PyTorch" verification (ongoing)

Stage 4 should drop a `bench_vs_pytorch_training` heavy benchmark
that locks in the perf gain. Regressions show up at next push.

## What we *don't* try to win at (honest)

- Multi-GPU training. PyTorch + NCCL is two decades of
  engineering. Toy participates as a single-GPU node in any DP
  setup; we don't build our own collective ops.
- Production inference serving at scale. vLLM / TensorRT-LLM are
  doing the right work there.
- General PyTorch-eager shape coverage. Our shapes are fixed at
  realize time by design; arbitrary-shape inference is not the goal.

## Where this connects to existing tracks

- **GH#7 batching** is a prereq for stages 2-4. Without `B>1`, even
  a libtorch matmul backend is testing the wrong shape regime.
- **GH#9 mixed-precision (bf16)** opens up tensor-core paths.
  Currently we're f32 everywhere on CUDA, which is leaving 4-8×
  on the floor (5090's tensor cores at bf16 vs f32).
- **GH#10 activation recomputation** is on the same axis as the
  per-step-time question — it trades 30% compute for ~half
  activation memory. Once a real-data training run hits the memory
  wall, recomputation becomes free perf.
- **decoupled-diloco-research** assumes a single-machine training
  loop that's been optimised. The training-backend work makes the
  individual-node loop genuinely fast, which makes the DP overhead
  on top look smaller.

## Recommendation

**Stage 1 (measure) now. Stages 2-4 staged after batching (GH#7)
lands.** The "beat PyTorch on simpler workloads" claim is testable
in 1-2 weeks (Stage 2) and either earns its way into the roadmap
or doesn't. No need to commit 6 months of work on speculation.

If Stage 2 says "libtorch matmul beats ggml-cuda matmul by ≥20%
on our shapes," the staircase up to Stage 4 follows naturally. If
it doesn't, we know to stay on the ggml-cuda track and push upstream
patches (the ggml#1514 path) instead.

## Open questions for the user

- Is "beat PyTorch" a single-GPU goal, or do you mean
  PyTorch-FSDP / Megatron-class distributed training too? (The
  former is feasible in ~3 months; the latter is years.)
- libtorch as a backend adds ~3 GB of `.so` to the deploy footprint.
  Is that acceptable for the workloads where you'd want it? (For
  the "Mac → CUDA" heterogeneous case it definitely isn't on the
  Mac side.)
- What's the workload that's most important to be fastest at?
  Answer probably shapes the priority of Stages 3-4. (SmolLM2-135M
  from-scratch? ViT-Tiny pure-emb-vs-scratch? Qwen-410M warm
  start?)
