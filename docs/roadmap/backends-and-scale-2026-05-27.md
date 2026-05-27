# Backends & scale — future-directions design

**Date:** 2026-05-27. **Status:** thinking-out-loud doc, not a commitment.
**Context:** companion to the open multi-backend issues (#3 / #4 / #5)
that already cover the obvious shape. This doc captures what those
issues *don't* — algorithmic prerequisites, hybrid CPU/GPU offload,
non-ggml backend options, and the strategic question of how much of
this toy should actually own.

## Where we are

Three backends shipped, all via ggml:

| Backend | Inference | LoRA fine-tune | Full FT | From-scratch train |
| --- | --- | --- | --- | --- |
| CPU (ggml-cpu) | ✅ | ✅ | ✅ | ✅ |
| CUDA (ggml-cuda) | ✅ | ✅ | ✅ | ❌ (issue #152) |
| Metal (ggml-metal) | ✅ | ✅ | ✅ | ✅ |

Two batches of "scale" work are already filed but not yet started:
- **#3 mode 1** — replicate inference across N GPUs (each session pinned
  to one device). Smallest lift; the device-selection plumbing is the
  only blocker.
- **#4 mode 2** — split one model across N GPUs via ggml-sched's
  multi-backend array. Buys *capacity* (model fits), not throughput
  (pipeline bubble, no microbatching).
- **#5 mode 3** — data-parallel training across replicas + multi-node
  later. Honest dependency chain: **batching → DP → multi-node**.

The bench-vs-pytorch heavy attribution (May 2026) found the 2× wallclock
gap to PyTorch is upstream graph-orchestration cost (single-stream
ggml-cuda dispatch), not application-layer overhead. That's filed as
[upstream issue 05](../archive/upstream/issues-vendor/05-ggml-cuda-stream-overlap.md);
nothing toy can close from its side.

## The pre-requisites the open issues call out but don't expand

#5 lists **batching** as the gating dependency for DP training. Three
sub-things hide behind that word:

### 1. Batched training input

Today's training path processes **one sequence per step** (batch=1).
LlamaSeqForwardFFICache's `realize_for_*` allocates input shape
`[T, 1]`. Real batching means:

- `realize_for_*(cfg, T, B, ...)` — accept a batch dim
- `t_seq_token_ids` becomes `[T, B]`
- attention mask & RoPE generalise (the inner-loop already does this on
  the ggml side — the wiring change is at the realize + upload sites)
- labels tensor becomes `[T*B, vocab]` (same one-hot pattern)
- loss reduce is mean-over-batch (already does mean-over-T, just extend)

About 100-200 LOC across the seq-forward builders (CPU + CUDA + Metal).
**Independently valuable** — even single-GPU throughput goes up with
batch>1 (better matmul utilization), so this is the cheapest "real
training speedup" toy can ship before any multi-GPU work.

### 2. Gradient accumulation

Once batching exists, the next free wins:

- Accumulate gradients across N micro-batches before the optimizer step
- Implemented purely in the Ruby loop (skip the `OPT_STEP_ADAMW` op
  until the Nth iter)
- Trivial — ~20 LOC change in the example train loops

**Effective batch size = micro_batch × accum_steps** without the
memory cost of a single big batch. Standard technique for fine-tuning
when GPU memory is tight.

### 3. Mixed-precision training

Today: all training math is f32. Standard practice: bf16/f16 weights
and activations, f32 master copy of weights + optimizer state.

- ggml supports `GGML_TYPE_BF16` / `F16` for weights already (inference
  uses these via dequant in the kernels)
- Training would need `realize_for_random_init` to accept a target dtype
- Master-copy f32 stays for the AdamW state (m, v) — already standard
  in PyTorch's `torch.amp`

Memory halves; throughput typically improves 1.3-1.5× on tensor-core
hardware. Real engineering work (~200-400 LOC) but well-trodden territory.

## Multi-GPU specifics not in #3 / #4

### Optimizer-state sharding (ZeRO-1 style)

Pure DP (#5 mode 3) replicates: every replica holds a full copy of
weights + optimizer state. For AdamW that's **12 bytes/param of state**
on top of the weights — for a 1B-param model on 4 GPUs, that's 48 GB of
duplicate state across replicas.

ZeRO-1 shards the optimizer state across replicas: each holds only its
1/N slice, and after the per-replica AdamW step they all-gather the
updated weight slices. Memory drops to ~(weights + 12/N × params) per
GPU. Compute cost: one extra all-gather per step.

ZeRO-2 also shards the gradients; ZeRO-3 shards the weights themselves.
2 and 3 require deeper graph changes; ZeRO-1 is mostly an optimizer-loop
rewrite once #5 lands.

### Activation recomputation (gradient checkpointing)

Trade compute for memory: drop intermediate activations during forward,
recompute them during backward. ggml supports this via the
`ggml_set_param` / sched-recompute logic. Standard technique for
training long sequences; halves activation memory at ~30 % compute
overhead. Self-contained — doesn't need any multi-GPU plumbing.

## Hybrid CPU+GPU offload — the missing strategic option

The open issues don't cover **offload**: keep some training state on the
host (CPU memory) and stream it to the GPU on demand. Useful when the
GPU's memory is the bottleneck — common for fine-tuning larger models
on smaller hardware.

ggml has the building blocks: a backend buffer can live on the CPU
backend while the compute backend is CUDA; the sched-routing inserts
`cpy` nodes at the boundary. Today toy uses this for the BYO-pointer
mmap (weights stay in host memory, mapped UVA to GPU). Extending it
for **optimizer state** would mean:

- `m` and `v` Adam tensors allocated on the CPU backend, not the GPU
- Per-step: download grad → apply AdamW on CPU → upload updated weight
  delta back to GPU
- Roughly halves GPU memory pressure for AdamW training (since `m+v`
  are the same size as weights)

DeepSpeed-Zero-Offload does exactly this. The toy-flavour cost is the
sched-routing change + a CPU AdamW codepath; the ggml machinery already
handles cross-backend ops.

**Why this isn't filed yet**: it's only useful at scales toy doesn't
yet target (≥1B params). But it's the right design escape valve for
when the GB10's 121 GB unified memory runs out on (say) a 7B full fine-
tune.

## Non-ggml backend options (the "yet more backends" question)

The open issues all assume ggml as the substrate. Stepping outside ggml
is real work, with mixed returns. Options:

### Triton kernels for the hot path

NVIDIA's Triton is a Python-embedded DSL that compiles to PTX. It's
the dominant choice for "fast attention kernels" (FlashAttention v2/v3,
flash-decoding, etc.). The relevant question: could toy's training
hot path (matmul + attention + ffn) be Triton kernels invoked from
Spinel-compiled Ruby?

- **Pros:** State-of-the-art kernel quality. Closes the gap to PyTorch
  on the cuBLAS-vs-ggml-cuda matmul efficiency axis.
- **Cons:** Need Python at build time (Triton's compiler), then
  cache PTX. Heavy build dependency. Loses some of the readable
  ggml-only purity.
- **Verdict:** Wait. The upstream ggml-cuda fixes (or the cudaGraph
  capture path filed upstream) probably get us most of the way without
  the Python dependency.

### cuBLASLt direct

Skip ggml-cuda's dispatch entirely for matmul. Talk to cuBLASLt from
our C side. Would unblock the multi-stream story without waiting for
ggml-cuda.

- **Pros:** Single, well-tested cuBLAS path. No vendor patching.
- **Cons:** Have to reimplement op-graph orchestration for matmul-heavy
  paths. ggml's value is the abstraction; bypassing it for one op kind
  unravels the model.
- **Verdict:** Defer unless ggml-cuda upstream stays stuck for months.

### MLX backend (Apple)

Apple's MLX (https://github.com/ml-explore/mlx) is often faster than
ggml-metal for training on Apple Silicon (mature autograd, BNS
optimizations). A `lib/llama_seq_forward_ffi_mlx.rb` would mirror the
ggml-metal one against MLX's C API.

- **Pros:** Faster training on Mac dev boxes.
- **Cons:** Another full backend (~1k LOC of mirror), and Mac training
  is mostly a dev convenience — production-scale work happens on the
  gx10's CUDA.
- **Verdict:** Skip. Metal works; Mac is for fast iteration not
  production training.

### A "research backend" that's pedagogical, not fast

The framing inversion: toy is a **readable** transformer. The bench-vs-
PyTorch gate exists to make sure we don't drift slow, not to win speed
wars. A new backend whose explicit goal is to **illustrate parallelism
patterns** — pipeline-parallel with visible stage bubbles, tensor-
parallel with the all-reduce visible in the graph, ZeRO with the
sharding visible in the snapshot — would fit toy's mission better
than another speed-chasing backend.

- **Pros:** Aligns with the project's "Phuong-Hutter pseudocode" ethos.
  Tao's compare/describe consumers benefit directly (architecture views
  showing where the data goes).
- **Cons:** No one's asking for this yet; could be over-engineered.
- **Verdict:** Worth thinking about as the *long* future direction.
  Issues #3-#5 are the obvious near-term lifts; this is what comes
  after them.

## Multi-machine specifics not in #5

#5 acknowledges multi-node as "much later, a different category." The
specific transports:

- **NCCL** — NVIDIA's collective comms. Standard for PyTorch DDP. Has
  good Python bindings; from Spinel'd Ruby we'd need C FFI to
  `ncclAllReduce` / `ncclBroadcast` etc. NCCL handles the topology
  (NVLink within node, InfiniBand across nodes). Cost: small NCCL
  wrapper (~100-200 LOC C) + a bootstrap dance (initialise comm,
  exchange ranks).
- **GLOO** — Facebook's portable collective lib, CPU-first but works on
  GPUs too. More forgiving heterogeneous setups (mixed GPU/CPU nodes).
  Lighter dependency than NCCL.
- **Sockets** — hand-rolled all-reduce over TCP. Educational, slow,
  used by some research codebases (e.g., older PyTorch backends). Fine
  for "watch the algorithm work" but not for real multi-node training.
- **MPI** — generic, runs everywhere, GPU-aware variants (OpenMPI,
  MVAPICH). Heavy dependency for our use; mostly chosen when integrating
  with HPC clusters.

If/when toy goes multi-node, NCCL is the obvious choice for the gx10
class of hardware; GLOO is the fallback if the network topology is
heterogeneous. Sockets aren't worth implementing.

## Strategic question for the project

The open issues' "Open questions" section in #5 puts it well:

> Does toy want to own distributed training at all? It might be more
> in keeping with the project's "readable, single-file forward pass"
> ethos to do excellent single-GPU (and mode-1/2 multi-GPU) training,
> and delegate large-scale DP to an external orchestrator.

That's the right tension. The most coherent path forward is probably:

1. **Earn the prerequisites** that have value regardless of the
   parallelism decision: batching, gradient accumulation, mixed
   precision, activation recomputation. These are not "yet another
   backend" — they're "the single-GPU training story toy actually
   needs," and they unblock everything below.
2. **Mode 1 multi-GPU inference** (#3) when there's a 2-GPU host to
   exercise it on. Smallest lift; clearest value.
3. **Mode 2 layer split** (#4) when a model bigger than one GPU
   matters.
4. **Single-host DP training** (#5 mode 3a) **only if** a real
   experiment needs it — and probably with NCCL bindings as the
   transport, ZeRO-1 optimizer sharding from day one.
5. **Multi-node** is a different category — when it matters, evaluate
   whether toy hosts it or hands off to torchrun / DeepSpeed for the
   distributed shell while keeping toy as the per-rank model. The
   latter would keep toy honest about its scope.

The hybrid CPU+GPU offload (Zero-Offload) is its own arc, useful at
larger model scales. Worth a placeholder issue.

## What to file as concrete issues

| Title | Priority | Pre-req for |
| --- | --- | --- |
| `training: batched input (B>1)` | HIGH — independently valuable | #5 (DP) |
| `training: gradient accumulation` | LOW — trivial | (none, but free win) |
| `training: mixed-precision (bf16/f16)` | MEDIUM | larger models |
| `training: activation recomputation` | LOW — well-trodden | long-seq training |
| `training: ZeRO-1 optimizer sharding` | LATER | follows #5 |
| `training: CPU offload (ZeRO-Offload-style)` | LATER | 1B+ model fine-tune |
| `multi-machine: NCCL FFI + rendezvous` | LATER | #5 (3b) |

The three with concrete LOC estimates (batching, grad accum, mixed
precision) would meaningfully extend toy's *single-machine* training
story before any multi-GPU work needs to happen, and they're each
small enough to ship as a one-session task.

## Recommended near-term path

If the goal is "add efficient training and lay groundwork for multi-
GPU," the cheapest-with-most-value order is:

1. Ship **batching** (the actual #5 pre-req — without it, DP is moot).
2. Ship **gradient accumulation** (~20 LOC of bonus).
3. Ship **mixed precision** for training (moderate; the most direct
   speedup we can get pre-multi-GPU).
4. **Then** decide on #3 / #4 / #5 based on what hardware shows up.

The "yet more backends" framing is largely settled by #2-#5 — what's
actually missing is the *single-machine training maturity* that
those issues implicitly assume.
