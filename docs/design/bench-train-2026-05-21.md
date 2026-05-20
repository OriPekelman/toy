# Training-step bench + grad-probe — 2026-05-21

Two related diagnostics shipped this session:

1. **Per-step training bench** — answers "is M3 (reusable decode graph,
   task #69) helpful for training the way it would be for inference?"
2. **Per-layer LoRA grad probe** — narrows down the CPU/CUDA
   gradient-magnitude divergence first observed in
   `demos/smollm2_lora_train_ce*` (task #70).

## Training step phase breakdown

Bench: `demos/smollm2_lora_train_bench[_cuda]`. Setup: SmolLM2-135M,
LoRA r=16 on Q, all 30 layers, 540 opt_step nodes. 10 timed steps
after 3 warmup.

| Phase           | CPU (ms) | CUDA (ms) |
|---              |---:|---:|
| graph_reset     | 0.085 | 0.051 |
| upload (inputs) | 0.021 | 0.042 |
| compute_backward| **34.45** | **30.62** |
| download (loss) | 0.0004 | 0.007 |
| **total/step**  | **34.6** | **30.7** |
| **steps/sec**   | 28.9 | 32.6 |

**compute_backward is 99.7 % of step time on both backends.** The
training graph is built ONCE by `realize_backward` (6 ms one-time
cost) and re-used across all steps; per-step is just a `graph_reset`
(cheap walk to zero grad slots) + the compute.

This is dramatically different from the inference path
(`docs/design/bench-cuda-2026-05-21.md`), where `decode_step` rebuilds
the entire forward graph every call. Training does NOT have that
problem — the F1.1 `tnn_build_forward_only` + persistent backward
graph pattern already gives us graph-reuse.

**Implication for M3 (task #69):** the reusable-decode-graph refactor
buys nothing for training. M3 is purely an inference-time speedup.

**Implication for performance work:** training perf is bounded by
ggml's compute. Faster training means faster ggml backward kernels,
not graph plumbing. On GB10, training at 30 ms/step = 33 SGD
steps/sec on a 135M model. For a 1.5B model the same step is ~10×
slower (~3 steps/sec) — that's about as expected for backprop
through 30 layers of f32 attention.

## Per-layer grad probe (task #70 progress)

Probe: `demos/smollm2_lora_grad_probe[_cuda]`. Setup: identical LoRA-Q
init (seed=42, init_scale=0.01), one prefilled position, single
`compute_backward` call. Downloads layer-N-head-0's LoRA-A and LoRA-B
gradient tensors and prints max-abs magnitude per layer.

**Finding 1: `gradA = 0.0` on both backends, all layers.**

Expected: standard LoRA init has B=0, and ∂L/∂A = B^T · (∂L/∂(B·A·h)) · h^T.
With B=0, B^T=0, ∂L/∂A=0 mathematically. **A only starts moving once
B becomes nonzero.** This is a known shape of LoRA convergence:
B "wakes up" first, then A starts receiving signal. Both backends
agree on this.

**Finding 2: `gradB` is nonzero on both backends but differs in
magnitude by 10× to 7500× per layer, with the ratio bouncing
wildly.**

Sample (full table in `/tmp/cpu_grads.txt` / `/tmp/cuda_grads.txt`):

| layer | CPU gradB     | CUDA gradB    | ratio (CUDA/CPU) |
|---:|---:|---:|---:|
|  0 | 2.22e-6 | 5.33e-4 | 240 |
| 13 | 5.24e-5 | 1.80e-3 | 34 |
| 20 | 2.07e-5 | 1.44e-2 | 696 |
| 22 | 1.39e-5 | 1.44e-2 | 1036 |
| 25 | 5.07e-6 | 1.13e-2 | 2239 |
| 27 | 6.07e-4 | 2.07e-4 | **0.34** (CPU LARGER) |
| 29 | 9.66e-5 | 6.65e-3 | 69 |

**This is not a consistent scaling bug.** A multiplicative
mis-application of (say) a `1/sqrt(d)` factor would give a flat ratio
across layers. We see ratios from 0.3 to 2239. Layer 27 even has
CPU > CUDA. The grads look noisy on both sides; on CPU they're just
much smaller in magnitude overall.

**Hypotheses still open:**

1. **CPU-side gradient underflow at one or more ops in the chain.**
   `ggml_rms_norm_back` was ruled out (POC at
   `~/tmp/f1_rms_norm_back_poc.c` shows bit-exact match against an
   analytical reference). Suspects remaining: `ggml_soft_max_back`,
   `ggml_rope_ext_back`, `ggml_mul_mat`'s backward through the
   K_hist view, the CONCAT backward at the long-context shape (the
   POC was at trivial shape).
2. **CUDA-side gradient inflation** — possible but less likely. The
   CUDA loss-decrease curve looks like real LoRA training (B grows
   exponentially, then plateaus); CPU's looks like noise-driven
   drift.
3. **An interaction with the `cpy` op writing into KV-cache views.**
   The backward chain may differ between backends if `cpy` backward
   handles strided destinations differently on CPU vs CUDA.

**Recommended next step when picking up #70:**

- Walk the backward chain at layer 0 with a finer probe. Pick a
  fixed intermediate (e.g. dL/dt_q_pre[0]) on each backend; download
  and compare element-by-element, not just max-abs. If element-wise
  match exists at one node but not at the next, that node's CPU
  backward implementation has a bug.
- Compare against a hand-coded reference for the attention block
  (manageable since we control the math): given fixed x, gamma, W_q,
  W_k, W_v, W_o, compute analytical dL/dB and check both backends.

Memory entry:
[[project_cpu_cuda_lora_train_divergence_2026_05_21]] — keep updated
as the investigation narrows down.

## Why this still passes the `monotonic decrease` gate

Both `demos/smollm2_lora_train_ce` (CPU) and `_cuda` (CUDA) print
`VERDICT: PASS` because the loss decreases from step 1 to step 20 on
either backend. The CPU decrease is ~5e-5 per step at LR=0.5 (CUDA is
0.05 to 0.3 per step). Both technically converge in the sense the
gate asks about. The CPU gate is not a useful training signal.

If you want a stricter gate, change the acceptance to e.g.
`final < initial - 0.5` (or `final < 0.5 × initial`). At those
thresholds CUDA passes and CPU fails — which is what task #70 needs
to be diagnosed for.
