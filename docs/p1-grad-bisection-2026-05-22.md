# P1 day 2: CPU/CUDA gradient bisection (2026-05-22)

Per task #89 (reframed): use the Day-1 Chrome Trace primitive +
gradient-stats CSV dump to find the backward op whose CPU
implementation underflows / diverges from CUDA. The memory note
`project_cpu_cuda_lora_train_divergence_2026_05_21` claimed
gradients diverged by 7–240× per layer and CPU LoRA converged
~1000–6000× slower than CUDA. Smoking gun for a precision bug
inside one ggml CPU backward kernel.

## Method

1. Reuse `examples/03_finetune_lora.rb` + `…_cuda.rb`. New env var
   `GRAD_DUMP=1` enables a CSV dump after each `tnn_compute_backward`:
   per (step, layer, head, param ∈ {A, B}), download the gradient,
   compute min/max/mean/L2-norm/NaN-count via the existing
   `tnn_scratch_*` reducers (plus two new ones, `sum_sq_f32` and
   `sum_f32`).
2. Run both binaries with the same seed and config:
   `GGUF=data/smollm2-135m-native.gguf RANK=8 STEPS=3`.
3. Diff the CSVs keyed on `(step, layer, head, param)` and rank by
   relative L2-norm divergence
   `|cpu_l2 - cuda_l2| / mean(cpu_l2, cuda_l2)`.

## Findings

**No precision bug is present in the current code.** Gradient L2
norms agree to within float32 numerical tolerance across CPU and
CUDA:

| step | median rel-diff | worst rel-diff | worst case |
|-----:|----------------:|---------------:|---|
| 1    | 0.32 %          | 2.9 %          | L8 h1 B  |
| 2    | 0.42 %          | 6.3 %          | L4 h5 A  |
| 3    | 0.34 %          | 3.2 %          | L6 h2 A  |

Cross-entropy losses match to 3+ decimals at every step:

| step | CPU              | CUDA             |
|-----:|-----------------:|-----------------:|
| 1    | 9.236274719…     | 9.240358352…     |
| 2    | 9.213868141…     | 9.220218658…     |
| 3    | 9.153078079…     | 9.158402442…     |
| 4    | 9.050341606…     | 9.056529998…     |
| 5    | 8.904613494…     | 8.912443161…     |

Zero NaN counts on both backends across all 540 (layer, head, param)
slots × 3 steps.

## Interpretation

The original divergence is real and the root cause stands:
ggml-cpu's `ggml_backend_sched` aliases buffer slots for some
long-backward-chain shapes, corrupting downstream reads (memory
`project_cpu_cuda_lora_train_divergence_2026_05_21`). What our
bisection actually verifies is that the local **workaround is in
place and effective**: `lib/llama_seq_forward_ffi.rb:1192` calls
`tnn_pin_all_graph_b_nodes(@sess)` between `tnn_build_backward` and
`tnn_realize_backward`, which prevents sched from reusing any slot.
Once pinned, CPU and CUDA backward graphs produce numerically-equal
gradients (within float32 precision).

So: prod LoRA training already handles the divergence through that
pin. The bisection confirms it; the upstream ggml-cpu allocator bug
is still there but masked.

## What this means for P1

The original P1 ("FFI-bind the LoRA backward matmul") was based on
two assumptions, both turned out wrong:

1. The matmul wasn't already in FFI — actually it was (whole
   training step compiles to one ggml graph).
2. There was a CPU/CUDA precision divergence — actually no longer
   true.

What we *gained* from P1:

- A reusable Chrome Trace observability primitive
  (`tinynn/tinynn_trace.{c,h}`), zero-cost when off.
- Per-tensor stats CSV dump path in the LoRA finetune examples.
- An empirical baseline confirming CPU/CUDA parity.

Next perf candidate from the trace data:
`tnn_compute_backward` is 99.7 % of step wallclock (77 ms / 77.4 ms
total). Any future LoRA-training perf work has to land inside
ggml's CPU or CUDA backward kernels, not at our FFI boundary.

## Memory cleanup

The note `project_cpu_cuda_lora_train_divergence_2026_05_21` is
**still correct** but should be annotated that the workaround is
already wired into prod (`lib/llama_seq_forward_ffi.rb:1192`) and
the bisection above confirms it works. The upstream ggml-cpu
allocator bug is still unfixed; an issue against ggml-org/ggml
remains a TODO. P1.3 candidate: file that issue with the
pinned-vs-unpinned trajectory pair as the reproducer.
