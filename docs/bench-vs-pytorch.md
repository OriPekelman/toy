# Comparing toy against PyTorch (the "old-stable" yardstick)

PyTorch is mature and well-optimised. We don't try to beat it — we use
it as a fixed reference to keep toy's design choices and defaults
honest, and to have a number we can call **"reasonable performance"**
in the single-machine single-GPU case.

The gate is the **ratio** (toy ÷ PyTorch), not absolute milliseconds.
Absolute step time is machine- and thermal-dependent (a fanless laptop
throttles ~7× under sustained load); the ratio is portable and is
exactly the signal we care about: *is the gap to old-stable reasonable,
and is it widening?*

## TL;DR

```sh
# On gx10 (where both engines live):
make bench-vs-pytorch          # measure both live, gate the ratio
make bench-vs-pytorch-update   # re-record the budget after an intended change
make bench-vs-pytorch-report   # measure + print, no gate
```

Latest GB10 (SmolLM2-135M, measured live, same session):

| Workload | toy (CUDA) | PyTorch (CUDA) | ratio toy/pt |
| --- | --- | --- | --- |
| Full-FT training step (T=4) | ~77 ms | ~77 ms | **~1.0× (parity)** |
| KV-cache decode | ~82 tok/s | ~113 tok/s | **~1.38×** |

For a transformer you can read top-to-bottom, parity on the training
step and ~1.4× on decode at 135M is a healthy place to be. (The CPU
story is different — see "Caveats".)

## What it measures

The same model on both sides — a Llama at **SmolLM2-135M** dims — across
two workloads that exercise toy's real code paths:

- **train** — one full fine-tune step (forward + cross-entropy +
  backward + AdamW), T=4 positions, batch=1. Matches
  `demos/seq_train_bench_cuda MODE=ft`.
- **infer** — KV-cache greedy decode, batch=1, one token per step,
  prefill + warmup excluded. Matches `demos/qwen25_bench_cuda`.

The PyTorch side uses `transformers`' `LlamaForCausalLM` with **random
weights** — throughput is weight-independent, so there's no checkpoint
download or tokenizer to match; we're comparing a faithful Llama
(RMSNorm / RoPE / GQA / SwiGLU / KV-cache), not an approximation.

## How it's wired (the two-environment reality)

On gx10, the two engines live in different places, and the orchestrator
bridges them — this is the only real complexity, and it's contained:

- **toy CUDA** runs **natively** on the gx10 host (Spinel-compiled
  binaries + `tinynn/libtinynn_ggml_cuda.a`). The orchestrator runs the
  existing `demos/*_cuda` binaries and parses their output.
- **PyTorch** runs in the **`gx10/dev-pytorch` container** (it has
  `torch` + `transformers`; the host does not). The orchestrator invokes
  it via `docker run`.

That container invocation is the single environment-specific knob,
isolated in one env var:

```sh
PT_CMD='docker run --rm --gpus all --ipc=host -v "$PWD":/w -w /w \
  gx10/dev-pytorch:latest python3 bench/ref_pytorch.py --workload both --device cuda'
```

Everything else is engine-agnostic. (toy has no `docker-compose.yml`, so
`gx-run` is not used for the toy side — toy builds/runs CUDA on the host
directly.)

## Files

| File | Role |
| --- | --- |
| [`bench/ref_pytorch.py`](../bench/ref_pytorch.py) | The PyTorch yardstick. `--workload train\|infer\|both`, `--device cuda\|cpu\|mps`. Emits `BENCH pt_*` lines. |
| [`bench/check_vs_pytorch.rb`](../bench/check_vs_pytorch.rb) | Orchestrator. Runs both engines, computes `ratio_*_toy_over_pt`, soft-gates against the budget. `--update` / `--report`. |
| [`bench/baselines_vs_pytorch.csv`](../bench/baselines_vs_pytorch.csv) | The ratio **budget** (same schema as `bench/baselines.csv`): `metric,value,unit,direction,tolerance_pct`. `direction=lower` (closer to / beating PyTorch is better); `tolerance_pct` is how much the ratio may grow before it fails. |

This mirrors the local `make bench` machinery (BENCH lines → CSV →
per-metric tolerance) so there's one mental model for both.

## Reading the gate

```
ratio_train_toy_over_pt =  0.975×  (budget 1.031×,  -5.4 %, tol +25 %)  [ok]
ratio_infer_toy_over_pt =  1.350×  (budget 1.379×,  -2.1 %, tol +25 %)  [ok]
ok — toy stays within budget of PyTorch
```

A ratio of `1.0×` means parity; `1.38×` means toy is 38 % slower than
PyTorch on that workload. The bench **fails** only if a ratio grows more
than `tolerance_pct` past its recorded budget — i.e. a change made the
gap to old-stable meaningfully worse. Re-baseline intended changes with
`make bench-vs-pytorch-update`.

Always measured **live, both engines, same session** — never compared to
a stale figure. (An older doc listed toy's train step at 108.5 ms; live
it's ~77 ms. Trust the live ratio.)

## Caveats and what to extend

- **Small models on CPU**: at ≤135M, toy's ggml-CPU path can *beat*
  CUDA (CUDA's per-step launch overhead dominates tiny matmuls;
  crossover ≈ d_model 896). So "use the GPU" is not automatically the
  reasonable default at small sizes.
- **CPU gap is larger (~2×)**: where matmuls are tiny, toy's per-head
  attention decomposition (separate GEMMs + concat per head, vs a fused
  qkv) costs more relative to kernel time. On the GPU at 135M this
  matters less — hence parity on train.
- **Extending** (each ~one line in the orchestrator + one budget row):
  a larger model (Qwen2.5-1.5B, where the GPU win is clearer), a CPU
  budget for the laptop case, or a backend-selection default that picks
  the faster toy backend per model size rather than assuming CUDA.
