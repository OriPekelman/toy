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

Latest GB10 (SmolLM2-135M, measured live, same session — 2026-06-12,
toy `67cf1e3`+toy#77 fix / spinel `a699cf9` / ggml `41e7949`):

| Workload | toy (CUDA) | PyTorch (CUDA) | ratio toy/pt |
| --- | --- | --- | --- |
| Full-FT training step (T=4) | 81.90 ms | 76.98 ms | **1.064×** |
| KV-cache decode | 80.7 tok/s | 112.8 tok/s | **1.399×** |

For a transformer you can read top-to-bottom, near-parity on the
training step at 135M and ~1.4× on decode is a healthy place to be.
(The CPU story is different — see "Caveats".)

> **Post-mortem (toy#77, fixed 2026-06-12).** The decode leg was dead
> 2026-05-31 → 2026-06-12: Spinel compiles `Bool#to_s` to a raw C
> string literal *without* the `\xff` static-string GC guard byte and
> registers the temp as a GC root, so two `KV_Q8.to_s`-style calls in
> one concat chain in `demos/qwen25_bench_cuda` made the next GC
> collection segfault in `sp_gc_mark` — or not, depending purely on
> link layout. Sidestepped with ternaries over guarded string
> literals; spinel-dev issue drafted. The same window also exposed the
> checker silently printing `ok` when a budgeted ratio had no
> measurement — `bench/check_vs_pytorch.rb` now prints a SKIPPED line
> per missing leg and exits non-zero (and `--update` refuses to
> silently shrink the budget).

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

## Heavy mode (ambitious workloads, optimization yardstick)

The light gate above runs in well under a minute and is meant to fire on
every meaningful commit. There is also a **heavy** variant that exercises
the libs at real scale — the kind of pressure you want when *choosing
between optimization strategies*.

```sh
# Toy-only, fast iteration loop (no PyTorch, no docker).
make bench-heavy                # gate vs bench/baselines_heavy.csv
make bench-heavy-update         # re-record after an intended change
make bench-heavy-report         # measure + print, no gate

# Adds the PyTorch ratio gate (slower; needs the docker image).
make bench-vs-pytorch-heavy
make bench-vs-pytorch-heavy-update
make bench-vs-pytorch-heavy-report
```

Two workloads (shared between both targets — no drift):

- **`heavy_train_lora_1p5b`** — LoRA-Q step on Qwen2.5-1.5B, seq=256,
  8 steps after warmup. Exercises the backward graph + AdamW at non-toy
  shape. Emits `step_ms_{mean,p95,stddev}`.
- **`heavy_infer_7b_q8`** — KV-cache greedy decode on Qwen2.5-7B-Q8 with
  `KV_Q8=1 FLASH_ATTN=1`, prefill=512, n_new=128. Exercises the
  Phase-2/3 mmap + Q8 dequant + the opt-in perf knobs at real model size.
  Emits `realize_ms`, `prefill_ms`, `decode_ms_{mean,p95}`,
  `decode_toks_per_sec`.

Latest GB10 numbers (recorded as both sets of baselines):

| Workload | toy | PyTorch | ratio toy/pt |
| --- | --- | --- | --- |
| LoRA-Q step, Qwen2.5-1.5B seq=256 | ~240 ms | ~120 ms | **~2.01×** |
| Decode, Qwen2.5-7B-Q8 (KV_Q8+FLASH) | ~12.7 tok/s | ~7.6 tok/s | **~0.60× (toy wins)** |

The 7B inference comparison is **toy-Q8 vs PT-F32**. That's the right
real-world comparison — Q8 is how toy users actually serve 7B — and it
shows up at decode where the workload is bandwidth-bound, so 4× less
weight bytes pays off. The training comparison is fair (both F32) and
identifies a ~2× headroom for optimization work on the seq-mode
forward+backward+AdamW path.

### Using heavy as an optimization yardstick

For run-over-run optimization iteration use `make bench-heavy-report`
(no gate, just numbers). The toy side has stddev < 1% on mean step time
and < 0.3% on mean decode time across runs on GB10, so deltas of 3–5%
from an optimization show up clearly. The CSV in
[`bench/baselines_heavy.csv`](../bench/baselines_heavy.csv) holds the
last `--update`-recorded numbers — re-run `--update` after an
intentional change to advance the baseline.

For weekly/release gating, `make bench-vs-pytorch-heavy` adds the
ratio-vs-PyTorch check, which catches the case where toy gets faster but
PyTorch gets faster by more (kernel cache refresh, ggml/torch upgrades).

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
