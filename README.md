# toy

<p align="center">
  <img src="toy_logo.png" alt="toy" width="240" />
</p>

**v0.7.0-pre-alpha** · early signal · not API-stable
&nbsp;·&nbsp; [CHANGELOG](CHANGELOG.md)
&nbsp;·&nbsp; [docs](docs/architecture.md)
&nbsp;·&nbsp; [op coverage](docs/coverage.md)

A transformer LM framework in Ruby, [Spinel](https://github.com/matz/spinel)-compiled
to native binaries. A plain CRuby CLI (`bin/toy`) drives Spinel-compiled
compute runners; the model algorithms are organized in five layers
(primitives → blocks → archs → recipes), and **every layer is gated
bit-identical** against a reference. Runs real HuggingFace models —
SmolLM2, Llama 3, Qwen 2.5, Qwen 3, Mistral, Gemma 2, OLMoE (MoE) — at
output-identical fidelity to PyTorch. CPU, CUDA, Metal.

End-to-end as single native binaries:

- **Inference** — KV-cache greedy decode on CPU/CUDA/Metal. F32 or Q8
  with zero-copy mmap. Opt-in: Q8 K+V cache, fused flash attention.
- **Training** — from-scratch trainer over a Llama-shape `TransformerLM`.
- **Eval** — per-token logprobs / top-K scoring.
- **Serve** — OpenAI-compatible HTTP API (CPU), token IDs in / out.
- **Discover** — auto-finds GGUFs in HuggingFace, Ollama, LM Studio,
  and project-local `data/` caches.

The goal is **readable**: the whole forward pass fits on one screen,
every shape is annotated, building blocks are named after the math.

## What it looks like

```ruby
# One transformer block (GPT-2 family — Llama swaps LN→RMSNorm and
# adds RoPE inside self_attention; same one-screen shape).
def transformer_block(x, block)
  h  = layer_norm(x, block.ln1_gamma, block.ln1_beta)
  x.add!(self_attention(h, block))
  h2 = layer_norm(x, block.ln2_gamma, block.ln2_beta)
  x.add!(feed_forward(h2, block))
  x
end
```

Every model has an `algorithm_card` (in `lib/toy_gpt2.rb`,
`lib/toy_smollm2.rb`) emitting Phuong–Hutter style pseudocode
(arXiv:2207.09238) with shape annotations:

```
Algorithm: Toy::GPT2.forward(x, p_start)      [HF GPT-2 family]
  Input:    x ∈ {1..V}^T   (token IDs)
  Output:   P ∈ R^{T×V}   (logits)
  Hyper:    V=50257 D=768 H=12 D_f=3072 N=6 ctx=1024
   1: e ← W_e[x] + W_p[p_start : p_start+T]                  e ∈ R^{T×D}
   2: for ℓ ← 1, …, N do
   3:    e ← e + Attn(LN(e; γ_ℓ^1, β_ℓ^1, ε); θ_ℓ^attn)      e ∈ R^{T×D}
   4:    e ← e + FFN (LN(e; γ_ℓ^2, β_ℓ^2, ε); θ_ℓ^ffn )      e ∈ R^{T×D}
   5: end for
   6: e ← LN(e; γ_f, β_f, ε)                                 e ∈ R^{T×D}
   7: P ← e · W_e^⊤                                          P ∈ R^{T×V}
```

`prep/card_to_code.rb` parses an algorithm card back into the Ruby that
constructs the model — the round-trip closes. `toy describe <model>`
renders the card straight from a GGUF's arch metadata.

## Quickstart

Requires Ruby, [Spinel](https://github.com/matz/spinel), and a C
compiler. The CLI is plain MRI Ruby; the compute runners are built on
demand by `toy install` (or `make`).

```sh
toy install                                  # build/verify the CPU backend
toy fetch ggml-org/models \
    tinyllamas/stories15M-q4_0.gguf          # grab a tiny model into ./data
toy infer data/stories15M-q4_0.gguf \
    --prompt "Once upon a time"              # greedy decode
```

The 9 commands (the `COMMANDS` registry in `lib/toy/core/cli.rb` is the
single source of truth — `toy --manifest` emits the same surface as
machine-readable JSON, format tag `toy/manifest-v1`):

```sh
toy new <path> [--force] [--json]            # scaffold a conventional project tree
toy install [--json]                         # build/verify the CPU backend
toy list [--json]                            # GGUFs in HF / Ollama / LM Studio / ./data
toy describe <model> [--json]                # GGUF metadata → arch-derived Card
toy fetch <hf-repo> [file.gguf] [--json]     # download a GGUF + data/ symlink
toy infer <model> [--prompt] [--n] [--json]  # greedy decode (defaults: "Once upon a time", n 16)
toy train from-scratch [--steps] [--seed] [--out] [--json]   # records runs/<id>/ + loss curve
toy eval <model> [--top-k] [--json]          # per-token logprobs (default top-K 5)
toy serve <model> [--port] [--name]          # OpenAI-compatible HTTP API (CPU)
```

Global flags: `--manifest`, `--help` / `-h`, `--version`. Exit codes:
`0` ok · `2` bad input (unknown command, missing required arg) · `1`
execution failure (GGUF unreadable, scaffold target exists, …).

If `toy list` shows nothing, populate a cache any of these ways:

```sh
toy fetch bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf
# …or huggingface-cli download / hf download, ollama pull, LM Studio —
# they all land in caches the next `toy list` will see.
```

`toy fetch` also drops a `data/<file>.gguf` symlink so default paths
resolve without extra typing. Set `TOY_MODEL_DIR` to add a search path.

Then run, train, eval, and serve:

```sh
toy infer data/llama-3.2-1b-tok.gguf --prompt "The capital of France is"
# → The capital of France is Paris. The capital of Germany is …

KV_Q8=1 FLASH_ATTN=1 \
  toy infer data/qwen3-1.7b-tok.gguf --prompt "Hi" --n 32   # opt-in perf knobs

toy train from-scratch --steps 20 --seed 0   # writes runs/<id>/ (weights + events + loss curve)
toy eval data/SmolLM2-135M-Instruct-Q8_0.gguf --top-k 5
toy serve data/SmolLM2-135M-Instruct-Q8_0.gguf --port 4567 --name smol
```

`toy infer` speaks text when the GGUF was converted with a tokenizer
(the `*-tok.gguf` variants). Without an embedded tokenizer the runner
falls back to a fixed token-ID prompt and prints raw IDs.

`toy serve` exposes OpenAI-shape `/v1/{models,completions,embeddings}`
over the same KV-cache inference path. It speaks **token IDs in / token
IDs out** (server-side chat-templating is deferred):

```sh
curl -X POST http://127.0.0.1:4567/v1/embeddings \
  -H 'Content-Type: application/json' -d '{"input":[1,2,3]}'
```

The serve runner lives at `lib/toy/serve/openai/` → `libexec/toy-serve`.

## Supported models

| Model              | Family               | Params      | F32 | Q8_0 | CPU | CUDA F32 | CUDA Q8 | Metal F32 |
| ------------------ | -------------------- | ----------- | --- | ---- | --- | -------- | ------- | --------- |
| DistilGPT-2        | GPT-2                | 82M         | ✓   |      | ✓   | ✓        |         | ‡         |
| GPT-2 small        | GPT-2                | 124M        | ✓   |      | ✓   | ✓        |         | ‡         |
| SmolLM2-135M       | Llama family         | 135M        | ✓   | ✓    | ✓   | ✓        |         | ✓         |
| SmolLM2-360M       | Llama family         | 360M        | ✓   |      | ✓   | ✓        |         | ‡         |
| TinyLlama-1.1B     | Llama + SPM-BPE tok  | 1.1B        | ✓   | ✓    | ✓   | ✓        |         | ‡         |
| Llama-3.2-1B       | Llama family         | 1B          | ✓   |      | ✓   | ✓        |         | ‡         |
| Llama-3.2-3B       | Llama family         | 3B          | ✓   |      | ✓   | ✓        |         | ‡         |
| Mistral-7B-v0.2    | Llama + SPM-BPE tok  | 7B          | ✓   | ✓    | ✓   |          | ✓       |           |
| Qwen2.5-0.5B       | Llama + QKV bias     | 0.5B        | ✓   | ✓    | ✓   | ✓        |         | ‡         |
| Qwen2.5-1.5B       | Llama + QKV bias     | 1.5B        | ✓   | ✓    | ✓   | ✓        | †       | ‡         |
| Qwen2.5-3B         | Llama + QKV bias     | 3B          | ✓   | ✓    | ✓   | ✓        | †       | ‡         |
| Qwen2.5-7B         | Llama + QKV bias     | 7B          | ✓   | ✓    | ✓   |          | ✓       |           |
| Qwen3-0.6B         | Qwen3 + QK-norm      | 0.6B        | ✓   |      | ✓   | ✓        |         | ‡         |
| Qwen3-1.7B         | Qwen3 + QK-norm      | 1.7B        | ✓   | ✓    | ✓   | ✓        |         | ‡         |
| Qwen3-4B           | Qwen3 + QK-norm      | 4B          | ✓   |      | ✓   |          |         | ‡         |
| OLMoE-1B-7B-Instr  | Llama + MoE + QK-norm⁂  | 7B (1B act) |     | ✓    | ✓   |          |         | ‡         |
| Gemma 2-2b-it      | Llama + Gemma extras⁂⁂  | 2B          |     | ✓    | ✓   |          |         | ‡         |

⁂ OLMoE uses OLMoE-style QK-norm (gamma at `[d_model]`, per-head
sliced) and 64 experts × top-8 routing. MoE expert weights **must be
Q8_0** — Q4_K / Q5_K / Q6_K trigger a known ggml `mul_mat_id` bug
([ggml#1506](https://github.com/ggml-org/ggml/issues/1506)). The loader
emits a runtime warning if expert weights are K-quantized.

⁂⁂ Gemma 2 needs four extras: embedding scaling by sqrt(d_model),
logit soft-cap (tanh) on attention + final logits, pre+post-norms on
each sublayer, and per-layer alternating SWA/full attention. All
auto-detected from the GGUF (`gemma2.*` metadata). Its tokenizer is
SPM-Unigram (no merges), enabled automatically.

† Qwen2.5-1.5B/3B Q8 abort on CUDA at weight-load time: ggml-cuda's
quantized matmul requires `d_ff` aligned to 512, and those models'
`d_ff` (8960, 11008) aren't. The F32 path works for all sizes.

‡ Metal: validated end-to-end only on SmolLM2-135M F32 (bit-identical
to CPU). Other Llama-family models should work — same FFI surface,
same graph builder, same ggml-metal kernel coverage — but haven't been
smoked. GPT-2-family isn't wired on Metal. Quantized weights on Metal
are untested. The Metal path uses copy-load rather than mmap, so
multi-GB models pay the copy cost.

The canonical "what's actually wired" reference is
[`docs/reference/coverage.md`](docs/coverage.md).

### Tokenizer / RoPE coverage

Three tokenizer flavors are auto-detected from the GGUF:

- **Byte-level BPE** (GPT-2 byte-map): GPT-2 family, SmolLM2,
  Llama-3.2, Qwen2.5, Qwen3.
- **SPM-BPE** (SentencePiece with merges): TinyLlama-1.1B, Mistral.
- **SPM-Unigram** (SentencePiece with scores, no merges): Gemma 2.
  Greedy longest-match; auto-prepends BOS when
  `tokenizer.ggml.add_bos_token` is set.

RoPE scaling (YaRN / llama3-style / linear) propagates through the
converter and loader, so Llama-3.x and Qwen3 long-context
configurations work without extra wiring.

### Opt-in performance features

Environment variables the runners read:

- **`KV_Q8=1`** — K and V caches stored as Q8_0 (3.75× smaller),
  auto-enables flash attention. Validated on Qwen3-1.7B: 13% faster
  wallclock at N_NEW=32, bit-identical token output.
- **`FLASH_ATTN=1`** — fused `ggml_flash_attn_ext` per Q-head. 12%
  faster on Qwen3-1.7B at N_NEW=32 vs the scale→softmax→matmul triplet.
- **`TOY_EVENTS=path.jsonl`** — emit the structured `toy/v1` event
  stream for live TUIs and post-run analysis (see
  [`docs/events.md`](docs/events.md)).

## Backends

CPU is the default and the gated reference. CUDA (GB10 / sm_121) and
Metal (Mac) mirror the L1–L3 algorithms; mirrors are generated from
`MIRRORABLE` markers and held bit-identical by `make verify-mirrors`.
See [`docs/reference/backends.md`](docs/reference/backends.md).

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — the five-layer
  algorithm stack and how the CLI shells to compute runners.
- [`docs/cli.md`](docs/cli.md) — the 9 commands, flags, exit codes,
  and the manifest contract.
- [`docs/authoring.md`](docs/authoring.md) — adding a primitive, block,
  arch, or recipe; the algorithm-card / `card_to_code` round-trip.
- [`docs/gating.md`](docs/gating.md) — the bit-identical gates and
  fixtures (`examples/smoke_*`, `prep/*_gate.rb`).
- [`docs/dependencies.md`](docs/dependencies.md) — Spinel, the vendored
  ggml patches, and consuming toy as a gem via spinelgems.
- [`docs/events.md`](docs/events.md) — the `toy/v1` event schema
  (`runs/<id>/events.jsonl`).
- [`docs/roadmap.md`](docs/roadmap.md) — deferred work and live
  research directions.
- [`docs/reference/`](docs/coverage.md) — op coverage,
  backends, the loader API, and memory design.
- [`examples/`](examples/README.md) — focused, single-file entry points
  compiled to native binaries.

## Reproducibility gates

- `make verify-mirrors` — CUDA/Metal mirrors stay bit-identical to CPU.
- `make check-cards` — Ripper-based drift detector: every `Toy::` class
  with both `def forward` and `def algorithm` keeps the two in
  lock-step.
- `make bench` — Spinel-compiled benches; emits `BENCH metric value`
  lines compared against `bench/baselines.csv`.
- `make bench-vs-pytorch` / `make bench-heavy` — ratio-not-ms comparison
  against a reference PyTorch run (see
  [`docs/reference/backends.md`](docs/reference/backends.md)).

A toy you can read top-to-bottom that happens to run real models.
