# toy

<p align="center">
  <img src="toy_logo.png" alt="toy" width="240" />
</p>

A small transformer language model in Ruby. AOT-compiled to a native
binary by [Spinel](https://github.com/matz/spinel) (matz's Ruby AOT
compiler). Runs real HuggingFace models — from SmolLM2-135M to
Qwen2.5-7B — at output-identical fidelity to PyTorch.

The goal is to be **readable**: the whole forward pass fits on one
screen, every shape is annotated inline, the building blocks are
named after the math.

## Supported models

| Family            | Model              | Params | F32 | Q8_0 | CPU | CUDA F32 | CUDA Q8 |
| ----------------- | ------------------ | ------ | --- | ---- | --- | -------- | ------- |
| GPT-2             | DistilGPT-2        | 82M    | ✓   |      | ✓   | ✓        |         |
| GPT-2             | GPT-2 small        | 124M   | ✓   |      | ✓   | ✓        |         |
| Llama family      | SmolLM2-135M       | 135M   | ✓   | ✓    | ✓   | ✓        |         |
| Llama family      | SmolLM2-360M       | 360M   | ✓   |      | ✓   | ✓        |         |
| Llama family      | TinyLlama-1.1B     | 1.1B   | ✓   | ✓    | ✓   | ✓        |         |
| Llama family      | Llama-3.2-1B       | 1B     | ✓   |      | ✓   | ✓        |         |
| Llama family      | Llama-3.2-3B       | 3B     | ✓   |      | ✓   | ✓        |         |
| Llama family      | Mistral-7B-v0.2    | 7B     | ✓   | ✓    | ✓   |          | ✓       |
| Llama + QKV bias  | Qwen2.5-0.5B       | 0.5B   | ✓   | ✓    | ✓   | ✓        |         |
| Llama + QKV bias  | Qwen2.5-1.5B       | 1.5B   | ✓   | ✓    | ✓   | ✓        | †       |
| Llama + QKV bias  | Qwen2.5-3B         | 3B     | ✓   | ✓    | ✓   | ✓        | †       |
| Llama + QKV bias  | Qwen2.5-7B         | 7B     | ✓   | ✓    | ✓   |          | ✓       |

† Qwen2.5-1.5B/3B Q8 abort on CUDA at weight-load time: ggml-cuda's
quantized matmul requires `d_ff` aligned to 512, and those models'
`d_ff` (8960, 11008) aren't. F32 path works for all sizes.

Today's bench:

| Model              | CPU tok/s | CPU RSS  |
| ------------------ | --------- | -------- |
| SmolLM2-135M       | **47**    | 0.55 GB  |
| SmolLM2-360M       | **24**    | 1.41 GB  |
| Qwen2.5-0.5B Q8    | **21**    | 0.91 GB  |
| Qwen2.5-0.5B       | 14.5      | 1.89 GB  |
| TinyLlama-1.1B     | 9.6       | 3.92 GB  |
| Qwen2.5-1.5B Q8    | 9.0       | 2.22 GB  |
| Llama-3.2-1B       | 7.1       | 4.69 GB  |
| Qwen2.5-1.5B       | 6.4       | 5.80 GB  |
| Qwen2.5-3B Q8      | 4.9       | 3.98 GB  |
| Llama-3.2-3B       | 3.8       | 12.08 GB |
| Qwen2.5-3B         | 3.7       | 11.57 GB |
| Mistral-7B Q8      | 2.4       | 7.17 GB  |
| Qwen2.5-7B Q8      | 2.4       | 7.09 GB  |

7B-Q8 inference: **7.1 GB RSS** via zero-copy mmap of the GGUF
weights into both CPU and CUDA buffers (UVA on GB10).
Full bench notes: [`docs/archive/bench-gx10-2026-05-20.md`](docs/archive/bench-gx10-2026-05-20.md).

Next targets: Qwen3 dense (Qwen3.6-27B + WebWorld-8B), Qwen3 MoE
(35B-A3B), GLM-4.7-Flash — see [`docs/architecture.md`](docs/architecture.md).

## What it looks like

```ruby
# One transformer block: pre-LN → MHA → residual → pre-LN → FFN → residual.
def transformer_block(x, block)
  h_norm  = layer_norm(x, block.ln1_gamma, block.ln1_beta)
  attn    = self_attention(h_norm, block)
  x.add!(attn)

  h_norm2 = layer_norm(x, block.ln2_gamma, block.ln2_beta)
  ff      = feed_forward(h_norm2, block)
  x.add!(ff)
  x
end
```

Every model has an `algorithm_card` that emits Phuong-Hutter style
pseudocode (arXiv:2207.09238) with shape annotations:

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

`prep/card_to_code.rb` parses an algorithm card back into the Ruby
that constructs the model — the round-trip closes.

## Quickstart

```sh
make setup-ggml                                # one-time, ~30 s
make example_inference                         # build the smallest example
GGUF=data/qwen25-0.5b-native.gguf ./examples/example_inference
# → ids: 9707 11 847 829 374 264 220 16 15 ...
```

The [`examples/`](examples/README.md) directory has four short,
focused entry points — inference / train-from-scratch / LoRA
fine-tune / HTTP serve. Each is one Ruby file under 80 lines that
compiles to a single native binary.

For CUDA + Qwen2.5: `make setup-ggml-cuda` then build a
[`demos/qwen25_*`](demos/README.md) demo or the `_cuda` examples.

Requires Ruby, [Spinel](https://github.com/matz/spinel) at
`~/sites/spinel`, and a C compiler. `uv` installs itself for the
Python converter; or `pip install uv` first.

## What's in the box

| Path | What |
| --- | --- |
| [`examples/`](examples/README.md) | **Start here.** Four short entry points: inference, train-from-scratch, LoRA fine-tune, HTTP serve. |
| [`lib/toy.rb`](lib/toy.rb) | Building blocks: `Mat`, `LayerNorm`, `RMSNorm`, `Linear`, `Embedding`, `CausalSelfAttention`, `GQAttention`, `FFN`, `SwiGLU`, `RoPE` |
| [`lib/toy_gpt2.rb`](lib/toy_gpt2.rb) | `Toy::GPT2` — full HF GPT-2 in ~80 lines |
| [`lib/toy_smollm2.rb`](lib/toy_smollm2.rb) | `Toy::SmolLM2` — Llama-family path (SmolLM2 / TinyLlama / Qwen2.5) |
| `lib/toy_smollm2_ffi_kv*.rb` | KV-cache FFI mirror (CPU + CUDA) — the perf path |
| [`lib/toy_trainer.rb`](lib/toy_trainer.rb) | `Toy::Trainer` — training-loop wrapper |
| [`sig/toy.rbs`](sig/toy.rbs) | RBS type signatures |
| [`tinynn/`](tinynn/) | C/CUDA shim over ggml — FFI bridge, KV cache, mmap loader |
| [`vendor/ggml`](vendor/ggml) | Vendored ggml + local CUDA + autograd patches (BYO-pointer mmap, strided-cpy fix, concat backward) |
| [`vendor-patches/`](vendor-patches/README.md) | The patches above, persisted; setup-ggml applies them automatically. |
| [`demos/`](demos/) | Per-model and per-feature drivers — see [`demos/README.md`](demos/README.md) |
| [`tep_demo/`](tep_demo/) | OpenAI-compatible HTTP API via Tep+Spinel |
| [`docs/`](docs/) | Long-form notes: design docs, benchmarks, upstream issues |

## Reading the rest

- [`docs/INDEX.md`](docs/INDEX.md) — full tour of the docs (current
  reference, archived investigations, future-work design notes).
- [`docs/architecture.md`](docs/architecture.md) — the generic
  `TransformerLM` + per-model `Arch` struct.
- [`docs/loader-api.md`](docs/loader-api.md) — Mat-mediated vs direct
  GGUF→FFI loaders.
- [`docs/memory-design.md`](docs/memory-design.md) /
  [`docs/cuda-byo-pointer-design.md`](docs/cuda-byo-pointer-design.md) —
  why a 7B-Q8 model fits in ~7 GB of RSS.
- [`docs/roadmap/`](docs/roadmap/) — finetuning roadmap, deferred
  Phase 0.6 refactor, model targets.
- [`docs/archive/upstream/`](docs/archive/upstream/) — ggml/Spinel/Tep
  upstream issues we've filed.
- [`tep_demo/README.md`](tep_demo/README.md) — OpenAI-compatible HTTP API.

A toy you can read top-to-bottom that happens to run real models.
