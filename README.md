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

| Family            | Model           | Params | F32 | Q8_0 | CPU | CUDA |
| ----------------- | --------------- | ------ | --- | ---- | --- | ---- |
| GPT-2             | DistilGPT-2     | 82M    | ✓   |      | ✓   | ✓    |
| GPT-2             | GPT-2 small     | 124M   | ✓   |      | ✓   | ✓    |
| Llama family      | SmolLM2-135M    | 135M   | ✓   | ✓    | ✓   | ✓    |
| Llama family      | SmolLM2-360M    | 360M   | ✓   |      | ✓   | ✓    |
| Llama family      | TinyLlama-1.1B  | 1.1B   | ✓   | ✓    | ✓   | ✓    |
| Llama + QKV bias  | Qwen2.5-0.5B    | 0.5B   | ✓   | ✓    | ✓   | ✓    |
| Llama + QKV bias  | Qwen2.5-1.5B    | 1.5B   | ✓   | ✓    | ✓   | ✓    |
| Llama + QKV bias  | Qwen2.5-3B      | 3B     | ✓   | ✓    | ✓   | ✓    |
| Llama + QKV bias  | Qwen2.5-7B      | 7B     | ✓   | ✓    | ✓   | ✓    |

7B-Q8 inference: **7.4 GB RSS** via zero-copy mmap of the GGUF
weights into both CPU and CUDA buffers (UVA on GB10). Next targets:
Llama-3.2, Qwen3 dense, Qwen3 MoE — see
[`docs/design/arch-struct.md`](docs/design/arch-struct.md).

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
make setup-ggml                                # ~30 s
./prep/convert_smollm2_to_gguf.py              # writes data/smollm2-135m-f32.gguf
./prep/smollm2_tokens.py encode "Once upon a time"

make smollm2_kv && ./demos/smollm2_kv
./prep/smollm2_tokens.py decode
# → "Once upon a time, there was a little girl named Lily..."
```

For CUDA + Qwen2.5: `make setup-ggml-cuda` then build a `demos/qwen25_*`
demo. See [`demos/README.md`](demos/README.md).

Requires Ruby, [Spinel](https://github.com/matz/spinel) at
`~/sites/spinel`, and a C compiler. `uv` installs itself for the
Python converter; or `pip install uv` first.

## What's in the box

| Path | What |
| --- | --- |
| [`lib/toy.rb`](lib/toy.rb) | Building blocks: `Mat`, `LayerNorm`, `RMSNorm`, `Linear`, `Embedding`, `CausalSelfAttention`, `GQAttention`, `FFN`, `SwiGLU`, `RoPE` |
| [`lib/toy_gpt2.rb`](lib/toy_gpt2.rb) | `Toy::GPT2` — full HF GPT-2 in ~80 lines |
| [`lib/toy_smollm2.rb`](lib/toy_smollm2.rb) | `Toy::SmolLM2` — Llama-family path (SmolLM2 / TinyLlama / Qwen2.5) |
| `lib/toy_smollm2_ffi_kv*.rb` | KV-cache FFI mirror (CPU + CUDA) — the perf path |
| [`lib/toy_trainer.rb`](lib/toy_trainer.rb) | `Toy::Trainer` — training-loop wrapper |
| [`sig/toy.rbs`](sig/toy.rbs) | RBS type signatures |
| [`tinynn/`](tinynn/) | C/CUDA shim over ggml — FFI bridge, KV cache, mmap loader |
| [`vendor/ggml`](vendor/ggml) | Vendored ggml + local CUDA patches (BYO-pointer mmap, strided-cpy fix) |
| [`demos/`](demos/) | End-to-end Ruby drivers — see [`demos/README.md`](demos/README.md) |
| [`tep_demo/`](tep_demo/) | OpenAI-compatible HTTP API via Tep+Spinel |
| [`docs/`](docs/) | Long-form notes: design docs, benchmarks, upstream issues |

## Reading the rest

- [`docs/design/arch-struct.md`](docs/design/arch-struct.md) — design for the generic `TransformerLM` + per-model `Arch` struct (next-up refactor)
- [`docs/HF_GPT2.md`](HF_GPT2.md) — the long story of getting GPT-2 to run identically to PyTorch
- [`docs/upstream-issues/`](docs/upstream-issues/) — ggml/llama.cpp upstream contributions (HF/GGUF byte-equivalence, CUDA buffer_from_ptr, cpy-into-strided fix)
- [`docs/bench-gx10-2026-05-16.md`](docs/bench-gx10-2026-05-16.md) — perf numbers
- [`tep_demo/README.md`](tep_demo/README.md) — OpenAI-compatible HTTP API

A toy you can read top-to-bottom that happens to run real models.
