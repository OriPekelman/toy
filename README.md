# toy

<p align="center">
  <img src="toy_logo.png" alt="toy" width="240" />
</p>

**v0.2.0-pre-alpha** — early signal. Not API-stable. See
[`CHANGELOG.md`](CHANGELOG.md) for what's working today.

A small transformer language model in Ruby. AOT-compiled to a native
binary by [Spinel](https://github.com/matz/spinel) (matz's Ruby AOT
compiler). Runs real HuggingFace models — from SmolLM2-135M to
Mistral-7B and Qwen3 — at output-identical fidelity to PyTorch.

It does these things end-to-end, each as a single native binary:

- **Run pretrained models** — KV-cache decode (CPU + CUDA) on F32 or Q8,
  zero-copy mmap from the GGUF on disk.
- **Reuse models already on the machine** — auto-discovers GGUFs in
  the HuggingFace, Ollama, LM Studio, and project-local caches.
- **Train from scratch** — `Toy::Trainer` over a `TransformerLM` (CPU).
- **Fine-tune with LoRA** — sequence-mode forward + backward + AdamW
  in one compute (CPU + CUDA).
- **QLoRA** — same fine-tune path against a Q8 base; only the adapter
  is F32 (CPU; CUDA QLoRA pending a vendor patch).
- **Serve over HTTP** — Tep+Spinel OpenAI-compatible API.

The goal is to be **readable**: the whole forward pass fits on one
screen, every shape is annotated inline, the building blocks are
named after the math.

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
make example_list_models                       # see what's already cached locally
./examples/example_list_models                 # HF / Ollama / LM Studio / ./data
```

If nothing shows up, grab one with the helper:

```sh
prep/fetch_model.sh bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf
```

…or use `huggingface-cli download`, `ollama pull <name>`, or LM
Studio directly. They all land in caches the next `example_list_models`
will see.

Then run a model:

```sh
make example_inference
GGUF=data/llama-3.2-1b-tok.gguf PROMPT="The capital of France is" ./examples/example_inference
# → text: The capital of France is Paris. The capital of Germany is …
```

`example_inference` speaks text when the GGUF was converted with
`--with-tokenizer` (the `*-tok.gguf` variants). The tokenizer
auto-detects byte-level BPE (GPT-2 / Llama-3 / Qwen) vs
SentencePiece (Llama-1/2 / Mistral / TinyLlama). Without an
embedded tokenizer it falls back to a fixed token-ID prompt and
prints raw IDs.

The [`examples/`](examples/README.md) directory has five focused entry
points: inference, train-from-scratch, LoRA / QLoRA fine-tune, HTTP
serve, model discovery. Each is one Ruby file under ~100 lines that
compiles to a single native binary.

For CUDA + larger models: `make setup-ggml-cuda` then `*_cuda`
example variants.

Requires Ruby, [Spinel](https://github.com/matz/spinel) at
`~/sites/spinel`, and a C compiler. `uv` installs itself for the
Python converter; or `pip install uv` first.

## Reading the rest

- [`examples/`](examples/README.md) — start here.
- [`docs/INDEX.md`](docs/INDEX.md) — tour of the docs (current
  reference, future work, archive).
- [`tep_demo/`](tep_demo/README.md) — OpenAI-compatible HTTP API.
- [`demos/`](demos/README.md) — per-model and per-feature drivers
  (one per concrete validation run).
- [`tinynn/`](tinynn/README.md) — the C/CUDA shim over ggml.

## Reproducibility gates

Three local-runnable gates catch regressions before they ship.
Run before pushing perf-sensitive or model-shape changes.

- `make bench` — three Spinel-compiled benches (LoRA training step,
  inference toks/sec, tokenizer encode µs/tok). Each emits
  `BENCH metric value` lines; the orchestrator compares to
  `bench/baselines.csv` and exits non-zero past a per-metric
  tolerance (±15 % default). `make bench-update` re-baselines.
- `make check-cards` — Ripper-based drift detector. Verifies every
  `Toy::` class with both `def forward` and `def algorithm` keeps
  the two in lock-step (activation-token mismatches + matmul-
  presence collapse). Catches the case where someone changes
  forward without updating the algorithm card or vice versa.
- `TRACE=path.json ./examples/example_finetune` — Chrome Trace
  Format observability primitive on the FFI hot path. ~5 ns per
  begin/end pair when off (zero-overhead measured), 12 % when on.
  Open the output in https://perfetto.dev .

## Supported models

| Model              | Family            | Params | F32 | Q8_0 | CPU | CUDA F32 | CUDA Q8 |
| ------------------ | ----------------- | ------ | --- | ---- | --- | -------- | ------- |
| DistilGPT-2        | GPT-2             | 82M    | ✓   |      | ✓   | ✓        |         |
| GPT-2 small        | GPT-2             | 124M   | ✓   |      | ✓   | ✓        |         |
| SmolLM2-135M       | Llama family      | 135M   | ✓   | ✓    | ✓   | ✓        |         |
| SmolLM2-360M       | Llama family      | 360M   | ✓   |      | ✓   | ✓        |         |
| TinyLlama-1.1B     | Llama family      | 1.1B   | ✓   | ✓    | ✓   | ✓        |         |
| Llama-3.2-1B       | Llama family      | 1B     | ✓   |      | ✓   | ✓        |         |
| Llama-3.2-3B       | Llama family      | 3B     | ✓   |      | ✓   | ✓        |         |
| Mistral-7B-v0.2    | Llama + SPM tok   | 7B     | ✓   | ✓    | ✓   |          | ✓       |
| Qwen2.5-0.5B       | Llama + QKV bias  | 0.5B   | ✓   | ✓    | ✓   | ✓        |         |
| Qwen2.5-1.5B       | Llama + QKV bias  | 1.5B   | ✓   | ✓    | ✓   | ✓        | †       |
| Qwen2.5-3B         | Llama + QKV bias  | 3B     | ✓   | ✓    | ✓   | ✓        | †       |
| Qwen2.5-7B         | Llama + QKV bias  | 7B     | ✓   | ✓    | ✓   |          | ✓       |
| Qwen3-0.6B         | Qwen3 + QK-norm   | 0.6B   | ✓   |      | ✓   | ✓        |         |

† Qwen2.5-1.5B/3B Q8 abort on CUDA at weight-load time: ggml-cuda's
quantized matmul requires `d_ff` aligned to 512, and those models'
`d_ff` (8960, 11008) aren't. F32 path works for all sizes.

Both byte-level BPE and SentencePiece tokenizers are supported in
text mode (auto-detected from the GGUF vocab). RoPE scaling
(YaRN / llama3-style / linear) propagates through the converter
and loader, so Llama-3.x and Qwen3 long-context configurations
work without extra wiring.

Next targets: Qwen3-4B / 14B / 32B (dense), Qwen3 MoE
(`MUL_MAT_ID`-gated), Gemma 2/3 (sliding-window attention), DeepSeek
V3 (MLA). See
[`docs/roadmap/modern-llm-primitives-2026-05-22.md`](docs/roadmap/modern-llm-primitives-2026-05-22.md).

A toy you can read top-to-bottom that happens to run real models.
