# toy

<p align="center">
  <img src="toy_logo.png" alt="toy" width="240" />
</p>

**v0.4.0-pre-alpha** · early signal · not API-stable
&nbsp;·&nbsp; [CHANGELOG](CHANGELOG.md)
&nbsp;·&nbsp; [docs](docs/INDEX.md)
&nbsp;·&nbsp; [op coverage](docs/coverage.md)

A transformer LM in Ruby, [Spinel](https://github.com/matz/spinel)-compiled
to native binaries. Runs real HuggingFace models — SmolLM2, Llama 3,
Qwen 2.5, Qwen 3, Mistral, Gemma 2, OLMoE (MoE) — at output-identical
fidelity to PyTorch. CPU, CUDA, Metal.

End-to-end as single native binaries:

- **Inference** — KV-cache decode on CPU/CUDA/Metal. F32 or Q8 with
  zero-copy mmap. Opt-in: Q8 K+V cache, fused flash attention.
- **Training** — from-scratch trainer over `TransformerLM`.
- **Fine-tune** — LoRA + QLoRA, one sequence-mode forward + backward +
  AdamW graph (CPU + CUDA).
- **Serve** — Tep+Spinel OpenAI-compatible HTTP.
- **Discover** — auto-finds GGUFs in HF, Ollama, LM Studio, and
  project-local caches.

Modern architecture coverage: MoE routing (OLMoE), SentencePiece
Unigram tokenizers (Gemma 2), pre+post-norm sandwich + logit
soft-cap + alternating sliding-window attention (Gemma 2), per-arch
QK-norm (Qwen 3 shared / OLMoE packed), RoPE scaling (YaRN /
llama3 / linear). See [supported models](#supported-models) for the
verified matrix.

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

Every model has an `algorithm_card` emitting Phuong-Hutter style
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

Requires Ruby, [Spinel](https://github.com/matz/spinel) at
`~/sites/spinel`, and a C compiler. `uv` self-installs for the
Python converter.

```sh
make setup-ggml                              # one-time, ~30 s
make example_list_models                     # what's cached locally
./examples/example_list_models               # HF / Ollama / LM Studio / ./data
```

If nothing shows up:

```sh
prep/fetch_model.sh bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf
```

…or use `huggingface-cli download`, `ollama pull`, or LM Studio
— they all land in caches the next list-models will see.

Then run:

```sh
make example_inference
GGUF=data/llama-3.2-1b-tok.gguf PROMPT="The capital of France is" \
  ./examples/example_inference
# → text: The capital of France is Paris. The capital of Germany is …

# Try the opt-in perf knobs:
KV_Q8=1 FLASH_ATTN=1 \
  GGUF=data/qwen3-1.7b-tok.gguf PROMPT="Hi" N_NEW=32 \
  ./examples/example_inference

# Modern MoE works (Q8 expert weights required — see footnote):
GGUF=data/OLMoE-1b-7b-0924-Instruct-q8_0.gguf PROMPT="The capital of France is" \
  ./examples/example_inference

# Gemma 2 (all four arch quirks auto-detected):
GGUF=data/gemma-2-2b-it-Q8_0.gguf PROMPT="The capital of France is" \
  ./examples/example_inference
```

`example_inference` speaks text when the GGUF was converted with
`--with-tokenizer` (the `*-tok.gguf` variants). Three tokenizer
flavors auto-detected: byte-level BPE, SPM-BPE, SPM-Unigram. Without
an embedded tokenizer the example falls back to a fixed token-ID
prompt and prints raw IDs.

[`examples/`](examples/README.md) has five focused entry points
(inference / train-from-scratch / LoRA fine-tune / HTTP serve /
model discovery), each one Ruby file under ~100 lines compiled to
a single native binary.

**CUDA:** `make setup-ggml-cuda` then `*_cuda` example variants.

**Metal:** `make setup-ggml-metal` then `make example_inference_metal`.
Works with Command Line Tools alone (shaders embed into the static
archive, JIT-compile on first device load ~15 s, then cached).
Validated end-to-end on SmolLM2-135M F32; [`coverage.md`](docs/coverage.md)
Metal column shows the rest.

## Reading the rest

- [`examples/`](examples/README.md) — start here.
- [`docs/INDEX.md`](docs/INDEX.md) — tour of the docs (current
  reference, future work, archive).
- [`tep_demo/`](tep_demo/README.md) — OpenAI-compatible HTTP API.
- [`demos/`](demos/README.md) — per-model and per-feature drivers
  (one per concrete validation run).
- [`tinynn/`](tinynn/README.md) — the C / CUDA / Metal shim over ggml.

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
| OLMoE-1B-7B-Instr  | Llama + MoE + QK-norm⁂ | 7B (1B act) |     | ✓    | ✓   |          |         | ‡         |
| Gemma 2-2b-it      | Llama + Gemma extras⁂⁂ | 2B          |     | ✓    | ✓   |          |         | ‡         |

⁂ OLMoE uses the OLMoE-style QK-norm (gamma at `[d_model]`,
per-head sliced) and 64 experts × top-8 routing. MoE expert weights
**must be Q8_0** — Q4_K / Q5_K / Q6_K trigger a known ggml
`mul_mat_id` bug ([ggml#1506](https://github.com/ggml-org/ggml/issues/1506)).
Our loader emits a runtime warning if the expert weights are
K-quantized.

⁂⁂ Gemma 2 needs four extras: embedding scaling by sqrt(d_model),
logit soft-cap (tanh) on attention + final logits, pre+post-norms
on each sublayer, and per-layer alternating SWA/full attention.
All auto-detected from the GGUF (`gemma2.*` metadata). The
tokenizer is SPM-Unigram (no merges); enabled automatically.

† Qwen2.5-1.5B/3B Q8 abort on CUDA at weight-load time: ggml-cuda's
quantized matmul requires `d_ff` aligned to 512, and those models'
`d_ff` (8960, 11008) aren't. F32 path works for all sizes.

‡ Metal: validated end-to-end only on SmolLM2-135M F32 (bit-identical
to the CPU path). Other Llama-family models should work — same FFI
surface, same graph builder, same ggml-metal kernel coverage — but
haven't been smoked yet. GPT-2-family hasn't been wired (no
`lib/gpt2_ffi_*_metal.rb` mirror has been built into a binary yet
even though the mirror files exist). Quantized weights on Metal are
untested. The current Metal path uses copy-load rather than mmap
(ggml-metal doesn't expose a public buffer-from-pointer API), so
multi-GB models pay the copy cost.

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

- **`KV_Q8=1`**: K and V caches stored as Q8_0 (3.75× smaller).
  Auto-enables flash attention. Validated on Qwen3-1.7B: 13%
  faster wallclock at N_NEW=32, bit-identical token output.
- **`FLASH_ATTN=1`**: fused `ggml_flash_attn_ext` per Q-head.
  12% faster on Qwen3-1.7B at N_NEW=32 vs the scale→softmax→matmul
  triplet (was a wash before V layout flip; now realized).
- **`TOY_EVENTS=path.jsonl`**: emit the structured v1 event stream
  (`docs/events-schema.md`) for live TUIs and post-run analysis.

### Next targets

Mixtral-8x7B (modern-stacked-tensor variants); Mamba/Jamba (SSM ops
bound, Cache class needed); Gemma 3; DeepSeek V3 (MLA); Llama 4 MoE.
Coverage matrix at [`docs/coverage.md`](docs/coverage.md) is the
canonical "what's actually wired" reference.

A toy you can read top-to-bottom that happens to run real models.
