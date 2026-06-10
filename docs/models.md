# Supported models

The canonical "what's actually wired" reference is
[`docs/coverage.md`](coverage.md) (per-op coverage vs PyTorch). This page
is the model-level view: which checkpoints run, in which precision, on
which backend — and the footnotes that keep the table honest.

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

## Tokenizer / RoPE coverage

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

## Opt-in performance features

Environment variables the runners read:

- **`KV_Q8=1`** — K and V caches stored as Q8_0 (3.75× smaller),
  auto-enables flash attention. Validated on Qwen3-1.7B: 13% faster
  wallclock at N_NEW=32, bit-identical token output.
- **`FLASH_ATTN=1`** — fused `ggml_flash_attn_ext` per Q-head. 12%
  faster on Qwen3-1.7B at N_NEW=32 vs the scale→softmax→matmul triplet.
- **`TOY_EVENTS=path.jsonl`** — emit the structured `toy/v1` event
  stream for live TUIs and post-run analysis (see
  [`events.md`](events.md)).
