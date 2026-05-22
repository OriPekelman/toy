# Changelog

## Unreleased

### Inference / DevEx

- `examples/example_inference` now speaks text end-to-end on GGUFs
  converted with `--with-tokenizer`: encode `PROMPT`, generate,
  decode the IDs back into text. Falls back to the hardcoded
  five-ID SmolLM2 prompt + raw IDs when no tokenizer is embedded.
- Verified Tokenizer round-trip on SmolLM2-135M, Llama-3.2-1B, and
  Qwen2.5-0.5B via `tinynn/ab_smoke_tokenizer.rb`. Llama and Qwen
  pass 5/5 representative English prompts; SmolLM2 fails 2/5 on
  edge-case chunks like `?\n` and `.txt` — its merge dict produces
  intermediate pieces not in the vocab and the encoder emits UNK.
  Known limitation; tracked as T1.2.

### RoPE scaling (FFI)

- `tnn_rope_ext` and `tnn_rope_ext_back` widened with the full
  YaRN/llama3 arg surface (freq_scale, ext_factor, attn_factor,
  beta_fast, beta_slow, freq_factors). New
  `tnn_rope_freq_factors_alloc` allocates the per-dim factors
  tensor; values computed in
  `Toy::RopeScaling.compute_llama3_freq_factors`.
- `Toy::RopeScaling` value class + `Toy::SmolLM2Config.@rope_scaling`
  field; `SmolLM2ConfigLoader` reads `llama.rope.scaling.*` from
  GGUF and dispatches to `none / linear / llama3 / yarn`.
- `prep/convert_smollm2_to_gguf.py` now propagates HF `rope_scaling.*`
  metadata to GGUF. Without this our Llama-3.x GGUFs had no scaling
  metadata; converter was the bottleneck.

## v0.1.0-pre-alpha — 2026-05-22

**First tagged cut.** Not API-stable; expect breaking changes in any
direction. The goal of this tag is to mark a coherent set of working
capabilities so external readers have a reference point.

### Inference

- Run pretrained models end-to-end as a single native binary:
  GPT-2 family, Llama-family (SmolLM2, TinyLlama, Llama-3.2,
  Mistral-7B), Qwen2.5 family (0.5B → 7B).
- KV-cache decode on CPU and CUDA, F32 and Q8.
- Zero-copy mmap of GGUF weights into both CPU and CUDA buffers
  (UVA on GB10).
- Discover cached models from HuggingFace / Ollama / LM Studio /
  `./data` / `$TOY_MODEL_DIR` via `examples/example_list_models`.

### Training

- From-scratch training of small GPTs via `Toy::Trainer`
  (`examples/example_train` on TinyStories).
- LoRA fine-tune on attention Q heads, CPU + CUDA
  (`examples/example_finetune` / `example_finetune_cuda`).
- QLoRA: Q8 base + F32 LoRA adapter, CPU (works through mmap) and
  CUDA (works through the new `realize_for_q8_copy` path that
  bypasses the BYO-pointer padding issue).
- Full fine-tune on CUDA: every per-block weight + optional
  embedding/output trainable, up to ~1.5B verified
  (`demos/smollm2_seq_full_finetune_cuda`).
- Sequence-mode forward graph (M3): `T` tokens in, `T` logits out;
  one forward + backward + opt_step per training step instead of
  T separate KV-decode rebuilds.

### Serving

- OpenAI-compatible HTTP API via Tep+Spinel
  (`tep_demo/openai_api_smollm2` and family).
- Lite HTTP example for direct token-IDs in / token-IDs out
  (`examples/example_serve` — see Known issues).

### Infrastructure

- Phase 0.6 CPU/CUDA mirror dedup: `prep/gen_cuda_mirror.rb`
  generates `*_cuda.rb` files from their CPU counterparts via
  mechanical substitution; `make verify-mirrors` catches drift.
- Vendored ggml patches (`vendor-patches/`):
  - `0001-0002` CUDA `buffer_from_ptr` for BYO-pointer mmap.
  - `0003` BYO copy-mode A/B selector.
  - `0004` CUDA `cpy` strided-destination fix.
  - `0005` `GGML_OP_CONCAT` backward.
  - `0006` chunked `get_rows_back` for vocab > 65535
    (fixes Qwen-class embedding training on CUDA).

### Bench reference (GB10, 2026-05-22)

| Model               | CPU tok/s | CUDA tok/s |
| ------------------- | --------- | ---------- |
| SmolLM2-135M F32    | 88        | 76         |
| Qwen2.5-0.5B F32    | 34        | 41         |
| Qwen2.5-1.5B F32    | 14        | 20         |
| Qwen2.5-7B Q8       | 4.2       | 14         |

LoRA training step time on SmolLM2-135M (T=4): 108 ms/step.
Full fine-tune step time on SmolLM2-135M (T=4): 108 ms/step.

Full details: `docs/archive/bench-gx10-2026-05-22.md`.

### Known issues

- `examples/example_serve` segfaults at startup on current main. The
  underlying capabilities (Tep + the KV-cache cache + the generate
  loop) all work independently and via `tep_demo/openai_api_smollm2`;
  the specific binding sequence in this example crashes during
  static init. Tracked for follow-up.
- CPU LoRA training requires `tnn_pin_all_graph_b_nodes` to work
  around a ggml-cpu scheduler aliasing bug on long backward chains
  (filed upstream as `ggml-org/ggml#1501`). Current cache classes
  apply the pin transparently.
- Full fine-tune memory ceiling: ~3B on a 121 GB unified-memory box
  (1.5B comfortable, 3B fits, 7B doesn't).

### Sibling projects

- [Spinel](https://github.com/matz/spinel) — Ruby AOT compiler
  matz is building; we live in `~/sites/spinel`. Issues filed
  during this work: `#644` (Range indexing codegen), `#645`
  (Optional<Int> narrowing).
- [Tep](https://github.com/OriPekelman/tep) — Sinatra-flavoured
  HTTP framework that compiles to a native binary via Spinel.
  Issues filed during this work: `#13`, `#16`, `#17`.
- Vendored [ggml](https://github.com/ggml-org/ggml). Upstream
  contributions filed: `#1500` (merged), `#1501` (open).
