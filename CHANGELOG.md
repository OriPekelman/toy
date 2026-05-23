# Changelog

## Unreleased

### M1.1 — explicit head_dim + tied-embeddings handling (partial)

- `Toy::SmolLM2Config` gains `head_dim` field (defaults to
  `d_model / n_heads`; loader overrides from GGUF).
- `SmolLM2ConfigLoader.read` reads `llama.attention.key_length`
  (llama.cpp convention) when present; falls back to the computed
  value for SmolLM2 / Llama-3.x / Qwen2.5 (all match
  hidden_size/num_heads).
- `lib/toy_smollm2_ffi_kv.rb` and `lib/llama_seq_forward_ffi.rb`
  now read `cfg.head_dim` everywhere they used to compute
  `cfg.d_model / cfg.n_heads`. Behaviour unchanged for models
  whose explicit head_dim matches the computed one.
- `prep/convert_smollm2_to_gguf.py` emits
  `llama.attention.key_length` + `llama.attention.value_length`
  GGUF keys (one value, written twice — every model on the
  roadmap has K-dim == V-dim).
- Converter also handles `tie_word_embeddings: true` correctly:
  skips emitting `output.weight` even when `lm_head.weight` is
  in the safetensors. Trusts the config flag (as llama.cpp does).

Qwen3-0.6B converts with the right head_dim=128 and tied
embeddings, but inference still crashes in
`sp_ToyLM_decode_step`. M1.1 stays open; root cause appears to
be deeper than the converter / loader layer (suspect ToyLM /
decode-step assumes specific shape invariants).

Bench passes (±2% across all metrics). Existing models —
SmolLM2, Llama-3.2, Qwen2.5 — produce identical text.

### M1 — Qwen3 dense plumbing (QK-norm, partial)

- `SmolLM2KVBlockFFI` now carries `t_q_norm_gamma` + `t_k_norm_gamma`
  (1D `[d_head]` shared across heads, per block). `SmolLM2KVFFICache`
  carries `@has_qk_norm` flag.
- `GGUFLoad.detect_smollm2_flags` detects QK-norm by presence of
  `blk.0.attn_q_norm.weight` (Qwen3-only; Qwen2.5 / Llama return false).
  `SmolLM2Flags` gains a `qk_norm` field.
- `realize_for_mmap` signature widened: `(gguf, cfg, max_T, untied,
  qkv_bias, qk_norm)`. Allocates the QK-norm gammas as mmap'd
  1D F32 tensors when set. Graph builder applies `tnn_rms_norm` to
  Q and K with the per-block gamma BEFORE `tnn_rope_ext`.
- `prep/convert_smollm2_to_gguf.py` propagates Qwen3's
  `self_attn.q_norm.weight` / `k_norm.weight` HF tensors to
  `attn_q_norm.weight` / `attn_k_norm.weight` GGUF tensors.
- Existing Qwen2.5 / SmolLM2 / Llama-3.2 / TinyLlama unchanged
  (no QK-norm path triggered). Bench passes within ±5%.
- Qwen3-0.6B converts and loads, but **inference is not yet
  correct**. Root cause: Qwen3 sets `head_dim = 128` explicitly in
  HF config, not `hidden_size / num_heads = 64`. Our converter
  computes the wrong head_dim and the Q/K/V projections come out
  half-sized. Tracked as M1.1 (next task). QK-norm itself appears
  correctly wired — the head_dim mismatch alone explains the
  garbage output.

- Re-converted `data/tinyllama-1.1b-tok.gguf` and
  `data/mistral-7b-instruct-v0.2-tok.gguf` with `--with-tokenizer`.
  Both load + run inference, but **text-mode I/O fails** because
  their tokenizers are SentencePiece (Llama-2 vocab), not the
  byte-level BPE our `lib/tokenizer.rb` handles. T1.2's "never mask"
  rule caught it cleanly: `WARN: tokenizer: piece "Ġ" not in vocab
  — emitting UNK` ⇒ Mistral output `"The<unk>capital<unk>of<unk>..."`.
- Tracked as T1.3 (new task). Adds tokenizer-flavor detection from
  `tokenizer.ggml.model` and a SentencePiece encoder path.
- **Current text-I/O coverage** (works end-to-end):
  SmolLM2-135M, Llama-3.2-1B, Qwen2.5-0.5B. All byte-level BPE.
- **Inference-only** (text I/O blocked on T1.3): TinyLlama-1.1B,
  Mistral-7B-v0.2. ID-mode `example_inference` (no PROMPT) still
  works fine.

### D1 — algorithm-card drift detector (instead of auto-emitter)

- `prep/card_drift_check.rb`: a Ripper-walker tripwire that
  verifies each `Toy::` class with both `def forward` and
  `def algorithm` keeps the two in lock-step. Run via
  `make check-cards`. Pure-Ruby stdlib, no extra deps.
- D1 was originally scoped as an auto-emitter that would delete
  the 209 LOC of hand-written `def algorithm` methods. Closer
  reading of those methods showed they're not 1:1 with the
  unrolled forward code — `FFN`'s 5-line forward becomes a
  curated 2-step card that fuses `matmul + add_bias + gelu`. An
  auto-emitter would produce faithful-but-ugly output that
  wouldn't actually replace the cards.
- The drift detector matches the real failure mode: forward
  changes, card doesn't (or vice versa). Catches the common
  `gelu` / `silu` / `⊙` activation-mismatch + the
  matmul-presence collapse case. Validated by deliberate-drift
  test (deleting `gelu(...)` from FFN's card → tool fails).
- For the original "delete the 209 LOC" goal (re-trigger
  condition (a) from task #95): an auto-emitter is still possible
  if/when we add a third architecture and feel the cost, but the
  drift detector covers the maintenance-during-edits case today.

### P2 — measured, not viable (skipped)

- `docs/roadmap/p2-ffi-matmul-2026-05-23.md`: planned to FFI-wrap
  `Mat#matmul` for a 5–10× win on `example_train`. Measured the
  actual cost: session-per-op FFI is **1.7× SLOWER** than pure-Ruby
  at training-toy shapes (32×8 matmul). 5 000 calls × ~180 µs
  session lifecycle overhead = ~0.9 s deficit vs the 1.3 s
  pure-Ruby baseline.
- Lesson: the 38× FFI gain on LLM-shape inference is for
  *whole-graph* FFI (one `tnn_compute` per step), not per-op FFI.
  At toy shapes, per-op FFI loses on session overhead; break-even
  is around `m*k*n ~ 100 000`. Real workloads (LoRA, KV decode,
  full FT) route through whole-graph FFI cache classes already
  and are unaffected.
- If we later want fast `example_train`, build a
  `TransformerLMTrainerFFI` mirror of `LlamaSeqForwardFFICache`
  for the custom-GPT shape (~500 LOC, ~5–20× expected). Queued
  as P2-α; not currently prioritised.

### Bench harness + Lowerer evidence

- `bench/` directory with three Spinel-compiled benches —
  `lora_step.rb` (training step ms), `inference.rb` (toks/sec), and
  `tokenizer.rb` (encode μs/token). Each emits `BENCH metric value`
  lines on stdout; the orchestrator at `bench/check.rb` runs them
  and compares to `bench/baselines.csv`, exiting 1 on any metric
  past its per-metric tolerance.
- `make bench` runs the gate; `make bench-update` rewrites the
  baselines; `make bench-report` runs without gating (handy for
  local exploration). Suggested use: invoke before pushing
  perf-sensitive changes; the CSV diffs cleanly in git so
  re-baselining is a normal commit.
- `docs/roadmap/lowerer-evidence-2026-05-23.md`: trace-driven
  evidence on whether the full Lowerer pays for itself.
  Measured `example_train` (native-Mat path) — 82.7 % of step
  wallclock is in three matmul methods, 51 % of all matmul
  calls share one (N, P) inner-dim pair. But the comparable
  alternative (FFI the matmul through ggml) gives 10–100×
  for an order of magnitude less code than the ~500-LOC
  Lowerer. Verdict: split the Lowerer's three claimed benefits
  into proportionate tools (P2 to FFI Mat ops; D1 standalone
  card emitter; Spinel landmines wait for upstream).
- `lib/transformer.rb`: six hot `Mat` methods (`matmul`,
  `matmul_t`, `t_matmul`, `plus`, `add!`, `scale!`) wrapped with
  `tnn_trace_begin/end`. Zero-cost when off (1.44 s baseline →
  1.44 s with traces present but inactive on a 87-sequence
  training run); 12 % overhead when on. `MAT_SHAPES=1` env
  enables a shape histogram printf for Lowerer-style evidence
  runs.

### Toolchain

- Bumped Spinel to master `d59926a`. Two changes affect us directly:
  - `0adca86` (matz/spinel#647): `examples/example_serve` no longer
    segfaults at startup. The bug was a Spinel codegen ordering issue
    on top-level `CONST = recv.method` when `recv` was a local;
    we'd kept the buggy form intentionally as a regression check.
    Confirmed: it loads the model, binds the port, accepts requests.
  - `97bf268` (rbs_extract): `--rbs sig` Mat-in-`Toy::` resolution
    now works. Warning overhead dropped from +45 (was unusable) to
    +3 (acceptable). `sig/toy.rbs` header reflects this.
  - The Hash#[missing] → 0 codegen behavior (T1.2's root cause)
    still requires the `has_key?` guards we landed yesterday;
    Spinel's matz/spinel#521 fix narrowed the symptoms but the
    structural workaround stays.

### Observability / training

- New `tinynn/tinynn_trace.{c,h}`: Chrome Trace Format emitter,
  ~5ns per begin/end when off, opens in https://perfetto.dev.
  Instrumented `tnn_realize{,_backward}`, `tnn_compute{,_backward}`,
  `tnn_upload{,_from_float_array,_from_int_array}`, `tnn_download`.
  `examples/03_finetune_lora.rb` and its CUDA mirror accept
  `TRACE=path.json` to wrap each step.
- New `tnn_scratch_sum_f32` and `tnn_scratch_sum_sq_f32` reducers
  (mean / L2-norm without a Mat round-trip).
- `examples/03_finetune_lora{,_cuda}.rb` accept `GRAD_DUMP=1` to
  emit per-(layer, head, A/B) gradient stats as CSV.
- **Finding** (`docs/p1-grad-bisection-2026-05-22.md`):
  CPU and CUDA LoRA gradients agree to 0.32–0.42 % median across
  steps 1–3; loss curves match to 3+ decimals. The underlying
  ggml-cpu sched aliasing bug is still upstream but the local
  workaround (`tnn_pin_all_graph_b_nodes`, wired into
  `lib/llama_seq_forward_ffi.rb:1192`) prevents it from biting
  prod LoRA training. Issue still wants filing against ggml-org/ggml.

### Inference / DevEx

- `examples/example_inference` now speaks text end-to-end on GGUFs
  converted with `--with-tokenizer`: encode `PROMPT`, generate,
  decode the IDs back into text. Falls back to the hardcoded
  five-ID SmolLM2 prompt + raw IDs when no tokenizer is embedded.
- `tinynn/ab_smoke_tokenizer.rb`: 15/15 prompts round-trip
  bit-identically on SmolLM2-135M, Llama-3.2-1B, Qwen2.5-0.5B.
  The earlier "SmolLM2 fails 2/5" bug (`?\n` and `.txt` patterns)
  was actually a **Spinel hash codegen bug**: `Hash#[missing_key]`
  returns integer 0 instead of nil, so missing merges appeared as
  rank-0 (top-priority) merges in the BPE loop. Fixed by guarding
  every hash read with `has_key?`. Tokenizer now warns loudly when
  emitting UNK so the next instance of this class of bug surfaces
  fast.

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
