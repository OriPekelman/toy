# Scout: next model targets

Where the bar is today: **GPT-2 / DistilGPT2** load from GGUF, decode via
KV cache (CPU + CUDA), parity-verified at F32-ULP against HF
`transformers`. Self-contained binary. ~14 ms/tok on M2 Air, ~22 ms/tok
on gx10 CUDA.

This doc compares modern small models as the next inference target and
ranks them by **new C/Ruby work**, not by parameter count.

## What the stack already does

Op surface exposed via `lib/tinynn.rb`:

| Need | Have | C-side helper |
|---|---|---|
| matmul (transposed weight) | ✅ | `tnn_matmul` |
| elementwise add (residual / bias) | ✅ | `tnn_add` |
| LayerNorm (γ, β, ε) | ✅ | `tnn_layer_norm` |
| RMSNorm (γ, ε) | ✅ | `tnn_rms_norm` |
| GeLU (`gelu_new`) | ✅ | `tnn_gelu` |
| softmax (last dim) | ✅ | `tnn_softmax`, `tnn_soft_max_ext` |
| causal mask | ✅ | `tnn_diag_mask_inf` |
| concat / view / cpy / scale | ✅ | `tnn_concat`, `tnn_view_2d`, `tnn_cpy`, `tnn_scale` |
| KV-cache write-in-place | ✅ | `tnn_cpy` + `tnn_view_2d` |
| GGUF load + dequant (F32 / Q8_0 / Q4_0 / Q5_0) | ✅ | `tnn_gguf_*` |
| BPE encode/decode | ✅ | `lib/bpe.rb` |

What the stack does **not** have yet (every llama-family model needs at
least the first three):

| Missing | Used by | ggml has it? | Effort |
|---|---|---|---|
| **SiLU** (`x * sigmoid(x)`) | TinyLlama, SmolLM2, Qwen2, Phi-3 | `ggml_silu` | 1-line C wrapper + FFI bind |
| **Elementwise mul** (for SwiGLU gate * up) | all SwiGLU FFNs | `ggml_mul` | 1-line C wrapper + FFI bind |
| **RoPE** (rotary position embedding) | everything modern except GPT-2/Pythia-no-rotary | `ggml_rope_ext` | ~20 lines C + freq/base wiring |
| **Grouped Query Attention** (`n_kv < n_heads`) | TinyLlama, SmolLM2, Qwen2 | trivial (head repeat = view stride) | KV layout change in `gpt2_ffi_kv` |
| **SentencePiece / tiktoken** tokenizer | non-GPT-2 vocab | n/a | new tokenizer or use HF byte-fallback |

GPT-NeoX (Pythia) also wants:
- **partial rotary** (only first 25% of head dims rotated) — supported by `ggml_rope_ext` via `n_dims < d_head`
- **parallel residual** (`x + attn(x) + mlp(norm(x))` instead of sequential) — wiring change only

## Candidate matrix

Architectures pulled from each model's `config.json` on HF (confirmed
2026-05-15):

| Model | Type | d_model | layers | heads | KV heads | d_ff | vocab | ctx | norm | act | RoPE | tied emb |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| **GPT-2 small** *(baseline)* | gpt2 | 768 | 12 | 12 | 12 | 3072 | 50257 | 1024 | LN | gelu | — | yes |
| **Pythia-160M** | gpt_neox | 768 | 12 | 12 | 12 | 3072 | 50304 | 2048 | LN | gelu | partial 25% | no |
| **SmolLM2-135M** | llama | 576 | 30 | 9 | 3 | 1536 | 49152 | 8192 | RMS | silu | full, θ=100k | yes |
| **TinyLlama-1.1B-Chat** | llama | 2048 | 22 | 32 | 4 | 5632 | 32000 | 2048 | RMS | silu | full, θ=10k | no |
| **Qwen2.5-0.5B** | qwen2 | 896 | 24 | 14 | 2 | 4864 | 151936 | 32768 | RMS | silu | full, θ=1M | yes |

## Effort per candidate

Each row reuses the existing GPT-2 KV-cache machinery; the deltas are
the *new* work.

### Pythia-160M

**New ops**: RoPE (partial, 25% of head dims) only. SiLU not needed.

**Wiring**:
- partial rotary embedding on Q and K (head dim 64 → rotate 16)
- parallel residual: `x + attn(norm1(x)) + mlp(norm2(x))` instead of sequential adds
- attention has a Q/K/V combined linear (split into 3 along the output dim in the loader)
- tied embeddings = **off**; needs a separate `embed_out.weight` GGUF tensor
- still uses 50k-ish BPE — could potentially reuse `lib/bpe.rb` with a different vocab/merges dump

**Estimated effort**: ~1 day. ~150 lines of new Ruby in `gpt2_ffi_kv_neox.rb` (mostly forks of the existing block), one new `tnn_rope` C wrapper, one converter for `EleutherAI/pythia-160m → gguf`.

### SmolLM2-135M

**New ops**: SiLU, elementwise mul, RoPE (full), GQA (n_kv=3 < n_heads=9, so each KV head is shared across 3 Q heads).

**Wiring**:
- SwiGLU FFN: `down_proj(silu(gate_proj(x)) * up_proj(x))` — 3 weight matrices instead of 2
- RMSNorm only (already supported)
- RoPE with θ=100k base (parameterizable through `ggml_rope_ext`)
- KV-cache layout: `K[L, n_kv, d_head, T]` instead of `K[L, n_heads, d_head, T]`; during attention, repeat each KV head 3× along the head dim (view stride trick, no data move)
- 49k vocab is byte-level BPE-compatible-ish; SmolLM2 uses a custom tokenizer that requires SentencePiece-like merges. **Tokenizer is the real cost here.**
- tied embeddings = yes (same code path as GPT-2)

**Estimated effort**: ~2-3 days. ~250 lines of new Ruby + the tokenizer port (or shell out to a host-side encoder for now). Three new C wrappers (`tnn_silu`, `tnn_mul`, `tnn_rope`). One converter (gguf-py already supports llama; mostly a metadata-key swap).

### TinyLlama-1.1B

**Same architecture as SmolLM2**, different shapes. Once SmolLM2 works,
TinyLlama is a config change + GGUF conversion. RoPE θ=10k (default).

**Caveat**: 1.1B params at F32 = 4.4 GB. Q8_0 brings it to ~1.2 GB. On
M2 Air with 8 GB this is feasible but tight; gx10 is comfortable.

**Tokenizer**: standard Llama tokenizer (SentencePiece). Same blocker as SmolLM2.

**Estimated effort**: ~1 day on top of SmolLM2.

### Qwen2.5-0.5B

**Same as SmolLM2**, plus:
- Q/K/V projections have **biases** (rare in llama family; standard
  llama leaves them off) — ✅ landed in step 43:
  `Toy::GQAttention.has_qkv_bias` toggled by the loader; both the
  native and FFI KV paths apply biases when present. The converter
  (`prep/convert_smollm2_to_gguf.py`) writes `attn_q.bias` /
  `attn_k.bias` / `attn_v.bias` when the HF safetensors has them.
- vocab is **151936** — needs Qwen's tokenizer (tiktoken-style BPE with
  a different merges table) — still pending; v1 can host-side tokenize
- RoPE θ = 1,000,000 (long-context-friendly) — already handled
- ctx 32768 (would blow our current `MAX_T=1024` ctx_buf) — pick a
  smaller `MAX_T` per-demo

**Remaining effort**: tokenizer only (the model itself runs today via
`prep/convert_smollm2_to_gguf.py --repo-id Qwen/Qwen2.5-0.5B`).

### Why not Phi-3-mini-4k

3.8B params, 32 layers, intermediate 8192. About 7× SmolLM2 in compute,
zero new architecture surface vs the llama path. Better as a stretch
target once the llama pipeline is solid.

## Recommendation

**Do SmolLM2-135M next.** It unlocks the entire llama family
(TinyLlama, Qwen2 minus QKV-bias, Phi-3 with light extras) and forces
all the right abstractions:

- SiLU + mul + RoPE — three small C wrappers that earn their keep on
  every future model
- GQA in the KV cache — a stride-only change that maps to a real
  perf win (KV cache is `3/9` the size of GPT-2-equivalent at this
  d_head, and that ratio gets more extreme on bigger models)
- a non-GPT-2 tokenizer — even if v1 shells out to a host-side
  encoder, it forces the inference layer to stop assuming Ruby BPE
  is the only path

Pythia-160M is closer to the current code but it's an evolutionary dead
end (no one ships GPT-NeoX-derived models anymore). Doing it would feel
like progress but wouldn't open the path to Qwen / Llama-3 / Phi.

## Plan-of-attack sketch (SmolLM2-135M)

1. **C-side ops** (`tinynn/tinynn_ggml.c`):
   - `tnn_silu(sess, t)` → `ggml_silu(s->ctx, t)`
   - `tnn_mul(sess, a, b)` → `ggml_mul(s->ctx, a, b)`
   - `tnn_rope(sess, t, t_pos, n_dims, base, scale, ne_past)` → `ggml_rope_ext(...)` with appropriate freq_base / freq_scale.
   - Bindings in `lib/tinynn.rb`.
2. **Converter** (`prep/convert_llama_to_gguf.py`):
   - HF safetensors → GGUF with llama metadata keys (`llama.attention.head_count_kv`, `llama.rope.freq_base`, etc.)
   - Reuse `--quantize q8_0` logic from the GPT-2 converter.
3. **Model class** (`lib/llama.rb` + `lib/llama_ffi_kv.rb`):
   - `LlamaConfig` (d_model, n_layers, n_heads, n_kv_heads, d_ff, rope_base, rms_eps, ...)
   - `LlamaKVFFICache` mirroring `GPT2KVFFICache` but with:
     - SwiGLU FFN block
     - RoPE on Q and K before storing
     - K/V buffers sized at `n_kv_heads * d_head` per layer
   - Same `decode_step` shape as GPT-2 so the API layer is reusable.
4. **Tokenizer**:
   - v1: host-side encoder via `prep/encode_prompt.py`, then load token IDs from a file. Lets us validate the inference path before sinking time into a SentencePiece port.
   - v2: pure-Ruby SentencePiece decoder (the encode side is much harder than decode, but for prompt-driven demos a one-shot Python-side encode is fine).
5. **Parity test**: mirror `prep/parity.py` against `HuggingFaceTB/SmolLM2-135M` and require argmax+top5 match like the GPT-2 path.

## Anti-targets (for the record)

- **Mamba / state-space models** — totally different op surface (selective scan), would mean re-doing everything. Skip.
- **Mixtral / MoE** — needs sparse routing; doable but adds a whole new dimension of work for no immediate win at toy scale.
- **DeepSeek-V2 MLA attention** — interesting but exotic; revisit once vanilla MQA/GQA is solid.
- **Encoder/decoder (T5, BART)** — different architecture; out of scope for "decoder-only toy".

## Open questions

- **Tokenizer**: pure-Ruby SentencePiece decoder is straightforward
  (vocab+merges → byte-level decode). The encoder is harder (Unigram
  vs BPE, prefix-token handling). Worth doing properly or punt with
  a Python-side encode shim?
- **RoPE in KV cache**: store K *post*-rotation (cheaper at inference,
  llama.cpp's choice) or *pre*-rotation (matches the HF reference
  exactly, easier parity). Pre-rotation for v1, swap later if perf
  motivates it.
- **GGUF metadata keys**: llama-family GGUFs use a *different* set of
  keys than GPT-2 (`llama.*` vs `gpt2.*`). The current `gguf_load.rb`
  hard-codes GPT-2 keys; need a small registry / dispatch.
