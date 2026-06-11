# Loader API surface

There are two ways to bring GGUF weights into memory. They are
**peers**, not "old vs new" — pick the one that matches the use case.
The functions live in `lib/toy/io/loaders/toy_smollm2_loader.rb` (module `GGUFLoad`),
the FFI cache in `lib/toy/llm/engine/llama_kv_engine.rb`, and the low-level
download primitive in `lib/toy/ffi/tinynn.rb`.

## 1. Mat-mediated path — `GGUFLoad.load_toy_smollm2`

For inspection, fine-tuning, parity checks — anything that wants the
full Ruby `Mat` object graph. Weights land as `Mat` (Float64) tensors
hung off a model object, and there is a training-side graph behind
them.

```ruby
cfg   = SmolLM2ConfigLoader.read(gguf_path)
model = <model_with_a_compatible_layout>.new(cfg)
GGUFLoad.load_toy_smollm2(model, gguf_path)

# Inspect / modify a weight as a Ruby Mat:
qw = model.token_embed.weight   # Mat[vocab, d_model]
qw.flat[0] += 1.0
```

Memory cost: ~12 bytes per parameter (8 in the Float64 `Mat` + 4 in
the ggml f32 shadow). That ceiling caps practical model size around
~3B params on a 121 GiB box. See
[`reference/memory-design.md`](memory-design.md) for the duplication
math.

## 2. Direct path — `kv.load_weights`

For inference at 7B-class scale, where the Float64 intermediate is
unaffordable. Skips `Mat` construction entirely; weights are ggml f32
(or stay Q8_0 — see memory-design) and never touch Ruby's heap.

```ruby
cfg   = SmolLM2ConfigLoader.read(gguf_path)
flags = GGUFLoad.detect_smollm2_flags(gguf_path)

kv = SmolLM2KVFFICache.new
kv.realize_for(max_T, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
               cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
               flags.untied, flags.qkv_bias)
kv.load_weights(gguf_path)
```

`load_weights` dispatches to `GGUFLoad.load_kv_cache_auto`, which picks
the right reader for the on-disk weight type (f32 / Q8_0 / mmap).
Memory cost: 4 bytes per parameter (ggml f32 only), or fewer when
quantized weights stay quantized. Verified to ~7–30 GiB peak RSS for
Qwen2.5-7B end-to-end depending on the memory path taken.

## Mat-roundtrip — pulling weights back out

The direct path is **not** a one-way trip. Any persistent FFI tensor
can be pulled back into a Ruby `Mat` for inspection, export, or as the
seed for a Mat-mediated fine-tune:

```ruby
emb_mat = kv.read_persistent_mat(kv.t_token_embed, cfg.vocab, cfg.d_model)
norm    = kv.read_persistent_mat(kv.t_final_norm_gamma, 1, cfg.d_model)
d_head  = cfg.d_model / cfg.n_heads
qhead0  = kv.read_persistent_mat(kv.kv_blocks_ffi[3].t_w_q[0],
                                 d_head, cfg.d_model)   # per-head: t_w_q is an array
```

`read_persistent_mat(t, rows, cols)` wraps `TinyNN.download_to_mat`,
which is backed by the FFI primitive `tnn_download_to_f64_array`. That
primitive chunks internally, bypassing the 16 MiB scratch buffer, so
it works on weight-sized tensors of arbitrary size — not just those
that fit in scratch. (For small graph intermediates — norms, per-step
logits — the scratch-based `download_row_major` is fine and slightly
faster.)

Verified bit-identical to the GGUF source on SmolLM2-135M's 28M-float
`token_embed`: max diff 0.0 across all elements.

The full-graph round-trip (write → reload → forward, asserting
bit-identical logits) is gated separately by
`examples/smoke_gguf_roundtrip.rb` (`make examples/smoke_gguf_roundtrip`).

## Why both, why not just one

- Mat-mediated is the **only** path with a training-side graph today.
  Removing it would lock the codebase into inference-only territory,
  which is explicitly not the direction.
- Direct is the **only** path that scales past ~3B parameters at
  current memory budgets.
- Mat-roundtrip on the direct path closes the loop: if you started via
  the direct path for memory reasons, you can still pull individual
  tensors back into `Mat` space for surgery.

The medium-term goal is a training-capable FFI cache (the KV-cache is
inference-shaped — separate forward/backward graphs are needed for
training). Until then, the two paths coexist.

## HF `nn.Linear` ↔ ggml byte equivalence

Both paths rely on the GGUF on-disk byte layout matching ggml's
column-major `ne=[in, out]` interpretation exactly, which is what
makes mmap zero-copy load possible (CPU `ggml_backend_cpu_buffer_from_ptr`,
CUDA `ggml_backend_cuda_buffer_from_ptr` — see
[`reference/backends.md`](backends.md)).

For a `mul_mat(W, x)` where ggml expects `W` with `ne0=in, ne1=out`:

- ggml stores element `(i, j)` at byte offset `(j*in + i)*sizeof(elem)`
  (column-major; `ne0` is fastest-varying).
- HF safetensors stores `nn.Linear.weight` with numpy shape
  `(out, in)` row-major: `(j, i)` at byte offset `(j*in + i)*sizeof(elem)`.

**Same byte offset for the same logical `(in_idx, out_idx)` element.**
So writing HF's bytes verbatim and declaring the ggml tensor with
`ne=[in, out]` yields a matmul-ready tensor with zero rearrangement —
no transpose at conversion.

Per-head slicing falls out for free: GGUF `attn_q.weight` has shape
`[n_heads*d_head, d_model]`; head `h` is rows
`[h*d_head, (h+1)*d_head)`, a **contiguous byte range** in HF-native
layout. A per-head ggml tensor with `ne=[d_model, d_head]` is
byte-identical to that slice — mmap-friendly with no per-head copy. The
same arithmetic holds for Q8_0 with the block-quantized stride
(`d_head * (d_model/32) * 34` bytes per head).

Caveat: GPT-2's `Conv1D.weight` is `[in_features, out_features]` —
the opposite of `nn.Linear` — so for that arch the transpose *is*
necessary. The two must be distinguished cleanly in any converter.
