# TinyLlama-1.1B FFI path produces NaN logits (FIXED)

**Status**: fixed in step 45 (2026-05-17). Root cause was a silent
scratch-buffer overflow in `stage_transposed_and_upload`, not an
f32 precision issue. Same bug, same fix as
[Qwen2.5-0.5B](qwen25-known-issue.md) — TinyLlama's `ffn_gate` is
2048 × 5632 = 11.5M floats, 3× larger than the 16 MiB scratch.

The diagnosis below ("f32 precision at L=5+") was wrong: the
non-monotonic NaN-at-L=5 / zeros-at-L=6 / finite-at-L=10 pattern was
random heap garbage in the scratch tail, not a precision threshold.
Now produces coherent text via `demos/tinyllama_kv`:

```
Once upon a time, there was a young girl named Lily. Lily was a kind and
```

The original investigation notes are kept below as a record of how
the misdiagnosis went. The trace tap that ultimately found it
(`SmolLM2KVFFICache#enable_trace!`) is in `lib/toy_smollm2_ffi_kv.rb`;
see `docs/qwen25-known-issue.md` for the trace excerpt and the fix.

---

## Original (incorrect) investigation

**Status (at the time)**: open. Native Mat path works; FFI (CPU and CUDA) NaNs.

## Symptoms

`demos/tinyllama_kv_cuda` (and `demos/tinyllama_kv` on CPU FFI) load
TinyLlama-1.1B-Chat-v1.0 and execute a forward pass at ~64 ms/token
(CUDA) / ~160 ms/token (CPU). Output logits are all `NaN`. Greedy
argmax over NaN picks token 0 (the `<unk>` token), so generation
produces a string of `<unk>` for every output position.

`demos/tinyllama_native` (pure Ruby Mat path on the same model) at
~4.5 s/token produces sensible logits (e.g. argmax=13, value=11.33
on the last prompt position) and reasonable continuations.

The exact same FFI code path runs `demos/smollm2_kv` /
`demos/smollm2_kv_cuda` correctly (`"Once upon a time, there was a
little girl named Lily..."`), so the FFI ops themselves are fine —
something about TinyLlama's shape exercises a different bug.

## What's been ruled out

- **Not the untied-output wiring.** Forcing the tied path on
  TinyLlama (the FFI uses `t_token_embed` for unembed instead of
  `t_output`) still produces NaN — the bug is upstream of the
  unembed step.
- **Not a load failure.** `tnn_finalize_weights` returns 0 and
  4.4 GB of weights are uploaded successfully. `model.output_proj`
  (verified via debug print) holds the right values from the GGUF.
- **Not a weight-conversion failure.** The native Mat path uses
  the same loaded weights and produces sensible logits.
- **Not a Spinel compile issue.** No warnings/errors on the new
  code; the binary builds clean.
- **Not architecture-bound.** TinyLlama uses the same Toy::SmolLM2 /
  SmolLM2KVFFI(Cuda) code as SmolLM2-135M. Both follow the
  llama-family pattern with RMSNorm + GQA + SwiGLU + RoPE.

## Suspects, not yet confirmed

- **Graph order / `tnn_add_to_graph` for 4 KV writes × 22 layers**
  may interact differently with ggml's scheduler than 3 × 30 did.
  Hypothesis: the K/V writes don't complete before the per-head
  attention reads, so attention reads zero-K and zero-V → softmax
  on near-zero scores produces a uniform distribution → matmul
  with zero-V produces zeros → eventual NaN in a later RMSNorm.
  Would explain "first decode is NaN even at pos=0".
- **Buffer overflow** in the compute ctx for the larger TinyLlama
  graph (32 Q heads × 22 layers vs 9 × 30). Estimated graph node
  count is ~8200 (under the 16384 cap), but the per-layer leaf
  count and scratch usage may differ.

## Reproduction

```sh
# on gx10 (has the GGUF + CUDA toolkit)
./prep/llama_tokens.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --ids data/tinyllama_prompt_ids.txt encode "Once upon a time"
./demos/tinyllama_kv_cuda
# → output ends in <unk><unk><unk>... ; argmax always 0

./demos/tinyllama_native
# → "Once upon a time" + sensible continuation (slow: ~4.5 s/token)
```

## Layer-count bisection — non-monotonic NaN pattern

Forcing `cfg.n_layers = N` in the demo before `realize_for` and the
model construction (so only N out of 22 layers are loaded and run):

| n_layers | logits[0..4] |
|---:|---|
| 1  | finite (0.22, -0.15, ...) |
| 2  | finite |
| 3  | finite |
| 4  | finite |
| 5  | **NaN** |
| 6–14 | finite |
| 15 | **NaN** |
| 20, 22 | **NaN** |

L=5 reproduces deterministically across multiple runs. L=10 is
consistently fine, even though it includes layers 0–4 (the same ones
that NaN in L=5). So this is *not* "layer 4 has a bad weight" — the
NaN depends on the total layer count, not which specific layers
participate.

That points away from a math/weights cause and toward a Spinel codegen
or ggml-graph-state interaction that varies with the size of the
`@kv_blocks_ffi` array or the total persistent-buffer layout. Possible
mechanisms:

* Spinel arrays with N elements may use different internal storage
  for some N (inline vs heap) and one path silently mis-aliases.
* `ggml_backend_alloc_ctx_tensors` lays out persistent tensors
  differently at different total sizes; some layouts may put
  adjacent tensors in a pattern that ggml's compute kernels
  mis-handle.
* Compute-graph buffer / scratch sizing crosses some threshold at
  specific node counts.

## Diagnosis: f32 precision overflow

Final bisection result:

| Path | Precision | L=5 result |
|---|---|---|
| `demos/tinyllama_native` (native Mat)  | f64 | finite (argmax=20358, max=3.27) |
| `demos/tinyllama_kv` (CPU FFI / ggml)  | f32 | NaN |
| `demos/tinyllama_kv_cuda` (ggml-CUDA)  | f32 | NaN |

Same model, same weights, same input. The only difference is the
compute precision. f64 has range ~1.8e308; f32 caps at 3.4e38.

TinyLlama-1.1B-Chat-v1.0 produces intermediate activations at L=5
that exceed f32 range. The non-monotonic pattern (L=5 NaN, L=6 zeros,
L=10 finite) reflects how subsequent layers re-normalize activations:
adding more RMSNorms brings magnitudes back into f32 range, but
truncating *exactly* at certain depths catches them mid-overflow.

L=6 producing all-zero logits is consistent: if rms_norm sees a
vector containing inf values, sumsq overflows to +inf, 1/sqrt(inf)
= 0, output = x * 0 = 0. From there, every downstream value is 0.

L=5 producing NaN: presumably layer 4's output has finite (but
huge) values; the final unembed matmul over 32k vocab × 2048
features overflows during accumulation, +inf - inf later = NaN.

This is a *known* phenomenon in some llama-family checkpoints —
specific layers develop "outlier features" with magnitudes that
need >f32 precision to compute stably. Quantization formats
(GGUF Q8_0, Q4_K) include per-block scale factors that effectively
normalize the dynamic range; running TinyLlama from a quantized
GGUF would likely avoid this entirely. SmolLM2-135M doesn't hit
the issue because its smaller weight magnitudes stay within f32
headroom even at full depth.

## Workarounds

1. **Use the native Mat path for full TinyLlama** (`demos/tinyllama_native`)
   — slow (~4.5 s/token on gx10 CPU) but numerically correct.
2. **SmolLM2-135M remains the canonical FFI benchmark** for the
   llama family in this project. Its activations stay in f32 range
   end-to-end.
3. **Quantized TinyLlama (future)** would likely sidestep the issue.
   Needs: extend `prep/convert_smollm2_to_gguf.py` with the same
   `--quantize q8_0` flag the GPT-2 converter has, then test through
   `demos/tinyllama_kv_cuda`. Quantization's per-block scales should
   keep dynamic range in check.

Not a Spinel issue; not a ggml issue. f32 limit of the chosen format.
