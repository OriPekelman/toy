# M3: DistilGPT2 inference through Ruby → Spinel → ggml-FFI

End-to-end inference of a real HuggingFace model — DistilGPT2 (82 M
params, 6 layers, d=768, vocab=50257) — running through three
backends with verified numerical parity against PyTorch transformers.

Branch: `hf-gpt2` (12 commits beyond `main`).

```
Generate "Hello, my name is" → "Hello, my name is J.J.K. Rowling."

Native Mat (f64, single-thread Ruby):  ~9.0 s   (~1100 ms/token)
FFI full-forward (T_SEQ=16):            451 ms  (   56 ms/token)
FFI KV cache:                            94 ms  (    7.4 ms/token)
```

## Quick start

One-time setup (downloads ~700 MB across distilgpt2 weights + Python
deps; converts to ~330 MB GGUF):

```sh
make setup-ggml                        # CPU ggml build (~30 s)
./prep/convert_distilgpt2_to_gguf.py   # HF → data/distilgpt2-f32.gguf
```

Run any of the three demos:

```sh
./prep/tokens.py encode "Hello, my name is"

make distilgpt2_demo      && ./demos/distilgpt2_demo       # native (slow)
make distilgpt2_demo_ffi  && ./demos/distilgpt2_demo_ffi   # full-forward FFI
make distilgpt2_demo_kv   && ./demos/distilgpt2_demo_kv    # KV-cache FFI

./prep/tokens.py decode   # any of the demos write generated IDs back
```

## Parity vs HF transformers

Reference produced by `prep/parity.py ref` (loads distilgpt2 via
PyTorch + transformers, dumps the last-row logits for the same
5-token prompt). All five Ruby paths match the reference at F32
round-trip precision:

| Path | max-abs diff | mean-abs diff | argmax | top-5 |
|---|---:|---:|---:|---:|
| Native Mat (f64) | `1.16e-4` | `1.93e-5` | ✓ | 5/5 |
| FFI-CPU full-forward (f32) | `5.42e-3` | `3.65e-3` | ✓ | 5/5 |
| FFI-CUDA full-forward (f32) | `5.42e-3` | `3.65e-3` | ✓ | 5/5 |
| FFI-CPU KV decode (f32) | `3.06e-3` | `1.36e-3` | ✓ | 5/5 |
| FFI-CUDA KV decode (f32) | `3.01e-3` | `1.37e-3` | ✓ | 5/5 |

(KV's max-abs is actually *better* than full-forward's because
per-position writes accumulate less F32 error than a full-sequence
matmul. CUDA vs CPU are equal to F32 ULP, as expected — both run
the same ggml graph on f32 inputs, just on different backends.)

Re-run on Mac:
- `make gpt2-parity` (native), `make gpt2-ffi-parity` (full-forward),
  `make gpt2-kv-parity` (KV).

Re-run on gx10:
- `make gpt2-ffi-parity-cuda`, `make gpt2-kv-parity-cuda`.

Then `./prep/parity.py compare --ours data/ours_*_logits.txt` from
the Mac.

## Bench

### M2 Air (single CPU worker, ggml-cpu backend)

`make gpt2-bench`, prompt = "Hello, my name is" (5 tokens):

| | ms/forward | forwards/sec | speedup vs native |
|---|---:|---:|---:|
| Native Mat, f64 | 1289.6 | 0.78 | 1× |
| FFI full-forward, T_SEQ=5 | 20.2 | 49.4 | 63.7× |
| FFI KV decode, pos=5..34 | 22.7 | 44.1 | 56.9× |

### gx10 (NVIDIA GB10, ggml-cuda backend)

`make gpt2-bench-cuda`, same prompt:

| | ms/forward | forwards/sec | speedup vs native (gx10) |
|---|---:|---:|---:|
| Native Mat, f64 | 374.8 | 2.67 | 1× |
| FFI-CUDA full-forward, T_SEQ=5 | 7.0 | 142.4 | 53.4× |
| FFI-CUDA KV decode, pos=5..34 | 6.2 | 160.6 | 60.2× |

End-to-end demo generation (`./demos/distilgpt2_demo_kv_cuda` on gx10) of
"Hello, my name is" → "Hello, my name is J.J.K. Rowling.":
- prefill (5 tokens): 33 ms
- 7 new tokens: 40 ms (5.6 ms/step)
- total: 73 ms (vs Mac CPU KV's 94 ms = ~22% faster)

At this toy shape (distilgpt2, T_SEQ=5..16) CUDA's advantage is
modest — the model is small and per-step compute is dominated by
graph-build overhead. CUDA pulls further ahead at larger models
(gpt2-medium 355M, gpt2-large 774M, gpt2-xl 1.5B) and longer
contexts, where the matmuls are big enough that GPU parallelism
matters.

### Long contexts (MAX_T up to 1024)

Demo defaults now run at MAX_T=1024 (GPT-2's full context window).
Tested with a 162-token prompt:

```
"The history of the modern computer is a remarkable journey
 spanning centuries... [162 tokens]... and finally"
   → "Today, computers are the most powerful technology in the
       world..."
```

Per-step decode stays ~11 ms at small positions, drifts up to
~14 ms by pos~190 (the linear-in-pos cost of the attention matmul
against K_history). At pos=1023 we'd project ~25 ms/step; full
1024-token generation from a 5-token prompt is ~15–20 s.

The greedy decoder will loop on repeated phrases at longer outputs
(classic small-model behaviour); temperature / top-k sampling is a
trivial follow-up and stays orthogonal to anything below.

One C-shim fix that bumping MAX_T exposed: `tnn_reset_for_rebuild`
used to swap graphs in the same compute ctx, leaving ~1300 dead
tensor headers per decode step. Now it tears ctx down and reinits
per step; ctx_w (persistent weights) is untouched so weights survive.

### Quantized GGUF (Q8_0 / Q4_0)

Same converter, add `--quantize q8_0` or `--quantize q4_0`:

```sh
./prep/convert_distilgpt2_to_gguf.py --repo-id gpt2 --out data/gpt2-q8_0.gguf --quantize q8_0
./prep/convert_distilgpt2_to_gguf.py --repo-id gpt2 --out data/gpt2-q4_0.gguf --quantize q4_0
```

Quantizes 2-D weight tensors only — Linear weights for q/k/v/o + ffn_up
+ ffn_down. Token + position embeddings, biases, and LayerNorm
gammas/betas stay f32 (small + quantizing them hurts more than it
saves). The existing `tnn_gguf_read_f32_to_doubles` FFI uses ggml's
type-traits `to_float` to dequantize per-tensor at load time; the rest
of the project doesn't change.

Size and behaviour on the canonical "Hello, my name is" prompt:

| | distilgpt2 (6 layers) | gpt2-small (12 layers) |
|---|---|---|
| F32 GGUF size | 327.7 MB | 497.8 MB |
| Q8_0 GGUF size | 202.9 MB (1.6×) | 248.3 MB (2.0×) |
| Q4_0 GGUF size | 181.7 MB (1.8×) | 205.8 MB (2.4×) |
| F32 generation | "Hello, my name is J.J.K. Rowling." | "Hello, my name is John. I'm a writer, and" |
| **Q8_0 generation** | "Hello, my name is J.J.J.K." (1 token drift) | **"Hello, my name is John. I'm a writer, and" (identical!)** |
| Q4_0 generation | (degenerates to loop) | "Hello, my name is J.J. Abrams. I'm" |

**Key result: gpt2-small Q8_0 produces byte-identical generation to
the F32 reference.** Distilgpt2 is small enough that even Q8_0's ~0.5
logit-unit noise occasionally flips an argmax; gpt2-small has enough
parameter redundancy that 8-bit quantization is effectively lossless
for greedy decode.

Q4_0 is too aggressive for distilgpt2 (model loops) and noticeable on
gpt2-small (different but coherent continuation). For comfortable
4-bit you want K-quants (Q4_K_M etc) which gguf-py doesn't have a
Python implementation of — those need llama.cpp's quantizer or a
C-side helper wrapping `ggml_quantize_chunk`. Listed as a follow-up.

Per-step cost increase (KV decode, gpt2-small):
- F32: 11 ms / step
- Q8_0: 15 ms / step (~35 % overhead from dequant)
- Q4_0: 12 ms / step (~10 % overhead; simpler dequant kernel)

Q4_0 being *faster* than Q8_0 is counter-intuitive but plausible —
fewer bytes through memory, fits more of the working set in cache.

### gpt2-small (124 M, 12 layers) — bigger model on both backends

Switching the demo's `GGUF` constant from `data/distilgpt2-f32.gguf`
to `data/gpt2-f32.gguf` is all it takes (hyperparams come from GGUF
metadata; see `GPT2ConfigLoader.read` in `lib/gguf_load.rb`).

Convert once:
```sh
./prep/convert_distilgpt2_to_gguf.py --repo-id gpt2 --out data/gpt2-f32.gguf
```

Bench results (T_SEQ=5, same prompt):

| | Mac CPU (M2) | gx10 CUDA (GB10) |
|---|---:|---:|
| Native Mat (f64) | 1667 ms / fwd | 1124 ms / fwd |
| FFI full-forward | 11.6 ms / fwd | 22.6 ms / fwd |
| FFI KV decode | 11.3 ms / step | 22.1 ms / step |

Interesting: at gpt2-small scale, Mac CPU FFI is **faster** than
gx10 CUDA. The per-step compute is small (12 layers × 12 heads at
d_head=64, T=5–34) and graph-build + kernel-launch overhead
dominates over arithmetic. The GPU advantage appears at larger
shapes (gpt2-medium / -large / -xl) and longer contexts; tiny
models stay CPU-friendly.

Parity vs HF gpt2 reference: max-abs diff `3.17e-3` (CPU) /
`6.11e-3` (CUDA); argmax + top-5 match exactly on both. End-to-end
generation "Hello, my name is" → "Hello, my name is John. I'm a
writer, and" — identical token sequence across CPU and CUDA.

### KV vs full-forward at growing T

The bench at T_SEQ=5 doesn't fully show the KV win: both FFI paths
are dominated by graph-build overhead at this scale. As T grows,
full-forward attention is O(T²), KV per-step is O(pos):

| | T_SEQ=5 (bench) | T_SEQ=16 (demo) |
|---|---:|---:|
| FFI-CPU full-forward | 20 ms / forward | 56 ms / forward |
| FFI-CPU KV decode | 22 ms / step (mean) | 7.4 ms / step |
| FFI-CUDA full-forward | 7 ms / forward | (n/a yet) |
| FFI-CUDA KV decode | 6.2 ms / step (mean) | 5.6 ms / step |

At T_SEQ=16 CPU KV is ~8× faster than CPU full-forward; CUDA KV
holds steady at ~6 ms regardless of T (per-step decode is O(pos)
but the constant is small at this model size).

## File map

```
├── prep/
│   ├── convert_distilgpt2_to_gguf.py  HF safetensors → data/*.gguf
│   ├── tokens.py                       BPE encode / decode (host-side)
│   └── parity.py                       HF reference logits + compare
│
├── lib/
│   ├── gpt2.rb              GPT2LM + GPT2Block (native, inference-only)
│   ├── gpt2_ffi.rb          GPT2FullForwardFFICache (full-fwd, CPU)
│   ├── gpt2_ffi_kv.rb       GPT2KVFFICache (per-step KV cache, CPU)
│   ├── gpt2_ffi_cuda.rb     GPT2FullForwardFFICacheCuda (full-fwd, CUDA)
│   ├── gpt2_ffi_kv_cuda.rb  GPT2KVFFICacheCuda (per-step KV cache, CUDA)
│   └── gguf_load.rb         Read GGUF tensors into a GPT2LM
│
├── tinynn/
│   ├── gguf_inspect.rb        List every tensor in a GGUF via FFI
│   ├── gpt2_build_smoke.rb    Toy-shape forward smoke
│   ├── gpt2_load_smoke.rb     Sentinel-value loader check
│   ├── gpt2_parity.rb         Native parity dump
│   ├── gpt2_ffi_parity.rb     CPU full-forward parity dump
│   ├── gpt2_kv_parity.rb      CPU KV-cache parity dump
│   ├── gpt2_ffi_parity_cuda.rb CUDA full-forward parity dump
│   ├── gpt2_kv_parity_cuda.rb  CUDA KV-cache parity dump
│   ├── kv_multi_cpy_smoke.rb  8-position single-head KV smoke
│   ├── gpt2_bench.rb          Native vs FFI vs KV bench (CPU)
│   └── gpt2_bench_cuda.rb     Same on CUDA
│
├── demos/distilgpt2_demo.rb           End-to-end generation (native)
├── demos/distilgpt2_demo_ffi.rb       End-to-end generation (full-fwd, CPU)
├── demos/distilgpt2_demo_kv.rb        End-to-end generation (KV, CPU)
├── demos/distilgpt2_demo_ffi_cuda.rb  End-to-end generation (full-fwd, CUDA)
└── demos/distilgpt2_demo_kv_cuda.rb   End-to-end generation (KV, CUDA)
```

## Architecture notes

### GPT2LM (lib/gpt2.rb)

Separate from `TransformerLM` because the existing class is heavy
with training scaffolding (Gradients, AdamState, layer caches, FFI
caches). For inference-only the simpler class is:

- learned token + absolute position embeddings, additive
- pre-LayerNorm with `gamma + beta` on every norm
- multi-head causal attention with `bias` on every Linear (q/k/v/o)
- GeLU FFN (the tanh approx, `gelu_new`) with bias on both Linears
- tied output embedding (`logits = x_final @ token_embedᵀ`)
- sequential residual

Per-head Q/K/V layout matches the existing `Block` so the FFI cache
can mirror what `FullForwardFFICache` already does.

### GGUF format & loader

Converter (`prep/convert_distilgpt2_to_gguf.py`) writes a llama.cpp-
compatible GGUF with naming `blk.N.{attn_norm,ffn_norm,attn_q,
attn_k,attn_v,attn_output,ffn_up,ffn_down}.{weight,bias}` plus
`token_embd.weight`, `position_embd.weight`, `output_norm.{weight,
bias}`. `c_attn` is pre-split into Q/K/V at conversion time so the
loader's per-head reshape is just a strided slice.

Loader (`lib/gguf_load.rb`) reads straight into the destination
`Mat.flat` / `Array<Float>` via `tnn_gguf_read_f32_to_doubles`
— no intermediate buffers, even for the 38.6 M-element `token_embd`.

### Full-forward FFI graph (lib/gpt2_ffi.rb)

`GPT2FullForwardFFICache` is the same shape as the existing
`FullForwardFFICache` (RMSNorm + no-bias), with two additions:

1. `tnn_layer_norm(x, gamma, beta, eps)` instead of `tnn_rms_norm`
2. Bias-adds after every Linear. ggml's `ggml_add` broadcasts via
   `ggml_can_repeat`, so the 1-D biases need no extra op — except
   the V bias must be shape `ne=[1, d_head]` (not `ne=[d_head, 1]`
   like Q/K) because the transposed-V matmul layout produces
   `ne=[T, d_head]` and the can_repeat rule cares about dim order.

Persistent T_SEQ at realize time; per-step pads input to T_SEQ.

### KV-cache decode (lib/gpt2_ffi_kv.rb)

`GPT2KVFFICache` extends the above with persistent K[block][head]
(d_head × max_T) and V[block][head] (max_T × d_head) buffers.

Per decode step at position `pos`:

1. `tnn_reset_for_rebuild(sess)` — clear compute graph, keep weights
2. Build one-position graph:
   - `get_rows(token_embed, [token_id])` + view of `pos_embed[pos]`
   - per block: LayerNorm, per-head Q/K/V matmuls, `cpy(k_new,
     view_2d(K, ..., pos))` + same for V, `matmul(K[0:pos+1], q)`,
     scale + softmax, `matmul(V[0:pos+1], attn)`, concat, out_proj,
     residual, LayerNorm, FFN, residual
   - final LayerNorm, tied unembed
3. `tnn_realize(t_logits)`, upload `token_id`, compute, download

Two primitives needed beyond the existing M2 work:

- **`tnn_add_to_graph(sess, tensor)`** (new) — exposes
  `ggml_build_forward_expand` so the `cpy(src, view_of_K)` ops are
  added to the graph even though their 1-row results aren't
  reachable from the realize target. Without this they'd be pruned
  and the K/V buffer never updates.
- **`cpy + view_2d` instead of `set_rows`/`set_2d`** — the pre-
  existing `ab_smoke_kv_attn_multi.rb` was a WIP that failed parity
  at pos≥1 because `ggml_set_2d`/`set_rows` use `ggml_dup_tensor`
  (returns a copy); the persistent K/V is never mutated.
  `ggml_cpy(src, view_of_K)` uses `ggml_view_tensor` on the
  destination, so the write lands in K's buffer. Multi-step
  attention then sees the accumulated history. Validated in
  `tinynn/kv_multi_cpy_smoke.rb` (8 positions, all pass).

Also: bumped `ctx_buf_size` in `tnn_session_new` from 8192 → 262144
tensor-header slots. The KV decode loop accumulates ~1280 compute-
graph node headers per step (6 layers × 12 heads × ~16 ops per
head + FFN/LN); old budget overflowed after ~30 rebuilds.

### Spinel name-collapse traps (recurring theme)

Spinel does whole-program type inference and unifies any name
(local var / method param / ivar) that appears with different
concrete types across the codebase. The same name in two scopes
becomes one polymorphic slot — and that boxing breaks `(void *)`
casts at FFI boundaries.

Renames needed during this work:

| was | became | reason |
|---|---|---|
| `m = GPT2LM.new(...)` (smoke) | `model = ...` | `m` was Mat in `softmax_rows!(m)` |
| `cache` in GPT2FFI / GPT2KV | `fwd_cache` / `kv_cache` | one was GPT2FullForwardFFICache, other GPT2KVFFICache |
| `tensor` param in `download_row_major` | `dl_handle` | dead `upload_transposed` mistyped its `tensor` param as mrb_int |
| `t_logits` ivar on GPT2KVStepResult | `kv_step_logits` | collided with FullForwardFFICache / GPT2FullForwardFFICache |
| `t` (Time.now) in bench | `iter_start` | `t = m.nrows` in transformer.rb |
| `null = TinyNN.tnn_null_ptr; [null]` | `[TinyNN.tnn_null_ptr]` | Spinel loses :ptr typing through local-var binding |

Diagnostic: errors like `'mrb_int' and 'sp_<OtherClass> *' in binary
expression` or `operand of type 'sp_RbVal' where arithmetic or
pointer type is required`. See
`~/.claude/projects/.../memory/feedback_spinel_var_name_collapse.md`
for the long-form note.

## Caveats / what's not in scope yet

- **Quantized GGUF unused.** Converter produces F32 (327.7 MB); the
  `tnn_gguf_read_f32_to_doubles` path dequantizes Q4_K/Q8_0/F16 via
  ggml's `to_float` traits but hasn't been exercised with a real
  model file. ~4× smaller distilgpt2 + similar arithmetic perf is
  on the table.

- **Reading hyperparams from GGUF metadata.** Caller currently
  hardcodes vocab/d_model/n_heads/etc to match what the converter
  wrote. Adding `tnn_gguf_kv_get_uint32` would close this.

- **BPE tokenizer is external.** `prep/tokens.py` is a Python
  shim (HF `tokenizers` package, rust-backed). A Ruby BPE encoder
  would let the Spinel binary be self-contained. ~200 lines of
  Ruby + a few Spinel name-collapse traps to fight.

- **No CUDA path yet for GPT-2.** `lib/tinynn_cuda.rb` already
  mirrors `lib/tinynn.rb` for the existing TransformerLM; the
  parallel `lib/gpt2_ffi_cuda.rb` is the natural next step.

- **Larger GPT-2 variants untested.** Same converter handles
  `--repo-id gpt2[-medium|-large|-xl]`. Memory cost scales linearly
  in params; gpt2-medium is ~700 MB F32 (RAM-tight on Mac at
  current disk pressure).

- **Context window is 1024** (DistilGPT2 / GPT-2 base). The
  KV-cache `max_T` parameter caps allocated buffer size; bench
  uses 32, demo uses 32. Larger contexts work but allocate more
  K/V memory per (block, head).

## CUDA path on gx10 (NVIDIA GB10): done

`lib/gpt2_ffi_cuda.rb` and `lib/gpt2_ffi_kv_cuda.rb` are 1:1 mirrors
of the CPU variants against the `TinyNNCuda` module. Class names
are suffixed `Cuda` per the existing `lib/tinynn_cuda.rb` pattern
(distinct from CPU classes to avoid Spinel's same-class-twice path,
since both modules end up loaded transitively via
`lib/transformer.rb`).

Generated by a Python regex pass from the CPU originals — the
class/module names, `TinyNN.` → `TinyNNCuda.`, and the
`tnn_session_new(0)` → `tnn_session_new(1)` flip (prefer-CUDA
backend) are the only differences. One hand edit on top: added
`tnn_add_to_graph` to `lib/tinynn_cuda.rb`'s `ffi_func` list, plus
an upload_int_array anchor block parallel to the one in
`lib/tinynn.rb` (Spinel needs the anchor to pin the `indices` param
type to `Array<Int>`, otherwise it defaults to `mrb_int` and the
`:int_array` FFI spec's `indices->data` access fails to compile).

Build + run pattern:

```sh
# On gx10 (NVIDIA GB10):
make setup-ggml-cuda                              # one-time
make gpt2-ffi-parity-cuda                         # parity probe
make gpt2-kv-parity-cuda                          # KV parity probe
make gpt2-bench-cuda                              # bench
make distilgpt2_demo_kv_cuda && \
  ./demos/distilgpt2_demo_kv_cuda                       # end-to-end demo
```

See the **Bench / gx10** section above for numbers.

## Future directions (smallest-effort first)

- **Quantized GGUF.** Converter produces F32 (327.7 MB); the
  `tnn_gguf_read_f32_to_doubles` path dequantizes Q4_K/Q8_0/F16 via
  ggml's `to_float` traits but hasn't been exercised with a real
  model file. ~4× smaller distilgpt2 on disk, identical arithmetic
  perf (ggml-cpu still computes in f32).

- **Reading hyperparams from GGUF metadata.** Caller currently
  hardcodes vocab/d_model/n_heads/etc to match what the converter
  wrote. Adding `tnn_gguf_kv_get_uint32` would close this.

- **BPE tokenizer in Ruby.** `prep/tokens.py` is a Python shim (HF
  `tokenizers`, rust-backed). A Ruby BPE encoder would let the
  Spinel binary be self-contained. ~200 lines + Spinel name-
  collapse traps.

- **Larger GPT-2 variants.** Same converter handles `--repo-id
  gpt2[-medium|-large|-xl]`. Memory cost scales linearly in params;
  gpt2-medium is ~700 MB F32 (RAM-tight on Mac), gpt2-large needs
  ggx10. CUDA backend would show its advantage at these shapes.

- **Longer contexts.** KV-cache `max_T` parameter caps allocated
  buffer size; bench uses 32, demo uses 32. Up to 1024 (DistilGPT2 /
  GPT-2 base context) works with proportionally more memory per
  (block, head).
