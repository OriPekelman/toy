# Qwen2.5-0.5B: scratch-buffer overflow in transposed upload (FIXED)

## Status

**Fixed in step 45 (2026-05-17)**: `tnn_upload_transposed_f64` does
the staging + upload in chunks of `floor(scratch_slots / br)` columns,
so weight tensors larger than the 16 MiB scratch buffer upload
correctly.

Qwen2.5-0.5B now runs end-to-end via the FFI KV path:

```
Hello, my name is a 10 year old boy. I have a question about my hair.
```

~80 ms/token on gx10 CPU, parity with the native Float64 path on
finite-logit stats.

## Root cause (what the trace tap found)

The old `stage_transposed_and_upload` in `lib/tinynn.rb` staged the
entire transposed matrix into `sess->scratch` via per-element
`tnn_scratch_set`, then called `tnn_upload(sess, target)` which copies
the full tensor's `ggml_nbytes(t)` from scratch to the backend buffer.

Scratch is 16 MiB = 4,194,304 f32 slots (`TNN_SCRATCH_BYTES /
sizeof(float)`). For Qwen's `ffn_gate.weight` shape (896 × 4864):

- `d_model × d_ff = 4,358,144` floats (~17.4 MB) — **exceeds scratch**
- `tnn_scratch_set` silently dropped writes at index ≥ 4,194,304
  (its guard clause returned with no error)
- `tnn_upload` then memcpy'd `ggml_nbytes(t) = 17.4 MB` from scratch
  to the backend — reading 0.6 MB **past the end** of the scratch
  buffer (heap garbage from adjacent allocations)
- The trailing rows of the matmul weight contained random f32 bit
  patterns; one of them was a denormal/large pattern that produced
  output values around ±1e+37 (just under f32 max ~3.4e+38)
- The next matmul in the FFN chain overflowed to inf, then to NaN

## What the trace tap revealed (the bisection)

`demos/qwen_probe.rb` with `TRACE=1 FORCE_N_LAYERS=1` printed:

```
L0.rn2  n= 896 min=-4.6 max=2.22  |mean|=0.53  nan=0
L0.gate n=4864 min=-8.77e+37 max=1.38e+37 |mean|=4.9e+34 nan=6
L0.up   n=4864 min=-8.77e+37 max=1.38e+37 |mean|=4.9e+34 nan=6
L0.silu_gate ... nan=1
L0.gated ... |mean|=Infinity nan=1
L0.dn   ... |mean|=NaN nan=896
```

`rn2` finite → `gate` overflow. The gate input (`rn2`) was healthy,
the gate weights were healthy (max ±0.5 in the GGUF), the matmul
arithmetic itself was healthy — the only explanation was a corrupted
weight upload. Verified: Qwen's ffn_gate is the first matmul whose
weight bytes exceed scratch in the SmolLM2-shaped path.

## Fix

`tnn_upload_transposed_f64(sess, target, src, br, bc)` in
`tinynn/tinynn_ggml.c`:

```c
int cols_per_chunk = max_slots / br;        // 4M / 896 = 4464
for (j_start = 0; j_start < bc; j_start += cols_per_chunk) {
    int j_end = min(j_start + cols_per_chunk, bc);
    // Stage columns [j_start, j_end) transposed into scratch[0..]
    for (j = j_start; j < j_end; j++)
        for (i = 0; i < br; i++)
            scratch[(j - j_start) * br + i] = src[i * bc + j];
    // Upload slice at byte offset j_start * br * 4
    ggml_backend_tensor_set(t, scratch,
        j_start * br * sizeof(float),
        (j_end - j_start) * br * sizeof(float));
}
```

For Qwen2.5-0.5B ffn_gate: 2 chunks (4464 cols, then 400 cols). For
TinyLlama-1.1B ffn_gate (2048 × 5632 = 11.5M floats): 3 chunks.

Same `:float_array` Spinel #474 zero-copy semantics — `mat.flat`
passes through as a `const double *` without per-element FFI calls.

## Trace tap infrastructure

The diagnostic that made the bisection possible is now reusable.
`SmolLM2KVFFICache#enable_trace!` switches the cache into trace mode;
the `build_decode_step` path inserts `trace_tap("name", t)` calls at
~12 named subblock outputs per layer. With trace off, `trace_tap`
is one `if @trace_on` branch — no extra ops enter the graph, no
extra `tnn_set_output` calls, hot path unchanged. With trace on, the
scheduler keeps the tapped tensors' buffers alive past compute; after
compute, `dump_trace` downloads each into scratch and prints
min/max/|mean|/NaN stats via four C-side O(n) reductions
(`tnn_scratch_{min,max,sum_abs,nan_count}_f32`).

Usage: `TRACE=1 FORCE_N_LAYERS=1 ./demos/qwen_probe`. See
`lib/toy_smollm2_ffi_kv.rb` for the build-path tap insertion points.

## Other models unblocked

The same bug affected:

- **TinyLlama-1.1B**: `ffn_gate` is 2048 × 5632 = 11.5M floats. The
  earlier `docs/tinyllama-known-issue.md` diagnosis ("f32 precision
  overflow at L=5") was wrong — same scratch overflow. Now produces
  coherent English ("Once upon a time, there was a young girl named
  Lily. Lily was a kind and").
- **Qwen2.5-1.5B**: 1536 × 8960 = 13.8M (untested, predicted to work).
- **Qwen2.5-7B and larger**: predicted to work for the upload path;
  separate memory question for the d_model × d_ff Mat allocation
  in Ruby (Qwen2.5-7B's down_proj alone is 67.9M floats = 543 MB
  in Ruby's Array<Float64>).

## Lesson

Silent-failure primitives are dangerous. `tnn_scratch_set` returning
`void` and ignoring out-of-range indices meant a 4% upload truncation
flowed through as a NaN cascade 4 layers deep. A logging fprintf or
return-an-error on the very first out-of-range write would have
caught this in minutes instead of weeks. The trace tap is the
right shape of tooling to land *with* primitives like this — it lets
you see when a layer's outputs diverge from expectation without
needing to commit to a specific failure-mode hypothesis.
