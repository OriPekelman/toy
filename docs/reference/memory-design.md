# Inference memory: from 12 B/w to zero-copy

The inference path used to pay **12 bytes per weight** — 8 in the Ruby
`Mat` (Float64) plus 4 in the FFI persistent buffer (f32). For a 7B
model that is 60 GB + 30 GB = 90 GB before activations, and Qwen2.5-7B
did not fit on gx10. The current loader pays **0 extra bytes**: the
weights *are* the mmap'd GGUF file pages.

This doc records the design that got us there and the layout/precision
constraints behind it.

## The original duplication

```
GGUF f32 on disk (4 B/w)
        ↓ read + f32→f64 widen
   Ruby Mat (8 B/w, Float64)        ← native (pure-Ruby) forward uses this
        ↓ f64→f32 narrow + transpose
   FFI buffer (4 B/w, f32)          ← FFI / CUDA forward uses this
```

The `Mat` only exists for the native forward path. For inference via
FFI it is allocated, populated from GGUF, uploaded once, then sits idle
for the whole decode loop. The fix is to stop materializing it.

PyTorch never has this problem because its tensors *are* the device
buffer: `safetensors` mmaps the file, tensors point at the mapped
pages, and on unified-memory hosts (Apple MPS, NVIDIA GB10) the device
buffer and host pointer reference the same physical bytes. We carried
two data types (Ruby `Mat` and ggml tensor) with a copy between them;
closing that gap is the whole story.

## Three paths — and where we landed

| Approach | Extra B/w | 7B total | Status |
|---|---|---|---|
| Mat-mediated (legacy) | 12 (Mat f64 + FFI f32) | 90 GB | superseded |
| **(A)** Skip Mat: GGUF → FFI directly | 4 (f32) | 30 GB | shipped |
| **(B)** Mat backed by f32 `:float_array` | 4 | 30 GB | not pursued (training-only benefit) |
| **(C)** mmap GGUF; ggml tensor data → mapped page | 0 | shared with file cache | shipped (CPU + CUDA) |

(B) is orthogonal — it would help training, where the `Mat` is the hot
data, but it is unnecessary for inference once (A)/(C) exist.

### (A) — direct GGUF → FFI loader

A set of C primitives walks each GGUF tensor and copies its bytes into
the corresponding FFI persistent buffer, skipping the Ruby `Mat`
entirely. Where a transpose is needed it does the chunked transposed
write; for 1-D γ / biases it is a plain `memcpy`. Result: **4 B/w**,
matching PyTorch at bf16, and Qwen2.5-7B fits at ~30 GB peak.

### (C) — mmap GGUF into ggml tensors

The llama.cpp model. ggml's CPU backend exposes
`ggml_backend_cpu_buffer_from_ptr(void *ptr, size_t size)`: wrap an
mmap'd region as a backend buffer, and tensors created within it point
at the mapped pages directly — no copy. Two things had to be true
first.

**Native layout.** The converter used to transpose 2D linear weights
from HF's `[out, in]` to `[in, out]` at write time — a legacy of the
old `Mat`-side `[in, out]` convention, and the only thing forcing a
load-time byte fixup. The converter's `--ggml-native` flag writes
bytes that already match ggml's column-major `ne=[in, out]` layout, so
the loader can `memcpy` or `mmap` without touching the bytes. Per-head
Q/K/V slices become contiguous byte ranges in this layout — ideal for
mmap. The loader auto-dispatches on the `toy.ggml_native` GGUF
metadata key, so callers stay layout-agnostic.

**Q8 stays Q8 (Phase 3).** Persistent 2D linear weights are allocated
with `GGML_TYPE_Q8_0` directly rather than dequantized to f32 at load.
ggml's `mul_mat` auto-dispatches to its Q8 vec_dot kernel for mixed
activation-F32 × weight-Q8 ops, so f32 parity is preserved while the
file stays quantized in memory.

The one structural change Q8 forced: ggml's `mul_mat` accepts the
quantized operand only in the `src0` position. The V matmul was the
only call site with the weight in `src1`; it was flipped to
weight-first (result shape `ne=[1, d_head]` → `ne=[d_head, 1]`), and a
zero-copy `view_2d` reinterprets the bytes for the V-cache write. The V
bias drops from 2-D to 1-D `[d_head]`.

## Memory profile (verified, native Q8, CPU)

```
                          peak RSS    steady-state    notes
Qwen2.5-0.5B native f32     3.87 GB     same          baseline
Qwen2.5-0.5B native Q8      1.82 GB     same          ~53% saving, ~26% faster
Qwen2.5-1.5B native Q8      4.56 GB     similar        (vs ~18 GB legacy)
Qwen2.5-7B  native Q8      19.0 GB      9.4 GB         (vs ~30 GB legacy)
```

On 7B the 19 GB load peak is a double-buffer effect: ggml's
`gguf_init_from_file` holds the f32-equivalent of the file while the
verbatim-copy loader fills the persistent buffer. With true mmap
(`ggml_backend_cpu_buffer_from_ptr`, weights ARE the file pages) the
double buffer is gone and the peak collapses toward steady-state — the
realize+attach step is just pointer wiring.

The CPU code path lives in `tinynn/tinynn_ggml.c`
(`tnn_session_attach_weight_mmap` → `ggml_backend_cpu_buffer_from_ptr`)
and `tinynn/tinynn_gguf.c`; the Ruby side is
`lib/toy/llm/engine/llama_seq_engine.rb` (`realize_for_mmap`). FFI bindings:
`lib/tinynn.rb`, `lib/tinynn_cuda.rb`, `lib/tinynn_metal.rb`.

## CUDA: UVA buffer-from-ptr (landed)

On GB10 / DGX Spark the physical memory is unified, so a host pointer
is directly addressable from device kernels via UVA. The CPU
zero-copy win carries over to CUDA through
`ggml_backend_cuda_buffer_from_ptr(void *host_ptr, size_t size, int device)`,
which stock ggml-cuda does not ship. We vendor it.

Mechanism: `cudaHostRegister(...HostRegisterMapped)` (with a
`HostRegisterReadOnly` attempt first, falling back for older drivers)
pins the mmap'd host pages, and `cudaHostGetDevicePointer` returns the
device-addressable pointer — on a UVA SKU this equals the host pointer,
adding no extra allocation. The buffer is read-only by convention
(`set_tensor` aborts; tensors are mmap'd file pages); `get_tensor`
reads host memory directly. The buffer takes ownership of the CUDA
*registration* state (`cudaHostUnregister` on free) but not the
underlying memory — the caller's mmap owns that.

On a discrete (non-UVA) card `cudaHostGetDevicePointer` returns a
distinct address and kernel reads cross PCIe per access: functionally
correct, slow. The patch detects nothing; the caller must know it is
on a unified-memory host.

Where it lives:
- API: `vendor/ggml/include/ggml-cuda.h`.
- Implementation: `vendor/ggml/src/ggml-cuda/ggml-cuda.cu`
  (`ggml_backend_cuda_buffer_from_ptr_*`).
- Vendored as patches **0001–0003** in `vendor-patches/`
  (buffer_from_ptr, iface reuse, copy-mode). See `../gating.md` for the
  full patch roster and the idempotent apply gate.
- Toy shim: `tinynn/tinynn_backend_cuda.c`
  (`tnn_cuda_buffer_from_ptr_internal`), with the weak hook in
  `tinynn/tinynn_ggml.c`.
- Smoke checks: `demos/cuda_byo_ptr_smoke.rb` (Ruby) and
  `tinynn/cuda_byo_smoke.c` (standalone C — mmap a GGUF, wrap it,
  `mul_mat`, compare to CPU).

Metal has no public buffer-from-ptr peer, so the Metal path routes
`tnn_session_attach_weight_mmap` through the CPU buffer and lets the
scheduler copy host pages to the device (`lib/tinynn_metal.rb`).
