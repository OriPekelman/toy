# Issue draft: ggml_backend_cuda_buffer_from_ptr (mmap zero-copy on UVA)

**Target repo:** `ggml-org/ggml` (or `ggml-org/llama.cpp` if you'd
prefer it lives there; the patch is in `ggml-cuda.cu` either way).

**Type:** Feature request + tentative PR (working patch attached
locally; willing to upstream).

---

## Title

ggml-cuda: add `ggml_backend_cuda_buffer_from_ptr` for UVA-direct mmap inference

## Body

### Use case

Phase 2 of a small mmap zero-copy refactor in a downstream toy
project lets the CPU backend keep weight tensors as direct views
into an mmap'd GGUF (`ggml_backend_cpu_buffer_from_ptr` + tensors
allocated via `ggml_backend_tensor_alloc(buf, t, mmap_base +
offset)`). Wins on Qwen2.5-7B-Q8 (gx10 CPU):

| Path | Load wall | Peak RSS |
|---|---|---|
| Dequant + persistent buffer | 33s | ~30 GB |
| mmap, copy into persistent buffer | 15s | ~18 GB |
| **BYO-pointer (CPU, mmap IS the buffer)** | **30 ms** | **7.4 GB** |

On GB10 / DGX Spark / Jetson (unified memory), the same trick
should work for the CUDA backend — host pointers are
device-addressable via UVA after `cudaHostRegister(...
HostRegisterMapped)`. But ggml-cuda has no equivalent of
`buffer_from_ptr`. The closest is
`ggml_backend_cuda_register_host_buffer`, which only pins host
pages for cudaMemcpy DMA; the device-side buffer is still a separate
allocation.

This issue is a feature request for a CUDA-side
`ggml_backend_cuda_buffer_from_ptr`.

### What the patch does

(Attached as a working downstream patch — see
[oripekelman/toy_ruby_neural_network commit
`e302d32`](https://github.com/oripekelman/toy_ruby_neural_network/commit/e302d32),
and the vendored ggml commit `5f3bee4` in the same project.)

```c
GGML_BACKEND_API ggml_backend_buffer_t
ggml_backend_cuda_buffer_from_ptr(void *host_ptr, size_t size, int device);
```

Implementation outline:

- `cudaHostRegister(host_ptr, size, Portable | Mapped [| ReadOnly])`
  to register the mmap'd region (fall back to no-ReadOnly for older
  runtimes).
- `cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0)` — returns
  `host_ptr` on UVA SKUs.
- Wrap in a backend buffer with a small read-only interface
  (`set_tensor` aborts, `get_tensor` is a host memcpy, `free_buffer`
  cudaHostUnregisters but doesn't free the underlying memory).
- Backed by a dedicated `buffer_type_t` whose `get_alloc_size` is
  `NULL` (defaults to `ggml_nbytes`) — no `MATRIX_ROW_PADDING`
  expansion past the tensor end, since the mmap'd file doesn't have
  padding after each tensor's bytes.

### Smoke test result (GB10)

```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 124546 MiB):
  Device 0: NVIDIA GB10, compute capability 12.1, VMM: yes
[1] mmap'd qwen25-1.5b-native-q8.gguf: 2326177120 bytes
[2] CUDA backend initialised
[3] cuda_buffer_from_ptr OK; buffer = 0xc38d2441e050
[4] buffer base = 0xecc52a600000 (host = 0xecc52a600000; UVA (host==dev))
[5] buffer size = 2326177120 bytes (file = 2326177120)
[6] first 4 bytes via dev_ptr: 'GGUF' (expect 'GGUF')
[7] buffer freed
[8] OK
```

### Known limitations

1. **MATRIX_ROW_PADDING for quantized matmul:** ggml-cuda's
   quantized matmul kernels read the last row up to a
   512-element-aligned boundary. The default
   `get_alloc_size` rounds quantized tensors up. The mmap'd file
   doesn't have that padding past each tensor. For tensors whose
   `ne0 % 512 != 0` (e.g. Qwen2.5-0.5B, d_model=896), the kernel
   would read into adjacent tensor bytes (not crash; the masked-out
   positions don't affect output, but it's an undocumented invariant
   to rely on).
   
   For d_model ∈ {1536, 2048, 3584} this is irrelevant; matmul
   reads stay within tensor bounds. Worth either (a) documenting
   the requirement, or (b) probing per-tensor and falling back to
   the regular allocated buffer when padding would be needed.

2. **Discrete GPUs:** On a non-UVA card,
   `cudaHostGetDevicePointer` returns a different address than
   `host_ptr` and kernel reads pull bytes over PCIe per access —
   functionally correct, performance terrible. The patch makes no
   attempt to detect this; could add a `cudaDeviceProp::
   pageableMemoryAccess` probe + warn.

3. **ReadOnly flag:** `cudaHostRegisterReadOnly` requires CUDA
   runtime ≥ 11.4. The patch falls back gracefully.

### Why "from_ptr" and not "register" + alloc-tensor-at-pointer?

Both APIs would work. `from_ptr` matches the CPU side's existing
shape and feels like the natural counterpart. The alternative — pin
externally, then `ggml_backend_tensor_alloc(some_cuda_buffer,
tensor, ptr)` — would require relaxing the
`ggml_backend_cuda_buffer_context::dev_ptr` ownership model in the
allocated buffer's destructor. `from_ptr` keeps the surgery local
to one new buffer type.

### Asks

- Sanity check on the approach.
- Permission to upstream the patch (happy to clean up + submit as a
  PR with a test).
- Guidance on the MATRIX_ROW_PADDING question — should
  buffer_from_ptr probe per-tensor and reject quantized weights
  with insufficient row padding, or document the caller's
  responsibility?

### Acknowledgement

Downstream patch was developed against ggml @ commit `5725fee` on
gx10 (GB10 unified memory). Happy to share the smoke test + the
Phase 2 design doc if useful.

---

## gh command (when auth'd)

```sh
gh issue create \
  --repo ggml-org/ggml \
  --title "ggml-cuda: add ggml_backend_cuda_buffer_from_ptr for UVA-direct mmap inference" \
  --body-file docs/upstream-issues/02-cuda-buffer-from-ptr.md
```
