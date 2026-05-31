# CUDA BYO-pointer: design + patch sketch for ggml-cuda

**Status:** scope + draft patch. Not landed. GPU was 96% utilized
during this session so the build+run validation step is deferred.

## Why this matters

Today's `ggml_backend_cuda_buffer_from_ptr` does not exist; the only
host-pointer story is `ggml_backend_cuda_register_host_buffer`, which
just pins host pages for fast DMA — the device buffer remains a
separate allocation. That means the Phase 2 BYO-pointer win
(7B-Q8 peak RSS 19 → 7.4 GB on CPU) does NOT carry over to CUDA.

On GB10 / DGX Spark this is a particularly painful gap: the physical
memory is unified, so a host pointer IS valid in device address space
via UVA. The cudaHostRegister + cudaHostGetDevicePointer dance lets
us route a host pointer through CUDA's virtual-memory plumbing for
near-zero extra cost. ggml-cuda just doesn't expose this as a buffer
constructor.

## Proposed API

In `ggml-cuda.h`:

```c
// Wrap an externally-managed host pointer as a CUDA backend buffer.
// On UVA-enabled platforms (most modern NVIDIA, all unified-memory
// SKUs including GB10), the host pointer is directly addressable
// from device kernels after cudaHostRegister(...HostRegisterMapped).
//
// The buffer takes ownership of the host pointer's CUDA registration
// state (cudaHostUnregister on free) but NOT of the underlying
// memory (caller manages that — typically via mmap + munmap).
//
// Returns NULL on failure (cudaHostRegister error, non-UVA platform,
// out-of-range size).
GGML_BACKEND_API ggml_backend_buffer_t
ggml_backend_cuda_buffer_from_ptr(void *host_ptr, size_t size, int device);
```

## Implementation sketch (ggml-cuda.cu)

```c++
struct ggml_backend_cuda_buffer_from_ptr_context {
    int    device;
    void  *dev_ptr;   // UVA-mapped device pointer (== host_ptr on
                      // unified-memory SKUs; different on discrete
                      // GPUs but still device-addressable)
    void  *host_ptr;  // original host pointer (for cudaHostUnregister)
    size_t size;
    std::string name;
};

static void from_ptr_free(ggml_backend_buffer_t buffer) {
    auto *ctx = (ggml_backend_cuda_buffer_from_ptr_context *)buffer->context;
    if (ctx->host_ptr) {
        ggml_cuda_set_device(ctx->device);
        cudaError_t err = cudaHostUnregister(ctx->host_ptr);
        if (err != cudaSuccess) cudaGetLastError();  // clear
    }
    delete ctx;
}

static void *from_ptr_get_base(ggml_backend_buffer_t buffer) {
    auto *ctx = (ggml_backend_cuda_buffer_from_ptr_context *)buffer->context;
    return ctx->dev_ptr;
}

// Read-only by convention: tensors created against this buffer
// are mmap'd file pages. set_tensor would corrupt the file or
// write to read-only mmap pages. For a generic implementation we
// could fall back to cudaMemcpy, but for the inference path this
// should never happen.
static void from_ptr_set_tensor(ggml_backend_buffer_t buffer,
                                  ggml_tensor *tensor,
                                  const void *data,
                                  size_t offset, size_t size) {
    GGML_ABORT("ggml_backend_cuda_buffer_from_ptr is read-only");
}

static void from_ptr_get_tensor(ggml_backend_buffer_t buffer,
                                  const ggml_tensor *tensor,
                                  void *data,
                                  size_t offset, size_t size) {
    // Host-readable directly — no kernel launch needed.
    memcpy(data, (const char *)tensor->data + offset, size);
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_from_ptr_iface = {
    .free_buffer     = from_ptr_free,
    .get_base        = from_ptr_get_base,
    .init_tensor     = NULL,           // no padding zero — caller owns bytes
    .memset_tensor   = NULL,           // read-only
    .set_tensor      = from_ptr_set_tensor,
    .get_tensor      = from_ptr_get_tensor,
    .set_tensor_2d   = NULL,
    .get_tensor_2d   = NULL,
    .cpy_tensor      = NULL,
    .clear           = NULL,
    .reset           = NULL,
};

ggml_backend_buffer_t ggml_backend_cuda_buffer_from_ptr(
    void *host_ptr, size_t size, int device)
{
    if (!host_ptr || size == 0) return NULL;
    if ((uintptr_t)host_ptr % TENSOR_ALIGNMENT != 0) return NULL;

    ggml_cuda_set_device(device);

    cudaError_t err = cudaHostRegister(host_ptr, size,
                                        cudaHostRegisterPortable |
                                        cudaHostRegisterMapped |
                                        cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // Try without ReadOnly for older drivers / runtimes.
        cudaGetLastError();
        err = cudaHostRegister(host_ptr, size,
                                cudaHostRegisterPortable |
                                cudaHostRegisterMapped);
        if (err != cudaSuccess) {
            cudaGetLastError();
            return NULL;
        }
    }

    void *dev_ptr = nullptr;
    err = cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
    if (err != cudaSuccess) {
        cudaGetLastError();
        cudaHostUnregister(host_ptr);
        return NULL;
    }

    auto *ctx = new ggml_backend_cuda_buffer_from_ptr_context{
        device, dev_ptr, host_ptr, size,
        GGML_CUDA_NAME + std::to_string(device) + "_mapped"
    };
    return ggml_backend_buffer_init(
        ggml_backend_cuda_buffer_type(device),   // reuse existing buffer-type
        ggml_backend_cuda_buffer_from_ptr_iface,
        ctx, size);
}
```

## Test plan (post-GPU-available)

1. **Build**: rebuild ggml with `GGML_USE_CUDA=ON`. Verify the new
   symbol is exported (`nm libggml-cuda.so | grep buffer_from_ptr`).

2. **Smoke test** (standalone C program against ggml-cuda):

   ```c
   // 1. Open a small GGUF, gguf_init_from_file(..., no_alloc=true)
   // 2. mmap the file (PROT_READ, MAP_PRIVATE)
   // 3. Create a CUDA backend on device 0
   // 4. ggml_backend_buffer_t buf =
   //      ggml_backend_cuda_buffer_from_ptr(mmap_base, file_size, 0)
   // 5. Allocate a single tensor into the buffer at the right offset
   // 6. Build a graph: mul_mat(weight_tensor, dummy_activation)
   // 7. ggml_backend_graph_compute. Read result.
   // 8. Compare to CPU compute on the same weights — must match
   //    to f32 precision (modulo per-block dequant order for Q8).
   ```

3. **Ruby integration**: mirror the CPU `tnn_session_attach_weight_mmap`
   in `lib/tinynn_cuda.rb` (paired with `tinynn_ggml_cuda.cu` for the
   CUDA shim). Reuse `realize_for_mmap` semantics on a CUDA session.

4. **End-to-end**: run `demos/qwen25_7b_native_mmap_q8.rb` adapted
   to CUDA. Expect:
   - Realize+attach ≈ 30 ms (same as CPU — just pointer wiring)
   - Generation ~10× faster than CPU (matmul kernel is GPU)
   - Peak RSS ≤ 9.7 GB (mmap pages — cuda registration adds no
     extra allocation on UVA SKUs)

## Open questions / risks

1. **`cudaHostRegisterReadOnly`** is only supported on driver ≥ 11.4.
   Fallback to non-ReadOnly is fine. The patch tries both.

2. **`init_tensor` skipped**: today's CUDA path zero-pads quantized
   tensors. For mmap'd Q8_0, the file should already have the right
   bytes (gguf pads to TENSOR_ALIGNMENT during write). If a tensor's
   `ggml_nbytes` doesn't equal its file-stored size we may need to
   handle padding explicitly. To verify: dump
   `ggml_backend_buft_get_alloc_size(...)` vs
   `gguf_get_tensor_size(...)` on a Q8 layer.

3. **Discrete GPU compat**: on a discrete card, `cudaHostGetDevicePointer`
   returns a different address than `host_ptr`, and kernel reads
   from that address pull bytes over PCIe per access. Functionally
   correct but slow. The patch detects nothing — it's the caller's
   problem to know they're on a UVA / GB10 SKU. Future: add a probe
   helper that returns whether the device can read from host memory
   at full speed (managed memory `cudaDeviceProp::pageableMemoryAccess`).

4. **Non-tensor-aligned mmap region**: mmap always returns
   page-aligned (4 KiB ≫ TENSOR_ALIGNMENT). GGUF tensor offsets
   are at least 32-aligned. So `host_ptr + tensor_offset` is
   tensor-aligned. The constructor only asserts on the base.

## Estimated effort to land

- Patch + build:                  2 hours
- Standalone CUDA smoke test:     2 hours (write + debug)
- Ruby/Spinel CUDA-side wiring:   3 hours (mirrors CPU work, but
                                  needs careful FFI types for the
                                  ggml-cuda symbol)
- End-to-end + parity verify:     1 hour

Total: ~8 hours. The patch itself is small (<100 lines); the time is
in test infrastructure + Ruby plumbing for the CUDA path.

## Upstream contribution path

This belongs upstream — there's no good reason for ggml-cuda not to
have a `buffer_from_ptr` peer to the CPU backend. Two paths:

1. **Issue first**: file against `ggml-org/llama.cpp` asking "why no
   ggml_backend_cuda_buffer_from_ptr?" — the maintainers may have a
   reason (e.g. they prefer cudaMallocManaged) and there might be a
   different recommended path.

2. **PR with this patch**: submit the implementation, with the smoke
   test as `tests/test-cuda-buffer-from-ptr.cu`.

Path 1 first; if no objection, follow up with path 2.
