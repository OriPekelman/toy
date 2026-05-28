/* tinynn_backend_cuda.c — CUDA backend init + BYO-pointer hook.
 *
 * Lives in its own .o so libtinynn_ggml_cuda.a contains only the
 * symbols that bridge to ggml-cuda — no overlap with libtinynn_ggml.a.
 * The common-side tnn_engine_get / tnn_session_attach_weight_mmap (in
 * tinynn_ggml.c) call these through weak references, so CPU-only
 * programs link cleanly without this archive and CUDA programs pull
 * the needed symbols.
 *
 * Compiled only into libtinynn_ggml_cuda.a (rule in Makefile).
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

ggml_backend_t tnn_backend_cuda_init_internal(void)
{
    return ggml_backend_cuda_init(0);
}

/* GH#3 — multi-GPU mode 1. Strong override of the weak stub in
 * tinynn_ggml.c. CPU-only builds keep the weak stub (returns NULL);
 * CUDA programs link the archive and get device-parametric init.
 *
 * Device validation lives upstream in tnn_engine_get_on; we just
 * pass through. Returns NULL if the device index isn't a valid
 * CUDA device (ggml_backend_cuda_init will assert/abort on
 * out-of-range, so callers should check device_count first). */
ggml_backend_t tnn_backend_cuda_init_internal_on(int device)
{
    return ggml_backend_cuda_init(device);
}

/* GH#3 — bind ggml_backend_cuda_get_device_count for runtime
 * enumeration. Weak stub in tinynn_ggml.c returns 0; this strong
 * override returns the real GPU count when the CUDA archive is
 * linked. */
int tnn_cuda_get_device_count_internal(void)
{
    return ggml_backend_cuda_get_device_count();
}

/* Phase 2 BYO-pointer on CUDA: wraps an mmap'd region in a
 * ggml_backend_cuda_buffer_from_ptr (vendored patch — see
 * docs/cuda-byo-pointer-design.md). Strong override of the weak
 * stub in tinynn_ggml.c. */
ggml_backend_buffer_t tnn_cuda_buffer_from_ptr_internal(void *host_ptr,
                                                         size_t size,
                                                         int device)
{
    return ggml_backend_cuda_buffer_from_ptr(host_ptr, size, device);
}

/* Forcing-reference symbol. Pass `-Wl,-u,tnn_cuda_force_link` to the
 * linker to force this object (and transitively libggml-cuda.a) to
 * be pulled in from libtinynn_ggml_cuda.a. Without this, the weak
 * tnn_backend_cuda_init_internal fallback in tinynn_ggml.c satisfies
 * the symbol table and the strong overrides here never get linked —
 * resulting in a "CUDA" binary that silently runs on CPU. */
void tnn_cuda_force_link(void)
{
    /* Reference ggml_backend_cuda_init so the linker also pulls in
     * libggml-cuda.a. */
    volatile void *p = (void *)&ggml_backend_cuda_init;
    (void)p;
}

/* Strong override of the weak tnn_pinned_alloc / tnn_pinned_free in
 * tinynn_ggml.c. cudaHostAlloc allocates pinned (page-locked) host
 * memory; cudaMemcpy from a pinned source bypasses the driver's
 * internal staging copy and can DMA directly. For toy's per-step
 * label-upload (heavy bench: t_labels = 155 MB at vocab=151936×T=256)
 * this halves the wall-clock cost of the transfer. */
void *tnn_pinned_alloc(size_t bytes)
{
    void *p = NULL;
    cudaError_t err = cudaHostAlloc(&p, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        fprintf(stderr, "[tnn-cuda] cudaHostAlloc(%zu) failed: %s — "
                        "falling back to pageable\n",
                        bytes, cudaGetErrorString(err));
        return calloc(1, bytes);
    }
    return p;
}

void tnn_pinned_free(void *p)
{
    if (!p) return;
    cudaError_t err = cudaFreeHost(p);
    if (err != cudaSuccess) {
        fprintf(stderr, "[tnn-cuda] cudaFreeHost failed: %s\n",
                cudaGetErrorString(err));
        free(p);  /* try pageable free as fallback */
    }
}
