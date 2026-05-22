#include "tinynn_ggml.h"
#include "tinynn_trace.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CUDA backend init lives in tinynn_backend_cuda.c (only present when
 * linking against libtinynn_ggml_cuda.a). Weak DEFINITION here returns
 * NULL — strong override in the CUDA archive provides the real impl.
 * Lets a single tinynn_ggml.o serve both CPU-only and CUDA programs
 * without symbol duplication, on both clang and gcc.
 */
__attribute__((weak)) ggml_backend_t tnn_backend_cuda_init_internal(void) {
    return NULL;
}

/* Weak hook: returns a CUDA-side ggml_backend_cuda_buffer_from_ptr
 * wrapping the given host region (typically an mmap'd GGUF). The
 * CPU-only build leaves this NULL; the CUDA archive
 * (tinynn_backend_cuda.c) overrides with a strong definition that
 * calls into the patched ggml-cuda. */
__attribute__((weak)) ggml_backend_buffer_t
tnn_cuda_buffer_from_ptr_internal(void *host_ptr, size_t size, int device) {
    (void)host_ptr; (void)size; (void)device;
    return NULL;
}

#define TNN_SCRATCH_BYTES (16 * 1024 * 1024)   /* 16 MiB: 4M f32 */

/* Engine: persistent across the program's lifetime. Holds the backend
 * objects + scheduler. Cached per (prefer_cuda) flavor so multiple
 * session_new calls share one backend init. */
typedef struct {
    ggml_backend_t       backend;        /* CUDA or CPU */
    ggml_backend_t       cpu_backend;    /* sched fallback when primary is CUDA */
    ggml_backend_sched_t sched;
    const char          *backend_name;
} tnn_engine;

static tnn_engine *g_engine_cpu  = NULL;
static tnn_engine *g_engine_cuda = NULL;

static tnn_engine *tnn_engine_get(int prefer_cuda)
{
    tnn_engine **slot = prefer_cuda ? &g_engine_cuda : &g_engine_cpu;
    if (*slot) return *slot;

    ggml_backend_load_all();
    tnn_engine *e = (tnn_engine *)calloc(1, sizeof(tnn_engine));
    if (!e) return NULL;

    if (prefer_cuda) {
        e->backend = tnn_backend_cuda_init_internal();
        if (e->backend) e->backend_name = "cuda";
    }
    if (!e->backend) {
        e->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);
        e->backend_name = "cpu";
    }
    if (!e->backend) { free(e); return NULL; }

    e->cpu_backend = (e->backend_name[0] == 'c' && e->backend_name[1] == 'p')
        ? NULL
        : ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);

    ggml_backend_t backends[2];
    int n_backends = 0;
    backends[n_backends++] = e->backend;
    if (e->cpu_backend) backends[n_backends++] = e->cpu_backend;
    /* Scheduler graph-size hint. Must be >= n_nodes + n_leafs of the
     * largest graph we'll alloc. 65536 covers seq-mode training of
     * Qwen2.5-3B at T<=32 with LoRA + AdamW + pin_all_graph_b_nodes
     * (~30K backward nodes once every grad-chain intermediate is
     * pinned-as-output for the ggml-cpu sched-alias workaround).
     * Older path (KV-cache decode, T=1) used 16384; we leave headroom. */
    e->sched = ggml_backend_sched_new(backends, NULL, n_backends,
                                       65536, false, true);

    *slot = e;
    return e;
}

/* Session: per "compute frame" — owns its ctx + graph + scratch, but
 * references a cached engine. tnn_session_free frees the per-frame
 * resources only; the engine persists for reuse.
 *
 * Two contexts:
 *  - ctx_w (weights_ctx): persistent tensors (parameters, moments).
 *    Allocated once via ggml_backend_alloc_ctx_tensors into a stable
 *    backend buffer that survives sched_reset cycles.
 *  - ctx (compute_ctx): per-step tensors (inputs, intermediates).
 *    Managed by backend_sched, re-allocated per compute cycle.
 *
 * Cross-ctx tensors in a single graph are supported by ggml — nodes
 * just hold tensor pointers. The compute graph references both ctxs;
 * sched skips persistent tensors (they already have a buffer). */
typedef struct {
    tnn_engine             *engine;       /* unowned */
    struct ggml_context    *ctx;          /* compute (no_alloc=true) */
    struct ggml_context    *ctx_w;        /* weights  (no_alloc=true until finalized) */
    struct ggml_context    *ctx_w_mmap;   /* mmap'd weights (no_alloc=true forever;
                                           * tensors get data via
                                           * ggml_backend_tensor_alloc against
                                           * weight_buf_mmap) */
    struct ggml_cgraph     *graph;        /* primary (e.g. forward) */
    struct ggml_cgraph     *graph_b;      /* secondary (e.g. adam_step) */
    uint8_t                *ctx_buf;
    size_t                  ctx_buf_size;
    uint8_t                *ctx_w_buf;
    size_t                  ctx_w_buf_size;
    uint8_t                *ctx_w_mmap_buf;
    size_t                  ctx_w_mmap_buf_size;
    ggml_backend_buffer_t   weights_buf;       /* set by tnn_finalize_weights */
    ggml_backend_buffer_t   weights_buf_mmap;  /* cpu_buffer_from_ptr wrapping
                                                * a caller-owned mmap region. We
                                                * free the buffer; we do NOT free
                                                * the underlying memory. */
    void                   *weights_map_base;  /* mmap base, caller-owned */
    size_t                  weights_map_size;
    float                  *scratch;
    int                     realized;
    int                     realized_b;
    int                     weights_finalized;
    int                     last_graph;            /* 0 = none, 1 = a, 2 = b */
    int                     scratch_overflow_warned; /* once-per-session diag */
} tnn_session;

void *tnn_session_new(int prefer_cuda)
{
    tnn_engine *e = tnn_engine_get(prefer_cuda);
    if (!e) return NULL;

    tnn_session *s = (tnn_session *)calloc(1, sizeof(tnn_session));
    if (!s) return NULL;
    s->engine = e;

    /* Reset the (shared) scheduler so any prior allocation state is
     * wiped before this session builds its graph. */
    ggml_backend_sched_reset(e->sched);

    /* Two cgraphs share ctx, so reserve room for both. ctx grows
     * monotonically across tnn_reset_for_rebuild cycles (each rebuild
     * allocates new compute-tensor metadata in the same ctx). At
     * GPT-2-distil shape one decode-step graph has ~1280 ops:
     *   6 layers × (12 heads × ~16 ops + concat/proj/FFN/LN/residual)
     * × N rebuilds = 1280 × N tensor headers (~376 B each).
     * Reserve enough headroom for ~10k rebuilds = ~5M tensor headers.
     * The no_alloc=true ctx only holds metadata so this is cheap
     * bytes-wise. */
    s->ctx_buf_size = ggml_tensor_overhead() * 262144
                      + ggml_graph_overhead_custom(GGML_DEFAULT_GRAPH_SIZE, false) * 4
                      + 32 * 1024 * 1024;
    s->ctx_buf = (uint8_t *)calloc(1, s->ctx_buf_size);
    struct ggml_init_params params = {
        /*.mem_size   =*/ s->ctx_buf_size,
        /*.mem_buffer =*/ s->ctx_buf,
        /*.no_alloc   =*/ true,
    };
    s->ctx = ggml_init(params);
    /* Graph node-count budget. Default GGML_DEFAULT_GRAPH_SIZE=2048
     * is enough for distilgpt2 (6 layers, ~1200 nodes/step) but not
     * for gpt2-small (12 layers, ~2500) and larger. 65536 covers
     * seq-mode training (Qwen2.5-3B, T<=32, LoRA + AdamW + pinned
     * graph_b) — matches the engine sched hash-set size. Cost is one
     * int slot per node header. */
    s->graph   = ggml_new_graph_custom(s->ctx, 65536, false);
    s->graph_b = ggml_new_graph_custom(s->ctx, 65536, false);

    /* Weights ctx pool. Sized for ~1024 weight tensors -- generous
     * upper bound that covers FullForwardFFICache at LLM scale
     * (per layer: 2 norms + 3*n_heads + 3 = up to ~50 tensors; for
     * 16 layers that's 800; plus global). no_alloc=true so this is
     * just metadata bytes. */
    /* Persistent-weights ctx. One slot per tensor declared via
     * tnn_input_*_f32_persistent. GPT-2 sizes:
     *   distilgpt2  6 layers  ~  636 tensors
     *   gpt2-small 12 layers  ~ 1272 tensors
     *   gpt2-large 36 layers  ~ 7560 tensors
     *   gpt2-xl    48 layers  ~10080 tensors  (KV cache per head adds)
     * 16384 covers up to gpt2-xl comfortably; the no_alloc ctx only
     * holds metadata so the extra bytes cost nothing on small models. */
    s->ctx_w_buf_size = ggml_tensor_overhead() * 16384;
    s->ctx_w_buf = (uint8_t *)calloc(1, s->ctx_w_buf_size);
    struct ggml_init_params w_params = {
        /*.mem_size   =*/ s->ctx_w_buf_size,
        /*.mem_buffer =*/ s->ctx_w_buf,
        /*.no_alloc   =*/ true,
    };
    s->ctx_w = ggml_init(w_params);

    /* ctx_w_mmap is created LAZILY (on first tnn_session_attach_weight_mmap
     * call) rather than at session_new. Eager creation has a CUDA
     * regression: even an empty no_alloc ggml_context with no
     * attached backend buffer causes ggml-cuda's scheduler to
     * produce wrong matmul output for downstream ops on the SAME
     * session (verified 2026-05-18 — CUDA inference goes from
     * wrong (top=112919) to correct (top=71 matching CPU) when
     * this ctx is absent). Lazy creation keeps the BYO-pointer
     * path working when needed without poisoning sessions that
     * don't use it. */
    s->ctx_w_mmap_buf_size = 0;
    s->ctx_w_mmap_buf      = NULL;
    s->ctx_w_mmap          = NULL;

    s->scratch = (float *)calloc(1, TNN_SCRATCH_BYTES);
    s->realized          = 0;
    s->realized_b        = 0;
    s->weights_finalized = 0;
    s->weights_buf       = NULL;
    s->weights_buf_mmap  = NULL;
    s->weights_map_base  = NULL;
    s->weights_map_size  = 0;
    s->last_graph        = 0;
    return (void *)s;
}

void tnn_session_free(void *sess)
{
    if (!sess) return;
    tnn_session *s = (tnn_session *)sess;
    if (s->weights_buf)      ggml_backend_buffer_free(s->weights_buf);
    if (s->weights_buf_mmap) ggml_backend_buffer_free(s->weights_buf_mmap);
    if (s->ctx)        ggml_free(s->ctx);
    if (s->ctx_w)      ggml_free(s->ctx_w);
    if (s->ctx_w_mmap) ggml_free(s->ctx_w_mmap);
    free(s->ctx_buf);
    free(s->ctx_w_buf);
    free(s->ctx_w_mmap_buf);
    free(s->scratch);
    free(s);
    /* Engine + sched are cached globally; do not free here. */
}

const char *tnn_backend_name(void *sess)
{
    if (!sess) return "(null)";
    return ((tnn_session *)sess)->engine->backend_name;
}

int tnn_link_check(void) { return 73; }

void *tnn_input_2d_f32(void *sess, int rows, int cols)
{
    if (!sess || rows <= 0 || cols <= 0) return NULL;
    tnn_session *s = (tnn_session *)sess;
    (void)s;   /* future: validate ctx hasn't been realized */
    return (void *)ggml_new_tensor_2d(((tnn_session *)sess)->ctx, GGML_TYPE_F32,
                                       (int64_t)cols, (int64_t)rows);
}

/* Create a PERSISTENT 2D F32 tensor in ctx_w. Its backend buffer is
 * allocated by tnn_finalize_weights (call once after all persistent
 * tensors are declared) and retained across sched_reset cycles, so
 * uploaded data survives multiple compute calls without re-upload. */
void *tnn_input_2d_f32_persistent(void *sess, int rows, int cols)
{
    if (!sess || rows <= 0 || cols <= 0) return NULL;
    tnn_session *s = (tnn_session *)sess;
    if (s->weights_finalized) return NULL;
    return (void *)ggml_new_tensor_2d(s->ctx_w, GGML_TYPE_F32,
                                       (int64_t)cols, (int64_t)rows);
}

/* Same shape as tnn_input_2d_f32_persistent but with a caller-chosen
 * ggml type (e.g. GGML_TYPE_Q8_0 for Q8-stays-Q8 inference). For
 * block-quantized types the column count (ne0) must be a multiple of
 * the block size — GGML_BLCK_SIZE handles this. Returns NULL on bad
 * shape; callers should sanity-check the result. */
void *tnn_input_2d_persistent_typed(void *sess, int rows, int cols, int ggml_type)
{
    if (!sess || rows <= 0 || cols <= 0) return NULL;
    tnn_session *s = (tnn_session *)sess;
    if (s->weights_finalized) return NULL;
    enum ggml_type t = (enum ggml_type)ggml_type;
    int blck = ggml_blck_size(t);
    if (blck > 1 && (cols % blck != 0)) return NULL;
    return (void *)ggml_new_tensor_2d(s->ctx_w, t,
                                       (int64_t)cols, (int64_t)rows);
}

/* Phase 2 BYO-pointer: register an mmap'd region as the backing
 * buffer for weight tensors created via tnn_input_*_persistent_mmap.
 * The session does NOT own the underlying memory — the caller (e.g.
 * a tnn_gguf_session) must keep `base` valid for the session's
 * lifetime. Returns 0 on success, -1 on already-attached / bad args. */
/* Lazy-create ctx_w_mmap. Called from tnn_session_attach_weight_mmap
 * (Phase 2 entry point). NOT called from tnn_session_new — see the
 * note there for why (eager creation breaks CUDA inference). */
static int ensure_ctx_w_mmap(tnn_session *s)
{
    if (s->ctx_w_mmap) return 0;
    s->ctx_w_mmap_buf_size = ggml_tensor_overhead() * 16384;
    s->ctx_w_mmap_buf = (uint8_t *)calloc(1, s->ctx_w_mmap_buf_size);
    if (!s->ctx_w_mmap_buf) return -1;
    struct ggml_init_params m_params = {
        /*.mem_size   =*/ s->ctx_w_mmap_buf_size,
        /*.mem_buffer =*/ s->ctx_w_mmap_buf,
        /*.no_alloc   =*/ true,
    };
    s->ctx_w_mmap = ggml_init(m_params);
    if (!s->ctx_w_mmap) {
        free(s->ctx_w_mmap_buf);
        s->ctx_w_mmap_buf = NULL;
        s->ctx_w_mmap_buf_size = 0;
        return -1;
    }
    return 0;
}

int tnn_session_attach_weight_mmap(void *sess, void *base, size_t size)
{
    if (!sess || !base || size == 0) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (s->weights_buf_mmap) return -1;  /* already attached */
    if (ensure_ctx_w_mmap(s) != 0) return -1;
    /* The buffer_from_ptr APIs assert ptr % TENSOR_ALIGNMENT == 0.
     * mmap returns page-aligned pointers (>= 4 KiB), so a GGUF mmap
     * always satisfies this.
     *
     * CUDA sessions get the patched ggml_backend_cuda_buffer_from_ptr
     * (vendored in this repo; see docs/cuda-byo-pointer-design.md).
     * The host region is cudaHostRegister'd and made device-addressable
     * via UVA; on GB10 unified memory the device pointer equals the
     * host pointer and kernels read the mmap'd file pages directly. */
    int is_cuda = (s->engine && s->engine->backend_name &&
                    s->engine->backend_name[0] == 'c' &&
                    s->engine->backend_name[1] == 'u');
    if (is_cuda) {
        s->weights_buf_mmap = tnn_cuda_buffer_from_ptr_internal(base, size, 0);
        if (!s->weights_buf_mmap) return -2;  /* CUDA archive not linked / GPU error */
    } else {
        s->weights_buf_mmap = ggml_backend_cpu_buffer_from_ptr(base, size);
        if (!s->weights_buf_mmap) return -1;
    }
    /* Store the buffer's "view" of the base, NOT the raw host pointer.
     * On CPU these are the same; on CUDA the buffer's base is the
     * UVA-mapped device pointer (equal to host_ptr on unified-memory
     * SKUs, different on discrete GPUs). Tensor data pointers are
     * computed as weights_map_base + offset; using the buffer's base
     * keeps ggml_backend_tensor_alloc's range-check happy. */
    s->weights_map_base = ggml_backend_buffer_get_base(s->weights_buf_mmap);
    s->weights_map_size = size;
    return 0;
}

/* Allocate a 2D persistent tensor in ctx_w_mmap whose `data` points
 * at `base + buf_offset` in the attached mmap region. The tensor's
 * `buffer` is set so the scheduler treats it as already-resident.
 * Returns NULL on bad args or out-of-range offset.
 *
 * For block-quantized types, `cols` (ne0) must be a multiple of the
 * type's block size and `buf_offset` must land on a 32-byte boundary
 * (GGUF guarantees this).
 *
 * Caller computes buf_offset as
 *   gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, idx).
 */
void *tnn_input_2d_persistent_mmap(void *sess, int rows, int cols,
                                    int ggml_type, size_t buf_offset)
{
    if (!sess || rows <= 0 || cols <= 0) return NULL;
    tnn_session *s = (tnn_session *)sess;
    if (!s->weights_buf_mmap || !s->weights_map_base) return NULL;
    enum ggml_type t = (enum ggml_type)ggml_type;
    int blck = ggml_blck_size(t);
    if (blck > 1 && (cols % blck != 0)) return NULL;
    if (buf_offset >= s->weights_map_size) return NULL;

    struct ggml_tensor *tensor = ggml_new_tensor_2d(s->ctx_w_mmap, t,
                                                    (int64_t)cols,
                                                    (int64_t)rows);
    if (!tensor) return NULL;

    void *addr = (char *)s->weights_map_base + buf_offset;
    enum ggml_status st = ggml_backend_tensor_alloc(s->weights_buf_mmap,
                                                     tensor, addr);
    if (st != GGML_STATUS_SUCCESS) return NULL;
    return (void *)tensor;
}

/* 1D variant for norms / biases — same semantics. */
void *tnn_input_1d_persistent_mmap(void *sess, int n, int ggml_type,
                                    size_t buf_offset)
{
    if (!sess || n <= 0) return NULL;
    tnn_session *s = (tnn_session *)sess;
    if (!s->weights_buf_mmap || !s->weights_map_base) return NULL;
    enum ggml_type t = (enum ggml_type)ggml_type;
    if (buf_offset >= s->weights_map_size) return NULL;

    struct ggml_tensor *tensor = ggml_new_tensor_1d(s->ctx_w_mmap, t,
                                                    (int64_t)n);
    if (!tensor) return NULL;

    void *addr = (char *)s->weights_map_base + buf_offset;
    enum ggml_status st = ggml_backend_tensor_alloc(s->weights_buf_mmap,
                                                     tensor, addr);
    if (st != GGML_STATUS_SUCCESS) return NULL;
    return (void *)tensor;
}

/* Same as above but 1D — used for the 7-elem adamw_params vector. */
void *tnn_input_1d_f32_persistent(void *sess, int n)
{
    if (!sess || n <= 0) return NULL;
    tnn_session *s = (tnn_session *)sess;
    if (s->weights_finalized) return NULL;
    return (void *)ggml_new_tensor_1d(s->ctx_w, GGML_TYPE_F32, (int64_t)n);
}

/* Allocate the backend buffer for all persistent tensors in ctx_w.
 * Must be called AFTER declaring all persistent tensors and BEFORE
 * any tnn_realize/compute. After this, the persistent tensors have
 * stable backend storage independent of sched.
 *
 * Returns 0 on success, negative on failure. */
int tnn_finalize_weights(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (s->weights_finalized) return -2;
    s->weights_buf = ggml_backend_alloc_ctx_tensors(s->ctx_w, s->engine->backend);
    if (!s->weights_buf) return -3;
    s->weights_finalized = 1;
    return 0;
}

/* Zero an entire persistent tensor via backend memset_tensor. Faster
 * than building a Mat-of-zeros + upload_row_major when the tensor is
 * big (e.g. Adam state for vocab×d_model embeddings: ~1 GB of zeros).
 * Works on both CPU (memset) and CUDA (cudaMemsetAsync). */
int tnn_zero_tensor(void *sess, void *tensor)
{
    if (!sess || !tensor) return -1;
    tnn_session *s = (tnn_session *)sess;
    (void)s;
    struct ggml_tensor *t = (struct ggml_tensor *)tensor;
    ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
    return 0;
}

void *tnn_matmul(void *sess, void *a, void *b)
{
    if (!sess || !a || !b) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_mul_mat(s->ctx,
                                 (struct ggml_tensor *)a,
                                 (struct ggml_tensor *)b);
}

void *tnn_matmul_axb(void *sess, void *a, void *b)
{
    /* Compute A · B (no transpose at the caller). ggml_mul_mat does
     * A · B^T natively, so we transpose B first.  ggml_transpose is a
     * stride-permutation view; ggml_cont materializes it as contiguous
     * so mul_mat's contiguity-required input is satisfied. */
    if (!sess || !a || !b) return NULL;
    tnn_session *s = (tnn_session *)sess;
    struct ggml_tensor *bT = ggml_cont(s->ctx, ggml_transpose(s->ctx, (struct ggml_tensor *)b));
    return (void *)ggml_mul_mat(s->ctx, (struct ggml_tensor *)a, bT);
}

void *tnn_add(void *sess, void *a, void *b)
{
    if (!sess || !a || !b) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_add(s->ctx,
                             (struct ggml_tensor *)a,
                             (struct ggml_tensor *)b);
}

void *tnn_gelu(void *sess, void *a)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    /* ggml_gelu uses the tanh approximation:
     *   0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
     * which matches the project's feed_forward GeLU exactly. */
    return (void *)ggml_gelu(s->ctx, (struct ggml_tensor *)a);
}

void *tnn_rms_norm(void *sess, void *x, void *gamma_row, double eps)
{
    if (!sess || !x || !gamma_row) return NULL;
    tnn_session *s = (tnn_session *)sess;
    /* ggml_rms_norm normalizes along ne[0] (the feature dim). The result
     * is the unscaled normalized tensor; we then multiply by gamma_row
     * (shape 1 x feature) which ggml_mul broadcasts over the leading dim. */
    struct ggml_tensor *normed = ggml_rms_norm(s->ctx,
                                                (struct ggml_tensor *)x,
                                                (float)eps);
    return (void *)ggml_mul(s->ctx, normed, (struct ggml_tensor *)gamma_row);
}

/* LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta. ggml_norm
 * computes the normalized (x - mean)/sqrt(var+eps) part; we then
 * multiply by gamma and add beta. Used for HF-style models (GPT-2 /
 * GPT-Neo / TinyStories) that use LayerNorm rather than RMSNorm. */
void *tnn_layer_norm(void *sess, void *x, void *gamma_row, void *beta_row, double eps)
{
    if (!sess || !x || !gamma_row || !beta_row) return NULL;
    tnn_session *s = (tnn_session *)sess;
    struct ggml_tensor *normed = ggml_norm(s->ctx,
                                             (struct ggml_tensor *)x,
                                             (float)eps);
    struct ggml_tensor *scaled = ggml_mul(s->ctx, normed,
                                            (struct ggml_tensor *)gamma_row);
    return (void *)ggml_add(s->ctx, scaled,
                              (struct ggml_tensor *)beta_row);
}

/* Write `b` into `a` at byte offset, with row stride nb1. Result has
 * `a`'s shape (unlike ggml_cpy which returns the small dst view) so
 * downstream ops can read the modified `a` directly. Used for V[:, pos]
 * column writes in KV cache (V layout = [max_T, d_head], offset =
 * pos * 4, nb1 = max_T * 4). */
void *tnn_set_2d(void *sess, void *a, void *b, long nb1, long offset)
{
    if (!sess || !a || !b) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_set_2d(s->ctx,
                                 (struct ggml_tensor *)a,
                                 (struct ggml_tensor *)b,
                                 (size_t)nb1,
                                 (size_t)offset);
}

/* Write `b`'s rows into `a` at row indices `idx`. For our KV cache:
 *   a   = persistent K (ne=[d_head, max_T])
 *   b   = compute k_new (ne=[d_head, 1])
 *   idx = compute (1,) int32 holding the current decode position
 * The new k row lands at K[idx[0]] (other rows untouched). Same shape
 * pattern for V. Position is a RUNTIME tensor — the graph stays
 * static across decode steps, so we don't need to rebuild it. */
void *tnn_set_rows(void *sess, void *a, void *b, void *idx)
{
    if (!sess || !a || !b || !idx) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_set_rows(s->ctx,
                                   (struct ggml_tensor *)a,
                                   (struct ggml_tensor *)b,
                                   (struct ggml_tensor *)idx);
}

/* Softmax-with-mask. Adds `mask` to `a`, scales by `scale`, then runs
 * softmax along ne[0]. For KV-cache attention: scores shape (max_T, 1),
 * mask shape (max_T, 1), result shape (max_T, 1). The mask is uploaded
 * per step with 0.0 for positions <= pos and -inf for positions > pos
 * so the softmax zeroes out future-key attention even though K's
 * future-position slots may hold stale or uninitialised values. */
void *tnn_soft_max_ext(void *sess, void *a, void *mask, double scale, double max_bias)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_soft_max_ext(s->ctx,
                                       (struct ggml_tensor *)a,
                                       (struct ggml_tensor *)mask,
                                       (float)scale,
                                       (float)max_bias);
}

/* Returns a NULL pointer typed as :ptr. Useful as an Array<:ptr> seed
 * value so Spinel infers the array as a PtrArray rather than typing
 * it from a `[nil]` literal (which can resolve to IntArray). */
void *tnn_null_ptr(void)
{
    return NULL;
}

/* 1-D view of a tensor at byte `offset`, of length `ne0`. Used to
 * slice a single row out of a (max_T, d_head) KV buffer at a runtime
 * position computed by the caller (offset = pos * d_head * 4). */
void *tnn_view_1d(void *sess, void *a, int ne0, long offset)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_view_1d(s->ctx, (struct ggml_tensor *)a,
                                  (int64_t)ne0, (size_t)offset);
}

/* 2-D view of a tensor: rows of length ne0 stride nb1, ne1 rows
 * total, starting at byte `offset`. Used for slicing K/V[0:pos+1] in
 * attention. nb1 = d_head * 4 for our row-of-floats KV layout. */
void *tnn_view_2d(void *sess, void *a, int ne0, int ne1, long nb1, long offset)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_view_2d(s->ctx, (struct ggml_tensor *)a,
                                  (int64_t)ne0, (int64_t)ne1,
                                  (size_t)nb1, (size_t)offset);
}

/* Reshape a contiguous tensor to (ne0, ne1, ne2). The total element
 * count must match. Used by the sequence-mode forward (M3) to lift
 * Q/K from ne=[d_head, T] to ne=[d_head, 1, T] before ggml_rope_ext —
 * rope_ext asserts a->ne[2] == positions->ne[0]. */
void *tnn_reshape_3d(void *sess, void *a, int ne0, int ne1, int ne2)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_reshape_3d(s->ctx, (struct ggml_tensor *)a,
                                     (int64_t)ne0, (int64_t)ne1, (int64_t)ne2);
}

/* Reshape a contiguous tensor back to (ne0, ne1). After rope_ext on
 * a [d_head, 1, T] tensor, downstream matmul wants [d_head, T] again. */
void *tnn_reshape_2d(void *sess, void *a, int ne0, int ne1)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_reshape_2d(s->ctx, (struct ggml_tensor *)a,
                                     (int64_t)ne0, (int64_t)ne1);
}

/* Copy a -> b. Used to write k_new into a view of the persistent K
 * buffer (b = view_2d(K, d_head, 1, ..., offset=pos*d_head*4)). */
void *tnn_cpy(void *sess, void *a, void *b)
{
    if (!sess || !a || !b) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_cpy(s->ctx, (struct ggml_tensor *)a,
                              (struct ggml_tensor *)b);
}

/* Concatenate `a` and `b` along the given dim (0 = ne[0], 1 = ne[1]).
 * Other dims must match. Used to glue per-head attention outputs into
 * a single (d_model, T) tensor by stacking d_head slices along ne0. */
void *tnn_concat(void *sess, void *a, void *b, int dim)
{
    if (!sess || !a || !b) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_concat(s->ctx,
                                (struct ggml_tensor *)a,
                                (struct ggml_tensor *)b,
                                dim);
}

/* Causal mask: sets elements ABOVE the diagonal (i.e. positions where
 * key_idx > query_idx + n_past) to -inf, so subsequent softmax zeroes
 * them. n_past = 0 gives the standard causal mask for training. For
 * KV-cache inference, n_past = current position so attention can see
 * cached keys plus the current token but not future tokens. */
void *tnn_diag_mask_inf(void *sess, void *a, int n_past)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_diag_mask_inf(s->ctx, (struct ggml_tensor *)a, n_past);
}

/* --- Llama-family ops -------------------------------------------------- */

/* SiLU activation: silu(x) = x * sigmoid(x). Used in SwiGLU FFNs
 * (Llama / SmolLM2 / Qwen / Phi). */
void *tnn_silu(void *sess, void *a)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_silu(s->ctx, (struct ggml_tensor *)a);
}

/* Elementwise multiply c = a * b. Used to combine the gate and up
 * projections of SwiGLU before the down projection. */
void *tnn_mul(void *sess, void *a, void *b)
{
    if (!sess || !a || !b) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_mul(s->ctx,
                             (struct ggml_tensor *)a,
                             (struct ggml_tensor *)b);
}

/* Rotary Position Embedding (rotate_half / NEOX mode), as used by
 * Llama / SmolLM2 / Qwen2 / Mistral. Applied to Q and K before the
 * dot product.
 *
 *   a:        input tensor, shape [Dh, T, ...]   (one head's worth)
 *   pos:      int32 tensor of length T, absolute positions per token
 *   n_dims:   number of dimensions to rotate (= Dh for full rotary,
 *             smaller for partial — Pythia uses Dh/4)
 *   freq_base: theta base. 10000 (Llama-1/2), 100000 (SmolLM2),
 *              1000000 (Qwen2 long-context)
 *
 * Pass freq_scale=1.0, ext_factor=0.0, attn_factor=1.0, beta_fast=32.0,
 * beta_slow=1.0, freq_factors=NULL for the no-scaling (vanilla GPT-2 /
 * SmolLM2 / Qwen2-short-context) default. YaRN tunes the scalars;
 * llama3 + LongRoPE supply freq_factors via tnn_rope_freq_factors_*. */
void *tnn_rope_ext(void *sess, void *a, void *pos, int n_dims,
                   double freq_base, double freq_scale,
                   double ext_factor, double attn_factor,
                   double beta_fast, double beta_slow,
                   void *freq_factors)
{
    if (!sess || !a || !pos) return NULL;
    tnn_session *s = (tnn_session *)sess;
    const int mode = 2;   /* GGML_ROPE_TYPE_NEOX — matches HF llama rotate_half */
    /* n_ctx_orig is only consulted when ext_factor != 0 (YaRN). Pass
     * 0 when no YaRN is in play; callers using YaRN encode orig_ctx
     * via the freq_factors path or pass it via attn_factor scaling. */
    const int n_ctx_orig = 0;
    return (void *)ggml_rope_ext(s->ctx,
                                  (struct ggml_tensor *)a,
                                  (struct ggml_tensor *)pos,
                                  (struct ggml_tensor *)freq_factors,
                                  n_dims,
                                  mode,
                                  n_ctx_orig,
                                  (float)freq_base,
                                  (float)freq_scale,
                                  (float)ext_factor,
                                  (float)attn_factor,
                                  (float)beta_fast,
                                  (float)beta_slow);
}

/* Allocate a persistent (n_dims/2)-element F32 tensor in ctx_w to hold
 * RoPE freq_factors for llama3-style or LongRoPE scaling. Must be
 * called BEFORE tnn_finalize_weights, like any other persistent.
 *
 * The values are computed by the Ruby side (see
 * Toy::RopeScaling.compute_llama3_freq_factors) and uploaded via the
 * standard tnn_upload_from_float_array path after finalize. Doing the
 * math in Ruby (i) keeps the C wrapper simple, (ii) avoids the
 * "write to t->data with no_alloc=true" trap, and (iii) makes the
 * scaling formula trivially testable from MRI without recompiling. */
void *tnn_rope_freq_factors_alloc(void *sess, int n_dims)
{
    if (!sess || n_dims <= 0) return NULL;
    tnn_session *s = (tnn_session *)sess;
    if (s->weights_finalized) return NULL;
    return (void *)ggml_new_tensor_1d(s->ctx_w, GGML_TYPE_F32,
                                      (int64_t)(n_dims / 2));
}

/* Allocate a 1-D int32 tensor in the *session* context. Used to hold
 * RoPE position indices. The caller fills it via tnn_scratch_set_i32 +
 * tnn_upload_int_array (or fills directly during graph build). */
void *tnn_input_1d_i32_ctx(void *sess, int n)
{
    if (!sess) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_new_tensor_1d(s->ctx, GGML_TYPE_I32, n);
}

void *tnn_softmax(void *sess, void *a)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    /* ggml_soft_max normalizes along ne[0]. With our convention
     * (ne0=cols, ne1=rows) this is per-row softmax, matching the
     * project's softmax_rows!. */
    return (void *)ggml_soft_max(s->ctx, (struct ggml_tensor *)a);
}

void *tnn_transpose(void *sess, void *a)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    /* ggml_transpose is a stride-permutation view (no data movement).
     * Wrap in ggml_cont so the result is contiguous f32 and downloadable. */
    return (void *)ggml_cont(s->ctx,
                              ggml_transpose(s->ctx, (struct ggml_tensor *)a));
}

void *tnn_scale(void *sess, void *a, double scale)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_scale(s->ctx, (struct ggml_tensor *)a, (float)scale);
}

void *tnn_rms_norm_back(void *sess, void *x, void *dy, double eps)
{
    if (!sess || !x || !dy) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_rms_norm_back(s->ctx,
                                       (struct ggml_tensor *)x,
                                       (struct ggml_tensor *)dy,
                                       (float)eps);
}

void *tnn_softmax_back(void *sess, void *a, void *dy)
{
    if (!sess || !a || !dy) return NULL;
    tnn_session *s = (tnn_session *)sess;
    /* Plain softmax backward: scale=1.0, max_bias=0.0 (no ALiBi). */
    return (void *)ggml_soft_max_ext_back(s->ctx,
                                           (struct ggml_tensor *)a,
                                           (struct ggml_tensor *)dy,
                                           1.0f, 0.0f);
}

/* Backward for SiLU activation. SiLU(x) = x * sigmoid(x);
 * dSiLU/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
 * Given x and dy (gradient from upstream), returns dx.
 *
 * NOTE: ggml_silu_back's public header comment swaps the args
 * ("a - x, b - dy"). Reading the actual CPU op, src[0]=dy and
 * src[1]=x. We pass (dy, x) to match the implementation. */
void *tnn_silu_back(void *sess, void *x, void *dy)
{
    if (!sess || !x || !dy) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_silu_back(s->ctx,
                                   (struct ggml_tensor *)dy,
                                   (struct ggml_tensor *)x);
}

/* Backward for RoPE-NEOX. Same arg convention as tnn_rope_ext but
 * also takes dy (gradient of the rope_ext output). Returns dx.
 * Callers must pass the same YaRN/scaling args used in the forward;
 * mismatch silently corrupts gradients. */
void *tnn_rope_ext_back(void *sess, void *dy, void *pos, int n_dims,
                        double freq_base, double freq_scale,
                        double ext_factor, double attn_factor,
                        double beta_fast, double beta_slow,
                        void *freq_factors)
{
    if (!sess || !dy || !pos) return NULL;
    tnn_session *s = (tnn_session *)sess;
    const int mode = 2;   /* GGML_ROPE_TYPE_NEOX */
    const int n_ctx_orig = 0;
    return (void *)ggml_rope_ext_back(s->ctx,
                                       (struct ggml_tensor *)dy,
                                       (struct ggml_tensor *)pos,
                                       (struct ggml_tensor *)freq_factors,
                                       n_dims,
                                       mode,
                                       n_ctx_orig,
                                       (float)freq_base,
                                       (float)freq_scale,
                                       (float)ext_factor,
                                       (float)attn_factor,
                                       (float)beta_fast,
                                       (float)beta_slow);
}

void *tnn_get_rows(void *sess, void *table, void *idx)
{
    if (!sess || !table || !idx) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_get_rows(s->ctx,
                                  (struct ggml_tensor *)table,
                                  (struct ggml_tensor *)idx);
}

void *tnn_get_rows_back(void *sess, void *d_out, void *idx, void *table_shape)
{
    if (!sess || !d_out || !idx || !table_shape) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_get_rows_back(s->ctx,
                                       (struct ggml_tensor *)d_out,
                                       (struct ggml_tensor *)idx,
                                       (struct ggml_tensor *)table_shape);
}

void *tnn_input_1d_i32(void *sess, int n)
{
    if (!sess || n <= 0) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_new_tensor_1d(s->ctx, GGML_TYPE_I32, (int64_t)n);
}

void tnn_gelu_back_scratch(void *sess, int n)
{
    if (!sess || n <= 0) return;
    tnn_session *s = (tnn_session *)sess;
    int max_slots = TNN_SCRATCH_BYTES / (int)sizeof(float);
    if (3 * n > max_slots) return;     /* not enough scratch */

    const float *x  = s->scratch + 0;
    const float *dh = s->scratch + n;
    float       *dx = s->scratch + 2 * n;

    const float c = 0.7978845608028654f;    /* sqrt(2/pi) */
    const float k = 0.044715f;

    for (int i = 0; i < n; ++i) {
        float xi  = x[i];
        float xi2 = xi * xi;
        float u   = c * (xi + k * xi * xi2);
        float tu  = tanhf(u);
        float sech2 = 1.0f - tu * tu;
        float dudx  = c * (1.0f + 3.0f * k * xi2);
        float dgelu = 0.5f * (1.0f + tu) + 0.5f * xi * sech2 * dudx;
        dx[i] = dh[i] * dgelu;
    }
}

void tnn_adam_step_scratch(void *sess, int n,
                            double lr, double b1, double b2, double eps,
                            double omc1, double omc2)
{
    if (!sess || n <= 0) return;
    tnn_session *s = (tnn_session *)sess;
    int max_slots = TNN_SCRATCH_BYTES / (int)sizeof(float);
    if (4 * n > max_slots) return;

    float *p = s->scratch + 0;
    const float *g = s->scratch + n;
    float *m = s->scratch + 2 * n;
    float *v = s->scratch + 3 * n;

    const float one_minus_b1 = (float)(1.0 - b1);
    const float one_minus_b2 = (float)(1.0 - b2);
    const float fb1   = (float)b1;
    const float fb2   = (float)b2;
    const float flr   = (float)lr;
    const float feps  = (float)eps;
    const float fomc1 = (float)omc1;
    const float fomc2 = (float)omc2;

    for (int i = 0; i < n; ++i) {
        float gi = g[i];
        float new_m = fb1 * m[i] + one_minus_b1 * gi;
        float new_v = fb2 * v[i] + one_minus_b2 * gi * gi;
        m[i] = new_m;
        v[i] = new_v;
        float m_hat = new_m / fomc1;
        float v_hat = new_v / fomc2;
        p[i] = p[i] - flr * m_hat / (sqrtf(v_hat) + feps);
    }
}

void tnn_set_output(void *tensor)
{
    if (!tensor) return;
    ggml_set_output((struct ggml_tensor *)tensor);
}

/* Sum all elements → scalar. Used to build a loss from a vector
 * output (e.g. sum(y * y) for an L2 squared loss). */
void *tnn_sum(void *sess, void *a)
{
    if (!sess || !a) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_sum(s->ctx, (struct ggml_tensor *)a);
}

/* Cross-entropy loss against a probability-distribution label tensor.
 * Wraps ggml_cross_entropy_loss: returns a scalar. The label tensor
 * has the same shape as the logits and should be a probability dist
 * (one-hot for hard targets, label-smoothed for soft). Output is the
 * mean negative log-likelihood across the columns of a (a column =
 * one example). Used for F1.2 SmolLM2 LoRA fine-tuning. */
void *tnn_cross_entropy_loss(void *sess, void *a, void *b)
{
    if (!sess || !a || !b) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_cross_entropy_loss(s->ctx,
                                            (struct ggml_tensor *)a,
                                            (struct ggml_tensor *)b);
}

void tnn_set_param(void *tensor)
{
    if (!tensor) return;
    ggml_set_param((struct ggml_tensor *)tensor);
}

/* Mark a tensor as the training loss. Required for autograd via
 * ggml_build_backward_expand — it asserts at least one node is marked
 * as loss and at least one as param. Typically the scalar output of
 * a sum-reduce or cross-entropy. */
void tnn_set_loss(void *tensor)
{
    if (!tensor) return;
    ggml_set_loss((struct ggml_tensor *)tensor);
}

/* Phase F0.4 autograd: after building a forward graph + marking params
 * + marking loss, call this to extend the graph with backward nodes.
 *
 * Workflow (caller side):
 *   1. tnn_input_*_persistent(...) for params, mark each with tnn_set_param
 *   2. Build forward ops (matmul, gelu, ...) ending in a scalar loss
 *   3. tnn_set_loss(loss_tensor); tnn_set_output(loss_tensor)
 *   4. tnn_realize(sess, loss_tensor)
 *   5. tnn_build_backward(sess)   ← extends s->graph_b with backward nodes
 *   6. tnn_compute_backward(sess) ← runs forward+backward
 *   7. tnn_tensor_grad(param)     ← retrieve the gradient tensor
 *
 * The backward extends s->graph_b (we keep s->graph as forward-only
 * for inference use); a freshly-duped copy of s->graph is taken with
 * grads=true so ggml_build_backward_expand has the slots it needs.
 * Returns 0 on success, -1 on failure. */
/* Split tnn_build_backward into two phases so callers can extend the
 * graph with optimizer-step nodes between build and alloc. Typical
 * in-graph-optimizer flow:
 *
 *   tnn_realize(sess, loss)            // forward graph
 *   tnn_build_backward(sess)           // dup + build_backward_expand
 *   for each param:
 *     opt_node = tnn_opt_step_adamw(sess, p, grad, m, v, hp)
 *     tnn_extend_backward_graph(sess, opt_node)
 *   tnn_realize_backward(sess)         // sched-alloc the final graph
 *   loop:
 *     tnn_compute_backward(sess)       // fwd + bwd + adam in one call
 *     read scalar loss; repeat
 */
int tnn_build_backward(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (!s->realized) return -2;   /* must build forward first */

    /* ggml_build_backward_expand requires cgraph->grads + grad_accs
     * to be non-NULL, which ggml_new_graph_custom only allocates when
     * `grads=true`. Our session's graph is created with grads=false
     * (forward-only). Solve by dup'ing with force_grads=true. The
     * duped graph SHARES tensor pointers with the original — leaves
     * and compute nodes alike. */
    s->graph_b = ggml_graph_dup(s->ctx, s->graph, /*force_grads=*/true);
    if (!s->graph_b) return -3;

    /* Expand with backward nodes for every node tagged as param. */
    ggml_build_backward_expand(s->ctx, s->graph_b, NULL);
    /* Note: NOT allocated yet — caller may extend with opt_step nodes,
     * then call tnn_realize_backward to finalize the allocation. */
    return 0;
}

/* Add a node to the backward graph (typically an opt_step output).
 * Used between tnn_build_backward and tnn_realize_backward. */
int tnn_extend_backward_graph(void *sess, void *node)
{
    if (!sess || !node) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (!s->graph_b) return -2;
    ggml_build_forward_expand(s->graph_b, (struct ggml_tensor *)node);
    return 0;
}

/* Finalize the backward graph allocation. Called once, after all
 * opt_step nodes have been added. Subsequent compute_backward calls
 * are cheap re-runs. */
int tnn_realize_backward(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (!s->graph_b) return -2;
    int64_t _t = tnn_trace_begin("realize_backward");
    ggml_backend_sched_reset(s->engine->sched);
    int ok = ggml_backend_sched_alloc_graph(s->engine->sched, s->graph_b) ? 1 : 0;
    tnn_trace_end("realize_backward", _t);
    if (!ok) return -3;
    s->realized_b = 1;
    return 0;
}

/* Initialize the backward-graph state: zero all gradient
 * accumulators + Adam moments (m, v) for any opt_step nodes; set the
 * loss tensor's incoming gradient to 1.0. Call this ONCE between
 * tnn_realize_backward and the first tnn_compute_backward. Subsequent
 * compute calls accumulate normally — momenta persist across steps. */
int tnn_graph_reset(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (!s->graph_b) return -2;
    ggml_graph_reset(s->graph_b);
    return 0;
}

/* F1.2 step 5: zero grad accumulators (and reset loss_grad to 1) but
 * leave opt_step's m / v momenta alone. Lets AdamW survive across
 * training steps without losing momentum, while still clearing the
 * grads between iterations so the next compute_backward recomputes
 * them from scratch (not accumulates).
 *
 * Mirrors ggml_graph_reset minus the GGML_OP_OPT_STEP_ADAMW arm that
 * zeros src[2] (m) and src[3] (v). For SGD this primitive and
 * tnn_graph_reset behave identically. For AdamW the difference is
 * load-bearing: graph_reset would clobber momentum every step. */
int tnn_graph_reset_grads_only(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (!s->graph_b) return -2;
    int n_nodes = ggml_graph_n_nodes(s->graph_b);
    int i = 0;
    while (i < n_nodes) {
        struct ggml_tensor * node     = ggml_graph_node(s->graph_b, i);
        struct ggml_tensor * grad_acc = ggml_graph_get_grad_acc(s->graph_b, node);
        if (grad_acc) {
            if (node->flags & GGML_TENSOR_FLAG_LOSS) {
                const float onef = 1.0f;
                if (grad_acc->buffer) {
                    ggml_backend_tensor_set(grad_acc, &onef, 0, sizeof(float));
                } else if (grad_acc->data) {
                    *((float *) grad_acc->data) = onef;
                }
            } else {
                ggml_set_zero(grad_acc);
            }
        }
        i++;
    }
    return 0;
}

/* Task #70 diagnostic — pin EVERY node in graph_b as an output, so
 * sched is forbidden from reusing any intermediate's buffer slot
 * once the node is computed. Used to test the hypothesis that the
 * CPU/CUDA training divergence is caused by sched aliasing of
 * intermediate grad tensors in long backward chains. Returns the
 * number of nodes pinned.
 *
 * Call AFTER tnn_build_backward (so the backward nodes exist) but
 * BEFORE tnn_realize_backward (so the sched sees the output flags
 * when it allocates buffers).
 *
 * This is a diagnostic primitive, NOT a recommended training path —
 * pinning every node defeats the sched's buffer-reuse optimization
 * and inflates memory by ~node-count tensors. Use only to localize
 * sched aliasing as the cause. */
int tnn_pin_all_graph_b_nodes(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (!s->graph_b) return -2;
    int n = ggml_graph_n_nodes(s->graph_b);
    int i = 0;
    while (i < n) {
        struct ggml_tensor *t = ggml_graph_node(s->graph_b, i);
        if (t) ggml_set_output(t);
        i++;
    }
    return n;
}

/* Run the backward graph (forward + backward in one compute call). */
int tnn_compute_backward(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (!s->realized_b) return -2;
    int64_t _t = tnn_trace_begin("compute_backward");
    enum ggml_status rc = ggml_backend_sched_graph_compute(s->engine->sched, s->graph_b);
    tnn_trace_end("compute_backward", _t);
    return (rc == GGML_STATUS_SUCCESS) ? 0 : (int)rc;
}

/* Return the gradient tensor for a param. Caller can then read its
 * data via tnn_download. Returns NULL if no gradient exists (param
 * wasn't marked, or backward wasn't built/computed). */
void *tnn_tensor_grad(void *sess, void *tensor)
{
    if (!sess || !tensor) return NULL;
    tnn_session *s = (tnn_session *)sess;
    if (!s->graph_b) return NULL;
    return (void *)ggml_graph_get_grad(s->graph_b, (struct ggml_tensor *)tensor);
}

void *tnn_input_1d_f32(void *sess, int n)
{
    if (!sess || n <= 0) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_new_tensor_1d(s->ctx, GGML_TYPE_F32, (int64_t)n);
}

void *tnn_opt_step_adamw(void *sess, void *a, void *grad, void *m, void *v, void *params)
{
    if (!sess || !a || !grad || !m || !v || !params) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_opt_step_adamw(s->ctx,
                                        (struct ggml_tensor *)a,
                                        (struct ggml_tensor *)grad,
                                        (struct ggml_tensor *)m,
                                        (struct ggml_tensor *)v,
                                        (struct ggml_tensor *)params);
}

/* SGD step: w = w - alpha * grad - alpha * wd * w. Simpler than Adam,
 * useful for sanity-checking the autograd gradient direction (no
 * momentum to obscure things). params is a 1-D 2-element tensor:
 * [alpha, weight_decay]. */
void *tnn_opt_step_sgd(void *sess, void *a, void *grad, void *params)
{
    if (!sess || !a || !grad || !params) return NULL;
    tnn_session *s = (tnn_session *)sess;
    return (void *)ggml_opt_step_sgd(s->ctx,
                                      (struct ggml_tensor *)a,
                                      (struct ggml_tensor *)grad,
                                      (struct ggml_tensor *)params);
}

int tnn_realize(void *sess, void *result)
{
    if (!sess || !result) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (s->realized) return -2;
    int64_t _t = tnn_trace_begin("realize");
    ggml_build_forward_expand(s->graph, (struct ggml_tensor *)result);
    ggml_backend_sched_reset(s->engine->sched);
    int ok = ggml_backend_sched_alloc_graph(s->engine->sched, s->graph) ? 1 : 0;
    tnn_trace_end("realize", _t);
    if (!ok) return -3;
    s->realized   = 1;
    s->last_graph = 1;
    return 0;
}

/* Same as tnn_realize minus the sched-alloc. Training callers use this:
 * tnn_build_forward_only(sess, loss) → tnn_build_backward(sess) →
 * (optional tnn_extend_backward_graph for opt_step) → tnn_realize_backward.
 * The follow-up tnn_realize_backward does the single sched-alloc on the
 * combined graph_b. Calling tnn_realize THEN tnn_realize_backward is
 * broken: the sched_reset between the two leaves tensor buffer pointers
 * stale and the second alloc lands tensors on freed-pool memory (validated
 * 2026-05-20 with a standalone ggml POC reproducing micro5's failure
 * byte-for-byte; see docs/design/phase-f1-status.md). */
int tnn_build_forward_only(void *sess, void *result)
{
    if (!sess || !result) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (s->realized) return -2;
    ggml_build_forward_expand(s->graph, (struct ggml_tensor *)result);
    s->realized   = 1;
    s->last_graph = 1;
    return 0;
}

/* Add an extra tensor's compute tree to the graph BEFORE tnn_realize.
 * Use for side-effect ops (ggml_cpy into a view) that aren't reachable
 * from the final result tensor — without this they'd be pruned. The
 * realize-target's tree is appended later by tnn_realize itself. */
int tnn_add_to_graph(void *sess, void *tensor)
{
    if (!sess || !tensor) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (s->realized) return -2;
    ggml_build_forward_expand(s->graph, (struct ggml_tensor *)tensor);
    return 0;
}

/* Reset for rebuild: free the compute ctx entirely and start fresh.
 * The persistent ctx_w + its backend buffer are untouched, so weights
 * keep their data. Previously this only swapped graphs in the same
 * ctx — that grew monotonically and overflowed after ~80 decode steps
 * at gpt2-small + max_T=1024 (each step creates ~1300 new tensor
 * headers, none get reclaimed). Tearing ctx down per step makes the
 * per-decode-step compute fully bounded in metadata footprint.
 *
 * The scheduler also has internal state tied to tensor pointers; we
 * reset it before realize, so this is safe. Per decode step:
 *   tnn_reset_for_rebuild(sess)
 *   ... build ops with current pos baked in ...
 *   tnn_realize(sess, result_tensor)
 *   ... upload, compute, download ... */
int tnn_reset_for_rebuild(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    /* Profile timing showed that free()+init() of the (now 130-ish MB)
     * ctx_buf adds ~500 ms per call — dominates compute. So we ONLY
     * teardown when the ctx is approaching capacity. The (small)
     * accumulated dead headers between teardowns are bounded by
     * ctx_used / ctx_buf_size, which we check before each rebuild
     * via ggml_used_mem.
     *
     * Threshold: half the buffer. Headroom ensures the *next* step's
     * graph build can complete without overflowing. */
    size_t used = ggml_used_mem(s->ctx);
    if (used > s->ctx_buf_size / 2) {
        ggml_free(s->ctx);
        struct ggml_init_params params = {
            /*.mem_size   =*/ s->ctx_buf_size,
            /*.mem_buffer =*/ s->ctx_buf,
            /*.no_alloc   =*/ true,
        };
        s->ctx        = ggml_init(params);
        s->graph_b    = ggml_new_graph_custom(s->ctx, 16384, false);
        s->realized_b = 0;
    }
    s->realized = 0;
    s->graph    = ggml_new_graph_custom(s->ctx, 16384, false);
    s->last_graph = 0;
    return 0;
}

int tnn_compute(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (!s->realized) return -2;
    int64_t _t = tnn_trace_begin("compute");
    enum ggml_status rc = ggml_backend_sched_graph_compute(s->engine->sched, s->graph);
    tnn_trace_end("compute", _t);
    return (rc == GGML_STATUS_SUCCESS) ? 0 : (int)rc;
}

/* Build a SECONDARY graph (graph_b) in the same session, sharing ctx
 * and tensors with the primary. Does NOT alloc — call tnn_switch_b
 * before tnn_compute_b each cycle. */
int tnn_realize_b(void *sess, void *result)
{
    if (!sess || !result) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (s->realized_b) return -2;
    ggml_build_forward_expand(s->graph_b, (struct ggml_tensor *)result);
    s->realized_b = 1;
    return 0;
}

/* Switch sched allocation to graph_b (or back to graph). Resets the
 * scheduler then allocates buffer slots for the requested graph's
 * compute tensors. Persistent tensors (allocated via ctx_w) keep
 * their stable buffer locations. Compute tensors (h, intermediates)
 * get fresh slots that may differ from prior cycles -- caller MUST
 * re-upload any compute inputs before tnn_compute*. */
int tnn_switch_b(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (!s->realized_b) return -2;
    ggml_backend_sched_reset(s->engine->sched);
    if (!ggml_backend_sched_alloc_graph(s->engine->sched, s->graph_b)) return -3;
    s->last_graph = 2;
    return 0;
}

int tnn_switch_a(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (!s->realized) return -2;
    ggml_backend_sched_reset(s->engine->sched);
    if (!ggml_backend_sched_alloc_graph(s->engine->sched, s->graph)) return -3;
    s->last_graph = 1;
    return 0;
}

int tnn_compute_b(void *sess)
{
    if (!sess) return -1;
    tnn_session *s = (tnn_session *)sess;
    if (!s->realized_b) return -2;
    enum ggml_status rc = ggml_backend_sched_graph_compute(s->engine->sched, s->graph_b);
    return (rc == GGML_STATUS_SUCCESS) ? 0 : (int)rc;
}

/* Out-of-range scratch_set used to silently drop writes — a stage+upload
 * pair operating on a tensor larger than the scratch buffer would
 * truncate at the boundary and then `tnn_upload` would memcpy past
 * the scratch end into the next backend buffer. That bug bit
 * Qwen2.5-0.5B (ffn_gate = 4.36M floats > 4M scratch slots; 17.4 MB
 * upload past a 16 MiB scratch) and produced NaN logits at L=1 with
 * no visible error. Now we fprintf a one-line warning the FIRST time
 * we see an out-of-range write per session — noisy enough to catch
 * future regressions without spamming the logs. */
void tnn_scratch_set(void *sess, int idx, double v)
{
    if (!sess) return;
    tnn_session *s = (tnn_session *)sess;
    int max_n = TNN_SCRATCH_BYTES / (int)sizeof(float);
    if (idx < 0 || idx >= max_n) {
        if (!s->scratch_overflow_warned) {
            fprintf(stderr, "[tnn] WARN: tnn_scratch_set idx=%d out of range "
                            "(max=%d, scratch=%d bytes). Subsequent uploads "
                            "from this scratch are corrupt — use a chunked "
                            "uploader (e.g. tnn_upload_transposed_f64).\n",
                            idx, max_n, TNN_SCRATCH_BYTES);
            s->scratch_overflow_warned = 1;
        }
        return;
    }
    s->scratch[idx] = (float)v;
}

/* Out-of-range reads used to silently return 0.0 — indistinguishable
 * from a legitimate zero in the scratch slot. Now we still return 0.0
 * for backward compatibility, but emit a once-per-session warning so
 * the failure is visible. Callers that need the legitimate zero/OOR
 * distinction should check bounds themselves. */
double tnn_scratch_get(void *sess, int idx)
{
    if (!sess) return 0.0;
    tnn_session *s = (tnn_session *)sess;
    int max_n = TNN_SCRATCH_BYTES / (int)sizeof(float);
    if (idx < 0 || idx >= max_n) {
        if (!s->scratch_overflow_warned) {
            fprintf(stderr, "[tnn] WARN: tnn_scratch_get idx=%d out of range "
                            "(max=%d). Returning 0.0 — but this is now a "
                            "silent zero, not a real one. Check your indexing.\n",
                            idx, max_n);
            s->scratch_overflow_warned = 1;
        }
        return 0.0;
    }
    return (double)s->scratch[idx];
}

/* The scratch buffer is just bytes; we let i32 values share it. Caller
 * must not mix i32 + f32 writes within a single tensor's upload window.
 * Same overflow warning as tnn_scratch_set — once-per-session fprintf. */
void tnn_scratch_set_i32(void *sess, int idx, int value)
{
    if (!sess) return;
    tnn_session *s = (tnn_session *)sess;
    int max_n = TNN_SCRATCH_BYTES / (int)sizeof(int32_t);
    if (idx < 0 || idx >= max_n) {
        if (!s->scratch_overflow_warned) {
            fprintf(stderr, "[tnn] WARN: tnn_scratch_set_i32 idx=%d out of "
                            "range (max=%d). Use a chunked uploader.\n",
                            idx, max_n);
            s->scratch_overflow_warned = 1;
        }
        return;
    }
    ((int32_t *)s->scratch)[idx] = (int32_t)value;
}

int tnn_scratch_get_i32(void *sess, int idx)
{
    if (!sess) return 0;
    tnn_session *s = (tnn_session *)sess;
    int max_n = TNN_SCRATCH_BYTES / (int)sizeof(int32_t);
    if (idx < 0 || idx >= max_n) {
        if (!s->scratch_overflow_warned) {
            fprintf(stderr, "[tnn] WARN: tnn_scratch_get_i32 idx=%d out of "
                            "range (max=%d). Returning 0 — but this is a "
                            "silent zero, not a real one.\n",
                            idx, max_n);
            s->scratch_overflow_warned = 1;
        }
        return 0;
    }
    return (int)((int32_t *)s->scratch)[idx];
}

/* Bounds-checked upload: tensor must fit in the 16 MiB scratch. Larger
 * tensors caused the silent UB that produced NaN logits at L=1 on
 * Qwen2.5-0.5B (ffn_gate = 17.4 MB > 16 MB scratch); the memcpy past
 * the scratch end overwrote adjacent heap. Use chunked uploaders for
 * anything that might be large:
 *   - tnn_upload_from_float_array (chunked f32 upload)
 *   - tnn_upload_transposed_f64   (chunked transposed f64 upload)
 * Returns 0 on success, -1 on null sess/tensor, -2 on size overflow. */
int tnn_upload(void *sess, void *tensor)
{
    if (!sess || !tensor) return -1;
    tnn_session *s = (tnn_session *)sess;
    struct ggml_tensor *t = (struct ggml_tensor *)tensor;
    size_t nbytes = ggml_nbytes(t);
    if (nbytes > (size_t)TNN_SCRATCH_BYTES) {
        if (!s->scratch_overflow_warned) {
            fprintf(stderr, "[tnn] WARN: tnn_upload tensor=%zu bytes exceeds "
                            "scratch=%d bytes. Skipping upload (was: silent UB). "
                            "Use tnn_upload_from_float_array or "
                            "tnn_upload_transposed_f64 for tensors > 16 MiB.\n",
                            nbytes, TNN_SCRATCH_BYTES);
            s->scratch_overflow_warned = 1;
        }
        return -2;
    }
    int64_t _t = tnn_trace_begin("upload");
    ggml_backend_tensor_set(t, s->scratch, 0, nbytes);
    tnn_trace_end("upload", _t);
    return 0;
}

/* Same bounds check as tnn_upload — a download into an oversized
 * tensor would memcpy past the scratch end into adjacent heap. */
int tnn_download(void *sess, void *tensor)
{
    if (!sess || !tensor) return -1;
    tnn_session *s = (tnn_session *)sess;
    struct ggml_tensor *t = (struct ggml_tensor *)tensor;
    size_t nbytes = ggml_nbytes(t);
    if (nbytes > (size_t)TNN_SCRATCH_BYTES) {
        if (!s->scratch_overflow_warned) {
            fprintf(stderr, "[tnn] WARN: tnn_download tensor=%zu bytes exceeds "
                            "scratch=%d bytes. Skipping download (was: silent UB). "
                            "Use tnn_download_to_f64_array for tensors > 16 MiB.\n",
                            nbytes, TNN_SCRATCH_BYTES);
            s->scratch_overflow_warned = 1;
        }
        return -2;
    }
    int64_t _t = tnn_trace_begin("download");
    ggml_backend_tensor_get(t, s->scratch, 0, nbytes);
    tnn_trace_end("download", _t);
    return 0;
}

/* Transpose-and-upload a row-major f64 Mat into a ggml f32 tensor of
 * shape ne=[br, bc] in chunked passes — so it works for tensors larger
 * than the 16 MiB scratch buffer.
 *
 * Source layout: src[i*bc + j] = (i, j) of an (br × bc) row-major Mat.
 * Destination ggml layout: T[ne0=k0, ne1=k1] at byte offset k1*br + k0
 * (in float positions). We want T[i, j] = src[i, j] (transpose semantics
 * is in the *consumer* — ggml_mul_mat treats (br, bc) as (K, M) where
 * the K axis is contracted; we get B^T · h that way).
 *
 * Chunking: pick `cols_per_chunk` ≤ scratch_slots / br. For each chunk
 * [j_start, j_end) of columns: stage src[i, j] → scratch[(j - j_start)*br + i]
 * for i ∈ [0, br) and j ∈ [j_start, j_end). Then upload that contiguous
 * slice into the tensor at byte offset j_start*br*sizeof(float).
 *
 * Same shape as tnn_upload_from_float_array's chunking, but for the
 * transposed-input case used by stage_transposed_and_upload. Fixes the
 * scratch-overflow bug that produced garbage uploads for Qwen's
 * ffn_gate / ffn_up / ffn_down (each ~17 MB, scratch is 16 MB). */
int tnn_upload_transposed_f64(void *sess, void *tensor,
                              const double *src, int br, int bc)
{
    if (!sess || !tensor || !src || br <= 0 || bc <= 0) return -1;
    tnn_session *s = (tnn_session *)sess;
    struct ggml_tensor *t = (struct ggml_tensor *)tensor;

    size_t expected_bytes = (size_t)br * (size_t)bc * sizeof(float);
    if (expected_bytes > ggml_nbytes(t)) return -2;

    const int max_slots = TNN_SCRATCH_BYTES / (int)sizeof(float);
    int cols_per_chunk = max_slots / br;
    if (cols_per_chunk <= 0) return -3;   /* br > scratch — wider than ~4M */

    int j_start = 0;
    while (j_start < bc) {
        int j_end = j_start + cols_per_chunk;
        if (j_end > bc) j_end = bc;

        int j = j_start;
        while (j < j_end) {
            int i = 0;
            const double *src_row_base = src + (size_t)j;
            float *dst_col = s->scratch + (size_t)(j - j_start) * (size_t)br;
            while (i < br) {
                dst_col[i] = (float)src_row_base[(size_t)i * (size_t)bc];
                i++;
            }
            j++;
        }

        size_t byte_off = (size_t)j_start * (size_t)br * sizeof(float);
        size_t byte_len = (size_t)(j_end - j_start) * (size_t)br * sizeof(float);
        ggml_backend_tensor_set(t, s->scratch, byte_off, byte_len);

        j_start = j_end;
    }
    return 0;
}

int tnn_upload_from_float_array(void *sess, void *tensor, const double *data, size_t n)
{
    if (!sess || !tensor || !data) return -1;
    int64_t _trace = tnn_trace_begin("upload_from_float_array");
    tnn_session *s = (tnn_session *)sess;
    struct ggml_tensor *t = (struct ggml_tensor *)tensor;
    const size_t chunk_floats = TNN_SCRATCH_BYTES / sizeof(float);

    /* Chunked f64 → f32 conversion into scratch, then ggml_backend_tensor_set
     * per chunk at the right byte offset. Lets us upload tensors larger
     * than scratch (e.g. distilgpt2's 38.6 M-element token_embd) without
     * growing the scratch buffer for everyone. */
    size_t off = 0;
    while (off < n) {
        size_t this_chunk = (n - off) < chunk_floats ? (n - off) : chunk_floats;
        for (size_t i = 0; i < this_chunk; ++i) {
            s->scratch[i] = (float)data[off + i];
        }
        ggml_backend_tensor_set(t, s->scratch,
                                  off * sizeof(float),
                                  this_chunk * sizeof(float));
        off += this_chunk;
    }
    tnn_trace_end("upload_from_float_array", _trace);
    return 0;
}

/* Mirror of tnn_upload_from_float_array: read a tensor's f32 contents
 * back into a host f64 buffer in scratch-sized chunks. Enables full
 * Mat-roundtrip on weights loaded via the direct GGUF→FFI path —
 * required by the user-stated rule that the API mustn't paint into
 * an inference-only corner. */
int tnn_download_to_f64_array(void *sess, void *tensor, double *dst, size_t n)
{
    if (!sess || !tensor || !dst) return -1;
    tnn_session *s = (tnn_session *)sess;
    struct ggml_tensor *t = (struct ggml_tensor *)tensor;
    size_t available = ggml_nelements(t);
    if (n > available) return -2;

    const size_t chunk_floats = TNN_SCRATCH_BYTES / sizeof(float);
    size_t off = 0;
    while (off < n) {
        size_t this_chunk = (n - off) < chunk_floats ? (n - off) : chunk_floats;
        ggml_backend_tensor_get(t, s->scratch,
                                  off * sizeof(float),
                                  this_chunk * sizeof(float));
        for (size_t i = 0; i < this_chunk; ++i) {
            dst[off + i] = (double)s->scratch[i];
        }
        off += this_chunk;
    }
    return 0;
}

int tnn_upload_from_int_array(void *sess, void *tensor, const long *data, size_t n)
{
    if (!sess || !tensor || !data) return -1;
    tnn_session *s = (tnn_session *)sess;
    struct ggml_tensor *t = (struct ggml_tensor *)tensor;
    size_t max_n = TNN_SCRATCH_BYTES / sizeof(int32_t);
    if (n > max_n) return -2;

    int64_t _trace = tnn_trace_begin("upload_from_int_array");
    int32_t *dst = (int32_t *)s->scratch;
    /* i64 → i32 narrowing. Spinel's :int_array is `const int64_t *`; ggml's
     * GGML_TYPE_I32 row-index tensors are 32-bit. Caller responsibility
     * not to pass out-of-range indices (vocab fits easily in int32). */
    for (size_t i = 0; i < n; ++i) dst[i] = (int32_t)data[i];

    ggml_backend_tensor_set(t, dst, 0, n * sizeof(int32_t));
    tnn_trace_end("upload_from_int_array", _trace);
    return 0;
}

/* Scratch-buffer stats. Caller has just done tnn_download(sess, t)
 * which copied a tensor's f32 contents into the session's scratch
 * buffer. These helpers reduce over the first `n` floats without
 * crossing the FFI boundary per element — one Ruby↔C call per stat,
 * O(n) in C. Used by the trace-tap diagnostic path; not on any
 * production hot path. */
double tnn_scratch_min_f32(void *sess, int n)
{
    if (!sess || n <= 0) return 0.0;
    tnn_session *s = (tnn_session *)sess;
    float mn = s->scratch[0];
    int i = 1;
    while (i < n) { if (s->scratch[i] < mn) mn = s->scratch[i]; i++; }
    return (double)mn;
}

double tnn_scratch_max_f32(void *sess, int n)
{
    if (!sess || n <= 0) return 0.0;
    tnn_session *s = (tnn_session *)sess;
    float mx = s->scratch[0];
    int i = 1;
    while (i < n) { if (s->scratch[i] > mx) mx = s->scratch[i]; i++; }
    return (double)mx;
}

double tnn_scratch_sum_abs_f32(void *sess, int n)
{
    if (!sess || n <= 0) return 0.0;
    tnn_session *s = (tnn_session *)sess;
    double sum = 0.0;
    int i = 0;
    while (i < n) {
        float v = s->scratch[i];
        sum += v < 0.0f ? -(double)v : (double)v;
        i++;
    }
    return sum;
}

/* Count of NaN-or-inf elements. NaN comparison: v != v is true iff NaN.
 * Inf: abs(v) > 1e30 is conservative (real f32 inf is 3.4e38). */
int tnn_scratch_nan_count_f32(void *sess, int n)
{
    if (!sess || n <= 0) return 0;
    tnn_session *s = (tnn_session *)sess;
    int c = 0;
    int i = 0;
    while (i < n) {
        float v = s->scratch[i];
        float av = v < 0.0f ? -v : v;
        if (v != v || av > 1.0e30f) c++;
        i++;
    }
    return c;
}

int tnn_tensor_ne0(void *t) { return t ? (int)((struct ggml_tensor *)t)->ne[0] : 0; }
int tnn_tensor_ne1(void *t) { return t ? (int)((struct ggml_tensor *)t)->ne[1] : 0; }
size_t tnn_tensor_nbytes(void *t) { return t ? ggml_nbytes((struct ggml_tensor *)t) : 0; }
int    tnn_tensor_nelements(void *t) { return t ? (int)ggml_nelements((struct ggml_tensor *)t) : 0; }
