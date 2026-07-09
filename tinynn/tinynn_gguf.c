#include "tinynn_gguf.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>

/* Phase 2 (memory-design): we now load GGUFs with no_alloc=true and
 * mmap the file ourselves. Tensor byte accessors point into the
 * mapped region directly — no double-buffer at load. The persistent
 * destination buffer is still a separate allocation, but the source
 * side no longer materializes a full f32-equivalent copy of the
 * file. */
typedef struct {
    struct gguf_context *gguf_ctx;
    struct ggml_context *ggml_ctx;   /* tensors have NULL data with no_alloc */
    int    fd;                       /* -1 for "empty" sessions */
    void  *map;                      /* NULL if no mmap (empty session) */
    size_t map_size;
    size_t data_offset;              /* gguf data section start, absolute */
} tnn_gguf_session;

void *tnn_gguf_load(const char *path)
{
    if (!path) return NULL;
    struct ggml_context *ggml_ctx = NULL;
    struct gguf_init_params params;
    params.no_alloc = true;   /* Phase 2: tensor data lives in mmap, not in ctx */
    params.ctx      = &ggml_ctx;
    struct gguf_context *gctx = gguf_init_from_file(path, params);
    if (!gctx) return NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0) { gguf_free(gctx); if (ggml_ctx) ggml_free(ggml_ctx); return NULL; }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd); gguf_free(gctx); if (ggml_ctx) ggml_free(ggml_ctx); return NULL;
    }
    size_t fsize = (size_t)st.st_size;
    void *map = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        close(fd); gguf_free(gctx); if (ggml_ctx) ggml_free(ggml_ctx); return NULL;
    }
    /* Hint the kernel that we'll touch most pages once. Keeps the
     * page-cache footprint friendlier under memory pressure. */
    /* MADV_WILLNEED: ask the kernel to read ahead. Required for the
     * CUDA-side cudaMemcpyAsync H→D path — it doesn't reliably
     * page-fault file-backed mmap pages on read, producing wrong
     * bytes in the device buffer if pages aren't already resident.
     * (Hypothesis under test 2026-05-18. Was MADV_RANDOM, which
     * told the kernel NOT to read ahead.) */
    madvise(map, fsize, MADV_WILLNEED);

    tnn_gguf_session *s = (tnn_gguf_session *)calloc(1, sizeof(*s));
    if (!s) {
        munmap(map, fsize); close(fd);
        gguf_free(gctx); if (ggml_ctx) ggml_free(ggml_ctx);
        return NULL;
    }
    s->gguf_ctx    = gctx;
    s->ggml_ctx    = ggml_ctx;
    s->fd          = fd;
    s->map         = map;
    s->map_size    = fsize;
    s->data_offset = gguf_get_data_offset(gctx);
    return s;
}

void *tnn_gguf_load_empty(void)
{
    tnn_gguf_session *s = (tnn_gguf_session *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->gguf_ctx = gguf_init_empty();
    s->ggml_ctx = NULL;
    s->fd       = -1;
    s->map      = NULL;
    return s;
}

void tnn_gguf_free(void *handle)
{
    if (!handle) return;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    if (s->map) munmap(s->map, s->map_size);
    if (s->fd > 0) close(s->fd);
    if (s->ggml_ctx) ggml_free(s->ggml_ctx);
    if (s->gguf_ctx) gguf_free(s->gguf_ctx);
    free(s);
}

/* Compute the pointer inside the mmap region for tensor `tensor_idx`'s
 * raw bytes. Returns NULL if the session has no mmap (empty session). */
static const void *gguf_tensor_mmap_ptr(tnn_gguf_session *s, int tensor_idx)
{
    if (!s->map) return NULL;
    size_t off = gguf_get_tensor_offset(s->gguf_ctx, (int64_t)tensor_idx);
    return (const char *)s->map + s->data_offset + off;
}

int tnn_gguf_n_tensors(void *handle)
{
    if (!handle) return 0;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    return (int)gguf_get_n_tensors(s->gguf_ctx);
}

const char *tnn_gguf_tensor_name(void *handle, int i)
{
    if (!handle) return NULL;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    return gguf_get_tensor_name(s->gguf_ctx, (int64_t)i);
}

int tnn_gguf_tensor_ne(void *handle, int i, int dim)
{
    if (!handle || dim < 0 || dim >= 4) return 0;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    if (!s->ggml_ctx) return 0;
    const char *name = gguf_get_tensor_name(s->gguf_ctx, (int64_t)i);
    if (!name) return 0;
    struct ggml_tensor *t = ggml_get_tensor(s->ggml_ctx, name);
    return t ? (int)t->ne[dim] : 0;
}

int tnn_gguf_tensor_type(void *handle, int i)
{
    if (!handle) return -1;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    return (int)gguf_get_tensor_type(s->gguf_ctx, (int64_t)i);
}

size_t tnn_gguf_tensor_nbytes(void *handle, int i)
{
    if (!handle) return 0;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    return gguf_get_tensor_size(s->gguf_ctx, (int64_t)i);
}

int tnn_gguf_read_f32_to_doubles(void *handle, int i, double *out, size_t n)
{
    if (!handle || !out) return -1;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    if (!s->ggml_ctx) return -2;
    const char *name = gguf_get_tensor_name(s->gguf_ctx, (int64_t)i);
    if (!name) return -3;
    struct ggml_tensor *t = ggml_get_tensor(s->ggml_ctx, name);
    if (!t) return -4;
    const void *src_bytes = gguf_tensor_mmap_ptr(s, i);
    if (!src_bytes) return -4;
    size_t available = ggml_nelements(t);
    if (n > available) n = available;

    /* F32: direct copy. */
    if (t->type == GGML_TYPE_F32) {
        const float *src = (const float *)src_bytes;
        for (size_t k = 0; k < n; ++k) out[k] = (double)src[k];
        return 0;
    }

    /* Quantized (Q8_0, Q4_K, Q5_K, F16, BF16, …): dequantize through
     * ggml's type_traits.to_float. Allocates a temporary f32 buffer
     * sized to the tensor's full element count (read past `n` not
     * supported per-block-aligned dequantization). */
    const struct ggml_type_traits *traits = ggml_get_type_traits(t->type);
    if (!traits || !traits->to_float) return -5;

    /* to_float must dequantize whole blocks; we ask for `available`
     * elements (the full tensor) and then only forward `n` of them. */
    float *fbuf = (float *)malloc(available * sizeof(float));
    if (!fbuf) return -6;
    traits->to_float(src_bytes, fbuf, (int64_t)available);
    for (size_t k = 0; k < n; ++k) out[k] = (double)fbuf[k];
    free(fbuf);
    return 0;
}

/* Convenience: tell the caller whether a given tensor index needs
 * dequantization (i.e. is not GGML_TYPE_F32). Useful for the Ruby
 * side to print "loaded Q8_0 tensor X" type diagnostics. */
int tnn_gguf_tensor_is_quantized(void *handle, int i)
{
    if (!handle) return 0;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    enum ggml_type t = gguf_get_tensor_type(s->gguf_ctx, (int64_t)i);
    return (t == GGML_TYPE_F32) ? 0 : 1;
}

/* Look up a uint32 metadata key (e.g. "gpt2.embedding_length"). Returns
 * the value, or -1 if the key is missing or has a non-u32 type. The
 * caller is expected to know which kv key names are valid for the
 * file's architecture. */
int tnn_gguf_get_u32(void *handle, const char *key)
{
    if (!handle || !key) return -1;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    int64_t kid = gguf_find_key(s->gguf_ctx, key);
    if (kid < 0) return -1;
    enum gguf_type t = gguf_get_kv_type(s->gguf_ctx, kid);
    if (t != GGUF_TYPE_UINT32) return -1;
    return (int)gguf_get_val_u32(s->gguf_ctx, kid);
}

/* Same shape for float-typed metadata (e.g. layer_norm_epsilon). */
double tnn_gguf_get_f32(void *handle, const char *key)
{
    if (!handle || !key) return 0.0;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    int64_t kid = gguf_find_key(s->gguf_ctx, key);
    if (kid < 0) return 0.0;
    enum gguf_type t = gguf_get_kv_type(s->gguf_ctx, kid);
    if (t != GGUF_TYPE_FLOAT32) return 0.0;
    return (double)gguf_get_val_f32(s->gguf_ctx, kid);
}

/* Phase 2 BYO-pointer accessors: expose the mmap base + size + per-
 * tensor file offsets so callers can wire weight tensors to point at
 * mmap'd file pages via tnn_session_attach_weight_mmap +
 * tnn_input_*_persistent_mmap. */
void *tnn_gguf_mmap_base(void *handle)
{
    if (!handle) return NULL;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    return s->map;
}

size_t tnn_gguf_mmap_size(void *handle)
{
    if (!handle) return 0;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    return s->map_size;
}

/* Absolute byte offset (from mmap base) where tensor `i`'s data
 * starts. Combines the file's data-section offset with the per-tensor
 * offset within that section. */
size_t tnn_gguf_tensor_file_offset(void *handle, int i)
{
    if (!handle) return 0;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    return s->data_offset + gguf_get_tensor_offset(s->gguf_ctx, (int64_t)i);
}

/* Bool metadata. Returns 1 if key is present and true, 0 if false or
 * missing. */
int tnn_gguf_get_bool(void *handle, const char *key)
{
    if (!handle || !key) return 0;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    int64_t kid = gguf_find_key(s->gguf_ctx, key);
    if (kid < 0) return 0;
    enum gguf_type t = gguf_get_kv_type(s->gguf_ctx, kid);
    if (t != GGUF_TYPE_BOOL) return 0;
    return gguf_get_val_bool(s->gguf_ctx, kid) ? 1 : 0;
}

/* String metadata. Used for general.architecture, general.name, and
 * the tokenizer's chat_template. Returns NULL if the key is missing
 * or has a non-string type. The returned pointer is owned by ggml
 * (lives until tnn_gguf_free); callers should not free it. */
const char *tnn_gguf_get_str(void *handle, const char *key)
{
    if (!handle || !key) return NULL;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    int64_t kid = gguf_find_key(s->gguf_ctx, key);
    if (kid < 0) return NULL;
    enum gguf_type t = gguf_get_kv_type(s->gguf_ctx, kid);
    if (t != GGUF_TYPE_STRING) return NULL;
    return gguf_get_val_str(s->gguf_ctx, kid);
}

/* Array metadata: length. Used to enumerate tokenizer.ggml.tokens
 * (string array, == vocab size) and .merges (string array of BPE
 * pairs). Returns -1 if the key is missing or non-array. */
int tnn_gguf_arr_n(void *handle, const char *key)
{
    if (!handle || !key) return -1;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    int64_t kid = gguf_find_key(s->gguf_ctx, key);
    if (kid < 0) return -1;
    enum gguf_type t = gguf_get_kv_type(s->gguf_ctx, kid);
    if (t != GGUF_TYPE_ARRAY) return -1;
    return (int)gguf_get_arr_n(s->gguf_ctx, kid);
}

/* Array element accessor (string-typed arrays only). Used to iterate
 * tokenizer vocab + merges. Returns NULL on bounds error or non-string
 * array. Caller must not free the returned pointer. */
const char *tnn_gguf_arr_str(void *handle, const char *key, int i)
{
    if (!handle || !key || i < 0) return NULL;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    int64_t kid = gguf_find_key(s->gguf_ctx, key);
    if (kid < 0) return NULL;
    enum gguf_type t = gguf_get_kv_type(s->gguf_ctx, kid);
    if (t != GGUF_TYPE_ARRAY) return NULL;
    enum gguf_type at = gguf_get_arr_type(s->gguf_ctx, kid);
    if (at != GGUF_TYPE_STRING) return NULL;
    if ((size_t)i >= gguf_get_arr_n(s->gguf_ctx, kid)) return NULL;
    return gguf_get_arr_str(s->gguf_ctx, kid, (size_t)i);
}

/* Helper: locate the raw bytes of a GGUF tensor in the mmap'd file.
 * No dequant — caller wants the on-disk type preserved (Phase 3 of
 * memory-design.md: Q8 stays Q8 in memory). Phase 2: the bytes live
 * in the mmap region, not a ggml-context-owned buffer. */
static const void *gguf_tensor_raw_bytes(tnn_gguf_session *s, int tensor_idx,
                                           size_t *out_nbytes,
                                           int *out_ggml_type)
{
    const char *name = gguf_get_tensor_name(s->gguf_ctx, (int64_t)tensor_idx);
    if (!name) return NULL;
    struct ggml_tensor *t = ggml_get_tensor(s->ggml_ctx, name);
    if (!t) return NULL;
    if (out_nbytes)    *out_nbytes    = ggml_nbytes(t);
    if (out_ggml_type) *out_ggml_type = (int)t->type;
    return gguf_tensor_mmap_ptr(s, tensor_idx);
}

/* Copy raw bytes from a GGUF tensor verbatim into a persistent tensor.
 * Used for quantized weights where the persistent tensor was allocated
 * with the same ggml type (e.g. GGML_TYPE_Q8_0). Asserts type + size
 * match the destination. Returns 0 on success. */
int tnn_gguf_copy_verbatim_to_persistent(void *handle, int tensor_idx,
                                          void *sess, void *target_tensor)
{
    (void)sess;
    if (!handle || !target_tensor || tensor_idx < 0) return -1;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    struct ggml_tensor *dst = (struct ggml_tensor *)target_tensor;

    size_t src_nbytes = 0;
    int    src_type   = 0;
    const void *src = gguf_tensor_raw_bytes(s, tensor_idx, &src_nbytes, &src_type);
    if (!src) return -2;
    if (src_type != (int)dst->type) return -3;
    if (src_nbytes != ggml_nbytes(dst)) return -4;

    ggml_backend_tensor_set(dst, src, 0, src_nbytes);
    return 0;
}

/* Per-head verbatim slice. Source HF-native [n_heads_total*d_head, d_model]
 * in whatever ggml type the GGUF stores. Head h occupies a contiguous
 * byte range — head_nbytes is ggml_nbytes(dst), starting offset is
 * head_idx * head_nbytes from the source tensor base. Works for f32
 * and any block-quantized type whose block size divides d_model. */
int tnn_gguf_copy_verbatim_head_slice_to_persistent(void *handle, int tensor_idx,
                                                      void *sess, void *target_tensor,
                                                      int head_idx, int n_heads_total)
{
    (void)sess;
    if (!handle || !target_tensor || head_idx < 0 || n_heads_total <= 0) return -1;
    if (head_idx >= n_heads_total) return -1;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    struct ggml_tensor *dst = (struct ggml_tensor *)target_tensor;

    size_t src_nbytes = 0;
    int    src_type   = 0;
    const void *src = gguf_tensor_raw_bytes(s, tensor_idx, &src_nbytes, &src_type);
    if (!src) return -2;
    if (src_type != (int)dst->type) return -3;

    size_t head_nbytes = ggml_nbytes(dst);
    if (head_nbytes * (size_t)n_heads_total != src_nbytes) return -4;

    size_t head_off = (size_t)head_idx * head_nbytes;
    ggml_backend_tensor_set(dst, (const char *)src + head_off, 0, head_nbytes);
    return 0;
}

/* --- direct GGUF → FFI persistent buffer loaders ------------------ */

/* Helper: get a pointer to the tensor's f32 data, dequantizing into
 * `*owned_buf` if the source is quantized/F16/BF16. Caller must free
 * `*owned_buf` if it's set to non-NULL on return. Returns NULL on error. */
static const float *gguf_tensor_as_f32(tnn_gguf_session *s, int tensor_idx,
                                        size_t expect_n,
                                        float **owned_buf, int *err)
{
    *owned_buf = NULL;
    *err = 0;
    const char *name = gguf_get_tensor_name(s->gguf_ctx, (int64_t)tensor_idx);
    if (!name) { *err = -3; return NULL; }
    struct ggml_tensor *t = ggml_get_tensor(s->ggml_ctx, name);
    if (!t) { *err = -4; return NULL; }
    const void *src_bytes = gguf_tensor_mmap_ptr(s, tensor_idx);
    if (!src_bytes) { *err = -4; return NULL; }
    size_t avail = ggml_nelements(t);
    if (expect_n > 0 && expect_n != avail) {
        /* Tolerated when expect_n==0 (caller doesn't know); otherwise
         * a mismatch likely means the caller has the wrong shape. */
        *err = -7;
        return NULL;
    }
    if (t->type == GGML_TYPE_F32) {
        return (const float *)src_bytes;
    }
    const struct ggml_type_traits *traits = ggml_get_type_traits(t->type);
    if (!traits || !traits->to_float) { *err = -5; return NULL; }
    float *fbuf = (float *)malloc(avail * sizeof(float));
    if (!fbuf) { *err = -6; return NULL; }
    traits->to_float(src_bytes, fbuf, (int64_t)avail);
    *owned_buf = fbuf;
    return fbuf;
}

int tnn_gguf_find_index(void *handle, const char *name)
{
    if (!handle || !name) return -1;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    int n = gguf_get_n_tensors(s->gguf_ctx);
    for (int i = 0; i < n; ++i) {
        const char *cur = gguf_get_tensor_name(s->gguf_ctx, (int64_t)i);
        if (cur && strcmp(cur, name) == 0) return i;
    }
    return -1;
}

int tnn_gguf_copy_to_persistent(void *handle, int tensor_idx,
                                 void *sess, void *target_tensor)
{
    if (!handle || !target_tensor || tensor_idx < 0) return -1;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    struct ggml_tensor *dst = (struct ggml_tensor *)target_tensor;
    size_t dst_n = ggml_nelements(dst);

    float *owned = NULL;
    int err = 0;
    const float *src = gguf_tensor_as_f32(s, tensor_idx, dst_n, &owned, &err);
    if (!src) return err;

    /* Stage through anonymous heap when src is the mmap region
     * (CUDA's cudaMemcpyAsync H→D from file-backed mmap produces
     * wrong bytes in the device buffer; calloc'd memory works). */
    if (!owned) {
        size_t bytes = dst_n * sizeof(float);
        float *tmp = (float *)malloc(bytes);
        if (!tmp) return -6;
        memcpy(tmp, src, bytes);
        ggml_backend_tensor_set(dst, tmp, 0, bytes);
        free(tmp);
    } else {
        ggml_backend_tensor_set(dst, src, 0, dst_n * sizeof(float));
    }
    (void)sess;

    free(owned);
    return 0;
}

int tnn_gguf_copy_1d_to_persistent(void *handle, int tensor_idx,
                                    void *sess, void *target_tensor)
{
    /* 1-D direct copy — same wire as 2-D for our purposes. */
    return tnn_gguf_copy_to_persistent(handle, tensor_idx, sess, target_tensor);
}

/* Chunked transposed copy. Source is (br × bc) row-major; destination
 * is a ggml tensor with ne=[br, bc] whose backend storage we fill with
 * the transposed byte order. Same shape as tnn_upload_transposed_f64. */
int tnn_gguf_copy_transposed_to_persistent(void *handle, int tensor_idx,
                                            void *sess, void *target_tensor,
                                            int br, int bc)
{
    (void)sess;
    if (!handle || !target_tensor || br <= 0 || bc <= 0) return -1;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    struct ggml_tensor *dst = (struct ggml_tensor *)target_tensor;
    size_t expected = (size_t)br * (size_t)bc;
    if (expected * sizeof(float) > ggml_nbytes(dst)) return -2;

    float *owned = NULL;
    int err = 0;
    const float *src = gguf_tensor_as_f32(s, tensor_idx, expected, &owned, &err);
    if (!src) return err;

    /* Stage one chunk of columns at a time into a single allocated
     * buffer (sized to hold the chunk), then ggml_backend_tensor_set
     * with byte offset. We allocate the staging buffer once per call
     * — 16 MiB max, same as the session scratch — and reuse it
     * across iterations of the column loop. */
    const size_t chunk_floats_max = 4 * 1024 * 1024;   /* 16 MiB */
    int cols_per_chunk = (int)(chunk_floats_max / (size_t)br);
    if (cols_per_chunk <= 0) {
        if (owned) free(owned);
        return -3;
    }
    size_t stage_bytes = (size_t)cols_per_chunk * (size_t)br * sizeof(float);
    float *stage = (float *)malloc(stage_bytes);
    if (!stage) {
        if (owned) free(owned);
        return -6;
    }

    int j_start = 0;
    while (j_start < bc) {
        int j_end = j_start + cols_per_chunk;
        if (j_end > bc) j_end = bc;
        int j = j_start;
        while (j < j_end) {
            const float *src_col_strided = src + j;
            float *dst_col = stage + (size_t)(j - j_start) * (size_t)br;
            int i = 0;
            while (i < br) {
                dst_col[i] = src_col_strided[(size_t)i * (size_t)bc];
                i++;
            }
            j++;
        }
        size_t byte_off = (size_t)j_start * (size_t)br * sizeof(float);
        size_t byte_len = (size_t)(j_end - j_start) * (size_t)br * sizeof(float);
        ggml_backend_tensor_set(dst, stage, byte_off, byte_len);
        j_start = j_end;
    }
    free(stage);
    free(owned);
    return 0;
}

/* Extract one head-slice of a [d_model, n_heads_total * d_head] tensor
 * and transpose-write it into a ne=[d_model, d_head] persistent target.
 *
 * Source row stride = n_heads_total * d_head; head h's columns are
 * [h*d_head, (h+1)*d_head). Per-head slice is (d_model × d_head)
 * row-major. We transpose-stage that into a chunked buffer, same
 * layout convention as tnn_upload_transposed_f64. */
int tnn_gguf_copy_head_slice_to_persistent(void *handle, int tensor_idx,
                                            void *sess, void *target_tensor,
                                            int head_idx, int n_heads_total,
                                            int d_model, int d_head)
{
    (void)sess;
    if (!handle || !target_tensor || head_idx < 0 || n_heads_total <= 0
        || d_model <= 0 || d_head <= 0) return -1;
    if (head_idx >= n_heads_total) return -1;

    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    struct ggml_tensor *dst = (struct ggml_tensor *)target_tensor;

    size_t total = (size_t)d_model * (size_t)n_heads_total * (size_t)d_head;
    float *owned = NULL;
    int err = 0;
    const float *src = gguf_tensor_as_f32(s, tensor_idx, total, &owned, &err);
    if (!src) return err;

    /* src layout: row i (i ∈ [0, d_model)) has n_heads_total*d_head
     * elements. Head h's d_head sub-row is at src + i*src_cols + h*d_head. */
    const int src_cols = n_heads_total * d_head;
    const int col_off  = head_idx * d_head;

    /* Stage transposed: dst ggml tensor ne=[d_model, d_head]. For
     * each "ne1 column" k (0..d_head), write d_model floats from
     * src[*, col_off + k]. Single chunk if (d_head * d_model) fits;
     * for our shapes d_model up to 4096, d_head 64..128 it always
     * fits in 16 MiB. */
    const size_t chunk_floats_max = 4 * 1024 * 1024;
    int cols_per_chunk = (int)(chunk_floats_max / (size_t)d_model);
    if (cols_per_chunk <= 0) {
        if (owned) free(owned);
        return -3;
    }
    if (cols_per_chunk > d_head) cols_per_chunk = d_head;

    size_t stage_bytes = (size_t)cols_per_chunk * (size_t)d_model * sizeof(float);
    float *stage = (float *)malloc(stage_bytes);
    if (!stage) {
        if (owned) free(owned);
        return -6;
    }

    int k_start = 0;
    while (k_start < d_head) {
        int k_end = k_start + cols_per_chunk;
        if (k_end > d_head) k_end = d_head;
        int k = k_start;
        while (k < k_end) {
            float *dst_col = stage + (size_t)(k - k_start) * (size_t)d_model;
            int i = 0;
            const float *src_row = src + col_off + k;
            while (i < d_model) {
                dst_col[i] = src_row[(size_t)i * (size_t)src_cols];
                i++;
            }
            k++;
        }
        size_t byte_off = (size_t)k_start * (size_t)d_model * sizeof(float);
        size_t byte_len = (size_t)(k_end - k_start) * (size_t)d_model * sizeof(float);
        ggml_backend_tensor_set(dst, stage, byte_off, byte_len);
        k_start = k_end;
    }
    free(stage);
    free(owned);
    return 0;
}

/* Native-layout per-head slice: HF-native source has shape
 * [n_heads_total * d_head, d_model] row-major. Head h occupies rows
 * [h*d_head, (h+1)*d_head); that block is a contiguous (d_head, d_model)
 * row-major slice. Those bytes already match ggml ne=[d_model, d_head]
 * column-major (because for j in d_head: for i in d_model: is the same
 * iteration order), so this is a plain ggml_backend_tensor_set with no
 * transpose stage. Used by GGUFLoad.load_kv_cache_directly_native to
 * fill the per-head t_w_q / t_w_k / t_w_v persistent tensors. */
int tnn_gguf_copy_head_slice_to_persistent_native(void *handle, int tensor_idx,
                                                    void *sess, void *target_tensor,
                                                    int head_idx, int n_heads_total,
                                                    int d_model, int d_head)
{
    (void)sess;
    if (!handle || !target_tensor || head_idx < 0 || n_heads_total <= 0
        || d_model <= 0 || d_head <= 0) return -1;
    if (head_idx >= n_heads_total) return -1;

    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    struct ggml_tensor *dst = (struct ggml_tensor *)target_tensor;

    size_t total = (size_t)d_model * (size_t)n_heads_total * (size_t)d_head;
    float *owned = NULL;
    int err = 0;
    const float *src = gguf_tensor_as_f32(s, tensor_idx, total, &owned, &err);
    if (!src) return err;

    size_t head_floats = (size_t)d_head * (size_t)d_model;
    size_t head_off    = (size_t)head_idx * head_floats;
    size_t bytes = head_floats * sizeof(float);

    /* Same staging as tnn_gguf_copy_to_persistent. */
    if (!owned) {
        float *tmp = (float *)malloc(bytes);
        if (!tmp) { free(owned); return -6; }
        memcpy(tmp, src + head_off, bytes);
        ggml_backend_tensor_set(dst, tmp, 0, bytes);
        free(tmp);
    } else {
        ggml_backend_tensor_set(dst, src + head_off, 0, bytes);
    }
    free(owned);
    return 0;
}

/* Extract one head-slice of a 1-D bias tensor [n_heads_total*d_head]
 * and copy d_head floats into the 1-D target (length d_head). */
int tnn_gguf_copy_head_bias_slice_to_persistent(void *handle, int tensor_idx,
                                                  void *sess, void *target_tensor,
                                                  int head_idx, int d_head)
{
    (void)sess;
    if (!handle || !target_tensor || head_idx < 0 || d_head <= 0) return -1;
    tnn_gguf_session *s = (tnn_gguf_session *)handle;
    struct ggml_tensor *dst = (struct ggml_tensor *)target_tensor;

    /* Don't require n_heads_total; instead read the tensor and slice
     * by byte offset based on head_idx*d_head*sizeof(float). The full
     * tensor is dequant'd (cheap — at most n_heads*d_head floats). */
    float *owned = NULL;
    int err = 0;
    const float *src = gguf_tensor_as_f32(s, tensor_idx, 0, &owned, &err);
    if (!src) return err;

    const float *head_slice = src + (size_t)head_idx * (size_t)d_head;
    ggml_backend_tensor_set(dst, head_slice, 0, (size_t)d_head * sizeof(float));
    free(owned);
    return 0;
}

int tnn_gguf_write_demo_file(const char *path)
{
    if (!path) return -1;

    /* Build a tiny ggml ctx, allocate one 2x3 f32 tensor, fill it. */
    size_t mem_size = 64 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ mem_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) return -2;
    struct ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);   /* ne0=3 cols, ne1=2 rows */
    ggml_set_name(t, "demo.tensor");
    float *d = (float *)t->data;
    for (int k = 0; k < 6; ++k) d[k] = (float)(k + 1);

    struct gguf_context *gctx = gguf_init_empty();
    gguf_add_tensor(gctx, t);
    int rc = gguf_write_to_file(gctx, path, false) ? 0 : -3;

    gguf_free(gctx);
    ggml_free(ctx);
    return rc;
}

/* Filesystem walking — minimal C shim so Spinel-compiled binaries can
 * scan model caches without a stdlib Dir.entries. Walks `root`
 * recursively, accumulates absolute paths whose name ends with
 * `.gguf` (case-sensitive), and returns them newline-separated in a
 * static buffer. Bails (returns "") if `root` is unreadable; bails if
 * the buffer would overflow.
 *
 * Returns: pointer to a NUL-terminated string of "path1\npath2\n..."
 *          (empty string if nothing found). The pointer is to a
 *          single static buffer reused across calls — copy if you
 *          need to keep the value across another call.
 *
 * Usage from Ruby: ffi_func :tnn_list_ggufs, [:str], :str. Split on
 * "\n" and discard empty trailing entry.
 */
#define TNN_LIST_BUF_BYTES (4 * 1024 * 1024)
static char tnn_list_buf[TNN_LIST_BUF_BYTES];
static size_t tnn_list_len = 0;

static void tnn_list_walk(const char *path) {
    DIR *d = opendir(path);
    if (!d) return;
    struct dirent *e;
    char buf[4096];
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.' &&
            (e->d_name[1] == '\0' ||
             (e->d_name[1] == '.' && e->d_name[2] == '\0'))) continue;
        int n = snprintf(buf, sizeof(buf), "%s/%s", path, e->d_name);
        if (n <= 0 || (size_t)n >= sizeof(buf)) continue;
        struct stat st;
        if (lstat(buf, &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            tnn_list_walk(buf);
        } else {
            int is_reg = S_ISREG(st.st_mode);
            if (S_ISLNK(st.st_mode)) {
                /* Hugging Face cache layout puts .gguf files as
                 * symlinks: hub/models--…/snapshots/<sha>/<name>.gguf
                 * → ../../blobs/<hash>. lstat says LNK; stat-follow
                 * to determine if the target is a regular file.
                 * Without this, HF-cached GGUFs were silently
                 * invisible to example_list_models on macOS (the
                 * Mac fresh-clone walkthrough 2026-05-28 hit this
                 * — the GGUF was downloaded but never listed). */
                struct stat tgt;
                if (stat(buf, &tgt) == 0 && S_ISREG(tgt.st_mode)) {
                    is_reg = 1;
                }
            }
            if (is_reg) {
                size_t name_len = strlen(e->d_name);
                if (name_len >= 5 &&
                    memcmp(e->d_name + name_len - 5, ".gguf", 5) == 0) {
                    size_t need = (size_t)n + 1;   /* path + newline */
                    if (tnn_list_len + need + 1 > TNN_LIST_BUF_BYTES) continue;
                    memcpy(tnn_list_buf + tnn_list_len, buf, (size_t)n);
                    tnn_list_len += (size_t)n;
                    tnn_list_buf[tnn_list_len++] = '\n';
                    tnn_list_buf[tnn_list_len] = '\0';
                }
            }
        }
    }
    closedir(d);
}

const char *tnn_list_ggufs(const char *root) {
    tnn_list_buf[0] = '\0';
    tnn_list_len = 0;
    if (root) tnn_list_walk(root);
    return tnn_list_buf;
}

/* File-size lookup. Returns 0 on stat failure (file missing,
 * unreadable). Used by ModelIndex for the size column. (Model
 * files are never 0 bytes; "not found" vs "empty" disambiguation
 * doesn't bite in practice. size_t keeps the FFI surface simple —
 * Spinel doesn't expose long long.) */
size_t tnn_file_size(const char *path) {
    if (!path) return 0;
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return (size_t)st.st_size;
}
