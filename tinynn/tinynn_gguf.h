#ifndef TINYNN_GGUF_H
#define TINYNN_GGUF_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Minimal GGUF loading wrapper around ggml's gguf_init_from_file —
 * needed because Spinel's FFI can't pass `struct gguf_init_params` by
 * value. The session owns both the gguf metadata context and the
 * ggml context that holds tensor headers + data.
 */
void  *tnn_gguf_load(const char *path);            /* NULL on error */
void  *tnn_gguf_load_empty(void);                   /* empty GGUF, 0 tensors */
void   tnn_gguf_free(void *handle);

int    tnn_gguf_n_tensors(void *handle);
const char *tnn_gguf_tensor_name(void *handle, int i);
int    tnn_gguf_tensor_ne(void *handle, int i, int dim);   /* dim 0..3 */
int    tnn_gguf_tensor_type(void *handle, int i);           /* GGML_TYPE_F32, etc. */
size_t tnn_gguf_tensor_nbytes(void *handle, int i);

/* Copy tensor data into a caller-owned double buffer (f64).
 * `n` is the number of elements (caller should query nbytes/elementsize).
 * Currently supports GGML_TYPE_F32 only — future: dequantize Q4/Q8/etc.
 */
/* Read tensor data as doubles. Handles F32 directly and Q8_0 / Q4_K /
 * Q5_K / Q6_K / F16 / BF16 / etc. via ggml's type_traits.to_float
 * dequantization path. */
int    tnn_gguf_read_f32_to_doubles(void *handle, int i, double *out, size_t n);
int    tnn_gguf_tensor_is_quantized(void *handle, int i);

/* Read scalar metadata kv pairs. The architecture decides the keys
 * (e.g. "gpt2.embedding_length", "gpt2.attention.layer_norm_epsilon").
 * Returns -1 / 0.0 if the key is missing or has the wrong type. */
int    tnn_gguf_get_u32(void *handle, const char *key);
double tnn_gguf_get_f32(void *handle, const char *key);
int    tnn_gguf_get_bool(void *handle, const char *key);

/* String metadata: returns NULL if key missing / not a string. The
 * returned pointer is owned by ggml; do not free. */
const char *tnn_gguf_get_str(void *handle, const char *key);

/* Array metadata. n() returns -1 on missing / non-array. arr_str()
 * returns NULL on bounds error or non-string-typed array. */
int         tnn_gguf_arr_n  (void *handle, const char *key);
const char *tnn_gguf_arr_str(void *handle, const char *key, int i);

/* Phase 2 BYO-pointer mmap accessors. Used to wire weight tensors to
 * point at file pages via tnn_session_attach_weight_mmap. */
void  *tnn_gguf_mmap_base(void *handle);
size_t tnn_gguf_mmap_size(void *handle);
size_t tnn_gguf_tensor_file_offset(void *handle, int i);

/* Create a tiny GGUF file at `path` with one 2x3 f32 tensor named
 * "demo.tensor" containing [1.0, 2.0, ..., 6.0]. Useful for the
 * smoke test to exercise the full load path without needing a
 * downloaded model file. Returns 0 on success. */
int    tnn_gguf_write_demo_file(const char *path);

/* Direct GGUF → FFI persistent buffer loaders. Bypass the Ruby Mat /
 * Array<Float64> intermediate so 7B-class models fit in RAM (3× win:
 * 12 B/w → 4 B/w). The session arg is needed only when we have to
 * stage a transpose; for direct memcpy the scratch buffer is unused.
 *
 * All four return 0 on success, negative on error (tensor missing,
 * shape mismatch, dequant failure). The source tensor is dequantized
 * to f32 on read if it isn't already; quantized GGUFs work but pay
 * one dequant per call. */

/* Single non-transposed tensor: token_embd.weight (V × D), and any
 * full-shape weight without per-head splitting. ggml_backend_tensor_set
 * handles large blits internally; no scratch chunking needed. */
int tnn_gguf_copy_to_persistent(void *handle, int tensor_idx,
                                 void *sess, void *target_tensor);

/* Single 1-D tensor: output_norm.weight, attn_norm.weight, ffn_norm.weight. */
int tnn_gguf_copy_1d_to_persistent(void *handle, int tensor_idx,
                                    void *sess, void *target_tensor);

/* Single transposed tensor: attn_output.weight, ffn_gate.weight,
 * ffn_up.weight, ffn_down.weight. Source is (br × bc) row-major;
 * target is the ggml ne=[br, bc] persistent tensor that expects the
 * transposed byte order (matches stage_transposed_and_upload). Chunked
 * by columns so it works for tensors larger than scratch.
 *
 * Convention reminder (same as tnn_upload_transposed_f64): the
 * "transposition" here is just so that ggml_mul_mat(target, x) computes
 * the mathematical (target^T · x) we want; bytes are physically
 * reorganized once, at load time. */
int tnn_gguf_copy_transposed_to_persistent(void *handle, int tensor_idx,
                                            void *sess, void *target_tensor,
                                            int br, int bc);

/* Extract one head-slice from a full Q (n_heads × d_head) or K/V
 * (n_kv × d_head) weight tensor and transpose-upload it to a single
 * per-head persistent buffer. Source layout: row i is
 * [head_0_col_0..d_head-1, head_1_col_0..d_head-1, ..., head_n-1_col_d_head-1]
 * — i.e. heads are concatenated along the column axis. To get head h:
 * src[i, h*d_head : (h+1)*d_head] for i in [0, d_model). Then
 * transpose-write that (d_model × d_head) sub-matrix into target. */
int tnn_gguf_copy_head_slice_to_persistent(void *handle, int tensor_idx,
                                            void *sess, void *target_tensor,
                                            int head_idx, int n_heads_total,
                                            int d_model, int d_head);

/* Native-layout per-head slice. Source has HF-native shape
 * [n_heads_total*d_head, d_model] row-major (i.e. heads concatenated
 * along the leading row axis). Head h is rows [h*d_head, (h+1)*d_head),
 * which is a contiguous (d_head, d_model) row-major block whose bytes
 * are already in ggml ne=[d_model, d_head] column-major order. Just a
 * memcpy — no transpose, no chunking. Used by
 * GGUFLoad.load_kv_cache_directly_native against GGUFs converted with
 * --ggml-native. */
int tnn_gguf_copy_head_slice_to_persistent_native(void *handle, int tensor_idx,
                                                    void *sess, void *target_tensor,
                                                    int head_idx, int n_heads_total,
                                                    int d_model, int d_head);

/* Verbatim (no-dequant) copy: bytes go from the GGUF tensor straight
 * into the persistent ggml tensor. Caller must pre-allocate the
 * destination with the same ggml type as the source. Used by Phase 3
 * (Q8-stays-Q8). The dst type / size must match the GGUF tensor
 * exactly; checked at runtime. */
int tnn_gguf_copy_verbatim_to_persistent(void *handle, int tensor_idx,
                                          void *sess, void *target_tensor);

/* Per-head verbatim slice. Source must be native-layout (head h is
 * rows [h*d_head, (h+1)*d_head) of an [n_heads_total*d_head, d_model]
 * tensor). The destination is one per-head tensor; its ggml_nbytes
 * must equal src_nbytes / n_heads_total. */
int tnn_gguf_copy_verbatim_head_slice_to_persistent(void *handle, int tensor_idx,
                                                      void *sess, void *target_tensor,
                                                      int head_idx, int n_heads_total);

/* Extract one head-slice from a 1-D bias tensor of length
 * (n_heads_total × d_head) and copy it to target_tensor (1-D length
 * d_head). Used for attn_q.bias / attn_k.bias / attn_v.bias in
 * Qwen2.x models. */
int tnn_gguf_copy_head_bias_slice_to_persistent(void *handle, int tensor_idx,
                                                  void *sess, void *target_tensor,
                                                  int head_idx, int d_head);

/* Locate a tensor by name. Returns its index in the GGUF, or -1 if not
 * found. Equivalent to the Ruby find_index helper but with no per-name
 * linear-scan-in-Ruby overhead. */
int tnn_gguf_find_index(void *handle, const char *name);

/* Recursively scan `root` for *.gguf files. Returns a static buffer
 * of newline-separated absolute paths (empty string if none found).
 * Buffer is reused across calls — copy if needed. Used by
 * lib/model_index.rb to find models in standard cache locations
 * (HF hub, Ollama, LM Studio, ./models, ...). */
const char *tnn_list_ggufs(const char *root);

/* File size in bytes via stat(2). Returns 0 if path doesn't exist
 * or stat fails (model files are never 0 bytes, so the ambiguity
 * doesn't bite). */
size_t tnn_file_size(const char *path);

#ifdef __cplusplus
}
#endif

#endif
