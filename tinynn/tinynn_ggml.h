#ifndef TINYNN_GGML_H
#define TINYNN_GGML_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Spinel-friendly C shim over ggml. Backend-aware (CPU now; CUDA when
 * compiled with -DTINYNN_HAVE_CUDA against the ggml-cuda build).
 *
 * Tensor lifetime: tied to the session. tnn_session_free disposes all
 * tensors created via the session.
 *
 * Result-shape convention for matmul: ggml's mul_mat result is
 * (ne0=m, ne1=n) where m = rows of A and n = rows of B (A,B given as
 * ne0=k, ne1=rows). Reading the logical (m,n) row-major result: index
 * scratch[j*m + i] after tnn_tensor_pull. The Ruby side does the swap.
 *
 * Scratch buffer is 16 MiB (4 M f32 slots). Tensors larger than that
 * MUST use the chunked uploaders (`tnn_upload_from_float_array`,
 * `tnn_upload_transposed_f64`) — calling tnn_upload directly on an
 * oversized tensor is bounds-checked now (was silent UB) but the call
 * still fails.
 */

/* TNN_ASSERT: debug-mode assert. In NDEBUG builds it compiles away.
 * Use at "shouldn't happen" boundaries (post-invariant checks, FFI
 * shape preconditions). Cheap. Pays for itself the first time it
 * catches a silent regression — the silent-failure family of bugs is
 * what makes this codebase hard to debug otherwise. */
#ifdef NDEBUG
#define TNN_ASSERT(cond, msg) ((void)0)
#else
#define TNN_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "[tnn] TNN_ASSERT failed: %s\n  at %s:%d (%s)\n", \
                (msg), __FILE__, __LINE__, #cond); \
        abort(); \
    } \
} while (0)
#endif

void  *tnn_session_new(int backend_kind);    /* 0 = CPU, 1 = CUDA, 2 = Metal — falls
                                              * back to CPU if the requested GPU
                                              * backend archive isn't linked in */
void   tnn_session_free(void *sess);

/* Idempotent teardown of cached engines (CPU + CUDA + Metal). Call
 * before main returns when targeting Metal — its static destructor
 * asserts on undrained residency sets otherwise. Safe to call on
 * any backend; no-op when caches are empty. */
void   tnn_shutdown_engines(void);

const char *tnn_backend_name(void *sess);
int    tnn_link_check(void);                 /* returns 73 */

/* Build phase: declare tensors and ops, then realize.
 * Tensors are created before tnn_realize; backend storage is allocated then. */
void  *tnn_input_2d_f32(void *sess, int rows, int cols);

/* Persistent F32 inputs. Created in ctx_w; backend buffer allocated by
 * tnn_finalize_weights. Data uploaded to these tensors survives across
 * sched_reset cycles — use for parameters, optimizer moments, anything
 * that should live on the device between training steps. */
void  *tnn_input_2d_f32_persistent(void *sess, int rows, int cols);
/* Same shape as the f32 persistent allocator but with a caller-chosen
 * ggml type (e.g. GGML_TYPE_Q8_0 = 8). Used by Phase 3 of the
 * memory-design plan: keep quantized weights quantized in memory. */
void  *tnn_input_2d_persistent_typed(void *sess, int rows, int cols, int ggml_type);

/* Row size in bytes for a given ggml type at the given element count
 * (ne0). For F32 this is ne0 * 4. For Q8_0 (block 32 × 34 bytes) at
 * d_head=64 this is 68. Used by callers that build view_2d offsets
 * for typed tensors — F32 byte stride and Q8 byte stride differ. */
long   tnn_row_size(int ggml_type, int ne0);
void  *tnn_input_1d_f32_persistent(void *sess, int n);
void  *tnn_input_3d_f32_persistent(void *sess, int ne0, int ne1, int ne2);  /* M2 MoE */
void  *tnn_input_3d_persistent_typed(void *sess, int ne0, int ne1, int ne2,
                                       int ggml_type);                         /* M2.3 MoE */

/* Phase 2 BYO-pointer: attach an mmap'd file region to the session.
 * Subsequent calls to tnn_input_*_persistent_mmap allocate tensors
 * whose `data` lives at byte offsets inside this region. The session
 * does NOT own the memory — caller (typically a tnn_gguf_session) is
 * responsible for keeping it valid for the session's lifetime.
 * Returns 0 on success. */
int    tnn_session_attach_weight_mmap(void *sess, void *base, size_t size);
void  *tnn_input_2d_persistent_mmap(void *sess, int rows, int cols,
                                     int ggml_type, size_t buf_offset);
void  *tnn_input_3d_persistent_mmap(void *sess, int ne0, int ne1, int ne2,
                                     int ggml_type, size_t buf_offset);
void  *tnn_input_1d_persistent_mmap(void *sess, int n, int ggml_type,
                                     size_t buf_offset);

/* Allocate the backend buffer for all persistent tensors. Call once,
 * after declaring persistent tensors and before any tnn_realize. */
int    tnn_finalize_weights(void *sess);

/* Zero an entire persistent tensor's backend buffer (CPU memset /
 * cudaMemsetAsync). Use for large Adam-state init where building a
 * Mat-of-zeros wastes RAM. */
int    tnn_zero_tensor(void *sess, void *tensor);

/* Build a SECONDARY graph (graph_b) sharing the session's ctx and
 * tensors. Used for adam_step / update passes that mutate persistent
 * weights between forward calls. Does NOT alloc -- call tnn_switch_b
 * before tnn_compute_b each iteration. */
int    tnn_realize_b(void *sess, void *result);

/* Reset sched and alloc the requested graph for the next compute.
 * Persistent tensors keep their stable buffer; compute tensors get
 * fresh slots (caller must re-upload compute inputs after switch). */
int    tnn_switch_a(void *sess);
int    tnn_switch_b(void *sess);

/* Compute the secondary graph (graph_b). Must be preceded by tnn_switch_b. */
int    tnn_compute_b(void *sess);
void  *tnn_matmul(void *sess, void *a, void *b);        /* A * B^T (ggml-native) */

/* --- M2: MoE primitives --------------------------------------------------- */
/* Sparse expert matmul. `as` is a 3-D weight tensor stacked across experts
 * (ne = [d_in, d_out, n_experts]). `b` is the input [d_in, T]. `ids` is
 * the per-token expert-index tensor (int32, [n_used, T]). Returns a tensor
 * of shape [d_out, n_used, T] — one matmul per selected expert per token. */
void  *tnn_mul_mat_id(void *sess, void *as, void *b, void *ids);

/* Gather-add: dst[i0, i1, i2] = a[i0, i1, i2] + b[i0, ids[i1, i2]].
 * Used to fold per-expert biases into routed activations. */
void  *tnn_add_id(void *sess, void *a, void *b, void *ids);

/* argsort along ne[0]. order=0 ascending, 1 descending. Returns int32 index
 * tensor of the same shape as input. */
void  *tnn_argsort(void *sess, void *a, int descending);

/* Top-K indices along ne[0]. Implemented in ggml as argsort + view. Result
 * is int32, shape [k, ...]. Order: descending values, but the k indices
 * themselves are not internally sorted (per ggml.h's note). */
void  *tnn_top_k(void *sess, void *a, int k);
void  *tnn_matmul_axb(void *sess, void *a, void *b);    /* A * B  (transposes B internally) */
void  *tnn_add(void *sess, void *a, void *b);           /* element-wise A + B (same shape) */
void  *tnn_gelu(void *sess, void *a);                   /* element-wise GeLU (tanh approx) */
void  *tnn_rms_norm(void *sess, void *x, void *gamma_row, double eps);
                                                         /* RMSNorm(x) * gamma_row, last-dim normalize, broadcast over the leading dim.
                                                            x: (n1, n0) with ne0=feature, ne1=batch_or_T
                                                            gamma_row: (1, n0) — a 1xfeature tensor */
void  *tnn_layer_norm(void *sess, void *x, void *gamma_row, void *beta_row, double eps);
                                                         /* LayerNorm: y = gamma * (x-mean)/sqrt(var+eps) + beta.
                                                            For HF-style models (GPT-2 / GPT-Neo / TinyStories). */
void  *tnn_softmax(void *sess, void *a);                /* per-row softmax along ne[0] */
void  *tnn_diag_mask_inf(void *sess, void *a, int n_past);

/* Flash attention forward (ggml_flash_attn_ext). P4.
 *
 * Fuses softmax(Q · Kᵀ / √d_head + mask) · V into one streaming kernel.
 * Significant win on long contexts (memory-bandwidth dominated) and on
 * the CUDA backend (purpose-built kernel). Drop-in replacement for the
 * scale → softmax → matmul triplet that builds attention output.
 *
 * Shapes (ggml ne[0..3]):
 *   q    [d_head, T_q, n_head,    batch]
 *   k    [d_head, T_k, n_head_kv, batch]
 *   v    [d_head, T_k, n_head_kv, batch]
 *   mask [T_k,    T_q, 1,         1    ]  type=F16, contiguous, optional (pass NULL)
 *   out  [d_head, n_head, T_q, batch]
 *
 * scale         : 1/sqrt(d_head) typically.
 * max_bias      : 0.0 disables ALiBi; >0 requires non-NULL mask.
 * logit_softcap : 0.0 disables; >0 applies tanh(x / softcap) * softcap
 *                 (Gemma 2 / Grok / Olmo). Cap NaN safe in ggml's impl.
 *
 * GQA (n_head_kv < n_head) is supported — ggml broadcasts K/V across
 * head groups internally.
 *
 * Backward is NOT supported in current vendored ggml
 * (ggml_flash_attn_back has GGML_ABORT). Use this for INFERENCE only;
 * training stays on the existing scale/softmax/matmul triplet. */
void  *tnn_flash_attn_ext(void *sess, void *q, void *k, void *v, void *mask,
                            double scale, double max_bias, double logit_softcap);

                                                         /* set elements above the diagonal (off by n_past) to -inf */

/* --- Llama-family ops --- */
void  *tnn_silu(void *sess, void *a);                    /* SiLU: x * sigmoid(x). SwiGLU activation. */
void  *tnn_mul (void *sess, void *a, void *b);           /* elementwise multiply c = a * b */
void  *tnn_rope_ext(void *sess, void *a, void *pos, int n_dims,
                    double freq_base, double freq_scale,
                    double ext_factor, double attn_factor,
                    double beta_fast, double beta_slow,
                    void *freq_factors);
                                                         /* RoPE (NEOX/rotate_half mode); pos is int32[T].
                                                          * freq_factors is a (n_dims/2) f32 tensor or NULL.
                                                          * For linear/dynamic: freq_scale = 1/factor, others default.
                                                          * For YaRN: all six scalars matter.
                                                          * For llama3-style: pass freq_factors built via
                                                          * tnn_rope_freq_factors_llama3. */
void  *tnn_rope_freq_factors_alloc(void *sess, int n_dims);
                                                         /* Allocate a (n_dims/2)-elem F32 persistent tensor
                                                          * in ctx_w. Ruby fills it via the standard upload
                                                          * path. Math for llama3 lives in
                                                          * Toy::RopeScaling.compute_llama3_freq_factors. */
void  *tnn_input_1d_i32_ctx(void *sess, int n);          /* int32 vector in session ctx (positions for RoPE) */
void  *tnn_concat(void *sess, void *a, void *b, int dim);
                                                         /* concat a and b along the given ne axis */
void  *tnn_null_ptr(void);                              /* :ptr-typed NULL seed for Spinel PtrArray inference */

/* KV-cache primitives. view_1d / view_2d / cpy let us slice into a
 * persistent (max_T, d_head) buffer and write a single row at the
 * current decode position. Offsets are baked in at graph build time;
 * to handle a runtime position, the caller rebuilds the decode graph
 * per step (cheap; just metadata). */
void  *tnn_view_1d(void *sess, void *a, int ne0, long offset);
void  *tnn_view_2d(void *sess, void *a, int ne0, int ne1, long nb1, long offset);
void  *tnn_cpy(void *sess, void *a, void *b);

/* Reshape a contiguous tensor to a new shape with the same total
 * element count. Used by the M3 sequence-mode forward to lift
 * (d_head, T) → (d_head, 1, T) for rope_ext, then back to (d_head, T)
 * for the downstream matmul. */
void  *tnn_reshape_3d(void *sess, void *a, int ne0, int ne1, int ne2);
void  *tnn_reshape_2d(void *sess, void *a, int ne0, int ne1);

/* set_rows + soft_max_ext: the canonical KV-cache attention primitives.
 * set_rows writes the new k/v row at a RUNTIME index, so the graph is
 * static across decode steps. soft_max_ext applies an additive mask
 * (typically -inf for keys > current pos) before softmax. */
void  *tnn_set_rows(void *sess, void *a, void *b, void *idx);
void  *tnn_soft_max_ext(void *sess, void *a, void *mask, double scale, double max_bias);
void  *tnn_set_2d(void *sess, void *a, void *b, long nb1, long offset);
void  *tnn_transpose(void *sess, void *a);              /* materialised transpose: (rows,cols) → (cols,rows) */
void  *tnn_scale(void *sess, void *a, double s);        /* element-wise a * s */

/* Backward ops. */
void  *tnn_rms_norm_back(void *sess, void *x, void *dy, double eps);

/* Phase F0 — backward ops for fine-tuning. Each one mirrors the
 * forward op's shape: same context, same tensor handles, returns a
 * new compute tensor representing dx. */
void  *tnn_silu_back(void *sess, void *x, void *dy);
void  *tnn_rope_ext_back(void *sess, void *dy, void *pos, int n_dims,
                         double freq_base, double freq_scale,
                         double ext_factor, double attn_factor,
                         double beta_fast, double beta_slow,
                         void *freq_factors);

/* Phase F0.4 — autograd via ggml_build_backward_expand. Mark loss +
 * params, build the forward graph (tnn_realize), then call build_backward
 * → compute_backward → tensor_grad to retrieve gradients. */
void  *tnn_sum            (void *sess, void *a);
/* Per-row sum along ne[0]. Input ne=[a,b,c,d] → output ne=[1,b,c,d].
 * Useful for collapsing the K axis of MoE outputs after a transpose
 * (since ggml has no axis-1 sum primitive). */
void  *tnn_sum_rows       (void *sess, void *a);
/* CE loss wrapper. b is the labels probability distribution
 * (one-hot for hard targets, label-smoothed otherwise). Returns a
 * scalar loss tensor. */
void  *tnn_cross_entropy_loss(void *sess, void *a, void *b);

/* Diagnostic — pin every node in graph_b as output (no sched reuse).
 * See task70-bisect doc for usage notes. */
int    tnn_pin_all_graph_b_nodes(void *sess);

/* AdamW-friendly graph reset — zero grads but preserve momenta.
 * See ggml_graph_reset for the baseline behaviour we differ from. */
int    tnn_graph_reset_grads_only(void *sess);
void   tnn_set_loss       (void *tensor);
int    tnn_build_backward (void *sess);                 /* dup + build_backward_expand, no alloc */
int    tnn_extend_backward_graph(void *sess, void *node); /* add node (typically opt_step output) */
int    tnn_realize_backward(void *sess);                /* sched-alloc the final backward graph */
int    tnn_graph_reset    (void *sess);                 /* zero grads + momenta; seed loss grad = 1 */
int    tnn_compute_backward(void *sess);                /* runs fwd + bwd + (opt) in one call */
void  *tnn_tensor_grad    (void *sess, void *tensor);
                                                         /* d/dx of plain RMSNorm(x) (no gamma); caller handles gamma chain. */
void  *tnn_softmax_back(void *sess, void *a, void *dy); /* d/dx of per-row softmax. a is softmax output. */
void  *tnn_get_rows(void *sess, void *table, void *idx);
                                                         /* gather rows: out[i] = table[idx[i]] */
void  *tnn_get_rows_back(void *sess, void *d_out, void *idx, void *table_shape);
                                                         /* scatter-add: out has table's shape, out[idx[i]] += d_out[i] */
void  *tnn_input_1d_i32(void *sess, int n);             /* int32 vector input (for row indices). */

/* Custom non-ggml kernel: dx = dh * d/dx GeLU(x) (tanh approximation).
 * Reads x from scratch[0..n) and dh from scratch[n..2n); writes
 * dx into scratch[2n..3n). Caller is responsible for staging and
 * reading back. CPU-only (no GPU path yet).
 *
 * No new tensor created; this is a side-channel op that skips the
 * ggml graph entirely. Useful because ggml has no gelu_back op.
 */
void   tnn_gelu_back_scratch(void *sess, int n);

/* Adam optimizer step (matches the project's adam_step_mat).
 * Scratch layout: [0..n) param, [n..2n) grad, [2n..3n) m, [3n..4n) v.
 * Updates all four in place:
 *   m_new = b1*m + (1-b1)*g
 *   v_new = b2*v + (1-b2)*g*g
 *   m_hat = m_new / omc1, v_hat = v_new / omc2
 *   param -= lr * m_hat / (sqrt(v_hat) + eps)
 * After return, read back scratch[0..n) (param), [2n..3n) (m), [3n..4n) (v).
 */
void   tnn_adam_step_scratch(void *sess, int n,
                              double lr, double b1, double b2, double eps,
                              double omc1, double omc2);

/* Realize the graph (allocates all tensors on the backend). Must be
 * called once after all ops are declared and before any upload. */
int    tnn_realize(void *sess, void *result);

/* Forward-graph build for training callers: appends nodes via
 * ggml_build_forward_expand but does NOT sched-alloc. The follow-up
 * tnn_build_backward + tnn_realize_backward does the single sched-alloc
 * on the combined graph_b. Calling tnn_realize before tnn_realize_backward
 * is broken: the sched_reset between the two leaves tensor buffer pointers
 * stale (the second alloc lands on freed pool memory). Inference callers
 * keep using tnn_realize; training callers use this. */
int    tnn_build_forward_only(void *sess, void *result);

/* Add a tensor's compute tree to the graph before tnn_realize. Used
 * for side-effect ops (ggml_cpy into a persistent view) that aren't
 * reachable from the realize target and would otherwise be pruned. */
int    tnn_add_to_graph(void *sess, void *tensor);

/* Reset the session for a fresh graph build. The persistent ctx_w
 * tensors keep their data; the compute graph and its node tensors
 * are replaced. Use for per-step decode where op offsets (e.g. V
 * set_2d's pos*sizeof(float)) need to change between steps. */
int    tnn_reset_for_rebuild(void *sess);

/* Mark a tensor as a graph output so the backend scheduler keeps its
 * buffer alive after compute (otherwise intermediate buffers may be
 * reused as scratch by later ops, in particular ggml_gelu can reuse
 * its input tensor's allocation). Call BEFORE tnn_realize. */
void   tnn_set_output(void *tensor);

/* Mark a tensor as a trainable parameter. Required by
 * ggml_opt_step_adamw on its weight argument. Call BEFORE tnn_realize. */
void   tnn_set_param(void *tensor);

/* 1-D F32 input tensor (length n). Used for the 7-element
 * adamw_params vector (alpha, b1, b2, eps, wd, beta1h, beta2h). */
void  *tnn_input_1d_f32(void *sess, int n);

/* In-place AdamW step. After compute:
 *   m = m*b1 + g*(1-b1)
 *   v = v*b2 + g*g*(1-b2)
 *   a = a*(1 - alpha*wd) - alpha * (m*beta1h) / (sqrt(v*beta2h) + eps)
 *
 * Matches the project's plain-Adam with wd=0 since `keep = 1 - alpha*0 = 1`
 * and beta1h = 1 / (1 - b1^t), beta2h = 1 / (1 - b2^t) -- so
 * mh = m * beta1h = m_hat, vh = sqrt(v * beta2h) + eps = sqrt(v_hat) + eps. */
void  *tnn_opt_step_adamw(void *sess, void *a, void *grad, void *m, void *v, void *params);
void  *tnn_opt_step_sgd  (void *sess, void *a, void *grad, void *params);

/* Compute the (already-built) graph. Must be called after upload. */
int    tnn_compute(void *sess);

/* Bulk upload/download via a session-owned scratch buffer (16 MiB).
 *   - tnn_scratch_set(sess, idx, value) fills scratch element by element
 *     (per-element FFI call; cheap on the Ruby side).
 *   - tnn_upload(sess, tensor) bulk-copies the staged data into the
 *     backend buffer (single backend_tensor_set => one cudaMemcpy).
 *   - tnn_download(sess, tensor) pulls in the other direction.
 *   - tnn_scratch_get reads from the host shadow.
 */
void   tnn_scratch_set(void *sess, int idx, double v);
double tnn_scratch_get(void *sess, int idx);
void   tnn_scratch_set_i32(void *sess, int idx, int value);
int    tnn_scratch_get_i32(void *sess, int idx);
int    tnn_upload(void *sess, void *tensor);
int    tnn_download(void *sess, void *tensor);

/* Bulk f64 → tensor upload using Spinel's :float_array spec (matz/spinel#474).
 * `data` is a `const double *` straight from an Array<Float>'s storage; we
 * convert to f32 and bulk-copy to the backend buffer in one go. Replaces
 * the per-element tnn_scratch_set loop for row-major uploads. */
int    tnn_upload_from_float_array(void *sess, void *tensor, const double *data, size_t n);

/* Transpose-and-upload a row-major f64 (br × bc) Mat into a ggml f32
 * tensor. Chunked so it works for tensors larger than scratch
 * (the per-element scratch_set + bulk-upload path silently truncated
 * at 4M floats — Qwen's ffn_gate is 4.36M). */
int    tnn_upload_transposed_f64(void *sess, void *tensor,
                                 const double *src, int br, int bc);

/* Bulk i64 → tensor upload, for row-index tensors (embedding lookup).
 * The :int_array spec gives us `const int64_t *`; we narrow to int32 (which
 * is what ggml's GGML_TYPE_I32 expects) during the copy. */
int    tnn_upload_from_int_array(void *sess, void *tensor, const long *data, size_t n);

/* Inverse of tnn_upload_from_float_array. Pulls a tensor's f32 contents
 * back to a host f64 buffer (the Ruby Mat's flat array) in scratch-sized
 * chunks — so it works for tensors larger than the 16 MiB scratch.
 * Enables Mat-roundtrip on weights loaded via the direct GGUF→FFI path
 * without growing scratch for everyone.
 *
 * Return values:
 *   0   ok
 *  -1   null arg
 *  -2   n exceeds tensor element count */
int    tnn_download_to_f64_array(void *sess, void *tensor, double *dst, size_t n);

int    tnn_tensor_ne0(void *t);
int    tnn_tensor_ne1(void *t);
size_t tnn_tensor_nbytes(void *t);
int    tnn_tensor_nelements(void *t);

/* Stats over the first `n` floats of sess->scratch. Caller has just
 * done tnn_download(sess, t). Used by FFI trace-tap diagnostics. */
double tnn_scratch_min_f32(void *sess, int n);
double tnn_scratch_max_f32(void *sess, int n);
double tnn_scratch_sum_abs_f32(void *sess, int n);
int    tnn_scratch_nan_count_f32(void *sess, int n);

#ifdef __cplusplus
}
#endif

#endif
