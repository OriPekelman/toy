/* ggml#1506 op-level reproducer: ggml_mul_mat_id with K-quant source weights,
 * at OLMoE topology (n_mats=64, n_used=8), on the CPU backend-sched path.
 *
 * NO FFI, NO model file. Synthesizes random expert weights, quantizes them to
 * the K-quant under test, runs mul_mat_id, and compares against the SAME op
 * with F32 (dequantized) weights via NMSE. Threshold 5e-4 matches ggml's own
 * test-backend-ops max_nmse_err for mul_mat_id.
 *
 * Since there are ZERO CPU-backend mul_mat_id changes between our pin
 * (41e7949) and current master (1e33fed), the CPU result here is the master
 * CPU result too.
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N_MATS 64
#define N_USED 8
#define K      2048   /* multiple of 256 (K-quant block) */
#define M      1024
#define N      5      /* tokens */

static unsigned int rng = 99991u;
static float frand(void) { rng = rng*1103515245u + 12345u; return ((float)((rng>>9)&0xFFFF)/32768.0f) - 1.0f; }

static double nmse(const float *a, const float *b, size_t n) {
    double se = 0, s2 = 0;
    for (size_t i = 0; i < n; i++) { double d = (double)a[i]-b[i]; se += d*d; s2 += (double)b[i]*b[i]; }
    return se / (s2 + 1e-12);
}

static int run_type(enum ggml_type qtype, const char *qname,
                    const float *as_f32, const float *bv, const int32_t *idv) {
    struct ggml_init_params ip = { 64*1024*1024, NULL, true /*no_alloc*/ };
    struct ggml_context *ctx = ggml_init(ip);

    struct ggml_tensor *as_q   = ggml_new_tensor_3d(ctx, qtype,          K, M, N_MATS);
    struct ggml_tensor *as_ref = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,  K, M, N_MATS);
    struct ggml_tensor *ids    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32,  N_MATS, N);
    ggml_set_input(as_q); ggml_set_input(as_ref); ggml_set_input(ids);
    struct ggml_tensor *idv2 = ggml_view_2d(ctx, ids, N_USED, N, ids->nb[1], 0);

    struct ggml_tensor *b_q  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N_USED, N);
    struct ggml_tensor *b_r  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, N_USED, N);
    ggml_set_input(b_q); ggml_set_input(b_r);

    struct ggml_tensor *out_q = ggml_mul_mat_id(ctx, as_q,   b_q, idv2);
    struct ggml_tensor *out_r = ggml_mul_mat_id(ctx, as_ref, b_r, idv2);
    ggml_set_output(out_q); ggml_set_output(out_r);

    struct ggml_cgraph *g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, out_q);
    ggml_build_forward_expand(g, out_r);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_t bs[1] = { backend };
    ggml_backend_sched_t s = ggml_backend_sched_new(bs, NULL, 1, ggml_graph_size(g), false, false);
    if (!ggml_backend_sched_alloc_graph(s, g)) { printf("alloc failed\n"); return 2; }

    /* quantize as_f32 -> qtype, row length = K */
    size_t qbytes = ggml_nbytes(as_q);
    void *qbuf = malloc(qbytes);
    ggml_quantize_chunk(qtype, as_f32, qbuf, 0, (int64_t)M*N_MATS, K, NULL);
    ggml_backend_tensor_set(as_q,   qbuf,   0, qbytes);
    ggml_backend_tensor_set(as_ref, as_f32, 0, ggml_nbytes(as_ref));
    ggml_backend_tensor_set(b_q, bv, 0, ggml_nbytes(b_q));
    ggml_backend_tensor_set(b_r, bv, 0, ggml_nbytes(b_r));
    ggml_backend_tensor_set(ids, idv, 0, ggml_nbytes(ids));
    free(qbuf);

    ggml_backend_sched_graph_compute(s, g);

    size_t no = (size_t)M*N_USED*N;
    float *oq = malloc(no*sizeof(float)), *orf = malloc(no*sizeof(float));
    ggml_backend_tensor_get(out_q, oq,  0, no*sizeof(float));
    ggml_backend_tensor_get(out_r, orf, 0, no*sizeof(float));
    double e = nmse(oq, orf, no);
    /* This is a quant-vs-F32 ACCURACY test, so the expected NMSE scales with
     * bit-width (Q8_0 < Q6_K < Q4_K is normal). The original #1506 symptom was
     * GROSS corruption (degenerate output); a kernel that mishandled the K-quant
     * block/index layout would give NMSE ~ O(1e-1..1), not 4-bit quant noise.
     * Flag only gross divergence. */
    int gross = e > 5e-2;
    printf("mul_mat_id  type_a=%-6s  scattered-ids  NMSE=%.3e  -> %s\n",
           qname, e, gross ? "GROSS / CORRUPT" : "within quant noise");
    free(oq); free(orf);
    ggml_backend_sched_free(s); ggml_backend_free(backend); ggml_free(ctx);
    return gross ? 1 : 0;
}

int main(void) {
    printf("ggml#1506 op-level: mul_mat_id K-quant vs F32 reference (CPU sched), pin 41e7949\n");
    size_t nas = (size_t)K*M*N_MATS;
    float *as_f32 = malloc(nas*sizeof(float));
    for (size_t i = 0; i < nas; i++) as_f32[i] = frand();
    size_t nb = (size_t)K*N_USED*N;
    float *bv = malloc(nb*sizeof(float));
    for (size_t i = 0; i < nb; i++) bv[i] = frand();
    /* ids: per token (row), a shuffled permutation of 0..n_mats-1; the view
     * takes the first n_used → scattered experts per token, like the real
     * router and like init_mul_mat_id_tensors in test-backend-ops. */
    int32_t idv[N_MATS*N];
    for (int r = 0; r < N; r++) {
        for (int i = 0; i < N_MATS; i++) idv[r*N_MATS+i] = i;
        for (int i = N_MATS-1; i > 0; i--) {           /* Fisher-Yates */
            rng = rng*1103515245u + 12345u;
            int j = (rng >> 8) % (i+1);
            int32_t t = idv[r*N_MATS+i]; idv[r*N_MATS+i] = idv[r*N_MATS+j]; idv[r*N_MATS+j] = t;
        }
    }

    int rc = 0;
    rc |= run_type(GGML_TYPE_Q8_0, "Q8_0", as_f32, bv, idv);  /* control: known-good */
    rc |= run_type(GGML_TYPE_Q4_K, "Q4_K", as_f32, bv, idv);  /* the suspect */
    rc |= run_type(GGML_TYPE_Q6_K, "Q6_K", as_f32, bv, idv);
    free(as_f32); free(bv);
    printf("%s\n", rc==0 ? "ALL COHERENT" : "CORRUPTION DETECTED");
    return rc;
}
