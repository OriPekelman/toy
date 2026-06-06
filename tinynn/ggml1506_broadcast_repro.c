/* ggml#1506 — the BROADCAST path of mul_mat_id with K-quant experts.
 *
 * build_moe_ffn (toy_smollm2_ffi_kv.rb) calls the gate/up projections as
 *   mul_mat_id(gate_exps, t_h2, top_idx)
 * where t_h2 is a SINGLE token [d_model, 1] — i.e. b->ne1 == 1 while
 * ids->ne0 == K (== n_experts_used). ggml broadcasts that one input to all K
 * routed experts (the `ids->ne0 % b->ne1 == 0` branch).
 *
 * NEITHER our op reproducer NOR ggml's test-backend-ops exercises that branch:
 * both build b with ne1 == n_used (distinct input per expert). So the broadcast
 * branch of mul_mat_id with K-quant source weights is UNTESTED. This isolates it.
 *
 * Source = real OLMoE q4_K ffn_gate_exps bytes (uniform q4_K). Compare the q4_K
 * broadcast result against the F32-dequant broadcast result of the SAME bytes,
 * and against Q8_0 of the same bytes (the known-good control).
 *
 *   cc -O2 -Ivendor/ggml/include -Ivendor/ggml/src tinynn/ggml1506_broadcast_repro.c \
 *      vendor/ggml/build/src/libggml.a vendor/ggml/build/src/libggml-cpu.a \
 *      vendor/ggml/build/src/libggml-base.a -lstdc++ -lpthread -lm -o /tmp/bc
 *   /tmp/bc data/OLMoE-1b-7b-0924-Instruct-Q4_K_M.gguf 0
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define N_USED 8
#define N      5      /* tokens */

static unsigned int rng = 12345u;
static float frand(void) { rng = rng*1103515245u + 12345u; return ((float)((rng>>9)&0xFFFF)/32768.0f) - 1.0f; }
static double nmse(const float *a, const float *b, size_t n) {
    double se = 0, s2 = 0;
    for (size_t i = 0; i < n; i++) { double d = (double)a[i]-b[i]; se += d*d; s2 += (double)b[i]*b[i]; }
    return se / (s2 + 1e-12);
}

/* one run. broadcast=1 -> b->ne1==1 (toy's gate/up path); 0 -> b->ne1==N_USED. */
static void run(enum ggml_type as_type, const void *bytes, int64_t k, int64_t m, int64_t nexp,
                const float *bv, const int32_t *idv, int broadcast, float *out) {
    struct ggml_init_params ip = { 256*1024*1024, NULL, true };
    struct ggml_context *ctx = ggml_init(ip);
    struct ggml_tensor *as = ggml_new_tensor_3d(ctx, as_type, k, m, nexp);
    ggml_set_input(as);
    struct ggml_tensor *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, nexp, N);
    ggml_set_input(ids);
    struct ggml_tensor *idv2 = ggml_view_2d(ctx, ids, N_USED, N, ids->nb[1], 0);
    int64_t bne1 = broadcast ? 1 : N_USED;
    struct ggml_tensor *b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, bne1, N);
    ggml_set_input(b);
    struct ggml_tensor *o = ggml_mul_mat_id(ctx, as, b, idv2);
    ggml_set_output(o);
    struct ggml_cgraph *gr = ggml_new_graph(ctx);
    ggml_build_forward_expand(gr, o);
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_sched_t s = ggml_backend_sched_new(&backend, NULL, 1, ggml_graph_size(gr), false, false);
    ggml_backend_sched_alloc_graph(s, gr);
    ggml_backend_tensor_set(as, bytes, 0, ggml_nbytes(as));
    ggml_backend_tensor_set(b, bv, 0, ggml_nbytes(b));
    ggml_backend_tensor_set(ids, idv, 0, ggml_nbytes(ids));
    ggml_backend_sched_graph_compute(s, gr);
    ggml_backend_tensor_get(o, out, 0, (size_t)m*N_USED*N*sizeof(float));
    ggml_backend_sched_free(s); ggml_backend_free(backend); ggml_free(ctx);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: bc <model.gguf> [layer=0]\n"); return 2; }
    int layer = (argc >= 3) ? atoi(argv[2]) : 0;
    struct ggml_context *mc = NULL;
    struct gguf_init_params gp; gp.no_alloc = true; gp.ctx = &mc;
    struct gguf_context *gg = gguf_init_from_file(argv[1], gp);
    char nm[128]; snprintf(nm, sizeof nm, "blk.%d.ffn_gate_exps.weight", layer);
    int ti = gguf_find_tensor(gg, nm);
    struct ggml_tensor *meta = ggml_get_tensor(mc, nm);
    enum ggml_type qtype = meta->type;
    int64_t k = meta->ne[0], m = meta->ne[1], nexp = meta->ne[2];   /* [d_model, d_ff, n_exp] */
    size_t file_off = gguf_get_data_offset(gg) + gguf_get_tensor_offset(gg, ti);
    printf("=== %s  %s  ne=[%lld,%lld,%lld] ===\n", nm, ggml_type_name(qtype),
           (long long)k,(long long)m,(long long)nexp);

    int fd = open(argv[1], O_RDONLY);
    struct stat st; fstat(fd, &st); size_t fsize = st.st_size;
    void *mbase = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    const void *qsrc = (const char *)mbase + file_off;

    size_t nelem = (size_t)k*m*nexp;
    float *ref = malloc(nelem*sizeof(float));
    ggml_get_type_traits(qtype)->to_float(qsrc, ref, (int64_t)nelem);
    /* also a Q8_0 of the same data — the known-good control */
    size_t q8bytes = ggml_row_size(GGML_TYPE_Q8_0, k) * m * nexp;
    void *q8 = malloc(q8bytes);
    ggml_quantize_chunk(GGML_TYPE_Q8_0, ref, q8, 0, (int64_t)m*nexp, k, NULL);

    size_t nb = (size_t)k*N_USED*N;
    float *bv = malloc(nb*sizeof(float));
    for (size_t i = 0; i < nb; i++) bv[i] = frand();
    int32_t *idv = malloc(sizeof(int32_t)*nexp*N);
    for (int r = 0; r < N; r++) {
        for (int i = 0; i < nexp; i++) idv[r*nexp+i] = i;
        for (int i=(int)nexp-1;i>0;i--){ rng=rng*1103515245u+12345u; int j=(rng>>8)%(i+1);
            int32_t t=idv[r*nexp+i]; idv[r*nexp+i]=idv[r*nexp+j]; idv[r*nexp+j]=t; }
    }

    size_t no = (size_t)m*N_USED*N;
    float *o_qK_bc = malloc(no*4), *o_ref_bc = malloc(no*4), *o_q8_bc = malloc(no*4);
    float *o_qK_nb = malloc(no*4), *o_ref_nb = malloc(no*4);

    /* BROADCAST path (b->ne1==1) — toy's gate/up path */
    run(qtype,           qsrc, k, m, nexp, bv, idv, 1, o_qK_bc);
    run(GGML_TYPE_F32,   ref,  k, m, nexp, bv, idv, 1, o_ref_bc);
    run(GGML_TYPE_Q8_0,  q8,   k, m, nexp, bv, idv, 1, o_q8_bc);
    /* NON-broadcast path (b->ne1==N_USED) — the tested path, control */
    run(qtype,           qsrc, k, m, nexp, bv, idv, 0, o_qK_nb);
    run(GGML_TYPE_F32,   ref,  k, m, nexp, bv, idv, 0, o_ref_nb);

    printf("\n-- BROADCAST path (b->ne1==1, ids->ne0==%d) : toy's gate/up call --\n", N_USED);
    printf("  q-K  vs F32ref  NMSE = %.3e\n", nmse(o_qK_bc, o_ref_bc, no));
    printf("  Q8_0 vs F32ref  NMSE = %.3e   (control)\n", nmse(o_q8_bc, o_ref_bc, no));
    printf("-- NON-broadcast path (b->ne1==%d) : op-test path --\n", N_USED);
    printf("  q-K  vs F32ref  NMSE = %.3e\n", nmse(o_qK_nb, o_ref_nb, no));

    double ebc = nmse(o_qK_bc, o_ref_bc, no);
    printf("\nVERDICT: %s\n", ebc > 5e-2
        ? "BROADCAST path corrupts K-quant -> this is the bug (toy gate/up call shape)."
        : "broadcast path clean for K-quant -> not here.");
    munmap(mbase, fsize); close(fd);
    return 0;
}
