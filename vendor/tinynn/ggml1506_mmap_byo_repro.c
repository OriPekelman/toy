/* ggml#1506 decisive test: mul_mat_id over the REAL OLMoE q6_K ffn_down_exps
 * bytes, attached three ways, all compared against the SAME-bytes F32 dequant
 * reference. The op-level synthetic reproducer (ggml1506_mul_mat_id_kquant_repro.c)
 * is clean, so the defect — if it's in the experts at all — must be in either the
 * data source (mmap BYO-pointer attach, our loader path) or the real q6_K bytes.
 *
 * Three source tensors, identical bytes, identical b + ids:
 *   A) mmap BYO-pointer q6_K  — exactly tnn_input_3d_persistent_mmap:
 *      ggml_backend_cpu_buffer_from_ptr(file) + ggml_backend_tensor_alloc(addr)
 *   B) copied  q6_K           — sched-allocated, ggml_backend_tensor_set (op-repro path)
 *   R) F32 dequant of A's bytes (ggml_get_type_traits(q6_K).to_float), the reference
 *
 * NMSE(A vs R) and NMSE(B vs R) should both be small q6_K noise (~2e-4). If A is
 * gross but B is fine → the BYO-pointer attach corrupts mul_mat_id reads → that's
 * the toy-side bug. If A == B (both fine) → experts are definitively correct and
 * the corruption is in the MoE GRAPH wiring (hunt build_moe_ffn), ggml is clean.
 *
 *   cc -O2 -Ivendor/ggml/include -Ivendor/ggml/src tinynn/ggml1506_mmap_byo_repro.c \
 *      vendor/ggml/build/src/libggml.a vendor/ggml/build/src/libggml-cpu.a \
 *      vendor/ggml/build/src/libggml-base.a -lstdc++ -lpthread -lm -o /tmp/byo
 *   /tmp/byo data/OLMoE-1b-7b-0924-Instruct-Q4_K_M.gguf 0
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

#define N_USED 8       /* OLMoE top-k */
#define N      5       /* tokens */

static unsigned int rng = 99991u;
static float frand(void) { rng = rng*1103515245u + 12345u; return ((float)((rng>>9)&0xFFFF)/32768.0f) - 1.0f; }

static double nmse(const float *a, const float *b, size_t n) {
    double se = 0, s2 = 0;
    for (size_t i = 0; i < n; i++) { double d = (double)a[i]-b[i]; se += d*d; s2 += (double)b[i]*b[i]; }
    return se / (s2 + 1e-12);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: byo <model.gguf> [layer=0]\n"); return 2; }
    int layer = (argc >= 3) ? atoi(argv[2]) : 0;

    /* --- metadata: data offset, tensor offset/type/shape --- */
    struct ggml_context *mc = NULL;
    struct gguf_init_params gp; gp.no_alloc = true; gp.ctx = &mc;
    struct gguf_context *gg = gguf_init_from_file(argv[1], gp);
    if (!gg) { fprintf(stderr, "open failed\n"); return 1; }

    char nm[128];
    snprintf(nm, sizeof nm, "blk.%d.ffn_down_exps.weight", layer);
    int ti = gguf_find_tensor(gg, nm);
    if (ti < 0) { fprintf(stderr, "no %s\n", nm); return 1; }
    struct ggml_tensor *meta = ggml_get_tensor(mc, nm);
    enum ggml_type qtype = meta->type;
    int64_t ne0 = meta->ne[0], ne1 = meta->ne[1], ne2 = meta->ne[2]; /* [d_ff, d_model, n_exp] */
    size_t data_off = gguf_get_data_offset(gg);
    size_t tens_off = gguf_get_tensor_offset(gg, ti);
    size_t file_off = data_off + tens_off;
    size_t qbytes   = ggml_nbytes(meta);
    printf("=== %s  %s  ne=[%lld,%lld,%lld]  file_off=%zu  nbytes=%zu ===\n",
           nm, ggml_type_name(qtype), (long long)ne0,(long long)ne1,(long long)ne2,
           file_off, qbytes);
    if (qtype != GGML_TYPE_Q6_K && qtype != GGML_TYPE_Q4_K)
        printf("(note: layer %d is %s, not a K-quant; pick a different layer)\n", layer, ggml_type_name(qtype));

    /* --- mmap the whole file (page-aligned base, like the loader) --- */
    int fd = open(argv[1], O_RDONLY);
    struct stat stbuf; fstat(fd, &stbuf);
    size_t fsize = (size_t)stbuf.st_size;
    void *mbase = mmap(NULL, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mbase == MAP_FAILED) { perror("mmap"); return 1; }

    /* the real q6_K bytes for this tensor */
    const void *qsrc = (const char *)mbase + file_off;

    /* --- F32 dequant reference of those same bytes --- */
    const struct ggml_type_traits *tr = ggml_get_type_traits(qtype);
    size_t nelem = (size_t)ne0*ne1*ne2;
    float *ref_f32 = malloc(nelem * sizeof(float));
    /* dequant row by row (to_float works on a full contiguous block here) */
    tr->to_float(qsrc, ref_f32, (int64_t)nelem);

    /* --- shared random b + ids (scattered experts per token) --- */
    size_t nb = (size_t)ne0*N_USED*N;
    float *bv = malloc(nb*sizeof(float));
    for (size_t i = 0; i < nb; i++) bv[i] = frand();
    int32_t *idv = malloc(sizeof(int32_t)*ne2*N);
    for (int r = 0; r < N; r++) {
        for (int i = 0; i < ne2; i++) idv[r*ne2+i] = i;
        for (int i = (int)ne2-1; i > 0; i--) {
            rng = rng*1103515245u + 12345u;
            int j = (rng >> 8) % (i+1);
            int32_t t = idv[r*ne2+i]; idv[r*ne2+i] = idv[r*ne2+j]; idv[r*ne2+j] = t;
        }
    }

    size_t no = (size_t)ne1*N_USED*N;
    float *out[3]; for (int i=0;i<3;i++) out[i]=malloc(no*sizeof(float));
    const char *label[3] = { "A mmap-BYO q-K ", "B copied  q-K  ", "R F32 dequant  " };

    ggml_backend_t backend = ggml_backend_cpu_init();

    /* run one of the three paths; path: 0=BYO mmap, 1=copied quant, 2=F32 ref */
    for (int path = 0; path < 3; path++) {
        struct ggml_init_params ip = { 64*1024*1024, NULL, true /*no_alloc*/ };
        struct ggml_context *ctx = ggml_init(ip);

        enum ggml_type as_type = (path == 2) ? GGML_TYPE_F32 : qtype;
        struct ggml_tensor *as = ggml_new_tensor_3d(ctx, as_type, ne0, ne1, ne2);

        struct ggml_tensor *ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, ne2, N);
        ggml_set_input(ids);
        struct ggml_tensor *idv2 = ggml_view_2d(ctx, ids, N_USED, N, ids->nb[1], 0);
        struct ggml_tensor *b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, N_USED, N);
        ggml_set_input(b);

        struct ggml_tensor *o = ggml_mul_mat_id(ctx, as, b, idv2);
        ggml_set_output(o);

        struct ggml_cgraph *gr = ggml_new_graph(ctx);
        ggml_build_forward_expand(gr, o);

        /* Path A: attach `as` to the mmap region BEFORE sched alloc, exactly
         * like tnn_input_3d_persistent_mmap. The other tensors stay sched-managed. */
        ggml_backend_buffer_t mmapbuf = NULL;
        if (path == 0) {
            mmapbuf = ggml_backend_cpu_buffer_from_ptr(mbase, fsize);
            void *addr = (char *)ggml_backend_buffer_get_base(mmapbuf) + file_off;
            if (ggml_backend_tensor_alloc(mmapbuf, as, addr) != GGML_STATUS_SUCCESS) {
                printf("tensor_alloc failed\n"); return 3;
            }
        } else {
            ggml_set_input(as);
        }

        ggml_backend_sched_t s = ggml_backend_sched_new(&backend, NULL, 1, ggml_graph_size(gr), false, false);
        if (!ggml_backend_sched_alloc_graph(s, gr)) { printf("alloc failed\n"); return 2; }

        if (path == 1) ggml_backend_tensor_set(as, qsrc,    0, qbytes);          /* copied quant */
        if (path == 2) ggml_backend_tensor_set(as, ref_f32, 0, nelem*sizeof(float)); /* F32 ref */
        ggml_backend_tensor_set(b,   bv,  0, ggml_nbytes(b));
        ggml_backend_tensor_set(ids, idv, 0, ggml_nbytes(ids));

        ggml_backend_sched_graph_compute(s, gr);
        ggml_backend_tensor_get(o, out[path], 0, no*sizeof(float));

        ggml_backend_sched_free(s);
        if (mmapbuf) ggml_backend_buffer_free(mmapbuf);
        ggml_free(ctx);
    }

    double eA = nmse(out[0], out[2], no);   /* mmap-BYO vs F32 ref */
    double eB = nmse(out[1], out[2], no);   /* copied   vs F32 ref */
    double eAB= nmse(out[0], out[1], no);   /* mmap-BYO vs copied  */
    printf("\n%s  NMSE(vs F32 ref) = %.3e\n", label[0], eA);
    printf("%s  NMSE(vs F32 ref) = %.3e\n", label[1], eB);
    printf("mmap-BYO vs copied-quant      NMSE = %.3e  (should be ~0: same bytes, same op)\n", eAB);

    int grossA = eA > 5e-2, grossB = eB > 5e-2;
    printf("\nVERDICT: ");
    if (grossA && !grossB)      printf("BYO-POINTER PATH IS THE BUG (mmap attach corrupts the op).\n");
    else if (!grossA && !grossB) printf("EXPERTS FINE both ways -> corruption is GRAPH-side (hunt build_moe_ffn).\n");
    else if (grossA && grossB)  printf("BOTH q-K paths corrupt vs F32 -> real op/data bug with these bytes.\n");
    else                        printf("copied corrupt but BYO fine -> unexpected; investigate.\n");

    munmap(mbase, fsize); close(fd);
    return 0;
}
