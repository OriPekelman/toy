/* gguf_requant_q4k.c — requantize every q8_0 2-D weight tensor in a GGUF to
 * Q4_K (rows whose ne0 % 256 == 0), copying all metadata and other tensors
 * verbatim. Purpose: produce a NON-MoE K-quant model to test whether toy's
 * end-to-end K-quant inference path is correct (ggml#1506 isolation).
 *
 *   cc -O2 -Ivendor/ggml/include -Ivendor/ggml/src tinynn/gguf_requant_q4k.c \
 *      vendor/ggml/build/src/libggml.a vendor/ggml/build/src/libggml-cpu.a \
 *      vendor/ggml/build/src/libggml-base.a -lstdc++ -lpthread -lm -o /tmp/rq
 *   /tmp/rq data/tinyllama-1.1b-q8_0.gguf /tmp/tinyllama-q4k.gguf
 */
#include "ggml.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: rq <in.gguf> <out.gguf>\n"); return 2; }
    struct ggml_context *src_ctx = NULL;
    struct gguf_init_params p; p.no_alloc = false; p.ctx = &src_ctx;
    struct gguf_context *src = gguf_init_from_file(argv[1], p);
    if (!src) { fprintf(stderr, "open failed\n"); return 1; }

    struct gguf_context *out = gguf_init_empty();
    gguf_set_kv(out, src);   /* copy ALL key/value metadata verbatim */

    /* a scratch ggml ctx to hold the requantized tensors */
    size_t overhead = ggml_tensor_overhead() * (gguf_get_n_tensors(src) + 16);
    struct ggml_init_params ip = { overhead, NULL, true /*no_alloc: we set data ptrs*/ };
    struct ggml_context *work = ggml_init(ip);

    int64_t n = gguf_get_n_tensors(src);
    int requant = 0, kept = 0;
    for (int64_t i = 0; i < n; i++) {
        const char *name = gguf_get_tensor_name(src, i);
        struct ggml_tensor *t = ggml_get_tensor(src_ctx, name);
        int do_q4k = (t->type == GGML_TYPE_Q8_0) && (t->ne[0] % 256 == 0) && (t->ne[1] >= 1);
        if (do_q4k) {
            int64_t k = t->ne[0], rows = t->ne[1]*t->ne[2]*t->ne[3];
            float *f32 = malloc((size_t)k*rows*sizeof(float));
            const struct ggml_type_traits *tr = ggml_get_type_traits(t->type);
            tr->to_float(t->data, f32, (int64_t)k*rows);
            size_t q4bytes = ggml_row_size(GGML_TYPE_Q4_K, k) * rows;
            void *q4 = malloc(q4bytes);
            ggml_quantize_chunk(GGML_TYPE_Q4_K, f32, q4, 0, rows, k, NULL);
            free(f32);
            struct ggml_tensor *nt = ggml_new_tensor(work, GGML_TYPE_Q4_K, GGML_MAX_DIMS, t->ne);
            ggml_set_name(nt, name);
            nt->data = q4;   /* gguf_add_tensor reads ->data at write time */
            gguf_add_tensor(out, nt);
            requant++;
        } else {
            gguf_add_tensor(out, t);   /* verbatim (data ptr from src_ctx) */
            kept++;
        }
    }
    printf("requantized %d tensors to Q4_K, kept %d verbatim\n", requant, kept);
    if (!gguf_write_to_file(out, argv[2], false)) { fprintf(stderr, "write failed\n"); return 1; }
    printf("wrote %s\n", argv[2]);
    return 0;
}
