/* gguf_all_types.c — list every tensor name + quant type in a GGUF.
 * Used to diff Q4_K_M vs q8_0 OLMoE: find which tensors (besides the
 * experts) actually change dtype between the two files. */
#include "ggml.h"
#include "gguf.h"
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]); return 2; }
    struct ggml_context *mc = NULL;
    struct gguf_init_params p; p.no_alloc = true; p.ctx = &mc;
    struct gguf_context *g = gguf_init_from_file(argv[1], p);
    if (!g) { fprintf(stderr, "open failed\n"); return 1; }
    int64_t n = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n; i++) {
        const char *name = gguf_get_tensor_name(g, i);
        struct ggml_tensor *t = ggml_get_tensor(mc, name);
        printf("%-40s %s\n", name, ggml_type_name(t->type));
    }
    gguf_free(g);
    return 0;
}
