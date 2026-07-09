/* gguf_expert_dump.c — dump the MoE expert tensors' type/ne/nb from a GGUF,
 * metadata only (gguf_init_from_file with no_alloc=true). The diagnostic
 * @devYRPauli suggested on ggml#1506: confirm ffn_down_exps is a MIXED quant
 * across layers (q4_K + q6_K) and that ggml's strides satisfy the invariants
 * (nb[1] == ggml_row_size(type, ne0), nb[2] == nb[1]*ne[1]).
 *
 *   cc -O2 -Ivendor/ggml/include -Ivendor/ggml/src tinynn/gguf_expert_dump.c \
 *      vendor/ggml/build/src/libggml.a vendor/ggml/build/src/libggml-cpu.a \
 *      vendor/ggml/build/src/libggml-base.a -lstdc++ -lpthread -lm -o /tmp/gdump
 *   /tmp/gdump data/OLMoE-1b-7b-0924-Instruct-Q4_K_M.gguf
 */
#include "ggml.h"
#include "gguf.h"
#include <stdio.h>
#include <string.h>

static void dump_one(struct ggml_context *mc, const char *name) {
    struct ggml_tensor *t = ggml_get_tensor(mc, name);
    if (!t) { printf("  %-28s (absent)\n", name); return; }
    long rs = (long)ggml_row_size(t->type, t->ne[0]);
    int nb1_ok = ((long)t->nb[1] == rs);
    int nb2_ok = ((long)t->nb[2] == (long)t->nb[1] * t->ne[1]);
    printf("  %-28s %-6s ne=[%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]  nb1_ok=%d nb2_ok=%d  row_size=%ld\n",
           name, ggml_type_name(t->type),
           (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
           t->nb[0], t->nb[1], t->nb[2], t->nb[3], nb1_ok, nb2_ok, rs);
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: gdump <model.gguf>\n"); return 2; }
    struct ggml_context *mc = NULL;
    struct gguf_init_params p; p.no_alloc = true; p.ctx = &mc;
    struct gguf_context *g = gguf_init_from_file(argv[1], p);
    if (!g) { fprintf(stderr, "failed to open %s\n", argv[1]); return 1; }

    printf("=== %s : MoE expert tensors per layer ===\n", argv[1]);
    int gate4 = 0, gate6 = 0, up4 = 0, up6 = 0, down4 = 0, down6 = 0, layers = 0;
    char nm[128];
    for (int li = 0; li < 64; li++) {
        snprintf(nm, sizeof nm, "blk.%d.ffn_down_exps.weight", li);
        struct ggml_tensor *d = ggml_get_tensor(mc, nm);
        if (!d) break;   /* past the last layer */
        layers++;
        printf("layer %d:\n", li);
        char g_nm[128], u_nm[128];
        snprintf(g_nm, sizeof g_nm, "blk.%d.ffn_gate_exps.weight", li);
        snprintf(u_nm, sizeof u_nm, "blk.%d.ffn_up_exps.weight", li);
        dump_one(mc, g_nm); dump_one(mc, u_nm); dump_one(mc, nm);
        struct ggml_tensor *gt = ggml_get_tensor(mc, g_nm);
        struct ggml_tensor *ut = ggml_get_tensor(mc, u_nm);
        if (gt) { if (gt->type == GGML_TYPE_Q6_K) gate6++; else gate4++; }
        if (ut) { if (ut->type == GGML_TYPE_Q6_K) up6++;   else up4++; }
        if (d->type == GGML_TYPE_Q6_K) down6++; else down4++;
    }
    printf("\n=== summary over %d layers ===\n", layers);
    printf("  gate_exps: q4_K-ish=%d  q6_K=%d\n", gate4, gate6);
    printf("  up_exps  : q4_K-ish=%d  q6_K=%d\n", up4, up6);
    printf("  down_exps: q4_K-ish=%d  q6_K=%d   <-- MIXED if both nonzero\n", down4, down6);
    gguf_free(g);
    return 0;
}
