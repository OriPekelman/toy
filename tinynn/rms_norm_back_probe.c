/*
 * Standalone probe for ggml_rms_norm_back: calls it directly with
 * known inputs (no shim, no Spinel) and prints the output. Compare
 * with the documented formula in ggml-cpu/ops.cpp:
 *
 *   dx[i] = (x[i] * (-sum_xdz/sum_eps) + dz[i]) * rrms
 *
 *   where: sum_xx = Σ x[i]²,  sum_eps = sum_xx + eps*n
 *          mean_eps = sum_xx/n + eps,  rrms = 1/sqrt(mean_eps)
 *          sum_xdz = Σ x[i]*dz[i]
 *
 * For x = [1,0,0,0], dz = [1,0,0,0], n=4, eps=1e-4:
 *   sum_xx = 1, sum_eps = 1.0004
 *   sum_xdz = 1, mean_eps = 0.2501, rrms ≈ 1.99960
 *   dx[0] = (1*(-1/1.0004) + 1) * 1.99960
 *         = (1 - 0.99960) * 1.99960
 *         = 0.0008
 *   dx[1..3] = (0*(-1/sum_eps) + 0) * rrms = 0
 *
 * Build: cc -O0 -g -Ivendor/ggml/include -Ivendor/ggml/src \
 *           rms_norm_back_probe.c \
 *           vendor/ggml/build/src/libggml.a \
 *           vendor/ggml/build/src/ggml-cpu/libggml-cpu.a \
 *           vendor/ggml/build/src/libggml-base.a \
 *           -lstdc++ -lpthread -lgomp -lm -o probe
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    /* Use the legacy single-context API (no_alloc=false) which is the
     * simplest path for a single op. */
    size_t mem_size = 256 * 1024;   /* 256 KB — ggml graph overhead is ~80 KB */
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context *ctx = ggml_init(params);
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); return 1; }

    /* Two 1x4 tensors. ggml_new_tensor_2d takes (ne0, ne1) = (cols, rows). */
    struct ggml_tensor *t_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    struct ggml_tensor *t_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);

    /* a = "gradients" per the impl source comment (src0 = dz = dy) */
    float a_data[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    /* b = "src1 from forward pass" = x */
    float b_data[4] = {1.0f, 0.0f, 0.0f, 0.0f};

    memcpy(t_a->data, a_data, sizeof a_data);
    memcpy(t_b->data, b_data, sizeof b_data);

    const float eps = 1e-4f;
    struct ggml_tensor *t_out = ggml_rms_norm_back(ctx, t_a, t_b, eps);

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    printf("ggml_rms_norm_back(a=%.2f %.2f %.2f %.2f, b=%.2f %.2f %.2f %.2f, eps=%g):\n",
           a_data[0], a_data[1], a_data[2], a_data[3],
           b_data[0], b_data[1], b_data[2], b_data[3], eps);
    float *out = (float *) t_out->data;
    printf("  out = [%.6f, %.6f, %.6f, %.6f]\n", out[0], out[1], out[2], out[3]);
    printf("  expected per impl source formula: [0.000800, 0, 0, 0]\n");

    /* Also try with reversed args (a=x, b=dy) per the header comment, */
    /* in case the source comments are wrong. */
    memcpy(t_a->data, b_data, sizeof b_data);   /* now a holds "x" */
    memcpy(t_b->data, a_data, sizeof a_data);   /* now b holds "dy" */
    /* Rebuild the graph for a fresh compute. */
    ctx = ggml_init(params);
    struct ggml_tensor *t_a2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    struct ggml_tensor *t_b2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    memcpy(t_a2->data, b_data, sizeof b_data);
    memcpy(t_b2->data, a_data, sizeof a_data);
    struct ggml_tensor *t_out2 = ggml_rms_norm_back(ctx, t_a2, t_b2, eps);
    struct ggml_cgraph *gf2 = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf2, t_out2);
    ggml_graph_compute_with_ctx(ctx, gf2, 1);
    float *out2 = (float *) t_out2->data;
    printf("\nReversed (a=x, b=dy per header comment):\n");
    printf("  out = [%.6f, %.6f, %.6f, %.6f]\n", out2[0], out2[1], out2[2], out2[3]);

    return 0;
}
