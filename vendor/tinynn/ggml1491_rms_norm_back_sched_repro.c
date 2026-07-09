/* ggml#1491 refresh: ggml_rms_norm_back differs between the legacy
 * ggml_graph_compute_with_ctx path (correct) and the backend-sched path
 * (wrong), on the same CPU backend + same inputs.
 *
 * Input (matches the original issue): a=[1,0,0,0] (dz), b=[1,0,0,0] (x),
 * eps=1e-4. Formula-correct result: dx = [0.000799, 0, 0, 0].
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <string.h>

static const float A[4] = {1.0f, 0.0f, 0.0f, 0.0f};
static const float B[4] = {1.0f, 0.0f, 0.0f, 0.0f};
static const float EPS  = 1e-4f;

static void legacy(void) {
    struct ggml_init_params p = { 256*1024, NULL, false };
    struct ggml_context *ctx = ggml_init(p);
    struct ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    struct ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    memcpy(a->data, A, sizeof A);
    memcpy(b->data, B, sizeof B);
    struct ggml_tensor *o = ggml_rms_norm_back(ctx, a, b, EPS);
    struct ggml_cgraph *g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, o);
    ggml_graph_compute_with_ctx(ctx, g, 1);
    float *out = (float *) o->data;
    printf("legacy  (compute_with_ctx) : [%g %g %g %g]\n", out[0], out[1], out[2], out[3]);
    ggml_free(ctx);
}

static void sched(void) {
    struct ggml_init_params p = { 256*1024, NULL, true /*no_alloc*/ };
    struct ggml_context *ctx = ggml_init(p);
    struct ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    struct ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
    ggml_set_input(a); ggml_set_input(b);
    struct ggml_tensor *o = ggml_rms_norm_back(ctx, a, b, EPS);
    ggml_set_output(o);
    struct ggml_cgraph *g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, o);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_t bs[1] = { backend };
    ggml_backend_sched_t s = ggml_backend_sched_new(bs, NULL, 1, 64, false, false);
    if (!ggml_backend_sched_alloc_graph(s, g)) { printf("alloc failed\n"); return; }
    ggml_backend_tensor_set(a, A, 0, sizeof A);
    ggml_backend_tensor_set(b, B, 0, sizeof B);
    ggml_backend_sched_graph_compute(s, g);
    float out[4];
    ggml_backend_tensor_get(o, out, 0, sizeof out);
    printf("sched   (backend_sched)    : [%g %g %g %g]\n", out[0], out[1], out[2], out[3]);
    ggml_backend_sched_free(s);
    ggml_backend_free(backend);
    ggml_free(ctx);
}

int main(void) {
    printf("expected (formula)         : [0.000799 0 0 0]\n");
    legacy();
    sched();
    return 0;
}
