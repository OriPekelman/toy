/*
 * Finite-difference probe for the two vendored GPT-2 backward kernels:
 *   ggml_gelu_back  (GGML_OP_GELU_BACK, autograd of GGML_UNARY_OP_GELU)
 *   ggml_norm_back  (GGML_OP_NORM_BACK, autograd of GGML_OP_NORM)
 *
 * Strategy (per docs/notes/gpt2-backward-patches.md, #1491 caution):
 * the autograd gradient is computed on the REAL backend-sched compute
 * path that training uses (NOT ggml_graph_compute_with_ctx), so the new
 * kernels run where the scheduler actually allocates the grad tensors.
 *
 * For each op we build a scalar loss with a non-trivial upstream grad:
 *     y    = op(x)              (gelu or norm)
 *     loss = sum( y .* r )      (r random => dy = r, not all-ones)
 * and read the autograd grad_x.
 *
 * The finite-difference REFERENCE is computed independently in pure C
 * against the EXACT functions the kernels differentiate:
 *   - gelu: the exact tanh-approx gelu (ggml_gelu_f32). NOTE: ggml's
 *     FORWARD graph uses an f16 *lookup table* (GGML_GELU_FP16), so
 *     differencing the graph forward would measure f16-bucket noise, not
 *     the analytic derivative. The kernel is the derivative of the exact
 *     tanh gelu, so that is what we difference here.
 *   - norm: the exact per-row normalize (matches ggml_norm f32 forward).
 *
 * Build (from repo root):
 *   cc -O2 -g -Ivendor/ggml/include -Ivendor/ggml/src \
 *      tinynn/gpt2_backward_probe.c \
 *      vendor/ggml/build/src/libggml.a \
 *      vendor/ggml/build/src/libggml-cpu.a \
 *      vendor/ggml/build/src/libggml-base.a \
 *      -lstdc++ -lpthread -lm -o /tmp/gpt2_backward_probe
 */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define D 8      /* columns (the normalized / activated dim) */
#define N 3      /* rows */
#define NEL (D*N)

enum probe_op { OP_GELU, OP_NORM, OP_RMS_NORM };

static const float EPS_NORM = 1e-5f;

/* --- exact reference functions (pure C, no ggml) --------------------- */

static const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
static const float GELU_COEF_A    = 0.044715f;

static float gelu_exact(float x) {
    return 0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

/* loss(x) = sum_i r_i * gelu_exact(x_i) */
static float loss_gelu(const float *x, const float *r) {
    float l = 0.0f;
    for (int i = 0; i < NEL; i++) l += r[i]*gelu_exact(x[i]);
    return l;
}

/* loss(x) = sum over rows of sum_i r_i * normalize(x)_i,
 * normalize per row of D with population variance + EPS_NORM inside sqrt. */
static float loss_norm(const float *x, const float *r) {
    float l = 0.0f;
    for (int row = 0; row < N; row++) {
        const float *xr = x + row*D;
        const float *rr = r + row*D;
        float mean = 0.0f;
        for (int i = 0; i < D; i++) mean += xr[i];
        mean /= D;
        float var = 0.0f;
        for (int i = 0; i < D; i++) { float c = xr[i]-mean; var += c*c; }
        var /= D;
        float rstd = 1.0f/sqrtf(var + EPS_NORM);
        for (int i = 0; i < D; i++) l += rr[i]*((xr[i]-mean)*rstd);
    }
    return l;
}

/* loss(x) = sum over rows of sum_i r_i * rmsnorm(x)_i,  rmsnorm per row of D:
 * y_i = x_i / sqrt(mean(x^2) + EPS_NORM).  Cross-check for upstream ggml#1491
 * (rms_norm_back backend-sched vs compute_with_ctx). */
static float loss_rmsnorm(const float *x, const float *r) {
    float l = 0.0f;
    for (int row = 0; row < N; row++) {
        const float *xr = x + row*D;
        const float *rr = r + row*D;
        float ms = 0.0f;
        for (int i = 0; i < D; i++) ms += xr[i]*xr[i];
        ms /= D;
        float rrms = 1.0f/sqrtf(ms + EPS_NORM);
        for (int i = 0; i < D; i++) l += rr[i]*(xr[i]*rrms);
    }
    return l;
}

typedef float (*loss_fn)(const float *, const float *);

/* --- ggml backward graph (the path under test) ----------------------- */

typedef struct {
    struct ggml_context *ctx;
    struct ggml_cgraph  *gb;
    struct ggml_tensor  *x;
    struct ggml_tensor  *r;
    struct ggml_tensor  *loss;
} probe_graph;

static probe_graph build_graph(enum probe_op op) {
    struct ggml_init_params ip = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
        .no_alloc   = true,
    };
    struct ggml_context *ctx = ggml_init(ip);

    struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, N);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    ggml_set_param(x);

    struct ggml_tensor *r = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, N);
    ggml_set_name(r, "r");
    ggml_set_input(r);

    struct ggml_tensor *y;
    switch (op) {
        case OP_GELU:     y = ggml_gelu(ctx, x);            break;
        case OP_NORM:     y = ggml_norm(ctx, x, EPS_NORM);  break;
        default:          y = ggml_rms_norm(ctx, x, EPS_NORM); break;
    }

    struct ggml_tensor *ry   = ggml_mul(ctx, y, r);
    struct ggml_tensor *loss = ggml_sum(ctx, ry);
    ggml_set_name(loss, "loss");
    ggml_set_loss(loss);
    ggml_set_output(loss);

    struct ggml_cgraph *gf = ggml_new_graph_custom(ctx, 2048, false);
    ggml_build_forward_expand(gf, loss);
    struct ggml_cgraph *gb = ggml_graph_dup(ctx, gf, /*force_grads=*/true);
    ggml_build_backward_expand(ctx, gb, NULL);

    probe_graph pg = { ctx, gb, x, r, loss };
    return pg;
}

static int probe(enum probe_op op, const char *name, loss_fn lf) {
    probe_graph pg = build_graph(op);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_t backends[1] = { backend };
    ggml_backend_sched_t sched =
        ggml_backend_sched_new(backends, NULL, 1, 2048, false, false);

    if (!ggml_backend_sched_alloc_graph(sched, pg.gb)) {
        fprintf(stderr, "sched_alloc_graph failed\n"); return 2;
    }

    /* deterministic pseudo-random inputs in [-1,1) */
    float xv[NEL], rv[NEL];
    unsigned int s = 12345u;
    for (int i = 0; i < NEL; i++) {
        s = s*1103515245u + 12345u; xv[i] = ((float)((s >> 9) & 0xFFFF)/32768.0f) - 1.0f;
        s = s*1103515245u + 12345u; rv[i] = ((float)((s >> 9) & 0xFFFF)/32768.0f) - 1.0f;
    }
    ggml_backend_tensor_set(pg.x, xv, 0, sizeof xv);
    ggml_backend_tensor_set(pg.r, rv, 0, sizeof rv);

    /* analytic gradient via autograd on the sched path */
    ggml_graph_reset(pg.gb);              /* zero grads, seed loss grad = 1 */
    if (ggml_backend_sched_graph_compute(sched, pg.gb) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n"); return 2;
    }
    struct ggml_tensor *gx = ggml_graph_get_grad(pg.gb, pg.x);
    if (!gx) { fprintf(stderr, "no grad for x\n"); return 2; }
    float ga[NEL];
    ggml_backend_tensor_get(gx, ga, 0, sizeof ga);

    /* forward-consistency: does the graph's loss match the pure-C reference?
     * (proves the reference loss fn matches ggml's forward; if forward agrees
     * but the gradient does not, the backward kernel is the culprit.) */
    float graph_loss = 0.0f;
    ggml_backend_tensor_get(pg.loss, &graph_loss, 0, sizeof(float));
    float ref_loss = lf(xv, rv);
    printf("%-10s  fwd: graph=% .6f ref=% .6f (|d|=%.2e)\n",
           name, graph_loss, ref_loss, fabsf(graph_loss - ref_loss));

    /* pure-C central finite differences of the exact reference loss.
     * Pass criterion is numpy-allclose style: |fd-ad| <= atol + rtol*|ad|.
     * A bare relative metric is meaningless where the gradient ~ 0; the
     * f32 central-difference rounding floor is ~ eps/h ~ 1e-4 absolute. */
    const float h    = 1e-3f;
    const float ATOL = 1e-3f;
    const float RTOL = 2e-2f;
    float max_abs = 0.0f, worst_margin = -1e30f;
    int worst = -1, fails = 0;
    for (int i = 0; i < NEL; i++) {
        float saved = xv[i];
        xv[i] = saved + h; float lp = lf(xv, rv);
        xv[i] = saved - h; float lm = lf(xv, rv);
        xv[i] = saved;
        float fd  = (lp - lm)/(2.0f*h);
        float ad  = ga[i];
        float abs = fabsf(fd - ad);
        float tol = ATOL + RTOL*fabsf(ad);
        float margin = abs - tol;                 /* >0 means fail */
        if (abs > max_abs) max_abs = abs;
        if (margin > worst_margin) { worst_margin = margin; worst = i; }
        if (margin > 0.0f) fails++;
    }

    int ok = (fails == 0);
    printf("%-10s  max_abs=%.3e  worst_margin=%+.3e  (idx %d)  fails=%d/%d  -> %s\n",
           name, max_abs, worst_margin, worst, fails, NEL, ok ? "PASS" : "FAIL");

    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    ggml_free(pg.ctx);
    return ok ? 0 : 1;
}

int main(void) {
    printf("GPT-2 backward kernels — autograd (backend-sched) vs exact finite difference\n");
    int rc = 0;
    rc |= probe(OP_GELU,     "gelu_back", loss_gelu);
    rc |= probe(OP_NORM,     "norm_back", loss_norm);
    rc |= probe(OP_RMS_NORM, "rms_back",  loss_rmsnorm);  /* upstream ggml#1491 cross-check */
    printf("%s\n", rc == 0 ? "ALL PASS" : "FAILURE");
    return rc;
}
