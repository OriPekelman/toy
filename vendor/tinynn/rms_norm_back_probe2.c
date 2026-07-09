/*
 * Probe that goes through tinynn_ggml.c's session API (the same path
 * the Spinel-emitted Ruby uses) and calls ggml_rms_norm_back. Compare
 * with the result the legacy compute_with_ctx path gave us:
 *
 *   x=[1,0,0,0], dy=[1,0,0,0], eps=1e-4 → expected ≈ [8e-4, 0, 0, 0]
 *
 * Build:
 *   cc -O0 -g -DTINYNN_DEBUG -Ivendor/ggml/include -Ivendor/ggml/src \
 *      tinynn/rms_norm_back_probe2.c tinynn/tinynn_ggml.c \
 *      vendor/ggml/build/src/libggml.a \
 *      vendor/ggml/build/src/libggml-cpu.a \
 *      vendor/ggml/build/src/libggml-base.a \
 *      -lstdc++ -lpthread -lgomp -lm -o /tmp/rms_probe2
 */
#include "tinynn_ggml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void)
{
    void *sess = tnn_session_new(0);
    if (!sess) { fprintf(stderr, "session_new failed\n"); return 1; }

    /* shape (3, 4): 3 rows of 4 (matches ggml test's [n, 5, 4, 3] flavor more closely) */
    void *tdy = tnn_input_2d_f32(sess, 3, 4);
    void *tx  = tnn_input_2d_f32(sess, 3, 4);

    /* Mark as inputs so backend_sched preserves their buffers. */
    ggml_set_input((struct ggml_tensor *) tdy);
    ggml_set_input((struct ggml_tensor *) tx);

    void *tc  = tnn_rms_norm_back(sess, tdy, tx, 1e-4);
    if (!tc) { fprintf(stderr, "rms_norm_back returned NULL\n"); return 1; }
    ggml_set_output((struct ggml_tensor *) tc);

    int rc = tnn_realize(sess, tc);
    printf("realize rc=%d\n", rc);

    /* Upload dy = [[1,0,0,0],[1,0,0,0],[1,0,0,0]] */
    for (int i = 0; i < 12; i++) tnn_scratch_set(sess, i, (i % 4 == 0) ? 1.0 : 0.0);
    tnn_upload(sess, tdy);

    /* Upload x = same */
    for (int i = 0; i < 12; i++) tnn_scratch_set(sess, i, (i % 4 == 0) ? 1.0 : 0.0);
    tnn_upload(sess, tx);

    rc = tnn_compute(sess);
    printf("compute rc=%d\n", rc);

    tnn_download(sess, tc);
    printf("result row 0: [%g %g %g %g]\n",
        tnn_scratch_get(sess, 0), tnn_scratch_get(sess, 1),
        tnn_scratch_get(sess, 2), tnn_scratch_get(sess, 3));
    printf("result row 1: [%g %g %g %g]\n",
        tnn_scratch_get(sess, 4), tnn_scratch_get(sess, 5),
        tnn_scratch_get(sess, 6), tnn_scratch_get(sess, 7));
    printf("result row 2: [%g %g %g %g]\n",
        tnn_scratch_get(sess, 8), tnn_scratch_get(sess, 9),
        tnn_scratch_get(sess, 10), tnn_scratch_get(sess, 11));
    printf("expected per ggml impl source for each row: ~[8e-4, 0, 0, 0]\n");

    tnn_session_free(sess);
    return 0;
}
