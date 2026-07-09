/* See tinynn_trace.h. */

#include "tinynn_trace.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* The ring is fixed-size for predictable memory. 1048576 * 24 bytes =
 * 24 MB. Sized for "one full native-Mat training run under trace"
 * which is ~6 Mat-ops × ~100/step × ~hundreds of steps. FFI workloads
 * use far fewer events; this just sets the ceiling. */
#define TNN_TRACE_BUF_SIZE (1024 * 1024)

typedef struct {
    const char *name;
    int64_t     ts_us;   /* offset from g_trace.epoch_us */
    int64_t     dur_us;
} tnn_trace_evt;

static struct {
    int             active;
    FILE           *out;
    tnn_trace_evt  *buf;
    int             head;
    int             wrap_warned;
    int64_t         epoch_us;
} g_trace = {0, NULL, NULL, 0, 0, 0};

/* P6 per-op capture state. Flag is independent of g_trace.active so a
 * caller can set it before tnn_trace_open and have it honoured once
 * the trace opens. tnn_trace_op_capture_active() ANDs both. */
static int     g_op_capture       = 0;
static int64_t g_op_pending_start = 0;

static int64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + (int64_t)(ts.tv_nsec / 1000);
}

int tnn_trace_active(void) {
    return g_trace.active;
}

int tnn_trace_open(const char *path) {
    if (g_trace.active) return -1;
    if (!path) return -2;
    g_trace.out = fopen(path, "w");
    if (!g_trace.out) return -3;
    g_trace.buf = (tnn_trace_evt *)calloc(TNN_TRACE_BUF_SIZE, sizeof(tnn_trace_evt));
    if (!g_trace.buf) {
        fclose(g_trace.out);
        g_trace.out = NULL;
        return -4;
    }
    /* Chrome Trace Format accepts either a top-level array or an
     * object with a "traceEvents" field. We emit the array form — it
     * streams cleanly with one record per line, and the close marker
     * is just a single "]". */
    fprintf(g_trace.out, "[\n");
    g_trace.epoch_us    = now_us();
    g_trace.head        = 0;
    g_trace.wrap_warned = 0;
    g_trace.active      = 1;
    return 0;
}

int64_t tnn_trace_begin(const char *name) {
    if (!g_trace.active) return 0;
    (void)name;   /* unused at begin; we capture at end so a single
                   * timestamp covers both push and pop. */
    return now_us();
}

void tnn_trace_end(const char *name, int64_t start_ts) {
    if (!g_trace.active) return;
    int64_t end_ts = now_us();
    int slot = g_trace.head;
    if (slot >= TNN_TRACE_BUF_SIZE) {
        if (!g_trace.wrap_warned) {
            fprintf(stderr, "[tnn-trace] event buffer full at %d "
                            "events; further events dropped.\n",
                    TNN_TRACE_BUF_SIZE);
            g_trace.wrap_warned = 1;
        }
        return;
    }
    g_trace.buf[slot].name   = name;
    g_trace.buf[slot].ts_us  = start_ts - g_trace.epoch_us;
    g_trace.buf[slot].dur_us = end_ts - start_ts;
    g_trace.head = slot + 1;
}

void tnn_trace_set_op_capture(int en) {
    g_op_capture = en ? 1 : 0;
}

int tnn_trace_op_capture_active(void) {
    return g_op_capture && g_trace.active;
}

void tnn_trace_op_record_begin(void) {
    if (!g_op_capture || !g_trace.active) return;
    g_op_pending_start = now_us();
}

void tnn_trace_op_record_end(const char *op_name) {
    if (!g_op_capture || !g_trace.active) return;
    int slot = g_trace.head;
    if (slot >= TNN_TRACE_BUF_SIZE) {
        if (!g_trace.wrap_warned) {
            fprintf(stderr, "[tnn-trace] event buffer full at %d "
                            "events; further events dropped.\n",
                    TNN_TRACE_BUF_SIZE);
            g_trace.wrap_warned = 1;
        }
        return;
    }
    int64_t end_ts = now_us();
    g_trace.buf[slot].name   = op_name;
    g_trace.buf[slot].ts_us  = g_op_pending_start - g_trace.epoch_us;
    g_trace.buf[slot].dur_us = end_ts - g_op_pending_start;
    g_trace.head = slot + 1;
}

void tnn_trace_mark(const char *name) {
    if (!g_trace.active) return;
    int slot = g_trace.head;
    if (slot >= TNN_TRACE_BUF_SIZE) return;   /* same overflow rule */
    int64_t ts = now_us();
    g_trace.buf[slot].name   = name;
    g_trace.buf[slot].ts_us  = ts - g_trace.epoch_us;
    g_trace.buf[slot].dur_us = 0;
    g_trace.head = slot + 1;
}

/* Escape JSON string for output. Names should be plain ASCII (they're
 * literals in our code) so the only escapes we expect are "\\" and
 * "\"". We bail loud if we see a control char — that's a programmer
 * error, not a runtime case. */
static void write_escaped(FILE *f, const char *s) {
    while (*s) {
        unsigned char c = (unsigned char)*s++;
        if (c == '"' || c == '\\') {
            fputc('\\', f); fputc((int)c, f);
        } else if (c < 0x20) {
            fprintf(f, "\\u%04x", c);
        } else {
            fputc((int)c, f);
        }
    }
}

void tnn_trace_close(void) {
    if (!g_trace.active) return;
    g_trace.active = 0;   /* stop accepting events first; serialization
                           * is unguarded. */
    for (int i = 0; i < g_trace.head; i++) {
        const tnn_trace_evt *e = &g_trace.buf[i];
        fprintf(g_trace.out, "%s {", i == 0 ? " " : ",");
        if (e->dur_us > 0) {
            fprintf(g_trace.out, "\"ph\":\"X\",\"ts\":%ld,\"dur\":%ld,\"name\":\"",
                    (long)e->ts_us, (long)e->dur_us);
        } else {
            /* instant event */
            fprintf(g_trace.out, "\"ph\":\"i\",\"ts\":%ld,\"name\":\"",
                    (long)e->ts_us);
        }
        write_escaped(g_trace.out, e->name ? e->name : "(null)");
        fprintf(g_trace.out, "\",\"pid\":1,\"tid\":1}\n");
    }
    fprintf(g_trace.out, "]\n");
    fclose(g_trace.out);
    free(g_trace.buf);
    g_trace.out = NULL;
    g_trace.buf = NULL;
    g_trace.head = 0;
}
