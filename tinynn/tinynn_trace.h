/* Chrome Trace Format emitter.
 *
 * One global trace target at a time. Open → instrument → close. The
 * "off" state is fast: each tnn_trace_begin/end checks a single static
 * int and returns immediately. We measured ~5ns/pair when off, dwarfed
 * by anything else in the inner loop.
 *
 * Storage is a fixed-size ring of events; when full, further events
 * are dropped with a one-shot stderr warning. Default buffer is 65536
 * events, which at v1 instrumentation (FFI boundaries + Ruby step
 * loop) is enough for ~1000 training steps.
 *
 * Output is a Chrome Trace JSON array on close — open in
 * https://perfetto.dev or chrome://tracing.
 *
 * IMPORTANT: every `name` passed to tnn_trace_begin/end must be a
 * string with stable address (string literal, or pointer to long-
 * lived storage). We do NOT copy; flush time re-reads the pointer.
 */

#ifndef TINYNN_TRACE_H
#define TINYNN_TRACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* Open a trace file. Returns 0 on success, negative on failure.
 * Idempotent calls when already-open return -1. */
int     tnn_trace_open(const char *path);

/* Push a region; returns a token (the start timestamp). Pair with
 * tnn_trace_end. When tracing is off, returns 0 and does no work. */
int64_t tnn_trace_begin(const char *name);

/* Pop a region. `name` and `start_ts` must match the
 * tnn_trace_begin call. When tracing is off, does nothing. */
void    tnn_trace_end(const char *name, int64_t start_ts);

/* One-shot instant event (no duration). Useful for markers. */
void    tnn_trace_mark(const char *name);

/* Flush events and close the file. */
void    tnn_trace_close(void);

/* Cheap predicate (one load) — callers can guard expensive
 * instrumentation (e.g. snprintf'd dynamic names) so the OFF path
 * is truly zero-cost. */
int     tnn_trace_active(void);

/* Per-op timing (P6). Opt-in: enabled by tnn_trace_set_op_capture(1)
 * AFTER tnn_trace_open. When enabled, ggml-backend's sched eval
 * callback (installed by tinynn_ggml.c at engine init) routes each
 * node through tnn_trace_op_record_{begin,end}, which emit one
 * Chrome-Trace duration event per ggml op (name = ggml_op_name(t->op)).
 *
 * Cost when OFF (capture flag = 0): one load + compare per ggml node.
 *   Measured: identical wall time to a build without the eval
 *   callback installed (within timing noise).
 * Cost when ON: ~5× slowdown of compute_backward at SmolLM2-135M
 *   LoRA shape on CPU. Most of the overhead is not the timestamp /
 *   ring-buffer writes (cheap) but ggml-backend's per-node
 *   callback dispatch path, which disables certain kernel fusion
 *   opportunities. Acceptable for diagnostic use; do NOT leave on
 *   for production profiling of throughput.
 *
 * Caveat (CUDA): eval callback fires after CPU enqueue, not after
 * kernel completion. CUDA per-op timings reflect launch latency
 * rather than kernel duration. CPU per-op timings are wall-accurate.
 *
 * Names are ggml_op_name(...) returns, which point into ggml's
 * static const table — stable for the trace lifetime as required by
 * the buffer contract above. */
void    tnn_trace_set_op_capture(int en);
int     tnn_trace_op_capture_active(void);
void    tnn_trace_op_record_begin(void);
void    tnn_trace_op_record_end(const char *op_name);

#ifdef __cplusplus
}
#endif

#endif
