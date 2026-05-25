/* JSON Lines event-stream emitter.
 *
 * Sibling to tinynn_trace.h (Chrome Trace format for per-op timing).
 * This primitive emits the run-level event stream documented in
 * docs/events-schema.md — one JSON object per line, append-only,
 * versioned `"schema": "toy/v1"`. The file is the contract between
 * Toy and downstream consumers (live TUIs, post-run analyzers,
 * W&B/MLflow adapters, the Tao project).
 *
 * Design choices:
 *
 * - Caller builds each event's JSON object (typically in Ruby via
 *   string concatenation). We append a newline and a single fprintf.
 *   Buffered fwrite under the hood; one fflush at close.
 *
 * - The OFF path (no file open) is a single load + compare in
 *   tnn_events_emit and tnn_events_active — same shape as the trace
 *   primitive. Cheap enough to leave instrumented in production.
 *
 * - tnn_events_now_seconds() returns time since open as a double,
 *   matching the schema's `"t"` field. CLOCK_MONOTONIC under the
 *   hood; immune to wall-clock skew.
 *
 * - One file per process. Multiple opens fail (returns -1). This
 *   matches the one-run-one-file convention from the schema.
 *
 * - We do NOT validate the JSON we're handed. Callers are expected
 *   to emit well-formed objects per the schema. A bad object turns
 *   into a torn line that consumers will skip.
 */

#ifndef TINYNN_EVENTS_H
#define TINYNN_EVENTS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Open the JSONL file for append-only writes. Path may be relative
 * or absolute. Parent directories must already exist (we don't
 * mkdir — keep the primitive simple). Returns 0 on success, negative
 * on failure (-1 already open, -2 null path, -3 fopen failed). */
int     tnn_events_open(const char *path);

/* Append a single JSON event followed by `\n`. The string MUST be a
 * complete, well-formed JSON object — we do not validate. When the
 * file isn't open, this is a single load + branch (no-op). */
void    tnn_events_emit(const char *json_obj);

/* Flush and close. Idempotent — closing an unopened stream is a
 * no-op. After close, subsequent emits are no-ops until a new
 * open. */
void    tnn_events_close(void);

/* Cheap predicate — callers can guard expensive JSON-building code
 * (string concats, .to_s on floats, etc.) so the OFF path is
 * truly zero-cost. */
int     tnn_events_active(void);

/* Seconds-since-open as a double (schema's `"t"` field). When the
 * file isn't open, returns 0.0. Uses CLOCK_MONOTONIC; safe under
 * clock skew. */
double  tnn_events_now_seconds(void);

/* Provenance helpers for the run_start / run_end fields the v1 schema
 * documents (tao#run-start-provenance). All return pointers to static
 * thread-unsafe buffers; copy or emit immediately. */
const char *tnn_events_iso8601_now(void);   /* "2026-05-25T17:42:13Z" */
const char *tnn_provenance_host_name(void); /* gethostname() */
const char *tnn_provenance_host_os(void);   /* uname.sysname lowercased */
const char *tnn_provenance_host_arch(void); /* uname.machine */

#ifdef __cplusplus
}
#endif

#endif
