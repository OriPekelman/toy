/* See tinynn_events.h. */

#include "tinynn_events.h"

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/utsname.h>

static struct {
    FILE   *out;
    double  epoch_sec;   /* CLOCK_MONOTONIC at open time */
} g_evt = { NULL, 0.0 };

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

static int events_open_mode(const char *path, const char *mode) {
    if (g_evt.out) return -1;
    if (!path)     return -2;
    g_evt.out = fopen(path, mode);
    if (!g_evt.out) return -3;
    /* Line-buffered so consumers tailing the file see new events
     * promptly — no waiting on a 4 KB stdio block boundary. */
    setvbuf(g_evt.out, NULL, _IOLBF, 0);
    g_evt.epoch_sec = monotonic_seconds();
    return 0;
}

int tnn_events_open(const char *path) {
    return events_open_mode(path, "a");
}

int tnn_events_open_trunc(const char *path) {
    return events_open_mode(path, "w");
}

void tnn_events_emit(const char *json_obj) {
    if (!g_evt.out || !json_obj) return;
    fputs(json_obj, g_evt.out);
    fputc('\n', g_evt.out);
}

void tnn_events_close(void) {
    if (!g_evt.out) return;
    fflush(g_evt.out);
    fclose(g_evt.out);
    g_evt.out = NULL;
    g_evt.epoch_sec = 0.0;
}

int tnn_events_active(void) {
    return g_evt.out != NULL;
}

double tnn_events_now_seconds(void) {
    if (!g_evt.out) return 0.0;
    return monotonic_seconds() - g_evt.epoch_sec;
}

/* Provenance helpers for run_start / run_end fields the v1 schema
 * documents (tao#run-start-provenance). All return pointers to
 * static thread-unsafe buffers (callers must copy before reusing). */

/* ISO-8601 UTC, second precision: "2026-05-25T17:42:13Z". */
const char *tnn_events_iso8601_now(void) {
    static char buf[32];
    time_t now = time(NULL);
    struct tm tm_buf;
    gmtime_r(&now, &tm_buf);
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
    return buf;
}

const char *tnn_provenance_host_name(void) {
    static char buf[256] = {0};
    if (buf[0] == '\0') {
        if (gethostname(buf, sizeof(buf) - 1) != 0) {
            strncpy(buf, "unknown", sizeof(buf) - 1);
        }
    }
    return buf;
}

const char *tnn_provenance_host_os(void) {
    static char buf[64] = {0};
    if (buf[0] == '\0') {
        struct utsname u;
        if (uname(&u) == 0) {
            /* Lowercase ("Linux" -> "linux") to match the schema's
             * `"os": "linux"` example. */
            size_t i = 0;
            while (u.sysname[i] != '\0' && i < sizeof(buf) - 1) {
                char c = u.sysname[i];
                if (c >= 'A' && c <= 'Z') c = c + ('a' - 'A');
                buf[i] = c;
                i++;
            }
            buf[i] = '\0';
        } else {
            strncpy(buf, "unknown", sizeof(buf) - 1);
        }
    }
    return buf;
}

const char *tnn_provenance_host_arch(void) {
    static char buf[64] = {0};
    if (buf[0] == '\0') {
        struct utsname u;
        if (uname(&u) == 0) {
            strncpy(buf, u.machine, sizeof(buf) - 1);
        } else {
            strncpy(buf, "unknown", sizeof(buf) - 1);
        }
    }
    return buf;
}
