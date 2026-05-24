/* See tinynn_events.h. */

#include "tinynn_events.h"

#include <stdio.h>
#include <time.h>

static struct {
    FILE   *out;
    double  epoch_sec;   /* CLOCK_MONOTONIC at open time */
} g_evt = { NULL, 0.0 };

static double monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1.0e9;
}

int tnn_events_open(const char *path) {
    if (g_evt.out) return -1;
    if (!path)     return -2;
    g_evt.out = fopen(path, "a");
    if (!g_evt.out) return -3;
    /* Line-buffered so consumers tailing the file see new events
     * promptly — no waiting on a 4 KB stdio block boundary. */
    setvbuf(g_evt.out, NULL, _IOLBF, 0);
    g_evt.epoch_sec = monotonic_seconds();
    return 0;
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
