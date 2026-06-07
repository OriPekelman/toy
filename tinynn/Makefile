# Standalone build for the tinynn shim archive — consumed by spinel-ext.json's
# "tinynn" build-unit (toy#45): `make -C tinynn` inside a consumer's vendor
# tree, where ../vendor/ggml is the vendored (patched) ggml. The root Makefile
# keeps its own copies of these rules for the dev flow; keep both in sync.
CC      ?= cc
CFLAGS  ?= -O2 -fPIC -Wall -Wextra
GGML_INC := -I../vendor/ggml/include -I../vendor/ggml/src

OBJS := tinynn_ggml.o tinynn_gguf.o tinynn_trace.o tinynn_events.o

libtinynn_ggml.a: $(OBJS)
	ar rcs $@ $(OBJS)

tinynn_ggml.o: tinynn_ggml.c tinynn_ggml.h tinynn_trace.h
	$(CC) $(CFLAGS) $(GGML_INC) -c $< -o $@

tinynn_gguf.o: tinynn_gguf.c tinynn_gguf.h
	$(CC) $(CFLAGS) $(GGML_INC) -c $< -o $@

tinynn_trace.o: tinynn_trace.c tinynn_trace.h
	$(CC) $(CFLAGS) -c $< -o $@

tinynn_events.o: tinynn_events.c tinynn_events.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) libtinynn_ggml.a
.PHONY: clean
