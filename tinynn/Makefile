# Standalone build for the tinynn shim archive — consumed by spinel-ext.json's
# "tinynn" build-unit (toy#45): `make -C tinynn` inside a consumer's vendor
# tree, where ../vendor/ggml is the vendored (patched) ggml. The root Makefile
# keeps its own copies of these rules for the dev flow; keep both in sync.
CC       ?= cc
CFLAGS   ?= -O2 -fPIC -Wall -Wextra
GGML_INC := -I../vendor/ggml/include -I../vendor/ggml/src
CUDA_DIR ?= /usr/local/cuda

OBJS := tinynn_ggml.o tinynn_gguf.o tinynn_trace.o tinynn_events.o

libtinynn_ggml.a: $(OBJS)
	ar rcs $@ $(OBJS)

# Optional backend shims (toy#45 Phase 3). NOT in the default target —
# they are separate build-unit targets ("targets": ["cuda"] / ["metal"]
# in the staged spinel-ext-gpu.json entries), opt-in per spinelgems#20.
cuda: libtinynn_ggml_cuda.a
.PHONY: cuda

libtinynn_ggml_cuda.a: tinynn_backend_cuda.o
	ar rcs $@ $<

tinynn_backend_cuda.o: tinynn_backend_cuda.c
	$(CC) $(CFLAGS) $(GGML_INC) -I$(CUDA_DIR)/include -c $< -o $@

# Metal (macOS-only; Mac-validation-pending — toy#27). Mirrors the root
# Makefile's rule: ObjC compile, no ARC objects, frameworks come in at
# the consumer's link line.
metal: libtinynn_ggml_metal.a
.PHONY: metal

libtinynn_ggml_metal.a: tinynn_backend_metal.o
	ar rcs $@ $<

tinynn_backend_metal.o: tinynn_backend_metal.m
	$(CC) $(CFLAGS) -x objective-c $(GGML_INC) -c $< -o $@

tinynn_ggml.o: tinynn_ggml.c tinynn_ggml.h tinynn_trace.h
	$(CC) $(CFLAGS) $(GGML_INC) -c $< -o $@

tinynn_gguf.o: tinynn_gguf.c tinynn_gguf.h
	$(CC) $(CFLAGS) $(GGML_INC) -c $< -o $@

tinynn_trace.o: tinynn_trace.c tinynn_trace.h
	$(CC) $(CFLAGS) -c $< -o $@

tinynn_events.o: tinynn_events.c tinynn_events.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) libtinynn_ggml.a \
	      tinynn_backend_cuda.o libtinynn_ggml_cuda.a \
	      tinynn_backend_metal.o libtinynn_ggml_metal.a
.PHONY: clean
