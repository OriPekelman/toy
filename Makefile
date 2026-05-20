# toy build system.
#
# Demo / training Ruby drivers live in demos/ and compile to native
# binaries via Spinel. CUDA / ggml acceleration is opt-in:
#
#   make                   # demos/train_minimal + demos/train_tinystories
#   make setup-ggml        # one-time clone + CPU build of vendored ggml
#   make setup-ggml-cuda   # one-time clone + CUDA build (needs CUDA toolkit)
#   make smoke             # tinynn FFI smoke test (4x3 ggml matmul demo)
#   make distilgpt2_demo_text  # → demos/distilgpt2_demo_text
#
# Vendored ggml lives at vendor/ggml/ (gitignored).
# The CUDA build expects sm_121 (NVIDIA GB10); override with
# GGML_CUDA_ARCH=NN on the command line.

SPINEL_DIR  ?= $(HOME)/sites/spinel
SPINEL      ?= $(SPINEL_DIR)/spinel

CC          ?= cc
CFLAGS      ?= -O2 -fPIC -Wall -Wextra
ARFLAGS      = rcs

# macOS Command Line Tools (as of 26.x) keep stale 2023 C++ stub headers
# at /Library/Developer/CommandLineTools/usr/include/c++/v1 which shadow
# the real headers in the SDK. Prepend the SDK's libc++ include path so
# ggml's C++ files can find <mutex>, <array>, etc. No-op on Linux.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  CMAKE_ENV := CPLUS_INCLUDE_PATH=$(shell xcrun --show-sdk-path)/usr/include/c++/v1
  NJOBS     := $(shell sysctl -n hw.logicalcpu)
else
  CMAKE_ENV :=
  NJOBS     := $(shell nproc)
endif

# --- vendored ggml ----------------------------------------------------------
GGML_DIR    := vendor/ggml
GGML_REPO   := https://github.com/ggml-org/ggml.git
GGML_CUDA_ARCH ?= 121
CUDA_DIR    ?= /usr/local/cuda

# --- pure-Spinel drivers ----------------------------------------------------
# Source lives in demos/. We expose short top-level target names
# (`make train_minimal`, `make distilgpt2_demo_text`) that build into
# demos/. Run the resulting binaries from the repo root.
all: demos/train demos/smollm2

# Parity-checks vs native TransformerLM.forward.

# Tep+Spinel HTTP server demos. See tep_demo/README.md. Builds bypass
# tep's translator (we use spinel directly on a layout-substituted
# copy of tep's lib in tep_demo/_tep_lib/).
tep_demo/hello: tep_demo/hello_api.rb tep_demo/_tep_lib/tep.rb
	$(SPINEL) tep_demo/hello_api.rb -o tep_demo/hello

# Inference API: /generate?n=N runs greedy generation via FullForwardFFICache.
tep_demo/api: tep_demo/inference_api.rb tep_demo/_tep_lib/tep.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tep_demo/inference_api.rb -o tep_demo/api

# OpenAI-compatible API backed by SmolLM2/Qwen-family models via the
# direct GGUF→FFI loader. Accepts pre-tokenized integer IDs (no
# server-side tokenizer — keeps the single-binary deployment story).
# Hard-codes the model GGUF path; copy + edit for other sizes.
tep_demo/openai_api_smollm2: tep_demo/openai_api_smollm2.rb tep_demo/_tep_lib/tep.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tep_demo/openai_api_smollm2.rb -o tep_demo/openai_api_smollm2

# Larger-size variants of the same API. Per-binary GGUF path because
# Spinel mistypes env-var-driven constants — copy + sed-edit is the
# convention. Add new sizes by following the pattern (see
# tep_demo/openai_api_qwen25_1.5b.rb for the canonical 2-line diff).
OPENAI_QWEN_SOURCES   = $(wildcard tep_demo/openai_api_qwen25_*.rb)
OPENAI_QWEN_TARGETS   = $(OPENAI_QWEN_SOURCES:.rb=)
OPENAI_QWEN_DEPS      = tep_demo/_tep_lib/tep.rb \
                        lib/toy_smollm2_ffi_kv.rb \
                        lib/toy_smollm2_loader.rb \
                        tinynn/libtinynn_ggml.a

openai-qwen-all: $(OPENAI_QWEN_TARGETS)

tep_demo/openai_api_qwen25_%: tep_demo/openai_api_qwen25_%.rb $(OPENAI_QWEN_DEPS)
	$(SPINEL) $< -o $@

# --- ggml vendor ------------------------------------------------------------
$(GGML_DIR)/CMakeLists.txt:
	mkdir -p vendor
	git clone --depth 1 $(GGML_REPO) $(GGML_DIR)

# GGML_OPENMP=OFF: avoid the libgomp link dependency. On macOS clang
# ships libomp (LLVM), not libgomp (GNU); ggml's own thread pool covers
# CPU parallelism either way. Same setting used on Linux for build
# parity (and so lib/tinynn.rb doesn't need ffi_lib "gomp").
setup-ggml: $(GGML_DIR)/CMakeLists.txt
	cd $(GGML_DIR) && $(CMAKE_ENV) cmake -B build \
	  -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON \
	  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF \
	  -DGGML_OPENCL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_ACCELERATE=OFF \
	  -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF \
	  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
	cd $(GGML_DIR) && $(CMAKE_ENV) cmake --build build -j$(NJOBS)

setup-ggml-cuda: $(GGML_DIR)/CMakeLists.txt
	cd $(GGML_DIR) && PATH=$(CUDA_DIR)/bin:$$PATH $(CMAKE_ENV) cmake -B build-cuda \
	  -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON \
	  -DGGML_CUDA=ON -DGGML_METAL=OFF -DGGML_VULKAN=OFF \
	  -DGGML_OPENCL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_ACCELERATE=OFF \
	  -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF \
	  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	  -DCMAKE_CUDA_ARCHITECTURES=$(GGML_CUDA_ARCH) -DGGML_NATIVE=OFF
	cd $(GGML_DIR) && PATH=$(CUDA_DIR)/bin:$$PATH $(CMAKE_ENV) cmake --build build-cuda -j$(NJOBS)

# --- tinynn shim (CPU build) ------------------------------------------------
GGML_INC := -I$(GGML_DIR)/include -I$(GGML_DIR)/src

tinynn/tinynn_ggml.o: tinynn/tinynn_ggml.c tinynn/tinynn_ggml.h
	$(CC) $(CFLAGS) $(GGML_INC) -c $< -o $@

tinynn/tinynn_gguf.o: tinynn/tinynn_gguf.c tinynn/tinynn_gguf.h
	$(CC) $(CFLAGS) $(GGML_INC) -c $< -o $@

tinynn/libtinynn_ggml.a: tinynn/tinynn_ggml.o tinynn/tinynn_gguf.o
	ar $(ARFLAGS) $@ tinynn/tinynn_ggml.o tinynn/tinynn_gguf.o

# --- smoke test -------------------------------------------------------------
# Builds tinynn/smoke.rb against the CPU shim. Requires `setup-ggml` to have
# been run once first.
smoke: tinynn/smoke
	./tinynn/smoke

tinynn/smoke: tinynn/smoke.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/smoke.rb -o tinynn/smoke

# A/B parity tests: native vs FFI (CPU) for one op each.
ab-smoke: tinynn/ab_smoke
	./tinynn/ab_smoke

ab-smoke-add: tinynn/ab_smoke_add
	./tinynn/ab_smoke_add

ab-smoke-gelu: tinynn/ab_smoke_gelu
	./tinynn/ab_smoke_gelu

# Llama-family ops (silu, mul, eventually rope) — added with the
# Toy::SmolLM2 FFI mirror work.
ab-smoke-silu: tinynn/ab_smoke_silu
	./tinynn/ab_smoke_silu

tinynn/ab_smoke_silu: tinynn/ab_smoke_silu.rb lib/toy_card.rb lib/toy.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

ab-smoke-mul: tinynn/ab_smoke_mul
	./tinynn/ab_smoke_mul

tinynn/ab_smoke_mul: tinynn/ab_smoke_mul.rb lib/toy_card.rb lib/toy.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

ab-smoke-rms-norm: tinynn/ab_smoke_rms_norm
	./tinynn/ab_smoke_rms_norm

ab-smoke-softmax: tinynn/ab_smoke_softmax
	./tinynn/ab_smoke_softmax

ab-smoke-transpose: tinynn/ab_smoke_transpose
	./tinynn/ab_smoke_transpose

ab-smoke-scale: tinynn/ab_smoke_scale
	./tinynn/ab_smoke_scale

# Chained-op pipeline: gelu(h·w1)·w2 in one ggml graph.
ab-smoke-pipeline: tinynn/ab_smoke_pipeline
	./tinynn/ab_smoke_pipeline

# Run every CPU smoke. (CUDA variants would need `make setup-ggml-cuda` first.)
# `ab-smoke-transpose` is omitted: ggml_cont(ggml_transpose(...)) trips
# the scheduler's buffer allocation; we fold transposes into consuming
# ops instead (see TinyNN.matmul's b-transposed upload).
test: smoke ab-smoke ab-smoke-add ab-smoke-gelu ab-smoke-rms-norm \
       ab-smoke-softmax ab-smoke-scale ab-smoke-pipeline \
       ab-smoke-matmul-variants ab-smoke-back ab-smoke-embed ab-smoke-sgd \
       ab-smoke-gelu-back ab-smoke-cegrad ab-smoke-adam

tinynn/ab_smoke: tinynn/ab_smoke.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke.rb -o tinynn/ab_smoke

tinynn/ab_smoke_add: tinynn/ab_smoke_add.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_add.rb -o tinynn/ab_smoke_add

tinynn/ab_smoke_gelu: tinynn/ab_smoke_gelu.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_gelu.rb -o tinynn/ab_smoke_gelu

tinynn/ab_smoke_rms_norm: tinynn/ab_smoke_rms_norm.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_rms_norm.rb -o tinynn/ab_smoke_rms_norm

tinynn/ab_smoke_softmax: tinynn/ab_smoke_softmax.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_softmax.rb -o tinynn/ab_smoke_softmax

tinynn/ab_smoke_transpose: tinynn/ab_smoke_transpose.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_transpose.rb -o tinynn/ab_smoke_transpose

tinynn/ab_smoke_scale: tinynn/ab_smoke_scale.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_scale.rb -o tinynn/ab_smoke_scale

tinynn/ab_smoke_pipeline: tinynn/ab_smoke_pipeline.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_pipeline.rb -o tinynn/ab_smoke_pipeline

# Chained FFNFFICache parity: pre, hidden, out vs hand-rolled native.
ab-smoke-ffncache: tinynn/ab_smoke_ffncache
	./tinynn/ab_smoke_ffncache

tinynn/ab_smoke_ffncache: tinynn/ab_smoke_ffncache.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_ffncache.rb -o tinynn/ab_smoke_ffncache

# ggml-native AdamW step (opt_step_adamw) parity vs project's plain-Adam.
ab-smoke-adamw-op: tinynn/ab_smoke_adamw_op
	./tinynn/ab_smoke_adamw_op

tinynn/ab_smoke_adamw_op: tinynn/ab_smoke_adamw_op.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_adamw_op.rb -o tinynn/ab_smoke_adamw_op

# Persistent-tensor architecture check: data uploaded to a ctx_w tensor
# survives a compute cycle.
ab-smoke-persistent: tinynn/ab_smoke_persistent
	./tinynn/ab_smoke_persistent

tinynn/ab_smoke_persistent: tinynn/ab_smoke_persistent.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_persistent.rb -o tinynn/ab_smoke_persistent

# Dual-cgraph + persistent-weights design check: forward reads t_w;
# adam mutates t_w in place; forward sees the new value.
ab-smoke-dual-graph: tinynn/ab_smoke_dual_graph
	./tinynn/ab_smoke_dual_graph

tinynn/ab_smoke_dual_graph: tinynn/ab_smoke_dual_graph.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_dual_graph.rb -o tinynn/ab_smoke_dual_graph

# M2 foundation: view_2d + cpy to write a single row into a persistent
# (max_T, d_head) KV buffer at a runtime-baked position.
ab-smoke-kv-write: tinynn/ab_smoke_kv_write
	./tinynn/ab_smoke_kv_write

tinynn/ab_smoke_kv_write: tinynn/ab_smoke_kv_write.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_kv_write.rb -o tinynn/ab_smoke_kv_write

# M2 prototype: single-step decode through a KV cache. Pre-fills K/V
# for positions 0..POS-1, writes k_new/v_new at POS, computes scores
# + soft_max_ext + head_out. Parity vs hand-rolled native.
ab-smoke-kv-attn: tinynn/ab_smoke_kv_attn
	./tinynn/ab_smoke_kv_attn

tinynn/ab_smoke_kv_attn: tinynn/ab_smoke_kv_attn.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_kv_attn.rb -o tinynn/ab_smoke_kv_attn

# M1.2: full single-block forward through the persistent graph.
# Parity vs native TransformerLM.forward() at n_layers=1, n_heads=2.
ab-smoke-full-forward-block: tinynn/ab_smoke_full_forward_block
	./tinynn/ab_smoke_full_forward_block

tinynn/ab_smoke_full_forward_block: tinynn/ab_smoke_full_forward_block.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_full_forward_block.rb -o tinynn/ab_smoke_full_forward_block

# Wallclock bench: native TransformerLM.forward vs FullForwardFFICache.
full-forward-bench: tinynn/full_forward_bench
	./tinynn/full_forward_bench

tinynn/full_forward_bench: tinynn/full_forward_bench.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/full_forward_bench.rb -o tinynn/full_forward_bench

full-forward-bench-cuda: tinynn/full_forward_bench_cuda
	./tinynn/full_forward_bench_cuda

tinynn/full_forward_bench_cuda: tinynn/full_forward_bench_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) tinynn/full_forward_bench_cuda.rb -o tinynn/full_forward_bench_cuda

ab-smoke-dual-graph-cuda: tinynn/ab_smoke_dual_graph_cuda
	./tinynn/ab_smoke_dual_graph_cuda

tinynn/ab_smoke_dual_graph_cuda: tinynn/ab_smoke_dual_graph_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) tinynn/ab_smoke_dual_graph_cuda.rb -o tinynn/ab_smoke_dual_graph_cuda

ab-smoke-adamw-op-cuda: tinynn/ab_smoke_adamw_op_cuda
	./tinynn/ab_smoke_adamw_op_cuda

tinynn/ab_smoke_adamw_op_cuda: tinynn/ab_smoke_adamw_op_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) tinynn/ab_smoke_adamw_op_cuda.rb -o tinynn/ab_smoke_adamw_op_cuda

# Transformer-shape sized parity + wallclock comparison.
ab-smoke-big: tinynn/ab_smoke_big
	./tinynn/ab_smoke_big

tinynn/ab_smoke_big: tinynn/ab_smoke_big.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_big.rb -o tinynn/ab_smoke_big

ab-smoke-matmul-variants: tinynn/ab_smoke_matmul_variants
	./tinynn/ab_smoke_matmul_variants

tinynn/ab_smoke_matmul_variants: tinynn/ab_smoke_matmul_variants.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_matmul_variants.rb -o tinynn/ab_smoke_matmul_variants

ab-smoke-back: tinynn/ab_smoke_back
	./tinynn/ab_smoke_back

tinynn/ab_smoke_back: tinynn/ab_smoke_back.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_back.rb -o tinynn/ab_smoke_back

ab-smoke-gelu-back: tinynn/ab_smoke_gelu_back
	./tinynn/ab_smoke_gelu_back

tinynn/ab_smoke_gelu_back: tinynn/ab_smoke_gelu_back.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_gelu_back.rb -o tinynn/ab_smoke_gelu_back

ab-smoke-cegrad: tinynn/ab_smoke_cegrad
	./tinynn/ab_smoke_cegrad

tinynn/ab_smoke_cegrad: tinynn/ab_smoke_cegrad.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_cegrad.rb -o tinynn/ab_smoke_cegrad

ab-smoke-adam: tinynn/ab_smoke_adam
	./tinynn/ab_smoke_adam

tinynn/ab_smoke_adam: tinynn/ab_smoke_adam.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_adam.rb -o tinynn/ab_smoke_adam

gguf-smoke: tinynn/gguf_smoke
	./tinynn/gguf_smoke

tinynn/gguf_smoke: tinynn/gguf_smoke.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gguf_smoke.rb -o tinynn/gguf_smoke

# Walks every tensor in data/distilgpt2-f32.gguf via tnn_gguf_*. Used to
# confirm large HF-converted GGUFs roundtrip through the project FFI.
gguf-inspect: tinynn/gguf_inspect
	./tinynn/gguf_inspect

tinynn/gguf_inspect: tinynn/gguf_inspect.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gguf_inspect.rb -o tinynn/gguf_inspect

# GPT2LM build smoke: confirm lib/gpt2.rb Spinel-compiles and the
# forward shapes line up. Toy dims, random weights — values mean nothing.
gpt2-build-smoke: tinynn/gpt2_build_smoke
	./tinynn/gpt2_build_smoke

tinynn/gpt2_build_smoke: tinynn/gpt2_build_smoke.rb lib/transformer.rb lib/gpt2.rb
	$(SPINEL) tinynn/gpt2_build_smoke.rb -o tinynn/gpt2_build_smoke

# Load distilgpt2-f32.gguf into a GPT2LM and print sentinel weights
# per category. Verifies name mapping + per-head split before forward.
gpt2-load-smoke: tinynn/gpt2_load_smoke
	./tinynn/gpt2_load_smoke

tinynn/gpt2_load_smoke: tinynn/gpt2_load_smoke.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_load_smoke.rb -o tinynn/gpt2_load_smoke

# data/prompt_ids.txt, loads weights from data/distilgpt2-f32.gguf,
# greedy-generates N_NEW tokens via native Mat forward, writes the
# full ID sequence back. Decode with prep/tokens.py decode.

# Native Mat GPT-2 inference (DistilGPT2 / GPT-2 family).
#
gpt2:        demos/gpt2
demos/gpt2: demos/gpt2.rb lib/toy_card.rb lib/toy.rb lib/toy_gpt2.rb lib/toy_gpt2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# SmolLM2-135M (llama-family) inference via Toy::SmolLM2.
# Tokenization is host-side: ./prep/smollm2_tokens.py encode "..."
smollm2:        demos/smollm2
demos/smollm2: demos/smollm2.rb lib/toy_card.rb lib/toy.rb lib/toy_smollm2.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# SmolLM2-135M FFI KV-cache (CPU).
smollm2_kv:        demos/smollm2_kv
demos/smollm2_kv: demos/smollm2_kv.rb lib/toy_card.rb lib/toy.rb lib/toy_smollm2.rb lib/toy_smollm2_loader.rb lib/toy_smollm2_ffi_kv.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Qwen2.5 Mat-mediated KV-cache (CPU). The slow, correct reference path.
# Run with `GGUF=data/qwen25-1.5b-f32.gguf ./demos/qwen25_kv` etc.
qwen25_kv:        demos/qwen25_kv
demos/qwen25_kv: demos/qwen25_kv.rb lib/toy_card.rb lib/toy.rb lib/toy_smollm2.rb lib/toy_smollm2_loader.rb lib/toy_smollm2_ffi_kv.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Qwen2.5 Phase-2 mmap inference (CPU). Canonical performance path.
qwen25_native_mmap:        demos/qwen25_native_mmap
demos/qwen25_native_mmap: demos/qwen25_native_mmap.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Phase 0.7 acceptance gates: 0.5B (f32 + Q8) + 1.5B + 3B greedy-decode
# parity against locked-in golden token-ID sequences. Run before tagging
# a release; see docs/design/phase-07-acceptance.md.
qwen25_acceptance:        demos/qwen25_acceptance
demos/qwen25_acceptance: demos/qwen25_acceptance.rb lib/arch.rb lib/transformer_lm.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Inference bench (CPU). Long warmup + long prefill + per-token stats.
# Pick model via GGUF env; see docs/design/bench-cuda-2026-05-21.md.
qwen25_bench_cpu:        demos/qwen25_bench_cpu
demos/qwen25_bench_cpu: demos/qwen25_bench_cpu.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Inference bench (CUDA). Same shape as the CPU bench for side-by-side.
qwen25_bench_cuda:        demos/qwen25_bench_cuda
demos/qwen25_bench_cuda: demos/qwen25_bench_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# Qwen2.5 Phase-2 mmap inference (CUDA). Requires `make setup-ggml-cuda`.
qwen25_native_mmap_cuda:        demos/qwen25_native_mmap_cuda
demos/qwen25_native_mmap_cuda: demos/qwen25_native_mmap_cuda.rb lib/toy.rb lib/toy_smollm2.rb lib/toy_smollm2_loader.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/training.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# SmolLM2-135M FFI KV-cache (CUDA).
smollm2_kv_cuda:        demos/smollm2_kv_cuda
demos/smollm2_kv_cuda: demos/smollm2_kv_cuda.rb lib/toy_card.rb lib/toy.rb lib/toy_smollm2.rb lib/toy_smollm2_loader.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/training.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) $< -o $@

# TinyLlama-1.1B demo. Uses the same Toy::SmolLM2 / FFI KV CUDA stack
# (llama-family architecture); just configured for the larger shape.
tinyllama_kv_cuda:        demos/tinyllama_kv_cuda
demos/tinyllama_kv_cuda: demos/tinyllama_kv_cuda.rb lib/toy_card.rb lib/toy.rb lib/toy_smollm2.rb lib/toy_smollm2_loader.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/training.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) $< -o $@

tinyllama:        demos/tinyllama
demos/tinyllama: demos/tinyllama.rb lib/toy_card.rb lib/toy.rb lib/toy_smollm2.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

tinyllama_kv:        demos/tinyllama_kv
demos/tinyllama_kv: demos/tinyllama_kv.rb lib/toy_card.rb lib/toy.rb lib/toy_smollm2.rb lib/toy_smollm2_loader.rb lib/toy_smollm2_ffi_kv.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Print the Phuong–Hutter algorithm cards for both models. No
# inference — just emit the structured pseudocode. Source-of-truth
# for the round-trip work (task #33).
algorithm_cards:        demos/algorithm_cards
demos/algorithm_cards: demos/algorithm_cards.rb lib/toy_card.rb lib/toy.rb lib/toy_gpt2.rb lib/toy_smollm2.rb lib/toy_gpt2_loader.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# TinyStories from-scratch training via Toy::Trainer.
#
train:        demos/train
demos/train: demos/train.rb lib/toy_trainer.rb lib/transformer.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Parity probe: one forward at distilgpt2 shape, dump last-row logits
# to data/ours_logits.txt. Pair with prep/parity.py for the HF reference.
gpt2-parity: tinynn/gpt2_parity
	./tinynn/gpt2_parity

tinynn/gpt2_parity: tinynn/gpt2_parity.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_parity.rb -o tinynn/gpt2_parity

# FFI parity probe: persistent ggml graph with LayerNorm + biases.
# Dumps last-row logits to data/ours_ffi_logits.txt.
gpt2-ffi-parity: tinynn/gpt2_ffi_parity
	./tinynn/gpt2_ffi_parity

tinynn/gpt2_ffi_parity: tinynn/gpt2_ffi_parity.rb lib/transformer.rb lib/gpt2.rb lib/gpt2_ffi.rb lib/gguf_load.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_ffi_parity.rb -o tinynn/gpt2_ffi_parity

# Apples-to-apples bench: native Mat vs FFI on the same forward.
# Re-encode data/prompt_ids.txt first so prompt length matches T_SEQ=5.
gpt2-bench: tinynn/gpt2_bench
	./tinynn/gpt2_bench

tinynn/gpt2_bench: tinynn/gpt2_bench.rb lib/transformer.rb lib/gpt2.rb lib/gpt2_ffi.rb lib/gpt2_ffi_kv.rb lib/gguf_load.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_bench.rb -o tinynn/gpt2_bench

# Ruby BPE smoke: load vocab/merges, encode + roundtrip-decode some
# fixed prompts. Compare against prep/tokens.py output.
bpe-smoke: tinynn/bpe_smoke
	./tinynn/bpe_smoke

tinynn/bpe_smoke: tinynn/bpe_smoke.rb lib/transformer.rb lib/bpe.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/bpe_smoke.rb -o tinynn/bpe_smoke

# KV-cache parity probe: prefill the prompt one token at a time through
# GPT2KVFFICache, dump last-position logits.
gpt2-kv-parity: tinynn/gpt2_kv_parity
	./tinynn/gpt2_kv_parity

tinynn/gpt2_kv_parity: tinynn/gpt2_kv_parity.rb lib/transformer.rb lib/gpt2.rb lib/gpt2_ffi_kv.rb lib/gguf_load.rb lib/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_kv_parity.rb -o tinynn/gpt2_kv_parity

# --- CUDA mirrors of the GPT-2 demos / parity / bench --------------
# All require `make setup-ggml-cuda` to have produced
# vendor/ggml/build-cuda first. Built on the gx10 (NVIDIA GB10);
# the Mac build doesn't have CUDA.

CUDA_GPT2_DEPS = lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb \
                 lib/training.rb lib/tinynn.rb lib/tinynn_cuda.rb \
                 tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a

gpt2-ffi-parity-cuda: tinynn/gpt2_ffi_parity_cuda
	./tinynn/gpt2_ffi_parity_cuda

tinynn/gpt2_ffi_parity_cuda: tinynn/gpt2_ffi_parity_cuda.rb lib/gpt2_ffi_cuda.rb $(CUDA_GPT2_DEPS)
	$(SPINEL) tinynn/gpt2_ffi_parity_cuda.rb -o tinynn/gpt2_ffi_parity_cuda

gpt2-kv-parity-cuda: tinynn/gpt2_kv_parity_cuda
	./tinynn/gpt2_kv_parity_cuda

tinynn/gpt2_kv_parity_cuda: tinynn/gpt2_kv_parity_cuda.rb lib/gpt2_ffi_kv_cuda.rb $(CUDA_GPT2_DEPS)
	$(SPINEL) tinynn/gpt2_kv_parity_cuda.rb -o tinynn/gpt2_kv_parity_cuda

gpt2-bench-cuda: tinynn/gpt2_bench_cuda
	./tinynn/gpt2_bench_cuda

tinynn/gpt2_bench_cuda: tinynn/gpt2_bench_cuda.rb lib/gpt2_ffi_cuda.rb lib/gpt2_ffi_kv_cuda.rb $(CUDA_GPT2_DEPS)
	$(SPINEL) tinynn/gpt2_bench_cuda.rb -o tinynn/gpt2_bench_cuda

ab-smoke-embed: tinynn/ab_smoke_embed
	./tinynn/ab_smoke_embed

tinynn/ab_smoke_embed: tinynn/ab_smoke_embed.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_embed.rb -o tinynn/ab_smoke_embed

ab-smoke-sgd: tinynn/ab_smoke_sgd
	./tinynn/ab_smoke_sgd

tinynn/ab_smoke_sgd: tinynn/ab_smoke_sgd.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_sgd.rb -o tinynn/ab_smoke_sgd

# Forward-only smoke: does TransformerLM#forward run at current Spinel
# master? (The #473 SIGBUS is in backward; forward might be OK.)
forward-smoke: tinynn/forward_smoke
	./tinynn/forward_smoke

tinynn/forward_smoke: tinynn/forward_smoke.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/forward_smoke.rb -o tinynn/forward_smoke

persistent-bench: tinynn/persistent_bench
	./tinynn/persistent_bench

tinynn/persistent_bench: tinynn/persistent_bench.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/persistent_bench.rb -o tinynn/persistent_bench

persistent-bench-cuda: tinynn/persistent_bench_cuda
	./tinynn/persistent_bench_cuda

tinynn/persistent_bench_cuda: tinynn/persistent_bench_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/persistent_bench_cuda.rb -o tinynn/persistent_bench_cuda

persistent-bench-big: tinynn/persistent_bench_big
	./tinynn/persistent_bench_big

tinynn/persistent_bench_big: tinynn/persistent_bench_big.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/persistent_bench_big.rb -o tinynn/persistent_bench_big

# A/B parity test against CUDA backend on the local GPU (sm_121 / GB10).
# Requires `make setup-ggml-cuda` to have produced vendor/ggml/build-cuda.
ab-smoke-cuda: tinynn/ab_smoke_cuda
	./tinynn/ab_smoke_cuda

tinynn/tinynn_backend_cuda.o: tinynn/tinynn_backend_cuda.c
	$(CC) $(CFLAGS) $(GGML_INC) -I$(CUDA_DIR)/include -c $< -o $@

# Only the CUDA backend init goes into the CUDA archive. Common
# wrappers stay in tinynn_ggml.o (CPU archive), referenced from CUDA
# programs via a weak link. Avoids the multi-archive multi-definition
# linker conflict that older two-fat-archive layout had.
tinynn/libtinynn_ggml_cuda.a: tinynn/tinynn_backend_cuda.o
	ar $(ARFLAGS) $@ $<

tinynn/ab_smoke_cuda: tinynn/ab_smoke_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_cuda.rb -o tinynn/ab_smoke_cuda

# Consolidated CUDA parity test: matmul + add + gelu + rms_norm + softmax + scale + ffn_pipeline.
ab-smoke-all-cuda: tinynn/ab_smoke_all_cuda
	./tinynn/ab_smoke_all_cuda

tinynn/ab_smoke_all_cuda: tinynn/ab_smoke_all_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_all_cuda.rb -o tinynn/ab_smoke_all_cuda

# Transformer-shape parity + wallclock bench on CUDA (GB10).
ab-smoke-big-cuda: tinynn/ab_smoke_big_cuda
	./tinynn/ab_smoke_big_cuda

tinynn/ab_smoke_big_cuda: tinynn/ab_smoke_big_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_big_cuda.rb -o tinynn/ab_smoke_big_cuda

# --- maintenance ------------------------------------------------------------
clean:
	rm -f demos/train_minimal demos/train_tinystories \
	      demos/inference_demo demos/inference_demo_cuda \
	      demos/distilgpt2_demo demos/distilgpt2_demo_ffi \
	      demos/distilgpt2_demo_kv demos/distilgpt2_demo_text \
	      demos/distilgpt2_demo_ffi_cuda demos/distilgpt2_demo_kv_cuda \
	      tinynn/tinynn_ggml.o tinynn/libtinynn_ggml.a \
	      tinynn/tinynn_backend_cuda.o tinynn/libtinynn_ggml_cuda.a \
	      tinynn/smoke tinynn/ab_smoke tinynn/ab_smoke_cuda tinynn/ab_smoke_all_cuda \
	      tinynn/ab_smoke_add tinynn/ab_smoke_gelu tinynn/ab_smoke_rms_norm \
	      tinynn/ab_smoke_softmax tinynn/ab_smoke_transpose tinynn/ab_smoke_scale \
	      tinynn/ab_smoke_pipeline tinynn/ab_smoke_big tinynn/ab_smoke_big_cuda \
	      tinynn/ab_smoke_matmul_variants tinynn/ab_smoke_back tinynn/ab_smoke_embed \
	      tinynn/ab_smoke_sgd tinynn/ab_smoke_gelu_back tinynn/ab_smoke_cegrad \
	      tinynn/ab_smoke_adam tinynn/forward_smoke tinynn/persistent_bench \
	      tinynn/persistent_bench_cuda tinynn/persistent_bench_big

distclean: clean
	rm -rf $(GGML_DIR)/build $(GGML_DIR)/build-cuda

.PHONY: all clean distclean setup-ggml setup-ggml-cuda smoke \
        ab-smoke ab-smoke-add ab-smoke-gelu ab-smoke-rms-norm \
        ab-smoke-softmax ab-smoke-transpose ab-smoke-scale ab-smoke-silu \
        ab-smoke-mul ab-smoke-pipeline ab-smoke-big ab-smoke-cuda \
        ab-smoke-all-cuda ab-smoke-big-cuda test \
        gpt2 smollm2 smollm2_kv smollm2_kv_cuda \
        tinyllama tinyllama_kv tinyllama_kv_cuda \
        train algorithm_cards
