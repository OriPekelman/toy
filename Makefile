# toy build system.
#
# Demo / training Ruby drivers live in demos/ and compile to native
# binaries via Spinel. GPU acceleration is opt-in:
#
#   make                   # demos/train_minimal + demos/train_tinystories
#   make setup-ggml        # one-time clone + CPU build of vendored ggml
#   make setup-ggml-cuda   # one-time clone + CUDA build (needs CUDA toolkit)
#   make setup-ggml-metal  # one-time Metal build (macOS only)
#   make smoke             # tinynn FFI smoke test (4x3 ggml matmul demo)
#   make distilgpt2_demo_text  # → demos/distilgpt2_demo_text
#
# Vendored ggml lives at vendor/ggml/ (gitignored).
# The CUDA build expects sm_121 (NVIDIA GB10); override with
# GGML_CUDA_ARCH=NN on the command line.
# The Metal build uses GGML_METAL_EMBED_LIBRARY=ON so it works with
# Command Line Tools only (the xcrun metal / metallib compilers ship
# only with full Xcode). Kernels get JIT-compiled at first device load.

SPINEL_DIR  ?= $(HOME)/sites/spinel
SPINEL_BIN  ?= $(SPINEL_DIR)/spinel

# --- DevEx polish knobs (cosmetic, never gate correctness) ----------------
# QUIET=1 (default) routes known-harmless build chatter through the
# prep/quietly + prep/progress helpers so the terminal stays readable
# on a fresh clone. QUIET=0 disables all filtering (useful when chasing
# a Spinel codegen issue or a cmake misconfig).
#   - prep/quietly silences exact-substring patterns; exit code is
#     ALWAYS the child's, so real errors still propagate.
#   - prep/progress draws a single-line [NN%] bar over cmake/make's
#     own progress markers; full output is tee'd to a .log file in
#     vendor/ggml/. On non-zero exit it dumps the log tail to stderr.
QUIET    ?= 1
QUIETLY  := $(CURDIR)/prep/quietly
PROGRESS := $(CURDIR)/prep/progress
ifeq ($(QUIET),0)
  SPINEL = $(SPINEL_BIN)
else
  SPINEL = $(QUIETLY) \
      'cannot resolve call to' \
      'ignoring duplicate libraries' \
      -- $(SPINEL_BIN)
endif
# Sentinel deps so example/demo Spinel-compiled binaries get re-spun
# when the Spinel compiler itself changes. Without this, stale .o /
# .a in tinynn/ combined with newer Spinel C codegen can produce
# misaligned binaries that segfault at init (Tao hit this 2026-05-26
# after pulling Spinel 2183a92 — the lib archives weren't rebuilt).
SPINEL_DEPS := $(SPINEL_DIR)/spinel_analyze $(SPINEL_DIR)/spinel_codegen

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

# --- Tep sibling sync (spinelgems convention) -------------------------------
# Tep is co-developed with this repo (sibling at ~/sites/tep). As of
# 2026-05-27 we vendor it via the bundler-spinel / spinelgems
# convention (see docs/roadmap/spinelgems-tep-adoption-2026-05-27.md):
#
#   1. `bundle lock`                                  (Gemfile → Gemfile.lock)
#   2. `../spinelgems/exe/spinel-compat vendor`       (lock → vendor/spinel/)
#   3. `./prep/post_vendor_tep.rb`                    (@TEP_*@ rewrite)
#
# All three roll up into the `vendor-tep` target. Bails loud on
# missing libpq / libsqlite3 — same diagnostic as the retired
# prep/sync_tep.rb did.
#
# Sibling-checkout precheck: bundle lock will gladly write garbage
# into Gemfile.lock if ../tep doesn't exist (Gemfile uses
# `path: "../tep"`). Short-circuit with a clear message instead.
vendor-tep:
	@if [ ! -d ../tep ] || [ ! -d ../spinelgems ]; then \
	    echo ""; \
	    echo "  ✗ vendor-tep needs sibling checkouts (see docs/roadmap/spinelgems-tep-adoption-2026-05-27.md):"; \
	    [ -d ../tep ]        || echo "      missing: ../tep"; \
	    [ -d ../spinelgems ] || echo "      missing: ../spinelgems"; \
	    echo ""; \
	    echo "    From this directory's parent ($$(cd .. && pwd)):"; \
	    echo "      git clone https://github.com/OriPekelman/tep"; \
	    echo "      git clone https://github.com/OriPekelman/spinelgems"; \
	    echo ""; \
	    echo "    Or symlink existing checkouts (common dev layout in ~/sites):"; \
	    echo "      ln -s ~/sites/tep        ../tep"; \
	    echo "      ln -s ~/sites/spinelgems ../spinelgems"; \
	    echo ""; \
	    exit 1; \
	fi
	bundle lock
	SPINEL_DIR=$(HOME)/sites/spinel ../spinelgems/exe/spinel-compat vendor
	./prep/post_vendor_tep.rb

# Build vendor/spinel/tep/lib/tep.rb on demand for tep_demo/* targets.
# Triggers vendor-tep, which gates on sibling checkouts.
vendor/spinel/tep/lib/tep.rb:
	@$(MAKE) vendor-tep

# --- pure-Spinel drivers ----------------------------------------------------
# Source lives in demos/. We expose short top-level target names
# (`make train_minimal`, `make distilgpt2_demo_text`) that build into
# demos/. Run the resulting binaries from the repo root.
# `make` with no args = `make help`. Previously it ran `all` which
# triggered vendor-tep and failed on machines without ../tep checked
# out (DevEx footgun a fresh-clone-on-Mac hit 2026-05-28). `make help`
# is always safe + discoverable; `make all` still works for the
# original behaviour.
.DEFAULT_GOAL := help

all: demos/train demos/smollm2

# `make setup` auto-detects the best backend for this host and runs
# the right setup-ggml-* variant. macOS → metal; nvcc on PATH → cuda;
# else CPU. Sentinels in setup-ggml-* make this a no-op if already
# done. Saves new users from picking the wrong setup target.
.PHONY: setup

setup:
	@uname_s="$$(uname -s)"; \
	if [ "$$uname_s" = "Darwin" ]; then \
	    echo "[setup] macOS detected → setup-ggml + setup-ggml-metal"; \
	    echo "          (CPU examples link against vendor/ggml/build/;"; \
	    echo "           Metal examples link against vendor/ggml/build-metal/.)"; \
	    $(MAKE) setup-ggml; \
	    $(MAKE) setup-ggml-metal; \
	elif command -v nvcc >/dev/null 2>&1; then \
	    echo "[setup] nvcc on PATH → setup-ggml + setup-ggml-cuda"; \
	    $(MAKE) setup-ggml; \
	    $(MAKE) setup-ggml-cuda; \
	else \
	    echo "[setup] CPU only → setup-ggml"; \
	    $(MAKE) setup-ggml; \
	fi; \
	echo ""; \
	echo "Done. Next: run 'make help' for the entry points."

# --- `make hello` — guided first-run experience ----------------------------
# One command from a fresh clone to "I see tokens on the screen":
#   1. setup the backend (CPU + Metal/CUDA if detected) — idempotent
#   2. convert HuggingFaceTB/SmolLM2-135M (the base, not Instruct) to
#      data/smollm2-135m-f32.gguf via the project's own converter, so
#      raw completion prompts produce coherent text. Requires `uv`
#      (autoinstalls deps inline).
#   3. build example_inference (Metal on macOS, CPU elsewhere) and run
#      it with a default prompt.
# Each step is a no-op if its output already exists. Safe to re-run.
.PHONY: hello _hello_model _hello_run

hello:
	@echo "▶ toy hello — guided first-run"
	@$(MAKE) -s setup
	@$(MAKE) -s _hello_model
	@$(MAKE) -s _hello_run
	@echo ""
	@echo "▶ done. Next:  make help     (see all entry points)"

_hello_model:
	@if [ ! -e data/smollm2-135m-f32.gguf ]; then \
	    if ! command -v uv >/dev/null 2>&1; then \
	        echo "[hello] uv not found. Install from https://docs.astral.sh/uv/"; \
	        echo "[hello] (or fetch any GGUF via prep/fetch_model.sh and re-run)."; \
	        exit 1; \
	    fi; \
	    echo "[hello] converting HuggingFaceTB/SmolLM2-135M → data/smollm2-135m-f32.gguf"; \
	    echo "[hello]   (first time pulls ~270 MB safetensors via huggingface_hub; ~30 s)"; \
	    ./prep/convert_smollm2_to_gguf.py --with-tokenizer >prep/_convert.log 2>&1 || { \
	        echo "[hello] converter failed — tail of prep/_convert.log:"; \
	        tail -30 prep/_convert.log; \
	        exit 1; \
	    }; \
	    echo "[hello] data/smollm2-135m-f32.gguf ready"; \
	else \
	    echo "[hello] data/smollm2-135m-f32.gguf already present"; \
	fi

_hello_run:
ifeq ($(UNAME_S),Darwin)
	@$(MAKE) -s example_inference_metal
	@echo ""
	@echo "[hello] running example_inference_metal — Metal-accelerated"
	@echo "[hello]   (first Metal run JIT-compiles kernels, ~15 s, then cached)"
	@echo ""
	@GGML_LOG_LEVEL=2 PROMPT="Once upon a time" $(QUIETLY) \
	    '^ggml_metal_' \
	    -- ./examples/example_inference_metal
else
	@$(MAKE) -s example_inference
	@echo ""
	@echo "[hello] running example_inference — CPU"
	@echo ""
	@PROMPT="Once upon a time" ./examples/example_inference
endif

# --- help / time-to-joy entry points --------------------------------------
# `make help` is the discoverable index for someone who just cloned.
# Keep it short — pointers to the heavier docs (examples/README.md,
# tep_demo/README.md, docs/INDEX.md) for the details.

.PHONY: help

help:
	@echo ""
	@echo "  toy — a transformer LM in Ruby, Spinel-compiled."
	@echo "  Full docs: README.md, examples/README.md, docs/INDEX.md."
	@echo ""
	@if [ ! -f vendor/ggml/build/src/libggml.a ] && \
	    [ ! -f vendor/ggml/build-metal/src/libggml.a ] && \
	    [ ! -f vendor/ggml/build-cuda/src/libggml.a ]; then \
	    echo "  ▶ FIRST TIME HERE?"; \
	    echo "      make hello              one-shot: setup + fetch a model + run inference"; \
	    echo "      make setup              just build the backend (CPU + Metal/CUDA if detected)"; \
	    echo ""; \
	fi
	@echo "  ONE-TIME SETUP"
	@echo "    make setup               auto-detect platform; pick CUDA/Metal/CPU"
	@echo "    make setup-ggml          force CPU build (~2 min)"
	@echo "    make setup-ggml-cuda     force CUDA backend"
	@echo "    make setup-ggml-metal    force Metal backend (macOS)"
	@echo ""
	@if [ "$$(uname -s)" = "Darwin" ]; then \
	    echo "  ⚡ macOS detected — for GPU acceleration use the _metal example"; \
	    echo "      variants below (they link against libggml-metal + KV kernels)."; \
	    echo "      The plain example_inference still works but is CPU-only."; \
	    echo ""; \
	fi
	@echo "  GETTING STARTED — examples/"
	@echo "    make example_list_models           list GGUFs cached locally / in HF / Ollama / LM Studio"
	@echo "    make example_inference             load a GGUF, generate 16 tokens (CPU)"
	@if [ "$$(uname -s)" = "Darwin" ]; then \
	    echo "    make example_inference_metal       same, Metal-accelerated (macOS) — use this on Mac"; \
	fi
	@echo "    make example_train                 tiny GPT trained from scratch on TinyStories"
	@echo "    make example_train_from_scratch    modern Llama-shape from-scratch trainer (CPU + CUDA)"
	@echo "    make example_finetune              LoRA / QLoRA fine-tune on a GGUF base"
	@echo "    make example_train_vit_tiny        ViT-Tiny image-classifier + warm-start from timm"
	@echo "    make example_warm_start_train      Qwen-style warm-start trainer with projection lens"
	@echo "    make example_lmc                   linear-mode-connectivity blend of two checkpoints"
	@echo ""
	@echo "  HTTP SERVING — tep_demo/"
	@echo "    make tep_demo/openai_api_llama     OpenAI-compatible server for any llama-family GGUF"
	@echo "                                       MODEL_PATH=… MODEL_NAME=… ./tep_demo/openai_api_llama -p 4567"
	@echo "    make tep_demo/hello                minimal Tep HTTP smoke"
	@if [ ! -f vendor/spinel/tep/lib/tep.rb ]; then \
	    printf "    (prereq: run %s first — needs ../tep + ../spinelgems checkouts)\n" "'make vendor-tep'"; \
	fi
	@if [ "$$(uname -s)" = "Darwin" ]; then \
	    echo ""; \
	    echo "    Note: tep_demo/openai_api_llama uses the CPU FFI bridge — Metal serving"; \
	    echo "    is a separate codepath (lib/tinynn_metal.rb) not yet wired into the HTTP"; \
	    echo "    server. CPU serving works fine on Mac, just won't hit the Metal kernels."; \
	fi
	@echo ""
	@echo "  BENCH + CHECKS"
	@echo "    make bench                         routine perf regression gate (vs bench/baselines.csv)"
	@echo "    make bench-vs-pytorch              same workloads, gated vs PyTorch (ratio, not absolute ms)"
	@echo "    make coverage                      regenerate the ggml-op coverage matrix"
	@echo "    make coverage-check                CI form (no diff means in sync)"
	@echo "    make test                          all tinynn FFI smoke binaries"
	@echo ""
	@echo "  COMMON MAKE FLAGS"
	@echo "    DEVICE=cuda                        on example_train_from_scratch / example_finetune_cuda"
	@echo "    GGUF=path/to/model.gguf            on example_inference / example_finetune"
	@echo ""

# --- examples/ getting-started entry points --------------------------------
# Compact, one-file demos covering the main use cases. See
# examples/README.md.
examples/example_inference: examples/01_inference.rb lib/arch.rb lib/transformer_lm.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb lib/tokenizer.rb lib/model_index.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@
example_inference: examples/example_inference

# toy#gguf-checkpoint-reload (#153) — smoke binary that loads a
# from-scratch toy GGUF and runs a tiny generation. No tokenizer.
examples/smoke_toy_ckpt_reload: examples/smoke_toy_ckpt_reload.rb lib/arch.rb lib/transformer_lm.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb lib/tokenizer.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# toy#embed-api (#145) — smoke for ToyLM#embed_lookup.
examples/smoke_embed_api: examples/smoke_embed_api.rb lib/arch.rb lib/transformer_lm.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb lib/tokenizer.rb lib/toy_logprobs.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# toy#decode-logprobs (#151) — smoke for ToyLM#decode_step_with_logprobs.
examples/smoke_decode_logprobs: examples/smoke_decode_logprobs.rb lib/arch.rb lib/transformer_lm.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb lib/tokenizer.rb lib/toy_logprobs.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# GH#18 — LMC interpolate-and-eval runner.
examples/example_lmc: examples/08_lmc.rb lib/llama_seq_forward_ffi.rb lib/toy_smollm2.rb lib/transformer.rb lib/tinynn.rb lib/toy_drift_grad.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
example_lmc: examples/example_lmc

# E2.3 (towards GH#14) — projection-lens smoke.
examples/smoke_projection_lens: examples/smoke_projection_lens.rb lib/llama_seq_forward_ffi.rb lib/toy_smollm2.rb lib/transformer.rb lib/tinynn.rb lib/toy_drift_grad.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# E2.4 (towards GH#14) — streaming corpus loader + cosine LR smoke.
examples/smoke_corpus_loader: examples/smoke_corpus_loader.rb lib/transformer.rb lib/tinynn.rb lib/toy_corpus_loader.rb lib/toy_lr_schedule.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# E2.5 (towards GH#14) — warm-start training driver.
examples/example_warm_start_train: examples/09_warm_start_train.rb lib/llama_seq_forward_ffi.rb lib/toy_smollm2.rb lib/transformer.rb lib/tinynn.rb lib/toy_drift_grad.rb lib/toy_gguf_writer.rb lib/toy_corpus_loader.rb lib/toy_lr_schedule.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
example_warm_start_train: examples/example_warm_start_train

# Auto-generated coverage matrix — ggml ops vs our FFI surface.
# Sources are vendor/ggml/include/ggml.h, tinynn/tinynn_ggml.c, and the
# two FFI binding files. See docs/coverage.md for the matrix.
coverage: docs/coverage.md
docs/coverage.md: prep/gen_coverage.rb vendor/ggml/include/ggml.h \
                  tinynn/tinynn_ggml.c lib/tinynn.rb lib/tinynn_cuda.rb \
                  lib/tinynn_metal.rb
	ruby prep/gen_coverage.rb
coverage-check:
	ruby prep/gen_coverage.rb --check
.PHONY: coverage coverage-check

examples/example_train: examples/02_train_custom_gpt.rb lib/transformer.rb lib/training.rb lib/toy_trainer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@
example_train: examples/example_train

examples/example_finetune: examples/03_finetune_lora.rb lib/llama_seq_forward_ffi.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@
example_finetune: examples/example_finetune

# CUDA mirror — same source, swap TinyNN → TinyNNCuda by including
# both libs. The example source uses TinyNN; the CUDA build link-step
# carries CUDA symbols too (no source change). For real GPU speedup
# users typically write a `_cuda` variant; this mirror is for the
# build-recipe story.
examples/example_finetune_cuda: examples/03_finetune_lora_cuda.rb lib/llama_seq_forward_ffi_cuda.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
example_finetune_cuda: examples/example_finetune_cuda

# Metal mirror of example_inference (macOS only). Uses TinyNNMetal.
# Same -Wl,-u trick as CUDA so the Metal backend init survives
# weak-symbol resolution. macOS expects a leading underscore on
# external symbols, hence `-Wl,-u,_tnn_metal_force_link`.
# Frameworks (Foundation/Metal/MetalKit) are linked via -framework.
examples/example_inference_metal: examples/01_inference_metal.rb lib/arch.rb lib/transformer_lm_metal.rb lib/toy_smollm2_ffi_kv_metal.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn_metal.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_metal.a
ifneq ($(UNAME_S),Darwin)
	@echo "example_inference_metal: macOS-only"; exit 1
endif
	$(SPINEL) --cc='cc -Wl,-u,_tnn_metal_force_link -framework Foundation -framework Metal -framework MetalKit' $< -o $@
example_inference_metal: examples/example_inference_metal

examples/example_serve: examples/04_serve_http.rb vendor/spinel/tep/lib/tep.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@
example_serve: examples/example_serve

examples/example_list_models: examples/05_list_models.rb lib/model_index.rb lib/arch.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@
example_list_models: examples/example_list_models

# DEVICE-aware entry point. Toy's per-backend Spinel binaries can't
# share a Ruby file (poly-dispatch landmines on LlamaSeqForwardFFICache
# vs *Cuda), so the entry point is a shell-script dispatcher.
#
# Today only DEVICE=cpu is supported for from-scratch training:
# LlamaSeqForwardFFICacheCuda implements realize_for_mmap (LoRA /
# fine-tune from a base GGUF) but NOT realize_for_random_init.
# Adding CUDA random-init is a real feature — tracked under
# toy#train-device-select-cuda follow-up. The dispatcher errors
# cleanly on DEVICE=cuda so Tao's `run_start.backend.kind=="cuda"`
# acceptance fails honestly rather than silently emitting cpu data.
examples/example_train_from_scratch_cpu: examples/06_train_from_scratch.rb lib/llama_seq_forward_ffi.rb lib/toy_smollm2.rb lib/transformer.rb lib/tinynn.rb lib/toy_describe_flow.rb lib/toy_drift_grad.rb lib/toy_gguf_writer.rb lib/toy_tap.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
examples/example_train_from_scratch_cuda: examples/06_train_from_scratch_cuda.rb lib/llama_seq_forward_ffi_cuda.rb lib/toy_smollm2.rb lib/transformer.rb lib/tinynn_cuda.rb lib/toy_describe_flow.rb lib/toy_drift_grad.rb lib/toy_gguf_writer.rb lib/toy_tap.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS)
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
examples/example_train_from_scratch: examples/example_train_from_scratch_cpu
	@printf '#!/bin/sh\n# Auto-generated by Makefile. DEVICE selects the backend binary.\n# Edit examples/06_train_from_scratch.rb (cpu) for behaviour; CUDA mirror is auto-generated by prep/gen_cuda_mirror.rb.\ncase "$${DEVICE:-cpu}" in\n  cpu|"") exec "$$(dirname "$$0")/example_train_from_scratch_cpu" "$$@" ;;\n  cuda)   exec "$$(dirname "$$0")/example_train_from_scratch_cuda" "$$@" ;;\n  metal)  echo "DEVICE=metal not yet supported for training (inference only)" >&2; exit 2 ;;\n  *)      echo "DEVICE=$${DEVICE} not recognised (want cpu|cuda)" >&2; exit 2 ;;\nesac\n' > $@
	@chmod +x $@
example_train_from_scratch: examples/example_train_from_scratch
example_train_from_scratch_cuda: examples/example_train_from_scratch_cuda

examples: example_inference example_train example_finetune example_finetune_cuda example_serve example_list_models

# Phase 0.6 — CUDA-mirror generator. The CPU file is the source of
# truth; the CUDA file is auto-generated by prep/gen_cuda_mirror.rb.
# `make verify-mirrors` exits non-zero if any committed CUDA mirror
# has drifted from what the generator would produce.
gen-mirrors:
	@ruby prep/gen_cuda_mirror.rb

verify-mirrors:
	@ruby prep/gen_cuda_mirror.rb --verify

# Parity-checks vs native TransformerLM.forward.

# Tep+Spinel HTTP server demos. See tep_demo/README.md. Builds bypass
# tep's translator (we use spinel directly on the spinelgems-vendored
# tep tree at vendor/spinel/tep/lib/, produced by `make vendor-tep`).
tep_demo/hello: tep_demo/hello_api.rb vendor/spinel/tep/lib/tep.rb
	$(SPINEL) tep_demo/hello_api.rb -o tep_demo/hello

# Inference API: /generate?n=N runs greedy generation via FullForwardFFICache.
tep_demo/api: tep_demo/inference_api.rb vendor/spinel/tep/lib/tep.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tep_demo/inference_api.rb -o tep_demo/api

# OpenAI-compatible API backed by SmolLM2/Qwen-family models via the
# direct GGUF→FFI loader. Accepts pre-tokenized integer IDs (no
# server-side tokenizer — keeps the single-binary deployment story).
# Model GGUF path comes from MODEL_PATH env at run time; one binary
# serves any llama-family GGUF. GH#188 consolidated from 7 near-
# duplicate per-model sources (openai_api_smollm2.rb + 6 Qwen2.5
# variants). Run:
#   make tep_demo/openai_api_llama
#   MODEL_PATH=data/qwen25-1.5b-native-q8.gguf MODEL_NAME=qwen25-1.5b-q8 \
#     ./tep_demo/openai_api_llama -p 4567 -w 1
tep_demo/openai_api_llama: tep_demo/openai_api_llama.rb vendor/spinel/tep/lib/tep.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tep_demo/openai_api_llama.rb -o tep_demo/openai_api_llama

# --- ggml vendor ------------------------------------------------------------
# Vendor patches that must land before any ggml build target. See
# vendor-patches/README.md for the per-patch rationale.
GGML_PATCHES := \
	vendor-patches/0001-cuda-buffer_from_ptr.patch \
	vendor-patches/0002-cuda-buffer_from_ptr-reuse-iface.patch \
	vendor-patches/0003-cuda-buffer_from_ptr-copy-mode.patch \
	vendor-patches/0004-cuda-cpy-strided.patch \
	vendor-patches/0005-concat-backward.patch \
	vendor-patches/0006-getrows-back-large-vocab.patch

# Sentinel file marking that all $(GGML_PATCHES) have been applied to
# the vendored tree. Build targets depend on it through CMakeLists.txt
# (which depends on this sentinel) so a fresh clone applies the patches
# exactly once, and re-runs of `make setup-ggml` are no-ops as long as
# the patch set is unchanged.
$(GGML_DIR)/.patched: $(GGML_DIR)/CMakeLists.txt $(GGML_PATCHES)
	@echo "  reset vendor/ggml to upstream HEAD (build/ untouched)"
	@cd $(GGML_DIR) && git reset --hard HEAD >/dev/null
	@cd $(GGML_DIR) && for p in $(GGML_PATCHES); do \
	  echo "  apply $$p"; \
	  git apply "$(CURDIR)/$$p" || { echo "    FAILED"; exit 1; }; \
	done
	touch $@

$(GGML_DIR)/CMakeLists.txt:
	mkdir -p vendor
	git clone --depth 1 $(GGML_REPO) $(GGML_DIR)

# GGML_OPENMP=OFF: avoid the libgomp link dependency. On macOS clang
# ships libomp (LLVM), not libgomp (GNU); ggml's own thread pool covers
# CPU parallelism either way. Same setting used on Linux for build
# parity (and so lib/tinynn.rb doesn't need ffi_lib "gomp").
#
# Build output is routed through prep/progress, which:
#   - tees full cmake/build output to vendor/ggml/<dir>.log
#   - draws a one-line [NN%] progress bar on a TTY (plain "[NN%] msg"
#     lines on CI / non-tty stdout, no overdraw)
#   - on non-zero exit, dumps the last 40 lines of the log + exits
#     with the child's status. NEVER swallows errors.
# Disable with QUIET=0 (passes through stdout unchanged).
# (PROGRESS / QUIET / QUIETLY are defined near the top of this file
# alongside SPINEL_BIN — see the DevEx polish knobs block.)

# Helper: run a `cd $(GGML_DIR) && cmake -B <DIR> <FLAGS>` configure
# step. Routes output to a logfile when QUIET=1; on failure dumps the
# log tail and propagates the exit code. QUIET=0 passes through.
# Args: $(1) = build dir name (build / build-metal / build-cuda)
#       $(2) = cmake invocation (everything after the cd)
define ggml_configure
	@if [ "$(QUIET)" = "1" ]; then \
	    log="$(CURDIR)/$(GGML_DIR)/$(1).config.log"; \
	    ( cd $(GGML_DIR) && $(2) ) >"$$log" 2>&1 || { \
	        echo "  ✗ cmake configure ($(1)) failed; tail of $$log:"; \
	        tail -30 "$$log"; exit 1; }; \
	else \
	    cd $(GGML_DIR) && $(2) ; \
	fi
endef

# Helper: run a `cmake --build <DIR> -j<N>` step. Routes through
# prep/progress when QUIET=1 (single-line [NN%] bar, log tee). QUIET=0
# passes through.
# Args: $(1) = build dir name; $(2) = label tag (cpu/metal/cuda);
#       $(3) = cmake --build command
define ggml_build
	@if [ "$(QUIET)" = "1" ]; then \
	    LOG="$(CURDIR)/$(GGML_DIR)/$(1).build.log" LABEL="ggml-$(2)" \
	        $(PROGRESS) -- sh -c "cd $(GGML_DIR) && $(3)"; \
	else \
	    cd $(GGML_DIR) && $(3) ; \
	fi
endef

# setup-ggml-* targets are user-facing phonies; the real work happens
# in the libggml.a sentinel rules below so re-running setup is a no-op
# once the static archive is built. Lets `make hello` chain through
# without redoing the ~5 s incremental cmake check on every invocation.
.PHONY: setup-ggml setup-ggml-cuda setup-ggml-metal

setup-ggml: $(GGML_DIR)/build/src/libggml.a
setup-ggml-cuda: $(GGML_DIR)/build-cuda/src/libggml.a
setup-ggml-metal: $(GGML_DIR)/build-metal/src/libggml.a

$(GGML_DIR)/build/src/libggml.a: $(GGML_DIR)/.patched
	@echo "  → configure  ggml (cpu)"
	$(call ggml_configure,build,$(CMAKE_ENV) cmake -B build \
	  -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON \
	  -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF \
	  -DGGML_OPENCL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_ACCELERATE=OFF \
	  -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF \
	  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON)
	@echo "  → build      ggml (cpu, $(NJOBS) jobs)"
	$(call ggml_build,build,cpu,$(CMAKE_ENV) cmake --build build -j$(NJOBS))

$(GGML_DIR)/build-cuda/src/libggml.a: $(GGML_DIR)/.patched
	@echo "  → configure  ggml (cuda, sm_$(GGML_CUDA_ARCH))"
	$(call ggml_configure,build-cuda,PATH=$(CUDA_DIR)/bin:$$PATH $(CMAKE_ENV) cmake -B build-cuda \
	  -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON \
	  -DGGML_CUDA=ON -DGGML_METAL=OFF -DGGML_VULKAN=OFF \
	  -DGGML_OPENCL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_ACCELERATE=OFF \
	  -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF \
	  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
	  -DCMAKE_CUDA_ARCHITECTURES=$(GGML_CUDA_ARCH) -DGGML_NATIVE=OFF)
	@echo "  → build      ggml (cuda, $(NJOBS) jobs)"
	$(call ggml_build,build-cuda,cuda,PATH=$(CUDA_DIR)/bin:$$PATH $(CMAKE_ENV) cmake --build build-cuda -j$(NJOBS))

# Metal build (macOS only). GGML_METAL_EMBED_LIBRARY=ON bakes the
# .metal shader source into the static archive as raw bytes; the
# Metal driver JIT-compiles it on first device load. This lets the
# whole pipeline work with the Command Line Tools (xcrun metal /
# metallib are full-Xcode-only). On a Mac with full Xcode you can
# flip GGML_METAL_EMBED_LIBRARY=OFF for AOT-compiled kernels.
$(GGML_DIR)/build-metal/src/libggml.a: $(GGML_DIR)/.patched
ifneq ($(UNAME_S),Darwin)
	@echo "setup-ggml-metal: Metal is macOS-only (uname -s = $(UNAME_S))"; exit 1
endif
	@echo "  → configure  ggml (metal)"
	$(call ggml_configure,build-metal,$(CMAKE_ENV) cmake -B build-metal \
	  -DBUILD_SHARED_LIBS=OFF -DGGML_STATIC=ON \
	  -DGGML_CUDA=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
	  -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF -DGGML_BLAS=OFF \
	  -DGGML_OPENMP=OFF -DGGML_ACCELERATE=OFF \
	  -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF \
	  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON)
	@echo "  → build      ggml (metal, $(NJOBS) jobs)"
	$(call ggml_build,build-metal,metal,$(CMAKE_ENV) cmake --build build-metal -j$(NJOBS))

# --- tinynn shim (CPU build) ------------------------------------------------
GGML_INC := -I$(GGML_DIR)/include -I$(GGML_DIR)/src

tinynn/tinynn_ggml.o: tinynn/tinynn_ggml.c tinynn/tinynn_ggml.h tinynn/tinynn_trace.h
	$(CC) $(CFLAGS) $(GGML_INC) -c $< -o $@

tinynn/tinynn_gguf.o: tinynn/tinynn_gguf.c tinynn/tinynn_gguf.h
	$(CC) $(CFLAGS) $(GGML_INC) -c $< -o $@

tinynn/tinynn_trace.o: tinynn/tinynn_trace.c tinynn/tinynn_trace.h
	$(CC) $(CFLAGS) -c $< -o $@

tinynn/tinynn_events.o: tinynn/tinynn_events.c tinynn/tinynn_events.h
	$(CC) $(CFLAGS) -c $< -o $@

tinynn/libtinynn_ggml.a: tinynn/tinynn_ggml.o tinynn/tinynn_gguf.o tinynn/tinynn_trace.o tinynn/tinynn_events.o
	ar $(ARFLAGS) $@ tinynn/tinynn_ggml.o tinynn/tinynn_gguf.o tinynn/tinynn_trace.o tinynn/tinynn_events.o

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

# E1.1 / GH#13 — Conv2D smoke + JSON dump for PyTorch parity.
tinynn/ab_smoke_conv2d: tinynn/ab_smoke_conv2d.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_conv2d.rb -o tinynn/ab_smoke_conv2d

# E1.2 / GH#13 — patch_embed composite smoke + parity dump.
tinynn/ab_smoke_patch_embed: tinynn/ab_smoke_patch_embed.rb lib/transformer.rb lib/tinynn.rb lib/toy_vit.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_patch_embed.rb -o tinynn/ab_smoke_patch_embed

# E1.3 / GH#13 — ViT-Tiny forward + training smoke.
examples/smoke_vit_tiny: examples/smoke_vit_tiny.rb lib/vit_tiny_forward_ffi.rb lib/toy_vit.rb lib/toy_smollm2.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# E1.5 / GH#13 — image-loader smoke.
examples/smoke_image_loader: examples/smoke_image_loader.rb lib/toy_image_loader.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# E1.6 / GH#13 — ViT-Tiny training driver.
examples/example_train_vit_tiny: examples/07_train_vit_tiny.rb lib/vit_tiny_forward_ffi.rb lib/toy_vit.rb lib/toy_smollm2.rb lib/toy_image_loader.rb lib/toy_lr_schedule.rb lib/toy_drift_grad.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
example_train_vit_tiny: examples/example_train_vit_tiny

tinynn/ab_smoke_gelu: tinynn/ab_smoke_gelu.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_gelu.rb -o tinynn/ab_smoke_gelu

tinynn/ab_smoke_rms_norm: tinynn/ab_smoke_rms_norm.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_rms_norm.rb -o tinynn/ab_smoke_rms_norm

tinynn/ab_smoke_softmax: tinynn/ab_smoke_softmax.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_softmax.rb -o tinynn/ab_smoke_softmax

tinynn/ab_smoke_flash_attn: tinynn/ab_smoke_flash_attn.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_flash_attn.rb -o tinynn/ab_smoke_flash_attn

tinynn/ab_smoke_q8_kv: tinynn/ab_smoke_q8_kv.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_q8_kv.rb -o tinynn/ab_smoke_q8_kv

tinynn/ab_smoke_moe_ffn: tinynn/ab_smoke_moe_ffn.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_moe_ffn.rb -o tinynn/ab_smoke_moe_ffn

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

# A/B harness for the "fuse-or-not" question: N_HEADS small matmuls vs
# 1 batched matmul at LoRA-Q shape. Override D_MODEL / N_HEADS / R / T
# via env to sweep launch-overhead vs compute-bound regimes. See
# docs/heavy-train-attribution-2026-05-24.md.
ab-smoke-lora-fused-cuda: tinynn/ab_smoke_lora_fused_cuda
	./tinynn/ab_smoke_lora_fused_cuda

tinynn/ab_smoke_lora_fused_cuda: tinynn/ab_smoke_lora_fused_cuda.rb lib/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' tinynn/ab_smoke_lora_fused_cuda.rb -o tinynn/ab_smoke_lora_fused_cuda

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
demos/qwen25_bench_cuda: demos/qwen25_bench_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS)
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F1.2 step 2: LoRA-Q forward-parity gate. Loads SmolLM2-135M twice
# (baseline + LoRA r=16 B=0), asserts bit-identical generated IDs.
smollm2_lora_forward:        demos/smollm2_lora_forward
demos/smollm2_lora_forward: demos/smollm2_lora_forward.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# F1.2 step 3: backward through the full SmolLM2 decode graph,
# layer-0 LoRA-Q updated via SGD. Requires the vendored CONCAT
# backward in vendor/ggml/src/ggml.c.
smollm2_lora_train_step:        demos/smollm2_lora_train_step
demos/smollm2_lora_train_step: demos/smollm2_lora_train_step.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# F1.2 step 4: all-layers LoRA-Q SGD on real CE loss against a rare
# target token. 540 opt_step nodes (30 layers × 9 heads × 2 params).
# Acceptance: monotonic decrease over 20 steps.
smollm2_lora_train_ce:        demos/smollm2_lora_train_ce
demos/smollm2_lora_train_ce: demos/smollm2_lora_train_ce.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# F2 step 1: CUDA mirror of the LoRA forward parity gate.
smollm2_lora_forward_cuda:        demos/smollm2_lora_forward_cuda
demos/smollm2_lora_forward_cuda: demos/smollm2_lora_forward_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F2 step 2: CUDA mirror of the multi-layer SGD CE training smoke.
smollm2_lora_train_ce_cuda:        demos/smollm2_lora_train_ce_cuda
demos/smollm2_lora_train_ce_cuda: demos/smollm2_lora_train_ce_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# Task #70 diagnostic — same CE smoke but with every graph_b node
# pinned. Confirms sched intermediate-grad aliasing is the CPU
# divergence's root cause. See docs/design/task70-root-cause-2026-05-21.md.
smollm2_lora_train_ce_pinned:        demos/smollm2_lora_train_ce_pinned
demos/smollm2_lora_train_ce_pinned: demos/smollm2_lora_train_ce_pinned.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# F1.2 step 5: AdamW training with per-step m/v preservation via
# tnn_graph_reset_grads_only. Converges 7.5 → 0.09 in 20 SGD steps
# at LR=1e-3 — proper SFT-shaped learning curve.
smollm2_lora_train_adamw_cuda:        demos/smollm2_lora_train_adamw_cuda
demos/smollm2_lora_train_adamw_cuda: demos/smollm2_lora_train_adamw_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F1.2 step 6a: multi-target AdamW SFT-shaped training. Cycles through
# 5 target tokens × 10 epochs at the same prefix; expects loss to
# drop on average + per-target. 10.8 → 3.6 in 10 epochs. Foundation
# for step 6b (multi-position) and step 7 (real alpaca dataset).
smollm2_lora_sft_multi_cuda:        demos/smollm2_lora_sft_multi_cuda
demos/smollm2_lora_sft_multi_cuda: demos/smollm2_lora_sft_multi_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F1.2 step 6b — multi-position SFT (cycle pos4 / pos5). Validates
# that persistent Adam m/v (allocated by enable_lora_q_adamw! +
# realize_for_mmap) survive tnn_reset_for_rebuild between cycles.
smollm2_lora_sft_multipos_cuda:        demos/smollm2_lora_sft_multipos_cuda
demos/smollm2_lora_sft_multipos_cuda: demos/smollm2_lora_sft_multipos_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# M3 step 1 — sequence-mode forward parity at T=1.
# LlamaSeqForwardFFICache.forward([id], [0]) must match
# SmolLM2KVFFICache + decode_step(id, 0). See
# docs/design/m3-seq-forward-2026-05-21.md.
smollm2_seq_parity:        demos/smollm2_seq_parity
demos/smollm2_seq_parity: demos/smollm2_seq_parity.rb lib/llama_seq_forward_ffi.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# M3 step 2 — T=4 trajectory parity (CPU). Per-position seq logits must
# match the decode_step trajectory; proves causal-mask + multi-pos RoPE.
smollm2_seq_parity_t4:        demos/smollm2_seq_parity_t4
demos/smollm2_seq_parity_t4: demos/smollm2_seq_parity_t4.rb lib/llama_seq_forward_ffi.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# M3 step 2 — CUDA mirror. T=1 and T=4 vs CPU decode_step trajectory.
smollm2_seq_parity_cuda:        demos/smollm2_seq_parity_cuda
demos/smollm2_seq_parity_cuda: demos/smollm2_seq_parity_cuda.rb lib/llama_seq_forward_ffi_cuda.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

smollm2_seq_parity_t4_cuda:        demos/smollm2_seq_parity_t4_cuda
demos/smollm2_seq_parity_t4_cuda: demos/smollm2_seq_parity_t4_cuda.rb lib/llama_seq_forward_ffi_cuda.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# M3 step 3 — seq-mode LoRA training smoke (CPU). One forward + backward
# + opt_step over T positions; loss should decrease over N steps.
smollm2_seq_train:        demos/smollm2_seq_train
demos/smollm2_seq_train: demos/smollm2_seq_train.rb lib/llama_seq_forward_ffi.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

smollm2_seq_train_cuda:        demos/smollm2_seq_train_cuda
demos/smollm2_seq_train_cuda: demos/smollm2_seq_train_cuda.rb lib/llama_seq_forward_ffi_cuda.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F3 — full fine-tune on CUDA. Every per-block weight tensor is
# writable F32 + AdamW state; opt_step on each. See
# docs/roadmap/f3-full-finetune-2026-05-21.md.
smollm2_seq_full_finetune_cuda:        demos/smollm2_seq_full_finetune_cuda
demos/smollm2_seq_full_finetune_cuda: demos/smollm2_seq_full_finetune_cuda.rb lib/llama_seq_forward_ffi_cuda.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F4 (QLoRA) on CUDA via realize_for_q8_copy. Q8 base in standard
# CUDA buffer + F32 LoRA adapter; bypasses the BYO-pointer padding bug.
smollm2_seq_qlora_cuda:        demos/smollm2_seq_qlora_cuda
demos/smollm2_seq_qlora_cuda: demos/smollm2_seq_qlora_cuda.rb lib/llama_seq_forward_ffi_cuda.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# Training step-time bench. MODE=lora|ft; STEPS=N; GGUF=path.
seq_train_bench_cuda:        demos/seq_train_bench_cuda
demos/seq_train_bench_cuda: demos/seq_train_bench_cuda.rb lib/llama_seq_forward_ffi_cuda.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS)
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# Per-phase training-step bench (CPU + CUDA). Times graph_reset /
# uploads / compute_backward / download separately. Doc:
# docs/design/bench-train-2026-05-21.md.
smollm2_lora_train_bench:        demos/smollm2_lora_train_bench
demos/smollm2_lora_train_bench: demos/smollm2_lora_train_bench.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

smollm2_lora_train_bench_cuda:        demos/smollm2_lora_train_bench_cuda
demos/smollm2_lora_train_bench_cuda: demos/smollm2_lora_train_bench_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# Task #70 grad-magnitude probes (per-layer maxabs(grad_A), maxabs(grad_B)).
smollm2_lora_grad_probe:        demos/smollm2_lora_grad_probe
demos/smollm2_lora_grad_probe: demos/smollm2_lora_grad_probe.rb lib/toy_smollm2_ffi_kv.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

smollm2_lora_grad_probe_cuda:        demos/smollm2_lora_grad_probe_cuda
demos/smollm2_lora_grad_probe_cuda: demos/smollm2_lora_grad_probe_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy_smollm2_loader.rb lib/transformer.rb lib/gpt2.rb lib/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
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

# F1.2 step 1: multi-step LoRA convergence via the F1.1 in-graph
# optimizer. Toy shape; SGD; 60 steps; asserts final loss < 10% of
# initial (passes at ~10e-13 of initial).
ab-smoke-lora-train: tinynn/ab_smoke_lora_train
	./tinynn/ab_smoke_lora_train

tinynn/ab_smoke_lora_train: tinynn/ab_smoke_lora_train.rb lib/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_lora_train.rb -o tinynn/ab_smoke_lora_train

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

# Metal backend mirror — same archive-isolation pattern as CUDA. The
# source is .m (Objective-C) since the Metal frameworks are ObjC; we
# compile with -fobjc-arc off (the file holds no ObjC objects of its
# own, just a C function calling into ggml-metal). Header search adds
# the Metal build dir so ggml-metal.h is reachable.
tinynn/tinynn_backend_metal.o: tinynn/tinynn_backend_metal.m
	$(CC) $(CFLAGS) -x objective-c $(GGML_INC) -c $< -o $@

tinynn/libtinynn_ggml_metal.a: tinynn/tinynn_backend_metal.o
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
	      tinynn/tinynn_backend_metal.o tinynn/libtinynn_ggml_metal.a \
	      examples/example_inference_metal \
	      tinynn/smoke tinynn/ab_smoke tinynn/ab_smoke_cuda tinynn/ab_smoke_all_cuda \
	      tinynn/ab_smoke_add tinynn/ab_smoke_gelu tinynn/ab_smoke_rms_norm \
	      tinynn/ab_smoke_softmax tinynn/ab_smoke_transpose tinynn/ab_smoke_scale \
	      tinynn/ab_smoke_pipeline tinynn/ab_smoke_big tinynn/ab_smoke_big_cuda \
	      tinynn/ab_smoke_matmul_variants tinynn/ab_smoke_back tinynn/ab_smoke_embed \
	      tinynn/ab_smoke_sgd tinynn/ab_smoke_gelu_back tinynn/ab_smoke_cegrad \
	      tinynn/ab_smoke_adam tinynn/forward_smoke tinynn/persistent_bench \
	      tinynn/persistent_bench_cuda tinynn/persistent_bench_big \
	      examples/example_train_from_scratch \
	      examples/example_train_from_scratch_cpu \
	      examples/example_train_from_scratch_cuda \
	      examples/example_finetune examples/example_finetune_cuda \
	      examples/example_inference examples/example_list_models \
	      examples/example_serve examples/example_train

distclean: clean
	rm -rf $(GGML_DIR)/build $(GGML_DIR)/build-cuda $(GGML_DIR)/build-metal

# --- Algorithm-card drift gate -----------------------------------------------
# Sanity-check that every Toy:: class with both `def forward` and
# `def algorithm` keeps the two in lock-step. Catches the common
# drift case where someone changes the forward without updating the
# card (or vice versa). Pure-Ruby, runs in a fraction of a second.
check-cards:
	ruby prep/card_drift_check.rb

# --- Perf regression gate -----------------------------------------------------
# Runs each bench/*.rb (LoRA step, inference, tokenizer) and compares the
# emitted BENCH lines against bench/baselines.csv. Exit 1 on any metric that
# regresses past its per-metric tolerance. `bench-update` re-records the
# current values as the new baseline.
#
# Run before pushing perf-sensitive changes; baselines.csv lives in the repo
# so anyone can re-run on the same hardware and compare.
bench: tinynn/libtinynn_ggml.a
	ruby bench/check.rb

bench-update: tinynn/libtinynn_ggml.a
	ruby bench/check.rb --update

bench-report: tinynn/libtinynn_ggml.a
	ruby bench/check.rb --report

# Routine comparison vs PyTorch — the "old-stable" yardstick — in the
# single-machine single-GPU case. Runs ON gx10: toy CUDA benches run
# native, the PyTorch reference (bench/ref_pytorch.py) runs in the
# dev-pytorch container. Gates the toy/PyTorch *ratio* (not absolute
# ms, which is machine-dependent) so a design change that quietly
# widens the gap fails. Budget in bench/baselines_vs_pytorch.csv;
# `--update` re-records it. Override the torch invocation with PT_CMD.
bench-vs-pytorch: demos/seq_train_bench_cuda demos/qwen25_bench_cuda
	ruby bench/check_vs_pytorch.rb

bench-vs-pytorch-update: demos/seq_train_bench_cuda demos/qwen25_bench_cuda
	ruby bench/check_vs_pytorch.rb --update

bench-vs-pytorch-report: demos/seq_train_bench_cuda demos/qwen25_bench_cuda
	ruby bench/check_vs_pytorch.rb --report

# Heavy bench — ambitious workloads that exercise the libs (LoRA on
# Qwen2.5-1.5B at seq=256, decode on Qwen2.5-7B-Q8 with KV_Q8+FLASH).
# ~3-5 min wallclock; meant as a yardstick for choosing between
# optimization strategies, not for every-commit gating.
#   bench-heavy            — toy-only, fast iteration loop (no PyTorch)
#   bench-vs-pytorch-heavy — same workloads + PyTorch ratio gate
bench-heavy: demos/seq_train_bench_cuda demos/qwen25_bench_cuda
	ruby bench/check_heavy.rb

bench-heavy-update: demos/seq_train_bench_cuda demos/qwen25_bench_cuda
	ruby bench/check_heavy.rb --update

bench-heavy-report: demos/seq_train_bench_cuda demos/qwen25_bench_cuda
	ruby bench/check_heavy.rb --report

bench-vs-pytorch-heavy: demos/seq_train_bench_cuda demos/qwen25_bench_cuda
	ruby bench/check_vs_pytorch.rb --heavy

bench-vs-pytorch-heavy-update: demos/seq_train_bench_cuda demos/qwen25_bench_cuda
	ruby bench/check_vs_pytorch.rb --heavy --update

bench-vs-pytorch-heavy-report: demos/seq_train_bench_cuda demos/qwen25_bench_cuda
	ruby bench/check_vs_pytorch.rb --heavy --report

.PHONY: all clean distclean setup-ggml setup-ggml-cuda setup-ggml-metal smoke \
        example_inference_metal \
        ab-smoke ab-smoke-add ab-smoke-gelu ab-smoke-rms-norm \
        ab-smoke-softmax ab-smoke-transpose ab-smoke-scale ab-smoke-silu \
        ab-smoke-mul ab-smoke-pipeline ab-smoke-big ab-smoke-cuda \
        ab-smoke-all-cuda ab-smoke-big-cuda test \
        gpt2 smollm2 smollm2_kv smollm2_kv_cuda \
        tinyllama tinyllama_kv tinyllama_kv_cuda \
        train algorithm_cards \
        examples gen-mirrors verify-mirrors \
        bench bench-update bench-report check-cards \
        bench-vs-pytorch bench-vs-pytorch-update bench-vs-pytorch-report \
        bench-heavy bench-heavy-update bench-heavy-report \
        bench-vs-pytorch-heavy bench-vs-pytorch-heavy-update bench-vs-pytorch-heavy-report
