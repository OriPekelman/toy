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
# Pinned upstream rev: what the vendor-patches/ set is proven against, and
# what ships inside the gem (toy#45). Bump deliberately, re-proving patches.
GGML_REV    := 41e7949
GGML_CUDA_ARCH ?= 121
CUDA_DIR    ?= /usr/local/cuda

# --- Tep dependency (spinelgems convention) ---------------------------------
# Tep is consumed as a RELEASED gem from RubyGems (Gemfile: `gem "tep",
# "~> 0.11"`; published at https://rubygems.org/gems/tep) via the
# bundler-spinel / spinelgems convention. Two steps:
#
#   1. `bundle lock`                              (Gemfile → Gemfile.lock)
#   2. `../spinelgems/exe/spinel-compat vendor`   (lock → vendor/spinel/:
#                                                  copies tep lib/ AND
#                                                  natively compiles+wires
#                                                  its C-exts from tep's
#                                                  spinel-ext.json, AND
#                                                  writes vendor/spinel/deps.rb)
#
# No step 3: the old `prep/post_vendor_tep.rb` @TEP_*@ substitution is
# RETIRED — spinel-compat vendor owns C-ext wiring now (tep#98). Spinel
# entrypoints do `require_relative "vendor/spinel/deps"`.
#
# Precheck: ../spinelgems (the vendor tool) must be present. tep itself
# comes from RubyGems (bundler fetches the released gem), so ../tep is NOT
# required for the vendor flow.
#
# `bundle` env note: use a user-managed Ruby (rbenv / rv / ruby-install
# with --user-install gems). With system-owned gems (e.g. Debian's
# /var/lib/gems), `bundle lock` can't write the git cache without sudo —
# that's an env-setup concern, not a toy bug.
#
# SPINEL_EXT_DISABLE=pg: tep's optional pg C-ext currently fails to
# compile under spinel-compat (its libpq pkg-config cflags aren't wired
# to the source .o compile — spinelgems#8). toy only uses tep for HTTP
# serving, not its pg adapter, so we opt out. Drop this once #8 lands.
vendor-tep:
	@if [ ! -d ../spinelgems ]; then \
	    echo ""; \
	    echo "  ✗ vendor-tep needs the spinelgems sibling checkout (the vendor tool):"; \
	    echo "      missing: ../spinelgems"; \
	    echo ""; \
	    echo "    From this directory's parent ($$(cd .. && pwd)):"; \
	    echo "      git clone https://github.com/OriPekelman/spinelgems"; \
	    echo "    Or symlink an existing checkout:"; \
	    echo "      ln -s ~/sites/spinelgems ../spinelgems"; \
	    echo ""; \
	    exit 1; \
	fi
	bundle lock
	SPINEL_EXT_DISABLE=pg SPINEL_DIR=$(HOME)/sites/spinel ../spinelgems/exe/spinel-compat vendor

# Build vendor/spinel/tep/lib/tep.rb on demand for tep_demo/* targets.
# Triggers vendor-tep, which gates on sibling checkouts.
vendor/spinel/tep/lib/tep.rb:
	@$(MAKE) vendor-tep

# SpinelKit (toy#44) is vendored by the SAME `vendor-tep` step (both gems are in
# the Gemfile/lock; `spinel-compat vendor` copies all of them). This rule lets
# any runner/example list the vendored spinel_kit/git.rb (toy_events' git
# provenance) as a build prereq and have it produced on demand. Pure Ruby, no
# C-ext — the vendor copy is just lib/ files.
vendor/spinel/spinel_kit/lib/spinel_kit/git.rb:
	@$(MAKE) vendor-tep

# SpinelKit JSON builder (toy#44) — the run_start/events JSON emitter, vendored
# by the same `vendor-tep` step. Replaces the retired lib/toy/io/toy_json.rb
# (Toy::Json → SpinelKit::Json::Builder; byte-identical output).
vendor/spinel/spinel_kit/lib/spinel_kit/json_builder.rb:
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
	@echo "  NEW HERE? Scaffold a project + discover models with the toy CLI:"
	@echo "    toy new <dir>            scaffold a conventional toy project tree"
	@echo "    toy install              build/verify the CPU backend"
	@echo "    toy list                 find GGUFs in caches + project data/"
	@echo "    toy fetch <repo> <file>  download a GGUF from HuggingFace"
	@echo ""
	@echo "  ONE-TIME SETUP"
	@echo "    make setup               auto-detect platform; pick CUDA/Metal/CPU"
	@echo "    make setup-ggml          force CPU build (~2 min)"
	@echo "    make setup-ggml-cuda     force CUDA backend"
	@echo "    make setup-ggml-metal    force Metal backend (macOS)"
	@echo ""
	@if [ "$$(uname -s)" = "Darwin" ]; then \
	    echo "  ⚡ macOS detected — for GPU acceleration use the _metal example"; \
	    echo "      variants below (they link against libggml-metal + KV kernels)."; \
	    echo "      The plain CPU runner (\`toy infer\`) still works but is CPU-only."; \
	    echo ""; \
	fi
	@echo "  GETTING STARTED — examples/"
	@echo "    toy list                           list GGUFs cached locally / in HF / Ollama / LM Studio"
	@echo "    toy infer <model.gguf>             load a GGUF, generate 16 tokens (CPU)"
	@if [ "$$(uname -s)" = "Darwin" ]; then \
	    echo "    make example_inference_metal       same, Metal-accelerated (macOS) — use this on Mac"; \
	fi
	@echo "    Most tasks are the CLI now: toy train|infer|eval|serve (see 'toy --help')."
	@echo "    Curated library/instrumentation examples:"
	@echo "    make example_train                 tiny GPT from scratch on TinyStories (pure-Ruby teaching path)"
	@echo "    make example_train_from_scratch    modern Llama-shape from-scratch trainer (instrumentation ref)"
	@echo "    make example_train_vit_tiny        ViT-Tiny image-classifier + warm-start from timm"
	@echo "    toy train from-scratch --arch gpt2 GPT-2 from-scratch (CPU/CUDA) — see examples/gpt2_train.rb"
	@echo "    (CLI-superseded demos moved to examples/legacy/: lora, warm-start, lmc, metal-infer.)"
	@echo ""
	@echo "  HTTP SERVING — tep_demo/"
	@echo "    make tep_demo/hello                minimal Tep HTTP smoke"
	@if [ ! -f vendor/spinel/tep/lib/tep.rb ]; then \
	    printf "    (prereq: run %s first — needs ../tep + ../spinelgems checkouts)\n" "'make vendor-tep'"; \
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
	@echo "    GGUF=path/to/model.gguf            on example_finetune (toy infer takes a positional path)"
	@echo ""

# --- examples/ getting-started entry points --------------------------------
# Compact, one-file demos covering the main use cases. See
# examples/README.md.
# `toy infer` COMPUTE runner — lib-side Spinel binary the CLI shells to.
# Lifted from the retired examples/01_inference.rb. Target name MUST equal
# the output path string: lib/toy/core/cli/infer.rb uses RUNNER_TARGET both
# as the make target (ensure_built) AND the joined binary path. CPU-only;
# NOT in MIRRORABLE (see prep/gen_cuda_mirror.rb).
libexec:
	mkdir -p libexec
libexec/toy-infer: lib/toy/run/infer.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb lib/toy/io/tokenizer.rb tinynn/libtinynn_ggml.a | libexec
	$(SPINEL) $< -o $@
toy-infer: libexec/toy-infer

# Diagnostic sibling of toy-infer: enables the cache trace and dumps per-tap
# min/max/|mean|/nan for every layer (used to localize ggml#1506 — the K-quant
# MoE attention head_nbytes collapse). See docs/notes/mul_mat_id_quants.md.
libexec/toy-infer-trace: lib/toy/run/infer_trace.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb lib/toy/io/tokenizer.rb tinynn/libtinynn_ggml.a | libexec
	$(SPINEL) $< -o $@
toy-infer-trace: libexec/toy-infer-trace

# P4 — `toy eval` COMPUTE runner (CRuby→runner COMPUTE BRIDGE, same shape as
# toy-infer). Spinel source lib/toy/run/eval.rb; the binary path EQUALS the
# make target so ToyRoot.ensure_built("libexec/toy-eval") both builds and
# locates it. Deps = infer's deps + lib/toy/dev/toy_logprobs.rb (a transitive require
# of transformer_lm; listed explicitly so a touch of it rebuilds the runner).
# CPU-only; NOT in MIRRORABLE (see prep/gen_cuda_mirror.rb).
libexec/toy-eval: lib/toy/run/eval.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb lib/toy/io/tokenizer.rb lib/toy/dev/toy_logprobs.rb tinynn/libtinynn_ggml.a | libexec
	$(SPINEL) $< -o $@
toy-eval: libexec/toy-eval

# LMC (Linear Mode Connectivity) eval runner — `toy eval lmc --ckpt A --other B`.
# Interpolates two checkpoints θ_α = (1-α)·θ_A + α·θ_B and evals CE per α.
# Spinel source lib/toy/run/eval_lmc.rb; the binary path EQUALS the make target
# so ToyRoot.ensure_built("libexec/toy-eval-lmc") both builds and locates it.
# Deps mirror example_lmc (Makefile:479) NOT toy-eval; order-only | libexec (no
# $(SPINEL_DEPS)) like the CPU toy-eval runner. CPU-only; NOT in MIRRORABLE (see
# prep/gen_cuda_mirror.rb); a cuda LMC twin is a later slice.
libexec/toy-eval-lmc: lib/toy/run/eval_lmc.rb lib/toy/llm/adamw.rb vendor/spinel/spinel_kit/lib/spinel_kit/json_builder.rb lib/toy/io/toy_events.rb vendor/spinel/spinel_kit/lib/spinel_kit/git.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy.rb lib/toy/models/transformer.rb lib/toy/train/toy_drift_grad.rb lib/tinynn.rb tinynn/libtinynn_ggml.a | libexec
	$(SPINEL) $< -o $@
toy-eval-lmc: libexec/toy-eval-lmc

# CUDA siblings of toy-infer / toy-eval — selected by the CRuby CLI shell when
# invoked with `--device cuda` (lib/toy/core/cli/{infer,eval}.rb derive the
# target). PER-DEVICE binaries (not one polymorphic runner): a single source
# requiring BOTH ToyLM and ToyLMCuda would force the CUDA archive onto the CPU
# binary's link line, changing it. Keeping separate binaries leaves
# libexec/toy-infer / toy-eval link lines BYTE-UNCHANGED. Source is the
# hand-written lib/toy/run/{infer,eval}_cuda.rb (ToyLMCuda ctor arity 1 →
# NOT mechanically mirrorable → ABSENT from MIRRORABLE, like the CPU runners).
# Force-link recipe matches every other cuda target (-Wl,-u,tnn_cuda_force_link).
libexec/toy-infer-cuda: lib/toy/run/infer_cuda.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn_cuda.rb lib/toy/io/tokenizer.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS) | libexec
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
toy-infer-cuda: libexec/toy-infer-cuda

libexec/toy-eval-cuda: lib/toy/run/eval_cuda.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn_cuda.rb lib/toy/dev/toy_logprobs.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS) | libexec
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
toy-eval-cuda: libexec/toy-eval-cuda

# Metal twins of the infer/eval cuda runners (macOS ONLY). Same single-type
# binary discipline (landmine #16): TinyNNMetal is the only compute module.
# Source is the hand-written lib/toy/run/{infer,eval}_metal.rb (ToyLMMetal ctor
# arity 1 -> NOT mechanically mirrorable -> ABSENT from MIRRORABLE, like the
# cuda/CPU runners). The macOS guard MUST come first so Linux/gx10 never touches
# the Apple frameworks; the metal --cc recipe links Foundation/Metal/MetalKit
# with the leading-underscore force-link symbol (_tnn_metal_force_link, macOS
# symbol convention) vs cuda's tnn_cuda_force_link. libtinynn_ggml.a (CPU
# archive) stays in deps for the base ggml symbols. gx10 RUNTIME-UNVERIFIED.
libexec/toy-infer-metal: lib/toy/run/infer_metal.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm_metal.rb lib/toy_smollm2_ffi_kv_metal.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn_metal.rb lib/toy/io/tokenizer.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_metal.a $(SPINEL_DEPS) | libexec
ifneq ($(UNAME_S),Darwin)
	@echo "toy-infer-metal: macOS-only"; exit 1
endif
	$(SPINEL) --cc='cc -Wl,-u,_tnn_metal_force_link -framework Foundation -framework Metal -framework MetalKit' $< -o $@
toy-infer-metal: libexec/toy-infer-metal

libexec/toy-eval-metal: lib/toy/run/eval_metal.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm_metal.rb lib/toy_smollm2_ffi_kv_metal.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn_metal.rb lib/toy/dev/toy_logprobs.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_metal.a $(SPINEL_DEPS) | libexec
ifneq ($(UNAME_S),Darwin)
	@echo "toy-eval-metal: macOS-only"; exit 1
endif
	$(SPINEL) --cc='cc -Wl,-u,_tnn_metal_force_link -framework Foundation -framework Metal -framework MetalKit' $< -o $@
toy-eval-metal: libexec/toy-eval-metal

# Convenience: run both functional gates with the CUDA parity arm enabled.
.PHONY: gate-cuda
gate-cuda:
	TOY_GATE_CUDA=1 ruby prep/infer_gate.rb
	TOY_GATE_CUDA=1 ruby prep/eval_gate.rb

# GPT-2 minimal inline training proof (toy#12 part-b foundation). Builds a
# self-contained forward+CE+backward+AdamW loop over the GPT-2-distinctive
# structure (wte+wpe learned embeddings, composite LayerNorm, GELU FFN, tied
# output) — exercising the two vendored backward kernels (ggml_gelu_back,
# ggml_norm_back; vendor-patches/0007) end-to-end. Attention is the next
# increment; this proves the kernels train. Asserts CE decreases (exit 1 if
# not). CPU-only. "record-from-inline-first" reference for prep/gpt2_train_gate.
libexec/gpt2-train-min: prep/gpt2_train_min.rb lib/toy.rb lib/tinynn.rb \
		lib/toy/models/transformer.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) | libexec
	$(SPINEL) $< -o $@
gpt2-train-min: libexec/gpt2-train-min
.PHONY: gpt2-train-min
gate-gpt2-min: libexec/gpt2-train-min
	./libexec/gpt2-train-min
.PHONY: gate-gpt2-min
# Byte-exact GPT-2 train gate: assert the CE loss curve is byte-identical to
# prep/fixtures/gpt2_train_baseline.txt (record-from-inline reference for the
# eventual `toy train --arch gpt2`). Re-record with `ruby prep/gpt2_train_gate.rb --record`.
gate-gpt2: libexec/gpt2-train-min
	ruby prep/gpt2_train_gate.rb
.PHONY: gate-gpt2
# Byte-exact gate for the GPT-2 ENGINE runner (libexec/toy-train-gpt2 →
# Toy::LLM::Engine::GPT2SeqEngine, the `toy train --arch gpt2` compute). Asserts
# the from-scratch loss curve is byte-identical + decreasing. Re-record with
# `ruby prep/gpt2_train_engine_gate.rb --record`.
gate-gpt2-train: libexec/toy-train-gpt2
	ruby prep/gpt2_train_engine_gate.rb
.PHONY: gate-gpt2-train
# CUDA arm: `toy train --arch gpt2 --device cuda`. Forward + most backward on
# CUDA; GELU/LayerNorm backward fall back to CPU (no GPU kernel). CUDA-vs-CUDA
# byte-exact (empirical on GB10) + decreasing. Re-record with
# `ruby prep/gpt2_train_cuda_gate.rb --record`.
gate-gpt2-train-cuda: libexec/toy-train-gpt2-cuda
	ruby prep/gpt2_train_cuda_gate.rb
.PHONY: gate-gpt2-train-cuda

# Deterministic train→infer ROUND-TRIP gate: train from-scratch --steps 5
# --seed 0, then infer a fixed numeric prompt greedily from the written
# checkpoint and assert the generated ids byte-equal the recorded fixture.
# Proves the from-scratch checkpoint is a standard fused-llama GGUF that
# `toy infer` loads. CPU-only (no CUDA arm); bin/toy auto-builds the runners.
.PHONY: gate-ckpt-roundtrip
gate-ckpt-roundtrip:
	ruby prep/ckpt_roundtrip_gate.rb

# Deterministic LMC gate: `toy eval lmc` interpolates two PINNED from-scratch
# checkpoints and evals CE per α. The curve is ggml-internal CE (no Ruby libm)
# → byte-exact everywhere. Run twice (determinism) and assert byte-identical to
# prep/fixtures/lmc_baseline.txt. CPU-only (no CUDA arm this slice).
.PHONY: gate-lmc
gate-lmc:
	ruby prep/lmc_gate.rb

# The 6th realize-gate (F3 full fine-tune) — past P2's accepted ceiling.
# Records the engine's full_finetune CE curve and re-verifies it byte-for-byte
# so the per-block alloc lift onto TransformerBlock is provably behavior-
# preserving. MODEL-GATED: needs data/smollm2-135m-native.gguf (gitignored dev
# artifact); SKIPs loudly when absent. Train losses are ggml-internal → byte-
# exact. Re-record with `ruby prep/full_finetune_gate.rb --record`.
.PHONY: gate-full-finetune
gate-full-finetune:
	ruby prep/full_finetune_gate.rb

# Mixed-precision training gate (GH#9, f16, CPU). Drives the from-scratch
# example at WEIGHT_DTYPE=1 vs =0 and asserts: f16 runs to completion (needs the
# 0008 mul_mat-backward-mixed-precision ggml patch — without it backward aborts),
# run_start.model.weight_type surfaces the dtype, and the f16 final loss lands
# within tolerance of the f32 baseline. TOLERANCE arm (dtype changes numerics),
# not byte-exact. bf16 is the CUDA/GB10 follow-up. Builds the example itself.
.PHONY: gate-mixed-precision
gate-mixed-precision:
	ruby prep/mixed_precision_gate.rb

# toy#64 item 6 — Toy::RunLog unit gate (CRuby-only, no Spinel build).
# Self-contained synthetic fixture + integration sniff of repo runs/.
.PHONY: gate-run-log
gate-run-log:
	ruby prep/run_log_gate.rb

# toy#42 full-API require gate. Builds examples/smoke_compute_surface (which
# requires ONLY lib/toy/compute.rb) and asserts it realizes a live engine —
# proving the one-require compute surface co-compiles + works for a library
# consumer. Builds the smoke itself.
.PHONY: gate-compute-surface
gate-compute-surface: examples/smoke_compute_surface
	@out="$$(./examples/smoke_compute_surface 2>&1)"; \
	echo "$$out" | tail -2; \
	echo "$$out" | grep -q "compute-surface: ok" \
	  && echo "GATE PASS [compute-surface]: lib/toy/compute.rb one-require surface is live" \
	  || { echo "GATE FAIL [compute-surface]"; exit 1; }

# toy#64 item 8 — CUDA twin of gate-compute-surface: build + run the
# consumer-ish CUDA entry smoke on the GPU (GB10 sm_121).
.PHONY: gate-compute-surface-cuda
gate-compute-surface-cuda: examples/smoke_compute_surface_cuda
	@out="$$(./examples/smoke_compute_surface_cuda 2>&1)"; \
	echo "$$out" | tail -2; \
	echo "$$out" | grep -q "compute-surface-cuda: ok" \
	  && echo "GATE PASS [compute-surface-cuda]: lib/toy/compute_cuda.rb device entry is live" \
	  || { echo "GATE FAIL [compute-surface-cuda]"; exit 1; }

# K-quant MoE attention regression gate (the bug long misfiled as ggml#1506):
# head_nbytes returned 0 for K-quant attention weights → per-head mmap stride
# collapsed every head onto head 0 → degenerate repeating decode on OLMoE
# Q4_K_M. Structural assertion (distinct-count + max single-token run), not
# byte-exact, so it survives benign K-quant drift. MODEL-GATED: needs the ~4 GB
# data/OLMoE-1b-7b-0924-Instruct-Q4_K_M.gguf (gitignored); SKIPs loudly when
# absent. bin/toy auto-builds the infer runner. See docs/notes/mul_mat_id_quants.md.
.PHONY: gate-moe-kquant
gate-moe-kquant:
	ruby prep/moe_kquant_gate.rb

# Silent poly-degradation gate (#32): compiles the canonical compute entrypoints
# with spinel and fails if a NEW `cannot resolve … on poly … (emitting 0)` warning
# appears vs the frozen baseline — i.e. a refactor just silently compiled a literal
# 0 into a numerical path (compiled != correct). Re-record the known-benign set with
# `ruby prep/poly_degrade_gate.rb --record`. See feedback_spinel_type_inference_landmines.
.PHONY: gate-poly-degrade
gate-poly-degrade:
	ruby prep/poly_degrade_gate.rb

# CUDA from-scratch TRAINING gate (STRONG arm, no epsilon): train
# from-scratch --device cuda --steps 5 --seed 0, assert the "step N: loss="
# curve byte-equals prep/fixtures/train_cuda_baseline.txt, loss decreases,
# and the CUDA checkpoint round-trips through CPU `toy infer`. Determinism is
# EMPIRICAL on this GB10 — see the fixture header. bin/toy auto-builds.
.PHONY: gate-train-cuda
gate-train-cuda:
	ruby prep/train_cuda_gate.rb

# Metal RUNTIME parity gate (macOS ONLY). Builds the three metal runners then
# runs prep/metal_gate.rb: infer (cpu-vs-metal byte-equal ids), eval (top-k id
# ORDER equality), train-from-scratch (run-twice byte-determinism OR a Mac-
# pinned baseline, loss-decrease, ckpt round-trip vs the SHARED fixture,
# events.jsonl run_start/run_end). On Linux/gx10 this SKIPS GREEN (exit 0) so
# umbrella `make gate-*` runs do not false-fail — Metal cannot build or run
# here. THIS is the gate that actually validates metal numerics; run it on the
# Mac. (The metal BUILD targets exit 1 on Linux — a gate that can't run skips
# green, a build target that can't build errors red.)
.PHONY: gate-metal
gate-metal:
ifneq ($(UNAME_S),Darwin)
	@echo "gate-metal: Metal is macOS-only (uname -s = $(UNAME_S)) — skipping"; exit 0
else
	$(MAKE) libexec/toy-infer-metal libexec/toy-eval-metal libexec/toy-train-metal
	ruby prep/metal_gate.rb
endif

# STRUCTURAL serving-telemetry gate: boot libexec/toy-serve with TAO_RUN_DIR
# set, POST /v1/completions, SIGTERM, then assert runs/<id>/events.jsonl carries
# the toy/v1 run_start(serve) + eval/serve/request + run_end stream (Tao #6).
# Honest STRUCTURAL (NOT byte-identical): t/latency_us/request_id are
# wall-clock/counter and cannot be byte-stable. Self-builds the runner.
.PHONY: gate-serve-events
gate-serve-events:
	ruby prep/serve_events_gate.rb

# Umbrella: the byte-baseline serve gate THEN the structural events gate.
.PHONY: gate-serve
gate-serve:
	ruby prep/serve_gate.rb
	ruby prep/serve_events_gate.rb

# P4 — from-scratch TRAINING compute runner (CRuby→runner COMPUTE BRIDGE,
# same shape as toy-infer). Spinel source lib/toy/run/train.rb; the binary
# path EQUALS the make target so ToyRoot.ensure_built("libexec/toy-train")
# both builds and locates it. Deps list every transitive require the runner
# pulls (the recipe → llama_seq_engine → transformer + toy + smollm2 +
# tinynn + the L1-L3 primitives/blocks/archs; plus gguf_writer + drift_grad
# for the checkpoint). CPU-only; NOT in MIRRORABLE (see prep/gen_cuda_mirror.rb).
libexec/toy-train: lib/toy/run/train.rb lib/toy/dev/toy_describe_flow.rb lib/toy.rb lib/toy/models/toy_smollm2.rb \
		vendor/spinel/spinel_kit/lib/spinel_kit/json_builder.rb lib/toy/io/toy_events.rb vendor/spinel/spinel_kit/lib/spinel_kit/git.rb \
		lib/toy/io/toy_corpus_loader.rb lib/toy/train/toy_lr_schedule.rb \
		lib/toy/llm/engine/llama_seq_engine.rb lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/from_scratch.rb \
		lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/warm_start.rb \
		lib/toy/llm/adamw.rb lib/toy/llm/labels.rb \
		lib/toy/train/toy_gguf_writer.rb lib/toy/train/toy_drift_grad.rb lib/toy/models/transformer.rb \
		lib/toy/llm/primitives/rms_norm.rb lib/toy/llm/primitives/rope.rb \
		lib/toy/llm/primitives/swiglu.rb lib/toy/llm/primitives/gqa.rb \
		lib/toy/llm/blocks/transformer_block.rb lib/toy/llm/archs/llama_arch.rb \
		lib/tinynn.rb tinynn/libtinynn_ggml.a | libexec
	$(SPINEL) $< -o $@
toy-train: libexec/toy-train

# `toy train lora` DEDICATED runner. Separate binary from toy-train: the
# LoRA realize_for_mmap path cannot share a Spinel compilation unit with the
# random-init path (cfg type-merge miscompile; see lib/toy/run/train_lora.rb
# header). CPU-only; NOT in MIRRORABLE.
libexec/toy-train-lora: lib/toy/run/train_lora.rb lib/toy/dev/toy_describe_flow.rb vendor/spinel/spinel_kit/lib/spinel_kit/json_builder.rb lib/toy/io/toy_events.rb vendor/spinel/spinel_kit/lib/spinel_kit/git.rb lib/toy.rb lib/toy/models/toy_smollm2.rb \
		lib/toy/llm/engine/llama_seq_engine.rb lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/lora.rb \
		lib/toy/llm/adamw.rb \
		lib/toy/train/toy_gguf_writer.rb lib/toy/train/toy_drift_grad.rb lib/toy/models/transformer.rb \
		lib/toy/llm/primitives/rms_norm.rb lib/toy/llm/primitives/rope.rb \
		lib/toy/llm/primitives/swiglu.rb lib/toy/llm/primitives/gqa.rb \
		lib/toy/llm/blocks/transformer_block.rb lib/toy/llm/archs/llama_arch.rb \
		lib/tinynn.rb tinynn/libtinynn_ggml.a | libexec
	$(SPINEL) $< -o $@
toy-train-lora: libexec/toy-train-lora

# `toy train from-scratch --arch gpt2` DEDICATED runner. Separate binary from
# toy-train (landmine #16: the GPT-2 realize path can't share a Spinel unit with
# the llama random-init path). Self-contained GPT2SeqEngine (no llama engine /
# primitives dep), so it also can't churn the llama gates. CPU-only this slice.
libexec/toy-train-gpt2: lib/toy/run/train_gpt2.rb lib/toy.rb \
		lib/toy/llm/engine/gpt2_seq_engine.rb lib/toy/llm/labels.rb lib/toy/llm/adamw.rb \
		lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a | libexec
	$(SPINEL) $< -o $@
toy-train-gpt2: libexec/toy-train-gpt2
.PHONY: toy-train-gpt2

# CUDA twin of toy-train-gpt2 (`--arch gpt2 --device cuda`). SEPARATE single-type
# binary (landmine #16): links the generated CUDA engine mirror + the CUDA TinyNN
# shim; the GELU/LayerNorm backward ops fall back to the CPU backend via the
# scheduler (no CUDA kernel). lib/tinynn.rb + transformer.rb stay in deps (Mat /
# CPU-TinyNN seam). NOT in MIRRORABLE (the engine mirror IS; the runner is hand-written).
libexec/toy-train-gpt2-cuda: lib/toy/run/train_gpt2_cuda.rb lib/toy.rb \
		lib/toy/llm/engine/gpt2_seq_engine_cuda.rb lib/toy/models/transformer.rb \
		lib/tinynn_cuda.rb lib/tinynn.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS) | libexec
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
toy-train-gpt2-cuda: libexec/toy-train-gpt2-cuda
.PHONY: toy-train-gpt2-cuda

# Metal twin (`--arch gpt2 --device metal`), macOS ONLY. Same structure; links
# the generated Metal engine mirror + the Metal TinyNN shim + Apple frameworks.
# gx10 RUNTIME-UNVERIFIED (codegen + structural parity here; runtime-gate on Mac).
libexec/toy-train-gpt2-metal: lib/toy/run/train_gpt2_metal.rb lib/toy.rb \
		lib/toy/llm/engine/gpt2_seq_engine_metal.rb lib/toy/models/transformer.rb \
		lib/tinynn_metal.rb lib/tinynn.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_metal.a $(SPINEL_DEPS) | libexec
ifneq ($(UNAME_S),Darwin)
	@echo "toy-train-gpt2-metal: macOS-only"; exit 1
endif
	$(SPINEL) --cc='cc -Wl,-u,_tnn_metal_force_link -framework Foundation -framework Metal -framework MetalKit' $< -o $@
toy-train-gpt2-metal: libexec/toy-train-gpt2-metal
.PHONY: toy-train-gpt2-metal

# P4/vit — ViT-Tiny from-scratch CPU TRAINING runner. SEPARATE binary
# (landmine #16): ViTTinyConfig must NOT share a Spinel compilation unit
# with SmolLM2Config. Source lib/toy/run/train_vit.rb; binary path EQUALS
# the make target. Reads STEPS/SEED/IMG_DIR/TAO_RUN_DIR/TOY_RUN_ID from ENV;
# trains random-init on the COMMITTED data/vit_smoke corpus. NO toy_gguf_writer
# dep (cfg.vocab/d_ff poly-collide with ViTTinyConfig — #169 checkpoint
# follow-up). CPU-only; absent from MIRRORABLE (no CUDA/Metal twin this slice).
libexec/toy-train-vit: lib/toy/run/train_vit.rb lib/toy/dev/toy_describe_flow.rb vendor/spinel/spinel_kit/lib/spinel_kit/json_builder.rb lib/toy/io/toy_events.rb vendor/spinel/spinel_kit/lib/spinel_kit/git.rb lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/vit_tiny.rb \
		lib/toy/llm/engine/vit_tiny_engine.rb lib/toy/models/toy_vit.rb lib/toy/models/toy_smollm2.rb \
		lib/toy/io/toy_image_loader.rb lib/toy/train/toy_lr_schedule.rb lib/toy/train/toy_drift_grad.rb \
		lib/toy/llm/adamw.rb \
		lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) | libexec
	$(SPINEL) $< -o $@
toy-train-vit: libexec/toy-train-vit

# P4/GPU — from-scratch CUDA TRAINING runner. CUDA twin of libexec/toy-train,
# from-scratch ONLY (warm_start dropped). SINGLE-TYPE binary (landmine #16):
# TinyNNCuda is the compute path; lib/tinynn.rb + lib/toy/models/transformer.rb stay in
# deps because transformer.rb requires tinynn -> defines CPU TinyNN for the
# checkpoint write/fuse/drift seam (dropping them breaks the writer). Links
# the CUDA ggml backend via -Wl,-u,tnn_cuda_force_link (every cuda target).
# CPU-only; NOT in MIRRORABLE (hand-written, see prep/gen_cuda_mirror.rb).
libexec/toy-train-cuda: lib/toy/run/train_cuda.rb lib/toy/dev/toy_describe_flow.rb vendor/spinel/spinel_kit/lib/spinel_kit/json_builder.rb lib/toy/io/toy_events.rb vendor/spinel/spinel_kit/lib/spinel_kit/git.rb lib/toy.rb lib/toy/models/toy_smollm2.rb \
		lib/toy/io/toy_corpus_loader.rb lib/toy/train/toy_lr_schedule.rb \
		lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/from_scratch_cuda.rb \
		lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/warm_start_cuda.rb \
		lib/toy/llm/adamw.rb lib/toy/llm/labels.rb \
		lib/toy/train/toy_gguf_writer.rb lib/toy/train/toy_drift_grad.rb lib/toy/train/toy_gguf_fuse.rb lib/toy/models/transformer.rb \
		lib/toy/llm/primitives/rms_norm_cuda.rb lib/toy/llm/primitives/rope_cuda.rb \
		lib/toy/llm/primitives/swiglu_cuda.rb lib/toy/llm/primitives/gqa_cuda.rb \
		lib/toy/llm/blocks/transformer_block_cuda.rb lib/toy/llm/archs/llama_arch_cuda.rb \
		lib/tinynn_cuda.rb lib/tinynn.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS) | libexec
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
toy-train-cuda: libexec/toy-train-cuda

# P4/GPU — LoRA CUDA TRAINING runner. CUDA twin of libexec/toy-train-lora.
# SEPARATE binary from libexec/toy-train-cuda: the LoRA realize_for_mmap path
# cannot share a Spinel compilation unit with the random-init path (cfg
# type-merge miscompile; landmine #16 — same reason toy-train-lora is split
# from toy-train). SINGLE-TYPE binary: TinyNNCuda is the compute path;
# lib/tinynn.rb + lib/toy/models/transformer.rb stay in deps because transformer.rb
# requires tinynn -> defines CPU TinyNN for the checkpoint write seam
# (ToyDriftGrad.params downloads via CPU TinyNN). toy_gguf_fuse is NOT a dep
# (lora uses ToyDriftGrad.params, not the lens-fold path). Links the CUDA
# ggml backend via -Wl,-u,tnn_cuda_force_link. NOT in MIRRORABLE (hand-written).
libexec/toy-train-lora-cuda: lib/toy/run/train_lora_cuda.rb lib/toy/dev/toy_describe_flow.rb vendor/spinel/spinel_kit/lib/spinel_kit/json_builder.rb lib/toy/io/toy_events.rb vendor/spinel/spinel_kit/lib/spinel_kit/git.rb lib/toy.rb lib/toy/models/toy_smollm2.rb \
		lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/lora_cuda.rb \
		lib/toy/llm/adamw.rb \
		lib/toy/train/toy_gguf_writer.rb lib/toy/train/toy_drift_grad.rb lib/toy/models/transformer.rb \
		lib/toy/llm/primitives/rms_norm_cuda.rb lib/toy/llm/primitives/rope_cuda.rb \
		lib/toy/llm/primitives/swiglu_cuda.rb lib/toy/llm/primitives/gqa_cuda.rb \
		lib/toy/llm/blocks/transformer_block_cuda.rb lib/toy/llm/archs/llama_arch_cuda.rb \
		lib/tinynn_cuda.rb lib/tinynn.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS) | libexec
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
toy-train-lora-cuda: libexec/toy-train-lora-cuda

# P4/GPU — from-scratch METAL TRAINING runner (macOS ONLY). Metal twin of
# libexec/toy-train-cuda, from-scratch ONLY. SINGLE-TYPE binary (landmine #16):
# TinyNNMetal is the compute path; lib/tinynn.rb + lib/toy/models/transformer.rb stay in
# deps because transformer.rb requires tinynn -> defines CPU TinyNN for the
# checkpoint write/fuse/drift seam (dropping them breaks the writer). The macOS
# guard MUST come first so Linux/gx10 never touches the Apple frameworks; the
# metal --cc recipe links Foundation/Metal/MetalKit with _tnn_metal_force_link
# (leading underscore, macOS symbol convention). libtinynn_ggml.a (CPU archive)
# stays in deps for the write seam + base ggml. NOT in MIRRORABLE (hand-written).
# gx10 RUNTIME-UNVERIFIED — pin baseline + gate on the Mac.
libexec/toy-train-metal: lib/toy/run/train_metal.rb lib/toy/dev/toy_describe_flow.rb vendor/spinel/spinel_kit/lib/spinel_kit/json_builder.rb lib/toy/io/toy_events.rb vendor/spinel/spinel_kit/lib/spinel_kit/git.rb lib/toy.rb lib/toy/models/toy_smollm2.rb \
		lib/toy/llm/engine/llama_seq_engine_metal.rb lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/from_scratch_metal.rb \
		lib/toy/llm/adamw.rb lib/toy/llm/labels.rb \
		lib/toy/train/toy_gguf_writer.rb lib/toy/train/toy_drift_grad.rb lib/toy/train/toy_gguf_fuse.rb lib/toy/models/transformer.rb \
		lib/toy/llm/primitives/rms_norm_metal.rb lib/toy/llm/primitives/rope_metal.rb \
		lib/toy/llm/primitives/swiglu_metal.rb lib/toy/llm/primitives/gqa_metal.rb \
		lib/toy/llm/blocks/transformer_block_metal.rb lib/toy/llm/archs/llama_arch_metal.rb \
		lib/tinynn_metal.rb lib/tinynn.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_metal.a $(SPINEL_DEPS) | libexec
ifneq ($(UNAME_S),Darwin)
	@echo "toy-train-metal: macOS-only"; exit 1
endif
	$(SPINEL) --cc='cc -Wl,-u,_tnn_metal_force_link -framework Foundation -framework Metal -framework MetalKit' $< -o $@
toy-train-metal: libexec/toy-train-metal

# P4 — `toy serve` PERSISTENT compute runner (OpenAI-compatible HTTP).
# Unlike infer/train/eval (compute-once), this runner blocks in Tep.run!.
# Spinel source lib/toy/run/serve.rb; the binary path EQUALS the make
# target so ToyRoot.ensure_built("libexec/toy-serve") both builds and
# locates it. The endpoint logic moved out of tep_demo/openai_api_llama.rb
# into lib/toy/serve/openai/* (Server/State + handlers + the embeddings
# handler; JSON via SpinelKit::Json, toy#44). vendor/spinel/tep/lib/tep.rb is the TEP BUILD-DEP
# edge — Tep is consumed purely as transport (built by `make vendor-tep`
# on a fresh tree; needs ../tep + ../spinelgems siblings). Deps mirror the
# tep_demo recipe (Makefile:486) + the KV stack. CPU-only; NOT in
# MIRRORABLE (see prep/gen_cuda_mirror.rb).
libexec/toy-serve: lib/toy/run/serve.rb vendor/spinel/spinel_kit/lib/spinel_kit/json_builder.rb lib/toy/io/toy_events.rb vendor/spinel/spinel_kit/lib/spinel_kit/git.rb \
		lib/toy/serve/openai/server.rb \
		lib/toy/serve/openai/handlers.rb lib/toy/serve/openai/embeddings_handler.rb \
		vendor/spinel/tep/lib/tep.rb \
		lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb \
		tinynn/libtinynn_ggml.a | libexec
	$(SPINEL) $< -o $@
toy-serve: libexec/toy-serve

# toy#gguf-checkpoint-reload (#153) — smoke binary that loads a
# from-scratch toy GGUF and runs a tiny generation. No tokenizer.
examples/smoke_toy_ckpt_reload: examples/smoke_toy_ckpt_reload.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb lib/toy/io/tokenizer.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# toy#embed-api (#145) — smoke for ToyLM#embed_lookup.
examples/smoke_embed_api: examples/smoke_embed_api.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb lib/toy/io/tokenizer.rb lib/toy/dev/toy_logprobs.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# P1 framework refactor — runtime Card derivation smoke. Loads a
# llama-family GGUF, realizes the seq-mode cache, derives a
# structural Toy::Card via ToyDescribeFlow.card, prints + gates.
examples/smoke_card_derive: examples/smoke_card_derive.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/dev/toy_describe_flow.rb lib/toy/train/toy_drift_grad.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb lib/toy/dev/toy_card.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# toy#decode-logprobs (#151) — smoke for ToyLM#decode_step_with_logprobs.
examples/smoke_decode_logprobs: examples/smoke_decode_logprobs.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb lib/toy/io/tokenizer.rb lib/toy/dev/toy_logprobs.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# GH#18 — LMC interpolate-and-eval runner.
examples/example_lmc: examples/legacy/08_lmc.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/tinynn.rb lib/toy/train/toy_drift_grad.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
example_lmc: examples/example_lmc

# E2.3 (towards GH#14) — projection-lens smoke.
examples/smoke_projection_lens: examples/smoke_projection_lens.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/tinynn.rb lib/toy/train/toy_drift_grad.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# toy#42 full-API require gate. Compiling this proves lib/toy/compute.rb's whole
# surface (all three engines + recipes + loaders) co-compiles in one program;
# running it realizes a LlamaSeqEngine to prove the surface is live. The prereq
# is just lib/toy/compute.rb — it pulls everything else transitively, and
# $(SPINEL) follows the require graph.
examples/smoke_compute_surface: examples/smoke_compute_surface.rb lib/toy/compute.rb lib/toy/llm/training_batch.rb lib/toy/llm/recipe_options.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# toy#64 item 8 — the CUDA compute entry (lib/toy/compute_cuda.rb), the
# consumer-ish device-at-compile-time gate. Same shape as the CPU
# compute-surface gate but requires compute_cuda + links the CUDA
# archives with the force-link flag. The generated CUDA mirrors in the
# dep list are kept fresh by the $(MIRROR_CUDA) pattern rules.
examples/smoke_compute_surface_cuda: examples/smoke_compute_surface_cuda.rb lib/toy/compute_cuda.rb \
		lib/toy/llm/training_batch.rb lib/toy/llm/recipe_options.rb \
		lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/llm/engine/gpt2_seq_engine_cuda.rb \
		lib/toy_smollm2_ffi_kv_cuda.rb \
		lib/toy/llm/recipes/from_scratch_cuda.rb lib/toy/llm/recipes/warm_start_cuda.rb \
		lib/toy/llm/primitives/rms_norm_cuda.rb lib/toy/llm/primitives/rope_cuda.rb \
		lib/toy/llm/primitives/swiglu_cuda.rb lib/toy/llm/primitives/gqa_cuda.rb \
		lib/toy/llm/blocks/transformer_block_cuda.rb lib/toy/llm/archs/llama_arch_cuda.rb \
		lib/tinynn_cuda.rb lib/tinynn.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS)
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# P2.6 — GQA-divergent (w_o) gate. Realizes a config with head_dim=24 so
# n_heads*head_dim (96) != d_model (64), proving the divergent w_o shape
# [d_model, n_heads*head_dim] allocates and runs forward+backward.
examples/smoke_gate_gqa_divergent: examples/smoke_gate_gqa_divergent.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/tinynn.rb lib/toy/train/toy_drift_grad.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# P2.6 — llama3 RoPE post-rope TENSOR parity gate. Builds a standalone
# post-rope subgraph from the SAME public primitive (RoPE.apply_2d) the
# model's K/Q paths call, with a NON-NULL, NON-TRIVIAL llama3 freq_factors
# ptr (computed via Toy::RopeScaling.compute_llama3_freq_factors). Logit-
# level is rope-angle-INSENSITIVE, so the gate taps the post-rope tensor:
# asserts (a) freq_factors non-uniform / kind==:llama3, (b) post-rope output
# byte-identical run-to-run, plus a contrast guard vs :none (NULL factors).
# No model file, no lib/ change, no mirror regen. Run from repo root.
examples/smoke_gate_llama3_tensor: examples/smoke_gate_llama3_tensor.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# P2.6 — B>1 (micro-batch) gate. Realizes with t_batch=2 so @seq_b=2,
# forcing the block-causal mask alloc + upload (gated on @seq_b>1) and the
# soft_max_ext attention path (gqa.rb:50). Proves the batched graph
# allocates the [T*B,T*B] mask and runs forward+backward; records a
# reproducible loss baseline. MUST run from repo root (data/ts_seqs.txt).
examples/smoke_gate_b_gt_1: examples/smoke_gate_b_gt_1.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/tinynn.rb lib/toy/train/toy_drift_grad.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# P2.6 — L4 FromScratch recipe gate. Drives the same random-init config
# as smoke_projection_lens THROUGH Toy::LLM::Recipes::FromScratch; its
# loss curve must byte-equal the projection-lens reference.
examples/smoke_recipe_from_scratch: examples/smoke_recipe_from_scratch.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/tinynn.rb lib/toy/train/toy_drift_grad.rb lib/toy/llm/adamw.rb lib/toy/llm/labels.rb lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/from_scratch.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# BLESSED from-scratch path — the short tutorial. Same gate-fixture
# config as smoke_recipe_from_scratch, but the clean tutorial read using
# the value objects (Toy::SmolLM2Config.mha + Toy::Labels + Toy::AdamW).
examples/example_train_from_scratch_blessed: examples/train_from_scratch.rb lib/toy.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/tinynn.rb lib/toy/llm/adamw.rb lib/toy/llm/labels.rb lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/from_scratch.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
example_train_from_scratch_blessed: examples/example_train_from_scratch_blessed

# L4 LoRA recipe gate. Drives the same LoRA fine-tune config as the
# frozen reference 03_finetune_lora THROUGH Toy::LLM::Recipes::LoRA; its
# loss curve must byte-equal the reference at the fixed config.
examples/smoke_recipe_lora: examples/smoke_recipe_lora.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb lib/toy/llm/adamw.rb lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/lora.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# L4 WarmStart recipe gate. Drives the same warm-start config as the
# frozen reference 09_warm_start_train (INIT=scratch) THROUGH
# Toy::LLM::Recipes::WarmStart; its loss curve must byte-equal 09's at
# the fixed config (SEED=0 STEPS=5). The fixture drives the cosine LR
# schedule + streaming corpus loader (deps below); the recipe stays thin.
examples/smoke_recipe_warm_start: examples/smoke_recipe_warm_start.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/tinynn.rb lib/toy/train/toy_drift_grad.rb lib/toy/io/toy_corpus_loader.rb lib/toy/train/toy_lr_schedule.rb lib/toy/llm/adamw.rb lib/toy/llm/labels.rb lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/warm_start.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# P2.6 gate — GGUF F32 mmap round-trip parity. Head-fuses a random_init
# model into the FUSED llama.cpp naming, writes a GGUF, reloads via
# realize_for_mmap, and asserts the reloaded forward is BIT-IDENTICAL to
# the in-memory forward. This is the behavioral gate for realize_for_mmap
# (previously only realize_for_random_init was gated). CPU-only: the GGUF
# WRITE half reads host data ptrs (tnn_gguf_w_add_tensor), which the CUDA
# writer doesn't implement — do NOT auto-mirror this to CUDA.
examples/smoke_gguf_roundtrip: examples/smoke_gguf_roundtrip.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/tinynn.rb lib/toy/train/toy_gguf_fuse.rb lib/toy/train/toy_gguf_writer.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

examples/smoke_full_finetune: examples/smoke_full_finetune.rb lib/toy/llm/adamw.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# P2.6 gate — qkv_bias mmap branch. Loads the real Qwen2.5-0.5B native GGUF
# (which DOES carry blk.N.attn_{q,k,v}.bias) and realizes via
# realize_for_mmap with qkv_bias=TRUE, untied=FALSE (output.weight absent =>
# tied), forcing the bias mmap branch (llama_seq_engine.rb:635-661) and
# its transformer_block tnn_add consumer — neither hit by smoke_gguf_roundtrip
# (qkv_bias=FALSE). Records a deterministic finite-logit baseline. CPU-only;
# DATA DEPENDENCY: data/qwen25-0.5b-native.gguf (not self-contained). MUST run
# from repo root. Do NOT auto-mirror to CUDA.
examples/smoke_gate_qkv_bias: examples/smoke_gate_qkv_bias.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# P2.6 gate — Q8-stays-Q8 realize_for_q8_copy branch. Loads the existing
# Q8 GGUF, asserts blk.0 attn_q weight stays Q8_0 in memory (NOT dequant
# to F32), deterministic forward x2 byte-identical baseline. Pure-Ruby
# fixture (no toy_drift_grad dep; seq_blocks_ffi directly).
examples/smoke_gate_q8_preserve: examples/smoke_gate_q8_preserve.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# P2.6 CUDA gate — GPU mirror of the projection-lens smoke. Exercises
# realize_for_random_init + seq forward on the CUDA backend so the
# realize-path refactor can be parity-gated on GPU (CUDA self-consistency
# before/after; CUDA floats don't bit-equal CPU). Mirror auto-generated
# by prep/gen_cuda_mirror.rb. Same force-link recipe as the 06 CUDA entry.
examples/smoke_projection_lens_cuda: examples/smoke_projection_lens_cuda.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/tinynn_cuda.rb lib/toy/train/toy_drift_grad.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS)
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# E2.4 (towards GH#14) — streaming corpus loader + cosine LR smoke.
examples/smoke_corpus_loader: examples/smoke_corpus_loader.rb lib/toy/models/transformer.rb lib/tinynn.rb lib/toy/io/toy_corpus_loader.rb lib/toy/train/toy_lr_schedule.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# E2.5 (towards GH#14) — warm-start training driver.
examples/example_warm_start_train: examples/legacy/09_warm_start_train.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/tinynn.rb lib/toy/train/toy_drift_grad.rb lib/toy/train/toy_gguf_writer.rb lib/toy/io/toy_corpus_loader.rb lib/toy/train/toy_lr_schedule.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
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

examples/example_train: examples/02_train_custom_gpt.rb lib/toy/models/transformer.rb lib/toy/train/training.rb lib/toy/train/toy_trainer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@
example_train: examples/example_train

examples/example_finetune: examples/legacy/03_finetune_lora.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@
example_finetune: examples/example_finetune

# CUDA mirror — same source, swap TinyNN → TinyNNCuda by including
# both libs. The example source uses TinyNN; the CUDA build link-step
# carries CUDA symbols too (no source change). For real GPU speedup
# users typically write a `_cuda` variant; this mirror is for the
# build-recipe story.
examples/example_finetune_cuda: examples/legacy/03_finetune_lora_cuda.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
example_finetune_cuda: examples/example_finetune_cuda

# Metal mirror of example_inference (macOS only). Uses TinyNNMetal.
# Same -Wl,-u trick as CUDA so the Metal backend init survives
# weak-symbol resolution. macOS expects a leading underscore on
# external symbols, hence `-Wl,-u,_tnn_metal_force_link`.
# Frameworks (Foundation/Metal/MetalKit) are linked via -framework.
examples/example_inference_metal: examples/legacy/01_inference_metal.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm_metal.rb lib/toy_smollm2_ffi_kv_metal.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn_metal.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_metal.a
ifneq ($(UNAME_S),Darwin)
	@echo "example_inference_metal: macOS-only"; exit 1
endif
	$(SPINEL) --cc='cc -Wl,-u,_tnn_metal_force_link -framework Foundation -framework Metal -framework MetalKit' $< -o $@
example_inference_metal: examples/example_inference_metal

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
examples/example_train_from_scratch_cpu: examples/06_train_from_scratch.rb vendor/spinel/spinel_kit/lib/spinel_kit/json_builder.rb lib/toy/io/toy_events.rb vendor/spinel/spinel_kit/lib/spinel_kit/git.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/tinynn.rb lib/toy/dev/toy_describe_flow.rb lib/toy/train/toy_drift_grad.rb lib/toy/train/toy_gguf_writer.rb lib/toy/dev/toy_tap.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
examples/example_train_from_scratch_cuda: examples/06_train_from_scratch_cuda.rb vendor/spinel/spinel_kit/lib/spinel_kit/json_builder.rb lib/toy/io/toy_events.rb vendor/spinel/spinel_kit/lib/spinel_kit/git.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/tinynn_cuda.rb lib/toy/dev/toy_describe_flow.rb lib/toy/train/toy_drift_grad.rb lib/toy/train/toy_gguf_writer.rb lib/toy/dev/toy_tap.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS)
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
examples/example_train_from_scratch: examples/example_train_from_scratch_cpu
	@printf '#!/bin/sh\n# Auto-generated by Makefile. DEVICE selects the backend binary.\n# Edit examples/06_train_from_scratch.rb (cpu) for behaviour; CUDA mirror is auto-generated by prep/gen_cuda_mirror.rb.\ncase "$${DEVICE:-cpu}" in\n  cpu|"") exec "$$(dirname "$$0")/example_train_from_scratch_cpu" "$$@" ;;\n  cuda)   exec "$$(dirname "$$0")/example_train_from_scratch_cuda" "$$@" ;;\n  metal)  echo "DEVICE=metal not yet supported for training (inference only)" >&2; exit 2 ;;\n  *)      echo "DEVICE=$${DEVICE} not recognised (want cpu|cuda)" >&2; exit 2 ;;\nesac\n' > $@
	@chmod +x $@
example_train_from_scratch: examples/example_train_from_scratch
example_train_from_scratch_cuda: examples/example_train_from_scratch_cuda

# GPT-2 from-scratch via the GPT2SeqEngine library API (the curated GPT-2 demo;
# CLI surface is `toy train from-scratch --arch gpt2`). Memorizes a synthetic
# sequence so CE visibly collapses; exercises the vendored LayerNorm/GELU kernels.
examples/gpt2_train: examples/gpt2_train.rb lib/toy.rb \
		lib/toy/llm/engine/gpt2_seq_engine.rb lib/toy/models/transformer.rb \
		lib/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) | libexec
	$(SPINEL) $< -o $@
gpt2_train: examples/gpt2_train
.PHONY: gpt2_train

examples: toy-infer example_train example_train_from_scratch gpt2_train

# Phase 0.6 — CUDA-mirror generator. The CPU file is the source of
# truth; the CUDA file is auto-generated by prep/gen_cuda_mirror.rb.
# `make verify-mirrors` exits non-zero if any committed CUDA mirror
# has drifted from what the generator would produce.
gen-mirrors:
	@ruby prep/gen_cuda_mirror.rb

# Mirrors are off-disk build artifacts (gitignored), so there is no committed
# copy to drift against. verify-mirrors now regenerates every mirror (incl. the
# Metal twins, which no Linux build consumes) and then re-runs the generator in
# --verify mode: this asserts the generator is healthy and IDEMPOTENT (generate
# == verify), the only invariant left once nothing is committed.
verify-mirrors:
	@ruby prep/gen_cuda_mirror.rb
	@ruby prep/gen_cuda_mirror.rb --verify

# Mirrors generated at build time (off-disk; gitignored). Every runner rule
# lists the mirror .rb as a prerequisite, so Make regenerates it on demand from
# the CPU source of truth + the generator. `--backend` writes one backend, so
# each target rebuilds exactly itself. These mirror MIRRORABLE in
# prep/gen_cuda_mirror.rb — keep the two lists in sync. STATIC pattern rules
# (targets restricted to this explicit list) so hand-written mirrors like
# lib/tinynn_cuda.rb / lib/toy/models/transformer_lm_cuda.rb are NOT captured.
MIRROR_CUDA := \
  lib/toy/llm/primitives/rms_norm_cuda.rb lib/toy/llm/primitives/rope_cuda.rb \
  lib/toy/llm/primitives/swiglu_cuda.rb lib/toy/llm/primitives/gqa_cuda.rb \
  lib/toy/llm/blocks/transformer_block_cuda.rb lib/toy/llm/archs/llama_arch_cuda.rb \
  lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/llm/engine/gpt2_seq_engine_cuda.rb \
  lib/toy/llm/recipes/from_scratch_cuda.rb lib/toy/llm/recipes/lora_cuda.rb \
  lib/toy/llm/recipes/warm_start_cuda.rb \
  lib/toy_smollm2_ffi_kv_cuda.rb \
  lib/gpt2_ffi_cuda.rb lib/gpt2_ffi_kv_cuda.rb \
  examples/06_train_from_scratch_cuda.rb examples/smoke_projection_lens_cuda.rb
MIRROR_METAL := $(MIRROR_CUDA:_cuda.rb=_metal.rb)

$(MIRROR_CUDA): %_cuda.rb: %.rb prep/gen_cuda_mirror.rb
	@ruby prep/gen_cuda_mirror.rb --backend cuda $<
$(MIRROR_METAL): %_metal.rb: %.rb prep/gen_cuda_mirror.rb
	@ruby prep/gen_cuda_mirror.rb --backend metal $<

# Parity-checks vs native TransformerLM.forward.

# Tep+Spinel HTTP server demos. See tep_demo/README.md. Builds bypass
# tep's translator (we use spinel directly on the spinelgems-vendored
# tep tree at vendor/spinel/tep/lib/, produced by `make vendor-tep`).
tep_demo/hello: tep_demo/hello_api.rb vendor/spinel/tep/lib/tep.rb
	$(SPINEL) tep_demo/hello_api.rb -o tep_demo/hello

# Inference API: /generate?n=N runs greedy generation via FullForwardFFICache.
tep_demo/api: tep_demo/legacy/inference_api.rb vendor/spinel/tep/lib/tep.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tep_demo/legacy/inference_api.rb -o tep_demo/api

# --- ggml vendor ------------------------------------------------------------
# Vendor patches that must land before any ggml build target. See
# vendor-patches/README.md for the per-patch rationale.
GGML_PATCHES := \
	vendor-patches/0001-cuda-buffer_from_ptr.patch \
	vendor-patches/0002-cuda-buffer_from_ptr-reuse-iface.patch \
	vendor-patches/0003-cuda-buffer_from_ptr-copy-mode.patch \
	vendor-patches/0004-cuda-cpy-strided.patch \
	vendor-patches/0005-concat-backward.patch \
	vendor-patches/0006-getrows-back-large-vocab.patch \
	vendor-patches/0007-gpt2-backward-kernels.patch \
	vendor-patches/0008-mul-mat-backward-mixed-precision.patch \
	vendor-patches/0009-sched-unsupported-node-diagnostic.patch

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
	git init -q $(GGML_DIR)
	cd $(GGML_DIR) && git remote add origin $(GGML_REPO) \
	  && git fetch -q --depth 1 origin $(GGML_REV) \
	  && git checkout -q FETCH_HEAD

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
# once the static archive is built. Lets `make setup` / `toy install`
# chain through without redoing the ~5 s incremental cmake check on
# every invocation.
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
# --- gem release prep (toy#45) ----------------------------------------------
# The gem ships PRISTINE pinned ggml (patches apply at the consumer's vendor
# step), so reset the working tree's ggml before `gem build`. Re-run setup-ggml
# afterwards to restore the dev build.
gem-prep: $(GGML_DIR)/CMakeLists.txt
	cd $(GGML_DIR) && git reset --hard FETCH_HEAD >/dev/null 2>&1 || git reset --hard HEAD >/dev/null
	rm -f $(GGML_DIR)/.patched
	@echo "ggml pristine at $$(cd $(GGML_DIR) && git rev-parse --short HEAD); now: gem build toy.gemspec"
.PHONY: gem-prep

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

tinynn/ab_smoke_silu: tinynn/ab_smoke_silu.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

ab-smoke-mul: tinynn/ab_smoke_mul
	./tinynn/ab_smoke_mul

tinynn/ab_smoke_mul: tinynn/ab_smoke_mul.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
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

tinynn/ab_smoke: tinynn/ab_smoke.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke.rb -o tinynn/ab_smoke

tinynn/ab_smoke_add: tinynn/ab_smoke_add.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_add.rb -o tinynn/ab_smoke_add

# E1.1 / GH#13 — Conv2D smoke + JSON dump for PyTorch parity.
tinynn/ab_smoke_conv2d: tinynn/ab_smoke_conv2d.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_conv2d.rb -o tinynn/ab_smoke_conv2d

# E1.2 / GH#13 — patch_embed composite smoke + parity dump.
tinynn/ab_smoke_patch_embed: tinynn/ab_smoke_patch_embed.rb lib/toy/models/transformer.rb lib/tinynn.rb lib/toy/models/toy_vit.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_patch_embed.rb -o tinynn/ab_smoke_patch_embed

# E1.3 / GH#13 — ViT-Tiny forward + training smoke.
examples/smoke_vit_tiny: examples/smoke_vit_tiny.rb lib/toy/llm/engine/vit_tiny_engine.rb lib/toy/models/toy_vit.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# E1.5 / GH#13 — image-loader smoke.
examples/smoke_image_loader: examples/smoke_image_loader.rb lib/toy/io/toy_image_loader.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# E1.6 / GH#13 — ViT-Tiny training driver.
examples/example_train_vit_tiny: examples/07_train_vit_tiny.rb lib/toy/llm/engine/vit_tiny_engine.rb lib/toy/models/toy_vit.rb lib/toy/models/toy_smollm2.rb lib/toy/io/toy_image_loader.rb lib/toy/train/toy_lr_schedule.rb lib/toy/train/toy_drift_grad.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
example_train_vit_tiny: examples/example_train_vit_tiny

tinynn/ab_smoke_gelu: tinynn/ab_smoke_gelu.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_gelu.rb -o tinynn/ab_smoke_gelu

tinynn/ab_smoke_rms_norm: tinynn/ab_smoke_rms_norm.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_rms_norm.rb -o tinynn/ab_smoke_rms_norm

tinynn/ab_smoke_softmax: tinynn/ab_smoke_softmax.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_softmax.rb -o tinynn/ab_smoke_softmax

tinynn/ab_smoke_flash_attn: tinynn/ab_smoke_flash_attn.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_flash_attn.rb -o tinynn/ab_smoke_flash_attn

tinynn/ab_smoke_q8_kv: tinynn/ab_smoke_q8_kv.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_q8_kv.rb -o tinynn/ab_smoke_q8_kv

tinynn/ab_smoke_moe_ffn: tinynn/ab_smoke_moe_ffn.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_moe_ffn.rb -o tinynn/ab_smoke_moe_ffn

tinynn/ab_smoke_transpose: tinynn/ab_smoke_transpose.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_transpose.rb -o tinynn/ab_smoke_transpose

tinynn/ab_smoke_scale: tinynn/ab_smoke_scale.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_scale.rb -o tinynn/ab_smoke_scale

tinynn/ab_smoke_pipeline: tinynn/ab_smoke_pipeline.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_pipeline.rb -o tinynn/ab_smoke_pipeline

# Chained FFNFFICache parity: pre, hidden, out vs hand-rolled native.
ab-smoke-ffncache: tinynn/ab_smoke_ffncache
	./tinynn/ab_smoke_ffncache

tinynn/ab_smoke_ffncache: tinynn/ab_smoke_ffncache.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_ffncache.rb -o tinynn/ab_smoke_ffncache

# ggml-native AdamW step (opt_step_adamw) parity vs project's plain-Adam.
ab-smoke-adamw-op: tinynn/ab_smoke_adamw_op
	./tinynn/ab_smoke_adamw_op

tinynn/ab_smoke_adamw_op: tinynn/ab_smoke_adamw_op.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_adamw_op.rb -o tinynn/ab_smoke_adamw_op

# Persistent-tensor architecture check: data uploaded to a ctx_w tensor
# survives a compute cycle.
ab-smoke-persistent: tinynn/ab_smoke_persistent
	./tinynn/ab_smoke_persistent

tinynn/ab_smoke_persistent: tinynn/ab_smoke_persistent.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_persistent.rb -o tinynn/ab_smoke_persistent

# Dual-cgraph + persistent-weights design check: forward reads t_w;
# adam mutates t_w in place; forward sees the new value.
ab-smoke-dual-graph: tinynn/ab_smoke_dual_graph
	./tinynn/ab_smoke_dual_graph

tinynn/ab_smoke_dual_graph: tinynn/ab_smoke_dual_graph.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_dual_graph.rb -o tinynn/ab_smoke_dual_graph

# M2 foundation: view_2d + cpy to write a single row into a persistent
# (max_T, d_head) KV buffer at a runtime-baked position.
ab-smoke-kv-write: tinynn/ab_smoke_kv_write
	./tinynn/ab_smoke_kv_write

tinynn/ab_smoke_kv_write: tinynn/ab_smoke_kv_write.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_kv_write.rb -o tinynn/ab_smoke_kv_write

# M2 prototype: single-step decode through a KV cache. Pre-fills K/V
# for positions 0..POS-1, writes k_new/v_new at POS, computes scores
# + soft_max_ext + head_out. Parity vs hand-rolled native.
ab-smoke-kv-attn: tinynn/ab_smoke_kv_attn
	./tinynn/ab_smoke_kv_attn

tinynn/ab_smoke_kv_attn: tinynn/ab_smoke_kv_attn.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_kv_attn.rb -o tinynn/ab_smoke_kv_attn

# M1.2: full single-block forward through the persistent graph.
# Parity vs native TransformerLM.forward() at n_layers=1, n_heads=2.
ab-smoke-full-forward-block: tinynn/ab_smoke_full_forward_block
	./tinynn/ab_smoke_full_forward_block

tinynn/ab_smoke_full_forward_block: tinynn/ab_smoke_full_forward_block.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_full_forward_block.rb -o tinynn/ab_smoke_full_forward_block

# Wallclock bench: native TransformerLM.forward vs FullForwardFFICache.
full-forward-bench: tinynn/full_forward_bench
	./tinynn/full_forward_bench

tinynn/full_forward_bench: tinynn/full_forward_bench.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/full_forward_bench.rb -o tinynn/full_forward_bench

full-forward-bench-cuda: tinynn/full_forward_bench_cuda
	./tinynn/full_forward_bench_cuda

tinynn/full_forward_bench_cuda: tinynn/full_forward_bench_cuda.rb lib/toy/models/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) tinynn/full_forward_bench_cuda.rb -o tinynn/full_forward_bench_cuda

ab-smoke-dual-graph-cuda: tinynn/ab_smoke_dual_graph_cuda
	./tinynn/ab_smoke_dual_graph_cuda

tinynn/ab_smoke_dual_graph_cuda: tinynn/ab_smoke_dual_graph_cuda.rb lib/toy/models/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) tinynn/ab_smoke_dual_graph_cuda.rb -o tinynn/ab_smoke_dual_graph_cuda

ab-smoke-adamw-op-cuda: tinynn/ab_smoke_adamw_op_cuda
	./tinynn/ab_smoke_adamw_op_cuda

tinynn/ab_smoke_adamw_op_cuda: tinynn/ab_smoke_adamw_op_cuda.rb lib/toy/models/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) tinynn/ab_smoke_adamw_op_cuda.rb -o tinynn/ab_smoke_adamw_op_cuda

# A/B harness for the "fuse-or-not" question: N_HEADS small matmuls vs
# 1 batched matmul at LoRA-Q shape. Override D_MODEL / N_HEADS / R / T
# via env to sweep launch-overhead vs compute-bound regimes. See
# docs/heavy-train-attribution-2026-05-24.md.
ab-smoke-lora-fused-cuda: tinynn/ab_smoke_lora_fused_cuda
	./tinynn/ab_smoke_lora_fused_cuda

tinynn/ab_smoke_lora_fused_cuda: tinynn/ab_smoke_lora_fused_cuda.rb lib/toy/models/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' tinynn/ab_smoke_lora_fused_cuda.rb -o tinynn/ab_smoke_lora_fused_cuda

# Transformer-shape sized parity + wallclock comparison.
ab-smoke-big: tinynn/ab_smoke_big
	./tinynn/ab_smoke_big

tinynn/ab_smoke_big: tinynn/ab_smoke_big.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_big.rb -o tinynn/ab_smoke_big

ab-smoke-matmul-variants: tinynn/ab_smoke_matmul_variants
	./tinynn/ab_smoke_matmul_variants

tinynn/ab_smoke_matmul_variants: tinynn/ab_smoke_matmul_variants.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_matmul_variants.rb -o tinynn/ab_smoke_matmul_variants

ab-smoke-back: tinynn/ab_smoke_back
	./tinynn/ab_smoke_back

tinynn/ab_smoke_back: tinynn/ab_smoke_back.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_back.rb -o tinynn/ab_smoke_back

ab-smoke-gelu-back: tinynn/ab_smoke_gelu_back
	./tinynn/ab_smoke_gelu_back

tinynn/ab_smoke_gelu_back: tinynn/ab_smoke_gelu_back.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_gelu_back.rb -o tinynn/ab_smoke_gelu_back

ab-smoke-cegrad: tinynn/ab_smoke_cegrad
	./tinynn/ab_smoke_cegrad

tinynn/ab_smoke_cegrad: tinynn/ab_smoke_cegrad.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_cegrad.rb -o tinynn/ab_smoke_cegrad

ab-smoke-adam: tinynn/ab_smoke_adam
	./tinynn/ab_smoke_adam

tinynn/ab_smoke_adam: tinynn/ab_smoke_adam.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_adam.rb -o tinynn/ab_smoke_adam

gguf-smoke: tinynn/gguf_smoke
	./tinynn/gguf_smoke

tinynn/gguf_smoke: tinynn/gguf_smoke.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gguf_smoke.rb -o tinynn/gguf_smoke

# Walks every tensor in data/distilgpt2-f32.gguf via tnn_gguf_*. Used to
# confirm large HF-converted GGUFs roundtrip through the project FFI.
gguf-inspect: tinynn/gguf_inspect
	./tinynn/gguf_inspect

tinynn/gguf_inspect: tinynn/gguf_inspect.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gguf_inspect.rb -o tinynn/gguf_inspect

# GPT2LM build smoke: confirm lib/toy/models/gpt2.rb Spinel-compiles and the
# forward shapes line up. Toy dims, random weights — values mean nothing.
gpt2-build-smoke: tinynn/gpt2_build_smoke
	./tinynn/gpt2_build_smoke

tinynn/gpt2_build_smoke: tinynn/gpt2_build_smoke.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb
	$(SPINEL) tinynn/gpt2_build_smoke.rb -o tinynn/gpt2_build_smoke

# Load distilgpt2-f32.gguf into a GPT2LM and print sentinel weights
# per category. Verifies name mapping + per-head split before forward.
gpt2-load-smoke: tinynn/gpt2_load_smoke
	./tinynn/gpt2_load_smoke

tinynn/gpt2_load_smoke: tinynn/gpt2_load_smoke.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_load_smoke.rb -o tinynn/gpt2_load_smoke

# data/prompt_ids.txt, loads weights from data/distilgpt2-f32.gguf,
# greedy-generates N_NEW tokens via native Mat forward, writes the
# full ID sequence back. Decode with prep/tokens.py decode.

# Native Mat GPT-2 inference (DistilGPT2 / GPT-2 family).
#
gpt2:        demos/gpt2
demos/gpt2: demos/gpt2.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_gpt2.rb lib/toy/models/toy_gpt2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# SmolLM2-135M (llama-family) inference via Toy::SmolLM2.
# Tokenization is host-side: ./prep/smollm2_tokens.py encode "..."
smollm2:        demos/smollm2
demos/smollm2: demos/smollm2.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# SmolLM2-135M FFI KV-cache (CPU).
smollm2_kv:        demos/smollm2_kv
demos/smollm2_kv: demos/smollm2_kv.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/models/toy_smollm2_loader.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Qwen2.5 Mat-mediated KV-cache (CPU). The slow, correct reference path.
# Run with `GGUF=data/qwen25-1.5b-f32.gguf ./demos/qwen25_kv` etc.
qwen25_kv:        demos/qwen25_kv
demos/qwen25_kv: demos/qwen25_kv.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/models/toy_smollm2_loader.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Qwen2.5 Phase-2 mmap inference (CPU). Canonical performance path.
qwen25_native_mmap:        demos/qwen25_native_mmap
demos/qwen25_native_mmap: demos/qwen25_native_mmap.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Phase 0.7 acceptance gates: 0.5B (f32 + Q8) + 1.5B + 3B greedy-decode
# parity against locked-in golden token-ID sequences. Run before tagging
# a release; see docs/design/phase-07-acceptance.md.
qwen25_acceptance:        demos/qwen25_acceptance
demos/qwen25_acceptance: demos/qwen25_acceptance.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Inference bench (CPU). Long warmup + long prefill + per-token stats.
# Pick model via GGUF env; see docs/design/bench-cuda-2026-05-21.md.
qwen25_bench_cpu:        demos/qwen25_bench_cpu
demos/qwen25_bench_cpu: demos/qwen25_bench_cpu.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Inference bench (CUDA). Same shape as the CPU bench for side-by-side.
qwen25_bench_cuda:        demos/qwen25_bench_cuda
demos/qwen25_bench_cuda: demos/qwen25_bench_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS)
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F1.2 step 2: LoRA-Q forward-parity gate. Loads SmolLM2-135M twice
# (baseline + LoRA r=16 B=0), asserts bit-identical generated IDs.
smollm2_lora_forward:        demos/smollm2_lora_forward
demos/smollm2_lora_forward: demos/smollm2_lora_forward.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# F1.2 step 3: backward through the full SmolLM2 decode graph,
# layer-0 LoRA-Q updated via SGD. Requires the vendored CONCAT
# backward in vendor/ggml/src/ggml.c.
smollm2_lora_train_step:        demos/smollm2_lora_train_step
demos/smollm2_lora_train_step: demos/smollm2_lora_train_step.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# F1.2 step 4: all-layers LoRA-Q SGD on real CE loss against a rare
# target token. 540 opt_step nodes (30 layers × 9 heads × 2 params).
# Acceptance: monotonic decrease over 20 steps.
smollm2_lora_train_ce:        demos/smollm2_lora_train_ce
demos/smollm2_lora_train_ce: demos/smollm2_lora_train_ce.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# F2 step 1: CUDA mirror of the LoRA forward parity gate.
smollm2_lora_forward_cuda:        demos/smollm2_lora_forward_cuda
demos/smollm2_lora_forward_cuda: demos/smollm2_lora_forward_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F2 step 2: CUDA mirror of the multi-layer SGD CE training smoke.
smollm2_lora_train_ce_cuda:        demos/smollm2_lora_train_ce_cuda
demos/smollm2_lora_train_ce_cuda: demos/smollm2_lora_train_ce_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# Task #70 diagnostic — same CE smoke but with every graph_b node
# pinned. Confirms sched intermediate-grad aliasing is the CPU
# divergence's root cause. See docs/design/task70-root-cause-2026-05-21.md.
smollm2_lora_train_ce_pinned:        demos/smollm2_lora_train_ce_pinned
demos/smollm2_lora_train_ce_pinned: demos/smollm2_lora_train_ce_pinned.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# F1.2 step 5: AdamW training with per-step m/v preservation via
# tnn_graph_reset_grads_only. Converges 7.5 → 0.09 in 20 SGD steps
# at LR=1e-3 — proper SFT-shaped learning curve.
smollm2_lora_train_adamw_cuda:        demos/smollm2_lora_train_adamw_cuda
demos/smollm2_lora_train_adamw_cuda: demos/smollm2_lora_train_adamw_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F1.2 step 6a: multi-target AdamW SFT-shaped training. Cycles through
# 5 target tokens × 10 epochs at the same prefix; expects loss to
# drop on average + per-target. 10.8 → 3.6 in 10 epochs. Foundation
# for step 6b (multi-position) and step 7 (real alpaca dataset).
smollm2_lora_sft_multi_cuda:        demos/smollm2_lora_sft_multi_cuda
demos/smollm2_lora_sft_multi_cuda: demos/smollm2_lora_sft_multi_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F1.2 step 6b — multi-position SFT (cycle pos4 / pos5). Validates
# that persistent Adam m/v (allocated by enable_lora_q_adamw! +
# realize_for_mmap) survive tnn_reset_for_rebuild between cycles.
smollm2_lora_sft_multipos_cuda:        demos/smollm2_lora_sft_multipos_cuda
demos/smollm2_lora_sft_multipos_cuda: demos/smollm2_lora_sft_multipos_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# M3 step 1 — sequence-mode forward parity at T=1.
# LlamaSeqForwardFFICache.forward([id], [0]) must match
# SmolLM2KVFFICache + decode_step(id, 0). See
# docs/design/m3-seq-forward-2026-05-21.md.
smollm2_seq_parity:        demos/smollm2_seq_parity
demos/smollm2_seq_parity: demos/smollm2_seq_parity.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# M3 step 2 — T=4 trajectory parity (CPU). Per-position seq logits must
# match the decode_step trajectory; proves causal-mask + multi-pos RoPE.
smollm2_seq_parity_t4:        demos/smollm2_seq_parity_t4
demos/smollm2_seq_parity_t4: demos/smollm2_seq_parity_t4.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# M3 step 2 — CUDA mirror. T=1 and T=4 vs CPU decode_step trajectory.
smollm2_seq_parity_cuda:        demos/smollm2_seq_parity_cuda
demos/smollm2_seq_parity_cuda: demos/smollm2_seq_parity_cuda.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

smollm2_seq_parity_t4_cuda:        demos/smollm2_seq_parity_t4_cuda
demos/smollm2_seq_parity_t4_cuda: demos/smollm2_seq_parity_t4_cuda.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# M3 step 3 — seq-mode LoRA training smoke (CPU). One forward + backward
# + opt_step over T positions; loss should decrease over N steps.
smollm2_seq_train:        demos/smollm2_seq_train
demos/smollm2_seq_train: demos/smollm2_seq_train.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

smollm2_seq_train_cuda:        demos/smollm2_seq_train_cuda
demos/smollm2_seq_train_cuda: demos/smollm2_seq_train_cuda.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F3 — full fine-tune on CUDA. Every per-block weight tensor is
# writable F32 + AdamW state; opt_step on each. See
# docs/roadmap/f3-full-finetune-2026-05-21.md.
smollm2_seq_full_finetune_cuda:        demos/smollm2_seq_full_finetune_cuda
demos/smollm2_seq_full_finetune_cuda: demos/smollm2_seq_full_finetune_cuda.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F4 (QLoRA) on CUDA via realize_for_q8_copy. Q8 base in standard
# CUDA buffer + F32 LoRA adapter; bypasses the BYO-pointer padding bug.
smollm2_seq_qlora_cuda:        demos/smollm2_seq_qlora_cuda
demos/smollm2_seq_qlora_cuda: demos/smollm2_seq_qlora_cuda.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# Training step-time bench. MODE=lora|ft; STEPS=N; GGUF=path.
seq_train_bench_cuda:        demos/seq_train_bench_cuda
demos/seq_train_bench_cuda: demos/seq_train_bench_cuda.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS)
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# Per-phase training-step bench (CPU + CUDA). Times graph_reset /
# uploads / compute_backward / download separately. Doc:
# docs/design/bench-train-2026-05-21.md.
smollm2_lora_train_bench:        demos/smollm2_lora_train_bench
demos/smollm2_lora_train_bench: demos/smollm2_lora_train_bench.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

smollm2_lora_train_bench_cuda:        demos/smollm2_lora_train_bench_cuda
demos/smollm2_lora_train_bench_cuda: demos/smollm2_lora_train_bench_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# Task #70 grad-magnitude probes (per-layer maxabs(grad_A), maxabs(grad_B)).
smollm2_lora_grad_probe:        demos/smollm2_lora_grad_probe
demos/smollm2_lora_grad_probe: demos/smollm2_lora_grad_probe.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

smollm2_lora_grad_probe_cuda:        demos/smollm2_lora_grad_probe_cuda
demos/smollm2_lora_grad_probe_cuda: demos/smollm2_lora_grad_probe_cuda.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# Qwen2.5 Phase-2 mmap inference (CUDA). Requires `make setup-ggml-cuda`.
qwen25_native_mmap_cuda:        demos/qwen25_native_mmap_cuda
demos/qwen25_native_mmap_cuda: demos/qwen25_native_mmap_cuda.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/models/toy_smollm2_loader.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# SmolLM2-135M FFI KV-cache (CUDA).
smollm2_kv_cuda:        demos/smollm2_kv_cuda
demos/smollm2_kv_cuda: demos/smollm2_kv_cuda.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/models/toy_smollm2_loader.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) $< -o $@

# TinyLlama-1.1B demo. Uses the same Toy::SmolLM2 / FFI KV CUDA stack
# (llama-family architecture); just configured for the larger shape.
tinyllama_kv_cuda:        demos/tinyllama_kv_cuda
demos/tinyllama_kv_cuda: demos/tinyllama_kv_cuda.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/models/toy_smollm2_loader.rb lib/toy_smollm2_ffi_kv_cuda.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) $< -o $@

tinyllama:        demos/tinyllama
demos/tinyllama: demos/tinyllama.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

tinyllama_kv:        demos/tinyllama_kv
demos/tinyllama_kv: demos/tinyllama_kv.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/models/toy_smollm2_loader.rb lib/toy_smollm2_ffi_kv.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Print the Phuong–Hutter algorithm cards for both models. No
# inference — just emit the structured pseudocode. Source-of-truth
# for the round-trip work (task #33).
algorithm_cards:        demos/algorithm_cards
demos/algorithm_cards: demos/algorithm_cards.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_gpt2.rb lib/toy/models/toy_smollm2.rb lib/toy/models/toy_gpt2_loader.rb lib/toy/models/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# TinyStories from-scratch training via Toy::Trainer.
#
train:        demos/train
demos/train: demos/train.rb lib/toy/train/toy_trainer.rb lib/toy/models/transformer.rb lib/toy/train/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Parity probe: one forward at distilgpt2 shape, dump last-row logits
# to data/ours_logits.txt. Pair with prep/parity.py for the HF reference.
gpt2-parity: tinynn/gpt2_parity
	./tinynn/gpt2_parity

tinynn/gpt2_parity: tinynn/gpt2_parity.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_parity.rb -o tinynn/gpt2_parity

# FFI parity probe: persistent ggml graph with LayerNorm + biases.
# Dumps last-row logits to data/ours_ffi_logits.txt.
gpt2-ffi-parity: tinynn/gpt2_ffi_parity
	./tinynn/gpt2_ffi_parity

tinynn/gpt2_ffi_parity: tinynn/gpt2_ffi_parity.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/gpt2_ffi.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_ffi_parity.rb -o tinynn/gpt2_ffi_parity

# Apples-to-apples bench: native Mat vs FFI on the same forward.
# Re-encode data/prompt_ids.txt first so prompt length matches T_SEQ=5.
gpt2-bench: tinynn/gpt2_bench
	./tinynn/gpt2_bench

tinynn/gpt2_bench: tinynn/gpt2_bench.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/gpt2_ffi.rb lib/gpt2_ffi_kv.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_bench.rb -o tinynn/gpt2_bench

# Ruby BPE smoke: load vocab/merges, encode + roundtrip-decode some
# fixed prompts. Compare against prep/tokens.py output.
bpe-smoke: tinynn/bpe_smoke
	./tinynn/bpe_smoke

tinynn/bpe_smoke: tinynn/bpe_smoke.rb lib/toy/models/transformer.rb lib/toy/io/bpe.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/bpe_smoke.rb -o tinynn/bpe_smoke

# KV-cache parity probe: prefill the prompt one token at a time through
# GPT2KVFFICache, dump last-position logits.
gpt2-kv-parity: tinynn/gpt2_kv_parity
	./tinynn/gpt2_kv_parity

tinynn/gpt2_kv_parity: tinynn/gpt2_kv_parity.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/gpt2_ffi_kv.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_kv_parity.rb -o tinynn/gpt2_kv_parity

# --- CUDA mirrors of the GPT-2 demos / parity / bench --------------
# All require `make setup-ggml-cuda` to have produced
# vendor/ggml/build-cuda first. Built on the gx10 (NVIDIA GB10);
# the Mac build doesn't have CUDA.

CUDA_GPT2_DEPS = lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb \
                 lib/toy/train/training.rb lib/tinynn.rb lib/tinynn_cuda.rb \
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

tinynn/ab_smoke_embed: tinynn/ab_smoke_embed.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_embed.rb -o tinynn/ab_smoke_embed

ab-smoke-sgd: tinynn/ab_smoke_sgd
	./tinynn/ab_smoke_sgd

tinynn/ab_smoke_sgd: tinynn/ab_smoke_sgd.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_sgd.rb -o tinynn/ab_smoke_sgd

# F1.2 step 1: multi-step LoRA convergence via the F1.1 in-graph
# optimizer. Toy shape; SGD; 60 steps; asserts final loss < 10% of
# initial (passes at ~10e-13 of initial).
ab-smoke-lora-train: tinynn/ab_smoke_lora_train
	./tinynn/ab_smoke_lora_train

tinynn/ab_smoke_lora_train: tinynn/ab_smoke_lora_train.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_lora_train.rb -o tinynn/ab_smoke_lora_train

# Forward-only smoke: does TransformerLM#forward run at current Spinel
# master? (The #473 SIGBUS is in backward; forward might be OK.)
forward-smoke: tinynn/forward_smoke
	./tinynn/forward_smoke

tinynn/forward_smoke: tinynn/forward_smoke.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/forward_smoke.rb -o tinynn/forward_smoke

persistent-bench: tinynn/persistent_bench
	./tinynn/persistent_bench

tinynn/persistent_bench: tinynn/persistent_bench.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/persistent_bench.rb -o tinynn/persistent_bench

persistent-bench-cuda: tinynn/persistent_bench_cuda
	./tinynn/persistent_bench_cuda

tinynn/persistent_bench_cuda: tinynn/persistent_bench_cuda.rb lib/toy/models/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/persistent_bench_cuda.rb -o tinynn/persistent_bench_cuda

persistent-bench-big: tinynn/persistent_bench_big
	./tinynn/persistent_bench_big

tinynn/persistent_bench_big: tinynn/persistent_bench_big.rb lib/toy/models/transformer.rb lib/tinynn.rb tinynn/libtinynn_ggml.a
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
ifneq ($(UNAME_S),Darwin)
	@echo "tinynn_backend_metal.o: macOS-only (Objective-C + Metal frameworks); uname -s = $(UNAME_S)"; exit 1
endif
	$(CC) $(CFLAGS) -x objective-c $(GGML_INC) -c $< -o $@

tinynn/libtinynn_ggml_metal.a: tinynn/tinynn_backend_metal.o
	ar $(ARFLAGS) $@ $<

tinynn/ab_smoke_cuda: tinynn/ab_smoke_cuda.rb lib/toy/models/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_cuda.rb -o tinynn/ab_smoke_cuda

# Consolidated CUDA parity test: matmul + add + gelu + rms_norm + softmax + scale + ffn_pipeline.
ab-smoke-all-cuda: tinynn/ab_smoke_all_cuda
	./tinynn/ab_smoke_all_cuda

tinynn/ab_smoke_all_cuda: tinynn/ab_smoke_all_cuda.rb lib/toy/models/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_all_cuda.rb -o tinynn/ab_smoke_all_cuda

# Transformer-shape parity + wallclock bench on CUDA (GB10).
ab-smoke-big-cuda: tinynn/ab_smoke_big_cuda
	./tinynn/ab_smoke_big_cuda

tinynn/ab_smoke_big_cuda: tinynn/ab_smoke_big_cuda.rb lib/toy/models/transformer.rb lib/tinynn_cuda.rb tinynn/libtinynn_ggml.a
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
	      libexec/toy-infer libexec/toy-train libexec/toy-train-cuda libexec/toy-train-lora-cuda libexec/toy-eval libexec/toy-eval-lmc libexec/toy-serve examples/example_train \
	      libexec/toy-infer-metal libexec/toy-eval-metal libexec/toy-train-metal

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
        toy-infer-metal toy-eval-metal toy-train-metal gate-metal \
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
