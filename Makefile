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

# Toolchain pin (toy#119; see docs/consuming-toy.md). toy is built and
# byte-gated against ONE spinel rev; compiling with any other rev is how
# gate-franken-llama "broke at HEAD" with zero source changes (an old
# ~/sites/spinel miscompiled the engine — backward never built,
# graph_reset hit grads==NULL). Advance PINNED_SPINEL in the same
# commit as the consuming-toy.md pin, per the toy#95 sweep protocol.
# The default SPINEL_DIR prefers the pinned scratch toolchain when it
# exists (gx10), falling back to ~/sites/spinel (which the guard then
# vets by rev, NOT by path — a Mac clone at the pin passes).
PINNED_SPINEL := dc5b8f17
SPINEL_DIR  ?= $(firstword $(wildcard /srv/data/scratch/spinel-dc5b8f17) $(HOME)/sites/spinel)
SPINEL_BIN  ?= $(SPINEL_DIR)/spinel

SPINEL_ACTUAL := $(shell git -C $(SPINEL_DIR) rev-parse --short=8 HEAD 2>/dev/null || echo unknown)
ifneq ($(SPINEL_SKIP_PIN_CHECK),1)
ifneq ($(SPINEL_ACTUAL),$(PINNED_SPINEL))
$(error toy is pinned to spinel $(PINNED_SPINEL) but SPINEL_DIR=$(SPINEL_DIR) is at '$(SPINEL_ACTUAL)'. Export SPINEL_DIR=<checkout at $(PINNED_SPINEL)> (gx10: /srv/data/scratch/spinel-dc5b8f17), or SPINEL_SKIP_PIN_CHECK=1 to bypass deliberately (e.g. a pin-advance sweep))
endif
endif

# toy#69 — sig/*.rbs type roots. Every Spinel compile seeds the analyzer
# with toy's shipped RBS tree (`--rbs sig`): uncalled public methods
# keep their DECLARED param/return/ivar types instead of widening to
# poly (the spinel-dev#11/#12 facet family). Seeds are ADVISORY —
# inference runs on top and widens on observed contradiction (spinel
# docs/RBS-EXTRACT.md), and the full gate sweep was byte-exact at
# adoption. Vendored gems' sig roots ride along through the gitignored
# sig/vendor symlink -> ../vendor/spinel/sig, refreshed by `make
# vendor-tep` (spinel takes ONE --rbs dir; spinel_rbs_extract walks it
# recursively and follows symlinks). Set SPINEL_RBS= (empty) to compile
# without seeds when chasing an analyzer issue.
SPINEL_RBS ?= --rbs $(CURDIR)/sig

# Gate harnesses resolve the toolchain THROUGH make (single source of
# truth + the pin guard runs at parse time) instead of a hardcoded
# ~/sites/spinel fallback — the toy#119 hazard class; consumer_gate and
# poly_degrade_gate were its last two carriers.
.PHONY: print-spinel-dir
print-spinel-dir:
	@echo $(SPINEL_DIR)

# spinel_kit resolves as a real require ("spinel_kit/git") on BOTH paths:
# spin provides the dependency root; the direct-spinel path points -I at
# the vendored gem's lib/ (gem layout). toy#107 deps leg.
SPINEL_INC := -I $(CURDIR)/vendor/spinel/spinel_kit/lib

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
  SPINEL = $(SPINEL_BIN) $(SPINEL_RBS) $(SPINEL_INC)
else
  SPINEL = $(QUIETLY) \
      'cannot resolve call to' \
      'ignoring duplicate libraries' \
      -- $(SPINEL_BIN) $(SPINEL_RBS) $(SPINEL_INC)
endif
# Sentinel deps so example/demo Spinel-compiled binaries get re-spun
# when the Spinel compiler itself changes. Without this, stale .o /
# .a in tinynn/ combined with newer Spinel C codegen can produce
# misaligned binaries that segfault at init (Tao hit this 2026-05-26
# after pulling Spinel 2183a92 — the lib archives weren't rebuilt).
# Track the compiler BINARY: post the Ruby→C rewrite there is no
# spinel_analyze/spinel_codegen at the checkout root (the Ruby backend
# moved to legacy/, oracle-only), just the single `spinel` binary —
# the right rebuild trigger, present on both the legacy and C layouts
# (verified byte-exact green on the union pin; toy#101 Part 1).
SPINEL_DEPS := $(SPINEL_BIN)

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
#
# MUST be the FULL 40-char SHA (toy#60 item 5): the clone rule does a
# shallow `git fetch origin $(GGML_REV)`, and GitHub only serves
# fetch-by-SHA for FULL SHAs (allowReachableSHA1InWant — a short SHA
# gets "fatal: couldn't find remote ref", which broke every pristine
# clone during #45). Full-SHA shallow fetch verified cold against
# github.com/ggml-org/ggml on 2026-06-11.
GGML_REV    := 41e7949d705fd5dfeac33f3804e1af2a136cebd9
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
	@# REMOVE WHEN spinelgems#26 LANDS: deps.rb's generated $$LOAD_PATH
	@# prelude aborts every Spinel binary at boot ("undefined method
	@# 'unshift' for unknown" — $$LOAD_PATH is an untyped global at
	@# f6a189e8). Spinel loads deps via the topo require_relatives; the
	@# prelude only serves plain-CRuby runs, which toy's serve never does.
	sed -i '/^\$$LOAD_PATH\.unshift/d' vendor/spinel/deps.rb
	@# toy#69 — fold the vendored gems' aggregated sig root (advertised
	@# by vendor/spinel/deps.rb, spinelgems#13) into toy's own --rbs
	@# root via a gitignored symlink: spinel accepts ONE --rbs dir and
	@# spinel_rbs_extract follows symlinks. Removed when no gem ships
	@# sig (a dangling link would warn on every compile).
	@if [ -d vendor/spinel/sig ]; then \
	    ln -sfn ../vendor/spinel/sig sig/vendor; \
	    echo "  sig/vendor -> ../vendor/spinel/sig (rbs ride-along)"; \
	else \
	    rm -f sig/vendor; \
	fi

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
	@echo "    Curated examples (narrated; examples/README.md is the tour):"
	@echo "    make example_01                    train a tiny Llama from scratch (start here; ~2 s)"
	@echo "    make example_02                    warm-start fine-tune from a real GGUF's embeddings"
	@echo "    make example_03                    LoRA adapters over a frozen mmap'd base"
	@echo "    make example_04                    load a GGUF, KV decode, print text"
	@echo "    make example_05                    per-token logprobs (the eval building block)"
	@echo "    make example_06                    compare your runs/ (CRuby, no build)"
	@echo "    make example_07                    ViT-Tiny image classifier (same recipe shape)"
	@echo "    (Superseded tutorials live on in examples/legacy/ — they still build.)"
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
	@echo "  GATES — the reproducibility battery"
	@echo "    make gates-fast                    ALL $(words $(GATES)) legs, phased, safe under -j (the sign-off run)"
	@echo "    make gates                         same legs, strictly serial"
	@echo "    make gates-framework               $(words $(GATES_FRAMEWORK)) framework legs only — fast iteration, NOT a sweep"
	@echo "    make gates-research                $(words $(GATES_RESEARCH)) fixture-lane legs only — NOT a sweep"
	@echo "    make gate-<name>                   one leg (e.g. make gate-mlp)"
	@echo "    make runs-prune                    shrink runs/ back down (dry run; APPLY=1 to act)"
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
libexec/toy-infer: lib/toy/run/infer.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/io/tokenizer.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) \
		lib/toy.rb lib/toy/dev/toy_card.rb lib/toy/dev/toy_logprobs.rb lib/toy/io/gguf_kv.rb lib/toy/models/toy_smollm2.rb lib/toy/train/sampler.rb lib/toy/version.rb | libexec
	$(SPINEL) $< -o $@
toy-infer: libexec/toy-infer

# Diagnostic sibling of toy-infer: enables the cache trace and dumps per-tap
# min/max/|mean|/nan for every layer (used to localize ggml#1506 — the K-quant
# MoE attention head_nbytes collapse). See docs/notes/mul_mat_id_quants.md.
libexec/toy-infer-trace: lib/toy/run/infer_trace.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/io/tokenizer.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) \
		lib/toy.rb lib/toy/dev/toy_card.rb lib/toy/dev/toy_logprobs.rb lib/toy/models/toy_smollm2.rb lib/toy/train/sampler.rb lib/toy/version.rb | libexec
	$(SPINEL) $< -o $@
toy-infer-trace: libexec/toy-infer-trace

# P4 — `toy eval` COMPUTE runner (CRuby→runner COMPUTE BRIDGE, same shape as
# toy-infer). Spinel source lib/toy/run/eval.rb; the binary path EQUALS the
# make target so ToyRoot.ensure_built("libexec/toy-eval") both builds and
# locates it. Deps = infer's deps + lib/toy/dev/toy_logprobs.rb (a transitive require
# of transformer_lm; listed explicitly so a touch of it rebuilds the runner).
# CPU-only; NOT in MIRRORABLE (see prep/gen_cuda_mirror.rb).
libexec/toy-eval: lib/toy/run/eval.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/io/tokenizer.rb lib/toy/dev/toy_logprobs.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) \
		lib/toy.rb lib/toy/dev/toy_card.rb lib/toy/models/toy_smollm2.rb lib/toy/train/sampler.rb lib/toy/version.rb | libexec
	$(SPINEL) $< -o $@
toy-eval: libexec/toy-eval

# LMC (Linear Mode Connectivity) eval runner — `toy eval lmc --ckpt A --other B`.
# Interpolates two checkpoints θ_α = (1-α)·θ_A + α·θ_B and evals CE per α.
# Spinel source lib/toy/run/eval_lmc.rb; the binary path EQUALS the make target
# so ToyRoot.ensure_built("libexec/toy-eval-lmc") both builds and locates it.
# Deps mirror example_lmc (Makefile:479) NOT toy-eval; order-only | libexec (no
# $(SPINEL_DEPS)) like the CPU toy-eval runner. CPU-only; NOT in MIRRORABLE (see
# prep/gen_cuda_mirror.rb); a cuda LMC twin is a later slice.
libexec/toy-eval-lmc: lib/toy/run/eval_lmc.rb lib/toy/llm/adamw.rb vendor/spinel/spinel_kit/lib/spinel_kit/json_builder.rb lib/toy/io/toy_events.rb vendor/spinel/spinel_kit/lib/spinel_kit/git.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy.rb lib/toy/models/transformer.rb lib/toy/train/toy_drift_grad.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) \
		lib/toy/dev/toy_card.rb lib/toy/io/json_builder.rb lib/toy/llm/archs/layer_spec.rb lib/toy/llm/archs/llama_arch.rb lib/toy/llm/blocks/gdn_block.rb lib/toy/llm/blocks/kda_block.rb lib/toy/llm/blocks/mla_block.rb lib/toy/llm/blocks/transformer_block.rb lib/toy/llm/primitives/gdn.rb lib/toy/llm/primitives/gqa.rb lib/toy/llm/primitives/kda.rb lib/toy/llm/primitives/mla.rb lib/toy/llm/primitives/muon.rb lib/toy/llm/primitives/rms_norm.rb lib/toy/llm/primitives/rope.rb lib/toy/llm/primitives/situ_glu.rb lib/toy/llm/primitives/swiglu.rb lib/toy/train/dfa_b.rb lib/toy/version.rb | libexec
	$(SPINEL) $< -o $@
toy-eval-lmc: libexec/toy-eval-lmc

# toy#130 — CE-over-pack evaluator (the consuming half of the toy#129
# item-3 eval seam). Own unit (landmine #16); CPU-only like eval/lmc.
libexec/toy-eval-ce: lib/toy/run/eval_ce.rb lib/toy/llm/adamw.rb vendor/spinel/spinel_kit/lib/spinel_kit/json_builder.rb lib/toy/io/toy_events.rb lib/toy/io/toy_corpus_loader.rb vendor/spinel/spinel_kit/lib/spinel_kit/git.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy.rb lib/toy/models/transformer.rb lib/toy/train/toy_drift_grad.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) \
		lib/toy/dev/toy_card.rb lib/toy/io/json_builder.rb lib/toy/llm/archs/layer_spec.rb lib/toy/llm/archs/llama_arch.rb lib/toy/llm/blocks/gdn_block.rb lib/toy/llm/blocks/kda_block.rb lib/toy/llm/blocks/mla_block.rb lib/toy/llm/blocks/transformer_block.rb lib/toy/llm/primitives/gdn.rb lib/toy/llm/primitives/gqa.rb lib/toy/llm/primitives/kda.rb lib/toy/llm/primitives/mla.rb lib/toy/llm/primitives/muon.rb lib/toy/llm/primitives/rms_norm.rb lib/toy/llm/primitives/rope.rb lib/toy/llm/primitives/situ_glu.rb lib/toy/llm/primitives/swiglu.rb lib/toy/train/dfa_b.rb lib/toy/version.rb | libexec
	$(SPINEL) $< -o $@
.PHONY: toy-eval-ce
toy-eval-ce: libexec/toy-eval-ce
.PHONY: gate-eval-ce
gate-eval-ce: libexec/toy-eval-ce libexec/toy-train-franken-llama
	ruby prep/eval_ce_gate.rb

# toy#132 — the flag x recipe matrix, asserted against reality (pure
# CRuby CLI validation; no runner builds — every probe fails pre-build).
.PHONY: gate-train-cli-matrix
gate-train-cli-matrix:
	ruby prep/train_cli_matrix_gate.rb

# toy#137 (K2a) — the KDA recurrence + decay parameterization, proven by
# REDUCTION against the (kernel-verified) GDN scalar-decay path.
prep/smokes/smoke_kda_recurrence: prep/smokes/smoke_kda_recurrence.rb lib/toy.rb lib/toy/ffi/tinynn.rb \
		lib/toy/llm/primitives/rms_norm.rb lib/toy/llm/primitives/gdn.rb \
		lib/toy/llm/primitives/kda.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
# toy#139 — Muon's Newton-Schulz orthogonalization, proven by its
# defining properties (fixed point / band / scale invariance).
prep/smokes/smoke_muon_ns: prep/smokes/smoke_muon_ns.rb lib/toy.rb lib/toy/ffi/tinynn.rb \
		lib/toy/llm/primitives/muon.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
.PHONY: gate-muon
gate-muon: prep/smokes/smoke_muon_ns
	@out="$$(./prep/smokes/smoke_muon_ns 2>&1)"; \
	echo "$$out"; \
	echo "$$out" | grep -q "^muon-ns: ok$$" || { echo "GATE FAIL [muon]"; exit 1; }; \
	echo "GATE PASS [muon]: NS fixed-point + orthogonality band + scale invariance + non-trivial (toy#139)"

# K-series M2 — the Gated-MLA pieces, proven by factorization exactness,
# parameter economy, causality under future-token perturbation, and that
# the output gate actually gates.
prep/smokes/smoke_mla_latent: prep/smokes/smoke_mla_latent.rb lib/toy.rb lib/toy/ffi/tinynn.rb \
		lib/toy/llm/primitives/rms_norm.rb lib/toy/llm/primitives/gqa.rb \
		lib/toy/llm/primitives/gdn.rb lib/toy/llm/primitives/kda.rb \
		lib/toy/llm/primitives/mla.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
.PHONY: gate-mla
gate-mla: prep/smokes/smoke_mla_latent
	@out="$$(./prep/smokes/smoke_mla_latent 2>&1)"; \
	echo "$$out"; \
	echo "$$out" | grep -q "^mla-latent: ok$$" || { echo "GATE FAIL [mla]"; exit 1; }; \
	echo "GATE PASS [mla]: latent factorization byte-exact + param economy + causality + gate (K-series M2)"

.PHONY: gate-kda
gate-kda: prep/smokes/smoke_kda_recurrence
	@out="$$(./prep/smokes/smoke_kda_recurrence 2>&1)"; \
	echo "$$out"; \
	echo "$$out" | grep -q "^kda-recurrence: ok$$" || { echo "GATE FAIL [kda]"; exit 1; }; \
	echo "GATE PASS [kda]: reduction-to-GDN + channel-wise live + lower-bounded decay + 2-head striding (toy#137)"

# CUDA siblings of toy-infer / toy-eval — selected by the CRuby CLI shell when
# invoked with `--device cuda` (lib/toy/core/cli/{infer,eval}.rb derive the
# target). PER-DEVICE binaries (not one polymorphic runner): a single source
# requiring BOTH ToyLM and ToyLMCuda would force the CUDA archive onto the CPU
# binary's link line, changing it. Keeping separate binaries leaves
# libexec/toy-infer / toy-eval link lines BYTE-UNCHANGED. Source is the
# hand-written lib/toy/run/{infer,eval}_cuda.rb (ToyLMCuda ctor arity 1 →
# NOT mechanically mirrorable → ABSENT from MIRRORABLE, like the CPU runners).
# Force-link recipe matches every other cuda target (-Wl,-u,tnn_cuda_force_link).
libexec/toy-infer-cuda: lib/toy/run/infer_cuda.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm_cuda.rb lib/toy/llm/engine/llama_kv_engine_cuda.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn_cuda.rb lib/toy/io/tokenizer.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS) \
		lib/toy.rb lib/toy/dev/toy_card.rb lib/toy/ffi/tinynn.rb lib/toy/io/gguf_kv.rb lib/toy/models/toy_smollm2.rb lib/toy/train/sampler.rb lib/toy/version.rb | libexec
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
toy-infer-cuda: libexec/toy-infer-cuda

libexec/toy-eval-cuda: lib/toy/run/eval_cuda.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm_cuda.rb lib/toy/llm/engine/llama_kv_engine_cuda.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn_cuda.rb lib/toy/dev/toy_logprobs.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS) \
		lib/toy.rb lib/toy/dev/toy_card.rb lib/toy/ffi/tinynn.rb lib/toy/models/toy_smollm2.rb lib/toy/train/sampler.rb lib/toy/version.rb | libexec
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
libexec/toy-infer-metal: lib/toy/run/infer_metal.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm_metal.rb lib/toy/llm/engine/llama_kv_engine_metal.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn_metal.rb lib/toy/io/tokenizer.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_metal.a $(SPINEL_DEPS) \
		lib/toy.rb lib/toy/dev/toy_card.rb lib/toy/ffi/tinynn.rb lib/toy/io/gguf_kv.rb lib/toy/models/toy_smollm2.rb lib/toy/train/sampler.rb lib/toy/version.rb | libexec
ifneq ($(UNAME_S),Darwin)
	@echo "toy-infer-metal: macOS-only"; exit 1
endif
	$(SPINEL) --cc='cc -Wl,-u,_tnn_metal_force_link -framework Foundation -framework Metal -framework MetalKit' $< -o $@
toy-infer-metal: libexec/toy-infer-metal

libexec/toy-eval-metal: lib/toy/run/eval_metal.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm_metal.rb lib/toy/llm/engine/llama_kv_engine_metal.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn_metal.rb lib/toy/dev/toy_logprobs.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_metal.a $(SPINEL_DEPS) \
		lib/toy.rb lib/toy/dev/toy_card.rb lib/toy/ffi/tinynn.rb lib/toy/models/toy_smollm2.rb lib/toy/train/sampler.rb lib/toy/version.rb | libexec
ifneq ($(UNAME_S),Darwin)
	@echo "toy-eval-metal: macOS-only"; exit 1
endif
	$(SPINEL) --cc='cc -Wl,-u,_tnn_metal_force_link -framework Foundation -framework Metal -framework MetalKit' $< -o $@
toy-eval-metal: libexec/toy-eval-metal

# Convenience: run both functional gates on the pure CPU path (no parity arm).
# These are the byte-exact infer/eval baselines. Until this target existed the
# CPU eval gate only ran behind gate-cuda's TOY_GATE_CUDA=1, so a CPU-only eval
# regression could reach main unnoticed — and did once (the decode_step
# PolyArray OOB, #104/#105). Self-builds the runners via bin/toy.
.PHONY: gate-cpu
gate-cpu:
	ruby prep/infer_gate.rb
	ruby prep/eval_gate.rb

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
libexec/gpt2-train-min: prep/gpt2_train_min.rb lib/toy.rb lib/toy/ffi/tinynn.rb \
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

# toy#71 — the MRI dev-run gate, BOTH arms (plain `ruby`, NO Spinel
# build, NO SPINEL_DIR). Stub leg (Stage A): `require "toy/mri"` loads
# the full compute surface under CRuby, the pure-Ruby teaching path
# genuinely trains, and crossing the native boundary raises the NAMED
# Toy::MRI::NativeCallError. Native leg (Stage B, the CRuby oracle;
# needs `make libtinynn_shared`, loud SKIP otherwise — MRI_GATE_STRICT=1
# turns the skip into a failure): MRI+Fiddle reproduces the recorded
# Spinel from-scratch gate curve BIT-EXACT (train_baseline.txt) and the
# smollm2-135m greedy decode ids byte-equal infer_baseline.txt.
# Prereq on the shared .so so a NEW FFI symbol (e.g. the #1449
# tnn_input_1d_i32_persistent) can't leave a STALE .so behind that
# fails the native leg with a missing-symbol NativeCallError — make
# rebuilds it from the .o's automatically.
.PHONY: gate-mri
gate-mri: tinynn/libtinynn_ggml_shared.so
	ruby prep/mri_gate.rb

# toy#60 item 4 — the COLD-START consumer gate: `toy new` scaffold →
# hello.rb compiles + runs (default ENV, then D_MODEL override without
# recompiling) → `toy train` prints losses + writes runs/<id>/ → the
# missing-corpus guard fails loud; PLUS the `toy new --lib` leg
# (bundle lock → spinel-compat vendor → ./build.sh cpu → run; skips
# loudly when bundler/spinel-compat are absent). Structural, not
# byte-exact. ~4 min (the lib leg builds ggml inside the tmp project).
.PHONY: gate-consumer
gate-consumer:
	ruby prep/consumer_gate.rb

# toy#42 full-API require gate. Builds prep/smokes/smoke_compute_surface (which
# requires ONLY lib/toy/compute.rb) and asserts it realizes a live engine —
# proving the one-require compute surface co-compiles + works for a library
# consumer. Builds the smoke itself.
.PHONY: gate-compute-surface
gate-compute-surface: prep/smokes/smoke_compute_surface
	@out="$$(./prep/smokes/smoke_compute_surface 2>&1)"; \
	echo "$$out" | tail -2; \
	echo "$$out" | grep -q "compute-surface: ok" \
	  && echo "GATE PASS [compute-surface]: lib/toy/compute.rb one-require surface is live" \
	  || { echo "GATE FAIL [compute-surface]"; exit 1; }

# toy#64 item 8 — CUDA twin of gate-compute-surface: build + run the
# consumer-ish CUDA entry smoke on the GPU (GB10 sm_121).
.PHONY: gate-compute-surface-cuda
gate-compute-surface-cuda: prep/smokes/smoke_compute_surface_cuda
	@out="$$(./prep/smokes/smoke_compute_surface_cuda 2>&1)"; \
	echo "$$out" | tail -2; \
	echo "$$out" | grep -q "compute-surface-cuda: ok" \
	  && echo "GATE PASS [compute-surface-cuda]: lib/toy/compute_cuda.rb device entry is live" \
	  || { echo "GATE FAIL [compute-surface-cuda]"; exit 1; }

# Projection-lens gate: train through W_proj only (token_embd frozen) and
# assert the loss drops (the smoke's own "is learning" verdict). The CPU
# smoke was an ungated diagnostic; this wires it into the gate surface.
.PHONY: gate-projection-lens
gate-projection-lens: prep/smokes/smoke_projection_lens
	@out="$$(STEPS=20 ./prep/smokes/smoke_projection_lens 2>&1)"; \
	echo "$$out" | tail -2; \
	echo "$$out" | grep -q "projection-lens training is learning" \
	  && echo "GATE PASS [projection-lens]: W_proj-only training learns (token_embd frozen)" \
	  || { echo "GATE FAIL [projection-lens]"; exit 1; }

# Metal twin of the projection-lens gate. The _metal smoke is an auto-
# generated mirror (MIRROR_METAL) that previously built but was reachable
# from no gate; this de-orphans it. macOS-only, skips green off Darwin
# exactly like gate-metal.
.PHONY: gate-projection-lens-metal
gate-projection-lens-metal:
ifneq ($(UNAME_S),Darwin)
	@echo "gate-projection-lens-metal: Metal is macOS-only (uname -s = $(UNAME_S)) — skipping"; exit 0
else
	$(MAKE) prep/smokes/smoke_projection_lens_metal
	@out="$$(STEPS=20 ./prep/smokes/smoke_projection_lens_metal 2>&1)"; \
	echo "$$out" | tail -2; \
	echo "$$out" | grep -q "projection-lens training is learning" \
	  && echo "GATE PASS [projection-lens-metal]: W_proj-only training learns on Metal" \
	  || { echo "GATE FAIL [projection-lens-metal]"; exit 1; }
endif

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

# Qwen3-MoE (qwen3moe arch) inference gate — Phase 1 routing variants
# (arch-prefix load + separate expert_ff + norm_topk renorm). Model-gated
# (~18 GB Qwen3-30B-A3B Q4_K_M, gitignored); SKIPs loudly when absent.
.PHONY: gate-qwen3moe
gate-qwen3moe:
	ruby prep/qwen3moe_gate.rb

# Shared-expert MoE gate — Phase 2 (Qwen1.5-MoE / qwen2moe: routed top-k +
# always-on gated shared expert). Model-gated (~9 GB, gitignored); SKIPs loudly.
.PHONY: gate-qwen2moe-shexp
gate-qwen2moe-shexp:
	ruby prep/qwen2moe_shexp_gate.rb

# DeepSeek-V2 MLA-A (deepseek2 arch) inference gate — Multi-head Latent
# Attention: latent c_kv + shared decoupled-RoPE key, asymmetric K(192)/V(128)
# cache, YaRN-mscale softmax, per-layer dense/MoE dispatch. Model-gated
# (~9.7 GB DeepSeek-V2-Lite-Chat Q4_K_M, gitignored); SKIPs loudly when absent.
.PHONY: gate-deepseek-mla
gate-deepseek-mla:
	ruby prep/deepseek_mla_gate.rb

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
		lib/toy/llm/blocks/transformer_block.rb lib/toy/llm/primitives/gdn.rb lib/toy/llm/blocks/gdn_block.rb \
		lib/toy/llm/archs/layer_spec.rb lib/toy/llm/archs/llama_arch.rb \
		lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) \
		lib/toy/dev/toy_card.rb lib/toy/io/json_builder.rb lib/toy/llm/blocks/kda_block.rb lib/toy/llm/blocks/mla_block.rb lib/toy/llm/primitives/kda.rb lib/toy/llm/primitives/mla.rb lib/toy/llm/primitives/muon.rb lib/toy/llm/primitives/situ_glu.rb lib/toy/train/dfa_b.rb lib/toy/train/toy_gguf_fuse.rb lib/toy/version.rb | libexec
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
		lib/toy/llm/blocks/transformer_block.rb lib/toy/llm/primitives/gdn.rb lib/toy/llm/blocks/gdn_block.rb \
		lib/toy/llm/archs/layer_spec.rb lib/toy/llm/archs/llama_arch.rb \
		lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) \
		lib/toy/dev/toy_card.rb lib/toy/io/json_builder.rb lib/toy/llm/blocks/kda_block.rb lib/toy/llm/blocks/mla_block.rb lib/toy/llm/primitives/kda.rb lib/toy/llm/primitives/mla.rb lib/toy/llm/primitives/muon.rb lib/toy/llm/primitives/situ_glu.rb lib/toy/train/dfa_b.rb lib/toy/version.rb | libexec
	$(SPINEL) $< -o $@
toy-train-lora: libexec/toy-train-lora

# `toy train from-scratch --arch gpt2` DEDICATED runner. Separate binary from
# toy-train (landmine #16: the GPT-2 realize path can't share a Spinel unit with
# the llama random-init path). Self-contained GPT2SeqEngine (no llama engine /
# primitives dep), so it also can't churn the llama gates. CPU-only this slice.
libexec/toy-train-gpt2: lib/toy/run/train_gpt2.rb lib/toy.rb \
		lib/toy/llm/engine/gpt2_seq_engine.rb lib/toy/llm/labels.rb lib/toy/llm/adamw.rb \
		lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) \
		lib/toy/dev/toy_card.rb lib/toy/version.rb | libexec
	$(SPINEL) $< -o $@
toy-train-gpt2: libexec/toy-train-gpt2
.PHONY: toy-train-gpt2

# CUDA twin of toy-train-gpt2 (`--arch gpt2 --device cuda`). SEPARATE single-type
# binary (landmine #16): links the generated CUDA engine mirror + the CUDA TinyNN
# shim; the GELU/LayerNorm backward ops fall back to the CPU backend via the
# scheduler (no CUDA kernel). lib/toy/ffi/tinynn.rb + transformer.rb stay in deps (Mat /
# CPU-TinyNN seam). NOT in MIRRORABLE (the engine mirror IS; the runner is hand-written).
libexec/toy-train-gpt2-cuda: lib/toy/run/train_gpt2_cuda.rb lib/toy.rb \
		lib/toy/llm/engine/gpt2_seq_engine_cuda.rb lib/toy/models/transformer.rb \
		lib/toy/ffi/tinynn_cuda.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS) \
		lib/toy/dev/toy_card.rb lib/toy/version.rb | libexec
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
toy-train-gpt2-cuda: libexec/toy-train-gpt2-cuda
.PHONY: toy-train-gpt2-cuda

# Metal twin (`--arch gpt2 --device metal`), macOS ONLY. Same structure; links
# the generated Metal engine mirror + the Metal TinyNN shim + Apple frameworks.
# gx10 RUNTIME-UNVERIFIED (codegen + structural parity here; runtime-gate on Mac).
libexec/toy-train-gpt2-metal: lib/toy/run/train_gpt2_metal.rb lib/toy.rb \
		lib/toy/llm/engine/gpt2_seq_engine_metal.rb lib/toy/models/transformer.rb \
		lib/toy/ffi/tinynn_metal.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_metal.a $(SPINEL_DEPS) \
		lib/toy/dev/toy_card.rb lib/toy/version.rb | libexec
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
		lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) \
		lib/toy.rb lib/toy/dev/toy_card.rb lib/toy/io/json_builder.rb lib/toy/version.rb | libexec
	$(SPINEL) $< -o $@
toy-train-vit: libexec/toy-train-vit

# toy#152 (DFA-arch T0) — the MLP-classifier ANCHOR runner. SEPARATE
# binary (landmine #16), CPU-only by decision (tao#18: T0–T3 get no
# CUDA twins — the anchors are small and twin-drift bit us twice in
# toy#150/#151). Reads STEPS/SEED/MLP_*/TAO_RUN_DIR/TOY_RUN_ID from
# ENV; trains on a seeded synthetic task (no corpus, no fixture to
# pin). Shares Toy::Train::DfaB with the franken lanes — the feedback
# machinery is the SAME code, only the output dim differs, which is
# the whole point of the anchor.
libexec/toy-train-mlp: lib/toy/run/train_mlp.rb lib/toy/dev/toy_describe_flow.rb \
		lib/toy/io/json_builder.rb lib/toy/io/json.rb lib/toy/io/toy_events.rb \
		vendor/spinel/spinel_kit/lib/spinel_kit/git.rb \
		lib/toy/io/toy_mlp_task.rb lib/toy/llm/engine/mlp_engine.rb \
		lib/toy/llm/recipes/mlp_classifier.rb lib/toy/llm/adamw.rb \
		lib/toy/train/dfa_b.rb \
		lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) | libexec
	$(SPINEL) $< -o $@
toy-train-mlp: libexec/toy-train-mlp
.PHONY: toy-train-mlp

# toy#154 (DFA-arch T1) — the CTR tower: per-field embedding tables +
# MLP tower + SCALAR sigmoid head. SEPARATE binary (landmine #16),
# CPU-only (tao#18). The DFA form is the toy#158 surrogate-root one,
# NOT toy#152's direct-gradient one: the tower's injected error has to
# propagate INTO the embedding tables for them to train at all, which
# is what "DFA the tower, embeddings stay chain" means.
libexec/toy-train-ctr: lib/toy/run/train_ctr.rb lib/toy/dev/toy_describe_flow.rb \
		lib/toy/io/json_builder.rb lib/toy/io/json.rb lib/toy/io/toy_events.rb \
		vendor/spinel/spinel_kit/lib/spinel_kit/git.rb \
		lib/toy/io/toy_ctr_task.rb lib/toy/llm/engine/ctr_engine.rb \
		lib/toy/llm/recipes/ctr_tower.rb lib/toy/llm/adamw.rb \
		lib/toy/train/dfa_b.rb \
		lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) | libexec
	$(SPINEL) $< -o $@
toy-train-ctr: libexec/toy-train-ctr
.PHONY: toy-train-ctr

# toy#160 (DFA-arch T4) — the GRAPH TRANSFORMER. SEPARATE binary
# (landmine #16), CPU-only. Its job is to DISAMBIGUATE the program's one
# unresolved negative: every transformer-LM run was DFA at a ~50k-vocab
# output, so "attention is DFA-hostile" and "the output dim was too big"
# were never separated. This lane keeps the attention and shrinks the
# head to 16 relation classes.
libexec/toy-train-gtx: lib/toy/run/train_gtx.rb lib/toy/dev/toy_describe_flow.rb \
		lib/toy/io/json_builder.rb lib/toy/io/json.rb lib/toy/io/toy_events.rb \
		vendor/spinel/spinel_kit/lib/spinel_kit/git.rb \
		lib/toy/io/toy_gtx_task.rb lib/toy/llm/engine/gtx_engine.rb \
		lib/toy/llm/recipes/gtx_graph.rb lib/toy/llm/adamw.rb \
		lib/toy/train/dfa_b.rb \
		lib/toy/io/toy_ae_task.rb \
		lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) | libexec
	$(SPINEL) $< -o $@
toy-train-gtx: libexec/toy-train-gtx
.PHONY: toy-train-gtx

# tao#24 — the CUDA twin, and it is a twin of the BYTE-LM TASK ONLY.
#
# The CPU-only decision for this lane was correct for gtx's relational
# task (d_model 64, 16 relation classes, 1500 steps — a GPU buys
# nothing). toy#170/P3's `--task bytelm` then changed the workload by
# orders of magnitude: vocab up to 4096, ctx 128, 4000 steps, ~3.2
# TFLOP/cell measured at ~32 GFLOP/s with the GB10 at 0% and ~6h per
# ladder. The scope LAPSED; it was not wrong. The relational/local tasks
# and every other cross-architecture lane stay CPU-only.
#
# The runner source is GENERATED (lib/toy/run/train_gtx_cuda.rb, from
# lib/toy/run/train_gtx.rb via prep/gen_cuda_mirror.rb) rather than
# hand-copied, so the twin cannot drift from the lane it is supposed to
# agree with numerically — which is this ticket's actual deliverable.
# The generated runner refuses every task but bytelm.
#
# SEPARATE single-type binary (landmine #16), force-linking the CUDA ggml
# backend via -Wl,-u,tnn_cuda_force_link like every other cuda target.
# BOTH shims are in the unit: TinyNNCuda drives the graph, and the CPU
# TinyNN arrives with Mat (lib/toy/models/transformer.rb) and
# ToyDescribeFlow's introspection — same shape as libexec/toy-train-cuda.
libexec/toy-train-gtx-cuda: lib/toy/run/train_gtx_cuda.rb lib/toy/dev/toy_describe_flow.rb \
		lib/toy/io/json_builder.rb lib/toy/io/json.rb lib/toy/io/toy_events.rb \
		vendor/spinel/spinel_kit/lib/spinel_kit/git.rb \
		lib/toy/io/toy_gtx_task.rb lib/toy/llm/engine/gtx_engine_cuda.rb \
		lib/toy/llm/recipes/gtx_graph_cuda.rb lib/toy/llm/adamw.rb \
		lib/toy/train/dfa_b.rb \
		lib/toy/io/toy_ae_task.rb \
		lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb lib/toy/ffi/tinynn_cuda.rb \
		tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS) | libexec
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
toy-train-gtx-cuda: libexec/toy-train-gtx-cuda
.PHONY: toy-train-gtx-cuda

# toy#160 gate: chain byte-null, determinism, the B seed, the MANDATORY
# success bar with each arm at ITS OWN best LR, and the two controls
# that make the lane mean anything — frozen must LOSE (the mask
# aggregates for free, so the task is built to make averaging useless)
# and the head must stay SMALL (re-entering the 50k regime would answer
# the wrong question).
.PHONY: gate-gtx
gate-gtx: libexec/toy-train-gtx
	ruby prep/gtx_gate.rb

# tao#24 gate: the byte-LM CUDA twin. CROSS-BACKEND NUMERIC AGREEMENT is
# the deliverable here, not the speedup — a twin that is 5x faster and
# quietly computes something else would put a wrong number into the
# E-series ladder wearing a plausible face. Asserts the run really used
# cuda (from run_start provenance, since ggml falls back silently), a
# BYTE-IDENTICAL graph on both backends, agreement at two stated
# tolerances (arithmetic at lr=0; trained at each arm's own LR), CUDA
# self-determinism, the bytelm-ONLY scope at both the binary and the CLI,
# and that the CPU lane is byte-identical to the pre-twin binary.
# Named *-cuda so the battery puts it in the serialised GPU phase and
# macOS filters it out.
.PHONY: gate-gtx-cuda
gate-gtx-cuda: libexec/toy-train-gtx libexec/toy-train-gtx-cuda
	ruby prep/gtx_cuda_gate.rb

# toy#171 — the prerequisite lists are hand-maintained and duplicate the
# runners' require_relative graphs, so they drift. Silently: a change to
# an undeclared file rebuilds NOTHING, `make` says "up to date", and the
# lane's own gate then passes against a stale binary. Found twice in two
# days on toy#170 (gtx and ssm, both missing toy_ae_task.rb), then across
# 26 of 36 targets — including toy-train-franken missing dfa_b.rb, i.e.
# the DFA program's dense lane not rebuilding when the feedback matrix
# changes. Needs no runner, so it costs nothing and runs first.
.PHONY: gate-prereq
gate-prereq:
	ruby prep/prereq_audit.rb

# toy#165 (capstone P1a) — the PER-TOKEN LATENT AUTOENCODER. All BP: no
# DFA and no diffusion in this lane. It asks whether a 4-8-dim per-token
# latent can carry a text token DECODABLY UNDER NOISE, which is the
# make-or-break the diffusion-text-LM capstone rests on. SEPARATE binary
# (landmine #16), CPU-only.
libexec/toy-train-ae: lib/toy/run/train_ae.rb lib/toy/dev/toy_describe_flow.rb \
		lib/toy/io/json_builder.rb lib/toy/io/json.rb lib/toy/io/toy_events.rb \
		vendor/spinel/spinel_kit/lib/spinel_kit/git.rb \
		lib/toy/io/toy_ae_task.rb lib/toy/llm/engine/ae_engine.rb \
		lib/toy/llm/recipes/ae_auto.rb lib/toy/llm/adamw.rb \
		lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) | libexec
	$(SPINEL) $< -o $@
toy-train-ae: libexec/toy-train-ae
.PHONY: toy-train-ae

# toy#166 (capstone P1b) — the LATENT-DIFFUSION BYTE-LM. Three models in
# ONE run (autoencoder, denoiser/AR arm, judge) but built SEQUENTIALLY:
# tnn_session_new resets the backend's SHARED scheduler, so two live
# sessions silently corrupt each other (prep/smokes/smoke_two_sessions.rb).
# SEPARATE binary (landmine #16), CPU-only, all BP.
libexec/toy-train-difflm: lib/toy/run/train_difflm.rb lib/toy/dev/toy_describe_flow.rb \
		lib/toy/io/json_builder.rb lib/toy/io/json.rb lib/toy/io/toy_events.rb \
		vendor/spinel/spinel_kit/lib/spinel_kit/git.rb \
		lib/toy/io/toy_ae_task.rb lib/toy/llm/engine/ae_engine.rb \
		lib/toy/llm/engine/ar_engine.rb lib/toy/llm/engine/difflm_engine.rb \
		lib/toy/llm/recipes/ae_auto.rb lib/toy/llm/adamw.rb \
		lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) | libexec
	$(SPINEL) $< -o $@
toy-train-difflm: libexec/toy-train-difflm
.PHONY: toy-train-difflm

.PHONY: gate-difflm
gate-difflm: libexec/toy-train-difflm
	ruby prep/difflm_gate.rb

# toy#166: the two-session probe. Its answer decides the difflm runner's
# whole shape, so it is a gate rather than a one-off.
prep/smokes/smoke_two_sessions: prep/smokes/smoke_two_sessions.rb \
		lib/toy/llm/engine/ae_engine.rb lib/toy/llm/adamw.rb \
		lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# toy#165 gate: determinism, the mandatory CONTROL-CAN-LOSE precondition
# (a SHUFFLED latent must fall to the unigram floor; the zeroed one is
# reported but is an identity, not an assertion), the corpus/alphabet
# provenance the margin curve is scoped by, and the probe's byte-null
# property — identity/ones/zeros must reproduce clean reconstruction
# exactly, or the measurement graph is not the training graph.
.PHONY: gate-ae
gate-ae: libexec/toy-train-ae
	ruby prep/ae_gate.rb

# toy#157 (DFA-arch T3) — the LSTM lane, the SSM rehearsal. SEPARATE
# binary (landmine #16), CPU-only. Reuses toy#155's delayed-cue task
# generator UNCHANGED so the two lanes are an architecture comparison
# rather than two anecdotes.
libexec/toy-train-lstm: lib/toy/run/train_lstm.rb lib/toy/dev/toy_describe_flow.rb \
		lib/toy/io/json_builder.rb lib/toy/io/json.rb lib/toy/io/toy_events.rb \
		vendor/spinel/spinel_kit/lib/spinel_kit/git.rb \
		lib/toy/io/toy_ssm_task.rb lib/toy/llm/engine/lstm_engine.rb \
		lib/toy/llm/recipes/lstm_seq.rb lib/toy/llm/adamw.rb \
		lib/toy/train/dfa_b.rb lib/toy/train/stream_bytes.rb \
		lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) | libexec
	$(SPINEL) $< -o $@
toy-train-lstm: libexec/toy-train-lstm
.PHONY: toy-train-lstm

# toy#157 gate: chain byte-null, forward-identity under detach, the B
# seed, the MANDATORY success bar, and the MEMORY measurement in BYTES
# that the ticket's success target is stated in.
#
# It also depends on the SSM runner, and that is the point of the lane:
# one leg runs the IDENTICAL per-step cut on toy#155's selective scan and
# asserts it collapses there while holding here. A cross-architecture
# claim that only ever runs one architecture is not a measurement.
.PHONY: gate-lstm
gate-lstm: libexec/toy-train-lstm libexec/toy-train-ssm
	ruby prep/lstm_gate.rb

# toy#156 (DFA-arch T2) — the latent diffusion denoiser. SEPARATE binary
# (landmine #16), CPU-only. Scoped strictly to a LOW-DIM latent: the
# eps-prediction target has the input's dimensionality, so pixel
# diffusion would defeat DFA the way vocab 50257 defeats the LM lanes.
libexec/toy-train-diff: lib/toy/run/train_diff.rb lib/toy/dev/toy_describe_flow.rb \
		lib/toy/io/json_builder.rb lib/toy/io/json.rb lib/toy/io/toy_events.rb \
		vendor/spinel/spinel_kit/lib/spinel_kit/git.rb \
		lib/toy/io/toy_diff_task.rb lib/toy/llm/engine/diff_engine.rb \
		lib/toy/llm/recipes/diff_denoiser.rb lib/toy/llm/adamw.rb \
		lib/toy/train/dfa_b.rb \
		lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) | libexec
	$(SPINEL) $< -o $@
toy-train-diff: libexec/toy-train-diff
.PHONY: toy-train-diff

# toy#156 gate: chain byte-null, determinism, the align phase, and the
# MANDATORY success bar stated on a GENERATIVE metric (energy distance,
# where LOWER is better — the bar inverts).
.PHONY: gate-diff
gate-diff: libexec/toy-train-diff
	ruby prep/diff_gate.rb

# toy#155 (DFA-arch T2) — the selective-scan / Mamba-lite lane. The
# recurrence is UNROLLED over T from differentiable primitives, because
# ggml's fused SSM_SCAN/SSM_CONV have no backward (ggml_compute_backward
# covers 43 ops and neither is among them) — and because the BPTT graph
# the DFA arm claims to avoid has to exist before avoiding it means
# anything. SEPARATE binary (landmine #16), CPU-only this slice: tao#19
# defers this lane's CUDA twin to the long-sequence memory measurement.
libexec/toy-train-ssm: lib/toy/run/train_ssm.rb lib/toy/dev/toy_describe_flow.rb \
		lib/toy/io/json_builder.rb lib/toy/io/json.rb lib/toy/io/toy_events.rb \
		vendor/spinel/spinel_kit/lib/spinel_kit/git.rb \
		lib/toy/io/toy_ssm_task.rb lib/toy/llm/engine/ssm_engine.rb \
		lib/toy/llm/recipes/ssm_seq.rb lib/toy/llm/adamw.rb \
		lib/toy/io/toy_ae_task.rb \
		lib/toy/train/dfa_b.rb lib/toy/train/stream_bytes.rb \
		lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) | libexec
	$(SPINEL) $< -o $@
toy-train-ssm: libexec/toy-train-ssm
.PHONY: toy-train-ssm

# toy#155 gate: chain byte-null, determinism, the per-step DFA cut
# (proved by the B seed moving the curve), the LTI control the ticket
# demands, and the MANDATORY success bar.
.PHONY: gate-ssm
gate-ssm: libexec/toy-train-ssm
	ruby prep/ssm_gate.rb

# toy#153 (DFA-arch T1) — the GNN node-classification lane: message
# passing over a symmetric-normalised adjacency, per-layer policy, and
# `--dfa-feedback structure` (DFA-GNN's error-along-the-graph feedback).
# SEPARATE binary (landmine #16), CPU-only (tao#18). Reads its graph
# either from a seed (contextual SBM + random GNN teacher) or from a
# bundle prep/fetch_cora.rb writes — no committed fixture either way.
libexec/toy-train-gnn: lib/toy/run/train_gnn.rb lib/toy/dev/toy_describe_flow.rb \
		lib/toy/io/json_builder.rb lib/toy/io/json.rb lib/toy/io/toy_events.rb \
		vendor/spinel/spinel_kit/lib/spinel_kit/git.rb \
		lib/toy/io/toy_gnn_task.rb lib/toy/llm/engine/gnn_engine.rb \
		lib/toy/llm/recipes/gnn_node.rb lib/toy/llm/adamw.rb \
		lib/toy/train/dfa_b.rb \
		lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) | libexec
	$(SPINEL) $< -o $@
toy-train-gnn: libexec/toy-train-gnn
.PHONY: toy-train-gnn

# toy#153 gate: chain byte-null, determinism, the masked-CE/semi-
# supervised wiring, structure-aware feedback as a REAL arm (it must
# move the curve AND reach the unlabelled nodes), and the MANDATORY
# success bar (all-DFA within gap of all-BP AND beating the frozen
# control, at matched init + seed).
# The Cora bundle is a REAL prerequisite, not an optional extra: the
# lane's mandatory success bar is stated on Cora (the seeded graph
# provably cannot carry it — see prep/gnn_gate.rb leg 8), so a gate that
# skipped it when the data were absent would not be a mandatory bar.
# One 168 KB download, then cached in data/cora.tgz.
data/gnn_cora.meta.i32:
	ruby prep/fetch_cora.rb

.PHONY: gate-gnn
gate-gnn: libexec/toy-train-gnn data/gnn_cora.meta.i32
	ruby prep/gnn_gate.rb

# toy#154 gate: the scalar-output CTR lane — logloss identity, chain
# byte-null, determinism, embeddings-train-under-DFA, and the MANDATORY
# success bar in AUC (~0.01 of BP, and beating the frozen control).
.PHONY: gate-ctr
gate-ctr: libexec/toy-train-ctr
	ruby prep/ctr_gate.rb

# toy#152 gate: chain byte-null vs absent policy, determinism, the dfa
# effect, align-event shape (WITH wname), and the MANDATORY success bar
# (all-DFA within gap of all-BP AND beating the frozen control, at
# matched init + seed).
.PHONY: gate-mlp
gate-mlp: libexec/toy-train-mlp
	ruby prep/mlp_gate.rb

# P4/GPU — from-scratch CUDA TRAINING runner. CUDA twin of libexec/toy-train,
# from-scratch ONLY (warm_start dropped). SINGLE-TYPE binary (landmine #16):
# TinyNNCuda is the compute path; lib/toy/ffi/tinynn.rb + lib/toy/models/transformer.rb stay in
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
		lib/toy/llm/primitives/situ_glu_cuda.rb \
		lib/toy/llm/blocks/transformer_block_cuda.rb lib/toy/llm/archs/llama_arch_cuda.rb \
		lib/toy/ffi/tinynn_cuda.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS) \
		lib/toy/dev/toy_card.rb lib/toy/io/json_builder.rb lib/toy/llm/archs/layer_spec.rb lib/toy/llm/blocks/gdn_block.rb lib/toy/llm/blocks/kda_block.rb lib/toy/llm/blocks/mla_block.rb lib/toy/llm/primitives/gdn.rb lib/toy/llm/primitives/kda.rb lib/toy/llm/primitives/mla.rb lib/toy/llm/primitives/muon.rb lib/toy/llm/primitives/situ_glu_cuda.rb lib/toy/train/dfa_b.rb lib/toy/version.rb | libexec
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
toy-train-cuda: libexec/toy-train-cuda

# P4/GPU — LoRA CUDA TRAINING runner. CUDA twin of libexec/toy-train-lora.
# SEPARATE binary from libexec/toy-train-cuda: the LoRA realize_for_mmap path
# cannot share a Spinel compilation unit with the random-init path (cfg
# type-merge miscompile; landmine #16 — same reason toy-train-lora is split
# from toy-train). SINGLE-TYPE binary: TinyNNCuda is the compute path;
# lib/toy/ffi/tinynn.rb + lib/toy/models/transformer.rb stay in deps because transformer.rb
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
		lib/toy/llm/primitives/situ_glu_cuda.rb \
		lib/toy/llm/blocks/transformer_block_cuda.rb lib/toy/llm/archs/llama_arch_cuda.rb \
		lib/toy/ffi/tinynn_cuda.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS) \
		lib/toy/dev/toy_card.rb lib/toy/io/json_builder.rb lib/toy/llm/archs/layer_spec.rb lib/toy/llm/blocks/gdn_block.rb lib/toy/llm/blocks/kda_block.rb lib/toy/llm/blocks/mla_block.rb lib/toy/llm/primitives/gdn.rb lib/toy/llm/primitives/kda.rb lib/toy/llm/primitives/mla.rb lib/toy/llm/primitives/muon.rb lib/toy/llm/primitives/situ_glu_cuda.rb lib/toy/train/dfa_b.rb lib/toy/version.rb | libexec
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
toy-train-lora-cuda: libexec/toy-train-lora-cuda

# P4/GPU — from-scratch METAL TRAINING runner (macOS ONLY). Metal twin of
# libexec/toy-train-cuda, from-scratch ONLY. SINGLE-TYPE binary (landmine #16):
# TinyNNMetal is the compute path; lib/toy/ffi/tinynn.rb + lib/toy/models/transformer.rb stay in
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
		lib/toy/ffi/tinynn_metal.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_metal.a $(SPINEL_DEPS) \
		lib/toy/dev/toy_card.rb lib/toy/io/json_builder.rb lib/toy/llm/archs/layer_spec.rb lib/toy/llm/blocks/gdn_block.rb lib/toy/llm/blocks/kda_block.rb lib/toy/llm/blocks/mla_block.rb lib/toy/llm/primitives/gdn.rb lib/toy/llm/primitives/kda.rb lib/toy/llm/primitives/mla.rb lib/toy/llm/primitives/muon.rb lib/toy/llm/primitives/situ_glu_metal.rb lib/toy/train/dfa_b.rb lib/toy/version.rb | libexec
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
# handler; JSON via the absorbed Toy::Json codec, toy#107). vendor/spinel/tep/lib/tep.rb is the TEP BUILD-DEP
# edge — Tep is consumed purely as transport (built by `make vendor-tep`
# on a fresh tree; needs ../tep + ../spinelgems siblings). Deps mirror the
# tep_demo recipe (Makefile:486) + the KV stack. CPU-only; NOT in
# MIRRORABLE (see prep/gen_cuda_mirror.rb).
libexec/toy-serve: lib/toy/run/serve.rb lib/toy/io/json_builder.rb lib/toy/io/json.rb lib/toy/io/json_decoder.rb lib/toy/io/toy_events.rb vendor/spinel/spinel_kit/lib/spinel_kit/git.rb \
		lib/toy/serve/openai/server.rb \
		lib/toy/serve/openai/handlers.rb lib/toy/serve/openai/toy_backend.rb \
		vendor/spinel/tep/lib/tep.rb \
		lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb \
		tinynn/libtinynn_ggml.a $(SPINEL_DEPS) \
		lib/toy.rb lib/toy/dev/toy_card.rb lib/toy/ffi/tinynn.rb lib/toy/io/gguf_load.rb lib/toy/models/gpt2.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/version.rb | libexec
	$(SPINEL) $< -o $@
toy-serve: libexec/toy-serve

# toy#gguf-checkpoint-reload (#153) — smoke binary that loads a
# from-scratch toy GGUF and runs a tiny generation. No tokenizer.
prep/smokes/smoke_toy_ckpt_reload: prep/smokes/smoke_toy_ckpt_reload.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/io/tokenizer.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# toy#embed-api (#145) — smoke for ToyLM#embed_lookup.
prep/smokes/smoke_embed_api: prep/smokes/smoke_embed_api.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/io/tokenizer.rb lib/toy/dev/toy_logprobs.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# P1 framework refactor — runtime Card derivation smoke. Loads a
# llama-family GGUF, realizes the seq-mode cache, derives a
# structural Toy::Card via ToyDescribeFlow.card, prints + gates.
prep/smokes/smoke_card_derive: prep/smokes/smoke_card_derive.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/dev/toy_describe_flow.rb lib/toy/train/toy_drift_grad.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/dev/toy_card.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# toy#decode-logprobs (#151) — smoke for ToyLM#decode_step_with_logprobs.
prep/smokes/smoke_decode_logprobs: prep/smokes/smoke_decode_logprobs.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/io/tokenizer.rb lib/toy/dev/toy_logprobs.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# GH#18 — LMC interpolate-and-eval runner.
examples/example_lmc: examples/legacy/08_lmc.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb lib/toy/train/toy_drift_grad.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
example_lmc: examples/example_lmc

# E2.3 (towards GH#14) — projection-lens smoke.
prep/smokes/smoke_projection_lens: prep/smokes/smoke_projection_lens.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb lib/toy/train/toy_drift_grad.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# toy#42 full-API require gate. Compiling this proves lib/toy/compute.rb's whole
# surface (all three engines + recipes + loaders) co-compiles in one program;
# running it realizes a LlamaSeqEngine to prove the surface is live. The prereq
# is just lib/toy/compute.rb — it pulls everything else transitively, and
# $(SPINEL) follows the require graph.
prep/smokes/smoke_compute_surface: prep/smokes/smoke_compute_surface.rb lib/toy/compute.rb lib/toy/llm/training_batch.rb lib/toy/llm/recipe_options.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# Dragon/GDN Phase 1 (docs/roadmap/dragon-gdn-arch-2026-06-20.md): prove the
# newly-wired tnn_gated_delta_net + tnn_conv_1d FFI ops compute through toy's
# stack on the in-tree ggml. Forward-only shape gate (the recurrence runs and
# emits the documented output shape).
.PHONY: gate-gdn-forward
gate-gdn-forward: prep/smokes/smoke_gdn_forward
	@out="$$(./prep/smokes/smoke_gdn_forward 2>&1)"; \
	echo "$$out" | tail -2; \
	echo "$$out" | grep -q "GDN smoke PASS" \
	  && echo "GATE PASS [gdn-forward]: tnn_gated_delta_net computes through the FFI" \
	  || { echo "GATE FAIL [gdn-forward]"; exit 1; }

prep/smokes/smoke_gdn_forward: prep/smokes/smoke_gdn_forward.rb lib/toy.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# Dragon/GDN Phase 2: the Toy::LLM::Primitives::GDN L1 composition (l2-norm,
# log-decay + sigmoid gates, recurrence, gated output norm). The gate+l2+recur
# chain is computed end-to-end; gated_out is shape-checked.
.PHONY: gate-gdn-primitive
gate-gdn-primitive: prep/smokes/smoke_gdn_primitive
	@out="$$(./prep/smokes/smoke_gdn_primitive 2>&1)"; \
	echo "$$out" | tail -2; \
	echo "$$out" | grep -q "GDN primitive smoke PASS" \
	  && echo "GATE PASS [gdn-primitive]: Toy::LLM::Primitives::GDN composes + computes" \
	  || { echo "GATE FAIL [gdn-primitive]"; exit 1; }

prep/smokes/smoke_gdn_primitive: prep/smokes/smoke_gdn_primitive.rb lib/toy.rb lib/toy/ffi/tinynn.rb lib/toy/llm/primitives/gdn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# Dragon/GDN Phase 2: the Dragon attention-side L1 primitives (DiffAttention,
# ScalableSoftmax, DepthScale).
.PHONY: gate-dragon-attn-prims
gate-dragon-attn-prims: prep/smokes/smoke_dragon_attn_prims
	@out="$$(./prep/smokes/smoke_dragon_attn_prims 2>&1)"; \
	echo "$$out" | tail -2; \
	echo "$$out" | grep -q "Dragon attn prims smoke PASS" \
	  && echo "GATE PASS [dragon-attn-prims]: diff-attn / ssmax / depth-scale compose" \
	  || { echo "GATE FAIL [dragon-attn-prims]"; exit 1; }

prep/smokes/smoke_dragon_attn_prims: prep/smokes/smoke_dragon_attn_prims.rb lib/toy.rb lib/toy/ffi/tinynn.rb lib/toy/llm/primitives/diff_attention.rb lib/toy/llm/primitives/scalable_softmax.rb lib/toy/llm/primitives/depth_scale.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# Dragon/GDN Phase 4 (Path B): numeric-parity gate — the UNROLLED,
# autograd-differentiable recurrence (GDN.recur_unrolled) reproduces the FUSED
# tnn_gated_delta_net token outputs within eps. This is what lets training use
# the composition (every op has a ggml backward) while inference keeps the fused
# kernel. See docs/roadmap/dragon-gdn-arch-2026-06-20.md (Phase 4).
.PHONY: gate-gdn-unrolled-parity
gate-gdn-unrolled-parity: prep/smokes/smoke_gdn_unrolled_parity
	@out="$$(./prep/smokes/smoke_gdn_unrolled_parity 2>&1)"; \
	echo "$$out" | tail -2; \
	echo "$$out" | grep -q "GDN unrolled-parity smoke PASS" \
	  && echo "GATE PASS [gdn-unrolled-parity]: recur_unrolled == fused kernel (eps)" \
	  || { echo "GATE FAIL [gdn-unrolled-parity]"; exit 1; }

prep/smokes/smoke_gdn_unrolled_parity: prep/smokes/smoke_gdn_unrolled_parity.rb lib/toy.rb lib/toy/ffi/tinynn.rb lib/toy/llm/primitives/gdn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# Dragon/GDN Phase 4 (Path B): the differentiability proof — ggml builds + runs
# a backward graph through recur_unrolled and yields finite non-zero dL/dq,k,v
# with NO hand-written fused-kernel backward. This is what makes GDN trainable.
.PHONY: gate-gdn-unrolled-backward
gate-gdn-unrolled-backward: prep/smokes/smoke_gdn_unrolled_backward
	@out="$$(./prep/smokes/smoke_gdn_unrolled_backward 2>&1)"; \
	echo "$$out" | tail -2; \
	echo "$$out" | grep -q "GDN unrolled-backward smoke PASS" \
	  && echo "GATE PASS [gdn-unrolled-backward]: recur_unrolled is differentiable" \
	  || { echo "GATE FAIL [gdn-unrolled-backward]"; exit 1; }

prep/smokes/smoke_gdn_unrolled_backward: prep/smokes/smoke_gdn_unrolled_backward.rb lib/toy.rb lib/toy/ffi/tinynn.rb lib/toy/llm/primitives/gdn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# Dragon/GDN Phase 5: multi-head parity — the per-head recur_unrolled looped over
# H heads + concat'd matches the fused kernel's head packing (strided slicing).
.PHONY: gate-gdn-unrolled-parity-mh
gate-gdn-unrolled-parity-mh: prep/smokes/smoke_gdn_unrolled_parity_mh
	@out="$$(./prep/smokes/smoke_gdn_unrolled_parity_mh 2>&1)"; \
	echo "$$out" | tail -2; \
	echo "$$out" | grep -q "GDN unrolled-parity-mh smoke PASS" \
	  && echo "GATE PASS [gdn-unrolled-parity-mh]: H-head recur_unrolled == fused kernel" \
	  || { echo "GATE FAIL [gdn-unrolled-parity-mh]"; exit 1; }

prep/smokes/smoke_gdn_unrolled_parity_mh: prep/smokes/smoke_gdn_unrolled_parity_mh.rb lib/toy.rb lib/toy/ffi/tinynn.rb lib/toy/llm/primitives/gdn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# Dragon/GDN Phase 5 capstone: a SELF-CONTAINED from-scratch HYBRID runner (one
# attention layer + one GDN layer, dispatched by the int-kind seam pattern) in
# its OWN compilation unit — CE loss decreases. Proves a heterogeneous
# attention+GDN stack trains from scratch. Separate unit so it can't corrupt the
# byte-exact llama engine (landmine #16). Reintegration into `toy train` waits on
# the union-pin Spinel codegen fix (master/spinelc).
.PHONY: gate-gdn-hybrid
gate-gdn-hybrid: libexec/toy-train-hybrid
	@out="$$(./libexec/toy-train-hybrid 2>&1)"; \
	echo "$$out" | tail -3; \
	echo "$$out" | grep -q "HYBRID train smoke PASS" \
	  && echo "GATE PASS [gdn-hybrid]: attention+GDN from-scratch hybrid trains" \
	  || { echo "GATE FAIL [gdn-hybrid]"; exit 1; }

libexec/toy-train-hybrid: lib/toy/run/train_hybrid.rb lib/toy.rb lib/toy/ffi/tinynn.rb \
		lib/toy/llm/primitives/rms_norm.rb lib/toy/llm/primitives/gdn.rb \
		lib/toy/llm/blocks/gdn_block.rb lib/toy/llm/archs/layer_spec.rb \
		tinynn/libtinynn_ggml.a $(SPINEL_DEPS) \
		lib/toy/dev/toy_card.rb lib/toy/models/transformer.rb lib/toy/version.rb | libexec
	$(SPINEL) $< -o $@
.PHONY: toy-train-hybrid
toy-train-hybrid: libexec/toy-train-hybrid

# Dragon/GDN Phase 5 (end-of-flow): a from-scratch model whose mixer is a
# trainable GDNBlock trains — CE loss decreases. Proves the GDN layer is an
# end-to-end trainable residual unit (no hand-written kernel backward).
.PHONY: gate-gdn-train
gate-gdn-train: prep/smokes/smoke_gdn_train
	@out="$$(./prep/smokes/smoke_gdn_train 2>&1)"; \
	echo "$$out" | tail -3; \
	echo "$$out" | grep -q "GDN train smoke PASS" \
	  && echo "GATE PASS [gdn-train]: from-scratch GDN-layer model trains (loss decreases)" \
	  || { echo "GATE FAIL [gdn-train]"; exit 1; }

prep/smokes/smoke_gdn_train: prep/smokes/smoke_gdn_train.rb lib/toy.rb lib/toy/ffi/tinynn.rb lib/toy/llm/primitives/rms_norm.rb lib/toy/llm/primitives/gdn.rb lib/toy/llm/blocks/gdn_block.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# toy#64 item 8 — the CUDA compute entry (lib/toy/compute_cuda.rb), the
# consumer-ish device-at-compile-time gate. Same shape as the CPU
# compute-surface gate but requires compute_cuda + links the CUDA
# archives with the force-link flag. The generated CUDA mirrors in the
# dep list are kept fresh by the $(MIRROR_CUDA) pattern rules.
# The mirror list here is the FULL require_relative closure of
# lib/toy/compute_cuda.rb (the one-require surface), not a subset:
# situ_glu_cuda and lora_cuda were both missing, and in a tree that had
# generated them once the gap is INVISIBLE — `make` says up to date and
# only a FRESH checkout fails. Same class as toy#171's prereq audit and
# b3bb7c1's dropped ggml patch. Written out rather than $(MIRROR_CUDA)
# because prerequisite lists are expanded when the makefile is READ and
# MIRROR_CUDA is defined several hundred lines below this rule, so the
# variable would expand to nothing here and silently declare no mirrors
# at all — a fix that looks like a fix and restores the original bug.
prep/smokes/smoke_compute_surface_cuda: prep/smokes/smoke_compute_surface_cuda.rb lib/toy/compute_cuda.rb \
		lib/toy/llm/training_batch.rb lib/toy/llm/recipe_options.rb \
		lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/llm/engine/gpt2_seq_engine_cuda.rb \
		lib/toy/llm/engine/llama_kv_engine_cuda.rb \
		lib/toy/llm/recipes/from_scratch_cuda.rb lib/toy/llm/recipes/warm_start_cuda.rb \
		lib/toy/llm/recipes/lora_cuda.rb \
		lib/toy/llm/primitives/rms_norm_cuda.rb lib/toy/llm/primitives/rope_cuda.rb \
		lib/toy/llm/primitives/swiglu_cuda.rb lib/toy/llm/primitives/gqa_cuda.rb \
		lib/toy/llm/primitives/situ_glu_cuda.rb \
		lib/toy/llm/blocks/transformer_block_cuda.rb lib/toy/llm/archs/llama_arch_cuda.rb \
		lib/toy/ffi/tinynn_cuda.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS)
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# P2.6 — GQA-divergent (w_o) gate. Realizes a config with head_dim=24 so
# n_heads*head_dim (96) != d_model (64), proving the divergent w_o shape
# [d_model, n_heads*head_dim] allocates and runs forward+backward.
prep/smokes/smoke_gate_gqa_divergent: prep/smokes/smoke_gate_gqa_divergent.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb lib/toy/train/toy_drift_grad.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# P2.6 — llama3 RoPE post-rope TENSOR parity gate. Builds a standalone
# post-rope subgraph from the SAME public primitive (RoPE.apply_2d) the
# model's K/Q paths call, with a NON-NULL, NON-TRIVIAL llama3 freq_factors
# ptr (computed via Toy::RopeScaling.compute_llama3_freq_factors). Logit-
# level is rope-angle-INSENSITIVE, so the gate taps the post-rope tensor:
# asserts (a) freq_factors non-uniform / kind==:llama3, (b) post-rope output
# byte-identical run-to-run, plus a contrast guard vs :none (NULL factors).
# No model file, no lib/ change, no mirror regen. Run from repo root.
prep/smokes/smoke_gate_llama3_tensor: prep/smokes/smoke_gate_llama3_tensor.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# P2.6 — B>1 (micro-batch) gate. Realizes with t_batch=2 so @seq_b=2,
# forcing the block-causal mask alloc + upload (gated on @seq_b>1) and the
# soft_max_ext attention path (gqa.rb:50). Proves the batched graph
# allocates the [T*B,T*B] mask and runs forward+backward; records a
# reproducible loss baseline. MUST run from repo root (data/ts_seqs.txt).
prep/smokes/smoke_gate_b_gt_1: prep/smokes/smoke_gate_b_gt_1.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb lib/toy/train/toy_drift_grad.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# P2.6 — L4 FromScratch recipe gate. Drives the same random-init config
# as smoke_projection_lens THROUGH Toy::LLM::Recipes::FromScratch; its
# loss curve must byte-equal the projection-lens reference.
prep/smokes/smoke_recipe_from_scratch: prep/smokes/smoke_recipe_from_scratch.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb lib/toy/train/toy_drift_grad.rb lib/toy/llm/adamw.rb lib/toy/llm/labels.rb lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/from_scratch.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# BLESSED from-scratch path — the short tutorial. Same gate-fixture
# config as smoke_recipe_from_scratch, but the clean tutorial read using
# the value objects (Toy::SmolLM2Config.mha + Toy::Labels + Toy::AdamW).
examples/example_train_from_scratch_blessed: examples/legacy/train_from_scratch.rb lib/toy.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb lib/toy/llm/adamw.rb lib/toy/llm/labels.rb lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/from_scratch.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
example_train_from_scratch_blessed: examples/example_train_from_scratch_blessed

# ── Curated examples (toy#60) — the narrated teaching set. One file,
# one make target, one binary each; see examples/README.md for the tour.
# 01 — from-scratch on the bundled tiny corpus via the one-require
# compute surface + the named value objects. THE showcase; the example
# in docs/framework.md must stay truthful to this file.
examples/example_01_train_tiny: examples/01_train_tiny.rb lib/toy/compute.rb lib/toy/io/toy_corpus_loader.rb lib/toy/io/run_bundle.rb lib/toy/llm/training_batch.rb lib/toy/llm/recipe_options.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
example_01: examples/example_01_train_tiny
.PHONY: example_01

# 02 — warm-start fine-tune: donor token_embd from a real GGUF through
# Toy::LLM::Recipes::WarmStart (realize_scratch! → realize_warm! → build!).
examples/example_02_finetune_warm_start: examples/02_finetune_warm_start.rb lib/toy/compute.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/io/toy_corpus_loader.rb lib/toy/train/toy_lr_schedule.rb lib/toy/llm/adamw.rb lib/toy/llm/labels.rb lib/toy/llm/training_batch.rb lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/warm_start.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
example_02: examples/example_02_finetune_warm_start
.PHONY: example_02

# 03 — LoRA adapters over a frozen mmap'd base GGUF, via the one-require
# compute surface (lora re-added to it by toy#52).
examples/example_03_lora: examples/03_lora.rb lib/toy/compute.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/llm/adamw.rb lib/toy/llm/labels.rb lib/toy/llm/training_batch.rb lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/lora.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
example_03: examples/example_03_lora
.PHONY: example_03

# 04 — load a GGUF, KV-cache decode, print text (the llama_kv_engine
# path the `toy infer` runner drives).
examples/example_04_generate: examples/04_generate.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/io/tokenizer.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
example_04: examples/example_04_generate
.PHONY: example_04

# 05 — per-token logprobs at a decode position (the `toy eval` compute).
examples/example_05_eval_logprobs: examples/05_eval_logprobs.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/io/tokenizer.rb lib/toy/dev/toy_logprobs.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
example_05: examples/example_05_eval_logprobs
.PHONY: example_05

# 06 — CRuby, NOT compiled: Toy::RunLog comparison table over runs/.
example_06:
	ruby examples/06_runlog_compare.rb
.PHONY: example_06

# 07 — ViT-Tiny on the committed data/vit_smoke corpus via Recipes::VitTiny.
examples/example_07_vit_tiny: examples/07_vit_tiny.rb lib/toy/compute.rb lib/toy/llm/engine/vit_tiny_engine.rb lib/toy/llm/recipes/vit_tiny.rb lib/toy/models/toy_vit.rb lib/toy/io/toy_image_loader.rb lib/toy/io/run_bundle.rb lib/toy/train/toy_lr_schedule.rb lib/toy/llm/adamw.rb lib/toy/llm/labels.rb lib/toy/llm/classify_batch.rb lib/toy/llm/recipe_options.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
example_07: examples/example_07_vit_tiny
.PHONY: example_07

examples/example_08_gdn_block: examples/08_gdn_block.rb lib/toy.rb lib/toy/ffi/tinynn.rb lib/toy/llm/primitives/rms_norm.rb lib/toy/llm/primitives/gdn.rb lib/toy/llm/blocks/gdn_block.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
example_08: examples/example_08_gdn_block
.PHONY: example_08

examples-curated: example_01 example_02 example_03 example_04 example_05 example_07 example_08
.PHONY: examples-curated

# L4 LoRA recipe gate. Drives the same LoRA fine-tune config as the
# frozen reference 03_finetune_lora THROUGH Toy::LLM::Recipes::LoRA; its
# loss curve must byte-equal the reference at the fixed config.
prep/smokes/smoke_recipe_lora: prep/smokes/smoke_recipe_lora.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/llm/adamw.rb lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/lora.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# spinel-dev#33 diagnostic — same deps as smoke_recipe_lora; dumps per-node
# value checksums after one LoRA step so master-vs-union can be diffed.
prep/smokes/lora_node_dump: prep/smokes/lora_node_dump.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/llm/adamw.rb lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/lora.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# L4 WarmStart recipe gate. Drives the same warm-start config as the
# frozen reference 09_warm_start_train (INIT=scratch) THROUGH
# Toy::LLM::Recipes::WarmStart; its loss curve must byte-equal 09's at
# the fixed config (SEED=0 STEPS=5). The fixture drives the cosine LR
# schedule + streaming corpus loader (deps below); the recipe stays thin.
prep/smokes/smoke_recipe_warm_start: prep/smokes/smoke_recipe_warm_start.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb lib/toy/train/toy_drift_grad.rb lib/toy/io/toy_corpus_loader.rb lib/toy/train/toy_lr_schedule.rb lib/toy/llm/adamw.rb lib/toy/llm/labels.rb lib/toy/llm/recipe_options.rb lib/toy/llm/recipes/warm_start.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# P2.6 gate — GGUF F32 mmap round-trip parity. Head-fuses a random_init
# model into the FUSED llama.cpp naming, writes a GGUF, reloads via
# realize_for_mmap, and asserts the reloaded forward is BIT-IDENTICAL to
# the in-memory forward. This is the behavioral gate for realize_for_mmap
# (previously only realize_for_random_init was gated). CPU-only: the GGUF
# WRITE half reads host data ptrs (tnn_gguf_w_add_tensor), which the CUDA
# writer doesn't implement — do NOT auto-mirror this to CUDA.
prep/smokes/smoke_gguf_roundtrip: prep/smokes/smoke_gguf_roundtrip.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb lib/toy/train/toy_gguf_fuse.rb lib/toy/train/toy_gguf_writer.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

prep/smokes/smoke_full_finetune: prep/smokes/smoke_full_finetune.rb lib/toy/llm/adamw.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# P2.6 gate — qkv_bias mmap branch. Loads the real Qwen2.5-0.5B native GGUF
# (which DOES carry blk.N.attn_{q,k,v}.bias) and realizes via
# realize_for_mmap with qkv_bias=TRUE, untied=FALSE (output.weight absent =>
# tied), forcing the bias mmap branch (llama_seq_engine.rb:635-661) and
# its transformer_block tnn_add consumer — neither hit by smoke_gguf_roundtrip
# (qkv_bias=FALSE). Records a deterministic finite-logit baseline. CPU-only;
# DATA DEPENDENCY: data/qwen25-0.5b-native.gguf (not self-contained). MUST run
# from repo root. Do NOT auto-mirror to CUDA.
prep/smokes/smoke_gate_qkv_bias: prep/smokes/smoke_gate_qkv_bias.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# P2.6 gate — Q8-stays-Q8 realize_for_q8_copy branch. Loads the existing
# Q8 GGUF, asserts blk.0 attn_q weight stays Q8_0 in memory (NOT dequant
# to F32), deterministic forward x2 byte-identical baseline. Pure-Ruby
# fixture (no toy_drift_grad dep; seq_blocks_ffi directly).
prep/smokes/smoke_gate_q8_preserve: prep/smokes/smoke_gate_q8_preserve.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# P2.6 CUDA gate — GPU mirror of the projection-lens smoke. Exercises
# realize_for_random_init + seq forward on the CUDA backend so the
# realize-path refactor can be parity-gated on GPU (CUDA self-consistency
# before/after; CUDA floats don't bit-equal CPU). Mirror auto-generated
# by prep/gen_cuda_mirror.rb. Same force-link recipe as the 06 CUDA entry.
prep/smokes/smoke_projection_lens_cuda: prep/smokes/smoke_projection_lens_cuda.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn_cuda.rb lib/toy/train/toy_drift_grad.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS)
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# E2.4 (towards GH#14) — streaming corpus loader + cosine LR smoke.
prep/smokes/smoke_corpus_loader: prep/smokes/smoke_corpus_loader.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb lib/toy/io/toy_corpus_loader.rb lib/toy/train/toy_lr_schedule.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# E2.5 (towards GH#14) — warm-start training driver.
examples/example_warm_start_train: examples/legacy/09_warm_start_train.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb lib/toy/train/toy_drift_grad.rb lib/toy/train/toy_gguf_writer.rb lib/toy/io/toy_corpus_loader.rb lib/toy/train/toy_lr_schedule.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
example_warm_start_train: examples/example_warm_start_train

# Auto-generated coverage matrix — ggml ops vs our FFI surface.
# Sources are vendor/ggml/include/ggml.h, tinynn/tinynn_ggml.c, and the
# two FFI binding files. See docs/coverage.md for the matrix.
coverage: docs/coverage.md
docs/coverage.md: prep/gen_coverage.rb vendor/ggml/include/ggml.h \
                  tinynn/tinynn_ggml.c lib/toy/ffi/tinynn.rb lib/toy/ffi/tinynn_cuda.rb \
                  lib/toy/ffi/tinynn_metal.rb
	ruby prep/gen_coverage.rb
coverage-check:
	ruby prep/gen_coverage.rb --check
.PHONY: coverage coverage-check

examples/example_train: examples/legacy/02_train_custom_gpt.rb lib/toy/models/transformer.rb lib/toy/train/training.rb lib/toy/train/toy_trainer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@
example_train: examples/example_train

examples/example_finetune: examples/legacy/03_finetune_lora.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@
example_finetune: examples/example_finetune

# CUDA mirror — same source, swap TinyNN → TinyNNCuda by including
# both libs. The example source uses TinyNN; the CUDA build link-step
# carries CUDA symbols too (no source change). For real GPU speedup
# users typically write a `_cuda` variant; this mirror is for the
# build-recipe story.
examples/example_finetune_cuda: examples/legacy/03_finetune_lora_cuda.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
example_finetune_cuda: examples/example_finetune_cuda

# Metal mirror of example_inference (macOS only). Uses TinyNNMetal.
# Same -Wl,-u trick as CUDA so the Metal backend init survives
# weak-symbol resolution. macOS expects a leading underscore on
# external symbols, hence `-Wl,-u,_tnn_metal_force_link`.
# Frameworks (Foundation/Metal/MetalKit) are linked via -framework.
examples/example_inference_metal: examples/legacy/01_inference_metal.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm_metal.rb lib/toy/llm/engine/llama_kv_engine_metal.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn_metal.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_metal.a
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
examples/example_train_from_scratch_cpu: examples/legacy/06_train_from_scratch.rb vendor/spinel/spinel_kit/lib/spinel_kit/json_builder.rb lib/toy/io/toy_events.rb vendor/spinel/spinel_kit/lib/spinel_kit/git.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb lib/toy/dev/toy_describe_flow.rb lib/toy/train/toy_drift_grad.rb lib/toy/train/toy_gguf_writer.rb lib/toy/dev/toy_tap.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
examples/example_train_from_scratch_cuda: examples/legacy/06_train_from_scratch_cuda.rb vendor/spinel/spinel_kit/lib/spinel_kit/json_builder.rb lib/toy/io/toy_events.rb vendor/spinel/spinel_kit/lib/spinel_kit/git.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn_cuda.rb lib/toy/dev/toy_describe_flow.rb lib/toy/train/toy_drift_grad.rb lib/toy/train/toy_gguf_writer.rb lib/toy/dev/toy_tap.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS)
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
examples/example_train_from_scratch: examples/example_train_from_scratch_cpu
	@printf '#!/bin/sh\n# Auto-generated by Makefile. DEVICE selects the backend binary.\n# Edit examples/legacy/06_train_from_scratch.rb (cpu) for behaviour; CUDA mirror is auto-generated by prep/gen_cuda_mirror.rb.\ncase "$${DEVICE:-cpu}" in\n  cpu|"") exec "$$(dirname "$$0")/example_train_from_scratch_cpu" "$$@" ;;\n  cuda)   exec "$$(dirname "$$0")/example_train_from_scratch_cuda" "$$@" ;;\n  metal)  echo "DEVICE=metal not yet supported for training (inference only)" >&2; exit 2 ;;\n  *)      echo "DEVICE=$${DEVICE} not recognised (want cpu|cuda)" >&2; exit 2 ;;\nesac\n' > $@
	@chmod +x $@
example_train_from_scratch: examples/example_train_from_scratch
example_train_from_scratch_cuda: examples/example_train_from_scratch_cuda

# GPT-2 from-scratch via the GPT2SeqEngine library API (the curated GPT-2 demo;
# CLI surface is `toy train from-scratch --arch gpt2`). Memorizes a synthetic
# sequence so CE visibly collapses; exercises the vendored LayerNorm/GELU kernels.
examples/gpt2_train: examples/legacy/gpt2_train.rb lib/toy.rb \
		lib/toy/llm/engine/gpt2_seq_engine.rb lib/toy/models/transformer.rb \
		lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS) | libexec
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
# lib/toy/ffi/tinynn_cuda.rb / lib/toy/models/transformer_lm_cuda.rb are NOT captured.
MIRROR_CUDA := \
  lib/toy/llm/primitives/rms_norm_cuda.rb lib/toy/llm/primitives/rope_cuda.rb \
  lib/toy/llm/primitives/swiglu_cuda.rb lib/toy/llm/primitives/gqa_cuda.rb \
  lib/toy/llm/primitives/mla_cuda.rb \
  lib/toy/llm/primitives/situ_glu_cuda.rb lib/toy/llm/primitives/kda_cuda.rb \
  lib/toy/llm/primitives/muon_cuda.rb \
  lib/toy/llm/recipes/franken_from_scratch_cuda.rb \
  lib/toy/llm/blocks/transformer_block_cuda.rb lib/toy/llm/archs/llama_arch_cuda.rb \
  lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/llm/engine/gpt2_seq_engine_cuda.rb \
  lib/toy/llm/engine/gtx_engine_cuda.rb lib/toy/llm/recipes/gtx_graph_cuda.rb \
  lib/toy/run/train_gtx_cuda.rb \
  lib/toy/llm/recipes/from_scratch_cuda.rb lib/toy/llm/recipes/lora_cuda.rb \
  lib/toy/llm/recipes/warm_start_cuda.rb \
  lib/toy/llm/engine/llama_kv_engine_cuda.rb \
  lib/toy/llm/engine/gpt2_fwd_engine_cuda.rb lib/toy/llm/engine/gpt2_kv_engine_cuda.rb \
  examples/legacy/06_train_from_scratch_cuda.rb prep/smokes/smoke_projection_lens_cuda.rb
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
tep_demo/api: tep_demo/legacy/inference_api.rb vendor/spinel/tep/lib/tep.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
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
	vendor-patches/0009-sched-unsupported-node-diagnostic.patch \
	vendor-patches/0010-cuda-buffer_from_ptr-skip-init_tensor-padding-memset.patch \
	vendor-patches/0011-tensor-flag-detached.patch \
	vendor-patches/0012-cuda-out-prod-k1-sger-fallback.patch \
	vendor-patches/0013-tanh-sigmoid-backward.patch \
	vendor-patches/0014-hip-symbol-map-for-toy-patches.patch

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
# parity (and so lib/toy/ffi/tinynn.rb doesn't need ffi_lib "gomp").
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

# --- toy#71 Stage B: the CRuby-oracle shared library ------------------------
# tinynn objects + the static CPU ggml archives linked into ONE self-
# contained shared object that plain MRI dlopens via Fiddle (lib/toy/mri.rb
# native arm). PIC is already on everywhere (CFLAGS -fPIC for tinynn,
# -DCMAKE_POSITION_INDEPENDENT_CODE=ON for ggml). -Wl,-Bsymbolic binds
# ggml's intra-library references locally — without it the aarch64 link
# rejects adrp relocations against ggml's C++ vtables ("may bind
# externally"); no interposition is wanted anyway. --whole-archive keeps
# every backend-registry object alive. Link order mirrors the Spinel
# ffi_lib list in lib/toy/ffi/tinynn.rb (stdc++/pthread, -lm TRAILING; no
# gomp — the CPU ggml build is -DGGML_OPENMP=OFF). CPU ONLY this stage:
# the CUDA/Metal shims stay static-archive-only (follow-up — a
# libtinynn_ggml_cuda_shared.so would whole-archive build-cuda/ + the CUDA
# stub libs; Metal additionally needs a Mac to verify -dynamiclib +
# -force_load). Artifact is gitignored (rebuild: make libtinynn_shared).
.PHONY: libtinynn_shared
libtinynn_shared: tinynn/libtinynn_ggml_shared.so

tinynn/libtinynn_ggml_shared.so: tinynn/tinynn_ggml.o tinynn/tinynn_gguf.o tinynn/tinynn_trace.o tinynn/tinynn_events.o $(GGML_DIR)/build/src/libggml.a
ifeq ($(UNAME_S),Darwin)
	# macOS variant (toy#71 Stage B follow-up, Mac-verified 2026-06-12):
	# -dynamiclib for -shared; -force_load per archive for GNU ld's
	# --whole-archive (pulls every ggml object so the Fiddle backend
	# resolves all tnn_* symbols); -lc++ for libc++ (not libstdc++); no
	# -Bsymbolic (macOS two-level namespace already binds internally).
	# Output keeps the .so name the gate/Fiddle loader expects.
	$(CC) -dynamiclib -o $@ \
	  tinynn/tinynn_ggml.o tinynn/tinynn_gguf.o tinynn/tinynn_trace.o tinynn/tinynn_events.o \
	  -Wl,-force_load,$(GGML_DIR)/build/src/libggml.a \
	  -Wl,-force_load,$(GGML_DIR)/build/src/libggml-cpu.a \
	  -Wl,-force_load,$(GGML_DIR)/build/src/libggml-base.a \
	  -lc++ -lpthread -lm
else
	$(CC) -shared -Wl,-Bsymbolic -o $@ \
	  tinynn/tinynn_ggml.o tinynn/tinynn_gguf.o tinynn/tinynn_trace.o tinynn/tinynn_events.o \
	  -L$(GGML_DIR)/build/src \
	  -Wl,--whole-archive -lggml -lggml-cpu -lggml-base -Wl,--no-whole-archive \
	  -lstdc++ -lpthread -lm
endif

# --- smoke test -------------------------------------------------------------
# Builds tinynn/smoke.rb against the CPU shim. Requires `setup-ggml` to have
# been run once first.
# --- gem release prep (toy#45) ----------------------------------------------
# The gem ships PRISTINE pinned ggml (patches apply at the consumer's vendor
# step), so reset the working tree's ggml before `gem build`. Re-run setup-ggml
# afterwards to restore the dev build. Also materialize the generated CUDA
# mirrors (gitignored; toy.gemspec ships lib/toy/llm/*_cuda.rb explicitly) —
# without them the gem's compute_cuda.rb requires point at missing files and
# Spinel silently compiles them to nothing (toy#70 finding).
# NB: reset to GGML_REV explicitly — NEVER FETCH_HEAD ("whatever was
# fetched last"): a cold-fetch test moved FETCH_HEAD to ggml master and
# this target silently staged UNVERIFIED ggml sources into the gem
# (caught at the v0.8.0 wire). The assert keeps it loud.
gem-prep: $(GGML_DIR)/CMakeLists.txt gen-mirrors
	cd $(GGML_DIR) && (git rev-parse --verify -q $(GGML_REV)^{commit} >/dev/null || git fetch -q --depth 1 origin $(GGML_REV)) && git reset --hard $(GGML_REV) >/dev/null
	rm -f $(GGML_DIR)/.patched
	@test "$$(cd $(GGML_DIR) && git rev-parse HEAD)" = "$(GGML_REV)" || { echo "FATAL: vendor/ggml HEAD != GGML_REV ($(GGML_REV)) after gem-prep"; exit 1; }
	@echo "ggml pristine at GGML_REV $$(cd $(GGML_DIR) && git rev-parse --short HEAD); now: gem build toy.gemspec"
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

tinynn/ab_smoke_silu: tinynn/ab_smoke_silu.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

ab-smoke-mul: tinynn/ab_smoke_mul
	./tinynn/ab_smoke_mul

tinynn/ab_smoke_mul: tinynn/ab_smoke_mul.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
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

tinynn/ab_smoke: tinynn/ab_smoke.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke.rb -o tinynn/ab_smoke

tinynn/ab_smoke_add: tinynn/ab_smoke_add.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_add.rb -o tinynn/ab_smoke_add

# E1.1 / GH#13 — Conv2D smoke + JSON dump for PyTorch parity.
tinynn/ab_smoke_conv2d: tinynn/ab_smoke_conv2d.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_conv2d.rb -o tinynn/ab_smoke_conv2d

# E1.2 / GH#13 — patch_embed composite smoke + parity dump.
tinynn/ab_smoke_patch_embed: tinynn/ab_smoke_patch_embed.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb lib/toy/models/toy_vit.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_patch_embed.rb -o tinynn/ab_smoke_patch_embed

# E1.3 / GH#13 — ViT-Tiny forward + training smoke.
prep/smokes/smoke_vit_tiny: prep/smokes/smoke_vit_tiny.rb lib/toy/llm/engine/vit_tiny_engine.rb lib/toy/models/toy_vit.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@

# toy#188 — the truck-and-trailer PLANT parity gate
# (dfa-for-dynamic-control's fixture; lib/toy/io/toy_truck_task.rb).
#
# Diffs 20 steps against a golden trajectory emitted by a verbatim
# transcription of the frontend's `drive()` under node, at 1e-12 per
# entry, then finite-difference-checks all 20 analytic Jacobian
# entries. NO tinynn dependency: the plant is arithmetic, and keeping
# the graph library out of its prereqs means a tinynn rebuild cannot
# make this leg look stale. toy#189's lane must not start until this
# is green.
prep/smokes/smoke_truck_plant: prep/smokes/smoke_truck_plant.rb lib/toy/io/toy_truck_task.rb
	$(SPINEL) $< -o $@
.PHONY: gate-truck-plant
gate-truck-plant: prep/smokes/smoke_truck_plant
	./prep/smokes/smoke_truck_plant

# E1.5 / GH#13 — image-loader smoke.
prep/smokes/smoke_image_loader: prep/smokes/smoke_image_loader.rb lib/toy/io/toy_image_loader.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# E1.6 / GH#13 — ViT-Tiny training driver.
examples/example_train_vit_tiny: examples/legacy/07_train_vit_tiny.rb lib/toy/llm/engine/vit_tiny_engine.rb lib/toy/models/toy_vit.rb lib/toy/models/toy_smollm2.rb lib/toy/io/toy_image_loader.rb lib/toy/train/toy_lr_schedule.rb lib/toy/train/toy_drift_grad.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a $(SPINEL_DEPS)
	$(SPINEL) $< -o $@
example_train_vit_tiny: examples/example_train_vit_tiny

tinynn/ab_smoke_gelu: tinynn/ab_smoke_gelu.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_gelu.rb -o tinynn/ab_smoke_gelu

tinynn/ab_smoke_rms_norm: tinynn/ab_smoke_rms_norm.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_rms_norm.rb -o tinynn/ab_smoke_rms_norm

tinynn/ab_smoke_softmax: tinynn/ab_smoke_softmax.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_softmax.rb -o tinynn/ab_smoke_softmax

tinynn/ab_smoke_flash_attn: tinynn/ab_smoke_flash_attn.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_flash_attn.rb -o tinynn/ab_smoke_flash_attn

tinynn/ab_smoke_q8_kv: tinynn/ab_smoke_q8_kv.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_q8_kv.rb -o tinynn/ab_smoke_q8_kv

tinynn/ab_smoke_moe_ffn: tinynn/ab_smoke_moe_ffn.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_moe_ffn.rb -o tinynn/ab_smoke_moe_ffn

tinynn/ab_smoke_transpose: tinynn/ab_smoke_transpose.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_transpose.rb -o tinynn/ab_smoke_transpose

tinynn/ab_smoke_scale: tinynn/ab_smoke_scale.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_scale.rb -o tinynn/ab_smoke_scale

tinynn/ab_smoke_pipeline: tinynn/ab_smoke_pipeline.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_pipeline.rb -o tinynn/ab_smoke_pipeline

# Chained FFNFFICache parity: pre, hidden, out vs hand-rolled native.
ab-smoke-ffncache: tinynn/ab_smoke_ffncache
	./tinynn/ab_smoke_ffncache

tinynn/ab_smoke_ffncache: tinynn/ab_smoke_ffncache.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_ffncache.rb -o tinynn/ab_smoke_ffncache

# ggml-native AdamW step (opt_step_adamw) parity vs project's plain-Adam.
ab-smoke-adamw-op: tinynn/ab_smoke_adamw_op
	./tinynn/ab_smoke_adamw_op

tinynn/ab_smoke_adamw_op: tinynn/ab_smoke_adamw_op.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_adamw_op.rb -o tinynn/ab_smoke_adamw_op

# Persistent-tensor architecture check: data uploaded to a ctx_w tensor
# survives a compute cycle.
ab-smoke-persistent: tinynn/ab_smoke_persistent
	./tinynn/ab_smoke_persistent

tinynn/ab_smoke_persistent: tinynn/ab_smoke_persistent.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_persistent.rb -o tinynn/ab_smoke_persistent

# Dual-cgraph + persistent-weights design check: forward reads t_w;
# adam mutates t_w in place; forward sees the new value.
ab-smoke-dual-graph: tinynn/ab_smoke_dual_graph
	./tinynn/ab_smoke_dual_graph

tinynn/ab_smoke_dual_graph: tinynn/ab_smoke_dual_graph.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_dual_graph.rb -o tinynn/ab_smoke_dual_graph

# M2 foundation: view_2d + cpy to write a single row into a persistent
# (max_T, d_head) KV buffer at a runtime-baked position.
ab-smoke-kv-write: tinynn/ab_smoke_kv_write
	./tinynn/ab_smoke_kv_write

tinynn/ab_smoke_kv_write: tinynn/ab_smoke_kv_write.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_kv_write.rb -o tinynn/ab_smoke_kv_write

# M2 prototype: single-step decode through a KV cache. Pre-fills K/V
# for positions 0..POS-1, writes k_new/v_new at POS, computes scores
# + soft_max_ext + head_out. Parity vs hand-rolled native.
ab-smoke-kv-attn: tinynn/ab_smoke_kv_attn
	./tinynn/ab_smoke_kv_attn

tinynn/ab_smoke_kv_attn: tinynn/ab_smoke_kv_attn.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_kv_attn.rb -o tinynn/ab_smoke_kv_attn

# M1.2: full single-block forward through the persistent graph.
# Parity vs native TransformerLM.forward() at n_layers=1, n_heads=2.
ab-smoke-full-forward-block: tinynn/ab_smoke_full_forward_block
	./tinynn/ab_smoke_full_forward_block

tinynn/ab_smoke_full_forward_block: tinynn/ab_smoke_full_forward_block.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_full_forward_block.rb -o tinynn/ab_smoke_full_forward_block

# Wallclock bench: native TransformerLM.forward vs FullForwardFFICache.
full-forward-bench: tinynn/full_forward_bench
	./tinynn/full_forward_bench

tinynn/full_forward_bench: tinynn/full_forward_bench.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/full_forward_bench.rb -o tinynn/full_forward_bench

full-forward-bench-cuda: tinynn/full_forward_bench_cuda
	./tinynn/full_forward_bench_cuda

tinynn/full_forward_bench_cuda: tinynn/full_forward_bench_cuda.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) tinynn/full_forward_bench_cuda.rb -o tinynn/full_forward_bench_cuda

ab-smoke-dual-graph-cuda: tinynn/ab_smoke_dual_graph_cuda
	./tinynn/ab_smoke_dual_graph_cuda

tinynn/ab_smoke_dual_graph_cuda: tinynn/ab_smoke_dual_graph_cuda.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) tinynn/ab_smoke_dual_graph_cuda.rb -o tinynn/ab_smoke_dual_graph_cuda

ab-smoke-adamw-op-cuda: tinynn/ab_smoke_adamw_op_cuda
	./tinynn/ab_smoke_adamw_op_cuda

tinynn/ab_smoke_adamw_op_cuda: tinynn/ab_smoke_adamw_op_cuda.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) tinynn/ab_smoke_adamw_op_cuda.rb -o tinynn/ab_smoke_adamw_op_cuda

# A/B harness for the "fuse-or-not" question: N_HEADS small matmuls vs
# 1 batched matmul at LoRA-Q shape. Override D_MODEL / N_HEADS / R / T
# via env to sweep launch-overhead vs compute-bound regimes. See
# docs/heavy-train-attribution-2026-05-24.md.
ab-smoke-lora-fused-cuda: tinynn/ab_smoke_lora_fused_cuda
	./tinynn/ab_smoke_lora_fused_cuda

tinynn/ab_smoke_lora_fused_cuda: tinynn/ab_smoke_lora_fused_cuda.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' tinynn/ab_smoke_lora_fused_cuda.rb -o tinynn/ab_smoke_lora_fused_cuda

# Transformer-shape sized parity + wallclock comparison.
ab-smoke-big: tinynn/ab_smoke_big
	./tinynn/ab_smoke_big

tinynn/ab_smoke_big: tinynn/ab_smoke_big.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_big.rb -o tinynn/ab_smoke_big

ab-smoke-matmul-variants: tinynn/ab_smoke_matmul_variants
	./tinynn/ab_smoke_matmul_variants

tinynn/ab_smoke_matmul_variants: tinynn/ab_smoke_matmul_variants.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_matmul_variants.rb -o tinynn/ab_smoke_matmul_variants

ab-smoke-back: tinynn/ab_smoke_back
	./tinynn/ab_smoke_back

tinynn/ab_smoke_back: tinynn/ab_smoke_back.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_back.rb -o tinynn/ab_smoke_back

ab-smoke-gelu-back: tinynn/ab_smoke_gelu_back
	./tinynn/ab_smoke_gelu_back

tinynn/ab_smoke_gelu_back: tinynn/ab_smoke_gelu_back.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_gelu_back.rb -o tinynn/ab_smoke_gelu_back

ab-smoke-cegrad: tinynn/ab_smoke_cegrad
	./tinynn/ab_smoke_cegrad

tinynn/ab_smoke_cegrad: tinynn/ab_smoke_cegrad.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_cegrad.rb -o tinynn/ab_smoke_cegrad

ab-smoke-adam: tinynn/ab_smoke_adam
	./tinynn/ab_smoke_adam

tinynn/ab_smoke_adam: tinynn/ab_smoke_adam.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_adam.rb -o tinynn/ab_smoke_adam

gguf-smoke: tinynn/gguf_smoke
	./tinynn/gguf_smoke

tinynn/gguf_smoke: tinynn/gguf_smoke.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gguf_smoke.rb -o tinynn/gguf_smoke

# Walks every tensor in data/distilgpt2-f32.gguf via tnn_gguf_*. Used to
# confirm large HF-converted GGUFs roundtrip through the project FFI.
gguf-inspect: tinynn/gguf_inspect
	./tinynn/gguf_inspect

tinynn/gguf_inspect: tinynn/gguf_inspect.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
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

tinynn/gpt2_load_smoke: tinynn/gpt2_load_smoke.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_load_smoke.rb -o tinynn/gpt2_load_smoke

# data/prompt_ids.txt, loads weights from data/distilgpt2-f32.gguf,
# greedy-generates N_NEW tokens via native Mat forward, writes the
# full ID sequence back. Decode with prep/tokens.py decode.

# Native Mat GPT-2 inference (DistilGPT2 / GPT-2 family).
#
gpt2:        demos/gpt2
demos/gpt2: demos/gpt2.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_gpt2.rb lib/toy/io/loaders/toy_gpt2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# SmolLM2-135M (llama-family) inference via Toy::SmolLM2.
# Tokenization is host-side: ./prep/smollm2_tokens.py encode "..."
smollm2:        demos/smollm2
demos/smollm2: demos/smollm2.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# SmolLM2-135M FFI KV-cache (CPU).
smollm2_kv:        demos/smollm2_kv
demos/smollm2_kv: demos/smollm2_kv.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Qwen2.5 Mat-mediated KV-cache (CPU). The slow, correct reference path.
# Run with `GGUF=data/qwen25-1.5b-f32.gguf ./demos/qwen25_kv` etc.
qwen25_kv:        demos/qwen25_kv
demos/qwen25_kv: demos/qwen25_kv.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Qwen2.5 Phase-2 mmap inference (CPU). Canonical performance path.
qwen25_native_mmap:        demos/qwen25_native_mmap
demos/qwen25_native_mmap: demos/qwen25_native_mmap.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Phase 0.7 acceptance gates: 0.5B (f32 + Q8) + 1.5B + 3B greedy-decode
# parity against locked-in golden token-ID sequences. Run before tagging
# a release; see docs/design/phase-07-acceptance.md.
qwen25_acceptance:        demos/qwen25_acceptance
demos/qwen25_acceptance: demos/qwen25_acceptance.rb lib/toy/models/arch.rb lib/toy/models/transformer_lm.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Inference bench (CPU). Long warmup + long prefill + per-token stats.
# Pick model via GGUF env; see docs/design/bench-cuda-2026-05-21.md.
qwen25_bench_cpu:        demos/qwen25_bench_cpu
demos/qwen25_bench_cpu: demos/qwen25_bench_cpu.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Inference bench (CUDA). Same shape as the CPU bench for side-by-side.
qwen25_bench_cuda:        demos/qwen25_bench_cuda
demos/qwen25_bench_cuda: demos/qwen25_bench_cuda.rb lib/toy/llm/engine/llama_kv_engine_cuda.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS)
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F1.2 step 2: LoRA-Q forward-parity gate. Loads SmolLM2-135M twice
# (baseline + LoRA r=16 B=0), asserts bit-identical generated IDs.
smollm2_lora_forward:        demos/smollm2_lora_forward
demos/smollm2_lora_forward: demos/smollm2_lora_forward.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# F1.2 step 3: backward through the full SmolLM2 decode graph,
# layer-0 LoRA-Q updated via SGD. Requires the vendored CONCAT
# backward in vendor/ggml/src/ggml.c.
smollm2_lora_train_step:        demos/smollm2_lora_train_step
demos/smollm2_lora_train_step: demos/smollm2_lora_train_step.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# F1.2 step 4: all-layers LoRA-Q SGD on real CE loss against a rare
# target token. 540 opt_step nodes (30 layers × 9 heads × 2 params).
# Acceptance: monotonic decrease over 20 steps.
smollm2_lora_train_ce:        demos/smollm2_lora_train_ce
demos/smollm2_lora_train_ce: demos/smollm2_lora_train_ce.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# F2 step 1: CUDA mirror of the LoRA forward parity gate.
smollm2_lora_forward_cuda:        demos/smollm2_lora_forward_cuda
demos/smollm2_lora_forward_cuda: demos/smollm2_lora_forward_cuda.rb lib/toy/llm/engine/llama_kv_engine_cuda.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F2 step 2: CUDA mirror of the multi-layer SGD CE training smoke.
smollm2_lora_train_ce_cuda:        demos/smollm2_lora_train_ce_cuda
demos/smollm2_lora_train_ce_cuda: demos/smollm2_lora_train_ce_cuda.rb lib/toy/llm/engine/llama_kv_engine_cuda.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# Task #70 diagnostic — same CE smoke but with every graph_b node
# pinned. Confirms sched intermediate-grad aliasing is the CPU
# divergence's root cause. See docs/design/task70-root-cause-2026-05-21.md.
smollm2_lora_train_ce_pinned:        demos/smollm2_lora_train_ce_pinned
demos/smollm2_lora_train_ce_pinned: demos/smollm2_lora_train_ce_pinned.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# F1.2 step 5: AdamW training with per-step m/v preservation via
# tnn_graph_reset_grads_only. Converges 7.5 → 0.09 in 20 SGD steps
# at LR=1e-3 — proper SFT-shaped learning curve.
smollm2_lora_train_adamw_cuda:        demos/smollm2_lora_train_adamw_cuda
demos/smollm2_lora_train_adamw_cuda: demos/smollm2_lora_train_adamw_cuda.rb lib/toy/llm/engine/llama_kv_engine_cuda.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F1.2 step 6a: multi-target AdamW SFT-shaped training. Cycles through
# 5 target tokens × 10 epochs at the same prefix; expects loss to
# drop on average + per-target. 10.8 → 3.6 in 10 epochs. Foundation
# for step 6b (multi-position) and step 7 (real alpaca dataset).
smollm2_lora_sft_multi_cuda:        demos/smollm2_lora_sft_multi_cuda
demos/smollm2_lora_sft_multi_cuda: demos/smollm2_lora_sft_multi_cuda.rb lib/toy/llm/engine/llama_kv_engine_cuda.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F1.2 step 6b — multi-position SFT (cycle pos4 / pos5). Validates
# that persistent Adam m/v (allocated by enable_lora_q_adamw! +
# realize_for_mmap) survive tnn_reset_for_rebuild between cycles.
smollm2_lora_sft_multipos_cuda:        demos/smollm2_lora_sft_multipos_cuda
demos/smollm2_lora_sft_multipos_cuda: demos/smollm2_lora_sft_multipos_cuda.rb lib/toy/llm/engine/llama_kv_engine_cuda.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# M3 step 1 — sequence-mode forward parity at T=1.
# LlamaSeqForwardFFICache.forward([id], [0]) must match
# SmolLM2KVFFICache + decode_step(id, 0). See
# docs/design/m3-seq-forward-2026-05-21.md.
smollm2_seq_parity:        demos/smollm2_seq_parity
demos/smollm2_seq_parity: demos/smollm2_seq_parity.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# M3 step 2 — T=4 trajectory parity (CPU). Per-position seq logits must
# match the decode_step trajectory; proves causal-mask + multi-pos RoPE.
smollm2_seq_parity_t4:        demos/smollm2_seq_parity_t4
demos/smollm2_seq_parity_t4: demos/smollm2_seq_parity_t4.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# M3 step 2 — CUDA mirror. T=1 and T=4 vs CPU decode_step trajectory.
smollm2_seq_parity_cuda:        demos/smollm2_seq_parity_cuda
demos/smollm2_seq_parity_cuda: demos/smollm2_seq_parity_cuda.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

smollm2_seq_parity_t4_cuda:        demos/smollm2_seq_parity_t4_cuda
demos/smollm2_seq_parity_t4_cuda: demos/smollm2_seq_parity_t4_cuda.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# M3 step 3 — seq-mode LoRA training smoke (CPU). One forward + backward
# + opt_step over T positions; loss should decrease over N steps.
smollm2_seq_train:        demos/smollm2_seq_train
demos/smollm2_seq_train: demos/smollm2_seq_train.rb lib/toy/llm/engine/llama_seq_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

smollm2_seq_train_cuda:        demos/smollm2_seq_train_cuda
demos/smollm2_seq_train_cuda: demos/smollm2_seq_train_cuda.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F3 — full fine-tune on CUDA. Every per-block weight tensor is
# writable F32 + AdamW state; opt_step on each. See
# docs/roadmap/f3-full-finetune-2026-05-21.md.
smollm2_seq_full_finetune_cuda:        demos/smollm2_seq_full_finetune_cuda
demos/smollm2_seq_full_finetune_cuda: demos/smollm2_seq_full_finetune_cuda.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# F4 (QLoRA) on CUDA via realize_for_q8_copy. Q8 base in standard
# CUDA buffer + F32 LoRA adapter; bypasses the BYO-pointer padding bug.
smollm2_seq_qlora_cuda:        demos/smollm2_seq_qlora_cuda
demos/smollm2_seq_qlora_cuda: demos/smollm2_seq_qlora_cuda.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# Training step-time bench. MODE=lora|ft; STEPS=N; GGUF=path.
# toy#77: the seq-engine mirror requires the primitives/blocks/archs
# mirrors; without them in the dep list a FRESH checkout generates only
# llama_seq_engine_cuda.rb, its require_relatives dangle (Spinel ignores
# them with a warning), every engine type degrades to int and .new
# returns nil — the demo then segfaults in the first attr setter.
seq_train_bench_cuda:        demos/seq_train_bench_cuda
demos/seq_train_bench_cuda: demos/seq_train_bench_cuda.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/llm/primitives/rms_norm_cuda.rb lib/toy/llm/primitives/rope_cuda.rb lib/toy/llm/primitives/swiglu_cuda.rb lib/toy/llm/primitives/gqa_cuda.rb lib/toy/llm/blocks/transformer_block_cuda.rb lib/toy/llm/archs/llama_arch_cuda.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS)
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# Per-phase training-step bench (CPU + CUDA). Times graph_reset /
# uploads / compute_backward / download separately. Doc:
# docs/design/bench-train-2026-05-21.md.
smollm2_lora_train_bench:        demos/smollm2_lora_train_bench
demos/smollm2_lora_train_bench: demos/smollm2_lora_train_bench.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

smollm2_lora_train_bench_cuda:        demos/smollm2_lora_train_bench_cuda
demos/smollm2_lora_train_bench_cuda: demos/smollm2_lora_train_bench_cuda.rb lib/toy/llm/engine/llama_kv_engine_cuda.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# Task #70 grad-magnitude probes (per-layer maxabs(grad_A), maxabs(grad_B)).
smollm2_lora_grad_probe:        demos/smollm2_lora_grad_probe
demos/smollm2_lora_grad_probe: demos/smollm2_lora_grad_probe.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

smollm2_lora_grad_probe_cuda:        demos/smollm2_lora_grad_probe_cuda
demos/smollm2_lora_grad_probe_cuda: demos/smollm2_lora_grad_probe_cuda.rb lib/toy/llm/engine/llama_kv_engine_cuda.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# Qwen2.5 Phase-2 mmap inference (CUDA). Requires `make setup-ggml-cuda`.
qwen25_native_mmap_cuda:        demos/qwen25_native_mmap_cuda
demos/qwen25_native_mmap_cuda: demos/qwen25_native_mmap_cuda.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/llm/engine/llama_kv_engine_cuda.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@

# SmolLM2-135M FFI KV-cache (CUDA).
smollm2_kv_cuda:        demos/smollm2_kv_cuda
demos/smollm2_kv_cuda: demos/smollm2_kv_cuda.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/llm/engine/llama_kv_engine_cuda.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) $< -o $@

# TinyLlama-1.1B demo. Uses the same Toy::SmolLM2 / FFI KV CUDA stack
# (llama-family architecture); just configured for the larger shape.
tinyllama_kv_cuda:        demos/tinyllama_kv_cuda
demos/tinyllama_kv_cuda: demos/tinyllama_kv_cuda.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/llm/engine/llama_kv_engine_cuda.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a
	$(SPINEL) $< -o $@

tinyllama:        demos/tinyllama
demos/tinyllama: demos/tinyllama.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

tinyllama_kv:        demos/tinyllama_kv
demos/tinyllama_kv: demos/tinyllama_kv.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_smollm2.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/llm/engine/llama_kv_engine.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Print the Phuong–Hutter algorithm cards for both models. No
# inference — just emit the structured pseudocode. Source-of-truth
# for the round-trip work (task #33).
algorithm_cards:        demos/algorithm_cards
demos/algorithm_cards: demos/algorithm_cards.rb lib/toy/dev/toy_card.rb lib/toy.rb lib/toy/models/toy_gpt2.rb lib/toy/models/toy_smollm2.rb lib/toy/io/loaders/toy_gpt2_loader.rb lib/toy/io/loaders/toy_smollm2_loader.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# TinyStories from-scratch training via Toy::Trainer.
#
train:        demos/train
demos/train: demos/train.rb lib/toy/train/toy_trainer.rb lib/toy/models/transformer.rb lib/toy/train/training.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) $< -o $@

# Parity probe: one forward at distilgpt2 shape, dump last-row logits
# to data/ours_logits.txt. Pair with prep/parity.py for the HF reference.
gpt2-parity: tinynn/gpt2_parity
	./tinynn/gpt2_parity

tinynn/gpt2_parity: tinynn/gpt2_parity.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_parity.rb -o tinynn/gpt2_parity

# FFI parity probe: persistent ggml graph with LayerNorm + biases.
# Dumps last-row logits to data/ours_ffi_logits.txt.
gpt2-ffi-parity: tinynn/gpt2_ffi_parity
	./tinynn/gpt2_ffi_parity

tinynn/gpt2_ffi_parity: tinynn/gpt2_ffi_parity.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/llm/engine/gpt2_fwd_engine.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_ffi_parity.rb -o tinynn/gpt2_ffi_parity

# Apples-to-apples bench: native Mat vs FFI on the same forward.
# Re-encode data/prompt_ids.txt first so prompt length matches T_SEQ=5.
gpt2-bench: tinynn/gpt2_bench
	./tinynn/gpt2_bench

tinynn/gpt2_bench: tinynn/gpt2_bench.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/llm/engine/gpt2_fwd_engine.rb lib/toy/llm/engine/gpt2_kv_engine.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_bench.rb -o tinynn/gpt2_bench

# Ruby BPE smoke: load vocab/merges, encode + roundtrip-decode some
# fixed prompts. Compare against prep/tokens.py output.
bpe-smoke: tinynn/bpe_smoke
	./tinynn/bpe_smoke

tinynn/bpe_smoke: tinynn/bpe_smoke.rb lib/toy/models/transformer.rb lib/toy/io/bpe.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/bpe_smoke.rb -o tinynn/bpe_smoke

# KV-cache parity probe: prefill the prompt one token at a time through
# GPT2KVFFICache, dump last-position logits.
gpt2-kv-parity: tinynn/gpt2_kv_parity
	./tinynn/gpt2_kv_parity

tinynn/gpt2_kv_parity: tinynn/gpt2_kv_parity.rb lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/llm/engine/gpt2_kv_engine.rb lib/toy/io/gguf_load.rb lib/toy/train/training.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/gpt2_kv_parity.rb -o tinynn/gpt2_kv_parity

# --- CUDA mirrors of the GPT-2 demos / parity / bench --------------
# All require `make setup-ggml-cuda` to have produced
# vendor/ggml/build-cuda first. Built on the gx10 (NVIDIA GB10);
# the Mac build doesn't have CUDA.

CUDA_GPT2_DEPS = lib/toy/models/transformer.rb lib/toy/models/gpt2.rb lib/toy/io/gguf_load.rb \
                 lib/toy/train/training.rb lib/toy/ffi/tinynn.rb lib/toy/ffi/tinynn_cuda.rb \
                 tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a

gpt2-ffi-parity-cuda: tinynn/gpt2_ffi_parity_cuda
	./tinynn/gpt2_ffi_parity_cuda

tinynn/gpt2_ffi_parity_cuda: tinynn/gpt2_ffi_parity_cuda.rb lib/toy/llm/engine/gpt2_fwd_engine_cuda.rb $(CUDA_GPT2_DEPS)
	$(SPINEL) tinynn/gpt2_ffi_parity_cuda.rb -o tinynn/gpt2_ffi_parity_cuda

gpt2-kv-parity-cuda: tinynn/gpt2_kv_parity_cuda
	./tinynn/gpt2_kv_parity_cuda

tinynn/gpt2_kv_parity_cuda: tinynn/gpt2_kv_parity_cuda.rb lib/toy/llm/engine/gpt2_kv_engine_cuda.rb $(CUDA_GPT2_DEPS)
	$(SPINEL) tinynn/gpt2_kv_parity_cuda.rb -o tinynn/gpt2_kv_parity_cuda

gpt2-bench-cuda: tinynn/gpt2_bench_cuda
	./tinynn/gpt2_bench_cuda

tinynn/gpt2_bench_cuda: tinynn/gpt2_bench_cuda.rb lib/toy/llm/engine/gpt2_fwd_engine_cuda.rb lib/toy/llm/engine/gpt2_kv_engine_cuda.rb $(CUDA_GPT2_DEPS)
	$(SPINEL) tinynn/gpt2_bench_cuda.rb -o tinynn/gpt2_bench_cuda

ab-smoke-embed: tinynn/ab_smoke_embed
	./tinynn/ab_smoke_embed

tinynn/ab_smoke_embed: tinynn/ab_smoke_embed.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_embed.rb -o tinynn/ab_smoke_embed

ab-smoke-sgd: tinynn/ab_smoke_sgd
	./tinynn/ab_smoke_sgd

tinynn/ab_smoke_sgd: tinynn/ab_smoke_sgd.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_sgd.rb -o tinynn/ab_smoke_sgd

# F1.2 step 1: multi-step LoRA convergence via the F1.1 in-graph
# optimizer. Toy shape; SGD; 60 steps; asserts final loss < 10% of
# initial (passes at ~10e-13 of initial).
ab-smoke-lora-train: tinynn/ab_smoke_lora_train
	./tinynn/ab_smoke_lora_train

tinynn/ab_smoke_lora_train: tinynn/ab_smoke_lora_train.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_lora_train.rb -o tinynn/ab_smoke_lora_train

# Forward-only smoke: does TransformerLM#forward run at current Spinel
# master? (The #473 SIGBUS is in backward; forward might be OK.)
forward-smoke: tinynn/forward_smoke
	./tinynn/forward_smoke

tinynn/forward_smoke: tinynn/forward_smoke.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/forward_smoke.rb -o tinynn/forward_smoke

persistent-bench: tinynn/persistent_bench
	./tinynn/persistent_bench

tinynn/persistent_bench: tinynn/persistent_bench.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/persistent_bench.rb -o tinynn/persistent_bench

persistent-bench-cuda: tinynn/persistent_bench_cuda
	./tinynn/persistent_bench_cuda

tinynn/persistent_bench_cuda: tinynn/persistent_bench_cuda.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/persistent_bench_cuda.rb -o tinynn/persistent_bench_cuda

persistent-bench-big: tinynn/persistent_bench_big
	./tinynn/persistent_bench_big

tinynn/persistent_bench_big: tinynn/persistent_bench_big.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn.rb tinynn/libtinynn_ggml.a
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

tinynn/ab_smoke_cuda: tinynn/ab_smoke_cuda.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_cuda.rb -o tinynn/ab_smoke_cuda

# Consolidated CUDA parity test: matmul + add + gelu + rms_norm + softmax + scale + ffn_pipeline.
ab-smoke-all-cuda: tinynn/ab_smoke_all_cuda
	./tinynn/ab_smoke_all_cuda

tinynn/ab_smoke_all_cuda: tinynn/ab_smoke_all_cuda.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a
	$(SPINEL) tinynn/ab_smoke_all_cuda.rb -o tinynn/ab_smoke_all_cuda

# Transformer-shape parity + wallclock bench on CUDA (GB10).
ab-smoke-big-cuda: tinynn/ab_smoke_big_cuda
	./tinynn/ab_smoke_big_cuda

tinynn/ab_smoke_big_cuda: tinynn/ab_smoke_big_cuda.rb lib/toy/models/transformer.rb lib/toy/ffi/tinynn_cuda.rb tinynn/libtinynn_ggml.a
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

# Metal perf leg (macOS only; #104 part C). Times the metal-vs-cpu infer
# runners via N-differencing — steady-state decode ms/token plus the
# metal-vs-cpu ratio on THIS machine. The baseline (bench/baselines_metal.csv)
# is Mac-pinned, like the metal_gate float baseline; capture it with
# `make bench-metal-update` on a QUIESCED machine (desktop load skews the
# numbers badly). Skips green off macOS, exactly like gate-metal.
.PHONY: bench-metal bench-metal-update bench-metal-report
bench-metal:
ifneq ($(UNAME_S),Darwin)
	@echo "bench-metal: Metal is macOS-only (uname -s = $(UNAME_S)) — skipping"; exit 0
else
	$(MAKE) libexec/toy-infer-metal libexec/toy-infer
	ruby bench/check_metal.rb
endif

bench-metal-update:
ifneq ($(UNAME_S),Darwin)
	@echo "bench-metal-update: Metal is macOS-only (uname -s = $(UNAME_S)) — skipping"; exit 0
else
	$(MAKE) libexec/toy-infer-metal libexec/toy-infer
	ruby bench/check_metal.rb --update
endif

bench-metal-report:
ifneq ($(UNAME_S),Darwin)
	@echo "bench-metal-report: Metal is macOS-only (uname -s = $(UNAME_S)) — skipping"; exit 0
else
	$(MAKE) libexec/toy-infer-metal libexec/toy-infer
	ruby bench/check_metal.rb --report
endif

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

# toy#109 P1 — the FrankenModel credit-assignment twin-lane runner.
# Lane A always :chain; lane B follows FRANKEN_POLICY (per-layer
# chain|dfa). Per-matmul DFA (design doc §4c) + shadow alignment
# telemetry. Own compilation unit (hybrid precedent).
libexec/toy-train-franken: lib/toy/run/train_franken.rb lib/toy.rb lib/toy/ffi/tinynn.rb \
		lib/toy/llm/primitives/rms_norm.rb \
		tinynn/libtinynn_ggml.a $(SPINEL_DEPS) \
		lib/toy/dev/toy_card.rb lib/toy/models/transformer.rb lib/toy/train/dfa_b.rb lib/toy/version.rb | libexec
	$(SPINEL) $< -o $@
.PHONY: toy-train-franken
toy-train-franken: libexec/toy-train-franken

# Gates: twin-parity (policy=chain,chain ⇒ lanes byte-identical),
# byte-repro (two runs identical), dfa-decreases, alignment well-formed.
.PHONY: gate-franken
gate-franken: libexec/toy-train-franken
	ruby prep/franken_gate.rb

# toy#109 P2 — engine-parity smoke: FromScratch vs FrankenFromScratch
# all-chain byte-parity + the per-head dfa arm through the real engine.
prep/smokes/smoke_franken_parity: prep/smokes/smoke_franken_parity.rb lib/toy/compute.rb \
		lib/toy/llm/recipes/franken_from_scratch.rb lib/toy/llm/engine/llama_seq_engine.rb \
		lib/toy/train/dfa_b.rb lib/toy/llm/recipe_options.rb \
		tinynn/libtinynn_ggml.a $(SPINEL_DEPS) | libexec
	$(SPINEL) $< -o $@
.PHONY: gate-franken-parity
gate-franken-parity: prep/smokes/smoke_franken_parity
	@out="$$(./prep/smokes/smoke_franken_parity 2>&1)"; \
	echo "$$out" | tail -6; \
	echo "$$out" | grep -q "^franken-parity: ok$$" || { echo "GATE FAIL [franken-parity]"; exit 1; }; \
	echo "GATE PASS [franken-parity]: engine all-chain byte-parity + dfa arm (toy#109 P2)"

# toy#109 P2b — the Franken-MoE arm: dense 2-expert soft mixture,
# BP-router + DFA-experts twin lanes.
libexec/toy-train-franken-moe: lib/toy/run/train_franken_moe.rb lib/toy/run/franken_moe_parts.rb lib/toy.rb lib/toy/ffi/tinynn.rb \
		lib/toy/llm/primitives/rms_norm.rb lib/toy/train/dfa_b.rb \
		tinynn/libtinynn_ggml.a $(SPINEL_DEPS) \
		lib/toy/dev/toy_card.rb lib/toy/llm/primitives/muon.rb lib/toy/llm/primitives/situ_glu.rb lib/toy/models/transformer.rb lib/toy/version.rb | libexec
	$(SPINEL) $< -o $@
.PHONY: gate-franken-moe
gate-franken-moe: libexec/toy-train-franken-moe
	ruby prep/franken_moe_gate.rb

# toy#120 — the spec-callable single-lane MoE runner (`toy train
# franken-moe`, the F4 surface). Shares franken_moe_parts.rb with the rig.
libexec/toy-train-franken-moe-cli: lib/toy/run/train_franken_moe_cli.rb lib/toy/run/franken_moe_parts.rb \
		lib/toy.rb lib/toy/ffi/tinynn.rb lib/toy/io/json_builder.rb lib/toy/io/json.rb \
		lib/toy/io/toy_events.rb lib/toy/dev/toy_describe_flow.rb \
		lib/toy/llm/primitives/rms_norm.rb lib/toy/train/dfa_b.rb \
		lib/toy/llm/primitives/muon.rb \
		lib/toy/llm/labels.rb lib/toy/io/toy_corpus_loader.rb \
		tinynn/libtinynn_ggml.a $(SPINEL_DEPS) \
		lib/toy/dev/toy_card.rb lib/toy/llm/primitives/situ_glu.rb lib/toy/models/transformer.rb lib/toy/version.rb | libexec
	$(SPINEL) $< -o $@
.PHONY: gate-franken-moe-cli
gate-franken-moe-cli: libexec/toy-train-franken-moe-cli libexec/toy-train-franken-moe
	ruby prep/franken_moe_cli_gate.rb

# toy#112 — the spec-callable franken runner (toy train franken): the
# from-scratch drive through FrankenFromScratch, TAO_RUN_DIR events with
# policy provenance + opt-in align events. Own unit (landmine #16).
libexec/toy-train-franken-llama: lib/toy/run/train_franken_llama.rb lib/toy.rb lib/toy/ffi/tinynn.rb \
		lib/toy/io/json_builder.rb lib/toy/io/json.rb lib/toy/io/toy_events.rb \
		lib/toy/llm/engine/llama_seq_engine.rb lib/toy/llm/recipes/franken_from_scratch.rb \
		lib/toy/llm/archs/llama_arch.rb lib/toy/llm/archs/layer_spec.rb \
		lib/toy/llm/blocks/transformer_block.rb lib/toy/llm/blocks/gdn_block.rb \
		lib/toy/llm/blocks/kda_block.rb lib/toy/llm/primitives/kda.rb \
		lib/toy/llm/blocks/mla_block.rb lib/toy/llm/primitives/mla.rb \
		lib/toy/llm/primitives/gdn.rb lib/toy/llm/primitives/situ_glu.rb \
		lib/toy/llm/primitives/swiglu.rb lib/toy/llm/primitives/rms_norm.rb \
		lib/toy/llm/primitives/rope.rb lib/toy/llm/primitives/gqa.rb \
		lib/toy/llm/recipe_options.rb lib/toy/dev/toy_describe_flow.rb lib/toy/train/dfa_b.rb \
		lib/toy/train/toy_gguf_writer.rb lib/toy/train/toy_gguf_fuse.rb \
		tinynn/libtinynn_ggml.a $(SPINEL_DEPS) \
		lib/toy/dev/toy_card.rb lib/toy/io/toy_corpus_loader.rb lib/toy/llm/adamw.rb lib/toy/llm/labels.rb lib/toy/llm/primitives/muon.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/train/toy_lr_schedule.rb lib/toy/version.rb | libexec
	$(SPINEL) $< -o $@
.PHONY: gate-franken-llama
gate-franken-llama: libexec/toy-train-franken-llama
	ruby prep/franken_llama_gate.rb

# GDN reintegration gate (docs/roadmap/gdn-hybrid-engine-reintegration.md,
# folded in 2026-07-25): all-attention byte-exact + GDN-through-the-engine.
.PHONY: gate-gdn-engine
gate-gdn-engine: libexec/toy-train
	ruby prep/gdn_engine_gate.rb

# The core training byte-gate matrix (prep/train_gate.rb: from-scratch /
# warm-start / lora / vit against frozen fixtures). Previously only the
# sweep scripts called it — toy#119 gave it a first-class target so the
# `gates` aggregate covers it.
.PHONY: gate-train
gate-train: libexec/toy-train
	ruby prep/train_gate.rb

# toy#119 process note: one command that runs EVERY gate leg. A shared-
# layer change (tinynn/*.c, the engine unit, a toolchain/pin move) must
# sweep ALL consumers, not just the leg being worked on — toy#113 and
# toy#119 both escaped through a green sibling gate. The list is
# self-parsed from this Makefile (any new gate-* target joins
# automatically); the off-platform backend legs are filtered by uname.
ALL_GATES := $(sort $(shell grep -oE '^gate-[a-z0-9-]+' $(firstword $(MAKEFILE_LIST))))
ifeq ($(shell uname),Darwin)
GATES := $(filter-out gate-cuda %-cuda,$(ALL_GATES))
else
GATES := $(filter-out gate-metal %-metal,$(ALL_GATES))
endif
.PHONY: gates
gates: $(GATES)
	@echo "ALL GATES PASS ($(words $(GATES)) legs)"

# ── gates-fast: the same legs, in two phases, safe under -j ──
#
# `make -j gates` is NOT safe, and the reason is not obvious. Several
# gate targets declare no prerequisites (gate-cpu, gate-cuda,
# gate-ckpt-roundtrip, gate-lmc, gate-consumer, ...), and 31 of the gate
# SCRIPTS shell out to `make` themselves to build a runner they find
# missing. Run those concurrently and two recursive makes can build the
# SAME libexec/* path at once — two spinel processes writing one output
# file. That is a corrupt binary, not a slow build, and it would surface
# later as an unrelated-looking gate failure.
#
# Phase 1 builds EVERY runner first, so by phase 2 no gate script's
# fallback can fire and the race has nothing to race over. Phase 2 then
# runs the legs concurrently. Both phases inherit the caller's -j through
# the jobserver, so `make -j8 gates-fast` parallelises both.
#
# WHY IT IS WORTH IT, and what it CANNOT buy: a serial battery leaves the
# box idle, and -j4 takes 70.3 min down to 52.8. But the floor is 49.8 min
# because gate-franken-moe-cli alone is that long, so the parallel win is
# nearly exhausted at -j4 and further job count makes it WORSE.
#
# (An earlier version of this comment named gate-lstm as the critical path
# at ~15 min. That came from reading a profile file while it was still
# being written — the run had not reached franken-moe-cli yet. The
# per-leg profile is now `prep/gate_profile.sh`, which is Makefile-derived,
# serial, and records exit codes, and whose total reconciles with the
# measured serial battery to within 2 seconds.)
#
# CUDA IS DELIBERATELY SERIALISED. The cuda legs share one GB10 and one
# unified 121 GiB pool; running them concurrently trades a small wall
# win for contention and OOM flakiness on a box that also hosts other
# services (see gx-status). They run after, serially, and that is a
# choice rather than an oversight.
GATE_RUNNERS := $(sort $(shell grep -oE '^libexec/toy-[a-z0-9-]+' $(firstword $(MAKEFILE_LIST))))
ifeq ($(shell uname),Darwin)
GATE_RUNNERS := $(filter-out %-cuda,$(GATE_RUNNERS))
else
GATE_RUNNERS := $(filter-out %-metal,$(GATE_RUNNERS))
endif
GATES_CUDA := $(filter %-cuda gate-cuda,$(GATES))
# gate-run-log SCANS THE WHOLE REPO runs/ TREE, so it cannot run beside
# any gate that writes a bundle: it will read one mid-write and fail with
# "has no run_start event" on a bundle that is perfectly valid seconds
# later. That is a flaky red battery caused by nothing but timing, and it
# bit once before being understood. It runs in the serial phase, after
# every bundle-writing leg has finished.
GATES_SERIAL := gate-run-log
GATES_CPU  := $(filter-out $(GATES_CUDA) $(GATES_SERIAL),$(GATES))
# Explicit, NOT inherited from the caller's -j. `make -j gates-fast` with
# an unlimited jobserver would launch all 49 CPU legs at once, and this
# box shares one 121 GiB pool with whatever else is resident (gx-status).
# Phase 1 is single-threaded spinel + cc1 per file, so it takes a wider
# fan-out than the legs, which mix 4-thread training runs in.
# MEASURED on an idle box, identical build state per variant (prep/gate_profile.sh
# for the per-leg numbers). Wall time for the full battery:
#
#   serial    70.3 min      -j6   60.6 min      -j4   52.8 min      -j10  76.8 min
#
# -j10 is SLOWER THAN SERIAL: runners are ~4-thread, so 10 jobs put 40 threads on
# 20 cores. -j4 is the measured optimum and sits within 6% of the hard floor, which
# is 49.8 min — the wall time of gate-franken-moe-cli, ONE LEG that is 71% of the
# battery. So there is nothing left for job count to buy; the only remaining lever
# is that leg. Phase 1 stays wider because it is single-threaded spinel + cc1.
BUILD_JOBS ?= 10
GATE_JOBS  ?= 4
.PHONY: gates-fast
gates-fast:
	@start=$$(date +%s); \
	 echo "== phase 1/4: building $(words $(GATE_RUNNERS)) runners (-j$(BUILD_JOBS)) =="; \
	 $(MAKE) -j$(BUILD_JOBS) $(GATE_RUNNERS) && \
	 echo "== phase 2/4: $(words $(GATES_CPU)) CPU gate legs (-j$(GATE_JOBS)) ==" && \
	 $(MAKE) -j$(GATE_JOBS) $(GATES_CPU) && \
	 echo "== phase 3/4: $(words $(GATES_CUDA)) CUDA gate legs (serial, one GPU) ==" && \
	 $(MAKE) -j1 $(GATES_CUDA) && \
	 echo "== phase 4/4: $(words $(GATES_SERIAL)) leg(s) that must run ALONE ==" && \
	 $(MAKE) -j1 $(GATES_SERIAL) && \
	 echo "ALL GATES PASS ($(words $(GATES)) legs, WALL $$(( $$(date +%s) - start ))s)"

# ── the capability/fixture split, as SUBSETS ───────────────────────────
#
# `gates` and `gates-fast` still run EVERY leg and that is deliberate.
# toy#119's note above is emphatic about why: a shared-layer change must
# sweep all consumers, and toy#113 and toy#119 BOTH escaped through a green
# sibling gate. Splitting the default aggregate would re-open exactly that
# hole. These two targets are for iterating, not for signing off.
#
# The lane list is READ FROM THE RUBY, not typed here. FIXTURE_RECIPES in
# lib/toy/core/cli/train.rb is the one place that decides what a fixture is;
# a second hand-maintained copy would drift, which is the failure this
# cleanup has now found five times (the recipe list, the prereq lists, the
# ggml patch, the `lr` literal, the matrix gate's own probe count).
FIXTURE_LANES := $(shell sed -n '/FIXTURE_RECIPES *= *%w\[/,/\]/p' lib/toy/core/cli/train.rb | tr '\n' ' ' | sed 's/.*%w\[//; s/\].*//')
# gate-lmc is the two-checkpoint linear-mode-connectivity instrument — a
# research instrument that is not named after a lane, so it is added by hand.
GATES_RESEARCH  := $(sort $(foreach l,$(FIXTURE_LANES),$(filter gate-$(l) gate-$(l)-%,$(GATES))) $(filter gate-lmc,$(GATES)))
GATES_FRAMEWORK := $(filter-out $(GATES_RESEARCH),$(GATES))

# Each aggregate reports its own WALL time. Measured 2026-08-25 on an idle
# gx10 with runners already built: gates-framework = 50s for 39 legs. That
# number is the whole point of the split — a framework change is now
# checkable in under a minute instead of behind the ~50-minute
# gate-franken-moe-cli floor, which lives in the research half. Printing it
# rather than leaving it to be re-measured by hand: the reason a subset
# target exists is its cost, so the cost should be in the output.
#
# WARM vs COLD: these run whatever builds their prerequisites demand, so a
# first run after touching a runner pays the compile. The 50s figure is the
# warm, iterating case the target is for.
.PHONY: gates-framework
gates-framework:
	@start=$$(date +%s); \
	 echo "== $(words $(GATES_FRAMEWORK)) FRAMEWORK legs (of $(words $(GATES)) total) =="; \
	 $(MAKE) -j$(GATE_JOBS) $(filter-out $(GATES_CUDA) $(GATES_SERIAL),$(GATES_FRAMEWORK)) && \
	 $(MAKE) -j1 $(filter $(GATES_CUDA),$(GATES_FRAMEWORK)) && \
	 $(MAKE) -j1 $(filter $(GATES_SERIAL),$(GATES_FRAMEWORK)) && \
	 echo "FRAMEWORK GATES PASS ($(words $(GATES_FRAMEWORK)) legs, WALL $$(( $$(date +%s) - start ))s) — NOT a full sweep; run 'make gates-fast' before shipping"

.PHONY: gates-research
gates-research:
	@start=$$(date +%s); \
	 echo "== $(words $(GATES_RESEARCH)) RESEARCH FIXTURE legs (of $(words $(GATES)) total) =="; \
	 $(MAKE) -j$(GATE_JOBS) $(filter-out $(GATES_CUDA) $(GATES_SERIAL),$(GATES_RESEARCH)) && \
	 $(MAKE) -j1 $(filter $(GATES_CUDA),$(GATES_RESEARCH)) && \
	 echo "RESEARCH GATES PASS ($(words $(GATES_RESEARCH)) legs, WALL $$(( $$(date +%s) - start ))s) — NOT a full sweep; run 'make gates-fast' before shipping"

# Partition assertion: every leg lands in exactly one half. Cheap, and it
# is what keeps the derived lane list honest — if the sed ever stops
# matching, GATES_RESEARCH silently empties and `gates-framework` quietly
# becomes the whole battery while still calling itself a subset.
# runs/ regrows: the battery writes bundles (~49 per gates-framework run,
# more for a full sweep), and gate-run-log scans the whole tree. That is
# how it reached 15,842 bundles / 2.6 GB. So keeping it small is a target
# rather than something someone remembers to do by hand.
#
# DRY RUN BY DEFAULT — deleting is irreversible (runs/ is gitignored, so
# there is no git checkout back). APPLY=1 to act, ARCHIVE=dir to move
# instead of delete, KEEP=n to change the per-prefix depth (default 3).
#
#   make runs-prune
#   make runs-prune APPLY=1
#   make runs-prune APPLY=1 ARCHIVE=/srv/data/scratch/toy-runs
.PHONY: runs-prune
runs-prune:
	@ruby prep/prune_runs.rb $(if $(APPLY),--apply) $(if $(ARCHIVE),--archive $(ARCHIVE))

.PHONY: gates-partition-check
gates-partition-check:
	@test $$(( $(words $(GATES_FRAMEWORK)) + $(words $(GATES_RESEARCH)) )) -eq $(words $(GATES)) \
	  || { echo "PARTITION BROKEN: $(words $(GATES_FRAMEWORK))+$(words $(GATES_RESEARCH)) != $(words $(GATES))"; exit 1; }
	@test $(words $(FIXTURE_LANES)) -eq 11 \
	  || { echo "FIXTURE_LANES read $(words $(FIXTURE_LANES)) lanes, expected 11 — the sed against FIXTURE_RECIPES has drifted"; exit 1; }
	@echo "ok: $(words $(GATES)) legs = $(words $(GATES_FRAMEWORK)) framework + $(words $(GATES_RESEARCH)) research; $(words $(FIXTURE_LANES)) lanes read from FIXTURE_RECIPES"

# toy#109 CUDA franken leg — the CUDA twin of the spec-callable runner
# (`toy train franken --device cuda`). Force-link keeps the CUDA backend
# registration alive (same as every CUDA trainer).
libexec/toy-train-franken-llama-cuda: lib/toy/run/train_franken_llama_cuda.rb lib/toy.rb \
		lib/toy/llm/primitives/kda.rb lib/toy/llm/blocks/kda_block.rb \
		lib/toy/llm/primitives/mla.rb lib/toy/llm/blocks/mla_block.rb \
		lib/toy/llm/primitives/situ_glu.rb lib/toy/llm/archs/llama_arch.rb \
		lib/toy/llm/blocks/transformer_block.rb \
		lib/toy/ffi/tinynn_cuda.rb lib/toy/io/json_builder.rb lib/toy/io/json.rb \
		lib/toy/io/toy_events.rb lib/toy/llm/recipe_options.rb lib/toy/dev/toy_describe_flow.rb lib/toy/train/dfa_b.rb \
		lib/toy/train/toy_gguf_writer.rb lib/toy/train/toy_gguf_fuse.rb \
		tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS) \
		lib/toy/dev/toy_card.rb lib/toy/ffi/tinynn.rb lib/toy/io/toy_corpus_loader.rb lib/toy/llm/adamw.rb lib/toy/llm/archs/layer_spec.rb lib/toy/llm/archs/llama_arch_cuda.rb lib/toy/llm/blocks/gdn_block.rb lib/toy/llm/blocks/transformer_block_cuda.rb lib/toy/llm/engine/llama_seq_engine_cuda.rb lib/toy/llm/labels.rb lib/toy/llm/primitives/gdn.rb lib/toy/llm/primitives/gqa_cuda.rb lib/toy/llm/primitives/muon.rb lib/toy/llm/primitives/rms_norm_cuda.rb lib/toy/llm/primitives/rope_cuda.rb lib/toy/llm/primitives/situ_glu_cuda.rb lib/toy/llm/primitives/swiglu_cuda.rb lib/toy/llm/recipes/franken_from_scratch_cuda.rb lib/toy/models/toy_smollm2.rb lib/toy/models/transformer.rb lib/toy/version.rb | libexec
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
.PHONY: gate-franken-llama-cuda
gate-franken-llama-cuda: libexec/toy-train-franken-llama-cuda libexec/toy-train-cuda
	ruby prep/franken_llama_cuda_gate.rb

# toy#134 — the CUDA twin of the spec-callable MoE runner (`toy train
# franken-moe --device cuda`): the F9-r2 critical path. Force-link keeps
# the CUDA backend registration alive (same as every CUDA trainer).
libexec/toy-train-franken-moe-cli-cuda: lib/toy/run/train_franken_moe_cli_cuda.rb lib/toy/run/franken_moe_parts_cuda.rb \
		lib/toy.rb lib/toy/ffi/tinynn_cuda.rb lib/toy/io/json_builder.rb lib/toy/io/json.rb \
		lib/toy/io/toy_events.rb lib/toy/dev/toy_describe_flow.rb \
		lib/toy/llm/primitives/rms_norm_cuda.rb lib/toy/train/dfa_b.rb \
		lib/toy/llm/labels.rb lib/toy/io/toy_corpus_loader.rb \
		tinynn/libtinynn_ggml_cuda.a $(SPINEL_DEPS) \
		lib/toy/dev/toy_card.rb lib/toy/ffi/tinynn.rb lib/toy/llm/primitives/muon_cuda.rb lib/toy/llm/primitives/situ_glu_cuda.rb lib/toy/models/transformer.rb lib/toy/version.rb | libexec
	$(SPINEL) --cc='cc -Wl,-u,tnn_cuda_force_link' $< -o $@
.PHONY: gate-franken-moe-cuda
gate-franken-moe-cuda: libexec/toy-train-franken-moe-cli-cuda
	ruby prep/franken_moe_cuda_gate.rb
