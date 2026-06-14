# lib/toy/compute.rb — the full Toy compute surface, in one require.
#
# WHY THIS EXISTS (toy#42). A library consumer composes models out of Toy's
# engines / primitives / recipes / loaders. Before this file they had to
# hand-`require_relative` each one (the generated `vendor/spinel/deps.rb` only
# pulls the top-level `toy.rb`, which is the MRI-safe CLI surface — Mat + Card +
# version, no FFI). This file is the SPINEL-ONLY compute entrypoint: require it
# once and the whole public composition API is loaded.
#
#   require_relative "vendor/spinel/toy/lib/toy/compute"   # one require
#   cfg    = Toy::SmolLM2Config.new(627, 64, 4, 4, 128, 2, 32, 10000.0, 1.0e-5)
#   engine = Toy::LLM::Engine::LlamaSeqEngine.new
#   engine.realize_for_random_init(cfg, 32, 1, 0, false, false, 0, 1.0)
#   # … build_training_step / KV decode / GGUF load …
#
# MRI-SAFETY BOUNDARY. Do NOT require this from `toy.rb` or anything under
# `lib/toy/core/` (the CRuby CLI): it pulls `tinynn` (FFI `ffi_lib`s) and the
# engines, which only load under a Spinel build. `toy.rb` must stay loadable by
# plain MRI so `bin/toy` can require it. This file is for Spinel consumers only.
#
# MINIMAL-SURFACE NOTE. Spinel has no tree-shaking, so this pulls the COMMON
# composition surface (all three engines + recipes + loaders). A consumer that
# only ever builds, say, a LlamaSeqEngine can still require the individual files
# instead to compile less; this is the convenience "everything" entry. The
# backend mirrors (`*_cuda` / `*_metal`) are NOT required here — a consumer
# selects its backend by requiring the matching engine variant.

# FFI compute layer (Mat, sessions, tnn_* ops) + the MRI-safe sugar.
require_relative "ffi/tinynn"
require_relative "../toy"

# Models + configs (Llama/SmolLM2, ViT) and the GGUF→engine loader.
require_relative "models/toy_smollm2"
require_relative "io/loaders/toy_smollm2_loader"
require_relative "models/toy_vit"

# Engines — the L4 forward/train drivers. Each pulls its L1 primitives,
# L2 block, and L3 arch transitively.
require_relative "llm/engine/llama_seq_engine"
require_relative "llm/engine/vit_tiny_engine"
require_relative "llm/engine/gpt2_seq_engine"

# KV-cache decode engine (inference).
require_relative "llm/engine/llama_kv_engine"

# Recipes — the named training/init compositions. Realize-path orchestration
# over the engines.
#
# HISTORY (toy#52, closed 2026-06-11): `recipes/lora` was excluded from this
# surface for three days — an UNCALLED LoRA#realize! forwarding `cfg` into a
# constructor slot poisoned the shared SmolLM2/RoPE ctor to poly program-wide
# (spinel-dev#12, the constructor-slot sibling of spinel-dev#11). The Phase-A
# RecipeOptions reshape of realize! (toy#64) dissolved the poison path: the
# re-add compiles + trains byte-identically even on pre-#12-fix Spinel revs
# (probed at 08a189c, 2026-06-11, toy#69). The compute-surface smoke keeps an
# instantiated-but-UNCALLED LoRA as a tripwire so any regression re-breaks
# that gate's C compile loudly.
require_relative "llm/recipe_options"
require_relative "llm/recipes/from_scratch"
require_relative "llm/recipes/warm_start"
require_relative "llm/recipes/lora"
require_relative "llm/recipes/vit_tiny"

# Inference I/O: GGUF model loading + the BPE tokenizer.
require_relative "io/gguf_load"
require_relative "io/tokenizer"

# Training I/O + schedule (toy#73 item 2): the streaming token-corpus
# loader, the ViT image/label loader, and the cosine LR schedule. The
# loaders are plain file I/O through C helpers in libtinynn_ggml.a
# (backend-agnostic; every backend links it) and ToyLR is pure Ruby, so
# all three are SHARED across the CPU/CUDA/Metal entries — no mirrors.
require_relative "io/toy_corpus_loader"
require_relative "io/toy_image_loader"
require_relative "train/toy_lr_schedule"

# AdamW hyper-parameter value object + training labels (used when driving
# build_training_step directly) + the validating per-step batch wrapper.
require_relative "llm/adamw"
require_relative "llm/labels"
require_relative "llm/training_batch"
require_relative "llm/classify_batch"

# ── Toy::Device — the device-agnostic construction seam (toy#64 item 8) ──
#
# DEVICE IS CHOSEN AT COMPILE TIME by which compute entry you require:
#   lib/toy/compute.rb        → CPU     (this file)
#   lib/toy/compute_cuda.rb   → CUDA    (GB10/sm_121 etc.)
#   lib/toy/compute_metal.rb  → Metal   (macOS only)
# Spinel does NOT resolve a require_relative inside a conditional — an
# `if ENV[...]`-guarded require is SILENTLY compiled to 0 (probed on
# a699cf9: "cannot resolve call to 'require_relative'... emitting 0"),
# so a TOY_DEVICE-switching single entry is impossible; per-device
# entry files are the mechanism.
#
# User source stays DEVICE-AGNOSTIC by constructing through these
# factories instead of naming the backend classes: each entry defines
# the SAME Toy::Device surface returning its backend's types, so the
# same experiment body compiles against any entry (the `toy new --lib`
# scaffold's build.sh loops devices over per-device main_<dev>.rb
# shims). Backends without a given engine/recipe variant simply do not
# ship it here: vit + warm-start recipes are CPU/CUDA-uneven (vit has
# no GPU engine; warm-start has no Metal recipe) — consumers that need
# them require the variant file directly.
module Toy
  module Device
    def self.name
      "cpu"
    end

    def self.llama_engine
      Toy::LLM::Engine::LlamaSeqEngine.new
    end

    def self.gpt2_engine
      Toy::LLM::Engine::GPT2SeqEngine.new
    end

    def self.from_scratch_recipe
      Toy::LLM::Recipes::FromScratch.new
    end

    def self.warm_start_recipe
      Toy::LLM::Recipes::WarmStart.new
    end

    # toy#90 — device teardown hook. CPU has no GPU-resource lifecycle to
    # drain, so this is a deliberate no-op; it exists only so a
    # device-agnostic experiment body can call Toy::Device.shutdown
    # portably before exit (the Metal entry's override is the one that
    # actually matters — see compute_metal.rb).
    def self.shutdown
      nil
    end
  end
end

# Run-bundle writer (toy#73 item 1): runs/<id>/events.jsonl in the toy/v1
# schema + the weights/ checkpoint-dir convention. SHARED (not mirrored):
# the events C symbols live in libtinynn_ggml.a, linked by every backend.
# Required AFTER the Toy::Device module above — run_start! stamps
# backend{kind: Toy::Device.name} (toy#73 A.3), and Spinel needs the
# constant defined before a file referencing it compiles.
require_relative "io/run_bundle"
