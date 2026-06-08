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
require_relative "../tinynn"
require_relative "../toy"

# Models + configs (Llama/SmolLM2, ViT) and the GGUF→engine loader.
require_relative "models/toy_smollm2"
require_relative "models/toy_smollm2_loader"
require_relative "models/toy_vit"

# Engines — the L4 forward/train drivers. Each pulls its L1 primitives,
# L2 block, and L3 arch transitively.
require_relative "llm/engine/llama_seq_engine"
require_relative "llm/engine/vit_tiny_engine"
require_relative "llm/engine/gpt2_seq_engine"

# KV-cache decode engine (inference).
require_relative "../toy_smollm2_ffi_kv"

# Recipes — the named training/init compositions (from-scratch, LoRA,
# warm-start, ViT). Realize-path orchestration over the engines.
require_relative "llm/recipes/from_scratch"
require_relative "llm/recipes/lora"
require_relative "llm/recipes/warm_start"
require_relative "llm/recipes/vit_tiny"

# Inference I/O: GGUF model loading + the BPE tokenizer.
require_relative "io/gguf_load"
require_relative "io/tokenizer"

# AdamW hyper-parameter value object + training labels (used when driving
# build_training_step directly).
require_relative "llm/adamw"
require_relative "llm/labels"
