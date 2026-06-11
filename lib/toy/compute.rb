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
require_relative "models/toy_smollm2_loader"
require_relative "models/toy_vit"

# Engines — the L4 forward/train drivers. Each pulls its L1 primitives,
# L2 block, and L3 arch transitively.
require_relative "llm/engine/llama_seq_engine"
require_relative "llm/engine/vit_tiny_engine"
require_relative "llm/engine/gpt2_seq_engine"

# KV-cache decode engine (inference).
require_relative "../toy_smollm2_ffi_kv"

# Recipes — the named training/init compositions. Realize-path orchestration
# over the engines.
#
# NOTE: `recipes/lora` is deliberately NOT required here — a SECOND Spinel facet,
# distinct from spinel-dev#11 (which matz/spinel#1385 fixed + landed). Re-adding
# the require and rebuilding against the #11-fixed Spinel STILL fails the C
# compile (verified 2026-06-09): `@cfg.d_model.to_s` → `sp_poly_to_s`,
# `RoPE.new(...)` arg → poly, `cache.token_ids = token_ids` → IntArray←PolyArray.
#
# Root cause (spinel-dev#12, isolated to a 22-line repro): #11's fix back-
# propagates a callee's param type to an UNCALLED forwarder's param — but ONLY
# when the shared callee is a regular instance method. It does NOT cover a
# CONSTRUCTOR slot: `Klass.new(x)` from an uncalled method does not pin `x` from
# `Klass#initialize`'s param type. LoRA#realize! forwards `cfg` (uncalled in the
# compute surface) into realize_for_mmap, which builds the model via `.new`
# (RoPE.new etc.); `cfg` never gets pinned, stays poly, and UNIONS into the
# shared SmolLM2/RoPE constructor — which FromScratch's live realize_for_random_init
# path also feeds concretely — poisoning it program-wide (`sp_Cfg` vs `sp_Cfg *`,
# same shape as #11 but on the constructor argument). The untyped FFI leading
# param on realize_for_mmap is INCIDENTAL (a no-handle, cfg-leading variant of
# the repro fails identically); it only means an RBS sig can't independently
# rescue it. A consumer that actually trains LoRA requires `toy/llm/recipes/lora`
# itself (and calls realize! with a concrete cfg, which pins the type). Re-add
# tracked by toy#52 — blocked on spinel-dev#12 (constructor-slot back-prop), not #11.
require_relative "llm/recipe_options"
require_relative "llm/recipes/from_scratch"
require_relative "llm/recipes/warm_start"
require_relative "llm/recipes/vit_tiny"

# Inference I/O: GGUF model loading + the BPE tokenizer.
require_relative "io/gguf_load"
require_relative "io/tokenizer"

# AdamW hyper-parameter value object + training labels (used when driving
# build_training_step directly) + the validating per-step batch wrapper.
require_relative "llm/adamw"
require_relative "llm/labels"
require_relative "llm/training_batch"

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
  end
end
