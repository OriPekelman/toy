# lib/toy/compute_cuda.rb — the Toy compute surface, CUDA entry.
#
# ── LOCKSTEP WARNING ── hand-maintained twin of lib/toy/compute.rb
# (NOT mechanically mirrorable: the CPU entry pulls vit + warm-start +
# loader files whose CUDA story is uneven — see the exclusions below —
# so prep/gen_cuda_mirror.rb's uniform substitution cannot produce this
# file). When compute.rb changes, update this file (and
# compute_metal.rb) in the same commit.
#
# DEVICE IS CHOSEN AT COMPILE TIME (toy#64 item 8): require THIS file
# instead of lib/toy/compute.rb and the same Toy::Device-driven
# experiment body compiles into a CUDA binary. Spinel cannot switch the
# require on ENV (a conditional require_relative silently compiles to
# 0 — probed on a699cf9), hence per-device entries. Link with the CUDA
# archives + force-link flag, exactly like every cuda target:
#   spinel --cc='cc -Wl,-u,tnn_cuda_force_link' <entry> -o <bin>
#   (deps: tinynn/libtinynn_ggml.a tinynn/libtinynn_ggml_cuda.a)
#
# SINGLE-TYPE BINARY discipline (landmine #16): TinyNNCuda is the
# compute module; the CPU TinyNN is still DEFINED (transformer.rb
# requires tinynn) for the checkpoint write/fuse seam, exactly like
# the hand-written cuda runners. Same MRI-safety boundary as
# compute.rb: Spinel-only, never require from the CRuby CLI.

# FFI compute layer: CUDA TinyNN + the MRI-safe sugar (Mat + Card).
require_relative "ffi/tinynn_cuda"
require_relative "../toy"

# Models + configs (shared — pure model/config code, no backend calls).
require_relative "models/toy_smollm2"
require_relative "io/loaders/toy_smollm2_loader"
require_relative "models/toy_vit"

# Engines — the CUDA mirrors (generated from the CPU files by
# prep/gen_cuda_mirror.rb; `make` pattern rules keep them fresh).
# NO ViT engine here: vit_tiny_engine has no CUDA mirror (absent from
# MIRRORABLE — vit training is CPU-only this arc).
require_relative "llm/engine/llama_seq_engine_cuda"
require_relative "llm/engine/gpt2_seq_engine_cuda"

# KV-cache decode engine (inference), CUDA mirror.
require_relative "llm/engine/llama_kv_engine_cuda"

# Recipes — the generated CUDA twins. lora_cuda included since toy#52
# closed (the spinel-dev#12 constructor-slot facet was dissolved by the
# toy#64 RecipeOptions reshape; see compute.rb's HISTORY note). Backend
# gap that remains: vit_tiny has no CUDA recipe (no CUDA vit engine).
require_relative "llm/recipe_options"
require_relative "llm/recipes/from_scratch_cuda"
require_relative "llm/recipes/warm_start_cuda"
require_relative "llm/recipes/lora_cuda"

# Inference I/O (shared): GGUF model loading + the BPE tokenizer.
require_relative "io/gguf_load"
require_relative "io/tokenizer"

# Training I/O + schedule (toy#73 item 2): corpus loader + image loader +
# cosine LR schedule. SHARED with the CPU entry (file I/O through C helpers
# in libtinynn_ggml.a + pure Ruby) — see compute.rb's note. The image
# loader rides along even though vit training stays CPU-only this arc
# (the loader is backend-free; the missing piece is the engine/recipe).
require_relative "io/toy_corpus_loader"
require_relative "io/toy_image_loader"
require_relative "train/toy_lr_schedule"

# AdamW + labels + the validating batch wrapper (shared, pure Ruby).
require_relative "llm/adamw"
require_relative "llm/labels"
require_relative "llm/training_batch"
require_relative "llm/classify_batch"

# Run-bundle writer (toy#73 item 1): runs/<id>/events.jsonl in the toy/v1
# schema + the weights/ checkpoint-dir convention. SHARED (not mirrored):
# the events C symbols live in libtinynn_ggml.a, linked by every backend
# (the CPU TinyNN module is defined in this binary too — see the
# checkpoint-seam note above).
require_relative "io/run_bundle"

# ── Toy::Device, CUDA arm — see the compute.rb doc block. Same
# surface, CUDA types; user source stays device-agnostic.
module Toy
  module Device
    def self.name
      "cuda"
    end

    def self.llama_engine
      Toy::LLM::Engine::LlamaSeqEngineCuda.new
    end

    def self.gpt2_engine
      Toy::LLM::Engine::GPT2SeqEngineCuda.new
    end

    def self.from_scratch_recipe
      Toy::LLM::Recipes::FromScratchCuda.new
    end

    def self.warm_start_recipe
      Toy::LLM::Recipes::WarmStartCuda.new
    end
  end
end
