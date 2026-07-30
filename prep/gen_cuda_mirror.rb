#!/usr/bin/env ruby
# Generate the GPU-backend mirror of a CPU FFI source file by
# mechanical substitution. The CPU file is the source of truth; the
# GPU files (CUDA, Metal) are build artifacts (committed so reviewers
# see both).
#
# Phase 0.6 — graph-inlining refactor. The Spinel ivar-collision that
# made an `include Module` approach impractical doesn't matter here:
# we still get distinct compiled classes per backend (Spinel-friendly),
# but one human-readable Ruby source file.
#
# Usage:
#   prep/gen_cuda_mirror.rb [file1 file2 ...]
#       -> regenerate both CUDA and Metal mirrors for each input
#          (default: every file in MIRRORABLE below).
#
#   prep/gen_cuda_mirror.rb --backend cuda|metal [file ...]
#       -> regenerate only the named backend's mirrors.
#
#   prep/gen_cuda_mirror.rb --verify [file ...]
#       -> compare each on-disk mirror against what we'd generate;
#          exit non-zero on drift. Use in CI / `make verify-mirrors`.
#
# === Sentinel syntax ===
#
# To carry a CPU-only block over the mirror line, wrap it:
#
#     # CUDA-MIRROR-SKIP-BEGIN: <reason>
#     <CPU-only code, any length>
#     # CUDA-MIRROR-SKIP-END
#
# The block is dropped from the GPU mirrors. Optionally precede the
# skipped lines with one or more STUB lines whose contents replace
# the block in the GPU mirror:
#
#     # CUDA-MIRROR-SKIP-BEGIN: trace-tap is CPU-only diagnostic
#     # CUDA-MIRROR-STUB: def trace_tap(_name, t)
#     # CUDA-MIRROR-STUB:   t
#     # CUDA-MIRROR-STUB: end
#     def trace_tap(name_, t)
#       if @trace_on
#         ...full CPU implementation...
#       end
#       t
#     end
#     # CUDA-MIRROR-SKIP-END
#
# In the GPU mirror, the three STUB lines become uncommented Ruby
# code in place of the original method body. STUB lines must appear
# *before* the first non-sentinel non-comment line of the block.
#
# Sentinels are spelled `CUDA-MIRROR-*` for backwards compatibility;
# they apply uniformly to every backend.

require "fileutils"

MIRRORABLE = [
  "lib/toy/llm/primitives/rms_norm.rb", # P2.x: L1 RMSNorm primitive (calls TinyNN.*)
  "lib/toy/llm/primitives/rope.rb",     # P2.3: L1 RoPE primitive (calls TinyNN.*)
  "lib/toy/llm/primitives/swiglu.rb",   # P2.4: L1 SwiGLU primitive (calls TinyNN.*)
  "lib/toy/llm/primitives/situ_glu.rb", # toy#136/K1: L1 SiTU-GLU primitive (calls TinyNN.*)
  "lib/toy/llm/primitives/kda.rb",      # toy#137/K2a: L1 KDA primitive (calls TinyNN.* + GDN.l2_train)
  "lib/toy/llm/primitives/muon.rb",     # toy#139: L1 Muon Newton-Schulz (calls TinyNN.*)
  "lib/toy/llm/primitives/gqa.rb",      # P2.x: L1 GQA primitive (calls TinyNN.*)
  "lib/toy/llm/blocks/transformer_block.rb", # P2.4: L2 transformer block (calls TinyNN.* + L1 primitives)
  "lib/toy/llm/archs/llama_arch.rb",    # P2.5: L3 Llama arch (orchestration lift; calls TinyNN.* + L2 block + L1 primitives)
  "lib/toy/llm/engine/llama_seq_engine.rb", # M3 engine (seq-mode forward + training + LoRA); was lib/llama_seq_forward_ffi.rb
  "lib/toy/llm/engine/gpt2_seq_engine.rb",  # GPT-2 from-scratch training engine (self-contained; calls TinyNN.*)
  "lib/toy/llm/recipes/from_scratch.rb", # toy#65 item 5: L4 from-scratch recipe (was a tracked hand-twin)
  "lib/toy/llm/recipes/franken_from_scratch.rb", # toy#109/#112: franken credit-assignment recipe
  "lib/toy/llm/recipes/lora.rb",         # toy#65 item 5: L4 LoRA recipe (was a tracked hand-twin)
  "lib/toy/llm/recipes/warm_start.rb",   # toy#65 item 5: L4 warm-start recipe (was a tracked hand-twin)
  "lib/toy/llm/engine/llama_kv_engine.rb", # KV-cache decode engine (was lib/toy_smollm2_ffi_kv.rb; toy#65 item 2)
  "lib/toy/llm/engine/gpt2_fwd_engine.rb", # GPT-2 full-forward FFI (was lib/gpt2_ffi.rb; toy#68)
  "lib/toy/llm/engine/gpt2_kv_engine.rb",  # GPT-2 KV-cache decode (was lib/gpt2_ffi_kv.rb; toy#68)
  "examples/legacy/06_train_from_scratch.rb",  # #152: from-scratch training entry (CPU/CUDA)
  "prep/smokes/smoke_projection_lens.rb", # P2.6 CUDA gate: realize_for_random_init forward on GPU
]

# Per-backend codegen parameters.
BACKENDS = {
  cuda:  { suffix: "Cuda",  session: 1, tinynn: "tinynn_cuda",  label: "CUDA"  },
  metal: { suffix: "Metal", session: 2, tinynn: "tinynn_metal", label: "Metal" },
}

# Build the substitution table for a given (file, backend). Each
# input file gets a per-file table because class names differ; the
# backend supplies the suffix + session + module rename. Order
# matters: longer-suffix variants must come first so we don't double-
# rename (e.g. don't run `TinyNN → TinyNNCuda` after we've already
# produced `TinyNNCuda`).
def subs_for(cpu_path, backend)
  cfg     = BACKENDS.fetch(backend)
  suffix  = cfg[:suffix]
  module_ = "TinyNN" + suffix
  session = cfg[:session]
  tinynn  = cfg[:tinynn]
  label   = cfg[:label]

  common_module_tail = [
    [/\bTinyNN\.\b/,                   module_ + "."],
    [/\bTinyNN\b/,                     module_],
    [/tnn_session_new\(0\)/,           "tnn_session_new(" + session.to_s + ")"],
    [/^require_relative "toy\/ffi\/tinynn"$/, 'require_relative "toy/ffi/' + tinynn + '"'],
  ]

  case cpu_path
  when "lib/toy/llm/primitives/rms_norm.rb"
    # P2.x L1 primitive. No class/module renames — Toy::LLM::Primitives::
    # RMSNorm collides with nothing in the backend namespace.
    # common_module_tail handles TinyNN. -> TinyNN<Backend>. (the only
    # backend-sensitive content). The require_relative "tinynn" rewrite
    # is harmless (the file has no such require by design).
    [
      [/^# lib\/toy\/llm\/primitives\/rms_norm\.rb.*$/,
       "# lib/toy/llm/primitives/rms_norm_#{backend}.rb — #{label} mirror of lib/toy/llm/primitives/rms_norm.rb.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand;\n" +
       "# edit the CPU source and re-run the generator. Same L1 contract\n" +
       "# on the GPU backend via #{module_}."],
    ] + common_module_tail

  when "lib/toy/llm/primitives/rope.rb"
    # P2.3 L1 primitive. No class/module renames — Toy::LLM::Primitives::
    # RoPE and its Cfg collide with nothing in the backend namespace.
    # common_module_tail handles TinyNN. -> TinyNN<Backend>. (the only
    # backend-sensitive content). The require_relative "tinynn" rewrite
    # is harmless (the file has no such require by design).
    [
      [/^# lib\/toy\/llm\/primitives\/rope\.rb.*$/,
       "# lib/toy/llm/primitives/rope_#{backend}.rb — #{label} mirror of lib/toy/llm/primitives/rope.rb.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand;\n" +
       "# edit the CPU source and re-run the generator. Same L1 contract\n" +
       "# on the GPU backend via #{module_}."],
    ] + common_module_tail

  when "lib/toy/llm/primitives/swiglu.rb"
    # P2.4 L1 primitive. No class/module renames — Toy::LLM::Primitives::
    # SwiGLU collides with nothing in the backend namespace.
    # common_module_tail handles TinyNN. -> TinyNN<Backend>. (the only
    # backend-sensitive content). The require_relative "tinynn" rewrite
    # is harmless (the file has no such require by design).
    [
      [/^# lib\/toy\/llm\/primitives\/swiglu\.rb.*$/,
       "# lib/toy/llm/primitives/swiglu_#{backend}.rb — #{label} mirror of lib/toy/llm/primitives/swiglu.rb.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand;\n" +
       "# edit the CPU source and re-run the generator. Same L1 contract\n" +
       "# on the GPU backend via #{module_}."],
    ] + common_module_tail

  when "lib/toy/llm/primitives/situ_glu.rb"
    # toy#136 K1 primitive — same shape as the swiglu rules.
    [
      [/^# lib\/toy\/llm\/primitives\/situ_glu\.rb.*$/,
       "# lib/toy/llm/primitives/situ_glu_#{backend}.rb — #{label} mirror of lib/toy/llm/primitives/situ_glu.rb.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand;\n" +
       "# edit the CPU source and re-run the generator. Same L1 contract\n" +
       "# on the GPU backend via #{module_}."],
    ] + common_module_tail

  when "lib/toy/llm/primitives/kda.rb"
    # toy#137 K2a primitive. It reaches Primitives::GDN.l2_train, so the
    # mirror must point at the mirrored GDN (handled by the shared
    # gdn-require rewrite in common_module_tail's sibling rules below);
    # module names collide with nothing per-backend.
    [
      [/^# lib\/toy\/llm\/primitives\/kda\.rb.*$/,
       "# lib/toy/llm/primitives/kda_#{backend}.rb — #{label} mirror of lib/toy/llm/primitives/kda.rb.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand;\n" +
       "# edit the CPU source and re-run the generator. Same L1 contract\n" +
       "# on the GPU backend via #{module_}."],
      [/Toy::LLM::Primitives::GDN\./, "Toy::LLM::Primitives::GDN."],
    ] + common_module_tail

  when "lib/toy/llm/primitives/muon.rb"
    # toy#139 primitive — same shape as the kda/swiglu rules.
    [
      [/^# lib\/toy\/llm\/primitives\/muon\.rb.*$/,
       "# lib/toy/llm/primitives/muon_#{backend}.rb — #{label} mirror of lib/toy/llm/primitives/muon.rb.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand;\n" +
       "# edit the CPU source and re-run the generator. Same L1 contract\n" +
       "# on the GPU backend via #{module_}."],
    ] + common_module_tail

  when "lib/toy/llm/primitives/gqa.rb"
    # P2.x L1 primitive. No class/module renames — Toy::LLM::Primitives::
    # GQA collides with nothing in the backend namespace.
    # common_module_tail handles TinyNN. -> TinyNN<Backend>. (the only
    # backend-sensitive content). The require_relative "tinynn" rewrite
    # is harmless (the file has no such require by design).
    [
      [/^# lib\/toy\/llm\/primitives\/gqa\.rb.*$/,
       "# lib/toy/llm/primitives/gqa_#{backend}.rb — #{label} mirror of lib/toy/llm/primitives/gqa.rb.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand;\n" +
       "# edit the CPU source and re-run the generator. Same L1 contract\n" +
       "# on the GPU backend via #{module_}."],
    ] + common_module_tail

  when "lib/toy/llm/blocks/transformer_block.rb"
    # P2.4 L2 block. No class/module renames — Toy::LLM::Blocks::
    # TransformerBlock and TransformerBlockCtx collide with nothing in
    # the backend namespace, and the block file references NEITHER
    # LlamaSeqBlockFFI nor LlamaSeqForwardFFICache as literals (the old
    # `blk` arg is gone; the class is self-named). common_module_tail
    # ALONE handles TinyNN. -> TinyNN<Backend>. (the only backend-
    # sensitive content). The require_relative "tinynn" rewrite is
    # harmless (the file has no such require by design).
    [
      [/^# lib\/toy\/llm\/blocks\/transformer_block\.rb.*$/,
       "# lib/toy/llm/blocks/transformer_block_#{backend}.rb — #{label} mirror of lib/toy/llm/blocks/transformer_block.rb.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand;\n" +
       "# edit the CPU source and re-run the generator. Same L2 contract\n" +
       "# on the GPU backend via #{module_}."],
    ] + common_module_tail

  when "lib/toy/llm/archs/llama_arch.rb"
    # P2.5 L3 arch. No class/module renames — Toy::LLM::Archs::LlamaArch
    # and LlamaArchForwardOut collide with nothing in the backend
    # namespace (the only Llama generator regex is the \b-anchored
    # \bLlamaSeqForwardFFICache\b, which does NOT match LlamaArch).
    # common_module_tail ALONE handles TinyNN. -> TinyNN<Backend>. (the
    # only backend-sensitive content). The require_relative "tinynn"
    # rewrite is harmless (the file has no such require by design).
    [
      [/^# lib\/toy\/llm\/archs\/llama_arch\.rb.*$/,
       "# lib/toy/llm/archs/llama_arch_#{backend}.rb — #{label} mirror of lib/toy/llm/archs/llama_arch.rb.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand;\n" +
       "# edit the CPU source and re-run the generator. Same L3 contract\n" +
       "# on the GPU backend via #{module_}."],
    ] + common_module_tail

  when "lib/toy/llm/engine/llama_seq_engine.rb"
    # P2-finish — the monolith retired into the tree as the L3-adjacent
    # engine (class Toy::LLM::Engine::LlamaSeqEngine). Was
    # lib/llama_seq_forward_ffi.rb / LlamaSeqForwardFFICache. The engine
    # lives at lib/toy/llm/engine/, so its mirrored require_relatives are
    # SIBLING-relative ("../primitives/...", "../blocks/...", "../archs/...").
    [
      [/^# lib\/toy\/llm\/engine\/llama_seq_engine\.rb.*$/,
       "# lib/toy/llm/engine/llama_seq_engine_#{backend}.rb — #{label} mirror of lib/toy/llm/engine/llama_seq_engine.rb.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand;\n" +
       "# edit the CPU source and re-run the generator. The CPU file's\n" +
       "# header explains the architecture; this mirror keeps the same\n" +
       "# contract on the GPU backend via #{module_}."],
      # Point the mirror at the mirrored primitive so the GPU engine
      # loads rope_<backend>.rb (TinyNN<Backend>.*), not the CPU rope.rb.
      [/^require_relative "\.\.\/primitives\/rms_norm"$/,
       'require_relative "../primitives/rms_norm_' + backend.to_s + '"'],
      [/^require_relative "\.\.\/primitives\/rope"$/,
       'require_relative "../primitives/rope_' + backend.to_s + '"'],
      [/^require_relative "\.\.\/primitives\/swiglu"$/,
       'require_relative "../primitives/swiglu_' + backend.to_s + '"'],
      [/^require_relative "\.\.\/primitives\/situ_glu"$/,
       'require_relative "../primitives/situ_glu_' + backend.to_s + '"'],
      [/^require_relative "\.\.\/primitives\/gqa"$/,
       'require_relative "../primitives/gqa_' + backend.to_s + '"'],
      # P2.4 — point the mirror at the mirrored L2 block so the GPU
      # engine loads transformer_block_<backend>.rb (TinyNN<Backend>.*),
      # not the CPU block.
      [/^require_relative "\.\.\/blocks\/transformer_block"$/,
       'require_relative "../blocks/transformer_block_' + backend.to_s + '"'],
      # P2.5 — point the mirror at the mirrored L3 arch so the GPU
      # engine loads llama_arch_<backend>.rb (TinyNN<Backend>.*), not
      # the CPU arch (else the GPU engine fail-quietly loads the CPU
      # arch and its TinyNN.* calls would hit the CPU backend).
      [/^require_relative "\.\.\/archs\/llama_arch"$/,
       'require_relative "../archs/llama_arch_' + backend.to_s + '"'],
      # The engine lives at lib/toy/llm/engine/ and TinyNN at lib/toy/ffi/
      # (toy#65 item 4), so its TinyNN require is "../../ffi/tinynn".
      # common_module_tail's rewrite only matches the lib-root form
      # ("toy/ffi/tinynn"), so rewrite the depth-prefixed path to the backend
      # TinyNN here — else the GPU mirror loads CPU TinyNN and every
      # TinyNN<Backend>.* fails to resolve.
      [/^require_relative "\.\.\/\.\.\/ffi\/tinynn"$/,
       'require_relative "../../ffi/tinynn_' + backend.to_s + '"'],
      [/\bLlamaSeqEngine\b/, "LlamaSeqEngine" + suffix],
    ] + common_module_tail

  when "lib/toy/llm/engine/gpt2_seq_engine.rb"
    # GPT-2 from-scratch training engine (class Toy::LLM::Engine::GPT2SeqEngine).
    # Self-contained: no L1/L2/L3 require_relatives and no `tinynn` require (the
    # runner loads the backend's TinyNN), so the mirror is just the class rename
    # + common_module_tail (TinyNN -> TinyNN<Backend>, session_new(0) -> N). The
    # LayerNorm/GELU backward ops are CPU-only; on a GPU session the scheduler
    # falls back to the CPU backend for them (correct, slower).
    [
      [/^# lib\/toy\/llm\/engine\/gpt2_seq_engine\.rb.*$/,
       "# lib/toy/llm/engine/gpt2_seq_engine_#{backend}.rb — #{label} mirror of lib/toy/llm/engine/gpt2_seq_engine.rb.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand;\n" +
       "# edit the CPU source and re-run the generator. Same contract on the\n" +
       "# GPU backend via #{module_} (LayerNorm/GELU backward fall back to CPU)."],
      [/\bGPT2SeqEngine\b/, "GPT2SeqEngine" + suffix],
    ] + common_module_tail

  when "lib/toy/llm/recipes/from_scratch.rb"
    # toy#65 item 5 — L4 recipe. Backend-sensitive content: the recipe
    # class name, the engine class it instantiates, and the TinyNN.*
    # calls inlined in step! (common_module_tail). The only require is
    # "../recipe_options" (backend-neutral pure Ruby) — no rewrite.
    [
      [/^# lib\/toy\/llm\/recipes\/from_scratch\.rb.*$/,
       "# lib/toy/llm/recipes/from_scratch_#{backend}.rb — #{label} mirror of lib/toy/llm/recipes/from_scratch.rb.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand;\n" +
       "# edit the CPU source and re-run the generator. Same L4 contract\n" +
       "# on the GPU backend via #{module_}."],
      [/\bLlamaSeqEngine\b/, "LlamaSeqEngine" + suffix],
      [/\bFromScratch\b/,    "FromScratch"    + suffix],
    ] + common_module_tail

  when "lib/toy/llm/recipes/franken_from_scratch.rb"
    # toy#109 — franken recipe: same shape as from_scratch (engine class +
    # recipe class + inlined TinyNN.* in step!). Requires recipe_options +
    # train/dfa_b — both backend-neutral pure Ruby, no rewrite.
    [
      [/^# lib\/toy\/llm\/recipes\/franken_from_scratch\.rb.*$/,
       "# lib/toy/llm/recipes/franken_from_scratch_#{backend}.rb — #{label} mirror.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand;\n" +
       "# edit the CPU source and re-run the generator."],
      [/\bLlamaSeqEngine\b/,    "LlamaSeqEngine"    + suffix],
      [/\bFrankenFromScratch\b/, "FrankenFromScratch" + suffix],
    ] + common_module_tail

  when "lib/toy/llm/recipes/lora.rb"
    # toy#65 item 5 — L4 recipe. Same shape as from_scratch: rename the
    # recipe class + the engine class; common_module_tail handles the
    # inlined TinyNN.* calls. \bLoRA\b does NOT match LoRAConfig/LoRA-Q
    # identifiers-with-suffix (word boundary), so only the bare class
    # name (and bare prose mentions, harmlessly) is renamed.
    [
      [/^# lib\/toy\/llm\/recipes\/lora\.rb.*$/,
       "# lib/toy/llm/recipes/lora_#{backend}.rb — #{label} mirror of lib/toy/llm/recipes/lora.rb.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand;\n" +
       "# edit the CPU source and re-run the generator. Same L4 contract\n" +
       "# on the GPU backend via #{module_}."],
      [/\bLlamaSeqEngine\b/, "LlamaSeqEngine" + suffix],
      [/\bLoRA\b/,           "LoRA"           + suffix],
    ] + common_module_tail

  when "lib/toy/llm/recipes/warm_start.rb"
    # toy#65 item 5 — L4 recipe. Same shape as from_scratch.
    [
      [/^# lib\/toy\/llm\/recipes\/warm_start\.rb.*$/,
       "# lib/toy/llm/recipes/warm_start_#{backend}.rb — #{label} mirror of lib/toy/llm/recipes/warm_start.rb.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand;\n" +
       "# edit the CPU source and re-run the generator. Same L4 contract\n" +
       "# on the GPU backend via #{module_}."],
      [/\bLlamaSeqEngine\b/, "LlamaSeqEngine" + suffix],
      [/\bWarmStart\b/,      "WarmStart"      + suffix],
    ] + common_module_tail

  when "lib/toy/llm/engine/llama_kv_engine.rb"
    # toy#65 item 2 — was lib/toy_smollm2_ffi_kv.rb. Class/module names
    # kept stable (SmolLM2KV*) in the path move; only the location is new.
    [
      [/^# lib\/toy\/llm\/engine\/llama_kv_engine\.rb.*$/,
       "# lib/toy/llm/engine/llama_kv_engine_#{backend}.rb — #{label} mirror of lib/toy/llm/engine/llama_kv_engine.rb.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand;\n" +
       "# edit the CPU source and re-run the generator. The CPU file's\n" +
       "# header explains the architecture; this mirror keeps the same\n" +
       "# contract on the GPU backend via #{module_}."],
      # Engine-depth TinyNN require (same rationale as llama_seq_engine):
      # common_module_tail only matches the lib-root "toy/ffi/tinynn" form.
      [/^require_relative "\.\.\/\.\.\/ffi\/tinynn"$/,
       'require_relative "../../ffi/tinynn_' + backend.to_s + '"'],
      # Class names — longer ones first (KV is a substring of KVBlock).
      [/\bSmolLM2KVStepResult\b/,    "SmolLM2KVStepResult" + suffix],
      [/\bSmolLM2KVFFICache\b/,      "SmolLM2KVFFICache"   + suffix],
      [/\bSmolLM2KVBlockFFI\b/,      "SmolLM2KVBlockFFI"   + suffix],
      # Module — the ".decode_step" wrapper.
      [/\bSmolLM2KV\.decode_step\b/, "SmolLM2KV" + suffix + ".decode_step"],
      [/\bSmolLM2KV\.upload_from\b/, "SmolLM2KV" + suffix + ".upload_from"],
      [/^module SmolLM2KV$/,         "module SmolLM2KV" + suffix],
    ] + common_module_tail

  when "lib/toy/llm/engine/gpt2_fwd_engine.rb"
    # toy#68 — was lib/gpt2_ffi.rb. Class/module names kept stable
    # (GPT2FullForwardFFICache, GPT2FFI) in the path move.
    [
      [/^# lib\/toy\/llm\/engine\/gpt2_fwd_engine\.rb.*$/,
       "# lib/toy/llm/engine/gpt2_fwd_engine_#{backend}.rb — #{label} mirror of lib/toy/llm/engine/gpt2_fwd_engine.rb.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand."],
      # Engine-depth TinyNN require (same rationale as llama_kv_engine):
      # common_module_tail only matches the lib-root "toy/ffi/tinynn" form.
      [/^require_relative "\.\.\/\.\.\/ffi\/tinynn"$/,
       'require_relative "../../ffi/tinynn_' + backend.to_s + '"'],
      [/\bGPT2FullForwardFFICache\b/, "GPT2FullForwardFFICache" + suffix],
      [/\bGPT2BlockFFI\b/,            "GPT2BlockFFI"            + suffix],
      [/\bGPT2FFI\b/,                 "GPT2FFI"                 + suffix],
    ] + common_module_tail

  when "lib/toy/llm/engine/gpt2_kv_engine.rb"
    # toy#68 — was lib/gpt2_ffi_kv.rb. Class/module names kept stable
    # (GPT2KVFFICache, GPT2KV) in the path move.
    [
      [/^# lib\/toy\/llm\/engine\/gpt2_kv_engine\.rb.*$/,
       "# lib/toy/llm/engine/gpt2_kv_engine_#{backend}.rb — #{label} mirror of lib/toy/llm/engine/gpt2_kv_engine.rb.\n" +
       "#\n" +
       "# AUTO-GENERATED by prep/gen_cuda_mirror.rb. Do not edit by hand."],
      [/^require_relative "\.\.\/\.\.\/ffi\/tinynn"$/,
       'require_relative "../../ffi/tinynn_' + backend.to_s + '"'],
      [/\bGPT2KVStepResult\b/,        "GPT2KVStepResult" + suffix],
      [/\bGPT2KVFFICache\b/,          "GPT2KVFFICache"   + suffix],
      [/\bGPT2KVBlockFFI\b/,          "GPT2KVBlockFFI"   + suffix],
      [/\bGPT2KV\.decode_step\b/,     "GPT2KV" + suffix + ".decode_step"],
      [/\bGPT2KV\.upload_from\b/,     "GPT2KV" + suffix + ".upload_from"],
      [/^module GPT2KV$/,             "module GPT2KV" + suffix],
    ] + common_module_tail

  when "examples/legacy/06_train_from_scratch.rb"
    # #152 — CUDA mirror of the from-scratch training entry. Rewrites
    # the require to the GPU-side engine mirror, renames LlamaSeqEngine
    # to its backend variant, and swaps every TinyNN. call to
    # TinyNN<Suffix>. (handled by common_module_tail). Lives two levels
    # below repo root (examples/legacy/) — hence the ../../ depth.
    [
      [/^require_relative "..\/..\/lib\/toy\/llm\/engine\/llama_seq_engine"$/,
       'require_relative "../../lib/toy/llm/engine/llama_seq_engine_' + backend.to_s + '"'],
      [/\bLlamaSeqEngine\b/, "LlamaSeqEngine" + suffix],
    ] + common_module_tail

  when "prep/smokes/smoke_projection_lens.rb"
    # P2.6 CUDA gate — mirror of the projection-lens smoke so the
    # realize_for_random_init forward can be parity-gated on the GPU
    # backend (CUDA self-consistency, not CPU bit-equality). Same
    # rewrites as the 06 example: require the GPU engine mirror,
    # rename the engine class, TinyNN. -> TinyNN<Suffix>. via the tail.
    # (Smokes live two levels below repo root — prep/smokes/ — hence
    # the ../../ require depth, unlike the examples/ entry above.)
    [
      [/^require_relative "..\/..\/lib\/toy\/llm\/engine\/llama_seq_engine"$/,
       'require_relative "../../lib/toy/llm/engine/llama_seq_engine_' + backend.to_s + '"'],
      [/\bLlamaSeqEngine\b/, "LlamaSeqEngine" + suffix],
    ] + common_module_tail

  else
    nil
  end
end

SKIP_OPEN   = /^\s*# CUDA-MIRROR-SKIP-BEGIN(?::.*)?$/
SKIP_CLOSE  = /^\s*# CUDA-MIRROR-SKIP-END\s*$/
STUB_LINE   = /^\s*# CUDA-MIRROR-STUB:\s?(.*)$/

def generate(cpu_path, subs)
  src = File.read(cpu_path)
  out_lines = []
  i = 0
  lines = src.lines
  while i < lines.length
    line = lines[i]
    if line =~ SKIP_OPEN
      # Scan the entire SKIP block, collect every STUB line (they may
      # appear anywhere inside, intermixed with comments and code),
      # then advance past SKIP-END.
      stub_lines = []
      j = i + 1
      while j < lines.length && lines[j] !~ SKIP_CLOSE
        if lines[j] =~ STUB_LINE
          stub_lines << $1 + "\n"
        end
        j += 1
      end
      stub_lines.each { |s| out_lines << s }
      i = j + 1  # past SKIP-END
      next
    end
    transformed = line.dup
    subs.each { |re, replacement| transformed.gsub!(re, replacement) }
    out_lines << transformed
    i += 1
  end
  out_lines.join
end

def derive_mirror_path(cpu_path, backend)
  # lib/X_ffi.rb → lib/X_ffi_<backend>.rb (original use case)
  # examples/NN_name.rb → examples/NN_name_<backend>.rb (#152)
  # lib/toy/llm/primitives/<name>.rb → ..._<backend>.rb (P2.3 L1 primitives)
  unless cpu_path =~ /_ffi(_[a-z]+)?\.rb$/ ||
         cpu_path =~ %r{^examples/(legacy/)?\d+_[a-z0-9_]+\.rb$} ||
         cpu_path =~ %r{^prep/smokes/smoke_[a-z0-9_]+\.rb$} ||
         cpu_path =~ %r{^lib/toy/llm/primitives/[a-z0-9_]+\.rb$} ||
         cpu_path =~ %r{^lib/toy/llm/blocks/[a-z0-9_]+\.rb$} ||
         cpu_path =~ %r{^lib/toy/llm/archs/[a-z0-9_]+\.rb$} ||
         cpu_path =~ %r{^lib/toy/llm/engine/[a-z0-9_]+\.rb$} ||
         cpu_path =~ %r{^lib/toy/llm/recipes/[a-z0-9_]+\.rb$}
    raise "unrecognised mirror path: #{cpu_path}"
  end
  cpu_path.sub(/\.rb$/, "_#{backend}.rb")
end

verify_only = ARGV.delete("--verify")
backend_arg = nil
if (idx = ARGV.index("--backend"))
  backend_arg = ARGV[idx + 1]&.to_sym
  ARGV.delete_at(idx + 1)
  ARGV.delete_at(idx)
  unless BACKENDS.key?(backend_arg)
    $stderr.puts "[gen_cuda_mirror] unknown backend: #{backend_arg.inspect} (want one of #{BACKENDS.keys.inspect})"
    exit 1
  end
end
inputs   = ARGV.empty? ? MIRRORABLE : ARGV
backends = backend_arg ? [backend_arg] : BACKENDS.keys

drift = false
inputs.each do |cpu_path|
  unless File.exist?(cpu_path)
    $stderr.puts "[gen_cuda_mirror] missing: #{cpu_path}"
    drift = true
    next
  end
  backends.each do |backend|
    subs = subs_for(cpu_path, backend)
    unless subs
      $stderr.puts "[gen_cuda_mirror] no substitution table for #{cpu_path} (backend #{backend})"
      drift = true
      next
    end
    mirror_path = derive_mirror_path(cpu_path, backend)
    generated   = generate(cpu_path, subs)
    if verify_only
      actual = File.exist?(mirror_path) ? File.read(mirror_path) : ""
      if actual == generated
        puts "[gen_cuda_mirror] ok: #{mirror_path} matches generator"
      else
        puts "[gen_cuda_mirror] DRIFT: #{mirror_path} does not match generator output"
        puts "                  edit #{cpu_path} (the source of truth) and re-run"
        puts "                  prep/gen_cuda_mirror.rb #{cpu_path}"
        drift = true
      end
    else
      File.write(mirror_path, generated)
      puts "[gen_cuda_mirror] wrote #{mirror_path} (#{generated.lines.size} lines)"
    end
  end
end

exit(drift ? 1 : 0) if verify_only
