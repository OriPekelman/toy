# lib/ — what loads where

Three loading worlds share this tree. Confusing them is the #1 way to
break the build, so know which one a file belongs to before requiring
anything:

1. **MRI-safe CLI tree — `lib/toy/core/`** (+ `lib/toy/core/cli/`).
   Plain CRuby, runs under the system Ruby with zero gems. It must
   NEVER require the FFI bridge or anything that transitively pulls it
   (the `ffi_lib` call only resolves inside Spinel-compiled binaries).
   The CLI shells out to the compiled runners in `libexec/` instead.

2. **Spinel compute tree — everything else under `lib/toy/`.**
   Compiled by Spinel into the `libexec/` runners, the examples and the
   demos. Layered as L1→L5 under `lib/toy/llm/` (primitives → blocks →
   archs → engines → recipes; see `docs/architecture.md`), with
   `models/` (Arch struct, model classes, pure-Ruby Mat in
   `models/transformer.rb`), `io/` (GGUF, tokenizer, loaders under
   `io/loaders/`), `train/`, `serve/`, `dev/`, and the entrypoints
   `compute*.rb` (library surface) and `run/` (runner mains —
   per-device twins there are hand-written and load-bearing, landmine
   #16; do not consolidate).

3. **FFI bridge — `lib/toy/ffi/tinynn{,_cuda,_metal}.rb`.**
   The TinyNN module(s): thin FFI bindings over `tinynn/` (the C shim
   over vendored ggml). One file per backend, same surface; module
   names are TinyNN / TinyNNCuda / TinyNNMetal. `_cuda`/`_metal` here
   are HAND-written (divergent), unlike the generated `_cuda`/`_metal`
   mirrors elsewhere (`prep/gen_cuda_mirror.rb`, gitignored).

Top-level stragglers: `lib/toy.rb` (sugar layer over Mat, Spinel tree)
and `lib/gpt2_ffi{,_kv}.rb` (GPT-2 full-forward / KV-decode caches —
retirement blocked on an engine-layer KV-decode replacement, toy#65
item 1; consumers are `tep_demo/openai_api.rb` and the `tinynn/gpt2_*`
parity harnesses).
