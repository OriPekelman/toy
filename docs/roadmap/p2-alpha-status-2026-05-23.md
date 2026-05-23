# P2-α: TransformerLMTrainerFFI — status (2026-05-23)

Goal (per task #96): build training infrastructure that supports
multiple architectures and scales — the foundation for "scale up
training to larger models". User picked the architecture-pluggable
trainer scope (~250 LOC estimate).

This session attempted **Phase A** (Llama-arch only,
`realize_for_random_init` on `LlamaSeqForwardFFICache`) as a stepping
stone. Got the method written and building, but hit a runtime
segfault during persistent-tensor allocation. Reverted to clean state
pending dedicated debug session.

## What was attempted

Added `realize_for_random_init(cfg, t_seq, untied, qkv_bias, seed, init_scale)`
to `LlamaSeqForwardFFICache`. Same persistent-tensor shape as
`realize_for_full_finetune`, but instead of loading weights from a
GGUF:

  - allocates all weights as writable F32 persistent tensors
  - runs `tnn_finalize_weights` to back them with a buffer
  - calls `upload_random_init!` to fill matmul weights with N(0, σ)
    where σ = `init_scale / sqrt(fan_in)`, biases with 0, norm
    gammas with 1
  - calls `ft_zero_init_adam` to zero the Adam moments
  - builds the forward+backward+optimizer graph and realizes it

Companion example `examples/06_train_from_scratch.rb` mirrors the
existing `examples/example_finetune` shape but on a smaller config
(D_MODEL=64, N_LAYERS=2, vocab=627 from TinyStories).

PRNG: xorshift64 with a single-element `[seed]` state array (Spinel
doesn't like class vars; the array gives mutable closure-style
state without globals). Box-Muller from two uniforms per pair of
Gaussians.

## Where it stuck

Runtime segfault (SIGSEGV) during `realize_for_random_init`. Bisect
via File.open(...) {|f| f.puts ...} logging localised the crash to
*after* `ft_add_global_2d(@t_seq_token_embed, ...)` and *before* the
next `tnn_input_1d_f32_persistent` call for `final_norm_gamma`.

The crash is not in `ft_add_global_2d` itself (the
`realize_for_full_finetune` path uses the same helper successfully).
Candidate causes worth investigating in a follow-up session:

1. Spinel poly-widening of `@ft_globals_weights` / `@ft_globals_m`
   when multiple realize_for_* paths reference them differently.
2. A missing `@ft_train_embeddings_enabled = true` somewhere — the
   full_finetune path conditions embedding allocation on it; the
   random-init path always allocates persistent, so something may
   be reading the unset flag as a poly value.
3. Codegen ordering: maybe `@t_seq_rope_freq_factors` being
   allocated before some other init causes a tensor-table mismatch
   at `tnn_finalize_weights` time.

## What was learned

- The Spinel codegen does poly-widen `@ivar` reads when the same
  ivar is set with different types across `realize_for_*` paths.
  Adding new variants needs careful audit of ivar-write call sites.
- File.open(path, "a") {|f| f.puts ...} works as a fast debug-log
  inside Spinel-compiled code where `puts` is buffered and
  unreliable across crashes.
- The 250-LOC estimate was tight; realistic for the architecture-
  pluggable middle option is ~400 LOC even if we keep both archs in
  one class.

## Recommended next step (for the future)

When picking this up again:

1. Start with a minimal `realize_for_random_init` that only sets up
   *one* tensor + `tnn_finalize_weights` + a smoke compute, to
   verify the persistent-tensor allocation path works in isolation.
2. Incrementally add the rest of the weights, running the smoke
   after each addition.
3. Once the realize path is stable, add the random-init upload
   loop, then connect to `build_training_step` and verify the
   training step converges on a small TinyStories corpus.
4. THEN consider the `arch=:gpt2` plumbing — easier to add when the
   Llama-only path is bulletproof.

Estimated time from a fresh session: half a day to get the working
random-init Llama-arch trainer; another half-day for the GPT-2
plumbing if desired.

## Status

- Reverted lib/llama_seq_forward_ffi.rb to clean state
- Removed examples/06_train_from_scratch.rb (not yet shipping)
- Task #96 remains open
- This document captures what to start from next time
