# P2: FFI Mat#matmul — measured, not viable as scoped (2026-05-23)

Surprise finding: the original Lowerer-evidence doc estimated
"5–10× on `example_train` wallclock, ~100 LOC" for routing
`Mat#matmul` through ggml. The actual measurement shows the
opposite — session-per-op FFI is **1.7× SLOWER** than the
pure-Ruby triple-nested loop at training-toy shapes.

## What I measured

Replaced `Mat#matmul`'s body with a one-line call to the existing
`TinyNN.matmul` helper (which does `session_new → input tensors →
realize → upload → compute → download → session_free` per call).
Ran `examples/example_train` on 87 TinyStories sequences.

| variant                              | wallclock |
|--------------------------------------|----------:|
| pure-Ruby `Mat#matmul`               |    1.3 s  |
| FFI per call (session-per-op)        |    2.2 s  |

The ~5 000 matmul calls × ~180 µs per-call FFI overhead =
~0.9 s of extra session lifecycle work — exactly the deficit.

## Why the estimate was wrong

The Lowerer-evidence doc cited the 38× full-forward FFI win from
`project_m1_full_forward_shipped_2026_05_14`. That number is for
**a whole forward pass packed into one ggml graph + one
`tnn_compute` call** — the FFI-cache pattern in
`lib/llama_seq_forward_ffi.rb` and `lib/toy_smollm2_ffi_kv.rb`.

The session-per-op pattern in `TinyNN.matmul` is the opposite shape:
each call pays ~500 µs of `tnn_session_new` + `tnn_realize` +
`tnn_session_free`, plus ~200 µs of per-element upload/download
loops. At a 32×8 matmul (the dominant training shape), that's
~10× more overhead than the ~50 µs of actual matmul math.

`project_ffi_perf_2026_05_13` memory note already said
"session-per-op is the wall for GPU"; we now confirm the same wall
applies on CPU at toy shapes.

## What would actually win

Two paths, both larger than P2's scoped 100 LOC:

1. **TransformerLMTrainerFFI** (~500 LOC) — mirrors
   `LlamaSeqForwardFFICache` for the custom-GPT shape used by
   `example_train`. One graph per step (forward + backward +
   optimizer), one `tnn_compute_backward` call. Estimated 5–20× on
   `example_train`. The architectural cost equals the original
   Lowerer estimate but for a much smaller surface (one model class,
   not every Mat usage).
2. **Session-pool MatFFI** (~150 LOC) — keep `Mat#matmul`'s
   shape-by-shape semantics but cache sessions by `(m, n, p)`.
   First call for a shape pays setup; later calls amortize. Modeled
   benefit: per-call FFI cost drops from ~800 µs to ~250 µs at
   steady state, still ~3× slower than pure Ruby's 77 µs at the
   32×8 shape. Net effect: **roughly break-even, possibly a small
   regression** at toy shapes. Win territory starts around
   matmul(64, 64) × matmul(64, 64), which `example_train` doesn't
   hit.

## Decision

Skip P2 as scoped. Pure-Ruby `Mat#matmul` is the right answer for
`example_train` and similar toy native-Ruby workloads.

The real workloads — LoRA training, KV-decode inference, full
fine-tune — already route through FFI cache classes
(`lib/llama_seq_forward_ffi.rb`, `lib/toy_smollm2_ffi_kv.rb`). Those
are the 38× speedups in production today; nothing to do.

If we later want to make `example_train` fast (e.g. as a tutorial
that doesn't take 2 s on a tiny GPT), build a TrainerFFI
(option 1 above) — tracked as **P2-α** (new task, not currently
prioritised).

## Bench-suite verdict

The bench harness still works correctly; this revert touches only
`lib/transformer.rb`. The baselines are unaffected (none of the
existing benches exercise the pure-Ruby Mat path):

| metric                       | unchanged |
|------------------------------|-----------|
| lora_step_ms                 | ✓ FFI graph already |
| infer_toks_per_sec           | ✓ FFI cache already |
| tokenizer_encode_us_per_tok  | ✓ no Mat dependency |

If we want a regression gate for the *native* training path, add
`bench/native_train.rb` invoking `example_train` with a fixed
seed and EPOCHS=1. Easy follow-up; doesn't block.

## Memory note

The Lowerer-evidence doc estimate ("5–10× on example_train via FFI
matmul") was wrong. The new memory entry below corrects it for
future planning. The 38× number applies to whole-graph-FFI, not
per-op-FFI.
