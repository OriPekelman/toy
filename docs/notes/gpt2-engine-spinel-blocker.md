# GPT-2 engine integration — Spinel poly-degradation blocker + spinel-dev asks

A spinel-dev-driven investigation of why `GPT2SeqEngine` (the
`toy train --arch gpt2` path) silently produces CE=0 at realistic model
dimensions. Done with `spinel doctor` (`~/sites/spinel-dev/tools/doctor`) as far
as the current tooling allows; the remainder is tool proposals for spinel-dev.

## Symptom

`prep/gpt2_engine_smoke.rb` (the engine, tiny dims VOCAB=32) **trains** — CE
3.46→0.96, 72 weights. The same engine at realistic dims (VOCAB=627, the
llama-from-scratch gate shape) silently produces **CE=0, no training**:
`tnn_realize_backward` and `tnn_compute_backward` both return success, but the
loss is identically 0. The inline trainer `prep/gpt2_train_min.rb` (`make
gate-gpt2`, same math, NOT in a class) is unaffected and is the working
byte-exact reference.

## What `spinel doctor` showed

`sh doctor.sh --no-cruby --no-bisect lib/toy/run/train_gpt2.rb`:
- **0** `on int (emitting 0)` warnings at tiny dims; **9** (engine) / **40+**
  (runner) at realistic dims. Each is a `TinyNN.tnn_*` / `Mat.new` / `.flat`
  call resolving "on int" — the *receivers* (the `TinyNN` module, `Mat`, and
  even the `engine` instance at top level) are typed as `int`, so the call
  "emits 0" (no-op / returns 0).
- The degradation is **size-dependent, not value-dependent**: ANY single dim
  raised above the tiny baseline (VOCAB, D_MODEL, D_FF, *or* CONTEXT) flips it
  on; it correlates with a Ruby array/alloc crossing a ~512–1024-element
  threshold, NOT with a specific value or env-vs-literal.
- It is **restructure-resistant**: the `Mat`-based init degrades the engine's
  `@g_weights` Array<:ptr> to empty at big dims; reworking init to the llama
  `flat-Array + tnn_upload_from_float_array` pattern (no Ruby `Mat`) FIXES the
  weight array (72 weights at 627) but then the **`tnn_upload_from_float_array`
  FFI calls themselves** degrade to emit-0 at ALL dims → weights+labels upload
  as zeros → logits=0, CE=0. Each fix trades one poly-degradation for another.
- The llama engine does the same FFI uploads at VOCAB=627 and works — so it's
  this *compilation unit's* complexity (many distinct Array<:ptr> ivars + large
  buffers + per-(layer,head) flat indexing in one class), not the operations.

## Why the current tooling didn't close it

`spinel doctor` localizes the **symptom** but not the **root**, and two of its
signals are misleading here:

1. **No provenance for the degradation.** The doctor lists dozens of downstream
   `X on int (emitting 0)` lines but never points at the operation/merge site
   that made the receiver `int` in the first place. With ~40 symptoms and no
   first-cause, you bisect by hand (dims, requires, corpus, Mat-vs-flat) — slow
   and inconclusive.
2. **`--emit-rbs` says CLEAN while the compile leg degrades.** The inference leg
   reports `realize!: (Integer×7) -> nil`, `@g_weights: Array[Integer]`, **0
   untyped slots** — i.e. inference is happy — yet the codegen leg emits-0. This
   inference-clean-but-codegen-degrades case is exactly the silent miscompile,
   and nothing flags the discrepancy.
3. **The `on int` count conflates benign and malign.** A trivial working FFI
   script also shows nonzero `on int` (the `:ptr`-as-`int` lowering is normal).
   So the count is a noisy metric — some "on int" calls work, some silently
   no-op, and the doctor doesn't distinguish them.

## Proposals to spinel-dev (tooling)

1. **Degradation provenance / `--explain <symbol>`.** When a receiver or slot
   degrades to `int`/poly, report the *first cause*: the merge site, the
   operation, and the triggering value/array. "Why is `engine` int here?" is the
   question the doctor should answer, not "here are 40 calls on int."
2. **Flag inference↔codegen disagreement.** A distinct, high-severity verdict
   when `--emit-rbs` resolves a slot but the codegen leg emits-0 for it — that's
   the silent-miscompile fingerprint and currently passes the inference leg.
3. **Authoritative pin (not advisory).** Make `--rbs DIR` (or a
   `# @spinel: monomorphic` / `keep` annotation) able to FORCE a class/method to
   stay monomorphic, so a known-good signature (e.g. emitted from the tiny-dim
   build) prevents the large-dim degradation.
4. **Severity-ranked `on int`.** Distinguish benign `:ptr`-lowering "on int"
   (the call works) from malign "on int" (the call silently no-ops / zeros
   data). The count today is unusable as a pass/fail signal.
5. **Reproducer minimizer (`spinel-reduce`).** Shrink a degrading program to the
   minimal trigger — here it would isolate array-count vs array-size vs
   ivar-count vs FFI-call-count as the cause.

## Status / next

The engine + runner are committed as WIP (NOT wired into the `toy train` CLI, so
nothing user-facing is broken). The GPT-2 **arch is proven and gated** (`make
gate-gpt2`); only the engine *surfacing* is blocked. Next, pending spinel-dev:
file proposals 1–5; if a provenance trace lands, re-localize and pin. Meanwhile
the inline trainer is the demo/reference. A non-Spinel workaround worth trying:
move the heavy init out of the engine class into a C-side random-fill primitive
(`tnn_fill_uniform(tensor, n, scale, seed)`), so the engine unit never builds
large Ruby arrays at all — sidesteps the trigger rather than fighting it.
