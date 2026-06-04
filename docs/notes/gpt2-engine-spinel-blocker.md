# GPT-2 engine integration — post-mortem (it was a require-path bug, not Spinel)

`toy train --arch gpt2` works (loss 6.44 → decreasing on the from-scratch
corpus; `make gate-gpt2-train`). This note records the false trail, because the
*shape* of the bug (a silent emit-0 cascade that looked exactly like a
poly-degradation) is worth recognizing fast next time.

## The bug

The runner `lib/toy/run/train_gpt2.rb` had `require_relative "../toy"` /
`"../tinynn"` — but the toplevel lib files are **two** levels up from
`lib/toy/run/` (`"../../toy"` / `"../../tinynn"`, as the working `train.rb`
uses). The wrong path resolves to a nonexistent file; Spinel **ignores the
require** (warning only) and keeps compiling. So `TinyNN` (the FFI module) and
`Mat` were **never loaded** — every `TinyNN.tnn_*` / `Mat.new` / `.flat` call had
no definition, resolved "on int", and **emitted 0** (no-op / returned 0). Result:
weights and labels uploaded as zeros → logits 0 → CE=0 → no training, no crash,
`realize`/`compute` both reporting success.

Fix: two characters per line (`../` → `../../`). The engine code was correct the
whole time.

## Why it looked like a deep Spinel poly-degradation (the false trail)

- The `prep/gpt2_engine_smoke.rb` smoke **worked** — because it's in `prep/` and
  its requires (`"../lib/toy"`) happen to resolve correctly from the repo root.
- Every "it fails at realistic dims" repro I built was a `/tmp/*.rb` copy, run
  **from `/tmp`**, where `"../lib/toy"` resolves to `/lib/toy` — the SAME broken
  require. So "big dims fail / small dims pass" was an artifact of *where the
  file lived*, not the dimensions. The dims were a complete red herring.
- `spinel doctor`'s `on int (emitting 0)` count rose with the (apparent) dims,
  reinforcing the wrong story — but that count also fires on benign `:ptr`-as-int
  FFI lowering, so it conflated "module not loaded" (fatal) with "normal FFI".

## What actually found it: `spinel <file>.rb --emit-types`

The delivered compiler flag printed:
`require_relative "../tinynn" … could not be resolved (no such file …); the call
is ignored`. That one line — plus diffing against a working runner's
`require_relative "../../toy"` — was the whole fix. `--emit-types` (and
`--emit-rbs`, `--debug`) ship in `matz/spinel`; the spinel-dev doctor wraps them.

## Lesson + the one surviving spinel-dev ask (see [OriPekelman/spinel-dev#9](https://github.com/OriPekelman/spinel-dev/issues/9))

An **ignored `require_relative` that defines the very module being called** is
almost always the root of an emit-0 cascade. `--emit-types` surfaces it but
`doctor`'s human summary buries it under the downstream symptoms. Ask: doctor
should hoist ignored/unresolved requires to the top as the prime suspect, and
severity-rank `on int` (unloaded-module → fatal, vs `:ptr` lowering → benign).
(Proposals 1–3 and 5 on #9 were withdrawn — they were premised on this being a
real poly-degradation, which it wasn't.)

**Debugging rule for this repo:** a silent CE=0 / all-zero-output with
`realize`/`compute` returning success ⇒ check `--emit-types` for an *ignored
require* before suspecting anything subtle. Run `spinel` from the repo root (or
build via `make`), never from `/tmp`, so relative requires resolve.
