# Metal runtime followup — Mac session handoff (2026-06-01)

**Branch:** `metal-source-wiring` (commit `68ac64b`, NOT merged to main). **Run this followup ON A MAC** — metal cannot build/run on the gx10 (Linux, no Apple frameworks). gx10 source-wired metal + verified everything checkable there (Spinel codegen-no-link, structural parity vs the `_cuda` twins, clean-error-on-Linux, no CPU/CUDA regression, verify-mirrors). The **runtime** is unverified and that's your job.

## What's wired (source)
- `lib/toy/run/{infer,eval,train}_metal.rb` — runtime twins of the shipped `_cuda` runners (substitution-only: `TinyNNCuda`→`TinyNNMetal`, `ToyLMCuda`→`ToyLMMetal`, `*Cuda` caches→`*Metal`, requires `_cuda`→`_metal`).
- `lib/toy/llm/recipes/from_scratch_metal.rb` — metal twin of `from_scratch_cuda.rb`.
- `lib/tinynn_metal.rb` — gained the 9 missing `tnn_events_*`/`tnn_provenance_*` ffi_funcs (the codegen blocker fix; byte-match `tinynn_cuda`).
- `lib/toy/core/cli/{infer,eval,train}.rb` — `--device metal` is platform-gated: on macOS it builds+dispatches `libexec/toy-*-metal`; on non-macOS it returns the clean "metal is only available in a macOS build" error.
- `Makefile` — `libexec/toy-{infer,eval,train}-metal` targets (metal `--cc` + `-Wl,-u,_tnn_metal_force_link -framework Foundation/Metal/MetalKit`, macOS-guarded) + `gate-metal`.
- `prep/metal_gate.rb` — metal-vs-cpu parity harness; SKIPs cleanly on non-macOS.

## The failure to fix
`make gate-metal` on the Mac builds all three binaries fine, but **ARM 1 `metal-infer` produces `nil`** (cpu emits ids, metal emits nothing). So the metal binary builds but **fails/crashes at runtime before output**. First step — get the actual error:
```bash
bin/toy infer data/smollm2-135m-f32.gguf --device metal --prompt-ids "6403 1980 253 655 28" --n 8 2>&1; echo "exit=$?"
```
- **exit 139 / segfault** → fault in the Metal backend init or the KV/decode path.
- **a clean error line** → a guard tripped (model load, backend select, buffer alloc).

**Highest-risk seam (flagged in `train_metal.rb` + the gate banner):** the train checkpoint round-trip downloads weights from the Metal training session through a CPU TinyNN handle (`write_sess = TinyNN.tnn_session_new(0)`, `ToyGGUFFuser`). If Metal buffers aren't host-addressable via the CPU FFI handle, that fails. But note: the FIRST failure is **infer**, not train — so the infer/decode metal path (likely `ToyLMMetal` / `SmolLM2KVFFICache*Metal` backend init) is the place to look first.

## Done criteria
1. `make gate-metal` → `GATE PASS [metal-infer]` (cpu-vs-metal **byte-equal token IDs**), `[metal-eval]` (top-k **id-order** equal; logprob floats may differ Metal-F32-vs-CPU-f64 and still pass), `[metal-train]` (run-twice determinism or a Mac-pinned `prep/fixtures/train_metal_baseline.txt`; loss-decrease; ckpt round-trip vs the SHARED `prep/fixtures/ckpt_roundtrip_baseline.txt`). Overall exit 0.
2. Portable no-regression on the Mac: `ruby prep/infer_gate.rb` PASS (token IDs are cross-platform). `make verify-mirrors` green.
3. Then merge `metal-source-wiring` → main.

## gx10-canonical gates (do NOT fight these on the Mac)
The **byte-exact FLOAT gates are gx10-canonical** (recorded on aarch64-Linux; they're deterministic there but `libm` differs cross-platform). On the Mac, `eval_gate`/`train_gate`'s float text diverges at ~1e-6 — that's EXPECTED, not a regression (the token IDs are identical → the model is correct). gx10-side is landing a **platform-aware** fix: strict byte-exact on Linux, discrete-invariant + float-tolerance + note elsewhere. **Rebase this branch on the updated `main` before the final no-regression run** so the Mac sees green `eval_gate`/`train_gate`.

## Coordination
- gx10 is doing the gate-portability fix (touches `prep/{eval,train}_gate.rb` only — no overlap with this branch's files) and will push it to `main`.
- gx10 is **holding** all `cli/{train,eval}.rb` + `Makefile`-touching work (lora/warm-start `--device cuda`, eval-lmc, ViT/GPT-2) until this metal branch merges, to avoid conflicts.
- After metal merges, gx10 resumes with `lora`/`warm-start --device cuda` (main target = Linux+CUDA).
