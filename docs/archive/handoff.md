# Session handoff (gx10 inference work)

This doc is for a fresh Claude session picking up where the Mac
session leaves off. It assumes the next session runs **on gx10**
(so it can iterate directly without rsync round-trips).

The Mac and gx10 share a git bare repo at `gx10:~/git/toy_ruby_neural_network.git`
(see `gx-sync --init`). Working trees are equal peers.

## Where we are

Inference path for llama-family decoder LMs (Toy::SmolLM2) now scales
to 7B-class models. Verified on gx10 CPU during this session:

| Model | Params | Decode | Memory | Output |
|---|---|---|---|---|
| SmolLM2-135M | 0.135B | 25 ms/tok | <2 GB | "Once upon a time, there was a little girl named Lily…" |
| Qwen2.5-0.5B | 0.49B  | 80 ms/tok | <4 GB | "Hello, my name is a 10 year old boy. I have a question about my hair." |
| TinyLlama-1.1B | 1.1B | 146 ms/tok | <8 GB | "Once upon a time, there was a young girl named Lily…" |
| Qwen2.5-1.5B | 1.54B | 220 ms/tok | <12 GB | "Hello, my name ishita and I am a 12th grade student from India." |
| Qwen2.5-3B   | 3.09B | 462 ms/tok | <25 GB | "Hello, my name is and I am a 10th grader. I am currently taking AP" |
| Qwen2.5-7B   | 7.62B | 1062 ms/tok | ~30 GB | "Hello, my name is a 19-year-old male. I have been having a problem with my" |

Scaling is linear in params — `decode_ms ≈ params_billion × 150 ms`
on gx10 CPU. CUDA path exists but was untested here (training was
holding the GPU). Expect 5–10× speedup once the GPU is free.

## What landed this session

Three commits on `main` (not yet pushed to origin — user explicitly
held back the GitHub push):

1. **`3427387 Toy::Card: structured IR …`** — algorithm pseudocode
   becomes structured data; round-trip parser consumes the IR
   directly. `docs/lowerer-design.md` records Sam Ruby's Roundhouse
   pitch from `tep#6`.
2. **`69f1c0c Qwen2.5-0.5B + TinyLlama-1.1B unblocked …`** — the big
   one. Fixed a silent scratch-buffer overflow in
   `stage_transposed_and_upload` that bit any tensor > 4M floats.
   Q/K/V bias support for Qwen2.x (per-head allocations + flag-gated
   apply through native + FFI). FFI trace-tap infrastructure
   (`SmolLM2KVFFICache#enable_trace!`) — zero cost when off, named
   per-subblock stats when on. Hardening: `tnn_scratch_set` now
   `fprintf`s on out-of-range writes.
3. **`f3a35d9 Qwen2.5-1.5B and Qwen2.5-3B run end-to-end`** —
   converter handles sharded safetensors (≥3B); 1.5B and 3B demos.
   Documents the memory-cost problem in `docs/memory-design.md`.

The unpushed-to-origin state was deliberate (CLI permission rule).
The user will push when ready.

## Open work items

### Just-landed (and now verified)
- `lib/tinynn.rb` + `lib/toy_smollm2_loader.rb` + `tinynn/tinynn_gguf.c`:
  direct GGUF→FFI loader (skip the Ruby Float64 Mat intermediate; cuts
  12 B/w to 4 B/w). Verified bit-identical to the Mat-mediated path on
  0.5B and 1.5B. **Qwen2.5-7B confirmed end-to-end** at ~30 GB peak
  RAM, 1062 ms/token, coherent text.

### Easy follow-ups
1. **Wire the direct loader into the chat / OpenAI-compat API**
   (`tep_demo/openai_api.rb`). Currently it uses the Mat-mediated path.
   For 7B serving we need direct.
2. **Run benchmarks** on the direct path at all sizes. Compare to
   Mat-mediated. The load time should be slightly faster (no Mat
   alloc); decode time should be identical (same FFI graph).
3. **Document the trace tap** in `tinynn/README.md`. Right now it
   only exists in `docs/qwen25-known-issue.md` and the source comments.
4. **Q8_0 conversion of the larger models**. `--quantize q8_0` works
   for SmolLM2; verify it for 1.5B/3B/7B. Cuts disk size 4× and the
   direct loader handles dequant transparently.

### Real work
5. **`docs/memory-design.md` option (C)**: mmap the GGUF into the
   ggml persistent buffer. On GB10 (unified memory) this means zero
   extra RAM beyond the file's page cache. Major upgrade. ggml has
   the primitives (llama.cpp uses them); we'd wire `mmap` + a new
   `tnn_input_from_mmap` C function.
6. **`docs/tinyllama-known-issue.md` corrections**: the original
   diagnosis ("f32 precision overflow") is annotated as FIXED but the
   archive still reads as if open. Could be cleaned up entirely.
7. **CUDA verification** of all models we now support. CUDA
   `realize_for` takes the new `qkv_bias` param; mirrors the CPU
   path. The GPU was busy this session; rerun the bench-gx10 table
   on CUDA once available.
8. **Llama-3.2-1B layer-count bug** still on the books. We have the
   GGUF (`data/llama-3.2-1b-f32.gguf`). The loader reads L=11 from
   metadata when the file claims L=16. Probably `tnn_gguf_get_u32`
   path or `llama.block_count` key naming. Hasn't been investigated.
9. **Fine-tuning examples**. TransformerLM still works for training
   (SmolLM2-shaped models — not yet wired into the trainer). The
   user wanted "fine-tuning examples" in the original loop direction.

### Optional / nice-to-have
10. The Card IR (Toy::Card) has a `render_pseudocode` renderer. A
    `render_ruby` renderer would close the round-trip loop
    structurally (today the parser regexes the rendered text). See
    `docs/lowerer-design.md`.
11. The lowerer itself (Sam Ruby's pitch) — Prism source walker that
    emits per-shape Mat specializations. Not urgent.

## Files map

```
lib/
├── toy_card.rb               Structured IR for algorithm cards (NEW this session)
├── toy.rb                    Building blocks; algorithm methods build Cards
├── toy_gpt2.rb               GPT-2 family model (Card IR)
├── toy_smollm2.rb            Llama family model (Card IR + Q/K/V bias)
├── toy_smollm2_loader.rb     GGUF loader (Mat-mediated + NEW direct-to-FFI)
├── toy_smollm2_ffi_kv.rb     CPU FFI KV-cache (+ trace taps NEW)
├── toy_smollm2_ffi_kv_cuda.rb  CUDA FFI KV-cache
├── tinynn.rb                 FFI binds; chunked transposed upload (NEW)
├── tinynn_cuda.rb            CUDA FFI binds
└── gguf_load.rb              Mat-mediated loader; bias reader helpers

tinynn/
├── tinynn_ggml.{c,h}         C shim; chunked upload (NEW); scratch stats (NEW); fprintf hardening (NEW)
├── tinynn_gguf.{c,h}         GGUF I/O; direct-to-FFI funcs (NEW)
└── README.md                 Refreshed inference path section

prep/
├── convert_smollm2_to_gguf.py  Sharded safetensors (NEW); bias writes (NEW)
├── qwen25_tokens.py             Tokenizer (NEW)
├── llama_tokens.py              TinyLlama / Llama-3.2 tokenizer
└── smollm2_tokens.py            SmolLM2 tokenizer

demos/
├── qwen25_kv.rb              0.5B (Mat-mediated)
├── qwen25_1.5b_kv.rb         1.5B (Mat-mediated)
├── qwen25_3b_kv.rb           3B (Mat-mediated)
├── qwen25_direct.rb          0.5B (NEW direct loader)
├── qwen25_1.5b_direct.rb     1.5B (NEW direct loader)
├── qwen25_7b_direct.rb       7B (NEW direct loader — requires direct path)
├── qwen_probe.rb             FFI trace bisection
├── qwen_native_probe.rb      Float64 oracle
└── (smollm2_kv / tinyllama_kv / …) other models

docs/
├── memory-design.md          NEW: 2× duplication analysis + path forward
├── qwen25-known-issue.md     NEW: scratch overflow story + trace tap walkthrough
├── tinyllama-known-issue.md  Annotated FIXED (root cause was the same)
├── lowerer-design.md         NEW: Sam Ruby's Roundhouse pitch
├── scout-small-models.md     Updated: Qwen 0.5B status FIXED, tokenizer pending
└── bench-gx10-2026-05-16.md  Pre-this-session benchmarks
```

## How to iterate (from gx10)

Repo lives at `~/sites/toy_ruby_neural_network`. Spinel at
`~/sites/spinel`. ggml vendored at `~/sites/toy_ruby_neural_network/vendor/ggml`
(via `make setup-ggml` — already built).

```sh
# Build the C archive (rebuilds when tinynn/tinynn_*.c changes):
make tinynn/libtinynn_ggml.a

# Build a demo (always direct invocation since some demos have no Make rule yet):
/home/oripekelman/sites/spinel/spinel demos/qwen25_7b_direct.rb \
    -o demos/qwen25_7b_direct

# Tokenize a prompt:
~/.local/bin/uv run --quiet prep/qwen25_tokens.py encode "Hello, my name is"

# Run:
./demos/qwen25_7b_direct

# Decode the IDs back:
~/.local/bin/uv run --quiet prep/qwen25_tokens.py decode
```

For CUDA versions:
```sh
make setup-ggml-cuda                # one-time
# Then use the demos/*_cuda variants.
```

## Gotchas / Spinel survival tips

1. **Local variable names collapse across the whole program.** If
   variable `x` is `String` in one method and `Int` in another,
   Spinel boxes both as `RbVal` and the int-context site fails to
   compile. Workarounds: rename to disambiguate (`tok` → `tok_id`),
   add `.to_i` casts in dead-code methods, or add an `if false`
   anchor block pinning the name to the right type.
2. **`tnn_scratch_set` now warns on out-of-range writes** (since
   step 45). If you see the `[tnn] WARN: tnn_scratch_set idx=…
   out of range` line, you've introduced a primitive that writes
   past the 4M-float scratch boundary. Use a chunked uploader
   (`tnn_upload_from_float_array`, `tnn_upload_transposed_f64`,
   or `tnn_gguf_copy_*_to_persistent`).
3. **Reassigning ivars after construction confuses Spinel.** Pin
   types at the initializer; flip flags later.
4. **Ternaries with env vars in module-level constants** (e.g.
   `GGUF = ENV["X"] || "default.gguf"`) can trigger a collapse.
   Per-size demo files are the workaround until Spinel handles this.
5. **The FFI trace tap is zero-cost when off.** To diagnose a new
   model's NaN, just set `kv.enable_trace!` before the first
   `decode_step` and read the per-subblock stats. See
   `docs/qwen25-known-issue.md` for the walkthrough.

## Pending GitHub issue

[Issue #1](https://github.com/OriPekelman/toy/issues/1) tracks the
broader "silent-failure FFI primitives" audit. The narrow fix
(noisy `tnn_scratch_set`) landed in commit `69f1c0c`. The deeper
work (chunking `tnn_upload` / `tnn_download` internally, adding a
`TNN_ASSERT` macro, etc.) is open.

## Running training (the gx10 was busy this session)

The training script the user is running (the one consuming 57 GB)
is in a separate codebase. Don't touch it. If you need to confirm
it's still running: `ps -ef | grep train.py | grep -v grep | head -1`.

If the gx10 frees up and the user wants to verify CUDA: the demos
mirror exactly (`smollm2_kv` ↔ `smollm2_kv_cuda`, etc.). Build the
CUDA archive (`make setup-ggml-cuda`) and the `*_cuda` Spinel
binaries.
