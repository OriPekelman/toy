# Phase D status — Ruby tokenizer

**Status:** D1 shipped, D2 blocked on a Spinel C-compile regression.
**Date:** 2026-05-20

## What's done (Phase D1)

`prep/convert_smollm2_to_gguf.py` now takes `--with-tokenizer`. When
set, the converter:

- Pulls `tokenizer.json` from HF
- Extracts vocab (`tokens`), merge rules (`merges`), special-token
  IDs (BOS/EOS/PAD/UNK), and chat_template
- Embeds them in the GGUF using `gguf.GGUFWriter.add_tokenizer_*`
  helpers (`tokenizer.ggml.{model,pre,tokens,merges,bos_token_id,
  eos_token_id,padding_token_id,unknown_token_id,chat_template}`)

Verified end-to-end on SmolLM2-135M:
- Re-converted with `--with-tokenizer` → 49152 tokens + 48900 merges
  embedded.
- Our existing `tnn_gguf_arr_n` / `tnn_gguf_arr_str` FFI primitives
  read the embedded vocab back correctly (via `tinynn/arch_probe.rb`
  smoke test).

## What's blocked (Phase D2)

Ruby-side BPE encoder (`text → token IDs`) was prototyped. The
algorithm is correct and works in isolation (verified via standalone
classes). But integrating it into `lib/tokenizer.rb` — even keeping
ONLY the existing decode side from the Phase 0 commit — fails to
compile via Spinel:

```
out.c: error: aggregate value used where an integer was expected
spinel: C compilation failed
```

The error is emitted in generated C for `tnn_input_2d_f32` (from
`lib/transformer.rb`'s training-side TransformerLM class). The
return type of the FFI function is being assigned to a non-pointer
local. This happens only when `lib/tokenizer.rb` is loaded
alongside `lib/transformer.rb` — neither file in isolation
triggers it.

This is an existing latent issue: `lib/tokenizer.rb` was committed
in Phase 0 (4e959ad) but was never actually integrated into a
built binary at that time. The bug has been hiding since.

### Hypothesis

Spinel's cross-module type inference gets confused by some
combination of:

- Two `[]` array literals in `Tokenizer.from_gguf` (one populated
  with strings, one returned empty) — type inference flips between
  `int_array` and `string_array`.
- The `:str` FFI return type of `tnn_gguf_arr_str` interacting with
  the array build loop.
- An interaction with `lib/transformer.rb`'s training-side
  `TransformerLM` class (warnings about `embed_backward` "cannot
  resolve" appear together with the compile error).

Trying `git checkout 7beeb54` to roll Spinel back didn't fix it.
The error reproduces with the exact Phase-0-commit-shape of
`lib/tokenizer.rb` and current `lib/transformer.rb`.

### Reproducer

```sh
cat > /tmp/_t.rb <<EOF
require_relative "lib/transformer"
require_relative "lib/tokenizer"
t = Tokenizer.from_gguf("data/smollm2-135m-tok.gguf")
puts t.vocab_size
EOF
cp /tmp/_t.rb tinynn/_t.rb
sed -i 's|"lib/|"../lib/|g' tinynn/_t.rb
/home/oripekelman/sites/spinel/spinel tinynn/_t.rb -o tinynn/_t   # fails
```

### Workarounds (none clean)

1. **Keep tokenization in Python via `prep/llama_tokens.py`**.
   Status quo before D1. Works fine; the Ruby side speaks token IDs.
   This is what every demo + the OpenAI API actually does today.

2. **Build a separate Spinel-friendly tokenizer module** that
   doesn't transitively require `lib/transformer.rb`. The current
   `lib/tokenizer.rb` requires `lib/tinynn.rb` which the demo
   already pulls in via `lib/transformer.rb`. Decoupling would
   require pulling the GGUF metadata reads into a smaller module.

3. **File the Spinel issue + wait for a fix.** A minimal repro
   (this file's section above) would help upstream isolate. Matz
   has been responsive on Spinel issues per
   [[project_spinel_upstream]].

### Recommended path forward

Option 1 (status quo) until either (a) we have time to file the
Spinel issue and iterate, or (b) we decouple the tokenizer module
to avoid the cross-module type-inference trap.

## Phase D3 (round-trip test)

Blocked on D2. Once the encoder is unblocked, we round-trip:

  text → encode → IDs → inference → IDs → decode → text

For SmolLM2-135M, Llama-3.2, Qwen2.5. All-Ruby; no Python at
runtime.
