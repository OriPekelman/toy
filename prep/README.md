# prep/ — gates, generators, converters

Everything in here is *development tooling*: nothing under `prep/` ships
in the library, and `lib/` never requires a `prep/` file. Three families
live side by side; the filename tells you which one you're holding.

## Gate smokes (`smokes/smoke_*.rb`) — the compiled fixtures

`prep/smokes/` holds the Spinel-compiled smoke fixtures the gates build
and run (formerly `examples/smoke_*.rb` — they are GATES, not tutorials,
so they live in gate-land). Each `smoke_<name>.rb` compiles to
`prep/smokes/smoke_<name>` via `make prep/smokes/smoke_<name>` and
asserts one primitive/recipe/branch end-to-end; `docs/gating.md` is the
index. Binaries are gitignored; run them from the repo root (they read
`data/` fixtures relative to the cwd).

## Gates (`*_gate.rb`) — the acceptance bar

A gate is a self-contained CRuby script that builds whatever it needs
(via `make`), runs it, and compares the output **byte-for-byte** against
a pinned baseline in `prep/fixtures/`. Exit 0 = pass, non-zero = fail,
and failures print the verbatim mismatch — gates never mask.

The convention:

- `prep/<surface>_gate.rb` runs the CPU arm; `prep/<surface>_cuda_gate.rb`
  (or a `TOY_GATE_CUDA=1` env arm inside the same script, see
  `infer_gate.rb` / `eval_gate.rb`) runs the CUDA twin.
- Baselines live in `prep/fixtures/<surface>_baseline.txt` plus any
  pinned corpora/ids fixtures the run needs (e.g. `ts_seqs_gate.bin`).
  Gate corpora MUST be pinned — an unpinned fixture is drift waiting to
  happen.
- Re-record intentionally with the gate's `--record` flag (where
  provided), never by hand-editing the baseline.
- Most gates have a `make gate-<name>` alias (see the Makefile's
  `gate-*` targets); CI-ish sweeps just chain those.

Train/serve gates are byte-exact across machines for ggml-computed
numbers; eval logprobs have a documented Ruby-libm tolerance arm (see
`eval_gate.rb`).

## Generators

- `gen_cuda_mirror.rb` — THE mirror generator. CPU sources listed in its
  `MIRRORABLE` table get `_cuda` / `_metal` siblings generated at build
  time (gitignored since ab11ca4; `make verify-mirrors` asserts the
  generator is healthy and idempotent). You write the CPU file only;
  carry CPU-only code over a mirror with the `CUDA-MIRROR-SKIP/STUB`
  sentinels documented at the top of the script. See also
  `docs/authoring.md` ("Mirror discipline").
- `gen_coverage.rb` — regenerates `docs/coverage.md` from the C shim +
  the three TinyNN bridge files (`lib/toy/ffi/tinynn*.rb`).

## Converters / data prep (Python + Ruby)

`convert_*_to_gguf.py`, `dump_bpe.py`, `*_tokens.py`, `pretokenize_*`,
`prep_tinystories.rb`, `preprocess_images.py`, `extract_vit_tiny.py` —
one-shot scripts that produce the regenerable artifacts under `data/`
(GGUFs, BPE tables, token id files, corpora). They run under the
system Python/CRuby, not Spinel.

## Everything else

Smokes and probes (`*_smoke.rb`, `gpt2_train_min.rb`), the build helpers
(`progress`, `quietly`, `build_tep_app.sh`), and `dataset_loader.rb` /
`tokenizer.rb` (CRuby-side helpers some gates share). Workflow `.js`
files are session-runner scratch, not tooling.
