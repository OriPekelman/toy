# Consuming `toy` as a vendored gem

**Status:** current reference (toy#19 shipped 2026-05-27).
**Audience:** maintainers of research projects (e.g. `tao_transfer`)
who want to compose toy's primitives into their own Spinel-compiled
experiment, rather than fork toy or hand-path `require_relative`s
into toy's tree.

This doc covers the recipe end-to-end. Skim it once, then bookmark
the "Quick reference" at the bottom for routine re-runs.

## Why vendor instead of `require_relative` toy directly?

You _can_ run `spinel my_experiment.rb -o exp` with a
`require_relative "../toy/lib/llama_seq_forward_ffi"` line — but
when toy's own build runs, it's compiled with a particular layout
of relative paths in `ffi_cflags`. A consumer running from a
different working directory gets stale link paths AND, more subtly,
hits a Spinel poly-dispatch crash around `Mat` (the failure mode
toy#19 was filed on). The vendored layout sidesteps both: every
consumer sees the same `vendor/spinel/toy/lib/` tree, with link
paths rewritten to absolute references into toy's own checkout.

The convention is `bundler-spinel` / [spinelgems](https://github.com/OriPekelman/spinelgems);
toy is the second flagship consumer after tep. See
[`docs/roadmap/spinelgems-tep-adoption-2026-05-27.md`](roadmap/spinelgems-tep-adoption-2026-05-27.md)
for the broader convention.

## Prereqs (one-time)

- A working toy checkout at e.g. `~/sites/toy_ruby_neural_network`,
  with `make tinynn/libtinynn_ggml.a` already run (so the `.a`
  archives the consumer links against exist). For CUDA/Metal,
  `make setup-ggml-cuda` / `setup-ggml-metal` similarly.
- Ruby ≥ 3.2.0 on the consumer side (`bundle lock` resolves under
  CRuby; the engine marker only fires under `bundle install`).
- Bundler installed: `gem install --user-install bundler`.
- spinelgems checkout at `~/sites/spinelgems` (or set
  `SPINEL_DIR=…` if elsewhere).

## The five-step workflow

```sh
# 1. Declare toy as a path: gem.
cat > Gemfile <<'EOF'
source "https://rubygems.org"
ruby "3.2.3", engine: "spinel", engine_version: "0.0.0"

gem "toy", path: "../toy_ruby_neural_network"   # or "../toy" if symlinked
EOF

# 2. Resolve — produces Gemfile.lock with a PATH source.
bundle lock

# 3. Vendor — copies toy/lib/*.rb into vendor/spinel/toy/lib/ +
#    writes vendor/spinel/deps.rb (which requires the top-level
#    vendor/spinel/toy/lib/toy.rb).
SPINEL_DIR=~/sites/spinel ~/sites/spinelgems/exe/spinel-compat vendor

# 4. Rewrite link paths — turns toy's relative `-L.` / `-Ltinynn`
#    flags into absolute paths anchored at TOY_SRC. Copy this hook
#    from toy's prep/ into your own prep/ (it's a template — same
#    shape for every consumer):
mkdir -p prep
cp ../toy_ruby_neural_network/prep/post_vendor_toy.rb prep/
./prep/post_vendor_toy.rb         # uses TOY_SRC=~/sites/toy_ruby_neural_network

# 5. Compile your experiment.
spinel experiment.rb -o experiment
./experiment
```

A typical `experiment.rb` looks like this. **Note the explicit
`require_relative`s** for the specific primitives you compose —
the auto-generated `vendor/spinel/deps.rb` only pulls in
`toy/lib/toy.rb` (the top-level), so you pick the rest yourself:

```ruby
require_relative "vendor/spinel/deps"
require_relative "vendor/spinel/toy/lib/toy_smollm2"
require_relative "vendor/spinel/toy/lib/llama_seq_forward_ffi"

cfg = Toy::SmolLM2Config.new(627, 64, 4, 4, 128, 2, 32, 10000.0, 1.0e-5)
fcache = LlamaSeqForwardFFICache.new
fcache.realize_for_random_init(cfg, 32, false, false, 0, 1.0)
# … build_training_step, upload, compute, …
```

That's it. From here you write whatever experiment loop you want
against `LlamaSeqForwardFFICache` / `ViTTinyForwardFFICache` /
`SmolLM2KVFFICache` / the GGUF loaders. The Tao project at
`~/sites/tao_transfer` is the canonical worked example.

## Env knobs

`prep/post_vendor_toy.rb` recognises:

| Env | Purpose |
| --- | --- |
| `TOY_SRC=…` | Absolute path to toy's checkout. Defaults to `~/sites/toy_ruby_neural_network`. The `-L` rewrites anchor here. |
| `TOY_DISABLE=cuda,metal` | Skip rewriting backend files you don't compile. CPU is always rewritten. |
| `CUDA_DIR_LIB=…` | Override the absolute CUDA libdir baked into `tinynn_cuda.rb`. Defaults to `/usr/local/cuda/lib64`. |

## What gets vendored, what doesn't

`spinel-compat vendor` copies the gem's `lib/` verbatim into
`vendor/spinel/toy/lib/`. That includes all 46 `lib/**/*.rb` files
plus `lib/toy/version.rb` + `lib/toy/ffi_manifest.rb`.

It does NOT copy:

- `vendor/ggml/build/` — 6 MB of build artifacts. The link paths
  rewritten by step 4 point at toy's own `vendor/ggml/build/src`
  (no copy; toy keeps owning the artifact).
- `data/`, model weights, pretokenized corpora. Caller-managed.
- `tinynn/libtinynn_ggml.a` itself — also referenced via absolute
  path. The consumer's link command picks it up from toy's tree.

Practically: if you delete toy's checkout, your consumer breaks
(the absolute paths in vendored `tinynn.rb` go stale). Re-run
the workflow against a fresh toy checkout to repoint.

## Updating toy on the consumer side

When toy moves (new commits in toy's tree), the consumer's vendored
copy goes stale. Re-run steps 2-4:

```sh
bundle lock                                                # refresh PATH spec metadata
SPINEL_DIR=~/sites/spinel ~/sites/spinelgems/exe/spinel-compat vendor
./prep/post_vendor_toy.rb
spinel experiment.rb -o experiment                         # rebuild
```

If toy's `lib/tinynn.rb` ever changes the literal `ffi_cflags`
string, `post_vendor_toy.rb` will warn that no substitution
matched. That means `lib/toy/ffi_manifest.rb`'s
`CURRENT_FFI_CFLAGS` constant is out of sync with the source — bump
both in lockstep in a single toy commit. The warning is the
canary; honor it.

## Disabling Tep batteries (Pg / Sqlite)

Toy includes tep_demo/* apps that link `Tep::PG` and `Tep::Sqlite`
batteries via the tep vendored tree (see
[`docs/roadmap/tep-ffi-manifest-2026-05-27.md`](roadmap/tep-ffi-manifest-2026-05-27.md)).
If your downstream box lacks libpq / libsqlite3, set
`TEP_DISABLE=pg` (or `pg,sqlite`) **on the toy side's
post-vendor**, not the toy#19 hook — toy's post_vendor_toy.rb
doesn't touch tep, only the tinynn link paths.

In practice: most research consumers don't use the tep_demo
HTTP-serving apps. They use the training cache classes directly.
No tep dependency at all from a pure training experiment.

## Why this matters (the "no Mat landmine" half)

Tao tried before this work landed: hand-pathed
`require_relative "../toy/lib/toy_smollm2"` etc. Compile-time it
PASSED, but generated a Spinel poly-dispatch crash referencing an
undefined `sp_Mat` type in an Array<Mat> declaration. Root cause:
when toy's `lib/transformer.rb` (where Mat lives) is loaded via a
non-vendored relative path, Spinel's whole-program inference
fragments the type table differently than for toy's own builds.
The vendored layout — same tree shape for every consumer — keeps
inference monomorphic in the way toy's own build expects.

If you ever see `sp_Mat undeclared` from your consumer compile,
that's the smell. Confirm you're using the vendored layout (not a
hand-pathed `require_relative` into toy's tree).

## Quick reference

```sh
# Initial:
gem build ../toy_ruby_neural_network/toy.gemspec   # optional — produces a .gem
                                                   # for off-machine consumption.
                                                   # For path:-based consumers
                                                   # (siblings on the same host)
                                                   # this isn't needed.

# Every consumer cycle (when toy moves or your experiment changes):
bundle lock
SPINEL_DIR=~/sites/spinel ~/sites/spinelgems/exe/spinel-compat vendor
./prep/post_vendor_toy.rb
spinel experiment.rb -o experiment
./experiment
```

## See also

- [`roadmap/spinelgems-tep-adoption-2026-05-27.md`](roadmap/spinelgems-tep-adoption-2026-05-27.md)
  — the broader convention, originally written for tep.
- [`roadmap/tep-ffi-manifest-2026-05-27.md`](roadmap/tep-ffi-manifest-2026-05-27.md)
  — the FFI-manifest design (toy's `lib/toy/ffi_manifest.rb` is the
  toy-side analog of `Tep::FFIManifest`).
- [`events-schema.md`](events-schema.md) — the `toy/v1` event
  contract your consumer can emit (`run_start`, `step`, `eval`,
  `tap`, `run_end`) for Tao-side consumption.
- toy#19, tep#97, spinelgems#3 — the issue trail.
