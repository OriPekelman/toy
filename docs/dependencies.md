# Dependencies & vendoring

How toy pulls in its two external Ruby dependencies (`tep` and
`spinel_kit`), the Spinel-codegen constraints toy permanently codes
around, and the vendored-ggml C divergences that live in `vendor/`.

Both are consumed identically — released RubyGems gems vendored via the
[spinelgems](https://github.com/OriPekelman/spinelgems) convention
(`bundler-spinel`), not hand-rolled vendoring — and both land under
`vendor/spinel/` with `vendor/spinel/deps.rb` requiring them in lock order
(`spinel_kit`, then `tep`).

---

## tep — consumed as a released RubyGems gem via spinelgems

`tep` is a **build-dep, used by `serve` alone**. `infer`, `train`, and
`eval` are tep-free. Since the SpinelKit consolidation (toy#44) serve uses
tep purely as the **HTTP transport** (`Tep::Handler` / `Tep.run!`); its JSON
encode/decode is `SpinelKit::Json`, not `Tep::Json`.

`Gemfile`:

```ruby
source "https://rubygems.org"
ruby "3.2.3", engine: "spinel", engine_version: "0.0.0"

gem "tep", "~> 0.11.2"
gem "spinel_kit", "~> 0.1"
```

## spinel_kit — the shared Spinel stdlib-surface gem (toy#44)

`spinel_kit` ([RubyGems](https://rubygems.org/gems/spinel_kit),
[repo](https://github.com/OriPekelman/spinelkit)) holds the pure-Ruby
"stdlib substitute" shims that Spinel-compiled projects can't get from CRuby
stdlib (the `json` gem is a C-ext, `Logger` won't lower, git gems are C). toy
and tep each grew these independently — the JSON escape/encode code came out
byte-identical — so they were consolidated into one gem.

toy uses three of its surfaces, all formerly hand-rolled in `lib/toy/io/`:

- **`SpinelKit::Json::Builder`** — the run_start/events JSON emitter (was
  `Toy::Json`; 15 consumers across the runners + examples).
- **`SpinelKit::Json`** — flat-key JSON encode (`encode_pair_*`/`from_*`) +
  decode (`get_int`/`get_int_array`/`has_key?`) for the serve handlers (was
  `Tep::Json` + a local `ApiJson` shim).
- **`SpinelKit::Git`** — `.git/HEAD` provenance for run_start (was `Toy::Git`).

Pure Ruby (`spinel-ext.json` is `[]`, no native ext), so the vendor is just
`lib/` copies under `vendor/spinel/spinel_kit/`. Runner/example sources require
the specific surface **by path** (e.g.
`vendor/spinel/spinel_kit/lib/spinel_kit/json_builder`) rather than via
`vendor/spinel/deps` — `deps.rb` pulls `tep` too, and the train/eval runners
must stay tep-free.

The **released gem** (https://rubygems.org/gems/tep) is the reproducible
cross-machine path — `Gemfile.lock` pins the version with a RubyGems sha256.
(The pre-publish `git: …branch: "main"` pin is retired.) tep ships a
`spinel-ext.json` so `spinel-compat vendor` natively compiles + wires its C
extensions; no `@TEP_*@` post-vendor substitution.

> serve is currently gated on a Spinel **pin-bump**: it won't compile at toy's
> current Spinel but is clean at tep's Spinel pin `f6d5eef` (a Spinel regression,
> not a tep bug — no tep release needed). Tracked as OriPekelman/tep#198; see
> `cli.md`.

### The flow (`make vendor-tep`)

```
bundle lock                                   # Gemfile -> Gemfile.lock
SPINEL_EXT_DISABLE=pg SPINEL_DIR=$(HOME)/sites/spinel \
  ../spinelgems/exe/spinel-compat vendor      # lock -> vendor/spinel/
```

`spinel-compat vendor` does two things: it copies tep's `lib/` into
`vendor/spinel/tep/lib/`, and it **natively compiles and wires tep's
C extensions** (driven by tep's shipped `spinel-ext.json`). It also
writes `vendor/spinel/deps.rb`, which a Spinel entrypoint pulls in
with `require_relative "../vendor/spinel/deps"` — that file in turn
`require_relative`s the vendored tep tree.

`make vendor-tep` gates on two sibling checkouts being present:
`../spinelgems` (the vendor tool) and the resolved tep source. It is
triggered transitively by the `serve` build path, not by the
tep-free runners.

### What got retired

The old rsync-and-substitute machinery is **gone**, not deprecated:

- `prep/sync_tep.rb` — DELETED (was: `rsync ~/sites/tep/lib/ →
  tep_demo/_tep_lib/`, then `require_relative "_tep_lib/tep"`).
- `prep/post_vendor_tep.rb` — DELETED.
- The `@TEP_*@` post-vendor placeholder trick and the `ffi_manifest`
  design (tep#97) it consumed — superseded by tep's `spinel-ext.json`
  (tep#98); `spinel-compat vendor` owns C-ext wiring now.
- tep#95 (move `prism` from a runtime to a development dependency;
  drop the required-Ruby floor to `>= 3.2.0`) is **landed** on tep
  `main`, so the lock resolves under gx10's Ruby 3.2.3 with zero
  toy-side tricks.

### Two environment notes

1. **`bundle` needs a user-managed Ruby env** (rbenv / rv /
   `gem install --user-install bundler`). System-owned gem dirs
   (`/var/lib/gems`) need sudo to write bundler's git cache, so
   `bundle lock` fails there. Doc-only — nothing in the build needs
   changing, just run from a user Ruby.
2. **tep's optional `pg` C-ext is opted out** via
   `SPINEL_EXT_DISABLE=pg` (baked into `vendor-tep`). Its libpq
   pkg-config cflags are not yet wired through spinel-compat
   (spinelgems#8). toy doesn't use tep's `pg` battery, so opting out
   is free.

> Serving is `toy serve` → `lib/toy/run/serve.rb` → `libexec/toy-serve`
> (the OpenAI-compatible HTTP API under `lib/toy/serve/openai/`). tep
> is transport plumbing for that binary only.

---

## toy itself as a consumable gem (toy#19)

toy is the **second** spinelgems consumer (after tep). Research
projects (e.g. `tao_transfer`) compose toy's primitives by vendoring
toy rather than hand-pathing `require_relative` into toy's tree.

The consumer-side flow mirrors the tep flow, plus one post-vendor
hook:

```sh
# Gemfile:  gem "toy", path: "../toy"
bundle lock
SPINEL_DIR=~/sites/spinel ~/sites/spinelgems/exe/spinel-compat vendor
# (post-vendor hook retired by toy#45 — build-units are self-contained)
spinel experiment.rb -o experiment
```

`prep/post_vendor_toy.rb` (present in this repo, shipped as a
template for consumers) rewrites the vendored
`tinynn{,_cuda,_metal}.rb` `ffi_cflags` from toy's relative `-L.` /
`-Ltinynn` / `-Lvendor/...` paths into absolute paths anchored at the
consumer's `TOY_SRC`. The path-rewrite logic is driven by
`lib/toy/ffi_manifest.rb`. The vendored `vendor/spinel/deps.rb` only
pulls in `toy/lib/toy.rb`; the consumer `require_relative`s the
specific primitives it composes.

Env knobs honored by `prep/post_vendor_toy.rb`:

| Env | Purpose |
| --- | --- |
| `TOY_SRC=…` | Absolute path to toy's checkout (default `~/sites/toy`). The `-L` rewrites anchor here. |
| `TOY_DISABLE=cuda,metal` | Skip rewriting backend files the consumer doesn't compile. CPU is always rewritten. |
| `CUDA_DIR_LIB=…` | Override the absolute CUDA libdir baked into `tinynn_cuda.rb` (default `/usr/local/cuda/lib64`). |

> The **"no Mat landmine"**: a consumer that hand-paths
> `require_relative "../toy/lib/..."` compiles but trips a Spinel
> poly-dispatch crash on an undefined `sp_Mat` (Array<Mat> type-table
> fragmentation). The vendored layout — identical tree shape for every
> consumer — keeps inference monomorphic the way toy's own build
> expects. If you see `sp_Mat undeclared` from a consumer compile,
> that is the smell; confirm you are on the vendored layout.

For the full consumer recipe see the worked example at
`~/sites/tao_transfer`.

---

## `Mat#add` was renamed to `Mat#plus`

In `lib/toy/models/transformer.rb`, the element-wise add on `Mat` is `#plus`,
not `#add`. The rename avoids a Spinel polymorphic-dispatch collision
with `Tep::Router#add` (a same-named method on an unrelated class):
Spinel's arg-type narrowing keys on method **name**, not signature, so
two unrelated `#add`s in one compilation widen each other. `#add!`
(mutating, suffix-banged) is unaffected — a distinct name.

---

## Spinel constraints toy permanently codes around

These are durable codegen/inference limits in the Spinel compiler.
toy works around each deliberately; they are not bugs in toy. They
form a regression budget — a Spinel change touching any of these
patterns deserves a re-test (`make bench-heavy` is the proof gate).

- **Seed-and-pop empty arrays (~94 sites).** A bare `[]` infers as
  `sp_IntArray *`; pushing 8-byte pointers or floats into 4-byte slots
  silently corrupts. toy writes
  `out = [TinyNN.tnn_null_ptr]; out.pop` (or `[0.0]; .pop`,
  `[""]; .pop`) to pin the element type. Heaviest in the
  `lib/toy/llm/engine/llama_seq_engine{,_cuda,_metal}.rb` and
  `lib/toy_smollm2_ffi_kv*.rb` files. Partial upstream fix (issue #688)
  promotes the local but locks the function-parameter type before the
  promotion is observed, so cross-function passes still need the seed.
- **`has_key?` guards on `IntHash`.** `*IntHash#[]` returns `0` on
  miss (and `*StrHash#[]` returns `""`), both truthy under Spinel
  coercion. toy guards lookups with `has_key?` (notably across
  `lib/toy/io/tokenizer.rb`). Upstream `*StrHash` now returns NULL, but
  `*IntHash` nil-semantics (`SP_INT_NIL`, the path tokenizer uses) is
  deferred, so the guards stay.
- **No `STDERR`.** Spinel has no `STDERR` constant; diagnostic warnings
  use `puts`. Cosmetic, but it means warnings land on stdout.
- **Filesystem C shim.** `Dir.entries` and friends are absent, so
  directory walking goes through a C shim
  (`tnn_list_dir_basenames`, in `tinynn/tinynn_gguf.c`), used by
  `lib/model_index.rb`. (`File.basename` and string interpolation are
  now available upstream but not bulk-adopted — the shim stays while
  `Dir.entries` is missing.)
- **Manual `.to_s` concatenation (~104 sites).** String interpolation
  (`"x=#{val}"`) is avoided in the compiled subset in favor of
  `"x=" + val.to_s` to dodge poly-cascade in mixed-type contexts.

---

## Vendored-ggml C divergences

toy carries a small, idempotent set of patches against the vendored
`ggml` checkout under `vendor/`. The patch set lives in
`vendor-patches/` (0001–0006) and is applied by the Makefile via a
`.patched` sentinel. The patch roster and which gate each one backs
is documented in **[gating.md](gating.md)** — in brief: 0001–0003 add
CUDA `buffer_from_ptr` (BYO-pointer, landed —
`ggml_backend_cuda_buffer_from_ptr` is present in
`vendor/ggml/include/ggml-cuda.h` + `src/ggml-cuda/ggml-cuda.cu`),
0004 cpy-strided (gates CUDA KV-cache bit-identity), 0005
concat-backward (live at `vendor/ggml/src/ggml.c`, gates training),
0006 getrows-back-large-vocab (gates training).

These divergences are deliberately confined to `vendor/`. The one
open upstream issue that toy masks at the application layer — the
ggml-cpu sched grad-aliasing bug, masked by
`tnn_pin_all_graph_b_nodes` in `lib/toy/llm/engine/llama_seq_engine.rb` — is
tracked under known-issues in **[roadmap.md](roadmap.md)**.
