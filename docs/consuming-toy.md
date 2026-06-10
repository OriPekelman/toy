# Consuming `toy` as a library

**Audience:** maintainers of research projects (e.g. `tao_transfer`) who want to
**compose toy's primitives** — engines, KV caches, GGUF loaders — into their own
Spinel-compiled experiment, rather than fork toy or hand-path `require_relative`s
into toy's tree.

> Not what you want? If you just want to **run** models (infer / train / eval /
> serve a GGUF), use the `toy` **CLI** — `toy new`, `toy infer model.gguf`, … —
> which is clean and self-contained. See [`cli.md`](cli.md). This doc is the
> *library-composition* surface, which is a different (and, today, rougher) path.

## Status: transitional — the current recipe works, but it's rougher than it should be

The honest state of play: the recipe below **works** and is what `tao_transfer`
uses, but it's more brittle and contrived than consuming a normal gem — a copied
post-vendor hook, absolute paths into toy's checkout, a manifest "canary", and
manual `require_relative`s. All of that traces to **one root cause**: toy doesn't
yet own its native-extension build the way [tep](https://rubygems.org/gems/tep)
does (toy ships no `spinel-ext.json`, so the vendor step can't build/link toy's
native archive into your tree). The cleanup is tracked across three repos —
see [The target, and how we get there](#the-target-and-how-we-get-there).

## The target (where this is going)

```sh
# Gemfile:  gem "toy", "~> X"        # published to RubyGems, like tep
bundle lock
spinel-compat vendor                 # copies lib/ AND builds toy's native archive
                                     #   INTO vendor/, wiring relative link paths
# experiment.rb:
#   require_relative "vendor/spinel/deps"   # pulls the full Toy compute surface
spinel experiment.rb -o experiment
./experiment
```

Three steps, no post-vendor hook, no absolute paths into toy's checkout, no
manifest canary, no `require_relative` soup. That's the bar: as friendly as the
CLI.

## The current recipe (works today)

### Prereqs (one-time)
- A toy checkout (e.g. `~/sites/toy`) with `make tinynn/libtinynn_ggml.a` already
  run (so the `.a` the consumer links against exists). For CUDA/Metal,
  `make setup-ggml-cuda` / `setup-ggml-metal`.
- Ruby ≥ 3.2 on the consumer side (`bundle lock` resolves under CRuby).
- Bundler (`gem install --user-install bundler`).
- A [spinelgems](https://github.com/OriPekelman/spinelgems) checkout at
  `~/sites/spinelgems` (the vendor tool), or set `SPINEL_DIR=…`.

### The five steps
```sh
# 1. Declare toy as a path: gem (a published gem is the target; see above).
cat > Gemfile <<'EOF'
source "https://rubygems.org"
ruby "3.2.3", engine: "spinel", engine_version: "0.0.0"

gem "toy", path: "../toy"
EOF

# 2. Resolve.
bundle lock

# 3. Vendor — copies toy/lib/*.rb into vendor/spinel/toy/lib/ + writes
#    vendor/spinel/deps.rb (which requires vendor/spinel/toy/lib/toy.rb).
SPINEL_DIR=~/sites/spinel ~/sites/spinelgems/exe/spinel-compat vendor

# 4. (THE CONTRIVED STEP — tracked for removal, toy#42) Rewrite link paths:
#    turns toy's relative -L flags into absolute paths anchored at TOY_SRC.
#    Copy the hook template from toy's prep/ into your own prep/:
# (nothing else — spinel-ext.json build-units build ggml + the tinynn
# archive inside vendor/spinel/toy/ during `spinel-compat vendor`; toy#45)

# 5. Compile.
spinel experiment.rb -o experiment
./experiment
```

A typical `experiment.rb`. One require — `toy/compute` (toy#42) — pulls the
whole composition surface (all three engines + recipes + GGUF/tokenizer
loaders), so you no longer hand-`require_relative` each primitive:

```ruby
require_relative "vendor/spinel/toy/lib/toy/compute"   # the full compute API

cfg    = Toy::SmolLM2Config.new(627, 64, 4, 4, 128, 2, 32, 10000.0, 1.0e-5)
engine = Toy::LLM::Engine::LlamaSeqEngine.new
engine.realize_for_random_init(cfg, 32, 1, 0, false, false, 0, 1.0)
# … build_training_step, upload, compute, …
```

`toy/compute` is the **Spinel-only** entry — it pulls `tinynn` (FFI) and the
engines. The top-level `toy.rb` stays MRI-safe (Mat + Card + version, no FFI)
so `bin/toy` can require it; don't use `toy.rb` for the compute surface. If you
build only one engine and want to compile less (Spinel has no tree-shaking),
require that engine's file directly instead of `toy/compute`. The
`gate-compute-surface` make target proves the one-require surface co-compiles +
runs a from-scratch training step (`examples/smoke_compute_surface`), with
`recipes/lora` co-loaded uncalled — historically the poisoning shape
([spinel-dev#11](https://github.com/OriPekelman/spinel-dev/issues/11) /
[spinel-dev#12](https://github.com/OriPekelman/spinel-dev/issues/12), fixed
upstream; re-added by toy#52).

From here you write your experiment loop against
`Toy::LLM::Engine::LlamaSeqEngine` / `Toy::LLM::Engine::ViTTinyEngine` /
`SmolLM2KVFFICache` / the GGUF loaders. `~/sites/tao_transfer` is the canonical
worked example. To emit Tao-consumable telemetry, see [`events.md`](events.md)
(and `SpinelKit::Json::Builder` / `Toy::Events.add_provenance` for building it).

### Env knobs (`prep/post_vendor_toy.rb`)
| Env | Purpose |
| --- | --- |
| `TOY_SRC=…` | Absolute path to toy's checkout. Default `~/sites/toy`. The `-L` rewrites anchor here. |
| `TOY_DISABLE=cuda,metal` | Skip rewriting backend files you don't compile. CPU always rewritten. |
| `CUDA_DIR_LIB=…` | Override the absolute CUDA libdir baked into `tinynn_cuda.rb`. Default `/usr/local/cuda/lib64`. |

### What gets vendored, what doesn't
`spinel-compat vendor` copies the gem's `lib/` verbatim into
`vendor/spinel/toy/lib/`. It does **not** copy `vendor/ggml/build/` (6 MB),
`data/`/weights, or `tinynn/libtinynn_ggml.a` — those are referenced by the
absolute paths step 4 writes (pointing into toy's own tree). **This is the
brittleness:** delete toy's checkout and your consumer breaks. Re-run the
workflow against a fresh toy checkout to repoint. (The fix — build the archive
*into your vendor tree* — is spinelgems#14 + toy#42.)

### Updating when toy moves
Re-run steps 2–4:
```sh
bundle lock
SPINEL_DIR=~/sites/spinel ~/sites/spinelgems/exe/spinel-compat vendor
./prep/post_vendor_toy.rb
spinel experiment.rb -o experiment
```
If toy's `lib/tinynn.rb` changes its literal `ffi_cflags`, `post_vendor_toy.rb`
warns that no substitution matched — that means `lib/toy/ffi_manifest.rb`'s
`CURRENT_FFI_CFLAGS` is out of sync with the source; bump both in one toy commit.
The warning is the canary; honor it. (This canary, too, disappears once toy owns
its ext wiring — toy#42.)

## The `sp_Mat undeclared` landmine

If your consumer compile fails with `sp_Mat undeclared` (or a poly-dispatch crash
around an `Array<Mat>`), you've hand-pathed a `require_relative` into toy's tree
instead of using the vendored layout. Root cause: Spinel's whole-program
inference is **require-path/layout sensitive** — loading `transformer.rb` (where
`Mat` lives) via a non-vendored path fragments the type table differently than
toy's own build. The vendored layout (same tree shape for every consumer) keeps
inference monomorphic the way toy's build expects. This layout-sensitivity is the
deeper root of *why* vendoring has to exist — filed as **matz/spinel#1367**.
toy's `make gate-poly-degrade` guards regressions of the related emit-0 class but
not this root.

## The target, and how we get there

Everything brittle above is a workaround for one gap — **toy doesn't own its
native-ext wiring like tep does**. The cleanup, tracked across three repos:

| Repo | Issue | What it unblocks |
| --- | --- | --- |
| spinelgems | **#14** | a native **build-step** in `spinel-ext.json` (build + link archives *into* the vendor tree), not just per-`.c` placeholder substitution — toy's ggml CMake build needs this. **The keystone.** |
| toy | **#42** | author `spinel-ext.json` (→ retire `post_vendor_toy.rb` + the `ffi_manifest` canary); a Spinel-only full-API `require`; **publish to RubyGems**; `toy new --lib` scaffold. Depends on #14. |
| matz/spinel | **#1367** | stable package identity so a user class (`Mat`) types the same regardless of require path — removes the layout-sensitivity the vendored layout works around. |

When #14 + #42 land, this doc collapses to the three-step
[target](#the-target-where-this-is-going) above.

## See also
- [`cli.md`](cli.md) — the friendly *app* surface (run models without any of this).
- [`events.md`](events.md) — the `toy/v1` event contract your consumer can emit.
- [`dependencies.md`](dependencies.md) — how toy itself consumes tep (the model toy is following).
- toy#19 (the original "no Mat landmine" work), tep#97, spinelgems#3 — the issue trail.
