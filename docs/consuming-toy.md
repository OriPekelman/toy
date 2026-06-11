# Consuming `toy` as a library

**Audience:** maintainers of research projects (e.g. `tao_transfer`) who want to
**compose toy's primitives** — engines, KV caches, GGUF loaders — into their own
Spinel-compiled experiment, rather than fork toy or hand-path `require_relative`s
into toy's tree.

> Not what you want? If you just want to **run** models (infer / train / eval /
> serve a GGUF), use the `toy` **CLI** — `toy new`, `toy infer model.gguf`, … —
> which is clean and self-contained. See [`cli.md`](cli.md). This doc is the
> *library-composition* surface.

## The recipe (toy#45: Gemfile + vendor — that's it)

Toy ships `spinel-ext.json` **build-units** (spinelgems#14): `spinel-compat
vendor` copies toy's `lib/`, then builds ggml + the tinynn shim archive
**inside your vendor tree**, and wires *project-relative* `-L` flags into the
vendored Ruby. Self-contained and relocatable — no post-vendor hook, no
absolute paths into toy's checkout, no canary. (`toy new --lib` scaffolds all
of this for you.)

```sh
# 1. Gemfile (path: for a checkout today; plain `gem "toy"` once published)
cat > Gemfile <<'EOF'
source "https://rubygems.org"
ruby "3.2.3", engine: "spinel", engine_version: "0.0.0"

gem "toy", path: "../toy"
EOF

# 2. Resolve (CRuby + Bundler).
bundle lock

# 3. Vendor — copies lib/ AND builds toy's native archives INTO vendor/
#    (ggml via its cmake build-unit + vendor-patches, tinynn via make).
SPINEL_DIR=~/sites/spinel ~/sites/spinelgems/exe/spinel-compat vendor

# 4. Compile from the project root (the -L flags are project-relative).
spinel experiment.rb -o experiment
./experiment
```

After step 3 you should see, with **zero absolute paths** into toy's checkout:

```
vendor/spinel/toy/lib/…                              # the primitives
vendor/spinel/toy/tinynn/libtinynn_ggml.a            # the shim archive
vendor/spinel/toy/vendor/ggml/build/src/libggml*.a   # 3 CPU archives
```

and the vendored `lib/toy/ffi/tinynn.rb` carries
`-Lvendor/spinel/toy/tinynn -Lvendor/spinel/toy/vendor/ggml/build/src …`.
You can move the whole project directory and recompile; it keeps working.
If the vendor step warns that a placeholder matched **no** vendored `.rb`,
toy's dev `ffi_cflags` line drifted from its manifest placeholder — that
warning is systemic (spinelgems), not a per-gem canary; report it.

A typical `experiment.rb`. One require — `toy/compute` (toy#42) — pulls the
whole composition surface (all three engines + recipes + GGUF/tokenizer
loaders), so you no longer hand-`require_relative` each primitive:

```ruby
require_relative "vendor/spinel/toy/lib/toy/compute"   # the full compute API

cfg    = Toy::SmolLM2Config.mha(627, 64, 4, 128, 2, 32, 10000.0, 1.0e-5)
recipe = Toy::LLM::Recipes::FromScratch.new
recipe.realize!(cfg, 32, 1, 0, false, false, 0, 1.0)
# … Toy::Labels.next_token, Toy::AdamW.new.hp(0), recipe.step!, …
```

`toy/compute` is the **Spinel-only** entry — it pulls `tinynn` (FFI) and the
engines. The top-level `toy.rb` stays MRI-safe (Mat + Card + version, no FFI)
so `bin/toy` can require it; don't use `toy.rb` for the compute surface. If you
build only one engine and want to compile less (Spinel has no tree-shaking),
require that engine's file directly instead of `toy/compute`. The
`gate-compute-surface` make target proves the one-require surface co-compiles +
runs a from-scratch training step (`examples/smoke_compute_surface`).

> One exception: `toy/compute` does **not** pull `recipes/lora` — loading it
> uncalled trips a Spinel analyzer bug that poisons `cfg` to poly program-wide
> ([spinel-dev#11](https://github.com/OriPekelman/spinel-dev/issues/11);
> re-add tracked by toy#52). A consumer that trains LoRA requires
> `toy/llm/recipes/lora` itself (and calls it, which pins the type).

From here you write your experiment loop against
`Toy::LLM::Engine::LlamaSeqEngine` / `Toy::LLM::Engine::ViTTinyEngine` /
`SmolLM2KVFFICache` / the GGUF loaders. `~/sites/tao_transfer` is the canonical
worked example. To emit Tao-consumable telemetry, see [`events.md`](events.md)
(and `SpinelKit::Json::Builder` / `Toy::Events.add_provenance` for building it).

### Updating when toy moves

Re-run `bundle lock` → `spinel-compat vendor` → recompile. The vendor step
rebuilds the native units in your tree (a `path:` dev checkout that is already
patched is detected; patches are not re-applied).

### CUDA / Metal (opt-in; staged)

The GPU backends are authored as **default-disabled** build-units in
`spinel-ext-gpu.json` (CUDA validated on gx10; Metal structural,
Mac-validation-pending — toy#27). They stay staged until spinelgems grows the
opt-in + variant-build-dir mechanics —
[spinelgems#20](https://github.com/OriPekelman/spinelgems/issues/20). Until
then, a CUDA consumer replicates what the staged entries declare: configure
ggml's `build-cuda/` with the entry's cmake args inside
`vendor/spinel/toy/vendor/ggml/`, `make -C vendor/spinel/toy/tinynn cuda`, and
substitute the `lib/toy/ffi/tinynn_cuda.rb` placeholder with the vendored-relative
`-L` set (see the `link` template in `spinel-ext-gpu.json`). CUDA compiles
need `spinel --cc='cc -Wl,-u,tnn_cuda_force_link' …` — without the force-link
symbol the linker drops the CUDA backend registration and ggml silently
computes on CPU (verified: the loss curve flips to the CPU fixture).

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

## History

The pre-#45 flow needed a per-consumer post-vendor hook (`post_vendor_toy.rb`)
that rewrote vendored `-L` flags to **absolute paths into toy's checkout**,
plus a `CURRENT_FFI_CFLAGS` canary in `lib/toy/ffi_manifest.rb`. Both are
retired: the hook by the build-units above, the canary by the vendor step's
zero-substitution warning. Consumers still carrying the hook: delete it and
re-vendor (tao_transfer: see OriPekelman/tao#8).

## See also
- [`cli.md`](cli.md) — the friendly *app* surface (run models without any of this).
- [`events.md`](events.md) — the `toy/v1` event contract your consumer can emit.
- [`dependencies.md`](dependencies.md) — how toy itself consumes tep (the model toy is following).
- toy#19 / toy#42 / toy#45, tep#97, spinelgems#3 / #14 / #20 — the issue trail.
