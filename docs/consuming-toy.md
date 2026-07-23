# Consuming `toy` as a library

**Audience:** maintainers of research projects (e.g. `tao_transfer`) who want to
**compose toy's primitives** — engines, KV caches, GGUF loaders — into their own
Spinel-compiled experiment, rather than fork toy or hand-path `require_relative`s
into toy's tree.

> Not what you want? If you just want to **run** models (infer / train / eval /
> serve a GGUF), use the `toy` **CLI** — `toy new`, `toy infer model.gguf`, … —
> which is clean and self-contained. See [`cli.md`](cli.md). This doc is the
> *library-composition* surface.

## Recommended Spinel revision

toy is verified against **Spinel master**, plain rev pin:

```sh
git clone https://github.com/matz/spinel && cd spinel
git checkout ecac633caa
make deps && make
```

At `ecac633caa` (verified 2026-07-23, gx10/GB10): the full gate matrix
passes — 31 of 33 legs green including all four training recipes
byte-exact (also under `SPINEL_GC_STRESS=1`), infer/eval CPU+CUDA,
GPT-2 minimal + engine trainers (CPU and CUDA), compute-surface, the
8 GDN/Dragon gates, the MLA/MoE gates, serve, consumer cold-start, and
poly-degrade. The one exclusion is a documented, pin-independent red:
mixed-precision tolerance (toy#108, identical at the previous pin).
Do NOT pin past this rev for now: the ~12 commits immediately after it
(`ecac633caa..16d6433c`) carry a 200×+ whole-program analysis blowup
(matz/spinel#3287 — toy's serve unit goes 23 s → 75+ min); this pin is
deliberately the last fast rev that includes the #3256 fix. Once #3287
is fixed upstream, the pin advances per the usual sweep protocol
(toy#95) — pin exactly this rev for byte-reproducibility.

Previous pin: `f6a189e8` (verified 2026-07-10, 28/28). The 2026-07-23
advance crossed the matz/spinel#3256 fix (boxed `sp_RbVal` into
`const char *`, hit by poly-degrade's bare compiles) and rode the
`Device.kind` rename (matz/spinel#3285 — builtin `Module#name` shadows
user overrides; toy no longer collides). tep's pin (`fd87f45b`,
tep PR #243) is one day behind this rev on the same master line.

Historical note: v0.8.0–v0.9.0 were verified against the
`toy-v0.8.0-pin` union branch (`fbb9beb` = spinel-dev#13 + #14 fixes,
26 gates, 2026-06-12). Master absorbed everything that branch carried;
the closing arc — matz/spinel#1855 (`:int_array` poly marshalling),
#1845 (`[[build]]` mechanics), #1867 (object-array narrowing across
instance-method edges) — is recorded on toy#95 and toy#107.

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
runs a from-scratch training step (`prep/smokes/smoke_compute_surface`).

> `toy/compute` pulls **all four** recipes, LoRA included (the historical
> lora exclusion — a Spinel uncalled-forwarder/constructor-slot facet,
> [spinel-dev#12](https://github.com/OriPekelman/spinel-dev/issues/12) —
> was dissolved by the toy#64 RecipeOptions reshape and closed as toy#52;
> the compute-surface gate keeps an uncalled-LoRA tripwire against
> regressions).

From here you write your experiment loop against
`Toy::LLM::Engine::LlamaSeqEngine` / `Toy::LLM::Engine::ViTTinyEngine` /
`SmolLM2KVFFICache` / the GGUF loaders. `~/sites/tao_transfer` is the canonical
worked example. To emit Tao-consumable telemetry, see [`events.md`](events.md)
(and `SpinelKit::Json::Builder` / `Toy::Events.add_provenance` for building it).

### Updating when toy moves

Re-run `bundle lock` → `spinel-compat vendor` → recompile. The vendor step
rebuilds the native units in your tree (a `path:` dev checkout that is already
patched is detected; patches are not re-applied).

### CUDA / Metal (opt-in build-units)

The GPU backends are **default-disabled** build-units in `spinel-ext.json`
(merged from the staged `spinel-ext-gpu.json` in toy#70, after
[spinelgems#20](https://github.com/OriPekelman/spinelgems/issues/20) shipped
the opt-in + variant-build-dir mechanics). A plain `spinel-compat vendor`
never attempts them — no CUDA cmake configure on a CUDA-less box; the
vendored `ffi_cflags` GPU lines are left byte-identical to the dev tree
(`disabled_cflags` = the dev line itself). Enable them at vendor time:

```sh
# CUDA (both units: the ggml build-cuda/ tree + the tinynn shim archive)
spinel-compat vendor --with-ext cuda --with-ext cuda-shim
# Metal (macOS)
spinel-compat vendor --with-ext metal --with-ext metal-shim
# or the env form: SPINEL_EXT_ENABLE=cuda,cuda-shim spinel-compat vendor
```

With the units enabled, `build-cuda/` (or `build-metal/`) artifacts appear
**alongside** the CPU unit's `build/` (copy-once shared source: the CPU
archives survive), and the `lib/toy/ffi/tinynn_cuda.rb` /
`tinynn_metal.rb` placeholder is substituted from the entry's `link`
template with project-relative `-L` flags. Then build the GPU binary —
CUDA compiles need `spinel --cc='cc -Wl,-u,tnn_cuda_force_link' …`
(the `toy new --lib` scaffold's `./build.sh cuda` does this): without the
force-link symbol the linker drops the CUDA backend registration and ggml
silently computes on CPU (verified: the loss curve flips to the CPU
fixture).

> **`path:` consumers — generated CUDA mirrors landmine.** The CUDA
> engine/recipe files (`*_cuda.rb` under `lib/toy/llm/`) are GENERATED
> (`make gen-mirrors`, prep/gen_cuda_mirror.rb) and gitignored. A fresh
> toy clone that never ran make has none on disk, the vendor step copies
> an incomplete `lib/`, and Spinel **silently compiles the missing
> `require_relative`s to nothing** — the symptom is a bizarre
> mis-resolution at the call site (observed: `recipe.step!` resolving to
> `Toy::RunBundle#step!`, "too many arguments"), NOT a missing-file
> error. Run `make gen-mirrors` in the toy checkout before vendoring
> with the CUDA units enabled. Published-gem consumers are unaffected
> (`make gem-prep` ships the mirrors).

Status: CUDA re-validated end-to-end through `vendor --with-ext` on gx10
(GB10 sm_121, CUDA 13.0) — toy#70; the loss curve reproduces
`prep/fixtures/train_cuda_baseline.txt`. Metal is **structural only**,
Mac-validation-pending (toy#27). Known macOS blocker for ANY cmake
build-unit (CPU included): the Vendorer's bare-env cmake lacks the SDK
libc++ include path (`fatal error: 'array' file not found`) — being fixed
tool-side as
[spinelgems#21](https://github.com/OriPekelman/spinelgems/issues/21); toy
deliberately does NOT work around it in its manifest. The CUDA unit pins
`CMAKE_CUDA_ARCHITECTURES=121` (GB10); other GPUs override the whole link
line via the `SPINEL_EXT_<PLACEHOLDER>` escape hatch.

## MRI dev-runs: `require "toy/mri"` (toy#71)

Plain CRuby can load — and, with one `make` target, **run** — the whole
compute surface; no Spinel:

```sh
ruby -Ilib -e 'require "toy/mri"'        # from a toy checkout
# or, in a consumer with toy on the load path:
ruby -e 'require "toy/mri"; ...'
```

`lib/toy/mri.rb` is the **MRI-only entry**: it gives the Spinel FFI
intrinsics (`ffi_lib` / `ffi_func` / `ffi_cflags`) a CRuby meaning, then
requires `toy/compute` unchanged. It is never compiled by Spinel and
must never appear in a compiled require chain (compiled code requires
`toy/compute*` directly). It has **two arms**, selected once at require
time:

- **Stub arm (Stage A — the default without a build).** `ffi_func`
  declarations are recorded (`Toy::MRI.declarations`) and the generated
  methods raise a named `Toy::MRI::NativeCallError` — "native call
  `tnn_session_new` requires the Spinel-compiled binary or the fiddle
  backend (`make libtinynn_shared`) — toy#71". Never a bare
  `NoMethodError`, never a silent wrong answer. Pure Ruby still works
  for real: configs (`SmolLM2Config.tiny/.mid`), `RecipeOptions`, cards,
  `RunLog`, the Mat path, and the whole teaching stack —
  `TransformerLM` + `Toy::Trainer` genuinely train under MRI (slowly;
  tiny shapes). This is what livebook / `tao notebook` dev-run cells
  use out of the box (a one-line stderr hint points at the native arm).

- **Native arm (Stage B — the CRuby oracle).** Build the shared
  library once:

  ```sh
  make libtinynn_shared      # → tinynn/libtinynn_ggml_shared.so
                             #   (needs `make setup-ggml` first;
                             #    Linux/CPU this stage, gitignored)
  ```

  When that artifact exists, `require "toy/mri"` dlopens it via
  **Fiddle** and every `ffi_func` declaration binds to the real C
  function — same 186-symbol surface the Spinel binaries link. MRI then
  runs *real* training and KV-decode inference at ggml speed:
  `Toy::LLM::Recipes::FromScratch`, GGUF load, `ToyLM#generate`, corpus
  loaders, the lot. `TOY_MRI_NATIVE=0` forces the stub arm back.
  The Stage A off-state carve-outs (`tnn_null_ptr`→nil, trace
  off-state) disappear in this arm — those functions go through Fiddle
  like everything else.

**Why "oracle":** the MRI+Fiddle arm and the Spinel-compiled binaries
run the *same Ruby program* over the *same C library* — only the Ruby
runtime differs. `make gate-mri`'s native leg pins this: the 5-step
from-scratch loss curve reproduces the recorded Spinel gate curve
**bit-exact** (`prep/fixtures/train_baseline.txt` — values *and*
formatted strings), and smollm2-135m greedy decode ids byte-equal
`prep/fixtures/infer_baseline.txt`. That makes any future MRI-vs-Spinel
divergence a **differential bisection oracle for Spinel codegen bugs**
(spinel-dev#6 phase 1): run the suspect program both ways; if MRI+Fiddle
agrees with the recorded truth and the compiled binary doesn't, the bug
is in compilation, by construction — no numerics argument needed.

Type lowering (pinned by probing Spinel's generated C): `:ptr`→`void*`
(integer-valued handles on the Ruby side), `:int`→`int`, `:long`→`long`,
`:size_t`→`size_t`, `:double`→`double`, `:str`→`const char*`,
`:int_array`→`const int64_t*`, `:float_array`→`const double*` (Spinel
arrays store i64/f64 — *not* f32). The Fiddle arm mirrors Spinel's
zero-copy array semantics by pack→call→unpack→`Array#replace`, so
C-filled output arrays (`tnn_read_i32_file`, `tnn_download_to_f64_array`)
work unchanged.

Gated by `make gate-mri` (plain `ruby`; see `gating.md`): stub leg
always; native leg when the `.so` exists (loud SKIP otherwise;
`MRI_GATE_STRICT=1` makes the skip a failure). CUDA/Metal shared
variants and the macOS dylib are follow-ups.

## RBS type roots ride along automatically (toy#69)

Toy ships `sig/*.rbs` covering its public surface — the pure-Ruby
models (`sig/toy.rbs`: Mat, the `Toy::` blocks, SmolLM2, GPT2LM,
Tokenizer) **and** the Spinel compute surface (`sig/toy_compute.rbs`:
`RecipeOptions`, `TrainingBatch`, `AdamW`, `Labels`, the recipes'
`realize!`/`step!`, the engines' scalar config + ptr-free methods,
`ViTTinyConfig`, the KV decode classes, `GGUFLoad::SmolLM2Flags`).

`spinel-compat vendor` copies each gem's `sig/` and aggregates ONE root
at `vendor/spinel/sig` (spinelgems#13), advertised in the generated
`vendor/spinel/deps.rb` header. Compile with it:

```sh
spinel --rbs vendor/spinel/sig experiment.rb -o experiment
```

(The `toy new --lib` scaffold's `build.sh` does this automatically when
the directory exists.) The seeds are **advisory** — Spinel's inference
runs on top and widens on observed contradiction — but they pin a
method's param/ivar/return types **without needing a call site**, which
defends against the uncalled-param poly-widening family (spinel-dev#11/
#12): a toy method your experiment never calls stays concretely typed
instead of unioning poly into shared callees.

Two honest limits (probed on spinel a699cf9):

- **FFI `:ptr` has no RBS spelling.** Methods with a ptr param or return
  (the `realize_for_mmap` family, tensor-handle accessors) can't be
  declared. (This briefly mattered for the historical lora-in-compute
  exclusion — spinel-dev#12 / toy#52, since dissolved by the toy#64
  reshape — and still means ptr-surfaces rely on call-site inference.)
- **Arity must be exact or absent.** A truncated declaration against a
  default-arg method is NOT a no-op: it breaks the C compile with a
  "too few arguments" error (the one non-advisory failure mode; toy's
  `Tokenizer#initialize` hit this during adoption).

## The `sp_Mat undeclared` landmine

If your consumer compile fails with `sp_Mat undeclared` (or a poly-dispatch crash
around an `Array<Mat>`), you've hand-pathed a `require_relative` into toy's tree
instead of using the vendored layout. Root cause: Spinel's whole-program
inference was **require-path/layout sensitive** — `spinel-flatten` deduplicated
inlined files by exact resolved-path *string*, so loading `transformer.rb` (where
`Mat` lives) via a non-canonical path (`.../models/../models/transformer`) compared
unequal to toy's own canonical require and got inlined **twice**, fragmenting the
type table differently than toy's own build. The vendored layout (same tree shape
for every consumer) sidesteps this by giving every consumer one canonical path.
Filed as **matz/spinel#1367**.

**Fixed on current Spinel master (≥ `cbde0f88`, fix #1601):** `spinel-flatten` now
canonicalizes `.`/`..` segments in `resolve`, so a file always maps to one string
and is inlined once regardless of the require path it's reached through. On master
the hand-path consumer flattens to a single `class Mat` and emits a fully-declared
`sp_Mat` — no `undeclared`, no poly-dispatch crash. (The name-collision axis is
covered separately by #1425.) If you're on an engine at or past that fix, the
landmine is gone; the vendored layout is still the recommended path for the native
archive / `-L` flag wiring, not for inference correctness. toy's
`make gate-poly-degrade` guards regressions of the related emit-0 class.

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
