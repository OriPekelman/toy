# Gating: bit-identical discipline

The spine of the framework is a simple rule: **every layer and every runner
is gated bit-identical against a frozen reference, and silent fallbacks are
forbidden.** When something cannot match, it fails loud rather than degrading
quietly. A refactor that changes a number is a regression, not a variation.

## Principle

The five-layer algorithm stack (L1 primitives → L2 blocks → L3 archs →
L4 recipes, see `docs/architecture.md`) is refactored under a single
constraint: the decomposed path must reproduce the inlined reference's output
**byte for byte**. The same applies to the four compute runners
(`infer` / `train` / `eval` / `serve`) bridging CRuby to the Spinel-compiled
`libexec/` binaries.

Two failure modes are explicitly rejected:

- **Silent numerical drift** — a loss curve that "looks close" is a failure.
  Gates assert exact equality (loss strings, decoded token lines, logprobs).
- **Silent capability fallback** — a missing kernel, an unbound GGUF key, a
  tokenizer UNK, or an FFI error must surface with context, never be masked.

## Smoke fixtures

The gate fixtures live in `prep/smokes/` as Spinel-compiled smoke binaries
(each with a `.rb` source built via its `make prep/smokes/<name>` target).
They run pure-Ruby CPU, SEED=0, F32 weights — deterministic by construction.
(They are gates, not tutorials — the narrated tutorials live in `examples/`.)

**Layer-decomposition gates** — prove a refactored path equals the inlined
reference loss curve:

- `prep/smokes/smoke_projection_lens` (+ `_cuda`, `_metal`) — the frozen
  reference forward+backward loss curve; the CUDA/Metal mirrors gate the
  generated backends against it.
- `prep/smokes/smoke_recipe_from_scratch` — drives the same random-init config
  through `Toy::LLM::Recipes::FromScratch`; its `step N: loss=` lines must
  byte-equal the projection-lens reference.
- `prep/smokes/smoke_recipe_lora`, `prep/smokes/smoke_recipe_warm_start` — the
  same discipline for the LoRA and warm-start recipes.

**Realize-divergence gates** — exercise graph slices the bulk-realize path
must not silently unify:

- `prep/smokes/smoke_gate_gqa_divergent` — the `n_heads*head_dim != d_model`
  case where `w_o` is non-square.
- `prep/smokes/smoke_gate_b_gt_1` — batch dimension > 1.
- `prep/smokes/smoke_gate_qkv_bias` — separate Q/K/V bias tensors.
- `prep/smokes/smoke_gate_q8_preserve` — Q8_0 weights stay Q8_0 (no silent
  dequant-to-F32 on the realize path).
- `prep/smokes/smoke_gate_llama3_tensor` — Llama-3 tensor layout.

**Runner / format gates:**

- `prep/smokes/smoke_gguf_roundtrip` — GGUF write→read bit-identity.
- `prep/smokes/smoke_decode_logprobs` — per-token logprob decode.

Build and run any fixture directly, e.g.:

```
make prep/smokes/smoke_recipe_from_scratch
SEED=0 STEPS=5 ./prep/smokes/smoke_recipe_from_scratch | grep '^step'
```

## Runner harnesses

The four compute runners are gated by `prep/{infer,train,eval,serve}_gate.rb`
against committed baselines in `prep/fixtures/`
(`{infer,train,eval,serve}_baseline.txt`). Each harness drives the real
`bin/toy` command end to end — building its `libexec/` runner if needed — and
asserts the output equals the recorded baseline byte for byte:

```
ruby prep/infer_gate.rb   # exit 0 on byte-for-byte match, 1 otherwise
```

`infer`/`train`/`eval` are tep-free. `serve_gate.rb` covers the
OpenAI-compatible HTTP path.

### Tolerance gates (numerics-varying)

Most gates are byte-exact. A few exercise a path where the numerics legitimately
change, so they assert a relative tolerance plus the discrete/structural
invariants (per the cross-platform-gate rule: discrete strict, numerical
tolerance):

- `make gate-mixed-precision` (`prep/mixed_precision_gate.rb`) — GH#9 f16
  train-from-scratch on CPU. Drives the from-scratch example at `WEIGHT_DTYPE=1`
  vs `=0` and asserts f16 runs to completion (depends on the `0008`
  mul_mat-backward ggml patch — without it backward aborts in sched-alloc),
  `run_start.model.weight_type` surfaces the dtype, and the f16 final loss is
  within 5% of the f32 baseline (~0.2% observed). bf16 is the CUDA/GB10
  follow-up (bf16 backward needs more than f16).

### Consumer cold-start gate

`make gate-consumer` (`prep/consumer_gate.rb`) proves the README /
`docs/framework.md` quickstart is TESTED: in a throwaway tmpdir it runs
`toy new gatelab` (asserting the scaffold seeds `data/ts_seqs.{bin,txt}`),
compiles `algos/recipes/hello.rb` with `$SPINEL_DIR`'s spinel, runs it with
default ENV and again with `D_MODEL=128 STEPS=10` (no recompile — one
compile, many runs), runs `toy train from-scratch` in the project (losses
print + `runs/<id>/events.jsonl` written), and asserts the missing-corpus
guard fails loud naming the path (the spinel-dev#17 class). The `--lib`
leg (`toy new gatelib --lib` → `bundle lock` → `spinel-compat vendor` →
`./build.sh cpu` → run) skips loudly when bundler / spinel-compat are
absent. Structural assertions (files exist, losses finite, curves react
to ENV), not byte-exact — numerics are `train_gate`'s job.

### MRI dev-run gate

`make gate-mri` (`prep/mri_gate.rb`, toy#71) proves the **plain-CRuby
entry** works with no Spinel anywhere — plain `ruby`, no `SPINEL_DIR`, no
Spinel build. Two subprocess legs:

**Stub leg** (Stage A, forced `TOY_MRI_NATIVE=0`): (1) `require
"toy/mri"` loads the full compute require chain under MRI (the
`ffi_lib`/`ffi_func`/`ffi_cflags` intrinsics resolve to the shim's
declaration recorders; 194 declarations land in `Toy::MRI.declarations`);
(2) the pure-Ruby teaching surface genuinely works — configs,
`RecipeOptions`, engine *construction*, and a real `Toy::Trainer` +
`TransformerLM` loop on the Mat path whose loss decreases; (3) crossing
the native boundary fails LOUD with the named `Toy::MRI::NativeCallError`
(never a bare `NoMethodError`).

**Native leg** (Stage B, the CRuby oracle; needs `make libtinynn_shared`
— loud SKIP when the `.so` is absent, `MRI_GATE_STRICT=1` turns the skip
into a failure): (4) the Fiddle arm binds and a real cpu session opens;
(5) **differential train** — the exact from-scratch gate shape run under
MRI+Fiddle must reproduce `prep/fixtures/train_baseline.txt` (the same
recorded curve `train_gate` holds the Spinel runner to) with losses
**bit-exact** (`pack("G")` comparison; formatted lines additionally
byte-compared, string-only drift reported not failed); (6) **KV decode**
— smollm2-135m greedy ids byte-equal `prep/fixtures/infer_baseline.txt`
(model-gated: loud SKIP without the gitignored GGUF). One program, two
Ruby runtimes, same C library — divergence isolates Spinel codegen
(spinel-dev#6 phase 1).

### Model-gated regression gates

Some gates exercise a REAL model whose GGUF is a gitignored multi-GB dev
artifact (present on gx10 / Mac dev boxes, absent from a fixture-only
checkout). These SKIP loudly (exit 0) when the model is missing rather than
failing CI:

- `make gate-full-finetune` (`prep/full_finetune_gate.rb`) — needs
  `data/smollm2-135m-native.gguf`; byte-exact engine full-finetune CE curve.
- `make gate-moe-kquant` (`prep/moe_kquant_gate.rb`) — needs
  `data/OLMoE-1b-7b-0924-Instruct-Q4_K_M.gguf` (~4 GB). Regression guard for
  the K-quant MoE attention head-stride collapse (`head_nbytes` returning 0 for
  K-quants — the bug long misfiled as ggml#1506; see
  `docs/notes/mul_mat_id_quants.md`). STRUCTURAL assertion (distinct-id count +
  max single-token run on a greedy OLMoE Q4_K_M decode), not byte-exact, so it
  guards "attention didn't collapse" without false-alarming on K-quant drift.

## Poly-degradation gate

`make gate-poly-degrade` (`prep/poly_degrade_gate.rb`) guards the most dangerous
Spinel failure mode: whole-program inference can't resolve a call's receiver,
emits `cannot resolve … on poly … (emitting 0)`, and compiles a literal `0` into
a numerical path — the binary builds clean and exits 0 but the result is silently
wrong (`compiled != correct`). It compiles the canonical compute entrypoints
(`train` / `train_gpt2` / `infer` / `eval` / `eval_lmc`) and fails on any **new**
emit-0 warning vs the frozen baseline (`prep/fixtures/poly_degrade_baseline.txt`,
the known-benign dead-path set). A missing/colliding `require` or an unconstrained
param that re-polys a hot path trips it immediately. Re-record the benign set with
`ruby prep/poly_degrade_gate.rb --record`. (Issue #32; see
`feedback_spinel_type_inference_landmines` for the landmine catalogue.)

## Mirror gate

The CPU algorithm file is the single source of truth; the CUDA and Metal
mirrors for L1/L2/L3 are auto-generated by `prep/gen_cuda_mirror.rb` from
`MIRRORABLE` markers. The gate:

```
make verify-mirrors
```

regenerates the mirrors in memory and exits non-zero if any committed mirror
has drifted from what the generator would produce. Hand-editing a generated
mirror is a gate failure; edit the CPU source and run `make gen-mirrors`.

## Vendored ggml patches

Nine local patches sit on top of vendored upstream `ggml@41e7949`, in
`vendor-patches/`. They are applied in filename order to `vendor/ggml/` by the
Makefile's `$(GGML_DIR)/.patched` sentinel target, which `setup-ggml` /
`setup-ggml-cuda` / `setup-ggml-metal` depend on through `CMakeLists.txt`.
A fresh clone applies them exactly once; re-running `make setup-ggml` is a
no-op while the patch set is unchanged. The rule resets the vendor tree to
upstream HEAD and re-applies, so editing or adding a patch re-runs the step on
the next build.

| Patch | Touches | Gates |
|---|---|---|
| `0001-cuda-buffer_from_ptr.patch` | `include/ggml-cuda.h`, `src/ggml-cuda/ggml-cuda.cu` | BYO-pointer mmap path for CUDA (mirror of the CPU `buffer_from_ptr`). |
| `0002-cuda-buffer_from_ptr-reuse-iface.patch` | `src/ggml-cuda/ggml-cuda.cu` | Refactor on 0001: reuse the standard CUDA buffer interface. |
| `0003-cuda-buffer_from_ptr-copy-mode.patch` | `src/ggml-cuda/ggml-cuda.cu` | `GGML_CUDA_BYO_PTR_MODE=copy` for non-UVA SKUs; UVA (GB10) stays zero-copy. |
| `0004-cuda-cpy-strided.patch` | `src/ggml-cuda/cpy.cu` | **CUDA KV-cache bit-identity** — guards `cpy_scalar_transpose` behind a contiguous-dst check; without it, KV writes into a strided `view_2d` silently miswrite. |
| `0005-concat-backward.patch` | `src/ggml.c` | **Training** — adds the `GGML_OP_CONCAT` case to `ggml_compute_backward`; live at `vendor/ggml/src/ggml.c:6514`. Without it autograd aborts on per-head concat before the O projection. |
| `0006-getrows-back-large-vocab.patch` | `src/ggml-cuda/getrows.cu` | **Training (large vocab)** — chunks the `get_rows_back` launch; the original `gridDim.y = vocab` aborts for Qwen-class vocabs (V > 65535). |
| `0007-gpt2-backward-kernels.patch` | `include/ggml.h`, `src/ggml.c`, `src/ggml-cpu/*` | **GPT-2 training** — `gelu_back` + `norm_back` (our ggml#1514); without them GPT-2 backward aborts. |
| `0008-mul-mat-backward-mixed-precision.patch` | `src/ggml.c` | **f16 training (GH#9)** — mul_mat backward for non-f32 weight dtypes; gated by `make gate-mixed-precision`. |
| `0009-sched-unsupported-node-diagnostic.patch` | `src/ggml-backend.cpp` | **Diagnostics** — names the offending node when sched-alloc hits an unsupported op instead of a bare abort. |

Patches 0001–0003 (the CUDA BYO-pointer path) are documented in
`docs/reference/memory-design.md`. Patch 0005 (concat-backward) has its
historical write-up in `docs/archive/concat-back-patch-2026-05-21.md`.
Patch 0006 carries its rationale in its own commit message.

### Regenerate on drift

If upstream ggml moves and a patch no longer applies:

1. Apply the existing series, make the edit, commit it inside `vendor/ggml/`,
   then `git format-patch -1 HEAD --stdout > vendor-patches/000N-name.patch`;
   or
2. delete `vendor/ggml` entirely and `make setup-ggml` again from scratch.

The upstream goal is to propose each patch as a separate ggml-org/ggml PR once
the work stabilises.

## CUDA / GGUF notes

- The CUDA mirrors (`*_cuda`) and Metal mirrors (`*_metal`) are gated against
  the same CPU reference loss curve via the projection-lens fixtures, so a
  backend that diverges fails immediately rather than producing
  "approximately correct" output.
- GGUF round-trip and Q8-preserve fixtures guarantee that quantized weights
  are not silently widened to F32 on load or on realize — a Q8_0 weight stays
  Q8_0 through the graph.

For known numerical-divergence issues currently masked rather than fixed
(CPU/CUDA LoRA-train grad aliasing), see the "Known issues" notes in
`docs/roadmap.md`. (The MoE K-quant `mul_mat_id` "bug" was resolved — it was
`head_nbytes` collapsing K-quant attention heads, not the op; guarded by
`make gate-moe-kquant`.)
