# vendor-patches/

Local patches applied to `vendor/ggml/` after the `make setup-ggml`
clone. The Makefile's `$(GGML_DIR)/.patched` target applies them in
filename order; each `make setup-ggml` / `make setup-ggml-cuda` invocation
checks the sentinel file and re-applies if needed. Patches are
idempotent (the rule skips one that's already applied).

These patches are local-only deltas on top of vendored upstream
`ggml@e484d08` (HEAD at first vendor). The goal is to upstream them
eventually — see each patch's commit message for the discussion notes.

## What's in here

| Patch | Touches | Why |
|---|---|---|
| `0001-cuda-buffer_from_ptr.patch`               | `include/ggml-cuda.h`, `src/ggml-cuda/ggml-cuda.cu` | BYO-pointer mmap path for the CUDA backend. Mirrors `ggml_backend_cpu_buffer_from_ptr`. Required for Phase 2 mmap inference on CUDA (per `docs/archive/cuda-byo-pointer-design.md`). |
| `0002-cuda-buffer_from_ptr-reuse-iface.patch`   | `src/ggml-cuda/ggml-cuda.cu` | Refactor on top of 0001: reuse the standard CUDA buffer interface, drop the read-only flag pattern. |
| `0003-cuda-buffer_from_ptr-copy-mode.patch`     | `src/ggml-cuda/ggml-cuda.cu` | Adds `GGML_CUDA_BYO_PTR_MODE=copy` for non-unified-memory SKUs that need a one-time host→device copy. UVA path (GB10 / DGX Spark / Jetson) stays zero-copy. |
| `0004-cuda-cpy-strided.patch`                   | `src/ggml-cuda/cpy.cu`       | Guard `cpy_scalar_transpose` dispatch behind a contiguous-dst check. Without it, KV-cache writes that go into a strided `view_2d` (the canonical pattern) silently miswrite. F32 KV-cache CUDA path was broken before this. |
| `0005-concat-backward.patch`                    | `src/ggml.c`                 | Adds `case GGML_OP_CONCAT` to `ggml_compute_backward`. Without it autograd aborts on SmolLM2 attention (per-head concat before the O projection). Required for F1.2 LoRA training. |
| `0006-getrows-back-large-vocab.patch`           | `src/ggml-cuda/getrows.cu`   | Chunks the `get_rows_back` kernel launch. The original code set `gridDim.y = vocab`; CUDA caps that at 65535 so Qwen-class vocabs (V=151936) aborted training with "invalid argument". Required for F3 with embedding training on Qwen-class models. |
| `0007-gpt2-backward-kernels.patch`              | `include/ggml.h`, `src/ggml.c`, `src/ggml-alloc.c`, `src/ggml-backend-meta.cpp`, `src/ggml-cpu/{ops.cpp,ops.h,vec.h,ggml-cpu.c}` | Two dedicated fused backward kernels for GPT-2 train-from-scratch (CPU reference): `GGML_OP_GELU_BACK` (autograd of `GGML_UNARY_OP_GELU`) and `GGML_OP_NORM_BACK` (autograd of `GGML_OP_NORM`, the pure normalize, no affine). Without them autograd aborts on GPT-2's GELU FFN / LayerNorm. 2 of the 3 gaps in our upstream ggml#1514; finite-diff validated via `tinynn/gpt2_backward_probe.c`. See `docs/notes/gpt2-backward-patches.md`. |

## How the application works

`Makefile`:

```
GGML_PATCHES := vendor-patches/0001-*.patch  vendor-patches/0002-*.patch  ...

$(GGML_DIR)/.patched: $(GGML_DIR)/CMakeLists.txt $(GGML_PATCHES)
    @for p in $(GGML_PATCHES); do \
      cd $(GGML_DIR) && git apply --check $$p \
        || (cd $(GGML_DIR) && git apply --check --reverse $$p && continue); \
      cd $(GGML_DIR) && git apply $$p; \
    done
    touch $@
```

`setup-ggml` and `setup-ggml-cuda` depend on `$(GGML_DIR)/.patched`,
which depends on the patch files themselves — adding a new patch (or
editing one) re-runs the apply step on the next build.

If the upstream ggml moves and a patch no longer applies cleanly,
either:
1. Update the patch file (regenerate from a manual rebase), or
2. Delete `vendor/ggml` entirely and `make setup-ggml` again from scratch.

## How to regenerate a patch (when it drifts)

Each patch was originally extracted with `git format-patch -1 <commit>`
from a local working tree where the patches were applied as proper
commits. If a patch needs editing:

1. Apply the existing patch series locally.
2. Make your edit; commit it inside `vendor/ggml/`.
3. `git format-patch -1 HEAD --stdout > vendor-patches/000N-name.patch`
   to overwrite the file.
4. Reset the vendor tree to clean upstream + re-run `make setup-ggml`
   to verify the new patch applies idempotently.

For the concat-backward case where the change was an uncommitted edit
(`0005`), `git diff <file>` was used directly. Either format is fine
— `git apply` accepts both.

## Upstream-ability

Each patch has a doc:
- `docs/archive/concat-back-patch-2026-05-21.md` (0005)
- `docs/archive/cuda-byo-pointer-design.md` (0001-0003)
- `docs/archive/ggml-cpy-patch-2026-05-18.md` (0004 — referenced from memory; not separately archived)
- `vendor-patches/0006-getrows-back-large-vocab.patch` carries its own
  rationale in the commit message.

When the F1/F2/F3 work stabilises, propose each as a separate ggml-org/ggml PR.
