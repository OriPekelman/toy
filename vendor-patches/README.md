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
| `0008-mul-mat-backward-mixed-precision.patch`   | `src/ggml.c`                 | Mixed-precision training (GH#9): the `MUL_MAT` backward emits the **activation** gradient as `out_prod(src0=weight, …)`, but `OUT_PROD` only implements an f32 `src0` (CPU `supports_op`; CUDA out_prod is f32-only). With f16/bf16 weight storage `src0` is low-precision → backward sched-alloc aborts (`*cur_backend_id == -1`). The fix branches on `src0->type`: non-f32 weights take the algebraically-equivalent `mul_mat(cont(transpose(src0)), cont(grad))` form (already present upstream as a comment) — `MUL_MAT` supports f16/bf16 `src0` + f32 `src1` on every backend, with tensor-core throughput on CUDA. f32 weights keep the original out_prod fast path. f16 train-from-scratch converges within ~0.2% of the f32 baseline. |
| `0009-sched-unsupported-node-diagnostic.patch`  | `src/ggml-backend.cpp`       | Fail-loud, not bare-assert: before `GGML_ASSERT(*cur_backend_id != -1)` in `ggml_backend_sched_split_graph`, dump the offending node (name/op/type) and its sources. That assert fires whenever no backend's `supports_op()` accepts a node — almost always a dtype an op's kernel doesn't implement — and the bare abort gives zero signal. Diagnostic-only (no behaviour change); how 0008's root cause was pinned. Keep until upstream surfaces the node itself. |
| `0010-cuda-buffer_from_ptr-skip-init_tensor-padding-memset.patch` | `src/ggml-cuda/ggml-cuda.cu` | Sets `buf->iface.init_tensor = NULL` on the BYO-pointer buffer (next to the existing `free_buffer` override). The standard init_tensor zeroes the quantized row-padding via `cudaMemset(data+nbytes, 0, padded-nbytes)`, but a read-only mmap has no padding bytes → the write lands past the tensor / in the read-only mapping → illegal memory access for any quantized tensor with `ne0 % MATRIX_ROW_PADDING(512) != 0` (DeepSeek/Qwen3-MoE `down_exps`, ne0=expert_ff=1408/768; OLMoE's 1024 is exempt — which is why only some MoE models crashed). The zeroing is only NaN-safety: matmul kernels zero-pad the input vector, so `weight_padding * 0 = 0` regardless. Unblocks every quantized MoE model on CUDA (DeepSeek-V2 MLA, Qwen3-MoE, Qwen2-MoE). |

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

## 0011-tensor-flag-detached.patch

`GGML_TENSOR_FLAG_DETACHED` (toy#121 bp-spine): an opaque-cut tensor
flag — forward-identity, gradient-opaque. `ggml_build_backward_expand`
treats a flagged node as a leaf for the needs-grad propagation: no
gradient is ever allocated for it, so nothing walks through it to its
sources (`ggml_compute_backward` early-returns on the NULL grad; zero
backward cases needed). Set via `tnn_detach` (a `ggml_dup` + flag —
DUP has forward kernels on every backend, so no new compute code).
Purpose: sever exactly one edge from a differentiable subgraph — the
expert-input path into `mul_mat_id` — while the residual and router
branches keep their chain grads (the design doc's deferred opaque-cut;
its 'an experiment demands it' trigger fired with F5).

## 0014 — HIP symbol map for toy's own patches

`src/ggml-cuda/vendors/hip.h` is ggml's CUDA→HIP spelling map. Four
symbols used by toy's *own* vendor patches are missing from it, so a
`-DGGML_HIP=ON` build fails compiling **our** patches rather than
anything in ggml:

| symbol | used by |
|---|---|
| `cublasSger` | 0012 (k=1 `out_prod`, the dense MoE gating backward) |
| `cudaMemset2DAsync` | 0012 (zeroing the strided dst) |
| `cudaHostRegisterMapped` | pinned-host path |
| `cudaHostGetDevicePointer` | pinned-host path |

Adding the four `#define`s is the whole patch. Verified on **evo**
(AMD Strix Halo, gfx1151, ROCm 7.0.2): with them,
`cmake -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151
-DBUILD_SHARED_LIBS=ON` builds `libggml.so` clean; without them it
fails on exactly those four symbols.

Two configure notes for anyone repeating it: ggml-hip **refuses static
linking** (`BUILD_SHARED_LIBS=ON` is required, which toy's static-archive
link model does not yet accommodate), and a stale CMake cache carries
`GGML_STATIC` forward — use a fresh build dir or it keeps failing for the
wrong reason.

INERT for CUDA and MUSA builds: `common.cuh` only includes this header
under `GGML_USE_HIP`, so the CUDA path never sees these lines. That is
why it is safe to carry permanently even though toy has no ROCm runner
yet — the remaining gap is `tinynn/tinynn_backend_cuda.c` (121 lines,
~12 CUDA calls, all with direct `hip*` equivalents), the shared-vs-static
build mode, and a ROCm sibling for the Ruby FFI mirror.
