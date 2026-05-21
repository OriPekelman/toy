# Issue draft: HF nn.Linear ↔ ggml ne=[in,out] byte equivalence

**Target repo:** `ggml-org/llama.cpp` (primary; the canonical
`convert_hf_to_gguf.py` lives there). Cross-link to `ggml-org/ggml`
if a spec note belongs there too.

**Type:** Documentation / clarification. May surface optimization
opportunities in existing converters.

---

## Title

`convert_hf_to_gguf.py`: are 2D nn.Linear transposes necessary for mmap?

## Body

### Background

I'm writing a from-scratch GGUF consumer (Ruby + Spinel + ggml). To
enable mmap zero-copy load (CPU `ggml_backend_cpu_buffer_from_ptr`,
CUDA `ggml_backend_cuda_buffer_from_ptr`), I needed the on-disk byte
layout of weight tensors to match ggml's column-major
`ne=[in, out]` interpretation exactly. Tracing through this, I think
HF's native `[out, in]` row-major bytes are already perfect — no
transpose needed at conversion. Filing to confirm + document.

### The math (for posterity)

For a `mul_mat(W, x)` op where ggml expects `W` with
`ne0=in, ne1=out`:

- ggml stores `(i, j)` at byte offset `(j * in + i) * sizeof(elem)`
  (column-major; `ne0` is fastest-varying).

HF safetensors stores `nn.Linear.weight` with numpy shape
`(out, in)` row-major:

- numpy `(j, i)` at byte offset `(j * in + i) * sizeof(elem)`.

**Same byte offset for the same logical (in_idx, out_idx) element.**
So writing HF's bytes verbatim and declaring the ggml tensor with
`ne=[in, out]` gives a usable matmul-ready tensor with zero
rearrangement.

### Per-head slicing falls out for free

For multi-head attention, GGUF's `attn_q.weight` has shape
`[n_heads * d_head, d_model]`. Head `h` is rows
`[h * d_head, (h+1) * d_head)`, which is a **contiguous byte
range** in HF-native layout. A per-head ggml tensor with
`ne=[d_model, d_head]` ends up byte-identical to this contiguous
slice. Mmap-friendly with no per-head copy.

Same arithmetic works for Q8_0 with the block-quantized stride
(`d_head * (d_model / 32) * 34` bytes per head).

### What I found in my own converter (now removed)

```python
def take_T(name):
    return np.ascontiguousarray(_load_f32(name).T)   # transposed write
```

This was a vestigial convention from when our loader used
`[in, out]` row-major in a hand-written matmul. Removing it
(replacing `take_T` with `take`) gave bit-identical greedy output
on Qwen2.5-0.5B + 1.5B + 7B, and unlocked mmap zero-copy:

- Qwen2.5-7B-Q8 inference RSS dropped from ~30 GB to ~7 GB (load +
  decode). The mmap region IS the persistent buffer; ggml's matmul
  kernels read it directly.

### Asks

1. **Audit `convert_hf_to_gguf.py`** (and arch subclasses) for `.T` /
   `.transpose()` calls on `nn.Linear` weight tensors. From a quick
   read I don't see obvious ones, but architecture-specific
   subclasses may differ. If any are present and unnecessary,
   removing them shrinks the converted GGUF (no padding to
   `TENSOR_ALIGNMENT` after a fresh-numpy contiguous copy) and
   enables mmap on more consumers.

2. **Spec note** (in ggml-org/ggml's `docs/`?) stating the byte
   equivalence:

   > 2D tensors written from row-major numpy `(A, B)` arrays are
   > readable as ggml `ne=[B, A]` column-major without any byte
   > reordering. Converters writing `nn.Linear.weight` should write
   > the HF-native `(out, in)` orientation directly.

3. **Test** that exercises this: read a small GGUF with
   `gguf_init_from_file(no_alloc=true)`, mmap the file, allocate a
   ggml tensor whose `data` points at `mmap_base + data_offset +
   tensor_offset`, run a small `mul_mat`, compare to a reference
   `mul_mat` from a freshly-allocated buffer. Asserting
   bit-identical output validates the mmap path end-to-end.

### Conv1D caveat

For projects converting GPT-2 (which uses `Conv1D.weight` =
`[in_features, out_features]`, opposite of `nn.Linear`), the
transpose IS necessary. The two should be distinguished cleanly in
any converter — this issue is specifically about `nn.Linear`.

### Acknowledgement

I'm a downstream consumer rebuilding this from scratch for
educational reasons. Happy to put together a PR for the audit + spec
note if you're amenable. If the answer is "yes we know, and the
canonical converter is correct" then maybe just a sentence in the
gguf format spec saying so explicitly would save the next person an
afternoon.

---

## gh command (when auth'd)

```sh
gh issue create \
  --repo ggml-org/llama.cpp \
  --title "convert_hf_to_gguf.py: are 2D nn.Linear transposes necessary for mmap?" \
  --body-file docs/upstream-issues/01-byte-equivalence.md
```

Strip the top header (`# Issue draft:` + target line) before
posting — the gh command sends the file as-is. Or copy-paste the
body section.
