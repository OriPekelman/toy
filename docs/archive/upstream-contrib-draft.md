# Upstream contribution draft — HF→GGUF zero-copy via byte-equivalence

Audience: maintainers of `gguf-org/gguf`, `ggml-org/llama.cpp`, and
anyone writing an HF→GGUF converter. The point is documentation /
clarification, not a bug report.

## The observation

GGUF converters often transpose `nn.Linear` weights from HF's
`[out_features, in_features]` to `[in_features, out_features]` before
writing. This transpose is **unnecessary** — the resulting bytes are
identical to writing the HF tensor as-is, given how ggml interprets
the storage.

### Math

For a `mul_mat(W, x)` op where `W` declared as `ne0=in,
ne1=out_features`:

- ggml stores tensors **column-major**: element `(a, b)` of a tensor
  with `ne=[ne0, ne1]` is at byte offset `(b * ne0 + a) * sizeof`.
- For weights, ggml's `mul_mat` semantic puts `(in, out)` as
  `ne=[in, out]`. So byte offset `(j_out * in + i_in) * sizeof` holds
  `W[i_in, j_out]`.

HF safetensors stores `nn.Linear.weight` as **`[out, in]` row-major**.
Element `(j_out, i_in)` at byte offset `(j_out * in + i_in) * sizeof`.

**These byte offsets are identical** for the same element. No
transpose is needed — write the HF tensor's bytes directly and they
slot straight into ggml's column-major `ne=[in, out]` interpretation.

### Why "transpose" appears in converters anyway

Two reasons we've seen:

1. **Conv1D vs Linear confusion**: GPT-2's `Conv1D.weight` is stored
   as `[in_features, out_features]` (opposite of `nn.Linear`). A
   converter handling GPT-2 needs to handle Conv1D correctly. If
   that code path leaks into Linear handling, you get an unnecessary
   transpose.

2. **Vestigial "logical" transposes**: An older codebase may have
   stored its own weights as `[in, out]` for some downstream API
   reason (matrix-vector convention in a hand-written matmul). The
   converter inherits that transpose even though ggml doesn't need
   it.

### Concrete win: mmap zero-copy

Once the converter writes HF-native bytes, the loader can `mmap()`
the GGUF file and point ggml tensor `data` pointers at file offsets:

```c
struct ggml_tensor *t = ggml_new_tensor_2d(no_alloc_ctx, type, in, out);
ggml_backend_buffer_t buf = ggml_backend_cpu_buffer_from_ptr(mmap_base,
                                                              file_size);
ggml_backend_tensor_alloc(buf, t,
                          (char *)mmap_base + data_offset + tensor_offset);
```

No copy, no dequant for quantized weights, zero load time. For Q8_0
the file IS the model in memory.

### Per-head slicing falls out for free

For multi-head attention, the `attn_q.weight` GGUF tensor with shape
`[n_heads * d_head, d_model]` has each head as a **contiguous byte
range** in the HF-native layout: head `h` occupies rows
`[h * d_head, (h+1) * d_head)`. To bind a per-head ggml tensor
`ne=[d_model, d_head]` to head `h`:

```c
void *head_h_ptr = mmap_base + tensor_offset + h * d_head * d_model * sizeof(elem);
```

Works identically for Q8_0 (the block-quantized stride is just
`d_head * (d_model / 32) * 34` bytes per head).

## Asks

1. **Documentation**: A clear note in `gguf` docs (probably in
   `docs/format.md` near the tensor-data section) stating the
   byte-equivalence rule:

   > 2D tensors written from row-major `[A, B]` numpy arrays are
   > readable as ggml `ne=[B, A]` column-major without any byte
   > reordering. Converters writing `nn.Linear.weight` should write
   > the HF-native `[out, in]` orientation directly.

2. **Audit**: Spot-check `convert_hf_to_gguf.py` and the various
   architecture-specific subclasses for `.T` / `.transpose()` calls
   on `nn.Linear` weight tensors. If any are unnecessary, removing
   them shrinks the converted GGUF (no padding to TENSOR_ALIGNMENT
   after a contiguous numpy copy) and unlocks mmap zero-copy load.

3. **Test**: A small test in `tests/test-gguf.cpp` that constructs a
   2D tensor with `ne=[B, A]`, writes it, reads it back with
   `mmap`+`no_alloc`, and asserts the tensor's `.data` pointer
   matches `mmap_base + data_offset + tensor_offset`.

## Where this was discovered

Implementing zero-copy mmap inference for a Ruby/Spinel-based ggml
binding (`https://github.com/oripekelman/toy_ruby_neural_network`,
private). Our converter had a vestigial `.T.contiguous()` step left
over from a row-vector matmul convention; once removed, a 7B-Q8
inference run dropped from 30 GB peak RSS to 7.4 GB peak with no
loss in numerical output.

The math is general — applies to any GGUF consumer that wants to
mmap weight tensors.

## Suggested venues

- `ggml-org/llama.cpp` issue: "convert_hf_to_gguf.py: are 2D-Linear
  transposes necessary for mmap?"
- `ggml-org/ggml` discussion: "Document byte-equivalence: HF
  `[out, in]` row-major == ggml `ne=[in, out]` column-major"

Filing as an issue (not a PR) first, since the audit / fix may have
many call sites and the maintainers know the codebase.
