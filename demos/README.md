# demos/

End-to-end Ruby drivers. Source here; build target names match the
file base; binaries land back here. **Run from the repo root** so
they can find `data/*.gguf`, prompts, BPE tables.

## Inference

| Source | Build | What |
|---|---|---|
| `gpt2.rb`            | `make gpt2`            | DistilGPT-2 / GPT-2 via `Toy::GPT2` (native Mat) |
| `smollm2.rb`         | `make smollm2`         | SmolLM2-135M via `Toy::SmolLM2` (native Mat) |
| `smollm2_kv.rb`      | `make smollm2_kv`      | SmolLM2-135M FFI KV-cache (CPU) |
| `smollm2_kv_cuda.rb` | `make smollm2_kv_cuda` | SmolLM2-135M FFI KV-cache (CUDA) |
| `tinyllama.rb`       | `make tinyllama`       | TinyLlama-1.1B via `Toy::SmolLM2` (native Mat) |
| `tinyllama_kv.rb`    | `make tinyllama_kv`    | TinyLlama-1.1B FFI KV-cache (CPU) |
| `tinyllama_kv_cuda.rb` | `make tinyllama_kv_cuda` | TinyLlama-1.1B FFI KV-cache (CUDA) |
| `qwen25_kv.rb`       | `make qwen25_kv`       | Qwen2.5 Mat-mediated KV (slow gold reference). `GGUF=…` picks 0.5B–7B. |
| `qwen25_native_mmap.rb` | `make qwen25_native_mmap` | Qwen2.5 Phase-2 mmap (CPU). Canonical CPU path. `GGUF=…` picks size + F32/Q8. |
| `qwen25_native_mmap_cuda.rb` | `make qwen25_native_mmap_cuda` | Qwen2.5 Phase-2 mmap (CUDA). Canonical CUDA path. F32 only today. |

## Parity tools

| Source | What |
|---|---|
| `qwen25_per_layer_cpu.rb`  | Per-layer logits dump for Qwen2.5 (CPU). For CPU↔CUDA divergence bisection. |
| `qwen25_per_layer_cuda.rb` | Per-layer logits dump for Qwen2.5 (CUDA). Pair with the CPU variant. |
| `algorithm_cards.rb`       | Print the Phuong-Hutter algorithm cards (no inference). |

## Training

| Source | Build | What |
|---|---|---|
| `train.rb` | `make train` | TinyStories from-scratch training via `Toy::Trainer` (CPU). Generate corpus first: `ruby prep/prep_tinystories.rb --max_lines 500`. |

CPU training is wired up; CUDA training is a planned effort —
backward kernels (RMS-norm-back, RoPE-back, attention-back) and a
backward-graph driver are needed. See
[`../docs/design/finetuning.md`](../docs/design/finetuning.md) for
the roadmap.

## Quickstart

```sh
make setup-ggml                                # one-time, builds vendored ggml
./prep/convert_smollm2_to_gguf.py              # writes data/smollm2-135m-f32.gguf
./prep/smollm2_tokens.py encode "Once upon a time"
make smollm2_kv && ./demos/smollm2_kv
./prep/smollm2_tokens.py decode
# → "Once upon a time, there was a little girl named Lily..."
```

For larger models on CUDA (after `make setup-ggml-cuda`):

```sh
make qwen25_native_mmap_cuda
GGUF=data/qwen25-1.5b-native.gguf ./demos/qwen25_native_mmap_cuda
```

The TinyLlama FFI paths produce NaN logits at full depth due to f32
overflow on that specific checkpoint. Use `demos/tinyllama` (native
Mat) for correct output; see
[`../docs/tinyllama-known-issue.md`](../docs/tinyllama-known-issue.md).
