# demos/

End-to-end Ruby drivers. Source here; build target names match the
file base; binaries land back here. **Run from the repo root** so
they can find `data/*.gguf`, prompts, BPE tables.

## Generic inference path (Phase 0 — preferred for new work)

| Source | What |
|---|---|
| `qwen25_transformer_lm.rb` | Qwen2.5 via the generic `ToyLM` (lib/transformer_lm.rb). `GGUF=…` picks size + F32/Q8. |
| `qwen25_toylm_cuda.rb`     | CUDA mirror of the above (`ToyLMCuda`). F32 + aligned-`d_ff` Q8. |
| `llama32_smoke.rb`         | Llama-3.2-1B via `ToyLM`. `GGUF=…` swaps to 3B. |
| `mistral_smoke.rb`         | Mistral-7B-Instruct-v0.2 via `ToyLM`. |
| `tinyllama_toylm.rb`       | TinyLlama-1.1B via `ToyLM`. |
| `smollm2_toylm.rb`         | SmolLM2-135M via `ToyLM`. |
| `sampler_smoke.rb`         | Sampler module smoke (temperature / top_k / top_p / determinism). `MODE=greedy\|topk\|topp` + `SEED=N`. |

## Legacy / per-model inference

| Source | Build | What |
|---|---|---|
| `gpt2.rb`              | `make gpt2`              | DistilGPT-2 / GPT-2 via `Toy::GPT2` (native Mat) |
| `smollm2.rb`           | `make smollm2`           | SmolLM2-135M via `Toy::SmolLM2` (native Mat) |
| `smollm2_kv.rb`        | `make smollm2_kv`        | SmolLM2-135M FFI KV-cache (CPU) |
| `smollm2_kv_cuda.rb`   | `make smollm2_kv_cuda`   | SmolLM2-135M FFI KV-cache (CUDA) |
| `tinyllama.rb`         | `make tinyllama`         | TinyLlama-1.1B via `Toy::SmolLM2` (native Mat) |
| `tinyllama_kv.rb`      | `make tinyllama_kv`      | TinyLlama-1.1B FFI KV-cache (CPU) |
| `tinyllama_kv_cuda.rb` | `make tinyllama_kv_cuda` | TinyLlama-1.1B FFI KV-cache (CUDA) |
| `qwen25_kv.rb`         | `make qwen25_kv`         | Qwen2.5 Mat-mediated KV (slow gold reference). |
| `qwen25_native_mmap.rb`      | `make qwen25_native_mmap`      | Qwen2.5 Phase-2 mmap (CPU). |
| `qwen25_native_mmap_cuda.rb` | `make qwen25_native_mmap_cuda` | Qwen2.5 Phase-2 mmap (CUDA). |

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
| `smollm2_seq_train_cuda.rb` | `make smollm2_seq_train_cuda` | Sequence-mode LoRA-Q training on CUDA — one forward+backward+opt_step over T positions. |
| `smollm2_seq_full_finetune_cuda.rb` | `make smollm2_seq_full_finetune_cuda` | Full fine-tune on CUDA (every per-block weight + optional embedding via `EMBED=1`). |
| `smollm2_seq_qlora_cuda.rb` | `make smollm2_seq_qlora_cuda` | QLoRA: Q8 base + F32 LoRA on CUDA. |
| `smollm2_lora_sft_multi_cuda.rb` / `smollm2_lora_sft_multipos_cuda.rb` | (make targets) | KV-decode multi-target / multi-position SFT smokes (F1.2 step 6). |

CPU LoRA training is wired up but currently slow due to the
ggml-cpu sched-aliasing workaround (`tnn_pin_all_graph_b_nodes`);
CUDA is the practical training path. See
[`../docs/archive/f3-full-finetune-2026-05-21.md`](../docs/archive/f3-full-finetune-2026-05-21.md)
for the full-finetune design notes.

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
[`../docs/archive/tinyllama-known-issue.md`](../docs/archive/tinyllama-known-issue.md).
