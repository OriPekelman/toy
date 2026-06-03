# `Toy::LLM` — layered transformer stdlib

The LLM algorithm stack. The CPU file is the source of truth in every
layer; the `_cuda` / `_metal` mirrors are generated at build time from
`MIRRORABLE` markers (`prep/gen_cuda_mirror.rb`) and held bit-identical by
`make verify-mirrors`.

```
toy/llm/
├── primitives/  # L1 — one named op per file        (RMSNorm, RoPE, SwiGLU, GQA)
├── blocks/      # L2 — one state-threading unit      (TransformerBlock)
├── archs/       # L3 — embed + N blocks + head       (LlamaArch)
├── engine/      # session + realize + training-step  (LlamaSeqEngine, ViTTinyEngine)
└── recipes/     # training entry points              (FromScratch, LoRA, WarmStart, ViTTiny)
```

The **engine** owns the FFI session, the four `realize_for_*` paths
(random_init / mmap / q8_copy / full_finetune), the forward + cross-entropy
+ backward + AdamW graph (`build_training_step`), and LoRA setup. It
composes L1–L3 for the compute and delegates per-block / per-global tensor
allocation to `TransformerBlock` / `LlamaArch`. The **recipes** instantiate
an engine and drive its `realize!` / `step!` surface.

See each subdirectory's README for the layer-specific contract.

## Reading order

1. `primitives/README.md` — a primitive: pure-function shape.
2. `blocks/README.md` — a block: state contract.
3. `archs/README.md` — an arch: embed + blocks + head orchestration.
4. `recipes/README.md` — a recipe: realize + step loop.

## Status

The five-layer refactor is complete. The former top-level monolith
(`lib/llama_seq_forward_ffi.rb`) is retired — it is now
`engine/llama_seq_engine.rb` (`Toy::LLM::Engine::LlamaSeqEngine`). All four
realize paths are decomposed onto `TransformerBlock` / `LlamaArch`, and the
`full_finetune` path is gated byte-exact by `prep/full_finetune_gate.rb`.
Every training path (from-scratch, LoRA, warm-start, ViT, full-finetune) is
gated bit-identical on CPU and CUDA.
