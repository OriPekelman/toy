# `Toy::LLM` — layered transformer/SSM stdlib

This directory is the destination of the P2 refactor (see
`docs/roadmap/toy-framework-roadmap-2026-05-28.md`).

The current monolith (`lib/llama_seq_forward_ffi.rb` + CUDA / Metal
mirrors) decomposes into:

```
toy/llm/
├── primitives/  # L1 — one file per named op       (RMSNorm, RoPE, GQA, ...)
├── blocks/      # L2 — one state-threading unit    (TransformerBlock, SSMBlock)
├── archs/       # L3 — embedding + N blocks + head (LlamaArch, GPT2Arch, ViTArch)
└── recipes/     # L4 — training plan (1..N stages) (FromScratch, LoRA, WarmStart)
```

See each subdirectory's README for the layer-specific contract.

## Reading order

1. `primitives/README.md` — what a primitive looks like; pure-function shape.
2. `blocks/README.md` — what a block looks like; state contract.
3. `archs/README.md` — what an arch looks like; per-layer overrides.
4. `recipes/README.md` — what a recipe looks like; stage sequence.

## Status (2026-05-28)

P2.2 — skeleton + contract READMEs landed. No code has moved yet.
The monolith `lib/llama_seq_forward_ffi.rb` is still authoritative.

P2.3 (next) will extract the **RoPE primitive** first as the pilot
— it's the right Goldilocks (8-arg signature, no weights, used in
both K and Q paths). See the roadmap §P2.0 survey findings.
