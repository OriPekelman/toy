# L3 — Archs

An Arch is the full network: embedding + stack of blocks + head.
One file per arch.

## What's here (2 archs)

| File | Class | Blocks | Notes |
| --- | --- | --- | --- |
| `llama_arch.rb` | `Toy::LLM::Archs::LlamaArch` | `TransformerBlock` × N | Llama / SmolLM2 / Qwen / Gemma share this. Per-cfg values (d_model, n_heads, n_kv, rope_base, …) select the exact model. |

Backend twins: `llama_arch_cuda.rb`, `llama_arch_metal.rb`.

## What lives on the ARCH

- Token embedding (shared across all blocks).
- Final RMSNorm + LM head (tied or untied).
- Block stacking across N layers.
- The realize path that allocates the whole-graph context.

## What lives on the RECIPE (L4), not here

- Training loop, optimizer, schedule.
- Data pipeline.

See `prep/smokes/smoke_recipe_*.rb` and the blessed
`examples/01_train_tiny.rb` for the arch driven end-to-end.
