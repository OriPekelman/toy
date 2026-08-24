# L2 — Blocks

A Block composes L1 primitives, owns its block-scoped weight tensors,
and builds the forward graph for one block instance.

## What's here (4 blocks)

| File | Class | Composes | Notes |
| --- | --- | --- | --- |
| `transformer_block.rb` | `Toy::LLM::Blocks::TransformerBlock` | `RMSNorm` × 2, `RoPE`, `GQA`, `SwiGLU` | Llama / Qwen / Gemma family (pre-norm, residual on each sublayer). |

Backend twins: `transformer_block_cuda.rb`, `transformer_block_metal.rb`.

## What lives on the BLOCK

- Block-scoped weight tensor handles.
- The forward graph for one block instance.
- State (KV-cache slice) input/output threading.

## What lives on the ARCH (L3), not here

- Token embedding, final norm, LM head.
- Block stacking across N layers.
- Whole-graph allocation.

See `prep/smokes/smoke_recipe_*.rb` and the blessed
`examples/01_train_tiny.rb` for the block driven end-to-end.
