# L1 — Primitives

A primitive is a single named op. One file per primitive, with a
CPU implementation plus `_cuda` and `_metal` backend variants
(mechanically mirrored by `prep/gen_cuda_mirror.rb`).

## What's here (12 primitives)

| File | Module | Purpose |
| --- | --- | --- |
| `rms_norm.rb` | `Toy::LLM::Primitives::RMSNorm` | Root-mean-square normalisation (gamma only, no beta) |
| `rope.rb` | `Toy::LLM::Primitives::RoPE` | Rotary positional embedding with extended (YaRN/llama3) scaling |
| `gqa.rb` | `Toy::LLM::Primitives::GQA` | Grouped-query attention (covers MHA at group_size = 1) |
| `swiglu.rb` | `Toy::LLM::Primitives::SwiGLU` | SiLU-gated FFN (Llama family) |

Each also ships `<name>_cuda.rb` and `<name>_metal.rb` backend twins.

## How they compose

The L2 transformer block calls these primitives directly to build its
forward graph into the active session; the L3 Llama arch stacks the
block N times. See the recipe exemplars that drive the whole stack
end-to-end:

- `prep/smokes/smoke_recipe_from_scratch.rb`
- `prep/smokes/smoke_recipe_warm_start.rb`
- `prep/smokes/smoke_recipe_lora.rb`

and the blessed tutorial `examples/01_train_tiny.rb`.
