# L5 — Recipes

A Recipe is a training plan: realize a model (random init, mmap'd
base, or warm-started), then drive `step!` over the data.

## What's here (4 recipes)

| File | Class | Realize | Notes |
| --- | --- | --- | --- |
| `from_scratch.rb` | `Toy::LLM::Recipes::FromScratch` | `realize!` | Random init → AdamW on cross-entropy. The blessed default for new models. |
| `lora.rb` | `Toy::LLM::Recipes::LoRA` | `realize!` (mmap'd base + LoRA adapter) | Frozen base + LoRA on Q. QLoRA-capable. Fine-tune a pretrained GGUF. |
| `warm_start.rb` | `Toy::LLM::Recipes::WarmStart` | `realize_scratch!` + (optional) `realize_warm!` + `build!` | Random init, then optionally graft a donor `token_embd` + PCA projection lens. |
| `vit_tiny.rb` | `Toy::LLM::Recipes::VitTiny` | `realize!` | ViT-Tiny image classifier (patch embed + CLS token + MLP head). |

Backend twins where they exist: `from_scratch_cuda.rb`,
`from_scratch_metal.rb`, `lora_cuda.rb`, `warm_start_cuda.rb`.

## The real API

Every recipe exposes `step!(inputs, positions, m_labels, m_hp, is_first)`
(ViT: `step!(m_image, cls_idx, m_labels, m_hp, is_first)`), returning the
scalar loss. The caller builds the two per-step Mats with the pure-Ruby
value objects:

- `Toy::Labels.next_token` / `Toy::Labels.next_token_guarded`
  (`lib/toy/llm/labels.rb`) — the shift-by-one one-hot label Mat.
- `Toy::AdamW` (`lib/toy/llm/adamw.rb`) — the named optimizer
  hyper-params; `adamw.hp(step)` builds the `Mat(1,7)` the FFI
  optimizer step reads.

Model shape is built with the named `Toy::SmolLM2Config.mha` /
`.gqa` factories (`lib/toy/models/toy_smollm2.rb`).

## Read these

The byte-gated exemplars users read:

- `examples/01_train_tiny.rb` — the blessed short tutorial.
- `prep/smokes/smoke_recipe_from_scratch.rb`
- `prep/smokes/smoke_recipe_warm_start.rb`
- `prep/smokes/smoke_recipe_lora.rb`
