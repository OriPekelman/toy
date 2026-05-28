# L3 — Archs

An Arch is the full network: embedding + stack of blocks + head.
One file per arch.

## Roster (target post-P2)

| File | Class | Blocks | Notes |
| --- | --- | --- | --- |
| `llama_arch.rb` | `Toy::LLM::Archs::LlamaArch` | `TransformerBlock` × N | Llama / SmolLM2 / Qwen / Gemma share this. Per-cfg defaults select the exact model. |
| `gpt2_arch.rb` | `Toy::LLM::Archs::GPT2Arch` | `GPT2Block` × N | GPT-2 family — learned token + position embeddings, tied / untied LM head. |
| `vit_arch.rb` | `Toy::LLM::Archs::ViTArch` | `ViTBlock` × N | ViT — patch embedding + CLS token + bidirectional attention. |
| `mamba_arch.rb` | `Toy::LLM::Archs::MambaArch` | `SSMBlock` × N | Mamba / Mamba-2. P2 stub; full landing later. |

## Contract

```ruby
class Toy::LLM::Archs::LlamaArch
  # Per-arch tensor handles owned at the top.
  attr_accessor :token_embed, :final_norm_gamma, :output_head,
                :positions, :rope_freq_factors,
                :blocks   # Array<TransformerBlock>

  # Build the full forward graph.
  def build_forward(sess, ids, positions, cfg)
    e = TinyNN.tnn_get_rows(sess, token_embed, ids)
    state_in = build_initial_state(cfg)

    x = e
    blocks.each_with_index do |blk, l|
      # Per-layer override applied here. cfg.per_layer(l) returns the
      # cfg with `with_hyper`-style overrides applied for this layer.
      x, state_in = blk.build_forward(sess, x, state_in, cfg.per_layer(l))
    end

    h_final = Primitives::RMSNorm.build(sess, x, final_norm_gamma, cfg.eps)
    logits  = TinyNN.tnn_mul_mat(sess, output_head, h_final)
    logits
  end
end
```

## Per-layer overrides — the killer demo

The arch resolves `per_layer` overrides at build time. The P2 gate
requires:

```ruby
arch = Toy::LLM::Archs::LlamaArch
  .with_hyper(:d_model, 512)
  .per_layer(:n_heads, [16, 16, 8, 4, 2, 1])

arch.realize(cfg, sess, :infer)   # must build + run a 6-block pyramid
```

The arch is the lowest layer that knows "stack of N blocks" — so
per-layer indexing lives here, not on the block.

## What lives on the ARCH

- Token embedding (shared across all blocks).
- Position embedding (when learned; RoPE is on the block / primitive).
- Final norm + LM head (lm-style) or pooling head (vit-style).
- The realize path that allocates whole-graph context.
- Per-layer override resolution.

## What lives on the RECIPE (L4), not here

- Training loop, optimizer, schedule.
- Data pipeline.
- Curriculum stages.

This file is a contract sketch. Real entries land in P2.5.
