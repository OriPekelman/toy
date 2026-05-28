# L1 — Primitives

A primitive is a single named op. One file per primitive. Each file
defines a module that registers itself with the framework registry on
load.

## Roster (target post-P2)

| File | Module | Purpose |
| --- | --- | --- |
| `rms_norm.rb` | `Toy::LLM::Primitives::RMSNorm` | Root-mean-square normalisation (gamma only, no beta) |
| `layer_norm.rb` | `Toy::LLM::Primitives::LayerNorm` | Mean/var normalisation (gamma + beta) |
| `rope.rb` | `Toy::LLM::Primitives::RoPE` | Rotary positional embedding with extended scaling |
| `gqa.rb` | `Toy::LLM::Primitives::GQA` | Grouped-query attention (covers MHA as group_size=1) |
| `mha.rb` | `Toy::LLM::Primitives::MHA` | Multi-head attention (delegates to GQA) |
| `swiglu.rb` | `Toy::LLM::Primitives::SwiGLU` | SiLU-gated FFN (Llama-family) |
| `gelu_ffn.rb` | `Toy::LLM::Primitives::GeLUFFN` | GeLU-gated FFN (GPT-2 family) |
| `patch_embed.rb` | `Toy::LLM::Primitives::PatchEmbed` | ViT patch + projection |
| `softmax.rb` | `Toy::LLM::Primitives::Softmax` | Numerically-stable softmax (with optional mask) |

## Contract

```ruby
module Toy::LLM::Primitives::RoPE
  # NAME used by replace_primitive / per_layer overrides.
  NAME = :rope

  # Build the op into the active session graph. Pure function of
  # tensor inputs + config — no ivars, no state.
  #
  # Returns the output tensor handle.
  def self.build(sess, x, positions, cfg)
    # cfg is a value object specific to this primitive:
    # RoPECfg{ d_head, base, freq_scale, ext_factor,
    #         attn_factor, beta_fast, beta_slow, freq_factors }
    TinyNN.tnn_rope_ext(sess, x, positions, cfg.d_head, ...)
  end

  # Optional: derive a Toy::Card fragment describing this primitive
  # at the supplied cfg. Used by upper layers to compose the full
  # algorithm Card.
  def self.card_fragment(cfg)
    # ...
  end
end
```

## Why pure-function shape

- No ivar typing landmines (Spinel) — modules with `self.` methods
  don't have ivar layout collisions.
- Composition is mechanical: an L2 block just calls
  `RMSNorm.build(sess, x, gamma, eps)` then `RoPE.build(sess, ...)`.
- Cards derive trivially: walk the resulting graph (already proven
  in P1).

## What lives on the BLOCK, not here

- Weight allocation (the L2 block owns the block-scoped param
  tensors).
- KV cache memory (the L2 block owns it for transformer; the SSM
  block owns its hidden state).
- Per-layer state threading.

This file is a contract sketch. Real entries land in P2.3.
