# L2 — Blocks

A Block is one state-threading unit. It composes L1 primitives,
owns its weight allocation, and threads state through itself.

## Roster (target post-P2)

| File | Class | Composes | Notes |
| --- | --- | --- | --- |
| `transformer_block.rb` | `Toy::LLM::Blocks::TransformerBlock` | `RMSNorm` × 2, `RoPE`, `GQA`, `SwiGLU` | Llama / Qwen / Gemma family. Cfg picks gate flavour (silu vs gelu). |
| `gpt2_block.rb` | `Toy::LLM::Blocks::GPT2Block` | `LayerNorm` × 2, `MHA`, `GeLUFFN` | GPT-2 family (no RoPE; learned position embeddings on the arch). |
| `vit_block.rb` | `Toy::LLM::Blocks::ViTBlock` | `LayerNorm` × 2, `MHA`, `GeLUFFN` | Vision transformer. Same shape as GPT2Block but no causal mask. |
| `ssm_block.rb` | `Toy::LLM::Blocks::SSMBlock` | Mamba selective-scan kernel | Proves the contract generalises beyond transformer. P2 gate. |

## Contract

```ruby
class Toy::LLM::Blocks::TransformerBlock
  # Per-block param tensors. Allocated by the arch when realising;
  # the block doesn't decide *which* dtype — the cfg does.
  attr_accessor :rn1_gamma, :rn2_gamma,
                :w_q, :w_k, :w_v, :w_o,
                :w_gate, :w_up, :w_down,
                :b_q, :b_k, :b_v   # nil unless cfg.qkv_bias

  # Build this block's forward graph into the active session.
  # `state` is the input state (KV cache slice for transformer;
  # SSM hidden state for ssm). Returns (output_tensor, state_out).
  def build_forward(sess, x, state, cfg)
    h    = Primitives::RMSNorm.build(sess, x, rn1_gamma, cfg.eps)
    a, state_out = Primitives::GQA.build(sess, h, qkv_weights, state, cfg.attn)
    x2   = TinyNN.tnn_add(sess, x, a)
    h2   = Primitives::RMSNorm.build(sess, x2, rn2_gamma, cfg.eps)
    ff   = Primitives::SwiGLU.build(sess, h2, ffn_weights, cfg.ffn)
    out  = TinyNN.tnn_add(sess, x2, ff)
    [out, state_out]
  end
end
```

State is **the** abstraction (per the design doc §6). For
transformer-family blocks state = KV cache slice; for SSM state =
hidden vector; future architectures define their own.

## What lives on the BLOCK

- Block-scoped weight tensor handles.
- The forward graph for one block instance.
- State input/output threading.

## What lives on the ARCH (L3), not here

- Token embedding, position embedding (when learned), final norm,
  LM head.
- Block stacking + per-layer override resolution.
- Whole-graph allocation (`build_forward_in_current_ctx`).

This file is a contract sketch. Real entries land in P2.4.
