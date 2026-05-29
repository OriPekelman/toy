# lib/toy/llm/primitives/gqa.rb — L1 primitive: GQA (grouped-query
# attention math; MHA falls out as group_size == 1).
#
# Pure module: `self.` methods only, no module ivars, no state, no
# config object. The block (L2) owns the KV-head selection
# (hkv = hq / group_size), the Q/K/V projection, bias, LoRA and RoPE
# steps, and the KV-cache tensors. This primitive composes only the
# attention math: scores -> scaled+masked softmax -> weighted V. See
# lib/toy/llm/primitives/README.md for the L1 contract.
#
# Spinel hygiene: no Cfg ctor and no default args (default-arg
# poisoning, landmine #4) — the single call site (build_seq_qhead)
# supplies all 7 positional args explicitly, so a fixed 7-arity
# signature is clean. `batch` is a concrete Int (pins the `> 1`
# comparison); `scale` is a Float threaded down from the block (no
# Math.sqrt here). No Card/step_bind calls, no FFI :str args
# (step_bind :str landmine 2026-05-28).
#
# Parity: the two branches (B>1 fused soft_max_ext vs B=1 separate
# scale+diag_mask_inf+softmax) are documented as bit-identical to
# pre-GH#7 at B=1 and MUST be preserved exactly — do not unify them.
# At B=1 the mask handle is NULL and is never read.
#
# This file does NOT `require_relative "tinynn"`: the loading module
# (lib/llama_seq_forward_ffi.rb) already loads the correct backend's
# TinyNN before requiring this primitive. The mirror generator picks
# the backend via the monolith's require rewrite (the CUDA/Metal
# mirror of the monolith requires gqa_<backend>, and that mirror's
# TinyNN. -> TinyNN<Backend>. rename handles the rest).

module Toy
  module LLM
    module Primitives
      module GQA
        NAME = :gqa

        # Pure attention math: scores -> scaled+masked softmax ->
        # weighted V. No weights, no KV cache, no ivars. All inputs are
        # tensor handles + scalars supplied by the L2 block.
        #
        #   t_k        : selected KV-head key tensor    ne=[d_head, T*B]
        #   t_q        : per-Q-head rotated query       ne=[d_head, T*B]
        #   t_vt       : selected KV-head V transpose   ne=[T*B, d_head]
        #   attn_mask  : block-causal mask tensor (used only when batch > 1)
        #   scale      : 1.0 / sqrt(d_head)
        #   batch      : @seq_b (selects soft_max_ext vs diag_mask_inf path)
        # Returns the per-head output tensor ne=[d_head, T*B].
        def self.attention(sess, t_k, t_q, t_vt, attn_mask, scale, batch)
          t_scores = TinyNN.tnn_matmul(sess, t_k, t_q)
          if batch > 1
            t_attn = TinyNN.tnn_soft_max_ext(sess, t_scores, attn_mask, scale, 0.0)
          else
            t_scaled = TinyNN.tnn_scale(sess, t_scores, scale)
            t_masked = TinyNN.tnn_diag_mask_inf(sess, t_scaled, 0)
            t_attn   = TinyNN.tnn_softmax(sess, t_masked)
          end
          TinyNN.tnn_matmul(sess, t_vt, t_attn)
        end
      end
    end
  end
end
