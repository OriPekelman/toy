# lib/toy/llm/primitives/scalable_softmax.rb — L1 primitive: Scalable-Softmax
# (SSMax, Nakanishi, arXiv 2501.19399) — anti-attention-fading softmax.
#
# Pure module: `self.` methods only. The BLOCK (L2) owns the learned per-head
# scalar s and computes the SSMax scale; this primitive is the scaled softmax
# itself. See README.md.
#
# Formula (no-bias form, Eq. 11): a = softmax((s*log n) * (q·kᵀ / sqrt(d))),
# i.e. the usual scaled logits are multiplied by the scalar s*log(n), where n
# is the number of keys in the causal prefix and s is learnable (init ~0.168).
# This is exactly the existing scaled-softmax op with a MODIFIED scale:
#   ssmax_scale = (1/sqrt(d)) * s * log(n).
# The block precomputes ssmax_scale (log(n) for a fixed context length is a
# CRuby-layer Float constant — no libm in the Spinel runner) and passes it here.
#
# Spinel hygiene: no Cfg / no default args. One FFI passthrough to
# tnn_soft_max_ext. Call via the full module path.

module Toy
  module LLM
    module Primitives
      module ScalableSoftmax
        NAME = :scalable_softmax

        # SSMax-scaled softmax over attention scores. scores is the raw q·kᵀ
        # map; mask the additive attention mask handle (or null); ssmax_scale
        # the block's precomputed (1/sqrt(d))*s*log(n) Float; max_bias the
        # ggml soft_max_ext ALiBi slope (0.0 when unused). Returns the
        # attention-weight map. (Plain softmax falls out when ssmax_scale is
        # the ordinary 1/sqrt(d) — so this also covers vanilla attention.)
        def self.attend(sess, scores, mask, ssmax_scale, max_bias)
          TinyNN.tnn_soft_max_ext(sess, scores, mask, ssmax_scale, max_bias)
        end
      end
    end
  end
end
