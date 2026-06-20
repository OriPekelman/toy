# lib/toy/llm/primitives/depth_scale.rb — L1 primitive: depth-dependent
# LayerNorm scaling (Sun et al., "The Curse of Depth", arXiv 2502.05795).
#
# Pure module: `self.` methods only. The BLOCK (L2) owns the norm and computes
# the per-layer constant; this primitive applies the parameter-free 1/sqrt(ell)
# scaling to a normalised sublayer INPUT. See README.md.
#
# Formula: h(ell) = LayerNorm(h_ell) * (1/sqrt(ell)), applied to BOTH the
# attention and FFN pre-norm outputs (the input fed to the sublayer), 1-indexed
# ell. Caps Pre-LN variance growth with depth so deep layers stay effective.
# Hyperparameter-free, no learned weights.
#
# Spinel hygiene: no Cfg / no default args. One FFI passthrough. inv_sqrt_depth
# is the block's precomputed Float (1/sqrt(layer_index), computed in the CRuby
# layer — no libm in the Spinel runner). Call via the full module path.

module Toy
  module LLM
    module Primitives
      module DepthScale
        NAME = :depth_scale

        # Scale a normalised tensor by the depth constant 1/sqrt(ell).
        # x is the block's RMSNorm output (the sublayer input); the block
        # passes inv_sqrt_depth = 1.0/Math.sqrt(layer) as a Float constant.
        # Returns the depth-scaled handle.
        def self.apply(sess, x, inv_sqrt_depth)
          TinyNN.tnn_scale(sess, x, inv_sqrt_depth)
        end
      end
    end
  end
end
