# lib/toy/llm/primitives/diff_attention.rb — L1 primitive: Differential
# Attention (DIFF Transformer, Ye et al.) — the diff-specific composition.
#
# Pure module: `self.` methods only, no module ivars, no state, no config
# object. The BLOCK (L2) owns the Q1/Q2/K1/K2/V projections, runs the two
# softmax attention maps (reusing the GQA primitive), and owns the learned
# lambda vectors / per-head subln gamma; this primitive composes only the
# DIFFERENTIAL pieces: the lambda scalar, the A1 - lambda*A2 combine, and the
# (1 - lambda_init)-scaled per-head sub-norm. See README.md and
# docs/roadmap/dragon-gdn-arch-2026-06-20.md.
#
# Formula (microsoft/unilm Diff-Transformer): each logical head owns two
# q/k subheads. lambda = exp(lq1·lk1) - exp(lq2·lk2) + lambda_init, where
# lambda_init = 0.8 - 0.6*exp(-0.3*depth) is a depth-constant the block passes
# in. A = A1 - lambda*A2 ; O = A@V ; O = rms_norm(O)*gamma * (1 - lambda_init).
#
# Spinel hygiene: no Cfg ctor / no default args, no Card/step_bind, no FFI
# :str. Fixed-arity FFI passthroughs. NOTE: call via the full module path
# (Spinel can't dispatch a module method through a constant alias).
#
# This file does NOT require_relative "tinynn": the loader loads the backend's
# TinyNN first (mirror generator handles the TinyNN.->TinyNN<Backend>. rename).

module Toy
  module LLM
    module Primitives
      module DiffAttention
        NAME = :diff_attention

        # The per-head differential lambda SCALAR:
        #   lambda = exp(sum(lq1*lk1)) - exp(sum(lq2*lk2)) + lambda_init
        # lq1/lk1/lq2/lk2 are the learned [head_dim] vectors (block-owned);
        # lambda_init is the depth-constant Float. The dot products reduce to
        # a [1] tensor via tnn_sum; the result lambda is a [1] tensor that
        # broadcast-multiplies A2 in `combine`. (scale_bias folds the
        # + lambda_init onto the first exp term, so the math is
        # (exp1 + lambda_init) - exp2 = exp1 - exp2 + lambda_init.)
        def self.lambda_scalar(sess, lq1, lk1, lq2, lk2, lambda_init)
          d1  = TinyNN.tnn_mul(sess, lq1, lk1)
          s1  = TinyNN.tnn_sum(sess, d1)
          e1  = TinyNN.tnn_exp(sess, s1)
          d2  = TinyNN.tnn_mul(sess, lq2, lk2)
          s2  = TinyNN.tnn_sum(sess, d2)
          e2  = TinyNN.tnn_exp(sess, s2)
          e1b = TinyNN.tnn_scale_bias(sess, e1, 1.0, lambda_init)  # exp1 + lambda_init
          TinyNN.tnn_sub(sess, e1b, e2)                            # (exp1+λ_init) - exp2
        end

        # Combine the two attention maps: A = A1 - lambda*A2. a1/a2 are the
        # block's two softmax score maps (same shape); lambda the [1] scalar
        # from `lambda_scalar` (broadcasts). a1 drives the shape under ggml
        # broadcast; the lambda*a2 term is subtracted.
        def self.combine(sess, a1, a2, lambda_t)
          la2 = TinyNN.tnn_mul(sess, a2, lambda_t)
          TinyNN.tnn_sub(sess, a1, la2)
        end

        # Per-head output sub-norm + the fixed (1 - lambda_init) scaling:
        #   O = rms_norm(O, gamma) * (1 - lambda_init).
        # o is the per-head attention output (block-sliced); gamma the subln
        # weight; eps the Float epsilon; one_minus_lambda_init the compile-time
        # Float (1 - lambda_init). tnn_rms_norm folds gamma; scale applies the
        # depth constant. Returns the normed/scaled head output.
        def self.subln(sess, o, gamma, eps, one_minus_lambda_init)
          n = TinyNN.tnn_rms_norm(sess, o, gamma, eps)
          TinyNN.tnn_scale(sess, n, one_minus_lambda_init)
        end
      end
    end
  end
end
