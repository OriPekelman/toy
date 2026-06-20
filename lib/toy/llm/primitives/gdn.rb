# lib/toy/llm/primitives/gdn.rb — L1 primitive: Gated DeltaNet (GDN)
# composition (Dragon / Qwen3-Next linear-attention mixer).
#
# Pure module: `self.` methods only, no module ivars, no state, no
# config object. The GDN BLOCK (L2) owns the q/k/v/z/b/a projections,
# the short causal conv, and the A_log / dt_bias / gamma weights; this
# primitive composes only the PARAMETER-FREE activation + recurrence
# steps that wrap them. See lib/toy/llm/primitives/README.md.
#
# The recurrence core is the in-tree ggml op tnn_gated_delta_net. Its
# CONTRACT (verified against ggml-cpu/ops.cpp:10634): the kernel applies
# exp(g) internally (g is the LOG-decay, passed raw), uses beta DIRECTLY
# (so it must be pre-sigmoid'd), uses q/k DIRECTLY (so they must be
# pre-L2-normed), and scales the attn output by 1/sqrt(S_v) internally.
# Output packs [token_outputs | state_snapshots]; the block slices the
# first T*B token columns before gated_out.
#
# Spinel hygiene: no Cfg ctor / no default args (landmine #4), no
# Card/step_bind, no FFI :str args. Fixed-arity FFI passthroughs only.
#
# This file does NOT require_relative "tinynn": the loading module loads
# the correct backend's TinyNN before requiring this primitive (mirror
# generator handles the TinyNN. -> TinyNN<Backend>. rename).

module Toy
  module LLM
    module Primitives
      module GDN
        NAME = :gdn

        # L2-normalise a projected q or k along its head dim (the delta
        # rule replaces softmax normalisation with L2-norm). x is the
        # block's already-projected (and conv'd) q or k tensor; eps the
        # Float epsilon. Returns the normalised handle. Called twice by
        # the block (once for q, once for k).
        def self.l2(sess, x, eps)
          TinyNN.tnn_l2_norm(sess, x, eps)
        end

        # Log-decay gate: g = -exp(A_log) * softplus(a + dt_bias). a is
        # the projected decay stream [1,H,T,B]; dt_bias and A_log are the
        # block's per-v-head weights ([1,H,1,1], broadcast). Returned g is
        # the raw LOG-decay the recurrence kernel exps internally. Op
        # order is fixed for ggml broadcast (the [1,H,T,B] softplus term
        # drives the shape; the [1,H,1,1] -exp(A_log) broadcasts onto it).
        def self.decay_gate(sess, a, dt_bias, a_log)
          a_db   = TinyNN.tnn_add(sess, a, dt_bias)
          sp     = TinyNN.tnn_softplus(sess, a_db)
          ea     = TinyNN.tnn_exp(sess, a_log)
          ea_neg = TinyNN.tnn_neg(sess, ea)
          TinyNN.tnn_mul(sess, sp, ea_neg)
        end

        # Update rate: beta = sigmoid(b). b is the projected update stream
        # [1,H,T,B]. The kernel uses beta directly, so the sigmoid lives
        # here. Returns beta.
        def self.update_gate(sess, b)
          TinyNN.tnn_sigmoid(sess, b)
        end

        # The recurrence core. q,k must be L2-normed; beta sigmoid'd; g the
        # raw log-decay; state the [S_v*S_v*H,K,B,1] carry. Returns the
        # packed [S_v*H, T*B + K*S_v*B] output (token outputs then state
        # snapshots). The block slices the leading T*B token columns.
        def self.recur(sess, q, k, v, g, beta, state)
          TinyNN.tnn_gated_delta_net(sess, q, k, v, g, beta, state)
        end

        # Gated output norm: GatedRMSNorm(o, z) = rms_norm(o) * gamma *
        # silu(z). o is the per-head token output (block-sliced from
        # recur); z the output-gate stream; gamma the block's norm weight;
        # eps the Float epsilon. tnn_rms_norm already folds the gamma
        # scale, so this is rms_norm(o,gamma) * silu(z). The normed term
        # drives the shape; silu(z) broadcasts/multiplies. Returns the
        # gated output (input to the block's out projection).
        def self.gated_out(sess, o, z, gamma, eps)
          n  = TinyNN.tnn_rms_norm(sess, o, gamma, eps)
          sz = TinyNN.tnn_silu(sess, z)
          TinyNN.tnn_mul(sess, n, sz)
        end
      end
    end
  end
end
