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

        # TRAINABLE L2 norm over ne0 — composed from ops that each have a ggml
        # backward (mul / sum_rows / scale_bias / sqrt / div), because the fused
        # `tnn_l2_norm` (GGML_OP_L2_NORM) has NO backward. Used by the trainable
        # GDN block; the fused `l2` above stays the inference path.
        #   y = x / sqrt(sum_ne0(x^2) + eps)
        def self.l2_train(sess, x, eps)
          sq     = TinyNN.tnn_mul(sess, x, x)            # x^2
          ss     = TinyNN.tnn_sum_rows(sess, sq)         # sum over ne0 -> [1,...]
          ss_eps = TinyNN.tnn_scale_bias(sess, ss, 1.0, eps)  # + eps
          denom  = TinyNN.tnn_sqrt(sess, ss_eps)         # [1,...]
          # DIV backward does NOT reduce a broadcast src1, so materialise denom to
          # x's full shape first (REPEAT backward sums the grad back correctly);
          # the div is then same-shape.
          denom_full = TinyNN.tnn_repeat(sess, denom, x)
          TinyNN.tnn_div(sess, x, denom_full)
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

        # TRAINABLE update gate — sigmoid(b) composed as exp(b)/(1+exp(b)) from
        # ops that each have a ggml backward, because GGML_UNARY_OP_SIGMOID has
        # none. Same-shape throughout (no broadcast). The fused `update_gate`
        # above (tnn_sigmoid) stays the inference path.
        def self.update_gate_train(sess, b)
          e = TinyNN.tnn_exp(sess, b)                    # exp(b)
          d = TinyNN.tnn_scale_bias(sess, e, 1.0, 1.0)   # 1 + exp(b)
          TinyNN.tnn_div(sess, e, d)
        end

        # The recurrence core. q,k must be L2-normed; beta sigmoid'd; g the
        # raw log-decay; state the [S_v*S_v*H,K,B,1] carry. Returns the
        # packed [S_v*H, T*B + K*S_v*B] output (token outputs then state
        # snapshots). The block slices the leading T*B token columns.
        def self.recur(sess, q, k, v, g, beta, state)
          TinyNN.tnn_gated_delta_net(sess, q, k, v, g, beta, state)
        end

        # Path-B TRAINABLE recurrence: the gated delta rule expressed as an
        # UNROLLED graph of ops that EACH have a ggml backward (mul / mul_mat /
        # sub / scale / exp / add / reshape) — so training backward comes free
        # and NO fused-kernel backward is needed (ggml has none for
        # GATED_DELTA_NET). The fused `recur` above stays the fast INFERENCE
        # path; this is its train-time twin, gated for numeric parity.
        #
        # Reproduces the fused kernel's token outputs for the SCALAR-decay path
        # (g->ne0 == 1, the Dragon/Qwen3-Next per-head gate). Single seq (B=1),
        # single head per call — the block loops heads/seqs around it in Phase 5.
        # Inputs are the packed projection tensors (q,k,v = [S_v,1,T,1]; g,beta =
        # [1,1,T,1]; state0 = [S_v,S_v]); per-token vectors are sliced via views
        # internally (no ptr-array params → no Spinel IntArray-lock landmine).
        # q/k must be pre-L2-normed and beta pre-sigmoid'd by the caller (the
        # kernel contract). Returns [S_v, T] — token outputs concat'd along ne1.
        #
        #   per token t (matching ops.cpp:10731 exactly):
        #     S = S * exp(g_t)                  decay  (scalar [1,1] broadcast)
        #     u = matmul(S, k_t)                u[j] = sum_i S[i,j] k[i]
        #     d = (v_t - u) * beta_t            delta
        #     S = S + matmul(k_row, d_row)      outer  (k⊗d)[i,j] = k[i] d[j]
        #     o_t = matmul(S, q_t)              o[j] = sum_i S[i,j] q[i]
        #
        # The kernel's 1/√S_v output scale is folded into a SINGLE pre-scale of q
        # (q enters only the output read, never the state, so o[j] = sum_i S[i,j]
        # (scale·q[i]) is exact). Done once on the contiguous q — NOT per-token on
        # o — because a per-token ggml_scale's BACKWARD receives a view-shaped grad
        # from the concat and asserts ggml_is_padded_1d (ggml.c:3392). One scale on
        # the whole tensor keeps the backward grad contiguous.
        #
        # ONE head of the recurrence. q,k,v are the packed [S_v, n_heads, T, 1]
        # projections; g,beta the packed [1, n_heads, T, 1] gates; state0 this
        # head's [S_v,S_v] carry. `head` selects the head; per-token vectors are
        # strided views into the packed tensors (token stride = S_v·n_heads, head
        # base = S_v·head — the ggml [S_v,H,T,B] layout). Returns [S_v, T] for this
        # head; the block concats heads along ne0. n_heads=1/head=0 is the plain
        # single-head case (contiguous per-token, the Phase-4 gate shape).
        def self.recur_unrolled(sess, q, k, v, g, beta, state0, s_v, n_heads, head, n_tokens)
          scale = 1.0 / Math.sqrt(s_v.to_f)
          fbytes = 4                          # sizeof(f32)
          tok_stride  = s_v * n_heads * fbytes # bytes between this head's tokens
          head_base   = s_v * head * fbytes    # byte offset to this head's col 0
          gtok_stride = n_heads * fbytes       # g/beta [1,H,T,1]: token stride
          ghead_base  = head * fbytes
          q_s = TinyNN.tnn_scale(sess, q, scale)   # pre-scaled q (contiguous)
          s_mat = state0
          t_out = TinyNN.tnn_null_ptr
          t = 0
          while t < n_tokens
            # Per-token slices: [S_v,1] vectors (S_v contiguous), [1,1] scalars.
            q_t = TinyNN.tnn_view_2d(sess, q_s, s_v, 1, tok_stride, head_base + t * tok_stride)
            k_t = TinyNN.tnn_view_2d(sess, k,   s_v, 1, tok_stride, head_base + t * tok_stride)
            v_t = TinyNN.tnn_view_2d(sess, v,   s_v, 1, tok_stride, head_base + t * tok_stride)
            g_t = TinyNN.tnn_view_2d(sess, g,    1, 1, gtok_stride, ghead_base + t * gtok_stride)
            b_t = TinyNN.tnn_view_2d(sess, beta, 1, 1, gtok_stride, ghead_base + t * gtok_stride)

            eg    = TinyNN.tnn_exp(sess, g_t)              # [1,1]
            s_dec = TinyNN.tnn_mul(sess, s_mat, eg)        # [S_v,S_v] * [1,1] bcast
            u     = TinyNN.tnn_matmul(sess, s_dec, k_t)    # [S_v,1]  u[j]
            diff  = TinyNN.tnn_sub(sess, v_t, u)           # [S_v,1]
            d     = TinyNN.tnn_mul(sess, diff, b_t)        # [S_v,1] * [1,1] bcast
            k_row = TinyNN.tnn_reshape_2d(sess, k_t, 1, s_v)  # [1,S_v]
            d_row = TinyNN.tnn_reshape_2d(sess, d, 1, s_v)    # [1,S_v]
            outer = TinyNN.tnn_matmul(sess, k_row, d_row)  # [S_v,S_v] [i,j]=k[i]d[j]
            s_mat = TinyNN.tnn_add(sess, s_dec, outer)     # state update
            o_t   = TinyNN.tnn_matmul(sess, s_mat, q_t)    # [S_v,1]  o[j] (already scaled)

            if t == 0
              t_out = o_t
            else
              t_out = TinyNN.tnn_concat(sess, t_out, o_t, 1)  # stack along ne1
            end
            t = t + 1
          end
          t_out
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
