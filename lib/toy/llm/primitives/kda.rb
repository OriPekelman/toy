# lib/toy/llm/primitives/kda.rb — L1 primitive: Kimi Delta Attention
# (KDA, Kimi K3 §2.1.1 — toy#137 / K-series M1).
#
# KDA extends the gated delta rule with a CHANNEL-WISE forget gate:
#
#   S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ
#   õ_t = S_tᵀ q_t
#
# where α_t ∈ (0,1)^{d_k} is the per-CHANNEL one-step retention factor
# (GDN/Dragon carries a per-HEAD SCALAR there) and β_t ∈ (0,1) the
# scalar write strength.
#
# ── The identity that makes this a small change ──
# Expand KDA's update with S' = Diag(α)S_{t−1}:
#   S_t = S' − β k kᵀS' + β k vᵀ = S' + k·(β(v − S'ᵀk))ᵀ
# which is EXACTLY the sequence Primitives::GDN.recur_unrolled already
# builds (u = S'ᵀk; d = β(v−u); S = S' + k⊗d; o = S ᵀq, verified against
# the fused ggml gated_delta_net kernel). The ONLY structural delta is
# the decay step: GDN multiplies the state by a [1,1] scalar, KDA by an
# [S_v,1] column that ggml broadcasts along ne1 — and because the state's
# ne0 IS the contracted key index, `mul(S, α)` gives S'[i,j] = S[i,j]·α[i]
# = Diag(α)S. Hence recur_unrolled_kda below is recur_unrolled with one
# view width changed; the reduction null (α uniform across channels ⇒
# byte-equal to the GDN scalar path) is what the gate pins.
#
# ── The decay parameterization (K3 eq 2 + eq 5) ──
#   z_t = W_α↑ W_α↓ x_t + b_α        (low-rank + per-head channel bias)
#   g_t = g_min · σ(e^{A_h} z_t)     ∈ (g_min, 0)^{d_k}   [LOWER-BOUNDED]
#   α_t = exp(g_t)                   ∈ (e^{g_min}, 1)
# with g_min = −5 fixed and A_h a learnable per-head log-scale (init 0).
# Kimi Linear used an UNBOUNDED −e^{A}·softplus(z); K3's scaled sigmoid
# bounds the log-decay from below so the chunkwise form's reciprocal
# cumulative decay stays inside BF16 range (their diagonal-tile
# tensor-core win). We keep the bound because it is also the numerically
# safer parameterization for the recurrent form, and because it is the
# mechanism as published.
#
# The recurrence here exponentiates the LOG-decay per token (the GDN
# kernel contract), so decay_logits returns g, not α.
#
# q/k/v prep (K3 eq 2): Swish then L2Norm on q/k, Swish on v. The
# ShortConv that K3 applies before Swish is deferred with the same
# rationale the GDN block already records for its own conv (it is a
# locality prior, orthogonal to the recurrence; K2b).
#
# Pure module, `self.` methods only; no Cfg ctor / default args
# (landmine #4); no require_relative "tinynn" (the loader picks the
# backend; the mirror generator renames TinyNN. → TinyNN<Backend>.).
# Every op used has a ggml backward — tanh/sigmoid arrived with
# vendor-patch 0013 — so training backward comes free.

module Toy
  module LLM
    module Primitives
      module KDA
        NAME = :kda

        # K3 fixes the log-decay floor at -5 (α ≥ e^-5 ≈ 6.7e-3, so the
        # cumulative decay over a 16-token tile stays in (-80, 0)).
        G_MIN = -5.0

        # Swish/SiLU then trainable L2Norm over ne0 — the q/k path.
        # (l2_train, not the fused L2_NORM op, because the fused op has
        # no ggml backward; GDN.l2_train established the composition.)
        def self.qk_prep(sess, x, eps)
          Toy::LLM::Primitives::GDN.l2_train(sess, TinyNN.tnn_silu(sess, x), eps)
        end

        # Swish only — the v path.
        def self.v_prep(sess, x)
          TinyNN.tnn_silu(sess, x)
        end

        # LOWER-BOUNDED channel-wise log-decay (K3 eq 5).
        #   g = G_MIN * sigmoid(exp(a_log) * z)
        # z is the projected decay stream packed like q/k/v ([S_v,H,T]);
        # a_log the learnable per-head log-scale ([1,H,1] — broadcast
        # over the channel and token axes). Returns g in (G_MIN, 0),
        # channel-wise. The caller hands g straight to the recurrence
        # (which exps it into α).
        def self.decay_logits(sess, z, a_log)
          ea = TinyNN.tnn_exp(sess, a_log)
          zs = TinyNN.tnn_mul(sess, z, ea)
          TinyNN.tnn_scale(sess, TinyNN.tnn_sigmoid(sess, zs), G_MIN)
        end

        # Scalar write strength β = sigmoid(b) (K3 eq 2). Composed via
        # tnn_sigmoid — patch 0013 gave it a backward, so unlike
        # GDN.update_gate_train there is no need for the exp/div
        # workaround.
        def self.write_gate(sess, b)
          TinyNN.tnn_sigmoid(sess, b)
        end

        # ONE head of the KDA recurrence — GDN.recur_unrolled with the
        # decay view widened from [1,1] to [S_v,1] (channel-wise). q,k
        # must be qk_prep'd, beta write_gate'd, g the channel-wise
        # log-decay. Layouts: q,k,v,g packed [S_v,H,T]; beta [1,H,T];
        # state0 this head's [S_v,S_v]. Returns [S_v,T].
        #
        # The 1/√S_v output scale is folded into ONE pre-scale of the
        # contiguous q (q enters only the output read) — same reason as
        # the GDN twin: a per-token scale's backward receives a
        # view-shaped grad from the concat and trips
        # ggml_is_padded_1d.
        def self.recur_unrolled(sess, q, k, v, g, beta, state0, s_v, n_heads, head, n_tokens)
          scale = 1.0 / Math.sqrt(s_v.to_f)
          fbytes = 4
          tok_stride  = s_v * n_heads * fbytes
          head_base   = s_v * head * fbytes
          gtok_stride = n_heads * fbytes
          ghead_base  = head * fbytes
          q_s = TinyNN.tnn_scale(sess, q, scale)
          s_mat = state0
          t_out = TinyNN.tnn_null_ptr
          t = 0
          while t < n_tokens
            q_t = TinyNN.tnn_view_2d(sess, q_s, s_v, 1, tok_stride, head_base + t * tok_stride)
            k_t = TinyNN.tnn_view_2d(sess, k,   s_v, 1, tok_stride, head_base + t * tok_stride)
            v_t = TinyNN.tnn_view_2d(sess, v,   s_v, 1, tok_stride, head_base + t * tok_stride)
            # THE KDA line: an [S_v,1] decay column (vs GDN's [1,1]).
            g_t = TinyNN.tnn_view_2d(sess, g,   s_v, 1, tok_stride, head_base + t * tok_stride)
            b_t = TinyNN.tnn_view_2d(sess, beta, 1, 1, gtok_stride, ghead_base + t * gtok_stride)

            alpha = TinyNN.tnn_exp(sess, g_t)              # [S_v,1] in (e^G_MIN,1)
            s_dec = TinyNN.tnn_mul(sess, s_mat, alpha)     # Diag(α)S: [i,j]*α[i]
            u     = TinyNN.tnn_matmul(sess, s_dec, k_t)    # u[j] = Σ_i S'[i,j]k[i]
            diff  = TinyNN.tnn_sub(sess, v_t, u)
            d     = TinyNN.tnn_mul(sess, diff, b_t)        # β(v−u), scalar bcast
            k_row = TinyNN.tnn_reshape_2d(sess, k_t, 1, s_v)
            d_row = TinyNN.tnn_reshape_2d(sess, d, 1, s_v)
            outer = TinyNN.tnn_matmul(sess, k_row, d_row)  # k⊗d
            s_mat = TinyNN.tnn_add(sess, s_dec, outer)
            o_t   = TinyNN.tnn_matmul(sess, s_mat, q_t)

            if t == 0
              t_out = o_t
            else
              t_out = TinyNN.tnn_concat(sess, t_out, o_t, 1)
            end
            t = t + 1
          end
          t_out
        end

        # FULL-RANK output gate (K3 eq 6): y_pre = σ(W_g x) ⊙ RMSNorm(õ).
        # The caller owns W_g (and applies the out projection). o is the
        # per-head-concat recurrence output, t_wgx the already-projected
        # gate stream, gamma the norm weight. K3 changed this from Kimi
        # Linear's low-rank gate to a full-rank projection; full-rank
        # here means the caller's W_g is [d_model, S_v*H] (a plain
        # matmul), which is what the block allocates.
        def self.gated_out(sess, o, t_wgx, gamma, eps)
          n = TinyNN.tnn_rms_norm(sess, o, gamma, eps)
          TinyNN.tnn_mul(sess, n, TinyNN.tnn_sigmoid(sess, t_wgx))
        end
      end
    end
  end
end
