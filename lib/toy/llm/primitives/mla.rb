# lib/toy/llm/primitives/mla.rb — L1 primitive: the Gated Multi-head
# Latent Attention pieces (Kimi K3 §2.1.2 — K-series M2, the TRAINING
# half).
#
# MLA already exists in toy as an INFERENCE path: the deepseek2 loader +
# KV engine read DeepSeek-V2's kv_lora_rank / asymmetric K(192)/V(128)
# cache / decoupled YaRN RoPE (prep/deepseek_mla_gate.rb). None of that
# is reachable from the SEQUENCE engine, which is where training lives —
# so a K3-shaped hybrid stack had no MLA to put in its "1" slot. This
# primitive is the training-side half.
#
# ── What K3 changes vs DeepSeek MLA ──
#   NoPE        no positional encoding on MLA layers AT ALL (M8) — the
#               position information lives in the KDA recurrence. That
#               deletes DeepSeek's whole decoupled-RoPE apparatus: no
#               nope/rope query split, no shared per-token rope key, no
#               YaRN mscale. What remains is a plain latent sandwich.
#   gate        the same channel-wise FULL-RANK output gate KDA uses:
#               y = W_o[σ(W_g x) ⊙ RMSNorm(õ)]. It is literally
#               Primitives::KDA.gated_out — shared here rather than
#               duplicated, because it IS one mechanism in the report
#               (§2.1.1 and §2.1.2 cite the same gate).
#   fp32 attn   K3 keeps the attention output in FP32 during training.
#               We are f32 throughout at toy scale, so this is satisfied
#               by construction rather than by a cast — noted, not hidden.
#
# ── The latent sandwich, and the null that proves it ──
#   c_t = W_kv_a · h          [r, T]      r = kv_lora_rank (r < inner)
#   c_t = RMSNorm(c_t, γ_kv)              DeepSeek's kv_a_norm
#   K   = W_k_b · c_t         [inner, T]
#   V   = W_v_b · c_t         [inner, T]
# The point is parameter economy: 2·inner·d_model becomes
# r·d_model + 2·inner·r, a win whenever r < (2·inner·d_model)/(d_model +
# 2·inner). With the norm OFF, r = inner and W_kv_a = I, the sandwich
# collapses to K = W_k_b · h — an ORDINARY projection. That reduction is
# exact and is what prep/smokes/smoke_mla_latent.rb pins byte-for-byte:
# it proves the factorization is a factorization and not merely a
# plausible-looking stack of matmuls. (kv_norm is a flag ONLY so this
# null is expressible; K3 and DeepSeek both keep the norm on, which is
# the default.)
#
# The attention math itself is NOT re-derived here: per head it is
# exactly Primitives::GQA.attention (scores → scale → causal
# diag_mask_inf → softmax → weighted V), the same core the Llama block
# uses. Reusing it means MLA inherits the causal masking that path
# already has gated, instead of growing a second mask implementation to
# get subtly wrong.
#
# Every op has a ggml backward, so training comes free.
#
# Pure module, `self.` methods only; no Cfg ctor / default args
# (landmine #4); no require_relative "tinynn" (the loader picks the
# backend; the mirror generator renames TinyNN. → TinyNN<Backend>.).

module Toy
  module LLM
    module Primitives
      module MLA
        NAME = :mla

        # The KV latent: down-project then (optionally) RMSNorm. h is
        # [d_model, T], w_kv_a is [r, d_model], returns [r, T].
        #
        # use_norm = 0 exists for the factorization null (see header);
        # DeepSeek/K3 run it at 1.
        def self.kv_latent(sess, h, w_kv_a, gamma, use_norm, eps)
          c = TinyNN.tnn_matmul(sess, w_kv_a, h)
          if use_norm == 1
            c = Toy::LLM::Primitives::RMSNorm.build(sess, c, gamma, eps)
          end
          c
        end

        # One head of NoPE causal latent attention. q_h/k_h/v_h are
        # CONTIGUOUS [s_v, T] per-head streams; returns [s_v, T].
        #
        # WHY PER-HEAD TENSORS AND NOT SLICES OF A PACKED [inner, T].
        # The obvious shape is one big projection plus a strided view
        # per head, made contiguous for the matmul. It does not survive
        # the backward: GGML_OP_CONT's gradient rule asserts BOTH that
        # the incoming grad and the source's accumulated grad are
        # contiguous (ggml.c:6709), and a cont-of-a-view or
        # cont-of-a-transpose in the loss path violates it. (Muon uses
        # cont_2d(transpose(...)) safely only because the optimizer step
        # lives OUTSIDE the loss graph and is never differentiated.)
        # Projecting per head — the TransformerBlock shape — produces
        # contiguous [s_v, T] operands with no view in the grad path at
        # all, which is the arrangement this codebase has already gated
        # for training.
        #
        # batch is pinned to 1 (the B=1 branch of GQA.attention, which
        # masks with diag_mask_inf), so MLA layers carry the same B=1
        # restriction the KDA/GDN blocks do.
        def self.head_attend(sess, q_h, k_h, v_h, s_v)
          # A bare transpose IS safe here: it feeds mul_mat directly, and
          # mul_mat's backward handles a transposed operand. It is
          # wrapping one in CONT that breaks (see above).
          v_t = TinyNN.tnn_transpose(sess, v_h)
          scale = 1.0 / Math.sqrt(s_v.to_f)
          Toy::LLM::Primitives::GQA.attention(sess, k_h, q_h, v_t,
                                              TinyNN.tnn_null_ptr, scale, 1)
        end

        # Parameter count of the latent K/V path vs the two ordinary
        # projections it replaces. Used by the cost accounting and by
        # the gate's economy leg — the latent is only worth its
        # complexity when this is a saving, and saying so in numbers
        # keeps the mechanism honest at toy shapes (where r is often NOT
        # small enough to win).
        def self.kv_params_latent(d_model, inner, r)
          r * d_model + 2 * (inner * r)
        end

        def self.kv_params_plain(d_model, inner)
          2 * (inner * d_model)
        end
      end
    end
  end
end
