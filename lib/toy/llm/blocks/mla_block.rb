# lib/toy/llm/blocks/mla_block.rb — L2 block: a TRAINABLE Gated
# Multi-head Latent Attention layer (Kimi K3 §2.1.2 — K-series M2, the
# training half). The KIND_MLA counterpart of TransformerBlock /
# GDNBlock / KDABlock.
#
# WHY THIS EXISTS. toy already had MLA — as INFERENCE, on the KV engine,
# for loaded DeepSeek-V2 weights (prep/deepseek_mla_gate.rb). The
# sequence engine, where training happens, had no MLA layer kind at all,
# so `--layer-pattern hybrid` filled K3's "1" slot with ORDINARY
# attention. That made the shipped hybrid 3 KDA : 1 Llama-attention,
# which is not the K3 contract (3 KDA : 1 Gated MLA). This block closes
# that gap; `--layer-pattern k3` is the faithful pattern, and `hybrid`
# is left exactly as it was so its gates stay byte-null.
#
# A SIBLING of KDABlock, same reasoning as KDA-vs-GDN: its own kind, its
# own typed block array, no mode flag on an existing block.
#
# Shapes (single seq, B=1):
#   x            [d_model, T]
#   h = rmsnorm  [d_model, T]
#   q  = W_q·h                 -> [S_v*H, T]
#   c  = RMSNorm(W_kv_a·h)     -> [r, T]        the LATENT (kv_a_norm)
#   k  = W_k_b·c               -> [S_v*H, T]
#   v  = W_v_b·c               -> [S_v*H, T]
#   per head: causal softmax attention, NoPE     -> [S_v, T]
#   o  = concat heads          -> [S_v*H, T]
#   gated = σ(W_g·h) ⊙ RMSNorm(o, γ_o)          the full-rank gate
#   out = W_o·gated -> [d_model, T] ; residual = x + out
#
# NO POSITIONAL ENCODING anywhere (K3's M8): position lives in the KDA
# recurrence, so an MLA layer in a k3 stack is deliberately
# permutation-equivariant on its own. That is a property, not an
# oversight — a lone MLA layer with no KDA beneath it has no notion of
# order at all, which the gate states rather than hides.
#
# NO FFN sublayer, matching KDABlock/GDNBlock: these kinds replace the
# ATTENTION sublayer, and the stack's feed-forward capacity comes from
# the KIND_ATTENTION layers (or, in the MoE instrument, the expert
# block). Noted because K3's own layers do carry an FFN/MoE sublayer —
# at toy scale the block seam is attention-substitution, and pretending
# otherwise would double-count parameters in the cost accounting.
#
# STILL B=1: the head slicing reshapes on seq_t exactly like KDABlock,
# so one window per step.
#
# Spinel hygiene: hand-written positional class, NEVER Struct.new
# (landmine #16); no Cfg ctor / default args (landmine #4); no
# Card/step_bind/FFI :str at class load; no require_relative "tinynn".

module Toy; module LLM; module Blocks
  class MLABlock
    attr_accessor :t_rn_gamma,
                  :t_w_q, :t_w_kv_a, :t_kv_gamma, :t_w_k_b, :t_w_v_b,
                  :t_w_g, :t_go_gamma, :t_w_o,
                  :mla_d_model, :mla_s_v, :mla_n_heads, :mla_rank,
                  :mla_kv_norm, :mla_gate,
                  :ft_weights, :ft_m, :ft_v

    def initialize
      @mla_d_model = 0; @mla_s_v = 0; @mla_n_heads = 0; @mla_rank = 0
      @mla_kv_norm = 1
      @mla_gate    = 1
      np = TinyNN.tnn_null_ptr
      @t_rn_gamma = np
      # PER-HEAD projection arrays (the TransformerBlock shape). See
      # Primitives::MLA.head_attend for why this is not one packed
      # matrix plus views: cont-of-a-view has no legal backward here.
      @t_w_q   = [np]; @t_w_q.pop
      @t_w_k_b = [np]; @t_w_k_b.pop
      @t_w_v_b = [np]; @t_w_v_b.pop
      @t_w_kv_a = np; @t_kv_gamma = np
      @t_w_g = np; @t_go_gamma = np; @t_w_o = np
      @ft_weights = [np]; @ft_weights.pop
      @ft_m       = [np]; @ft_m.pop
      @ft_v       = [np]; @ft_v.pop
    end

    # d_model = residual width; n_heads × s_v = the attention inner
    # width; rank = the KV latent width r (DeepSeek's kv_lora_rank).
    # kv_norm / gate are 0/1 knobs that exist so the reduction nulls are
    # expressible (see Primitives::MLA's header); K3 runs both at 1.
    def alloc_trainable_f32_weights!(sess, d_model, s_v, n_heads, rank, kv_norm, gate_on)
      @mla_d_model = d_model; @mla_s_v = s_v; @mla_n_heads = n_heads
      @mla_rank = rank; @mla_kv_norm = kv_norm; @mla_gate = gate_on
      inner = s_v * n_heads
      # input_2d_f32_persistent(rows, cols) -> ne0=cols, ne1=rows: pass
      # (out, in) so matmul(W, h) contracts ne0=in and yields [out, T].
      @t_rn_gamma = reg1(sess, d_model)
      @t_w_kv_a = reg2(sess, rank, d_model)      # latent down-projection
      hq = 0
      while hq < n_heads
        @t_w_q.push(reg2(sess, s_v, d_model))    # per-head query
        @t_w_k_b.push(reg2(sess, s_v, rank))     # latent -> this head's K
        @t_w_v_b.push(reg2(sess, s_v, rank))     # latent -> this head's V
        hq = hq + 1
      end
      if kv_norm == 1
        @t_kv_gamma = reg1(sess, rank)           # DeepSeek's kv_a_norm
      end
      if gate_on == 1
        @t_w_g      = reg2(sess, inner, d_model) # full-rank output gate
        @t_go_gamma = reg1(sess, inner)
      end
      @t_w_o = reg2(sess, d_model, inner)
    end

    def reg1(sess, n)
      w = TinyNN.tnn_input_1d_f32_persistent(sess, n)
      @ft_weights.push(w)
      @ft_m.push(TinyNN.tnn_input_1d_f32_persistent(sess, n))
      @ft_v.push(TinyNN.tnn_input_1d_f32_persistent(sess, n))
      w
    end

    def reg2(sess, rows, cols)
      w = TinyNN.tnn_input_2d_f32_persistent(sess, rows, cols)
      @ft_weights.push(w)
      @ft_m.push(TinyNN.tnn_input_2d_f32_persistent(sess, rows, cols))
      @ft_v.push(TinyNN.tnn_input_2d_f32_persistent(sess, rows, cols))
      w
    end

    # Mark every projection weight a trainable param. Call BEFORE
    # finalize_weights (load-bearing order).
    def set_params!
      wi = 0
      while wi < @ft_weights.length
        TinyNN.tnn_set_param(@ft_weights[wi])
        wi = wi + 1
      end
    end

    # Forward: residual update for x [d_model, T] (B=1). Returns
    # [d_model, T].
    def build_forward(sess, t_x, seq_t, eps)
      s_v     = @mla_s_v
      n_heads = @mla_n_heads
      inner   = s_v * n_heads
      h = Toy::LLM::Primitives::RMSNorm.build(sess, t_x, @t_rn_gamma, eps)

      # ONE shared latent for every head — that IS the mechanism (K/V
      # capacity is bought once at width r, then read out per head).
      c = Toy::LLM::Primitives::MLA.kv_latent(sess, h, @t_w_kv_a,
                                              @t_kv_gamma, @mla_kv_norm, eps)

      o = TinyNN.tnn_null_ptr
      hh = 0
      while hh < n_heads
        q_h = TinyNN.tnn_matmul(sess, @t_w_q[hh],   h)   # [s_v, T]
        k_h = TinyNN.tnn_matmul(sess, @t_w_k_b[hh], c)   # [s_v, T]
        v_h = TinyNN.tnn_matmul(sess, @t_w_v_b[hh], c)   # [s_v, T]
        o_h = Toy::LLM::Primitives::MLA.head_attend(sess, q_h, k_h, v_h, s_v)
        if hh == 0
          o = o_h
        else
          o = TinyNN.tnn_concat(sess, o, o_h, 0)
        end
        hh = hh + 1
      end

      gated = o
      if @mla_gate == 1
        # THE SAME gate KDA uses — σ(W_g h) ⊙ RMSNorm(õ). Shared, not
        # duplicated: it is one mechanism in the report.
        wgx   = TinyNN.tnn_matmul(sess, @t_w_g, h)            # [inner, T]
        gated = Toy::LLM::Primitives::KDA.gated_out(sess, o, wgx, @t_go_gamma, eps)
      end
      out = TinyNN.tnn_matmul(sess, @t_w_o, gated)
      TinyNN.tnn_add(sess, t_x, out)
    end
  end
end; end; end
