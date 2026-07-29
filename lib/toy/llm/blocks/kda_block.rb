# lib/toy/llm/blocks/kda_block.rb — L2 block: a TRAINABLE Kimi Delta
# Attention layer (Kimi K3 §2.1.1 — toy#137 / K-series M1, phase K2b).
# The KIND_KDA counterpart of TransformerBlock / GDNBlock.
#
# A SIBLING of GDNBlock, deliberately not a mode flag on it: GDN keeps its
# byte-gated Dragon/Qwen3-Next shape (gate-gdn-engine stays trivially
# safe), KDA gets its own layer kind, and the hybrid layer patterns K3
# uses (3 KDA : 1 global attention) become a per-layer kind list rather
# than a mode matrix.
#
# What differs from GDNBlock (the K3 deltas, all in Primitives::KDA):
#   decay      CHANNEL-wise α ∈ (e^-5,1)^{d_k} from a LOW-RANK projection
#              (W_α↑W_α↓x + b_α) through the lower-bounded scaled sigmoid
#              g = -5·σ(e^{A_h}z)   — GDN carries a per-head SCALAR from
#              an unbounded -e^{A}·softplus.
#   q/k/v      Swish before the L2Norm (q/k) and on v.
#   out gate   FULL-RANK σ(W_g h) ⊙ RMSNorm(õ)  — GDN uses silu(W_z h).
# The recurrence itself is Primitives::KDA.recur_unrolled, which is
# GDN.recur_unrolled with the decay view widened to [S_v,1]; the identity
# and its byte-exact reduction null live in prep/smokes/smoke_kda_recurrence.rb.
#
# ShortConv (K3 eq 2) IS wired (toy#137 K2c): a depthwise causal K=4
# conv on q/k/v before Swish, composed from shifted views
# (Primitives::KDA.short_conv) so its backward comes free. IDENTITY
# INIT (w[:,0]=1, rest 0) makes it a step-1 forward no-op — the null
# the gate pins — after which the taps train. kda_conv=0 disables it.
#
# STILL B=1: the projections reshape on seq_t (like GDNBlock), so one
# window per step. The runner fails loud on --batch > 1 + KDA layers.
#
# Shapes (single seq, B=1):
#   x            [d_model, T]
#   h = rmsnorm  [d_model, T]
#   q/k/v/wg     = W·h        -> [S_v*H, T]
#   z (decay)    = W_au·(W_ad·h) + b_α -> [S_v*H, T]   (low-rank, r = S_v)
#   b (write)    = W_b·h      -> [H, T]
#   per head h:  KDA.recur_unrolled(qn,kn,vs,g,beta, state0_h) -> [S_v, T]
#   o            concat heads -> [S_v*H, T]
#   gated        = σ(wg) ⊙ RMSNorm(o, γ) -> [S_v*H, T]
#   out = W_o·gated -> [d_model, T] ; residual = x + out
#
# Spinel hygiene: hand-written positional class, NEVER Struct.new
# (landmine #16); no Cfg ctor / default args (landmine #4); no
# Card/step_bind/FFI :str at class load; no require_relative "tinynn"
# (the loader picks the backend, as for the L1 primitives).

module Toy; module LLM; module Blocks
  class KDABlock
    attr_accessor :t_rn_gamma,
                  :t_w_q, :t_w_k, :t_w_v, :t_w_g, :t_w_b,
                  :t_w_ad, :t_w_au, :t_b_alpha, :t_a_log,
                  :t_go_gamma, :t_w_o,
                  :t_cq0, :t_cq1, :t_cq2, :t_cq3,
                  :t_ck0, :t_ck1, :t_ck2, :t_ck3,
                  :t_cv0, :t_cv1, :t_cv2, :t_cv3,
                  :t_zpad,
                  :t_state0,
                  :kda_d_model, :kda_s_v, :kda_n_heads, :kda_conv,
                  :ft_weights, :ft_m, :ft_v

    def initialize
      @kda_d_model = 0; @kda_s_v = 0; @kda_n_heads = 0; @kda_conv = 1
      np = TinyNN.tnn_null_ptr
      @t_cq0 = np; @t_cq1 = np; @t_cq2 = np; @t_cq3 = np
      @t_ck0 = np; @t_ck1 = np; @t_ck2 = np; @t_ck3 = np
      @t_cv0 = np; @t_cv1 = np; @t_cv2 = np; @t_cv3 = np
      @t_rn_gamma = TinyNN.tnn_null_ptr
      @t_w_q = TinyNN.tnn_null_ptr; @t_w_k = TinyNN.tnn_null_ptr; @t_w_v = TinyNN.tnn_null_ptr
      @t_w_g = TinyNN.tnn_null_ptr; @t_w_b = TinyNN.tnn_null_ptr
      @t_w_ad = TinyNN.tnn_null_ptr; @t_w_au = TinyNN.tnn_null_ptr
      @t_b_alpha = TinyNN.tnn_null_ptr; @t_a_log = TinyNN.tnn_null_ptr
      @t_go_gamma = TinyNN.tnn_null_ptr; @t_w_o = TinyNN.tnn_null_ptr
      @t_zpad = TinyNN.tnn_null_ptr
      @t_state0 = TinyNN.tnn_null_ptr
      @ft_weights = [TinyNN.tnn_null_ptr]; @ft_weights.pop
      @ft_m       = [TinyNN.tnn_null_ptr]; @ft_m.pop
      @ft_v       = [TinyNN.tnn_null_ptr]; @ft_v.pop
    end

    # d_model = residual width; n_heads × s_v = the KDA inner width. The
    # decay projection is LOW-RANK with r = s_v (K3's W_α↓/W_α↑ pair —
    # its purpose is parameter economy at d=7168; keeping the structure
    # here keeps the mechanism faithful and the rank an honest knob).
    # state0 is a zeroed [s_v, s_v*n_heads] constant carry, NOT a param.
    # KSIZE: K3's ShortConv kernel width (4, from Kimi Linear).
    KSIZE = 4

    def alloc_trainable_f32_weights!(sess, d_model, s_v, n_heads, conv_on)
      @kda_d_model = d_model; @kda_s_v = s_v; @kda_n_heads = n_heads
      @kda_conv = conv_on
      inner = s_v * n_heads
      rank  = s_v
      # input_2d_f32_persistent(rows, cols) -> ne0=cols, ne1=rows: pass
      # (out, in) so matmul(W, h) contracts ne0=in and yields [out, T].
      @t_rn_gamma = reg1(sess, d_model)
      @t_w_q  = reg2(sess, inner,   d_model)
      @t_w_k  = reg2(sess, inner,   d_model)
      @t_w_v  = reg2(sess, inner,   d_model)
      @t_w_g  = reg2(sess, inner,   d_model)   # full-rank output gate
      @t_w_b  = reg2(sess, n_heads, d_model)   # scalar write strength
      @t_w_ad = reg2(sess, rank,    d_model)   # decay down-proj
      @t_w_au = reg2(sess, inner,   rank)      # decay up-proj
      @t_b_alpha  = reg1(sess, inner)          # per-channel decay bias
      @t_a_log    = reg4(sess, 1, n_heads, 1, 1)  # per-head log-scale A_h
      @t_go_gamma = reg1(sess, inner)
      @t_w_o = reg2(sess, d_model, inner)
      if conv_on == 1
        # WHOLE 1d [inner] tensors per tap — never views of one packed
        # weight (a view as a mul operand breaks the backward; see
        # Primitives::KDA.short_conv4).
        @t_cq0 = reg1(sess, inner); @t_cq1 = reg1(sess, inner)
        @t_cq2 = reg1(sess, inner); @t_cq3 = reg1(sess, inner)
        @t_ck0 = reg1(sess, inner); @t_ck1 = reg1(sess, inner)
        @t_ck2 = reg1(sess, inner); @t_ck3 = reg1(sess, inner)
        @t_cv0 = reg1(sess, inner); @t_cv1 = reg1(sess, inner)
        @t_cv2 = reg1(sess, inner); @t_cv3 = reg1(sess, inner)
        # The conv's zero pad: a [inner, KSIZE-1] NON-param leaf (zeroed
        # in zero_state!). Not scale(x,0) — see short_conv4's note.
        @t_zpad = TinyNN.tnn_input_2d_f32_persistent(sess, KSIZE - 1, inner)
      end
      @t_state0 = TinyNN.tnn_input_2d_f32_persistent(sess, s_v, s_v * n_heads)
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

    def reg4(sess, a, b, c, d)
      w = TinyNN.tnn_input_4d_f32_persistent(sess, a, b, c, d)
      @ft_weights.push(w)
      @ft_m.push(TinyNN.tnn_input_4d_f32_persistent(sess, a, b, c, d))
      @ft_v.push(TinyNN.tnn_input_4d_f32_persistent(sess, a, b, c, d))
      w
    end

    # Mark every projection weight a trainable param. Call BEFORE
    # finalize_weights (load-bearing order). a_log + b_alpha ARE trained
    # (the decay shape); state0 is NOT (absent from ft_weights).
    def set_params!
      wi = 0
      while wi < @ft_weights.length
        TinyNN.tnn_set_param(@ft_weights[wi])
        wi = wi + 1
      end
    end

    def zero_state!(sess)
      TinyNN.tnn_zero_tensor(sess, @t_state0)
      if @kda_conv == 1
        TinyNN.tnn_zero_tensor(sess, @t_zpad)
      end
    end

    # Forward: residual update for x [d_model, T] (B=1). Returns [d_model, T].
    def build_forward(sess, t_x, seq_t, eps)
      d_model = @kda_d_model
      s_v     = @kda_s_v
      n_heads = @kda_n_heads
      fbytes  = 4
      inner   = s_v * n_heads
      h = Toy::LLM::Primitives::RMSNorm.build(sess, t_x, @t_rn_gamma, eps)

      q2 = TinyNN.tnn_matmul(sess, @t_w_q, h)   # [inner, T]
      k2 = TinyNN.tnn_matmul(sess, @t_w_k, h)
      v2 = TinyNN.tnn_matmul(sess, @t_w_v, h)
      g2 = TinyNN.tnn_matmul(sess, @t_w_g, h)   # [inner, T] gate stream
      b2 = TinyNN.tnn_matmul(sess, @t_w_b, h)   # [H, T] write stream
      # low-rank decay logits + per-channel bias (K3 eq 2's z term).
      zd = TinyNN.tnn_matmul(sess, @t_w_ad, h)  # [rank, T]
      zu = TinyNN.tnn_matmul(sess, @t_w_au, zd) # [inner, T]
      z2 = TinyNN.tnn_add(sess, zu, @t_b_alpha) # + b_α  ([inner] broadcasts)

      # ShortConv on q/k/v (K3 eq 2), before Swish. Identity-inited, so
      # step 1 is a forward no-op vs kda_conv=0.
      if @kda_conv == 1
        q2 = Toy::LLM::Primitives::KDA.short_conv4(sess, q2, @t_cq0, @t_cq1, @t_cq2, @t_cq3, @t_zpad, inner, seq_t)
        k2 = Toy::LLM::Primitives::KDA.short_conv4(sess, k2, @t_ck0, @t_ck1, @t_ck2, @t_ck3, @t_zpad, inner, seq_t)
        v2 = Toy::LLM::Primitives::KDA.short_conv4(sess, v2, @t_cv0, @t_cv1, @t_cv2, @t_cv3, @t_zpad, inner, seq_t)
      end
      q3 = TinyNN.tnn_reshape_3d(sess, q2, s_v, n_heads, seq_t)
      k3 = TinyNN.tnn_reshape_3d(sess, k2, s_v, n_heads, seq_t)
      v3 = TinyNN.tnn_reshape_3d(sess, v2, s_v, n_heads, seq_t)
      z3 = TinyNN.tnn_reshape_3d(sess, z2, s_v, n_heads, seq_t)
      b3 = TinyNN.tnn_reshape_3d(sess, b2, 1,   n_heads, seq_t)

      qn = Toy::LLM::Primitives::KDA.qk_prep(sess, q3, eps)
      kn = Toy::LLM::Primitives::KDA.qk_prep(sess, k3, eps)
      vs = Toy::LLM::Primitives::KDA.v_prep(sess, v3)
      gg = Toy::LLM::Primitives::KDA.decay_logits(sess, z3, @t_a_log)
      bt = Toy::LLM::Primitives::KDA.write_gate(sess, b3)

      o = TinyNN.tnn_null_ptr
      hh = 0
      while hh < n_heads
        st_h = TinyNN.tnn_view_2d(sess, @t_state0, s_v, s_v,
                                  s_v * fbytes, hh * s_v * s_v * fbytes)
        o_h = Toy::LLM::Primitives::KDA.recur_unrolled(sess, qn, kn, vs, gg, bt,
                                                       st_h, s_v, n_heads, hh, seq_t)
        if hh == 0
          o = o_h
        else
          o = TinyNN.tnn_concat(sess, o, o_h, 0)
        end
        hh = hh + 1
      end

      gated = Toy::LLM::Primitives::KDA.gated_out(sess, o, g2, @t_go_gamma, eps)
      out   = TinyNN.tnn_matmul(sess, @t_w_o, gated)
      TinyNN.tnn_add(sess, t_x, out)
    end
  end
end; end; end
