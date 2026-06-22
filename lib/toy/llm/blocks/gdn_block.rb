# lib/toy/llm/blocks/gdn_block.rb — L2 block: a TRAINABLE Gated-DeltaNet layer
# (Dragon / Qwen3-Next linear-attention mixer), the KIND_GDN counterpart of the
# attention TransformerBlock. Composes the L1 GDN primitives around its own
# projection weights and the autograd-differentiable recurrence
# (Toy::LLM::Primitives::GDN.recur_unrolled, Phase 4 / Path B) — so the whole
# layer trains with NO hand-written kernel backward.
#
# DEFERRED (Phase 5 minimal-trainable scope; revisit for Dragon bit-match): the
# short causal conv on q/k/v (ggml_conv_1d is FFI-wired from Phase 1) and any
# Dragon-exact stream layout. This block proves a GDN layer is a correct,
# trainable residual unit; it is not yet a bit-faithful Dragon block.
#
# Shapes (single seq, B=1):
#   x            [d_model, T]
#   h = rmsnorm  [d_model, T]
#   q/k/v/z      = W·h -> [S_v*H, T]   (W : [d_model, S_v*H])
#   a/b          = W·h -> [H, T]       (W : [d_model, H]) ; per-head scalars
#   per head h:  recur_unrolled(qn,kn,v,g,beta, state0_h) -> [S_v, T]
#   o            concat heads -> [S_v*H, T]
#   gated        = GatedRMSNorm(o, z) -> [S_v*H, T]
#   out = W_o·gated -> [d_model, T] ; residual = x + out
#
# Spinel hygiene: hand-written positional class, NEVER Struct.new (landmine #16);
# no Cfg ctor / default args (landmine #4); no Card/step_bind/FFI :str at class
# load. This file does NOT require_relative "tinynn" (the loader picks the
# backend before requiring this block, as for the L1 primitives + L2 attention
# block).

module Toy; module LLM; module Blocks
  class GDNBlock
    attr_accessor :t_rn_gamma,
                  :t_w_q, :t_w_k, :t_w_v, :t_w_z, :t_w_a, :t_w_b,
                  :t_a_log, :t_dt_bias, :t_go_gamma, :t_w_o,
                  :t_state0,
                  # F3 full-finetune parallel arrays (weight, m, v) — same
                  # convention as TransformerBlock so the engine's opt_step
                  # walker reaches them by name.
                  :ft_weights, :ft_m, :ft_v

    def initialize
      @t_rn_gamma = TinyNN.tnn_null_ptr
      @t_w_q = TinyNN.tnn_null_ptr; @t_w_k = TinyNN.tnn_null_ptr; @t_w_v = TinyNN.tnn_null_ptr
      @t_w_z = TinyNN.tnn_null_ptr; @t_w_a = TinyNN.tnn_null_ptr; @t_w_b = TinyNN.tnn_null_ptr
      @t_a_log = TinyNN.tnn_null_ptr; @t_dt_bias = TinyNN.tnn_null_ptr
      @t_go_gamma = TinyNN.tnn_null_ptr; @t_w_o = TinyNN.tnn_null_ptr
      @t_state0 = TinyNN.tnn_null_ptr
      @ft_weights = [TinyNN.tnn_null_ptr]; @ft_weights.pop
      @ft_m       = [TinyNN.tnn_null_ptr]; @ft_m.pop
      @ft_v       = [TinyNN.tnn_null_ptr]; @ft_v.pop
    end

    # Allocate the block's trainable persistent F32 weights + their Adam moments
    # (parallel ft_weights/ft_m/ft_v arrays, populated in lockstep so the engine
    # / a train loop can opt_step generically). d_model is the residual width;
    # n_heads × s_v = the GDN inner width. state0 is a zeroed [s_v, s_v*n_heads]
    # constant carry (one [s_v,s_v] block per head), NOT a param. Each weight's
    # m/v match its shape (opt_step_adamw asserts same-shape).
    def alloc_trainable_f32_weights!(sess, d_model, s_v, n_heads)
      inner = s_v * n_heads
      # W : [d_model, out]  (matmul(W, h) contracts ne0=d_model -> [out, T]).
      # input_2d_f32_persistent(rows, cols) -> ne0=cols, ne1=rows, so pass
      # (out, d_model) to get ne0=d_model, ne1=out.
      @t_rn_gamma = reg1(sess, d_model)
      @t_w_q = reg2(sess, inner,   d_model)
      @t_w_k = reg2(sess, inner,   d_model)
      @t_w_v = reg2(sess, inner,   d_model)
      @t_w_z = reg2(sess, inner,   d_model)
      @t_w_a = reg2(sess, n_heads, d_model)
      @t_w_b = reg2(sess, n_heads, d_model)
      @t_a_log    = reg4(sess, 1, n_heads, 1, 1)
      @t_dt_bias  = reg4(sess, 1, n_heads, 1, 1)
      @t_go_gamma = reg1(sess, inner)
      @t_w_o = reg2(sess, d_model, inner)
      # Constant zero initial state (NOT registered as a trainable param).
      @t_state0 = TinyNN.tnn_input_2d_f32_persistent(sess, s_v, s_v * n_heads)
    end

    # reg{1,2,4}: alloc a weight of the given rank + matching m/v, push the
    # triple into ft_weights/ft_m/ft_v, return the weight handle.
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

    # Mark every projection weight a trainable param. Call BEFORE finalize_weights
    # (load-bearing order, gpt2_seq_engine.rb:128). a_log + dt_bias ARE trained
    # (per-head decay shape); state0 is NOT (it is not in ft_weights).
    def set_params!
      wi = 0
      while wi < @ft_weights.length
        TinyNN.tnn_set_param(@ft_weights[wi])
        wi = wi + 1
      end
    end

    # Zero the constant initial state (after finalize_weights).
    def zero_state!(sess)
      TinyNN.tnn_zero_tensor(sess, @t_state0)
    end

    # Forward: residual update for x [d_model, T] (B=1). Returns [d_model, T].
    def build_forward(sess, t_x, d_model, s_v, n_heads, seq_t, eps)
      fbytes = 4
      h = Toy::LLM::Primitives::RMSNorm.build(sess, t_x, @t_rn_gamma, eps)

      q2 = TinyNN.tnn_matmul(sess, @t_w_q, h)   # [S_v*H, T]
      k2 = TinyNN.tnn_matmul(sess, @t_w_k, h)
      v2 = TinyNN.tnn_matmul(sess, @t_w_v, h)
      z2 = TinyNN.tnn_matmul(sess, @t_w_z, h)   # [S_v*H, T] output gate
      a2 = TinyNN.tnn_matmul(sess, @t_w_a, h)   # [H, T] decay stream
      b2 = TinyNN.tnn_matmul(sess, @t_w_b, h)   # [H, T] update stream

      # Reshape projections into the recurrence's packed [S_v, H, T] / [1, H, T].
      q3 = TinyNN.tnn_reshape_3d(sess, q2, s_v, n_heads, seq_t)
      k3 = TinyNN.tnn_reshape_3d(sess, k2, s_v, n_heads, seq_t)
      v3 = TinyNN.tnn_reshape_3d(sess, v2, s_v, n_heads, seq_t)
      a3 = TinyNN.tnn_reshape_3d(sess, a2, 1,   n_heads, seq_t)
      b3 = TinyNN.tnn_reshape_3d(sess, b2, 1,   n_heads, seq_t)

      qn = Toy::LLM::Primitives::GDN.l2_train(sess, q3, eps)
      kn = Toy::LLM::Primitives::GDN.l2_train(sess, k3, eps)
      g  = Toy::LLM::Primitives::GDN.decay_gate(sess, a3, @t_dt_bias, @t_a_log)
      bt = Toy::LLM::Primitives::GDN.update_gate_train(sess, b3)

      # Per-head recurrence; concat head outputs along ne0 -> [S_v*H, T].
      o = TinyNN.tnn_null_ptr
      hh = 0
      while hh < n_heads
        st_h = TinyNN.tnn_view_2d(sess, @t_state0, s_v, s_v,
                                  s_v * fbytes, hh * s_v * s_v * fbytes)
        o_h = Toy::LLM::Primitives::GDN.recur_unrolled(sess, qn, kn, v3, g, bt,
                                                       st_h, s_v, n_heads, hh, seq_t)
        if hh == 0
          o = o_h
        else
          o = TinyNN.tnn_concat(sess, o, o_h, 0)
        end
        hh = hh + 1
      end

      gated = Toy::LLM::Primitives::GDN.gated_out(sess, o, z2, @t_go_gamma, eps)
      out   = TinyNN.tnn_matmul(sess, @t_w_o, gated)   # [d_model, T]
      TinyNN.tnn_add(sess, t_x, out)
    end
  end
end; end; end
