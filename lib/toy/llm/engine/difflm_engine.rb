# lib/toy/llm/engine/difflm_engine.rb — the toy#166 (capstone P1b)
# LATENT DIFFUSION DENOISER over PER-POSITION latents.
#
# ── WHY THIS IS NOT toy#156's ENGINE ──
#
# The P1b ticket describes this lane as composing toy#165's autoencoder
# with toy#156's denoiser. The autoencoder half is literally true — this
# lane instantiates AeEngine unchanged. The denoiser half is not:
# `diff_engine.rb` is a time-conditioned MLP over ONE latent vector,
#
#     input = [x_t (latent_dim) ; time features] -> silu MLP -> eps^
#
# with no positions and no attention, its batch axis being independent
# samples. A text latent is a SEQUENCE of per-position latents, and a
# denoiser that treats positions independently can only ever produce
# per-position noise that decodes to per-position garbage — every byte
# individually plausible, the sequence incoherent. Sequence coherence
# has to come from the denoiser MIXING ACROSS POSITIONS, so this is a
# transformer over the T latent columns, not an MLP over one of them.
#
# What DOES compose from toy#156 is the diffusion plumbing — the beta
# schedule, the eps parameterisation, the ancestral sampler and the
# abar_T guard — and that lives in the runner.
#
# ── THE MODEL ──
#
#   in     = [ x_t (d) ; x_selfcond (d) ]              [2d, T]
#   h_0    = W_in . in  +  W_time[t]  +  P             [d_model, T]
#   per block (pre-norm): a = h + MHA(RMS(h))   BIDIRECTIONAL
#                         h' = a + FFN_silu(RMS(a))
#   eps^   = W_out . RMS(h_L)                          [d, T]
#   loss   = mean( (eps^ - eps)^2 )
#
# Attention is bidirectional and unmasked: diffusion denoises every
# position at once and has no generation order, which is the whole
# structural difference from the AR arm.
#
# ── SELF-CONDITIONING ──
#
# The input carries a SECOND d-wide block: the model's own previous
# x_0 estimate (Analog Bits 2208.04202 / SED 2211.04236). It is the
# literature's lever for low-dimensional latents, which is exactly the
# regime this capstone lives in, so `diff-plain` (the ablation) must
# differ from `diff-selfcond` by DATA and not by graph — the self-cond
# block is uploaded as ZEROS when the arm is off. One realized graph,
# two arms, so the ablation cannot accidentally compare two models.
#
# Training follows the standard recipe: with probability 1/2 the
# self-cond block is zeros, otherwise it is a detached x_0 estimate from
# a first forward pass. Detached because the point is to condition on
# the estimate, not to backprop through a second denoiser call.
#
# CPU-only (tao#18). ALL BP — no DFA here. P1c attaches DFA to THIS
# module (its output dim is the latent d, squarely in F20's window)
# while the 256-way decode head stays BP, which is why the denoiser is
# a separate engine from the autoencoder rather than extra layers on it.
#
# Spinel hygiene: plain class, no-arg ctor, no Struct, typed-empty array
# seeds, while loops, no #{}.

require_relative "../../models/transformer"

module Toy; module LLM; module Engine
class DifflmEngine
  attr_accessor :sess,
                :dl_d_model, :dl_heads, :dl_d_head, :dl_d_ff, :dl_blocks,
                :dl_context, :dl_latent, :dl_tsteps,
                :ft_weights, :ft_m, :ft_v, :ft_din, :ft_dout, :ft_names,
                :t_xin, :t_temb, :t_eps, :t_hp, :t_pred, :t_loss,
                :dl_graph_nodes, :dl_graph_bytes

  def initialize
    @sess       = TinyNN.tnn_null_ptr
    @dl_d_model = 0
    @dl_heads   = 0
    @dl_d_head  = 0
    @dl_d_ff    = 0
    @dl_blocks  = 0
    @dl_context = 0
    @dl_latent  = 0
    @dl_tsteps  = 0
    @ft_weights = [TinyNN.tnn_null_ptr]; @ft_weights.pop
    @ft_m       = [TinyNN.tnn_null_ptr]; @ft_m.pop
    @ft_v       = [TinyNN.tnn_null_ptr]; @ft_v.pop
    @ft_din     = [0]; @ft_din.pop
    @ft_dout    = [0]; @ft_dout.pop
    @ft_names   = [""]; @ft_names.pop
    @t_xin  = TinyNN.tnn_null_ptr
    @t_temb = TinyNN.tnn_null_ptr
    @t_eps  = TinyNN.tnn_null_ptr
    @t_hp   = TinyNN.tnn_null_ptr
    @t_pred = TinyNN.tnn_null_ptr
    @t_loss = TinyNN.tnn_null_ptr
    @dl_graph_nodes = 0
    @dl_graph_bytes = 0
  end

  def realize_for_random_init(d_model, heads, d_ff, n_blocks, context,
                              latent, tsteps, seed, init_scale)
    @dl_d_model = d_model
    @dl_heads   = heads
    @dl_d_head  = d_model / heads
    @dl_d_ff    = d_ff
    @dl_blocks  = n_blocks
    @dl_context = context
    @dl_latent  = latent
    @dl_tsteps  = tsteps

    @sess = TinyNN.tnn_session_new(0)
    TinyNN.tnn_session_set_graph_capacity(@sess, n_blocks * 400 + 262144)

    # The input projection reads BOTH d-wide blocks: x_t and the
    # self-conditioning estimate. See the header on why the ablation is
    # a zero upload rather than a narrower matrix.
    add_w(d_model, 2 * latent, "in")
    add_w(context, d_model, "pos")
    bi = 0
    while bi < n_blocks
      p = "b" + bi.to_s + "."
      h = 0
      while h < heads
        add_w(@dl_d_head, d_model, p + "q" + h.to_s)
        add_w(@dl_d_head, d_model, p + "k" + h.to_s)
        add_w(@dl_d_head, d_model, p + "v" + h.to_s)
        h = h + 1
      end
      add_w(d_model, d_model, p + "o")
      add_w(d_ff,    d_model, p + "up")
      add_w(d_model, d_ff,    p + "down")
      add_w(1, d_model, p + "ln1")
      add_w(1, d_model, p + "ln2")
      bi = bi + 1
    end
    add_w(1, d_model, "ln_f")
    # THE OUTPUT IS THE LATENT'S OWN WIDTH — d, not the 256-way vocab.
    # That is the whole reason P1c can attach DFA here and not to the
    # decode head (F20's window; F22's finding that the LM negative was
    # the output dim).
    add_w(latent, d_model, "out")

    wi = 0
    while wi < @ft_weights.length
      TinyNN.tnn_set_param(@ft_weights[wi])
      wi = wi + 1
    end

    @t_xin  = TinyNN.tnn_input_2d_f32_persistent(@sess, context, 2 * latent)
    TinyNN.tnn_tensor_set_name(@t_xin, "x_t_and_selfcond")
    # The timestep embedding is ONE d_model vector broadcast over all T
    # positions — every position of a diffusion step shares its t, so a
    # per-position time input would be capacity to learn something that
    # does not vary. Uploaded per step from a fixed sinusoidal table.
    @t_temb = TinyNN.tnn_input_2d_f32_persistent(@sess, context, d_model)
    TinyNN.tnn_tensor_set_name(@t_temb, "time_embedding")

    TinyNN.tnn_finalize_weights(@sess)
    upload_random_init!(seed, init_scale)
    nil
  end

  def build_training_step
    TinyNN.tnn_reset_for_rebuild(@sess)

    t_h = TinyNN.tnn_add(@sess,
            TinyNN.tnn_add(@sess,
              TinyNN.tnn_matmul(@sess, @ft_weights[0], @t_xin),
              @t_temb),
            @ft_weights[1])

    scale = 1.0 / Math.sqrt(@dl_d_head.to_f)
    bj = 0
    while bj < @dl_blocks
      base = 2 + bj * (3 * @dl_heads + 5)
      t_n1 = TinyNN.tnn_rms_norm(@sess, t_h,
               @ft_weights[base + 3 * @dl_heads + 3], 1.0e-5)
      t_heads = [TinyNN.tnn_null_ptr]; t_heads.pop
      hh = 0
      while hh < @dl_heads
        t_q = TinyNN.tnn_matmul(@sess, @ft_weights[base + hh * 3],     t_n1)
        t_k = TinyNN.tnn_matmul(@sess, @ft_weights[base + hh * 3 + 1], t_n1)
        t_v = TinyNN.tnn_matmul(@sess, @ft_weights[base + hh * 3 + 2], t_n1)
        # NO MASK: the denoiser sees every position at once. That is the
        # structural difference from the AR arm, not an oversight.
        t_sc = TinyNN.tnn_softmax(@sess,
                 TinyNN.tnn_scale(@sess,
                   TinyNN.tnn_matmul(@sess, t_k, t_q), scale))
        t_heads.push(TinyNN.tnn_matmul(@sess, TinyNN.tnn_transpose(@sess, t_v), t_sc))
        hh = hh + 1
      end
      t_cat = t_heads[0]
      hc = 1
      while hc < @dl_heads
        t_cat = TinyNN.tnn_concat(@sess, t_cat, t_heads[hc], 0)
        hc = hc + 1
      end
      t_a = TinyNN.tnn_add(@sess, t_h,
              TinyNN.tnn_matmul(@sess, @ft_weights[base + 3 * @dl_heads], t_cat))
      t_n2 = TinyNN.tnn_rms_norm(@sess, t_a,
               @ft_weights[base + 3 * @dl_heads + 4], 1.0e-5)
      t_up = TinyNN.tnn_silu(@sess,
               TinyNN.tnn_matmul(@sess, @ft_weights[base + 3 * @dl_heads + 1], t_n2))
      t_h  = TinyNN.tnn_add(@sess, t_a,
               TinyNN.tnn_matmul(@sess, @ft_weights[base + 3 * @dl_heads + 2], t_up))
      bj = bj + 1
    end

    nw = @ft_weights.length
    t_f = TinyNN.tnn_rms_norm(@sess, t_h, @ft_weights[nw - 2], 1.0e-5)
    @t_pred = TinyNN.tnn_matmul(@sess, @ft_weights[nw - 1], t_f)   # [d, T]
    TinyNN.tnn_set_output(@t_pred)

    # eps-prediction MSE, mean over (d, T). Built from primitives rather
    # than a fused op so the reduction is explicit and matches toy#156's.
    @t_eps = TinyNN.tnn_input_2d_f32(@sess, @dl_context, @dl_latent)
    @t_hp  = TinyNN.tnn_input_1d_f32(@sess, 7)
    t_d    = TinyNN.tnn_sub(@sess, @t_pred, @t_eps)
    @t_loss = TinyNN.tnn_scale(@sess,
                TinyNN.tnn_sum_rows(@sess,
                  TinyNN.tnn_reshape_2d(@sess,
                    TinyNN.tnn_mul(@sess, t_d, t_d),
                    @dl_latent * @dl_context, 1)),
                1.0 / (@dl_latent * @dl_context).to_f)
    TinyNN.tnn_set_output(@t_loss)

    TinyNN.tnn_set_loss(@t_loss)
    TinyNN.tnn_build_forward_only(@sess, @t_loss)
    TinyNN.tnn_build_backward(@sess)

    wk = 0
    while wk < @ft_weights.length
      tg = TinyNN.tnn_tensor_grad(@sess, @ft_weights[wk])
      to = TinyNN.tnn_opt_step_adamw(@sess, @ft_weights[wk], tg,
                                      @ft_m[wk], @ft_v[wk], @t_hp)
      TinyNN.tnn_extend_backward_graph(@sess, to)
      wk = wk + 1
    end

    puts "difflm: blocks=" + @dl_blocks.to_s +
         " d_model=" + @dl_d_model.to_s +
         " heads=" + @dl_heads.to_s +
         " d_ff=" + @dl_d_ff.to_s +
         " context=" + @dl_context.to_s +
         " latent=" + @dl_latent.to_s +
         " tsteps=" + @dl_tsteps.to_s +
         " attn=bidirectional out_dim=" + @dl_latent.to_s +
         " params=" + param_count.to_s

    TinyNN.tnn_pin_all_graph_b_nodes(@sess)
    TinyNN.tnn_realize_backward(@sess)
    measure_graph!
    [@t_loss, @t_eps, @t_hp]
  end

  def measure_graph!
    @dl_graph_nodes = TinyNN.tnn_graph_n_nodes(@sess)
    total = 0
    i = 0
    while i < @dl_graph_nodes
      total = total + TinyNN.tnn_tensor_nbytes(TinyNN.tnn_graph_node(@sess, i))
      i = i + 1
    end
    @dl_graph_bytes = total
    nil
  end

  def param_count
    n = 0
    wi = 0
    while wi < @ft_weights.length
      n = n + @ft_din[wi] * @ft_dout[wi]
      wi = wi + 1
    end
    n
  end

  def add_w(dout, din, name)
    w = TinyNN.tnn_input_2d_f32_persistent(@sess, dout, din)
    m = TinyNN.tnn_input_2d_f32_persistent(@sess, dout, din)
    v = TinyNN.tnn_input_2d_f32_persistent(@sess, dout, din)
    TinyNN.tnn_tensor_set_name(w, name)
    TinyNN.tnn_tensor_set_name(m, name + ".m")
    TinyNN.tnn_tensor_set_name(v, name + ".v")
    @ft_weights.push(w)
    @ft_m.push(m)
    @ft_v.push(v)
    @ft_din.push(din)
    @ft_dout.push(dout)
    @ft_names.push(name)
    nil
  end

  def upload_random_init!(seed, init_scale)
    state = xorshift_seed_state(seed)
    wi = 0
    while wi < @ft_weights.length
      n  = @ft_din[wi] * @ft_dout[wi]
      nm = @ft_names[wi]
      if nm == "ln_f" || (nm.length > 3 &&
         (nm[nm.length - 3, 3] == "ln1" || nm[nm.length - 3, 3] == "ln2"))
        upload_const(@ft_weights[wi], n, 1.0)
      else
        std = init_scale / Math.sqrt(@ft_din[wi].to_f)
        upload_gaussian(@ft_weights[wi], n, std, state)
      end
      zero_tensor(@ft_m[wi])
      zero_tensor(@ft_v[wi])
      wi = wi + 1
    end
    nil
  end

  def xorshift_seed_state(seed)
    s = ((seed + 104729) * 2654435761) % 2147483647
    if s <= 0
      s = seed + 104729
    end
    w = 0
    while w < 8
      s = (s * 1103515245 + 12345) & 0x7FFFFFFF
      w = w + 1
    end
    [s]
  end

  def upload_gaussian(tensor, n, std, state)
    buf = Array.new(n, 0.0)
    i = 0
    while i < n
      s = state[0]
      s = (s * 1103515245 + 12345) & 0x7FFFFFFF
      state[0] = s
      u1 = (s.to_f + 1.0) / 2147483648.0
      s = state[0]
      s = (s * 1103515245 + 12345) & 0x7FFFFFFF
      state[0] = s
      u2 = (s.to_f + 1.0) / 2147483648.0
      if u1 < 1.0e-12; u1 = 1.0e-12; end
      z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
      buf[i] = z * std
      i = i + 1
    end
    TinyNN.tnn_upload_from_float_array(@sess, tensor, buf, n)
  end

  def upload_const(tensor, n, v)
    buf = Array.new(n, v)
    TinyNN.tnn_upload_from_float_array(@sess, tensor, buf, n)
  end

  def zero_tensor(tensor)
    n = TinyNN.tnn_tensor_nelements(tensor)
    buf = Array.new(n, 0.0)
    TinyNN.tnn_upload_from_float_array(@sess, tensor, buf, n)
  end
end
end; end; end
