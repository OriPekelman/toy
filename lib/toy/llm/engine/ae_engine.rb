# lib/toy/llm/engine/ae_engine.rb — the toy#165 (capstone P1a) PER-TOKEN
# LATENT AUTOENCODER: a bidirectional transformer encoder, a d-dim
# per-position bottleneck, and a PER-POSITION decode head back to the
# byte.
#
# ── WHAT THIS MEASURES ──
#
# The capstone (a diffusion text LM) needs a per-token continuous latent
# of 4-8 dims, because F20/toy#156 put DFA's advantage in exactly that
# output-dim window. Whether text SURVIVES such a latent is unrun in the
# literature. P1a is the cheapest decisive form of the question and it is
# ALL BP — no DFA anywhere in this file, no diffusion. DFA arrives in
# P1c, on the denoiser, whose output dim is the latent itself.
#
#   emb    = E[tokens] + P                          [d_model, T]
#   per block (pre-norm):  a = h + MHA(RMS(h));  h' = a + FFN_silu(RMS(a))
#   lat    = W_lat . RMS(h_L)                       [d, T]   <- THE LATENT
#   lat'   = (lat[:, perm] (*) gain) + noise        [d, T]   <- the probe
#   logits = W_head . lat'                          [256, T]
#
# ── THE DECODER IS PER-POSITION, AND THAT IS THE WHOLE POINT ──
#
# `W_head` is ONE matrix applied to each position's latent independently.
# No attention, no convolution, no neighbour of any kind after the
# bottleneck. If the decoder could see context it would simply predict
# token i from its neighbours and the latent would be bypassed — the lane
# would be measuring language modelling and reporting it as capacity.
#
# The encoder is bidirectional (no causal mask): reconstruction is not
# prediction, and a diffusion sampler denoises all positions at once, so
# a causal encoder would model a generation order this capstone does not
# have.
#
# ── THE PROBE IS DATA, NOT A SECOND GRAPH ──
#
# Three persistent inputs sit between the latent and the head:
#
#   perm  [T] i32   gather over POSITIONS   — identity, or a shuffle
#   gain  [d, T]    elementwise multiplier  — ones, or zeros
#   noise [d, T]    elementwise addend      — zeros, or sigma * std * z
#
# Training uploads identity / ones / zeros, so the training graph is the
# measurement graph. That is deliberate and it is the mirror rule
# (toy#139/#146) taken one step further: instead of remembering to zero a
# second hp vector on the eval path, there IS no second path — clean
# reconstruction, every noise level, and both controls are the SAME
# realized graph under different uploads. A control that needed its own
# graph could differ from the trained model in ways no assertion covers.
#
# The gain multiply costs one op per step in training, where it multiplies
# by one. That is the price of the property above and it is worth it.
#
# ── WHY THE HEAD IS 256-WIDE ON EVERY CORPUS ──
#
# The alphabet is an axis (tao#22): the same lane runs over corpora with
# 27, 65 and 201 distinct byte values. Sizing the head to the observed
# alphabet would shrink the output matrix as N shrinks and confound the
# alphabet axis with head capacity. Fixed at 256, only the DATA changes
# along the axis. Unused classes never receive mass; the reported
# alphabet size is what says so.
#
# CPU-only (tao#18). Spinel hygiene (landmine #16): plain class, no-arg
# ctor, no Struct, typed-empty array seeds, while loops, no #{}.

# This lane builds its own encoder and uses nothing FROM transformer.rb,
# but it still has to be required — and `require_relative "../../ffi/tinynn"`
# in its place is NOT enough. Measured, not assumed:
#
#   * with neither, `TinyNN.tnn_events_active == 1` fails to compile as
#     `unsupported equality ... recv=CallNode/ty0` — an unrequired TinyNN
#     is not a missing-constant error under Spinel, every call just loses
#     its type and the failure surfaces somewhere unrelated;
#   * with the FFI alone it gets further and then dies inside tinynn.rb
#     itself, on `mat.nrows * mat.ncols` in `upload_row_major` — the
#     helper's `mat` parameter has no call site that pins it to Mat
#     (landmine #5), and transformer.rb is what pins it for every other
#     lane.
#
# So: require the transformer, and do not "tidy" this into the narrower
# require. The cost is a compiled-in file this lane never calls.
require_relative "../../models/transformer"

module Toy; module LLM; module Engine
class AeEngine
  VOCAB = 256

  attr_accessor :sess,
                :ae_d_model, :ae_heads, :ae_d_head, :ae_d_ff, :ae_blocks,
                :ae_context, :ae_latent, :ae_kl_beta, :ae_kl_learned,
                :ix_ln_f, :ix_lat, :ix_head, :ix_bias, :ix_logvar,
                :ft_weights, :ft_m, :ft_v, :ft_din, :ft_dout, :ft_names,
                :t_tokens, :t_perm, :t_gain, :t_noise,
                :t_latent, :t_labels, :t_hp, :t_logits, :t_loss,
                :ae_graph_nodes, :ae_graph_bytes

  def initialize
    @sess       = TinyNN.tnn_null_ptr
    @ae_d_model = 0
    @ae_heads   = 0
    @ae_d_head  = 0
    @ae_d_ff    = 0
    @ae_blocks  = 0
    @ae_context = 0
    @ae_latent  = 0
    @ae_kl_beta = 0.0
    @ae_kl_learned = 0
    @ix_ln_f    = 0
    @ix_lat     = 0
    @ix_head    = 0
    @ix_bias    = 0
    @ix_logvar  = -1
    @ft_weights = [TinyNN.tnn_null_ptr]; @ft_weights.pop
    @ft_m       = [TinyNN.tnn_null_ptr]; @ft_m.pop
    @ft_v       = [TinyNN.tnn_null_ptr]; @ft_v.pop
    @ft_din     = [0]; @ft_din.pop
    @ft_dout    = [0]; @ft_dout.pop
    @ft_names   = [""]; @ft_names.pop
    @t_tokens = TinyNN.tnn_null_ptr
    @t_perm   = TinyNN.tnn_null_ptr
    @t_gain   = TinyNN.tnn_null_ptr
    @t_noise  = TinyNN.tnn_null_ptr
    @t_latent = TinyNN.tnn_null_ptr
    @t_labels = TinyNN.tnn_null_ptr
    @t_hp     = TinyNN.tnn_null_ptr
    @t_logits = TinyNN.tnn_null_ptr
    @t_loss   = TinyNN.tnn_null_ptr
    @ae_graph_nodes = 0
    @ae_graph_bytes = 0
  end

  # toy#168 followup — `kl_beta` adds a fixed-sigma KL penalty on the
  # latent. 0.0 is the default and is BYTE-NULL: the extra nodes are not
  # built at all, so the toy#165 ae lane is bit-identical to before.
  #
  # WHY FIXED SIGMA. With q(z|x) = N(mu, s^2 I) and p = N(0, I),
  #   KL = 0.5 * sum( mu^2 + s^2 - 1 - 2 ln s )
  # and at CONSTANT s every term but 0.5*sum(mu^2) is a constant. So a
  # fixed-sigma VAE is exactly "L2 on the latent mean + Gaussian noise
  # injection at scale s" — which needs no second projection head and no
  # exp/log ops, and is the "small KL/smoothness term" P1b's spec asked
  # for. The noise half is uploaded by the runner through the EXISTING
  # probe tensor (t_noise), so the reparameterisation costs no new graph.
  #
  # A LEARNED per-position sigma is the next step up if the aggregate
  # still misses the prior; it is deliberately not the first cut.
  def realize_for_random_init(d_model, heads, d_ff, n_blocks, context,
                              latent, seed, init_scale, kl_beta, kl_learned)
    @ae_d_model = d_model
    @ae_heads   = heads
    @ae_d_head  = d_model / heads
    @ae_d_ff    = d_ff
    @ae_blocks  = n_blocks
    @ae_context = context
    @ae_latent  = latent
    @ae_kl_beta = kl_beta
    @ae_kl_learned = kl_learned

    @sess = TinyNN.tnn_session_new(0)
    cap = n_blocks * 400 + 262144
    TinyNN.tnn_session_set_graph_capacity(@sess, cap)

    # ne0 is the CONTRACTED dim for matmul, so a weight allocated as
    # add_w(dout, din) has ne=[din, dout] and matmul(W, x) with x
    # ne=[din, T] gives ne=[dout, T]. `emb` and `pos` are never
    # contracted — they are gathered and added — so their ne is read the
    # other way: [d_model, VOCAB] and [d_model, T].
    add_w(VOCAB,  d_model, "emb")
    add_w(context, d_model, "pos")
    bi = 0
    while bi < n_blocks
      p = "b" + bi.to_s + "."
      h = 0
      while h < heads
        add_w(@ae_d_head, d_model, p + "q" + h.to_s)
        add_w(@ae_d_head, d_model, p + "k" + h.to_s)
        add_w(@ae_d_head, d_model, p + "v" + h.to_s)
        h = h + 1
      end
      add_w(d_model, d_model, p + "o")
      add_w(d_ff,    d_model, p + "up")
      add_w(d_model, d_ff,    p + "down")
      add_w(1, d_model, p + "ln1")
      add_w(1, d_model, p + "ln2")
      bi = bi + 1
    end
    # NAMED INDICES. These used to be read as `nw - 2`, `nw - 3` ... which
    # is fine until a weight is added and every offset silently shifts by
    # one — the head would still be a matrix of the right shape, so
    # nothing would fail loudly. The learned-sigma arm adds exactly such a
    # weight, so the arithmetic goes away.
    @ix_ln_f = @ft_weights.length
    add_w(1, d_model, "ln_f")
    @ix_lat = @ft_weights.length
    add_w(latent, d_model, "lat")
    # THE DECODE HEAD: latent -> byte, one matrix, applied per position.
    @ix_head = @ft_weights.length
    add_w(VOCAB, latent, "head")
    # ...plus a BIAS, and it is load-bearing rather than conventional.
    #
    # The controls are stated against the UNIGRAM FLOOR — "a decode that
    # cannot see the latent should do no better than always guessing the
    # commonest byte". Without a bias the head CANNOT express that prior:
    # a zeroed latent yields an all-zero logit vector, argmax breaks the
    # tie at class 0, and the control scores ~0.000 instead of the floor.
    # Measured on the first smoke run, and it makes the comparison
    # meaningless in the flattering direction — every control would look
    # like it lost badly no matter how much the head had memorised.
    #
    # ne=[VOCAB, 1], repeated over the T positions by the add. One bias
    # for all positions, deliberately: a per-position bias would be free
    # capacity to learn a positional prior, which is the very thing the
    # shuffled-latent control exists to detect.
    @ix_bias = @ft_weights.length
    add_w(1, VOCAB, "head_bias")
    # THE LEARNED POSTERIOR SCALE. A second projection from the same
    # encoder output, giving log(sigma^2) per position and dim. Built
    # ONLY for the learned arm, so every other configuration keeps its
    # exact previous graph.
    #
    # Why it matters, measured: with sigma FIXED the KL collapses to
    # 0.5*sum(mu^2), i.e. plain L2 on the mean, which SPARSIFIES a
    # discrete code rather than Gaussianising it (kurtosis went to 4.29
    # at the beta that fixed the scale). With sigma learned, an
    # uninformative dimension can sit at (mu=0, sigma=1) — which IS the
    # prior — instead of being squashed toward zero and then re-inflated
    # by standardisation.
    if @ae_kl_learned == 1
      @ix_logvar = @ft_weights.length
      add_w(latent, d_model, "logvar")
    end

    wi = 0
    while wi < @ft_weights.length
      TinyNN.tnn_set_param(@ft_weights[wi])
      wi = wi + 1
    end

    # Persistent inputs, ALLOCATED BEFORE finalize_weights (toy#133): an
    # input allocated after it reads zeros in SILENCE.
    @t_tokens = TinyNN.tnn_input_1d_i32_persistent(@sess, context)
    @t_perm   = TinyNN.tnn_input_1d_i32_persistent(@sess, context)
    @t_gain   = TinyNN.tnn_input_2d_f32_persistent(@sess, context, latent)
    TinyNN.tnn_tensor_set_name(@t_gain, "latent_gain")
    @t_noise  = TinyNN.tnn_input_2d_f32_persistent(@sess, context, latent)
    TinyNN.tnn_tensor_set_name(@t_noise, "latent_noise")

    TinyNN.tnn_finalize_weights(@sess)
    upload_random_init!(seed, init_scale)
    nil
  end

  def build_training_step
    TinyNN.tnn_reset_for_rebuild(@sess)

    t_emb = TinyNN.tnn_get_rows(@sess, @ft_weights[0], @t_tokens)  # [d_model, T]
    t_h   = TinyNN.tnn_add(@sess, t_emb, @ft_weights[1])

    scale = 1.0 / Math.sqrt(@ae_d_head.to_f)
    bj = 0
    while bj < @ae_blocks
      base = 2 + bj * (3 * @ae_heads + 5)
      t_n1 = TinyNN.tnn_rms_norm(@sess, t_h,
               @ft_weights[base + 3 * @ae_heads + 3], 1.0e-5)

      t_heads = [TinyNN.tnn_null_ptr]; t_heads.pop
      hh = 0
      while hh < @ae_heads
        t_q = TinyNN.tnn_matmul(@sess, @ft_weights[base + hh * 3],     t_n1)
        t_k = TinyNN.tnn_matmul(@sess, @ft_weights[base + hh * 3 + 1], t_n1)
        t_v = TinyNN.tnn_matmul(@sess, @ft_weights[base + hh * 3 + 2], t_n1)
        # No mask: the encoder is BIDIRECTIONAL. Reconstruction is not
        # prediction, and the diffusion sampler this feeds denoises every
        # position at once, so a causal mask would model a generation
        # order the capstone does not have.
        t_sc = TinyNN.tnn_softmax(@sess,
                 TinyNN.tnn_scale(@sess,
                   TinyNN.tnn_matmul(@sess, t_k, t_q), scale))
        t_heads.push(TinyNN.tnn_matmul(@sess, TinyNN.tnn_transpose(@sess, t_v), t_sc))
        hh = hh + 1
      end
      t_cat = t_heads[0]
      hc = 1
      while hc < @ae_heads
        t_cat = TinyNN.tnn_concat(@sess, t_cat, t_heads[hc], 0)
        hc = hc + 1
      end
      t_a = TinyNN.tnn_add(@sess, t_h,
              TinyNN.tnn_matmul(@sess, @ft_weights[base + 3 * @ae_heads], t_cat))

      t_n2 = TinyNN.tnn_rms_norm(@sess, t_a,
               @ft_weights[base + 3 * @ae_heads + 4], 1.0e-5)
      t_up = TinyNN.tnn_silu(@sess,
               TinyNN.tnn_matmul(@sess, @ft_weights[base + 3 * @ae_heads + 1], t_n2))
      t_h  = TinyNN.tnn_add(@sess, t_a,
               TinyNN.tnn_matmul(@sess, @ft_weights[base + 3 * @ae_heads + 2], t_up))
      bj = bj + 1
    end

    t_f = TinyNN.tnn_rms_norm(@sess, t_h, @ft_weights[@ix_ln_f], 1.0e-5)

    # THE BOTTLENECK. Set as an output so the runner can download it and
    # calibrate the noise against the latent's own per-dim spread — the
    # noise-margin curve has to be scale-invariant across d, or a wider
    # latent would look more robust merely for having larger activations.
    @t_latent = TinyNN.tnn_matmul(@sess, @ft_weights[@ix_lat], t_f)  # [d, T]
    TinyNN.tnn_set_output(@t_latent)

    # THE PROBE. Identity/ones/zeros during training, so this IS the
    # training graph; see the header.
    # z = mu + sigma * eps. With a learned sigma the uploaded noise is
    # UNIT gaussian and is scaled in-graph; otherwise it is the addend
    # itself and this is exactly the pre-existing probe.
    t_noise_term = @t_noise
    t_logvar = TinyNN.tnn_null_ptr
    if @ae_kl_learned == 1
      t_logvar = TinyNN.tnn_matmul(@sess, @ft_weights[@ix_logvar], t_f)
      t_noise_term = TinyNN.tnn_mul(@sess,
                       TinyNN.tnn_exp(@sess,
                         TinyNN.tnn_scale(@sess, t_logvar, 0.5)),
                       @t_noise)
    end
    t_probe = TinyNN.tnn_add(@sess,
                TinyNN.tnn_mul(@sess,
                  TinyNN.tnn_get_rows(@sess, @t_latent, @t_perm),
                  @t_gain),
                t_noise_term)

    @t_logits = TinyNN.tnn_add(@sess,
                  TinyNN.tnn_matmul(@sess, @ft_weights[@ix_head], t_probe),
                  @ft_weights[@ix_bias])
    TinyNN.tnn_set_output(@t_logits)

    @t_labels = TinyNN.tnn_input_2d_f32(@sess, @ae_context, VOCAB)
    @t_hp     = TinyNN.tnn_input_1d_f32(@sess, 7)
    @t_loss   = TinyNN.tnn_cross_entropy_loss(@sess, @t_logits, @t_labels)
    TinyNN.tnn_set_output(@t_loss)

    # The KL penalty, built ONLY when asked — at beta 0 the graph is
    # unchanged and every pre-toy#168 number reproduces exactly.
    t_obj = @t_loss
    if @ae_kl_beta > 0.0
      # fixed sigma:   KL ~ mu^2                      (+ const)
      # learned sigma: KL ~ mu^2 + exp(logvar) - logvar  (+ const)
      # The "-1" per element is dropped in both: it is a constant, so it
      # shifts the reported objective and changes no gradient.
      t_el = TinyNN.tnn_mul(@sess, @t_latent, @t_latent)
      if @ae_kl_learned == 1
        t_el = TinyNN.tnn_sub(@sess,
                 TinyNN.tnn_add(@sess, t_el, TinyNN.tnn_exp(@sess, t_logvar)),
                 t_logvar)
      end
      t_sq = TinyNN.tnn_sum_rows(@sess,
               TinyNN.tnn_reshape_2d(@sess, t_el,
                 @ae_latent * @ae_context, 1))
      t_obj = TinyNN.tnn_add(@sess, @t_loss,
                TinyNN.tnn_scale(@sess, t_sq,
                  0.5 * @ae_kl_beta / (@ae_latent * @ae_context).to_f))
      TinyNN.tnn_set_output(t_obj)
    end

    TinyNN.tnn_set_loss(t_obj)
    TinyNN.tnn_build_forward_only(@sess, t_obj)
    TinyNN.tnn_build_backward(@sess)

    wk = 0
    while wk < @ft_weights.length
      tg = TinyNN.tnn_tensor_grad(@sess, @ft_weights[wk])
      to = TinyNN.tnn_opt_step_adamw(@sess, @ft_weights[wk], tg,
                                      @ft_m[wk], @ft_v[wk], @t_hp)
      TinyNN.tnn_extend_backward_graph(@sess, to)
      wk = wk + 1
    end

    puts "ae: blocks=" + @ae_blocks.to_s +
         " d_model=" + @ae_d_model.to_s +
         " heads=" + @ae_heads.to_s +
         " d_ff=" + @ae_d_ff.to_s +
         " context=" + @ae_context.to_s +
         " latent=" + @ae_latent.to_s +
         " vocab=" + VOCAB.to_s +
         " decoder=per_position" +
         " encoder=bidirectional" +
         " kl_beta=" + @ae_kl_beta.to_s +
         " kl_sigma=" + (@ae_kl_learned == 1 ? "learned" : "fixed")

    TinyNN.tnn_pin_all_graph_b_nodes(@sess)
    TinyNN.tnn_realize_backward(@sess)
    measure_graph!
    [@t_loss, @t_labels, @t_hp]
  end

  def measure_graph!
    @ae_graph_nodes = TinyNN.tnn_graph_n_nodes(@sess)
    total = 0
    i = 0
    while i < @ae_graph_nodes
      total = total + TinyNN.tnn_tensor_nbytes(TinyNN.tnn_graph_node(@sess, i))
      i = i + 1
    end
    @ae_graph_bytes = total
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

  # --- bookkeeping helpers ---

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
      if nm == "head_bias" || nm == "logvar"
        # head_bias: starts at ZERO so step 1 is exactly the no-bias
        # model; the unigram prior is LEARNED, not handed over.
        # logvar: starts at ZERO so sigma = exp(0) = 1 — the posterior
        # begins AT the prior and has to be moved away from it, rather
        # than starting somewhere arbitrary and being dragged back.
        upload_const(@ft_weights[wi], n, 0.0)
      elsif nm == "ln_f" || (nm.length > 3 &&
         (nm[nm.length - 3, 3] == "ln1" || nm[nm.length - 3, 3] == "ln2"))
        # RMSNorm gain starts at 1.0, or the first block scales its own
        # input to nothing and the residual stream never gets going.
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
