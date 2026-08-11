# lib/toy/llm/engine/diff_engine.rb — the toy#156 (DFA-arch T2) latent
# diffusion denoiser: a time-conditioned eps-prediction MLP over a
# LOW-DIMENSIONAL latent, with a per-layer credit-assignment policy
# (chain | dfa | frozen).
#
# WHY THIS LANE, AND WHY IT IS SCOPED TO LATENTS. toy#156's hard
# constraint is that the eps-prediction target has the INPUT's
# dimensionality — small for latent / tabular / time-series diffusion,
# ~2e5 for pixels. So pixel diffusion would defeat DFA exactly the way
# vocab 50257 defeats our LM lanes, and the ticket scopes strictly to
# the low-dim regime. That makes this lane a DIRECT test of the
# output-dim lens toy#152 measured: the feedback matrices here are
# [latent_dim, d_hidden] with latent_dim ~ 16.
#
# ── THE MODEL ──
#
#   input  = [ x_t (latent_dim) ; time features (n_time) ]
#   h_l    = silu(W_l . h_{l-1})                          L hidden layers
#   eps^   = W_head . h_L                                 [latent_dim, B]
#   loss   = mean_over(B, latent_dim) ( eps^ - eps )^2
#
# ── THE DFA RULE IS toy#152's, NOT toy#158's, AND THAT IS DELIBERATE ──
#
# This lane uses the canonical Nokland direct-gradient rule
#
#     e      = 2 (eps^ - eps) / (B * latent_dim)       [latent_dim, B]
#     da_l   = (B_l . e) (*) f'(a_l)                   [d_out, B]
#     grad W = h_{l-1}^T . da_l^T
#
# — the same construction as the MLP anchor, with `e` the exact gradient
# of the MSE wrt the prediction rather than of a cross-entropy. The
# surrogate-root form toy#154/#158 use is unnecessary here because
# nothing below the policied stack needs to train through it (there are
# no embedding tables and no recurrence), and using the literature rule
# means a result on this lane is comparable to toy#152's straight
# across: only the output dimension and the loss differ.
#
# NOTE the (*) f'(a_l) factor, which the franken lane omits. Same reason
# as toy#152: if this lane failed with our per-matmul surrogate instead,
# we could not tell "DFA does not work on a denoiser" from "our
# surrogate is not DFA".
#
# ARMS: chain (BP), dfa (above), frozen (no optimizer step — the layer
# stays at init and only the head trains). The head is always chain: at
# the output layer DFA and BP coincide.
#
# Spinel hygiene (landmine #16): plain class, no-arg ctor, no Struct,
# typed-empty array seeds, while loops, no #{} interpolation.
#
# LANDMINES honoured here:
#   - the persistent input is ALLOCATED BEFORE finalize_weights
#     (toy#133);
#   - every extend_backward_graph comes AFTER tnn_build_backward
#     (toy#150);
#   - the chain grad-acc read back for align telemetry is PINNED with
#     set_output: for a pure-dfa weight it has NO consumer, so sched
#     would alias its slot and the align download would read zeros.

require_relative "../../models/transformer"
require_relative "../../train/dfa_b"

module Toy; module LLM; module Engine
class DiffEngine
  POLICY_CHAIN  = 0
  POLICY_DFA    = 1
  POLICY_FROZEN = 2

  attr_accessor :sess,
                :df_d_in, :df_latent, :df_hidden, :df_layers, :df_batch,
                :ft_weights, :ft_m, :ft_v, :ft_din, :ft_dout, :ft_names,
                :t_x, :t_eps, :t_hp, :t_pred, :t_loss,
                :df_align_grads, :df_align_accs, :df_align_lis,
                :df_align_wnames,
                :df_b_handles, :df_b_seeds, :df_b_douts,
                :df_b_dist, :df_b_scale, :df_b_sigma,
                :df_dfa_wired, :df_frozen_count

  def initialize
    @sess       = TinyNN.tnn_null_ptr
    @df_d_in    = 0
    @df_latent  = 0
    @df_hidden  = 0
    @df_layers  = 0
    @df_batch   = 0
    @ft_weights = [TinyNN.tnn_null_ptr]; @ft_weights.pop
    @ft_m       = [TinyNN.tnn_null_ptr]; @ft_m.pop
    @ft_v       = [TinyNN.tnn_null_ptr]; @ft_v.pop
    @ft_din     = [0]; @ft_din.pop
    @ft_dout    = [0]; @ft_dout.pop
    @ft_names   = [""]; @ft_names.pop
    @t_x    = TinyNN.tnn_null_ptr
    @t_eps  = TinyNN.tnn_null_ptr
    @t_hp   = TinyNN.tnn_null_ptr
    @t_pred = TinyNN.tnn_null_ptr
    @t_loss = TinyNN.tnn_null_ptr
    @df_align_grads  = [TinyNN.tnn_null_ptr]; @df_align_grads.pop
    @df_align_accs   = [TinyNN.tnn_null_ptr]; @df_align_accs.pop
    @df_align_lis    = [0]; @df_align_lis.pop
    @df_align_wnames = [""]; @df_align_wnames.pop
    @df_b_handles = [TinyNN.tnn_null_ptr]; @df_b_handles.pop
    @df_b_seeds   = [0]; @df_b_seeds.pop
    @df_b_douts   = [0]; @df_b_douts.pop
    @df_b_dist    = 0
    @df_b_scale   = 0
    @df_b_sigma   = 0.0
    @df_dfa_wired = 0
    @df_frozen_count = 0
  end

  def realize_for_random_init(d_in, latent_dim, d_hidden, n_layers,
                              batch, seed, init_scale)
    @df_d_in   = d_in
    @df_latent = latent_dim
    @df_hidden = d_hidden
    @df_layers = n_layers
    @df_batch  = batch

    @sess = TinyNN.tnn_session_new(0)
    cap = (n_layers + 1) * 2000 + 65536
    TinyNN.tnn_session_set_graph_capacity(@sess, cap)

    li = 0
    while li < n_layers
      din = li == 0 ? d_in : d_hidden
      add_weight(d_hidden, din, "w" + (li + 1).to_s)
      li = li + 1
    end
    add_weight(latent_dim, d_hidden, "head")

    wi = 0
    while wi < @ft_weights.length
      TinyNN.tnn_set_param(@ft_weights[wi])
      wi = wi + 1
    end

    # toy#133: the persistent input must be allocated BEFORE
    # finalize_weights or it silently reads zeros at compute time. It
    # holds [x_t ; time features] and is re-uploaded every step AND
    # every ancestral-sampling step.
    @t_x = TinyNN.tnn_input_2d_f32_persistent(@sess, batch, d_in)
    TinyNN.tnn_tensor_set_name(@t_x, "x")

    TinyNN.tnn_finalize_weights(@sess)
    upload_random_init!(seed, init_scale)
    nil
  end

  def build_training_step(policy, b_seed, b_dist, b_scale, b_sigma)
    @df_b_dist  = b_dist
    @df_b_scale = b_scale
    @df_b_sigma = b_sigma

    TinyNN.tnn_reset_for_rebuild(@sess)

    taps = [TinyNN.tnn_null_ptr]; taps.pop
    pres = [TinyNN.tnn_null_ptr]; pres.pop
    t_h = @t_x
    li = 0
    while li < @df_layers
      taps.push(t_h)
      TinyNN.tnn_set_output(t_h)
      t_pre = TinyNN.tnn_matmul(@sess, @ft_weights[li], t_h)
      TinyNN.tnn_set_output(t_pre)
      pres.push(t_pre)
      t_h = TinyNN.tnn_silu(@sess, t_pre)
      TinyNN.tnn_set_output(t_h)
      li = li + 1
    end
    taps.push(t_h)

    @t_pred = TinyNN.tnn_matmul(@sess, @ft_weights[@df_layers], t_h)
    TinyNN.tnn_set_output(@t_pred)

    @t_eps = TinyNN.tnn_input_2d_f32(@sess, @df_batch, @df_latent)
    @t_hp  = TinyNN.tnn_input_1d_f32(@sess, 7)

    # MSE, built by hand: there is no mean-squared-error op in the shim,
    # and sum_rows over a flattened square is the form toy#158 already
    # proved differentiable in a loss root.
    n_out = @df_latent * @df_batch
    t_diff = TinyNN.tnn_sub(@sess, @t_pred, @t_eps)
    TinyNN.tnn_set_output(t_diff)
    @t_loss = TinyNN.tnn_scale(@sess,
                TinyNN.tnn_sum_rows(@sess,
                  TinyNN.tnn_reshape_2d(@sess,
                    TinyNN.tnn_mul(@sess, t_diff, t_diff), n_out, 1)),
                1.0 / n_out.to_f)
    TinyNN.tnn_set_output(@t_loss)
    TinyNN.tnn_set_loss(@t_loss)
    TinyNN.tnn_build_forward_only(@sess, @t_loss)
    # toy#150: every extend_backward_graph below MUST come after this.
    TinyNN.tnn_build_backward(@sess)

    any_dfa = false
    pi = 0
    while pi < policy.length
      if policy[pi] == POLICY_DFA
        any_dfa = true
      end
      pi = pi + 1
    end

    t_e = TinyNN.tnn_null_ptr
    if any_dfa
      # The EXACT gradient of the MSE wrt the prediction: 2 (pred - eps)
      # / (B * latent_dim). The DFA arm and the BP arm therefore differ
      # ONLY in how that signal reaches the hidden layers, never in what
      # the signal is — the same property the MLP anchor has.
      t_e = TinyNN.tnn_scale(@sess, t_diff, 2.0 / n_out.to_f)
      TinyNN.tnn_set_output(t_e)
    end

    @df_align_grads  = [TinyNN.tnn_null_ptr]; @df_align_grads.pop
    @df_align_accs   = [TinyNN.tnn_null_ptr]; @df_align_accs.pop
    @df_align_lis    = [0]; @df_align_lis.pop
    @df_align_wnames = [""]; @df_align_wnames.pop
    @df_b_handles = [TinyNN.tnn_null_ptr]; @df_b_handles.pop
    @df_b_seeds   = [0]; @df_b_seeds.pop
    @df_b_douts   = [0]; @df_b_douts.pop
    @df_frozen_count = 0

    lj = 0
    while lj < @df_layers
      mode = POLICY_CHAIN
      if lj < policy.length
        mode = policy[lj]
      end
      tw   = @ft_weights[lj]
      din  = @ft_din[lj]
      dout = @ft_dout[lj]
      if mode == POLICY_DFA
        t_b = TinyNN.tnn_input_2d_f32(@sess, dout, @df_latent)
        TinyNN.tnn_set_output(t_b)
        t_delta = TinyNN.tnn_matmul(@sess, t_b, t_e)              # [dout, B]
        t_dpre  = TinyNN.tnn_silu_back(@sess, pres[lj], t_delta)
        t_tap_t = TinyNN.tnn_cont_2d(@sess,
                    TinyNN.tnn_transpose(@sess, taps[lj]), @df_batch, din)
        t_del_t = TinyNN.tnn_cont_2d(@sess,
                    TinyNN.tnn_transpose(@sess, t_dpre), @df_batch, dout)
        t_g = TinyNN.tnn_matmul(@sess, t_tap_t, t_del_t)          # ne=[din,dout]
        TinyNN.tnn_set_output(t_g)
        to = TinyNN.tnn_opt_step_adamw(@sess, tw, t_g,
                                        @ft_m[lj], @ft_v[lj], @t_hp)
        TinyNN.tnn_extend_backward_graph(@sess, to)
        @df_b_handles.push(t_b)
        @df_b_seeds.push(b_seed + lj * 1000)
        @df_b_douts.push(dout)
        # PIN the shadow acc — a pure-dfa weight's chain grad has no
        # consumer, so without this the align download reads zeros.
        t_acc = TinyNN.tnn_tensor_grad(@sess, tw)
        TinyNN.tnn_set_output(t_acc)
        @df_align_grads.push(t_g)
        @df_align_accs.push(t_acc)
        @df_align_lis.push(lj)
        @df_align_wnames.push(@ft_names[lj])
      elsif mode == POLICY_FROZEN
        @df_frozen_count = @df_frozen_count + 1
      else
        tg = TinyNN.tnn_tensor_grad(@sess, tw)
        to = TinyNN.tnn_opt_step_adamw(@sess, tw, tg,
                                        @ft_m[lj], @ft_v[lj], @t_hp)
        TinyNN.tnn_extend_backward_graph(@sess, to)
      end
      lj = lj + 1
    end

    hw = @df_layers
    thg = TinyNN.tnn_tensor_grad(@sess, @ft_weights[hw])
    tho = TinyNN.tnn_opt_step_adamw(@sess, @ft_weights[hw], thg,
                                     @ft_m[hw], @ft_v[hw], @t_hp)
    TinyNN.tnn_extend_backward_graph(@sess, tho)

    @df_dfa_wired = @df_b_handles.length
    puts "diff: latent=" + @df_latent.to_s +
         " layers=" + @df_layers.to_s +
         " policy_len=" + policy.length.to_s +
         " dfa_wired=" + @df_dfa_wired.to_s +
         " frozen=" + @df_frozen_count.to_s

    TinyNN.tnn_pin_all_graph_b_nodes(@sess)
    TinyNN.tnn_realize_backward(@sess)

    refresh_b!
    [@t_loss, @t_eps, @t_hp]
  end

  def refresh_b!
    bi = 0
    while bi < @df_b_handles.length
      dout = @df_b_douts[bi]
      nb   = dout * @df_latent
      sig  = Toy::Train::DfaB.sigma_for(@df_b_scale, @df_latent,
                                        dout, @df_b_sigma)
      TinyNN.tnn_upload_from_float_array(@sess, @df_b_handles[bi],
        Toy::Train::DfaB.fill(nb, @df_b_seeds[bi], @df_b_dist, sig), nb)
      bi = bi + 1
    end
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

  # --- bookkeeping helpers (same shape as mlp_engine.rb) ---

  def add_weight(dout, din, name)
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
      n   = @ft_din[wi] * @ft_dout[wi]
      std = init_scale / Math.sqrt(@ft_din[wi].to_f)
      upload_gaussian(@ft_weights[wi], n, std, state)
      zero_tensor(@ft_m[wi])
      zero_tensor(@ft_v[wi])
      wi = wi + 1
    end
    nil
  end

  # toy#114 — mixer-seeded stream state.
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

  def zero_tensor(tensor)
    n = TinyNN.tnn_tensor_nelements(tensor)
    buf = Array.new(n, 0.0)
    TinyNN.tnn_upload_from_float_array(@sess, tensor, buf, n)
  end
end
end; end; end
