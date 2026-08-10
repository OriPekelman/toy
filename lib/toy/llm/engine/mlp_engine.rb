# lib/toy/llm/engine/mlp_engine.rb — the toy#152 (DFA-arch T0) MLP
# classifier engine: an N-hidden-layer bias-free MLP + small-#classes
# head + cross-entropy, with a PER-LAYER credit-assignment policy
# (chain | dfa | frozen).
#
# WHY THIS LANE EXISTS. The whole F4–F14 body of work is NEGATIVE for
# DFA on transformer LMs (vocab 50257 = a huge output dim). Our lens
# (F13/F13b) and the theory (Refinetti et al. 2021) say DFA works when
# the OUTPUT DIMENSION is small. Until this lane reproduces a KNOWN DFA
# positive, those negatives are not findings — they could be harness
# artifacts. This engine is the control for everything already shipped.
#
# THE DFA RULE IS THE CANONICAL (Nøkland 2016) ONE, and it is the whole
# reason this lane is cheap:
#
#     e      = (softmax(logits) - onehot) / B         [n_classes, B]
#     δa_l   = (B_l · e) ⊙ f'(a_l)                    [d_out, B]
#     ∇W_l   = h_{l-1}ᵀ · δa_lᵀ                       [d_in, d_out]
#
# with B_l shaped [n_classes, d_out] — the SMALL output dim is the
# point, and Toy::Train::DfaB (fill / sigma_for) is the SAME feedback
# machinery the franken lanes use, so no new B code exists here.
#
# NOTE the ⊙ f'(a_l) factor. The franken (transformer) lane omits it —
# there the per-matmul DFA surrogate projects straight onto the weight
# — but for the anchor we want the LITERATURE rule, not our surrogate:
# if this lane failed we could not tell "DFA does not work at small
# output dim" from "our surrogate is not DFA". f' comes from
# ggml_silu_back(x, dy), i.e. the exact derivative of the forward
# activation, not an approximation.
#
# ARMS (per hidden layer, policy codes):
#   0 = chain  — plain backprop (the BP arm)
#   1 = dfa    — the rule above (the DFA arm)
#   2 = frozen — NO optimizer step at all: the layer stays at init and
#                only the head trains. This is the FROZEN CONTROL the
#                success bar needs (docs/roadmap/dfa-arch-program-
#                2026-08-10.md): "near-BP" alone cannot distinguish
#                "DFA taught the hidden layers something" from "the
#                head alone could do this task" — the trap that
#                inflated our own early MoE numbers, and the shape of
#                toy#141 where the frozen arm BEAT both dfa and chain.
#
# The HEAD is always chain: at the output layer DFA and BP coincide
# (both get e), so policying it would be a no-op with extra machinery.
# Under an ALL-CHAIN policy (or an empty one) the emitted graph is
# IDENTICAL to a plain BP build — prep/mlp_gate.rb pins that byte-for-
# byte, the same "default that changes nothing" discipline as toy#151.
#
# Spinel hygiene (landmine #16): plain class, no-arg ctor, no Struct,
# typed-empty array seeds, while loops, no #{} interpolation.
#
# LANDMINES honoured here:
#   - persistent inputs (t_x) are ALLOCATED BEFORE finalize_weights
#     (toy#133) — a compute-context input allocated later reads zeros
#     in silence;
#   - every extend_backward_graph comes AFTER tnn_build_backward
#     (toy#150) — earlier is silently discarded;
#   - the chain grad-acc read back for align telemetry is PINNED with
#     set_output (the P0 pin-read-backs lesson): for a pure-dfa weight
#     it has NO consumer, so sched would alias its slot and the align
#     download would read zeros.

require_relative "../../models/transformer"
require_relative "../../train/dfa_b"

module Toy; module LLM; module Engine
class MlpEngine
  POLICY_CHAIN  = 0
  POLICY_DFA    = 1
  POLICY_FROZEN = 2

  attr_accessor :sess,
                :mlp_d_in, :mlp_d_hidden, :mlp_layers, :mlp_classes,
                :mlp_batch,
                :ft_weights, :ft_m, :ft_v, :ft_din, :ft_dout, :ft_names,
                :t_x, :t_labels, :t_hp, :t_logits, :t_loss,
                :mlp_align_grads, :mlp_align_accs, :mlp_align_lis,
                :mlp_align_wnames,
                :mlp_b_handles, :mlp_b_seeds, :mlp_b_douts,
                :mlp_b_dist, :mlp_b_scale, :mlp_b_sigma,
                :mlp_dfa_wired, :mlp_frozen_count

  def initialize
    @sess         = TinyNN.tnn_null_ptr
    @mlp_d_in     = 0
    @mlp_d_hidden = 0
    @mlp_layers   = 0
    @mlp_classes  = 0
    @mlp_batch    = 0
    @ft_weights   = [TinyNN.tnn_null_ptr]; @ft_weights.pop
    @ft_m         = [TinyNN.tnn_null_ptr]; @ft_m.pop
    @ft_v         = [TinyNN.tnn_null_ptr]; @ft_v.pop
    @ft_din       = [0]; @ft_din.pop
    @ft_dout      = [0]; @ft_dout.pop
    @ft_names     = [""]; @ft_names.pop
    @t_x      = TinyNN.tnn_null_ptr
    @t_labels = TinyNN.tnn_null_ptr
    @t_hp     = TinyNN.tnn_null_ptr
    @t_logits = TinyNN.tnn_null_ptr
    @t_loss   = TinyNN.tnn_null_ptr
    @mlp_align_grads  = [TinyNN.tnn_null_ptr]; @mlp_align_grads.pop
    @mlp_align_accs   = [TinyNN.tnn_null_ptr]; @mlp_align_accs.pop
    @mlp_align_lis    = [0]; @mlp_align_lis.pop
    @mlp_align_wnames = [""]; @mlp_align_wnames.pop
    @mlp_b_handles = [TinyNN.tnn_null_ptr]; @mlp_b_handles.pop
    @mlp_b_seeds   = [0]; @mlp_b_seeds.pop
    @mlp_b_douts   = [0]; @mlp_b_douts.pop
    @mlp_b_dist    = 0
    @mlp_b_scale   = 0
    @mlp_b_sigma   = 0.0
    @mlp_dfa_wired = 0
    @mlp_frozen_count = 0
  end

  # Allocate every PARAM + its Adam moments, the persistent input, and
  # random-init the weights. Takes plain Ints (NO config object): a cfg
  # class here would be one more type to keep out of every other
  # compilation unit for no gain (landmine #16).
  def realize_for_random_init(d_in, d_hidden, n_layers, n_classes,
                              batch, seed, init_scale)
    @mlp_d_in     = d_in
    @mlp_d_hidden = d_hidden
    @mlp_layers   = n_layers
    @mlp_classes  = n_classes
    @mlp_batch    = batch

    @sess = TinyNN.tnn_session_new(0)
    cap = (n_layers + 1) * 2000 + 65536
    TinyNN.tnn_session_set_graph_capacity(@sess, cap)

    # Hidden layers: W_l is [d_out, d_in] logically → ne=[d_in, d_out].
    li = 0
    while li < n_layers
      din = li == 0 ? d_in : d_hidden
      add_weight(d_hidden, din, "w" + (li + 1).to_s)
      li = li + 1
    end
    # Head — always the last entry, name "head".
    add_weight(n_classes, d_hidden, "head")

    wi = 0
    while wi < @ft_weights.length
      TinyNN.tnn_set_param(@ft_weights[wi])
      wi = wi + 1
    end

    # toy#133: the persistent graph input must be allocated BEFORE
    # finalize_weights or it silently reads zeros at compute time.
    @t_x = TinyNN.tnn_input_2d_f32_persistent(@sess, batch, d_in)
    TinyNN.tnn_tensor_set_name(@t_x, "x")

    TinyNN.tnn_finalize_weights(@sess)
    upload_random_init!(seed, init_scale)
    nil
  end

  # Build forward + CE + backward + the per-layer update rules.
  # `policy` is a flat Int array (one code per HIDDEN layer; missing
  # entries default to chain). Returns [t_loss, t_labels, t_hp].
  def build_training_step(policy, b_seed, b_dist, b_scale, b_sigma)
    @mlp_b_dist  = b_dist
    @mlp_b_scale = b_scale
    @mlp_b_sigma = b_sigma

    TinyNN.tnn_reset_for_rebuild(@sess)

    # ---- forward, recording the per-layer tap (input activation) and
    # pre-activation. Both are pinned as outputs: the DFA branch reads
    # them from OUTSIDE the autodiff graph, so a sched slot alias would
    # feed it stale/zero data.
    taps = [TinyNN.tnn_null_ptr]; taps.pop
    pres = [TinyNN.tnn_null_ptr]; pres.pop
    t_h = @t_x
    li = 0
    while li < @mlp_layers
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

    @t_logits = TinyNN.tnn_matmul(@sess, @ft_weights[@mlp_layers], t_h)
    TinyNN.tnn_set_output(@t_logits)

    @t_labels = TinyNN.tnn_input_2d_f32(@sess, @mlp_batch, @mlp_classes)
    @t_hp     = TinyNN.tnn_input_1d_f32(@sess, 7)

    @t_loss = TinyNN.tnn_cross_entropy_loss(@sess, @t_logits, @t_labels)
    TinyNN.tnn_set_output(@t_loss)
    TinyNN.tnn_set_loss(@t_loss)
    TinyNN.tnn_build_forward_only(@sess, @t_loss)
    # toy#150: everything below is extend_backward_graph territory and
    # MUST come after this call, or it is silently discarded.
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
      # The CE error signal, identical in form to the franken lane's:
      # (p - y)/B. This is the EXACT gradient of the loss wrt the
      # logits, so the DFA arm and the BP arm differ ONLY in how that
      # signal reaches the hidden layers.
      t_p = TinyNN.tnn_softmax(@sess, @t_logits)
      t_e = TinyNN.tnn_scale(@sess,
              TinyNN.tnn_sub(@sess, t_p, @t_labels),
              1.0 / @mlp_batch.to_f)
      TinyNN.tnn_set_output(t_e)
    end

    @mlp_align_grads  = [TinyNN.tnn_null_ptr]; @mlp_align_grads.pop
    @mlp_align_accs   = [TinyNN.tnn_null_ptr]; @mlp_align_accs.pop
    @mlp_align_lis    = [0]; @mlp_align_lis.pop
    @mlp_align_wnames = [""]; @mlp_align_wnames.pop
    @mlp_b_handles = [TinyNN.tnn_null_ptr]; @mlp_b_handles.pop
    @mlp_b_seeds   = [0]; @mlp_b_seeds.pop
    @mlp_b_douts   = [0]; @mlp_b_douts.pop
    @mlp_frozen_count = 0

    lj = 0
    while lj < @mlp_layers
      mode = POLICY_CHAIN
      if lj < policy.length
        mode = policy[lj]
      end
      tw   = @ft_weights[lj]
      din  = @ft_din[lj]
      dout = @ft_dout[lj]
      if mode == POLICY_DFA
        t_b = TinyNN.tnn_input_2d_f32(@sess, dout, @mlp_classes)
        TinyNN.tnn_set_output(t_b)
        # B_l · e  →  [d_out, B]
        t_delta = TinyNN.tnn_matmul(@sess, t_b, t_e)
        # ⊙ f'(a_l): ggml_silu_back(x=pre, dy=delta) is exactly this.
        t_dpre  = TinyNN.tnn_silu_back(@sess, pres[lj], t_delta)
        t_tap_t = TinyNN.tnn_cont_2d(@sess,
                    TinyNN.tnn_transpose(@sess, taps[lj]), @mlp_batch, din)
        t_del_t = TinyNN.tnn_cont_2d(@sess,
                    TinyNN.tnn_transpose(@sess, t_dpre), @mlp_batch, dout)
        t_g = TinyNN.tnn_matmul(@sess, t_tap_t, t_del_t)   # ne=[din, dout]
        TinyNN.tnn_set_output(t_g)
        to = TinyNN.tnn_opt_step_adamw(@sess, tw, t_g,
                                        @ft_m[lj], @ft_v[lj], @t_hp)
        TinyNN.tnn_extend_backward_graph(@sess, to)
        @mlp_b_handles.push(t_b)
        @mlp_b_seeds.push(b_seed + lj * 1000)
        @mlp_b_douts.push(dout)
        # PIN the shadow acc — a pure-dfa weight's chain grad has no
        # consumer, so without this the align download reads zeros.
        t_acc = TinyNN.tnn_tensor_grad(@sess, tw)
        TinyNN.tnn_set_output(t_acc)
        @mlp_align_grads.push(t_g)
        @mlp_align_accs.push(t_acc)
        @mlp_align_lis.push(lj)
        @mlp_align_wnames.push(@ft_names[lj])
      elsif mode == POLICY_FROZEN
        # No optimizer step: the layer stays at init. Its chain grad is
        # still computed (it is a PARAM and the head's backward runs
        # through it) — it is simply never applied.
        @mlp_frozen_count = @mlp_frozen_count + 1
      else
        tg = TinyNN.tnn_tensor_grad(@sess, tw)
        to = TinyNN.tnn_opt_step_adamw(@sess, tw, tg,
                                        @ft_m[lj], @ft_v[lj], @t_hp)
        TinyNN.tnn_extend_backward_graph(@sess, to)
      end
      lj = lj + 1
    end

    # The head always trains by BP (DFA == BP at the output layer).
    hw = @mlp_layers
    thg = TinyNN.tnn_tensor_grad(@sess, @ft_weights[hw])
    tho = TinyNN.tnn_opt_step_adamw(@sess, @ft_weights[hw], thg,
                                     @ft_m[hw], @ft_v[hw], @t_hp)
    TinyNN.tnn_extend_backward_graph(@sess, tho)

    @mlp_dfa_wired = @mlp_b_handles.length
    # Loud wiring summary (never-mask: a zero dfa count under a
    # non-empty dfa policy is a bug, not a quiet fallback).
    puts "mlp: layers=" + @mlp_layers.to_s +
         " policy_len=" + policy.length.to_s +
         " dfa_wired=" + @mlp_dfa_wired.to_s +
         " frozen=" + @mlp_frozen_count.to_s

    TinyNN.tnn_pin_all_graph_b_nodes(@sess)
    TinyNN.tnn_realize_backward(@sess)

    # B uploads: the buffers exist only after the sched alloc above.
    # Read-only leaves — uploaded once, stable across computes.
    refresh_b!
    [@t_loss, @t_labels, @t_hp]
  end

  # Re-upload every feedback matrix from its recorded seed/axes. Called
  # once after realize; re-callable (the franken lane re-uploads per
  # step because its B leaves are graph inputs — here they are stable,
  # so this is the one-shot fill plus a hook for future per-step axes).
  def refresh_b!
    bi = 0
    while bi < @mlp_b_handles.length
      dout = @mlp_b_douts[bi]
      nb   = dout * @mlp_classes
      sig  = Toy::Train::DfaB.sigma_for(@mlp_b_scale, @mlp_classes,
                                        dout, @mlp_b_sigma)
      TinyNN.tnn_upload_from_float_array(@sess, @mlp_b_handles[bi],
        Toy::Train::DfaB.fill(nb, @mlp_b_seeds[bi], @mlp_b_dist, sig), nb)
      bi = bi + 1
    end
    nil
  end

  # Total trainable parameter count (the run_start cost object).
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

  # toy#114 — mixer-seeded stream state (a raw [seed] state has a zero
  # fixed point and poor small-seed avalanche). Same 31-bit LCG as the
  # llama/vit engines (matz/spinel#3371: 64-bit masks wrap negative
  # under Spinel).
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
