# lib/toy/llm/engine/gnn_engine.rb — the toy#153 (DFA-arch T1) GNN
# node-classification engine: an L-layer GCN-class message-passing net
# over a normalised adjacency, a small-#classes head, a MASKED
# cross-entropy over the labelled nodes only, and a PER-LAYER
# credit-assignment policy (chain | dfa | frozen).
#
# WHY THIS LANE. toy#153 calls it the highest-confidence real positive
# in the cross-architecture survey: node classification has a tiny
# output dim (Cora 7, PubMed 3), which is the regime our lens says DFA
# works in and where toy#152 found the program's first positive. And
# full-graph backprop is where DFA has a genuine value prop — it stores
# every node's activations and is update-locked across layers, while a
# DFA layer needs only the error and its own local activations.
#
# ── THE FORWARD (GCN, bias-free) ──
#
#   tap_1    = S-hat X                (HOST-precomputed; see below)
#   z_l      = W_l . tap_l                                 [d_out, N]
#   h_l      = silu(z_l)
#   tap_l+1  = S-hat h_l              (in-graph propagation)
#   logits   = W_head . tap_L+1                            [C, N]
#
# S-hat = D^-1/2 (A+I) D^-1/2 is symmetric, so ONE dense [N, N] tensor
# serves the forward propagation and the structure-aware feedback both.
# Propagation is `cont_2d(transpose(h)) . S-hat` — the same transpose
# idiom the MLP anchor uses for its gradient outer product, because
# ggml's mul_mat contracts ne0 and the node axis is ne1 here.
#
# The FIRST propagation is folded into the host preprocessing (the
# engine is handed S-hat X, not X). S-hat and X are both constant, so
# S-hat(X W) == (S-hat X) W is an exact identity; doing it once on the
# host keeps an N^2 x feat_dim matmul out of every step, which on Cora
# is the difference between 10.5 GFLOP and 0. It changes no gradient:
# neither BP nor DFA differentiates a constant input.
#
# ── THE LOSS IS MASKED, AND THAT IS THE POINT OF THE LANE ──
#
# This is the TRANSDUCTIVE semi-supervised setting: the forward pass
# sees the whole graph, but only ~20 nodes per class carry a label. The
# CE therefore runs on a get_rows GATHER of the labelled columns, so a
# node with no label contributes no loss and no gradient. Its exact
# gradient wrt the logits is
#
#     e = (softmax(logits) - Y) (*) M / n_train              [C, N]
#
# with Y the one-hot (zero off the training set) and M the 0/1 column
# mask — which is precisely the DFA error signal below, so the BP arm
# and the DFA arm differ ONLY in how that signal reaches the hidden
# layers, never in what the signal is.
#
# ── THE TWO FEEDBACK MODES ──
#
#   direct    : delta_l = (B_l . e) (*) f'(z_l)     — canonical DFA
#   structure : delta_l = (B_l . (S-hat^k e)) (*) f'(z_l)
#
# `structure` is DFA-GNN's (Zhao et al., NeurIPS 2024) mechanism in its
# simplest honest form: the error is spread along the graph BEFORE the
# random projection, so an unlabelled node — which has exactly zero
# direct error — receives a pseudo-error assembled from its labelled
# neighbourhood. Under `direct`, every unlabelled node contributes
# nothing to any hidden-layer update, and with 20 labels per class that
# is most of the graph. This is the whole reason structure-aware
# feedback exists, and the contrast is crisp HERE because the
# propagation sits BEFORE the weight in each layer: no S-hat appears in
# the DFA backward path at all except the one the feedback mode puts
# there.
#
# ARMS (per hidden layer):
#   0 = chain  — plain backprop
#   1 = dfa    — the rule above
#   2 = frozen — no optimizer step; the layer stays at init and only the
#                head trains. The MANDATORY control (tao#19 item 4), and
#                an unusually demanding one in a GNN: aggregation is
#                ARCHITECTURE, so a frozen random stack still smooths
#                features over the graph. See toy_gnn_task.rb.
#
# The HEAD is always chain (at the output layer DFA and BP coincide).
# Under an all-chain policy the emitted graph is IDENTICAL to a plain BP
# build — prep/gnn_gate.rb pins that byte-for-byte.
#
# Spinel hygiene (landmine #16): plain class, no-arg ctor, no Struct,
# typed-empty array seeds, while loops, no #{} interpolation.
#
# LANDMINES honoured here:
#   - every persistent input (t_x, t_s, t_train_idx, t_y, t_mask,
#     t_labels) is ALLOCATED BEFORE finalize_weights (toy#133);
#   - every extend_backward_graph comes AFTER tnn_build_backward
#     (toy#150) — earlier is silently discarded;
#   - the chain grad-acc read back for align telemetry is PINNED with
#     set_output: for a pure-dfa weight it has NO consumer, so sched
#     would alias its slot and the align download would read zeros.

require_relative "../../models/transformer"
require_relative "../../train/dfa_b"

module Toy; module LLM; module Engine
class GnnEngine
  POLICY_CHAIN  = 0
  POLICY_DFA    = 1
  POLICY_FROZEN = 2

  FEEDBACK_DIRECT    = 0
  FEEDBACK_STRUCTURE = 1

  attr_accessor :sess,
                :gnn_nodes, :gnn_feat, :gnn_hidden, :gnn_layers,
                :gnn_classes, :gnn_train,
                :ft_weights, :ft_m, :ft_v, :ft_din, :ft_dout, :ft_names,
                :t_x, :t_s, :t_train_idx, :t_y, :t_mask, :t_labels,
                :t_hp, :t_logits, :t_loss,
                :gnn_align_grads, :gnn_align_accs, :gnn_align_lis,
                :gnn_align_wnames,
                :gnn_b_handles, :gnn_b_seeds, :gnn_b_douts,
                :gnn_b_dist, :gnn_b_scale, :gnn_b_sigma,
                :gnn_dfa_wired, :gnn_frozen_count, :gnn_feedback_hops

  def initialize
    @sess        = TinyNN.tnn_null_ptr
    @gnn_nodes   = 0
    @gnn_feat    = 0
    @gnn_hidden  = 0
    @gnn_layers  = 0
    @gnn_classes = 0
    @gnn_train   = 0
    @ft_weights  = [TinyNN.tnn_null_ptr]; @ft_weights.pop
    @ft_m        = [TinyNN.tnn_null_ptr]; @ft_m.pop
    @ft_v        = [TinyNN.tnn_null_ptr]; @ft_v.pop
    @ft_din      = [0]; @ft_din.pop
    @ft_dout     = [0]; @ft_dout.pop
    @ft_names    = [""]; @ft_names.pop
    @t_x         = TinyNN.tnn_null_ptr
    @t_s         = TinyNN.tnn_null_ptr
    @t_train_idx = TinyNN.tnn_null_ptr
    @t_y         = TinyNN.tnn_null_ptr
    @t_mask      = TinyNN.tnn_null_ptr
    @t_labels    = TinyNN.tnn_null_ptr
    @t_hp        = TinyNN.tnn_null_ptr
    @t_logits    = TinyNN.tnn_null_ptr
    @t_loss      = TinyNN.tnn_null_ptr
    @gnn_align_grads  = [TinyNN.tnn_null_ptr]; @gnn_align_grads.pop
    @gnn_align_accs   = [TinyNN.tnn_null_ptr]; @gnn_align_accs.pop
    @gnn_align_lis    = [0]; @gnn_align_lis.pop
    @gnn_align_wnames = [""]; @gnn_align_wnames.pop
    @gnn_b_handles = [TinyNN.tnn_null_ptr]; @gnn_b_handles.pop
    @gnn_b_seeds   = [0]; @gnn_b_seeds.pop
    @gnn_b_douts   = [0]; @gnn_b_douts.pop
    @gnn_b_dist    = 0
    @gnn_b_scale   = 0
    @gnn_b_sigma   = 0.0
    @gnn_dfa_wired = 0
    @gnn_frozen_count = 0
    @gnn_feedback_hops = 0
  end

  # Allocate every PARAM + its Adam moments, every persistent input, and
  # random-init the weights. Plain Ints only (no cfg object — landmine
  # #16 keeps this unit's type set minimal).
  def realize_for_random_init(n_nodes, feat_dim, d_hidden, n_layers,
                              n_classes, n_train, seed, init_scale)
    @gnn_nodes   = n_nodes
    @gnn_feat    = feat_dim
    @gnn_hidden  = d_hidden
    @gnn_layers  = n_layers
    @gnn_classes = n_classes
    @gnn_train   = n_train

    @sess = TinyNN.tnn_session_new(0)
    cap = (n_layers + 2) * 4000 + 131072
    TinyNN.tnn_session_set_graph_capacity(@sess, cap)

    li = 0
    while li < n_layers
      din = li == 0 ? feat_dim : d_hidden
      add_weight(d_hidden, din, "w" + (li + 1).to_s)
      li = li + 1
    end
    add_weight(n_classes, d_hidden, "head")

    wi = 0
    while wi < @ft_weights.length
      TinyNN.tnn_set_param(@ft_weights[wi])
      wi = wi + 1
    end

    # toy#133: EVERY persistent input is allocated here, before
    # finalize_weights — one allocated later reads zeros in silence.
    # All six are constant for the whole run (the "batch" is the graph),
    # so they are uploaded once below and never touched per step.
    @t_x = TinyNN.tnn_input_2d_f32_persistent(@sess, n_nodes, feat_dim)
    TinyNN.tnn_tensor_set_name(@t_x, "x")
    @t_s = TinyNN.tnn_input_2d_f32_persistent(@sess, n_nodes, n_nodes)
    TinyNN.tnn_tensor_set_name(@t_s, "s_hat")
    @t_train_idx = TinyNN.tnn_input_1d_i32_persistent(@sess, n_train)
    TinyNN.tnn_tensor_set_name(@t_train_idx, "train_idx")
    @t_y = TinyNN.tnn_input_2d_f32_persistent(@sess, n_nodes, n_classes)
    TinyNN.tnn_tensor_set_name(@t_y, "y")
    @t_mask = TinyNN.tnn_input_2d_f32_persistent(@sess, n_nodes, n_classes)
    TinyNN.tnn_tensor_set_name(@t_mask, "train_mask")
    @t_labels = TinyNN.tnn_input_2d_f32_persistent(@sess, n_train, n_classes)
    TinyNN.tnn_tensor_set_name(@t_labels, "labels")

    TinyNN.tnn_finalize_weights(@sess)
    upload_random_init!(seed, init_scale)
    nil
  end

  # Upload the graph. Called once, after realize_for_random_init and
  # BEFORE build_training_step (the buffers exist from
  # finalize_weights). `s_flat` is row-major S-hat, `x_flat` is the
  # PRE-PROPAGATED features (node-major), `train_idx` the labelled node
  # ids, `y_flat`/`mask_flat` the [C, N] one-hot and column mask, and
  # `lab_flat` the [C, n_train] gathered one-hot the CE reads.
  def upload_graph!(x_flat, s_flat, train_idx, y_flat, mask_flat, lab_flat)
    TinyNN.tnn_upload_from_float_array(@sess, @t_x, x_flat,
                                       @gnn_nodes * @gnn_feat)
    TinyNN.tnn_upload_from_float_array(@sess, @t_s, s_flat,
                                       @gnn_nodes * @gnn_nodes)
    TinyNN.upload_int_array(@sess, @t_train_idx, train_idx)
    TinyNN.tnn_upload_from_float_array(@sess, @t_y, y_flat,
                                       @gnn_nodes * @gnn_classes)
    TinyNN.tnn_upload_from_float_array(@sess, @t_mask, mask_flat,
                                       @gnn_nodes * @gnn_classes)
    TinyNN.tnn_upload_from_float_array(@sess, @t_labels, lab_flat,
                                       @gnn_train * @gnn_classes)
    nil
  end

  # Build forward + masked CE + backward + the per-layer update rules.
  # `policy` is a flat Int array (one code per hidden layer; missing
  # entries default to chain). Returns [t_loss, t_hp].
  def build_training_step(policy, feedback, hops, b_seed, b_dist,
                          b_scale, b_sigma)
    @gnn_b_dist  = b_dist
    @gnn_b_scale = b_scale
    @gnn_b_sigma = b_sigma
    @gnn_feedback_hops = feedback == FEEDBACK_STRUCTURE ? hops : 0

    TinyNN.tnn_reset_for_rebuild(@sess)

    # ---- forward. taps[l] is layer l's INPUT (already propagated),
    # pres[l] its pre-activation. Both are pinned as outputs: the DFA
    # branch reads them from OUTSIDE the autodiff graph, so a sched slot
    # alias would feed it stale or zero data.
    taps = [TinyNN.tnn_null_ptr]; taps.pop
    pres = [TinyNN.tnn_null_ptr]; pres.pop
    t_tap = @t_x
    li = 0
    while li < @gnn_layers
      taps.push(t_tap)
      TinyNN.tnn_set_output(t_tap)
      t_z = TinyNN.tnn_matmul(@sess, @ft_weights[li], t_tap)
      TinyNN.tnn_set_output(t_z)
      pres.push(t_z)
      t_h = TinyNN.tnn_silu(@sess, t_z)
      TinyNN.tnn_set_output(t_h)
      t_tap = propagate(t_h, @gnn_hidden)
      TinyNN.tnn_set_output(t_tap)
      li = li + 1
    end
    taps.push(t_tap)

    @t_logits = TinyNN.tnn_matmul(@sess, @ft_weights[@gnn_layers], t_tap)
    TinyNN.tnn_set_output(@t_logits)

    # The MASKED loss: gather the labelled columns and run the ordinary
    # CE on them. A node outside the training set contributes no loss
    # and no gradient — which is what "semi-supervised" means here, and
    # what makes structure-aware feedback a real question rather than a
    # decoration.
    t_log_tr = TinyNN.tnn_get_rows(@sess, @t_logits, @t_train_idx)
    TinyNN.tnn_set_output(t_log_tr)
    @t_hp   = TinyNN.tnn_input_1d_f32(@sess, 7)
    @t_loss = TinyNN.tnn_cross_entropy_loss(@sess, t_log_tr, @t_labels)
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

    t_fb = TinyNN.tnn_null_ptr
    if any_dfa
      # e = (softmax(logits) - Y) (*) M / n_train — the EXACT gradient of
      # the masked CE wrt the logits, so the two arms differ only in how
      # it is routed.
      t_p = TinyNN.tnn_softmax(@sess, @t_logits)
      t_e = TinyNN.tnn_scale(@sess,
              TinyNN.tnn_mul(@sess,
                TinyNN.tnn_sub(@sess, t_p, @t_y), @t_mask),
              1.0 / @gnn_train.to_f)
      TinyNN.tnn_set_output(t_e)
      t_fb = t_e
      hi = 0
      while hi < @gnn_feedback_hops
        t_fb = propagate(t_fb, @gnn_classes)
        TinyNN.tnn_set_output(t_fb)
        hi = hi + 1
      end
    end

    @gnn_align_grads  = [TinyNN.tnn_null_ptr]; @gnn_align_grads.pop
    @gnn_align_accs   = [TinyNN.tnn_null_ptr]; @gnn_align_accs.pop
    @gnn_align_lis    = [0]; @gnn_align_lis.pop
    @gnn_align_wnames = [""]; @gnn_align_wnames.pop
    @gnn_b_handles = [TinyNN.tnn_null_ptr]; @gnn_b_handles.pop
    @gnn_b_seeds   = [0]; @gnn_b_seeds.pop
    @gnn_b_douts   = [0]; @gnn_b_douts.pop
    @gnn_frozen_count = 0

    lj = 0
    while lj < @gnn_layers
      mode = POLICY_CHAIN
      if lj < policy.length
        mode = policy[lj]
      end
      tw   = @ft_weights[lj]
      din  = @ft_din[lj]
      dout = @ft_dout[lj]
      if mode == POLICY_DFA
        t_b = TinyNN.tnn_input_2d_f32(@sess, dout, @gnn_classes)
        TinyNN.tnn_set_output(t_b)
        t_delta = TinyNN.tnn_matmul(@sess, t_b, t_fb)          # [dout, N]
        t_dpre  = TinyNN.tnn_silu_back(@sess, pres[lj], t_delta)
        t_tap_t = TinyNN.tnn_cont_2d(@sess,
                    TinyNN.tnn_transpose(@sess, taps[lj]), @gnn_nodes, din)
        t_del_t = TinyNN.tnn_cont_2d(@sess,
                    TinyNN.tnn_transpose(@sess, t_dpre), @gnn_nodes, dout)
        t_g = TinyNN.tnn_matmul(@sess, t_tap_t, t_del_t)       # ne=[din, dout]
        TinyNN.tnn_set_output(t_g)
        to = TinyNN.tnn_opt_step_adamw(@sess, tw, t_g,
                                        @ft_m[lj], @ft_v[lj], @t_hp)
        TinyNN.tnn_extend_backward_graph(@sess, to)
        @gnn_b_handles.push(t_b)
        @gnn_b_seeds.push(b_seed + lj * 1000)
        @gnn_b_douts.push(dout)
        # PIN the shadow acc — a pure-dfa weight's chain grad has no
        # consumer, so without this the align download reads zeros.
        t_acc = TinyNN.tnn_tensor_grad(@sess, tw)
        TinyNN.tnn_set_output(t_acc)
        @gnn_align_grads.push(t_g)
        @gnn_align_accs.push(t_acc)
        @gnn_align_lis.push(lj)
        @gnn_align_wnames.push(@ft_names[lj])
      elsif mode == POLICY_FROZEN
        # No optimizer step: the layer stays at init. Its chain grad is
        # still computed (it is a PARAM in the head's backward path) —
        # it is simply never applied.
        @gnn_frozen_count = @gnn_frozen_count + 1
      else
        tg = TinyNN.tnn_tensor_grad(@sess, tw)
        to = TinyNN.tnn_opt_step_adamw(@sess, tw, tg,
                                        @ft_m[lj], @ft_v[lj], @t_hp)
        TinyNN.tnn_extend_backward_graph(@sess, to)
      end
      lj = lj + 1
    end

    hw = @gnn_layers
    thg = TinyNN.tnn_tensor_grad(@sess, @ft_weights[hw])
    tho = TinyNN.tnn_opt_step_adamw(@sess, @ft_weights[hw], thg,
                                     @ft_m[hw], @ft_v[hw], @t_hp)
    TinyNN.tnn_extend_backward_graph(@sess, tho)

    @gnn_dfa_wired = @gnn_b_handles.length
    # Loud wiring summary (never-mask: a zero dfa count under a
    # non-empty dfa policy is a bug, not a quiet fallback).
    puts "gnn: nodes=" + @gnn_nodes.to_s +
         " layers=" + @gnn_layers.to_s +
         " policy_len=" + policy.length.to_s +
         " dfa_wired=" + @gnn_dfa_wired.to_s +
         " frozen=" + @gnn_frozen_count.to_s +
         " fb_hops=" + @gnn_feedback_hops.to_s

    TinyNN.tnn_pin_all_graph_b_nodes(@sess)
    TinyNN.tnn_realize_backward(@sess)

    refresh_b!
    [@t_loss, @t_hp]
  end

  # out = S-hat . h, for h laid out [dim, N]. ggml's mul_mat contracts
  # ne0 and the node axis is ne1, hence the transpose + cont: S-hat is
  # symmetric so no separate transposed copy is needed.
  def propagate(t_h, dim)
    t_ht = TinyNN.tnn_cont_2d(@sess,
             TinyNN.tnn_transpose(@sess, t_h), @gnn_nodes, dim)
    TinyNN.tnn_matmul(@sess, t_ht, @t_s)
  end

  # Re-upload every feedback matrix from its recorded seed/axes.
  def refresh_b!
    bi = 0
    while bi < @gnn_b_handles.length
      dout = @gnn_b_douts[bi]
      nb   = dout * @gnn_classes
      sig  = Toy::Train::DfaB.sigma_for(@gnn_b_scale, @gnn_classes,
                                        dout, @gnn_b_sigma)
      TinyNN.tnn_upload_from_float_array(@sess, @gnn_b_handles[bi],
        Toy::Train::DfaB.fill(nb, @gnn_b_seeds[bi], @gnn_b_dist, sig), nb)
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

  # toy#114 — mixer-seeded stream state (a raw [seed] state has a zero
  # fixed point and poor small-seed avalanche).
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
