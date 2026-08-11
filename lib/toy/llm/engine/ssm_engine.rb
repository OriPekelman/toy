# lib/toy/llm/engine/ssm_engine.rb — the toy#155 (DFA-arch T2)
# selective-scan / Mamba-lite engine: a stack of channel-wise selective
# linear recurrences over a sequence, a SMALL-output-dim
# sequence-classification head, and a per-layer credit-assignment policy
# (chain | dfa | frozen) where `dfa` CUTS BACKPROP THROUGH TIME.
#
# WHY THIS LANE. toy#155 calls it the top novelty pick: there is no
# DFA/forward-forward precedent on Mamba/S4/S5 at all (only vanilla-RNN
# work — RFLO 2019, e-prop 2020). The structural argument is that the
# SSM core is a LINEAR RECURRENCE, so a random-feedback error injected
# per step composes through the same linear operator, which makes random
# feedback mathematically natural here in a way it is not for attention.
#
# ── THE RECURRENCE IS UNROLLED, AND THAT IS NOT A SHORTCUT ──
#
# ggml ships fused SSM_SCAN / SSM_CONV kernels and the shim exposes them
# (tnn_ssm_scan / tnn_ssm_conv), but ggml_compute_backward covers 43 ops
# and NEITHER of those is among them — they are inference-only and abort
# under autodiff. So a TRAINABLE SSM in this tree has to be built from
# differentiable primitives, i.e. explicitly unrolled over T. That is
# also the honest construction for this ticket: the BPTT graph the DFA
# arm claims to avoid has to actually exist before avoiding it means
# anything.
#
# ── THE BLOCK (channel-wise selective recurrence) ──
#
# Per layer, per step t, with x_t the residual stream [d_model, B]:
#
#   u_t  = W_in . x_t                                    [d_inner, B]
#   uc_t = sum_k cw_k (*) u_{t-k}          causal depthwise conv, width K
#   z_t  = silu(uc_t)
#   dt_t = softplus(W_dt . z_t + dt_bias)   <- THE SELECTION (nonlinear,
#                                              input-dependent)
#   a_t  = exp(-dt_t)                       <- input-dependent decay
#   b_t  = (W_b . z_t) (*) dt_t
#   h_t  = h_{t-1} (*) a_t + b_t            <- the scan
#   c_t  = W_c . z_t
#   y_t  = (h_t (*) c_t) (*) silu(W_gate . x_t)          <- the gate
#   o_t  = W_out . y_t
#   x_t' = x_t + o_t                                     (residual)
#
# The state is one scalar per inner channel (Mamba-2 / GLA / RWKV shape)
# rather than Mamba-1's [d_inner, d_state] matrix. That keeps every
# tensor 2-D and cheap while retaining the property the whole ticket
# turns on: A, B and C are all INPUT-DEPENDENT.
#
# Readout is the LAST timestep, never a mean-pool — see toy_ssm_task.rb.
#
# ── --selection lti IS THE TICKET'S OWN CONTROL, NOT A CONVENIENCE ──
#
# toy#155's sharp caveat: "in the purely-LINEAR case FA collapses to
# plain gradient descent, so the interesting alignment MUST live in the
# NONLINEAR parts (selection A,B,C / gating / conv). The experiment must
# isolate that." `--selection lti` does exactly that: dt and c become
# learned CONSTANTS (input-independent), the silu and the gate are
# dropped, and the whole block becomes LINEAR in x. W_dt / W_c / W_gate
# are then not allocated at all, so an lti run cannot accidentally carry
# a nonlinear path. The DFA-vs-BP gap in `selective` MINUS the
# DFA-vs-BP gap in `lti` is the finding.
#
# ── HOW `dfa` CUTS, AND WHERE THE ERROR GOES IN ──
#
# Both cuts use `tnn_detach` (forward-identity, gradient-opaque —
# vendor-patch 0011) and a surrogate root
#
#     L = sum( tap (*) (B . e) ),      e detached
#
# whose gradient AT the tap is exactly the random-projected error; then
# ordinary autodiff does full BP from the tap down and stops at the
# cuts. e is DETACHED because it is the fixed error signal, not
# something to differentiate through. Both roots stay live: the CE
# trains the head, the surrogates train everything below, and they
# cannot double-count because the cuts are what stop the CE reaching a
# policied layer's internals.
#
# WHICH TAP, and this is the part that is easy to get wrong: a linear
# surrogate is only meaningful at a tap that can CHANGE THE PREDICTION.
# The head reads the LAST timestep, so the final layer's o_t for
# t < T-1 has no functional path to the prediction at all — injecting
# the global error there gives a surrogate with nothing to balance it
# and the weights run away (measured: loss 1e24, then NaN). So:
#
#   CUT_LAYER — one tap, at the layer's output on the READOUT step.
#     BPTT inside the layer carries that single injection back through
#     the whole recurrence, so every step's weights still get temporal
#     credit. Bounded, and the arm that WORKS (.996 against BP's 1.000).
#   CUT_STEP  — h_{t-1} detached at every step, so no gradient crosses
#     time. Taps: the STATE h_t at every step (the only per-step
#     quantity that always reaches the prediction, since it is what
#     propagates forward in time), plus the layer output at the steps
#     where that output is live. Measured: free under `lti`,
#     catastrophic under `selective`. See prep/ssm_gate.rb leg 7.
#
# The layer input is detached under BOTH cuts (the update-locking half).
# CUT_LAYER is toy#158's macro-DFA transposed from depth to a recurrent
# block; CUT_STEP is the one with no feedforward analogue.
#
# NOTE there is no cos(g_dfa, g_bp) telemetry on this lane, and that is
# structural rather than an omission: the DFA update arrives through
# autodiff from the surrogate roots, so it lands in the SAME accumulator
# a BP run would use and there is no second tensor to compare against.
# toy#158 hit this first; its answer, and this lane's, is to gate on
# "the B seed moves the curve" instead — the assertion that actually
# catches a silently-unwired build.
#
# Spinel hygiene (landmine #16): plain class, no-arg ctor, no Struct,
# typed-empty array seeds, while loops, no #{} interpolation.
#
# LANDMINES honoured here:
#   - every persistent input is ALLOCATED BEFORE finalize_weights
#     (toy#133);
#   - every extend_backward_graph comes AFTER tnn_build_backward
#     (toy#150);
#   - ggml_mul/ggml_add repeat src1 into src0, never the other way, so
#     the [d_inner, B] operand is ALWAYS src0 and the broadcast
#     [d_inner, 1] constant is ALWAYS src1. Getting that backwards is a
#     shape abort under `selective` and silently wrong under `lti`.

require_relative "../../models/transformer"
require_relative "../../train/dfa_b"
require_relative "../../train/stream_bytes"

module Toy; module LLM; module Engine
class SsmEngine
  POLICY_CHAIN  = 0
  POLICY_DFA    = 1
  POLICY_FROZEN = 2

  SELECT_SELECTIVE = 0
  SELECT_LTI       = 1

  # WHERE the DFA cut goes, and it is the lane's real axis.
  #
  #   CUT_LAYER — cut only the LAYER boundary. BPTT stays intact inside
  #     the layer and the random feedback is injected ONCE, at the
  #     layer's output on the readout step. This is toy#158's macro-DFA
  #     transposed from depth to a recurrent block: bounded, because the
  #     tap genuinely determines the prediction, so as the model gets it
  #     right the injected error shrinks and the pressure stops. It does
  #     NOT reduce activation memory — the whole point of BPTT-within is
  #     that the T states are still retained.
  #
  #   CUT_STEP  — additionally detach h_{t-1}, so NO gradient crosses a
  #     timestep, and inject per step. This is the construction the
  #     ticket's value prop needs (step t's update depends only on the
  #     error and step t's own activations) and the one with no
  #     feedforward analogue: see the note on boundedness in
  #     build_training_step.
  CUT_LAYER = 0
  CUT_STEP  = 1

  attr_accessor :sess,
                :ssm_d_model, :ssm_d_inner, :ssm_t, :ssm_batch,
                :ssm_classes, :ssm_layers, :ssm_conv_k, :ssm_selection,
                :ssm_cut,
                :ft_weights, :ft_m, :ft_v, :ft_din, :ft_dout, :ft_names,
                :ft_layer, :ft_role,
                :t_x, :t_labels, :t_hp, :t_logits, :t_loss,
                :ssm_b_handles, :ssm_b_seeds, :ssm_b_douts,
                :ssm_b_dist, :ssm_b_scale, :ssm_b_sigma,
                :ssm_dfa_wired, :ssm_frozen_count, :ssm_graph_nodes,
                :ssm_stream_bptt, :ssm_stream_sqrt, :ssm_stream_cut,
                :ssm_dt_init

  ROLE_IN    = 0
  ROLE_GATE  = 1
  ROLE_DT    = 2
  ROLE_B     = 3
  ROLE_C     = 4
  ROLE_OUT   = 5
  ROLE_DTB   = 6
  ROLE_CB    = 7
  ROLE_CONV  = 8
  ROLE_HEAD  = 9

  def initialize
    @sess         = TinyNN.tnn_null_ptr
    @ssm_d_model  = 0
    @ssm_d_inner  = 0
    @ssm_t        = 0
    @ssm_batch    = 0
    @ssm_classes  = 0
    @ssm_layers   = 0
    @ssm_conv_k   = 0
    @ssm_selection = SELECT_SELECTIVE
    @ssm_cut      = CUT_LAYER
    @ssm_dt_init  = 0.0
    @ft_weights   = [TinyNN.tnn_null_ptr]; @ft_weights.pop
    @ft_m         = [TinyNN.tnn_null_ptr]; @ft_m.pop
    @ft_v         = [TinyNN.tnn_null_ptr]; @ft_v.pop
    @ft_din       = [0]; @ft_din.pop
    @ft_dout      = [0]; @ft_dout.pop
    @ft_names     = [""]; @ft_names.pop
    @ft_layer     = [0]; @ft_layer.pop
    @ft_role      = [0]; @ft_role.pop
    @t_x      = TinyNN.tnn_null_ptr
    @t_labels = TinyNN.tnn_null_ptr
    @t_hp     = TinyNN.tnn_null_ptr
    @t_logits = TinyNN.tnn_null_ptr
    @t_loss   = TinyNN.tnn_null_ptr
    @ssm_b_handles = [TinyNN.tnn_null_ptr]; @ssm_b_handles.pop
    @ssm_b_seeds   = [0]; @ssm_b_seeds.pop
    @ssm_b_douts   = [0]; @ssm_b_douts.pop
    @ssm_b_dist    = 0
    @ssm_b_scale   = 0
    @ssm_b_sigma   = 0.0
    @ssm_dfa_wired = 0
    @ssm_frozen_count = 0
    @ssm_graph_nodes  = 0
    @ssm_stream_bptt  = 0
    @ssm_stream_sqrt  = 0
    @ssm_stream_cut   = 0
  end

  # Allocate every PARAM + its Adam moments, the persistent sequence
  # input, and random-init the weights.
  #
  # `selection` decides WHICH weights exist: under lti the
  # input-dependent projections (W_dt, W_c, W_gate) are never allocated,
  # so an lti run cannot silently keep a nonlinear path alive.
  def realize_for_random_init(d_model, d_inner, t_len, batch, n_classes,
                              n_layers, conv_k, selection, seed,
                              init_scale, dt_init)
    @ssm_d_model = d_model
    @ssm_d_inner = d_inner
    @ssm_t       = t_len
    @ssm_batch   = batch
    @ssm_classes = n_classes
    @ssm_layers  = n_layers
    @ssm_conv_k  = conv_k
    @ssm_selection = selection
    @ssm_dt_init = dt_init

    @sess = TinyNN.tnn_session_new(0)
    cap = t_len * n_layers * 200 + 262144
    TinyNN.tnn_session_set_graph_capacity(@sess, cap)

    li = 0
    while li < n_layers
      add_w(d_inner, d_model, "l" + li.to_s + ".in",   li, ROLE_IN)
      add_w(d_inner, d_inner, "l" + li.to_s + ".b",    li, ROLE_B)
      add_w(d_model, d_inner, "l" + li.to_s + ".out",  li, ROLE_OUT)
      add_w(1,       d_inner, "l" + li.to_s + ".dt_bias", li, ROLE_DTB)
      add_w(conv_k,  d_inner, "l" + li.to_s + ".conv", li, ROLE_CONV)
      if selection == SELECT_SELECTIVE
        add_w(d_inner, d_inner, "l" + li.to_s + ".dt",   li, ROLE_DT)
        add_w(d_inner, d_inner, "l" + li.to_s + ".c",    li, ROLE_C)
        add_w(d_inner, d_model, "l" + li.to_s + ".gate", li, ROLE_GATE)
      else
        add_w(1, d_inner, "l" + li.to_s + ".c_bias", li, ROLE_CB)
      end
      li = li + 1
    end
    add_w(n_classes, d_model, "head", n_layers, ROLE_HEAD)

    wi = 0
    while wi < @ft_weights.length
      TinyNN.tnn_set_param(@ft_weights[wi])
      wi = wi + 1
    end

    # toy#133: the persistent sequence input is allocated BEFORE
    # finalize_weights. Column order is (t * batch + b), which is what
    # the per-step views below slice and what the task generator writes.
    @t_x = TinyNN.tnn_input_2d_f32_persistent(@sess, t_len * batch, d_model)
    TinyNN.tnn_tensor_set_name(@t_x, "x")

    TinyNN.tnn_finalize_weights(@sess)
    upload_random_init!(seed, init_scale)
    nil
  end

  # Build forward + CE + backward + the per-layer update rules.
  # Returns [t_loss, t_labels, t_hp].
  def build_training_step(policy, cut, b_seed, b_dist, b_scale, b_sigma)
    @ssm_cut     = cut
    @ssm_b_dist  = b_dist
    @ssm_b_scale = b_scale
    @ssm_b_sigma = b_sigma

    TinyNN.tnn_reset_for_rebuild(@sess)

    d  = @ssm_d_model
    dn = @ssm_d_inner
    bt = @ssm_batch

    # The residual stream, one handle per timestep. Layer 0 reads views
    # into the persistent input; each layer replaces them.
    xs = [TinyNN.tnn_null_ptr]; xs.pop
    ti = 0
    while ti < @ssm_t
      xs.push(TinyNN.tnn_view_2d(@sess, @t_x, d, bt, d * 4, ti * bt * d * 4))
      ti = ti + 1
    end

    @ssm_b_handles = [TinyNN.tnn_null_ptr]; @ssm_b_handles.pop
    @ssm_b_seeds   = [0]; @ssm_b_seeds.pop
    @ssm_b_douts   = [0]; @ssm_b_douts.pop
    @ssm_frozen_count = 0
    dfa_layers = [0]; dfa_layers.pop
    # The surrogate roots are attached AFTER the forward, because they
    # need the error signal the head has not produced yet. Two tap
    # families (see where they are pushed): `taps_h` is the recurrent
    # STATE at every step, `taps_o` the layer output at the steps that
    # can actually reach the readout.
    taps_h = [TinyNN.tnn_null_ptr]; taps_h.pop
    taps_o = [TinyNN.tnn_null_ptr]; taps_o.pop
    taps_count = 0

    lj = 0
    while lj < @ssm_layers
      mode = POLICY_CHAIN
      if lj < policy.length
        mode = policy[lj]
      end
      is_dfa = mode == POLICY_DFA
      if mode == POLICY_FROZEN
        @ssm_frozen_count = @ssm_frozen_count + 1
      end

      w_in   = weight_of(lj, ROLE_IN)
      w_b    = weight_of(lj, ROLE_B)
      w_out  = weight_of(lj, ROLE_OUT)
      w_dtb  = weight_of(lj, ROLE_DTB)
      w_conv = weight_of(lj, ROLE_CONV)

      us  = [TinyNN.tnn_null_ptr]; us.pop
      nxs = [TinyNN.tnn_null_ptr]; nxs.pop
      t_h = TinyNN.tnn_null_ptr

      tk = 0
      while tk < @ssm_t
        # CUT 1 (dfa): the layer boundary. Forward-identity, so the
        # forward pass is bit-for-bit what a chain build computes.
        x_in = xs[tk]
        if is_dfa
          x_in = TinyNN.tnn_detach(@sess, x_in)
        end
        t_u = TinyNN.tnn_matmul(@sess, w_in, x_in)              # [dn, B]
        us.push(t_u)
        # Causal depthwise conv, written as K shifted per-channel scales
        # of activations this loop already has. ggml_conv_1d is not
        # depthwise, and a real depthwise kernel would be a new shim op
        # for something that is K multiply-adds here.
        t_uc = TinyNN.tnn_mul(@sess, t_u, conv_row(w_conv, 0))
        ck = 1
        while ck < @ssm_conv_k
          if tk - ck >= 0
            t_uc = TinyNN.tnn_add(@sess, t_uc,
                     TinyNN.tnn_mul(@sess, us[tk - ck], conv_row(w_conv, ck)))
          end
          ck = ck + 1
        end
        t_z = t_uc
        if @ssm_selection == SELECT_SELECTIVE
          t_z = TinyNN.tnn_silu(@sess, t_uc)
        end

        # SELECTION. Under lti dt is the bias alone — a learned constant
        # in t, so the recurrence is LTI and the block is linear in x.
        t_pre = w_dtb
        if @ssm_selection == SELECT_SELECTIVE
          t_pre = TinyNN.tnn_add(@sess,
                    TinyNN.tnn_matmul(@sess, weight_of(lj, ROLE_DT), t_z), w_dtb)
        end
        t_dt = TinyNN.tnn_softplus(@sess, t_pre)
        t_a  = TinyNN.tnn_exp(@sess, TinyNN.tnn_neg(@sess, t_dt))
        # [dn, B] operand FIRST — ggml repeats src1 into src0, never the
        # other way, and under lti t_dt/t_a are [dn, 1].
        t_bb = TinyNN.tnn_mul(@sess,
                 TinyNN.tnn_matmul(@sess, w_b, t_z), t_dt)      # [dn, B]

        # CUT 2 (dfa): the timestep boundary. This is the value prop —
        # with it, step tk's update cannot depend on any later step.
        if tk == 0
          t_h = t_bb
        else
          h_prev = t_h
          if is_dfa && @ssm_cut == CUT_STEP
            h_prev = TinyNN.tnn_detach(@sess, h_prev)
          end
          t_h = TinyNN.tnn_add(@sess, TinyNN.tnn_mul(@sess, h_prev, t_a), t_bb)
        end

        t_c = weight_of(lj, ROLE_CB)
        if @ssm_selection == SELECT_SELECTIVE
          t_c = TinyNN.tnn_matmul(@sess, weight_of(lj, ROLE_C), t_z)
        end
        t_y = TinyNN.tnn_mul(@sess, t_h, t_c)                   # [dn, B]
        if @ssm_selection == SELECT_SELECTIVE
          t_g = TinyNN.tnn_silu(@sess,
                  TinyNN.tnn_matmul(@sess, weight_of(lj, ROLE_GATE), x_in))
          t_y = TinyNN.tnn_mul(@sess, t_y, t_g)
        end
        t_o = TinyNN.tnn_matmul(@sess, w_out, t_y)              # [d, B]
        if is_dfa
          if @ssm_cut == CUT_STEP
            # TAP 1 — the STATE, every step. h_t is the only quantity in
            # this architecture that influences the prediction at EVERY
            # step (it is what propagates forward in time), so it is the
            # only per-step injection point that is well-posed at all.
            TinyNN.tnn_set_output(t_h)
            taps_h.push(t_h)
            # TAP 2 — the layer output, but ONLY where it can reach the
            # readout. NOT an optimisation, correctness: the head reads
            # the LAST timestep, so the last layer's o_t for t < T-1 has
            # no functional path to the prediction, and injecting the
            # global error into a tap that cannot change the prediction
            # leaves a linear surrogate with nothing to balance it.
            if lj < @ssm_layers - 1 || tk == @ssm_t - 1
              TinyNN.tnn_set_output(t_o)
              taps_o.push(t_o)
            end
            taps_count = taps_count + 1
          elsif tk == @ssm_t - 1
            # CUT_LAYER: ONE tap per layer, at the readout step. BPTT
            # inside the layer then carries that single injection back
            # through the whole recurrence, so every step's weights get
            # temporal credit — and the tap determines the prediction,
            # which is what keeps the surrogate bounded.
            TinyNN.tnn_set_output(t_o)
            taps_o.push(t_o)
            taps_count = taps_count + 1
          end
        end
        nxs.push(TinyNN.tnn_add(@sess, x_in, t_o))
        tk = tk + 1
      end

      if is_dfa
        dfa_layers.push(lj)
      end
      xs = nxs
      lj = lj + 1
    end

    # Readout at the LAST timestep (never a pool — toy_ssm_task.rb).
    @t_logits = TinyNN.tnn_matmul(@sess,
                  weight_of(@ssm_layers, ROLE_HEAD), xs[@ssm_t - 1])
    TinyNN.tnn_set_output(@t_logits)

    @t_labels = TinyNN.tnn_input_2d_f32(@sess, bt, @ssm_classes)
    @t_hp     = TinyNN.tnn_input_1d_f32(@sess, 7)
    @t_loss   = TinyNN.tnn_cross_entropy_loss(@sess, @t_logits, @t_labels)
    TinyNN.tnn_set_output(@t_loss)

    if dfa_layers.length > 0
      # e = (softmax(logits) - y)/B, DETACHED: it is the fixed error
      # signal, not something to differentiate through (differentiating
      # it would feed the head a second, wrong gradient).
      t_p = TinyNN.tnn_softmax(@sess, @t_logits)
      e_det = TinyNN.tnn_detach(@sess,
                TinyNN.tnn_scale(@sess,
                  TinyNN.tnn_sub(@sess, t_p, @t_labels), 1.0 / bt.to_f))
      t_sur = TinyNN.tnn_null_ptr
      sur_started = false
      # ONE feedback matrix per tap family, shared across layers and
      # steps within a family: the injected signal is the SAME global
      # error at every step, which is what "direct feedback" means. The
      # two families need different shapes (d_inner vs d_model), which
      # is why there are two and not one.
      # ALLOCATE A FEEDBACK MATRIX ONLY IF ITS TAP FAMILY IS NON-EMPTY.
      # Under CUT_LAYER there are no state taps, and an input tensor
      # that nothing in the graph consumes gets NO BACKEND BUFFER — the
      # refresh_b! upload then aborts inside ggml_backend_tensor_set
      # rather than failing gracefully. Same shape as toy#154's
      # scalar-error label under an all-chain policy and franken-moe's
      # t_hp under --optimizer sgd: guard on "does the graph use it".
      t_ph = TinyNN.tnn_null_ptr
      t_po = TinyNN.tnn_null_ptr
      if taps_h.length > 0
        t_bh = TinyNN.tnn_input_2d_f32(@sess, dn, @ssm_classes)  # ne=[C, dn]
        TinyNN.tnn_set_output(t_bh)
        @ssm_b_handles.push(t_bh)
        @ssm_b_seeds.push(b_seed + 77)
        @ssm_b_douts.push(dn)
        t_ph = TinyNN.tnn_matmul(@sess, t_bh, e_det)             # [dn, B]
      end
      if taps_o.length > 0
        t_bo = TinyNN.tnn_input_2d_f32(@sess, d, @ssm_classes)   # ne=[C, d]
        TinyNN.tnn_set_output(t_bo)
        @ssm_b_handles.push(t_bo)
        @ssm_b_seeds.push(b_seed + 991)
        @ssm_b_douts.push(d)
        t_po = TinyNN.tnn_matmul(@sess, t_bo, e_det)             # [d, B]
      end
      hi = 0
      while hi < taps_h.length
        t_term = TinyNN.tnn_sum_rows(@sess,
                   TinyNN.tnn_reshape_2d(@sess,
                     TinyNN.tnn_mul(@sess, taps_h[hi], t_ph), dn * bt, 1))
        if sur_started
          t_sur = TinyNN.tnn_add(@sess, t_sur, t_term)
        else
          t_sur = t_term
          sur_started = true
        end
        hi = hi + 1
      end
      oi = 0
      while oi < taps_o.length
        t_term = TinyNN.tnn_sum_rows(@sess,
                   TinyNN.tnn_reshape_2d(@sess,
                     TinyNN.tnn_mul(@sess, taps_o[oi], t_po), d * bt, 1))
        if sur_started
          t_sur = TinyNN.tnn_add(@sess, t_sur, t_term)
        else
          t_sur = t_term
          sur_started = true
        end
        oi = oi + 1
      end
      TinyNN.tnn_set_output(t_sur)
      TinyNN.tnn_set_loss(@t_loss)
      TinyNN.tnn_set_loss(t_sur)
      TinyNN.tnn_add_to_graph(@sess, @t_loss)
      TinyNN.tnn_build_forward_only(@sess, t_sur)
    else
      TinyNN.tnn_set_loss(@t_loss)
      TinyNN.tnn_build_forward_only(@sess, @t_loss)
    end
    # toy#150: every extend_backward_graph below MUST come after this.
    TinyNN.tnn_build_backward(@sess)

    wk = 0
    while wk < @ft_weights.length
      lyr = @ft_layer[wk]
      mode = POLICY_CHAIN
      if lyr < @ssm_layers && lyr < policy.length
        mode = policy[lyr]
      end
      # The head (layer index == n_layers) always trains by BP.
      if lyr < @ssm_layers && mode == POLICY_FROZEN
        # No optimizer step: the layer stays at init.
      else
        tg = TinyNN.tnn_tensor_grad(@sess, @ft_weights[wk])
        to = TinyNN.tnn_opt_step_adamw(@sess, @ft_weights[wk], tg,
                                        @ft_m[wk], @ft_v[wk], @t_hp)
        TinyNN.tnn_extend_backward_graph(@sess, to)
      end
      wk = wk + 1
    end

    @ssm_dfa_wired = dfa_layers.length
    puts "ssm: layers=" + @ssm_layers.to_s +
         " t=" + @ssm_t.to_s +
         " selection=" + (@ssm_selection == SELECT_LTI ? "lti" : "selective") +
         " cut=" + (@ssm_cut == CUT_STEP ? "step" : "layer") +
         " policy_len=" + policy.length.to_s +
         " dfa_wired=" + @ssm_dfa_wired.to_s +
         " frozen=" + @ssm_frozen_count.to_s +
         " step_taps=" + taps_count.to_s +
         " tap_h=" + taps_h.length.to_s + " tap_o=" + taps_o.length.to_s

    TinyNN.tnn_pin_all_graph_b_nodes(@sess)
    TinyNN.tnn_realize_backward(@sess)
    # Recorded AFTER realize: the size of the graph the two arms
    # actually build is the only activation-memory proxy this harness
    # can measure honestly (see the runner).
    @ssm_graph_nodes = TinyNN.tnn_graph_n_nodes(@sess)
    measure_stream!

    refresh_b!
    [@t_loss, @t_labels, @t_hp]
  end

  # toy#159 — THE OTHER HALF OF THE MEMORY QUESTION, and it is ANALYTIC.
  # The node count above is what this harness BUILDS; a graph autodiff
  # materialises every forward tensor whatever the credit rule, which is
  # why this lane had to report the ticket's memory target as a negative.
  # These three figures come from the cell's own shapes instead. Note the
  # conv window lands in the CARRY, not the per-step term: it is O(K),
  # fixed by the kernel width, so it does not make the arm O(T).
  def measure_stream!
    sel   = @ssm_selection == SELECT_SELECTIVE
    live  = Toy::Train::StreamBytes.ssm_step_live(@ssm_d_inner, @ssm_d_model,
                                                  @ssm_batch, @ssm_layers, sel)
    inp   = Toy::Train::StreamBytes.bytes_2d(@ssm_d_model, @ssm_batch)
    carry = Toy::Train::StreamBytes.ssm_carry(@ssm_d_inner, @ssm_batch,
                                              @ssm_layers, @ssm_conv_k)
    head  = Toy::Train::StreamBytes.head(@ssm_classes, @ssm_batch)
    @ssm_stream_bptt = Toy::Train::StreamBytes.bptt(live, inp, head, @ssm_t)
    @ssm_stream_sqrt = Toy::Train::StreamBytes.bptt_sqrt_checkpointed(live, inp, carry, head, @ssm_t)
    @ssm_stream_cut  = Toy::Train::StreamBytes.cut_replay(live, inp, carry, head)
    nil
  end

  # Row k of the depthwise conv kernel as a broadcastable [d_inner, 1].
  def conv_row(w_conv, k)
    TinyNN.tnn_view_2d(@sess, w_conv, @ssm_d_inner, 1,
                       @ssm_d_inner * 4, k * @ssm_d_inner * 4)
  end

  def weight_of(layer, role)
    wi = 0
    while wi < @ft_weights.length
      if @ft_layer[wi] == layer && @ft_role[wi] == role
        return @ft_weights[wi]
      end
      wi = wi + 1
    end
    TinyNN.tnn_null_ptr
  end

  def refresh_b!
    bi = 0
    while bi < @ssm_b_handles.length
      dout = @ssm_b_douts[bi]
      nb  = dout * @ssm_classes
      sig = Toy::Train::DfaB.sigma_for(@ssm_b_scale, @ssm_classes,
                                       dout, @ssm_b_sigma)
      TinyNN.tnn_upload_from_float_array(@sess, @ssm_b_handles[bi],
        Toy::Train::DfaB.fill(nb, @ssm_b_seeds[bi], @ssm_b_dist, sig), nb)
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

  # --- bookkeeping helpers ---

  def add_w(dout, din, name, layer, role)
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
    @ft_layer.push(layer)
    @ft_role.push(role)
    nil
  end

  # Two weights get a NON-generic init, both for the same reason: the
  # generic fan-in gaussian would put this model in a regime where the
  # task is unlearnable by anybody, which would make every arm tie.
  #
  #   dt_bias -> `dt_init` (default -5): softplus(-5) = 0.0067, so the
  #     initial per-step decay is exp(-0.0067) = 0.993 and the state
  #     retains ~0.7 of a cue over a 48-step delay. At a zero-mean init
  #     the decay starts at exp(-0.69) = 0.5, a one-step half-life, and
  #     NOTHING can learn a delayed-cue task from there.
  #   conv    -> near-identity (tap 0 at 1.0, later taps small): a
  #     random depthwise kernel smears the one marked cue across K
  #     steps before the recurrence ever sees it.
  def upload_random_init!(seed, init_scale)
    state = xorshift_seed_state(seed)
    wi = 0
    while wi < @ft_weights.length
      n   = @ft_din[wi] * @ft_dout[wi]
      if @ft_role[wi] == ROLE_DTB
        upload_gaussian_offset(@ft_weights[wi], n, 0.1, @ssm_dt_init, state)
      elsif @ft_role[wi] == ROLE_CONV
        upload_conv_init(@ft_weights[wi], state)
      elsif @ft_role[wi] == ROLE_CB
        upload_gaussian_offset(@ft_weights[wi], n, 0.1, 1.0, state)
      else
        std = init_scale / Math.sqrt(@ft_din[wi].to_f)
        upload_gaussian_offset(@ft_weights[wi], n, std, 0.0, state)
      end
      zero_tensor(@ft_m[wi])
      zero_tensor(@ft_v[wi])
      wi = wi + 1
    end
    nil
  end

  def upload_conv_init(tensor, state)
    n = @ssm_d_inner * @ssm_conv_k
    buf = Array.new(n, 0.0)
    i = 0
    while i < n
      buf[i] = next_gauss(state) * 0.1
      if i < @ssm_d_inner
        buf[i] = buf[i] + 1.0
      end
      i = i + 1
    end
    TinyNN.tnn_upload_from_float_array(@sess, tensor, buf, n)
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

  def next_gauss(state)
    s = state[0]
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    state[0] = s
    u1 = (s.to_f + 1.0) / 2147483648.0
    s = state[0]
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    state[0] = s
    u2 = (s.to_f + 1.0) / 2147483648.0
    if u1 < 1.0e-12; u1 = 1.0e-12; end
    Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
  end

  def upload_gaussian_offset(tensor, n, std, offset, state)
    buf = Array.new(n, 0.0)
    i = 0
    while i < n
      buf[i] = next_gauss(state) * std + offset
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
