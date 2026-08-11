# lib/toy/llm/engine/lstm_engine.rb — the toy#157 (DFA-arch T3) LSTM
# sequence classifier: a stack of standard LSTM cells unrolled over T, a
# small-output classification head reading the LAST timestep, and a
# per-layer credit-assignment policy (chain | dfa | frozen) on the same
# `--dfa-cut layer|step` axis toy#155 introduced.
#
# WHY THIS LANE. toy#157 calls it the companion / SSM rehearsal: DFA on
# RNNs is not novel in existence (RFLO 2019; Folchini et al. ISC-HPC
# 2025 show DFA updates apply in PARALLEL across time, removing
# sequential BPTT — with BP still ahead on accuracy). The
# under-measured claim is the trade: "DFA matches BP accuracy at
# k-times-less memory for sequence length L". So this lane's job is to
# MEASURE the trade, on a gated recurrence, with the same cut axis the
# SSM lane used — which makes the two directly comparable.
#
# ── THE CELL IS THE TEXTBOOK ONE, WITH EIGHT SEPARATE WEIGHTS ──
#
#   i_t = sigmoid(W_xi x_t + W_hi h_{t-1})
#   f_t = sigmoid(W_xf x_t + W_hf h_{t-1})
#   o_t = sigmoid(W_xo x_t + W_ho h_{t-1})
#   g_t = tanh   (W_xg x_t + W_hg h_{t-1})
#   c_t = f_t (*) c_{t-1} + i_t (*) g_t
#   h_t = o_t (*) tanh(c_t)
#
# Real implementations fuse the four gates into one [d, 4H] matmul and
# slice the result. Here they are EIGHT separate matrices, deliberately:
# slicing a [4H, B] tensor along ne0 gives a strided view, and ggml's
# unary ops (sigmoid/tanh) want contiguity — the fused form would be a
# correctness risk for a speed gain that does not matter at this size.
# It also makes every gate individually addressable in the align
# telemetry's `wname`, which the fused form would not.
#
# ── THE CUT AXIS, SHARED WITH toy#155 ──
#
#   CUT_LAYER — cut only the layer boundary; inject the random feedback
#     ONCE, at the layer's hidden state on the readout step, with BPTT
#     intact inside the layer. Bounded, because that tap determines the
#     prediction.
#   CUT_STEP  — additionally detach h_{t-1} AND c_{t-1}, so NO gradient
#     crosses a timestep, and tap h_t at every step. This is the
#     "parallel across time" form the ticket cites, and the one whose
#     memory story is the ticket's success target.
#
# NOTE that unlike the SSM lane, ONE tap family suffices here: h_t IS
# the layer's output (there is no separate output projection), and it
# reaches the prediction from every step because it is what propagates
# forward in time. The SSM lane needed a second family precisely because
# its per-step layer output was inert at the final layer.
#
# ── THE MEMORY INSTRUMENT IS BYTES, NOT NODE COUNT ──
#
# toy#155 reported graph NODE COUNT as its activation-memory proxy and
# had to caveat it heavily. This lane sums ggml_nbytes over every node
# of the realized graph instead, which is the actual materialised
# activation footprint and is what the ticket's success target is
# stated in. The caveat that remains is real and is reported with the
# number: in a graph autodiff every forward tensor is materialised
# whatever the credit rule, so this measures what the harness BUILDS,
# not what a streaming implementation COULD get away with.
#
# Spinel hygiene (landmine #16): plain class, no-arg ctor, no Struct,
# typed-empty array seeds, while loops, no #{} interpolation.

require_relative "../../models/transformer"
require_relative "../../train/dfa_b"

module Toy; module LLM; module Engine
class LstmEngine
  POLICY_CHAIN  = 0
  POLICY_DFA    = 1
  POLICY_FROZEN = 2

  CUT_LAYER = 0
  CUT_STEP  = 1

  GATE_I = 0
  GATE_F = 1
  GATE_O = 2
  GATE_G = 3

  attr_accessor :sess,
                :lstm_d_in, :lstm_hidden, :lstm_t, :lstm_batch,
                :lstm_classes, :lstm_layers, :lstm_cut,
                :ft_weights, :ft_m, :ft_v, :ft_din, :ft_dout, :ft_names,
                :ft_layer,
                :t_x, :t_labels, :t_hp, :t_logits, :t_loss,
                :lstm_b_handles, :lstm_b_seeds,
                :lstm_b_dist, :lstm_b_scale, :lstm_b_sigma,
                :lstm_dfa_wired, :lstm_frozen_count,
                :lstm_graph_nodes, :lstm_graph_bytes, :lstm_taps

  def initialize
    @sess         = TinyNN.tnn_null_ptr
    @lstm_d_in    = 0
    @lstm_hidden  = 0
    @lstm_t       = 0
    @lstm_batch   = 0
    @lstm_classes = 0
    @lstm_layers  = 0
    @lstm_cut     = CUT_LAYER
    @ft_weights = [TinyNN.tnn_null_ptr]; @ft_weights.pop
    @ft_m       = [TinyNN.tnn_null_ptr]; @ft_m.pop
    @ft_v       = [TinyNN.tnn_null_ptr]; @ft_v.pop
    @ft_din     = [0]; @ft_din.pop
    @ft_dout    = [0]; @ft_dout.pop
    @ft_names   = [""]; @ft_names.pop
    @ft_layer   = [0]; @ft_layer.pop
    @t_x      = TinyNN.tnn_null_ptr
    @t_labels = TinyNN.tnn_null_ptr
    @t_hp     = TinyNN.tnn_null_ptr
    @t_logits = TinyNN.tnn_null_ptr
    @t_loss   = TinyNN.tnn_null_ptr
    @lstm_b_handles = [TinyNN.tnn_null_ptr]; @lstm_b_handles.pop
    @lstm_b_seeds   = [0]; @lstm_b_seeds.pop
    @lstm_b_dist  = 0
    @lstm_b_scale = 0
    @lstm_b_sigma = 0.0
    @lstm_dfa_wired = 0
    @lstm_frozen_count = 0
    @lstm_graph_nodes = 0
    @lstm_graph_bytes = 0
    @lstm_taps = 0
  end

  def realize_for_random_init(d_in, hidden, t_len, batch, n_classes,
                              n_layers, seed, init_scale)
    @lstm_d_in    = d_in
    @lstm_hidden  = hidden
    @lstm_t       = t_len
    @lstm_batch   = batch
    @lstm_classes = n_classes
    @lstm_layers  = n_layers

    @sess = TinyNN.tnn_session_new(0)
    cap = t_len * n_layers * 260 + 262144
    TinyNN.tnn_session_set_graph_capacity(@sess, cap)

    li = 0
    while li < n_layers
      din = li == 0 ? d_in : hidden
      # Order matters only for weight_at: x-gates then h-gates, i/f/o/g.
      add_w(hidden, din,    "l" + li.to_s + ".xi", li)
      add_w(hidden, din,    "l" + li.to_s + ".xf", li)
      add_w(hidden, din,    "l" + li.to_s + ".xo", li)
      add_w(hidden, din,    "l" + li.to_s + ".xg", li)
      add_w(hidden, hidden, "l" + li.to_s + ".hi", li)
      add_w(hidden, hidden, "l" + li.to_s + ".hf", li)
      add_w(hidden, hidden, "l" + li.to_s + ".ho", li)
      add_w(hidden, hidden, "l" + li.to_s + ".hg", li)
      # GATE BIASES, and the forget one is NOT optional. A bias-free
      # LSTM has f_t = sigmoid(0) = 0.5 at init, so the cell state
      # HALVES every step and 0.5^64 is nothing — the carry is dead
      # before training starts and BPTT's gradient vanishes long before
      # it reaches a cue 48 steps back. MEASURED: without these, BPTT
      # scored .199 on a 4-class task (below chance) while per-step DFA
      # scored .969, which would have read as a spectacular result and
      # was really a crippled LSTM. b_f is initialised to 1.0
      # (Jozefowicz et al. 2015), the rest to 0.
      add_w(1, hidden, "l" + li.to_s + ".bi", li)
      add_w(1, hidden, "l" + li.to_s + ".bf", li)
      add_w(1, hidden, "l" + li.to_s + ".bo", li)
      add_w(1, hidden, "l" + li.to_s + ".bg", li)
      li = li + 1
    end
    add_w(n_classes, hidden, "head", n_layers)

    wi = 0
    while wi < @ft_weights.length
      TinyNN.tnn_set_param(@ft_weights[wi])
      wi = wi + 1
    end

    # toy#133: the persistent sequence input is allocated BEFORE
    # finalize_weights. Column order is (t * batch + b) — the same
    # step-major layout the SSM lane and its task generator use, which
    # is why this lane can reuse SsmTask unchanged.
    @t_x = TinyNN.tnn_input_2d_f32_persistent(@sess, t_len * batch, d_in)
    TinyNN.tnn_tensor_set_name(@t_x, "x")

    TinyNN.tnn_finalize_weights(@sess)
    upload_random_init!(seed, init_scale)
    nil
  end

  def build_training_step(policy, cut, b_seed, b_dist, b_scale, b_sigma)
    @lstm_cut     = cut
    @lstm_b_dist  = b_dist
    @lstm_b_scale = b_scale
    @lstm_b_sigma = b_sigma

    TinyNN.tnn_reset_for_rebuild(@sess)

    hid = @lstm_hidden
    bt  = @lstm_batch

    xs = [TinyNN.tnn_null_ptr]; xs.pop
    ti = 0
    while ti < @lstm_t
      xs.push(TinyNN.tnn_view_2d(@sess, @t_x, @lstm_d_in, bt,
                                 @lstm_d_in * 4, ti * bt * @lstm_d_in * 4))
      ti = ti + 1
    end

    @lstm_b_handles = [TinyNN.tnn_null_ptr]; @lstm_b_handles.pop
    @lstm_b_seeds   = [0]; @lstm_b_seeds.pop
    @lstm_frozen_count = 0
    taps = [TinyNN.tnn_null_ptr]; taps.pop
    any_dfa = false

    lj = 0
    while lj < @lstm_layers
      mode = POLICY_CHAIN
      if lj < policy.length
        mode = policy[lj]
      end
      is_dfa = mode == POLICY_DFA
      if is_dfa
        any_dfa = true
      end
      if mode == POLICY_FROZEN
        @lstm_frozen_count = @lstm_frozen_count + 1
      end

      nxs = [TinyNN.tnn_null_ptr]; nxs.pop
      t_h = TinyNN.tnn_null_ptr
      t_c = TinyNN.tnn_null_ptr

      tk = 0
      while tk < @lstm_t
        x_in = xs[tk]
        if is_dfa
          x_in = TinyNN.tnn_detach(@sess, x_in)
        end
        if tk == 0
          # h_{-1} = c_{-1} = 0, so the recurrent terms vanish and
          # c_0 = i_0 (*) g_0. Building explicit zero tensors instead
          # would add T*4 nodes to say nothing.
          t_i = TinyNN.tnn_sigmoid(@sess, TinyNN.tnn_add(@sess,
                  TinyNN.tnn_matmul(@sess, weight_at(lj, GATE_I, false), x_in),
                  bias_at(lj, GATE_I)))
          t_g = TinyNN.tnn_tanh(@sess, TinyNN.tnn_add(@sess,
                  TinyNN.tnn_matmul(@sess, weight_at(lj, GATE_G, false), x_in),
                  bias_at(lj, GATE_G)))
          t_o = TinyNN.tnn_sigmoid(@sess, TinyNN.tnn_add(@sess,
                  TinyNN.tnn_matmul(@sess, weight_at(lj, GATE_O, false), x_in),
                  bias_at(lj, GATE_O)))
          t_c = TinyNN.tnn_mul(@sess, t_i, t_g)
          t_h = TinyNN.tnn_mul(@sess, t_o, TinyNN.tnn_tanh(@sess, t_c))
        else
          h_prev = t_h
          c_prev = t_c
          if is_dfa && @lstm_cut == CUT_STEP
            # THE CUT: no gradient crosses a timestep. BOTH carries have
            # to be cut — leaving c_{t-1} attached would keep a full
            # BPTT path alive through the cell state, which is exactly
            # the path this arm claims not to need.
            h_prev = TinyNN.tnn_detach(@sess, h_prev)
            c_prev = TinyNN.tnn_detach(@sess, c_prev)
          end
          t_i = gate(lj, GATE_I, x_in, h_prev, true)
          t_f = gate(lj, GATE_F, x_in, h_prev, true)
          t_o = gate(lj, GATE_O, x_in, h_prev, true)
          t_g = gate(lj, GATE_G, x_in, h_prev, false)
          t_c = TinyNN.tnn_add(@sess,
                  TinyNN.tnn_mul(@sess, t_f, c_prev),
                  TinyNN.tnn_mul(@sess, t_i, t_g))
          t_h = TinyNN.tnn_mul(@sess, t_o, TinyNN.tnn_tanh(@sess, t_c))
        end
        if is_dfa
          # h_t reaches the prediction from EVERY step (it is what
          # propagates forward in time), so unlike the SSM lane one tap
          # family is enough and every tap is well-posed.
          if @lstm_cut == CUT_STEP || tk == @lstm_t - 1
            TinyNN.tnn_set_output(t_h)
            taps.push(t_h)
          end
        end
        nxs.push(t_h)
        tk = tk + 1
      end
      xs = nxs
      lj = lj + 1
    end

    @t_logits = TinyNN.tnn_matmul(@sess,
                  @ft_weights[@ft_weights.length - 1], xs[@lstm_t - 1])
    TinyNN.tnn_set_output(@t_logits)

    @t_labels = TinyNN.tnn_input_2d_f32(@sess, bt, @lstm_classes)
    @t_hp     = TinyNN.tnn_input_1d_f32(@sess, 7)
    @t_loss   = TinyNN.tnn_cross_entropy_loss(@sess, @t_logits, @t_labels)
    TinyNN.tnn_set_output(@t_loss)

    @lstm_taps = taps.length
    if any_dfa && taps.length > 0
      t_p = TinyNN.tnn_softmax(@sess, @t_logits)
      e_det = TinyNN.tnn_detach(@sess,
                TinyNN.tnn_scale(@sess,
                  TinyNN.tnn_sub(@sess, t_p, @t_labels), 1.0 / bt.to_f))
      t_b = TinyNN.tnn_input_2d_f32(@sess, hid, @lstm_classes)   # ne=[C, hid]
      TinyNN.tnn_set_output(t_b)
      @lstm_b_handles.push(t_b)
      @lstm_b_seeds.push(b_seed + 77)
      t_proj = TinyNN.tnn_matmul(@sess, t_b, e_det)              # [hid, B]
      t_sur = TinyNN.tnn_null_ptr
      started = false
      hi = 0
      while hi < taps.length
        t_term = TinyNN.tnn_sum_rows(@sess,
                   TinyNN.tnn_reshape_2d(@sess,
                     TinyNN.tnn_mul(@sess, taps[hi], t_proj), hid * bt, 1))
        if started
          t_sur = TinyNN.tnn_add(@sess, t_sur, t_term)
        else
          t_sur = t_term
          started = true
        end
        hi = hi + 1
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
      if lyr < @lstm_layers && lyr < policy.length
        mode = policy[lyr]
      end
      if lyr < @lstm_layers && mode == POLICY_FROZEN
        # No optimizer step: the layer stays at init.
      else
        tg = TinyNN.tnn_tensor_grad(@sess, @ft_weights[wk])
        to = TinyNN.tnn_opt_step_adamw(@sess, @ft_weights[wk], tg,
                                        @ft_m[wk], @ft_v[wk], @t_hp)
        TinyNN.tnn_extend_backward_graph(@sess, to)
      end
      wk = wk + 1
    end

    @lstm_dfa_wired = @lstm_b_handles.length > 0 ? dfa_layer_count(policy) : 0
    puts "lstm: layers=" + @lstm_layers.to_s +
         " t=" + @lstm_t.to_s +
         " cut=" + (@lstm_cut == CUT_STEP ? "step" : "layer") +
         " policy_len=" + policy.length.to_s +
         " dfa_wired=" + @lstm_dfa_wired.to_s +
         " frozen=" + @lstm_frozen_count.to_s +
         " taps=" + @lstm_taps.to_s

    TinyNN.tnn_pin_all_graph_b_nodes(@sess)
    TinyNN.tnn_realize_backward(@sess)
    measure_graph!
    refresh_b!
    [@t_loss, @t_labels, @t_hp]
  end

  # THE MEMORY INSTRUMENT. Sum ggml_nbytes over every node of the
  # realized graph: the actual materialised activation footprint, which
  # is what toy#157's success target is stated in. Read it as "what this
  # harness BUILDS" — in a graph autodiff every forward tensor is
  # materialised whatever the credit rule, so a streaming implementation
  # could do better than any number here.
  def measure_graph!
    @lstm_graph_nodes = TinyNN.tnn_graph_n_nodes(@sess)
    total = 0
    i = 0
    while i < @lstm_graph_nodes
      total = total + TinyNN.tnn_tensor_nbytes(TinyNN.tnn_graph_node(@sess, i))
      i = i + 1
    end
    @lstm_graph_bytes = total
    nil
  end

  def dfa_layer_count(policy)
    n = 0
    i = 0
    while i < policy.length
      if policy[i] == POLICY_DFA
        n = n + 1
      end
      i = i + 1
    end
    n
  end

  # z = W_x . x + W_h . h_prev, then the gate nonlinearity.
  def gate(layer, which, x_in, h_prev, is_sigmoid)
    t_z = TinyNN.tnn_add(@sess,
            TinyNN.tnn_add(@sess,
              TinyNN.tnn_matmul(@sess, weight_at(layer, which, false), x_in),
              TinyNN.tnn_matmul(@sess, weight_at(layer, which, true), h_prev)),
            bias_at(layer, which))
    if is_sigmoid
      return TinyNN.tnn_sigmoid(@sess, t_z)
    end
    TinyNN.tnn_tanh(@sess, t_z)
  end

  # 12 weights per layer: 4 input gates, 4 recurrent gates, 4 biases.
  def weight_at(layer, which, recurrent)
    base = layer * 12
    if recurrent
      base = base + 4
    end
    @ft_weights[base + which]
  end

  def bias_at(layer, which)
    @ft_weights[layer * 12 + 8 + which]
  end

  def refresh_b!
    bi = 0
    while bi < @lstm_b_handles.length
      nb  = @lstm_hidden * @lstm_classes
      sig = Toy::Train::DfaB.sigma_for(@lstm_b_scale, @lstm_classes,
                                       @lstm_hidden, @lstm_b_sigma)
      TinyNN.tnn_upload_from_float_array(@sess, @lstm_b_handles[bi],
        Toy::Train::DfaB.fill(nb, @lstm_b_seeds[bi], @lstm_b_dist, sig), nb)
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

  def add_w(dout, din, name, layer)
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
    nil
  end

  def upload_random_init!(seed, init_scale)
    state = xorshift_seed_state(seed)
    wi = 0
    while wi < @ft_weights.length
      n   = @ft_din[wi] * @ft_dout[wi]
      nm  = @ft_names[wi]
      if nm.length > 3 && nm[nm.length - 3, 3] == ".bf"
        upload_const(@ft_weights[wi], n, 1.0)
      elsif nm.length > 3 && nm[nm.length - 3, 2] == ".b"
        upload_const(@ft_weights[wi], n, 0.0)
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
