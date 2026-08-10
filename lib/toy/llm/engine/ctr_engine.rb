# lib/toy/llm/engine/ctr_engine.rb — the toy#154 (DFA-arch T1) CTR
# tower: per-field embedding tables -> concat -> N-layer MLP tower ->
# SCALAR sigmoid head -> logloss, with a per-tower-layer credit
# assignment policy (chain | dfa | frozen).
#
# WHY THIS LANE. LightOn 2020 (arXiv:2006.12878) put DFA within ~0.01
# AUC of BP on Criteo — effectively matching, and their most convincing
# non-vision result. It is a SCALAR output (the extreme of the
# small-output-dim regime our lens predicts DFA likes), it has huge
# sparse embedding tables (the thing you would most want to update
# without a backward pass), and nobody has revisited it since 2020.
#
# ── THE DFA FORM HERE IS toy#158's, NOT toy#152's, AND THAT IS FORCED ──
#
# toy#152's MLP lane computes each policied weight's gradient DIRECTLY
# (grad = a_in^T (B e)^T) and propagates NOTHING. That cannot express
# this ticket's requirement — "DFA the MLP tower, embeddings stay
# chain" — because with nothing propagating out of tower layer 1, the
# embeddings below it would receive no gradient at all and would sit
# frozen at init.
#
# So this lane uses the toy#158 (macro) construction instead: a
# SURROGATE LOSS ROOT per policied tower layer,
#
#     L_l = sum( tap_l (.) B_l·e ),      e detached
#
# whose gradient AT tap_l is exactly the random-projected output error.
# Ordinary autodiff then carries it through that layer's weights and —
# for the LOWEST tower layer — onward into the embedding tables, which
# is precisely "the tower is DFA'd, the embeddings train by backprop".
# That is also what a tinydfa DFALayer stack does. The cut at each
# tower-layer output is what stops the roots double-counting.
#
# ── THE OUTPUT IS GENUINELY SCALAR ──
#
# The head is ONE unit producing z. The loss is exact binary logloss,
# obtained by concatenating a CONSTANT ZERO row and taking a 2-way
# cross-entropy: softmax([0, z]) == [1-sigmoid(z), sigmoid(z)], so
# CE([0,z], [1-y, y]) IS the binary logloss. The zero row is not a
# parameter and never trains. The DFA error signal is built directly as
# the scalar e = (sigmoid(z) - y)/B with shape [1, B], and B_l is
# [1, d_out] — so the feedback really is a projection of a
# ONE-DIMENSIONAL error, which is the claim under test.
#
# ARMS (per TOWER layer):
#   0 = chain   — plain backprop
#   1 = dfa     — the surrogate root above
#   2 = frozen  — no optimizer step for that layer's weights. The
#                 embeddings and head STILL train by backprop through
#                 the frozen weights, so this control isolates exactly
#                 "did training the tower's own weights buy anything".
#
# LANDMINES honoured:
#   - #1449: every embedding INDEX leaf is tnn_input_1d_i32_persistent
#     and allocated BEFORE finalize_weights. A compute-ctx index leaf
#     gets its slot reused for the loss output, and backward get_rows
#     then reads loss bits as an index -> wild OOB abort, layout-flaky.
#     This lane has one index leaf PER FIELD, so it would have hit it
#     n_fields times over.
#   - toy#133: persistent inputs alloc before finalize_weights.
#   - toy#150: extend_backward_graph only AFTER tnn_build_backward.
#
# Spinel hygiene: plain class, no-arg ctor, no Struct, typed-empty
# seeds, while loops, no #{} interpolation.

require_relative "../../models/transformer"
require_relative "../../train/dfa_b"

module Toy; module LLM; module Engine
class CtrEngine
  POLICY_CHAIN  = 0
  POLICY_DFA    = 1
  POLICY_FROZEN = 2

  attr_accessor :sess,
                :ctr_fields, :ctr_card, :ctr_numeric, :ctr_d_emb,
                :ctr_d_hidden, :ctr_layers, :ctr_batch, :ctr_d_in,
                :ft_weights, :ft_m, :ft_v, :ft_din, :ft_dout, :ft_names,
                :ft_is_tower,
                :t_idx, :t_numeric, :t_labels, :t_y, :t_hp,
                :t_logit, :t_loss,
                :ctr_b_handles, :ctr_b_seeds, :ctr_b_douts,
                :ctr_b_dist, :ctr_b_scale, :ctr_b_sigma,
                :ctr_dfa_wired, :ctr_frozen_count,
                # 1 only when some tower layer is :dfa. `t_y` (the
                # scalar label the DFA error reads) is then part of the
                # graph and has a backend buffer; under an all-chain
                # policy it is NOT, and uploading to it would abort
                # inside ggml_backend_tensor_set — the same failure the
                # franken-moe eval hit when t_hp went unreachable under
                # --optimizer sgd. The recipe reads this before
                # uploading rather than the engine growing a dead op.
                :ctr_uses_y,
                # toy#154: DeepFM's FM BRANCH — first-order (linear
                # over the embeddings) PLUS the second-order pairwise
                # term, straight to the logit, bypassing the tower.
                # Always chain. This is the reference architecture's
                # other half, and the reason its DFA result looks the
                # way it does.
                :ctr_wide, :ctr_wide_idx

  def initialize
    @sess         = TinyNN.tnn_null_ptr
    @ctr_fields   = 0
    @ctr_card     = 0
    @ctr_numeric  = 0
    @ctr_d_emb    = 0
    @ctr_d_hidden = 0
    @ctr_layers   = 0
    @ctr_batch    = 0
    @ctr_d_in     = 0
    @ft_weights = [TinyNN.tnn_null_ptr]; @ft_weights.pop
    @ft_m       = [TinyNN.tnn_null_ptr]; @ft_m.pop
    @ft_v       = [TinyNN.tnn_null_ptr]; @ft_v.pop
    @ft_din     = [0]; @ft_din.pop
    @ft_dout    = [0]; @ft_dout.pop
    @ft_names   = [""]; @ft_names.pop
    @ft_is_tower = [0]; @ft_is_tower.pop
    @t_idx     = [TinyNN.tnn_null_ptr]; @t_idx.pop
    @t_numeric = TinyNN.tnn_null_ptr
    @t_labels  = TinyNN.tnn_null_ptr
    @t_y       = TinyNN.tnn_null_ptr
    @t_hp      = TinyNN.tnn_null_ptr
    @t_logit   = TinyNN.tnn_null_ptr
    @t_loss    = TinyNN.tnn_null_ptr
    @ctr_b_handles = [TinyNN.tnn_null_ptr]; @ctr_b_handles.pop
    @ctr_b_seeds   = [0]; @ctr_b_seeds.pop
    @ctr_b_douts   = [0]; @ctr_b_douts.pop
    @ctr_b_dist    = 0
    @ctr_b_scale   = 0
    @ctr_b_sigma   = 0.0
    @ctr_dfa_wired = 0
    @ctr_frozen_count = 0
    @ctr_uses_y = 0
    @ctr_wide = 0
    @ctr_wide_idx = -1
  end

  # Allocate every PARAM + Adam moments + the persistent inputs, then
  # random-init. Plain Ints, no cfg object (landmine #16).
  #
  # Weight order: [emb_0 … emb_{F-1}, w1 … wL, head]. `ft_is_tower`
  # marks which entries the per-layer policy applies to — the tables
  # and the head are never policied (the ticket: DFA the tower,
  # embeddings stay chain; and at the output layer DFA == BP anyway).
  def realize_for_random_init(n_fields, cardinality, n_numeric, d_emb,
                              d_hidden, n_layers, batch, seed, init_scale,
                              wide)
    @ctr_fields   = n_fields
    @ctr_card     = cardinality
    @ctr_numeric  = n_numeric
    @ctr_d_emb    = d_emb
    @ctr_d_hidden = d_hidden
    @ctr_layers   = n_layers
    @ctr_batch    = batch
    @ctr_d_in     = n_fields * d_emb + n_numeric

    @sess = TinyNN.tnn_session_new(0)
    cap = (n_layers + n_fields + 4) * 2000 + 65536
    TinyNN.tnn_session_set_graph_capacity(@sess, cap)

    fi = 0
    while fi < n_fields
      add_weight(cardinality, d_emb, "emb_" + fi.to_s, 0)
      fi = fi + 1
    end
    li = 0
    while li < n_layers
      din = li == 0 ? @ctr_d_in : d_hidden
      add_weight(d_hidden, din, "w" + (li + 1).to_s, 1)
      li = li + 1
    end
    # The SCALAR head: one output unit.
    add_weight(1, d_hidden, "head", 0)
    # DeepFM's "wide"/FM branch: embeddings -> logit, around the tower.
    # Never policied — it is the BP path the reference architecture
    # keeps alongside the DNN tower.
    @ctr_wide = wide
    if wide == 1
      add_weight(1, @ctr_d_in, "wide", 0)
      @ctr_wide_idx = @ft_weights.length - 1
    end

    wi = 0
    while wi < @ft_weights.length
      TinyNN.tnn_set_param(@ft_weights[wi])
      wi = wi + 1
    end

    # #1449: the per-field index leaves MUST be galloc-external and
    # allocated before finalize, or backward get_rows reads a reused
    # slot as an index and aborts on a wild row.
    fj = 0
    while fj < n_fields
      t = TinyNN.tnn_input_1d_i32_persistent(@sess, batch)
      TinyNN.tnn_tensor_set_name(t, "idx_" + fj.to_s)
      @t_idx.push(t)
      fj = fj + 1
    end
    @t_numeric = TinyNN.tnn_input_2d_f32_persistent(@sess, batch, n_numeric)
    TinyNN.tnn_tensor_set_name(@t_numeric, "numeric")
    # The constant ZERO row that turns a 2-way CE into binary logloss.
    # A persistent leaf uploaded once — never a parameter.
    @t_zero = TinyNN.tnn_input_2d_f32_persistent(@sess, batch, 1)
    TinyNN.tnn_tensor_set_name(@t_zero, "logit_zero")

    TinyNN.tnn_finalize_weights(@sess)
    upload_random_init!(seed, init_scale)
    zbuf = Array.new(batch, 0.0)
    TinyNN.tnn_upload_from_float_array(@sess, @t_zero, zbuf, batch)
    nil
  end

  attr_accessor :t_zero

  # Build forward + logloss + backward + the per-layer update rules.
  # `policy` has one code per TOWER layer. Returns [t_loss, t_hp].
  def build_training_step(policy, b_seed, b_dist, b_scale, b_sigma)
    @ctr_b_dist  = b_dist
    @ctr_b_scale = b_scale
    @ctr_b_sigma = b_sigma
    @ctr_b_handles = [TinyNN.tnn_null_ptr]; @ctr_b_handles.pop
    @ctr_b_seeds   = [0]; @ctr_b_seeds.pop
    @ctr_b_douts   = [0]; @ctr_b_douts.pop
    @ctr_frozen_count = 0
    @ctr_uses_y = 0

    TinyNN.tnn_reset_for_rebuild(@sess)

    # ---- embeddings: one get_rows per field, concat along dim 0. ----
    t_x = TinyNN.tnn_null_ptr
    fi = 0
    while fi < @ctr_fields
      t_e = TinyNN.tnn_get_rows(@sess, @ft_weights[fi], @t_idx[fi])  # [d_emb, B]
      if fi == 0
        t_x = t_e
      else
        t_x = TinyNN.tnn_concat(@sess, t_x, t_e, 0)
      end
      fi = fi + 1
    end
    if @ctr_numeric > 0
      t_x = TinyNN.tnn_concat(@sess, t_x, @t_numeric, 0)                # [d_in, B]
    end
    TinyNN.tnn_set_output(t_x)

    # ---- tower, with the DFA cut at every layer output. ----
    taps = [TinyNN.tnn_null_ptr]; taps.pop
    any_dfa = false
    pk = 0
    while pk < policy.length
      if policy[pk] == POLICY_DFA
        any_dfa = true
      end
      pk = pk + 1
    end

    t_h = t_x
    li = 0
    while li < @ctr_layers
      widx = @ctr_fields + li
      t_pre  = TinyNN.tnn_matmul(@sess, @ft_weights[widx], t_h)
      t_post = TinyNN.tnn_silu(@sess, t_pre)
      TinyNN.tnn_set_output(t_post)
      taps.push(t_post)
      mode = li < policy.length ? policy[li] : POLICY_CHAIN
      # The cut goes at a DFA'd layer's output ONLY. A chain layer must
      # keep its backward path or it would silently stop training —
      # which is why this is not an unconditional cut like the
      # transformer lane's (there the recipe is all-or-nothing).
      if mode == POLICY_DFA
        t_h = TinyNN.tnn_detach(@sess, t_post)
      else
        t_h = t_post
      end
      li = li + 1
    end

    # ---- scalar head + exact binary logloss. ----
    hidx = @ctr_fields + @ctr_layers
    t_deep = TinyNN.tnn_matmul(@sess, @ft_weights[hidx], t_h)        # [1, B]
    if @ctr_wide == 1
      # logit = deep(tower) + FM(embeddings). Both FM terms read
      # activations BEFORE any cut, so they train by ordinary BP into
      # the tables regardless of the tower's policy.
      #
      #   first order:  w · [e_1 … e_F, numeric]
      #   second order: 1/2 * sum_j [ (sum_f e_f[j])^2 - sum_f e_f[j]^2 ]
      #                 == sum over field PAIRS of <e_f, e_g>
      #
      # The second-order term is the whole point: it computes the
      # pairwise crosses EXPLICITLY, without the tower. That identity
      # is why DeepFM can score well with a barely-trained DNN branch.
      t_deep = TinyNN.tnn_add(@sess, t_deep,
                 TinyNN.tnn_matmul(@sess, @ft_weights[@ctr_wide_idx], t_x))
      t_sum = TinyNN.tnn_null_ptr
      t_sqs = TinyNN.tnn_null_ptr
      fk = 0
      while fk < @ctr_fields
        t_ef = TinyNN.tnn_get_rows(@sess, @ft_weights[fk], @t_idx[fk])
        t_sq = TinyNN.tnn_mul(@sess, t_ef, t_ef)
        if fk == 0
          t_sum = t_ef
          t_sqs = t_sq
        else
          t_sum = TinyNN.tnn_add(@sess, t_sum, t_ef)
          t_sqs = TinyNN.tnn_add(@sess, t_sqs, t_sq)
        end
        fk = fk + 1
      end
      t_fm = TinyNN.tnn_sum_rows(@sess,
               TinyNN.tnn_scale(@sess,
                 TinyNN.tnn_sub(@sess, TinyNN.tnn_mul(@sess, t_sum, t_sum), t_sqs),
                 0.5))
      t_deep = TinyNN.tnn_add(@sess, t_deep, t_fm)
    end
    @t_logit = t_deep
    TinyNN.tnn_set_output(@t_logit)
    # softmax([0, z]) = [1-sigmoid(z), sigmoid(z)] -> CE IS the logloss.
    t_logits2 = TinyNN.tnn_concat(@sess, @t_zero, @t_logit, 0)       # [2, B]
    TinyNN.tnn_set_output(t_logits2)

    @t_labels = TinyNN.tnn_input_2d_f32(@sess, @ctr_batch, 2)
    @t_y      = TinyNN.tnn_input_2d_f32(@sess, @ctr_batch, 1)
    @t_hp     = TinyNN.tnn_input_1d_f32(@sess, 7)

    @t_loss = TinyNN.tnn_cross_entropy_loss(@sess, t_logits2, @t_labels)
    TinyNN.tnn_set_output(@t_loss)
    TinyNN.tnn_set_loss(@t_loss)

    # ---- the surrogate roots: one per DFA'd tower layer. ----
    t_root = TinyNN.tnn_null_ptr
    if any_dfa
      @ctr_uses_y = 1
      # The SCALAR error, built directly: e = (sigmoid(z) - y)/B, [1,B].
      # Detached — it is the fixed error signal, not something to
      # differentiate through (differentiating it would feed the head a
      # second, wrong gradient).
      t_p = TinyNN.tnn_sigmoid(@sess, @t_logit)
      e_det = TinyNN.tnn_detach(@sess,
                TinyNN.tnn_scale(@sess, TinyNN.tnn_sub(@sess, t_p, @t_y),
                                 1.0 / @ctr_batch.to_f))
      TinyNN.tnn_set_output(e_det)
      lj = 0
      while lj < @ctr_layers
        mode = lj < policy.length ? policy[lj] : POLICY_CHAIN
        if mode == POLICY_DFA
          # B_l is [1, d_hidden]: a projection of a ONE-DIMENSIONAL
          # error. That is the scalar-output claim, literally.
          t_b = TinyNN.tnn_input_2d_f32(@sess, @ctr_d_hidden, 1)
          TinyNN.tnn_set_output(t_b)
          delta = TinyNN.tnn_matmul(@sess, t_b, e_det)                # [d_hidden, B]
          prod  = TinyNN.tnn_mul(@sess, taps[lj], delta)
          term  = TinyNN.tnn_sum_rows(@sess,
                    TinyNN.tnn_reshape_2d(@sess, prod, @ctr_d_hidden * @ctr_batch, 1))
          if t_root == TinyNN.tnn_null_ptr
            t_root = term
          else
            t_root = TinyNN.tnn_add(@sess, t_root, term)
          end
          @ctr_b_handles.push(t_b)
          @ctr_b_seeds.push(b_seed + 101 + lj * 1000)
          @ctr_b_douts.push(@ctr_d_hidden)
        end
        lj = lj + 1
      end
    end

    if t_root == TinyNN.tnn_null_ptr
      TinyNN.tnn_build_forward_only(@sess, @t_loss)
    else
      TinyNN.tnn_set_output(t_root)
      TinyNN.tnn_set_loss(t_root)
      TinyNN.tnn_add_to_graph(@sess, @t_loss)
      TinyNN.tnn_build_forward_only(@sess, t_root)
    end
    # toy#150: every extend_backward_graph below MUST follow this.
    TinyNN.tnn_build_backward(@sess)

    # ---- optimizer steps. Tables + head always; tower per policy. ----
    wi = 0
    while wi < @ft_weights.length
      skip = false
      if @ft_is_tower[wi] == 1
        tl = wi - @ctr_fields
        mode = tl < policy.length ? policy[tl] : POLICY_CHAIN
        if mode == POLICY_FROZEN
          # No step: the layer stays at init. Its grad is still
          # computed (it is a PARAM and the head backprops through it),
          # it is simply never applied — so the embeddings below still
          # train, which is what isolates the tower's contribution.
          skip = true
          @ctr_frozen_count = @ctr_frozen_count + 1
        end
      end
      if !skip
        tw = @ft_weights[wi]
        tg = TinyNN.tnn_tensor_grad(@sess, tw)
        to = TinyNN.tnn_opt_step_adamw(@sess, tw, tg, @ft_m[wi], @ft_v[wi], @t_hp)
        TinyNN.tnn_extend_backward_graph(@sess, to)
      end
      wi = wi + 1
    end

    @ctr_dfa_wired = @ctr_b_handles.length
    puts "ctr: fields=" + @ctr_fields.to_s +
         " tower=" + @ctr_layers.to_s +
         " d_in=" + @ctr_d_in.to_s +
         " dfa_wired=" + @ctr_dfa_wired.to_s +
         " frozen=" + @ctr_frozen_count.to_s

    TinyNN.tnn_pin_all_graph_b_nodes(@sess)
    TinyNN.tnn_realize_backward(@sess)
    refresh_b!
    [@t_loss, @t_hp]
  end

  # Upload every feedback matrix. fan_in is 1 (the scalar error), which
  # is the whole point of this lane — inv_sqrt_fan therefore gives
  # sigma 1.0 here rather than 1/sqrt(vocab).
  def refresh_b!
    bi = 0
    while bi < @ctr_b_handles.length
      dout = @ctr_b_douts[bi]
      sig  = Toy::Train::DfaB.sigma_for(@ctr_b_scale, 1, dout, @ctr_b_sigma)
      TinyNN.tnn_upload_from_float_array(@sess, @ctr_b_handles[bi],
        Toy::Train::DfaB.fill(dout, @ctr_b_seeds[bi], @ctr_b_dist, sig), dout)
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

  def tower_param_count
    n = 0
    wi = 0
    while wi < @ft_weights.length
      if @ft_is_tower[wi] == 1
        n = n + @ft_din[wi] * @ft_dout[wi]
      end
      wi = wi + 1
    end
    n
  end

  # --- bookkeeping ---

  def add_weight(dout, din, name, is_tower)
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
    @ft_is_tower.push(is_tower)
    nil
  end

  def upload_random_init!(seed, init_scale)
    state = xorshift_seed_state(seed)
    wi = 0
    while wi < @ft_weights.length
      n = @ft_din[wi] * @ft_dout[wi]
      # Embedding tables get the conventional small constant std; the
      # tower and head are fan-in scaled.
      std = @ft_is_tower[wi] == 0 && @ft_dout[wi] > 1 ?
              0.05 * init_scale : init_scale / Math.sqrt(@ft_din[wi].to_f)
      upload_gaussian(@ft_weights[wi], n, std, state)
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

  def zero_tensor(tensor)
    n = TinyNN.tnn_tensor_nelements(tensor)
    buf = Array.new(n, 0.0)
    TinyNN.tnn_upload_from_float_array(@sess, tensor, buf, n)
  end
end
end; end; end
