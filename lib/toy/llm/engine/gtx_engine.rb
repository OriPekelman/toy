# lib/toy/llm/engine/gtx_engine.rb — the toy#160 (DFA-arch T4) GRAPH
# TRANSFORMER: masked self-attention whose mask IS an adjacency, a SMALL
# relation-classification head, and a per-BLOCK credit-assignment policy
# (chain | dfa | frozen) on a `--dfa-cut layer|step` axis.
#
# ── WHY THIS LANE EXISTS ──
#
# The DFA program has one unresolved negative: the transformer LM. But
# every LM run was `scope=all` DFA at a ~50k-vocab output, and F13/F18
# established that DFA's alignment collapses as the OUTPUT DIMENSION
# grows. So that negative confounds two things — ATTENTION, and a huge
# head — and they have never been separated. This lane removes the
# output-dim half (16 relation classes by default; GLM's head is 17) and
# asks what is left:
#
#     Is ATTENTION itself DFA-hostile, or was it only ever the vocab?
#
# ── THE FORWARD ──
#
#   h_0     = W_in . x                                   [d_model, N]
#   per block:
#     a     = h + MHA(RMSNorm(h))    masked by the ADJACENCY
#     h'    = a + FFN_silu(RMSNorm(a))
#   logits  = W_head . concat(h_L[:, i], h_L[:, j])       [R, P]
#
# The mask is ADDITIVE and comes from the task as a [N, N] constant
# (0 where an edge exists, -30 where it does not; see toy_gtx_task.rb on
# why -30 and not -inf). Attention is therefore structurally confined to
# the graph, exactly like gGLM's global attention with an edge bias.
#
# The head reads a PAIR of nodes — get_rows gathers the two endpoint
# columns, concat along ne0 — so the output dim stays R = TY*TY however
# large the graph gets. That is the property the lane is built on and
# the gate asserts it.
#
# ── THE ERROR LIVES ON PAIRS; THE ACTIVATIONS LIVE ON NODES ──
#
# This is the lane's one genuinely new wiring problem. The other lanes'
# taps and error share a batch axis; here the CE error is [R, P] over
# PAIRS while every tap is [d_model, N] over NODES. The error is routed
# onto nodes through a CONSTANT incidence matrix S [P, N] (1 where a
# node is an endpoint of that pair):
#
#     e_nodes = e^T . S            [R, N]
#     delta_l = B_l . e_nodes      [d_model, N]
#     sur_l   = sum(tap_l (*) delta_l)
#
# which is toy#153's structure-aware route in its simplest form: credit
# reaches a node because it PARTICIPATED, not because it was labelled.
# Note the consequence, and it is honest rather than incidental —
# ATTRIBUTE nodes are endpoints of no pair, so their columns receive
# zero direct error. The weights are shared across nodes, so they still
# learn from the entity columns; a `structure` route that spreads
# e_nodes along the adjacency is the obvious next axis if that turns out
# to matter (toy#153 has the precedent).
#
# ── THE TWO CUTS ──
#
#   CUT_LAYER — detach the BLOCK's input and tap its OUTPUT once. BP is
#     intact INSIDE the block, attention included: the block still
#     learns its own attention pattern from the true local gradient.
#   CUT_STEP  — additionally detach the attention PROBABILITIES, so no
#     gradient crosses the token-mixing at all, and give Q and K their
#     own random-feedback taps so they still learn *something*. This is
#     the direct analogue of toy#155/#157's per-step cut: there the cut
#     was across TIME, here it is across the MIXING axis. Without the
#     Q/K taps this arm would be "attention frozen", which is a
#     different and much less interesting claim.
#
# Spinel hygiene (landmine #16): plain class, no-arg ctor, no Struct,
# typed-empty array seeds, while loops, no #{} interpolation.

require_relative "../../models/transformer"
require_relative "../../train/dfa_b"

module Toy; module LLM; module Engine
class GtxEngine
  POLICY_CHAIN  = 0
  POLICY_DFA    = 1
  POLICY_FROZEN = 2

  CUT_LAYER = 0
  CUT_STEP  = 1

  attr_accessor :sess,
                :gx_d_in, :gx_d_model, :gx_heads, :gx_d_head, :gx_d_ff,
                :gx_blocks, :gx_nodes, :gx_pairs, :gx_classes, :gx_cut,
                :ft_weights, :ft_m, :ft_v, :ft_din, :ft_dout, :ft_names,
                :ft_layer,
                :t_x, :t_mask, :t_idx_a, :t_idx_b, :t_inc,
                :t_labels, :t_hp, :t_logits, :t_loss,
                :gx_b_handles, :gx_b_seeds, :gx_b_douts,
                :gx_b_dist, :gx_b_scale, :gx_b_sigma,
                :gx_dfa_wired, :gx_frozen_count, :gx_taps,
                :gx_graph_nodes, :gx_graph_bytes,
                :gx_retro_classes, :gx_adapters, :gx_adapter_rank,
                :gx_retrofit, :gx_freeze_backbone, :gx_adapter_policy,
                :gx_backbone_first, :gx_backbone_count, :gx_active_classes

  def initialize
    @sess       = TinyNN.tnn_null_ptr
    @gx_d_in    = 0
    @gx_d_model = 0
    @gx_heads   = 0
    @gx_d_head  = 0
    @gx_d_ff    = 0
    @gx_blocks  = 0
    @gx_nodes   = 0
    @gx_pairs   = 0
    @gx_classes = 0
    @gx_cut     = CUT_LAYER
    @ft_weights = [TinyNN.tnn_null_ptr]; @ft_weights.pop
    @ft_m       = [TinyNN.tnn_null_ptr]; @ft_m.pop
    @ft_v       = [TinyNN.tnn_null_ptr]; @ft_v.pop
    @ft_din     = [0]; @ft_din.pop
    @ft_dout    = [0]; @ft_dout.pop
    @ft_names   = [""]; @ft_names.pop
    @ft_layer   = [0]; @ft_layer.pop
    @t_x      = TinyNN.tnn_null_ptr
    @t_mask   = TinyNN.tnn_null_ptr
    @t_idx_a  = TinyNN.tnn_null_ptr
    @t_idx_b  = TinyNN.tnn_null_ptr
    @t_inc    = TinyNN.tnn_null_ptr
    @t_labels = TinyNN.tnn_null_ptr
    @t_hp     = TinyNN.tnn_null_ptr
    @t_logits = TinyNN.tnn_null_ptr
    @t_loss   = TinyNN.tnn_null_ptr
    @gx_b_handles = [TinyNN.tnn_null_ptr]; @gx_b_handles.pop
    @gx_b_seeds   = [0]; @gx_b_seeds.pop
    @gx_b_douts   = [0]; @gx_b_douts.pop
    @gx_b_dist  = 0
    @gx_b_scale = 0
    @gx_b_sigma = 0.0
    @gx_dfa_wired    = 0
    @gx_frozen_count = 0
    @gx_taps         = 0
    @gx_graph_nodes  = 0
    @gx_graph_bytes  = 0
    @gx_retro_classes   = 0
    @gx_adapters        = 0
    @gx_adapter_rank    = 0
    @gx_retrofit        = 0
    @gx_freeze_backbone = 0
    @gx_adapter_policy  = POLICY_CHAIN
    @gx_backbone_first  = 0
    @gx_backbone_count  = 0
    @gx_active_classes  = 0
  end

  def realize_for_random_init(d_in, d_model, heads, d_ff, n_blocks,
                              n_nodes, n_pairs, n_classes, seed, init_scale,
                              retro_classes, adapters, adapter_rank)
    @gx_d_in    = d_in
    @gx_d_model = d_model
    @gx_heads   = heads
    @gx_d_head  = d_model / heads
    @gx_d_ff    = d_ff
    @gx_blocks  = n_blocks
    @gx_nodes   = n_nodes
    @gx_pairs   = n_pairs
    @gx_classes = n_classes
    @gx_retro_classes = retro_classes
    @gx_adapters      = adapters
    @gx_adapter_rank  = adapter_rank

    @sess = TinyNN.tnn_session_new(0)
    cap = n_blocks * 400 + 262144
    TinyNN.tnn_session_set_graph_capacity(@sess, cap)

    # Input projection is layer -1's business but belongs to no block;
    # it is parked at block index 0 so a `dfa` policy on block 0 also
    # detaches ITS input, which is what "the policy applies from here
    # down" has to mean.
    add_w(d_model, d_in, "in.proj", 0)
    bi = 0
    while bi < n_blocks
      p = "b" + bi.to_s + "."
      h = 0
      while h < heads
        add_w(@gx_d_head, d_model, p + "q" + h.to_s, bi)
        add_w(@gx_d_head, d_model, p + "k" + h.to_s, bi)
        add_w(@gx_d_head, d_model, p + "v" + h.to_s, bi)
        h = h + 1
      end
      add_w(d_model, d_model, p + "o",    bi)
      add_w(d_ff,    d_model, p + "up",   bi)
      add_w(d_model, d_ff,    p + "down", bi)
      add_w(1, d_model, p + "ln1", bi)
      add_w(1, d_model, p + "ln2", bi)
      bi = bi + 1
    end
    # The head reads TWO node columns, so its input is 2 * d_model. The
    # output dim is R and stays R however large the graph gets — the
    # property this whole lane is built to exploit.
    add_w(n_classes, 2 * d_model, "head", n_blocks)
    # Everything allocated so far is the BACKBONE (+ its pretrain head):
    # what a retrofit freezes and reuses. Recorded as a span so the
    # freeze, the checkpoint and the "did it move" signature all agree on
    # what the word means.
    @gx_backbone_count = @ft_weights.length

    # toy#161 — the RETROFIT capacity, allocated ALWAYS (persistent
    # inputs must exist before finalize_weights, toy#133) and used only
    # in retrofit mode. A pretrain run is byte-identical with them
    # present because nothing consumes them.
    #
    # The adapter stack sits at the PAIR site, on concat(h_a, h_b), and
    # that placement is forced: the retrofit label is a MODULAR SUM, so
    # it needs an interaction between the two endpoints, and the pair is
    # the only place that interaction exists. Node-level adapters inside
    # the backbone could not express it at all (see toy_gtx_task.rb).
    #
    # Each layer is a residual bottleneck  z += W_up . silu(W_down . z),
    # with W_up initialised to ZERO so the retrofit STARTS as exactly the
    # pretrained function. That makes `--adapter-policy frozen` precisely
    # "the frozen backbone plus a retrained head" — the control the bar
    # is stated against.
    @gx_backbone_first = @ft_weights.length
    ai = 0
    while ai < @gx_adapters
      add_w(@gx_adapter_rank, 2 * d_model, "ad" + ai.to_s + ".down", n_blocks + 1)
      add_w(2 * d_model, @gx_adapter_rank, "ad" + ai.to_s + ".up",   n_blocks + 1)
      ai = ai + 1
    end
    add_w(@gx_retro_classes, 2 * d_model, "retro_head", n_blocks + 2)

    wi = 0
    while wi < @ft_weights.length
      TinyNN.tnn_set_param(@ft_weights[wi])
      wi = wi + 1
    end

    # Persistent inputs, ALLOCATED BEFORE finalize_weights (toy#133): a
    # compute-context input allocated later reads zeros in SILENCE.
    @t_x    = TinyNN.tnn_input_2d_f32_persistent(@sess, n_nodes, d_in)
    TinyNN.tnn_tensor_set_name(@t_x, "x")
    @t_mask = TinyNN.tnn_input_2d_f32_persistent(@sess, n_nodes, n_nodes)
    TinyNN.tnn_tensor_set_name(@t_mask, "adj_mask")
    @t_inc  = TinyNN.tnn_input_2d_f32_persistent(@sess, n_nodes, n_pairs)
    TinyNN.tnn_tensor_set_name(@t_inc, "pair_incidence")
    @t_idx_a = TinyNN.tnn_input_1d_i32_persistent(@sess, n_pairs)
    @t_idx_b = TinyNN.tnn_input_1d_i32_persistent(@sess, n_pairs)

    TinyNN.tnn_finalize_weights(@sess)
    upload_random_init!(seed, init_scale)
    nil
  end

  # toy#161 — RETROFIT. Same backbone graph, then:
  #   * the backbone output is DETACHED when frozen, so no gradient enters
  #     it at all (freezing without detaching would still BUILD the
  #     backward — the cost claim has to be structural, not bookkeeping);
  #   * a residual bottleneck ADAPTER STACK on the pair representation;
  #   * a fresh SMALL head for the retrofit label.
  # `adapter_policy` is chain | dfa | frozen, exactly the lane vocabulary.
  def build_retrofit_step(policy, cut, b_seed, b_dist, b_scale, b_sigma,
                          adapter_policy, freeze_backbone)
    @gx_retrofit        = 1
    @gx_adapter_policy  = adapter_policy
    @gx_freeze_backbone = freeze_backbone
    build_training_step(policy, cut, b_seed, b_dist, b_scale, b_sigma)
  end

  def build_training_step(policy, cut, b_seed, b_dist, b_scale, b_sigma)
    @gx_cut     = cut
    @gx_b_dist  = b_dist
    @gx_b_scale = b_scale
    @gx_b_sigma = b_sigma

    TinyNN.tnn_reset_for_rebuild(@sess)

    @gx_b_handles = [TinyNN.tnn_null_ptr]; @gx_b_handles.pop
    @gx_b_seeds   = [0]; @gx_b_seeds.pop
    @gx_b_douts   = [0]; @gx_b_douts.pop
    @gx_frozen_count = 0
    taps  = [TinyNN.tnn_null_ptr]; taps.pop
    tapd  = [0]; tapd.pop
    any_dfa = false

    t_h = TinyNN.tnn_matmul(@sess, @ft_weights[0], @t_x)   # [d_model, N]

    scale = 1.0 / Math.sqrt(@gx_d_head.to_f)
    bj = 0
    while bj < @gx_blocks
      mode = POLICY_CHAIN
      if bj < policy.length
        mode = policy[bj]
      end
      is_dfa = mode == POLICY_DFA
      if is_dfa
        any_dfa = true
      end
      if mode == POLICY_FROZEN
        @gx_frozen_count = @gx_frozen_count + 1
      end

      blk_in = t_h
      if is_dfa
        blk_in = TinyNN.tnn_detach(@sess, blk_in)
      end

      base = 1 + bj * (3 * @gx_heads + 5)
      t_n1 = TinyNN.tnn_rms_norm(@sess, blk_in, @ft_weights[base + 3 * @gx_heads + 3], 1.0e-5)

      t_heads = [TinyNN.tnn_null_ptr]; t_heads.pop
      hh = 0
      while hh < @gx_heads
        t_q = TinyNN.tnn_matmul(@sess, @ft_weights[base + hh * 3],     t_n1)  # [d_head, N]
        t_k = TinyNN.tnn_matmul(@sess, @ft_weights[base + hh * 3 + 1], t_n1)
        t_v = TinyNN.tnn_matmul(@sess, @ft_weights[base + hh * 3 + 2], t_n1)
        # scores ne=[N_key, N_query]; the adjacency mask is ADDITIVE and
        # applied before the softmax, so attention cannot leave the graph.
        t_sc = TinyNN.tnn_softmax(@sess,
                 TinyNN.tnn_add(@sess,
                   TinyNN.tnn_scale(@sess, TinyNN.tnn_matmul(@sess, t_k, t_q), scale),
                   @t_mask))
        if is_dfa && @gx_cut == CUT_STEP
          # THE MIXING CUT: no gradient crosses the attention pattern.
          # Q and K then get their own random-feedback taps below, or
          # this arm would just be "attention frozen".
          t_sc = TinyNN.tnn_detach(@sess, t_sc)
          TinyNN.tnn_set_output(t_q)
          taps.push(t_q); tapd.push(@gx_d_head)
          TinyNN.tnn_set_output(t_k)
          taps.push(t_k); tapd.push(@gx_d_head)
        end
        t_heads.push(TinyNN.tnn_matmul(@sess, TinyNN.tnn_transpose(@sess, t_v), t_sc))
        hh = hh + 1
      end
      t_cat = t_heads[0]
      hc = 1
      while hc < @gx_heads
        t_cat = TinyNN.tnn_concat(@sess, t_cat, t_heads[hc], 0)
        hc = hc + 1
      end
      t_a = TinyNN.tnn_add(@sess, blk_in,
              TinyNN.tnn_matmul(@sess, @ft_weights[base + 3 * @gx_heads], t_cat))

      t_n2 = TinyNN.tnn_rms_norm(@sess, t_a, @ft_weights[base + 3 * @gx_heads + 4], 1.0e-5)
      t_up = TinyNN.tnn_silu(@sess,
               TinyNN.tnn_matmul(@sess, @ft_weights[base + 3 * @gx_heads + 1], t_n2))
      t_h  = TinyNN.tnn_add(@sess, t_a,
               TinyNN.tnn_matmul(@sess, @ft_weights[base + 3 * @gx_heads + 2], t_up))

      if is_dfa
        TinyNN.tnn_set_output(t_h)
        taps.push(t_h); tapd.push(@gx_d_model)
      end
      bj = bj + 1
    end

    # THE HEAD: gather the two endpoint columns and concat along ne0, so
    # the output dim is R whatever N is.
    if @gx_retrofit == 1 && @gx_freeze_backbone == 1
      # THE FREEZE IS A DETACH, not just a skipped optimizer step. Under
      # a skipped step the backward through the whole backbone would
      # still be BUILT and computed — so "no backward through the frozen
      # backbone" would be a claim about bookkeeping rather than about
      # the graph. Detaching makes it true of the graph, for BOTH arms,
      # which is exactly what the measured node/byte counts then show.
      t_h = TinyNN.tnn_detach(@sess, t_h)
    end
    t_ha = TinyNN.tnn_get_rows(@sess, t_h, @t_idx_a)       # [d_model, P]
    t_hb = TinyNN.tnn_get_rows(@sess, t_h, @t_idx_b)
    t_pair = TinyNN.tnn_concat(@sess, t_ha, t_hb, 0)       # [2*d_model, P]

    n_out = @gx_classes
    if @gx_retrofit == 1
      n_out = @gx_retro_classes
      ad_dfa = @gx_adapter_policy == POLICY_DFA
      aj = 0
      while aj < @gx_adapters
        w_dn = @ft_weights[@gx_backbone_first + aj * 2]
        w_up = @ft_weights[@gx_backbone_first + aj * 2 + 1]
        a_in = t_pair
        if ad_dfa
          # A dfa adapter layer detaches its INPUT: its credit comes from
          # the random projection of the output error, never from the
          # layer above it.
          a_in = TinyNN.tnn_detach(@sess, a_in)
        end
        t_pair = TinyNN.tnn_add(@sess, a_in,
                   TinyNN.tnn_matmul(@sess, w_up,
                     TinyNN.tnn_silu(@sess, TinyNN.tnn_matmul(@sess, w_dn, a_in))))
        if ad_dfa
          TinyNN.tnn_set_output(t_pair)
          taps.push(t_pair); tapd.push(2 * @gx_d_model)
          any_dfa = true
        end
        aj = aj + 1
      end
      @t_logits = TinyNN.tnn_matmul(@sess,
                    @ft_weights[@ft_weights.length - 1], t_pair)
    else
      @t_logits = TinyNN.tnn_matmul(@sess,
                    @ft_weights[@gx_backbone_count - 1], t_pair)
    end
    TinyNN.tnn_set_output(@t_logits)

    @t_labels = TinyNN.tnn_input_2d_f32(@sess, @gx_pairs, n_out)
    @t_hp     = TinyNN.tnn_input_1d_f32(@sess, 7)
    @t_loss   = TinyNN.tnn_cross_entropy_loss(@sess, @t_logits, @t_labels)
    TinyNN.tnn_set_output(@t_loss)

    # The ACTIVE output dim: the retrofit head is TY-wide, not TY*TY.
    # refresh_b! sizes its uploads from this — using @gx_classes there
    # wrote a 16-wide B into a 4-wide tensor and ggml caught it as an
    # out-of-bounds write.
    active_classes = @gx_retrofit == 1 ? @gx_retro_classes : @gx_classes
    @gx_active_classes = active_classes
    @gx_taps = taps.length
    if any_dfa && taps.length > 0
      t_p = TinyNN.tnn_softmax(@sess, @t_logits)
      e_det = TinyNN.tnn_detach(@sess,
                TinyNN.tnn_scale(@sess,
                  TinyNN.tnn_sub(@sess, t_p, @t_labels), 1.0 / @gx_pairs.to_f))
      # PAIR error -> NODE error through the constant incidence matrix.
      # e^T is [P, R]; S is [P, N]; the product is [R, N].
      e_nodes = TinyNN.tnn_matmul(@sess,
                  TinyNN.tnn_cont_2d(@sess, TinyNN.tnn_transpose(@sess, e_det),
                                     @gx_pairs, active_classes),
                  @t_inc)
      t_sur = TinyNN.tnn_null_ptr
      started = false
      ti = 0
      while ti < taps.length
        dout = tapd[ti]
        on_pairs = dout == 2 * @gx_d_model && @gx_retrofit == 1
        t_b = TinyNN.tnn_input_2d_f32(@sess, dout, active_classes)  # ne=[C, dout]
        TinyNN.tnn_set_output(t_b)
        @gx_b_handles.push(t_b)
        @gx_b_seeds.push(b_seed + 77 + ti)
        @gx_b_douts.push(dout)
        # An ADAPTER tap lives on the PAIR axis, so its error needs no
        # incidence routing at all — the error is already per-pair. Only
        # the backbone's node-axis taps go through S.
        if on_pairs
          t_delta = TinyNN.tnn_matmul(@sess, t_b, e_det)           # [dout, P]
          t_term = TinyNN.tnn_sum_rows(@sess,
                     TinyNN.tnn_reshape_2d(@sess,
                       TinyNN.tnn_mul(@sess, taps[ti], t_delta),
                       dout * @gx_pairs, 1))
        else
          t_delta = TinyNN.tnn_matmul(@sess, t_b, e_nodes)         # [dout, N]
          t_term = TinyNN.tnn_sum_rows(@sess,
                     TinyNN.tnn_reshape_2d(@sess,
                       TinyNN.tnn_mul(@sess, taps[ti], t_delta),
                       dout * @gx_nodes, 1))
        end
        if started
          t_sur = TinyNN.tnn_add(@sess, t_sur, t_term)
        else
          t_sur = t_term
          started = true
        end
        ti = ti + 1
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
      if lyr < @gx_blocks && lyr < policy.length
        mode = policy[lyr]
      end
      step_it = true
      if @gx_retrofit == 1
        # RETROFIT: what trains is the ADDED capacity plus the new head.
        # The backbone (and its pretrain head) is out; the adapters obey
        # --adapter-policy; the retrofit head ALWAYS trains, because it
        # is the readout for a task the backbone has never seen, not
        # "added capacity" whose credit rule is under test.
        if wk < @gx_backbone_count
          step_it = @gx_freeze_backbone == 0
        elsif wk < @ft_weights.length - 1
          step_it = @gx_adapter_policy != POLICY_FROZEN
        else
          step_it = true
        end
      elsif lyr < @gx_blocks && mode == POLICY_FROZEN
        # No optimizer step: the block stays at init.
        step_it = false
      elsif wk >= @gx_backbone_count
        # Pretrain never touches the retrofit capacity — and it must not,
        # or a "pretrained backbone" would quietly carry trained adapters
        # into a run whose whole point is that they start at identity.
        step_it = false
      end
      if step_it
        tg = TinyNN.tnn_tensor_grad(@sess, @ft_weights[wk])
        to = TinyNN.tnn_opt_step_adamw(@sess, @ft_weights[wk], tg,
                                        @ft_m[wk], @ft_v[wk], @t_hp)
        TinyNN.tnn_extend_backward_graph(@sess, to)
      end
      wk = wk + 1
    end

    @gx_dfa_wired = dfa_block_count(policy)
    puts "gtx: " + (@gx_retrofit == 1 ? "retrofit " : "") +
         "blocks=" + @gx_blocks.to_s +
         " nodes=" + @gx_nodes.to_s +
         " pairs=" + @gx_pairs.to_s +
         " classes=" + @gx_classes.to_s +
         " cut=" + (@gx_cut == CUT_STEP ? "step" : "layer") +
         " policy_len=" + policy.length.to_s +
         " dfa_wired=" + @gx_dfa_wired.to_s +
         " frozen=" + @gx_frozen_count.to_s +
         " taps=" + @gx_taps.to_s +
         (@gx_retrofit == 1 ?
           (" adapters=" + @gx_adapters.to_s +
            " adapter_policy=" + (@gx_adapter_policy == POLICY_DFA ? "dfa" :
                                  (@gx_adapter_policy == POLICY_FROZEN ? "frozen" : "chain")) +
            " freeze_backbone=" + @gx_freeze_backbone.to_s +
            " retro_classes=" + @gx_retro_classes.to_s) : "")

    TinyNN.tnn_pin_all_graph_b_nodes(@sess)
    TinyNN.tnn_realize_backward(@sess)
    measure_graph!
    refresh_b!
    [@t_loss, @t_labels, @t_hp]
  end

  # toy#161 — sum of squares over every BACKBONE weight. Emitted before
  # and after a retrofit: --freeze-backbone must leave the two
  # BIT-IDENTICAL, which is the gate's proof that nothing moved, rather
  # than a promise that nothing was stepped.
  # toy#161 — the gradient bytes a FROZEN backbone never materialises.
  # Analytic, and it has to be: the realized-graph node/byte counters do
  # not include the backward extension, so a measured count cannot see
  # this (freezing actually reads +1 node, the detach itself). tao#21's
  # caveat was that a measured graph cannot show a STREAMING win; the
  # mirror caveat applies here — a measured graph cannot show an ABSENT
  # backward either.
  #
  # Read it as what FREEZING buys, not what DFA buys: at this adapter
  # site both arms detach the backbone, so the figure is identical for
  # chain and dfa. DFA's own additional saving is the backward through
  # the ADAPTER STACK only, which is two small matrices per layer.
  def backbone_grad_bytes
    n = 0
    i = 0
    while i < @gx_backbone_count
      n = n + @ft_din[i] * @ft_dout[i]
      i = i + 1
    end
    n * 4
  end

  def adapter_grad_bytes
    n = 0
    i = @gx_backbone_first
    while i < @ft_weights.length - 1
      n = n + @ft_din[i] * @ft_dout[i]
      i = i + 1
    end
    n * 4
  end

  def backbone_sig
    acc = 0.0
    i = 0
    while i < @gx_backbone_count
      n = TinyNN.tnn_tensor_nelements(@ft_weights[i])
      buf = Array.new(n, 0.0)
      TinyNN.tnn_download_to_f64_array(@sess, @ft_weights[i], buf, n)
      k = 0
      while k < n
        acc = acc + buf[k] * buf[k]
        k = k + 1
      end
      i = i + 1
    end
    acc
  end

  def measure_graph!
    @gx_graph_nodes = TinyNN.tnn_graph_n_nodes(@sess)
    total = 0
    i = 0
    while i < @gx_graph_nodes
      total = total + TinyNN.tnn_tensor_nbytes(TinyNN.tnn_graph_node(@sess, i))
      i = i + 1
    end
    @gx_graph_bytes = total
    nil
  end

  def dfa_block_count(policy)
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

  def refresh_b!
    bi = 0
    while bi < @gx_b_handles.length
      dout = @gx_b_douts[bi]
      nb   = dout * @gx_active_classes
      sig  = Toy::Train::DfaB.sigma_for(@gx_b_scale, @gx_active_classes, dout, @gx_b_sigma)
      TinyNN.tnn_upload_from_float_array(@sess, @gx_b_handles[bi],
        Toy::Train::DfaB.fill(nb, @gx_b_seeds[bi], @gx_b_dist, sig), nb)
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
      n  = @ft_din[wi] * @ft_dout[wi]
      nm = @ft_names[wi]
      if nm.length > 3 && nm[0, 2] == "ad" && nm[nm.length - 3, 3] == ".up"
        # toy#161: W_up starts at ZERO so the adapter is the identity and
        # the retrofit begins at exactly the pretrained function. A
        # non-zero start would make even the `frozen` control a different
        # model from the backbone it is supposed to control for.
        #
        # The `nm[0,2] == "ad"` guard is NOT redundant: every block's FFN
        # up-projection is named "b<i>.up" and also ends in ".up", so a
        # suffix-only match zeroed the whole backbone's FFN. It was
        # invisible in the retrofit numbers and caught only by toy#160's
        # byte-gated fixture — which is precisely what that fixture is
        # for.
        upload_const(@ft_weights[wi], n, 0.0)
      elsif nm.length > 3 && (nm[nm.length - 3, 3] == "ln1" || nm[nm.length - 3, 3] == "ln2")
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
