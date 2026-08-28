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
                :gx_bytelm, :t_tokens, :ix_emb, :ix_lmhead, :gx_gnorm, :t_gnorm, :gx_gnorm_built,
                :gx_retrofit, :gx_freeze_backbone, :gx_adapter_policy,
                :gx_backbone_first, :gx_backbone_count, :gx_active_classes,
                :gx_b_rank, :gx_b_adapt, :gx_b_pseed, :gx_p_map, :gx_p_csup,
                :gx_ld_fro_eff, :gx_ld_fro_full, :gx_ld_rank_eff, :gx_ld_dout

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
    @gx_b_sr    = ""
    @gx_dfa_wired    = 0
    @gx_frozen_count = 0
    @gx_taps         = 0
    @gx_graph_nodes  = 0
    @gx_graph_bytes  = 0
    @gx_retro_classes   = 0
    @gx_bytelm  = 0
    @gx_gnorm   = 0
    @t_gnorm    = TinyNN.tnn_null_ptr
    @gx_gnorm_built = 0
    @t_tokens   = TinyNN.tnn_null_ptr
    @ix_emb     = -1
    @ix_lmhead  = -1
    @gx_adapters        = 0
    @gx_adapter_rank    = 0
    @gx_retrofit        = 0
    @gx_freeze_backbone = 0
    @gx_adapter_policy  = POLICY_CHAIN
    @gx_backbone_first  = 0
    @gx_backbone_count  = 0
    @gx_active_classes  = 0
    # toy#172 (E2) — the LDFA low-rank factorisation. 0 means FULL WIDTH,
    # i.e. exactly the pre-E2 path, and every branch below is keyed on
    # that so the default binary is byte-identical.
    @gx_b_rank   = 0
    @gx_b_adapt  = 0
    @gx_b_pseed  = 0
    @gx_p_map    = ""
    @gx_p_csup   = ""
    @gx_p        = [0.0]; @gx_p.pop
    @gx_p_ready  = 0
    @gx_ld_tgt2  = [0.0]; @gx_ld_tgt2.pop
    @gx_ld_fro_eff  = 0.0
    @gx_ld_fro_full = 0.0
    @gx_ld_rank_eff = 0
    @gx_ld_dout     = 0
  end

  def realize_for_random_init(d_in, d_model, heads, d_ff, n_blocks,
                              n_nodes, n_pairs, n_classes, seed, init_scale,
                              retro_classes, adapters, adapter_rank, bytelm,
                              gnorm)
    @gx_d_in    = d_in
    @gx_d_model = d_model
    @gx_heads   = heads
    @gx_d_head  = d_model / heads
    @gx_d_ff    = d_ff
    @gx_blocks  = n_blocks
    @gx_nodes   = n_nodes
    @gx_pairs   = n_pairs
    @gx_classes = n_classes
    @gx_bytelm = bytelm
    # Threaded but INERT on this lane — see the note in train_gtx.rb.
    @gx_gnorm  = gnorm
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
    # toy#170 (P3) — the byte-LM surface. The embedding replaces the
    # continuous node features; the LM head is [n_classes, d_model]
    # because there is no pair concat to widen it. Both are built ONLY
    # under bytelm, so the toy#160 relational path keeps its exact graph.
    #
    # toy#170 (P5) — this width was literally 256 until the nominal-head
    # axis needed it. n_classes IS 256 under the byte-LM default, so the
    # change is byte-null there; `--vocab k` is what moves it, and it
    # moves the embedding, the head AND the DFA feedback matrix B
    # together, because the whole point of the output-dim law is that
    # those three widths are the same number.
    if @gx_bytelm == 1
      @ix_emb = @ft_weights.length
      add_w(n_classes, d_model, "emb", 0)
      @ix_lmhead = @ft_weights.length
      add_w(n_classes, d_model, "lm_head", n_blocks)
    end
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
    if @gx_bytelm == 1
      @t_tokens = TinyNN.tnn_input_1d_i32_persistent(@sess, n_nodes)
      TinyNN.tnn_tensor_set_name(@t_tokens, "tokens")
    end
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
    # toy#172 (E2) — the LDFA scale targets are keyed by TAP INDEX, so a
    # rebuild that changes the tap set must not inherit them. LDFA refuses
    # retrofit outright, which is the only caller that rebuilds; this line
    # is here so the invariant does not depend on that refusal staying.
    @gx_ld_tgt2   = [0.0]; @gx_ld_tgt2.pop
    @gx_frozen_count = 0
    taps  = [TinyNN.tnn_null_ptr]; taps.pop
    tapd  = [0]; tapd.pop
    any_dfa = false

    # bytelm feeds EMBEDDED BYTES; the relational path keeps the feature
    # projection. Note the embedding lands BELOW block 0's detach under a
    # dfa policy — the toy#169 trap — so it gets its own tap below.
    t_h = TinyNN.tnn_matmul(@sess, @ft_weights[0], @t_x)   # [d_model, N]
    t_emb_out = TinyNN.tnn_null_ptr
    if @gx_bytelm == 1
      t_h = TinyNN.tnn_get_rows(@sess, @ft_weights[@ix_emb], @t_tokens)
      t_emb_out = t_h
    end

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
    if @gx_bytelm == 1
      # PER-POSITION readout: position i predicts byte i+1. No pair
      # gather, so the head is [n_classes, d_model] and the error is
      # already per position — the incidence routing below becomes the
      # identity.
      @t_logits = TinyNN.tnn_matmul(@sess, @ft_weights[@ix_lmhead], t_h)
      TinyNN.tnn_set_output(@t_logits)
      # Must match the lm_head's output dim: cross_entropy_loss asserts
      # same-shape, so a literal here is an ABORT under --vocab k rather
      # than a wrong number — but only because ggml checks. Nothing in
      # this file would have caught it.
      @t_labels = TinyNN.tnn_input_2d_f32(@sess, @gx_nodes, @gx_classes)
      @t_hp     = TinyNN.tnn_input_1d_f32(@sess, 7)
      @t_loss   = TinyNN.tnn_cross_entropy_loss(@sess, @t_logits, @t_labels)
      TinyNN.tnn_set_output(@t_loss)
      return build_bytelm_tail!(policy, taps, tapd, any_dfa, t_emb_out,
                                b_seed, b_dist, b_scale, b_sigma)
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

  # toy#170 (P3) — the bytelm tail. Split out because it differs from the
  # relational path in exactly two ways, and both are simplifications:
  #
  #  * THE INCIDENCE ROUTING DROPS OUT. The relational lane routes a
  #    PAIR error onto nodes through the constant S (e_nodes = e^T . S).
  #    Under bytelm the error is ALREADY per position, so that step is
  #    the identity and the pair machinery is not built at all.
  #  * THE TAPS NEED NO FOLDING. gtx taps [d_model, N] per block, and
  #    with N = positions a tap is already aligned column-for-column
  #    with the per-position error. This is the piece that cost toy#169
  #    a fold and an ordering hazard; on attention it is free.
  #
  # THE EMBEDDING TAP is built here from the start rather than
  # discovered later: `emb` sits BELOW block 0's `detach`, so under a
  # dfa policy the chain path gives it nothing. toy#169 paid two rounds
  # and two retractions for that. Its output is [d_model, N] — already
  # the tap family's shape and column order — so it just joins the list.
  def build_bytelm_tail!(policy, taps, tapd, any_dfa, t_emb_out,
                         b_seed, b_dist, b_scale, b_sigma)
    # The error dim the feedback matrices are sized and scaled from. Under
    # bytelm there is no retrofit head, so it is just the class count —
    # which is 256 by default and `--vocab k` otherwise.
    @gx_active_classes = @gx_classes
    if any_dfa && policy.length > 0 && policy[0] == POLICY_DFA &&
       t_emb_out != TinyNN.tnn_null_ptr
      TinyNN.tnn_set_output(t_emb_out)
      taps.push(t_emb_out)
      tapd.push(@gx_d_model)
    end
    @gx_taps = taps.length
    if any_dfa && taps.length > 0
      t_p = TinyNN.tnn_softmax(@sess, @t_logits)
      e_det = TinyNN.tnn_detach(@sess,
                TinyNN.tnn_scale(@sess,
                  TinyNN.tnn_sub(@sess, t_p, @t_labels), 1.0 / @gx_nodes.to_f))
      t_sur = TinyNN.tnn_null_ptr
      started = false
      ti = 0
      while ti < taps.length
        dout = tapd[ti]
        # B is [dout, error_dim]. refresh_b! sizes and scales its uploads
        # from @gx_active_classes, so this MUST be that same number — a
        # literal here would upload a differently-shaped B in silence.
        t_b = TinyNN.tnn_input_2d_f32(@sess, dout, @gx_active_classes)
        TinyNN.tnn_set_output(t_b)
        @gx_b_handles.push(t_b)
        @gx_b_seeds.push(b_seed + 77 + ti)
        @gx_b_douts.push(dout)
        t_delta = TinyNN.tnn_matmul(@sess, t_b, e_det)          # [dout, N]
        t_term = TinyNN.tnn_sum_rows(@sess,
                   TinyNN.tnn_reshape_2d(@sess,
                     TinyNN.tnn_mul(@sess, taps[ti], t_delta),
                     dout * @gx_nodes, 1))
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
    TinyNN.tnn_build_backward(@sess)

    wk = 0
    while wk < @ft_weights.length
      lyr = @ft_layer[wk]
      mode = POLICY_CHAIN
      if lyr < @gx_blocks && lyr < policy.length
        mode = policy[lyr]
      end
      step_it = true
      if lyr < @gx_blocks && mode == POLICY_FROZEN
        step_it = false
      elsif wk >= @gx_backbone_count && wk != @ix_emb && wk != @ix_lmhead
        # retrofit capacity is never touched by a bytelm run
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

    # b_dim is the width of the DFA feedback matrix B, and it is printed
    # SEPARATELY from vocab even though they are the same number today.
    # The output-dim law is a claim about B; if a later change ever lets
    # the head and B disagree, this line is where it shows up instead of
    # being absorbed into a plausible-looking result.
    puts "gtx: bytelm blocks=" + @gx_blocks.to_s +
         " nodes=" + @gx_nodes.to_s +
         " vocab=" + @gx_classes.to_s +
         " b_dim=" + @gx_active_classes.to_s +
         " attn=causal readout=per_position" +
         " routing=position_t" +
         " cut=" + (@gx_cut == CUT_STEP ? "step" : "layer") +
         " dfa_wired=" + dfa_block_count(policy).to_s +
         " frozen=" + @gx_frozen_count.to_s +
         " taps=" + @gx_taps.to_s +
         " emb_tapped=" + ((any_dfa && policy.length > 0 && policy[0] == POLICY_DFA) ? "1" : "0")

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

  # toy#164 — write the BACKBONE (everything up to and including the
  # pretrain head, i.e. the span toy#161 already calls
  # @gx_backbone_count) as a GGUF, so one pretrain can serve many
  # retrofit arms. The retrofit capacity is deliberately NOT written: an
  # adapter that travelled with a "pretrained backbone" would silently
  # start a retrofit somewhere other than the pretrained function, which
  # is the one property the whole comparison rests on.
  #
  # The shape metadata is not decoration — load_backbone_ckpt refuses a
  # mismatch on it. A backbone loaded under a different width or a
  # different task shape would produce confident garbage.
  def write_backbone_ckpt(path, run_id, step)
    ctxw = TinyNN.tnn_gguf_w_init
    if ctxw == nil || ctxw == TinyNN.tnn_null_ptr
      return -1
    end
    TinyNN.tnn_gguf_w_set_str(ctxw, "general.architecture", "toy-gtx")
    TinyNN.tnn_gguf_w_set_str(ctxw, "general.name", "gtx-backbone")
    TinyNN.tnn_gguf_w_set_str(ctxw, "general.run_id", run_id)
    TinyNN.tnn_gguf_w_set_u32(ctxw, "general.step", step)
    TinyNN.tnn_gguf_w_set_str(ctxw, "toy.checkpoint_format", "toy-gtx/v1")
    TinyNN.tnn_gguf_w_set_u32(ctxw, "toy.gtx.d_model", @gx_d_model)
    TinyNN.tnn_gguf_w_set_u32(ctxw, "toy.gtx.heads",   @gx_heads)
    TinyNN.tnn_gguf_w_set_u32(ctxw, "toy.gtx.d_ff",    @gx_d_ff)
    TinyNN.tnn_gguf_w_set_u32(ctxw, "toy.gtx.blocks",  @gx_blocks)
    TinyNN.tnn_gguf_w_set_u32(ctxw, "toy.gtx.d_in",    @gx_d_in)
    TinyNN.tnn_gguf_w_set_u32(ctxw, "toy.gtx.nodes",   @gx_nodes)
    TinyNN.tnn_gguf_w_set_u32(ctxw, "toy.gtx.classes", @gx_classes)
    gi = 0
    while gi < @gx_backbone_count
      TinyNN.tnn_gguf_w_add_tensor(ctxw, @ft_weights[gi])
      gi = gi + 1
    end
    rc = TinyNN.tnn_gguf_w_finalize(ctxw, path)
    TinyNN.tnn_gguf_w_free(ctxw)
    rc
  end

  # toy#164 — overwrite every backbone tensor BY NAME from a checkpoint.
  # Returns 0 on success; on any mismatch it prints both sides and
  # returns non-zero, because a silently-wrong backbone is the worst
  # possible failure here: every downstream number would look healthy.
  def load_backbone_ckpt(path)
    gg = TinyNN.tnn_gguf_load(path)
    if gg == nil || gg == TinyNN.tnn_null_ptr
      puts "toy-train-gtx: cannot open checkpoint " + path
      return 1
    end
    fmt = TinyNN.tnn_gguf_get_str(gg, "toy.checkpoint_format")
    if fmt != "toy-gtx/v1"
      puts "toy-train-gtx: " + path + " is not a toy-gtx/v1 checkpoint" +
           " (toy.checkpoint_format=" + fmt.to_s + ")"
      return 1
    end
    ck_dm = TinyNN.tnn_gguf_get_u32(gg, "toy.gtx.d_model")
    ck_hd = TinyNN.tnn_gguf_get_u32(gg, "toy.gtx.heads")
    ck_ff = TinyNN.tnn_gguf_get_u32(gg, "toy.gtx.d_ff")
    ck_bl = TinyNN.tnn_gguf_get_u32(gg, "toy.gtx.blocks")
    ck_di = TinyNN.tnn_gguf_get_u32(gg, "toy.gtx.d_in")
    ck_nd = TinyNN.tnn_gguf_get_u32(gg, "toy.gtx.nodes")
    ck_cl = TinyNN.tnn_gguf_get_u32(gg, "toy.gtx.classes")
    if ck_dm != @gx_d_model || ck_hd != @gx_heads || ck_ff != @gx_d_ff ||
       ck_bl != @gx_blocks || ck_di != @gx_d_in || ck_nd != @gx_nodes ||
       ck_cl != @gx_classes
      puts "toy-train-gtx: checkpoint shape mismatch — ckpt d_model=" + ck_dm.to_s +
           " heads=" + ck_hd.to_s + " d_ff=" + ck_ff.to_s + " blocks=" + ck_bl.to_s +
           " d_in=" + ck_di.to_s + " nodes=" + ck_nd.to_s + " classes=" + ck_cl.to_s +
           " vs instrument d_model=" + @gx_d_model.to_s + " heads=" + @gx_heads.to_s +
           " d_ff=" + @gx_d_ff.to_s + " blocks=" + @gx_blocks.to_s +
           " d_in=" + @gx_d_in.to_s + " nodes=" + @gx_nodes.to_s +
           " classes=" + @gx_classes.to_s +
           " (pass the matching --d-model/--heads/--d-ff/--layers/--features/--entities/--types)"
      return 1
    end
    gi = 0
    while gi < @gx_backbone_count
      nm = TinyNN.tnn_tensor_name(@ft_weights[gi])
      idx = TinyNN.tnn_gguf_find_index(gg, nm)
      if idx < 0
        puts "toy-train-gtx: checkpoint missing tensor " + nm
        return 1
      end
      nel = TinyNN.tnn_tensor_nelements(@ft_weights[gi])
      buf = Array.new(nel, 0.0)
      TinyNN.tnn_gguf_read_f32_to_doubles(gg, idx, buf, nel)
      TinyNN.tnn_upload_from_float_array(@sess, @ft_weights[gi], buf, nel)
      gi = gi + 1
    end
    0
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
    # toy#172 (E2) — the LDFA fork. Rank 0 IS full width and falls through
    # to the original body untouched, which is what makes
    # `--dfa-feedback-rank full` a byte identity rather than a promise.
    if @gx_b_rank > 0
      refresh_b_lowrank!
      return nil
    end
    bi = 0
    @gx_b_sr = ""
    while bi < @gx_b_handles.length
      dout = @gx_b_douts[bi]
      nb   = dout * @gx_active_classes
      sig  = Toy::Train::DfaB.sigma_for(@gx_b_scale, @gx_active_classes, dout, @gx_b_sigma)
      vals = Toy::Train::DfaB.fill(nb, @gx_b_seeds[bi], @gx_b_dist, sig)
      TinyNN.tnn_upload_from_float_array(@sess, @gx_b_handles[bi], vals, nb)
      # toy#172 (E1) — B's stable rank, computed HERE because B is
      # generated on the host and never has to be read back. It is the
      # instrument's flat CONTROL: B is a fixed i.i.d. draw that does not
      # move during training, so this should not vary across rungs.
      #
      # ||B||_F^2 is exact; ||B||_2 is bounded below by the largest row
      # norm, which is what is used — so this is an UPPER bound on the
      # stable rank, and it is labelled as such rather than presented as
      # the spectral norm.
      fro2 = 0.0
      k = 0
      while k < nb
        fro2 = fro2 + vals[k] * vals[k]
        k = k + 1
      end
      rmax = 0.0
      r = 0
      while r < dout
        rn = 0.0
        c = 0
        base = r * @gx_active_classes
        while c < @gx_active_classes
          rn = rn + vals[base + c] * vals[base + c]
          c = c + 1
        end
        if rn > rmax
          rmax = rn
        end
        r = r + 1
      end
      sr = rmax > 0.0 ? (fro2 / rmax) : 0.0
      @gx_b_sr = @gx_b_sr + (bi > 0 ? " " : "") +
                 "B" + bi.to_s + "=[" + dout.to_s + "x" + @gx_active_classes.to_s +
                 "]sr_ub=" + sr.to_s
      bi = bi + 1
    end
    nil
  end

  # The control line, assembled at refresh time. Named `_ub` throughout
  # because it is an upper bound (largest row norm <= spectral norm), not
  # the stable rank itself — a number that quietly overstates itself is
  # the failure this program keeps paying for.
  def b_stable_rank_report
    @gx_b_sr.length > 0 ? @gx_b_sr : "none (no DFA taps wired)"
  end

  # ---- toy#172 / E2: ADAPTIVE LOW-RANK FEEDBACK (LDFA) ----
  #
  # LDFA (Hanut & Kadmon 2026, arXiv:2502.20580) factorises the feedback
  # matrix as B_eff = Q . P with Q [dout x r] and P [r x V]. P is either
  # a fixed random draw or ADAPTED to the error stream by Oja's rule; the
  # fixed-vs-adaptive contrast at each width IS the hypothesis.
  #
  # ── NO NEW GRAPH NODE, FOR THE THIRD TIME ──
  #
  # Same argument as nDFA's: B is generated HOST-SIDE by refresh_b! and
  # uploaded, so a factorisation is a change to what gets uploaded and
  # nothing else. The ||g|| instrument on this lane realized as ZEROS
  # through three separate readout paths and had to be removed; the
  # E1 instrument and nDFA both avoided a new node and both worked.
  #
  # ── THE SCALE CONTROL, WHICH IS WHAT MAKES ANY OF THIS MEAN ANYTHING ──
  #
  # Q.P has a completely different Frobenius norm from the full-width B it
  # replaces (it is a product of two draws, and its scale depends on r).
  # Left alone, "low rank hurts" would be indistinguishable from "the
  # updates got smaller" — a global gain on B is an LR change wearing a
  # rank costume, exactly the confound nDFA's `preserve` gain was built to
  # remove. So the full-width draw this arm REPLACES is regenerated from
  # the SAME seed and B_eff is rescaled to its realised ||B||_F. Both
  # norms are emitted, so a reader can verify the arms differ in RANK and
  # not in SCALE rather than take it on faith.
  #
  # ── P IS ORTHONORMALISED AT INIT IN *BOTH* MODES ──
  #
  # A deliberate deviation from the spec's "fixed random draw": Oja's rule
  # REQUIRES orthonormal rows, so an un-orthonormalised `none` would
  # differ from `oja` in two ways at once and the contrast would not be a
  # clean B-test. At r << V a Gaussian [r x V] draw is already
  # near-orthogonal, so this changes the fixed arm very little — but
  # "very little" is not "nothing", and the point of the arm is that the
  # ONLY difference is the adaptivity. Stated in provenance as
  # `p_ortho=init`.
  #
  # ── rank_eff IS NOT r ──
  #
  # rank(Q.P) <= min(dout, r), and dout is d_model = 128 on this fixture.
  # So r = 256 does NOT give a rank-256 feedback matrix — it gives the
  # SAME rank as full width, with the row space confined to P's 256-dim
  # span. That is still a real intervention under `oja` (the span is the
  # error's dominant subspace rather than a random one) but it is NOT a
  # rank reduction, and reading "r=256" as "rank 256" would misstate the
  # whole ladder. `rank_eff` and `dout` are both on the provenance line.
  def refresh_b_lowrank!
    v = @gx_active_classes
    r = @gx_b_rank
    if @gx_p_ready == 0
      init_p!(v, r)
      @gx_p_ready = 1
    end
    @gx_b_sr = ""
    eff2  = 0.0
    full2 = 0.0
    rke   = 0
    dmax  = 0
    bi = 0
    while bi < @gx_b_handles.length
      dout = @gx_b_douts[bi]
      nb   = dout * v
      # THE SCALE TARGET: the realised Frobenius norm of the full-width
      # draw this tap would have had. It is a function of the SEED alone,
      # so it is computed once and cached — the draw itself is 5 x 5e5
      # Box-Muller samples and paying for it at every Oja refresh would
      # cost more than the adaptation it normalises.
      if bi >= @gx_ld_tgt2.length
        sigf  = Toy::Train::DfaB.sigma_for(@gx_b_scale, v, dout, @gx_b_sigma)
        bfull = Toy::Train::DfaB.fill(nb, @gx_b_seeds[bi], @gx_b_dist, sigf)
        acc0 = 0.0
        k = 0
        while k < nb
          acc0 = acc0 + bfull[k] * bfull[k]
          k = k + 1
        end
        @gx_ld_tgt2.push(acc0)
      end
      tf2 = @gx_ld_tgt2[bi]
      # Q [dout x r], from the tap's OWN seed, so --b-seed keeps meaning
      # what it meant on every other arm of this lane.
      sigq = Toy::Train::DfaB.sigma_for(@gx_b_scale, r, dout, @gx_b_sigma)
      q = Toy::Train::DfaB.fill(dout * r, @gx_b_seeds[bi], @gx_b_dist, sigq)

      # B_eff = Q P, accumulated row by row: the inner loop walks P and
      # B_eff contiguously, which matters at dout*r*v ~ 1.3e8 per tap.
      vals = Array.new(nb, 0.0)
      rr = 0
      while rr < dout
        base = rr * v
        t = 0
        while t < r
          qq = q[rr * r + t]
          pb = t * v
          c = 0
          while c < v
            vals[base + c] = vals[base + c] + qq * @gx_p[pb + c]
            c = c + 1
          end
          t = t + 1
        end
        rr = rr + 1
      end

      ef2 = 0.0
      k2 = 0
      while k2 < nb
        ef2 = ef2 + vals[k2] * vals[k2]
        k2 = k2 + 1
      end
      if ef2 > 0.0
        gsc = Math.sqrt(tf2 / ef2)
        k3 = 0
        while k3 < nb
          vals[k3] = vals[k3] * gsc
          k3 = k3 + 1
        end
      end
      # Re-measured AFTER the rescale, not assumed from it: the emitted
      # ||B_eff||_F must describe what was actually uploaded.
      ef2b = 0.0
      k4 = 0
      while k4 < nb
        ef2b = ef2b + vals[k4] * vals[k4]
        k4 = k4 + 1
      end
      eff2  = eff2 + ef2b
      full2 = full2 + tf2

      TinyNN.tnn_upload_from_float_array(@sess, @gx_b_handles[bi], vals, nb)

      rk = r < dout ? r : dout
      if rk > rke
        rke = rk
      end
      if dout > dmax
        dmax = dout
      end

      fro2 = ef2b
      rmax = 0.0
      r5 = 0
      while r5 < dout
        rn = 0.0
        c5 = 0
        base5 = r5 * v
        while c5 < v
          rn = rn + vals[base5 + c5] * vals[base5 + c5]
          c5 = c5 + 1
        end
        if rn > rmax
          rmax = rn
        end
        r5 = r5 + 1
      end
      sr = rmax > 0.0 ? (fro2 / rmax) : 0.0
      @gx_b_sr = @gx_b_sr + (bi > 0 ? " " : "") +
                 "B" + bi.to_s + "=[" + dout.to_s + "x" + v.to_s +
                 "]sr_ub=" + sr.to_s
      bi = bi + 1
    end
    @gx_ld_fro_eff  = Math.sqrt(eff2)
    @gx_ld_fro_full = Math.sqrt(full2)
    @gx_ld_rank_eff = rke
    @gx_ld_dout     = dmax
    nil
  end

  # P [r x V]: one i.i.d. draw from the SAME DfaB machinery, then MGS.
  # Its seed is separate from every tap's, because P is SHARED across taps
  # — the compression is a property of the error stream, not of the tap,
  # which is the whole structure LDFA proposes.
  def init_p!(v, r)
    if @gx_p_map.length > 0
      init_p_from_map!(v, r)
      return 0
    end
    sigp = Toy::Train::DfaB.sigma_for(@gx_b_scale, v, r, @gx_b_sigma)
    @gx_p = Toy::Train::DfaB.fill(r * v, @gx_b_pseed, @gx_b_dist, sigp)
    ortho_p!(v, r)
  end

  # rev2026-08-23/D1 — ORACLE pooling P, from a two-file map pack:
  #   <prefix>.meta.i32  [n_codes, n_base]
  #   <prefix>.tok.i32   n_codes row ids (code c -> its base symbol)
  # Rows have disjoint support by construction and are normalised to unit
  # norm, so P is exactly orthonormal; adapt_p! (apply=0) measures it
  # unchanged and the scale gate in refresh_b_lowrank! applies as-is.
  # Codes beyond n_codes are padded head columns and stay all-zero (they
  # carry no error).
  def init_p_from_map!(v, r)
    hdr = Array.new(2, 0)
    got = TinyNN.tnn_read_i32_file(@gx_p_map + ".meta.i32", 0, 2, hdr)
    if got != 2
      puts "gtx_engine: could not read oracle P map " + @gx_p_map + ".meta.i32"
      exit 1
    end
    n_codes = hdr[0]
    n_base  = hdr[1]
    if n_base > r
      puts "gtx_engine: oracle P map has " + n_base.to_s +
           " base symbols but GTX_DFA_RANK is " + r.to_s +
           " — rank must be at least the map's base-symbol count"
      exit 1
    end
    if n_base < r && @gx_p_csup.length == 0
      puts "gtx_engine: GTX_DFA_RANK " + r.to_s + " exceeds the map's " +
           n_base.to_s + " base symbols — set GTX_LDFA_CSUP" +
           " (active|dead|full) to say where the " + (r - n_base).to_s +
           " random complement rows draw their support"
      exit 1
    end
    if n_codes > v
      puts "gtx_engine: oracle P map covers " + n_codes.to_s +
           " codes but the head has only " + v.to_s + " classes"
      exit 1
    end
    map = Array.new(n_codes, 0)
    got2 = TinyNN.tnn_read_i32_file(@gx_p_map + ".tok.i32", 0, n_codes, map)
    if got2 != n_codes
      puts "gtx_engine: oracle P map short read (" + got2.to_s + "/" +
           n_codes.to_s + ")"
      exit 1
    end
    @gx_p = Array.new(r * v, 0.0)
    counts = Array.new(n_base, 0)
    c = 0
    while c < n_codes
      b = map[c]
      if b < 0 || b >= n_base
        puts "gtx_engine: oracle P map code " + c.to_s + " row " + b.to_s +
             " out of range"
        exit 1
      end
      @gx_p[b * v + c] = 1.0
      counts[b] = counts[b] + 1
      c = c + 1
    end
    i = 0
    while i < n_base
      if counts[i] < 1
        puts "gtx_engine: oracle P map leaves row " + i.to_s + " empty"
        exit 1
      end
      nrm = Math.sqrt(counts[i].to_f)
      pb = i * v
      cc = 0
      while cc < v
        @gx_p[pb + cc] = @gx_p[pb + cc] / nrm
        cc = cc + 1
      end
      i = i + 1
    end
    if r > n_base
      fill_p_complement!(v, r, n_codes, n_base)
    end
    0
  end

  # rev2026-08-28/D1b — the RANK CONTROL for D1.
  #
  # D1's oracle P had exactly n_base = 65 rows, so rank(Q.P) = 65 < dout =
  # d_model = 128: "perfect routing" was confounded with a rank
  # restriction, which the LDFA literature independently predicts is
  # harmful (spurious fixed points). This adds r - n_base random rows,
  # orthonormalised against the pooling rows and each other, so the SAME
  # oracle routing can be run at rank_eff = dout.
  #
  # WHERE THE COMPLEMENT DRAWS ITS SUPPORT IS THE WHOLE DESIGN, because on
  # an inflation fixture the learnable error is only n_base-dimensional —
  # you cannot have both full rank and a noise-free error stream:
  #   active : columns [0, n_codes) — the extra directions are within-base
  #            contrasts, i.e. the planted uniform noise. Rank is fully
  #            lifted, at the price of re-admitting dilution.
  #   dead   : columns [n_codes, v) — head classes that are never a target,
  #            so the extra directions carry only softmax leak. Rank is
  #            lifted formally with the signal left clean; the price is
  #            that those directions see almost no error.
  #   full   : columns [0, v) — the unconstrained draw, for completeness.
  # Running `active` and `dead` brackets the confound from both sides.
  def fill_p_complement!(v, r, n_codes, n_base)
    c0 = 0
    c1 = v
    if @gx_p_csup == "active"
      c0 = 0
      c1 = n_codes
    end
    if @gx_p_csup == "dead"
      c0 = n_codes
      c1 = v
    end
    wid = c1 - c0
    nrows = r - n_base
    if wid < nrows
      puts "gtx_engine: complement support " + @gx_p_csup + " spans " +
           wid.to_s + " columns, too few for " + nrows.to_s +
           " independent rows"
      exit 1
    end
    # Same draw machinery and same P-seed as the plain random P, so a
    # complement row is distributionally the row init_p! would have made.
    # Sigma is irrelevant after MGS normalises, but keep it consistent.
    sigp = Toy::Train::DfaB.sigma_for(@gx_b_scale, v, r, @gx_b_sigma)
    fillv = Toy::Train::DfaB.fill(nrows * wid, @gx_b_pseed, @gx_b_dist, sigp)
    i = n_base
    while i < r
      pb = i * v
      k = 0
      while k < wid
        @gx_p[pb + c0 + k] = fillv[(i - n_base) * wid + k]
        k = k + 1
      end
      i = i + 1
    end
    # Rows 0..n_base-1 are left bit-untouched: the oracle pooling rows have
    # disjoint support, are already unit-norm, and must stay byte-identical
    # to D1 so the D1 cells remain the reproduction gate for this branch.
    if ortho_p_rows!(v, r, n_base) != 0
      puts "gtx_engine: complement rows collapsed under MGS (support " +
           @gx_p_csup + ", " + nrows.to_s + " rows in " + wid.to_s +
           " columns)"
      exit 1
    end
    0
  end

  # Modified Gram-Schmidt over the ROWS of P, in place. Returns 0, or 1 if
  # a row collapses to zero (which at r << V means the draw or an Oja step
  # destroyed the basis, and it must not be silently renormalised).
  def ortho_p!(v, r)
    ortho_p_rows!(v, r, 0)
  end

  # MGS over rows i0..r-1 only, orthogonalising each against EVERY earlier
  # row (including rows < i0, which are left bit-untouched). rev2026-08-28
  # D1b needs exactly this: the oracle rows must survive unchanged while a
  # random complement is orthogonalised into what is left.
  def ortho_p_rows!(v, r, i0)
    i = i0
    while i < r
      bi2 = i * v
      j = 0
      while j < i
        bj = j * v
        d = 0.0
        c = 0
        while c < v
          d = d + @gx_p[bi2 + c] * @gx_p[bj + c]
          c = c + 1
        end
        c2 = 0
        while c2 < v
          @gx_p[bi2 + c2] = @gx_p[bi2 + c2] - d * @gx_p[bj + c2]
          c2 = c2 + 1
        end
        j = j + 1
      end
      nrm = 0.0
      c3 = 0
      while c3 < v
        nrm = nrm + @gx_p[bi2 + c3] * @gx_p[bi2 + c3]
        c3 = c3 + 1
      end
      nrm = Math.sqrt(nrm)
      d0 = nrm - nrm
      if nrm <= 0.0 || d0 != 0.0
        return 1
      end
      c4 = 0
      while c4 < v
        @gx_p[bi2 + c4] = @gx_p[bi2 + c4] / nrm
        c4 = c4 + 1
      end
      i = i + 1
    end
    0
  end

  # ── ONE ADAPTATION / MEASUREMENT PASS OVER m ERROR SAMPLES ──
  #
  # Oja's subspace rule, per error sample e (a V-vector), with P's rows
  # orthonormal:
  #
  #     y = P e                                  [r]
  #     P += eta (y e' - y y' P)  ==  eta y (e - P'y)'
  #
  # The right-hand form is what is implemented and it is why this is
  # cheap: y y' P is [r x r][r x V] read naively, but (y y')P = y (P'y)'
  # is a RANK-1 update, so the whole step is 3 r V rather than r^2 V.
  #
  # `apply` is 0 for the FIXED arm, which still runs the measurement pass
  # (the energy fraction below is the two arms' only common yardstick) and
  # updates nothing. Both arms therefore download the same logits on the
  # same steps — the fixed/adaptive contrast is the Oja update and nothing
  # else.
  #
  # The orthonormality diagnostics are taken BEFORE re-orthonormalising,
  # deliberately: after MGS the row norms are 1 and the off-diagonals ~0
  # BY CONSTRUCTION, so measuring there would print a tautology while a
  # diverging eta went unnoticed. What is reported is the drift Oja
  # actually accumulated over this refresh's m samples.
  #
  # Returns SEVEN FLOATS and nothing else (a mixed-type array comes back
  # `unknown` under Spinel and the first .to_f on it fails at runtime):
  #   [0] status  0 ok / 1 basis collapsed or non-finite / 2 not configured
  #   [1] min row norm of P before re-orthonormalisation
  #   [2] max row norm     "
  #   [3] max |P_i . P_j|, i != j, before re-orthonormalisation
  #   [4] captured energy fraction  sum||P e||^2 / sum||e||^2
  #   [5] ||B_eff||_F after the rebuild
  #   [6] ||B_full||_F, the scale target
  def adapt_p!(ebuf, n, eta, apply)
    out = [0.0]
    out.push(0.0); out.push(0.0); out.push(0.0)
    out.push(0.0); out.push(0.0); out.push(0.0)
    if @gx_b_rank < 1 || @gx_p_ready == 0 || n < 1
      out[0] = 2.0
      return out
    end
    v = @gx_active_classes
    r = @gx_b_rank
    y = Array.new(r, 0.0)
    w = Array.new(v, 0.0)
    cap = 0.0
    tot = 0.0
    s = 0
    while s < n
      eb = s * v
      t = 0
      while t < r
        acc = 0.0
        pb = t * v
        c = 0
        while c < v
          acc = acc + @gx_p[pb + c] * ebuf[eb + c]
          c = c + 1
        end
        y[t] = acc
        cap = cap + acc * acc
        t = t + 1
      end
      c0 = 0
      while c0 < v
        tot = tot + ebuf[eb + c0] * ebuf[eb + c0]
        c0 = c0 + 1
      end
      if apply == 1 && eta > 0.0
        c1 = 0
        while c1 < v
          w[c1] = 0.0
          c1 = c1 + 1
        end
        t2 = 0
        while t2 < r
          yy = y[t2]
          pb2 = t2 * v
          c3 = 0
          while c3 < v
            w[c3] = w[c3] + yy * @gx_p[pb2 + c3]
            c3 = c3 + 1
          end
          t2 = t2 + 1
        end
        t3 = 0
        while t3 < r
          g = eta * y[t3]
          pb3 = t3 * v
          c4 = 0
          while c4 < v
            @gx_p[pb3 + c4] = @gx_p[pb3 + c4] + g * (ebuf[eb + c4] - w[c4])
            c4 = c4 + 1
          end
          t3 = t3 + 1
        end
      end
      s = s + 1
    end

    # Drift, measured on the CURRENT P (post-Oja, pre-MGS).
    rn_min = 0.0
    rn_max = 0.0
    off    = 0.0
    i = 0
    while i < r
      bi2 = i * v
      nn = 0.0
      c5 = 0
      while c5 < v
        nn = nn + @gx_p[bi2 + c5] * @gx_p[bi2 + c5]
        c5 = c5 + 1
      end
      nn = Math.sqrt(nn)
      if i == 0 || nn < rn_min
        rn_min = nn
      end
      if nn > rn_max
        rn_max = nn
      end
      j = 0
      while j < i
        bj = j * v
        d = 0.0
        c6 = 0
        while c6 < v
          d = d + @gx_p[bi2 + c6] * @gx_p[bj + c6]
          c6 = c6 + 1
        end
        if d < 0.0
          d = 0.0 - d
        end
        if d > off
          off = d
        end
        j = j + 1
      end
      i = i + 1
    end
    # FINITE OR NOTHING. A diverged eta uploads inf/nan into B and the
    # loss simply stops being a number partway through a 4000-step run,
    # with nothing in the output saying so.
    dchk = (rn_max - rn_max) + (off - off) + (cap - cap)
    if dchk != 0.0
      out[0] = 1.0
      return out
    end
    if apply == 1 && eta > 0.0
      if ortho_p!(v, r) != 0
        out[0] = 1.0
        return out
      end
      refresh_b_lowrank!
    end
    out[1] = rn_min
    out[2] = rn_max
    out[3] = off
    out[4] = tot > 0.0 ? (cap / tot) : 0.0
    out[5] = @gx_ld_fro_eff
    out[6] = @gx_ld_fro_full
    out
  end

  # ---- toy#172 / E1 Phase 1.2: the nDFA ERROR-SIDE PRECONDITIONER ----
  #
  # nDFA (Safaai et al. 2026, arXiv:2607.18574) left-multiplies the
  # broadcast error by the inverse local-error second moment. E1 Phase
  # 1.1 measured that the thing being inverted is real: C_E occupies
  # 5-11% of the dimensions available to it and relatively less as the
  # head widens. So there IS anisotropy to whiten.
  #
  # ── WHY THIS IS FOLDED INTO B AND NOT ADDED TO THE GRAPH ──
  #
  # The preconditioner acts on the error, and the error reaches every tap
  # through exactly one product — B . e. So
  #
  #     (B) . (P e)  ==  (B P) . e
  #
  # and B is generated HOST-SIDE by refresh_b! and uploaded. Folding P
  # into B therefore needs NO new graph node, NO new readout path and no
  # new backend surface — which is the right prior on this lane, where
  # the ||g|| instrument realized as ZEROS through three separate readout
  # paths and had to be removed rather than shipped reading 0.0.
  #
  # ── THE IDENTITY, AND WHY NOTHING [V x V] IS EVER FORMED ──
  #
  #   C_E = (1/m) E E'          E = [V x m] error samples
  #   P   = lambda (C_E + lambda I)^-1
  #       = I - E (lambda m I_m + G)^-1 E'          G = E'E, [m x m]
  #   B'  = B P = B - (B E) (lambda m I_m + G)^-1 E'
  #
  # (Woodbury.) At V=4096, m=256, dout=128 that is ~1.4e9 MACs for five
  # taps against 2.3e10 for the [V x V] inverse — and the [V x V] inverse
  # is where f32 accumulation would go wrong SILENTLY, which is the whole
  # reason the spec asked for a finite-norm assertion.
  #
  # ── THE lambda NORMALISATION IS LOAD-BEARING ──
  #
  # P is lambda(C_E + lambda I)^-1, NOT (C_E + lambda I)^-1. The bracket
  # above goes to ZERO as lambda grows, so B' -> B EXACTLY (the
  # correction underflows against B's own magnitude in f64 long before
  # the f32 upload), which makes "large lambda changes nothing" a BYTE
  # IDENTITY rather than an approximation. The unnormalised form tends to
  # I/lambda instead and would silently rescale every DFA update — an LR
  # change wearing a conditioning costume.
  #
  # ── AND SO IS THE GAIN RULE ──
  #
  # P's eigenvalues are lambda/(lambda + s_i) in (0, 1], so nDFA can only
  # SHRINK B — hardest along the error's dominant directions, which is
  # the point, but with a global gain drop riding along. On this lane a
  # global gain on B is indistinguishable from an LR change (toy#152's
  # B-scale finding restated), so `preserve` renormalises B' back to
  # ||B||_F and leaves ONLY the direction reweighting under test. `raw`
  # keeps the shrinkage, and the measured pre-renorm ratio is emitted
  # either way so the size of what was removed is never hidden.
  #
  # Returns FOUR FLOATS and nothing else (a mixed-type array comes back
  # `unknown` under Spinel and the first .to_f on it fails at runtime):
  #   [0] status  0 ok / 1 Cholesky failed / 2 no DFA taps wired
  #   [1] ||B E||_F   before preconditioning, summed over taps
  #   [2] ||B' E||_F  after, i.e. the error as the taps actually see it
  #   [3] ||B'||_F / ||B||_F  BEFORE any gain restoration
  def precondition_b!(ebuf, n, lam, preserve_gain)
    out = [0.0]
    out.push(0.0)
    out.push(0.0)
    out.push(1.0)
    if @gx_b_handles.length == 0 || n < 1
      out[0] = 2.0
      return out
    end
    v = @gx_active_classes

    # G = E'E. Symmetric, so only the upper triangle is computed and
    # mirrored — same shape as the instrument's bc_gram!, kept here
    # rather than shared because the engine is its own mirrored unit and
    # a cross-file top-level call from an instance method is exactly the
    # kind of thing Spinel's inference gets wrong quietly.
    g = Array.new(n * n, 0.0)
    i = 0
    while i < n
      j = i
      while j < n
        s = 0.0
        ai = i * v
        aj = j * v
        c = 0
        while c < v
          s = s + ebuf[ai + c] * ebuf[aj + c]
          c = c + 1
        end
        g[i * n + j] = s
        g[j * n + i] = s
        j = j + 1
      end
      i = i + 1
    end

    # M = lambda m I + G, then its Cholesky IN PLACE. M is SPD for any
    # lambda > 0 (G is PSD by construction), so a failure here means the
    # ridge is too small for f64 at this width, and it FAILS rather than
    # falling back to something that still returns a number.
    mm = Array.new(n * n, 0.0)
    k = 0
    lim = n * n
    while k < lim
      mm[k] = g[k]
      k = k + 1
    end
    d = 0
    while d < n
      mm[d * n + d] = mm[d * n + d] + lam * n.to_f
      d = d + 1
    end
    if nd_chol!(mm, n) != 0
      out[0] = 1.0
      return out
    end

    pre2  = 0.0
    post2 = 0.0
    shr_n = 0.0
    shr_d = 0.0
    @gx_b_sr = ""
    bi = 0
    while bi < @gx_b_handles.length
      dout = @gx_b_douts[bi]
      nb   = dout * v
      sig  = Toy::Train::DfaB.sigma_for(@gx_b_scale, v, dout, @gx_b_sigma)
      # REGENERATED from the seed, never read back and never accumulated
      # across refreshes: B' is a function of (B, E) alone, so a second
      # refresh must precondition the ORIGINAL draw and not last
      # refresh's output. Compounding would make the flag's meaning
      # depend on the cadence.
      vals = Toy::Train::DfaB.fill(nb, @gx_b_seeds[bi], @gx_b_dist, sig)

      fro_pre = 0.0
      fk = 0
      while fk < nb
        fro_pre = fro_pre + vals[fk] * vals[fk]
        fk = fk + 1
      end

      # be = B E   [dout x n]
      be = Array.new(dout * n, 0.0)
      r = 0
      while r < dout
        br = r * v
        sc = 0
        while sc < n
          acc = 0.0
          er = sc * v
          c2 = 0
          while c2 < v
            acc = acc + vals[br + c2] * ebuf[er + c2]
            c2 = c2 + 1
          end
          be[r * n + sc] = acc
          pre2 = pre2 + acc * acc
          sc = sc + 1
        end
        r = r + 1
      end

      # z = (B E) M^-1, one Cholesky solve per row of B E.
      z = Array.new(dout * n, 0.0)
      x = Array.new(n, 0.0)
      r2 = 0
      while r2 < dout
        q = 0
        while q < n
          x[q] = be[r2 * n + q]
          q = q + 1
        end
        nd_chol_solve!(mm, x, n)
        q2 = 0
        while q2 < n
          z[r2 * n + q2] = x[q2]
          q2 = q2 + 1
        end
        r2 = r2 + 1
      end

      # ||B' E||_F, from B'E = BE - z G. [dout x n x n] and so cheap
      # next to the two [dout x n x v] products either side of it. This
      # is THE number the finite-norm assertion is stated on, because it
      # is the error as the taps actually receive it rather than an
      # abstract P e.
      r4 = 0
      while r4 < dout
        s4 = 0
        while s4 < n
          acc4 = be[r4 * n + s4]
          t4 = 0
          while t4 < n
            acc4 = acc4 - z[r4 * n + t4] * g[t4 * n + s4]
            t4 = t4 + 1
          end
          post2 = post2 + acc4 * acc4
          s4 = s4 + 1
        end
        r4 = r4 + 1
      end

      # B' = B - z E'
      r3 = 0
      while r3 < dout
        br2 = r3 * v
        s3 = 0
        while s3 < n
          zz = z[r3 * n + s3]
          er2 = s3 * v
          c3 = 0
          while c3 < v
            vals[br2 + c3] = vals[br2 + c3] - zz * ebuf[er2 + c3]
            c3 = c3 + 1
          end
          s3 = s3 + 1
        end
        r3 = r3 + 1
      end

      fro_post = 0.0
      fk2 = 0
      while fk2 < nb
        fro_post = fro_post + vals[fk2] * vals[fk2]
        fk2 = fk2 + 1
      end
      shr_n = shr_n + fro_post
      shr_d = shr_d + fro_pre
      # At large lambda vals is BIT-IDENTICAL to the draw, so this ratio
      # is exactly 1.0 and the rescale below is exactly the identity —
      # which is what makes the byte-null gate a byte gate.
      if preserve_gain == 1 && fro_post > 0.0
        gsc = Math.sqrt(fro_pre / fro_post)
        gk = 0
        while gk < nb
          vals[gk] = vals[gk] * gsc
          gk = gk + 1
        end
      end

      TinyNN.tnn_upload_from_float_array(@sess, @gx_b_handles[bi], vals, nb)

      # The stable-rank CONTROL is recomputed here on purpose. It is
      # emitted once, at the end of the run; leaving it describing the
      # initial draw while the live matrix is B' would make the control
      # describe a matrix the run stopped using.
      fro2 = 0.0
      k5 = 0
      while k5 < nb
        fro2 = fro2 + vals[k5] * vals[k5]
        k5 = k5 + 1
      end
      rmax = 0.0
      r5 = 0
      while r5 < dout
        rn = 0.0
        c5 = 0
        base5 = r5 * v
        while c5 < v
          rn = rn + vals[base5 + c5] * vals[base5 + c5]
          c5 = c5 + 1
        end
        if rn > rmax
          rmax = rn
        end
        r5 = r5 + 1
      end
      sr = rmax > 0.0 ? (fro2 / rmax) : 0.0
      @gx_b_sr = @gx_b_sr + (bi > 0 ? " " : "") +
                 "B" + bi.to_s + "=[" + dout.to_s + "x" + v.to_s +
                 "]sr_ub=" + sr.to_s
      bi = bi + 1
    end

    out[1] = Math.sqrt(pre2)
    out[2] = post2 > 0.0 ? Math.sqrt(post2) : post2
    out[3] = shr_d > 0.0 ? Math.sqrt(shr_n / shr_d) : 1.0
    out
  end

  # Lower Cholesky of a SYMMETRIC POSITIVE-DEFINITE flat row-major [n x n],
  # in place: the lower triangle becomes L, the upper is left as it was
  # (nd_chol_solve! only ever reads the lower half). Returns 0 on
  # success, 1 if a pivot goes non-positive.
  #
  # Written out rather than reached for because there is no dense linear
  # algebra in this program: everything else goes through ggml, and ggml
  # has no Cholesky. n is the SAMPLE count (256-1024), so n^3/3 is
  # 5.6e6-3.6e8 — small beside the [dout x n x v] products it enables.
  def nd_chol!(a, n)
    j = 0
    while j < n
      s = a[j * n + j]
      k = 0
      while k < j
        s = s - a[j * n + k] * a[j * n + k]
        k = k + 1
      end
      if s <= 0.0
        return 1
      end
      dg = Math.sqrt(s)
      a[j * n + j] = dg
      i = j + 1
      while i < n
        s2 = a[i * n + j]
        k2 = 0
        while k2 < j
          s2 = s2 - a[i * n + k2] * a[j * n + k2]
          k2 = k2 + 1
        end
        a[i * n + j] = s2 / dg
        i = i + 1
      end
      j = j + 1
    end
    0
  end

  # Solve L L' x = x in place for the L that nd_chol! left in `l`.
  def nd_chol_solve!(l, x, n)
    i = 0
    while i < n
      s = x[i]
      k = 0
      while k < i
        s = s - l[i * n + k] * x[k]
        k = k + 1
      end
      x[i] = s / l[i * n + i]
      i = i + 1
    end
    i2 = n - 1
    while i2 >= 0
      s2 = x[i2]
      k2 = i2 + 1
      while k2 < n
        s2 = s2 - l[k2 * n + i2] * x[k2]
        k2 = k2 + 1
      end
      x[i2] = s2 / l[i2 * n + i2]
      i2 = i2 - 1
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
