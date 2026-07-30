# lib/toy/run/franken_moe_parts.rb — the Franken-MoE instrument's shared
# builders (toy#120): consts, tower registration, attention/MoE blocks
# (dense soft-mixture + top1 hard routing), and the chain/dfa/top1 wire
# helpers. Split out of train_franken_moe.rb so the twin-lane rig and the
# spec-callable single-lane runner (train_franken_moe_cli.rb) compile the
# SAME graph builders — the module is reopened by both.

require_relative "../../toy"
require_relative "../ffi/tinynn"
require_relative "../llm/primitives/rms_norm"
require_relative "../train/dfa_b"
require_relative "../llm/primitives/muon"

module Toy
  module LLM
    module Run
      module TrainFrankenMoe
        VOCAB = 16
        DM_BASE  = 8
        DFF_BASE = 16
        NE    = 2      # experts
        T     = 4
        EPS   = 1.0e-5

        # toy#124 shape presets: DM/DFF are runtime module state so the
        # spec-callable runner can widen (base d8/ff16, wide d256/ff512)
        # without a second compiled unit. EVERY runner calls shape_init
        # FIRST (the rig pins base; the CLI reads FRANKEN_SHAPE).
        # NE stays pinned per shape preset; T (context) joined the
        # runtime state in toy#129 item 1 (the corpus feed at ctx>=8;
        # the fixed-seq feed pins T=4, byte-null).
        #
        # toy#125: vocab joined the runtime state — the corpus feed
        # streams the frozen-vocab contract (627, toy#123), and vocab-16
        # embeds cannot take those token ids (get_rows would read past
        # the table). The rig and the CLI's fixed-seq feed pin VOCAB
        # (16, byte-null); --corpus pins 627.
        #
        # toy#128: expert count joined too (the demonstrator's E axis;
        # E>=8 is the F9 target). The rig pins NE (2, byte-null); the
        # CLI reads FRANKEN_MOE_EXPERTS. Experts live at pp[9+2i]
        # (up_i) / pp[10+2i] (down_i); the spine stays 0..8.
        def self.shape_init(dm, dff, vocab, ne, t)
          @sh_dm = dm
          @sh_df = dff
          @sh_vocab = vocab
          @sh_ne = ne
          @sh_t = t
          0
        end

        def self.dmv
          @sh_dm
        end

        def self.dfv
          @sh_df
        end

        def self.vocabv
          @sh_vocab
        end

        def self.nev
          @sh_ne
        end

        def self.tv
          @sh_t
        end

        # toy#139 optimizer axis. 0 = adamw (byte-null default), 1 =
        # muon, 2 = sgd. Muon uses the STANDARD recipe (Jordan): the
        # orthogonalized step applies to 2D HIDDEN matrices only —
        # attention q/k/v/o, the router, the experts, the output gate —
        # while embeddings, norms and scalars stay on AdamW. Naive
        # Muon-on-everything underperforms and would make the F9m
        # comparison a strawman.
        #
        # In this instrument that rule is an index rule: pp[0] is the
        # (tied) embedding and pp[1]/pp[2]/pp[7] are norms, so those
        # four stay AdamW; everything else is a 2D hidden matrix.
        def self.opt_init(v)
          @sh_opt = v
          0
        end

        def self.optv
          @sh_opt
        end

        def self.muon_eligible(idx)
          idx != 0 && idx != 1 && idx != 2 && idx != 7
        end

        # ONE parameter's update, routed by optimizer + param class.
        # t_hp is the AdamW hp vector; t_hp_sgd the [lr, wd] pair the
        # ggml SGD step takes. Returns the opt node (already extended
        # into the backward graph).
        def self.apply_step(sess, tw, t_hp, t_hp_sgd, idx, grad)
          to = TinyNN.tnn_null_ptr
          if optv == 1 && muon_eligible(idx)
            nel = TinyNN.tnn_tensor_nelements(tw.pp[idx])
            ne0 = TinyNN.tnn_tensor_ne0(tw.pp[idx])
            ne1 = nel / ne0
            # The momentum buffer IS the Adam m slot (already allocated
            # and zeroed) — Muon needs one buffer, not two.
            step = Toy::LLM::Primitives::Muon.update(sess, grad, tw.pm[idx], ne0, ne1)
            to = TinyNN.tnn_opt_step_sgd(sess, tw.pp[idx], step, t_hp_sgd)
          elsif optv == 2
            to = TinyNN.tnn_opt_step_sgd(sess, tw.pp[idx], grad, t_hp_sgd)
          else
            to = TinyNN.tnn_opt_step_adamw(sess, tw.pp[idx], grad, tw.pm[idx], tw.pv[idx], t_hp)
          end
          TinyNN.tnn_extend_backward_graph(sess, to)
          0
        end

        # toy#136 K1.1: attention output gate (K3 eq 7, the MLA form —
        # y = W_o[σ(W_g x) ⊙ õ], no RMSNorm; the KDA form's head-wise
        # RMSNorm arrives with KDA in K2). 0 = off, byte-null: the gate
        # weight is not even ALLOCATED, so every other weight keeps its
        # index AND its init stream. When on, W_g lands at the TAIL
        # (9 + 2E) — appending is what keeps the spine 0..8 and the
        # expert pairs 9+2i/10+2i (and their stable DfaB seeds) intact.
        def self.attn_gate_init(v)
          @sh_gate = v
          0
        end

        def self.gatev
          @sh_gate
        end

        def self.gate_idx
          9 + 2 * nev
        end

        def self.fillv(n, seed)
          a = [0.0]; a.pop
          i = 0
          while i < n
            a.push(((((i + seed) * 1103515245 + 12345) % 1000) - 500).to_f * 0.001)
            i = i + 1
          end
          a
        end

        def self.zeros(n)
          a = [0.0]; a.pop
          i = 0
          while i < n
            a.push(0.0)
            i = i + 1
          end
          a
        end

        def self.b_dist_code
          d = ENV["FRANKEN_B_DIST"] || ""
          if d == "uniform"
            return Toy::Train::DfaB::DIST_UNIFORM
          end
          if d == "rademacher"
            return Toy::Train::DfaB::DIST_RADEMACHER
          end
          Toy::Train::DfaB::DIST_GAUSSIAN
        end

        # Per-tower handles (uniform-typed arrays; no Struct/Card).
        class MoeTower
          attr_accessor :pp, :pm, :pv
          attr_accessor :t_loss, :t_logits, :t_gates
          attr_accessor :tap_h2, :tap_a1, :tap_a2, :tap_ah, :tap_ctx, :t_onehots
          attr_accessor :t_rlogits
          attr_accessor :tap_as
          attr_accessor :dfa_grads, :dfa_accs, :dfa_names

          def initialize
            np = TinyNN.tnn_null_ptr
            @pp = [np]; @pp.pop
            @pm = [np]; @pm.pop
            @pv = [np]; @pv.pop
            @t_loss   = np
            @t_logits = np
            @t_gates  = np
            @tap_h2   = np
            @tap_a1   = np
            @tap_a2   = np
            @tap_ah   = np
            @tap_ctx  = np
            @t_onehots = np
            @t_rlogits = np   # toy#136: raw router logits (QB reads margins)
            @tap_as    = [np]; @tap_as.pop   # toy#128: per-expert dense acts
            @dfa_grads = [np]; @dfa_grads.pop
            @dfa_accs  = [np]; @dfa_accs.pop
            @dfa_names = [""]; @dfa_names.pop
          end
        end

        def self.reg2(sess, tw, rows, cols)
          w = TinyNN.tnn_input_2d_f32_persistent(sess, rows, cols)
          tw.pp.push(w)
          tw.pm.push(TinyNN.tnn_input_2d_f32_persistent(sess, rows, cols))
          tw.pv.push(TinyNN.tnn_input_2d_f32_persistent(sess, rows, cols))
          w
        end

        def self.reg1(sess, tw, n)
          w = TinyNN.tnn_input_1d_f32_persistent(sess, n)
          tw.pp.push(w)
          tw.pm.push(TinyNN.tnn_input_1d_f32_persistent(sess, n))
          tw.pv.push(TinyNN.tnn_input_1d_f32_persistent(sess, n))
          w
        end

        # weight order (parity depends on identical registration in both
        # towers): embed, fnorm, attn{rn1,wq,wk,wv,wo}, moe{rn2, wr,
        # then per expert i: up_i, down_i}  → pp indices 0..8+2E
        # (toy#128: the expert tail loops; E=2 reproduces 9..12 exactly)
        def self.alloc_tower(sess)
          tw = MoeTower.new
          reg2(sess, tw, vocabv, dmv)  # 0 embed
          reg1(sess, tw, dmv)          # 1 fnorm
          reg1(sess, tw, dmv)          # 2 attn rn1
          reg2(sess, tw, dmv, dmv)      # 3 wq
          reg2(sess, tw, dmv, dmv)      # 4 wk
          reg2(sess, tw, dmv, dmv)      # 5 wv
          reg2(sess, tw, dmv, dmv)      # 6 wo
          reg1(sess, tw, dmv)          # 7 moe rn2
          reg2(sess, tw, nev, dmv)     # 8 wr (router)
          ei = 0
          while ei < nev
            reg2(sess, tw, dfv, dmv)   # 9+2i  up_i
            reg2(sess, tw, dmv, dfv)   # 10+2i down_i
            ei = ei + 1
          end
          if gatev == 1
            reg2(sess, tw, dmv, dmv)   # 9+2E  wg (attention output gate)
          end
          tw
        end

        # returns [out, h, ctx] — the DFA taps (dense mode ignores 1/2).
        # toy#133: `mask` — NULL keeps the byte-gated diag_mask_inf path
        # (B=1); a real [tb, tb] block-causal mask (0 within-window
        # causal, -1e30 elsewhere — the GH#7 values) is ADDED before
        # softmax so B windows never attend across each other. The
        # duplicated-window isolation null gates the orientation.
        def self.attention_block(sess, t_x, rn, wq, wk, wv, wo, mask, wg)
          h = Toy::LLM::Primitives::RMSNorm.build(sess, t_x, rn, EPS)
          q = TinyNN.tnn_matmul(sess, wq, h)
          k = TinyNN.tnn_matmul(sess, wk, h)
          v = TinyNN.tnn_matmul(sess, wv, h)
          scores = TinyNN.tnn_matmul(sess, k, q)
          scaled = TinyNN.tnn_scale(sess, scores, 1.0 / Math.sqrt(dmv.to_f))
          masked = TinyNN.tnn_null_ptr
          if mask == TinyNN.tnn_null_ptr
            masked = TinyNN.tnn_diag_mask_inf(sess, scaled, 0)
          else
            masked = TinyNN.tnn_add(sess, scaled, mask)
          end
          attn   = TinyNN.tnn_softmax(sess, masked)
          v_t    = TinyNN.tnn_transpose(sess, v)
          ctx    = TinyNN.tnn_matmul(sess, v_t, attn)
          # toy#136 K1.1: data-dependent channel-wise output gate. The
          # tap returned as tap_ctx is W_o's ACTUAL input (gated when
          # the gate is on) — the DFA wire for wo reads that tap, so
          # the local grad stays (B·e)·a_inᵀ with the true a_in.
          ctx_in = ctx
          if wg != TinyNN.tnn_null_ptr
            g_att  = TinyNN.tnn_sigmoid(sess, TinyNN.tnn_matmul(sess, wg, t_x))
            ctx_in = TinyNN.tnn_mul(sess, g_att, ctx)
          end
          out    = TinyNN.tnn_matmul(sess, wo, ctx_in)
          [TinyNN.tnn_add(sess, t_x, out), h, ctx_in]
        end

        # Dense soft-mixture MoE block; records taps + gates on the tower.
        # sels = the E uploaded one-hot row selectors [1, NE] (toy#128:
        # an array; the E=2 graph is the original one — sum tree
        # (t_x + (g1 + g2)) reproduced by the accumulator loop).
        # tap_a1/tap_a2 keep the rig's two-expert names; tap_as carries
        # all E for the CLI's generalized dfa wires.
        def self.moe_block(sess, tw, t_x, sels)
          rn2 = tw.pp[7]
          wr  = tw.pp[8]
          h2 = Toy::LLM::Primitives::RMSNorm.build(sess, t_x, rn2, EPS)
          tw.tap_h2 = h2
          r_logits = TinyNN.tnn_matmul(sess, wr, h2)          # [NE, T]
          gates    = TinyNN.tnn_softmax(sess, r_logits)       # [NE, T]
          TinyNN.tnn_set_output(gates)
          tw.t_gates = gates
          acc = TinyNN.tnn_null_ptr
          ei = 0
          while ei < nev
            g_i = TinyNN.tnn_matmul(sess, sels[ei], gates)    # [1, T]
            a_i = TinyNN.tnn_gelu(sess, TinyNN.tnn_matmul(sess, tw.pp[9 + 2 * ei], h2))  # [DFF,T]
            tw.tap_as.push(a_i)
            if ei == 0
              tw.tap_a1 = a_i
            end
            if ei == 1
              tw.tap_a2 = a_i
            end
            o_i = TinyNN.tnn_matmul(sess, tw.pp[10 + 2 * ei], a_i)               # [DM,T]
            gated_i = TinyNN.tnn_mul(sess, o_i, g_i)  # broadcast [DM,T]*[1,T]
            if ei == 0
              acc = gated_i
            else
              acc = TinyNN.tnn_add(sess, acc, gated_i)
            end
            ei = ei + 1
          end
          TinyNN.tnn_add(sess, t_x, acc)
        end

        # HARD top-1 MoE block: argmax routing + mul_mat_id dispatch.
        # eye = uploaded identity [NE,NE]; records taps + onehots + gates.
        # cut=1 (toy#121 bp-spine): the expert INPUT goes through
        # tnn_detach — forward-identity, gradient-opaque — so chain
        # grads reach attention/embeds via the residual + router
        # branches while the walker never needs mul_mat_id backward.
        # toy#136 (K1): qb_bias — NULL = legacy argmax(r_logits)
        # (byte-null); a real [NE,1] bias tensor shifts the SELECTION
        # scores only (K3 §2.3.3 Quantile Balancing: the mixture prob
        # stays softmax of the UNBIASED logits — routing dispatch moves,
        # mixture weights and router gradients do not).
        def self.moe_block_top1(sess, tw, t_x, eye, cut, qb_bias)
          rn2 = tw.pp[7]
          wr  = tw.pp[8]
          h2 = Toy::LLM::Primitives::RMSNorm.build(sess, t_x, rn2, EPS)
          tw.tap_h2 = h2
          r_logits = TinyNN.tnn_matmul(sess, wr, h2)              # [NE, T]
          TinyNN.tnn_set_output(r_logits)
          tw.t_rlogits = r_logits
          probs    = TinyNN.tnn_softmax(sess, r_logits)
          TinyNN.tnn_set_output(probs)
          tw.t_gates = probs
          sel_scores = r_logits
          if qb_bias != TinyNN.tnn_null_ptr
            sel_scores = TinyNN.tnn_add(sess, r_logits, qb_bias)  # broadcast [NE,T]+[NE,1]
          end
          ids   = TinyNN.tnn_argmax(sess, sel_scores)             # I32 [T]
          ids2  = TinyNN.tnn_reshape_3d(sess, ids, 1, tv, 1)       # [1, T]
          oneh  = TinyNN.tnn_get_rows(sess, eye, ids)             # [NE, T]
          TinyNN.tnn_set_output(oneh)
          tw.t_onehots = oneh
          gate  = TinyNN.tnn_sum_rows(sess, TinyNN.tnn_mul(sess, oneh, probs)) # [1,T]

          # toy#128: stack the E experts by chained concat along dim 2
          # (E=2 == the original single concat pair).
          up_stack = TinyNN.tnn_reshape_3d(sess, tw.pp[9],  dmv, dfv, 1)
          dn_stack = TinyNN.tnn_reshape_3d(sess, tw.pp[10], dfv, dmv, 1)
          ei = 1
          while ei < nev
            up_stack = TinyNN.tnn_concat(sess, up_stack,
              TinyNN.tnn_reshape_3d(sess, tw.pp[9 + 2 * ei],  dmv, dfv, 1), 2)   # [DM,DFF,E]
            dn_stack = TinyNN.tnn_concat(sess, dn_stack,
              TinyNN.tnn_reshape_3d(sess, tw.pp[10 + 2 * ei], dfv, dmv, 1), 2)   # [DFF,DM,E]
            ei = ei + 1
          end

          h_exp = h2
          if cut == 1
            h_exp = TinyNN.tnn_detach(sess, h2)
          end
          h3    = TinyNN.tnn_reshape_3d(sess, h_exp, dmv, 1, tv)
          upo   = TinyNN.tnn_mul_mat_id(sess, up_stack, h3, ids2)    # [DFF,1,T]
          a     = TinyNN.tnn_gelu(sess, upo)
          tw.tap_a1 = TinyNN.tnn_reshape_3d(sess, a, dfv, tv, 1)      # routed acts [DFF,T]
          dno   = TinyNN.tnn_mul_mat_id(sess, dn_stack, a, ids2)     # [DM,1,T]
          eo    = TinyNN.tnn_reshape_3d(sess, dno, dmv, tv, 1)         # [DM,T]
          TinyNN.tnn_add(sess, t_x, TinyNN.tnn_mul(sess, eo, gate))
        end

        def self.forward_tower(sess, tw, t_tok, t_labels, sels, eye, top1, cut, mask, qb_bias)
          x = TinyNN.tnn_get_rows(sess, tw.pp[0], t_tok)
          wg_t = TinyNN.tnn_null_ptr
          if gatev == 1
            wg_t = tw.pp[gate_idx]
          end
          atrip = attention_block(sess, x, tw.pp[2], tw.pp[3], tw.pp[4], tw.pp[5], tw.pp[6], mask, wg_t)
          x = atrip[0]
          tw.tap_ah  = atrip[1]
          tw.tap_ctx = atrip[2]
          if top1
            x = moe_block_top1(sess, tw, x, eye, cut, qb_bias)
          else
            x = moe_block(sess, tw, x, sels)
          end
          xf  = Toy::LLM::Primitives::RMSNorm.build(sess, x, tw.pp[1], EPS)
          lgt = TinyNN.tnn_matmul(sess, tw.pp[0], xf)
          tw.t_logits = lgt
          tw.t_loss   = TinyNN.tnn_cross_entropy_loss(sess, lgt, t_labels)
          TinyNN.tnn_set_output(tw.t_loss)
          # top1 tower B is a NO-AUTODIFF tower: its loss is read-only
          # output, never a backward root — a LOSS flag there leaves an
          # unallocated grad-acc that ggml_graph_reset asserts on
          # (GGML_ASSERT(grad_acc->data), found the hard way).
          if !top1
            TinyNN.tnn_set_loss(tw.t_loss)
          end
          0
        end

        # DFA grad for one weight: (B·e)·a_inᵀ. a_in [d_in, T] with the
        # weight's ne=[d_in, d_out]; B ne=[VOCAB, d_out].
        def self.dfa_grad(sess, b, e, a_in, d_in, d_out)
          delta  = TinyNN.tnn_matmul(sess, b, e)                        # [d_out, T]
          a_in_t = TinyNN.tnn_cont_2d(sess, TinyNN.tnn_transpose(sess, a_in), tv, d_in)
          delt_t = TinyNN.tnn_cont_2d(sess, TinyNN.tnn_transpose(sess, delta), tv, d_out)
          TinyNN.tnn_matmul(sess, a_in_t, delt_t)                       # [d_in, d_out]
        end

        def self.wire_chain(sess, tw, t_hp, t_hp_sgd, idx)
          tg = TinyNN.tnn_tensor_grad(sess, tw.pp[idx])
          apply_step(sess, tw, t_hp, t_hp_sgd, idx, tg)
        end

        # top1 lane-B wire: LATE-param (P0 idiom — tower B has zero params
        # at build_backward time) + optional routing mask on delta. No align
        # recording (no autodiff accs exist in this tower).
        def self.wire_dfa_top1(sess, tw, t_hp, t_hp_sgd, idx, b, e, a_in, d_in, d_out, mask)
          delta = TinyNN.tnn_matmul(sess, b, e)
          if mask != TinyNN.tnn_null_ptr
            delta = TinyNN.tnn_mul(sess, delta, mask)
          end
          a_in_t = TinyNN.tnn_cont_2d(sess, TinyNN.tnn_transpose(sess, a_in), tv, d_in)
          delt_t = TinyNN.tnn_cont_2d(sess, TinyNN.tnn_transpose(sess, delta), tv, d_out)
          g = TinyNN.tnn_matmul(sess, a_in_t, delt_t)
          TinyNN.tnn_set_output(g)
          TinyNN.tnn_set_param(tw.pp[idx])
          apply_step(sess, tw, t_hp, t_hp_sgd, idx, g)
        end

        def self.wire_dfa(sess, tw, t_hp, t_hp_sgd, idx, b, e, a_in, d_in, d_out, name)
          g = dfa_grad(sess, b, e, a_in, d_in, d_out)
          TinyNN.tnn_set_output(g)
          apply_step(sess, tw, t_hp, t_hp_sgd, idx, g)
          tw.dfa_grads.push(g)
          acc = TinyNN.tnn_tensor_grad(sess, tw.pp[idx])
          TinyNN.tnn_set_output(acc)   # shadow acc: unconsumed, pin it
          tw.dfa_accs.push(acc)
          tw.dfa_names.push(name)
          0
        end

        def self.cosv(a, b)
          dot = 0.0; na = 0.0; nb = 0.0
          i = 0
          while i < a.length
            dot = dot + a[i] * b[i]
            na = na + a[i] * a[i]
            nb = nb + b[i] * b[i]
            i = i + 1
          end
          d = Math.sqrt(na) * Math.sqrt(nb)
          if d <= 0.0
            return 0.0
          end
          dot / d
        end

      end
    end
  end
end
