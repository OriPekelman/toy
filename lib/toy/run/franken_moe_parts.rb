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

module Toy
  module LLM
    module Run
      module TrainFrankenMoe
        VOCAB = 16
        DM    = 8
        DFF   = 16
        NE    = 2      # experts
        T     = 4
        EPS   = 1.0e-5

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
        # up1, down1, up2, down2}   → pp indices 0..12
        def self.alloc_tower(sess)
          tw = MoeTower.new
          reg2(sess, tw, VOCAB, DM)   # 0 embed
          reg1(sess, tw, DM)          # 1 fnorm
          reg1(sess, tw, DM)          # 2 attn rn1
          reg2(sess, tw, DM, DM)      # 3 wq
          reg2(sess, tw, DM, DM)      # 4 wk
          reg2(sess, tw, DM, DM)      # 5 wv
          reg2(sess, tw, DM, DM)      # 6 wo
          reg1(sess, tw, DM)          # 7 moe rn2
          reg2(sess, tw, NE, DM)      # 8 wr (router)
          reg2(sess, tw, DFF, DM)     # 9 up1
          reg2(sess, tw, DM, DFF)     # 10 down1
          reg2(sess, tw, DFF, DM)     # 11 up2
          reg2(sess, tw, DM, DFF)     # 12 down2
          tw
        end

        # returns [out, h, ctx] — the DFA taps (dense mode ignores 1/2).
        def self.attention_block(sess, t_x, rn, wq, wk, wv, wo)
          h = Toy::LLM::Primitives::RMSNorm.build(sess, t_x, rn, EPS)
          q = TinyNN.tnn_matmul(sess, wq, h)
          k = TinyNN.tnn_matmul(sess, wk, h)
          v = TinyNN.tnn_matmul(sess, wv, h)
          scores = TinyNN.tnn_matmul(sess, k, q)
          scaled = TinyNN.tnn_scale(sess, scores, 1.0 / Math.sqrt(DM.to_f))
          masked = TinyNN.tnn_diag_mask_inf(sess, scaled, 0)
          attn   = TinyNN.tnn_softmax(sess, masked)
          v_t    = TinyNN.tnn_transpose(sess, v)
          ctx    = TinyNN.tnn_matmul(sess, v_t, attn)
          out    = TinyNN.tnn_matmul(sess, wo, ctx)
          [TinyNN.tnn_add(sess, t_x, out), h, ctx]
        end

        # Dense soft-mixture MoE block; records taps + gates on the tower.
        # sel1/sel2 are the uploaded one-hot row selectors [1, NE].
        def self.moe_block(sess, tw, t_x, sel1, sel2)
          rn2 = tw.pp[7]
          wr  = tw.pp[8]
          h2 = Toy::LLM::Primitives::RMSNorm.build(sess, t_x, rn2, EPS)
          tw.tap_h2 = h2
          r_logits = TinyNN.tnn_matmul(sess, wr, h2)          # [NE, T]
          gates    = TinyNN.tnn_softmax(sess, r_logits)       # [NE, T]
          TinyNN.tnn_set_output(gates)
          tw.t_gates = gates
          g1 = TinyNN.tnn_matmul(sess, sel1, gates)           # [1, T]
          g2 = TinyNN.tnn_matmul(sess, sel2, gates)           # [1, T]

          a1 = TinyNN.tnn_gelu(sess, TinyNN.tnn_matmul(sess, tw.pp[9], h2))   # [DFF,T]
          tw.tap_a1 = a1
          o1 = TinyNN.tnn_matmul(sess, tw.pp[10], a1)                          # [DM,T]
          a2 = TinyNN.tnn_gelu(sess, TinyNN.tnn_matmul(sess, tw.pp[11], h2))
          tw.tap_a2 = a2
          o2 = TinyNN.tnn_matmul(sess, tw.pp[12], a2)

          gated1 = TinyNN.tnn_mul(sess, o1, g1)   # broadcast [DM,T]*[1,T]
          gated2 = TinyNN.tnn_mul(sess, o2, g2)
          TinyNN.tnn_add(sess, t_x, TinyNN.tnn_add(sess, gated1, gated2))
        end

        # HARD top-1 MoE block: argmax routing + mul_mat_id dispatch.
        # eye = uploaded identity [NE,NE]; records taps + onehots + gates.
        # cut=1 (toy#121 bp-spine): the expert INPUT goes through
        # tnn_detach — forward-identity, gradient-opaque — so chain
        # grads reach attention/embeds via the residual + router
        # branches while the walker never needs mul_mat_id backward.
        def self.moe_block_top1(sess, tw, t_x, eye, cut)
          rn2 = tw.pp[7]
          wr  = tw.pp[8]
          h2 = Toy::LLM::Primitives::RMSNorm.build(sess, t_x, rn2, EPS)
          tw.tap_h2 = h2
          r_logits = TinyNN.tnn_matmul(sess, wr, h2)              # [NE, T]
          probs    = TinyNN.tnn_softmax(sess, r_logits)
          TinyNN.tnn_set_output(probs)
          tw.t_gates = probs
          ids   = TinyNN.tnn_argmax(sess, r_logits)               # I32 [T]
          ids2  = TinyNN.tnn_reshape_3d(sess, ids, 1, T, 1)       # [1, T]
          oneh  = TinyNN.tnn_get_rows(sess, eye, ids)             # [NE, T]
          TinyNN.tnn_set_output(oneh)
          tw.t_onehots = oneh
          gate  = TinyNN.tnn_sum_rows(sess, TinyNN.tnn_mul(sess, oneh, probs)) # [1,T]

          up_stack = TinyNN.tnn_concat(sess,
            TinyNN.tnn_reshape_3d(sess, tw.pp[9],  DM, DFF, 1),
            TinyNN.tnn_reshape_3d(sess, tw.pp[11], DM, DFF, 1), 2)   # [DM,DFF,NE]
          dn_stack = TinyNN.tnn_concat(sess,
            TinyNN.tnn_reshape_3d(sess, tw.pp[10], DFF, DM, 1),
            TinyNN.tnn_reshape_3d(sess, tw.pp[12], DFF, DM, 1), 2)   # [DFF,DM,NE]

          h_exp = h2
          if cut == 1
            h_exp = TinyNN.tnn_detach(sess, h2)
          end
          h3    = TinyNN.tnn_reshape_3d(sess, h_exp, DM, 1, T)
          upo   = TinyNN.tnn_mul_mat_id(sess, up_stack, h3, ids2)    # [DFF,1,T]
          a     = TinyNN.tnn_gelu(sess, upo)
          tw.tap_a1 = TinyNN.tnn_reshape_3d(sess, a, DFF, T, 1)      # routed acts [DFF,T]
          dno   = TinyNN.tnn_mul_mat_id(sess, dn_stack, a, ids2)     # [DM,1,T]
          eo    = TinyNN.tnn_reshape_3d(sess, dno, DM, T, 1)         # [DM,T]
          TinyNN.tnn_add(sess, t_x, TinyNN.tnn_mul(sess, eo, gate))
        end

        def self.forward_tower(sess, tw, t_tok, t_labels, sel1, sel2, eye, top1, cut)
          x = TinyNN.tnn_get_rows(sess, tw.pp[0], t_tok)
          atrip = attention_block(sess, x, tw.pp[2], tw.pp[3], tw.pp[4], tw.pp[5], tw.pp[6])
          x = atrip[0]
          tw.tap_ah  = atrip[1]
          tw.tap_ctx = atrip[2]
          if top1
            x = moe_block_top1(sess, tw, x, eye, cut)
          else
            x = moe_block(sess, tw, x, sel1, sel2)
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
          a_in_t = TinyNN.tnn_cont_2d(sess, TinyNN.tnn_transpose(sess, a_in), T, d_in)
          delt_t = TinyNN.tnn_cont_2d(sess, TinyNN.tnn_transpose(sess, delta), T, d_out)
          TinyNN.tnn_matmul(sess, a_in_t, delt_t)                       # [d_in, d_out]
        end

        def self.wire_chain(sess, tw, t_hp, idx)
          tg = TinyNN.tnn_tensor_grad(sess, tw.pp[idx])
          to = TinyNN.tnn_opt_step_adamw(sess, tw.pp[idx], tg, tw.pm[idx], tw.pv[idx], t_hp)
          TinyNN.tnn_extend_backward_graph(sess, to)
          0
        end

        # top1 lane-B wire: LATE-param (P0 idiom — tower B has zero params
        # at build_backward time) + optional routing mask on delta. No align
        # recording (no autodiff accs exist in this tower).
        def self.wire_dfa_top1(sess, tw, t_hp, idx, b, e, a_in, d_in, d_out, mask)
          delta = TinyNN.tnn_matmul(sess, b, e)
          if mask != TinyNN.tnn_null_ptr
            delta = TinyNN.tnn_mul(sess, delta, mask)
          end
          a_in_t = TinyNN.tnn_cont_2d(sess, TinyNN.tnn_transpose(sess, a_in), T, d_in)
          delt_t = TinyNN.tnn_cont_2d(sess, TinyNN.tnn_transpose(sess, delta), T, d_out)
          g = TinyNN.tnn_matmul(sess, a_in_t, delt_t)
          TinyNN.tnn_set_output(g)
          TinyNN.tnn_set_param(tw.pp[idx])
          to = TinyNN.tnn_opt_step_adamw(sess, tw.pp[idx], g, tw.pm[idx], tw.pv[idx], t_hp)
          TinyNN.tnn_extend_backward_graph(sess, to)
          0
        end

        def self.wire_dfa(sess, tw, t_hp, idx, b, e, a_in, d_in, d_out, name)
          g = dfa_grad(sess, b, e, a_in, d_in, d_out)
          TinyNN.tnn_set_output(g)
          to = TinyNN.tnn_opt_step_adamw(sess, tw.pp[idx], g, tw.pm[idx], tw.pv[idx], t_hp)
          TinyNN.tnn_extend_backward_graph(sess, to)
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
