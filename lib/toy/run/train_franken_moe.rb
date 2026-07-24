# lib/toy/run/train_franken_moe.rb — toy#109 P2b: the Franken-MoE arm.
#
# TWIN-LANE, ONE GRAPH (the P1 discipline: the global sched makes
# multi-session interleaving unsafe; two towers in one graph_b step in
# lockstep). Tower = embed → attention block → MoE-FFN block (residual)
# → final norm → tied logits. The MoE block is a DENSE 2-expert SOFT
# mixture — deliberately differentiable end-to-end (no mul_mat_id, no
# routed-token gather), so lane A (:chain everywhere, experts included)
# is a clean BP baseline:
#
#   h2      = rmsnorm(x, rn2)
#   gates   = softmax(Wr·h2)              [E, T]   (router)
#   a_i     = gelu(up_i·h2)               [DFF, T]
#   o_i     = down_i·a_i                  [DM, T]
#   out     = x + Σ_i o_i ⊙ g_i           g_i = sel_i·gates  [1, T]
#
# Lane B policy (FRANKEN_MOE env):
#   chain        — null config: byte-identical to lane A (parity gate)
#   dfa-experts  — the design-doc "MoE bonus" demonstration: router Wr
#                  (and everything else) stays :chain — its gradient
#                  flows through softmax × the gated combine, never
#                  through expert INTERNALS — while the four expert
#                  weights (up_i, down_i) update via per-matmul DFA:
#                  grad_up_i = (B·e)·h2ᵀ, grad_down_i = (B·e)·a_iᵀ,
#                  pure forward ops (P0 mechanics; B via Toy::Train::DfaB,
#                  FRANKEN_B_DIST/FRANKEN_B_SCALE axes).
#
# Shadow alignment telemetry on the four dfa weights (all weights stay
# in the autodiff param set — shadow-shaped build) + a router-health
# line (mean gate share of expert 0) per step:
#
#   align step=<s> w=<up1|down1|up2|down2> cos=<cos∠(g_dfa, g_chain)>
#   router step=<s> g0_mean=<mean gate[0] over T>
#
# Spinel hygiene as train_franken.rb (while loops, typed-empty seeds,
# monomorphic helpers, no default args, ENV nil-guards).

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
          attr_accessor :tap_h2, :tap_a1, :tap_a2
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
          TinyNN.tnn_add(sess, t_x, out)
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

        def self.forward_tower(sess, tw, t_tok, t_labels, sel1, sel2)
          x = TinyNN.tnn_get_rows(sess, tw.pp[0], t_tok)
          x = attention_block(sess, x, tw.pp[2], tw.pp[3], tw.pp[4], tw.pp[5], tw.pp[6])
          x = moe_block(sess, tw, x, sel1, sel2)
          xf  = Toy::LLM::Primitives::RMSNorm.build(sess, x, tw.pp[1], EPS)
          lgt = TinyNN.tnn_matmul(sess, tw.pp[0], xf)
          tw.t_logits = lgt
          tw.t_loss   = TinyNN.tnn_cross_entropy_loss(sess, lgt, t_labels)
          TinyNN.tnn_set_output(tw.t_loss)
          TinyNN.tnn_set_loss(tw.t_loss)
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

        def self.wire_dfa(sess, tw, t_hp, idx, b, e, a_in, d_in, d_out, name)
          g = dfa_grad(sess, b, e, a_in, d_in, d_out)
          TinyNN.tnn_set_output(g)
          to = TinyNN.tnn_opt_step_adamw(sess, tw.pp[idx], g, tw.pm[idx], tw.pv[idx], t_hp)
          TinyNN.tnn_extend_backward_graph(sess, to)
          tw.dfa_grads.push(g)
          tw.dfa_accs.push(TinyNN.tnn_tensor_grad(sess, tw.pp[idx]))
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

        def self.run
          mode_s = ENV["FRANKEN_MOE"] || ""
          if mode_s.length == 0
            mode_s = "dfa-experts"
          end
          dfa_experts = mode_s == "dfa-experts"
          steps_s = ENV["STEPS"] || ""
          steps = steps_s.length > 0 ? steps_s.to_i : 40
          seed_s = ENV["FRANKEN_SEED"] || ""
          seed = seed_s.length > 0 ? seed_s.to_i : 1234
          lr = 0.02

          puts "franken-moe twin-lane: experts=" + NE.to_s + " policy_b=" + mode_s +
               " steps=" + steps.to_s + " seed=" + seed.to_s

          sess = TinyNN.tnn_session_new(0)
          TinyNN.tnn_session_set_graph_capacity(sess, 262144)

          tow_a = alloc_tower(sess)
          tow_b = alloc_tower(sess)

          # dfa B mats for lane B's four expert weights (idx 9..12):
          # ne=[VOCAB, d_out] via (rows=d_out, cols=VOCAB).
          b_up1   = TinyNN.tnn_input_2d_f32_persistent(sess, DFF, VOCAB)
          b_down1 = TinyNN.tnn_input_2d_f32_persistent(sess, DM,  VOCAB)
          b_up2   = TinyNN.tnn_input_2d_f32_persistent(sess, DFF, VOCAB)
          b_down2 = TinyNN.tnn_input_2d_f32_persistent(sess, DM,  VOCAB)
          # one-hot row selectors [1, NE]
          sel1 = TinyNN.tnn_input_2d_f32_persistent(sess, 1, NE)
          sel2 = TinyNN.tnn_input_2d_f32_persistent(sess, 1, NE)

          gi = 0
          while gi < tow_a.pp.length
            TinyNN.tnn_set_param(tow_a.pp[gi])
            TinyNN.tnn_set_param(tow_b.pp[gi])
            gi = gi + 1
          end
          TinyNN.tnn_finalize_weights(sess)

          gi = 0
          while gi < tow_a.pp.length
            n = TinyNN.tnn_tensor_nelements(tow_a.pp[gi])
            vals = fillv(n, gi * 7 + 1)
            TinyNN.tnn_upload_from_float_array(sess, tow_a.pp[gi], vals, n)
            TinyNN.tnn_upload_from_float_array(sess, tow_b.pp[gi], vals, n)
            TinyNN.tnn_zero_tensor(sess, tow_a.pm[gi]); TinyNN.tnn_zero_tensor(sess, tow_a.pv[gi])
            TinyNN.tnn_zero_tensor(sess, tow_b.pm[gi]); TinyNN.tnn_zero_tensor(sess, tow_b.pv[gi])
            gi = gi + 1
          end
          dist_c = b_dist_code
          sig_up   = Toy::Train::DfaB.sigma_for(Toy::Train::DfaB::SCALE_INV_SQRT_FAN, VOCAB, DFF, 0.0)
          sig_down = Toy::Train::DfaB.sigma_for(Toy::Train::DfaB::SCALE_INV_SQRT_FAN, VOCAB, DM, 0.0)
          TinyNN.tnn_upload_from_float_array(sess, b_up1,
            Toy::Train::DfaB.fill(DFF * VOCAB, seed + 11, dist_c, sig_up), DFF * VOCAB)
          TinyNN.tnn_upload_from_float_array(sess, b_down1,
            Toy::Train::DfaB.fill(DM * VOCAB, seed + 12, dist_c, sig_down), DM * VOCAB)
          TinyNN.tnn_upload_from_float_array(sess, b_up2,
            Toy::Train::DfaB.fill(DFF * VOCAB, seed + 13, dist_c, sig_up), DFF * VOCAB)
          TinyNN.tnn_upload_from_float_array(sess, b_down2,
            Toy::Train::DfaB.fill(DM * VOCAB, seed + 14, dist_c, sig_down), DM * VOCAB)
          s1 = [1.0, 0.0]
          s2 = [0.0, 1.0]
          TinyNN.tnn_upload_from_float_array(sess, sel1, s1, NE)
          TinyNN.tnn_upload_from_float_array(sess, sel2, s2, NE)

          t_tok    = TinyNN.tnn_input_1d_i32(sess, T)
          t_labels = TinyNN.tnn_input_2d_f32(sess, T, VOCAB)
          t_hp     = TinyNN.tnn_input_1d_f32(sess, 7)
          forward_tower(sess, tow_a, t_tok, t_labels, sel1, sel2)
          forward_tower(sess, tow_b, t_tok, t_labels, sel1, sel2)

          # P1 finding: extra roots FIRST, then the realizing call.
          TinyNN.tnn_add_to_graph(sess, tow_a.t_loss)
          TinyNN.tnn_build_forward_only(sess, tow_b.t_loss)
          TinyNN.tnn_build_backward(sess)

          # lane A: all chain
          idx = 0
          while idx < tow_a.pp.length
            wire_chain(sess, tow_a, t_hp, idx)
            idx = idx + 1
          end
          # lane B: chain everywhere except (policy) the four expert weights
          p_sm = TinyNN.tnn_softmax(sess, tow_b.t_logits)
          e_b  = TinyNN.tnn_scale(sess, TinyNN.tnn_sub(sess, p_sm, t_labels), 1.0 / T.to_f)
          idx = 0
          while idx < 9
            wire_chain(sess, tow_b, t_hp, idx)
            idx = idx + 1
          end
          if dfa_experts
            wire_dfa(sess, tow_b, t_hp, 9,  b_up1,   e_b, tow_b.tap_h2, DM,  DFF, "w=up1")
            wire_dfa(sess, tow_b, t_hp, 10, b_down1, e_b, tow_b.tap_a1, DFF, DM,  "w=down1")
            wire_dfa(sess, tow_b, t_hp, 11, b_up2,   e_b, tow_b.tap_h2, DM,  DFF, "w=up2")
            wire_dfa(sess, tow_b, t_hp, 12, b_down2, e_b, tow_b.tap_a2, DFF, DM,  "w=down2")
          else
            idx = 9
            while idx < 13
              wire_chain(sess, tow_b, t_hp, idx)
              idx = idx + 1
            end
          end

          TinyNN.tnn_pin_all_graph_b_nodes(sess)
          TinyNN.tnn_realize_backward(sess)

          ids = [1, 2, 3, 4]
          labels = zeros(VOCAB * T)
          tt = 0
          while tt < T
            tgt = (ids[tt] + 1) % VOCAB
            labels[tgt + VOCAB * tt] = 1.0
            tt = tt + 1
          end

          n_dfa = tow_b.dfa_grads.length
          gbuf_up   = zeros(DM * DFF)
          abuf_up   = zeros(DM * DFF)
          gates_buf = zeros(NE * T)
          first_a = 0.0; last_a = 0.0; first_b = 0.0; last_b = 0.0
          b1 = 0.9; b2 = 0.95
          s = 0
          while s < steps
            if s == 0
              TinyNN.tnn_graph_reset(sess)
            else
              TinyNN.tnn_graph_reset_grads_only(sess)
            end
            t = (s + 1).to_f
            hp = [lr, b1, b2, 1.0e-8, 0.0,
                  1.0 / (1.0 - (b1 ** t)), 1.0 / (1.0 - (b2 ** t))]
            TinyNN.upload_int_array(sess, t_tok, ids)
            TinyNN.tnn_upload_from_float_array(sess, t_labels, labels, VOCAB * T)
            TinyNN.tnn_upload_from_float_array(sess, t_hp, hp, 7)
            TinyNN.tnn_compute_backward(sess)
            TinyNN.tnn_download(sess, tow_a.t_loss)
            la = TinyNN.tnn_scratch_get(sess, 0)
            TinyNN.tnn_download(sess, tow_b.t_loss)
            lb = TinyNN.tnn_scratch_get(sess, 0)
            if s == 0
              first_a = la; first_b = lb
            end
            last_a = la; last_b = lb
            puts "step " + s.to_s + ": lane_a=" + la.to_s + " lane_b=" + lb.to_s
            di = 0
            while di < n_dfa
              nw = TinyNN.tnn_tensor_nelements(tow_b.dfa_grads[di])
              TinyNN.tnn_download_to_f64_array(sess, tow_b.dfa_grads[di], gbuf_up, nw)
              TinyNN.tnn_download_to_f64_array(sess, tow_b.dfa_accs[di], abuf_up, nw)
              # cosv over the first nw entries (buffers are max-size).
              dot = 0.0; na = 0.0; nb2 = 0.0
              ii = 0
              while ii < nw
                dot = dot + gbuf_up[ii] * abuf_up[ii]
                na = na + gbuf_up[ii] * gbuf_up[ii]
                nb2 = nb2 + abuf_up[ii] * abuf_up[ii]
                ii = ii + 1
              end
              den = Math.sqrt(na) * Math.sqrt(nb2)
              cv = 0.0
              if den > 0.0
                cv = dot / den
              end
              puts "align step=" + s.to_s + " " + tow_b.dfa_names[di] + " cos=" + cv.to_s
              di = di + 1
            end
            TinyNN.tnn_download_to_f64_array(sess, tow_b.t_gates, gates_buf, NE * T)
            gsum = 0.0
            ti = 0
            while ti < T
              gsum = gsum + gates_buf[ti * NE]
              ti = ti + 1
            end
            puts "router step=" + s.to_s + " g0_mean=" + (gsum / T.to_f).to_s
            s = s + 1
          end

          puts "franken-moe summary: lane_a " + first_a.to_s + " -> " + last_a.to_s +
               " | lane_b " + first_b.to_s + " -> " + last_b.to_s
          if last_a != last_a || last_b != last_b
            puts "FRANKEN-MOE FAIL: NaN loss"
            return
          end
          puts "FRANKEN-MOE DONE"
        end
      end
    end
  end
end

Toy::LLM::Run::TrainFrankenMoe.run
