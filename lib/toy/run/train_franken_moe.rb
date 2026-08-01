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
# ROUTING (FRANKEN_MOE_ROUTING env, toy#109 hard-routed leg):
#   dense (default) — the P2b soft mixture, byte-identical to before.
#   top1            — HARD routing via in-graph argmax + mul_mat_id
#                     (tnn_argmax is the program's first new shim bind).
#                     Lane B is then FULLY DFA (attention q/k/v/o +
#                     router + routing-masked experts; embed/norms
#                     frozen): mul_mat_id has NO ggml backward, so no
#                     grad path may touch it — zero tower-B params at
#                     build_backward time keeps the walker away
#                     (grads_needed short-circuit), and the DFA updates
#                     are forward ops with LATE-param opt nodes (P0
#                     idiom). Expert updates mask to their ROUTED
#                     tokens via get_rows(I_E, ids) one-hots. Router
#                     grads: none from chain — router is DFA too
#                     (B_r: vocab→E). Lane A stays the dense
#                     all-chain baseline; per-expert utilization is
#                     emitted per step ("route step=N shares=...").
#
# LOAD-BALANCING AUX LOSS (FRANKEN_MOE_AUX env, top1 only; the router-
# collapse fix — without it the DFA-fed top1 router collapses to one
# expert by ~step 20):
#   Switch-style L_aux = α·NE·Σ_e f_e·P̄_e, where f_e = routed-token
#   fraction (non-differentiable — computed CPU-side from the previous
#   step's one-hots and re-uploaded per step as a plain graph input,
#   the toy#117 labels contract; uniform at step 0 — with this rig's
#   fixed batch the lag-1 f equals the current f after step 0) and
#   P̄_e = mean router prob (differentiable). The aux is a second LOSS
#   root: Wr joins the autodiff param set (the ONLY tower-B param —
#   the aux backward path is Wr→logits→softmax→mean, nowhere near
#   mul_mat_id; argmax's output is I32 so the walker allocates no grad
#   through it), and the router update becomes
#   g_total = g_dfa(task) + g_bp(aux) into one opt_step_adamw.
#   α=0 (default) builds the exact pre-aux graph — byte-null.
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

require_relative "franken_moe_parts"

module Toy
  module LLM
    module Run
      module TrainFrankenMoe
        def self.run
          mode_s = ENV["FRANKEN_MOE"] || ""
          if mode_s.length == 0
            mode_s = "dfa-experts"
          end
          dfa_experts = mode_s == "dfa-experts"
          routing_s = ENV["FRANKEN_MOE_ROUTING"] || ""
          top1 = routing_s == "top1"
          aux_s = ENV["FRANKEN_MOE_AUX"] || ""
          aux_alpha = aux_s.length > 0 ? aux_s.to_f : 0.0
          aux_on = top1 && aux_alpha > 0.0
          steps_s = ENV["STEPS"] || ""
          steps = steps_s.length > 0 ? steps_s.to_i : 40
          seed_s = ENV["FRANKEN_SEED"] || ""
          seed = seed_s.length > 0 ? seed_s.to_i : 1234
          lr = 0.02

          puts "franken-moe twin-lane: experts=" + NE.to_s + " policy_b=" + mode_s +
               " routing=" + (top1 ? "top1" : "dense") + " aux=" + aux_alpha.to_s +
               " steps=" + steps.to_s + " seed=" + seed.to_s

          shape_init(DM_BASE, DFF_BASE, VOCAB, NE, T)   # the rig IS the base instrument
          attn_gate_init(0)                      # toy#136 K1.1: rig = ungated
          opt_init(0)                            # toy#139: the rig is AdamW-only
          latent_init(0, 0)                      # toy#142 K4: rig = no latent, no shared
          sess = TinyNN.tnn_session_new(0)
          TinyNN.tnn_session_set_graph_capacity(sess, 262144)

          tow_a = alloc_tower(sess)
          tow_b = alloc_tower(sess)

          # dfa B mats for lane B's four expert weights (idx 9..12):
          # ne=[VOCAB, d_out] via (rows=d_out, cols=VOCAB).
          b_up1   = TinyNN.tnn_input_2d_f32_persistent(sess, dfv, VOCAB)
          b_down1 = TinyNN.tnn_input_2d_f32_persistent(sess, dmv,  VOCAB)
          b_up2   = TinyNN.tnn_input_2d_f32_persistent(sess, dfv, VOCAB)
          b_down2 = TinyNN.tnn_input_2d_f32_persistent(sess, dmv,  VOCAB)
          # one-hot row selectors [1, NE]
          sel1 = TinyNN.tnn_input_2d_f32_persistent(sess, 1, NE)
          sel2 = TinyNN.tnn_input_2d_f32_persistent(sess, 1, NE)
          # top1 extras: identity table + B mats for the fully-DFA lane B
          # (attention q/k/v/o + router; expert B mats reuse b_up/b_down).
          eye   = TinyNN.tnn_input_2d_f32_persistent(sess, NE, NE)
          b_aq  = TinyNN.tnn_input_2d_f32_persistent(sess, dmv, VOCAB)
          b_ak  = TinyNN.tnn_input_2d_f32_persistent(sess, dmv, VOCAB)
          b_av  = TinyNN.tnn_input_2d_f32_persistent(sess, dmv, VOCAB)
          b_ao  = TinyNN.tnn_input_2d_f32_persistent(sess, dmv, VOCAB)
          b_r   = TinyNN.tnn_input_2d_f32_persistent(sess, NE, VOCAB)
          # aux-loss constants: ones column for the sum-over-T contraction
          # (persistent = real storage; the per-step f vector is a plain
          # graph input, declared below with tok/labels).
          ones_t = TinyNN.tnn_input_2d_f32_persistent(sess, 1, T)   # ne=[T,1]
          # toy#139: ggml's SGD step takes exactly [alpha, wd]; muon and
          # sgd both apply through it. PERSISTENT (alloc BEFORE
          # finalize_weights) — a compute-context input gets no backing
          # buffer when the graph does not reach it, and the per-step
          # upload then aborts inside ggml_backend_tensor_set.
          t_hp_sgd = TinyNN.tnn_input_1d_f32_persistent(sess, 2)

          gi = 0
          while gi < tow_a.pp.length
            TinyNN.tnn_set_param(tow_a.pp[gi])
            # top1: tower B gets ZERO params at build_backward time — the
            # backward walker must never need a grad through mul_mat_id
            # (no ggml backward case). DFA opt nodes late-param instead.
            if !top1
              TinyNN.tnn_set_param(tow_b.pp[gi])
            end
            gi = gi + 1
          end
          if aux_on
            # Wr is the sole tower-B autodiff param: the aux backward
            # stops at Wr; everything else stays late-param DFA.
            TinyNN.tnn_set_param(tow_b.pp[8])
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
          sig_up   = Toy::Train::DfaB.sigma_for(Toy::Train::DfaB::SCALE_INV_SQRT_FAN, VOCAB, dfv, 0.0)
          sig_down = Toy::Train::DfaB.sigma_for(Toy::Train::DfaB::SCALE_INV_SQRT_FAN, VOCAB, dmv, 0.0)
          TinyNN.tnn_upload_from_float_array(sess, b_up1,
            Toy::Train::DfaB.fill(dfv * VOCAB, seed + 11, dist_c, sig_up), dfv * VOCAB)
          TinyNN.tnn_upload_from_float_array(sess, b_down1,
            Toy::Train::DfaB.fill(dmv * VOCAB, seed + 12, dist_c, sig_down), dmv * VOCAB)
          TinyNN.tnn_upload_from_float_array(sess, b_up2,
            Toy::Train::DfaB.fill(dfv * VOCAB, seed + 13, dist_c, sig_up), dfv * VOCAB)
          TinyNN.tnn_upload_from_float_array(sess, b_down2,
            Toy::Train::DfaB.fill(dmv * VOCAB, seed + 14, dist_c, sig_down), dmv * VOCAB)
          s1 = [1.0, 0.0]
          s2 = [0.0, 1.0]
          TinyNN.tnn_upload_from_float_array(sess, sel1, s1, NE)
          TinyNN.tnn_upload_from_float_array(sess, sel2, s2, NE)
          ey = [1.0, 0.0, 0.0, 1.0]
          TinyNN.tnn_upload_from_float_array(sess, eye, ey, NE * NE)
          if top1
            sig_dm = Toy::Train::DfaB.sigma_for(Toy::Train::DfaB::SCALE_INV_SQRT_FAN, VOCAB, dmv, 0.0)
            sig_e  = Toy::Train::DfaB.sigma_for(Toy::Train::DfaB::SCALE_INV_SQRT_FAN, VOCAB, NE, 0.0)
            TinyNN.tnn_upload_from_float_array(sess, b_aq, Toy::Train::DfaB.fill(dmv * VOCAB, seed + 21, dist_c, sig_dm), dmv * VOCAB)
            TinyNN.tnn_upload_from_float_array(sess, b_ak, Toy::Train::DfaB.fill(dmv * VOCAB, seed + 22, dist_c, sig_dm), dmv * VOCAB)
            TinyNN.tnn_upload_from_float_array(sess, b_av, Toy::Train::DfaB.fill(dmv * VOCAB, seed + 23, dist_c, sig_dm), dmv * VOCAB)
            TinyNN.tnn_upload_from_float_array(sess, b_ao, Toy::Train::DfaB.fill(dmv * VOCAB, seed + 24, dist_c, sig_dm), dmv * VOCAB)
            TinyNN.tnn_upload_from_float_array(sess, b_r,  Toy::Train::DfaB.fill(NE * VOCAB, seed + 25, dist_c, sig_e),  NE * VOCAB)
            onesv = zeros(T)
            oi = 0
            while oi < T
              onesv[oi] = 1.0
              oi = oi + 1
            end
            TinyNN.tnn_upload_from_float_array(sess, ones_t, onesv, T)
          end

          t_tok    = TinyNN.tnn_input_1d_i32(sess, T)
          t_labels = TinyNN.tnn_input_2d_f32(sess, T, VOCAB)
          t_hp     = TinyNN.tnn_input_1d_f32(sess, 7)
          t_f      = TinyNN.tnn_input_2d_f32(sess, 1, NE)   # ne=[NE,1]
          forward_tower(sess, tow_a, t_tok, t_labels, [sel1, sel2], eye, false, 0, TinyNN.tnn_null_ptr, TinyNN.tnn_null_ptr, 0)
          forward_tower(sess, tow_b, t_tok, t_labels, [sel1, sel2], eye, top1, 0, TinyNN.tnn_null_ptr, TinyNN.tnn_null_ptr, 0)

          # aux loss: Σ_e Σ_t f'_e·probs[e,t] with the α·NE/T scaling
          # folded into the uploaded f' vector; [NE,1] broadcasts over T,
          # sum_rows + ones-contraction scalarize it (all ops have ggml
          # backwards; nothing here touches the argmax/mul_mat_id path).
          t_aux = TinyNN.tnn_null_ptr
          if aux_on
            m_fp  = TinyNN.tnn_mul(sess, tow_b.t_gates, t_f)      # [NE,T]
            s_tok = TinyNN.tnn_sum_rows(sess, m_fp)               # [1,T]
            s_col = TinyNN.tnn_reshape_3d(sess, s_tok, T, 1, 1)   # [T,1]
            t_aux = TinyNN.tnn_matmul(sess, s_col, ones_t)        # scalar
            TinyNN.tnn_set_output(t_aux)
            TinyNN.tnn_set_loss(t_aux)
          end

          # P1 finding: extra roots FIRST, then the realizing call.
          TinyNN.tnn_add_to_graph(sess, tow_a.t_loss)
          if aux_on
            TinyNN.tnn_add_to_graph(sess, t_aux)
          end
          TinyNN.tnn_build_forward_only(sess, tow_b.t_loss)
          TinyNN.tnn_build_backward(sess)

          # lane A: all chain
          idx = 0
          while idx < tow_a.pp.length
            wire_chain(sess, tow_a, t_hp, t_hp_sgd, idx)
            idx = idx + 1
          end
          # lane B: chain everywhere except (policy) the four expert weights
          p_sm = TinyNN.tnn_softmax(sess, tow_b.t_logits)
          e_b  = TinyNN.tnn_scale(sess, TinyNN.tnn_sub(sess, p_sm, t_labels), 1.0 / T.to_f)
          if top1
            # FULLY-DFA tower B: attention + router + routed-masked experts;
            # embed/norms frozen. Routing masks from the recorded one-hots.
            np2 = TinyNN.tnn_null_ptr
            wire_dfa_top1(sess, tow_b, t_hp, t_hp_sgd, 3, b_aq, e_b, tow_b.tap_ah,  dmv, dmv, np2)
            wire_dfa_top1(sess, tow_b, t_hp, t_hp_sgd, 4, b_ak, e_b, tow_b.tap_ah,  dmv, dmv, np2)
            wire_dfa_top1(sess, tow_b, t_hp, t_hp_sgd, 5, b_av, e_b, tow_b.tap_ah,  dmv, dmv, np2)
            wire_dfa_top1(sess, tow_b, t_hp, t_hp_sgd, 6, b_ao, e_b, tow_b.tap_ctx, dmv, dmv, np2)
            if aux_on
              # router = DFA task signal + BP aux signal, one opt step.
              # Wr is an EARLY param (set before build_backward) — no
              # late set_param here; its autodiff acc carries the aux
              # grad only (lane-B CE is not a loss root).
              g_dfa_r = dfa_grad(sess, b_r, e_b, tow_b.tap_h2, dmv, NE)
              acc_r   = TinyNN.tnn_tensor_grad(sess, tow_b.pp[8])
              g_tot   = TinyNN.tnn_add(sess, g_dfa_r, acc_r)
              TinyNN.tnn_set_output(g_tot)
              to_r = TinyNN.tnn_opt_step_adamw(sess, tow_b.pp[8], g_tot, tow_b.pm[8], tow_b.pv[8], t_hp)
              TinyNN.tnn_extend_backward_graph(sess, to_r)
            else
              wire_dfa_top1(sess, tow_b, t_hp, t_hp_sgd, 8, b_r, e_b, tow_b.tap_h2, dmv, NE, np2)
            end
            m1 = TinyNN.tnn_matmul(sess, sel1, tow_b.t_onehots)   # [1,T]
            m2 = TinyNN.tnn_matmul(sess, sel2, tow_b.t_onehots)
            wire_dfa_top1(sess, tow_b, t_hp, t_hp_sgd, 9,  b_up1,   e_b, tow_b.tap_h2, dmv,  dfv, m1)
            wire_dfa_top1(sess, tow_b, t_hp, t_hp_sgd, 10, b_down1, e_b, tow_b.tap_a1, dfv, dmv,  m1)
            wire_dfa_top1(sess, tow_b, t_hp, t_hp_sgd, 11, b_up2,   e_b, tow_b.tap_h2, dmv,  dfv, m2)
            wire_dfa_top1(sess, tow_b, t_hp, t_hp_sgd, 12, b_down2, e_b, tow_b.tap_a1, dfv, dmv,  m2)
          else
          idx = 0
          while idx < 9
            wire_chain(sess, tow_b, t_hp, t_hp_sgd, idx)
            idx = idx + 1
          end
          if dfa_experts
            wire_dfa(sess, tow_b, t_hp, t_hp_sgd, 9,  b_up1,   e_b, tow_b.tap_h2, dmv,  dfv, "w=up1")
            wire_dfa(sess, tow_b, t_hp, t_hp_sgd, 10, b_down1, e_b, tow_b.tap_a1, dfv, dmv,  "w=down1")
            wire_dfa(sess, tow_b, t_hp, t_hp_sgd, 11, b_up2,   e_b, tow_b.tap_h2, dmv,  dfv, "w=up2")
            wire_dfa(sess, tow_b, t_hp, t_hp_sgd, 12, b_down2, e_b, tow_b.tap_a2, dfv, dmv,  "w=down2")
          else
            idx = 9
            while idx < 13
              wire_chain(sess, tow_b, t_hp, t_hp_sgd, idx)
              idx = idx + 1
            end
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
          gbuf_up   = zeros(dmv * dfv)
          abuf_up   = zeros(dmv * dfv)
          gates_buf = zeros(NE * T)
          # f' vector for the aux loss: α·NE·f_e/T, uniform before the
          # first compute, then lag-1 from the routed one-hots.
          fvec = zeros(NE)
          fi = 0
          while fi < NE
            fvec[fi] = aux_alpha / T.to_f
            fi = fi + 1
          end
          aux_buf = zeros(1)
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
            if aux_on
              TinyNN.tnn_upload_from_float_array(sess, t_f, fvec, NE)
            end
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
            if top1
              TinyNN.tnn_download_to_f64_array(sess, tow_b.t_onehots, gates_buf, NE * T)
              e0 = 0.0
              ti2 = 0
              while ti2 < T
                e0 = e0 + gates_buf[ti2 * NE]
                ti2 = ti2 + 1
              end
              puts "route step=" + s.to_s + " e0_share=" + (e0 / T.to_f).to_s
              if aux_on
                # lag-1 f' from this step's routing (α·NE·count_e/T²)
                ei = 0
                while ei < NE
                  cnt = 0.0
                  ti3 = 0
                  while ti3 < T
                    cnt = cnt + gates_buf[ti3 * NE + ei]
                    ti3 = ti3 + 1
                  end
                  fvec[ei] = aux_alpha * NE.to_f * cnt / (T.to_f * T.to_f)
                  ei = ei + 1
                end
                TinyNN.tnn_download_to_f64_array(sess, t_aux, aux_buf, 1)
                puts "auxbal step=" + s.to_s + " loss=" + aux_buf[0].to_s
              end
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
