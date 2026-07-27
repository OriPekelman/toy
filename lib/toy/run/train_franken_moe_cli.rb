# lib/toy/run/train_franken_moe_cli.rb — toy#120: the spec-callable
# Franken-MoE runner (`toy train franken-moe`, the F4 surface). SINGLE
# lane at the rig shape (franken_moe_parts.rb builders), under the
# TAO_RUN_DIR contract (toy#112 precedent).
#
# ENV (the CLI's controlled env):
#   STEPS / SEED / TAO_RUN_DIR / TOY_RUN_ID       — as train.rb
#   FRANKEN_MOE_ROUTING  dense (default) | top1
#   FRANKEN_MOE          dense: chain (default) | dfa-experts
#                        top1:  "" (fully-DFA lane, the F4 arm) |
#                               bp-router-dfa-experts (toy#121, the F5
#                               core: task-BP router credit through the
#                               p-scale edge — the forward is IDENTICAL
#                               to the fully-DFA lane (out = x +
#                               p_sel*expert_out, already Switch-scaled);
#                               ONLY credit flow changes. Wr's acc
#                               accumulates task-BP + aux-BP (two loss
#                               roots, one autodiff acc); experts and
#                               attention stay per-matmul DFA; the
#                               backward walker still never needs
#                               mul_mat_id (same grads_needed
#                               short-circuit as the aux path).
#   FRANKEN_MOE_AUX      <alpha> (top1 only; load-balancing aux-loss)
#   FRANKEN_B_SEED / FRANKEN_B_DIST / FRANKEN_B_SCALE — DfaB axes
#   FRANKEN_ALIGN=1      opt-in align events (dense dfa-experts only:
#                        top1 has no autodiff accs to compare against)
#
# SEED seeds the weight-init stream (seed*131 mixed into the rig's
# fillv); SEED=0 reproduces the rig's fixed init BYTE-exactly, so the
# dense-chain arm at SEED=0 byte-equals the rig's lane A — the gate's
# cross-binary null anchor.
#
# EVENTS (toy/v1, additive per toy#120):
#   run_start.franken_moe = {routing, policy, aux_alpha, b_seed,
#                            b_dist, b_scale, b_sigma}
#   step  {step, loss}                        — loss num_or_null (#118)
#   route {step, shares:[e0,e1], g0_mean, aux} — every step, both
#         routings (dense: shares = mean soft gate per expert; top1:
#         hard one-hot token shares; aux null unless top1 aux>0)
#   align {step, w, cos, dfa_norm, bp_norm}   — dense dfa-experts + opt-in
#   run_end; flow.json post-build. NO checkpoint: the MoE instrument
#   has no GGUF writer (the vit-tiny #169 precedent).
#
# top1 + aux: the aux loss graph is ALWAYS built in top1 mode (Wr is
# the sole autodiff param); FRANKEN_MOE_AUX=0 uploads an all-zero f'
# vector — a VALUE-null (zero aux gradient, router update = pure DFA),
# not a graph-null. The graph-null anchor stays pinned rig-side
# (gate-franken-moe); collapse behavior is gated CLI-side.

require_relative "franken_moe_parts"
require_relative "../io/json_builder"
require_relative "../io/json"
require_relative "../io/toy_events"
require_relative "../dev/toy_describe_flow"

module Toy
  module LLM
    module Run
      module TrainFrankenMoe
        def self.num_or_null_cli(x)
          d = x - x
          if d == 0.0
            x.to_s
          else
            "null"
          end
        end

        def self.run_cli
          steps_s = ENV["STEPS"] || ""
          steps = steps_s.length > 0 ? steps_s.to_i : 40
          seed_s = ENV["SEED"] || ""
          seed = seed_s.length > 0 ? seed_s.to_i : 0
          run_dir = ENV["TAO_RUN_DIR"] || ""
          run_id_s = ENV["TOY_RUN_ID"] || ""
          routing_s = ENV["FRANKEN_MOE_ROUTING"] || ""
          top1 = routing_s == "top1"
          mode_s = ENV["FRANKEN_MOE"] || ""
          if mode_s.length == 0
            mode_s = "chain"
          end
          dfa_experts = mode_s == "dfa-experts"
          bp_router = top1 && mode_s == "bp-router-dfa-experts"
          bp_spine  = top1 && mode_s == "bp-spine"
          aux_s = ENV["FRANKEN_MOE_AUX"] || ""
          aux_alpha = aux_s.length > 0 ? aux_s.to_f : 0.0
          bseed_s = ENV["FRANKEN_B_SEED"] || ""
          b_seed = bseed_s.length > 0 ? bseed_s.to_i : 1234
          align_on = (ENV["FRANKEN_ALIGN"] || "") == "1"
          events = run_dir.length > 0 ? (run_dir + "/events.jsonl") : ""
          lr = 0.02

          pol_name = mode_s
          if top1
            pol_name = "top1-dfa"
            if bp_router
              pol_name = "bp-router-dfa-experts"
            end
            if bp_spine
              pol_name = "bp-spine"
            end
          end
          puts "franken-moe: routing=" + (top1 ? "top1" : "dense") +
               " policy=" + pol_name +
               " aux=" + aux_alpha.to_s + " steps=" + steps.to_s +
               " seed=" + seed.to_s + " b_seed=" + b_seed.to_s

          shape_s = ENV["FRANKEN_SHAPE"] || "base"
          if shape_s != "base" && shape_s != "wide"
            puts "unknown FRANKEN_SHAPE: " + shape_s + " (franken-moe takes base|wide)"
            return 1
          end
          if shape_s == "wide"
            shape_init(256, 512)
          else
            shape_init(DM_BASE, DFF_BASE)
          end

          sess = TinyNN.tnn_session_new(0)
          TinyNN.tnn_session_set_graph_capacity(sess, 262144)

          tw = alloc_tower(sess)

          b_up1   = TinyNN.tnn_input_2d_f32_persistent(sess, dfv, VOCAB)
          b_down1 = TinyNN.tnn_input_2d_f32_persistent(sess, dmv,  VOCAB)
          b_up2   = TinyNN.tnn_input_2d_f32_persistent(sess, dfv, VOCAB)
          b_down2 = TinyNN.tnn_input_2d_f32_persistent(sess, dmv,  VOCAB)
          sel1 = TinyNN.tnn_input_2d_f32_persistent(sess, 1, NE)
          sel2 = TinyNN.tnn_input_2d_f32_persistent(sess, 1, NE)
          eye   = TinyNN.tnn_input_2d_f32_persistent(sess, NE, NE)
          b_aq  = TinyNN.tnn_input_2d_f32_persistent(sess, dmv, VOCAB)
          b_ak  = TinyNN.tnn_input_2d_f32_persistent(sess, dmv, VOCAB)
          b_av  = TinyNN.tnn_input_2d_f32_persistent(sess, dmv, VOCAB)
          b_ao  = TinyNN.tnn_input_2d_f32_persistent(sess, dmv, VOCAB)
          b_r   = TinyNN.tnn_input_2d_f32_persistent(sess, NE, VOCAB)
          ones_t = TinyNN.tnn_input_2d_f32_persistent(sess, 1, T)   # ne=[T,1]

          # params: dense = every weight (shadow-shaped when dfa-experts);
          # top1 = Wr ONLY (the aux backward path; everything else is
          # late-param DFA — the walker must never need mul_mat_id).
          gi = 0
          while gi < tw.pp.length
            # dense: every weight. bp-spine: the whole spine 0..8 (embed,
            # fnorm, attention, rn2, Wr) — experts 9..12 stay late-param
            # DFA behind the detach cut.
            if !top1 || (bp_spine && gi < 9)
              TinyNN.tnn_set_param(tw.pp[gi])
            end
            gi = gi + 1
          end
          if top1 && !bp_spine
            TinyNN.tnn_set_param(tw.pp[8])
          end
          TinyNN.tnn_finalize_weights(sess)

          # weight init: rig stream, seed-mixed (seed=0 == rig lane A)
          gi = 0
          while gi < tw.pp.length
            n = TinyNN.tnn_tensor_nelements(tw.pp[gi])
            vals = fillv(n, seed * 131 + gi * 7 + 1)
            TinyNN.tnn_upload_from_float_array(sess, tw.pp[gi], vals, n)
            TinyNN.tnn_zero_tensor(sess, tw.pm[gi])
            TinyNN.tnn_zero_tensor(sess, tw.pv[gi])
            gi = gi + 1
          end
          dist_c = b_dist_code
          sig_up   = Toy::Train::DfaB.sigma_for(Toy::Train::DfaB::SCALE_INV_SQRT_FAN, VOCAB, dfv, 0.0)
          sig_down = Toy::Train::DfaB.sigma_for(Toy::Train::DfaB::SCALE_INV_SQRT_FAN, VOCAB, dmv, 0.0)
          TinyNN.tnn_upload_from_float_array(sess, b_up1,
            Toy::Train::DfaB.fill(dfv * VOCAB, b_seed + 11, dist_c, sig_up), dfv * VOCAB)
          TinyNN.tnn_upload_from_float_array(sess, b_down1,
            Toy::Train::DfaB.fill(dmv * VOCAB, b_seed + 12, dist_c, sig_down), dmv * VOCAB)
          TinyNN.tnn_upload_from_float_array(sess, b_up2,
            Toy::Train::DfaB.fill(dfv * VOCAB, b_seed + 13, dist_c, sig_up), dfv * VOCAB)
          TinyNN.tnn_upload_from_float_array(sess, b_down2,
            Toy::Train::DfaB.fill(dmv * VOCAB, b_seed + 14, dist_c, sig_down), dmv * VOCAB)
          s1 = [1.0, 0.0]
          s2 = [0.0, 1.0]
          TinyNN.tnn_upload_from_float_array(sess, sel1, s1, NE)
          TinyNN.tnn_upload_from_float_array(sess, sel2, s2, NE)
          ey = [1.0, 0.0, 0.0, 1.0]
          TinyNN.tnn_upload_from_float_array(sess, eye, ey, NE * NE)
          if top1
            sig_dm = Toy::Train::DfaB.sigma_for(Toy::Train::DfaB::SCALE_INV_SQRT_FAN, VOCAB, dmv, 0.0)
            sig_e  = Toy::Train::DfaB.sigma_for(Toy::Train::DfaB::SCALE_INV_SQRT_FAN, VOCAB, NE, 0.0)
            TinyNN.tnn_upload_from_float_array(sess, b_aq, Toy::Train::DfaB.fill(dmv * VOCAB, b_seed + 21, dist_c, sig_dm), dmv * VOCAB)
            TinyNN.tnn_upload_from_float_array(sess, b_ak, Toy::Train::DfaB.fill(dmv * VOCAB, b_seed + 22, dist_c, sig_dm), dmv * VOCAB)
            TinyNN.tnn_upload_from_float_array(sess, b_av, Toy::Train::DfaB.fill(dmv * VOCAB, b_seed + 23, dist_c, sig_dm), dmv * VOCAB)
            TinyNN.tnn_upload_from_float_array(sess, b_ao, Toy::Train::DfaB.fill(dmv * VOCAB, b_seed + 24, dist_c, sig_dm), dmv * VOCAB)
            TinyNN.tnn_upload_from_float_array(sess, b_r,  Toy::Train::DfaB.fill(NE * VOCAB, b_seed + 25, dist_c, sig_e),  NE * VOCAB)
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
          forward_tower(sess, tw, t_tok, t_labels, sel1, sel2, eye, top1, bp_spine ? 1 : 0)
          if bp_router || bp_spine
            # toy#121: the task CE joins the loss roots — backward reaches
            # Wr through gate = sum_rows(oneh*probs) -> softmax -> matmul,
            # never mul_mat_id (eo's subtree stays out of grads_needed).
            TinyNN.tnn_set_loss(tw.t_loss)
          end

          # top1: aux loss root (always built; alpha=0 -> zero f' vector).
          t_aux = TinyNN.tnn_null_ptr
          if top1
            m_fp  = TinyNN.tnn_mul(sess, tw.t_gates, t_f)
            s_tok = TinyNN.tnn_sum_rows(sess, m_fp)
            s_col = TinyNN.tnn_reshape_3d(sess, s_tok, T, 1, 1)
            t_aux = TinyNN.tnn_matmul(sess, s_col, ones_t)
            TinyNN.tnn_set_output(t_aux)
            TinyNN.tnn_set_loss(t_aux)
            TinyNN.tnn_add_to_graph(sess, tw.t_loss)
            TinyNN.tnn_build_forward_only(sess, t_aux)
          else
            TinyNN.tnn_build_forward_only(sess, tw.t_loss)
          end
          TinyNN.tnn_build_backward(sess)

          if top1
            p_sm = TinyNN.tnn_softmax(sess, tw.t_logits)
            e_b  = TinyNN.tnn_scale(sess, TinyNN.tnn_sub(sess, p_sm, t_labels), 1.0 / T.to_f)
            np2 = TinyNN.tnn_null_ptr
            if bp_spine
              # the whole spine is chain: embed/fnorm/attention/rn2 + Wr
              sp = 0
              while sp < 9
                wire_chain(sess, tw, t_hp, sp)
                sp = sp + 1
              end
            else
              wire_dfa_top1(sess, tw, t_hp, 3, b_aq, e_b, tw.tap_ah,  dmv, dmv, np2)
              wire_dfa_top1(sess, tw, t_hp, 4, b_ak, e_b, tw.tap_ah,  dmv, dmv, np2)
              wire_dfa_top1(sess, tw, t_hp, 5, b_av, e_b, tw.tap_ah,  dmv, dmv, np2)
              wire_dfa_top1(sess, tw, t_hp, 6, b_ao, e_b, tw.tap_ctx, dmv, dmv, np2)
            end
            if bp_spine
              # router already chain-wired above; experts follow below
            elsif bp_router
              # toy#121: router credit is PURE BP — the acc already holds
              # task-BP + aux-BP (both loss roots backward into it).
              wire_chain(sess, tw, t_hp, 8)
            else
              # F4 lane: router = DFA task signal + BP aux signal
              g_dfa_r = dfa_grad(sess, b_r, e_b, tw.tap_h2, dmv, NE)
              acc_r   = TinyNN.tnn_tensor_grad(sess, tw.pp[8])
              g_tot   = TinyNN.tnn_add(sess, g_dfa_r, acc_r)
              TinyNN.tnn_set_output(g_tot)
              to_r = TinyNN.tnn_opt_step_adamw(sess, tw.pp[8], g_tot, tw.pm[8], tw.pv[8], t_hp)
              TinyNN.tnn_extend_backward_graph(sess, to_r)
            end
            m1 = TinyNN.tnn_matmul(sess, sel1, tw.t_onehots)
            m2 = TinyNN.tnn_matmul(sess, sel2, tw.t_onehots)
            wire_dfa_top1(sess, tw, t_hp, 9,  b_up1,   e_b, tw.tap_h2, dmv,  dfv, m1)
            wire_dfa_top1(sess, tw, t_hp, 10, b_down1, e_b, tw.tap_a1, dfv, dmv,  m1)
            wire_dfa_top1(sess, tw, t_hp, 11, b_up2,   e_b, tw.tap_h2, dmv,  dfv, m2)
            wire_dfa_top1(sess, tw, t_hp, 12, b_down2, e_b, tw.tap_a1, dfv, dmv,  m2)
          else
            if dfa_experts
              p_sm = TinyNN.tnn_softmax(sess, tw.t_logits)
              e_b  = TinyNN.tnn_scale(sess, TinyNN.tnn_sub(sess, p_sm, t_labels), 1.0 / T.to_f)
              idx = 0
              while idx < 9
                wire_chain(sess, tw, t_hp, idx)
                idx = idx + 1
              end
              wire_dfa(sess, tw, t_hp, 9,  b_up1,   e_b, tw.tap_h2, dmv,  dfv, "up1")
              wire_dfa(sess, tw, t_hp, 10, b_down1, e_b, tw.tap_a1, dfv, dmv,  "down1")
              wire_dfa(sess, tw, t_hp, 11, b_up2,   e_b, tw.tap_h2, dmv,  dfv, "up2")
              wire_dfa(sess, tw, t_hp, 12, b_down2, e_b, tw.tap_a2, dfv, dmv,  "down2")
            else
              idx = 0
              while idx < 13
                wire_chain(sess, tw, t_hp, idx)
                idx = idx + 1
              end
            end
          end

          TinyNN.tnn_pin_all_graph_b_nodes(sess)
          TinyNN.tnn_realize_backward(sess)

          ToyDescribeFlow.emit_flow_json(run_dir, sess)

          # run_start
          if events.length > 0
            rc = TinyNN.tnn_events_open(events)
            if rc == 0
              rid = run_id_s.length > 0 ? run_id_s : "anonymous"
              rs = Toy::Json::Builder.new
              rs.add_str("kind", "run_start")
              rs.add_str("schema", "toy/v1")
              rs.add_num("t", TinyNN.tnn_events_now_seconds)
              rs.add_str("started_at", TinyNN.tnn_events_iso8601_now)
              rs.add_str("run_id", rid)
              rs.add_str("phase", "train")
              Toy::Events.add_provenance(rs,
                TinyNN.tnn_provenance_host_name, TinyNN.tnn_provenance_host_os,
                TinyNN.tnn_provenance_host_arch, TinyNN.tnn_backend_name(sess))
              model = Toy::Json::Builder.new
              model.add_str("arch", "franken-moe")
              model.add_str("shape", shape_s)
              model.add_str("name", "franken-moe-instrument")
              model.add_num("vocab",    VOCAB)
              model.add_num("d_model",  dmv)
              model.add_num("n_experts", NE)
              model.add_num("d_ff",     dfv)
              rs.add_obj("model", model)
              config = Toy::Json::Builder.new
              config.add_num("context", T)
              config.add_num("steps",   steps)
              config.add_num("lr",      lr)
              config.add_num("seed",    seed)
              rs.add_obj("config", config)
              fm = Toy::Json::Builder.new
              fm.add_str("routing",   top1 ? "top1" : "dense")
              fm.add_str("policy",    pol_name)
              fm.add_num("aux_alpha", aux_alpha)
              fm.add_num("b_seed",    b_seed)
              fm.add_num("b_dist",    dist_c)
              fm.add_num("b_scale",   Toy::Train::DfaB::SCALE_INV_SQRT_FAN)
              fm.add_num("b_sigma",   0.0)
              rs.add_obj("franken_moe", fm)
              TinyNN.tnn_events_emit(rs.dump)
            else
              puts "events_open failed: rc=" + rc.to_s + " (path=" + events + ")"
            end
          end

          ids = [1, 2, 3, 4]
          labels = zeros(VOCAB * T)
          tt = 0
          while tt < T
            tgt = (ids[tt] + 1) % VOCAB
            labels[tgt + VOCAB * tt] = 1.0
            tt = tt + 1
          end

          n_dfa = tw.dfa_grads.length
          gbuf = zeros(dmv * dfv)
          abuf = zeros(dmv * dfv)
          gates_buf = zeros(NE * T)
          fvec = zeros(NE)
          fi = 0
          while fi < NE
            fvec[fi] = aux_alpha / T.to_f
            fi = fi + 1
          end
          aux_buf = zeros(1)
          b1 = 0.9; b2 = 0.95
          final_loss = 0.0
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
            if top1
              TinyNN.tnn_upload_from_float_array(sess, t_f, fvec, NE)
            end
            TinyNN.tnn_compute_backward(sess)
            TinyNN.tnn_download(sess, tw.t_loss)
            loss = TinyNN.tnn_scratch_get(sess, 0)
            final_loss = loss
            puts "step " + (s + 1).to_s + ": loss=" + loss.to_s

            if events.length > 0
              es = Toy::Json::Builder.new
              es.add_str("kind",  "step")
              es.add_str("phase", "train")
              es.add_num("t",     TinyNN.tnn_events_now_seconds)
              es.add_num("step",  s + 1)
              es.add_raw("loss",  num_or_null_cli(loss))
              TinyNN.tnn_events_emit(es.dump)
            end

            # route event: shares + router health, both routings
            sh0 = 0.0
            if top1
              TinyNN.tnn_download_to_f64_array(sess, tw.t_onehots, gates_buf, NE * T)
              ti2 = 0
              while ti2 < T
                sh0 = sh0 + gates_buf[ti2 * NE]
                ti2 = ti2 + 1
              end
              sh0 = sh0 / T.to_f
            end
            TinyNN.tnn_download_to_f64_array(sess, tw.t_gates, gates_buf, NE * T)
            g0 = 0.0
            ti = 0
            while ti < T
              g0 = g0 + gates_buf[ti * NE]
              ti = ti + 1
            end
            g0 = g0 / T.to_f
            if !top1
              sh0 = g0
            end
            aux_v = "null"
            if top1
              TinyNN.tnn_download_to_f64_array(sess, t_aux, aux_buf, 1)
              aux_v = num_or_null_cli(aux_buf[0])
              # lag-1 f' from this step's routing
              TinyNN.tnn_download_to_f64_array(sess, tw.t_onehots, gates_buf, NE * T)
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
            end
            if events.length > 0
              re2 = Toy::Json::Builder.new
              re2.add_str("kind",  "route")
              re2.add_str("phase", "train")
              re2.add_num("t",     TinyNN.tnn_events_now_seconds)
              re2.add_num("step",  s + 1)
              re2.add_raw("shares", "[" + num_or_null_cli(sh0) + "," +
                                    num_or_null_cli(1.0 - sh0) + "]")
              re2.add_raw("g0_mean", num_or_null_cli(g0))
              re2.add_raw("aux",     aux_v)
              TinyNN.tnn_events_emit(re2.dump)
            end

            # align events (dense dfa-experts, opt-in)
            if align_on && events.length > 0 && n_dfa > 0
              ai = 0
              while ai < n_dfa
                nw = TinyNN.tnn_tensor_nelements(tw.dfa_grads[ai])
                rc_g = TinyNN.tnn_download_to_f64_array(sess, tw.dfa_grads[ai], gbuf, nw)
                rc_a = TinyNN.tnn_download_to_f64_array(sess, tw.dfa_accs[ai], abuf, nw)
                if rc_g != 0 || rc_a != 0
                  puts "align download failed: step=" + (s + 1).to_s +
                       " w=" + tw.dfa_names[ai] + " rc_g=" + rc_g.to_s + " rc_a=" + rc_a.to_s
                end
                dot = 0.0; na = 0.0; nb = 0.0
                ii = 0
                while ii < nw
                  dot = dot + gbuf[ii] * abuf[ii]
                  na = na + gbuf[ii] * gbuf[ii]
                  nb = nb + abuf[ii] * abuf[ii]
                  ii = ii + 1
                end
                sa = Math.sqrt(na)
                sb = Math.sqrt(nb)
                cv = 0.0
                d = sa * sb
                if d > 0.0
                  cv = dot / d
                end
                ae = Toy::Json::Builder.new
                ae.add_str("kind",  "align")
                ae.add_str("phase", "train")
                ae.add_num("t",     TinyNN.tnn_events_now_seconds)
                ae.add_num("step",  s + 1)
                ae.add_str("w",     tw.dfa_names[ai])
                ae.add_raw("cos",      num_or_null_cli(cv))
                ae.add_raw("dfa_norm", num_or_null_cli(sa))
                ae.add_raw("bp_norm",  num_or_null_cli(sb))
                TinyNN.tnn_events_emit(ae.dump)
                ai = ai + 1
              end
            end
            s = s + 1
          end

          if events.length > 0 && TinyNN.tnn_events_active == 1
            re = Toy::Json::Builder.new
            re.add_str("kind", "run_end")
            re.add_num("t",          TinyNN.tnn_events_now_seconds)
            re.add_str("ended_at",   TinyNN.tnn_events_iso8601_now)
            re.add_str("reason",     "completed")
            re.add_num("final_step", steps)
            re.add_raw("final_loss", num_or_null_cli(final_loss))
            re.add_raw("exit_code",  "0")
            TinyNN.tnn_events_emit(re.dump)
            TinyNN.tnn_events_close
          end
          if final_loss != final_loss
            puts "FRANKEN-MOE-CLI: NaN loss"
          end
          puts "FRANKEN-MOE-CLI DONE"
        end
      end
    end
  end
end

Toy::LLM::Run::TrainFrankenMoe.run_cli
