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
#   FRANKEN_ALIGN_EVERY  toy#127: thin align downloads + emissions to
#                        every Nth step (the toy#122 thinning, MoE-side;
#                        step/route events stay per-step; default 1)
#   FRANKEN_MOE_EXPERTS  toy#128 (the demonstrator's E axis): expert
#                        count >= 2 (default 2, byte-null — the rig
#                        shape). Experts at pp[9+2i]/pp[10+2i]; B seeds
#                        stable per expert index (legacy +11..+14 for
#                        experts 0/1, +101+10i/+102+10i from expert 2 —
#                        the legacy stride would hit the attention B
#                        offsets +21..+25 at expert 5). route.shares
#                        becomes the true length-E vector.
#   CORPUS               toy#125 (the F8 data surface): stream the
#                        packed-i32 corpus per step (warm-start's
#                        reader, rotating T-token windows, pre-EOF
#                        rotation — the toy#122 stuck-window fix);
#                        labels rebuilt per step (next_token_guarded).
#                        Sets the instrument to the frozen-vocab
#                        contract (VOCAB 627, toy#123) — the fixed-seq
#                        feed's vocab-16 embed cannot take the stream's
#                        token ids. Absent flag = the byte-gated fixed
#                        sequence at vocab 16, untouched.
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
require_relative "../llm/labels"
require_relative "../io/toy_corpus_loader"
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
          # toy#129 item 2: FRANKEN_NO_SHADOW=1 — dense dfa-experts
          # drop the expert weights from the autodiff param set
          # (late-param via the top1 wire, no chain grad-acc, no
          # backward expansion through the experts). top1 lanes are
          # ALREADY shadow-free by construction (zero-param build, the
          # F4/F5 design) — the flag is meaningless there, fail loud.
          no_shadow = (ENV["FRANKEN_NO_SHADOW"] || "") == "1"
          # toy#127: the toy#122 thinning, MoE-side — every Nth step
          # skips the align DOWNLOADS and emissions both (the shadow
          # readback is the cost); step/route events stay per-step.
          ae_raw = (ENV["FRANKEN_ALIGN_EVERY"] || "1").to_i
          align_every = ae_raw < 1 ? 1 : ae_raw
          events = run_dir.length > 0 ? (run_dir + "/events.jsonl") : ""
          lr = 0.02
          if no_shadow && top1
            puts "toy-train-franken-moe: --no-shadow is meaningless under --routing top1 — top1 lanes are already shadow-free (zero-param build, the F4/F5 design)"
            return 1
          end
          if no_shadow && !dfa_experts
            puts "toy-train-franken-moe: --no-shadow requires --moe-policy dfa-experts (dense chain has no DFA segments to unshadow)"
            return 1
          end
          if no_shadow && align_on
            puts "toy-train-franken-moe: --no-shadow + align events — align compares DFA grads against the chain shadow acc, which a no-shadow build does not create. Drop one."
            return 1
          end

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
               " seed=" + seed.to_s + " b_seed=" + b_seed.to_s +
               " experts=" + (ENV["FRANKEN_MOE_EXPERTS"] || "2")

          shape_s = ENV["FRANKEN_SHAPE"] || "base"
          if shape_s != "base" && shape_s != "wide"
            puts "unknown FRANKEN_SHAPE: " + shape_s + " (franken-moe takes base|wide)"
            return 1
          end
          corpus_s = ENV["CORPUS"] || ""
          if corpus_s.length > 0 && !File.exist?(corpus_s)
            puts "toy-train-franken-moe: corpus not found: " + corpus_s
            puts "  --corpus streams packed-i32 tokens (the warm-start reader)."
            return 1
          end
          # toy#129 item 1: context joins the runtime state — corpus
          # mode takes FRANKEN_CONTEXT (>= 2); the fixed-seq feed is
          # the byte-gated 4-token contract, so a context override
          # without a corpus fails loud.
          ctx_s = ENV["FRANKEN_CONTEXT"] || ""
          ctx_raw = ctx_s.length > 0 ? ctx_s.to_i : 0
          ctx = ctx_raw > 0 ? ctx_raw : T   # 0 = unset (the CLI's default)
          if ctx != T && corpus_s.length == 0
            puts "toy-train-franken-moe: --context needs --corpus (the fixed-seq feed is the byte-gated 4-token contract)"
            return 1
          end
          if ctx < 2
            puts "toy-train-franken-moe: --context must be >= 2, got " + ctx.to_s
            return 1
          end
          # toy#125/#129: corpus vocab — a TOYC pack declares it in the
          # header (authoritative; a conflicting --vocab fails loud);
          # headerless packs default to the frozen-vocab contract (627,
          # toy#123) unless --vocab overrides. Fixed-seq stays vocab 16.
          vsel = VOCAB
          if corpus_s.length > 0
            hdr_v = ToyCorpusLoader.probe_vocab(corpus_s)
            env_v = (ENV["FRANKEN_VOCAB"] || "0").to_i
            if hdr_v > 0 && env_v > 0 && hdr_v != env_v
              puts "toy-train-franken-moe: --vocab " + env_v.to_s + " conflicts with the pack header (TOYC vocab " + hdr_v.to_s + ")"
              return 1
            end
            vsel = 627
            if hdr_v > 0
              vsel = hdr_v
            end
            if hdr_v == 0 && env_v > 0
              vsel = env_v
            end
          end
          # toy#128: expert count (the demonstrator's E axis; default 2
          # byte-null — the rig shape).
          ex_s = ENV["FRANKEN_MOE_EXPERTS"] || ""
          esel = ex_s.length > 0 ? ex_s.to_i : NE
          if esel < 2
            puts "FRANKEN_MOE_EXPERTS must be >= 2, got " + ex_s
            return 1
          end
          if shape_s == "wide"
            shape_init(256, 512, vsel, esel, ctx)
          else
            shape_init(DM_BASE, DFF_BASE, vsel, esel, ctx)
          end

          sess = TinyNN.tnn_session_new(0)
          TinyNN.tnn_session_set_graph_capacity(sess, 262144)

          tw = alloc_tower(sess)

          # toy#128: per-expert B mats + one-hot selectors as arrays
          # (E=2 reproduces the b_up1/b_down1/b_up2/b_down2 + sel1/sel2
          # allocation order exactly).
          np0 = TinyNN.tnn_null_ptr
          b_ups   = [np0]; b_ups.pop
          b_downs = [np0]; b_downs.pop
          ei = 0
          while ei < nev
            b_ups.push(TinyNN.tnn_input_2d_f32_persistent(sess, dfv, vocabv))
            b_downs.push(TinyNN.tnn_input_2d_f32_persistent(sess, dmv, vocabv))
            ei = ei + 1
          end
          sels = [np0]; sels.pop
          ei = 0
          while ei < nev
            sels.push(TinyNN.tnn_input_2d_f32_persistent(sess, 1, nev))
            ei = ei + 1
          end
          eye   = TinyNN.tnn_input_2d_f32_persistent(sess, nev, nev)
          b_aq  = TinyNN.tnn_input_2d_f32_persistent(sess, dmv, vocabv)
          b_ak  = TinyNN.tnn_input_2d_f32_persistent(sess, dmv, vocabv)
          b_av  = TinyNN.tnn_input_2d_f32_persistent(sess, dmv, vocabv)
          b_ao  = TinyNN.tnn_input_2d_f32_persistent(sess, dmv, vocabv)
          b_r   = TinyNN.tnn_input_2d_f32_persistent(sess, nev, vocabv)
          ones_t = TinyNN.tnn_input_2d_f32_persistent(sess, 1, tv)   # ne=[tv,1]

          # params: dense = every weight (shadow-shaped when dfa-experts);
          # top1 = Wr ONLY (the aux backward path; everything else is
          # late-param DFA — the walker must never need mul_mat_id).
          gi = 0
          while gi < tw.pp.length
            # dense: every weight — EXCEPT no-shadow drops the experts
            # (gi >= 9; late-param at wire time). bp-spine: the whole
            # spine 0..8 (embed, fnorm, attention, rn2, Wr) — experts
            # stay late-param DFA behind the detach cut.
            mark = false
            if top1
              mark = bp_spine && gi < 9
            else
              mark = !(no_shadow && gi >= 9)
            end
            if mark
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
          sig_up   = Toy::Train::DfaB.sigma_for(Toy::Train::DfaB::SCALE_INV_SQRT_FAN, vocabv, dfv, 0.0)
          sig_down = Toy::Train::DfaB.sigma_for(Toy::Train::DfaB::SCALE_INV_SQRT_FAN, vocabv, dmv, 0.0)
          # toy#128 per-expert B seeds — STABLE in the expert index (the
          # standing DfaB discipline: re-policying one expert never
          # reshuffles another). Experts 0/1 keep the legacy offsets
          # +11/+12/+13/+14 (E=2 byte-null); experts >= 2 jump to
          # +101+10i / +102+10i — the legacy stride would collide with
          # the attention/router B offsets (+21..+25) from expert 5 on.
          ei = 0
          while ei < nev
            us = b_seed + 11 + 2 * ei
            ds = b_seed + 12 + 2 * ei
            if ei >= 2
              us = b_seed + 101 + 10 * ei
              ds = b_seed + 102 + 10 * ei
            end
            TinyNN.tnn_upload_from_float_array(sess, b_ups[ei],
              Toy::Train::DfaB.fill(dfv * vocabv, us, dist_c, sig_up), dfv * vocabv)
            TinyNN.tnn_upload_from_float_array(sess, b_downs[ei],
              Toy::Train::DfaB.fill(dmv * vocabv, ds, dist_c, sig_down), dmv * vocabv)
            ei = ei + 1
          end
          ei = 0
          while ei < nev
            sv = zeros(nev)
            sv[ei] = 1.0
            TinyNN.tnn_upload_from_float_array(sess, sels[ei], sv, nev)
            ei = ei + 1
          end
          ey = zeros(nev * nev)
          ei = 0
          while ei < nev
            ey[ei * nev + ei] = 1.0
            ei = ei + 1
          end
          TinyNN.tnn_upload_from_float_array(sess, eye, ey, nev * nev)
          if top1
            sig_dm = Toy::Train::DfaB.sigma_for(Toy::Train::DfaB::SCALE_INV_SQRT_FAN, vocabv, dmv, 0.0)
            sig_e  = Toy::Train::DfaB.sigma_for(Toy::Train::DfaB::SCALE_INV_SQRT_FAN, vocabv, nev, 0.0)
            TinyNN.tnn_upload_from_float_array(sess, b_aq, Toy::Train::DfaB.fill(dmv * vocabv, b_seed + 21, dist_c, sig_dm), dmv * vocabv)
            TinyNN.tnn_upload_from_float_array(sess, b_ak, Toy::Train::DfaB.fill(dmv * vocabv, b_seed + 22, dist_c, sig_dm), dmv * vocabv)
            TinyNN.tnn_upload_from_float_array(sess, b_av, Toy::Train::DfaB.fill(dmv * vocabv, b_seed + 23, dist_c, sig_dm), dmv * vocabv)
            TinyNN.tnn_upload_from_float_array(sess, b_ao, Toy::Train::DfaB.fill(dmv * vocabv, b_seed + 24, dist_c, sig_dm), dmv * vocabv)
            TinyNN.tnn_upload_from_float_array(sess, b_r,  Toy::Train::DfaB.fill(nev * vocabv, b_seed + 25, dist_c, sig_e),  nev * vocabv)
            onesv = zeros(tv)
            oi = 0
            while oi < tv
              onesv[oi] = 1.0
              oi = oi + 1
            end
            TinyNN.tnn_upload_from_float_array(sess, ones_t, onesv, tv)
          end

          t_tok    = TinyNN.tnn_input_1d_i32(sess, tv)
          t_labels = TinyNN.tnn_input_2d_f32(sess, tv, vocabv)
          t_hp     = TinyNN.tnn_input_1d_f32(sess, 7)
          t_f      = TinyNN.tnn_input_2d_f32(sess, 1, nev)   # ne=[NE,1]
          forward_tower(sess, tw, t_tok, t_labels, sels, eye, top1, bp_spine ? 1 : 0)
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
            s_col = TinyNN.tnn_reshape_3d(sess, s_tok, tv, 1, 1)
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
            e_b  = TinyNN.tnn_scale(sess, TinyNN.tnn_sub(sess, p_sm, t_labels), 1.0 / tv.to_f)
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
              g_dfa_r = dfa_grad(sess, b_r, e_b, tw.tap_h2, dmv, nev)
              acc_r   = TinyNN.tnn_tensor_grad(sess, tw.pp[8])
              g_tot   = TinyNN.tnn_add(sess, g_dfa_r, acc_r)
              TinyNN.tnn_set_output(g_tot)
              to_r = TinyNN.tnn_opt_step_adamw(sess, tw.pp[8], g_tot, tw.pm[8], tw.pv[8], t_hp)
              TinyNN.tnn_extend_backward_graph(sess, to_r)
            end
            # toy#128: per-expert routed-mask DFA wires (E=2 == the old
            # m1/m2 + 9..12 sequence; both downs read tap_a1 = the
            # routed acts from mul_mat_id).
            ei = 0
            while ei < nev
              m_i = TinyNN.tnn_matmul(sess, sels[ei], tw.t_onehots)
              wire_dfa_top1(sess, tw, t_hp, 9 + 2 * ei,  b_ups[ei],   e_b, tw.tap_h2, dmv,  dfv, m_i)
              wire_dfa_top1(sess, tw, t_hp, 10 + 2 * ei, b_downs[ei], e_b, tw.tap_a1, dfv, dmv,  m_i)
              ei = ei + 1
            end
          else
            if dfa_experts
              p_sm = TinyNN.tnn_softmax(sess, tw.t_logits)
              e_b  = TinyNN.tnn_scale(sess, TinyNN.tnn_sub(sess, p_sm, t_labels), 1.0 / tv.to_f)
              idx = 0
              while idx < 9
                wire_chain(sess, tw, t_hp, idx)
                idx = idx + 1
              end
              # toy#128: per-expert dense DFA wires; each down_i reads
              # its own dense act tap (tap_as[i]). toy#129: no-shadow
              # reuses the top1 wire (late-param, NO acc recording,
              # null mask) — the DFA grad math is identical, so applied
              # updates byte-match the shadow build (gated).
              np3 = TinyNN.tnn_null_ptr
              ei = 0
              while ei < nev
                if no_shadow
                  wire_dfa_top1(sess, tw, t_hp, 9 + 2 * ei,  b_ups[ei],   e_b, tw.tap_h2,     dmv,  dfv, np3)
                  wire_dfa_top1(sess, tw, t_hp, 10 + 2 * ei, b_downs[ei], e_b, tw.tap_as[ei], dfv, dmv,  np3)
                else
                  wire_dfa(sess, tw, t_hp, 9 + 2 * ei,  b_ups[ei],   e_b, tw.tap_h2,    dmv,  dfv, "up" + (ei + 1).to_s)
                  wire_dfa(sess, tw, t_hp, 10 + 2 * ei, b_downs[ei], e_b, tw.tap_as[ei], dfv, dmv,  "down" + (ei + 1).to_s)
                end
                ei = ei + 1
              end
            else
              idx = 0
              while idx < 9 + 2 * nev
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
              model.add_num("vocab",    vocabv)
              model.add_num("d_model",  dmv)
              model.add_num("n_experts", nev)
              model.add_num("d_ff",     dfv)
              rs.add_obj("model", model)
              config = Toy::Json::Builder.new
              config.add_num("context", tv)
              config.add_num("steps",   steps)
              config.add_num("lr",      lr)
              config.add_num("seed",    seed)
              rs.add_obj("config", config)
              # toy#129 item 4: derived cost accounting. total = tied
              # embed vocab*dm + norms 3dm + attention 4dm^2 + router
              # E*dm + E expert pairs 2*dm*dff. active swaps E -> the
              # per-token expert count (top1 routes 1; dense mixes all
              # E). flops_per_token = FORWARD, 2 flops/MAC: attention
              # matmuls + scores/combine at T + router + active experts
              # + tied logits.
              act_e = top1 ? 1 : nev
              cost_total  = vocabv * dmv + 3 * dmv + 4 * dmv * dmv +
                            nev * dmv + nev * 2 * dmv * dfv
              cost_active = vocabv * dmv + 3 * dmv + 4 * dmv * dmv +
                            nev * dmv + act_e * 2 * dmv * dfv
              cost_flops  = 2 * (4 * dmv * dmv) + 4 * dmv * tv +
                            2 * nev * dmv + act_e * 2 * (2 * dmv * dfv) +
                            2 * vocabv * dmv
              cost = Toy::Json::Builder.new
              cost.add_num("total_params",    cost_total)
              cost.add_num("active_params",   cost_active)
              cost.add_num("flops_per_token", cost_flops)
              rs.add_obj("cost", cost)
              fm = Toy::Json::Builder.new
              fm.add_str("routing",   top1 ? "top1" : "dense")
              fm.add_str("policy",    pol_name)
              fm.add_num("aux_alpha", aux_alpha)
              fm.add_num("b_seed",    b_seed)
              fm.add_num("b_dist",    dist_c)
              fm.add_num("b_scale",   Toy::Train::DfaB::SCALE_INV_SQRT_FAN)
              fm.add_num("b_sigma",   0.0)
              # toy#129: shadow = whether dfa segments carry chain
              # grad-accs (dense dfa-experts without --no-shadow).
              # top1 lanes and dense chain have none — false.
              fm.add_bool("shadow",   dfa_experts && !top1 && !no_shadow)
              rs.add_obj("franken_moe", fm)
              TinyNN.tnn_events_emit(rs.dump)
            else
              puts "events_open failed: rc=" + rc.to_s + " (path=" + events + ")"
            end
          end

          ids = [1, 2, 3, 4]
          labels = zeros(vocabv * tv)
          tt = 0
          while tt < tv
            tgt = (ids[tt] + 1) % vocabv
            labels[tgt + vocabv * tt] = 1.0
            tt = tt + 1
          end
          corpus_base  = corpus_s.length > 0 ? ToyCorpusLoader.data_offset(corpus_s) : 0
          corpus_off   = corpus_base
          corpus_bytes = corpus_s.length > 0 ? File.size(corpus_s) : 0

          n_dfa = tw.dfa_grads.length
          gbuf = zeros(dmv * dfv)
          abuf = zeros(dmv * dfv)
          gates_buf = zeros(nev * tv)
          share_v = zeros(nev)
          fvec = zeros(nev)
          fi = 0
          while fi < nev
            fvec[fi] = aux_alpha / tv.to_f
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
            if corpus_s.length > 0
              # rotating-window stream: restart at 0 BEFORE the window
              # would run past EOF (read_seq's own EOF-wrap otherwise
              # pins every later step to the first window — the toy#122
              # stuck-window failure mode).
              if corpus_off + tv * 4 > corpus_bytes
                corpus_off = corpus_base
              end
              ids = ToyCorpusLoader.read_seq(corpus_s, corpus_off, tv)
              corpus_off = corpus_off + tv * 4
              m_lab = Toy::Labels.next_token_guarded(ids, vocabv, tv, 1)
              labels = m_lab.flat
            end
            TinyNN.upload_int_array(sess, t_tok, ids)
            TinyNN.tnn_upload_from_float_array(sess, t_labels, labels, vocabv * tv)
            TinyNN.tnn_upload_from_float_array(sess, t_hp, hp, 7)
            if top1
              TinyNN.tnn_upload_from_float_array(sess, t_f, fvec, nev)
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

            # route event: shares + router health, both routings.
            # toy#128: shares is the true per-expert vector (length E) —
            # top1 = hard one-hot token shares, dense = mean soft gate
            # per expert (E=2 keeps shares[0] exactly; shares[1] is now
            # measured, not 1-shares[0]).
            if top1
              TinyNN.tnn_download_to_f64_array(sess, tw.t_onehots, gates_buf, nev * tv)
              ei = 0
              while ei < nev
                cs = 0.0
                ti2 = 0
                while ti2 < tv
                  cs = cs + gates_buf[ti2 * nev + ei]
                  ti2 = ti2 + 1
                end
                share_v[ei] = cs / tv.to_f
                ei = ei + 1
              end
            end
            TinyNN.tnn_download_to_f64_array(sess, tw.t_gates, gates_buf, nev * tv)
            g0 = 0.0
            ti = 0
            while ti < tv
              g0 = g0 + gates_buf[ti * nev]
              ti = ti + 1
            end
            g0 = g0 / tv.to_f
            if !top1
              ei = 0
              while ei < nev
                cs = 0.0
                ti2 = 0
                while ti2 < tv
                  cs = cs + gates_buf[ti2 * nev + ei]
                  ti2 = ti2 + 1
                end
                share_v[ei] = cs / tv.to_f
                ei = ei + 1
              end
            end
            aux_v = "null"
            if top1
              TinyNN.tnn_download_to_f64_array(sess, t_aux, aux_buf, 1)
              aux_v = num_or_null_cli(aux_buf[0])
              # lag-1 f' from this step's routing
              TinyNN.tnn_download_to_f64_array(sess, tw.t_onehots, gates_buf, nev * tv)
              ei = 0
              while ei < nev
                cnt = 0.0
                ti3 = 0
                while ti3 < tv
                  cnt = cnt + gates_buf[ti3 * nev + ei]
                  ti3 = ti3 + 1
                end
                fvec[ei] = aux_alpha * nev.to_f * cnt / (tv.to_f * tv.to_f)
                ei = ei + 1
              end
            end
            if events.length > 0
              re2 = Toy::Json::Builder.new
              re2.add_str("kind",  "route")
              re2.add_str("phase", "train")
              re2.add_num("t",     TinyNN.tnn_events_now_seconds)
              re2.add_num("step",  s + 1)
              sh_s = "["
              ei = 0
              while ei < nev
                if ei > 0
                  sh_s = sh_s + ","
                end
                sh_s = sh_s + num_or_null_cli(share_v[ei])
                ei = ei + 1
              end
              sh_s = sh_s + "]"
              re2.add_raw("shares", sh_s)
              re2.add_raw("g0_mean", num_or_null_cli(g0))
              re2.add_raw("aux",     aux_v)
              TinyNN.tnn_events_emit(re2.dump)
            end

            # align events (dense dfa-experts, opt-in; toy#127 thinned)
            if align_on && events.length > 0 && n_dfa > 0 && (s % align_every) == 0
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
          0
        end
      end
    end
  end
end

# toy#129 fallout fix: run_cli's error paths `return 1` — that return
# value was silently DROPPED, so validation failures exited 0 (latent
# since toy#124's unknown-shape guard; first observable when the gate
# asserted a rejection). The exit status now carries it.
rc_main = Toy::LLM::Run::TrainFrankenMoe.run_cli
if rc_main != 0
  exit 1
end
