# lib/toy/run/train_franken.rb — toy#109 P1: the FrankenModel credit-assignment
# runner, in its OWN Spinel compilation unit (the toy-train-hybrid precedent).
#
# TWIN-LANE trainer, ONE GRAPH: two identical attention towers (disjoint
# weight sets, bit-identical seeded init) live in the SAME session/graph and
# advance in true lockstep — one compute_backward per step trains both, off
# the SAME uploaded batch. Lane A is always :chain (backprop). Lane B follows
# FRANKEN_POLICY — per-layer, comma-separated:
#
#   FRANKEN_POLICY=chain,chain        # null: lane B byte-equals lane A
#   FRANKEN_POLICY=dfa,dfa            # per-matmul DFA on attention weights
#   FRANKEN_POLICY=chain,dfa          # bottom-BP / top-DFA (any mix)
#   FRANKEN_POLICY=mix:0.5,mix:0.5    # α·chain + (1−α)·dfa per weight (P3)
#   FRANKEN_POLICY=maskdfa:0.001,...  # dfa gated by 1[|g_bp|>τ]  (P3 —
#   FRANKEN_POLICY=maskbp:0.001,...   # bp gated by 1[|g_dfa|>τ]   the
#                                     # "activation function on the update";
#                                     # smooth gate 0.5(1+tanh(k(|g|−τ))),
#                                     # k=1e4 ⇒ near-hard; τ=−1 saturates to
#                                     # EXACTLY 1.0 ⇒ byte-equals dfa (null),
#                                     # τ huge ⇒ exactly 0.0 ⇒ frozen arm)
#
# WHY one graph (P1 finding, load-bearing): tinynn's engine — and therefore
# THE SCHED — is a process-global singleton (g_engine_cpu). Two sessions
# alternating realize/compute on one sched leave the first session's graph
# allocation stale (the F1.1 / dual-cgraph single-shot-per-reset class):
# identical weights + identical graphs computed DIFFERENT losses. Two towers
# in one graph_b get ONE sched alloc and cannot desynchronize. (A second
# benefit: ggml autodiff handles the two loss roots in one backward sweep —
# tower-disjoint params keep the grads from crossing.)
#
# v1 DFA variant (design doc §4c): PER-MATMUL, identity-boundary — each
# attention weight W gets its own fixed random B_W (VOCAB→DM), update
# grad_W = (B_W · e_b) · h_inᵀ with e_b = (softmax(logits_b) − labels)/T,
# built as PURE FORWARD OPS appended to graph_b (P0 mechanics; no autodiff,
# no cuts). RMSNorm gammas + embedding + final norm stay :chain in every
# policy. B_W sampling: gaussian via Box–Muller on xorshift, σ = 1/√VOCAB
# (inv_sqrt_fan), seeded from (FRANKEN_SEED, stable weight index) — so
# re-policying one layer never reshuffles another's B. Distribution/scale
# become RecipeOptions axes in P2.
#
# SHADOW ALIGNMENT (design doc §4b): all weights stay in the autodiff param
# set (shadow-shaped build), so lane B's true chain gradients exist even for
# :dfa weights — the APPLIED update is wired to the DFA tensor; the chain
# grad-acc is downloaded as telemetry:
#
#   align step=<s> li=<layer> w=<q|k|v|o> cos=<cos∠(g_dfa, g_chain)>
#
# (the FA-literature alignment diagnostic). The membership CUT (production
# shape, no shadow) needs tnn_unset_param — deferred with the cut primitive.
#
# P0 idioms carried: pin_all_graph_b_nodes before realize (sched aliasing);
# hp slots 5/6 = 1/(1−β^t) bias correction (lora convention), per step.
#
# Spinel hygiene: while loops, string-concat output, typed-empty seeds, no
# default args, ENV[] nil-guards.

require_relative "../../toy"
require_relative "../ffi/tinynn"
require_relative "../llm/primitives/rms_norm"
require_relative "../train/dfa_b"

module Toy
  module LLM
    module Run
      module TrainFranken
        VOCAB  = 16
        DM     = 8
        T      = 4
        LAYERS = 2
        EPS    = 1.0e-5

        # deterministic weight init (same helper family as train_hybrid);
        # seed depends only on the WITHIN-TOWER weight index so both towers
        # start bit-identical.
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

        # B fill via the shared axes module (toy#109 P2). ENV knobs:
        # FRANKEN_B_DIST=gaussian|uniform|rademacher,
        # FRANKEN_B_SCALE=inv_sqrt_fan|glorot|fixed:<sigma>.
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

        def self.b_sigma
          sc = ENV["FRANKEN_B_SCALE"] || ""
          if sc.length >= 6 && sc[0, 6] == "fixed:"
            return Toy::Train::DfaB.sigma_for(Toy::Train::DfaB::SCALE_FIXED,
                                              VOCAB, DM, sc[6, sc.length - 6].to_f)
          end
          if sc == "glorot"
            return Toy::Train::DfaB.sigma_for(Toy::Train::DfaB::SCALE_GLOROT, VOCAB, DM, 0.0)
          end
          Toy::Train::DfaB.sigma_for(Toy::Train::DfaB::SCALE_INV_SQRT_FAN, VOCAB, DM, 0.0)
        end

        # Per-tower handle bundle (uniform-typed arrays; no Struct/Card).
        class Tower
          attr_accessor :pp, :pm, :pv
          attr_accessor :rns, :wqs, :wks, :wvs, :wos
          attr_accessor :hs, :ctxs, :t_loss, :t_logits
          attr_accessor :dfa_grads, :dfa_accs, :dfa_names
          attr_accessor :masks, :mask_names

          def initialize
            np = TinyNN.tnn_null_ptr
            @pp = [np]; @pp.pop
            @pm = [np]; @pm.pop
            @pv = [np]; @pv.pop
            @rns = [np]; @rns.pop
            @wqs = [np]; @wqs.pop
            @wks = [np]; @wks.pop
            @wvs = [np]; @wvs.pop
            @wos = [np]; @wos.pop
            @hs   = [np]; @hs.pop
            @ctxs = [np]; @ctxs.pop
            @t_loss   = np
            @t_logits = np
            @dfa_grads = [np]; @dfa_grads.pop
            @dfa_accs  = [np]; @dfa_accs.pop
            @dfa_names = [""]; @dfa_names.pop
            @masks      = [np]; @masks.pop
            @mask_names = [""]; @mask_names.pop
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

        # Allocate one tower's persistent weights (embed, fnorm, per-layer
        # rn/wq/wk/wv/wo). Same order in both towers => same init streams.
        def self.alloc_tower(sess)
          tw = Tower.new
          reg2(sess, tw, VOCAB, DM)     # pp[0] embed
          reg1(sess, tw, DM)            # pp[1] fnorm
          li = 0
          while li < LAYERS
            tw.rns.push(reg1(sess, tw, DM))
            tw.wqs.push(reg2(sess, tw, DM, DM))
            tw.wks.push(reg2(sess, tw, DM, DM))
            tw.wvs.push(reg2(sess, tw, DM, DM))
            tw.wos.push(reg2(sess, tw, DM, DM))
            li = li + 1
          end
          tw
        end

        # Inline single-head causal self-attention, returning [out, h, ctx]
        # so the DFA taps are exposed: h feeds wq/wk/wv, ctx feeds wo.
        def self.attention_taps(sess, t_x, rn, wq, wk, wv, wo)
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
          added  = TinyNN.tnn_add(sess, t_x, out)
          [added, h, ctx]
        end

        # Forward one tower over the SHARED token input; record taps + loss.
        def self.forward_tower(sess, tw, t_tok, t_labels)
          x = TinyNN.tnn_get_rows(sess, tw.pp[0], t_tok)
          li = 0
          while li < LAYERS
            trip = attention_taps(sess, x, tw.rns[li], tw.wqs[li], tw.wks[li],
                                  tw.wvs[li], tw.wos[li])
            x = trip[0]
            tw.hs.push(trip[1])
            tw.ctxs.push(trip[2])
            li = li + 1
          end
          xf  = Toy::LLM::Primitives::RMSNorm.build(sess, x, tw.pp[1], EPS)
          lgt = TinyNN.tnn_matmul(sess, tw.pp[0], xf)
          tw.t_logits = lgt
          tw.t_loss   = TinyNN.tnn_cross_entropy_loss(sess, lgt, t_labels)
          TinyNN.tnn_set_output(tw.t_loss)
          TinyNN.tnn_set_loss(tw.t_loss)
          0
        end

        # DFA update for one weight: grad = (B·e)·a_inᵀ, all forward ops.
        def self.dfa_grad(sess, b, e, a_in)
          delta  = TinyNN.tnn_matmul(sess, b, e)                       # [DM, T]
          a_in_t = TinyNN.tnn_cont_2d(sess, TinyNN.tnn_transpose(sess, a_in), T, DM)
          delt_t = TinyNN.tnn_cont_2d(sess, TinyNN.tnn_transpose(sess, delta), T, DM)
          TinyNN.tnn_matmul(sess, a_in_t, delt_t)                      # [DM, DM]
        end

        # Smooth near-hard gate: 0.5·(1 + tanh(k·(|g| − τ))), |g| via
        # sqrt(g⊙g), k = 1e4. τ = −1 saturates to EXACTLY 1.0 (tanh(≥1e4)
        # == 1.0f); τ huge saturates to exactly 0.0 — the byte-null anchors.
        def self.smooth_mask(sess, g, tau)
          absg = TinyNN.tnn_sqrt(sess, TinyNN.tnn_mul(sess, g, g))
          y    = TinyNN.tnn_scale_bias(sess, absg, 1.0, 0.0 - tau)
          TinyNN.tnn_scale_bias(sess, TinyNN.tnn_tanh(sess, TinyNN.tnn_scale(sess, y, 10000.0)), 0.5, 0.5)
        end

        def self.wire_chain(sess, tw, t_hp, idx)
          tg = TinyNN.tnn_tensor_grad(sess, tw.pp[idx])
          to = TinyNN.tnn_opt_step_adamw(sess, tw.pp[idx], tg, tw.pm[idx], tw.pv[idx], t_hp)
          TinyNN.tnn_extend_backward_graph(sess, to)
          0
        end

        def self.run
          pol_s = ENV["FRANKEN_POLICY"] || ""
          if pol_s.length == 0
            pol_s = "dfa,dfa"
          end
          parts = pol_s.split(",")
          policy = [0]; policy.pop
          p_alpha = [0.0]; p_alpha.pop
          p_tau   = [0.0]; p_tau.pop
          i = 0
          while i < LAYERS
            m = 0
            al = 0.0
            ta = 0.0
            if i < parts.length
              tk = parts[i]
              if tk == "dfa"
                m = 1
              elsif tk.length > 4 && tk[0, 4] == "mix:"
                m = 2
                al = tk[4, tk.length - 4].to_f
              elsif tk.length > 8 && tk[0, 8] == "maskdfa:"
                m = 3
                ta = tk[8, tk.length - 8].to_f
              elsif tk.length > 7 && tk[0, 7] == "maskbp:"
                m = 4
                ta = tk[7, tk.length - 7].to_f
              end
            end
            policy.push(m)
            p_alpha.push(al)
            p_tau.push(ta)
            i = i + 1
          end
          steps_s = ENV["STEPS"] || ""
          steps = steps_s.length > 0 ? steps_s.to_i : 40
          seed_s = ENV["FRANKEN_SEED"] || ""
          seed = seed_s.length > 0 ? seed_s.to_i : 1234
          lr = 0.02

          chain_all = [0]; chain_all.pop
          i = 0
          while i < LAYERS
            chain_all.push(0)
            i = i + 1
          end

          puts "franken twin-lane(one-graph): layers=" + LAYERS.to_s +
               " policy_b=" + pol_s + " steps=" + steps.to_s + " seed=" + seed.to_s

          sess = TinyNN.tnn_session_new(0)
          TinyNN.tnn_session_set_graph_capacity(sess, 262144)

          # ---- persistent allocations (both towers + all B mats) ----
          tow_a = alloc_tower(sess)
          tow_b = alloc_tower(sess)
          # B mats for lane B's dfa layers (alloc BEFORE finalize; count
          # depends only on policy).
          bmats = [TinyNN.tnn_null_ptr]; bmats.pop
          bseeds = [0]; bseeds.pop
          li = 0
          while li < LAYERS
            if policy[li] > 0
              wi = 0
              while wi < 4
                bmats.push(TinyNN.tnn_input_2d_f32_persistent(sess, DM, VOCAB))
                bseeds.push(seed + li * 40 + 11 + wi)
                wi = wi + 1
              end
            end
            li = li + 1
          end

          gi = 0
          while gi < tow_a.pp.length
            TinyNN.tnn_set_param(tow_a.pp[gi])
            TinyNN.tnn_set_param(tow_b.pp[gi])
            gi = gi + 1
          end
          TinyNN.tnn_finalize_weights(sess)

          # identical init: same per-index stream in both towers
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
          sigma  = b_sigma
          bi = 0
          while bi < bmats.length
            nb = DM * VOCAB
            TinyNN.tnn_upload_from_float_array(sess, bmats[bi],
              Toy::Train::DfaB.fill(nb, bseeds[bi], dist_c, sigma), nb)
            bi = bi + 1
          end

          # ---- shared inputs + both forwards in ONE graph ----
          t_tok    = TinyNN.tnn_input_1d_i32(sess, T)
          t_labels = TinyNN.tnn_input_2d_f32(sess, T, VOCAB)
          t_hp     = TinyNN.tnn_input_1d_f32(sess, 7)
          forward_tower(sess, tow_a, t_tok, t_labels)
          forward_tower(sess, tow_b, t_tok, t_labels)

          # forward roots: both losses; backward: one sweep, two loss roots,
          # tower-disjoint params. ORDER is load-bearing: tnn_add_to_graph
          # refuses once realized, and tnn_build_forward_only SETS realized —
          # so the extra root goes in FIRST (silent -2 otherwise; found the
          # hard way via "tensor buffer not set" on the missing tower's loss).
          TinyNN.tnn_add_to_graph(sess, tow_a.t_loss)
          TinyNN.tnn_build_forward_only(sess, tow_b.t_loss)
          TinyNN.tnn_build_backward(sess)

          # ---- optimizer wiring ----
          # lane A: all chain
          idx = 0
          while idx < tow_a.pp.length
            wire_chain(sess, tow_a, t_hp, idx)
            idx = idx + 1
          end
          # lane B: policy
          p_sm = TinyNN.tnn_softmax(sess, tow_b.t_logits)
          e_b  = TinyNN.tnn_scale(sess, TinyNN.tnn_sub(sess, p_sm, t_labels), 1.0 / T.to_f)
          names = ["q", "k", "v", "o"]
          wire_chain(sess, tow_b, t_hp, 0)
          wire_chain(sess, tow_b, t_hp, 1)
          bi = 0
          li = 0
          while li < LAYERS
            base = 2 + li * 5
            wire_chain(sess, tow_b, t_hp, base)
            if policy[li] == 0
              wi = 1
              while wi < 5
                wire_chain(sess, tow_b, t_hp, base + wi)
                wi = wi + 1
              end
            else
              acts = [tow_b.hs[li], tow_b.hs[li], tow_b.hs[li], tow_b.ctxs[li]]
              wi = 0
              while wi < 4
                w   = tow_b.pp[base + 1 + wi]
                acc = TinyNN.tnn_tensor_grad(sess, w)
                gd  = dfa_grad(sess, bmats[bi], e_b, acts[wi])
                g   = gd
                if policy[li] == 2
                  al = p_alpha[li]
                  g = TinyNN.tnn_add(sess,
                        TinyNN.tnn_scale(sess, acc, al),
                        TinyNN.tnn_scale(sess, gd, 1.0 - al))
                elsif policy[li] == 3
                  mk = smooth_mask(sess, acc, p_tau[li])
                  TinyNN.tnn_set_output(mk)
                  tow_b.masks.push(mk)
                  tow_b.mask_names.push("li=" + li.to_s + " w=" + names[wi])
                  g = TinyNN.tnn_mul(sess, gd, mk)
                elsif policy[li] == 4
                  mk = smooth_mask(sess, gd, p_tau[li])
                  TinyNN.tnn_set_output(mk)
                  tow_b.masks.push(mk)
                  tow_b.mask_names.push("li=" + li.to_s + " w=" + names[wi])
                  g = TinyNN.tnn_mul(sess, acc, mk)
                end
                TinyNN.tnn_set_output(g)
                to = TinyNN.tnn_opt_step_adamw(sess, w, g,
                                                tow_b.pm[base + 1 + wi],
                                                tow_b.pv[base + 1 + wi], t_hp)
                TinyNN.tnn_extend_backward_graph(sess, to)
                tow_b.dfa_grads.push(g)
                tow_b.dfa_accs.push(acc)
                tow_b.dfa_names.push("li=" + li.to_s + " w=" + names[wi])
                bi = bi + 1
                wi = wi + 1
              end
            end
            li = li + 1
          end

          TinyNN.tnn_pin_all_graph_b_nodes(sess)
          TinyNN.tnn_realize_backward(sess)

          # ---- fixed batch (overfit task, hybrid-style) ----
          ids = [1, 2, 3, 4]
          labels = zeros(VOCAB * T)
          tt = 0
          while tt < T
            tgt = (ids[tt] + 1) % VOCAB
            labels[tgt + VOCAB * tt] = 1.0
            tt = tt + 1
          end

          n_dfa = tow_b.dfa_grads.length
          ga = zeros(DM * DM)
          gb = zeros(DM * DM)
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
              TinyNN.tnn_download_to_f64_array(sess, tow_b.dfa_grads[di], gb, DM * DM)
              TinyNN.tnn_download_to_f64_array(sess, tow_b.dfa_accs[di], ga, DM * DM)
              puts "align step=" + s.to_s + " " + tow_b.dfa_names[di] +
                   " cos=" + cosv(gb, ga).to_s
              di = di + 1
            end
            mi = 0
            while mi < tow_b.masks.length
              TinyNN.tnn_download_to_f64_array(sess, tow_b.masks[mi], gb, DM * DM)
              msum = 0.0
              ii = 0
              while ii < DM * DM
                msum = msum + gb[ii]
                ii = ii + 1
              end
              puts "mask step=" + s.to_s + " " + tow_b.mask_names[mi] +
                   " density=" + (msum / (DM * DM).to_f).to_s
              mi = mi + 1
            end
            s = s + 1
          end

          puts "franken summary: lane_a " + first_a.to_s + " -> " + last_a.to_s +
               " | lane_b " + first_b.to_s + " -> " + last_b.to_s
          if last_a != last_a || last_b != last_b
            puts "FRANKEN FAIL: NaN loss"
            return
          end
          puts "FRANKEN DONE"
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

Toy::LLM::Run::TrainFranken.run
