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

        # toy#131: native-form MoE checkpoint — per-tensor moe.* names,
        # toy-moe/v1 metadata, NO fusion and NO second session
        # (tnn_gguf_w_add_tensor reads the CPU session buffers in
        # place, so the write cannot touch the sched).
        def self.write_moe_ckpt(tw, path, run_id, step, shape_s, pol_name, routing_name, dm2, df2, vo2, ne2, ctx2, batch2)
          ctxw = TinyNN.tnn_gguf_w_init
          if ctxw == nil || ctxw == TinyNN.tnn_null_ptr
            return -1
          end
          TinyNN.tnn_gguf_w_set_str(ctxw, "general.architecture", "toy-moe")
          TinyNN.tnn_gguf_w_set_str(ctxw, "general.name", "franken-moe-instrument")
          TinyNN.tnn_gguf_w_set_str(ctxw, "general.run_id", run_id)
          TinyNN.tnn_gguf_w_set_u32(ctxw, "general.step", step)
          TinyNN.tnn_gguf_w_set_str(ctxw, "toy.checkpoint_format", "toy-moe/v1")
          TinyNN.tnn_gguf_w_set_u32(ctxw, "toy.moe.d_model", dm2)
          TinyNN.tnn_gguf_w_set_u32(ctxw, "toy.moe.d_ff", df2)
          TinyNN.tnn_gguf_w_set_u32(ctxw, "toy.moe.vocab_size", vo2)
          TinyNN.tnn_gguf_w_set_u32(ctxw, "toy.moe.n_experts", ne2)
          TinyNN.tnn_gguf_w_set_u32(ctxw, "toy.moe.context", ctx2)
          TinyNN.tnn_gguf_w_set_u32(ctxw, "toy.moe.batch", batch2)
          TinyNN.tnn_gguf_w_set_str(ctxw, "toy.moe.shape", shape_s)
          TinyNN.tnn_gguf_w_set_str(ctxw, "toy.moe.routing", routing_name)
          TinyNN.tnn_gguf_w_set_str(ctxw, "toy.moe.policy", pol_name)
          gi2 = 0
          while gi2 < tw.pp.length
            TinyNN.tnn_gguf_w_add_tensor(ctxw, tw.pp[gi2])
            gi2 = gi2 + 1
          end
          rc = TinyNN.tnn_gguf_w_finalize(ctxw, path)
          TinyNN.tnn_gguf_w_free(ctxw)
          rc
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
          # toy#132 (toy#126 parity): FRANKEN_LR overrides the recipe's
          # lr (default 0.02, byte-null without the flag); FRANKEN_WARMUP
          # ramps linearly over the first N steps (lr_t = LR*(t+1)/N,
          # reaching LR exactly at step N) — the hp vector is rebuilt
          # per step already, so the ramp is zero engine work (the
          # 8ba447d pattern). The end-of-run eval (toy#130) pins its own
          # lr=0 hp and is untouched.
          # toy#136 (K1): FRANKEN_MOE_BALANCE ""|aux (legacy default —
          # the Switch aux-loss, α from FRANKEN_MOE_AUX) | qb (K3
          # §2.3.3 Quantile Balancing: aux-loss-FREE — a per-expert
          # bias on the SELECTION scores only, set each step from the
          # (1−k/n)-quantile of score margins vs the Top-(k+1) cutoff,
          # lag-1 like the aux f'; exact quantile at toy scale; bias
          # frozen for the end-of-run eval) | none. qb requires top1.
          # toy#136 K1.1: FRANKEN_ATTN_GATE=1 — K3's data-dependent
          # attention output gate (eq 7 MLA form). W_g joins the SPINE
          # (chain-wired under bp-spine, param under dense) at the tail
          # index 9+2E; off = not allocated (byte-null).
          attn_gate = (ENV["FRANKEN_ATTN_GATE"] || "") == "1"
          # toy#139: FRANKEN_OPTIMIZER adamw (default, byte-null) | muon
          # | sgd. Muon rides the STANDARD recipe — orthogonalized steps
          # on 2D hidden matrices only, AdamW on embeddings/norms (see
          # franken_moe_parts.muon_eligible). The scientifically novel
          # bit, per tao#139: under --moe-policy dfa-experts/bp-spine
          # this orthogonalizes the DFA PSEUDO-gradient (B·e·hᵀ), a
          # rank-structured near-random direction that Muon's premise
          # (momentum ≈ a meaningful gradient) does not describe —
          # whether that helps or is inert IS the experiment.
          # toy#140 (F10): --donor <gguf> initialises the spine's input
          # embedding from a SAME-VOCAB donor, projected to this
          # instrument's width (see parts.donor_embed_values for the
          # projection + scale-matching choices). --freeze-embed keeps
          # it fixed (granite's strongest arm in several results).
          donor_s = ENV["FRANKEN_DONOR"] || ""
          if donor_s.length > 0 && !File.exist?(donor_s)
            puts "toy-train-franken-moe: donor not found: " + donor_s
            return 1
          end
          donor_mode = ENV["FRANKEN_DONOR_MODE"] || ""
          if donor_mode.length > 0 && donor_mode != "tied"
            puts "toy-train-franken-moe: --donor-mode " + donor_mode +
                 " unsupported — this instrument is TIED-embedding by construction (logits = embed^T x); an untied head is a follow-up, say if F10 needs it"
            return 1
          end
          freeze_embed = (ENV["FRANKEN_FREEZE_EMBED"] || "") == "1"
          if freeze_embed && donor_s.length == 0
            puts "toy-train-franken-moe: --freeze-embed without --donor freezes a RANDOM embedding — pass a donor, or drop the flag"
            return 1
          end
          opt_s = ENV["FRANKEN_OPTIMIZER"] || ""
          if opt_s.length > 0 && opt_s != "adamw" && opt_s != "muon" && opt_s != "sgd"
            puts "toy-train-franken-moe: unknown FRANKEN_OPTIMIZER " + opt_s + " (adamw|muon|sgd)"
            return 1
          end
          opt_code = 0
          if opt_s == "muon"
            opt_code = 1
          end
          if opt_s == "sgd"
            opt_code = 2
          end
          bal_s = ENV["FRANKEN_MOE_BALANCE"] || ""
          if bal_s.length > 0 && bal_s != "aux" && bal_s != "qb" && bal_s != "none"
            puts "toy-train-franken-moe: unknown FRANKEN_MOE_BALANCE " + bal_s + " (aux|qb|none)"
            return 1
          end
          qb_on = bal_s == "qb"
          # FRANKEN_SCHEDULE ""|const|cosine (K3 M9; cosine decays LR
          # -> 0.1*LR post-warmup; libm cos => platform-scoped curves).
          sched_s = ENV["FRANKEN_SCHEDULE"] || ""
          if sched_s.length > 0 && sched_s != "const" && sched_s != "cosine"
            puts "toy-train-franken-moe: unknown FRANKEN_SCHEDULE " + sched_s + " (const|cosine)"
            return 1
          end
          cosine_on = sched_s == "cosine"
          lr_s = ENV["FRANKEN_LR"] || ""
          lr = lr_s.length > 0 ? lr_s.to_f : 0.02
          # toy#131: the toy#120 deviation come due — a NATIVE-form GGUF
          # checkpoint (per-tensor moe.* names, NO fusion, no second
          # session: tnn_gguf_w_add_tensor reads the CPU session buffers
          # directly, so the write cannot touch the sched). Checkpoints
          # carry WEIGHTS only (no Adam moments) — FRANKEN_MOE_LOAD is
          # therefore EVAL-ONLY (STEPS=0 + the toy#130 eval loop);
          # resume fails loud.
          ck_raw = (ENV["FRANKEN_CKPT_EVERY"] || "0").to_i
          ckpt_every = ck_raw < 0 ? 0 : ck_raw
          load_ckpt = ENV["FRANKEN_MOE_LOAD"] || ""
          if load_ckpt.length > 0 && !File.exist?(load_ckpt)
            puts "toy-train-franken-moe: no such checkpoint: " + load_ckpt
            return 1
          end
          if load_ckpt.length > 0 && steps != 0
            puts "toy-train-franken-moe: FRANKEN_MOE_LOAD is eval-only (checkpoints carry weights, not Adam moments — resume is unsupported). Pass STEPS=0 + FRANKEN_EVAL_CORPUS."
            return 1
          end
          wu_raw = (ENV["FRANKEN_WARMUP"] || "0").to_i
          warmup = wu_raw < 0 ? 0 : wu_raw
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
          if qb_on && !top1
            puts "toy-train-franken-moe: --moe-balance qb requires --routing top1 (QB biases hard SELECTION; dense has no selection to bias)"
            return 1
          end
          if qb_on && aux_alpha > 0.0
            puts "toy-train-franken-moe: --moe-balance qb replaces the aux loss — drop --moe-aux (K3 runs QB aux-free)"
            return 1
          end
          # toy#130: END-OF-RUN held-out eval — the MoE instrument has no
          # GGUF writer (the vit-tiny #169 precedent), so checkpoint-
          # boundary evals cannot carry this lane; instead the runner
          # evals AFTER training completes (lr=0 windows on the SAME
          # graph — the Adam moments it mutates are dead at that point,
          # so the toy#122 pollution objection is moot). Unset = off,
          # byte-null.
          ev_corpus = ENV["FRANKEN_EVAL_CORPUS"] || ""
          ev_tok_s = ENV["FRANKEN_EVAL_TOKENS"] || ""
          ev_tokens_raw = ev_tok_s.length > 0 ? ev_tok_s.to_i : 0
          ev_tokens = ev_tokens_raw > 0 ? ev_tokens_raw : 4096   # 0 = unset (the CLI's default)
          ev_off_s = ENV["FRANKEN_EVAL_OFFSET"] || ""
          ev_offset = ev_off_s.length > 0 ? ev_off_s.to_i : 0
          if ev_corpus.length > 0 && !File.exist?(ev_corpus)
            puts "toy-train-franken-moe: eval corpus not found: " + ev_corpus
            return 1
          end
          if load_ckpt.length > 0 && ev_corpus.length == 0
            puts "toy-train-franken-moe: FRANKEN_MOE_LOAD without FRANKEN_EVAL_CORPUS does nothing — pass the held-out pack"
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
          # toy#133: B windows per step. Reading ctx*B contiguous tokens
          # IS reading B windows (the MoE instrument is position-free);
          # only the attention MASK (block-causal, window-isolated) and
          # the labels (window-local) distinguish batched from one long
          # window. B=1 is byte-null (NULL mask = the diag_mask path).
          bat_s = ENV["FRANKEN_BATCH"] || ""
          bat_raw = bat_s.length > 0 ? bat_s.to_i : 0
          batch = bat_raw > 0 ? bat_raw : 1
          if batch > 1 && corpus_s.length == 0
            puts "toy-train-franken-moe: FRANKEN_BATCH > 1 needs --corpus (the fixed-seq feed is the byte-gated single-window contract)"
            return 1
          end
          tbv = ctx * batch
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
          if ev_corpus.length > 0
            ev_hdr_v = ToyCorpusLoader.probe_vocab(ev_corpus)
            if ev_hdr_v > 0 && ev_hdr_v != vsel
              puts "toy-train-franken-moe: eval pack vocab " + ev_hdr_v.to_s +
                   " (TOYC header) != instrument vocab " + vsel.to_s
              return 1
            end
          end
          if shape_s == "wide"
            shape_init(256, 512, vsel, esel, tbv)
          else
            shape_init(DM_BASE, DFF_BASE, vsel, esel, tbv)
          end
          attn_gate_init(attn_gate ? 1 : 0)
          opt_init(opt_code)
          # toy#136 K1.1: the spine bound. Without the gate the spine is
          # 0..8 (embed/fnorm/attention/rn2/Wr); with it, W_g at 9+2E
          # joins — so spine membership is "gi < 9 || gi == gate_idx"
          # and the all-chain wire bound grows by one.
          gate_i = attn_gate ? gate_idx : -1
          n_all = 9 + 2 * esel + (attn_gate ? 1 : 0)

          sess = TinyNN.tnn_session_new(0)
          TinyNN.tnn_session_set_graph_capacity(sess, 262144)

          tw = alloc_tower(sess)
          # toy#131: names are the checkpoint contract (write by name,
          # load by name — no positional coupling).
          TinyNN.tnn_tensor_set_name(tw.pp[0], "moe.embed")
          TinyNN.tnn_tensor_set_name(tw.pp[1], "moe.fnorm")
          TinyNN.tnn_tensor_set_name(tw.pp[2], "moe.attn_rn1")
          TinyNN.tnn_tensor_set_name(tw.pp[3], "moe.wq")
          TinyNN.tnn_tensor_set_name(tw.pp[4], "moe.wk")
          TinyNN.tnn_tensor_set_name(tw.pp[5], "moe.wv")
          TinyNN.tnn_tensor_set_name(tw.pp[6], "moe.wo")
          TinyNN.tnn_tensor_set_name(tw.pp[7], "moe.rn2")
          TinyNN.tnn_tensor_set_name(tw.pp[8], "moe.wr")
          nmi = 0
          while nmi < nev
            TinyNN.tnn_tensor_set_name(tw.pp[9 + 2 * nmi],  "moe.expert_" + nmi.to_s + ".up")
            TinyNN.tnn_tensor_set_name(tw.pp[10 + 2 * nmi], "moe.expert_" + nmi.to_s + ".down")
            nmi = nmi + 1
          end
          if attn_gate
            TinyNN.tnn_tensor_set_name(tw.pp[gate_idx], "moe.attn_gate")
          end

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
          # toy#139: ggml's SGD step takes exactly [alpha, wd]; muon and
          # sgd both apply through it. PERSISTENT (alloc BEFORE
          # finalize_weights) — a compute-context input gets no backing
          # buffer when the graph does not reach it, and the per-step
          # upload then aborts inside ggml_backend_tensor_set.
          t_hp_sgd = TinyNN.tnn_input_1d_f32_persistent(sess, 2)
          # toy#133: block-causal attention mask — ALLOCATED here
          # (persistent inputs must precede finalize_weights; an alloc
          # after finalize has no backing buffer and silently reads
          # ZEROS — found the hard way: add(scaled, 0) is a no-op and
          # B=2 ran fully-unmasked).
          attn_mask = TinyNN.tnn_null_ptr
          if batch > 1
            attn_mask = TinyNN.tnn_input_2d_f32_persistent(sess, tv, tv)
          end
          # toy#136: QB selection-bias — persistent (alloc BEFORE
          # finalize, the toy#133 lesson), uploaded per step (lag-1).
          qb_bias_t = TinyNN.tnn_null_ptr
          if qb_on
            qb_bias_t = TinyNN.tnn_input_2d_f32_persistent(sess, 1, nev)   # ne=[NE,1] — broadcasts over T
          end

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
              mark = bp_spine && (gi < 9 || gi == gate_i)
            else
              mark = !(no_shadow && gi >= 9)
            end
            # toy#140: a frozen donor embedding is not a param at all —
            # no grad, no opt node, and (never-mask) no silent update.
            if freeze_embed && gi == 0
              mark = false
            end
            if mark
              TinyNN.tnn_set_param(tw.pp[gi])
            end
            gi = gi + 1
          end
          if top1 && !bp_spine
            TinyNN.tnn_set_param(tw.pp[8])
          end
          # toy#136 K1.1: in the fully-DFA / bp-router top1 lanes W_g is
          # FROZEN at init (no DFA wire): its input is the embedding
          # output, a tap the DFA plumbing does not carry, and the gate
          # is a bp-spine-context mechanism in K3. Fail loud rather than
          # silently training a dead weight.
          if attn_gate && top1 && !bp_spine
            puts "toy-train-franken-moe: --attn-gate needs --moe-policy bp-spine under top1 (the fully-DFA/bp-router lanes have no tap for W_g's input — it would sit frozen)"
            return 1
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
          # toy#131: eval-only reload — overwrite every pp tensor by
          # NAME from the checkpoint; shape metadata must match the
          # instrument exactly (fail loud, listing both sides).
          if load_ckpt.length > 0
            gg_l = TinyNN.tnn_gguf_load(load_ckpt)
            if gg_l == nil || gg_l == TinyNN.tnn_null_ptr
              puts "toy-train-franken-moe: cannot open checkpoint " + load_ckpt
              return 1
            end
            ck_dm = TinyNN.tnn_gguf_get_u32(gg_l, "toy.moe.d_model")
            ck_df = TinyNN.tnn_gguf_get_u32(gg_l, "toy.moe.d_ff")
            ck_vo = TinyNN.tnn_gguf_get_u32(gg_l, "toy.moe.vocab_size")
            ck_ne = TinyNN.tnn_gguf_get_u32(gg_l, "toy.moe.n_experts")
            if ck_dm != dmv || ck_df != dfv || ck_vo != vocabv || ck_ne != nev
              puts "toy-train-franken-moe: checkpoint shape mismatch — ckpt d_model=" + ck_dm.to_s +
                   " d_ff=" + ck_df.to_s + " vocab=" + ck_vo.to_s + " experts=" + ck_ne.to_s +
                   " vs instrument d_model=" + dmv.to_s + " d_ff=" + dfv.to_s +
                   " vocab=" + vocabv.to_s + " experts=" + nev.to_s +
                   " (pass the matching --shape/--corpus/--vocab/--experts)"
              return 1
            end
            gi3 = 0
            while gi3 < tw.pp.length
              nm = TinyNN.tnn_tensor_name(tw.pp[gi3])
              idx3 = TinyNN.tnn_gguf_find_index(gg_l, nm)
              if idx3 < 0
                puts "toy-train-franken-moe: checkpoint missing tensor " + nm
                return 1
              end
              nel3 = TinyNN.tnn_tensor_nelements(tw.pp[gi3])
              mv3 = Mat.new(1, nel3)
              TinyNN.tnn_gguf_read_f32_to_doubles(gg_l, idx3, mv3.flat, nel3)
              TinyNN.tnn_upload_from_float_array(sess, tw.pp[gi3], mv3.flat, nel3)
              gi3 = gi3 + 1
            end
            puts "loaded checkpoint: " + load_ckpt
          end
          # toy#140: overwrite the randomly-initialised embed with the
          # projected donor. AFTER the init loop so every other weight
          # keeps its seeded stream byte-for-byte — the donor arm
          # differs from the scratch arm in the EMBEDDING ONLY.
          if donor_s.length > 0
            dvals = donor_embed_values(donor_s, vocabv, dmv, seed)
            if dvals.length != dmv * vocabv
              puts "toy-train-franken-moe: donor projection failed"
              return 1
            end
            TinyNN.tnn_upload_from_float_array(sess, tw.pp[0], dvals, dmv * vocabv)
            puts "donor: loaded " + donor_s + " -> embed " + vocabv.to_s + "x" + dmv.to_s +
                 (freeze_embed ? " (frozen)" : " (trainable)")
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
          qb_bias = zeros(nev)
          if qb_on
            TinyNN.tnn_upload_from_float_array(sess, qb_bias_t, qb_bias, nev)
          end
          # toy#133: block-causal mask VALUES (the GH#7 fill, verbatim
          # orientation — the order-swap isolation null gates it).
          if batch > 1
            mvals = zeros(tv * tv)
            mi1 = 0
            while mi1 < tv
              bq = mi1 / ctx
              pq = mi1 % ctx
              mi0 = 0
              while mi0 < tv
                bk = mi0 / ctx
                pk = mi0 % ctx
                if bk == bq && pk <= pq
                  mvals[mi1 * tv + mi0] = 0.0
                else
                  mvals[mi1 * tv + mi0] = -1.0e30
                end
                mi0 = mi0 + 1
              end
              mi1 = mi1 + 1
            end
            TinyNN.tnn_upload_from_float_array(sess, attn_mask, mvals, tv * tv)
          end
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
          forward_tower(sess, tw, t_tok, t_labels, sels, eye, top1, bp_spine ? 1 : 0, attn_mask, qb_bias_t)
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
                if !(freeze_embed && sp == 0)
                  wire_chain(sess, tw, t_hp, t_hp_sgd, sp)
                end
                sp = sp + 1
              end
              if gate_i >= 0
                wire_chain(sess, tw, t_hp, t_hp_sgd, gate_i)
              end
            else
              wire_dfa_top1(sess, tw, t_hp, t_hp_sgd, 3, b_aq, e_b, tw.tap_ah,  dmv, dmv, np2)
              wire_dfa_top1(sess, tw, t_hp, t_hp_sgd, 4, b_ak, e_b, tw.tap_ah,  dmv, dmv, np2)
              wire_dfa_top1(sess, tw, t_hp, t_hp_sgd, 5, b_av, e_b, tw.tap_ah,  dmv, dmv, np2)
              wire_dfa_top1(sess, tw, t_hp, t_hp_sgd, 6, b_ao, e_b, tw.tap_ctx, dmv, dmv, np2)
            end
            if bp_spine
              # router already chain-wired above; experts follow below
            elsif bp_router
              # toy#121: router credit is PURE BP — the acc already holds
              # task-BP + aux-BP (both loss roots backward into it).
              wire_chain(sess, tw, t_hp, t_hp_sgd, 8)
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
              wire_dfa_top1(sess, tw, t_hp, t_hp_sgd, 9 + 2 * ei,  b_ups[ei],   e_b, tw.tap_h2, dmv,  dfv, m_i)
              wire_dfa_top1(sess, tw, t_hp, t_hp_sgd, 10 + 2 * ei, b_downs[ei], e_b, tw.tap_a1, dfv, dmv,  m_i)
              ei = ei + 1
            end
          else
            if dfa_experts
              p_sm = TinyNN.tnn_softmax(sess, tw.t_logits)
              e_b  = TinyNN.tnn_scale(sess, TinyNN.tnn_sub(sess, p_sm, t_labels), 1.0 / tv.to_f)
              idx = 0
              while idx < 9
                if !(freeze_embed && idx == 0)
                  wire_chain(sess, tw, t_hp, t_hp_sgd, idx)
                end
                idx = idx + 1
              end
              if gate_i >= 0
                wire_chain(sess, tw, t_hp, t_hp_sgd, gate_i)
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
                  wire_dfa_top1(sess, tw, t_hp, t_hp_sgd, 9 + 2 * ei,  b_ups[ei],   e_b, tw.tap_h2,     dmv,  dfv, np3)
                  wire_dfa_top1(sess, tw, t_hp, t_hp_sgd, 10 + 2 * ei, b_downs[ei], e_b, tw.tap_as[ei], dfv, dmv,  np3)
                else
                  wire_dfa(sess, tw, t_hp, t_hp_sgd, 9 + 2 * ei,  b_ups[ei],   e_b, tw.tap_h2,    dmv,  dfv, "up" + (ei + 1).to_s)
                  wire_dfa(sess, tw, t_hp, t_hp_sgd, 10 + 2 * ei, b_downs[ei], e_b, tw.tap_as[ei], dfv, dmv,  "down" + (ei + 1).to_s)
                end
                ei = ei + 1
              end
            else
              idx = 0
              while idx < n_all
                if !(freeze_embed && idx == 0)
                  wire_chain(sess, tw, t_hp, t_hp_sgd, idx)
                end
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
              config.add_num("context", ctx)
              config.add_num("batch",   batch)
              config.add_num("steps",   steps)
              config.add_num("lr",      lr)
              config.add_num("warmup",  warmup)
              config.add_str("schedule", cosine_on ? "cosine" : "const")
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
              gate_p = attn_gate ? dmv * dmv : 0
              cost_total  = vocabv * dmv + 3 * dmv + 4 * dmv * dmv + gate_p +
                            nev * dmv + nev * 2 * dmv * dfv
              cost_active = vocabv * dmv + 3 * dmv + 4 * dmv * dmv + gate_p +
                            nev * dmv + act_e * 2 * dmv * dfv
              cost_flops  = 2 * (4 * dmv * dmv) + 2 * gate_p + 4 * dmv * tv +
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
              fm.add_str("balance",   qb_on ? "qb" : (bal_s == "none" ? "none" : "aux"))
              fm.add_bool("attn_gate", attn_gate)
              fm.add_str("optimizer", opt_s.length > 0 ? opt_s : "adamw")
              fm.add_str("donor", donor_s)
              fm.add_str("donor_mode", donor_s.length > 0 ? "tied" : "")
              fm.add_str("donor_projection", donor_s.length > 0 ? "random-jl,scale-matched" : "")
              fm.add_bool("freeze_embed", freeze_embed)
              # the per-param-class routing, recorded so a bundle can be
              # audited without re-reading the runner (tao#139 asked).
              fm.add_str("optimizer_routing",
                         opt_code == 1 ? "muon:2d-hidden,adamw:embed+norms" :
                         (opt_code == 2 ? "sgd:all" : "adamw:all"))
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
          # toy#133: incremental batched one-hot (the eval_ce trick) —
          # clear only the previous tv scatter positions per step, never
          # a tv*vocab refill. Values byte-identical to the builder.
          lab_inc = corpus_s.length > 0 ? Mat.new(tv, vocabv) : Mat.new(1, 1)
          lab_prev = [0]; lab_prev.pop
          lp = 0; while lp < tv; lab_prev.push(-1); lp = lp + 1; end
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
            lr_t = lr
            if warmup > 0 && s < warmup
              # linear ramp; at step warmup-1 the factor is exactly 1.0
              lr_t = lr * ((s + 1).to_f / warmup.to_f)
            elsif cosine_on
              span = steps - warmup
              prog = span > 0 ? ((s - warmup).to_f / span.to_f) : 1.0
              min_lr = lr * 0.1
              lr_t = min_lr + 0.5 * (lr - min_lr) * (1.0 + Math.cos(3.141592653589793 * prog))
            end
            hp = [lr_t, b1, b2, 1.0e-8, 0.0,
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
              k2 = 0
              while k2 < tv
                if lab_prev[k2] >= 0
                  lab_inc.flat[k2 * vocabv + lab_prev[k2]] = 0.0
                end
                wpos = k2 % ctx
                tgt = (wpos + 1 < ctx) ? ids[k2 + 1] : ids[k2]
                if tgt >= 0 && tgt < vocabv
                  lab_inc.flat[k2 * vocabv + tgt] = 1.0
                  lab_prev[k2] = tgt
                else
                  lab_prev[k2] = -1
                end
                k2 = k2 + 1
              end
              labels = lab_inc.flat
            end
            TinyNN.upload_int_array(sess, t_tok, ids)
            TinyNN.tnn_upload_from_float_array(sess, t_labels, labels, vocabv * tv)
            # At --optimizer sgd NOTHING uses opt_step_adamw, so t_hp
            # (a compute-context input) is unreachable from the graph
            # and has no buffer — uploading it aborts in
            # ggml_backend_tensor_set. muon still routes embeds/norms
            # through adamw, so it keeps the upload.
            if opt_code != 2
              TinyNN.tnn_upload_from_float_array(sess, t_hp, hp, 7)
            end
            if opt_code != 0
              hp_sgd = [lr_t, 0.0]
              TinyNN.tnn_upload_from_float_array(sess, t_hp_sgd, hp_sgd, 2)
            end
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
              es.add_raw("lr",    lr_t.to_s)
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
            # toy#136 QB update (lag-1, exact at toy scale): for k=1 the
            # Top-(k+1) cutoff per token is the 2nd-largest BIASED score;
            # margins m_ij = s_ij − cutoff_i; next bias b_j = −quantile
            # at rank ceil(m·k/n) of expert j's margins (so its expected
            # load matches q = m·k/n tokens), then mean-centered.
            if qb_on
              rl_buf = zeros(nev * tv)
              TinyNN.tnn_download_to_f64_array(sess, tw.t_rlogits, rl_buf, nev * tv)
              cutoffs = zeros(tv)
              ti4 = 0
              while ti4 < tv
                best = -1.0e30
                second = -1.0e30
                ei4 = 0
                while ei4 < nev
                  v = rl_buf[ti4 * nev + ei4] + qb_bias[ei4]
                  if v > best
                    second = best
                    best = v
                  elsif v > second
                    second = v
                  end
                  ei4 = ei4 + 1
                end
                cutoffs[ti4] = second
                ti4 = ti4 + 1
              end
              qtarget = tv / nev
              if qtarget < 1
                qtarget = 1
              end
              nb_sum = 0.0
              nb = zeros(nev)
              ei4 = 0
              while ei4 < nev
                margins = zeros(tv)
                ti4 = 0
                while ti4 < tv
                  margins[ti4] = rl_buf[ti4 * nev + ei4] - cutoffs[ti4]
                  ti4 = ti4 + 1
                end
                margins.sort!
                # rank: exactly qtarget margins should exceed -b_j ->
                # -b_j = the (qtarget+1)-th largest = margins[tv-qtarget-1]
                idx4 = tv - qtarget - 1
                if idx4 < 0
                  idx4 = 0
                end
                nb[ei4] = 0.0 - margins[idx4]
                nb_sum = nb_sum + nb[ei4]
                ei4 = ei4 + 1
              end
              nb_mean = nb_sum / nev.to_f
              ei4 = 0
              while ei4 < nev
                qb_bias[ei4] = nb[ei4] - nb_mean
                ei4 = ei4 + 1
              end
              TinyNN.tnn_upload_from_float_array(sess, qb_bias_t, qb_bias, nev)
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
            if ckpt_every > 0 && run_dir.length > 0 &&
               ((s + 1) % ckpt_every) == 0 && (s + 1) < steps
              ck_dir = run_dir + "/weights"
              TinyNN.tnn_filesystem_mkdir(ck_dir)
              ck_rid = run_id_s.length > 0 ? run_id_s : "anonymous"
              rc_ck = write_moe_ckpt(tw, ck_dir + "/step_" + (s + 1).to_s + ".gguf", ck_rid, s + 1,
                                     shape_s, pol_name, (top1 ? "top1" : "dense"),
                                     dmv, dfv, vocabv, nev, ctx, batch)
              if rc_ck != 0
                puts "checkpoint write failed: step=" + (s + 1).to_s + " rc=" + rc_ck.to_s
              end
            end
            s = s + 1
          end
          if ckpt_every > 0 && run_dir.length > 0 && steps > 0
            ck_dir = run_dir + "/weights"
            TinyNN.tnn_filesystem_mkdir(ck_dir)
            ck_rid = run_id_s.length > 0 ? run_id_s : "anonymous"
            rc_ck = write_moe_ckpt(tw, ck_dir + "/step_" + steps.to_s + ".gguf", ck_rid, steps,
                                   shape_s, pol_name, (top1 ? "top1" : "dense"),
                                   dmv, dfv, vocabv, nev, ctx, batch)
            if rc_ck != 0
              puts "checkpoint write failed: step=" + steps.to_s + " rc=" + rc_ck.to_s
            end
          end

          # toy#130: end-of-run held-out eval (lr=0 windows; weights
          # frozen — AdamW with lr=0 is a weight no-op).
          if ev_corpus.length > 0
            ev_base  = ToyCorpusLoader.data_offset(ev_corpus)
            ev_bytes = File.size(ev_corpus)
            ev_hp = [0.0, b1, b2, 1.0e-8, 0.0, b1, b2]
            ev_want = ev_tokens / tv
            ev_sum = 0.0
            ev_done = 0
            ev_o = ev_base + ev_offset * 4
            evw = 0
            while evw < ev_want
              if ev_o + tv * 4 > ev_bytes
                break
              end
              ev_ids = ToyCorpusLoader.read_seq(ev_corpus, ev_o, tv)
              ev_o = ev_o + tv * 4
              evk = 0
              while evk < tv
                if ev_ids[evk] < 0 || ev_ids[evk] >= vocabv
                  puts "toy-train-franken-moe: eval token id " + ev_ids[evk].to_s +
                       " outside [0, " + vocabv.to_s + ") — pack/instrument mismatch"
                  return 1
                end
                evk = evk + 1
              end
              ev_lab = Toy::Labels.next_token_guarded_batched(ev_ids, vocabv, ctx, batch)
              TinyNN.tnn_graph_reset_grads_only(sess)
              TinyNN.upload_int_array(sess, t_tok, ev_ids)
              TinyNN.tnn_upload_from_float_array(sess, t_labels, ev_lab.flat, vocabv * tv)
              TinyNN.tnn_upload_from_float_array(sess, t_hp, ev_hp, 7)
              if top1
                TinyNN.tnn_upload_from_float_array(sess, t_f, fvec, nev)
              end
              TinyNN.tnn_compute_backward(sess)
              TinyNN.tnn_download(sess, tw.t_loss)
              ev_sum = ev_sum + TinyNN.tnn_scratch_get(sess, 0)
              ev_done = ev_done + 1
              evw = evw + 1
            end
            if ev_done == 0
              puts "toy-train-franken-moe: zero eval windows (offset past the pack end)"
              return 1
            end
            ev_ce = ev_sum / ev_done.to_f
            puts "eval_ce: windows=" + ev_done.to_s +
                 " tokens=" + (ev_done * tv).to_s +
                 " ce=" + ev_ce.to_s
            if events.length > 0
              eve = Toy::Json::Builder.new
              eve.add_str("kind",  "eval")
              eve.add_str("phase", "eval")
              eve.add_num("t",       TinyNN.tnn_events_now_seconds)
              eve.add_str("name",    "eval-ce")
              eve.add_num("step",    steps)
              eve.add_raw("loss",    num_or_null_cli(ev_ce))
              eve.add_num("windows", ev_done)
              eve.add_num("tokens",  ev_done * tv)
              TinyNN.tnn_events_emit(eve.dump)
            end
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
