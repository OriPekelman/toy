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
require_relative "../llm/primitives/situ_glu"

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
        # toy#146: which transformer layer owns weight `idx`.
        # -1 = a GLOBAL (embed, fnorm) — outside the depth stack, so it
        # is not on the ramp and keeps the base LR.
        def self.layer_of(idx)
          if idx < 2
            return -1
          end
          (idx - 2) / per_layer_count
        end

        # toy#146: THE seam for per-layer learning rates, and the reason
        # this is a general mechanism rather than a DFA one. Every
        # optimizer step for every weight — chain, DFA, block-DFA, top1,
        # adamw/muon/sgd — funnels through apply_step, so swapping the hp
        # vector here reaches all of them at once and nothing else has to
        # know the schedule exists.
        #
        # An EMPTY t_hps means "uniform": the caller's single shared
        # vector is returned unchanged, so the default path builds the
        # identical graph it always did.
        def self.hp_for(tw, idx, t_hp_default)
          if tw.t_hps.length == 0
            return t_hp_default
          end
          l = layer_of(idx)
          if l < 0 || l >= tw.t_hps.length
            return t_hp_default
          end
          tw.t_hps[l]
        end

        def self.hp_sgd_for(tw, idx, t_hp_default)
          if tw.t_hps_sgd.length == 0
            return t_hp_default
          end
          l = layer_of(idx)
          if l < 0 || l >= tw.t_hps_sgd.length
            return t_hp_default
          end
          tw.t_hps_sgd[l]
        end

        def self.apply_step(sess, tw, t_hp, t_hp_sgd, idx, grad)
          t_hp     = hp_for(tw, idx, t_hp)
          t_hp_sgd = hp_sgd_for(tw, idx, t_hp_sgd)
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

        def self.gate_idx(l)
          layer_base(l) + LOFF_EXP + 2 * nev
        end

        # toy#142 (K4) Stable LatentMoE, K3 §2.3:
        #   u = Σ_i p_i · E_i^routed(W↓ x)        (routed experts in LATENT space ℓ)
        #   y = Σ_j E_j^shared(x) + W↑ RMSNorm(u)  (shared experts at FULL width)
        # The RMSNorm between aggregation and up-projection IS the
        # "Stable" of the name (§2.3.1): without it W↑ sees a scale that
        # varies with which experts fired and how confidently.
        #
        # lat = 0 disables the sandwich (the byte-null legacy shape); ns
        # = 0 disables shared experts. They are SEPARATE axes on
        # purpose — K3 ships both together, but "does the latent
        # bottleneck cost anything" and "do shared experts carry the
        # common transformation" are different questions.
        #
        # Layout: everything NEW is APPENDED past the experts (and past
        # the attn gate), so the spine stays 0..8, the expert pairs stay
        # 9+2i/10+2i, and every DfaB seed keeps its index — the toy#128
        # discipline. The ROUTED expert SHAPES do change under latent
        # (they now map ℓ→dff→ℓ), which is a mode difference, not an
        # index one.
        def self.latent_init(lat, ns)
          @sh_lat = lat
          @sh_ns = ns
          0
        end

        # K4b / M6: 0 = GELU experts (the byte-null default), 1 =
        # SiTU-GLU experts. A GLU needs a SECOND projection per expert
        # (the gate branch), which the 9+2i/10+2i pair has no room for —
        # so the E gate matrices are appended at the VERY tail, past the
        # spine tail, and every existing index keeps its meaning and its
        # DfaB seed. They are EXPERT weights, not spine: DFA-wired like
        # up_i, frozen by --freeze-experts, counted in experts_sig.
        def self.expert_act_init(v)
          @sh_eact = v
          0
        end

        # toy#145: DEPTH. The tower was one attention block + one MoE
        # block with a hard-coded layout (embed 0, fnorm 1, spine 2..8,
        # experts 9+2i, tails past that). `--shape deep` repeats the
        # WHOLE (attention + MoE) layer L times, because the block-DFA
        # question is about how many blocks the random-projected error
        # spans — deepening only the attention spine would leave exactly
        # one routed-expert block and answer nothing.
        #
        # The layout is now uniform and layer-relative:
        #   0 embed, 1 fnorm, then L blocks of
        #   [rn1 wq wk wv wo rn2 wr | E expert pairs | spine tail | glu tail]
        # At L=1, layer_base(0) == 2 reproduces the old indices
        # EXACTLY — same order, same DfaB seeds, same checkpoint names —
        # which is what keeps every recorded fixture and the cross-binary
        # rig null valid.
        def self.layers_init(l)
          @sh_nl = l
          0
        end

        def self.nlv
          @sh_nl
        end

        def self.eactv
          @sh_eact
        end

        def self.latv
          @sh_lat
        end

        def self.nsv
          @sh_ns
        end

        # expert width: the routed experts read/write ℓ under latent,
        # d_model otherwise.
        def self.ewidth
          latv > 0 ? latv : dmv
        end

        # within-layer offsets of the seven spine weights
        LOFF_RN1 = 0
        LOFF_WQ  = 1
        LOFF_WK  = 2
        LOFF_WV  = 3
        LOFF_WO  = 4
        LOFF_RN2 = 5
        LOFF_WR  = 6
        LOFF_EXP = 7    # first expert pair

        def self.per_layer_count
          LOFF_EXP + 2 * nev + tail_spine_count + eglu_count
        end

        def self.layer_base(l)
          2 + l * per_layer_count
        end

        # expert i's up / down within layer l
        def self.up_idx(l, i)
          layer_base(l) + LOFF_EXP + 2 * i
        end

        def self.down_idx(l, i)
          layer_base(l) + LOFF_EXP + 2 * i + 1
        end

        def self.lat_base(l)
          layer_base(l) + LOFF_EXP + 2 * nev + (gatev == 1 ? 1 : 0)
        end

        def self.shared_base(l)
          lat_base(l) + (latv > 0 ? 3 : 0)
        end

        # every tail index that belongs to the SPINE (chain-wired under
        # bp-spine, param under dense): the attn gate, the latent
        # sandwich, and the shared experts.
        def self.tail_spine_count
          (gatev == 1 ? 1 : 0) + (latv > 0 ? 3 : 0) + 2 * nsv
        end

        # First index of the EXPERT-GATE tail (K4b) — immediately after
        # the spine tail. Zero of them unless eactv == 1, in which case
        # eglu_base + i is expert i's gate projection.
        def self.eglu_base(l)
          shared_base(l) + 2 * nsv
        end

        # toy#145: layer-aware membership. With L repeats the old
        # "gi < 9 is spine, 9..8+2E is experts" arithmetic no longer
        # holds, and getting it wrong is SILENT — a weight in the wrong
        # class simply trains under the wrong credit rule. These two are
        # the single source of truth for that split.
        #
        # Per layer the block is
        #   [7 spine][2E experts][spine tail][E glu]
        # so within-layer offset decides the class.
        def self.is_expert_idx(gi)
          if gi < 2
            return false
          end
          off = (gi - 2) % per_layer_count
          if off >= LOFF_EXP && off < LOFF_EXP + 2 * nev
            return true
          end
          off >= per_layer_count - eglu_count && eglu_count > 0
        end

        def self.is_spine_idx(gi)
          !is_expert_idx(gi)
        end

        # total registered weights
        def self.n_weights
          2 + nlv * per_layer_count
        end

        def self.eglu_count
          eactv == 1 ? nev : 0
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

        # toy#140 (F10): DONOR EMBEDDING TRANSFER. Read a donor GGUF's
        # token_embd.weight (V x D_donor) and project it to the
        # instrument's width (V x D_target), returning the values ready
        # to upload into pp[0].
        #
        # SAME VOCAB REQUIRED — this does NO token remapping. Our
        # fixture is GPT-2 BPE (50257) and data/distilgpt2-f32.gguf is a
        # GPT-2-vocab donor, so the ids line up by construction; a
        # donor with a different vocab fails loud rather than
        # transferring garbage.
        #
        # PROJECTION = FIXED SEEDED RANDOM (Johnson-Lindenstrauss), not
        # PCA. Stated plainly because it is a deviation from granite's
        # recipe: a random projection preserves PAIRWISE GEOMETRY in
        # expectation (which is the transfer signal — tokens that were
        # neighbours stay neighbours), while PCA would additionally
        # concentrate variance in the leading directions. PCA needs an
        # SVD of a 50257x768 matrix; JL needs one matmul. If F10 shows
        # the transfer signal is weak, PCA is the first thing to try.
        #
        # SCALE-MATCHED: after projecting, the result is rescaled to the
        # std the instrument's own random init would have produced
        # (fillv is uniform on +/-0.5, std = 1/sqrt(12)). Without this a
        # pure SCALE difference would move the loss and the gate would
        # be measuring the wrong thing.
        #
        # The projection runs in a THROWAWAY session (one matmul on
        # V x D_donor), freed before returning.
        def self.donor_embed_values(donor_path, vocab, dm, seed)
          gg = TinyNN.tnn_gguf_load(donor_path)
          if gg == nil || gg == TinyNN.tnn_null_ptr
            puts "donor: cannot open " + donor_path
            return zeros(0)
          end
          idx = TinyNN.tnn_gguf_find_index(gg, "token_embd.weight")
          if idx < 0
            puts "donor: " + donor_path + " has no token_embd.weight"
            return zeros(0)
          end
          d_donor = TinyNN.tnn_gguf_tensor_ne(gg, idx, 0)
          v_donor = TinyNN.tnn_gguf_tensor_ne(gg, idx, 1)
          if v_donor != vocab
            puts "donor: vocab mismatch — donor " + v_donor.to_s +
                 " vs instrument " + vocab.to_s +
                 " (this transfer does NO token remapping; use a same-vocab donor)"
            return zeros(0)
          end
          n_src = d_donor * v_donor
          src = zeros(n_src)
          TinyNN.tnn_gguf_read_f32_to_doubles(gg, idx, src, n_src)

          dsess = TinyNN.tnn_session_new(0)
          TinyNN.tnn_session_set_graph_capacity(dsess, 262144)
          t_src = TinyNN.tnn_input_2d_f32_persistent(dsess, v_donor, d_donor)  # ne=[d_donor, V]
          t_p   = TinyNN.tnn_input_2d_f32_persistent(dsess, dm, d_donor)       # ne=[d_donor, dm]
          TinyNN.tnn_finalize_weights(dsess)
          TinyNN.tnn_upload_from_float_array(dsess, t_src, src, n_src)
          # JL projection matrix, seeded off the run seed so the arm is
          # reproducible; 1/sqrt(d_donor) keeps the projected norms in
          # the donor's range before the explicit rescale below.
          pn = d_donor * dm
          pv = Toy::Train::DfaB.fill(pn, seed + 909, Toy::Train::DfaB::DIST_GAUSSIAN,
                                     1.0 / Math.sqrt(d_donor.to_f))
          TinyNN.tnn_upload_from_float_array(dsess, t_p, pv, pn)
          t_out = TinyNN.tnn_matmul(dsess, t_p, t_src)   # ne=[dm, V]
          TinyNN.tnn_set_output(t_out)
          TinyNN.tnn_build_forward_only(dsess, t_out)
          TinyNN.tnn_compute(dsess)
          n_dst = dm * vocab
          dst = zeros(n_dst)
          TinyNN.tnn_download_to_f64_array(dsess, t_out, dst, n_dst)
          TinyNN.tnn_session_free(dsess)

          # scale-match to the instrument's own init std (uniform +/-0.5)
          ss = 0.0
          di = 0
          while di < n_dst
            ss = ss + dst[di] * dst[di]
            di = di + 1
          end
          cur = Math.sqrt(ss / n_dst.to_f)
          if cur > 0.0
            k = 0.28867513459481287 / cur
            di = 0
            while di < n_dst
              dst[di] = dst[di] * k
              di = di + 1
            end
          end
          dst
        end

        # Per-tower handles (uniform-typed arrays; no Struct/Card).
        class MoeTower
          attr_accessor :pp, :pm, :pv
          attr_accessor :t_loss, :t_logits, :t_gates
          attr_accessor :tap_h2, :tap_a1, :tap_a2, :tap_ah, :tap_ctx, :t_onehots
          attr_accessor :t_rlogits
          attr_accessor :tap_as
          # toy#145: every activation tap is now PER LAYER — flat arrays
          # indexed by layer (tap_as is layer-major, l * E + i). The
          # scalar tap_a1/tap_a2/tap_ah/tap_ctx/tap_h2 accessors are kept
          # and hold the LAST layer's value, which is what the rig's
          # two-expert names and the single-layer gates already mean.
          attr_accessor :tap_zs     # toy#142: the routed experts' input, per layer
          attr_accessor :tap_blks   # toy#143: the routed contribution (block-DFA target), per layer
          attr_accessor :tap_ahs, :tap_ctxs, :tap_h2s
          attr_accessor :tap_z      # last layer's, for the single-layer readers
          attr_accessor :tap_blk
          # toy#146: per-layer optimizer hyper-parameters. EMPTY under the
          # uniform default, which is what makes that path byte-null —
          # hp_for falls straight through to the single shared vector.
          # toy#147: per-layer router state. At depth each MoE block has
          # its OWN router, so it needs its own gates/onehots/logits and
          # its own aux term — one shared set only ever described the
          # last block.
          attr_accessor :t_gates_l, :t_onehots_l, :t_rlogits_l, :t_auxgates_l
          attr_accessor :qb_biases   # toy#147: one QB bias per block's router
          attr_accessor :t_hps, :t_hps_sgd
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
            @tap_zs    = [np]; @tap_zs.pop
            @tap_blks  = [np]; @tap_blks.pop
            @tap_ahs   = [np]; @tap_ahs.pop
            @tap_ctxs  = [np]; @tap_ctxs.pop
            @tap_h2s   = [np]; @tap_h2s.pop
            @tap_z     = np
            @tap_blk   = np
            @qb_biases    = [np]; @qb_biases.pop
            @t_gates_l    = [np]; @t_gates_l.pop
            @t_onehots_l  = [np]; @t_onehots_l.pop
            @t_rlogits_l  = [np]; @t_rlogits_l.pop
            @t_auxgates_l = [np]; @t_auxgates_l.pop
            @t_hps     = [np]; @t_hps.pop
            @t_hps_sgd = [np]; @t_hps_sgd.pop
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
        # Registration ORDER is the layout contract: at L=1 this emits
        # exactly the pre-toy#145 sequence, so every index, DfaB seed
        # and checkpoint name is unchanged.
        def self.alloc_tower(sess)
          tw = MoeTower.new
          reg2(sess, tw, vocabv, dmv)  # 0 embed
          reg1(sess, tw, dmv)          # 1 fnorm
          ew = ewidth
          li = 0
          while li < nlv
            reg1(sess, tw, dmv)         # +0 attn rn1
            reg2(sess, tw, dmv, dmv)    # +1 wq
            reg2(sess, tw, dmv, dmv)    # +2 wk
            reg2(sess, tw, dmv, dmv)    # +3 wv
            reg2(sess, tw, dmv, dmv)    # +4 wo
            reg1(sess, tw, dmv)         # +5 moe rn2
            reg2(sess, tw, nev, dmv)    # +6 wr (router)
            ei = 0
            while ei < nev
              reg2(sess, tw, dfv, ew)   # +7+2i  up_i   (ℓ->dff under latent)
              reg2(sess, tw, ew, dfv)   # +8+2i  down_i (dff->ℓ under latent)
              ei = ei + 1
            end
            if gatev == 1
              reg2(sess, tw, dmv, dmv)  # gate_idx(l)  wg (attention output gate)
            end
            if latv > 0
              reg2(sess, tw, latv, dmv) # lat_base+0  W↓  d->ℓ
              reg2(sess, tw, dmv, latv) # lat_base+1  W↑  ℓ->d
              reg1(sess, tw, latv)      # lat_base+2  latent RMSNorm gamma
            end
            sj = 0
            while sj < nsv
              reg2(sess, tw, dfv, dmv)  # shared_base+2j    up   (full width)
              reg2(sess, tw, dmv, dfv)  # shared_base+2j+1  down
              sj = sj + 1
            end
            # K4b: the SiTU-GLU gate branch, one per routed expert, at
            # the layer's tail so no earlier index moves.
            gj = 0
            while gj < eglu_count
              reg2(sess, tw, dfv, ew)   # eglu_base(l)+i  gate_i (ℓ->dff)
              gj = gj + 1
            end
            li = li + 1
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
        # toy#142: the N_s always-on FULL-WIDTH shared experts (K3 eq
        # 11's Σ_j E_j^shared(x)). gelu-MLP like the routed ones; the
        # SiTU-GLU swap is K4b (a GLU needs a second input projection
        # per expert, i.e. another layout change).
        def self.shared_experts(sess, tw, h2, l)
          acc = TinyNN.tnn_null_ptr
          sj = 0
          while sj < nsv
            a = TinyNN.tnn_gelu(sess, TinyNN.tnn_matmul(sess, tw.pp[shared_base(l) + 2 * sj], h2))
            o = TinyNN.tnn_matmul(sess, tw.pp[shared_base(l) + 2 * sj + 1], a)
            if sj == 0
              acc = o
            else
              acc = TinyNN.tnn_add(sess, acc, o)
            end
            sj = sj + 1
          end
          acc
        end

        # toy#142: aggregate -> RMSNorm -> up-project. `u` is the routed
        # aggregate in latent space; returns the FULL-WIDTH contribution.
        # Without latent this is the identity (u is already full width).
        def self.latent_up(sess, tw, u, l)
          if latv == 0
            return u
          end
          un = Toy::LLM::Primitives::RMSNorm.build(sess, u, tw.pp[lat_base(l) + 2], EPS)
          TinyNN.tnn_matmul(sess, tw.pp[lat_base(l) + 1], un)
        end

        # toy#143 (block-DFA): blk_cut = 1 isolates the routed-expert
        # sub-block from the CE graph in BOTH directions —
        #   input  detached: no gradient crosses the block boundary
        #                    downward (Launay's "no backward chain
        #                    across blocks");
        #   output detached IN THE MAIN PATH: the CE loss cannot reach
        #                    the experts from above.
        # Both detaches are FORWARD-IDENTITY (vendor-patch 0011), so the
        # prediction is unchanged — only the backward path is replaced.
        # The runner then attaches the block's own loss root
        # (sum(routed ⊙ B·e)), whose gradient at `routed` is exactly the
        # random-projected output error; autodiff computes the expert
        # up/down gradients from it by ordinary BP INSIDE the block.
        # tap_blk carries the UNDETACHED routed contribution for that.
        def self.moe_block(sess, tw, t_x, sels, blk_cut, l)
          rn2 = tw.pp[layer_base(l) + LOFF_RN2]
          wr  = tw.pp[layer_base(l) + LOFF_WR]
          h2 = Toy::LLM::Primitives::RMSNorm.build(sess, t_x, rn2, EPS)
          tw.tap_h2 = h2
          tw.tap_h2s.push(h2)
          r_logits = TinyNN.tnn_matmul(sess, wr, h2)          # [NE, T]
          gates    = TinyNN.tnn_softmax(sess, r_logits)       # [NE, T]
          TinyNN.tnn_set_output(gates)
          tw.t_gates = gates
          # toy#142: the routed experts read the LATENT projection of h2
          # (K3 eq 11's W↓x); the router still scores the FULL-width h2.
          z = h2
          if latv > 0
            z = TinyNN.tnn_matmul(sess, tw.pp[lat_base(l) + 0], h2)   # [ℓ, T]
          end
          if blk_cut == 1
            z = TinyNN.tnn_detach(sess, z)
          end
          tw.tap_z = z
          tw.tap_zs.push(z)
          acc = TinyNN.tnn_null_ptr
          ei = 0
          while ei < nev
            g_i = TinyNN.tnn_matmul(sess, sels[ei], gates)    # [1, T]
            u_i = TinyNN.tnn_matmul(sess, tw.pp[up_idx(l, ei)], z)              # [DFF,T]
            a_i = TinyNN.tnn_gelu(sess, u_i)
            if eactv == 1
              # K4b/M6: the EXISTING up_i stays the UP branch (β₂=25 cap)
              # and the appended matrix is the GATE branch (β₁=4 cap +
              # sigmoid) — the SwiGLU correspondence, so up_i keeps its
              # meaning, its DfaB seed, and its checkpoint name.
              a_i = Toy::LLM::Primitives::SiTUGLU.gate(sess,
                      TinyNN.tnn_matmul(sess, tw.pp[eglu_base(l) + ei], z), u_i)
            end
            tw.tap_as.push(a_i)
            if ei == 0
              tw.tap_a1 = a_i
            end
            if ei == 1
              tw.tap_a2 = a_i
            end
            o_i = TinyNN.tnn_matmul(sess, tw.pp[down_idx(l, ei)], a_i)   # [ℓ or DM, T]
            gated_i = TinyNN.tnn_mul(sess, o_i, g_i)  # broadcast * [1,T]
            if ei == 0
              acc = gated_i
            else
              acc = TinyNN.tnn_add(sess, acc, gated_i)
            end
            ei = ei + 1
          end
          routed = latent_up(sess, tw, acc, l)
          tw.tap_blk = routed
          tw.tap_blks.push(routed)
          out = routed
          if blk_cut == 1
            # the MAIN path sees a gradient-opaque copy; the surrogate
            # root (runner-side) owns the real one.
            out = TinyNN.tnn_detach(sess, routed)
          end
          if nsv > 0
            out = TinyNN.tnn_add(sess, out, shared_experts(sess, tw, h2, l))
          end
          TinyNN.tnn_add(sess, t_x, out)
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
        def self.moe_block_top1(sess, tw, t_x, eye, cut, qb_bias, l)
          rn2 = tw.pp[layer_base(l) + LOFF_RN2]
          wr  = tw.pp[layer_base(l) + LOFF_WR]
          h2 = Toy::LLM::Primitives::RMSNorm.build(sess, t_x, rn2, EPS)
          tw.tap_h2 = h2
          tw.tap_h2s.push(h2)
          r_logits = TinyNN.tnn_matmul(sess, wr, h2)              # [NE, T]
          TinyNN.tnn_set_output(r_logits)
          tw.t_rlogits = r_logits
          tw.t_rlogits_l.push(r_logits)
          probs    = TinyNN.tnn_softmax(sess, r_logits)
          TinyNN.tnn_set_output(probs)
          tw.t_gates = probs
          tw.t_gates_l.push(probs)
          # toy#147: QB biases the SELECTION, so each block's router
          # needs its own — a shared bias would balance every block by
          # the last one's load statistics.
          qbb = qb_bias
          if tw.qb_biases.length > l
            qbb = tw.qb_biases[l]
          end
          sel_scores = r_logits
          if qbb != TinyNN.tnn_null_ptr
            sel_scores = TinyNN.tnn_add(sess, r_logits, qbb)  # broadcast [NE,T]+[NE,1]
          end
          ids   = TinyNN.tnn_argmax(sess, sel_scores)             # I32 [T]
          ids2  = TinyNN.tnn_reshape_3d(sess, ids, 1, tv, 1)       # [1, T]
          oneh  = TinyNN.tnn_get_rows(sess, eye, ids)             # [NE, T]
          TinyNN.tnn_set_output(oneh)
          tw.t_onehots = oneh
          tw.t_onehots_l.push(oneh)
          gate  = TinyNN.tnn_sum_rows(sess, TinyNN.tnn_mul(sess, oneh, probs)) # [1,T]

          # toy#147: the AUX gate stream for THIS block's router.
          #
          # Layer 0 reuses the forward `probs` outright, so at L=1 the
          # graph is bit-identical to what shipped — nothing below layer
          # 0 is a mul_mat_id, so nothing needs cutting and the aux
          # gradient keeps reaching the attention block and embedding
          # exactly as before.
          #
          # Layers >= 1 recompute the router logits from a DETACHED block
          # input. The cut sits exactly at the mul_mat_id barrier and
          # nowhere else: without it the aux root at layer l must reach
          # params below it, and that path crosses the previous block's
          # mul_mat_id, which has no ggml backward (toy#110) — that is
          # the toy#145 abort. rn2 and Wr still receive their aux credit
          # (they are above the cut); what is severed is only the aux
          # loss pushing the REPRESENTATION of earlier blocks around,
          # which at depth is not expressible at all rather than being a
          # preference.
          ag = probs
          if l > 0
            h2a = Toy::LLM::Primitives::RMSNorm.build(sess, TinyNN.tnn_detach(sess, t_x), rn2, EPS)
            ag  = TinyNN.tnn_softmax(sess, TinyNN.tnn_matmul(sess, wr, h2a))
          end
          tw.t_auxgates_l.push(TinyNN.tnn_sum_rows(sess, TinyNN.tnn_mul(sess, oneh, ag)))

          # toy#128: stack the E experts by chained concat along dim 2
          # (E=2 == the original single concat pair).
          ew = ewidth
          up_stack = TinyNN.tnn_reshape_3d(sess, tw.pp[up_idx(l, 0)],  ew, dfv, 1)
          dn_stack = TinyNN.tnn_reshape_3d(sess, tw.pp[down_idx(l, 0)], dfv, ew, 1)
          ei = 1
          while ei < nev
            up_stack = TinyNN.tnn_concat(sess, up_stack,
              TinyNN.tnn_reshape_3d(sess, tw.pp[up_idx(l, ei)],  ew, dfv, 1), 2)   # [ℓ,DFF,E]
            dn_stack = TinyNN.tnn_concat(sess, dn_stack,
              TinyNN.tnn_reshape_3d(sess, tw.pp[down_idx(l, ei)], dfv, ew, 1), 2)   # [DFF,ℓ,E]
            ei = ei + 1
          end

          # toy#142: latent projection before the routed experts. The
          # detach cut stays on the EXPERT INPUT, so under latent it
          # cuts after W↓ — W↓ itself is spine and keeps its chain grad.
          z_t = h2
          if latv > 0
            z_t = TinyNN.tnn_matmul(sess, tw.pp[lat_base(l) + 0], h2)
          end
          tw.tap_z = z_t
          tw.tap_zs.push(z_t)
          h_exp = z_t
          if cut == 1
            h_exp = TinyNN.tnn_detach(sess, z_t)
          end
          h3    = TinyNN.tnn_reshape_3d(sess, h_exp, ew, 1, tv)
          upo   = TinyNN.tnn_mul_mat_id(sess, up_stack, h3, ids2)    # [DFF,1,T]
          a     = TinyNN.tnn_gelu(sess, upo)
          if eactv == 1
            # The gate branch stacks and dispatches exactly like up —
            # one more mul_mat_id through the SAME ids2 selection, so
            # both branches are guaranteed to read the same expert.
            gt_stack = TinyNN.tnn_reshape_3d(sess, tw.pp[eglu_base(l)], ew, dfv, 1)
            gi2 = 1
            while gi2 < nev
              gt_stack = TinyNN.tnn_concat(sess, gt_stack,
                TinyNN.tnn_reshape_3d(sess, tw.pp[eglu_base(l) + gi2], ew, dfv, 1), 2)
              gi2 = gi2 + 1
            end
            gto = TinyNN.tnn_mul_mat_id(sess, gt_stack, h3, ids2)     # [DFF,1,T]
            a   = Toy::LLM::Primitives::SiTUGLU.gate(sess, gto, upo)
          end
          tw.tap_a1 = TinyNN.tnn_reshape_3d(sess, a, dfv, tv, 1)      # routed acts [DFF,T]
          dno   = TinyNN.tnn_mul_mat_id(sess, dn_stack, a, ids2)     # [ℓ,1,T]
          eo    = TinyNN.tnn_reshape_3d(sess, dno, ew, tv, 1)         # [ℓ,T]
          routed = latent_up(sess, tw, TinyNN.tnn_mul(sess, eo, gate), l)
          if nsv > 0
            routed = TinyNN.tnn_add(sess, routed, shared_experts(sess, tw, h2, l))
          end
          TinyNN.tnn_add(sess, t_x, routed)
        end

        def self.forward_tower(sess, tw, t_tok, t_labels, sels, eye, top1, cut, mask, qb_bias, blk_cut)
          x = TinyNN.tnn_get_rows(sess, tw.pp[0], t_tok)
          # toy#145: L repeats of (attention + MoE). At L=1 this is the
          # original single-block tower, op for op.
          li = 0
          while li < nlv
            lb = layer_base(li)
            wg_t = TinyNN.tnn_null_ptr
            if gatev == 1
              wg_t = tw.pp[gate_idx(li)]
            end
            atrip = attention_block(sess, x, tw.pp[lb + LOFF_RN1], tw.pp[lb + LOFF_WQ],
                                    tw.pp[lb + LOFF_WK], tw.pp[lb + LOFF_WV],
                                    tw.pp[lb + LOFF_WO], mask, wg_t)
            x = atrip[0]
            tw.tap_ah  = atrip[1]
            tw.tap_ctx = atrip[2]
            tw.tap_ahs.push(atrip[1])
            tw.tap_ctxs.push(atrip[2])
            if top1
              x = moe_block_top1(sess, tw, x, eye, cut, qb_bias, li)
            else
              x = moe_block(sess, tw, x, sels, blk_cut, li)
            end
            li = li + 1
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

        # toy#143: scalar sum over every element of a 2d tensor —
        # reshape to a column then sum_rows (the Muon Frobenius idiom).
        # Used to turn the block's error inner-product into a LOSS ROOT
        # whose gradient at `routed` is exactly the projected error.
        def self.full_sum(sess, x, n_elem)
          col = TinyNN.tnn_reshape_2d(sess, x, n_elem, 1)
          TinyNN.tnn_sum_rows(sess, col)
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
