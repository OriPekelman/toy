# lib/toy/run/train_hybrid.rb — Phase 5 capstone: a SELF-CONTAINED from-scratch
# HYBRID trainer (one attention layer + one Gated-DeltaNet layer), in its OWN
# Spinel compilation unit.
#
# Why its own runner (not the llama engine): pulling GDNBlock alloc/train code
# into Toy::LLM::Engine::LlamaSeqEngine's unit miscompiles the proven byte-exact
# attention path on the union pin (landmine #16 family — the same reason
# toy-train-lora / toy-train-gpt2 are separate binaries). A dedicated unit can't
# corrupt that path. See docs/roadmap/dragon-gdn-arch-2026-06-20.md (Phase 5).
#
# The forward dispatches per layer on a flat INT kind (the LayerSpec seam
# pattern, monomorphic per call site): KIND_ATTENTION → inline causal
# self-attention; KIND_GDN → Toy::LLM::Blocks::GDNBlock (Path-B autograd
# recurrence). All params are flattened into one uniform ptr array so the
# AdamW opt_step loop never touches two block types in one method.
#
#   x   = get_rows(embed, ids)                 [d_model, T]
#   x   = attention_layer(x)                   [d_model, T]  (KIND_ATTENTION)
#   x   = GDNBlock.build_forward(x)            [d_model, T]  (KIND_GDN)
#   xf  = rmsnorm(x, final_gamma)              [d_model, T]
#   lgt = matmul(embed, xf)                    [vocab, T]    (tied)
#   loss= cross_entropy(lgt, labels)           overfit one fixed batch
#
# Asserts the CE loss DECREASES — the heterogeneous trainable stack works.

require_relative "../../toy"
require_relative "../ffi/tinynn"
require_relative "../llm/primitives/rms_norm"
require_relative "../llm/primitives/gdn"
require_relative "../llm/blocks/gdn_block"
require_relative "../llm/archs/layer_spec"

module Toy
  module LLM
    module Run
      module TrainHybrid
        VOCAB = 16
        DM    = 8
        H     = 2
        S_V   = 4     # H*S_V == DM
        T     = 4
        STEPS = 16
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

        # Inline single-head causal self-attention (no RoPE/GQA — minimal,
        # trainable). Weights arrive as explicit handles. Returns x + Wo·ctx.
        def self.attention_layer(sess, t_x, rn, wq, wk, wv, wo, eps)
          h = Toy::LLM::Primitives::RMSNorm.build(sess, t_x, rn, eps)
          q = TinyNN.tnn_matmul(sess, wq, h)            # [DM, T]
          k = TinyNN.tnn_matmul(sess, wk, h)            # [DM, T]
          v = TinyNN.tnn_matmul(sess, wv, h)            # [DM, T]
          scores = TinyNN.tnn_matmul(sess, k, q)        # [T_k, T_q]
          scaled = TinyNN.tnn_scale(sess, scores, 1.0 / Math.sqrt(DM.to_f))
          masked = TinyNN.tnn_diag_mask_inf(sess, scaled, 0)
          attn   = TinyNN.tnn_softmax(sess, masked)     # [T_k, T_q]
          v_t    = TinyNN.tnn_transpose(sess, v)        # [T, DM]
          ctx    = TinyNN.tnn_matmul(sess, v_t, attn)   # [DM, T_q]
          out    = TinyNN.tnn_matmul(sess, wo, ctx)     # [DM, T]
          TinyNN.tnn_add(sess, t_x, out)
        end

        def self.run
          sess = TinyNN.tnn_session_new(0)
          TinyNN.tnn_session_set_graph_capacity(sess, 262144)

          # Flat param arrays (uniform ptr) so opt_step never sees two block types.
          pp = [TinyNN.tnn_null_ptr]; pp.pop
          pm = [TinyNN.tnn_null_ptr]; pm.pop
          pv = [TinyNN.tnn_null_ptr]; pv.pop

          # reg2/reg1: alloc a weight + matching m/v, register, return the weight.
          embed  = reg2(sess, pp, pm, pv, VOCAB, DM)   # ne0=DM, ne1=VOCAB
          fnorm  = reg1(sess, pp, pm, pv, DM)
          # Attention layer weights.
          a_rn   = reg1(sess, pp, pm, pv, DM)
          a_wq   = reg2(sess, pp, pm, pv, DM, DM)
          a_wk   = reg2(sess, pp, pm, pv, DM, DM)
          a_wv   = reg2(sess, pp, pm, pv, DM, DM)
          a_wo   = reg2(sess, pp, pm, pv, DM, DM)

          # GDN layer (its own weights live in ft_weights/ft_m/ft_v; flatten in).
          gblk = Toy::LLM::Blocks::GDNBlock.new
          gblk.alloc_trainable_f32_weights!(sess, DM, S_V, H)
          bi = 0
          while bi < gblk.ft_weights.length
            pp.push(gblk.ft_weights[bi]); pm.push(gblk.ft_m[bi]); pv.push(gblk.ft_v[bi])
            bi = bi + 1
          end

          # set_param BEFORE finalize (load-bearing order).
          gi = 0
          while gi < pp.length
            TinyNN.tnn_set_param(pp[gi])
            gi = gi + 1
          end
          TinyNN.tnn_finalize_weights(sess)
          gblk.zero_state!(sess)

          # Init weights + zero moments.
          gi = 0
          while gi < pp.length
            n = TinyNN.tnn_tensor_nelements(pp[gi])
            TinyNN.tnn_upload_from_float_array(sess, pp[gi], fillv(n, gi * 7 + 1), n)
            TinyNN.tnn_zero_tensor(sess, pm[gi])
            TinyNN.tnn_zero_tensor(sess, pv[gi])
            gi = gi + 1
          end

          # Forward — per-layer INT-kind dispatch (the seam pattern).
          t_tok = TinyNN.tnn_input_1d_i32(sess, T)
          x = TinyNN.tnn_get_rows(sess, embed, t_tok)
          kinds = [Toy::LLM::Archs::LayerSpec::KIND_ATTENTION,
                   Toy::LLM::Archs::LayerSpec::KIND_GDN]
          li = 0
          while li < kinds.length
            if kinds[li] == Toy::LLM::Archs::LayerSpec::KIND_ATTENTION
              x = attention_layer(sess, x, a_rn, a_wq, a_wk, a_wv, a_wo, EPS)
            else
              x = gblk.build_forward(sess, x, T, EPS)
            end
            li = li + 1
          end
          xf  = Toy::LLM::Primitives::RMSNorm.build(sess, x, fnorm, EPS)
          lgt = TinyNN.tnn_matmul(sess, embed, xf)         # [VOCAB, T] tied

          t_labels = TinyNN.tnn_input_2d_f32(sess, T, VOCAB)
          t_hp     = TinyNN.tnn_input_1d_f32(sess, 7)
          t_loss   = TinyNN.tnn_cross_entropy_loss(sess, lgt, t_labels)
          TinyNN.tnn_set_output(t_loss)
          TinyNN.tnn_set_loss(t_loss)

          TinyNN.tnn_build_forward_only(sess, t_loss)
          TinyNN.tnn_build_backward(sess)
          gj = 0
          while gj < pp.length
            tg = TinyNN.tnn_tensor_grad(sess, pp[gj])
            to = TinyNN.tnn_opt_step_adamw(sess, pp[gj], tg, pm[gj], pv[gj], t_hp)
            TinyNN.tnn_extend_backward_graph(sess, to)
            gj = gj + 1
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
          hp = [0.02, 0.9, 0.95, 1.0e-8, 0.0, 0.9, 0.95]

          first_loss = 0.0
          last_loss  = 0.0
          s = 0
          while s < STEPS
            if s == 0
              TinyNN.tnn_graph_reset(sess)
            else
              TinyNN.tnn_graph_reset_grads_only(sess)
            end
            TinyNN.upload_int_array(sess, t_tok, ids)
            TinyNN.tnn_upload_from_float_array(sess, t_labels, labels, VOCAB * T)
            TinyNN.tnn_upload_from_float_array(sess, t_hp, hp, 7)
            TinyNN.tnn_compute_backward(sess)
            TinyNN.tnn_download(sess, t_loss)
            lv = TinyNN.tnn_scratch_get(sess, 0)
            if s == 0
              first_loss = lv
            end
            last_loss = lv
            puts "step " + s.to_s + ": loss=" + lv.to_s
            s = s + 1
          end

          ok = true
          if first_loss != first_loss || last_loss != last_loss
            puts "FAIL: loss is NaN"
            ok = false
          end
          if last_loss >= first_loss - 0.05
            puts "FAIL: loss did not decrease (first=" + first_loss.to_s + " last=" + last_loss.to_s + ")"
            ok = false
          end
          if ok
            puts "HYBRID train smoke PASS: attention+GDN from-scratch hybrid trains — CE loss " +
                 first_loss.to_s + " -> " + last_loss.to_s + " over " + STEPS.to_s + " steps"
          else
            puts "HYBRID train smoke FAIL"
          end
        end

        def self.reg1(sess, pp, pm, pv, n)
          w = TinyNN.tnn_input_1d_f32_persistent(sess, n)
          pp.push(w); pm.push(TinyNN.tnn_input_1d_f32_persistent(sess, n))
          pv.push(TinyNN.tnn_input_1d_f32_persistent(sess, n))
          w
        end

        def self.reg2(sess, pp, pm, pv, rows, cols)
          w = TinyNN.tnn_input_2d_f32_persistent(sess, rows, cols)
          pp.push(w); pm.push(TinyNN.tnn_input_2d_f32_persistent(sess, rows, cols))
          pv.push(TinyNN.tnn_input_2d_f32_persistent(sess, rows, cols))
          w
        end
      end
    end
  end
end

Toy::LLM::Run::TrainHybrid.run
