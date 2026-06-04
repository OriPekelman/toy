# lib/toy/llm/engine/gpt2_seq_engine.rb — the GPT-2 training engine.
#
# Formalizes the byte-exact-gated inline GPT-2 trainer (prep/gpt2_train_min.rb)
# into the engine structure: a `realize!` that builds the whole
# forward + CE + backward + AdamW graph, and a `step!` that drives one
# training step. The GPT-2-distinctive structure: wte+wpe learned positional
# embeddings, composite LayerNorm (ggml_norm + mul γ + add β), multi-head causal
# self-attention (per-head weights + concat, qkv biases), GELU FFN, tied output.
# Backward of the LayerNorm + GELU rides the two vendored kernels
# (ggml_norm_back / ggml_gelu_back; vendor-patches/0007).
#
# DELIBERATELY SEPARATE from LlamaSeqEngine (NOT a subclass / not a shared
# realize path). Reasons: (1) it protects the Llama byte-exact gates from any
# GPT-2 churn; (2) compiling two different realize paths into one Spinel unit
# merges the `cfg`/engine receiver types (landmine #16, the same reason
# train_lora is its own binary), so the GPT-2 runner is its own binary too. The
# per-step graph drive (graph_reset → uploads → compute_backward → download)
# mirrors LlamaSeqEngine but is inlined here for monomorphic compilation.
#
# Spinel hygiene: NO Struct; NO default-arg ctor (no-arg ctor + explicit
# realize! args); explicit while loops; weights in flat Array<:ptr> ivars
# (the @ft_globals_* precedent); the per-(layer,head) q/k/v weights are
# flat-indexed [li*n_heads + h]; random init via a seeded LCG → Mat →
# upload (no C-side random-fill at the Ruby level); the per-realize Array<Mat>
# of inits stays LOCAL to realize! (never crosses a function boundary —
# Spinel has no sp_Mat_ptr_array).
#
# This file does NOT require_relative "tinynn": the loading runner
# (lib/toy/run/train_gpt2.rb) loads the CPU TinyNN first, like the L1-L4 tree.

module Toy; module LLM; module Engine
  class GPT2SeqEngine
    attr_accessor :sess,
                  :g_vocab, :g_d_model, :g_n_heads, :g_d_head, :g_d_ff,
                  :g_n_layers, :g_context,
                  :g_t_tok, :g_t_pos, :g_t_labels, :g_t_hp, :g_t_loss, :g_t_logits,
                  # arch handles (flat Array<:ptr>)
                  :g_wte, :g_wpe, :g_lnf_g, :g_lnf_b,
                  :g_ln1_g, :g_ln1_b, :g_ln2_g, :g_ln2_b,
                  :g_w_q, :g_b_q, :g_w_k, :g_b_k, :g_w_v, :g_b_v, :g_w_o, :g_b_o,
                  :g_fc_w, :g_fc_b, :g_pr_w, :g_pr_b,
                  # optimizer triples (weight, m, v) parallel to g_weights
                  :g_weights, :g_opt_m, :g_opt_v,
                  :g_rng, :g_rb_rc, :g_cb_rc

    LN_EPS = 1.0e-5

    def initialize
      @sess = TinyNN.tnn_null_ptr
      @g_vocab = 0; @g_d_model = 0; @g_n_heads = 0; @g_d_head = 0
      @g_d_ff = 0; @g_n_layers = 0; @g_context = 0
      @g_t_tok = TinyNN.tnn_null_ptr; @g_t_pos = TinyNN.tnn_null_ptr
      @g_t_labels = TinyNN.tnn_null_ptr; @g_t_hp = TinyNN.tnn_null_ptr
      @g_t_loss = TinyNN.tnn_null_ptr; @g_t_logits = TinyNN.tnn_null_ptr
      @g_wte = TinyNN.tnn_null_ptr; @g_wpe = TinyNN.tnn_null_ptr
      @g_lnf_g = TinyNN.tnn_null_ptr; @g_lnf_b = TinyNN.tnn_null_ptr
      @g_ln1_g = [TinyNN.tnn_null_ptr]; @g_ln1_g.pop
      @g_ln1_b = [TinyNN.tnn_null_ptr]; @g_ln1_b.pop
      @g_ln2_g = [TinyNN.tnn_null_ptr]; @g_ln2_g.pop
      @g_ln2_b = [TinyNN.tnn_null_ptr]; @g_ln2_b.pop
      @g_w_q = [TinyNN.tnn_null_ptr]; @g_w_q.pop
      @g_b_q = [TinyNN.tnn_null_ptr]; @g_b_q.pop
      @g_w_k = [TinyNN.tnn_null_ptr]; @g_w_k.pop
      @g_b_k = [TinyNN.tnn_null_ptr]; @g_b_k.pop
      @g_w_v = [TinyNN.tnn_null_ptr]; @g_w_v.pop
      @g_b_v = [TinyNN.tnn_null_ptr]; @g_b_v.pop
      @g_w_o = [TinyNN.tnn_null_ptr]; @g_w_o.pop
      @g_b_o = [TinyNN.tnn_null_ptr]; @g_b_o.pop
      @g_fc_w = [TinyNN.tnn_null_ptr]; @g_fc_w.pop
      @g_fc_b = [TinyNN.tnn_null_ptr]; @g_fc_b.pop
      @g_pr_w = [TinyNN.tnn_null_ptr]; @g_pr_w.pop
      @g_pr_b = [TinyNN.tnn_null_ptr]; @g_pr_b.pop
      @g_weights = [TinyNN.tnn_null_ptr]; @g_weights.pop
      @g_opt_m   = [TinyNN.tnn_null_ptr]; @g_opt_m.pop
      @g_opt_v   = [TinyNN.tnn_null_ptr]; @g_opt_v.pop
      @g_rng = 0
    end

    # seeded LCG → ~[-1,1)
    def rand_unit
      @g_rng = ((@g_rng * 1103515245) + 12345) & 0x7fffffff
      ((@g_rng >> 8).to_f / 8388608.0) - 1.0
    end

    def random_mat(rows, cols, scale)
      m = Mat.new(rows, cols)
      n = rows * cols
      i = 0
      while i < n
        m.flat[i] = rand_unit * scale
        i = i + 1
      end
      m
    end

    def const_mat(rows, cols, value)
      m = Mat.new(rows, cols)
      n = rows * cols
      i = 0
      while i < n
        m.flat[i] = value
        i = i + 1
      end
      m
    end

    # alloc-only (buffers don't exist until tnn_finalize_weights); records
    # (weight, m, v) into the optimizer arrays and the init Mat into `inits`.
    def alloc_w2(inits, rows, cols, init_mat)
      w = TinyNN.tnn_input_2d_f32_persistent(@sess, rows, cols)
      @g_weights.push(w)
      @g_opt_m.push(TinyNN.tnn_input_2d_f32_persistent(@sess, rows, cols))
      @g_opt_v.push(TinyNN.tnn_input_2d_f32_persistent(@sess, rows, cols))
      inits.push(init_mat)
      w
    end

    def alloc_w1(inits, n, init_mat)
      w = TinyNN.tnn_input_1d_f32_persistent(@sess, n)
      @g_weights.push(w)
      @g_opt_m.push(TinyNN.tnn_input_1d_f32_persistent(@sess, n))
      @g_opt_v.push(TinyNN.tnn_input_1d_f32_persistent(@sess, n))
      inits.push(init_mat)
      w
    end

    # Build the full random-init training graph. Realize ordering is
    # load-bearing (alloc → set_param → finalize_weights → upload → backward →
    # realize_backward); uploading a persistent weight before finalize aborts
    # ("tensor buffer not set").
    def realize!(vocab, d_model, n_heads, d_ff, n_layers, context, seed)
      @g_vocab = vocab; @g_d_model = d_model; @g_n_heads = n_heads
      @g_d_head = d_model / n_heads; @g_d_ff = d_ff
      @g_n_layers = n_layers; @g_context = context
      @g_rng = seed
      @sess = TinyNN.tnn_session_new(0)
      # Per-head decomposition makes node count scale O(n_layers × n_heads);
      # budget like LlamaSeqEngine (the default cap overflows at backward-expand
      # on bigger shapes). Must precede realize (no compute tensors stored yet).
      TinyNN.tnn_session_set_graph_capacity(@sess, n_layers * n_heads * 1000 + 65536)

      inits = [Mat.new(1, 1)]; inits.pop

      @g_wte = alloc_w2(inits, vocab,   d_model, random_mat(vocab,   d_model, 0.02))
      @g_wpe = alloc_w2(inits, context, d_model, random_mat(context, d_model, 0.02))

      li = 0
      while li < n_layers
        @g_ln1_g.push(alloc_w1(inits, d_model, const_mat(1, d_model, 1.0)))
        @g_ln1_b.push(alloc_w1(inits, d_model, const_mat(1, d_model, 0.0)))
        hh = 0
        while hh < n_heads
          @g_w_q.push(alloc_w2(inits, @g_d_head, d_model, random_mat(@g_d_head, d_model, 0.02)))
          @g_b_q.push(alloc_w1(inits, @g_d_head, const_mat(1, @g_d_head, 0.0)))
          @g_w_k.push(alloc_w2(inits, @g_d_head, d_model, random_mat(@g_d_head, d_model, 0.02)))
          @g_b_k.push(alloc_w1(inits, @g_d_head, const_mat(1, @g_d_head, 0.0)))
          @g_w_v.push(alloc_w2(inits, @g_d_head, d_model, random_mat(@g_d_head, d_model, 0.02)))
          @g_b_v.push(alloc_w1(inits, @g_d_head, const_mat(1, @g_d_head, 0.0)))
          hh = hh + 1
        end
        @g_w_o.push(alloc_w2(inits, d_model, d_model, random_mat(d_model, d_model, 0.02)))
        @g_b_o.push(alloc_w1(inits, d_model, const_mat(1, d_model, 0.0)))
        @g_ln2_g.push(alloc_w1(inits, d_model, const_mat(1, d_model, 1.0)))
        @g_ln2_b.push(alloc_w1(inits, d_model, const_mat(1, d_model, 0.0)))
        @g_fc_w.push(alloc_w2(inits, d_ff, d_model, random_mat(d_ff, d_model, 0.02)))
        @g_fc_b.push(alloc_w1(inits, d_ff, const_mat(1, d_ff, 0.0)))
        @g_pr_w.push(alloc_w2(inits, d_model, d_ff, random_mat(d_model, d_ff, 0.02)))
        @g_pr_b.push(alloc_w1(inits, d_model, const_mat(1, d_model, 0.0)))
        li = li + 1
      end
      @g_lnf_g = alloc_w1(inits, d_model, const_mat(1, d_model, 1.0))
      @g_lnf_b = alloc_w1(inits, d_model, const_mat(1, d_model, 0.0))

      gi = 0
      while gi < @g_weights.length
        TinyNN.tnn_set_param(@g_weights[gi])
        gi = gi + 1
      end
      TinyNN.tnn_finalize_weights(@sess)

      gk = 0
      while gk < @g_weights.length
        TinyNN.upload_row_major(@sess, @g_weights[gk], inits[gk])
        TinyNN.tnn_zero_tensor(@sess, @g_opt_m[gk])
        TinyNN.tnn_zero_tensor(@sess, @g_opt_v[gk])
        gk = gk + 1
      end

      build_forward!
      build_train_step!
      nil
    end

    # GPT-2 forward → @g_t_logits (tied unembed).
    def build_forward!
      @g_t_tok = TinyNN.tnn_input_1d_i32(@sess, @g_context)
      @g_t_pos = TinyNN.tnn_input_1d_i32(@sess, @g_context)

      x = TinyNN.tnn_add(@sess,
            TinyNN.tnn_get_rows(@sess, @g_wte, @g_t_tok),
            TinyNN.tnn_get_rows(@sess, @g_wpe, @g_t_pos))
      TinyNN.tnn_set_output(x)

      att_scale = 1.0 / Math.sqrt(@g_d_head.to_f)
      li = 0
      while li < @g_n_layers
        # attention sub-block (per-head loop + concat)
        h1 = TinyNN.tnn_layer_norm(@sess, x, @g_ln1_g[li], @g_ln1_b[li], LN_EPS)
        head_out = TinyNN.tnn_null_ptr
        hh = 0
        while hh < @g_n_heads
          hi = li * @g_n_heads + hh
          q = TinyNN.tnn_add(@sess, TinyNN.tnn_matmul(@sess, @g_w_q[hi], h1), @g_b_q[hi])
          k = TinyNN.tnn_add(@sess, TinyNN.tnn_matmul(@sess, @g_w_k[hi], h1), @g_b_k[hi])
          v = TinyNN.tnn_matmul(@sess, @g_w_v[hi], h1)   # bias added to output
          scores = TinyNN.tnn_scale(@sess, TinyNN.tnn_matmul(@sess, k, q), att_scale)
          scores = TinyNN.tnn_diag_mask_inf(@sess, scores, 0)
          probs  = TinyNN.tnn_softmax(@sess, scores)
          v_t    = TinyNN.tnn_cont_2d(@sess, TinyNN.tnn_transpose(@sess, v), @g_context, @g_d_head)
          head   = TinyNN.tnn_add(@sess, TinyNN.tnn_matmul(@sess, v_t, probs), @g_b_v[hi])
          if hh == 0
            head_out = head
          else
            head_out = TinyNN.tnn_concat(@sess, head_out, head, 0)
          end
          hh = hh + 1
        end
        ao = TinyNN.tnn_add(@sess, TinyNN.tnn_matmul(@sess, @g_w_o[li], head_out), @g_b_o[li])
        x  = TinyNN.tnn_add(@sess, x, ao)
        TinyNN.tnn_set_output(x)

        # FFN sub-block
        h2  = TinyNN.tnn_layer_norm(@sess, x, @g_ln2_g[li], @g_ln2_b[li], LN_EPS)
        pre = TinyNN.tnn_add(@sess, TinyNN.tnn_matmul(@sess, @g_fc_w[li], h2), @g_fc_b[li])
        act = TinyNN.tnn_gelu(@sess, pre)
        mlp = TinyNN.tnn_add(@sess, TinyNN.tnn_matmul(@sess, @g_pr_w[li], act), @g_pr_b[li])
        x   = TinyNN.tnn_add(@sess, x, mlp)
        TinyNN.tnn_set_output(x)
        li = li + 1
      end

      x_final = TinyNN.tnn_layer_norm(@sess, x, @g_lnf_g, @g_lnf_b, LN_EPS)
      TinyNN.tnn_set_output(x_final)
      @g_t_logits = TinyNN.tnn_matmul(@sess, @g_wte, x_final)   # tied
      TinyNN.tnn_set_output(@g_t_logits)
      nil
    end

    # CE loss + backward + opt_step_adamw per weight.
    def build_train_step!
      @g_t_labels = TinyNN.tnn_input_2d_f32(@sess, @g_context, @g_vocab)
      @g_t_hp     = TinyNN.tnn_input_1d_f32(@sess, 7)
      @g_t_loss   = TinyNN.tnn_cross_entropy_loss(@sess, @g_t_logits, @g_t_labels)
      TinyNN.tnn_set_output(@g_t_loss)
      TinyNN.tnn_set_loss(@g_t_loss)

      TinyNN.tnn_build_forward_only(@sess, @g_t_loss)
      TinyNN.tnn_build_backward(@sess)

      gj = 0
      while gj < @g_weights.length
        tw = @g_weights[gj]
        tg = TinyNN.tnn_tensor_grad(@sess, tw)
        to = TinyNN.tnn_opt_step_adamw(@sess, tw, tg, @g_opt_m[gj], @g_opt_v[gj], @g_t_hp)
        TinyNN.tnn_extend_backward_graph(@sess, to)
        gj = gj + 1
      end
      @g_rb_rc = TinyNN.tnn_realize_backward(@sess)
      nil
    end

    # One training step. is_first selects full reset vs grads-only (momenta
    # persist). Returns the loss Float.
    def step!(seq_ids, positions, m_labels, m_hp, is_first)
      if is_first
        TinyNN.tnn_graph_reset(@sess)
      else
        TinyNN.tnn_graph_reset_grads_only(@sess)
      end
      TinyNN.upload_int_array(@sess, @g_t_tok, seq_ids)
      TinyNN.upload_int_array(@sess, @g_t_pos, positions)
      TinyNN.upload_row_major(@sess, @g_t_labels, m_labels)
      TinyNN.upload_row_major(@sess, @g_t_hp, m_hp)
      @g_cb_rc = TinyNN.tnn_compute_backward(@sess)
      TinyNN.tnn_download(@sess, @g_t_loss)
      TinyNN.tnn_scratch_get(@sess, 0)
    end
  end
end; end; end
