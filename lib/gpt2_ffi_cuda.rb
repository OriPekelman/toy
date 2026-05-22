# Persistent ggml graph for GPT2LM inference. Mirrors the existing
# FullForwardFFICache in lib/tinynn.rb (which is RMSNorm + no-bias),
# but with the additions GPT-2 needs:
#
#   - LayerNorm (gamma + beta) instead of RMSNorm
#   - Bias terms after every Linear: q/k/v/o, ff_up/ff_down
#   - All biases are 1-D tensors that ggml_add broadcasts across T
#     via ggml_can_repeat(bias, activation) — no extra op needed.
#
# Single persistent session holds all weights in ctx_w; one compute
# graph at a fixed T_SEQ; per-step inputs are token_ids only. Pad to
# T_SEQ if the actual sequence is shorter.
#
# No KV cache yet — each forward recomputes attention over the full
# T_SEQ. Per-step cost is constant in prompt length but linear in
# T_SEQ. The KV-cache version reuses the same persistent weights and
# only rebuilds the compute graph (cheap — metadata only).

require_relative "transformer"
require_relative "gpt2"
require_relative "tinynn_cuda"

# Per-block tensor handles. Distinct class from BlockFFICache (in
# lib/tinynn.rb) to avoid Spinel collapsing :ptr-array ivars across
# the two cache types.
class GPT2BlockFFICuda
  attr_accessor :t_ln1_gamma, :t_ln1_beta, :t_ln2_gamma, :t_ln2_beta,
                :t_w_q, :t_w_k, :t_w_v,
                :t_b_q, :t_b_k, :t_b_v,
                :t_w_o, :t_b_o,
                :t_w_ff1, :t_b_ff1,
                :t_w_ff2, :t_b_ff2

  def initialize
    # Inline literal seed pattern (not `null = ...; [null]` — Spinel
    # loses :ptr typing through the local-var binding and re-types the
    # array as poly_array).
    @t_ln1_gamma = TinyNNCuda.tnn_null_ptr
    @t_ln1_beta  = TinyNNCuda.tnn_null_ptr
    @t_ln2_gamma = TinyNNCuda.tnn_null_ptr
    @t_ln2_beta  = TinyNNCuda.tnn_null_ptr
    @t_w_q = [TinyNNCuda.tnn_null_ptr]
    @t_w_k = [TinyNNCuda.tnn_null_ptr]
    @t_w_v = [TinyNNCuda.tnn_null_ptr]
    @t_b_q = [TinyNNCuda.tnn_null_ptr]
    @t_b_k = [TinyNNCuda.tnn_null_ptr]
    @t_b_v = [TinyNNCuda.tnn_null_ptr]
    @t_w_o   = TinyNNCuda.tnn_null_ptr
    @t_b_o   = TinyNNCuda.tnn_null_ptr
    @t_w_ff1 = TinyNNCuda.tnn_null_ptr
    @t_b_ff1 = TinyNNCuda.tnn_null_ptr
    @t_w_ff2 = TinyNNCuda.tnn_null_ptr
    @t_b_ff2 = TinyNNCuda.tnn_null_ptr
  end
end

class GPT2FullForwardFFICacheCuda
  attr_accessor :sess, :t_token_embed, :t_pos_slice, :t_token_ids,
                :t_ln_f_gamma, :t_ln_f_beta,
                :gpt2_blocks_ffi,
                :t_x_embed, :t_x_final, :t_logits,
                :t_seq, :d_model, :d_ff, :n_heads, :d_head, :n_layers,
                :vocab_size, :realized

  def initialize
    @realized   = false
    @t_seq      = 0
    @d_model    = 0
    @d_ff       = 0
    @n_heads    = 0
    @d_head     = 0
    @n_layers   = 0
    @vocab_size = 0
    @sess          = TinyNNCuda.tnn_null_ptr
    @t_token_embed = TinyNNCuda.tnn_null_ptr
    @t_pos_slice   = TinyNNCuda.tnn_null_ptr
    @t_token_ids   = TinyNNCuda.tnn_null_ptr
    @t_ln_f_gamma  = TinyNNCuda.tnn_null_ptr
    @t_ln_f_beta   = TinyNNCuda.tnn_null_ptr
    @t_x_embed     = TinyNNCuda.tnn_null_ptr
    @t_x_final     = TinyNNCuda.tnn_null_ptr
    @t_logits      = TinyNNCuda.tnn_null_ptr
    @gpt2_blocks_ffi = [GPT2BlockFFICuda.new]
  end

  # Allocate persistent ctx_w, declare all weights, build the compute
  # graph. After this, only token_ids changes per call. Call once per
  # T_SEQ choice; rebuild for a different T_SEQ.
  def realize_for(t_seq, d_model, d_ff, n_heads, n_layers, vocab_size)
    @t_seq      = t_seq
    @d_model    = d_model
    @d_ff       = d_ff
    @n_heads    = n_heads
    @d_head     = d_model / n_heads
    @n_layers   = n_layers
    @vocab_size = vocab_size

    @sess = TinyNNCuda.tnn_session_new(1)

    # === Persistent weights (ctx_w) ===
    @t_token_embed = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, vocab_size, d_model)
    @t_pos_slice   = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, t_seq,      d_model)
    @t_ln_f_gamma  = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)
    @t_ln_f_beta   = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)

    # Per-block handles — seed-then-push so Spinel types as Array<GPT2BlockFFICuda>.
    @gpt2_blocks_ffi = [GPT2BlockFFICuda.new]
    li = 1
    while li < n_layers
      @gpt2_blocks_ffi.push(GPT2BlockFFICuda.new)
      li = li + 1
    end

    li = 0
    while li < n_layers
      blk = @gpt2_blocks_ffi[li]
      blk.t_ln1_gamma = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)
      blk.t_ln1_beta  = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)
      blk.t_ln2_gamma = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)
      blk.t_ln2_beta  = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)

      # Per-head Q/K/V weights. Uploaded TRANSPOSED so ne=[d_model, d_head]
      # holds W.elem(r, c) = mat[r][c]. matmul(w_q_t, h) then yields
      # ne=[d_head, T] — same trick as FullForwardFFICache.
      #
      # Bias shapes:
      #   b_q / b_k  ne=[d_head, 1]    — broadcasts against (d_head, T)
      #                                  matmul result, the QK layout
      #   b_v        ne=[1, d_head]    — broadcasts against (T, d_head)
      #                                  matmul result, the transposed-V
      #                                  layout (needed for head_out = v @ attn)
      # Both declarations are 2D under the hood; data is still a flat
      # length-d_head Array<Float>.
      blk.t_w_q = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model)]
      blk.t_w_k = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model)]
      blk.t_w_v = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model)]
      blk.t_b_q = [TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_head)]
      blk.t_b_k = [TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_head)]
      blk.t_b_v = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, 1)]
      h = 1
      while h < n_heads
        blk.t_w_q.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model))
        blk.t_w_k.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model))
        blk.t_w_v.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model))
        blk.t_b_q.push(TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_head))
        blk.t_b_k.push(TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_head))
        blk.t_b_v.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, 1))
        h = h + 1
      end

      blk.t_w_o   = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_model, d_model)
      blk.t_b_o   = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)
      blk.t_w_ff1 = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_ff,    d_model)
      blk.t_b_ff1 = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_ff)
      blk.t_w_ff2 = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_model, d_ff)
      blk.t_b_ff2 = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)
      li = li + 1
    end

    TinyNNCuda.tnn_finalize_weights(@sess)

    # === Compute input ===
    @t_token_ids = TinyNNCuda.tnn_input_1d_i32(@sess, t_seq)

    # === Forward graph ===
    # x_embed = token_embed[ids] + pos_slice  (ne=[d_model, T])
    t_embedded = TinyNNCuda.tnn_get_rows(@sess, @t_token_embed, @t_token_ids)
    @t_x_embed = TinyNNCuda.tnn_add(@sess, t_embedded, @t_pos_slice)
    TinyNNCuda.tnn_set_output(@t_x_embed)

    eps = 1.0e-5
    scale = 1.0 / Math.sqrt(@d_head.to_f)

    t_cur = @t_x_embed
    li = 0
    while li < n_layers
      t_cur = build_block(t_cur, @gpt2_blocks_ffi[li], eps, scale)
      li = li + 1
    end

    # Final LayerNorm.
    @t_x_final = TinyNNCuda.tnn_layer_norm(@sess, t_cur, @t_ln_f_gamma, @t_ln_f_beta, eps)
    TinyNNCuda.tnn_set_output(@t_x_final)

    # Tied unembed: logits = mul_mat(token_embed, x_final)  ne=[vocab, T]
    @t_logits = TinyNNCuda.tnn_matmul(@sess, @t_token_embed, @t_x_final)
    TinyNNCuda.tnn_set_output(@t_logits)

    TinyNNCuda.tnn_realize(@sess, @t_logits)
    @realized = true
  end

  # Build one GPT-2 block's graph nodes.
  #
  #   h1 = LayerNorm(x, ln1_gamma, ln1_beta)
  #   per head h:
  #     q_h = mul_mat(w_q_t_h, h1) + b_q_h  ne=[d_head, T]
  #     k_h = mul_mat(w_k_t_h, h1) + b_k_h
  #     v_h = mul_mat(h1, w_v_t_h) + b_v_h  ne=[T, d_head] (transposed)
  #     scores_h = mul_mat(k_h, q_h)
  #     attn_h   = softmax(causal_mask(scale(scores_h)))
  #     head_out = mul_mat(v_h, attn_h)
  #   concat_h = concat along ne0 (d_head -> d_model)
  #   x_attn = x + (mul_mat(w_o_t, concat) + b_o)
  #   h2 = LayerNorm(x_attn, ln2_gamma, ln2_beta)
  #   ff_up = mul_mat(w_ff1_t, h2) + b_ff1
  #   ff_g  = gelu(ff_up)
  #   ff_dn = mul_mat(w_ff2_t, ff_g) + b_ff2
  #   x_out = x_attn + ff_dn
  def build_block(t_x, blk, eps, scale)
    t_h1 = TinyNNCuda.tnn_layer_norm(@sess, t_x, blk.t_ln1_gamma, blk.t_ln1_beta, eps)

    t_head0 = build_attention_head(t_h1, blk.t_w_q[0], blk.t_w_k[0], blk.t_w_v[0],
                                    blk.t_b_q[0], blk.t_b_k[0], blk.t_b_v[0], scale)
    t_head_outs = [t_head0]
    h = 1
    while h < @n_heads
      t_head_outs.push(build_attention_head(t_h1,
                                             blk.t_w_q[h], blk.t_w_k[h], blk.t_w_v[h],
                                             blk.t_b_q[h], blk.t_b_k[h], blk.t_b_v[h],
                                             scale))
      h = h + 1
    end

    # Concat along ne0 (d_head -> d_model).
    t_concat = t_head_outs[0]
    h = 1
    while h < @n_heads
      t_concat = TinyNNCuda.tnn_concat(@sess, t_concat, t_head_outs[h], 0)
      h = h + 1
    end

    # Output projection + bias + residual.
    t_out_proj_raw = TinyNNCuda.tnn_matmul(@sess, blk.t_w_o, t_concat)
    t_out_proj     = TinyNNCuda.tnn_add(@sess, t_out_proj_raw, blk.t_b_o)
    t_x_attn       = TinyNNCuda.tnn_add(@sess, t_x, t_out_proj)

    # FFN.
    t_h2     = TinyNNCuda.tnn_layer_norm(@sess, t_x_attn, blk.t_ln2_gamma, blk.t_ln2_beta, eps)
    t_pre_raw= TinyNNCuda.tnn_matmul(@sess, blk.t_w_ff1, t_h2)
    t_pre    = TinyNNCuda.tnn_add(@sess, t_pre_raw, blk.t_b_ff1)
    t_hidden = TinyNNCuda.tnn_gelu(@sess, t_pre)
    t_dn_raw = TinyNNCuda.tnn_matmul(@sess, blk.t_w_ff2, t_hidden)
    t_dn     = TinyNNCuda.tnn_add(@sess, t_dn_raw, blk.t_b_ff2)

    TinyNNCuda.tnn_add(@sess, t_x_attn, t_dn)
  end

  def build_attention_head(t_x, t_w_q, t_w_k, t_w_v, t_b_q, t_b_k, t_b_v, scale)
    t_q_raw = TinyNNCuda.tnn_matmul(@sess, t_w_q, t_x)        # ne=[d_head, T]
    t_q     = TinyNNCuda.tnn_add(@sess, t_q_raw, t_b_q)
    t_k_raw = TinyNNCuda.tnn_matmul(@sess, t_w_k, t_x)
    t_k     = TinyNNCuda.tnn_add(@sess, t_k_raw, t_b_k)
    # v in transposed pattern (ne=[T, d_head]) so head_out's k_dim matches.
    t_v_raw = TinyNNCuda.tnn_matmul(@sess, t_x, t_w_v)
    t_v     = TinyNNCuda.tnn_add(@sess, t_v_raw, t_b_v)

    t_scores = TinyNNCuda.tnn_matmul(@sess, t_k, t_q)
    t_scaled = TinyNNCuda.tnn_scale(@sess, t_scores, scale)
    t_masked = TinyNNCuda.tnn_diag_mask_inf(@sess, t_scaled, 0)
    t_attn   = TinyNNCuda.tnn_softmax(@sess, t_masked)

    TinyNNCuda.tnn_matmul(@sess, t_v, t_attn)                  # ne=[d_head, T_query]
  end
end

module GPT2FFICuda
  # Upload all weights from a populated GPT2LM into a freshly-realized
  # GPT2FullForwardFFICacheCuda. Transposed-upload for the per-head Q/K/V
  # and for w_o/w_ff1/w_ff2; row-major bulk for token_embed/pos_slice;
  # direct 1-D upload for biases and LayerNorm params.
  def self.upload_from(fwd_cache, model, pos_slice_mat)
    sess = fwd_cache.sess
    n    = fwd_cache.n_layers
    n_heads = fwd_cache.n_heads
    d_model = fwd_cache.d_model

    TinyNNCuda.upload_row_major(sess, fwd_cache.t_token_embed, model.token_embed)
    TinyNNCuda.upload_row_major(sess, fwd_cache.t_pos_slice,   pos_slice_mat)
    TinyNNCuda.tnn_upload_from_float_array(sess, fwd_cache.t_ln_f_gamma, model.ln_f_gamma, d_model)
    TinyNNCuda.tnn_upload_from_float_array(sess, fwd_cache.t_ln_f_beta,  model.ln_f_beta,  d_model)

    li = 0
    while li < n
      blk_n = model.gpt2_blocks[li]
      blk_f = fwd_cache.gpt2_blocks_ffi[li]

      TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_ln1_gamma, blk_n.ln1_gamma, d_model)
      TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_ln1_beta,  blk_n.ln1_beta,  d_model)
      TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_ln2_gamma, blk_n.ln2_gamma, d_model)
      TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_ln2_beta,  blk_n.ln2_beta,  d_model)

      d_head = fwd_cache.d_head
      h = 0
      while h < n_heads
        TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_q[h], blk_n.w_q[h])
        TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_k[h], blk_n.w_k[h])
        TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_v[h], blk_n.w_v[h])
        TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_b_q[h], blk_n.b_q[h], d_head)
        TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_b_k[h], blk_n.b_k[h], d_head)
        TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_b_v[h], blk_n.b_v[h], d_head)
        h = h + 1
      end

      TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_o,   blk_n.w_o)
      TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_ff1, blk_n.w_ff1)
      TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_ff2, blk_n.w_ff2)
      TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_b_o,   blk_n.b_o,   d_model)
      TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_b_ff1, blk_n.b_ff1, fwd_cache.d_ff)
      TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_b_ff2, blk_n.b_ff2, d_model)

      li = li + 1
    end
  end

  # Build the (t_seq, d_model) pos_slice that pairs with token_ids
  # padded to t_seq. Slice rows 0..t_seq-1 of model.pos_embed.
  def self.make_pos_slice(model, t_seq)
    out = Mat.new(t_seq, model.d_model)
    n = t_seq * model.d_model
    i = 0
    while i < n
      out.flat[i] = model.pos_embed.flat[i]
      i = i + 1
    end
    out
  end

  # Pad an Array<Int> of token IDs to length t_seq with zeros (the
  # "<unk>" / EOS-style fallback). Returns a new Array.
  def self.pad_ids(ids, t_seq)
    out = Array.new(t_seq, 0)
    n   = ids.length
    if n > t_seq
      n = t_seq
    end
    i = 0
    while i < n
      out[i] = ids[i]
      i = i + 1
    end
    out
  end

  # Run forward. token_ids is a length-t_seq padded Array<Int>.
  # Returns the (t_seq, vocab) logits Mat. ggml's mul_mat result has
  # ne=[vocab, t_seq] which, interpreted row-major with rows=t_seq /
  # cols=vocab, is the layout Mat#flat[t*vocab + v] expects.
  def self.forward(fwd_cache, token_ids)
    TinyNNCuda.upload_int_array(fwd_cache.sess, fwd_cache.t_token_ids, token_ids)
    rc = TinyNNCuda.tnn_compute(fwd_cache.sess)
    if rc != 0
      puts "tnn_compute failed: rc=" + rc.to_s
    end
    TinyNNCuda.download_row_major(fwd_cache.sess, fwd_cache.t_logits, fwd_cache.t_seq, fwd_cache.vocab_size)
  end

end
