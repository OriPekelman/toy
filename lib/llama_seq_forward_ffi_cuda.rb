# lib/llama_seq_forward_ffi_cuda.rb — CUDA mirror of lib/llama_seq_forward_ffi.rb.
#
# Same contract; differs only by routing through TinyNNCuda (which selects
# the CUDA backend at session creation) and using distinct class names so
# Spinel keeps the two cache classes separate. See the CPU twin for the
# block-by-block math.

require_relative "transformer"
require_relative "toy"
require_relative "toy_smollm2"
require_relative "tinynn_cuda"

class LlamaSeqBlockFFICuda
  attr_accessor :t_seq_rn1_gamma, :t_seq_rn2_gamma,
                :t_seq_w_q,  :t_seq_w_k,  :t_seq_w_v,  :t_seq_w_o,
                :t_seq_b_q,  :t_seq_b_k,  :t_seq_b_v,
                :t_seq_w_gate, :t_seq_w_up, :t_seq_w_down

  def initialize
    @t_seq_rn1_gamma = TinyNNCuda.tnn_null_ptr
    @t_seq_rn2_gamma = TinyNNCuda.tnn_null_ptr
    @t_seq_w_q = [TinyNNCuda.tnn_null_ptr]
    @t_seq_w_k = [TinyNNCuda.tnn_null_ptr]
    @t_seq_w_v = [TinyNNCuda.tnn_null_ptr]
    @t_seq_b_q = [TinyNNCuda.tnn_null_ptr]
    @t_seq_b_k = [TinyNNCuda.tnn_null_ptr]
    @t_seq_b_v = [TinyNNCuda.tnn_null_ptr]
    @t_seq_w_o    = TinyNNCuda.tnn_null_ptr
    @t_seq_w_gate = TinyNNCuda.tnn_null_ptr
    @t_seq_w_up   = TinyNNCuda.tnn_null_ptr
    @t_seq_w_down = TinyNNCuda.tnn_null_ptr
  end
end

class LlamaSeqForwardFFICacheCuda
  attr_accessor :sess,
                :t_seq_token_embed, :t_seq_final_norm_gamma, :t_seq_output,
                :seq_has_untied_output, :seq_has_qkv_bias,
                :seq_blocks_ffi,
                :seq_t, :seq_d_model, :seq_d_ff, :seq_n_heads, :seq_n_kv,
                :seq_d_head, :seq_group_size, :seq_n_layers, :seq_vocab_size,
                :seq_rope_base, :seq_rms_eps, :seq_realized,
                :t_seq_token_ids, :t_seq_positions,
                :t_seq_x_embed, :t_seq_x_final, :t_seq_logits,
                :seq_gguf_handle_keepalive

  def initialize
    @seq_realized   = false
    @seq_t          = 0
    @seq_d_model    = 0
    @seq_d_ff       = 0
    @seq_n_heads    = 0
    @seq_n_kv       = 0
    @seq_d_head     = 0
    @seq_group_size = 0
    @seq_n_layers   = 0
    @seq_vocab_size = 0
    @seq_rope_base  = 10000.0
    @seq_rms_eps    = 1.0e-5
    @sess                  = TinyNNCuda.tnn_null_ptr
    @t_seq_token_embed     = TinyNNCuda.tnn_null_ptr
    @t_seq_final_norm_gamma = TinyNNCuda.tnn_null_ptr
    @t_seq_output          = TinyNNCuda.tnn_null_ptr
    @seq_has_untied_output = false
    @seq_has_qkv_bias      = false
    @seq_blocks_ffi        = [LlamaSeqBlockFFICuda.new]
    @seq_gguf_handle_keepalive = TinyNNCuda.tnn_null_ptr
    @t_seq_token_ids = TinyNNCuda.tnn_null_ptr
    @t_seq_positions = TinyNNCuda.tnn_null_ptr
    @t_seq_x_embed   = TinyNNCuda.tnn_null_ptr
    @t_seq_x_final   = TinyNNCuda.tnn_null_ptr
    @t_seq_logits    = TinyNNCuda.tnn_null_ptr
  end

  def realize_for_mmap(gguf_handle, cfg, t_seq, untied, qkv_bias)
    @seq_t          = t_seq
    @seq_d_model    = cfg.d_model
    @seq_d_ff       = cfg.d_ff
    @seq_n_heads    = cfg.n_heads
    @seq_n_kv       = cfg.n_kv
    @seq_d_head     = cfg.d_model / cfg.n_heads
    @seq_group_size = cfg.n_heads / cfg.n_kv
    @seq_n_layers   = cfg.n_layers
    @seq_vocab_size = cfg.vocab
    @seq_rope_base  = cfg.rope_base
    @seq_rms_eps    = cfg.rms_eps

    @seq_gguf_handle_keepalive = gguf_handle
    @sess                  = TinyNNCuda.tnn_session_new(1)   # 1 = CUDA backend
    @seq_has_untied_output = untied
    @seq_has_qkv_bias      = qkv_bias

    map_base = TinyNNCuda.tnn_gguf_mmap_base(gguf_handle)
    map_size = TinyNNCuda.tnn_gguf_mmap_size(gguf_handle)
    TinyNNCuda.tnn_session_attach_weight_mmap(@sess, map_base, map_size)

    eidx = TinyNNCuda.tnn_gguf_find_index(gguf_handle, "token_embd.weight")
    eoff = TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_handle, eidx)
    etyp = TinyNNCuda.tnn_gguf_tensor_type(gguf_handle, eidx)
    @t_seq_token_embed = TinyNNCuda.tnn_input_2d_persistent_mmap(@sess,
                           @seq_vocab_size, @seq_d_model, etyp, eoff)

    fnidx = TinyNNCuda.tnn_gguf_find_index(gguf_handle, "output_norm.weight")
    fnoff = TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_handle, fnidx)
    @t_seq_final_norm_gamma = TinyNNCuda.tnn_input_1d_persistent_mmap(@sess,
                                @seq_d_model, 0, fnoff)

    if untied
      oidx = TinyNNCuda.tnn_gguf_find_index(gguf_handle, "output.weight")
      ooff = TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_handle, oidx)
      otyp = TinyNNCuda.tnn_gguf_tensor_type(gguf_handle, oidx)
      @t_seq_output = TinyNNCuda.tnn_input_2d_persistent_mmap(@sess,
                        @seq_vocab_size, @seq_d_model, otyp, ooff)
    end

    @seq_blocks_ffi = [LlamaSeqBlockFFICuda.new]
    li_init = 1
    while li_init < @seq_n_layers
      @seq_blocks_ffi.push(LlamaSeqBlockFFICuda.new)
      li_init = li_init + 1
    end

    li = 0
    while li < @seq_n_layers
      blk = @seq_blocks_ffi[li]
      prefix = "blk." + li.to_s

      rn1_idx = TinyNNCuda.tnn_gguf_find_index(gguf_handle, prefix + ".attn_norm.weight")
      rn2_idx = TinyNNCuda.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_norm.weight")
      blk.t_seq_rn1_gamma = TinyNNCuda.tnn_input_1d_persistent_mmap(@sess,
                              @seq_d_model, 0,
                              TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_handle, rn1_idx))
      blk.t_seq_rn2_gamma = TinyNNCuda.tnn_input_1d_persistent_mmap(@sess,
                              @seq_d_model, 0,
                              TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_handle, rn2_idx))

      q_idx      = TinyNNCuda.tnn_gguf_find_index(gguf_handle, prefix + ".attn_q.weight")
      q_off_base = TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_handle, q_idx)
      q_type     = TinyNNCuda.tnn_gguf_tensor_type(gguf_handle, q_idx)
      q_stride   = head_nbytes(q_type, @seq_d_head, @seq_d_model)
      blk.t_seq_w_q = [TinyNNCuda.tnn_input_2d_persistent_mmap(@sess,
                         @seq_d_head, @seq_d_model, q_type, q_off_base)]
      hq = 1
      while hq < @seq_n_heads
        blk.t_seq_w_q.push(TinyNNCuda.tnn_input_2d_persistent_mmap(@sess,
                             @seq_d_head, @seq_d_model, q_type,
                             q_off_base + hq * q_stride))
        hq = hq + 1
      end

      k_idx      = TinyNNCuda.tnn_gguf_find_index(gguf_handle, prefix + ".attn_k.weight")
      v_idx      = TinyNNCuda.tnn_gguf_find_index(gguf_handle, prefix + ".attn_v.weight")
      k_off_base = TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_handle, k_idx)
      v_off_base = TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_handle, v_idx)
      k_type     = TinyNNCuda.tnn_gguf_tensor_type(gguf_handle, k_idx)
      v_type     = TinyNNCuda.tnn_gguf_tensor_type(gguf_handle, v_idx)
      k_stride   = head_nbytes(k_type, @seq_d_head, @seq_d_model)
      v_stride   = head_nbytes(v_type, @seq_d_head, @seq_d_model)
      blk.t_seq_w_k = [TinyNNCuda.tnn_input_2d_persistent_mmap(@sess,
                         @seq_d_head, @seq_d_model, k_type, k_off_base)]
      blk.t_seq_w_v = [TinyNNCuda.tnn_input_2d_persistent_mmap(@sess,
                         @seq_d_head, @seq_d_model, v_type, v_off_base)]
      hkv = 1
      while hkv < @seq_n_kv
        blk.t_seq_w_k.push(TinyNNCuda.tnn_input_2d_persistent_mmap(@sess,
                             @seq_d_head, @seq_d_model, k_type,
                             k_off_base + hkv * k_stride))
        blk.t_seq_w_v.push(TinyNNCuda.tnn_input_2d_persistent_mmap(@sess,
                             @seq_d_head, @seq_d_model, v_type,
                             v_off_base + hkv * v_stride))
        hkv = hkv + 1
      end

      if qkv_bias
        qb_idx = TinyNNCuda.tnn_gguf_find_index(gguf_handle, prefix + ".attn_q.bias")
        kb_idx = TinyNNCuda.tnn_gguf_find_index(gguf_handle, prefix + ".attn_k.bias")
        vb_idx = TinyNNCuda.tnn_gguf_find_index(gguf_handle, prefix + ".attn_v.bias")
        qb_off = TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_handle, qb_idx)
        kb_off = TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_handle, kb_idx)
        vb_off = TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_handle, vb_idx)
        bias_stride = @seq_d_head * 4

        blk.t_seq_b_q = [TinyNNCuda.tnn_input_1d_persistent_mmap(@sess, @seq_d_head, 0, qb_off)]
        hq = 1
        while hq < @seq_n_heads
          blk.t_seq_b_q.push(TinyNNCuda.tnn_input_1d_persistent_mmap(@sess, @seq_d_head, 0,
                               qb_off + hq * bias_stride))
          hq = hq + 1
        end
        blk.t_seq_b_k = [TinyNNCuda.tnn_input_1d_persistent_mmap(@sess, @seq_d_head, 0, kb_off)]
        blk.t_seq_b_v = [TinyNNCuda.tnn_input_1d_persistent_mmap(@sess, @seq_d_head, 0, vb_off)]
        hkv = 1
        while hkv < @seq_n_kv
          blk.t_seq_b_k.push(TinyNNCuda.tnn_input_1d_persistent_mmap(@sess, @seq_d_head, 0,
                               kb_off + hkv * bias_stride))
          blk.t_seq_b_v.push(TinyNNCuda.tnn_input_1d_persistent_mmap(@sess, @seq_d_head, 0,
                               vb_off + hkv * bias_stride))
          hkv = hkv + 1
        end
      end

      o_idx    = TinyNNCuda.tnn_gguf_find_index(gguf_handle, prefix + ".attn_output.weight")
      gate_idx = TinyNNCuda.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_gate.weight")
      up_idx   = TinyNNCuda.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_up.weight")
      down_idx = TinyNNCuda.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_down.weight")
      blk.t_seq_w_o    = TinyNNCuda.tnn_input_2d_persistent_mmap(@sess, @seq_d_model, @seq_d_model,
                           TinyNNCuda.tnn_gguf_tensor_type(gguf_handle, o_idx),
                           TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_handle, o_idx))
      blk.t_seq_w_gate = TinyNNCuda.tnn_input_2d_persistent_mmap(@sess, @seq_d_ff, @seq_d_model,
                           TinyNNCuda.tnn_gguf_tensor_type(gguf_handle, gate_idx),
                           TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_handle, gate_idx))
      blk.t_seq_w_up   = TinyNNCuda.tnn_input_2d_persistent_mmap(@sess, @seq_d_ff, @seq_d_model,
                           TinyNNCuda.tnn_gguf_tensor_type(gguf_handle, up_idx),
                           TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_handle, up_idx))
      blk.t_seq_w_down = TinyNNCuda.tnn_input_2d_persistent_mmap(@sess, @seq_d_model, @seq_d_ff,
                           TinyNNCuda.tnn_gguf_tensor_type(gguf_handle, down_idx),
                           TinyNNCuda.tnn_gguf_tensor_file_offset(gguf_handle, down_idx))

      li = li + 1
    end

    TinyNNCuda.tnn_finalize_weights(@sess)

    @t_seq_token_ids = TinyNNCuda.tnn_input_1d_i32(@sess, @seq_t)
    @t_seq_positions = TinyNNCuda.tnn_input_1d_i32_ctx(@sess, @seq_t)

    eps   = @seq_rms_eps
    scale = 1.0 / Math.sqrt(@seq_d_head.to_f)

    @t_seq_x_embed = TinyNNCuda.tnn_get_rows(@sess, @t_seq_token_embed, @t_seq_token_ids)
    TinyNNCuda.tnn_set_output(@t_seq_x_embed)

    t_cur = @t_seq_x_embed
    li_g = 0
    while li_g < @seq_n_layers
      t_cur = build_seq_block(t_cur, @seq_blocks_ffi[li_g], scale, eps)
      li_g = li_g + 1
    end

    @t_seq_x_final = TinyNNCuda.tnn_rms_norm(@sess, t_cur, @t_seq_final_norm_gamma, eps)
    TinyNNCuda.tnn_set_output(@t_seq_x_final)

    if @seq_has_untied_output
      @t_seq_logits = TinyNNCuda.tnn_matmul(@sess, @t_seq_output, @t_seq_x_final)
    else
      @t_seq_logits = TinyNNCuda.tnn_matmul(@sess, @t_seq_token_embed, @t_seq_x_final)
    end
    TinyNNCuda.tnn_set_output(@t_seq_logits)

    TinyNNCuda.tnn_realize(@sess, @t_seq_logits)
    @seq_realized = true
  end

  def head_nbytes(ggml_type, d_head, d_model)
    if ggml_type == 0
      d_head * d_model * 4
    elsif ggml_type == 8
      d_head * (d_model / 32) * 34
    else
      0
    end
  end

  def build_seq_block(t_x, blk, scale, eps)
    t_h = TinyNNCuda.tnn_rms_norm(@sess, t_x, blk.t_seq_rn1_gamma, eps)

    t_k_per_kv  = []
    t_vt_per_kv = []
    hkv = 0
    while hkv < @seq_n_kv
      t_k_raw = TinyNNCuda.tnn_matmul(@sess, blk.t_seq_w_k[hkv], t_h)
      if @seq_has_qkv_bias
        t_k_pre = TinyNNCuda.tnn_add(@sess, t_k_raw, blk.t_seq_b_k[hkv])
      else
        t_k_pre = t_k_raw
      end
      t_k_pre3 = TinyNNCuda.tnn_reshape_3d(@sess, t_k_pre, @seq_d_head, 1, @seq_t)
      t_k3     = TinyNNCuda.tnn_rope_ext(@sess, t_k_pre3, @t_seq_positions,
                                           @seq_d_head, @seq_rope_base)
      t_k      = TinyNNCuda.tnn_reshape_2d(@sess, t_k3, @seq_d_head, @seq_t)
      t_k_per_kv.push(t_k)

      t_v_raw = TinyNNCuda.tnn_matmul(@sess, blk.t_seq_w_v[hkv], t_h)
      if @seq_has_qkv_bias
        t_v = TinyNNCuda.tnn_add(@sess, t_v_raw, blk.t_seq_b_v[hkv])
      else
        t_v = t_v_raw
      end
      t_v_t = TinyNNCuda.tnn_transpose(@sess, t_v)
      t_vt_per_kv.push(t_v_t)
      hkv = hkv + 1
    end

    t_head_out0 = build_seq_qhead(t_h, blk, 0, t_k_per_kv, t_vt_per_kv, scale)
    t_head_outs = [t_head_out0]
    hq = 1
    while hq < @seq_n_heads
      t_head_outs.push(build_seq_qhead(t_h, blk, hq, t_k_per_kv, t_vt_per_kv, scale))
      hq = hq + 1
    end

    t_concat = t_head_outs[0]
    hq2 = 1
    while hq2 < @seq_n_heads
      t_concat = TinyNNCuda.tnn_concat(@sess, t_concat, t_head_outs[hq2], 0)
      hq2 = hq2 + 1
    end

    t_out_proj = TinyNNCuda.tnn_matmul(@sess, blk.t_seq_w_o, t_concat)
    t_x_attn   = TinyNNCuda.tnn_add(@sess, t_x, t_out_proj)

    t_h2    = TinyNNCuda.tnn_rms_norm(@sess, t_x_attn, blk.t_seq_rn2_gamma, eps)
    t_gate  = TinyNNCuda.tnn_matmul(@sess, blk.t_seq_w_gate, t_h2)
    t_up    = TinyNNCuda.tnn_matmul(@sess, blk.t_seq_w_up,   t_h2)
    t_silug = TinyNNCuda.tnn_silu(@sess, t_gate)
    t_gated = TinyNNCuda.tnn_mul(@sess, t_silug, t_up)
    t_dn    = TinyNNCuda.tnn_matmul(@sess, blk.t_seq_w_down, t_gated)

    TinyNNCuda.tnn_add(@sess, t_x_attn, t_dn)
  end

  def build_seq_qhead(t_h, blk, hq, t_k_per_kv, t_vt_per_kv, scale)
    hkv = hq / @seq_group_size
    t_q_raw = TinyNNCuda.tnn_matmul(@sess, blk.t_seq_w_q[hq], t_h)
    if @seq_has_qkv_bias
      t_q_pre = TinyNNCuda.tnn_add(@sess, t_q_raw, blk.t_seq_b_q[hq])
    else
      t_q_pre = t_q_raw
    end
    t_q_pre3 = TinyNNCuda.tnn_reshape_3d(@sess, t_q_pre, @seq_d_head, 1, @seq_t)
    t_q3     = TinyNNCuda.tnn_rope_ext(@sess, t_q_pre3, @t_seq_positions,
                                         @seq_d_head, @seq_rope_base)
    t_q      = TinyNNCuda.tnn_reshape_2d(@sess, t_q3, @seq_d_head, @seq_t)

    t_scores = TinyNNCuda.tnn_matmul(@sess, t_k_per_kv[hkv], t_q)
    t_scaled = TinyNNCuda.tnn_scale(@sess, t_scores, scale)
    t_masked = TinyNNCuda.tnn_diag_mask_inf(@sess, t_scaled, 0)
    t_attn   = TinyNNCuda.tnn_softmax(@sess, t_masked)
    TinyNNCuda.tnn_matmul(@sess, t_vt_per_kv[hkv], t_attn)
  end

  def forward(ids, positions)
    TinyNNCuda.upload_int_array(@sess, @t_seq_token_ids, ids)
    TinyNNCuda.upload_int_array(@sess, @t_seq_positions, positions)
    TinyNNCuda.tnn_compute(@sess)
    @t_seq_logits
  end
end
