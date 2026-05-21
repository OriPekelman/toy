# lib/llama_seq_forward_ffi.rb — sequence-mode forward graph for the
# llama-family architecture (RMSNorm + SwiGLU + RoPE + GQA).
#
# Sibling of lib/toy_smollm2_ffi_kv.rb (KV-cache decode for single
# positions). This class builds ONE forward graph that processes T
# token positions in a single compute — the prerequisite for real SFT
# (F1.2 step 6c/6d) where masked CE across all T positions is the loss.
#
# Step 1 (this file): forward only, no LoRA, no training. Parity gate
# is T=1 vs SmolLM2KVFFICache#decode_step at pos=0; full-T parity smoke
# (T>1) follows in step 2.
#
# Spinel hygiene — see docs/design/m3-seq-forward-2026-05-21.md, risk #6:
# the M1 full-forward integration was blocked by ivar-type unification
# across cache classes (t_seq / d_model / d_ff fields colliding). To
# sidestep it cleanly, every dimension/parameter ivar here uses a
# `@seq_*` prefix. Verbose but type-isolated.

require_relative "transformer"
require_relative "toy"
require_relative "toy_smollm2"
require_relative "tinynn"

# Per-block tensor handles. Distinct from SmolLM2KVBlockFFI so Spinel
# treats them as independent classes (no shared layout pressure).
class LlamaSeqBlockFFI
  attr_accessor :t_seq_rn1_gamma, :t_seq_rn2_gamma,
                :t_seq_w_q,  :t_seq_w_k,  :t_seq_w_v,  :t_seq_w_o,
                :t_seq_b_q,  :t_seq_b_k,  :t_seq_b_v,
                :t_seq_w_gate, :t_seq_w_up, :t_seq_w_down

  def initialize
    @t_seq_rn1_gamma = TinyNN.tnn_null_ptr
    @t_seq_rn2_gamma = TinyNN.tnn_null_ptr
    @t_seq_w_q = [TinyNN.tnn_null_ptr]
    @t_seq_w_k = [TinyNN.tnn_null_ptr]
    @t_seq_w_v = [TinyNN.tnn_null_ptr]
    @t_seq_b_q = [TinyNN.tnn_null_ptr]
    @t_seq_b_k = [TinyNN.tnn_null_ptr]
    @t_seq_b_v = [TinyNN.tnn_null_ptr]
    @t_seq_w_o    = TinyNN.tnn_null_ptr
    @t_seq_w_gate = TinyNN.tnn_null_ptr
    @t_seq_w_up   = TinyNN.tnn_null_ptr
    @t_seq_w_down = TinyNN.tnn_null_ptr
  end
end

class LlamaSeqForwardFFICache
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
    @sess                  = TinyNN.tnn_null_ptr
    @t_seq_token_embed     = TinyNN.tnn_null_ptr
    @t_seq_final_norm_gamma = TinyNN.tnn_null_ptr
    @t_seq_output          = TinyNN.tnn_null_ptr
    @seq_has_untied_output = false
    @seq_has_qkv_bias      = false
    @seq_blocks_ffi        = [LlamaSeqBlockFFI.new]
    @seq_gguf_handle_keepalive = TinyNN.tnn_null_ptr
    @t_seq_token_ids = TinyNN.tnn_null_ptr
    @t_seq_positions = TinyNN.tnn_null_ptr
    @t_seq_x_embed   = TinyNN.tnn_null_ptr
    @t_seq_x_final   = TinyNN.tnn_null_ptr
    @t_seq_logits    = TinyNN.tnn_null_ptr
  end

  # Allocate persistent weights mmap'd from `gguf_handle` (caller is
  # responsible for keeping the handle alive — we keepalive it via
  # @seq_gguf_handle_keepalive), compute inputs, and the full forward
  # graph for T = `t_seq` positions. Fixed T; rebuild for a different T.
  #
  # Weight layout matches SmolLM2KVFFICache#realize_for_mmap exactly
  # (same byte offsets + per-head split), so a sharded GGUF can be
  # loaded by either class.
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
    @sess                  = TinyNN.tnn_session_new(0)
    @seq_has_untied_output = untied
    @seq_has_qkv_bias      = qkv_bias

    map_base = TinyNN.tnn_gguf_mmap_base(gguf_handle)
    map_size = TinyNN.tnn_gguf_mmap_size(gguf_handle)
    TinyNN.tnn_session_attach_weight_mmap(@sess, map_base, map_size)

    # Embeddings + final norm + optional untied LM head.
    eidx = TinyNN.tnn_gguf_find_index(gguf_handle, "token_embd.weight")
    eoff = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, eidx)
    etyp = TinyNN.tnn_gguf_tensor_type(gguf_handle, eidx)
    @t_seq_token_embed = TinyNN.tnn_input_2d_persistent_mmap(@sess,
                           @seq_vocab_size, @seq_d_model, etyp, eoff)

    fnidx = TinyNN.tnn_gguf_find_index(gguf_handle, "output_norm.weight")
    fnoff = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, fnidx)
    @t_seq_final_norm_gamma = TinyNN.tnn_input_1d_persistent_mmap(@sess,
                                @seq_d_model, 0, fnoff)

    if untied
      oidx = TinyNN.tnn_gguf_find_index(gguf_handle, "output.weight")
      ooff = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, oidx)
      otyp = TinyNN.tnn_gguf_tensor_type(gguf_handle, oidx)
      @t_seq_output = TinyNN.tnn_input_2d_persistent_mmap(@sess,
                        @seq_vocab_size, @seq_d_model, otyp, ooff)
    end

    @seq_blocks_ffi = [LlamaSeqBlockFFI.new]
    li_init = 1
    while li_init < @seq_n_layers
      @seq_blocks_ffi.push(LlamaSeqBlockFFI.new)
      li_init = li_init + 1
    end

    li = 0
    while li < @seq_n_layers
      blk = @seq_blocks_ffi[li]
      prefix = "blk." + li.to_s

      rn1_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_norm.weight")
      rn2_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_norm.weight")
      blk.t_seq_rn1_gamma = TinyNN.tnn_input_1d_persistent_mmap(@sess,
                              @seq_d_model, 0,
                              TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, rn1_idx))
      blk.t_seq_rn2_gamma = TinyNN.tnn_input_1d_persistent_mmap(@sess,
                              @seq_d_model, 0,
                              TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, rn2_idx))

      # Q heads — per-head [d_head, d_model] tensor, n_heads of them.
      q_idx      = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_q.weight")
      q_off_base = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, q_idx)
      q_type     = TinyNN.tnn_gguf_tensor_type(gguf_handle, q_idx)
      q_stride   = head_nbytes(q_type, @seq_d_head, @seq_d_model)
      blk.t_seq_w_q = [TinyNN.tnn_input_2d_persistent_mmap(@sess,
                         @seq_d_head, @seq_d_model, q_type, q_off_base)]
      hq = 1
      while hq < @seq_n_heads
        blk.t_seq_w_q.push(TinyNN.tnn_input_2d_persistent_mmap(@sess,
                             @seq_d_head, @seq_d_model, q_type,
                             q_off_base + hq * q_stride))
        hq = hq + 1
      end

      # K, V heads — per-KV-head [d_head, d_model].
      k_idx      = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_k.weight")
      v_idx      = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_v.weight")
      k_off_base = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, k_idx)
      v_off_base = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, v_idx)
      k_type     = TinyNN.tnn_gguf_tensor_type(gguf_handle, k_idx)
      v_type     = TinyNN.tnn_gguf_tensor_type(gguf_handle, v_idx)
      k_stride   = head_nbytes(k_type, @seq_d_head, @seq_d_model)
      v_stride   = head_nbytes(v_type, @seq_d_head, @seq_d_model)
      blk.t_seq_w_k = [TinyNN.tnn_input_2d_persistent_mmap(@sess,
                         @seq_d_head, @seq_d_model, k_type, k_off_base)]
      blk.t_seq_w_v = [TinyNN.tnn_input_2d_persistent_mmap(@sess,
                         @seq_d_head, @seq_d_model, v_type, v_off_base)]
      hkv = 1
      while hkv < @seq_n_kv
        blk.t_seq_w_k.push(TinyNN.tnn_input_2d_persistent_mmap(@sess,
                             @seq_d_head, @seq_d_model, k_type,
                             k_off_base + hkv * k_stride))
        blk.t_seq_w_v.push(TinyNN.tnn_input_2d_persistent_mmap(@sess,
                             @seq_d_head, @seq_d_model, v_type,
                             v_off_base + hkv * v_stride))
        hkv = hkv + 1
      end

      # Optional Q/K/V biases (Qwen2.x). 1D [d_head] per head, contiguous.
      if qkv_bias
        qb_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_q.bias")
        kb_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_k.bias")
        vb_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_v.bias")
        qb_off = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, qb_idx)
        kb_off = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, kb_idx)
        vb_off = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, vb_idx)
        bias_stride = @seq_d_head * 4

        blk.t_seq_b_q = [TinyNN.tnn_input_1d_persistent_mmap(@sess, @seq_d_head, 0, qb_off)]
        hq = 1
        while hq < @seq_n_heads
          blk.t_seq_b_q.push(TinyNN.tnn_input_1d_persistent_mmap(@sess, @seq_d_head, 0,
                               qb_off + hq * bias_stride))
          hq = hq + 1
        end
        blk.t_seq_b_k = [TinyNN.tnn_input_1d_persistent_mmap(@sess, @seq_d_head, 0, kb_off)]
        blk.t_seq_b_v = [TinyNN.tnn_input_1d_persistent_mmap(@sess, @seq_d_head, 0, vb_off)]
        hkv = 1
        while hkv < @seq_n_kv
          blk.t_seq_b_k.push(TinyNN.tnn_input_1d_persistent_mmap(@sess, @seq_d_head, 0,
                               kb_off + hkv * bias_stride))
          blk.t_seq_b_v.push(TinyNN.tnn_input_1d_persistent_mmap(@sess, @seq_d_head, 0,
                               vb_off + hkv * bias_stride))
          hkv = hkv + 1
        end
      end

      # O, FFN — full 2D weights, no per-head split.
      o_idx    = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_output.weight")
      gate_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_gate.weight")
      up_idx   = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_up.weight")
      down_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_down.weight")
      blk.t_seq_w_o    = TinyNN.tnn_input_2d_persistent_mmap(@sess, @seq_d_model, @seq_d_model,
                           TinyNN.tnn_gguf_tensor_type(gguf_handle, o_idx),
                           TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, o_idx))
      blk.t_seq_w_gate = TinyNN.tnn_input_2d_persistent_mmap(@sess, @seq_d_ff, @seq_d_model,
                           TinyNN.tnn_gguf_tensor_type(gguf_handle, gate_idx),
                           TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, gate_idx))
      blk.t_seq_w_up   = TinyNN.tnn_input_2d_persistent_mmap(@sess, @seq_d_ff, @seq_d_model,
                           TinyNN.tnn_gguf_tensor_type(gguf_handle, up_idx),
                           TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, up_idx))
      blk.t_seq_w_down = TinyNN.tnn_input_2d_persistent_mmap(@sess, @seq_d_model, @seq_d_ff,
                           TinyNN.tnn_gguf_tensor_type(gguf_handle, down_idx),
                           TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, down_idx))

      li = li + 1
    end

    TinyNN.tnn_finalize_weights(@sess)

    # Compute inputs. Token IDs and positions live in the session (compute)
    # context, length T each. Caller fills them via upload_int_array.
    @t_seq_token_ids = TinyNN.tnn_input_1d_i32(@sess, @seq_t)
    @t_seq_positions = TinyNN.tnn_input_1d_i32_ctx(@sess, @seq_t)

    eps   = @seq_rms_eps
    scale = 1.0 / Math.sqrt(@seq_d_head.to_f)

    # Embed lookup. ggml's get_rows over an int32 vector of length T
    # gives ne=[d_model, T] — same shape the per-block ops expect.
    @t_seq_x_embed = TinyNN.tnn_get_rows(@sess, @t_seq_token_embed, @t_seq_token_ids)
    TinyNN.tnn_set_output(@t_seq_x_embed)

    t_cur = @t_seq_x_embed
    li_g = 0
    while li_g < @seq_n_layers
      t_cur = build_seq_block(t_cur, @seq_blocks_ffi[li_g], scale, eps)
      li_g = li_g + 1
    end

    @t_seq_x_final = TinyNN.tnn_rms_norm(@sess, t_cur, @t_seq_final_norm_gamma, eps)
    TinyNN.tnn_set_output(@t_seq_x_final)

    if @seq_has_untied_output
      @t_seq_logits = TinyNN.tnn_matmul(@sess, @t_seq_output, @t_seq_x_final)
    else
      @t_seq_logits = TinyNN.tnn_matmul(@sess, @t_seq_token_embed, @t_seq_x_final)
    end
    TinyNN.tnn_set_output(@t_seq_logits)

    TinyNN.tnn_realize(@sess, @t_seq_logits)
    @seq_realized = true
  end

  # GGUF type → bytes-per-row stride for per-head slicing. Mirrors the
  # SmolLM2KVFFICache helper of the same name. F32=0, Q8_0=8.
  def head_nbytes(ggml_type, d_head, d_model)
    if ggml_type == 0
      d_head * d_model * 4
    elsif ggml_type == 8
      d_head * (d_model / 32) * 34
    else
      0
    end
  end

  # One transformer block.
  #
  #   h1   = RMSNorm(x)
  #   per KV head kv_h:
  #     k_pre = w_k[kv_h] @ h1  (+ b_k[kv_h])         ne=[d_head, T]
  #     k     = RoPE(k_pre, positions)
  #     v     = w_v[kv_h] @ h1  (+ b_v[kv_h])         ne=[d_head, T]
  #     v_t   = transpose(v)                          ne=[T, d_head]
  #   per Q head q_h (kv_h = q_h / group_size):
  #     q_pre = w_q[q_h] @ h1  (+ b_q[q_h])           ne=[d_head, T]
  #     q     = RoPE(q_pre, positions)
  #     scores = k[kv_h] @ q                          ne=[T_keys, T_queries]
  #     scaled = scores / sqrt(d_head)
  #     masked = diag_mask_inf(scaled, 0)              causal triangle
  #     attn   = softmax(masked)                       ne=[T, T]
  #     head_h = v_t[kv_h] @ attn                     ne=[d_head, T]
  #   concat heads along ne0 → ne=[d_model, T]
  #   x_attn = x + (w_o @ concat)
  #   h2     = RMSNorm(x_attn)
  #   ff     = w_down @ (silu(w_gate @ h2) * (w_up @ h2))
  #   x_out  = x_attn + ff
  def build_seq_block(t_x, blk, scale, eps)
    t_h = TinyNN.tnn_rms_norm(@sess, t_x, blk.t_seq_rn1_gamma, eps)

    # K, V over all KV heads. Pre-compute v_t per head so the per-Q-head
    # attention loop can index it (avoids n_heads × transpose).
    t_k_per_kv = []
    t_vt_per_kv = []
    hkv = 0
    while hkv < @seq_n_kv
      t_k_raw = TinyNN.tnn_matmul(@sess, blk.t_seq_w_k[hkv], t_h)
      if @seq_has_qkv_bias
        t_k_pre = TinyNN.tnn_add(@sess, t_k_raw, blk.t_seq_b_k[hkv])
      else
        t_k_pre = t_k_raw
      end
      # ggml_rope_ext requires a->ne[2] == positions->ne[0]. Our K is
      # ne=[d_head, T] (ne[2]=1); reshape to ne=[d_head, 1, T] so ne[2]==T,
      # then reshape back after rope. Reshape is metadata-only (no copy)
      # on contiguous tensors. At T=1 this is a no-op (1 == 1).
      t_k_pre3 = TinyNN.tnn_reshape_3d(@sess, t_k_pre, @seq_d_head, 1, @seq_t)
      t_k3     = TinyNN.tnn_rope_ext(@sess, t_k_pre3, @t_seq_positions,
                                       @seq_d_head, @seq_rope_base)
      t_k      = TinyNN.tnn_reshape_2d(@sess, t_k3, @seq_d_head, @seq_t)
      t_k_per_kv.push(t_k)

      t_v_raw = TinyNN.tnn_matmul(@sess, blk.t_seq_w_v[hkv], t_h)
      if @seq_has_qkv_bias
        t_v = TinyNN.tnn_add(@sess, t_v_raw, blk.t_seq_b_v[hkv])
      else
        t_v = t_v_raw
      end
      # head_out = v_t @ attn. v has ne=[d_head, T]; transpose to
      # ne=[T, d_head] so the second matmul's contraction lines up.
      t_v_t = TinyNN.tnn_transpose(@sess, t_v)
      t_vt_per_kv.push(t_v_t)
      hkv = hkv + 1
    end

    # Per-Q-head attention. GQA: each Q head reads from kv_h = q_h / group_size.
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
      t_concat = TinyNN.tnn_concat(@sess, t_concat, t_head_outs[hq2], 0)
      hq2 = hq2 + 1
    end

    t_out_proj = TinyNN.tnn_matmul(@sess, blk.t_seq_w_o, t_concat)
    t_x_attn   = TinyNN.tnn_add(@sess, t_x, t_out_proj)

    # SwiGLU FFN.
    t_h2    = TinyNN.tnn_rms_norm(@sess, t_x_attn, blk.t_seq_rn2_gamma, eps)
    t_gate  = TinyNN.tnn_matmul(@sess, blk.t_seq_w_gate, t_h2)
    t_up    = TinyNN.tnn_matmul(@sess, blk.t_seq_w_up,   t_h2)
    t_silug = TinyNN.tnn_silu(@sess, t_gate)
    t_gated = TinyNN.tnn_mul(@sess, t_silug, t_up)
    t_dn    = TinyNN.tnn_matmul(@sess, blk.t_seq_w_down, t_gated)

    TinyNN.tnn_add(@sess, t_x_attn, t_dn)
  end

  def build_seq_qhead(t_h, blk, hq, t_k_per_kv, t_vt_per_kv, scale)
    hkv = hq / @seq_group_size
    t_q_raw = TinyNN.tnn_matmul(@sess, blk.t_seq_w_q[hq], t_h)
    if @seq_has_qkv_bias
      t_q_pre = TinyNN.tnn_add(@sess, t_q_raw, blk.t_seq_b_q[hq])
    else
      t_q_pre = t_q_raw
    end
    # Same rope-shape lift as the K path; see comment in build_seq_block.
    t_q_pre3 = TinyNN.tnn_reshape_3d(@sess, t_q_pre, @seq_d_head, 1, @seq_t)
    t_q3     = TinyNN.tnn_rope_ext(@sess, t_q_pre3, @t_seq_positions,
                                     @seq_d_head, @seq_rope_base)
    t_q      = TinyNN.tnn_reshape_2d(@sess, t_q3, @seq_d_head, @seq_t)

    # scores ne=[T_keys, T_queries]. Same shape as decode_step's
    # matmul(K_hist, q) at T_keys = pos+1, T_queries = 1.
    t_scores = TinyNN.tnn_matmul(@sess, t_k_per_kv[hkv], t_q)
    t_scaled = TinyNN.tnn_scale(@sess, t_scores, scale)
    # Causal mask: pos j > pos i (col > row) becomes -inf. n_past=0 is
    # the full-sequence case (no prior context outside the T window).
    t_masked = TinyNN.tnn_diag_mask_inf(@sess, t_scaled, 0)
    t_attn   = TinyNN.tnn_softmax(@sess, t_masked)
    # head ne=[d_head, T]
    TinyNN.tnn_matmul(@sess, t_vt_per_kv[hkv], t_attn)
  end

  # Run one forward pass. `ids` and `positions` are length-T Int arrays.
  # Returns the t_seq_logits handle; caller downloads via download_row_major
  # against (vocab, T) shape.
  def forward(ids, positions)
    TinyNN.upload_int_array(@sess, @t_seq_token_ids, ids)
    TinyNN.upload_int_array(@sess, @t_seq_positions, positions)
    TinyNN.tnn_compute(@sess)
    @t_seq_logits
  end
end
