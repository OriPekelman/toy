# FFI bridge between the project's Mat (row-major f64) and ggml's
# backend-aware tensor library. Loads the static archives produced by
# the Makefile's `setup-ggml` (or `setup-ggml-cuda`) target.
#
# Usage (from a driver script):
#
#   require_relative "lib/transformer"   # defines Mat
#   require_relative "lib/tinynn"        # adds TinyNN.matmul(a, b)
#
# TinyNN.matmul(a, b) computes a ** b and returns a fresh Mat with the
# same row-major layout. Internally it spins up a one-shot ggml session,
# uploads, computes, downloads, frees. Performance is bad for many small
# calls (kernel-launch + backend-init per call); the eventual fix is to
# share a persistent session across the training step. For S2 *** a single
# A/B smoke check *** one-shot is fine.

# Persistent FFI cache for one transformer block's FFN. Single ggml
# session holding the full chain `matmul -> gelu -> matmul`. Activations
# stay inside ggml between the two matmuls; only the three outputs
# (pre, hidden, out) are downloaded at the end.
#
# Lazy-realized: T (sequence length) isn't known until the first
# forward call. realize_for(t_seq, d_model, d_ff) sets up the graph;
# subsequent calls with the same T reuse it.
#
# Operand layout: we feed matmul1 as `matmul(t_w1_t, t_h)` so its
# result has ne0=d_ff -- which is the k-dim of matmul2 -- so the
# chain doesn't need an intermediate transpose. Downloads of all
# three result tensors are then a straight row-major memcpy.
class FFNFFICache
  attr_accessor :sess, :t_h, :t_w1_t, :t_w2_t,
                :t_pre, :t_hidden, :t_out,
                :t_seq, :d_model, :d_ff, :realized

  def initialize
    @realized = false
    @t_seq    = 0
    @d_model  = 0
    @d_ff     = 0
    # `:ptr` ivars seed with TinyNN.tnn_null_ptr (a typed NULL `void *`)
    # rather than `nil`. Post-spinel `85a4670`, mixing `nil` with `:ptr`
    # boxes the ivar as `sp_RbVal`, which then fails the `(void *)` cast
    # at every FFI call site downstream. The typed-NULL seed keeps the
    # ivar as plain `void *` end-to-end.
    @sess     = TinyNN.tnn_null_ptr
    @t_h      = TinyNN.tnn_null_ptr
    @t_w1_t   = TinyNN.tnn_null_ptr
    @t_w2_t   = TinyNN.tnn_null_ptr
    @t_pre    = TinyNN.tnn_null_ptr
    @t_hidden = TinyNN.tnn_null_ptr
    @t_out    = TinyNN.tnn_null_ptr
  end

  def realize_for(t_seq, d_model, d_ff)
    @t_seq   = t_seq
    @d_model = d_model
    @d_ff    = d_ff

    @sess = TinyNN.tnn_session_new(0)
    # t_h:    ne=[d_model, T] -- h uploaded row-major (data[k] = h.flat[k]).
    # t_w1_t: ne=[d_model, d_ff] -- w1 uploaded transposed.
    # t_w2_t: ne=[d_ff, d_model] -- w2 uploaded transposed.
    @t_h    = TinyNN.tnn_input_2d_f32(@sess, t_seq,  d_model)
    @t_w1_t = TinyNN.tnn_input_2d_f32(@sess, d_ff,   d_model)
    @t_w2_t = TinyNN.tnn_input_2d_f32(@sess, d_model, d_ff)

    # Chain: mul_mat(w1_t, h) -> gelu -> mul_mat(w2_t, hidden).
    # Result shapes (ggml ne):  [d_ff, T] -> [d_ff, T] -> [d_model, T].
    @t_pre    = TinyNN.tnn_matmul(@sess, @t_w1_t, @t_h)
    @t_hidden = TinyNN.tnn_gelu(@sess, @t_pre)
    @t_out    = TinyNN.tnn_matmul(@sess, @t_w2_t, @t_hidden)
    # Mark intermediates as outputs so the scheduler doesn't alias
    # their buffers with later ops -- backward needs pre and hidden.
    TinyNN.tnn_set_output(@t_pre)
    TinyNN.tnn_set_output(@t_hidden)
    TinyNN.tnn_set_output(@t_out)
    TinyNN.tnn_realize(@sess, @t_out)

    @realized = true
  end
end

# Per-block tensor handles for FullForwardFFICache. One instance per
# transformer block. All ivars are :ptr handles (or arrays thereof);
# the actual ggml tensors live in the FullForwardFFICache's session.
class BlockFFICache
  attr_accessor :t_norm1_gamma, :t_norm2_gamma,
                :t_w_q, :t_w_k, :t_w_v,   # Array<:ptr>, one per head
                :t_w_o, :t_w_ff1, :t_w_ff2

  def initialize
    # Post-spinel `85a4670` (Array<:ptr> -> PtrArray): every `:ptr`
    # ivar seeds with TinyNN.tnn_null_ptr so the inferred type stays
    # `void *` end-to-end. A bare `nil` seed boxes the ivar as
    # `sp_RbVal` (nilable) and FFI dispatch then fails the `(void *)`
    # cast.
    @t_norm1_gamma = TinyNN.tnn_null_ptr
    @t_norm2_gamma = TinyNN.tnn_null_ptr
    @t_w_q   = [TinyNN.tnn_null_ptr]
    @t_w_k   = [TinyNN.tnn_null_ptr]
    @t_w_v   = [TinyNN.tnn_null_ptr]
    @t_w_o   = TinyNN.tnn_null_ptr
    @t_w_ff1 = TinyNN.tnn_null_ptr
    @t_w_ff2 = TinyNN.tnn_null_ptr
  end
end

# Full forward of a TransformerLM as one persistent ggml graph. Built
# incrementally; M1.1 covered embed + positional embedding + tied
# unembed (the bookends). M1.2 adds one full transformer block:
# pre-RMSNorm, multi-head causal attention, residual, pre-RMSNorm, FFN,
# residual. M1.3+ will scale to n_layers blocks.
#
# Layout conventions (see project_chained_ffn_2026_05_14):
#   - Mat (rows, cols) row-major upload  -> ggml ne=[cols, rows]
#   - Per-block intermediates carry ne=[d_model, T]: elem(d, t) is the
#     logical value at (row=t, col=d).
#
# Persistent (ctx_w):
#   - t_token_embed (vocab, d_model)
#   - t_pos_slice   (T, d_model)
#   - t_final_norm_gamma (d_model)
#   - per block (in @blocks_ffi):
#     - t_norm1_gamma, t_norm2_gamma (d_model)
#     - t_w_q[h], t_w_k[h], t_w_v[h] (d_model, d_head) per head
#     - t_w_o   (d_model, d_model)
#     - t_w_ff1 (d_model, d_ff), t_w_ff2 (d_ff, d_model)
#
# Compute (ctx):       t_token_ids (T int32), intermediates, t_logits
class FullForwardFFICache
  attr_accessor :sess, :t_token_embed, :t_pos_slice, :t_token_ids,
                :t_final_norm_gamma,
                :blocks_ffi,
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
    @sess               = TinyNN.tnn_null_ptr
    @t_token_embed      = TinyNN.tnn_null_ptr
    @t_pos_slice        = TinyNN.tnn_null_ptr
    @t_token_ids        = TinyNN.tnn_null_ptr
    @t_final_norm_gamma = TinyNN.tnn_null_ptr
    @t_x_embed          = TinyNN.tnn_null_ptr
    @t_x_final          = TinyNN.tnn_null_ptr
    @t_logits           = TinyNN.tnn_null_ptr
    @blocks_ffi         = [BlockFFICache.new]
  end

  def realize_for(t_seq, d_model, d_ff, n_heads, n_layers, vocab_size)
    @t_seq      = t_seq
    @d_model    = d_model
    @d_ff       = d_ff
    @n_heads    = n_heads
    @d_head     = d_model / n_heads
    @n_layers   = n_layers
    @vocab_size = vocab_size

    @sess = TinyNN.tnn_session_new(0)

    # === Persistent weights (ctx_w) ===
    @t_token_embed      = TinyNN.tnn_input_2d_f32_persistent(@sess, vocab_size, d_model)
    @t_pos_slice        = TinyNN.tnn_input_2d_f32_persistent(@sess, t_seq,      d_model)
    @t_final_norm_gamma = TinyNN.tnn_input_1d_f32_persistent(@sess, d_model)

    # Build per-block tensor handles (seed-then-push for Spinel's
    # Array<BlockFFICache> inference).
    @blocks_ffi = [BlockFFICache.new]
    li = 1
    while li < n_layers
      @blocks_ffi.push(BlockFFICache.new)
      li = li + 1
    end

    li = 0
    while li < n_layers
      blk = @blocks_ffi[li]
      blk.t_norm1_gamma = TinyNN.tnn_input_1d_f32_persistent(@sess, d_model)
      blk.t_norm2_gamma = TinyNN.tnn_input_1d_f32_persistent(@sess, d_model)
      # Per-head Q/K/V: shape (d_model, d_head). Uploaded TRANSPOSED so
      # ggml ne=[d_model, d_head] holds w.elem(r, c) = w[r][c].
      blk.t_w_q = [TinyNN.tnn_input_2d_f32_persistent(@sess, d_head, d_model)]
      blk.t_w_k = [TinyNN.tnn_input_2d_f32_persistent(@sess, d_head, d_model)]
      blk.t_w_v = [TinyNN.tnn_input_2d_f32_persistent(@sess, d_head, d_model)]
      h = 1
      while h < n_heads
        blk.t_w_q.push(TinyNN.tnn_input_2d_f32_persistent(@sess, d_head, d_model))
        blk.t_w_k.push(TinyNN.tnn_input_2d_f32_persistent(@sess, d_head, d_model))
        blk.t_w_v.push(TinyNN.tnn_input_2d_f32_persistent(@sess, d_head, d_model))
        h = h + 1
      end
      blk.t_w_o   = TinyNN.tnn_input_2d_f32_persistent(@sess, d_model, d_model)
      blk.t_w_ff1 = TinyNN.tnn_input_2d_f32_persistent(@sess, d_ff,    d_model)
      blk.t_w_ff2 = TinyNN.tnn_input_2d_f32_persistent(@sess, d_model, d_ff)
      li = li + 1
    end

    TinyNN.tnn_finalize_weights(@sess)

    # === Compute input ===
    @t_token_ids = TinyNN.tnn_input_1d_i32(@sess, t_seq)

    # === Forward graph ===
    # x_embed = token_embed[ids] + pos_slice  (ne=[d_model, T])
    t_embedded = TinyNN.tnn_get_rows(@sess, @t_token_embed, @t_token_ids)
    @t_x_embed = TinyNN.tnn_add(@sess, t_embedded, @t_pos_slice)
    TinyNN.tnn_set_output(@t_x_embed)

    # Through each block.
    t_cur = @t_x_embed
    eps   = 1.0e-5
    scale = 1.0 / Math.sqrt(d_head.to_f)
    li = 0
    while li < n_layers
      t_cur = build_block(t_cur, @blocks_ffi[li], eps, scale)
      li = li + 1
    end

    # Final RMSNorm on the post-blocks x.
    @t_x_final = TinyNN.tnn_rms_norm(@sess, t_cur, @t_final_norm_gamma, eps)
    TinyNN.tnn_set_output(@t_x_final)

    # Tied unembed: logits = mul_mat(token_embed, x_final)  ne=[vocab, T]
    @t_logits = TinyNN.tnn_matmul(@sess, @t_token_embed, @t_x_final)
    TinyNN.tnn_set_output(@t_logits)

    TinyNN.tnn_realize(@sess, @t_logits)
    @realized = true
  end

  # Build one transformer block's graph nodes. Returns the block's
  # output tensor (post-FFN residual). Mathematics:
  #   h1 = rms_norm(x, norm1_gamma)
  #   per head h:
  #     q_h = w_q[h]^T @ h1     (mul_mat(w_q_t_h, h1)  ne=[d_head, T])
  #     k_h = w_k[h]^T @ h1
  #     v_h = h1 @ w_v[h]       (mul_mat(h1, w_v_t_h)  ne=[T, d_head])
  #     scores_h = mul_mat(k_h, q_h)   ne=[T_key, T_query]
  #     scaled_h = scale(scores_h, 1/sqrt(d_head))
  #     masked_h = diag_mask_inf(scaled_h, 0)         -- causal
  #     attn_h   = soft_max(masked_h)  -- per-query softmax over keys
  #     head_out_h = mul_mat(v_h, attn_h)  ne=[d_head, T_query]
  #   concat = concat_along_d(head_out_h for h in heads)  ne=[d_model, T]
  #   out_proj = mul_mat(w_o_t, concat)  ne=[d_model, T]
  #   x_attn = x + out_proj
  #   h2 = rms_norm(x_attn, norm2_gamma)
  #   ffn:
  #     pre    = mul_mat(w_ff1_t, h2)   ne=[d_ff,    T]
  #     hidden = gelu(pre)
  #     ffn_out= mul_mat(w_ff2_t, hidden) ne=[d_model, T]
  #   x_out = x_attn + ffn_out
  def build_block(t_x, blk, eps, scale)
    # Pre-norm before attention.
    t_h1 = TinyNN.tnn_rms_norm(@sess, t_x, blk.t_norm1_gamma, eps)

    # Per-head attention. Build each head's output, then concat.
    t_head_outs = [build_attention_head(t_h1, blk.t_w_q[0], blk.t_w_k[0], blk.t_w_v[0], scale)]
    h = 1
    while h < @n_heads
      t_head_outs.push(build_attention_head(t_h1, blk.t_w_q[h], blk.t_w_k[h], blk.t_w_v[h], scale))
      h = h + 1
    end

    # Concat along ne0 (d_head -> d_model).
    t_concat = t_head_outs[0]
    h = 1
    while h < @n_heads
      t_concat = TinyNN.tnn_concat(@sess, t_concat, t_head_outs[h], 0)
      h = h + 1
    end

    # Output projection + residual.
    t_out_proj = TinyNN.tnn_matmul(@sess, blk.t_w_o, t_concat)
    t_x_attn   = TinyNN.tnn_add(@sess, t_x, t_out_proj)

    # Pre-norm before FFN.
    t_h2 = TinyNN.tnn_rms_norm(@sess, t_x_attn, blk.t_norm2_gamma, eps)

    # FFN (matches FFNFFICache's chained design).
    t_pre    = TinyNN.tnn_matmul(@sess, blk.t_w_ff1, t_h2)
    t_hidden = TinyNN.tnn_gelu(@sess, t_pre)
    t_ffn    = TinyNN.tnn_matmul(@sess, blk.t_w_ff2, t_hidden)

    # Second residual.
    TinyNN.tnn_add(@sess, t_x_attn, t_ffn)
  end

  # Single attention head, given pre-normed x and the head's persistent
  # Q/K/V weights. See build_block's docstring for the math.
  def build_attention_head(t_x, t_w_q, t_w_k, t_w_v, scale)
    t_q = TinyNN.tnn_matmul(@sess, t_w_q, t_x)   # ne=[d_head, T]
    t_k = TinyNN.tnn_matmul(@sess, t_w_k, t_x)   # ne=[d_head, T]
    # v in Pattern A (ne=[T, d_head]) so head_out's k_dim matches.
    # mul_mat(x, w_v_t) where x.ne=[d_model, T] and w_v_t.ne=[d_model, d_head]
    # yields ne=[T, d_head]. ✓
    t_v = TinyNN.tnn_matmul(@sess, t_x, t_w_v)

    t_scores = TinyNN.tnn_matmul(@sess, t_k, t_q)            # ne=[T_key, T_query]
    t_scaled = TinyNN.tnn_scale(@sess, t_scores, scale)
    t_masked = TinyNN.tnn_diag_mask_inf(@sess, t_scaled, 0)
    t_attn   = TinyNN.tnn_softmax(@sess, t_masked)           # softmax along ne0 = key dim

    TinyNN.tnn_matmul(@sess, t_v, t_attn)                    # ne=[d_head, T_query]
  end
end

# Holder for adam_step's three return values. (Spinel doesn't reliably
# handle tuple/array returns of mixed-shape Mats — same workaround as
# lib/transformer.rb's NormResult/FFResult/etc.)
#
# Field names deliberately avoid `m`/`v` to dodge a name collision with
# AdamState.m / AdamState.v in lib/transformer.rb — Spinel's iterative
# type inference unifies field types by name, and AdamState.m holds a
# Gradients (not a Mat), so naming the field `m` here makes the codegen
# mistype this slot.
class AdamStepResult
  attr_accessor :param, :mom_m, :mom_v
  def initialize(param, mom_m, mom_v)
    @param = param
    @mom_m = mom_m
    @mom_v = mom_v
  end
end

module TinyNN
  ffi_lib "tinynn_ggml"
  ffi_lib "ggml"
  ffi_lib "ggml-cpu"
  ffi_lib "ggml-base"
  ffi_lib "stdc++"
  ffi_lib "pthread"
  # spinel wrapper adds -lm *before* FFI libs; ggml-cpu needs it again.
  ffi_lib "m"

  # `-Wno-int-conversion` keeps newer gccs (ubuntu noble, Ruby 3.4
  # container, etc.) from upgrading `Array<:ptr>` -> IntArray cc
  # warnings to errors. The pointer round-trips through mrb_int
  # cleanly on 64-bit; filed upstream as matz/spinel#492.
  ffi_cflags "-L. -Ltinynn -Lvendor/ggml/build/src -Lvendor/ggml/build/src/ggml-cpu -Wno-int-conversion"

  ffi_func :tnn_session_new,      [:int],                   :ptr
  ffi_func :tnn_session_free,     [:ptr],                   :void
  ffi_func :tnn_backend_name,     [:ptr],                   :str
  ffi_func :tnn_input_2d_f32,     [:ptr, :int, :int],       :ptr
  ffi_func :tnn_matmul,           [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_add,              [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_gelu,             [:ptr, :ptr],             :ptr
  ffi_func :tnn_rms_norm,         [:ptr, :ptr, :ptr, :double], :ptr
  ffi_func :tnn_softmax,          [:ptr, :ptr],             :ptr
  ffi_func :tnn_diag_mask_inf,    [:ptr, :ptr, :int],       :ptr
  # Llama-family ops
  ffi_func :tnn_silu,             [:ptr, :ptr],             :ptr
  ffi_func :tnn_mul,              [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_rope_ext,         [:ptr, :ptr, :ptr, :int, :double], :ptr
  ffi_func :tnn_input_1d_i32_ctx, [:ptr, :int],             :ptr
  ffi_func :tnn_concat,           [:ptr, :ptr, :ptr, :int], :ptr
  ffi_func :tnn_null_ptr,         [],                       :ptr
  ffi_func :tnn_layer_norm,       [:ptr, :ptr, :ptr, :ptr, :double], :ptr
  ffi_func :tnn_view_1d,          [:ptr, :ptr, :int, :long],         :ptr
  ffi_func :tnn_view_2d,          [:ptr, :ptr, :int, :int, :long, :long], :ptr
  ffi_func :tnn_reshape_3d,       [:ptr, :ptr, :int, :int, :int], :ptr
  ffi_func :tnn_reshape_2d,       [:ptr, :ptr, :int, :int],       :ptr
  ffi_func :tnn_cpy,              [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_set_rows,         [:ptr, :ptr, :ptr, :ptr], :ptr
  ffi_func :tnn_soft_max_ext,     [:ptr, :ptr, :ptr, :double, :double], :ptr
  ffi_func :tnn_set_2d,           [:ptr, :ptr, :ptr, :long, :long], :ptr
  ffi_func :tnn_reset_for_rebuild, [:ptr],                  :int
  ffi_func :tnn_transpose,        [:ptr, :ptr],             :ptr
  ffi_func :tnn_scale,            [:ptr, :ptr, :double],    :ptr
  ffi_func :tnn_rms_norm_back,    [:ptr, :ptr, :ptr, :double], :ptr
  ffi_func :tnn_softmax_back,     [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_silu_back,        [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_rope_ext_back,    [:ptr, :ptr, :ptr, :int, :double], :ptr
  # Phase F0.4 — autograd
  ffi_func :tnn_sum,                  [:ptr, :ptr],             :ptr
  ffi_func :tnn_set_loss,             [:ptr],                   :void
  ffi_func :tnn_build_backward,       [:ptr],                   :int
  ffi_func :tnn_extend_backward_graph,[:ptr, :ptr],             :int
  ffi_func :tnn_realize_backward,     [:ptr],                   :int
  ffi_func :tnn_graph_reset,          [:ptr],                   :int
  ffi_func :tnn_compute_backward,     [:ptr],                   :int
  ffi_func :tnn_tensor_grad,          [:ptr, :ptr],             :ptr
  ffi_func :tnn_get_rows,         [:ptr, :ptr, :ptr],       :ptr
  ffi_func :tnn_get_rows_back,    [:ptr, :ptr, :ptr, :ptr], :ptr
  ffi_func :tnn_input_1d_i32,     [:ptr, :int],             :ptr
  ffi_func :tnn_gelu_back_scratch,[:ptr, :int],             :void
  ffi_func :tnn_set_output,       [:ptr],                   :void
  ffi_func :tnn_set_param,        [:ptr],                   :void
  ffi_func :tnn_input_1d_f32,     [:ptr, :int],             :ptr
  ffi_func :tnn_input_2d_f32_persistent, [:ptr, :int, :int],   :ptr
  ffi_func :tnn_input_2d_persistent_typed, [:ptr, :int, :int, :int], :ptr
  # Phase 2 BYO-pointer: attach an mmap'd region as the source of
  # weight bytes, then allocate persistent tensors that point at
  # offsets inside it (no copy).
  ffi_func :tnn_session_attach_weight_mmap, [:ptr, :ptr, :size_t], :int
  ffi_func :tnn_input_2d_persistent_mmap, [:ptr, :int, :int, :int, :size_t], :ptr
  ffi_func :tnn_input_1d_persistent_mmap, [:ptr, :int, :int, :size_t], :ptr
  ffi_func :tnn_input_1d_f32_persistent, [:ptr, :int],         :ptr
  ffi_func :tnn_finalize_weights, [:ptr],                   :int
  ffi_func :tnn_zero_tensor,      [:ptr, :ptr],             :int
  ffi_func :tnn_realize_b,        [:ptr, :ptr],             :int
  ffi_func :tnn_switch_a,         [:ptr],                   :int
  ffi_func :tnn_switch_b,         [:ptr],                   :int
  ffi_func :tnn_compute_b,        [:ptr],                   :int
  ffi_func :tnn_opt_step_adamw,   [:ptr, :ptr, :ptr, :ptr, :ptr, :ptr], :ptr
  ffi_func :tnn_opt_step_sgd,     [:ptr, :ptr, :ptr, :ptr],             :ptr
  ffi_func :tnn_adam_step_scratch,[:ptr, :int, :double, :double, :double, :double, :double, :double], :void

  # GGUF model-file loader (S5). Path-based; pass nil-friendly via tnn_gguf_load_empty
  # when you just want to validate the binding works.
  ffi_func :tnn_gguf_load,                  [:str],           :ptr
  ffi_func :tnn_gguf_load_empty,            [],               :ptr
  ffi_func :tnn_gguf_free,                  [:ptr],           :void
  ffi_func :tnn_gguf_n_tensors,             [:ptr],           :int
  ffi_func :tnn_gguf_tensor_name,           [:ptr, :int],     :str
  ffi_func :tnn_gguf_tensor_ne,             [:ptr, :int, :int], :int
  ffi_func :tnn_gguf_tensor_type,           [:ptr, :int],     :int
  ffi_func :tnn_gguf_tensor_nbytes,         [:ptr, :int],     :size_t
  ffi_func :tnn_gguf_read_f32_to_doubles,   [:ptr, :int, :float_array, :size_t], :int
  ffi_func :tnn_gguf_tensor_is_quantized,   [:ptr, :int],     :int
  ffi_func :tnn_gguf_get_u32,               [:ptr, :str],     :int
  ffi_func :tnn_gguf_get_f32,               [:ptr, :str],     :double
  ffi_func :tnn_gguf_get_bool,              [:ptr, :str],     :int
  ffi_func :tnn_gguf_get_str,               [:ptr, :str],     :str
  ffi_func :tnn_gguf_arr_n,                 [:ptr, :str],     :int
  ffi_func :tnn_gguf_arr_str,               [:ptr, :str, :int], :str
  ffi_func :tnn_gguf_mmap_base,             [:ptr],           :ptr
  ffi_func :tnn_gguf_mmap_size,             [:ptr],           :size_t
  ffi_func :tnn_gguf_tensor_file_offset,    [:ptr, :int],     :size_t
  ffi_func :tnn_gguf_write_demo_file,       [:str],           :int
  # Direct GGUF→FFI persistent loaders (see tinynn/tinynn_gguf.h).
  # These skip the Ruby Float64 Mat intermediate, dropping per-weight
  # cost from 12 B to 4 B and unblocking 7B inference. See
  # docs/memory-design.md.
  ffi_func :tnn_gguf_find_index,                    [:ptr, :str],           :int
  # Filesystem walk for model discovery (lib/model_index.rb).
  ffi_func :tnn_list_ggufs,                         [:str],                 :str
  ffi_func :tnn_file_size,                          [:str],                 :size_t
  ffi_func :tnn_gguf_copy_to_persistent,            [:ptr, :int, :ptr, :ptr], :int
  ffi_func :tnn_gguf_copy_1d_to_persistent,         [:ptr, :int, :ptr, :ptr], :int
  ffi_func :tnn_gguf_copy_transposed_to_persistent,
           [:ptr, :int, :ptr, :ptr, :int, :int], :int
  ffi_func :tnn_gguf_copy_head_slice_to_persistent,
           [:ptr, :int, :ptr, :ptr, :int, :int, :int, :int], :int
  ffi_func :tnn_gguf_copy_head_slice_to_persistent_native,
           [:ptr, :int, :ptr, :ptr, :int, :int, :int, :int], :int
  ffi_func :tnn_gguf_copy_verbatim_to_persistent,
           [:ptr, :int, :ptr, :ptr], :int
  ffi_func :tnn_gguf_copy_verbatim_head_slice_to_persistent,
           [:ptr, :int, :ptr, :ptr, :int, :int], :int
  ffi_func :tnn_gguf_copy_head_bias_slice_to_persistent,
           [:ptr, :int, :ptr, :ptr, :int, :int], :int
  ffi_func :tnn_scratch_set_i32,  [:ptr, :int, :int],       :void
  ffi_func :tnn_scratch_get_i32,  [:ptr, :int],             :int
  ffi_func :tnn_realize,          [:ptr, :ptr],             :int
  ffi_func :tnn_build_forward_only, [:ptr, :ptr],           :int
  ffi_func :tnn_cross_entropy_loss, [:ptr, :ptr, :ptr],     :ptr
  ffi_func :tnn_pin_all_graph_b_nodes, [:ptr],              :int
  ffi_func :tnn_graph_reset_grads_only, [:ptr],             :int
  ffi_func :tnn_add_to_graph,     [:ptr, :ptr],             :int
  ffi_func :tnn_compute,          [:ptr],                   :int
  ffi_func :tnn_scratch_set,      [:ptr, :int, :double],    :void
  ffi_func :tnn_scratch_get,      [:ptr, :int],             :double
  ffi_func :tnn_upload,           [:ptr, :ptr],             :int
  ffi_func :tnn_download,         [:ptr, :ptr],             :int
  # Zero-copy bulk upload via Spinel's :float_array / :int_array specs
  # (matz/spinel#474). Replaces the per-element tnn_scratch_set loops.
  ffi_func :tnn_upload_from_float_array, [:ptr, :ptr, :float_array, :size_t], :int
  ffi_func :tnn_upload_from_int_array,   [:ptr, :ptr, :int_array,   :size_t], :int
  # Mirror of tnn_upload_from_float_array. Chunked f32→f64 download
  # into a Ruby Array<Float>'s storage; works for tensors larger than
  # the 16 MiB scratch. Used by `SmolLM2KVFFICache#read_persistent_mat`
  # so weights loaded via the direct GGUF→FFI path stay inspectable.
  ffi_func :tnn_download_to_f64_array,   [:ptr, :ptr, :float_array, :size_t], :int
  # Chunked transpose+upload — supersedes the per-element
  # `stage_transposed_and_upload` Ruby loop for tensors > scratch
  # (4M f32 slots). See tnn_upload_transposed_f64 in tinynn_ggml.c.
  ffi_func :tnn_upload_transposed_f64,   [:ptr, :ptr, :float_array, :int, :int], :int
  ffi_func :tnn_tensor_ne0,       [:ptr],                   :int
  ffi_func :tnn_tensor_ne1,       [:ptr],                   :int
  ffi_func :tnn_tensor_nelements, [:ptr],                   :int

  # Scratch stats — fast O(n) reductions in C over the bytes just
  # placed in sess->scratch by tnn_download. Used by trace-tap.
  ffi_func :tnn_scratch_min_f32,        [:ptr, :int], :double
  ffi_func :tnn_scratch_max_f32,        [:ptr, :int], :double
  ffi_func :tnn_scratch_sum_abs_f32,    [:ptr, :int], :double
  ffi_func :tnn_scratch_nan_count_f32,  [:ptr, :int], :int

  # a ** b where both are project Mats (row-major f64). Returns a Mat
  # (rows = a.nrows, cols = b.ncols).
  #
  # Implementation note: ggml_mul_mat computes A ** B^T. To get A ** B we
  # upload b TRANSPOSED *** b is (br x bc) row-major; we present it to
  # ggml as a (bc x br) tensor whose rows are b's columns. Then ggml's
  # A ** B^T = A ** B (because the "B^T" inside ggml lines up with the
  # original b shape).
  def self.matmul(a, b)
    sess = TinyNN.tnn_session_new(0)   # 0 = CPU

    ta = TinyNN.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    # ggml-side tensor for b^T: rows=b.ncols, cols=b.nrows.
    tb_t = TinyNN.tnn_input_2d_f32(sess, b.ncols, b.nrows)
    tc = TinyNN.tnn_matmul(sess, ta, tb_t)
    TinyNN.tnn_realize(sess, tc)

    # Upload a (row-major flat).
    i = 0
    na = a.nrows * a.ncols
    while i < na
      TinyNN.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, ta)

    # Upload b TRANSPOSED into scratch: scratch[j*b.nrows + i] = b[i,j].
    bc = b.ncols
    br = b.nrows
    i = 0
    while i < br
      j = 0
      while j < bc
        TinyNN.tnn_scratch_set(sess, j * br + i, b.flat[i * bc + j])
        j = j + 1
      end
      i = i + 1
    end
    TinyNN.tnn_upload(sess, tb_t)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    # Result tensor ggml shape: ne0=m=a.nrows, ne1=n=b.ncols. Read into
    # row-major Mat[i][j] (= flat[i*ncols+j]) from scratch[j*m + i].
    out = Mat.new(a.nrows, b.ncols)
    m = a.nrows
    n = b.ncols
    i = 0
    while i < m
      j = 0
      while j < n
        out.flat[i * n + j] = TinyNN.tnn_scratch_get(sess, j * m + i)
        j = j + 1
      end
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end

  # Element-wise a + b. Both Mats must have the same shape.
  def self.add(a, b)
    sess = TinyNN.tnn_session_new(0)
    ta = TinyNN.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tb = TinyNN.tnn_input_2d_f32(sess, b.nrows, b.ncols)
    tc = TinyNN.tnn_add(sess, ta, tb)
    TinyNN.tnn_realize(sess, tc)

    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, ta)

    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, b.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, tb)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    # Result is row-major same shape as a (ne0=cols, ne1=rows, flat
    # is row-major already since ggml_add preserves layout).
    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end

  # Element-wise GeLU (tanh approximation, matches project's feed_forward).
  def self.gelu(a)
    sess = TinyNN.tnn_session_new(0)
    ta = TinyNN.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tc = TinyNN.tnn_gelu(sess, ta)
    TinyNN.tnn_realize(sess, tc)

    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, ta)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end

  # Element-wise SiLU (x * sigmoid(x)), llama-family activation.
  # One-shot wrapper (slow per-call: session + graph + free); used by
  # ab_smoke_silu and as a building block. The persistent-session FFN
  # path doesn't go through this — it builds silu into a fused graph.
  def self.silu(a)
    sess = TinyNN.tnn_session_new(0)
    ta = TinyNN.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tc = TinyNN.tnn_silu(sess, ta)
    TinyNN.tnn_realize(sess, tc)

    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, ta)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end

  # Element-wise multiply c = a * b. Matching shape required.
  # One-shot wrapper. Used in SwiGLU between silu(gate) and up.
  def self.mul(a, b)
    sess = TinyNN.tnn_session_new(0)
    ta = TinyNN.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tb = TinyNN.tnn_input_2d_f32(sess, b.nrows, b.ncols)
    tc = TinyNN.tnn_mul(sess, ta, tb)
    TinyNN.tnn_realize(sess, tc)

    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, ta)
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, b.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, tb)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end

  # RMSNorm(x) * gamma. x is (T, d_model), gamma is Array<Float> of
  # length d_model. eps defaults to 1e-5 (matches the project's
  # rms_norm helper).
  def self.rms_norm(x, gamma, eps)
    sess = TinyNN.tnn_session_new(0)
    tx = TinyNN.tnn_input_2d_f32(sess, x.nrows, x.ncols)
    # gamma as a 1-row tensor: shape (1, d_model). ggml will broadcast
    # across x's leading dimension during the mul.
    tg = TinyNN.tnn_input_2d_f32(sess, 1, x.ncols)
    tc = TinyNN.tnn_rms_norm(sess, tx, tg, eps)
    TinyNN.tnn_realize(sess, tc)

    # Upload x.
    nx = x.nrows * x.ncols
    i = 0
    while i < nx
      TinyNN.tnn_scratch_set(sess, i, x.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, tx)

    # Upload gamma (length d_model).
    i = 0
    while i < x.ncols
      TinyNN.tnn_scratch_set(sess, i, gamma[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, tg)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    out = Mat.new(x.nrows, x.ncols)
    i = 0
    while i < nx
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end

  # Per-row softmax. Matches the project's softmax_rows! (out-of-place).
  def self.softmax(a)
    sess = TinyNN.tnn_session_new(0)
    ta = TinyNN.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tc = TinyNN.tnn_softmax(sess, ta)
    TinyNN.tnn_realize(sess, tc)

    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, ta)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end

  # Transpose. Returns a Mat with rows/cols swapped.
  def self.transpose(a)
    sess = TinyNN.tnn_session_new(0)
    ta = TinyNN.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tc = TinyNN.tnn_transpose(sess, ta)
    TinyNN.tnn_realize(sess, tc)

    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, ta)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    # Result shape: (a.ncols, a.nrows) *** rows and cols swapped.
    # ggml stores it contiguous after ggml_cont; row-major readout is
    # straightforward since the transposed tensor's ne0/ne1 already
    # match the target Mat's cols/rows.
    out = Mat.new(a.ncols, a.nrows)
    rin  = a.nrows
    cin  = a.ncols
    i = 0
    while i < cin
      j = 0
      while j < rin
        out.flat[i * rin + j] = TinyNN.tnn_scratch_get(sess, i * rin + j)
        j = j + 1
      end
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end

  # ----------------------------------------------------------------------
  # Persistent-session API: build a graph once, run it many times.
  #
  # Workflow:
  #   sess = TinyNN.persistent_new(0)
  #   ta   = TinyNN.alloc_2d(sess, rows, cols)
  #   tb   = TinyNN.alloc_2d(sess, rows, cols)
  #   tc   = TinyNN.build_matmul(sess, ta, tb)   # or build_add / build_gelu / ...
  #   TinyNN.realize(sess, tc)                    # allocates all backend buffers
  #   # Upload weights once:
  #   TinyNN.upload_row_major(sess, tb, w_mat)
  #   # Per training step:
  #   loop do
  #     TinyNN.upload_row_major(sess, ta, input_mat)
  #     TinyNN.compute(sess)
  #     result = TinyNN.download_matmul(sess, tc, m, n)    # transposed readback
  #   end
  #   TinyNN.persistent_free(sess)
  #
  # The win over the one-shot wrappers (TinyNN.matmul etc.) is that
  # ggml_init / ggml_backend_sched_alloc_graph runs once instead of per
  # op, and backend buffers (the cuda-side storage for tensors) are
  # allocated once instead of per call. At the toy LM's transformer
  # shapes (see ab_smoke_big), this should flip CUDA from losing to
  # native at small shapes.

  def self.persistent_new(prefer_cuda)
    TinyNN.tnn_session_new(prefer_cuda)
  end

  def self.persistent_free(sess)
    TinyNN.tnn_session_free(sess)
  end

  def self.alloc_2d(sess, rows, cols)
    TinyNN.tnn_input_2d_f32(sess, rows, cols)
  end

  def self.alloc_1d_i32(sess, n)
    TinyNN.tnn_input_1d_i32(sess, n)
  end

  def self.build_matmul(sess, ta, tb)
    TinyNN.tnn_matmul(sess, ta, tb)
  end

  def self.build_add(sess, ta, tb)
    TinyNN.tnn_add(sess, ta, tb)
  end

  def self.build_gelu(sess, ta)
    TinyNN.tnn_gelu(sess, ta)
  end

  def self.build_softmax(sess, ta)
    TinyNN.tnn_softmax(sess, ta)
  end

  def self.build_scale(sess, ta, s)
    TinyNN.tnn_scale(sess, ta, s)
  end

  def self.build_rms_norm(sess, tx, tgamma, eps)
    TinyNN.tnn_rms_norm(sess, tx, tgamma, eps)
  end

  def self.realize(sess, result)
    TinyNN.tnn_realize(sess, result)
  end

  def self.compute(sess)
    TinyNN.tnn_compute(sess)
  end

  # Stage a Mat row-major into scratch and upload to `tensor`. Use for
  # elementwise inputs or for matmul's A operand. For matmul's B we
  # also have upload_transposed below.
  #
  # Uses Spinel's :float_array spec (matz/spinel#474) for zero-copy
  # transfer of mat.flat — single FFI call replaces O(n) per-element
  # tnn_scratch_set loop.
  def self.upload_row_major(sess, tensor, mat)
    TinyNN.tnn_upload_from_float_array(sess, tensor, mat.flat, mat.nrows * mat.ncols)
  end

  # Upload an Array<Int> to a 1D int32 tensor in one FFI call.
  # Uses Spinel's :int_array spec (matz/spinel#474).
  def self.upload_int_array(sess, tensor, indices)
    TinyNN.tnn_upload_from_int_array(sess, tensor, indices, indices.length)
  end

  # Stage a Mat TRANSPOSED into scratch and upload. Use this for the
  # `b` operand of build_matmul to get logical A*B semantics (ggml's
  # mul_mat is A*B^T natively).
  def self.upload_transposed(sess, tensor, mat)
    br = mat.nrows
    bc = mat.ncols
    i = 0
    while i < br
      j = 0
      while j < bc
        TinyNN.tnn_scratch_set(sess, j * br + i, mat.flat[i * bc + j])
        j = j + 1
      end
      i = i + 1
    end
    TinyNN.tnn_upload(sess, tensor)
  end

  # Download a tensor whose data is row-major (output of elementwise
  # ops like add, gelu, rms_norm, softmax, scale).
  #
  # Param name `dl_handle` (not `tensor`) intentionally — Spinel
  # unifies param-name types across the whole program, and `tensor`
  # collides with a dead `upload_transposed` definition whose param
  # got mistyped as mrb_int. Result: download_row_major's tensor arg
  # gets boxed at call sites and the (void *) cast inside fails.
  def self.download_row_major(sess, dl_handle, rows, cols)
    TinyNN.tnn_download(sess, dl_handle)
    out = Mat.new(rows, cols)
    n = rows * cols
    i = 0
    while i < n
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end
    out
  end

  # Chunked Mat-roundtrip for large tensors. Unlike download_row_major
  # this bypasses the 16 MiB scratch (via tnn_download_to_f64_array's
  # internal chunking) and so works on weight-sized tensors loaded via
  # the direct GGUF→FFI path. Mirrors `upload_row_major`.
  #
  # Use this when you want a Mat copy of a persistent FFI tensor —
  # inspection, Mat-side fine-tuning, export. For small graph
  # intermediates (norms / per-step logits) the scratch-based
  # download_row_major is fine and slightly faster.
  def self.download_to_mat(sess, dl_handle, rows, cols)
    out = Mat.new(rows, cols)
    TinyNN.tnn_download_to_f64_array(sess, dl_handle, out.flat, rows * cols)
    out
  end

  # Download a matmul result. ggml's mul_mat result has ne0=m, ne1=n;
  # reading row-major (rows=m, cols=n) means scratch[j*m + i].
  def self.download_matmul(sess, tensor, m, n)
    TinyNN.tnn_download(sess, tensor)
    out = Mat.new(m, n)
    i = 0
    while i < m
      j = 0
      while j < n
        out.flat[i * n + j] = TinyNN.tnn_scratch_get(sess, j * m + i)
        j = j + 1
      end
      i = i + 1
    end
    out
  end

  # Internal: stage b TRANSPOSED into scratch, then bulk-upload to `target`.
  # The C side does both the transpose and a chunked upload so the call
  # works for tensors larger than the 16 MiB scratch buffer
  # (Qwen2.5-0.5B's ffn_gate is 17.4 MB; the old per-element + single
  # bulk-upload path silently truncated at the 4M-float boundary,
  # leaving the tail uninitialised and producing 1e+37 magnitudes
  # in the subsequent matmul output).
  def self.stage_transposed_and_upload(sess, target, b)
    TinyNN.tnn_upload_transposed_f64(sess, target, b.flat, b.nrows, b.ncols)
  end

  # Internal: stage `m` row-major into scratch, then bulk-upload to `target`.
  def self.stage_row_major_and_upload(sess, target, m)
    n = m.nrows * m.ncols
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, m.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, target)
  end

  # d/dx of plain RMSNorm(x) given dy (= grad of normalized output).
  # No gamma — caller is responsible for the gamma part of the chain rule.
  #
  # Note on arg order: ggml's header says "a - x, b - dy" but the CPU
  # source (ggml-cpu/ops.cpp ggml_compute_forward_rms_norm_back_f32)
  # treats src0 as gradients and src1 as the forward input. We pass
  # (dy, x) to match the source.
  def self.rms_norm_back(x, dy, eps)
    sess = TinyNN.tnn_session_new(0)
    tdy = TinyNN.tnn_input_2d_f32(sess, dy.nrows, dy.ncols)
    tx  = TinyNN.tnn_input_2d_f32(sess, x.nrows, x.ncols)
    tc  = TinyNN.tnn_rms_norm_back(sess, tdy, tx, eps)
    TinyNN.tnn_realize(sess, tc)
    TinyNN.stage_row_major_and_upload(sess, tdy, dy)
    TinyNN.stage_row_major_and_upload(sess, tx,  x)
    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)
    out = Mat.new(x.nrows, x.ncols)
    n = x.nrows * x.ncols
    i = 0
    while i < n
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNN.tnn_session_free(sess)
    out
  end

  # GeLU backward: dx = dh * d/dx GeLU(x) (tanh approx).
  # Skips ggml entirely — uses tnn_gelu_back_scratch which operates
  # on the session's scratch buffer directly. CPU-only.
  def self.gelu_back(x, dh)
    sess = TinyNN.tnn_session_new(0)
    n = x.nrows * x.ncols
    # Stage x at [0..n), dh at [n..2n)
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, x.flat[i])
      i = i + 1
    end
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, n + i, dh.flat[i])
      i = i + 1
    end
    TinyNN.tnn_gelu_back_scratch(sess, n)
    out = Mat.new(x.nrows, x.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNN.tnn_scratch_get(sess, 2 * n + i)
      i = i + 1
    end
    TinyNN.tnn_session_free(sess)
    out
  end

  # Fused softmax-cross-entropy gradient:
  #   dlogits[i, v] = (softmax(logits)[i, v] - one_hot(targets[i])[v]) / n_pred
  #
  # Composable from existing ops:
  #   sm  = softmax(logits)
  #   oh  = one_hot mat (built on the Ruby side; cheap — n_pred sets)
  #   dlg = (sm - oh) / n_pred = scale(sm, 1/n_pred) + scale(oh, -1/n_pred)
  #
  # `logits` is (n_pred, vocab); `targets` is Array<Int> of length n_pred
  # where targets[i] in [0, vocab) is the desired class at row i.
  def self.cross_entropy_grad(logits, targets, n_pred)
    # 1. one-hot in Ruby.
    oh = Mat.new(logits.nrows, logits.ncols)
    i = 0
    while i < n_pred
      oh.flat[i * logits.ncols + targets[i]] = 1.0
      i = i + 1
    end
    # 2. softmax + scale + scale + add through FFI.
    sm = TinyNN.softmax(logits)
    inv_n = 1.0 / n_pred.to_f
    sm_s  = TinyNN.scale(sm, inv_n)
    oh_s  = TinyNN.scale(oh, -inv_n)
    TinyNN.add(sm_s, oh_s)
  end

  # Adam optimizer step. Matches the project's adam_step_mat.
  #
  # Returns three new Mats: [param_new, m_new, v_new]. Caller is
  # responsible for swapping them back into wherever they came from
  # (no persistent storage yet — once persistent sessions are wired
  # into transformer.rb, m/v can stay on-device).
  #
  # omc1, omc2 are pre-computed bias-correction divisors:
  #   omc1 = 1 - beta1^t,  omc2 = 1 - beta2^t
  # where t is the step number. (The project tracks them as running
  # products in AdamState.bc1 / bc2; both conventions work.)
  def self.adam_step(param, grad, m, v, lr, b1, b2, eps, omc1, omc2)
    sess = TinyNN.tnn_session_new(0)
    n = param.nrows * param.ncols
    # Stage param at [0..n), grad at [n..2n), m at [2n..3n), v at [3n..4n).
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, param.flat[i])
      i = i + 1
    end
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, n + i, grad.flat[i])
      i = i + 1
    end
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, 2 * n + i, m.flat[i])
      i = i + 1
    end
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, 3 * n + i, v.flat[i])
      i = i + 1
    end

    TinyNN.tnn_adam_step_scratch(sess, n, lr, b1, b2, eps, omc1, omc2)

    new_param = Mat.new(param.nrows, param.ncols)
    new_mom_m = Mat.new(param.nrows, param.ncols)
    new_mom_v = Mat.new(param.nrows, param.ncols)
    i = 0
    while i < n
      new_param.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      new_mom_m.flat[i] = TinyNN.tnn_scratch_get(sess, 2 * n + i)
      new_mom_v.flat[i] = TinyNN.tnn_scratch_get(sess, 3 * n + i)
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    AdamStepResult.new(new_param, new_mom_m, new_mom_v)
  end

  # SGD parameter update: param_new = param - lr * grad.
  # Returns a fresh Mat with the updated parameter (caller is
  # responsible for swapping it back into wherever param came from —
  # we don't have persistent-session storage yet).
  #
  # Composed from TinyNN.add and TinyNN.scale rather than ggml_opt_step_sgd
  # (which would need an sgd_params tensor with (alpha, weight_decay)).
  # Faster path is a single fused op; this version is the cleanest one
  # with the primitives we already have.
  def self.sgd_step(param, grad, lr)
    TinyNN.add(param, TinyNN.scale(grad, -lr))
  end

  # Embedding lookup: gather table rows by indices.
  # `table` is (vocab, d_model) Mat; `indices` is Array<Int>.
  # Returns (indices.length, d_model) Mat with table[indices[i]] in row i.
  def self.embed_lookup(table, indices)
    n_idx = indices.length
    sess  = TinyNN.tnn_session_new(0)
    ttab  = TinyNN.tnn_input_2d_f32(sess, table.nrows, table.ncols)
    tidx  = TinyNN.tnn_input_1d_i32(sess, n_idx)
    tout  = TinyNN.tnn_get_rows(sess, ttab, tidx)
    TinyNN.tnn_realize(sess, tout)

    TinyNN.stage_row_major_and_upload(sess, ttab, table)

    i = 0
    while i < n_idx
      TinyNN.tnn_scratch_set_i32(sess, i, indices[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, tidx)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tout)

    out = Mat.new(n_idx, table.ncols)
    n = n_idx * table.ncols
    i = 0
    while i < n
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNN.tnn_session_free(sess)
    out
  end

  # Embedding backward: scatter-add d_out rows into a vocab-sized table.
  # `d_out` is (n_idx, d_model). `indices` is Array<Int>. Returns
  # (vocab_size, d_model) Mat where out[indices[i]] += d_out[i].
  def self.embed_back(d_out, indices, vocab_size)
    n_idx = indices.length
    sess  = TinyNN.tnn_session_new(0)
    td    = TinyNN.tnn_input_2d_f32(sess, d_out.nrows, d_out.ncols)
    tidx  = TinyNN.tnn_input_1d_i32(sess, n_idx)
    # Shape reference for the result: a freshly-allocated (vocab, d) tensor.
    tshape = TinyNN.tnn_input_2d_f32(sess, vocab_size, d_out.ncols)
    tout  = TinyNN.tnn_get_rows_back(sess, td, tidx, tshape)
    TinyNN.tnn_realize(sess, tout)

    TinyNN.stage_row_major_and_upload(sess, td, d_out)

    i = 0
    while i < n_idx
      TinyNN.tnn_scratch_set_i32(sess, i, indices[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, tidx)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tout)

    out = Mat.new(vocab_size, d_out.ncols)
    n = vocab_size * d_out.ncols
    i = 0
    while i < n
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNN.tnn_session_free(sess)
    out
  end

  # Backward for SiLU: given x (the input to silu) and dy (gradient
  # from upstream), returns dx. dx = dy * (sigmoid(x) * (1 + x * (1 - sigmoid(x)))).
  def self.silu_back(x, dy)
    sess = TinyNN.tnn_session_new(0)
    tx  = TinyNN.tnn_input_2d_f32(sess, x.nrows, x.ncols)
    tdy = TinyNN.tnn_input_2d_f32(sess, dy.nrows, dy.ncols)
    tc  = TinyNN.tnn_silu_back(sess, tx, tdy)
    TinyNN.tnn_realize(sess, tc)
    TinyNN.stage_row_major_and_upload(sess, tx,  x)
    TinyNN.stage_row_major_and_upload(sess, tdy, dy)
    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)
    out = Mat.new(x.nrows, x.ncols)
    n = x.nrows * x.ncols
    i = 0
    while i < n
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNN.tnn_session_free(sess)
    out
  end

  # d/dx of per-row softmax. `a_softmax` is the softmax output;
  # `dy` is grad of output. (ggml source: src0=dy, src1=y_softmax.)
  def self.softmax_back(a_softmax, dy)
    sess = TinyNN.tnn_session_new(0)
    tdy = TinyNN.tnn_input_2d_f32(sess, dy.nrows, dy.ncols)
    ta  = TinyNN.tnn_input_2d_f32(sess, a_softmax.nrows, a_softmax.ncols)
    tc  = TinyNN.tnn_softmax_back(sess, tdy, ta)
    TinyNN.tnn_realize(sess, tc)
    TinyNN.stage_row_major_and_upload(sess, tdy, dy)
    TinyNN.stage_row_major_and_upload(sess, ta,  a_softmax)
    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)
    out = Mat.new(a_softmax.nrows, a_softmax.ncols)
    n = a_softmax.nrows * a_softmax.ncols
    i = 0
    while i < n
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end
    TinyNN.tnn_session_free(sess)
    out
  end

  # FFN-shaped chain: result = gelu(h * w1) * w2.
  #
  # Calls three op-sized sessions, each reusing the cached engine (the
  # backend + scheduler init runs once, not three times). One ggml-graph
  # chaining is theoretically possible but needs explicit intermediate
  # transposes because mul_mat's result has ne0 swapped relative to the
  # next op's k-dim. Sticking to three sessions until we have a clean
  # chain-friendly layout convention.
  def self.ffn_pipeline(h, w1, w2)
    pre    = TinyNN.matmul(h, w1)
    hidden = TinyNN.gelu(pre)
    TinyNN.matmul(hidden, w2)
  end

  # a * b^T natively (matches Mat#matmul_t). Faster than .matmul(b) for the
  # same shapes because there's no Ruby-side transpose of b on upload.
  def self.matmul_t(a, b)
    sess = TinyNN.tnn_session_new(0)
    ta = TinyNN.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tb = TinyNN.tnn_input_2d_f32(sess, b.nrows, b.ncols)
    tc = TinyNN.tnn_matmul(sess, ta, tb)
    TinyNN.tnn_realize(sess, tc)

    TinyNN.stage_row_major_and_upload(sess, ta, a)
    TinyNN.stage_row_major_and_upload(sess, tb, b)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    out = Mat.new(a.nrows, b.nrows)
    m = a.nrows
    n = b.nrows
    i = 0
    while i < m
      j = 0
      while j < n
        out.flat[i * n + j] = TinyNN.tnn_scratch_get(sess, j * m + i)
        j = j + 1
      end
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end

  # a^T * b (matches Mat#t_matmul). Both inputs uploaded transposed so
  # ggml's ne0 lines up with the summed-over K dimension.
  def self.t_matmul(a, b)
    sess = TinyNN.tnn_session_new(0)
    # Both tensors created as their transposed shape:
    #   ta_t: ne0=a.nrows (=K), ne1=a.ncols (=M)
    #   tb_t: ne0=b.nrows (=K), ne1=b.ncols (=N)
    ta_t = TinyNN.tnn_input_2d_f32(sess, a.ncols, a.nrows)
    tb_t = TinyNN.tnn_input_2d_f32(sess, b.ncols, b.nrows)
    tc = TinyNN.tnn_matmul(sess, ta_t, tb_t)
    TinyNN.tnn_realize(sess, tc)

    TinyNN.stage_transposed_and_upload(sess, ta_t, a)
    TinyNN.stage_transposed_and_upload(sess, tb_t, b)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    out = Mat.new(a.ncols, b.ncols)
    m = a.ncols
    n = b.ncols
    i = 0
    while i < m
      j = 0
      while j < n
        out.flat[i * n + j] = TinyNN.tnn_scratch_get(sess, j * m + i)
        j = j + 1
      end
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end

  # Element-wise a * s for scalar s. Returns a new Mat (out-of-place).
  def self.scale(a, s)
    sess = TinyNN.tnn_session_new(0)
    ta = TinyNN.tnn_input_2d_f32(sess, a.nrows, a.ncols)
    tc = TinyNN.tnn_scale(sess, ta, s)
    TinyNN.tnn_realize(sess, tc)

    n = a.nrows * a.ncols
    i = 0
    while i < n
      TinyNN.tnn_scratch_set(sess, i, a.flat[i])
      i = i + 1
    end
    TinyNN.tnn_upload(sess, ta)

    TinyNN.tnn_compute(sess)
    TinyNN.tnn_download(sess, tc)

    out = Mat.new(a.nrows, a.ncols)
    i = 0
    while i < n
      out.flat[i] = TinyNN.tnn_scratch_get(sess, i)
      i = i + 1
    end

    TinyNN.tnn_session_free(sess)
    out
  end
end



# Spinel anchor block: library-style methods (upload_int_array,
# embed_lookup) with no reachable caller default their `indices`
# param to `mrb_int` (or, with Spinel post-6b2ae3b body-usage
# inference, to a bitset — produces `(x >> i) & 1` for `x[i]`).
# Pin them to Array<Int> with concrete callsites in a guarded block.
if false
  _ai_sess   = TinyNN.tnn_null_ptr
  _ai_tensor = TinyNN.tnn_null_ptr
  _ai_ids    = [0]
  _ai_table  = Mat.new(1, 1)
  TinyNN.upload_int_array(_ai_sess, _ai_tensor, _ai_ids)
  TinyNN.embed_lookup(_ai_table, _ai_ids)
  # Pin `indices` (literal name) as Array<Int> so embed_back's
  # `indices[i]` returns Int. Spinel post-6b2ae3b body-usage inference
  # would otherwise infer it as RbVal-boxed and emit invalid `((int)(_t))`
  # against an aggregate. The local-name collapse means the anchor's
  # var MUST be named `indices` — `_ai_ids` doesn't unify with it.
  indices = [0]
  _ai_d   = Mat.new(1, 1)
  TinyNN.embed_back(_ai_d, indices, 1)
end
