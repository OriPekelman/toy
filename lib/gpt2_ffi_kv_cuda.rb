# GPT2KVFFICacheCuda — CUDA mirror of `lib/gpt2_ffi_kv.rb`.
#
# This file is mechanically the same as gpt2_ffi_kv.rb modulo:
#   * `TinyNN` → `TinyNNCuda` (CUDA FFI module)
#   * `GPT2KVFFICache` → `GPT2KVFFICacheCuda` (Spinel collapses same-named
#     classes across modules; the suffix prevents that)
#   * `require_relative "tinynn"` → `..."tinynn_cuda"`
#
# Phase 0 replaces both files with one generic `TransformerLM` taking
# a backend flag (PyTorch-style); see `docs/design/arch-struct.md`.
# Until then: any change to gpt2_ffi_kv.rb MUST be mirrored here, or
# the CPU and CUDA paths will silently diverge (the Qwen V-matmul flip
# is the canonical example of this drift — see
# `docs/design/arch-struct.md` and the cpy-strided incident).

require_relative "transformer"
require_relative "gpt2"
require_relative "tinynn_cuda"

# Per-block persistent tensor handles. Includes per-head K/V buffers
# alongside the standard GPT-2 weights.
class GPT2KVBlockFFICuda
  attr_accessor :t_ln1_gamma, :t_ln1_beta, :t_ln2_gamma, :t_ln2_beta,
                :t_w_q, :t_w_k, :t_w_v,
                :t_b_q, :t_b_k, :t_b_v,
                :t_w_o, :t_b_o,
                :t_w_ff1, :t_b_ff1,
                :t_w_ff2, :t_b_ff2,
                :t_K, :t_V

  def initialize
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
    @t_K   = [TinyNNCuda.tnn_null_ptr]
    @t_V   = [TinyNNCuda.tnn_null_ptr]
    @t_w_o   = TinyNNCuda.tnn_null_ptr
    @t_b_o   = TinyNNCuda.tnn_null_ptr
    @t_w_ff1 = TinyNNCuda.tnn_null_ptr
    @t_b_ff1 = TinyNNCuda.tnn_null_ptr
    @t_w_ff2 = TinyNNCuda.tnn_null_ptr
    @t_b_ff2 = TinyNNCuda.tnn_null_ptr
  end
end

class GPT2KVFFICacheCuda
  attr_accessor :sess, :t_token_embed, :t_pos_embed,
                :t_ln_f_gamma, :t_ln_f_beta,
                :kv_blocks_ffi,
                :max_T, :d_model, :d_ff, :n_heads, :d_head, :n_layers,
                :vocab_size, :context_length, :realized

  def initialize
    @realized   = false
    @max_T      = 0
    @d_model    = 0
    @d_ff       = 0
    @n_heads    = 0
    @d_head     = 0
    @n_layers   = 0
    @vocab_size = 0
    @context_length = 0
    @sess          = TinyNNCuda.tnn_null_ptr
    @t_token_embed = TinyNNCuda.tnn_null_ptr
    @t_pos_embed   = TinyNNCuda.tnn_null_ptr
    @t_ln_f_gamma  = TinyNNCuda.tnn_null_ptr
    @t_ln_f_beta   = TinyNNCuda.tnn_null_ptr
    @kv_blocks_ffi = [GPT2KVBlockFFICuda.new]
  end

  # Declare all persistent tensors (weights + K/V buffers) and finalize
  # the backend buffer. After this, weights can be uploaded; compute
  # graphs are built per decode step via build_decode_step.
  def realize_for(max_T, d_model, d_ff, n_heads, n_layers,
                  vocab_size, context_length)
    @max_T          = max_T
    @d_model        = d_model
    @d_ff           = d_ff
    @n_heads        = n_heads
    @d_head         = d_model / n_heads
    @n_layers       = n_layers
    @vocab_size     = vocab_size
    @context_length = context_length

    @sess = TinyNNCuda.tnn_session_new(1)

    @t_token_embed = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, vocab_size, d_model)
    @t_pos_embed   = TinyNNCuda.tnn_input_2d_f32_persistent(@sess, context_length, d_model)
    @t_ln_f_gamma  = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)
    @t_ln_f_beta   = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)

    @kv_blocks_ffi = [GPT2KVBlockFFICuda.new]
    li = 1
    while li < n_layers
      @kv_blocks_ffi.push(GPT2KVBlockFFICuda.new)
      li = li + 1
    end

    li = 0
    while li < n_layers
      blk = @kv_blocks_ffi[li]
      blk.t_ln1_gamma = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)
      blk.t_ln1_beta  = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)
      blk.t_ln2_gamma = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)
      blk.t_ln2_beta  = TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_model)

      # Per-head: weights, biases, and KV buffers.
      blk.t_w_q = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model)]
      blk.t_w_k = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model)]
      blk.t_w_v = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model)]
      blk.t_b_q = [TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_head)]
      blk.t_b_k = [TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_head)]
      blk.t_b_v = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, 1)]   # ne=[1, d_head]
      # K: ne=[d_head, max_T]; V: ne=[max_T, d_head] (transposed layout).
      blk.t_K   = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, max_T,  d_head)]
      blk.t_V   = [TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, max_T)]
      h = 1
      while h < n_heads
        blk.t_w_q.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model))
        blk.t_w_k.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model))
        blk.t_w_v.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, d_model))
        blk.t_b_q.push(TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_head))
        blk.t_b_k.push(TinyNNCuda.tnn_input_1d_f32_persistent(@sess, d_head))
        blk.t_b_v.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, 1))
        blk.t_K.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, max_T,  d_head))
        blk.t_V.push(TinyNNCuda.tnn_input_2d_f32_persistent(@sess, d_head, max_T))
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
    @realized = true
  end

  # Build the compute graph for one decode position. Returns the logits
  # tensor handle. Caller calls tnn_compute then download_row_major.
  def build_decode_step(pos)
    eps   = 1.0e-5
    scale = 1.0 / Math.sqrt(@d_head.to_f)
    d_model = @d_model
    d_head  = @d_head
    max_T   = @max_T
    bytes_d_head    = d_head * 4
    bytes_d_model   = d_model * 4
    bytes_max_T     = max_T * 4

    # Single-token input.
    t_token_id = TinyNNCuda.tnn_input_1d_i32(@sess, 1)

    # x = embed[token_id] + pos_embed[pos]
    t_embed_row = TinyNNCuda.tnn_get_rows(@sess, @t_token_embed, t_token_id)  # ne=[d_model, 1]
    t_pos_row   = TinyNNCuda.tnn_view_2d(@sess, @t_pos_embed,
                                      d_model, 1, bytes_d_model,
                                      pos * bytes_d_model)
    t_x = TinyNNCuda.tnn_add(@sess, t_embed_row, t_pos_row)

    li = 0
    while li < @n_layers
      t_x = build_block_step(t_x, @kv_blocks_ffi[li], pos, scale, eps,
                              bytes_d_head, bytes_max_T)
      li = li + 1
    end

    t_x_final = TinyNNCuda.tnn_layer_norm(@sess, t_x, @t_ln_f_gamma, @t_ln_f_beta, eps)
    t_kv_logits = TinyNNCuda.tnn_matmul(@sess, @t_token_embed, t_x_final)  # ne=[vocab, 1]
    TinyNNCuda.tnn_set_output(t_kv_logits)
    GPT2KVStepResultCuda.new(t_token_id, t_kv_logits)
  end

  def build_block_step(t_x, blk, pos, scale, eps, bytes_d_head, bytes_max_T)
    t_h = TinyNNCuda.tnn_layer_norm(@sess, t_x, blk.t_ln1_gamma, blk.t_ln1_beta, eps)

    t_head_out0 = build_attention_head_step(t_h, blk, 0, pos, scale,
                                             bytes_d_head, bytes_max_T)
    t_head_outs = [t_head_out0]
    h = 1
    while h < @n_heads
      t_head_outs.push(build_attention_head_step(t_h, blk, h, pos, scale,
                                                  bytes_d_head, bytes_max_T))
      h = h + 1
    end

    t_concat = t_head_outs[0]
    h = 1
    while h < @n_heads
      t_concat = TinyNNCuda.tnn_concat(@sess, t_concat, t_head_outs[h], 0)
      h = h + 1
    end

    t_out_proj_raw = TinyNNCuda.tnn_matmul(@sess, blk.t_w_o, t_concat)
    t_out_proj     = TinyNNCuda.tnn_add(@sess, t_out_proj_raw, blk.t_b_o)
    t_x_attn       = TinyNNCuda.tnn_add(@sess, t_x, t_out_proj)

    t_h2     = TinyNNCuda.tnn_layer_norm(@sess, t_x_attn, blk.t_ln2_gamma, blk.t_ln2_beta, eps)
    t_up_raw = TinyNNCuda.tnn_matmul(@sess, blk.t_w_ff1, t_h2)
    t_up     = TinyNNCuda.tnn_add(@sess, t_up_raw, blk.t_b_ff1)
    t_g      = TinyNNCuda.tnn_gelu(@sess, t_up)
    t_dn_raw = TinyNNCuda.tnn_matmul(@sess, blk.t_w_ff2, t_g)
    t_dn     = TinyNNCuda.tnn_add(@sess, t_dn_raw, blk.t_b_ff2)

    TinyNNCuda.tnn_add(@sess, t_x_attn, t_dn)
  end

  def build_attention_head_step(t_h, blk, head_idx, pos, scale,
                                 bytes_d_head, bytes_max_T)
    # q_new, k_new, v_new for the single new position.
    t_q_raw = TinyNNCuda.tnn_matmul(@sess, blk.t_w_q[head_idx], t_h)   # ne=[d_head, 1]
    t_q     = TinyNNCuda.tnn_add(@sess, t_q_raw, blk.t_b_q[head_idx])
    t_k_raw = TinyNNCuda.tnn_matmul(@sess, blk.t_w_k[head_idx], t_h)
    t_k_new = TinyNNCuda.tnn_add(@sess, t_k_raw, blk.t_b_k[head_idx])
    t_v_raw = TinyNNCuda.tnn_matmul(@sess, t_h, blk.t_w_v[head_idx])   # ne=[1, d_head]
    t_v_new = TinyNNCuda.tnn_add(@sess, t_v_raw, blk.t_b_v[head_idx])

    # Write k_new → K[pos], v_new → V[:, pos] via cpy-into-view.
    t_K_slot = TinyNNCuda.tnn_view_2d(@sess, blk.t_K[head_idx],
                                    @d_head, 1, bytes_d_head, pos * bytes_d_head)
    t_cpy_k  = TinyNNCuda.tnn_cpy(@sess, t_k_new, t_K_slot)
    t_V_slot = TinyNNCuda.tnn_view_2d(@sess, blk.t_V[head_idx],
                                    1, @d_head, bytes_max_T, pos * 4)
    t_cpy_v  = TinyNNCuda.tnn_cpy(@sess, t_v_new, t_V_slot)
    # The cpy tensors aren't reachable from head_out; force them into
    # the graph so the scheduler runs them before the attn matmuls
    # read K/V history.
    TinyNNCuda.tnn_add_to_graph(@sess, t_cpy_k)
    TinyNNCuda.tnn_add_to_graph(@sess, t_cpy_v)

    # Attention over K[0:pos+1] / V[0:pos+1].
    t_K_hist = TinyNNCuda.tnn_view_2d(@sess, blk.t_K[head_idx],
                                    @d_head, pos + 1, bytes_d_head, 0)
    t_V_hist = TinyNNCuda.tnn_view_2d(@sess, blk.t_V[head_idx],
                                    pos + 1, @d_head, bytes_max_T, 0)

    t_scores = TinyNNCuda.tnn_matmul(@sess, t_K_hist, t_q)        # ne=[pos+1, 1]
    t_scaled = TinyNNCuda.tnn_scale(@sess, t_scores, scale)
    # No causal mask: K_hist already covers only valid past positions.
    t_attn   = TinyNNCuda.tnn_softmax(@sess, t_scaled)
    TinyNNCuda.tnn_matmul(@sess, t_V_hist, t_attn)                 # ne=[d_head, 1]
  end
end

# Single-step graph artefacts. Returned by build_decode_step so the
# caller can upload the token_id input and download the logits.
#
# Init-param names deliberately differ from the ivar names to dodge
# Spinel's whole-program local-name collapse (e.g. `t_logits` as a
# param could unify with `t_logits` locals elsewhere and box the
# slot as sp_RbVal — breaks the (void *) cast in download_row_major).
class GPT2KVStepResultCuda
  attr_accessor :t_token_id, :kv_step_logits
  def initialize(tok_ptr, logits_ptr)
    @t_token_id     = tok_ptr
    @kv_step_logits = logits_ptr
  end
end

module GPT2KVCuda
  # Upload all GPT-2 weights (+ zero-init the K/V buffers) into a
  # realized GPT2KVFFICacheCuda. Counterpart to GPT2FFICuda.upload_from for
  # the KV cache variant. Note: pos_embed is uploaded in FULL (all
  # context_length rows), not sliced.
  def self.upload_from(kv_cache, model)
    sess    = kv_cache.sess
    n       = kv_cache.n_layers
    n_heads = kv_cache.n_heads
    d_model = kv_cache.d_model
    d_head  = kv_cache.d_head
    max_T   = kv_cache.max_T

    TinyNNCuda.upload_row_major(sess, kv_cache.t_token_embed, model.token_embed)
    TinyNNCuda.upload_row_major(sess, kv_cache.t_pos_embed,   model.pos_embed)
    TinyNNCuda.tnn_upload_from_float_array(sess, kv_cache.t_ln_f_gamma, model.ln_f_gamma, d_model)
    TinyNNCuda.tnn_upload_from_float_array(sess, kv_cache.t_ln_f_beta,  model.ln_f_beta,  d_model)

    # Zero buffers for K/V (ggml_backend_alloc_ctx_tensors typically
    # zeros, but be explicit so reuse across multiple decode runs has
    # a clean starting state).
    kv_zero_k = Mat.new(max_T,  d_head)
    kv_zero_v = Mat.new(d_head, max_T)

    li = 0
    while li < n
      blk_n = model.gpt2_blocks[li]
      blk_f = kv_cache.kv_blocks_ffi[li]

      TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_ln1_gamma, blk_n.ln1_gamma, d_model)
      TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_ln1_beta,  blk_n.ln1_beta,  d_model)
      TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_ln2_gamma, blk_n.ln2_gamma, d_model)
      TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_ln2_beta,  blk_n.ln2_beta,  d_model)

      h = 0
      while h < n_heads
        TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_q[h], blk_n.w_q[h])
        TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_k[h], blk_n.w_k[h])
        TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_v[h], blk_n.w_v[h])
        TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_b_q[h], blk_n.b_q[h], d_head)
        TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_b_k[h], blk_n.b_k[h], d_head)
        TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_b_v[h], blk_n.b_v[h], d_head)
        TinyNNCuda.upload_row_major(sess, blk_f.t_K[h], kv_zero_k)
        TinyNNCuda.upload_row_major(sess, blk_f.t_V[h], kv_zero_v)
        h = h + 1
      end

      TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_o,   blk_n.w_o)
      TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_ff1, blk_n.w_ff1)
      TinyNNCuda.stage_transposed_and_upload(sess, blk_f.t_w_ff2, blk_n.w_ff2)
      TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_b_o,   blk_n.b_o,   d_model)
      TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_b_ff1, blk_n.b_ff1, kv_cache.d_ff)
      TinyNNCuda.tnn_upload_from_float_array(sess, blk_f.t_b_ff2, blk_n.b_ff2, d_model)

      li = li + 1
    end
  end

  # Decode one new token at position `pos`. Writes K[pos], V[:, pos]
  # as a side effect, returns the (vocab,) logits Mat for the new
  # position. The caller can argmax (greedy) or sample.
  def self.decode_step(kv_cache, token_id, pos)
    TinyNNCuda.tnn_reset_for_rebuild(kv_cache.sess)
    step = kv_cache.build_decode_step(pos)
    TinyNNCuda.tnn_realize(kv_cache.sess, step.kv_step_logits)
    TinyNNCuda.upload_int_array(kv_cache.sess, step.t_token_id, [token_id])
    TinyNNCuda.tnn_compute(kv_cache.sess)
    # Logits ne=[vocab, 1]. Download as (1, vocab) row-major — same
    # layout as a single-row Mat with vocab columns.
    TinyNNCuda.download_row_major(kv_cache.sess, step.kv_step_logits, 1, kv_cache.vocab_size)
  end
end
