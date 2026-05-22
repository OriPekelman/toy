# lib/toy_smollm2_loader.rb — GGUF weight load + config reader for Toy::SmolLM2.
#
# Kept separate from lib/gguf_load.rb so a demo that only uses
# GPT-2 doesn't have to pull SmolLM2 types into Spinel's compile graph.

require_relative "gguf_load"
require_relative "toy_smollm2"

module GGUFLoad
  # Llama-family weight load into a Toy::SmolLM2.
  #
  # Tensor name conventions match prep/convert_smollm2_to_gguf.py.
  # The converter has already transposed every nn.Linear weight from
  # HF's [out, in] to our [in, out] orientation.
  def self.load_toy_smollm2(model, path)
    handle = TinyNN.tnn_gguf_load(path)
    if handle == nil
      puts "open failed: " + path
      return false
    end
    n_tensors = TinyNN.tnn_gguf_n_tensors(handle)
    puts "loading " + path + " (" + n_tensors.to_s + " tensors)"

    cfg     = model.cfg
    d_model = cfg.d_model
    n_heads = cfg.n_heads
    n_kv    = cfg.n_kv
    d_head  = d_model / n_heads

    read_mat(handle,   "token_embd.weight",  model.token_embed.weight, n_tensors)
    read_array(handle, "output_norm.weight", model.final_norm.gamma,   n_tensors)

    # Untied output (`output.weight`) is present for TinyLlama / Llama-2
    # but not for SmolLM2 / Qwen2.5. Detect via tensor presence; the
    # converter omits it for tied models.
    output_idx = find_index(handle, "output.weight", n_tensors)
    if output_idx >= 0
      puts "  untied output: output.weight present"
      model.enable_untied_output!
      read_mat(handle, "output.weight", model.output_proj, n_tensors)
    end

    # Q/K/V biases are a Qwen2.x trait (Llama / SmolLM2 / TinyLlama lack
    # them). Detect via attn_q.bias in block 0; the converter writes all
    # three when any are present in the HF safetensors.
    qkv_bias_idx = find_index(handle, "blk.0.attn_q.bias", n_tensors)
    has_qkv_bias = qkv_bias_idx >= 0
    if has_qkv_bias
      puts "  Q/K/V biases present (Qwen2.x-style)"
    end

    li = 0
    while li < cfg.n_layers
      blk    = model.stack[li]
      prefix = "blk." + li.to_s

      read_array(handle, prefix + ".attn_norm.weight", blk.rn1.gamma, n_tensors)
      read_array(handle, prefix + ".ffn_norm.weight",  blk.rn2.gamma, n_tensors)

      # Q: full [d_model, n_heads * d_head] = [d_model, d_model]
      read_split_heads_weight(handle, prefix + ".attn_q.weight",
                               blk.attn.w_q, n_heads, d_model, d_head, n_tensors)
      # K, V: narrower [d_model, n_kv * d_head] — uses the GQA reader.
      read_split_kv_weight(handle, prefix + ".attn_k.weight",
                            blk.attn.w_k, n_kv, d_model, d_head, n_tensors)
      read_split_kv_weight(handle, prefix + ".attn_v.weight",
                            blk.attn.w_v, n_kv, d_model, d_head, n_tensors)
      read_mat(handle,   prefix + ".attn_output.weight", blk.attn.w_o, n_tensors)

      if has_qkv_bias
        # Q bias: [n_heads * d_head] split into per-Q-head arrays.
        read_split_heads_bias(handle, prefix + ".attn_q.bias",
                               blk.attn.b_q, n_heads, d_head, n_tensors)
        # K/V biases: [n_kv * d_head] split into per-KV-head arrays.
        read_split_kv_bias(handle, prefix + ".attn_k.bias",
                            blk.attn.b_k, n_kv, d_head, n_tensors)
        read_split_kv_bias(handle, prefix + ".attn_v.bias",
                            blk.attn.b_v, n_kv, d_head, n_tensors)
        blk.attn.enable_qkv_bias!
      end

      read_mat(handle,   prefix + ".ffn_gate.weight", blk.ffn.w_gate, n_tensors)
      read_mat(handle,   prefix + ".ffn_up.weight",   blk.ffn.w_up,   n_tensors)
      read_mat(handle,   prefix + ".ffn_down.weight", blk.ffn.w_down, n_tensors)

      li = li + 1
    end

    TinyNN.tnn_gguf_free(handle)
    true
  end
end

# Read llama-family hyperparameters from a GGUF's kv metadata. Mirrors
# GPT2ConfigLoader but for `llama.*` keys (set by convert_smollm2_to_gguf.py).
module GGUFLoad
  # Detect llama-family GGUF capability flags without instantiating the
  # Ruby Mat-backed model. Returns has_untied_output + has_qkv_bias so
  # the caller can build the FFI cache directly.
  class SmolLM2Flags
    attr_accessor :untied, :qkv_bias
    def initialize(untied, qkv_bias)
      @untied   = untied
      @qkv_bias = qkv_bias
    end
  end

  def self.detect_smollm2_flags(path)
    handle = TinyNN.tnn_gguf_load(path)
    if handle == nil
      return SmolLM2Flags.new(false, false)
    end
    untied   = TinyNN.tnn_gguf_find_index(handle, "output.weight")       >= 0
    qkv_bias = TinyNN.tnn_gguf_find_index(handle, "blk.0.attn_q.bias")   >= 0
    TinyNN.tnn_gguf_free(handle)
    SmolLM2Flags.new(untied, qkv_bias)
  end

  # Inference-only loader: stream GGUF weights directly into the FFI
  # persistent buffers, skipping the Ruby Float64 Mat allocation. 4 B/w
  # vs the Mat-mediated 12 B/w; required for 7B-class models.
  #
  # The kv_cache MUST already be realized via realize_for. We do not
  # construct Toy::SmolLM2 at all — callers that need `describe` /
  # `algorithm_card` should still use the Mat-mediated path on a
  # 1×1-stub config.
  def self.load_kv_cache_directly(kv_cache, path)
    handle = TinyNN.tnn_gguf_load(path)
    if handle == nil
      puts "open failed: " + path
      return false
    end
    n_tensors = TinyNN.tnn_gguf_n_tensors(handle)
    puts "loading " + path + " → FFI direct (" + n_tensors.to_s + " tensors)"

    sess     = kv_cache.sess
    n_heads  = kv_cache.n_heads
    n_kv     = kv_cache.n_kv
    d_model  = kv_cache.d_model
    d_head   = kv_cache.d_head
    d_ff     = kv_cache.d_ff

    # --- Globals -----
    embed_idx = TinyNN.tnn_gguf_find_index(handle, "token_embd.weight")
    TinyNN.tnn_gguf_copy_to_persistent(handle, embed_idx,
                                        sess, kv_cache.t_token_embed)

    fn_idx = TinyNN.tnn_gguf_find_index(handle, "output_norm.weight")
    TinyNN.tnn_gguf_copy_1d_to_persistent(handle, fn_idx,
                                           sess, kv_cache.t_final_norm_gamma)

    if kv_cache.has_untied_output
      out_idx = TinyNN.tnn_gguf_find_index(handle, "output.weight")
      TinyNN.tnn_gguf_copy_to_persistent(handle, out_idx,
                                          sess, kv_cache.t_output)
    end

    # --- Per-block -----
    li = 0
    while li < kv_cache.n_layers
      blk_f  = kv_cache.kv_blocks_ffi[li]
      prefix = "blk." + li.to_s

      rn1_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".attn_norm.weight")
      rn2_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".ffn_norm.weight")
      TinyNN.tnn_gguf_copy_1d_to_persistent(handle, rn1_idx, sess, blk_f.t_rn1_gamma)
      TinyNN.tnn_gguf_copy_1d_to_persistent(handle, rn2_idx, sess, blk_f.t_rn2_gamma)

      # Q (n_heads per-head slices of attn_q.weight)
      q_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".attn_q.weight")
      hq = 0
      while hq < n_heads
        TinyNN.tnn_gguf_copy_head_slice_to_persistent(handle, q_idx, sess,
                                                       blk_f.t_w_q[hq],
                                                       hq, n_heads, d_model, d_head)
        hq = hq + 1
      end

      # K, V (n_kv per-head slices each)
      k_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".attn_k.weight")
      v_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".attn_v.weight")
      hkv = 0
      while hkv < n_kv
        TinyNN.tnn_gguf_copy_head_slice_to_persistent(handle, k_idx, sess,
                                                       blk_f.t_w_k[hkv],
                                                       hkv, n_kv, d_model, d_head)
        TinyNN.tnn_gguf_copy_head_slice_to_persistent(handle, v_idx, sess,
                                                       blk_f.t_w_v[hkv],
                                                       hkv, n_kv, d_model, d_head)
        hkv = hkv + 1
      end

      # Optional Q/K/V biases (Qwen2.x)
      if kv_cache.has_qkv_bias
        qb_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".attn_q.bias")
        kb_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".attn_k.bias")
        vb_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".attn_v.bias")
        hq = 0
        while hq < n_heads
          TinyNN.tnn_gguf_copy_head_bias_slice_to_persistent(handle, qb_idx, sess,
                                                              blk_f.t_b_q[hq], hq, d_head)
          hq = hq + 1
        end
        hkv = 0
        while hkv < n_kv
          TinyNN.tnn_gguf_copy_head_bias_slice_to_persistent(handle, kb_idx, sess,
                                                              blk_f.t_b_k[hkv], hkv, d_head)
          TinyNN.tnn_gguf_copy_head_bias_slice_to_persistent(handle, vb_idx, sess,
                                                              blk_f.t_b_v[hkv], hkv, d_head)
          hkv = hkv + 1
        end
      end

      # O (attn_output.weight) — single transposed
      o_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".attn_output.weight")
      TinyNN.tnn_gguf_copy_transposed_to_persistent(handle, o_idx, sess,
                                                     blk_f.t_w_o, d_model, d_model)

      # FFN — gate, up, down (each single transposed)
      gate_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".ffn_gate.weight")
      up_idx   = TinyNN.tnn_gguf_find_index(handle, prefix + ".ffn_up.weight")
      down_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".ffn_down.weight")
      TinyNN.tnn_gguf_copy_transposed_to_persistent(handle, gate_idx, sess,
                                                     blk_f.t_w_gate, d_model, d_ff)
      TinyNN.tnn_gguf_copy_transposed_to_persistent(handle, up_idx,   sess,
                                                     blk_f.t_w_up,   d_model, d_ff)
      TinyNN.tnn_gguf_copy_transposed_to_persistent(handle, down_idx, sess,
                                                     blk_f.t_w_down, d_ff, d_model)

      li = li + 1
    end

    # Zero-init K/V buffers (matches the Mat-mediated path's kv_zero_*
    # uploads — without this the persistent K/V tensors contain
    # garbage from the backend's initial allocation).
    kv_zero_k = Mat.new(kv_cache.max_T, d_head)
    kv_zero_v = Mat.new(d_head, kv_cache.max_T)
    li = 0
    while li < kv_cache.n_layers
      blk_f = kv_cache.kv_blocks_ffi[li]
      hkv = 0
      while hkv < n_kv
        TinyNN.upload_row_major(sess, blk_f.t_K[hkv], kv_zero_k)
        TinyNN.upload_row_major(sess, blk_f.t_V[hkv], kv_zero_v)
        hkv = hkv + 1
      end
      li = li + 1
    end

    TinyNN.tnn_gguf_free(handle)
    true
  end

  # Native-layout direct loader. Same shape as load_kv_cache_directly
  # but the source GGUF was written with --ggml-native — 2D linear
  # weights are stored in HF-native [out, in] row-major, which already
  # matches ggml's column-major ne=[in, out] byte order. All transposes
  # are gone; per-head Q/K/V slices are contiguous byte ranges.
  #
  # See [[project_mmap_phase1_2026_05_18]] / docs/memory-design.md for
  # the rationale.
  def self.load_kv_cache_directly_native(kv_cache, path)
    handle = TinyNN.tnn_gguf_load(path)
    if handle == nil
      puts "open failed: " + path
      return false
    end
    n_tensors = TinyNN.tnn_gguf_n_tensors(handle)
    puts "loading " + path + " → FFI direct (native, " + n_tensors.to_s + " tensors)"

    sess     = kv_cache.sess
    n_heads  = kv_cache.n_heads
    n_kv     = kv_cache.n_kv
    d_model  = kv_cache.d_model
    d_head   = kv_cache.d_head
    d_ff     = kv_cache.d_ff

    # Globals (token_embd, output_norm, optional untied output) — these
    # were already non-transposed even in the old converter; loader is
    # identical to the legacy path.
    embed_idx = TinyNN.tnn_gguf_find_index(handle, "token_embd.weight")
    TinyNN.tnn_gguf_copy_to_persistent(handle, embed_idx,
                                        sess, kv_cache.t_token_embed)

    fn_idx = TinyNN.tnn_gguf_find_index(handle, "output_norm.weight")
    TinyNN.tnn_gguf_copy_1d_to_persistent(handle, fn_idx,
                                           sess, kv_cache.t_final_norm_gamma)

    if kv_cache.has_untied_output
      out_idx = TinyNN.tnn_gguf_find_index(handle, "output.weight")
      if kv_cache.weight_type != 0
        TinyNN.tnn_gguf_copy_verbatim_to_persistent(handle, out_idx,
                                                     sess, kv_cache.t_output)
      else
        TinyNN.tnn_gguf_copy_to_persistent(handle, out_idx,
                                            sess, kv_cache.t_output)
      end
    end

    li = 0
    while li < kv_cache.n_layers
      blk_f  = kv_cache.kv_blocks_ffi[li]
      prefix = "blk." + li.to_s

      rn1_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".attn_norm.weight")
      rn2_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".ffn_norm.weight")
      TinyNN.tnn_gguf_copy_1d_to_persistent(handle, rn1_idx, sess, blk_f.t_rn1_gamma)
      TinyNN.tnn_gguf_copy_1d_to_persistent(handle, rn2_idx, sess, blk_f.t_rn2_gamma)

      # Per-head Q/K/V — native layout: contiguous byte range. When the
      # cache is in Q8 mode (Phase 3) we use the verbatim head-slice
      # helper, which is type-agnostic and just memcpys the right
      # contiguous range. For F32 mode the f32 helper does the same
      # plus a dequant fallback (in case the GGUF is Q8 but the cache
      # is F32 — old behavior).
      use_verbatim = kv_cache.weight_type != 0
      q_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".attn_q.weight")
      hq = 0
      while hq < n_heads
        if use_verbatim
          TinyNN.tnn_gguf_copy_verbatim_head_slice_to_persistent(handle, q_idx, sess,
                                                                  blk_f.t_w_q[hq],
                                                                  hq, n_heads)
        else
          TinyNN.tnn_gguf_copy_head_slice_to_persistent_native(handle, q_idx, sess,
                                                                blk_f.t_w_q[hq],
                                                                hq, n_heads, d_model, d_head)
        end
        hq = hq + 1
      end

      k_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".attn_k.weight")
      v_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".attn_v.weight")
      hkv = 0
      while hkv < n_kv
        if use_verbatim
          TinyNN.tnn_gguf_copy_verbatim_head_slice_to_persistent(handle, k_idx, sess,
                                                                  blk_f.t_w_k[hkv], hkv, n_kv)
          TinyNN.tnn_gguf_copy_verbatim_head_slice_to_persistent(handle, v_idx, sess,
                                                                  blk_f.t_w_v[hkv], hkv, n_kv)
        else
          TinyNN.tnn_gguf_copy_head_slice_to_persistent_native(handle, k_idx, sess,
                                                                blk_f.t_w_k[hkv],
                                                                hkv, n_kv, d_model, d_head)
          TinyNN.tnn_gguf_copy_head_slice_to_persistent_native(handle, v_idx, sess,
                                                                blk_f.t_w_v[hkv],
                                                                hkv, n_kv, d_model, d_head)
        end
        hkv = hkv + 1
      end

      # Q/K/V biases: 1-D, identical loader (biases were already
      # take()'d untransposed even in the legacy converter).
      if kv_cache.has_qkv_bias
        qb_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".attn_q.bias")
        kb_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".attn_k.bias")
        vb_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".attn_v.bias")
        hq = 0
        while hq < n_heads
          TinyNN.tnn_gguf_copy_head_bias_slice_to_persistent(handle, qb_idx, sess,
                                                              blk_f.t_b_q[hq], hq, d_head)
          hq = hq + 1
        end
        hkv = 0
        while hkv < n_kv
          TinyNN.tnn_gguf_copy_head_bias_slice_to_persistent(handle, kb_idx, sess,
                                                              blk_f.t_b_k[hkv], hkv, d_head)
          TinyNN.tnn_gguf_copy_head_bias_slice_to_persistent(handle, vb_idx, sess,
                                                              blk_f.t_b_v[hkv], hkv, d_head)
          hkv = hkv + 1
        end
      end

      # O / FFN gate / up / down — native: plain memcpy. Q8 mode
      # uses the verbatim primitive (same shape; type-preserving).
      o_idx    = TinyNN.tnn_gguf_find_index(handle, prefix + ".attn_output.weight")
      gate_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".ffn_gate.weight")
      up_idx   = TinyNN.tnn_gguf_find_index(handle, prefix + ".ffn_up.weight")
      down_idx = TinyNN.tnn_gguf_find_index(handle, prefix + ".ffn_down.weight")
      if use_verbatim
        TinyNN.tnn_gguf_copy_verbatim_to_persistent(handle, o_idx,    sess, blk_f.t_w_o)
        TinyNN.tnn_gguf_copy_verbatim_to_persistent(handle, gate_idx, sess, blk_f.t_w_gate)
        TinyNN.tnn_gguf_copy_verbatim_to_persistent(handle, up_idx,   sess, blk_f.t_w_up)
        TinyNN.tnn_gguf_copy_verbatim_to_persistent(handle, down_idx, sess, blk_f.t_w_down)
      else
        TinyNN.tnn_gguf_copy_to_persistent(handle, o_idx,    sess, blk_f.t_w_o)
        TinyNN.tnn_gguf_copy_to_persistent(handle, gate_idx, sess, blk_f.t_w_gate)
        TinyNN.tnn_gguf_copy_to_persistent(handle, up_idx,   sess, blk_f.t_w_up)
        TinyNN.tnn_gguf_copy_to_persistent(handle, down_idx, sess, blk_f.t_w_down)
      end

      li = li + 1
    end

    # Zero-init K/V buffers (same as the legacy path).
    kv_zero_k = Mat.new(kv_cache.max_T, d_head)
    kv_zero_v = Mat.new(d_head, kv_cache.max_T)
    li = 0
    while li < kv_cache.n_layers
      blk_f = kv_cache.kv_blocks_ffi[li]
      hkv = 0
      while hkv < n_kv
        TinyNN.upload_row_major(sess, blk_f.t_K[hkv], kv_zero_k)
        TinyNN.upload_row_major(sess, blk_f.t_V[hkv], kv_zero_v)
        hkv = hkv + 1
      end
      li = li + 1
    end

    TinyNN.tnn_gguf_free(handle)
    true
  end

  # Detect the GGUF's 2D linear weight type. Peeks at
  # blk.0.attn_q.weight (always present for llama-family models).
  # Returns the ggml type integer (0=F32, 8=Q8_0). Callers should
  # pass this to kv.set_weight_type before kv.realize_for to enable
  # the Q8-stays-Q8 path.
  def self.detect_weight_type(path)
    handle = TinyNN.tnn_gguf_load(path)
    if handle == nil
      return 0
    end
    idx = TinyNN.tnn_gguf_find_index(handle, "blk.0.attn_q.weight")
    t   = if idx >= 0
            TinyNN.tnn_gguf_tensor_type(handle, idx)
          else
            0
          end
    TinyNN.tnn_gguf_free(handle)
    t
  end

  # Auto-dispatcher: peek at the toy.ggml_native metadata key and pick
  # the matching loader. Keeps callers ignorant of the file layout.
  def self.load_kv_cache_auto(kv_cache, path)
    handle = TinyNN.tnn_gguf_load(path)
    if handle == nil
      puts "open failed: " + path
      return false
    end
    is_native = TinyNN.tnn_gguf_get_bool(handle, "toy.ggml_native") == 1
    TinyNN.tnn_gguf_free(handle)
    if is_native
      load_kv_cache_directly_native(kv_cache, path)
    else
      load_kv_cache_directly(kv_cache, path)
    end
  end
end

module SmolLM2ConfigLoader
  # Read rope_scaling.* metadata from the GGUF and pick a RopeScaling
  # variant. GGUF keys (when present):
  #   llama.rope.scaling.type          (str: "linear", "yarn", "llama3", "longrope")
  #   llama.rope.scaling.factor        (f32)
  #   llama.rope.scaling.original_context_length        (u32)
  #   llama.rope.scaling.low_freq_factor                (f32, llama3)
  #   llama.rope.scaling.high_freq_factor               (f32, llama3)
  #   llama.rope.scaling.attn_factor   (f32, yarn)
  #   llama.rope.scaling.beta_fast     (f32, yarn)
  #   llama.rope.scaling.beta_slow     (f32, yarn)
  #
  # Returns Toy::RopeScaling.none when the type key is missing (gguf
  # accessors return -1/0 sentinels). LongRoPE not yet supported —
  # falls back to .none with a warning.
  def self.read_rope_scaling(handle)
    kind_str = TinyNN.tnn_gguf_get_str(handle, "llama.rope.scaling.type")
    if kind_str == nil
      return Toy::RopeScaling.none
    end
    if kind_str == "linear"
      f = TinyNN.tnn_gguf_get_f32(handle, "llama.rope.scaling.factor")
      puts "rope_scaling: linear (factor=" + f.to_s + ")"
      return Toy::RopeScaling.linear(f)
    end
    if kind_str == "llama3"
      f    = TinyNN.tnn_gguf_get_f32(handle, "llama.rope.scaling.factor")
      lo   = TinyNN.tnn_gguf_get_f32(handle, "llama.rope.scaling.low_freq_factor")
      hi   = TinyNN.tnn_gguf_get_f32(handle, "llama.rope.scaling.high_freq_factor")
      omp  = TinyNN.tnn_gguf_get_u32(handle, "llama.rope.scaling.original_context_length")
      if omp < 0
        omp = TinyNN.tnn_gguf_get_u32(handle, "llama.context_length")
      end
      puts "rope_scaling: llama3 (factor=" + f.to_s +
           " low=" + lo.to_s + " high=" + hi.to_s +
           " orig_max=" + omp.to_s + ")"
      return Toy::RopeScaling.llama3(f, lo, hi, omp)
    end
    if kind_str == "yarn"
      f   = TinyNN.tnn_gguf_get_f32(handle, "llama.rope.scaling.factor")
      omp = TinyNN.tnn_gguf_get_u32(handle, "llama.rope.scaling.original_context_length")
      af  = TinyNN.tnn_gguf_get_f32(handle, "llama.rope.scaling.attn_factor")
      bf  = TinyNN.tnn_gguf_get_f32(handle, "llama.rope.scaling.beta_fast")
      bs  = TinyNN.tnn_gguf_get_f32(handle, "llama.rope.scaling.beta_slow")
      if af == 0.0; af = 1.0; end
      if bf == 0.0; bf = 32.0; end
      if bs == 0.0; bs = 1.0; end
      puts "rope_scaling: yarn (factor=" + f.to_s + " orig_max=" + omp.to_s +
           " attn_factor=" + af.to_s + ")"
      return Toy::RopeScaling.new(:yarn, 1.0 / f, omp, f, 1.0, 4.0, 1.0, af, bf, bs)
    end
    # longrope and any unknown kind: fall back, with a warning.
    puts "rope_scaling: unsupported type '" + kind_str + "' — using no-scaling (likely degraded long-context quality)"
    Toy::RopeScaling.none
  end

  def self.read(path)
    handle = TinyNN.tnn_gguf_load(path)
    if handle == nil
      puts "SmolLM2ConfigLoader: failed to open " + path
      return Toy::SmolLM2Config.new(0, 0, 0, 0, 0, 0, 0, 10000.0, 1.0e-5)
    end
    vocab     = TinyNN.tnn_gguf_get_u32(handle, "llama.vocab_size")
    d_model   = TinyNN.tnn_gguf_get_u32(handle, "llama.embedding_length")
    d_ff      = TinyNN.tnn_gguf_get_u32(handle, "llama.feed_forward_length")
    n_head    = TinyNN.tnn_gguf_get_u32(handle, "llama.attention.head_count")
    n_kv      = TinyNN.tnn_gguf_get_u32(handle, "llama.attention.head_count_kv")
    n_layer   = TinyNN.tnn_gguf_get_u32(handle, "llama.block_count")
    ctx       = TinyNN.tnn_gguf_get_u32(handle, "llama.context_length")
    rope_base = TinyNN.tnn_gguf_get_f32(handle, "llama.rope.freq_base")
    rms_eps   = TinyNN.tnn_gguf_get_f32(handle, "llama.attention.layer_norm_rms_epsilon")
    scaling   = read_rope_scaling(handle)
    TinyNN.tnn_gguf_free(handle)
    cfg = Toy::SmolLM2Config.new(vocab, d_model, n_head, n_kv, d_ff, n_layer,
                                 ctx, rope_base, rms_eps)
    cfg.rope_scaling = scaling
    cfg
  end
end
