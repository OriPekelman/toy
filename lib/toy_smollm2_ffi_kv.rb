# lib/toy_smollm2_ffi_kv.rb — Toy::SmolLM2 KV-cache decode via ggml FFI.
#
# Mirror of lib/gpt2_ffi_kv.rb but for the llama-family architecture:
#   - RMSNorm (no beta) instead of LayerNorm
#   - No biases on Q / K / V / O / FFN projections
#   - SwiGLU FFN: down( silu(gate(x)) * up(x) )
#   - RoPE applied to Q and K before the dot product
#   - GQA: K and V are stored per-`n_kv`-head, not per-`n_heads`-head.
#     Each KV head is shared by group_size = n_heads / n_kv query heads.
#
# Per decode step builds a single-position compute graph; K and V at
# the current position are written into persistent per-layer buffers
# via cpy-into-view (same pattern as the GPT-2 cache). Cost per step:
# constant in prompt length.

require_relative "transformer"
require_relative "toy"
require_relative "toy_smollm2"
require_relative "tinynn"
# NOTE: not requiring "toy_smollm2_loader" here. Requiring it from
# inside this file triggers a Spinel GC mark crash in decode_step
# (sp_gc_mark / sp_PtrArray_new_scan) for reasons we haven't fully
# isolated — likely something about require-order interaction with
# Spinel's type inference around GGUFLoad. Callers that use
# realize_and_load_auto (or any method here that references
# GGUFLoad) must `require_relative "toy_smollm2_loader"` from their
# top-level driver file BEFORE this file is loaded. The OpenAI API
# binaries and the realize-mmap demos already do.

# Per-block persistent tensors for the SmolLM2 KV cache.
#
# Q is split per query head (n_heads of them).
# K, V, and their persistent buffers are split per KV head (n_kv of them).
class SmolLM2KVBlockFFI
  attr_accessor :t_rn1_gamma, :t_rn2_gamma,
                :t_w_q, :t_w_k, :t_w_v, :t_w_o,
                :t_b_q, :t_b_k, :t_b_v,
                # M1: per-block QK-norm (Qwen3). RMSNorm on Q and K with
                # a shared [d_head] gamma applied to every head before
                # RoPE. Allocated only when has_qk_norm is set. The
                # null-ptr seed lets graph-builder code branch cleanly.
                :t_q_norm_gamma, :t_k_norm_gamma,
                :t_w_gate, :t_w_up, :t_w_down,
                :t_K, :t_V,
                # F1.2: optional LoRA adapters on Q projection (one
                # rank-R pair per Q head). t_w_lora_a_q[hq] has shape
                # (r, d_model); t_w_lora_b_q[hq] has shape (d_head, r).
                # Allocated only when cache.lora_q_enabled at realize
                # time. Trainable f32 tensors in ctx_w (not mmap'd from
                # GGUF — adapters are session-local).
                :t_w_lora_a_q, :t_w_lora_b_q,
                # F1.2 step 6b: optional persistent Adam moments paired
                # with the LoRA-A/B tensors above. Allocated in ctx_w
                # (NOT compute ctx) so they survive tnn_reset_for_rebuild
                # between multi-position SFT steps. Same shapes as A/B.
                # Allocated only when cache.lora_q_adamw_enabled. The
                # m/v live next to A/B so a future "save adapter +
                # optimizer state" hook can serialize them together.
                :t_w_lora_a_q_m, :t_w_lora_a_q_v,
                :t_w_lora_b_q_m, :t_w_lora_b_q_v

  def initialize
    @t_rn1_gamma = TinyNN.tnn_null_ptr
    @t_rn2_gamma = TinyNN.tnn_null_ptr
    @t_q_norm_gamma = TinyNN.tnn_null_ptr
    @t_k_norm_gamma = TinyNN.tnn_null_ptr
    @t_w_q  = [TinyNN.tnn_null_ptr]
    @t_w_k  = [TinyNN.tnn_null_ptr]
    @t_w_v  = [TinyNN.tnn_null_ptr]
    @t_b_q  = [TinyNN.tnn_null_ptr]   # per-Q-head bias (Qwen2.x)
    @t_b_k  = [TinyNN.tnn_null_ptr]   # per-KV-head bias
    @t_b_v  = [TinyNN.tnn_null_ptr]   # per-KV-head bias (1-D [d_head])
    @t_K    = [TinyNN.tnn_null_ptr]
    @t_V    = [TinyNN.tnn_null_ptr]
    @t_w_o    = TinyNN.tnn_null_ptr
    @t_w_gate = TinyNN.tnn_null_ptr
    @t_w_up   = TinyNN.tnn_null_ptr
    @t_w_down = TinyNN.tnn_null_ptr
    @t_w_lora_a_q = [TinyNN.tnn_null_ptr]
    @t_w_lora_b_q = [TinyNN.tnn_null_ptr]
    @t_w_lora_a_q_m = [TinyNN.tnn_null_ptr]
    @t_w_lora_a_q_v = [TinyNN.tnn_null_ptr]
    @t_w_lora_b_q_m = [TinyNN.tnn_null_ptr]
    @t_w_lora_b_q_v = [TinyNN.tnn_null_ptr]
  end
end

class SmolLM2KVFFICache
  attr_accessor :sess, :t_token_embed, :t_final_norm_gamma,
                :t_output, :has_untied_output, :has_qkv_bias,
                # M1: Qwen3 added per-block QK-norm. When true, the
                # graph builder applies tnn_rms_norm to Q and K with
                # blk.t_q_norm_gamma / blk.t_k_norm_gamma (shape
                # [d_head], shared across heads) BEFORE tnn_rope_ext.
                # Detect by presence of "blk.0.attn_q_norm.weight" in
                # the GGUF. Always false on Qwen2.5 / Llama-family.
                :has_qk_norm,
                :kv_blocks_ffi,
                :max_T, :d_model, :d_ff, :n_heads, :n_kv, :d_head,
                :group_size, :n_layers, :vocab_size, :rope_base,
                :rope_scaling, :t_rope_freq_factors,
                :rms_eps, :realized,
                # CUDA-MIRROR-SKIP-BEGIN: trace-tap is CPU-only diagnostic
                :trace_on, :trace_names, :trace_tensors,
                # CUDA-MIRROR-SKIP-END
                # Phase 3: ggml type for 2D linear weights. Default
                # 0 = GGML_TYPE_F32 (legacy). 8 = GGML_TYPE_Q8_0. Set
                # via #set_weight_type before #realize_for to keep
                # quantized weights quantized in memory.
                :weight_type,
                :gguf_handle_keepalive,
                # F1.2: LoRA on Q projection. enable_lora_q!(r) sets
                # both flags BEFORE realize. When enabled, each block
                # gets per-Q-head trainable A/B adapter pairs spliced
                # into the Q matmul: q_eff = w_q[h]@h + B[h]@A[h]@h.
                :lora_q_enabled, :lora_q_rank,
                # F1.2 step 6b: when true, realize_for_mmap also
                # allocates persistent AdamW moments (m, v) for every
                # LoRA-A/B pair in ctx_w. Required for multi-position
                # SFT: between graph rebuilds the compute ctx is freed,
                # so moments held there would be lost (NaN on cycle 2+).
                :lora_q_adamw_enabled

  def initialize
    @realized   = false
    @max_T      = 0
    @d_model    = 0
    @d_ff       = 0
    @n_heads    = 0
    @n_kv       = 0
    @d_head     = 0
    @group_size = 0
    @n_layers   = 0
    @vocab_size = 0
    @rope_base  = 10000.0
    @rope_scaling        = Toy::RopeScaling.none
    @t_rope_freq_factors = TinyNN.tnn_null_ptr
    @rms_eps    = 1.0e-5
    @sess               = TinyNN.tnn_null_ptr
    @t_token_embed      = TinyNN.tnn_null_ptr
    @t_final_norm_gamma = TinyNN.tnn_null_ptr
    @t_output           = TinyNN.tnn_null_ptr
    @has_untied_output  = false
    @has_qkv_bias       = false
    @has_qk_norm        = false
    @kv_blocks_ffi      = [SmolLM2KVBlockFFI.new]
    # CUDA-MIRROR-SKIP-BEGIN: trace-tap is CPU-only diagnostic
    # --- trace-tap diagnostics (zero cost when off) ---
    # When @trace_on is true, trace_tap() pushes (name, tensor) onto
    # parallel arrays AND calls tnn_set_output so the scheduler keeps
    # the tensor's buffer alive. After tnn_compute, dump_trace() walks
    # the arrays, downloads each, and prints min/max/|mean|/nan stats.
    # When off, trace_tap() is a single bool branch — the graph is
    # unchanged from production.
    @trace_on      = false
    @trace_names   = [""]
    @trace_names.pop
    @trace_tensors = [TinyNN.tnn_null_ptr]
    @trace_tensors.pop
    # CUDA-MIRROR-SKIP-END
    @weight_type   = 0                # GGML_TYPE_F32; legacy default
    @gguf_handle_keepalive = TinyNN.tnn_null_ptr  # set by realize_for_mmap
    @lora_q_enabled = false
    @lora_q_rank    = 0
    @lora_q_adamw_enabled = false
  end

  # F1.2: enable per-Q-head LoRA on this session's forward graph. Call
  # BEFORE realize_for_mmap. Adapter A is (r, d_model), adapter B is
  # (d_head, r); both trainable F32 tensors in ctx_w (not mmap'd, so
  # writes survive). Standard LoRA init: A = small Gaussian, B = 0,
  # which makes the adapter a no-op at step 0 (forward output ==
  # baseline). Use upload_lora_zero!(seed) to set up that init.
  def enable_lora_q!(r)
    @lora_q_enabled = true
    @lora_q_rank    = r
  end

  # F1.2 step 6b: allocate persistent AdamW moments (m, v) alongside
  # each LoRA-A/B pair, in ctx_w. Requires enable_lora_q!(...) to have
  # been called first (so the rank is known). Call BEFORE
  # realize_for_mmap. Without this, multi-position SFT loses Adam
  # state at every graph rebuild and diverges to NaN.
  def enable_lora_q_adamw!
    @lora_q_adamw_enabled = true
  end

  # Phase 3 opt-in: set the ggml type used for 2D linear weights when
  # realize_for runs. 0 = F32, 8 = Q8_0. Call BEFORE realize_for —
  # the persistent tensors are allocated there.
  def set_weight_type(t)
    @weight_type = t
  end

  # Allocate one persistent 2D linear weight tensor at the configured
  # type. Used by realize_for; keeps the Q8/F32 branch in one place.
  # Non-2D-linear tensors (norms, biases, K/V cache, t_output) stay
  # F32 even in Q8 mode — quantizing them costs accuracy with no
  # compute saving.
  def alloc_2d_w(rows, cols)
    if @weight_type == 0
      TinyNN.tnn_input_2d_f32_persistent(@sess, rows, cols)
    else
      TinyNN.tnn_input_2d_persistent_typed(@sess, rows, cols, @weight_type)
    end
  end

  # Phase 2 BYO-pointer realization. Like realize_for but every
  # GGUF-resident tensor (token_embed, norms, biases, all 2D linears,
  # untied output) is allocated to POINT AT the file's mmap'd pages
  # rather than copied into a backend buffer. Only K/V cache and the
  # compute scratch live in backend-allocated memory. The kv_cache
  # holds the GGUF handle so the mmap stays alive for its lifetime.
  #
  # Caller flow:
  #   gguf  = TinyNN.tnn_gguf_load(path)        # mmap'd, no_alloc
  #   flags = GGUFLoad.detect_smollm2_flags(path)
  #   wtype = GGUFLoad.detect_weight_type(path)
  #   kv = SmolLM2KVFFICache.new
  #   kv.realize_for_mmap(gguf, cfg, MAX_T, flags.untied, flags.qkv_bias)
  #   # weights are already in place; no load_weights call needed.
  def realize_for_mmap(gguf_handle, cfg, max_T, untied, qkv_bias, qk_norm)
    @max_T      = max_T
    @d_model    = cfg.d_model
    @d_ff       = cfg.d_ff
    @n_heads    = cfg.n_heads
    @n_kv       = cfg.n_kv
    @d_head     = cfg.head_dim
    @group_size = cfg.n_heads / cfg.n_kv
    @n_layers   = cfg.n_layers
    @vocab_size = cfg.vocab
    @rope_base    = cfg.rope_base
    @rope_scaling = cfg.rope_scaling
    @rms_eps      = cfg.rms_eps

    @gguf_handle_keepalive = gguf_handle   # prevent GC; mmap must outlive @sess
    @sess              = TinyNN.tnn_session_new(0)
    @has_untied_output = untied
    @has_qkv_bias      = qkv_bias
    @has_qk_norm       = qk_norm

    # llama3 / LongRoPE: allocate the (d_head/2)-elem freq_factors
    # tensor in ctx_w before finalize_weights. We compute and upload
    # the values after finalize (see below). For all other rope_scaling
    # kinds the FFI call still needs a pointer — pass tnn_null_ptr.
    if @rope_scaling.kind == :llama3
      @t_rope_freq_factors = TinyNN.tnn_rope_freq_factors_alloc(@sess, @d_head)
    else
      @t_rope_freq_factors = TinyNN.tnn_null_ptr
    end

    # Wire the GGUF's mmap region into the session as the source of
    # weight bytes. Subsequent tnn_input_*_persistent_mmap calls
    # allocate tensors with .data inside this region — no copy.
    map_base = TinyNN.tnn_gguf_mmap_base(gguf_handle)
    map_size = TinyNN.tnn_gguf_mmap_size(gguf_handle)
    TinyNN.tnn_session_attach_weight_mmap(@sess, map_base, map_size)

    # Globals — embeddings + final norm + optional untied output.
    eidx = TinyNN.tnn_gguf_find_index(gguf_handle, "token_embd.weight")
    eoff = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, eidx)
    etyp = TinyNN.tnn_gguf_tensor_type(gguf_handle, eidx)
    @t_token_embed = TinyNN.tnn_input_2d_persistent_mmap(@sess,
                       @vocab_size, @d_model, etyp, eoff)

    fnidx = TinyNN.tnn_gguf_find_index(gguf_handle, "output_norm.weight")
    fnoff = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, fnidx)
    @t_final_norm_gamma = TinyNN.tnn_input_1d_persistent_mmap(@sess,
                            @d_model, 0, fnoff)   # 0 = GGML_TYPE_F32

    if untied
      oidx = TinyNN.tnn_gguf_find_index(gguf_handle, "output.weight")
      ooff = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, oidx)
      otyp = TinyNN.tnn_gguf_tensor_type(gguf_handle, oidx)
      @t_output = TinyNN.tnn_input_2d_persistent_mmap(@sess,
                    @vocab_size, @d_model, otyp, ooff)
    end

    @kv_blocks_ffi = [SmolLM2KVBlockFFI.new]
    li = 1
    while li < @n_layers
      @kv_blocks_ffi.push(SmolLM2KVBlockFFI.new)
      li = li + 1
    end

    li = 0
    while li < @n_layers
      blk = @kv_blocks_ffi[li]
      prefix = "blk." + li.to_s

      # Norms — 1D F32 mmap'd directly.
      rn1_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_norm.weight")
      rn2_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_norm.weight")
      blk.t_rn1_gamma = TinyNN.tnn_input_1d_persistent_mmap(@sess, @d_model, 0,
                          TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, rn1_idx))
      blk.t_rn2_gamma = TinyNN.tnn_input_1d_persistent_mmap(@sess, @d_model, 0,
                          TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, rn2_idx))

      # M1: Qwen3 QK-norm gammas (shape [d_head], shared across heads).
      # Tensor names follow llama.cpp: blk.X.attn_q_norm.weight etc.
      if @has_qk_norm
        qn_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_q_norm.weight")
        kn_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_k_norm.weight")
        blk.t_q_norm_gamma = TinyNN.tnn_input_1d_persistent_mmap(@sess, @d_head, 0,
                               TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, qn_idx))
        blk.t_k_norm_gamma = TinyNN.tnn_input_1d_persistent_mmap(@sess, @d_head, 0,
                               TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, kn_idx))
      end

      # Q per-head — each head is a contiguous slice of the full
      # [n_heads*d_head, d_model] tensor in HF-native row-major.
      q_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_q.weight")
      q_off_base = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, q_idx)
      q_type     = TinyNN.tnn_gguf_tensor_type(gguf_handle, q_idx)
      q_stride   = head_nbytes(q_type, @d_head, @d_model)
      blk.t_w_q = [TinyNN.tnn_input_2d_persistent_mmap(@sess,
                     @d_head, @d_model, q_type, q_off_base)]
      hq = 1
      while hq < @n_heads
        blk.t_w_q.push(TinyNN.tnn_input_2d_persistent_mmap(@sess,
                         @d_head, @d_model, q_type,
                         q_off_base + hq * q_stride))
        hq = hq + 1
      end

      # F1.2: per-Q-head LoRA adapter slots. F32-only, allocated in
      # ctx_w (trainable, not mmap'd). A: (r, d_model). B: (d_head, r).
      # Standard init (A small Gaussian + B zero) makes the adapter
      # equal to zero at step 0 → forward output matches the base
      # model exactly. Caller seeds via upload_lora_q_init!(seed).
      if @lora_q_enabled
        blk.t_w_lora_a_q = [TinyNN.tnn_input_2d_f32_persistent(@sess,
                              @lora_q_rank, @d_model)]
        blk.t_w_lora_b_q = [TinyNN.tnn_input_2d_f32_persistent(@sess,
                              @d_head, @lora_q_rank)]
        hq = 1
        while hq < @n_heads
          blk.t_w_lora_a_q.push(TinyNN.tnn_input_2d_f32_persistent(@sess,
                                  @lora_q_rank, @d_model))
          blk.t_w_lora_b_q.push(TinyNN.tnn_input_2d_f32_persistent(@sess,
                                  @d_head, @lora_q_rank))
          hq = hq + 1
        end

        # F1.2 step 6b: persistent AdamW moments paired with the LoRA
        # adapter tensors above. Same shapes. Live in ctx_w so they
        # survive tnn_reset_for_rebuild across multi-position SFT.
        if @lora_q_adamw_enabled
          blk.t_w_lora_a_q_m = [TinyNN.tnn_input_2d_f32_persistent(@sess,
                                  @lora_q_rank, @d_model)]
          blk.t_w_lora_a_q_v = [TinyNN.tnn_input_2d_f32_persistent(@sess,
                                  @lora_q_rank, @d_model)]
          blk.t_w_lora_b_q_m = [TinyNN.tnn_input_2d_f32_persistent(@sess,
                                  @d_head, @lora_q_rank)]
          blk.t_w_lora_b_q_v = [TinyNN.tnn_input_2d_f32_persistent(@sess,
                                  @d_head, @lora_q_rank)]
          hqm = 1
          while hqm < @n_heads
            blk.t_w_lora_a_q_m.push(TinyNN.tnn_input_2d_f32_persistent(@sess,
                                      @lora_q_rank, @d_model))
            blk.t_w_lora_a_q_v.push(TinyNN.tnn_input_2d_f32_persistent(@sess,
                                      @lora_q_rank, @d_model))
            blk.t_w_lora_b_q_m.push(TinyNN.tnn_input_2d_f32_persistent(@sess,
                                      @d_head, @lora_q_rank))
            blk.t_w_lora_b_q_v.push(TinyNN.tnn_input_2d_f32_persistent(@sess,
                                      @d_head, @lora_q_rank))
            hqm = hqm + 1
          end
        end
      end

      # K, V per-kv-head — same slicing math.
      k_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_k.weight")
      v_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_v.weight")
      k_off_base = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, k_idx)
      v_off_base = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, v_idx)
      k_type     = TinyNN.tnn_gguf_tensor_type(gguf_handle, k_idx)
      v_type     = TinyNN.tnn_gguf_tensor_type(gguf_handle, v_idx)
      k_stride   = head_nbytes(k_type, @d_head, @d_model)
      v_stride   = head_nbytes(v_type, @d_head, @d_model)
      blk.t_w_k = [TinyNN.tnn_input_2d_persistent_mmap(@sess,
                     @d_head, @d_model, k_type, k_off_base)]
      blk.t_w_v = [TinyNN.tnn_input_2d_persistent_mmap(@sess,
                     @d_head, @d_model, v_type, v_off_base)]
      blk.t_K   = [TinyNN.tnn_input_2d_f32_persistent(@sess, max_T,  @d_head)]
      blk.t_V   = [TinyNN.tnn_input_2d_f32_persistent(@sess, @d_head, max_T)]
      hkv = 1
      while hkv < @n_kv
        blk.t_w_k.push(TinyNN.tnn_input_2d_persistent_mmap(@sess,
                         @d_head, @d_model, k_type,
                         k_off_base + hkv * k_stride))
        blk.t_w_v.push(TinyNN.tnn_input_2d_persistent_mmap(@sess,
                         @d_head, @d_model, v_type,
                         v_off_base + hkv * v_stride))
        blk.t_K.push(TinyNN.tnn_input_2d_f32_persistent(@sess, max_T,  @d_head))
        blk.t_V.push(TinyNN.tnn_input_2d_f32_persistent(@sess, @d_head, max_T))
        hkv = hkv + 1
      end

      # Q/K/V biases — 1D F32 per head, contiguous in the file.
      if qkv_bias
        qb_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_q.bias")
        kb_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_k.bias")
        vb_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_v.bias")
        qb_off = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, qb_idx)
        kb_off = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, kb_idx)
        vb_off = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, vb_idx)
        bias_stride = @d_head * 4  # f32

        blk.t_b_q = [TinyNN.tnn_input_1d_persistent_mmap(@sess, @d_head, 0, qb_off)]
        hq = 1
        while hq < @n_heads
          blk.t_b_q.push(TinyNN.tnn_input_1d_persistent_mmap(@sess, @d_head, 0,
                           qb_off + hq * bias_stride))
          hq = hq + 1
        end

        blk.t_b_k = [TinyNN.tnn_input_1d_persistent_mmap(@sess, @d_head, 0, kb_off)]
        blk.t_b_v = [TinyNN.tnn_input_1d_persistent_mmap(@sess, @d_head, 0, vb_off)]
        hkv = 1
        while hkv < @n_kv
          blk.t_b_k.push(TinyNN.tnn_input_1d_persistent_mmap(@sess, @d_head, 0,
                           kb_off + hkv * bias_stride))
          blk.t_b_v.push(TinyNN.tnn_input_1d_persistent_mmap(@sess, @d_head, 0,
                           vb_off + hkv * bias_stride))
          hkv = hkv + 1
        end
      end

      # O / FFN — full 2D weights, no per-head slicing.
      o_idx    = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_output.weight")
      gate_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_gate.weight")
      up_idx   = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_up.weight")
      down_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_down.weight")
      blk.t_w_o    = TinyNN.tnn_input_2d_persistent_mmap(@sess, @d_model, @d_model,
                       TinyNN.tnn_gguf_tensor_type(gguf_handle, o_idx),
                       TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, o_idx))
      blk.t_w_gate = TinyNN.tnn_input_2d_persistent_mmap(@sess, @d_ff, @d_model,
                       TinyNN.tnn_gguf_tensor_type(gguf_handle, gate_idx),
                       TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, gate_idx))
      blk.t_w_up   = TinyNN.tnn_input_2d_persistent_mmap(@sess, @d_ff, @d_model,
                       TinyNN.tnn_gguf_tensor_type(gguf_handle, up_idx),
                       TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, up_idx))
      blk.t_w_down = TinyNN.tnn_input_2d_persistent_mmap(@sess, @d_model, @d_ff,
                       TinyNN.tnn_gguf_tensor_type(gguf_handle, down_idx),
                       TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, down_idx))

      li = li + 1
    end

    # F1.2: mark LoRA tensors as trainable BEFORE finalize_weights.
    # set_param flips a flag on the tensor; the build_backward pass
    # later walks PARAM-flagged nodes to emit grad nodes. Doing it
    # here (rather than in the smoke) keeps the cache class as the
    # single source of truth for what's trainable in a session.
    if @lora_q_enabled
      li2 = 0
      while li2 < @n_layers
        blk2 = @kv_blocks_ffi[li2]
        hq = 0
        while hq < @n_heads
          TinyNN.tnn_set_param(blk2.t_w_lora_a_q[hq])
          TinyNN.tnn_set_param(blk2.t_w_lora_b_q[hq])
          hq = hq + 1
        end
        li2 = li2 + 1
      end
    end

    # Finalize the regular persistent context (K/V cache buffers).
    # Mmap'd tensors don't need finalization — they were allocated
    # against weights_buf_mmap inline.
    TinyNN.tnn_finalize_weights(@sess)

    # Upload llama3-style RoPE freq_factors once the backend buffer
    # for @t_rope_freq_factors exists (post-finalize). The values are
    # a per-model constant — never re-uploaded across rebuild cycles.
    if @rope_scaling.kind == :llama3
      ff = Toy::RopeScaling.compute_llama3_freq_factors(
        @d_head, @rope_base,
        @rope_scaling.orig_max_pos, @rope_scaling.factor,
        @rope_scaling.low_freq_factor, @rope_scaling.high_freq_factor)
      TinyNN.tnn_upload_from_float_array(@sess, @t_rope_freq_factors,
                                         ff, ff.length)
    end

    # F1.2 step 6b: zero-init persistent Adam moments. AdamW's update
    # rule assumes m = v = 0 at step 0 (otherwise the first step picks
    # up garbage from the buffer). The bias-correction term beta1h/beta2h
    # then ramps in as the moments accumulate.
    if @lora_q_adamw_enabled
      za = Mat.new(@lora_q_rank, @d_model)
      zb = Mat.new(@d_head,      @lora_q_rank)
      i = 0
      while i < @lora_q_rank * @d_model; za.flat[i] = 0.0; i = i + 1; end
      j = 0
      while j < @d_head * @lora_q_rank; zb.flat[j] = 0.0; j = j + 1; end
      li_z = 0
      while li_z < @n_layers
        blk_z = @kv_blocks_ffi[li_z]
        hqz = 0
        while hqz < @n_heads
          TinyNN.upload_row_major(@sess, blk_z.t_w_lora_a_q_m[hqz], za)
          TinyNN.upload_row_major(@sess, blk_z.t_w_lora_a_q_v[hqz], za)
          TinyNN.upload_row_major(@sess, blk_z.t_w_lora_b_q_m[hqz], zb)
          TinyNN.upload_row_major(@sess, blk_z.t_w_lora_b_q_v[hqz], zb)
          hqz = hqz + 1
        end
        li_z = li_z + 1
      end
    end

    # Zero-init K/V cache buffers (same as realize_for + legacy load).
    kv_zero_k = Mat.new(max_T, @d_head)
    kv_zero_v = Mat.new(@d_head, max_T)
    li = 0
    while li < @n_layers
      blk_f = @kv_blocks_ffi[li]
      hkv = 0
      while hkv < @n_kv
        TinyNN.upload_row_major(@sess, blk_f.t_K[hkv], kv_zero_k)
        TinyNN.upload_row_major(@sess, blk_f.t_V[hkv], kv_zero_v)
        hkv = hkv + 1
      end
      li = li + 1
    end

    @realized = true
  end

  # Auto-dispatch: open the GGUF, peek at its `toy.ggml_native` flag,
  # and route to either the BYO-pointer mmap path (Phase 2) or the
  # legacy realize_for + load_weights copy path. Returns the GGUF
  # handle (or null for the legacy path); the kv_cache holds it via
  # @gguf_handle_keepalive so the mmap stays valid for inference.
  #
  # Caller must have `require_relative "toy_smollm2_loader"` at the
  # top-level driver — this file deliberately does NOT require it
  # (require-order with GGUFLoad's methods that touch `weight_type`
  # was triggering a Spinel GC crash in decode_step).
  # F1.2: standard LoRA init for the Q adapters. A = small Gaussian
  # (scale = init_scale, default 0.01); B = zero. With B=0 the LoRA
  # contribution is exactly zero, so forward output matches the base
  # model bit-for-bit at step 0. Call AFTER realize_for_mmap.
  def upload_lora_q_init!(seed, init_scale)
    if !@lora_q_enabled; return; end
    s = seed
    m_a = Mat.new(@lora_q_rank, @d_model)
    m_b = Mat.new(@d_head, @lora_q_rank)
    z_b = m_b
    i_b = 0
    while i_b < @d_head * @lora_q_rank
      z_b.flat[i_b] = 0.0
      i_b = i_b + 1
    end
    li = 0
    while li < @n_layers
      blk = @kv_blocks_ffi[li]
      hq = 0
      while hq < @n_heads
        # Per-(layer, head) Gaussian for A via Box-Muller on an LCG.
        ii = 0
        while ii < @lora_q_rank * @d_model
          s = (s * 1103515245 + 12345) & 0x7FFFFFFF
          u1 = (s.to_f + 1.0) / 2147483648.0
          s = (s * 1103515245 + 12345) & 0x7FFFFFFF
          u2 = (s.to_f + 1.0) / 2147483648.0
          m_a.flat[ii] = init_scale * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
          ii = ii + 1
        end
        TinyNN.upload_row_major(@sess, blk.t_w_lora_a_q[hq], m_a)
        TinyNN.upload_row_major(@sess, blk.t_w_lora_b_q[hq], z_b)
        hq = hq + 1
      end
      li = li + 1
    end
  end

  def realize_and_load_auto(gguf_path, max_T, cfg, flags)
    gguf = TinyNN.tnn_gguf_load(gguf_path)
    is_native = TinyNN.tnn_gguf_get_bool(gguf, "toy.ggml_native") == 1
    if is_native
      wtype = GGUFLoad.detect_weight_type(gguf_path)
      set_weight_type(wtype)
      realize_for_mmap(gguf, cfg, max_T, flags.untied, flags.qkv_bias)
      puts "  BYO-pointer mmap (weight_type=" + wtype.to_s + ")"
      gguf
    else
      TinyNN.tnn_gguf_free(gguf)
      realize_for(max_T, cfg.d_model, cfg.d_ff,
                  cfg.n_heads, cfg.n_kv,
                  cfg.n_layers, cfg.vocab,
                  cfg.rope_base, cfg.rms_eps,
                  flags.untied, flags.qkv_bias)
      load_weights(gguf_path)
      puts "  legacy copy load"
      TinyNN.tnn_null_ptr
    end
  end

  # Per-head byte stride for slicing a full [n_heads*d_head, d_model]
  # tensor into n_heads contiguous Dh×D blocks. Matches ggml_nbytes()
  # of a per-head ne=[d_model, d_head] tensor of `ggml_type`.
  def head_nbytes(ggml_type, d_head, d_model)
    if ggml_type == 0
      # F32: d_head * d_model * 4
      d_head * d_model * 4
    elsif ggml_type == 8
      # Q8_0: blocks of 32 elements stored as (half + 32 int8) = 34 bytes.
      # Per-head has d_head rows of d_model elements each.
      # bytes = d_head * (d_model / 32) * 34
      d_head * (d_model / 32) * 34
    else
      # Unknown: refuse rather than guess.
      0
    end
  end

  # CUDA-MIRROR-SKIP-BEGIN: trace-tap diagnostic ivars are CPU-only;
  # the CUDA mirror gets a no-op stub for trace_tap and an empty
  # dump_trace + enable_trace! so callers don't have to backend-switch.
  # CUDA-MIRROR-STUB:   def enable_trace!; end
  # CUDA-MIRROR-STUB:   def trace_tap(_name, t); t; end
  # CUDA-MIRROR-STUB:   def dump_trace; end
  def enable_trace!
    @trace_on = true
  end

  # Insert a tap at a named point in the graph. Returns `t` unchanged
  # so callers can write `t = trace_tap("L0.rn1", t)` inline. With
  # tracing off this just returns t; with tracing on it also pushes
  # the (name, tensor) pair and marks the tensor as a scheduler output.
  def trace_tap(name_, t)
    if @trace_on
      @trace_names.push(name_)
      @trace_tensors.push(t)
      TinyNN.tnn_set_output(t)
    end
    t
  end

  # Walk the captured taps after compute. Resets the arrays at the
  # end so the next decode_step starts fresh.
  def dump_trace
    if !@trace_on
      return
    end
    i = 0
    total = @trace_names.length
    while i < total
      nm = @trace_names[i]
      t  = @trace_tensors[i]
      n  = TinyNN.tnn_tensor_nelements(t)
      TinyNN.tnn_download(@sess, t)
      mn   = TinyNN.tnn_scratch_min_f32(@sess, n)
      mx   = TinyNN.tnn_scratch_max_f32(@sess, n)
      sa   = TinyNN.tnn_scratch_sum_abs_f32(@sess, n)
      nan  = TinyNN.tnn_scratch_nan_count_f32(@sess, n)
      mean_abs = sa / n.to_f
      puts "    " + nm.ljust(24) + " n=" + n.to_s.rjust(6) +
           " min=" + mn.to_s +
           " max=" + mx.to_s +
           " |mean|=" + mean_abs.to_s +
           " nan=" + nan.to_s
      i = i + 1
    end
    # Reset for the next decode_step. Spinel-friendly: pop everything
    # rather than reassign the ivar.
    while @trace_names.length > 0
      @trace_names.pop
    end
    while @trace_tensors.length > 0
      @trace_tensors.pop
    end
  end
  # CUDA-MIRROR-SKIP-END

  # Ruby-OO entry point for "load weights into this realized cache."
  # Auto-detects layout: GGUFs with the `toy.ggml_native` metadata key
  # take the memcpy path (no transpose); legacy GGUFs take the
  # transposing path. Callers stay layout-agnostic.
  def load_weights(path)
    GGUFLoad.load_kv_cache_auto(self, path)
  end

  # Pull any persistent FFI tensor back to a Ruby Mat (chunked download,
  # works for weight-sized tensors). Required by the design rule that
  # the direct-loader path must keep Mat-roundtrip open — see
  # docs/loader-api.md.
  #
  # `t` is any tensor handle exposed on this cache or its blocks
  # (e.g. `kv.t_token_embed`, `kv.kv_blocks_ffi[3].t_w_o`). `rows` and
  # `cols` are the logical shape; we trust the caller.
  def read_persistent_mat(t, rows, cols)
    TinyNN.download_to_mat(@sess, t, rows, cols)
  end

  # Declare every persistent tensor (weights + K/V buffers) and finalize.
  # `untied` is true for TinyLlama-shape models that have a separate
  # `output.weight` (lm_head); false for SmolLM2 / Qwen2.5 with tied
  # embeddings. When false we skip the (vocab × d_model) t_output
  # allocation entirely. `qkv_bias` is true for Qwen2.x; when false the
  # b_q/b_k/b_v tensors aren't allocated and Q/K/V matmuls land
  # without an add.
  def realize_for(max_T, d_model, d_ff, n_heads, n_kv, n_layers,
                  vocab_size, rope_base, rms_eps, untied, qkv_bias)
    @max_T      = max_T
    @d_model    = d_model
    @d_ff       = d_ff
    @n_heads    = n_heads
    @n_kv       = n_kv
    @d_head     = d_model / n_heads
    @group_size = n_heads / n_kv
    @n_layers   = n_layers
    @vocab_size = vocab_size
    @rope_base  = rope_base
    @rms_eps    = rms_eps

    @sess               = TinyNN.tnn_session_new(0)
    @t_token_embed      = TinyNN.tnn_input_2d_f32_persistent(@sess, vocab_size, d_model)
    @t_final_norm_gamma = TinyNN.tnn_input_1d_f32_persistent(@sess, d_model)
    @has_untied_output  = untied
    @has_qkv_bias       = qkv_bias
    if untied
      @t_output = alloc_2d_w(vocab_size, d_model)
    end

    @kv_blocks_ffi = [SmolLM2KVBlockFFI.new]
    li = 1
    while li < n_layers
      @kv_blocks_ffi.push(SmolLM2KVBlockFFI.new)
      li = li + 1
    end

    li = 0
    while li < n_layers
      blk = @kv_blocks_ffi[li]
      blk.t_rn1_gamma = TinyNN.tnn_input_1d_f32_persistent(@sess, d_model)
      blk.t_rn2_gamma = TinyNN.tnn_input_1d_f32_persistent(@sess, d_model)

      # Q: n_heads per-head matrices of (d_head, d_model). Quantizable.
      blk.t_w_q = [alloc_2d_w(d_head, d_model)]
      if qkv_bias
        blk.t_b_q = [TinyNN.tnn_input_1d_f32_persistent(@sess, d_head)]
      end
      hq = 1
      while hq < n_heads
        blk.t_w_q.push(alloc_2d_w(d_head, d_model))
        if qkv_bias
          blk.t_b_q.push(TinyNN.tnn_input_1d_f32_persistent(@sess, d_head))
        end
        hq = hq + 1
      end

      # K, V (and the persistent K/V buffers): n_kv per-head. Linear
      # weights quantizable; K/V cache buffers and biases stay F32.
      blk.t_w_k = [alloc_2d_w(d_head, d_model)]
      blk.t_w_v = [alloc_2d_w(d_head, d_model)]
      blk.t_K   = [TinyNN.tnn_input_2d_f32_persistent(@sess, max_T,  d_head)]
      blk.t_V   = [TinyNN.tnn_input_2d_f32_persistent(@sess, d_head, max_T)]
      if qkv_bias
        # K bias: 1-D (broadcasts over [d_head, 1] k matmul result).
        # V bias: 1-D too (the V matmul is now ordered weight-first, so
        # its result is [d_head, 1] like K — matches a 1-D bias).
        blk.t_b_k = [TinyNN.tnn_input_1d_f32_persistent(@sess, d_head)]
        blk.t_b_v = [TinyNN.tnn_input_1d_f32_persistent(@sess, d_head)]
      end
      hkv = 1
      while hkv < n_kv
        blk.t_w_k.push(alloc_2d_w(d_head, d_model))
        blk.t_w_v.push(alloc_2d_w(d_head, d_model))
        blk.t_K.push(TinyNN.tnn_input_2d_f32_persistent(@sess, max_T,  d_head))
        blk.t_V.push(TinyNN.tnn_input_2d_f32_persistent(@sess, d_head, max_T))
        if qkv_bias
          blk.t_b_k.push(TinyNN.tnn_input_1d_f32_persistent(@sess, d_head))
          blk.t_b_v.push(TinyNN.tnn_input_1d_f32_persistent(@sess, d_head))
        end
        hkv = hkv + 1
      end

      blk.t_w_o    = alloc_2d_w(d_model, d_model)
      blk.t_w_gate = alloc_2d_w(d_ff,    d_model)
      blk.t_w_up   = alloc_2d_w(d_ff,    d_model)
      blk.t_w_down = alloc_2d_w(d_model, d_ff)
      li = li + 1
    end

    TinyNN.tnn_finalize_weights(@sess)
    @realized = true
  end

  # Build the compute graph for one decode position.
  def build_decode_step(pos)
    eps     = @rms_eps
    scale   = 1.0 / Math.sqrt(@d_head.to_f)
    d_model = @d_model
    d_head  = @d_head
    max_T   = @max_T
    bytes_d_head = d_head * 4
    bytes_max_T  = max_T * 4

    # Inputs: token id + RoPE position. Both length 1.
    t_token_id  = TinyNN.tnn_input_1d_i32(@sess, 1)
    t_pos       = TinyNN.tnn_input_1d_i32_ctx(@sess, 1)

    t_x = TinyNN.tnn_get_rows(@sess, @t_token_embed, t_token_id)   # ne=[d_model, 1]
    t_x = trace_tap("embed", t_x)

    li = 0
    while li < @n_layers
      t_x = build_block_step(t_x, @kv_blocks_ffi[li], t_pos, pos,
                              scale, eps, bytes_d_head, bytes_max_T, li)
      li = li + 1
    end

    t_x_final = TinyNN.tnn_rms_norm(@sess, t_x, @t_final_norm_gamma, eps)
    t_x_final = trace_tap("final_norm", t_x_final)
    # Logits: untied path matmuls against t_output (lm_head); tied
    # path against t_token_embed. Both tensors are [vocab, d_model],
    # so the matmul shape is identical either way.
    if @has_untied_output
      t_kv_logits = TinyNN.tnn_matmul(@sess, @t_output, t_x_final)
    else
      t_kv_logits = TinyNN.tnn_matmul(@sess, @t_token_embed, t_x_final)
    end
    TinyNN.tnn_set_output(t_kv_logits)
    SmolLM2KVStepResult.new(t_token_id, t_pos, t_kv_logits)
  end

  def build_block_step(t_x, blk, t_pos, pos, scale, eps,
                        bytes_d_head, bytes_max_T, layer_idx)
    # Layer-tag prefix for tap names (e.g. "L00."). String concat of an
    # int needs explicit .to_s; ljust pads so all names align in output.
    tag = "L" + layer_idx.to_s + "."

    t_h = TinyNN.tnn_rms_norm(@sess, t_x, blk.t_rn1_gamma, eps)
    t_h = trace_tap(tag + "rn1", t_h)

    # --- compute K, V for each KV head (n_kv times), rope K, cpy into buffers ---
    hkv = 0
    while hkv < @n_kv
      t_k_raw = TinyNN.tnn_matmul(@sess, blk.t_w_k[hkv], t_h)         # ne=[d_head, 1]
      if @has_qkv_bias
        t_k_pre = TinyNN.tnn_add(@sess, t_k_raw, blk.t_b_k[hkv])
      else
        t_k_pre = t_k_raw
      end
      # Tap K (head 0 only) post-bias, pre-RoPE.
      if hkv == 0
        t_k_pre = trace_tap(tag + "k_pre", t_k_pre)
      end
      # M1: Qwen3 QK-norm — RMSNorm K with the per-block gamma before
      # rope. Shared gamma across heads (shape [d_head]).
      if @has_qk_norm
        t_k_pre = TinyNN.tnn_rms_norm(@sess, t_k_pre, blk.t_k_norm_gamma, @rms_eps)
      end
      t_k_rot = TinyNN.tnn_rope_ext(@sess, t_k_pre, t_pos, @d_head,
                                    @rope_base, @rope_scaling.freq_scale,
                                    @rope_scaling.ext_factor,
                                    @rope_scaling.attn_factor,
                                    @rope_scaling.beta_fast,
                                    @rope_scaling.beta_slow,
                                    @t_rope_freq_factors)
      if hkv == 0
        t_k_rot = trace_tap(tag + "k_rot", t_k_rot)
      end
      # V matmul: weight in A position so ggml's matmul kernel can
      # dispatch to Q8 (and other quantized) kernels. Result is
      # [d_head, 1] instead of the legacy [1, d_head]; a contiguous
      # view_2d before the cpy reinterprets it as a [1, d_head] row
      # without moving bytes.
      t_v_raw = TinyNN.tnn_matmul(@sess, blk.t_w_v[hkv], t_h)         # ne=[d_head, 1]
      if @has_qkv_bias
        t_v_new = TinyNN.tnn_add(@sess, t_v_raw, blk.t_b_v[hkv])      # bias is 1-D [d_head]
      else
        t_v_new = t_v_raw
      end
      if hkv == 0
        t_v_new = trace_tap(tag + "v_new", t_v_new)
      end

      t_K_slot = TinyNN.tnn_view_2d(@sess, blk.t_K[hkv],
                                      @d_head, 1, bytes_d_head, pos * bytes_d_head)
      t_cpy_k = TinyNN.tnn_cpy(@sess, t_k_rot, t_K_slot)
      t_V_slot = TinyNN.tnn_view_2d(@sess, blk.t_V[hkv],
                                      1, @d_head, bytes_max_T, pos * 4)
      # Re-interpret [d_head, 1] as [1, d_head] (same contiguous bytes).
      t_v_row = TinyNN.tnn_view_2d(@sess, t_v_new, 1, @d_head, 4, 0)
      t_cpy_v = TinyNN.tnn_cpy(@sess, t_v_row, t_V_slot)
      TinyNN.tnn_add_to_graph(@sess, t_cpy_k)
      TinyNN.tnn_add_to_graph(@sess, t_cpy_v)
      hkv = hkv + 1
    end

    # --- per-Q-head attention ---
    t_head_out0 = build_attention_qhead_step(t_h, blk, 0, t_pos, pos,
                                              scale, bytes_d_head, bytes_max_T,
                                              tag, true)
    t_head_outs = [t_head_out0]
    hq = 1
    while hq < @n_heads
      t_head_outs.push(build_attention_qhead_step(t_h, blk, hq, t_pos, pos,
                                                    scale, bytes_d_head, bytes_max_T,
                                                    tag, false))
      hq = hq + 1
    end

    t_concat = t_head_outs[0]
    hq = 1
    while hq < @n_heads
      t_concat = TinyNN.tnn_concat(@sess, t_concat, t_head_outs[hq], 0)
      hq = hq + 1
    end
    t_concat = trace_tap(tag + "concat", t_concat)

    t_out_proj = TinyNN.tnn_matmul(@sess, blk.t_w_o, t_concat)
    t_out_proj = trace_tap(tag + "attn_out", t_out_proj)
    t_x_attn   = TinyNN.tnn_add(@sess, t_x, t_out_proj)
    t_x_attn   = trace_tap(tag + "post_attn", t_x_attn)

    # --- SwiGLU FFN ---
    t_h2     = TinyNN.tnn_rms_norm(@sess, t_x_attn, blk.t_rn2_gamma, eps)
    t_h2     = trace_tap(tag + "rn2", t_h2)
    t_gate   = TinyNN.tnn_matmul(@sess, blk.t_w_gate, t_h2)        # ne=[d_ff, 1]
    t_gate   = trace_tap(tag + "gate", t_gate)
    t_up     = TinyNN.tnn_matmul(@sess, blk.t_w_up,   t_h2)        # ne=[d_ff, 1]
    t_up     = trace_tap(tag + "up", t_up)
    t_silug  = TinyNN.tnn_silu(@sess, t_gate)
    t_silug  = trace_tap(tag + "silu_gate", t_silug)
    t_gated  = TinyNN.tnn_mul(@sess, t_silug, t_up)
    t_gated  = trace_tap(tag + "gated", t_gated)
    t_dn     = TinyNN.tnn_matmul(@sess, blk.t_w_down, t_gated)     # ne=[d_model, 1]
    t_dn     = trace_tap(tag + "dn", t_dn)

    t_post_ffn = TinyNN.tnn_add(@sess, t_x_attn, t_dn)
    trace_tap(tag + "post_ffn", t_post_ffn)
  end

  # One query head. Uses the (already-written) K and V of the
  # corresponding KV head — index = hq / group_size. `tag` is the
  # "L<i>." layer prefix; `tap_this_head` is true only for head 0 so we
  # don't multiply taps by n_heads in trace mode.
  def build_attention_qhead_step(t_h, blk, hq, t_pos, pos, scale,
                                  bytes_d_head, bytes_max_T,
                                  tag, tap_this_head)
    hkv = hq / @group_size

    t_q_raw = TinyNN.tnn_matmul(@sess, blk.t_w_q[hq], t_h)   # ne=[d_head, 1]
    # F1.2: optional LoRA on Q. Standard placement is BEFORE the bias
    # add (HF LoRA practice — the bias stays a property of the base
    # projection, LoRA only adjusts the linear part). Math:
    #   q_lora = w_lora_b[hq] @ (w_lora_a[hq] @ t_h)
    #   q_raw  := q_raw + q_lora
    # With B init to zero, q_lora == 0 and q_raw is unchanged.
    if @lora_q_enabled
      t_lora_a_h    = TinyNN.tnn_matmul(@sess, blk.t_w_lora_a_q[hq], t_h)      # ne=[r, 1]
      t_lora_b_a_h  = TinyNN.tnn_matmul(@sess, blk.t_w_lora_b_q[hq], t_lora_a_h)# ne=[d_head, 1]
      t_q_raw       = TinyNN.tnn_add(@sess, t_q_raw, t_lora_b_a_h)
    end
    if @has_qkv_bias
      t_q_pre = TinyNN.tnn_add(@sess, t_q_raw, blk.t_b_q[hq])
    else
      t_q_pre = t_q_raw
    end
    if tap_this_head
      t_q_pre = trace_tap(tag + "q_pre", t_q_pre)
    end
    if @has_qk_norm
      t_q_pre = TinyNN.tnn_rms_norm(@sess, t_q_pre, blk.t_q_norm_gamma, @rms_eps)
    end
    t_q     = TinyNN.tnn_rope_ext(@sess, t_q_pre, t_pos, @d_head,
                                  @rope_base, @rope_scaling.freq_scale,
                                  @rope_scaling.ext_factor,
                                  @rope_scaling.attn_factor,
                                  @rope_scaling.beta_fast,
                                  @rope_scaling.beta_slow,
                                  @t_rope_freq_factors)
    if tap_this_head
      t_q = trace_tap(tag + "q_rot", t_q)
    end

    t_K_hist = TinyNN.tnn_view_2d(@sess, blk.t_K[hkv],
                                    @d_head, pos + 1, bytes_d_head, 0)
    t_V_hist = TinyNN.tnn_view_2d(@sess, blk.t_V[hkv],
                                    pos + 1, @d_head, bytes_max_T, 0)

    t_scores = TinyNN.tnn_matmul(@sess, t_K_hist, t_q)
    if tap_this_head
      t_scores = trace_tap(tag + "scores", t_scores)
    end
    t_scaled = TinyNN.tnn_scale(@sess, t_scores, scale)
    t_attn   = TinyNN.tnn_softmax(@sess, t_scaled)
    if tap_this_head
      t_attn = trace_tap(tag + "softmax", t_attn)
    end
    t_head = TinyNN.tnn_matmul(@sess, t_V_hist, t_attn)
    if tap_this_head
      t_head = trace_tap(tag + "head0", t_head)
    end
    t_head
  end
end

# Init-param names deliberately differ from the ivar names — same
# defensive pattern as GPT2KVStepResult.
class SmolLM2KVStepResult
  attr_accessor :t_token_id, :t_pos, :kv_step_logits
  def initialize(tok_ptr, pos_ptr, logits_ptr)
    @t_token_id     = tok_ptr
    @t_pos          = pos_ptr
    @kv_step_logits = logits_ptr
  end
end

module SmolLM2KV
  # Upload all Toy::SmolLM2 weights into a realized cache (+ zero-init
  # the K/V buffers).
  def self.upload_from(kv_cache, model)
    sess     = kv_cache.sess
    n        = kv_cache.n_layers
    n_heads  = kv_cache.n_heads
    n_kv     = kv_cache.n_kv
    d_model  = kv_cache.d_model
    d_head   = kv_cache.d_head
    max_T    = kv_cache.max_T

    TinyNN.upload_row_major(sess, kv_cache.t_token_embed, model.token_embed.weight)
    TinyNN.tnn_upload_from_float_array(sess, kv_cache.t_final_norm_gamma,
                                        model.final_norm.gamma, d_model)
    if kv_cache.has_untied_output
      TinyNN.upload_row_major(sess, kv_cache.t_output, model.output_proj)
    end

    kv_zero_k = Mat.new(max_T,  d_head)
    kv_zero_v = Mat.new(d_head, max_T)

    li = 0
    while li < n
      blk_n = model.stack[li]
      blk_f = kv_cache.kv_blocks_ffi[li]

      TinyNN.tnn_upload_from_float_array(sess, blk_f.t_rn1_gamma, blk_n.rn1.gamma, d_model)
      TinyNN.tnn_upload_from_float_array(sess, blk_f.t_rn2_gamma, blk_n.rn2.gamma, d_model)

      hq = 0
      while hq < n_heads
        TinyNN.stage_transposed_and_upload(sess, blk_f.t_w_q[hq], blk_n.attn.w_q[hq])
        if kv_cache.has_qkv_bias
          TinyNN.tnn_upload_from_float_array(sess, blk_f.t_b_q[hq], blk_n.attn.b_q[hq], d_head)
        end
        hq = hq + 1
      end

      hkv = 0
      while hkv < n_kv
        TinyNN.stage_transposed_and_upload(sess, blk_f.t_w_k[hkv], blk_n.attn.w_k[hkv])
        TinyNN.stage_transposed_and_upload(sess, blk_f.t_w_v[hkv], blk_n.attn.w_v[hkv])
        if kv_cache.has_qkv_bias
          TinyNN.tnn_upload_from_float_array(sess, blk_f.t_b_k[hkv], blk_n.attn.b_k[hkv], d_head)
          TinyNN.tnn_upload_from_float_array(sess, blk_f.t_b_v[hkv], blk_n.attn.b_v[hkv], d_head)
        end
        TinyNN.upload_row_major(sess, blk_f.t_K[hkv], kv_zero_k)
        TinyNN.upload_row_major(sess, blk_f.t_V[hkv], kv_zero_v)
        hkv = hkv + 1
      end

      TinyNN.stage_transposed_and_upload(sess, blk_f.t_w_o,    blk_n.attn.w_o)
      TinyNN.stage_transposed_and_upload(sess, blk_f.t_w_gate, blk_n.ffn.w_gate)
      TinyNN.stage_transposed_and_upload(sess, blk_f.t_w_up,   blk_n.ffn.w_up)
      TinyNN.stage_transposed_and_upload(sess, blk_f.t_w_down, blk_n.ffn.w_down)

      li = li + 1
    end
  end

  # Decode one new token at position `pos`. Returns the (1, vocab)
  # logits Mat for the new position. If `kv_cache.trace_on` is set the
  # rebuild path inserts taps and we dump stats before reading logits.
  def self.decode_step(kv_cache, token_id, pos)
    TinyNN.tnn_reset_for_rebuild(kv_cache.sess)
    step = kv_cache.build_decode_step(pos)
    TinyNN.tnn_realize(kv_cache.sess, step.kv_step_logits)
    TinyNN.upload_int_array(kv_cache.sess, step.t_token_id, [token_id])
    TinyNN.upload_int_array(kv_cache.sess, step.t_pos,      [pos])
    TinyNN.tnn_compute(kv_cache.sess)
    kv_cache.dump_trace
    TinyNN.download_row_major(kv_cache.sess, step.kv_step_logits, 1, kv_cache.vocab_size)
  end
end
