# lib/toy/llm/engine/llama_kv_engine.rb (was lib/toy_smollm2_ffi_kv.rb) — Toy::SmolLM2 KV-cache decode via ggml FFI.
#
# Mirror of lib/toy/llm/engine/gpt2_kv_engine.rb (was lib/gpt2_ffi_kv.rb)
# but for the llama-family architecture:
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

require_relative "../../models/transformer"
require_relative "../../../toy"
require_relative "../../models/toy_smollm2"
require_relative "../../ffi/tinynn"
# NOTE: not requiring "toy_smollm2_loader" here. Requiring it from
# inside this file triggers a Spinel GC mark crash in decode_step
# (sp_gc_mark / sp_PtrArray_new_scan) for reasons we haven't fully
# isolated — likely something about require-order interaction with
# Spinel's type inference around GGUFLoad. Callers that use
# realize_and_load_auto (or any method here that references
# GGUFLoad) must `require_relative "toy/models/toy_smollm2_loader"` from their
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
                # I-Gemma (#113): post-attention and post-FFN RMSNorm
                # gammas. Gemma 2 sandwiches each sublayer between a
                # pre-norm (the existing t_rn1_gamma / t_rn2_gamma)
                # and a post-norm (these). Shape [d_model] each.
                # Allocated only when cache.has_post_norms is set.
                :t_post_attn_norm_gamma, :t_post_ffn_norm_gamma,
                :t_w_gate, :t_w_up, :t_w_down,
                # M2.3 MoE. When SmolLM2KVFFICache#is_moe is true, the
                # FFN block is replaced with a Mixtral-style routed FFN:
                #   t_w_router    : 2D [d_model, n_experts] — gating
                #   t_w_gate_exps : 3D [d_model, d_ff, n_experts]
                #   t_w_up_exps   : 3D [d_model, d_ff, n_experts]
                #   t_w_down_exps : 3D [d_ff, d_model, n_experts]
                # Set by realize_for_mmap when GGUF carries
                # blk.0.ffn_gate_inp.weight (the MoE-presence sentinel).
                :t_w_router, :t_w_gate_exps, :t_w_up_exps, :t_w_down_exps,
                # P2: always-on shared experts (Qwen2-MoE / DeepSeek-V2 / GLM-MoE).
                # Loaded when blk carries ffn_gate_shexp; t_w_shexp_gate is the
                # optional Qwen-style sigmoid gate (ffn_gate_inp_shexp), null otherwise.
                :t_w_gate_shexp, :t_w_up_shexp, :t_w_down_shexp, :t_w_shexp_gate,
                # DeepSeek-V2 MLA (Multi-head Latent Attention). Allocated
                # only when cache.is_mla. Replaces the per-head t_w_q/k/v:
                #   t_w_q_mla        : full attn_q [d_model → n_heads·d_head_k]
                #   t_w_kv_a         : attn_kv_a_mqa [d_model → kv_lora_rank+rope_dim]
                #   t_kv_a_norm_gamma: RMSNorm over the latent c_kv [kv_lora_rank]
                #   t_w_kv_b         : attn_kv_b [kv_lora_rank → n_heads·(d_head_nope+d_head_v)]
                # attn_output reuses t_w_o (sized n_heads·d_head_v → d_model).
                :t_w_q_mla, :t_w_kv_a, :t_kv_a_norm_gamma, :t_w_kv_b,
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
    @t_post_attn_norm_gamma = TinyNN.tnn_null_ptr
    @t_post_ffn_norm_gamma  = TinyNN.tnn_null_ptr
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
    @t_w_router    = TinyNN.tnn_null_ptr
    @t_w_gate_exps = TinyNN.tnn_null_ptr
    @t_w_up_exps   = TinyNN.tnn_null_ptr
    @t_w_down_exps = TinyNN.tnn_null_ptr
    @t_w_gate_shexp = TinyNN.tnn_null_ptr
    @t_w_up_shexp   = TinyNN.tnn_null_ptr
    @t_w_down_shexp = TinyNN.tnn_null_ptr
    @t_w_shexp_gate = TinyNN.tnn_null_ptr
    @t_w_q_mla        = TinyNN.tnn_null_ptr
    @t_w_kv_a         = TinyNN.tnn_null_ptr
    @t_kv_a_norm_gamma = TinyNN.tnn_null_ptr
    @t_w_kv_b         = TinyNN.tnn_null_ptr
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
                # #110: which QK-norm flavor — 1 = per-head shared
                # gamma (Qwen3, gamma shape [d_head]); 2 = full-Q
                # gamma (OLMoE / Granite, gamma shape [d_model],
                # applied to the concatenated Q before head split).
                # 0 = none. Set by realize_for_mmap from the detected
                # flags. The graph builder branches on this.
                :qk_norm_kind,
                # I-Gemma (#113): Gemma 2-specific knobs. All default
                # to inert values (no-op) for non-Gemma models.
                #   has_post_norms: blk.X has post_attention_norm +
                #     post_ffw_norm tensors after the residual adds.
                #   embed_scale: post-token-embed multiplier
                #     (sqrt(d_model) for Gemma 2; 1.0 otherwise).
                #   attn_softcap: tanh-softcap on attention logits
                #     (50.0 for Gemma 2; 0.0 disables).
                #   final_softcap: tanh-softcap on the final output
                #     logits (30.0 for Gemma 2; 0.0 disables).
                #   swa_alternates: when true, only EVEN layers apply
                #     sliding window; odd layers see full attention.
                :has_post_norms, :embed_scale,
                :attn_softcap, :final_softcap, :swa_alternates,
                # M3: SWA window. 0 = no sliding window (full causal).
                # >0 = attend only to the last `swa_window` positions
                # in the K/V cache. Phi-3-mini-4k sets this to 2048;
                # Gemma 2 local layers set it to 4096. Realize-time
                # parameter (set via realize_for_mmap or post-init).
                :swa_window,
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
                # P5.1+P5.2: KV cache dtype. 0 = F32 (legacy), 8 = Q8_0.
                # `enable_kv_q8!` sets both to Q8_0; finer-grained
                # control is reserved for future debugging. Per-position
                # writes go through ggml_cpy which quantizes f32→Q8 at
                # the destination view. P5.2 flipped V's layout to
                # match K (`ne=[d_head, max_T]`, positions on ne1), so
                # both write paths span contiguous d_head-vectors —
                # block-aligned for Q8 at d_head=64 (=2 blocks of 32).
                :kv_type_k, :kv_type_v,
                # P4.1: opt into ggml_flash_attn_ext in the attention
                # step (default false → existing scale→softmax→matmul
                # triplet). When true, each Q head's attention is one
                # fused kernel call. Backward NOT supported (flash_back
                # aborts in vendored ggml), so this is INFERENCE only.
                # Set via enable_flash_attn! BEFORE realize_for_*.
                :use_flash_attn,
                # M2.3: MoE flags. is_moe → replace SwiGLU FFN with the
                # routed expert FFN (router → softmax → top_k → 3× mul_mat_id
                # → silu·up → weighted sum). Set by detect_smollm2_flags
                # when GGUF carries blk.0.ffn_gate_inp.weight.
                :is_moe, :n_experts, :n_experts_used,
                # DeepSeek-V2 MLA. is_mla flips the attention path to the
                # latent-projection builder (build_mla_block_step). dims:
                #   kv_lora_rank, d_head_k (qk=192), d_head_v (v=128),
                #   rope_dim (decoupled-RoPE width=64). leading_dense is
                #   the # of dense layers before MoE starts (per-layer FFN
                #   dispatch). mla_kq_scale is the YaRN-mscale-adjusted
                #   softmax scale precomputed at realize.
                :is_mla, :kv_lora_rank, :d_head_k, :d_head_v,
                :rope_dim, :leading_dense, :mla_kq_scale,
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
    @qk_norm_kind       = 0
    @swa_window         = 0
    @has_post_norms     = false
    @embed_scale        = 1.0
    @attn_softcap       = 0.0
    @final_softcap      = 0.0
    @swa_alternates     = false
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
    @kv_type_k     = 0                # GGML_TYPE_F32; opt in via enable_kv_q8!
    @kv_type_v     = 0                # GGML_TYPE_F32; opt in via enable_kv_q8!
    @use_flash_attn = false            # opt in via enable_flash_attn!
    @is_moe         = false
    @n_experts      = 0
    @n_experts_used = 0
    @moe_norm_topk  = false   # norm_topk_prob; set in realize_for_mmap (per-arch)
    # DeepSeek-V2 MLA — inert by default (realize_for_mmap sets from cfg).
    @is_mla         = false
    @kv_lora_rank   = 0
    @d_head_k       = 0
    @d_head_v       = 0
    @rope_dim       = 0
    @leading_dense  = 0
    @mla_kq_scale   = 0.0
    @gguf_handle_keepalive = TinyNN.tnn_null_ptr  # set by realize_for_mmap
    @lora_q_enabled = false
    @lora_q_rank    = 0
    @lora_q_adamw_enabled = false
  end

  # P5.1: opt into Q8_0 storage for the K cache. Must be called BEFORE
  # realize_for_mmap. V stays F32 in this phase — its layout
  # (positions along ne0) makes per-position Q8 writes non-block-
  # aligned. K's layout (positions along ne1, d_head along ne0)
  # writes whole d_head-vectors at a time, which for d_head=64
  # spans exactly 2 Q8_0 blocks of 32 elements each → aligned. The
  # write path uses ggml_cpy which quantizes on f32→Q8 destination;
  # the read path (attention matmul) dequantizes block-by-block
  # inside ggml's kernel. Cuts K-cache memory & bandwidth ~4×.
  # P5.1+P5.2: opt into Q8_0 for the K and V caches. Halves K and V
  # memory + bandwidth (3.75× smaller at d_head=64).
  #
  # Auto-enables flash attention. Reason: the non-flash V matmul
  # requires a transpose-cont of V_hist, which is structurally
  # impossible for Q8_0 (transposing flips the d_head and hist_count
  # axes; hist_count generally isn't a multiple of 32, so the
  # contiguous Q8 destination can't be allocated). flash_attn_ext
  # consumes V in its natural [d_head, hist_count] orientation,
  # which dodges the transpose entirely — so Q8 V works there.
  #
  # Inference-only. flash_attn's backward aborts in vendored ggml.
  def enable_kv_q8!
    @kv_type_k      = 8   # GGML_TYPE_Q8_0
    @kv_type_v      = 8
    @use_flash_attn = true
  end

  # P4.1: opt into ggml_flash_attn_ext for inference. Per-Q-head it
  # replaces the (scale → softmax → matmul) triplet with one fused
  # call. The V cache stays in its current [max_T, d_head] layout —
  # we transpose-materialize it per step (cheap; one ggml_cont). A
  # future cleanup (P5.2) flips V's layout to remove the transpose
  # and unlock V Q8.
  #
  # Backward is unsupported in vendored ggml (flash_attn_back aborts),
  # so this path is INFERENCE only. Call BEFORE realize_for_mmap.
  def enable_flash_attn!
    @use_flash_attn = true
  end

  # M2.3: opt into the MoE FFN graph. Must be called BEFORE realize_for_mmap.
  # n_experts is the total count in the GGUF; n_experts_used is the
  # top-K routed per token. Mixtral-8x7B: enable_moe!(8, 2). Qwen3-30B-
  # A3B: enable_moe!(128, 8) (with optional shared expert — not yet
  # supported in this path).
  def enable_moe!(n_experts, n_experts_used)
    @is_moe         = true
    @n_experts      = n_experts
    @n_experts_used = n_experts_used
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

    # DeepSeek-V2 MLA. d_head_k (qk=192) / d_head_v (v=128) make the K and
    # V caches ASYMMETRIC; the attention output width is n_heads·d_head_v.
    # mla_kq_scale folds the YaRN mscale into the softmax scale:
    #   mscale   = 1 + yarn_log_mul·ln(factor)
    #   kq_scale = mscale² / √d_head_k
    @is_mla        = cfg.is_mla
    if @is_mla
      @kv_lora_rank  = cfg.kv_lora_rank
      @d_head_k      = cfg.d_head_k
      @d_head_v      = cfg.d_head_v
      @rope_dim      = cfg.rope_dim
      @leading_dense = cfg.leading_dense
      mscale         = 1.0 + @rope_scaling.yarn_log_mul * Math.log(@rope_scaling.factor)
      @mla_kq_scale  = mscale * mscale / Math.sqrt(@d_head_k.to_f)
      puts "MLA detected: kv_lora_rank=" + @kv_lora_rank.to_s +
           " d_head_k=" + @d_head_k.to_s + " d_head_v=" + @d_head_v.to_s +
           " rope_dim=" + @rope_dim.to_s + " leading_dense=" + @leading_dense.to_s +
           " kq_scale=" + @mla_kq_scale.to_s
    end

    @gguf_handle_keepalive = gguf_handle   # prevent GC; mmap must outlive @sess
    @sess              = TinyNN.tnn_session_new(0)
    @has_untied_output = untied
    @has_qkv_bias      = qkv_bias
    @has_qk_norm       = qk_norm
    # #110: if caller didn't pre-set qk_norm_kind via the
    # attr_accessor, default to 1 (per-head shared) for backward
    # compat with the Qwen3 detection that established the qk_norm
    # path. Models that want full-Q (OLMoE / Granite) must set
    # kv.qk_norm_kind = 2 BEFORE calling realize_for_mmap.
    if @has_qk_norm && @qk_norm_kind == 0
      @qk_norm_kind = 1
    end

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

    # toy#gguf-checkpoint-reload (#153) — from-scratch checkpoints
    # written by ToyGGUFWriter store one tensor per head
    # (blk.N.attn_q.head_H.weight) rather than the fused llama.cpp
    # shape. Detect via the head_0 sentinel; the per-Q-head/K/V
    # loaders below branch on it.
    @per_head_attn = TinyNN.tnn_gguf_find_index(gguf_handle, "blk.0.attn_q.head_0.weight") >= 0
    if @per_head_attn
      puts "  per-head tensors detected (toy from-scratch checkpoint)"
    end

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

      # I-Gemma (#113): post-attention and post-FFN RMSNorm gammas
      # (Gemma 2 sandwiches each sublayer between pre+post norms).
      # Tensor names: blk.X.post_attention_norm.weight, blk.X.post_ffw_norm.weight.
      if @has_post_norms
        pa_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".post_attention_norm.weight")
        pf_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".post_ffw_norm.weight")
        blk.t_post_attn_norm_gamma = TinyNN.tnn_input_1d_persistent_mmap(@sess, @d_model, 0,
                                       TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, pa_idx))
        blk.t_post_ffn_norm_gamma  = TinyNN.tnn_input_1d_persistent_mmap(@sess, @d_model, 0,
                                       TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, pf_idx))
      end

      if @is_mla
        # --- DeepSeek-V2 MLA attention weights (whole tensors) ---
        # attn_q [d_model → n_heads·d_head_k]; per-head split into
        # q_nope(d_head_k-rope_dim) + q_rope(rope_dim) happens in the graph.
        q_idx  = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_q.weight")
        blk.t_w_q_mla = TinyNN.tnn_input_2d_persistent_mmap(@sess,
                          @n_heads * @d_head_k, @d_model,
                          TinyNN.tnn_gguf_tensor_type(gguf_handle, q_idx),
                          TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, q_idx))
        # attn_kv_a_mqa [d_model → kv_lora_rank + rope_dim]; the latent
        # c_kv(kv_lora_rank) + the shared decoupled-RoPE key k_rope(rope_dim).
        kva_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_kv_a_mqa.weight")
        blk.t_w_kv_a = TinyNN.tnn_input_2d_persistent_mmap(@sess,
                         @kv_lora_rank + @rope_dim, @d_model,
                         TinyNN.tnn_gguf_tensor_type(gguf_handle, kva_idx),
                         TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, kva_idx))
        # attn_kv_a_norm: RMSNorm over the latent [kv_lora_rank] (F32).
        kvan_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_kv_a_norm.weight")
        blk.t_kv_a_norm_gamma = TinyNN.tnn_input_1d_persistent_mmap(@sess,
                                  @kv_lora_rank, 0,
                                  TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, kvan_idx))
        # attn_kv_b [kv_lora_rank → n_heads·(d_head_nope + d_head_v)];
        # up-projects the latent to per-head k_nope(d_head_k-rope_dim) + v(d_head_v).
        kvb_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_kv_b.weight")
        d_head_nope = @d_head_k - @rope_dim
        blk.t_w_kv_b = TinyNN.tnn_input_2d_persistent_mmap(@sess,
                         @n_heads * (d_head_nope + @d_head_v), @kv_lora_rank,
                         TinyNN.tnn_gguf_tensor_type(gguf_handle, kvb_idx),
                         TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, kvb_idx))
        # attn_output [n_heads·d_head_v → d_model].
        o_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_output.weight")
        blk.t_w_o = TinyNN.tnn_input_2d_persistent_mmap(@sess,
                      @d_model, @n_heads * @d_head_v,
                      TinyNN.tnn_gguf_tensor_type(gguf_handle, o_idx),
                      TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, o_idx))
        # Asymmetric per-(query-)head K(d_head_k) / V(d_head_v) cache.
        # MLA-A reuses the existing per-head attention by caching the
        # expanded K/V (no memory win; that is MLA-B). F32 only.
        blk.t_K = [TinyNN.tnn_input_2d_f32_persistent(@sess, max_T, @d_head_k)]
        blk.t_V = [TinyNN.tnn_input_2d_f32_persistent(@sess, max_T, @d_head_v)]
        hmla = 1
        while hmla < @n_heads
          blk.t_K.push(TinyNN.tnn_input_2d_f32_persistent(@sess, max_T, @d_head_k))
          blk.t_V.push(TinyNN.tnn_input_2d_f32_persistent(@sess, max_T, @d_head_v))
          hmla = hmla + 1
        end
      else
      # M1 + #110: QK-norm gammas. Two flavors detected via shape:
      #   kind=1: Qwen3 — gamma shape [d_head], shared across heads.
      #   kind=2: OLMoE / Granite — gamma shape [d_model], applied to
      #          the full Q before head split. Allocate the full
      #          [d_model] tensor; the graph builder either does a
      #          full-Q rms_norm OR views per-head d_head slices.
      gamma_nelems = (@qk_norm_kind == 2) ? @d_model : @d_head
      if @has_qk_norm
        qn_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_q_norm.weight")
        kn_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_k_norm.weight")
        blk.t_q_norm_gamma = TinyNN.tnn_input_1d_persistent_mmap(@sess, gamma_nelems, 0,
                               TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, qn_idx))
        # K norm follows the same flavor as Q.
        k_gamma_nelems = (@qk_norm_kind == 2) ? (@n_kv * @d_head) : @d_head
        blk.t_k_norm_gamma = TinyNN.tnn_input_1d_persistent_mmap(@sess, k_gamma_nelems, 0,
                               TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, kn_idx))
      end

      # Q per-head — two layouts:
      # 1) Fused (llama.cpp): single attn_q.weight tensor; each head
      #    is a contiguous slice at offset q_base + h * head_nbytes.
      # 2) Per-head (toy from-scratch ckpt, #153): each head has its
      #    own attn_q.head_H.weight tensor with its own file offset.
      if @per_head_attn
        q0_idx  = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_q.head_0.weight")
        q0_type = TinyNN.tnn_gguf_tensor_type(gguf_handle, q0_idx)
        blk.t_w_q = [TinyNN.tnn_input_2d_persistent_mmap(@sess,
                       @d_head, @d_model, q0_type,
                       TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, q0_idx))]
        hq = 1
        while hq < @n_heads
          qh_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_q.head_" + hq.to_s + ".weight")
          blk.t_w_q.push(TinyNN.tnn_input_2d_persistent_mmap(@sess,
                           @d_head, @d_model, q0_type,
                           TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, qh_idx)))
          hq = hq + 1
        end
      else
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

      # K, V per-kv-head — same dual-layout split (#153).
      if @per_head_attn
        k0_idx  = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_k.head_0.weight")
        v0_idx  = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_v.head_0.weight")
        k_type  = TinyNN.tnn_gguf_tensor_type(gguf_handle, k0_idx)
        v_type  = TinyNN.tnn_gguf_tensor_type(gguf_handle, v0_idx)
        blk.t_w_k = [TinyNN.tnn_input_2d_persistent_mmap(@sess,
                       @d_head, @d_model, k_type,
                       TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, k0_idx))]
        blk.t_w_v = [TinyNN.tnn_input_2d_persistent_mmap(@sess,
                       @d_head, @d_model, v_type,
                       TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, v0_idx))]
        k_stride = 0  # unused in per-head branch but referenced later
        v_stride = 0
      else
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
      end
      # P5.1+P5.2: K and V allocs both follow @kv_type_*. Layout is
      # `ne=[d_head, max_T]` for both — positions on ne1, d_head on
      # ne0. Per-position writes span a contiguous d_head-vector
      # which is Q8-block-aligned at d_head=64 (=2 blocks of 32).
      # See the struct comment on :kv_type_k / :kv_type_v.
      if @kv_type_k == 8
        blk.t_K = [TinyNN.tnn_input_2d_persistent_typed(@sess, max_T, @d_head, 8)]
      else
        blk.t_K = [TinyNN.tnn_input_2d_f32_persistent(@sess, max_T, @d_head)]
      end
      if @kv_type_v == 8
        blk.t_V = [TinyNN.tnn_input_2d_persistent_typed(@sess, max_T, @d_head, 8)]
      else
        blk.t_V = [TinyNN.tnn_input_2d_f32_persistent(@sess, max_T, @d_head)]
      end
      hkv = 1
      while hkv < @n_kv
        if @per_head_attn
          kh_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_k.head_" + hkv.to_s + ".weight")
          vh_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_v.head_" + hkv.to_s + ".weight")
          blk.t_w_k.push(TinyNN.tnn_input_2d_persistent_mmap(@sess,
                           @d_head, @d_model, k_type,
                           TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, kh_idx)))
          blk.t_w_v.push(TinyNN.tnn_input_2d_persistent_mmap(@sess,
                           @d_head, @d_model, v_type,
                           TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, vh_idx)))
        else
          blk.t_w_k.push(TinyNN.tnn_input_2d_persistent_mmap(@sess,
                           @d_head, @d_model, k_type,
                           k_off_base + hkv * k_stride))
          blk.t_w_v.push(TinyNN.tnn_input_2d_persistent_mmap(@sess,
                           @d_head, @d_model, v_type,
                           v_off_base + hkv * v_stride))
        end
        if @kv_type_k == 8
          blk.t_K.push(TinyNN.tnn_input_2d_persistent_typed(@sess, max_T, @d_head, 8))
        else
          blk.t_K.push(TinyNN.tnn_input_2d_f32_persistent(@sess, max_T, @d_head))
        end
        if @kv_type_v == 8
          blk.t_V.push(TinyNN.tnn_input_2d_persistent_typed(@sess, max_T, @d_head, 8))
        else
          blk.t_V.push(TinyNN.tnn_input_2d_f32_persistent(@sess, max_T, @d_head))
        end
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

      # O — full 2D weight, no per-head slicing.
      # M1.1: o_proj maps [n_heads * d_head] → [d_model]. For models
      # where d_head = d_model / n_heads (SmolLM2 / Llama / Qwen2.5)
      # these are equal; for Qwen3 with explicit head_dim=128 they
      # differ (n_heads * d_head = 2048, d_model = 1024).
      o_idx    = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_output.weight")
      blk.t_w_o    = TinyNN.tnn_input_2d_persistent_mmap(@sess, @d_model, @n_heads * @d_head,
                       TinyNN.tnn_gguf_tensor_type(gguf_handle, o_idx),
                       TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, o_idx))
      end   # !is_mla (MLA allocated its own q/kv_a/kv_b/o + K/V cache above)

      # FFN: per-layer dispatch. DeepSeek-V2's leading_dense_block_count
      # makes the first @leading_dense layers dense even though the model
      # is MoE; @leading_dense=0 for every non-deepseek model, so the
      # condition is unchanged there.
      if @is_moe && li >= @leading_dense
        # M2.3: MoE FFN. Per-expert weight matrices are stacked along
        # ne2 in the GGUF (llama.cpp convention):
        #   ffn_gate_inp.weight : ne=[d_model, n_experts]
        #   ffn_gate_exps.weight: ne=[d_model, d_ff,    n_experts]
        #   ffn_up_exps.weight  : ne=[d_model, d_ff,    n_experts]
        #   ffn_down_exps.weight: ne=[d_ff,    d_model, n_experts]
        # All mmap'd in place — Mixtral-8x7B Q4_K_M (26 GB) loads without
        # any RAM copy.
        router_idx    = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_gate_inp.weight")
        gate_exps_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_gate_exps.weight")
        up_exps_idx   = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_up_exps.weight")
        down_exps_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_down_exps.weight")
        # #112 (RESOLVED): K-quant MoE experts work. The old warning here
        # blamed ggml's mul_mat_id kernel for the OLMoE-Q4_K_M corruption,
        # but the op was always correct for K-quants (verified by op-level
        # and real-bytes reproducers in tinynn/ggml1506_*). The actual bug
        # was head_nbytes() returning 0 for K-quant ATTENTION weights,
        # collapsing every head onto head 0 — fixed there. K-quant expert
        # stacks (gate/up/down, including OLMoE's mixed q4_K+q6_K down_exps)
        # load and run coherently. See docs/notes/mul_mat_id_quants.md.
        # Expert intermediate size = the gate_exps tensor's ne[1], read
        # straight from the GGUF rather than the dense feed_forward_length:
        # MoE models (Qwen3-30B-A3B: expert_ff=768 vs dense d_ff=6144) size
        # experts independently. OLMoE's expert_ff == its d_ff, so this is
        # byte-identical there; robust for any family without a metadata key.
        moe_d_ff = TinyNN.tnn_gguf_tensor_ne(gguf_handle, gate_exps_idx, 1)
        @moe_norm_topk = GGUFLoad.moe_norm_topk?(gguf_handle)
        blk.t_w_router    = TinyNN.tnn_input_2d_persistent_mmap(@sess,
                              @n_experts, @d_model,
                              TinyNN.tnn_gguf_tensor_type(gguf_handle, router_idx),
                              TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, router_idx))
        blk.t_w_gate_exps = TinyNN.tnn_input_3d_persistent_mmap(@sess,
                              @d_model, moe_d_ff, @n_experts,
                              TinyNN.tnn_gguf_tensor_type(gguf_handle, gate_exps_idx),
                              TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, gate_exps_idx))
        blk.t_w_up_exps   = TinyNN.tnn_input_3d_persistent_mmap(@sess,
                              @d_model, moe_d_ff, @n_experts,
                              TinyNN.tnn_gguf_tensor_type(gguf_handle, up_exps_idx),
                              TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, up_exps_idx))
        blk.t_w_down_exps = TinyNN.tnn_input_3d_persistent_mmap(@sess,
                              moe_d_ff, @d_model, @n_experts,
                              TinyNN.tnn_gguf_tensor_type(gguf_handle, down_exps_idx),
                              TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, down_exps_idx))

        # P2: always-on shared experts — present when the block carries
        # ffn_gate_shexp (Qwen2-MoE / DeepSeek-V2 / GLM-MoE). A plain SwiGLU
        # of intermediate size sh_ff = gate_shexp ne[1] (Qwen1.5-MoE: 5632).
        # gguf ne: gate/up_shexp [d_model, sh_ff] → mmap(rows=sh_ff, cols=d_model);
        # down_shexp [sh_ff, d_model] → mmap(rows=d_model, cols=sh_ff).
        gate_shexp_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_gate_shexp.weight")
        if gate_shexp_idx >= 0
          up_shexp_idx   = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_up_shexp.weight")
          down_shexp_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_down_shexp.weight")
          sh_ff = TinyNN.tnn_gguf_tensor_ne(gguf_handle, gate_shexp_idx, 1)
          blk.t_w_gate_shexp = TinyNN.tnn_input_2d_persistent_mmap(@sess,
                                 sh_ff, @d_model,
                                 TinyNN.tnn_gguf_tensor_type(gguf_handle, gate_shexp_idx),
                                 TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, gate_shexp_idx))
          blk.t_w_up_shexp   = TinyNN.tnn_input_2d_persistent_mmap(@sess,
                                 sh_ff, @d_model,
                                 TinyNN.tnn_gguf_tensor_type(gguf_handle, up_shexp_idx),
                                 TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, up_shexp_idx))
          blk.t_w_down_shexp = TinyNN.tnn_input_2d_persistent_mmap(@sess,
                                 @d_model, sh_ff,
                                 TinyNN.tnn_gguf_tensor_type(gguf_handle, down_shexp_idx),
                                 TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, down_shexp_idx))
          # Optional Qwen-style sigmoid gate on the shared output (1D [d_model]).
          # Absent in DeepSeek-V2 (ungated shared experts) → stays null.
          shexp_gate_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_gate_inp_shexp.weight")
          if shexp_gate_idx >= 0
            blk.t_w_shexp_gate = TinyNN.tnn_input_1d_persistent_mmap(@sess,
                                   @d_model,
                                   TinyNN.tnn_gguf_tensor_type(gguf_handle, shexp_gate_idx),
                                   TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, shexp_gate_idx))
          end
        end
      else
        gate_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_gate.weight")
        up_idx   = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_up.weight")
        down_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_down.weight")
        blk.t_w_gate = TinyNN.tnn_input_2d_persistent_mmap(@sess, @d_ff, @d_model,
                         TinyNN.tnn_gguf_tensor_type(gguf_handle, gate_idx),
                         TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, gate_idx))
        blk.t_w_up   = TinyNN.tnn_input_2d_persistent_mmap(@sess, @d_ff, @d_model,
                         TinyNN.tnn_gguf_tensor_type(gguf_handle, up_idx),
                         TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, up_idx))
        blk.t_w_down = TinyNN.tnn_input_2d_persistent_mmap(@sess, @d_model, @d_ff,
                         TinyNN.tnn_gguf_tensor_type(gguf_handle, down_idx),
                         TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, down_idx))
      end

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
    # P5.1: skip K zero-init when K is Q8_0. upload_row_major writes
    # F32 row-major bytes which would corrupt a Q8 tensor's quantization
    # blocks. The K cache is read only at positions [0, pos+1], and
    # every position is written before it's read, so unset trailing
    # positions are never observed — zero-init is paranoia and safe
    # to skip for Q8. P5.2 flipped V to mirror K's layout, so V's
    # zero-init Mat now has the same shape as K's, and the same Q8
    # skip rule applies.
    # MLA caches are asymmetric (K=d_head_k, V=d_head_v) and span
    # n_heads buffers; non-MLA caches are symmetric (@d_head) over @n_kv.
    if @is_mla
      kv_zero_k = Mat.new(max_T, @d_head_k)
      kv_zero_v = Mat.new(max_T, @d_head_v)
      n_kv_buf  = @n_heads
    else
      kv_zero_k = Mat.new(max_T, @d_head)
      kv_zero_v = kv_zero_k
      n_kv_buf  = @n_kv
    end
    li = 0
    while li < @n_layers
      blk_f = @kv_blocks_ffi[li]
      hkv = 0
      while hkv < n_kv_buf
        if @kv_type_k != 8
          TinyNN.upload_row_major(@sess, blk_f.t_K[hkv], kv_zero_k)
        end
        if @kv_type_v != 8
          TinyNN.upload_row_major(@sess, blk_f.t_V[hkv], kv_zero_v)
        end
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
  # Caller must have `require_relative "toy/models/toy_smollm2_loader"` at the
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
      realize_for_mmap(gguf, cfg, max_T, flags.untied, flags.qkv_bias, flags.qk_norm)
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
  # tensor into n_heads contiguous Dh×D blocks. A per-head slice is
  # d_head rows of d_model elements, so the stride is d_head row-sizes.
  #
  # tnn_row_size delegates to ggml_row_size, which is correct for EVERY
  # type — F32, Q8_0, and the K-quants (Q4_K/Q5_K/Q6_K). The previous
  # hand-coded F32/Q8_0-only branches returned 0 for any other type,
  # which silently made the per-head offset `off_base + hq*0 == off_base`
  # — i.e. every attention head read head 0's weight slice. That
  # collapsed multi-head attention on K-quant MoE models (forced down the
  # realize_for_mmap path), compounding across layers into degenerate
  # output. This was misdiagnosed as a ggml mul_mat_id K-quant bug
  # (ggml#1506); it was ours. Block alignment holds because each row is a
  # whole number of quant blocks (requires d_model % block == 0, which
  # the per-head tnn_input_2d_persistent_mmap also enforces via ne0).
  def head_nbytes(ggml_type, d_head, d_model)
    rs = TinyNN.tnn_row_size(ggml_type, d_model)
    if rs <= 0
      # Fail loud per the never-mask rule: a 0 stride would collapse all
      # heads. tnn_row_size only returns 0 on a bad type/shape.
      puts "FATAL: head_nbytes got row_size<=0 for ggml_type=" +
           ggml_type.to_s + " d_model=" + d_model.to_s +
           " — per-head attention stride would collapse. Aborting."
      exit 1
    end
    d_head * rs
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
      # weights quantizable; K/V cache buffers follow @kv_type_*
      # (P5.1 K, P5.2 V); biases stay F32.
      blk.t_w_k = [alloc_2d_w(d_head, d_model)]
      blk.t_w_v = [alloc_2d_w(d_head, d_model)]
      # P5.1: Q8 K alloc when enabled (see realize_for_mmap parallel path).
      if @kv_type_k == 8
        blk.t_K = [TinyNN.tnn_input_2d_persistent_typed(@sess, max_T, d_head, 8)]
      else
        blk.t_K = [TinyNN.tnn_input_2d_f32_persistent(@sess, max_T, d_head)]
      end
      # P5.2: V now mirrors K's layout (ne=[d_head, max_T]).
      if @kv_type_v == 8
        blk.t_V = [TinyNN.tnn_input_2d_persistent_typed(@sess, max_T, d_head, 8)]
      else
        blk.t_V = [TinyNN.tnn_input_2d_f32_persistent(@sess, max_T, d_head)]
      end
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
        if @kv_type_k == 8
          blk.t_K.push(TinyNN.tnn_input_2d_persistent_typed(@sess, max_T, d_head, 8))
        else
          blk.t_K.push(TinyNN.tnn_input_2d_f32_persistent(@sess, max_T, d_head))
        end
        if @kv_type_v == 8
          blk.t_V.push(TinyNN.tnn_input_2d_persistent_typed(@sess, max_T, d_head, 8))
        else
          blk.t_V.push(TinyNN.tnn_input_2d_f32_persistent(@sess, max_T, d_head))
        end
        if qkv_bias
          blk.t_b_k.push(TinyNN.tnn_input_1d_f32_persistent(@sess, d_head))
          blk.t_b_v.push(TinyNN.tnn_input_1d_f32_persistent(@sess, d_head))
        end
        hkv = hkv + 1
      end

      blk.t_w_o    = alloc_2d_w(d_model, @n_heads * @d_head)
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
    # P5.1+P5.2: row size for K and V. F32 → d_head*4; Q8_0 →
    # ggml_row_size(Q8_0, d_head) (block 32 × 34 bytes; 68 at d_head=64).
    # V is in the same layout as K post-P5.2 so the math is symmetric.
    bytes_d_head_k = @kv_type_k == 8 ? TinyNN.tnn_row_size(8, d_head) : bytes_d_head
    bytes_d_head_v = @kv_type_v == 8 ? TinyNN.tnn_row_size(8, d_head) : bytes_d_head

    # Inputs: token id + RoPE position. Both length 1.
    t_token_id  = TinyNN.tnn_input_1d_i32(@sess, 1)
    t_pos       = TinyNN.tnn_input_1d_i32_ctx(@sess, 1)

    t_x = TinyNN.tnn_get_rows(@sess, @t_token_embed, t_token_id)   # ne=[d_model, 1]
    # I-Gemma (#113): Gemma 2 scales token embeddings by sqrt(d_model)
    # post-lookup. Non-Gemma archs use @embed_scale = 1.0 (no-op
    # branch). The scalar is computed at flag-detection time so we
    # don't pay a Math.sqrt landmine in the hot path.
    if @embed_scale != 1.0
      t_x = TinyNN.tnn_scale(@sess, t_x, @embed_scale)
    end
    t_x = trace_tap("embed", t_x)

    li = 0
    while li < @n_layers
      if @is_mla
        t_x = build_mla_block_step(t_x, @kv_blocks_ffi[li], t_pos, pos, eps, li)
      else
        t_x = build_block_step(t_x, @kv_blocks_ffi[li], t_pos, pos,
                                scale, eps, bytes_d_head, bytes_d_head_k,
                                bytes_d_head_v, bytes_max_T, li)
      end
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
    # I-Gemma (#113): final logit soft-cap. Gemma 2 applies
    # tanh(logits / final_softcap) * final_softcap to the output
    # logits before argmax / sampling. No-op for other models.
    if @final_softcap > 0.0
      t_kv_logits = TinyNN.tnn_scale(@sess, t_kv_logits, 1.0 / @final_softcap)
      t_kv_logits = TinyNN.tnn_tanh(@sess, t_kv_logits)
      t_kv_logits = TinyNN.tnn_scale(@sess, t_kv_logits, @final_softcap)
    end
    TinyNN.tnn_set_output(t_kv_logits)
    SmolLM2KVStepResult.new(t_token_id, t_pos, t_kv_logits)
  end

  def build_block_step(t_x, blk, t_pos, pos, scale, eps,
                        bytes_d_head, bytes_d_head_k, bytes_d_head_v,
                        bytes_max_T, layer_idx)
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
      # M1 + #110: QK-norm. Two flavors:
      #   kind=1 (Qwen3): blk.t_k_norm_gamma is [d_head], shared
      #     across all KV heads; pass directly.
      #   kind=2 (OLMoE / Granite, per-head approximation):
      #     blk.t_k_norm_gamma is [n_kv * d_head] = [d_model_kv];
      #     view the per-head [d_head] slice at byte offset
      #     hkv*d_head*4. This computes per-head variance (not the
      #     true full-Q-vector variance) but applies the correct
      #     per-element gamma scaling. Cheap and close-enough for
      #     models where per-head magnitudes are similar (which they
      #     typically are for projections of a single input).
      if @has_qk_norm
        if @qk_norm_kind == 2
          k_gamma_view = TinyNN.tnn_view_1d(@sess, blk.t_k_norm_gamma,
                                              @d_head, hkv * @d_head * 4)
          t_k_pre = TinyNN.tnn_rms_norm(@sess, t_k_pre, k_gamma_view, @rms_eps)
        else
          t_k_pre = TinyNN.tnn_rms_norm(@sess, t_k_pre, blk.t_k_norm_gamma, @rms_eps)
        end
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

      # P5.1+P5.2: K and V both use the same per-position write pattern.
      # bytes_d_head_{k,v} reflect each cache's dtype (F32 → d_head*4,
      # Q8_0 → type-aware row size from tnn_row_size). cpy quantizes
      # f32 source → Q8 destination automatically when types differ.
      t_K_slot = TinyNN.tnn_view_2d(@sess, blk.t_K[hkv],
                                      @d_head, 1, bytes_d_head_k, pos * bytes_d_head_k)
      t_cpy_k = TinyNN.tnn_cpy(@sess, t_k_rot, t_K_slot)
      t_V_slot = TinyNN.tnn_view_2d(@sess, blk.t_V[hkv],
                                      @d_head, 1, bytes_d_head_v, pos * bytes_d_head_v)
      t_cpy_v = TinyNN.tnn_cpy(@sess, t_v_new, t_V_slot)
      TinyNN.tnn_add_to_graph(@sess, t_cpy_k)
      TinyNN.tnn_add_to_graph(@sess, t_cpy_v)
      hkv = hkv + 1
    end

    # --- per-Q-head attention ---
    t_head_out0 = build_attention_qhead_step(t_h, blk, 0, t_pos, pos,
                                              scale, bytes_d_head, bytes_d_head_k,
                                              bytes_d_head_v, bytes_max_T, tag, true,
                                              layer_idx)
    t_head_outs = [t_head_out0]
    hq = 1
    while hq < @n_heads
      t_head_outs.push(build_attention_qhead_step(t_h, blk, hq, t_pos, pos,
                                                    scale, bytes_d_head, bytes_d_head_k,
                                                    bytes_d_head_v, bytes_max_T, tag, false,
                                                    layer_idx))
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
    # I-Gemma (#113): post-attention RMSNorm applied to the attention
    # output BEFORE the residual add. Gemma 2's sandwich structure:
    #   pre_norm(x) → attention → post_norm → residual + …
    # No-op when has_post_norms is false (every non-Gemma arch).
    if @has_post_norms
      t_out_proj = TinyNN.tnn_rms_norm(@sess, t_out_proj, blk.t_post_attn_norm_gamma, eps)
      t_out_proj = trace_tap(tag + "post_attn_norm", t_out_proj)
    end
    t_x_attn   = TinyNN.tnn_add(@sess, t_x, t_out_proj)
    t_x_attn   = trace_tap(tag + "post_attn", t_x_attn)

    # --- FFN ---
    t_h2     = TinyNN.tnn_rms_norm(@sess, t_x_attn, blk.t_rn2_gamma, eps)
    t_h2     = trace_tap(tag + "rn2", t_h2)

    if @is_moe
      t_dn = build_moe_ffn(blk, t_h2, tag)
    else
      # --- SwiGLU FFN (dense) ---
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
    end

    # I-Gemma (#113): post-FFN RMSNorm on the FFN output before the
    # residual add. Same pattern as the post-attn norm above.
    if @has_post_norms
      t_dn = TinyNN.tnn_rms_norm(@sess, t_dn, blk.t_post_ffn_norm_gamma, eps)
      t_dn = trace_tap(tag + "post_ffn_norm", t_dn)
    end
    t_post_ffn = TinyNN.tnn_add(@sess, t_x_attn, t_dn)
    trace_tap(tag + "post_ffn", t_post_ffn)
  end

  # DeepSeek-V2 MLA (Multi-head Latent Attention) decode block. Same
  # residual structure as build_block_step (rn1 → attention → +residual →
  # rn2 → FFN → +residual) but the attention sublayer projects the hidden
  # state to a small latent c_kv + a shared decoupled-RoPE key, up-projects
  # the latent per head to k_nope/v, and runs per-head attention with the
  # YaRN-mscale softmax scale (@mla_kq_scale). MLA-A reuses the per-head
  # KV cache (expanded K[d_head_k]/V[d_head_v]) — no memory win yet.
  # FFN: dense for the first @leading_dense layers, MoE otherwise.
  def build_mla_block_step(t_x, blk, t_pos, pos, eps, layer_idx)
    tag = "L" + layer_idx.to_s + "."
    bf  = 4                              # f32 bytes
    d_head_nope    = @d_head_k - @rope_dim          # 192 - 64 = 128
    head_kv_stride = d_head_nope + @d_head_v         # 128 + 128 = 256

    t_h = TinyNN.tnn_rms_norm(@sess, t_x, blk.t_rn1_gamma, eps)
    t_h = trace_tap(tag + "rn1", t_h)

    # Q: full projection; per-head split into q_nope/q_rope in the loop.
    t_q_all = TinyNN.tnn_matmul(@sess, blk.t_w_q_mla, t_h)    # [n_heads*d_head_k, 1]

    # KV-a: latent c_kv (RMSNorm'd) + the shared decoupled-RoPE key k_rope.
    t_kv_a   = TinyNN.tnn_matmul(@sess, blk.t_w_kv_a, t_h)    # [kv_lora_rank+rope_dim, 1]
    t_c_kv   = TinyNN.tnn_view_1d(@sess, t_kv_a, @kv_lora_rank, 0)
    t_c_kv   = TinyNN.tnn_rms_norm(@sess, t_c_kv, blk.t_kv_a_norm_gamma, eps)
    t_k_rope = TinyNN.tnn_view_1d(@sess, t_kv_a, @rope_dim, @kv_lora_rank * bf)
    # DeepSeek-V2 decoupled-RoPE uses NORM mode (0; GPT-J interleaved pairs),
    # NOT the NEOX (2) the rest of toy uses — the GGUF q_pe/k_pe slices are
    # laid out for it. Empirically NEOX garbles fact recall; NORM matches
    # llama.cpp's reference top-1 token.
    t_k_rope_rot = TinyNN.tnn_rope_ext_yarn(@sess, t_k_rope, t_pos, @rope_dim,
                     0, @rope_scaling.orig_max_pos,
                     @rope_base, @rope_scaling.freq_scale,
                     @rope_scaling.ext_factor, @rope_scaling.attn_factor,
                     @rope_scaling.beta_fast, @rope_scaling.beta_slow,
                     TinyNN.tnn_null_ptr)

    # KV-b: up-project the normed latent to per-head k_nope + v.
    t_kv_b = TinyNN.tnn_matmul(@sess, blk.t_w_kv_b, t_c_kv)   # [n_heads*256, 1]

    bytes_dk = @d_head_k * bf
    bytes_dv = @d_head_v * bf
    t_head_outs = [TinyNN.tnn_null_ptr]; t_head_outs.pop
    h = 0
    while h < @n_heads
      q_base   = h * @d_head_k
      t_q_nope = TinyNN.tnn_view_1d(@sess, t_q_all, d_head_nope, q_base * bf)
      t_q_rope = TinyNN.tnn_view_1d(@sess, t_q_all, @rope_dim, (q_base + d_head_nope) * bf)
      t_q_rope_rot = TinyNN.tnn_rope_ext_yarn(@sess, t_q_rope, t_pos, @rope_dim,
                       0, @rope_scaling.orig_max_pos,
                       @rope_base, @rope_scaling.freq_scale,
                       @rope_scaling.ext_factor, @rope_scaling.attn_factor,
                       @rope_scaling.beta_fast, @rope_scaling.beta_slow,
                       TinyNN.tnn_null_ptr)
      t_q = TinyNN.tnn_concat(@sess, t_q_nope, t_q_rope_rot, 0)   # [d_head_k, 1]

      kv_base  = h * head_kv_stride
      t_k_nope = TinyNN.tnn_view_1d(@sess, t_kv_b, d_head_nope, kv_base * bf)
      t_v      = TinyNN.tnn_view_1d(@sess, t_kv_b, @d_head_v, (kv_base + d_head_nope) * bf)
      t_k = TinyNN.tnn_concat(@sess, t_k_nope, t_k_rope_rot, 0)   # [d_head_k, 1]

      # Cache writes (K=d_head_k, V=d_head_v) at this position.
      t_K_slot = TinyNN.tnn_view_2d(@sess, blk.t_K[h], @d_head_k, 1, bytes_dk, pos * bytes_dk)
      TinyNN.tnn_add_to_graph(@sess, TinyNN.tnn_cpy(@sess, t_k, t_K_slot))
      t_V_slot = TinyNN.tnn_view_2d(@sess, blk.t_V[h], @d_head_v, 1, bytes_dv, pos * bytes_dv)
      TinyNN.tnn_add_to_graph(@sess, TinyNN.tnn_cpy(@sess, t_v, t_V_slot))

      # Attention over cached positions [0, pos].
      hist     = pos + 1
      t_K_hist = TinyNN.tnn_view_2d(@sess, blk.t_K[h], @d_head_k, hist, bytes_dk, 0)
      t_V_hist = TinyNN.tnn_view_2d(@sess, blk.t_V[h], @d_head_v, hist, bytes_dv, 0)
      t_scores = TinyNN.tnn_matmul(@sess, t_K_hist, t_q)          # [hist, 1]
      t_scaled = TinyNN.tnn_scale(@sess, t_scores, @mla_kq_scale)
      t_attn   = TinyNN.tnn_softmax(@sess, t_scaled)
      if layer_idx == 0 && h == 0
        t_scaled = trace_tap(tag + "scores", t_scaled)
        t_attn   = trace_tap(tag + "softmax", t_attn)
      end
      t_V_T    = TinyNN.tnn_transpose(@sess, t_V_hist)            # [hist, d_head_v]
      t_head   = TinyNN.tnn_matmul(@sess, t_V_T, t_attn)          # [d_head_v, 1]
      t_head_outs.push(t_head)
      h = h + 1
    end

    t_concat = t_head_outs[0]
    h = 1
    while h < @n_heads
      t_concat = TinyNN.tnn_concat(@sess, t_concat, t_head_outs[h], 0)
      h = h + 1
    end
    t_concat   = trace_tap(tag + "concat", t_concat)
    t_out_proj = TinyNN.tnn_matmul(@sess, blk.t_w_o, t_concat)    # [d_model, 1]
    t_x_attn   = TinyNN.tnn_add(@sess, t_x, t_out_proj)
    t_x_attn   = trace_tap(tag + "post_attn", t_x_attn)

    # FFN: dense for the leading dense layers, MoE otherwise.
    t_h2 = TinyNN.tnn_rms_norm(@sess, t_x_attn, blk.t_rn2_gamma, eps)
    if @is_moe && layer_idx >= @leading_dense
      t_dn = build_moe_ffn(blk, t_h2, tag)
    else
      t_gate  = TinyNN.tnn_matmul(@sess, blk.t_w_gate, t_h2)
      t_up    = TinyNN.tnn_matmul(@sess, blk.t_w_up,   t_h2)
      t_silug = TinyNN.tnn_silu(@sess, t_gate)
      t_gated = TinyNN.tnn_mul(@sess, t_silug, t_up)
      t_dn    = TinyNN.tnn_matmul(@sess, blk.t_w_down, t_gated)
    end
    t_post_ffn = TinyNN.tnn_add(@sess, t_x_attn, t_dn)
    trace_tap(tag + "post_ffn", t_post_ffn)
  end

  # M2.3: Mixtral / Qwen-MoE routed FFN. Ports the validated graph from
  # tinynn/ab_smoke_moe_ffn into the production decode path. Shapes:
  #   t_h2          [d_model, 1]                    input (post-norm)
  #   router_logits [n_experts, 1]                  matmul(w_router, h2)
  #   probs         [n_experts, 1]                  softmax(logits)
  #   top_idx       [n_experts_used, 1]             top_k(probs)
  #   weights       [1, n_experts_used, 1]          get_rows(reshape_3d(probs,1,n_exp,1), top_idx)
  #   e_gate / e_up [d_ff,    n_experts_used, 1]    mul_mat_id(...exps, h2, top_idx)
  #   e_down        [d_model, n_experts_used, 1]    after weight × sum
  #
  # The (mul/transpose/sum_rows/reshape) sum-across-K is the same trick
  # the smoke uses; ggml has no axis-1 reduce primitive.
  def build_moe_ffn(blk, t_h2, tag)
    t_logits     = TinyNN.tnn_matmul(@sess, blk.t_w_router, t_h2)        # ne=[n_exp, 1]
    t_logits     = trace_tap(tag + "moe_logits", t_logits)
    t_probs      = TinyNN.tnn_softmax(@sess, t_logits)                   # ne=[n_exp, 1]
    t_top_idx    = TinyNN.tnn_top_k(@sess, t_probs, @n_experts_used)     # ne=[K, 1]
    t_probs_3d   = TinyNN.tnn_reshape_3d(@sess, t_probs, 1, @n_experts, 1)
    t_w_route    = TinyNN.tnn_get_rows(@sess, t_probs_3d, t_top_idx)     # ne=[1, K, 1]

    # norm_topk_prob (Qwen3-MoE / DeepSeek): renormalize the K selected weights
    # to sum 1. Mixtral/OLMoE leave this false → byte-identical to before. Sum
    # over K via reshape→sum_rows (ne0-only reduce), then broadcast-divide.
    if @moe_norm_topk
      t_w_flat   = TinyNN.tnn_reshape_2d(@sess, t_w_route, @n_experts_used, 1)  # [K, 1]
      t_w_ssum   = TinyNN.tnn_sum_rows(@sess, t_w_flat)                         # [1, 1]
      t_w_ssum3  = TinyNN.tnn_reshape_3d(@sess, t_w_ssum, 1, 1, 1)              # [1, 1, 1]
      t_w_route  = TinyNN.tnn_div(@sess, t_w_route, t_w_ssum3)                  # broadcast over K
    end

    t_e_gate     = TinyNN.tnn_mul_mat_id(@sess, blk.t_w_gate_exps, t_h2, t_top_idx)
    t_e_up       = TinyNN.tnn_mul_mat_id(@sess, blk.t_w_up_exps,   t_h2, t_top_idx)
    t_e_silu     = TinyNN.tnn_silu(@sess, t_e_gate)
    t_e_gated    = TinyNN.tnn_mul(@sess, t_e_silu, t_e_up)               # ne=[d_ff, K, 1]
    t_e_down     = TinyNN.tnn_mul_mat_id(@sess, blk.t_w_down_exps, t_e_gated, t_top_idx)
    t_e_down     = trace_tap(tag + "moe_e_down", t_e_down)               # ne=[d_model, K, 1]

    # Broadcast weights over d_model: [d_model, K, 1] × [1, K, 1] → [d_model, K, 1].
    t_weighted   = TinyNN.tnn_mul(@sess, t_e_down, t_w_route)

    # Sum across K (axis 1). Reshape to 2D (T=1 collapses), transpose
    # [d_model, K] → [K, d_model], sum_rows along ne0=K → [1, d_model],
    # reshape back to [d_model, 1].
    t_weighted_2d = TinyNN.tnn_reshape_2d(@sess, t_weighted, @d_model, @n_experts_used)
    t_weighted_T  = TinyNN.tnn_transpose(@sess, t_weighted_2d)
    t_summed_T    = TinyNN.tnn_sum_rows(@sess, t_weighted_T)             # ne=[1, d_model]
    t_dn          = TinyNN.tnn_reshape_2d(@sess, t_summed_T, @d_model, 1)

    # P2: add the always-on shared expert (Qwen2-MoE / DeepSeek-V2 / GLM-MoE).
    # A plain SwiGLU on the *_shexp weights; Qwen2-MoE further scales it by a
    # per-token sigmoid gate (ffn_gate_inp_shexp). Null handles → skipped.
    if blk.t_w_gate_shexp != TinyNN.tnn_null_ptr
      t_sh_gate  = TinyNN.tnn_matmul(@sess, blk.t_w_gate_shexp, t_h2)    # [sh_ff, 1]
      t_sh_up    = TinyNN.tnn_matmul(@sess, blk.t_w_up_shexp,   t_h2)    # [sh_ff, 1]
      t_sh_silu  = TinyNN.tnn_silu(@sess, t_sh_gate)
      t_sh_gated = TinyNN.tnn_mul(@sess, t_sh_silu, t_sh_up)
      t_sh_dn    = TinyNN.tnn_matmul(@sess, blk.t_w_down_shexp, t_sh_gated)  # [d_model, 1]
      if blk.t_w_shexp_gate != TinyNN.tnn_null_ptr
        t_sh_logit = TinyNN.tnn_matmul(@sess, blk.t_w_shexp_gate, t_h2)  # [1, 1]
        t_sh_g     = TinyNN.tnn_sigmoid(@sess, t_sh_logit)
        t_sh_dn    = TinyNN.tnn_mul(@sess, t_sh_dn, t_sh_g)              # broadcast scalar
      end
      t_dn = TinyNN.tnn_add(@sess, t_dn, t_sh_dn)
    end
    trace_tap(tag + "moe_out", t_dn)
  end

  # One query head. Uses the (already-written) K and V of the
  # corresponding KV head — index = hq / group_size. `tag` is the
  # "L<i>." layer prefix; `tap_this_head` is true only for head 0 so we
  # don't multiply taps by n_heads in trace mode.
  def build_attention_qhead_step(t_h, blk, hq, t_pos, pos, scale,
                                  bytes_d_head, bytes_d_head_k, bytes_d_head_v,
                                  bytes_max_T, tag, tap_this_head,
                                  layer_idx)
    hkv = hq / @group_size

    # I-Gemma (#113): per-layer SWA toggle. Gemma 2 alternates layers
    # between full attention and sliding-window. When @swa_alternates
    # is true, only EVEN layers see the SWA window; odd layers get
    # effectively full attention (window = 0 ⇒ hist_count = pos+1).
    # Non-Gemma archs: @swa_alternates is false; all layers apply
    # @swa_window uniformly (or 0 for no-SWA models).
    swa_for_this_layer = @swa_window
    if @swa_alternates && layer_idx.odd?
      swa_for_this_layer = 0
    end

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
      if @qk_norm_kind == 2
        # OLMoE / Granite per-head gamma slice (see build_block_step's
        # K-norm comment). The gamma tensor is [d_model]; head hq's
        # slice lives at byte offset hq*d_head*4.
        q_gamma_view = TinyNN.tnn_view_1d(@sess, blk.t_q_norm_gamma,
                                            @d_head, hq * @d_head * 4)
        t_q_pre = TinyNN.tnn_rms_norm(@sess, t_q_pre, q_gamma_view, @rms_eps)
      else
        t_q_pre = TinyNN.tnn_rms_norm(@sess, t_q_pre, blk.t_q_norm_gamma, @rms_eps)
      end
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

    # M3 + I-Gemma: sliding-window attention. When swa_for_this_layer
    # > 0, restrict the K/V view to the last `min(pos+1, swa_window)`
    # positions. swa_for_this_layer differs from @swa_window only
    # when @swa_alternates is set (Gemma 2's even/odd layer pattern).
    if swa_for_this_layer > 0 && (pos + 1) > swa_for_this_layer
      hist_start = pos + 1 - swa_for_this_layer
      hist_count = swa_for_this_layer
    else
      hist_start = 0
      hist_count = pos + 1
    end
    # P5.1+P5.2: K and V views share the same byte-stride math.
    # ggml_mul_mat dequantizes Q8 source on the fly when reads happen.
    t_K_hist = TinyNN.tnn_view_2d(@sess, blk.t_K[hkv],
                                    @d_head, hist_count, bytes_d_head_k,
                                    hist_start * bytes_d_head_k)
    # P5.2: V is now ne=[d_head, max_T] (positions on ne1, mirror of K).
    # The history view at [d_head, hist_count] is what flash_attn_ext
    # expects natively — no transpose-cont in the flash path now.
    t_V_hist = TinyNN.tnn_view_2d(@sess, blk.t_V[hkv],
                                    @d_head, hist_count, bytes_d_head_v,
                                    hist_start * bytes_d_head_v)

    if @use_flash_attn
      # P4.1+P5.2: fused softmax(Q·Kᵀ·scale + mask)·V via
      # ggml_flash_attn_ext. Reshape Q/K/V to the 3D shapes
      # flash_attn_ext expects (ne[3] defaults to 1 so we don't need
      # a fourth dim). V's layout is already correct post-P5.2 — no
      # transpose needed.
      t_q_3d   = TinyNN.tnn_reshape_3d(@sess, t_q,      @d_head, 1, 1)
      t_K_3d   = TinyNN.tnn_reshape_3d(@sess, t_K_hist, @d_head, hist_count, 1)
      t_V_3d   = TinyNN.tnn_reshape_3d(@sess, t_V_hist, @d_head, hist_count, 1)
      # I-Gemma (#113): pass logit soft-cap to flash_attn_ext. The
      # kernel applies tanh(x/softcap)*softcap to attention logits
      # internally. 0.0 disables (every non-Gemma model).
      t_out_4d = TinyNN.tnn_flash_attn_ext(@sess, t_q_3d, t_K_3d, t_V_3d, nil,
                                            scale, 0.0, @attn_softcap)
      # Output ne=[d_head, n_head=1, T_q=1, batch=1]; collapse to 2D.
      t_head = TinyNN.tnn_reshape_2d(@sess, t_out_4d, @d_head, 1)
      if tap_this_head
        t_head = trace_tap(tag + "head0_flash", t_head)
      end
      return t_head
    end

    t_scores = TinyNN.tnn_matmul(@sess, t_K_hist, t_q)
    if tap_this_head
      t_scores = trace_tap(tag + "scores", t_scores)
    end
    t_scaled = TinyNN.tnn_scale(@sess, t_scores, scale)
    # I-Gemma (#113): logit soft-cap in the non-flash path.
    #   y = softcap * tanh(x / softcap)
    # Composed via two scales + tanh. No-op when @attn_softcap == 0.
    if @attn_softcap > 0.0
      t_scaled = TinyNN.tnn_scale(@sess, t_scaled, 1.0 / @attn_softcap)
      t_scaled = TinyNN.tnn_tanh(@sess, t_scaled)
      t_scaled = TinyNN.tnn_scale(@sess, t_scaled, @attn_softcap)
    end
    t_attn   = TinyNN.tnn_softmax(@sess, t_scaled)
    if tap_this_head
      t_attn = trace_tap(tag + "softmax", t_attn)
    end
    # P5.2: V is now [d_head, hist_count]; ggml_mul_mat needs the
    # matching k axis (hist_count) on both inputs, so transpose V_hist
    # (free view; tnn_transpose materializes via ggml_cont — one copy
    # of d_head × hist_count × 4 bytes per Q-head per layer). Cheap
    # at decode (typical hist_count ~ a few hundred) and uniform with
    # how flash takes V — both paths see the same V layout now.
    t_V_T  = TinyNN.tnn_transpose(@sess, t_V_hist)
    t_head = TinyNN.tnn_matmul(@sess, t_V_T, t_attn)
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

    # P5.2: K and V share the same layout ne=[d_head, max_T] now,
    # so they share the same zero-init Mat.
    kv_zero = Mat.new(max_T, d_head)

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
        # P5.1+P5.2: same Q8 skip rule as realize_for_mmap.
        if kv_cache.kv_type_k != 8
          TinyNN.upload_row_major(sess, blk_f.t_K[hkv], kv_zero)
        end
        if kv_cache.kv_type_v != 8
          TinyNN.upload_row_major(sess, blk_f.t_V[hkv], kv_zero)
        end
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
    # Spinel landmine: in whole-program inference contexts where `token_id`
    # poly-collapses to sp_RbVal (e.g. the eval runner, where generate's
    # sampler-fed `last_id` unifies decode_step's param to RbVal), the literal
    # `[token_id]` compiles to a PolyArray. upload_int_array takes :int_array
    # (sp_IntArray), so the PolyArray is then mis-read as an IntArray → garbage
    # length → ggml "tensor write out of bounds" abort. Narrowing to a clean
    # mrb_int via `.to_i` forces the IntArray codegen (as `[pos]` already gets).
    tid = token_id.to_i
    TinyNN.upload_int_array(kv_cache.sess, step.t_token_id, [tid])
    TinyNN.upload_int_array(kv_cache.sess, step.t_pos,      [pos])
    TinyNN.tnn_compute(kv_cache.sess)
    kv_cache.dump_trace
    TinyNN.download_row_major(kv_cache.sess, step.kv_step_logits, 1, kv_cache.vocab_size)
  end
end
