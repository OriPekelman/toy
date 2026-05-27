# TransformerLM — generic decoder-only LM driven by an Arch struct.
#
# Phase 0 implementation: thin wrapper that delegates to existing
# SmolLM2KVFFICache(_Cuda) classes based on the backend flag. The
# architectural decisions live in the Arch struct (read from GGUF);
# the graph code is still the architecture-aware Qwen-shaped code
# from lib/toy_smollm2_ffi_kv*.rb until Phase 0.6 inlines it here.
#
# After Phase 0.6: this class owns the decode-graph builder directly
# and reads arch fields to decide what to emit. lib/toy_smollm2_*.rb
# gets deleted.
#
# Usage:
#   arch = Arch.from_gguf("data/qwen25-1.5b-native.gguf")
#   lm   = ToyLM.new(arch, :cpu)             # or :cuda
#   lm.load("data/qwen25-1.5b-native.gguf")
#   ids  = lm.generate([9707, 11, 847, 829, 374], 8) # 8 new tokens
#
# Sampler integration is via the optional SamplerConfig argument. When
# nil (default), greedy argmax is used.

require_relative "transformer"
require_relative "arch"
require_relative "sampler"
require_relative "toy_smollm2_loader"
require_relative "toy_smollm2_ffi_kv"
require_relative "toy_logprobs"

class ToyLM
  attr_reader :arch, :backend, :tokenizer, :max_T

  def initialize(arch, backend)
    @arch    = arch
    @backend = backend
    @max_T   = 256
    @kv_cpu  = nil
    @kv_cuda = nil
    @gguf_handle = nil
    @loaded  = false
  end

  def max_T=(t)
    @max_T = t
  end

  # Load weights from the GGUF (mmap path). Path must match the path
  # used to construct the Arch (we re-open the GGUF for the mmap'd
  # weight pages).
  def load(path)
    if @backend == :cuda
      load_cuda(path)
    else
      load_cpu(path)
    end
    @loaded = true
  end

  def load_cpu(path)
    flags = GGUFLoad.detect_smollm2_flags(path)
    wtype = GGUFLoad.detect_weight_type(path)
    cfg   = SmolLM2ConfigLoader.read(path)

    # Format dispatch. Phase 2 mmap requires the GGUF to be in native
    # layout (toy.ggml_native flag set by --ggml-native at convert
    # time). Legacy GGUFs have transposed bytes and would produce
    # garbage if read directly; fall back to the Mat-mediated direct
    # loader for those.
    probe = TinyNN.tnn_gguf_load(path)
    is_native = false
    if probe != nil
      is_native = (TinyNN.tnn_gguf_get_bool(probe, "toy.ggml_native") == 1)
      TinyNN.tnn_gguf_free(probe)
    end
    # M2.3: MoE GGUFs (Mixtral / Qwen-MoE etc.) are always in standard
    # ggml-native layout; the legacy "transposed-load" path doesn't
    # know how to dequantize the 3D expert stacks anyway. Force mmap
    # whenever MoE is detected, regardless of the toy.ggml_native flag.
    if flags.is_moe
      is_native = true
    end
    # #113: same reasoning for Gemma 2 — third-party GGUFs (bartowski /
    # ggml-org / etc.) don't carry toy.ggml_native, but their layout is
    # standard. Force mmap when we see Gemma 2 sentinels (post-norm
    # tensors) so we get the post-norm and softcap paths.
    if flags.has_post_norms
      is_native = true
    end

    kv = SmolLM2KVFFICache.new
    # P5.1: KV_Q8=1 opts into Q8_0 storage for the K cache. Must be set
    # BEFORE realize_for(_mmap). Saves ~half the K-cache bytes &
    # bandwidth; V stays F32 until P5.2 (layout flip needed for V Q8).
    if (ENV["KV_Q8"] || "") == "1"
      kv.enable_kv_q8!
    end
    # P4.1: FLASH_ATTN=1 opts into ggml_flash_attn_ext for the per-Q-head
    # attention step. Inference only — vendored ggml's flash backward
    # aborts.
    if (ENV["FLASH_ATTN"] || "") == "1"
      kv.enable_flash_attn!
    end
    # M2.3: MoE — detected by GGUF tensor presence; enables the routed
    # FFN graph (router → softmax → top_k → 3× mul_mat_id → silu·up
    # → weighted sum). Must come BEFORE realize_for_mmap.
    if flags.is_moe
      kv.enable_moe!(flags.n_experts, flags.n_experts_used)
      puts "MoE detected: n_experts=" + flags.n_experts.to_s +
           " top_k=" + flags.n_experts_used.to_s
    end
    # #110: pass through the detected qk_norm flavor BEFORE realize.
    # 1 = Qwen3-style ([d_head] shared), 2 = OLMoE/Granite-style
    # ([d_model] per-head packed; per-head sliced gamma).
    kv.qk_norm_kind = flags.qk_norm_kind
    # NO_QK_NORM=1 turns the norm off entirely as a diagnostic.
    if (ENV["NO_QK_NORM"] || "") == "1"
      kv.has_qk_norm  = false
      kv.qk_norm_kind = 0
    end
    # #113: Gemma 2 extras. All inert by default — non-Gemma callers
    # pass embed_scale=1.0, softcaps=0.0, has_post_norms=false,
    # swa_alternates=false, and the graph paths skip the extras.
    kv.has_post_norms = flags.has_post_norms
    kv.embed_scale    = flags.embed_scale
    kv.attn_softcap   = flags.attn_softcap
    kv.final_softcap  = flags.final_softcap
    kv.swa_alternates = flags.swa_alternates
    if flags.has_post_norms || flags.attn_softcap > 0.0 || flags.swa_alternates
      puts "Gemma-2 features: post_norms=" + flags.has_post_norms.to_s +
           " embed_scale=" + flags.embed_scale.to_s +
           " attn_softcap=" + flags.attn_softcap.to_s +
           " final_softcap=" + flags.final_softcap.to_s +
           " swa_alt=" + flags.swa_alternates.to_s
    end

    if is_native
      # Native layout: mmap weights at their stored ggml type. Q8_0
      # tensors stay quantized; matmul kernels read them in place.
      kv.set_weight_type(wtype)
      @gguf_handle = TinyNN.tnn_gguf_load(path)
      kv.realize_for_mmap(@gguf_handle, cfg, @max_T, flags.untied, flags.qkv_bias, flags.qk_norm)
      kv.swa_window = flags.swa_window
    else
      # Legacy layout: dequantize-to-F32 on copy. The
      # tnn_gguf_copy_head_slice_to_persistent helper writes F32 bytes
      # into the dst, so dst tensors must be F32-typed — there's no
      # quantize-on-write code path yet. We deliberately do NOT call
      # set_weight_type here; the default (F32) is what holds.
      kv.realize_for(@max_T, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
                     cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
                     flags.untied, flags.qkv_bias)
      GGUFLoad.load_kv_cache_auto(kv, path)
    end
    @kv_cpu = kv
  end

  # The CUDA branch lives in load_cuda; we keep it conditional so the
  # CPU-only builds don't pull TinyNNCuda in.
  def load_cuda(path)
    puts "TransformerLM: CUDA path requires a CUDA-linked build. " +
         "Use lib/transformer_lm_cuda.rb (mirror); not implemented inline."
    nil
  end

  # Single-step decode → logits Mat (1 × vocab).
  def decode_step(token_id, pos)
    if @backend == :cuda
      puts "decode_step: CUDA backend not wired in this build"
      return nil
    end
    SmolLM2KV.decode_step(@kv_cpu, token_id, pos)
  end

  # toy#decode-logprobs (#151) — single-step decode that also returns
  # log_softmax(logits) + the top-K (id, logprob) pairs. Building block
  # for Tep's future /v1/chat/completions with `logprobs=true`.
  #
  # Returns [logits_mat, logprobs_mat, top_ids, top_vals] where:
  #   logits_mat   — Mat[1, vocab] raw logits (same as decode_step)
  #   logprobs_mat — Mat[1, vocab] numerically stable log-softmax
  #   top_ids      — Array<Int>   length top_k, sorted by logprob desc
  #   top_vals     — Array<Float> length top_k, parallel to top_ids
  def decode_step_with_logprobs(token_id, pos, top_k)
    logits = decode_step(token_id, pos)
    if logits == nil
      return [nil, nil, [0], [0.0]]   # never reached on CPU; CUDA prints+returns
    end
    logprobs = ToyLogProbs.log_softmax(logits)
    pair     = ToyLogProbs.top_k(logprobs, top_k)
    [logits, logprobs, pair[0], pair[1]]
  end

  # toy#embed-api (#145) — return the token-embedding row for each
  # input ID as a flat Array<Float> of length n_tokens * d_model.
  # Callers (Tep /v1/embeddings) can reshape / pool client-side. Works
  # regardless of backend because the GGUF mmap region is CPU-readable.
  # Single-row lookup is dequantize-aware (Q4/Q5/Q6/Q8/F16/F32).
  def embed_lookup(token_ids)
    if !@loaded
      puts "ToyLM.embed_lookup: model not loaded; call .load(path) first"
      return [0.0]
    end
    if @backend == :cuda
      puts "ToyLM.embed_lookup: CUDA backend not wired in this build " +
           "(use lib/transformer_lm_cuda.rb mirror once it lands; #145)"
      return Array.new(token_ids.length * @arch.d_model, 0.0)
    end
    d_model = @arch.d_model
    out = Array.new(token_ids.length * d_model, 0.0)
    row = Array.new(d_model, 0.0)
    handle = @kv_cpu.sess
    tensor = @kv_cpu.t_token_embed
    i = 0
    while i < token_ids.length
      rc = TinyNN.tnn_embed_lookup_to_doubles(handle, tensor, token_ids[i], row, d_model)
      if rc != 0
        puts "embed_lookup: rc=" + rc.to_s + " token=" + token_ids[i].to_s
        return out
      end
      j = 0
      while j < d_model
        out[i * d_model + j] = row[j]
        j = j + 1
      end
      i = i + 1
    end
    out
  end

  # Run prefill + generate. Returns the full ID array (prompt + N_NEW
  # generated). Uses greedy argmax if sampler_config is nil; otherwise
  # applies the configured sampler pipeline.
  def generate(prompt_ids, n_new, sampler_config = nil)
    if !@loaded
      puts "TransformerLM.generate: model not loaded; call .load(path) first"
      return prompt_ids
    end
    ids = []
    j = 0
    while j < prompt_ids.length
      ids.push(prompt_ids[j])
      j = j + 1
    end

    # Prefill: feed every prompt token through decode_step. Final
    # logits from the last prefill step ARE the first sampling target.
    i = 0
    while i < prompt_ids.length
      decode_step(prompt_ids[i], i)
      i = i + 1
    end

    ctx = nil
    if sampler_config != nil
      ctx = SamplerContext.new(ids, sampler_config.seed)
    end

    n = 0
    while n < n_new
      pos = ids.length
      last_id = ids[pos - 1]
      logits = decode_step(last_id, pos)
      pick = -1
      if sampler_config == nil
        pick = Sampler.argmax(logits)
      else
        logits = Sampler.repetition_penalty(logits, ctx, sampler_config.rep_penalty)
        logits = Sampler.temperature(logits, sampler_config.temperature)
        logits = Sampler.top_k(logits, sampler_config.top_k)
        logits = Sampler.top_p(logits, sampler_config.top_p)
        pick   = Sampler.pick(logits, sampler_config, ctx)
        ctx.generated_ids.push(pick)
      end
      ids.push(pick)
      if pick == @arch.eos_id
        break
      end
      n = n + 1
    end
    ids
  end
end
