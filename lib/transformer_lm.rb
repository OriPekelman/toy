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

    kv = SmolLM2KVFFICache.new

    if is_native
      # Native layout: mmap weights at their stored ggml type. Q8_0
      # tensors stay quantized; matmul kernels read them in place.
      kv.set_weight_type(wtype)
      @gguf_handle = TinyNN.tnn_gguf_load(path)
      kv.realize_for_mmap(@gguf_handle, cfg, @max_T, flags.untied, flags.qkv_bias)
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
