# ToyLMCuda — CUDA mirror of lib/transformer_lm.rb.
#
# Mechanically identical to ToyLM modulo:
#   * TinyNN     → TinyNNCuda
#   * SmolLM2KV* → SmolLM2KV*Cuda
#   * Class name suffixed Cuda (Spinel collapses same-named classes
#     across modules; the suffix prevents that)
#
# Phase 0 implementation: delegates to SmolLM2KVFFICacheCuda. Phase 0.6
# will collapse both files into one generic class with a backend flag.
# Until then, callers requiring CUDA inference use this file; callers
# wanting CPU use lib/transformer_lm.rb (ToyLM).
#
# Usage:
#   arch = Arch.from_gguf("data/qwen25-1.5b-native.gguf")
#   lm   = ToyLMCuda.new(arch)
#   lm.load("data/qwen25-1.5b-native.gguf")
#   ids  = lm.generate([9707, 11, 847, 829, 374], 8)
#
# F32-only on CUDA today; Q8 needs the V-matmul flip mirrored to the
# CUDA decode_step (see docs/design/arch-struct.md for the open
# divergence note).

require_relative "transformer"
require_relative "arch"
require_relative "toy/train/sampler"
require_relative "toy"
require_relative "toy_smollm2"
require_relative "toy_smollm2_loader"
require_relative "toy_smollm2_ffi_kv_cuda"

class ToyLMCuda
  attr_reader :arch, :tokenizer, :max_T

  def initialize(arch)
    @arch    = arch
    @max_T   = 256
    @kv      = nil
    @gguf_handle = nil
    @loaded  = false
  end

  def max_T=(t)
    @max_T = t
  end

  def load(path)
    flags = GGUFLoad.detect_smollm2_flags(path)
    cfg   = SmolLM2ConfigLoader.read(path)

    probe = TinyNNCuda.tnn_gguf_load(path)
    is_native = false
    if probe != nil
      is_native = (TinyNNCuda.tnn_gguf_get_bool(probe, "toy.ggml_native") == 1)
      TinyNNCuda.tnn_gguf_free(probe)
    end

    kv = SmolLM2KVFFICacheCuda.new
    # P5.1: KV_Q8=1 opts into Q8_0 storage for the K cache. See the CPU
    # mirror in lib/transformer_lm.rb for the full rationale.
    if (ENV["KV_Q8"] || "") == "1"
      kv.enable_kv_q8!
    end
    # P4.1: FLASH_ATTN=1 → ggml_flash_attn_ext for attention.
    if (ENV["FLASH_ATTN"] || "") == "1"
      kv.enable_flash_attn!
    end

    if is_native
      @gguf_handle = TinyNNCuda.tnn_gguf_load(path)
      kv.realize_for_mmap(@gguf_handle, cfg, @max_T, flags.untied, flags.qkv_bias)
    else
      kv.realize_for(@max_T, cfg.d_model, cfg.d_ff, cfg.n_heads, cfg.n_kv,
                     cfg.n_layers, cfg.vocab, cfg.rope_base, cfg.rms_eps,
                     flags.untied, flags.qkv_bias)
      GGUFLoad.load_kv_cache_auto(kv, path)
    end
    @kv = kv
    @loaded = true
  end

  def decode_step(token_id, pos)
    SmolLM2KVCuda.decode_step(@kv, token_id, pos)
  end

  def generate(prompt_ids, n_new, sampler_config = nil)
    if !@loaded
      puts "ToyLMCuda.generate: model not loaded; call .load(path) first"
      return prompt_ids
    end
    ids = []
    j = 0
    while j < prompt_ids.length
      ids.push(prompt_ids[j])
      j = j + 1
    end

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
