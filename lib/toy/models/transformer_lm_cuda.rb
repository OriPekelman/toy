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
require_relative "../train/sampler"
require_relative "../../toy"
require_relative "toy_smollm2"
require_relative "../io/loaders/toy_smollm2_loader"
require_relative "../llm/engine/llama_kv_engine_cuda"

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
    # #76 fix: wire QK-norm through to the engine, parity with
    # load_cpu in lib/toy/models/transformer_lm.rb. This call site
    # previously passed only 5 of realize_for_mmap's 6 args; Spinel
    # zero-fills missing call args WITHOUT a diagnostic, so qk_norm
    # arrived as false and Qwen3's per-head Q/K RMS-norms were never
    # built on CUDA → degenerate decode (CPU was coherent).
    # 1 = Qwen3-style ([d_head] shared), 2 = OLMoE/Granite-style
    # ([d_model] packed, per-head sliced gamma). Must be set BEFORE
    # realize_for_mmap.
    kv.qk_norm_kind = flags.qk_norm_kind
    qk_norm_on = flags.qk_norm
    # NO_QK_NORM=1 turns the norm off entirely as a diagnostic
    # (same env knob as the CPU loader).
    if (ENV["NO_QK_NORM"] || "") == "1"
      qk_norm_on = false
      kv.qk_norm_kind = 0
    end

    if is_native
      @gguf_handle = TinyNNCuda.tnn_gguf_load(path)
      kv.realize_for_mmap(@gguf_handle, cfg, @max_T, flags.untied, flags.qkv_bias, qk_norm_on)
    else
      if qk_norm_on
        # Fail loud (never mask): the legacy copy-load path has no
        # QK-norm support — realize_for can't allocate the gamma
        # tensors, so decode would be silently degenerate.
        puts "ToyLMCuda.load: " + path + " needs QK-norm but is not in " +
             "toy.ggml_native layout; the legacy copy-load path cannot " +
             "apply QK-norm. Re-convert with --ggml-native. Aborting."
        exit 1
      end
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
