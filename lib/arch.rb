# Arch — per-model architecture struct.
#
# One instance per loaded GGUF, populated by Arch.from_gguf(path). The
# generic `TransformerLM` reads its fields to build the decode graph;
# different models differ in DATA (Arch fields) not CODE (graph).
#
# Design doc: docs/design/arch-struct.md
#
# Spinel-friendly: plain attr_reader fields, no metaprogramming, no
# nil defaults — the factory must fill every field explicitly.

require_relative "tinynn"

class Arch
  # Identity — :qwen2, :llama, :smollm. The label comes from tensor-
  # presence detection (NOT general.architecture: our converter writes
  # "llama" for every model, so it's unreliable).
  attr_reader :family
  attr_reader :name

  # Dimensions
  attr_reader :vocab_size
  attr_reader :d_model
  attr_reader :n_layers
  attr_reader :n_heads_q
  attr_reader :n_heads_kv
  attr_reader :d_head
  attr_reader :d_ff
  attr_reader :max_position
  attr_reader :untied_lm_head

  # Attention
  attr_reader :qkv_bias
  attr_reader :qk_norm
  attr_reader :swa_window      # nil when no sliding-window

  # RoPE
  attr_reader :rope_freq_base
  attr_reader :rope_freq_scale
  attr_reader :rope_partial_factor  # 1.0 default; 0.5 for GLM/Phi

  # Norm
  attr_reader :norm_kind        # :rms or :layer
  attr_reader :norm_eps

  # FFN
  attr_reader :ffn_kind         # :swiglu | :geglu | :gelu_mlp
  attr_reader :ffn_bias

  # MoE (zeros / false when not MoE)
  attr_reader :moe
  attr_reader :n_experts
  attr_reader :n_experts_used
  attr_reader :n_shared_experts
  attr_reader :expert_gating    # :softmax | :sigmoid

  # Tokenizer (Phase 0: GGUF metadata or nil)
  attr_reader :tokenizer_kind   # :gguf_embedded | :external
  attr_reader :bos_id
  attr_reader :eos_id
  attr_reader :pad_id
  attr_reader :unk_id
  attr_reader :add_bos_by_default

  # Embed scale (some models multiply token_embd by sqrt(d_model);
  # Llama-family does not).
  attr_reader :embed_scale

  def initialize(family, name,
                 vocab_size, d_model, n_layers, n_heads_q, n_heads_kv, d_head, d_ff,
                 max_position, untied_lm_head,
                 qkv_bias, qk_norm, swa_window,
                 rope_freq_base, rope_freq_scale, rope_partial_factor,
                 norm_kind, norm_eps,
                 ffn_kind, ffn_bias,
                 moe, n_experts, n_experts_used, n_shared_experts, expert_gating,
                 tokenizer_kind, bos_id, eos_id, pad_id, unk_id, add_bos_by_default,
                 embed_scale)
    @family               = family
    @name                 = name
    @vocab_size           = vocab_size
    @d_model              = d_model
    @n_layers             = n_layers
    @n_heads_q            = n_heads_q
    @n_heads_kv           = n_heads_kv
    @d_head               = d_head
    @d_ff                 = d_ff
    @max_position         = max_position
    @untied_lm_head       = untied_lm_head
    @qkv_bias             = qkv_bias
    @qk_norm              = qk_norm
    @swa_window           = swa_window
    @rope_freq_base       = rope_freq_base
    @rope_freq_scale      = rope_freq_scale
    @rope_partial_factor  = rope_partial_factor
    @norm_kind            = norm_kind
    @norm_eps             = norm_eps
    @ffn_kind             = ffn_kind
    @ffn_bias             = ffn_bias
    @moe                  = moe
    @n_experts            = n_experts
    @n_experts_used       = n_experts_used
    @n_shared_experts     = n_shared_experts
    @expert_gating        = expert_gating
    @tokenizer_kind       = tokenizer_kind
    @bos_id               = bos_id
    @eos_id               = eos_id
    @pad_id               = pad_id
    @unk_id               = unk_id
    @add_bos_by_default   = add_bos_by_default
    @embed_scale          = embed_scale
  end

  def moe?
    @moe
  end

  def gqa?
    @n_heads_kv < @n_heads_q
  end

  def swa?
    @swa_window != nil
  end

  # Pretty one-line summary for log lines / startup.
  def summary
    "Arch(" + @family.to_s +
      ", vocab=" + @vocab_size.to_s +
      ", d=" + @d_model.to_s +
      ", L=" + @n_layers.to_s +
      ", n_q=" + @n_heads_q.to_s +
      ", n_kv=" + @n_heads_kv.to_s +
      ", d_ff=" + @d_ff.to_s +
      ", qkv_bias=" + @qkv_bias.to_s +
      ", rope_base=" + @rope_freq_base.to_s +
      ", " + @norm_kind.to_s + " eps=" + @norm_eps.to_s +
      ", " + @ffn_kind.to_s + ")"
  end

  # Detect the architecture family by reading the GGUF and inspecting
  # what's there. The general.architecture key is unreliable (our
  # converter writes "llama" for every model), so we use tensor
  # presence + RoPE freq_base as the actual signal.
  def self.from_gguf(path)
    handle = TinyNN.tnn_gguf_load(path)
    if handle == nil
      puts "Arch.from_gguf: failed to open " + path
      return nil
    end

    # Llama-family GGUF keys are the canonical scalar metadata (the
    # converter writes "llama.*" for SmolLM2/TinyLlama/Qwen2.5/Llama3
    # alike). Read once and reuse.
    vocab    = TinyNN.tnn_gguf_get_u32(handle, "llama.vocab_size")
    d_model  = TinyNN.tnn_gguf_get_u32(handle, "llama.embedding_length")
    d_ff     = TinyNN.tnn_gguf_get_u32(handle, "llama.feed_forward_length")
    n_q      = TinyNN.tnn_gguf_get_u32(handle, "llama.attention.head_count")
    n_kv     = TinyNN.tnn_gguf_get_u32(handle, "llama.attention.head_count_kv")
    n_layers = TinyNN.tnn_gguf_get_u32(handle, "llama.block_count")
    ctx      = TinyNN.tnn_gguf_get_u32(handle, "llama.context_length")
    if ctx < 0
      ctx = 8192   # default if metadata missing
    end
    rope_base = TinyNN.tnn_gguf_get_f32(handle, "llama.rope.freq_base")
    rms_eps   = TinyNN.tnn_gguf_get_f32(handle, "llama.attention.layer_norm_rms_epsilon")
    d_head    = d_model / n_q

    # Tensor-presence flags.
    has_qkv_bias = TinyNN.tnn_gguf_find_index(handle, "blk.0.attn_q.bias") >= 0
    untied       = TinyNN.tnn_gguf_find_index(handle, "output.weight")     >= 0

    # Tokenizer metadata (most current GGUFs in this repo don't embed
    # it — our converter skips it. Read anyway for forward-compat).
    bos = TinyNN.tnn_gguf_get_u32(handle, "tokenizer.ggml.bos_token_id")
    eos = TinyNN.tnn_gguf_get_u32(handle, "tokenizer.ggml.eos_token_id")
    pad = TinyNN.tnn_gguf_get_u32(handle, "tokenizer.ggml.padding_token_id")
    unk = TinyNN.tnn_gguf_get_u32(handle, "tokenizer.ggml.unknown_token_id")
    vocab_n = TinyNN.tnn_gguf_arr_n(handle, "tokenizer.ggml.tokens")
    tok_kind = :external
    if vocab_n > 0
      tok_kind = :gguf_embedded
    end

    # Family detection — see the comment above. The current set of
    # models all share the Llama-family graph; the only structural
    # delta we care about is QKV bias.
    family = :llama
    if has_qkv_bias
      family = :qwen2
    end

    TinyNN.tnn_gguf_free(handle)

    # Arch.new positional args: family, name, vocab, d_model, n_layers,
    # n_q, n_kv, d_head, d_ff, max_pos, untied, qkv_bias, qk_norm,
    # swa_window, rope_freq_base, rope_scale, rope_partial, norm_kind,
    # norm_eps, ffn_kind, ffn_bias, moe, n_experts, n_experts_used,
    # n_shared_experts, expert_gating, tokenizer_kind, bos, eos, pad,
    # unk, add_bos, embed_scale.
    Arch.new(family, path,
             vocab, d_model, n_layers, n_q, n_kv, d_head, d_ff,
             ctx, untied,
             has_qkv_bias, false, nil,
             rope_base, 1.0, 1.0,
             :rms, rms_eps,
             :swiglu, false,
             false, 0, 0, 0, :softmax,
             tok_kind, bos, eos, pad, unk, false,
             1.0)
  end
end
