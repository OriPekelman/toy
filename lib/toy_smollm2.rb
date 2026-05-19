# lib/toy_smollm2.rb — Toy::SmolLM2: llama-family decoder LM.
#
# Same block shape as Toy::GPT2 (pre-norm, two sublayers, residual on
# each) but with the llama-family substitutions:
#   - RMSNorm instead of LayerNorm
#   - GQAttention (grouped-query) instead of plain MHA
#   - SwiGLU instead of GeLU FFN
#   - RoPE instead of learned absolute position embeddings
#   - no biases anywhere
#   - tied embeddings (token_embed used as the unembed)
#
# Default config matches HuggingFaceTB/SmolLM2-135M:
#   d_model=576, n_heads=9, n_kv=3, d_ff=1536, n_layers=30,
#   vocab=49152, ctx=8192, rope_base=100000, rms_eps=1e-5

require_relative "toy"

module Toy
  class SmolLM2Config
    attr_accessor :vocab, :d_model, :n_heads, :n_kv, :d_ff,
                  :n_layers, :ctx, :rope_base, :rms_eps

    def initialize(vocab, d_model, n_heads, n_kv, d_ff, n_layers,
                   ctx, rope_base, rms_eps)
      @vocab     = vocab
      @d_model   = d_model
      @n_heads   = n_heads
      @n_kv      = n_kv
      @d_ff      = d_ff
      @n_layers  = n_layers
      @ctx       = ctx
      @rope_base = rope_base
      @rms_eps   = rms_eps
    end

    # Convenience: the default that matches SmolLM2-135M on HF.
    def self.smollm2_135m
      Toy::SmolLM2Config.new(49152, 576, 9, 3, 1536, 30,
                             8192, 100000.0, 1.0e-5)
    end
  end

  # Llama-style block: pre-norm + residual on each sublayer.
  class SmolLM2Block
    attr_accessor :rn1, :rn2, :attn, :ffn

    def initialize(cfg, rope_obj)
      @rn1     = Toy::RMSNorm.new(cfg.d_model)
      @rn1.eps = cfg.rms_eps
      @rn2     = Toy::RMSNorm.new(cfg.d_model)
      @rn2.eps = cfg.rms_eps
      @attn    = Toy::GQAttention.new(cfg.d_model, cfg.n_heads, cfg.n_kv, rope_obj)
      @ffn     = Toy::SwiGLU.new(cfg.d_model, cfg.d_ff)
    end

    # x: [T, D] → [T, D].  pos_start: absolute position of row 0 of x.
    def forward(x, pos_start)
      x.add!(@attn.forward(@rn1.forward(x), pos_start))   # residual after attention
      x.add!(@ffn.forward(@rn2.forward(x)))               # residual after FFN
      x
    end

    def param_count
      @rn1.param_count + @rn2.param_count +
        @attn.param_count + @ffn.param_count
    end

    def algorithm
      c = Toy::Card.new("SmolLM2Block.forward(x, p_start)", "")
      c.add_input("x",       "R^{T×D}", "")
      c.add_input("p_start", "ℕ",       "")
      c.add_output("x",      "R^{T×D}", "")
      c.step_update("x", "x + GQAttn(RMSNorm(x; γ_1, ε), p_start)",
                    "", "residual; RoPE inside attn")
      c.step_update("x", "x + SwiGLU(RMSNorm(x; γ_2, ε))",
                    "", "residual")
      c.step_return("x")
      c
    end

    def algorithm_card; algorithm.render_pseudocode; end
  end

  # SmolLM2 / generic llama-family decoder LM.
  #
  # Supports both tied and untied output embeddings:
  #   - SmolLM2 / Qwen2.5 / Gemma: tied (logits = x · token_embed.T)
  #   - TinyLlama / Llama-2/3 / Mistral: untied (logits = x · lm_head.T)
  #
  # Untied is opt-in via enable_untied_output! after construction.
  # The output_proj weight is stored as [V, D] (matches token_embed
  # layout) so the same matmul_t code path works for both.
  class SmolLM2
    attr_accessor :cfg, :token_embed, :final_norm, :stack, :rope,
                  :output_proj, :has_untied_output

    def initialize(cfg)
      @cfg         = cfg
      @token_embed = Toy::Embedding.new(cfg.vocab, cfg.d_model)
      @final_norm  = Toy::RMSNorm.new(cfg.d_model)
      @final_norm.eps = cfg.rms_eps
      @rope        = Toy::RoPE.new(cfg.d_model / cfg.n_heads,
                                   cfg.ctx, cfg.rope_base)

      @stack = [Toy::SmolLM2Block.new(cfg, @rope)]
      li = 1
      while li < cfg.n_layers
        @stack.push(Toy::SmolLM2Block.new(cfg, @rope))
        li += 1
      end

      # Always allocate the output projection at full [V, D] shape so
      # Spinel sees a stable Mat with known dimensions from the very
      # first reference. Costs vocab*d_model floats of memory even on
      # tied models (a few MB on SmolLM2, 256MB on TinyLlama) — small
      # next to the actual weights and avoids reassign-after-construct
      # surprises in the AOT type model.
      @output_proj       = Mat.new(cfg.vocab, cfg.d_model)
      @has_untied_output = false
    end

    # Called by the GGUF loader when `output.weight` is present. The
    # Mat is already allocated; this just flips the flag so the
    # forward uses it.
    def enable_untied_output!
      @has_untied_output = true
    end

    # ids: Array<Int> (length T), pos_start: Int → logits [T, V]
    def forward(ids, pos_start)
      x = @token_embed.lookup(ids)                           # [T, D]
      li = 0
      while li < @cfg.n_layers
        x = @stack[li].forward(x, pos_start)                 # [T, D]
        li += 1
      end
      x_final = @final_norm.forward(x)                       # [T, D]
      if @has_untied_output
        x_final.matmul_t(@output_proj)                       # [T, V]  (untied)
      else
        x_final.matmul_t(@token_embed.weight)                # [T, V]  (tied)
      end
    end

    # Total trainable parameter count (tied embeddings counted once).
    def param_count
      total = @token_embed.param_count + @final_norm.param_count
      li = 0
      while li < @cfg.n_layers
        total = total + @stack[li].param_count
        li += 1
      end
      total
    end

    # Introspection: `model.algorithm_card` (Phuong-Hutter pseudocode
    # with shape annotations) is the single canonical path. See
    # demos/algorithm_cards.rb.

    # Phuong–Hutter style algorithm card. Reads like the paper —
    # tensor shapes annotated on the right, ←  for assignment, ▷ for
    # commentary. See arXiv:2207.09238 §4 for the canonical form.
    #
    # `algorithm` returns the structured form (Toy::Card); `algorithm_card`
    # renders it to the human-readable Phuong–Hutter text. The structured
    # form is what prep/card_to_code.rb consumes for round-trip parsing.
    def algorithm
      c = Toy::Card.new("Toy::SmolLM2.forward(x, p_start)",
                        "Llama-family decoder")
      c.add_input("x",       "{1..V}^T", "token IDs")
      c.add_input("p_start", "ℕ",        "absolute position of x[0]; for RoPE")
      c.add_output("P",      "R^{T×V}",  "logits")
      c.add_hyper("V",      @cfg.vocab.to_s)
      c.add_hyper("D",      @cfg.d_model.to_s)
      c.add_hyper("H",      @cfg.n_heads.to_s)
      c.add_hyper("H_kv",   @cfg.n_kv.to_s)
      c.add_hyper("D_f",    @cfg.d_ff.to_s)
      c.add_hyper("N",      @cfg.n_layers.to_s)
      c.add_hyper("ctx",    @cfg.ctx.to_s)
      c.add_hyper("θ_base", @cfg.rope_base.to_s)
      c.add_param("W_e", "R^{V×D}", "token embeddings")
      if @has_untied_output
        c.add_param("W_out", "R^{V×D}", "separate lm_head")
      end
      c.add_param("θ_block_ℓ", "(ℓ=1..N)", "per-block; see SmolLM2Block")
      c.add_param("γ_f",       "R^D",      "final RMSNorm")
      c.add_param_extra("(total " + Toy.fmt_count(param_count) + ")")
      c.step_bind("e", "W_e[x]", "e ∈ R^{T×D}")
      c.step_loop("ℓ ← 1, …, N", "")
      c.step_update("e", "e + GQAttn(RMSNorm(e; γ_ℓ^1, ε), p_start; θ_ℓ^attn)",
                    "e ∈ R^{T×D}", "")
      c.step_update("e", "e + SwiGLU(RMSNorm(e; γ_ℓ^2, ε); θ_ℓ^ffn)",
                    "e ∈ R^{T×D}", "")
      c.step_loop_close
      c.step_update("e", "RMSNorm(e; γ_f, ε)", "e ∈ R^{T×D}", "")
      if @has_untied_output
        c.step_bind("P", "e · W_out^⊤", "P ∈ R^{T×V}  (untied)")
      else
        c.step_bind("P", "e · W_e^⊤",   "P ∈ R^{T×V}  (tied)")
      end
      c.step_return("P")
      c
    end

    def algorithm_card; algorithm.render_pseudocode; end

    # Recursive card: top-level forward + block + every sub-op
    # (RMSNorm, GQAttention, RoPE, SwiGLU) inlined. Useful for the
    # "full pseudocode" view; the top-level alone is the "section-1
    # overview" view.
    def algorithm_card_full
      blk = @stack[0]
      s = algorithm_card + "\n\n"
      s = s + "─── sub-algorithms ─────────────────────────────────────────────────────\n\n"
      s = s + blk.algorithm_card    + "\n\n"
      s = s + blk.rn1.algorithm_card  + "\n\n"
      s = s + blk.attn.algorithm_card + "\n\n"
      s = s + @rope.algorithm_card    + "\n\n"
      s = s + blk.ffn.algorithm_card
      s
    end
  end
end
