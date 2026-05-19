# lib/toy_gpt2.rb — Toy::GPT2: HF-shape GPT-2 in the sugar API.
#
# The whole model is the three lines of `forward` at the bottom of this
# file. Everything above is configuration + a `GPT2Block` whose `call`
# is two lines.
#
# ┌─ Architecture (same as HF transformers.GPT2Model) ──────────────┐
# │   ids  →  embed (token + position)  →  N × GPT2Block            │
# │        →  final LayerNorm  →  unembed (tied to token_embed.T)   │
# │                                                                  │
# │   GPT2Block (pre-norm):                                          │
# │     x ← x + attn(ln1(x))                                         │
# │     x ← x + ffn (ln2(x))                                         │
# └──────────────────────────────────────────────────────────────────┘

require_relative "toy"

module Toy
  # Hyperparameter bag. Use a class (not Hash) so Spinel sees a
  # concrete struct with Int / Float fields.
  class GPT2Config
    attr_accessor :vocab, :d_model, :n_heads, :d_ff, :n_layers, :ctx

    def initialize(vocab, d_model, n_heads, d_ff, n_layers, ctx)
      @vocab    = vocab
      @d_model  = d_model
      @n_heads  = n_heads
      @d_ff     = d_ff
      @n_layers = n_layers
      @ctx      = ctx
    end
  end

  # One transformer block: pre-norm, residual after attention,
  # pre-norm, residual after FFN.
  class GPT2Block
    attr_accessor :ln1, :ln2, :attn, :ffn

    def initialize(cfg)
      @ln1  = Toy::LayerNorm.new(cfg.d_model)
      @ln2  = Toy::LayerNorm.new(cfg.d_model)
      @attn = Toy::CausalSelfAttention.new(cfg.d_model, cfg.n_heads)
      @ffn  = Toy::FFN.new(cfg.d_model, cfg.d_ff, :gelu_new)
    end

    # x: [T, D] → [T, D]
    def forward(x)
      x.add!(@attn.forward(@ln1.forward(x)))    # residual after attention
      x.add!(@ffn.forward(@ln2.forward(x)))       # residual after FFN
      x
    end

    def param_count
      @ln1.param_count + @ln2.param_count +
        @attn.param_count + @ffn.param_count
    end

    def algorithm
      c = Toy::Card.new("GPT2Block.forward(x)", "")
      c.add_input("x",  "R^{T×D}", "")
      c.add_output("x", "R^{T×D}", "")
      c.step_update("x", "x + Attn(LN(x; γ_1, β_1, ε))", "", "residual after attention")
      c.step_update("x", "x + FFN (LN(x; γ_2, β_2, ε))", "", "residual after FFN")
      c.step_return("x")
      c
    end

    def algorithm_card; algorithm.render_pseudocode; end
  end

  # GPT-2: decoder-only transformer LM. `stack` (not `blocks`) is kept
  # as the field name for readability — "the stack of N transformer
  # blocks" — independent of the older Spinel field-name-collapse
  # constraint that originally forced it.
  class GPT2
    attr_accessor :cfg, :token_embed, :pos_embed, :stack, :final_norm

    def initialize(cfg)
      @cfg         = cfg
      @token_embed = Toy::Embedding.new(cfg.vocab, cfg.d_model)
      @pos_embed   = Toy::Embedding.new(cfg.ctx,   cfg.d_model)
      @final_norm  = Toy::LayerNorm.new(cfg.d_model)

      # Block stack: literal-seed + push so Spinel infers
      # PtrArray<Toy::GPT2Block>.
      @stack = [Toy::GPT2Block.new(cfg)]
      li = 1
      while li < cfg.n_layers
        @stack.push(Toy::GPT2Block.new(cfg))
        li += 1
      end
    end

    # ids: Array<Int> (length T), start_pos: Int → logits [T, V]
    def forward(ids, start_pos)
      x = @token_embed.lookup(ids)                              # [T, D]
      x.add!(@pos_embed.slice(start_pos, ids.length))           # [T, D]

      li = 0
      while li < @cfg.n_layers
        x = @stack[li].forward(x)                                   # [T, D]
        li += 1
      end

      x_final = @final_norm.forward(x)                              # [T, D]
      x_final.matmul_t(@token_embed.weight)                      # [T, V]
    end

    # Total trainable parameter count. Tied embeddings counted once.
    def param_count
      total = @token_embed.param_count + @pos_embed.param_count +
              @final_norm.param_count
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

    # Phuong–Hutter style algorithm card for the whole model.
    # See arXiv:2207.09238 for the formalism. Mamba (arXiv:2312.00752)
    # and FlashAttention (arXiv:2205.14135) Algorithm 1 are the modern
    # exemplars for shape-annotated pseudocode.
    #
    # `algorithm` returns the structured form (Toy::Card); `algorithm_card`
    # renders it to the human-readable Phuong–Hutter text. The structured
    # form is what prep/card_to_code.rb consumes for round-trip parsing.
    def algorithm
      c = Toy::Card.new("Toy::GPT2.forward(x, p_start)", "HF GPT-2 family")
      c.add_input("x",       "{1..V}^T", "token IDs")
      c.add_input("p_start", "ℕ",        "absolute position of x[0]")
      c.add_output("P",      "R^{T×V}",  "logits")
      c.add_hyper("V",   @cfg.vocab.to_s)
      c.add_hyper("D",   @cfg.d_model.to_s)
      c.add_hyper("H",   @cfg.n_heads.to_s)
      c.add_hyper("D_f", @cfg.d_ff.to_s)
      c.add_hyper("N",   @cfg.n_layers.to_s)
      c.add_hyper("ctx", @cfg.ctx.to_s)
      c.add_param("W_e",         "R^{V×D}",   "token embeddings")
      c.add_param("W_p",         "R^{ctx×D}", "learned absolute positions")
      c.add_param("θ_block_ℓ",   "(ℓ=1..N)",  "per-block; see GPT2Block")
      c.add_param("γ_f, β_f",    "R^D",       "final LayerNorm")
      c.add_param_extra("(total " + Toy.fmt_count(param_count) +
                        ", embeddings tied: logits = e · W_e^⊤)")
      c.step_bind("e", "W_e[x] + W_p[p_start : p_start+T]", "e ∈ R^{T×D}")
      c.step_loop("ℓ ← 1, …, N", "")
      c.step_update("e", "e + Attn(LN(e; γ_ℓ^1, β_ℓ^1, ε); θ_ℓ^attn)",
                    "e ∈ R^{T×D}", "")
      c.step_update("e", "e + FFN (LN(e; γ_ℓ^2, β_ℓ^2, ε); θ_ℓ^ffn )",
                    "e ∈ R^{T×D}", "")
      c.step_loop_close
      c.step_update("e", "LN(e; γ_f, β_f, ε)", "e ∈ R^{T×D}", "")
      c.step_bind("P", "e · W_e^⊤",            "P ∈ R^{T×V}")
      c.step_return("P")
      c
    end

    def algorithm_card; algorithm.render_pseudocode; end

    # Recursive card — model + block + sub-ops inlined.
    def algorithm_card_full
      blk = @stack[0]
      s = algorithm_card + "\n\n"
      s = s + "─── sub-algorithms ─────────────────────────────────────────────────────\n\n"
      s = s + blk.algorithm_card    + "\n\n"
      s = s + blk.ln1.algorithm_card  + "\n\n"
      s = s + blk.attn.algorithm_card + "\n\n"
      s = s + blk.ffn.algorithm_card
      s
    end
  end
end
