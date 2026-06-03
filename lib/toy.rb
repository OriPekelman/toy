# lib/toy.rb — transformer building blocks.
#
# A thin sugar layer on top of Mat (lib/transformer.rb). Each class
# here owns its own parameters as ivars and has one entry point —
# `call(x)`. Models compose these so the forward pass reads like the
# transformer paper: pre-norm, attention with residual, FFN with
# residual.
#
# Shapes throughout follow this convention in comments:
#   [T, D]   T = sequence length     D = d_model
#   [T, Dh]  Dh = d_head             H = n_heads
#   [V]      V = vocab               Df = d_ff
#
# Spinel notes (relevant if you read the source):
#   • Each class's ivars are concrete types (Mat or Array<Float>).
#     No polymorphic containers.
#   • Per-head Q/K/V uses the literal-seed + push pattern
#     (Array<Mat>) — same as lib/transformer.rb's Block.
#   • Methods return concrete types (always Mat for tensor-valued ops)
#     so Spinel's type inference flows cleanly.

require_relative "toy/models/transformer"   # Mat lives here
require_relative "toy/dev/toy_card"      # Toy::Card structured IR

require_relative "toy/version"   # defines Toy::VERSION; gem-loadable

module Toy
  # =========================================================================
  # Toy::LayerNorm
  #   y = (x - mean) / sqrt(var + eps) * gamma + beta,  row-wise.
  #   gamma, beta: length D. eps default 1e-5 (HF GPT-2).
  # =========================================================================
  class LayerNorm
    attr_accessor :gamma, :beta, :d, :eps

    def initialize(d)
      @d     = d
      @eps   = RMS_EPS_DEFAULT
      @gamma = Array.new(d, 1.0)
      @beta  = Array.new(d, 0.0)
    end

    # x: [T, D] → [T, D]
    def forward(x)
      t   = x.nrows
      d   = @d
      out = Mat.new(t, d)
      i = 0
      while i < t
        # mean over row
        sum = 0.0
        j = 0
        while j < d
          sum = sum + x.flat[i * d + j]
          j += 1
        end
        mean = sum / d
        # variance over row
        sumsq = 0.0
        j = 0
        while j < d
          v = x.flat[i * d + j] - mean
          sumsq = sumsq + v * v
          j += 1
        end
        inv = 1.0 / Math.sqrt(sumsq / d + @eps)
        # normalize, scale, shift
        j = 0
        while j < d
          n = (x.flat[i * d + j] - mean) * inv
          out.flat[i * d + j] = n * @gamma[j] + @beta[j]
          j += 1
        end
        i += 1
      end
      out
    end

    def summary;     "LayerNorm(d=" + @d.to_s + ", eps=" + @eps.to_s + ")"; end
    def param_count; 2 * @d; end   # gamma + beta

    # One-line algorithm-card body (Phuong–Hutter style).
    #   LN(x; γ, β) = (x - mean(x)) / sqrt(var(x) + ε) ⊙ γ + β
    def algorithm_card
      "LN(x; γ, β, ε) := (x − mean(x)) / √(var(x) + ε) ⊙ γ + β"
    end
  end

  # =========================================================================
  # Toy::Linear
  #   y = x · W + b  (b optional; bias=false skips it).
  # =========================================================================
  class Linear
    attr_accessor :w, :b, :has_bias, :in_dim, :out_dim

    def initialize(in_dim, out_dim, with_bias)
      @in_dim   = in_dim
      @out_dim  = out_dim
      @has_bias = with_bias
      @w        = Mat.new(in_dim, out_dim)
      @b        = Array.new(out_dim, 0.0)
    end

    # x: [T, in_dim] → [T, out_dim]
    def forward(x)
      out = x.matmul(@w)
      if @has_bias
        Toy.add_bias!(out, @b)
      end
      out
    end

    def summary
      bs = @has_bias ? "true" : "false"
      "Linear(in=" + @in_dim.to_s + ", out=" + @out_dim.to_s + ", bias=" + bs + ")"
    end
    def param_count
      n = @in_dim * @out_dim
      if @has_bias
        n = n + @out_dim
      end
      n
    end
  end

  # =========================================================================
  # Toy::Embedding
  #   Lookup table: row[i] = weight[i, :].
  # =========================================================================
  class Embedding
    attr_accessor :weight, :vocab, :d

    def initialize(vocab, d)
      @vocab  = vocab
      @d      = d
      @weight = Mat.new(vocab, d)
    end

    # ids: [T] (Array<Int>) → [T, D]
    def lookup(ids)
      t   = ids.length
      d   = @d
      out = Mat.new(t, d)
      i = 0
      while i < t
        row = ids[i]
        j = 0
        while j < d
          out.flat[i * d + j] = @weight.flat[row * d + j]
          j += 1
        end
        i += 1
      end
      out
    end

    # Slice rows [start, start + count) as a fresh Mat.
    # Used for absolute position embeddings.
    def slice(start, count)
      d   = @d
      out = Mat.new(count, d)
      i = 0
      while i < count
        j = 0
        while j < d
          out.flat[i * d + j] = @weight.flat[(start + i) * d + j]
          j += 1
        end
        i += 1
      end
      out
    end

    def summary;     "Embedding(vocab=" + @vocab.to_s + ", d=" + @d.to_s + ")"; end
    def param_count; @vocab * @d; end
  end

  # =========================================================================
  # Toy::CausalSelfAttention
  #
  #   Per-head storage (Array<Mat> for Q/K/V) matches lib/transformer.rb's
  #   Block layout, so the GGUF loader's split-heads helper works.
  #
  #     attn(x) =
  #       q_h = (x · W_q^h + b_q^h)       for h in 0..H
  #       k_h = (x · W_k^h + b_k^h)
  #       v_h = (x · W_v^h + b_v^h)
  #       s_h = softmax(causal(q_h · k_h^T / sqrt(Dh))) · v_h
  #       y   = hstack(s_0..s_H) · W_o + b_o
  # =========================================================================
  class CausalSelfAttention
    attr_accessor :w_q, :w_k, :w_v, :b_q, :b_k, :b_v, :w_o, :b_o,
                  :n_heads, :d_model, :d_head, :inv_sqrt

    def initialize(d_model, n_heads)
      @d_model  = d_model
      @n_heads  = n_heads
      @d_head   = d_model / n_heads
      @inv_sqrt = 1.0 / Math.sqrt(@d_head)

      # Per-head: literal-seed + push so Spinel types as PtrArray of Mat /
      # PtrArray of FloatArray.
      @w_q = [Mat.new(d_model, @d_head)]
      @w_k = [Mat.new(d_model, @d_head)]
      @w_v = [Mat.new(d_model, @d_head)]
      @b_q = [Array.new(@d_head, 0.0)]
      @b_k = [Array.new(@d_head, 0.0)]
      @b_v = [Array.new(@d_head, 0.0)]
      h = 1
      while h < n_heads
        @w_q.push(Mat.new(d_model, @d_head))
        @w_k.push(Mat.new(d_model, @d_head))
        @w_v.push(Mat.new(d_model, @d_head))
        @b_q.push(Array.new(@d_head, 0.0))
        @b_k.push(Array.new(@d_head, 0.0))
        @b_v.push(Array.new(@d_head, 0.0))
        h += 1
      end
      @w_o = Mat.new(d_model, d_model)
      @b_o = Array.new(d_model, 0.0)
    end

    # x: [T, D] → [T, D]
    def forward(x)
      head0    = head(x, 0)               # [T, Dh]
      per_head = [head0]
      h = 1
      while h < @n_heads
        per_head.push(head(x, h))         # [T, Dh] each
        h += 1
      end
      concat = Toy.hstack_heads(per_head, @n_heads, @d_head, @d_model)  # [T, D]
      out    = concat.matmul(@w_o)        # [T, D]
      Toy.add_bias!(out, @b_o)
      out
    end

    # One attention head. x: [T, D] → [T, Dh].
    def head(x, h)
      q = x.matmul(@w_q[h])
      Toy.add_bias!(q, @b_q[h])           # [T, Dh]
      k = x.matmul(@w_k[h])
      Toy.add_bias!(k, @b_k[h])           # [T, Dh]
      v = x.matmul(@w_v[h])
      Toy.add_bias!(v, @b_v[h])           # [T, Dh]

      scores = q.matmul_t(k)              # [T, T]
      scores.scale!(@inv_sqrt)
      Toy.causal_mask!(scores)
      Toy.softmax_rows!(scores)
      scores.matmul(v)                    # [T, Dh]
    end

    def summary
      "CausalSelfAttention(d_model=" + @d_model.to_s +
        ", heads=" + @n_heads.to_s + ", d_head=" + @d_head.to_s + ")"
    end
    def param_count
      # n_heads × (d_model * d_head + d_head) for Q/K/V, plus d_model² + d_model for W_o + b_o
      per_head = (@d_model * @d_head + @d_head) * 3
      per_head * @n_heads + @d_model * @d_model + @d_model
    end

    # Algorithm card. Shapes: x ∈ R^{T×D}; D_h = D/H.
    def algorithm
      c = Toy::Card.new("CausalSelfAttention.forward(x)", "")
      c.add_input("x",  "R^{T×D}", "")
      c.add_output("y", "R^{T×D}", "")
      c.add_hyper("D",   @d_model.to_s)
      c.add_hyper("H",   @n_heads.to_s)
      c.add_hyper("D_h", @d_head.to_s)
      c.add_param("W_Q^h, W_K^h, W_V^h", "R^{D×D_h}", "")
      c.add_param("b_Q^h, b_K^h, b_V^h", "R^{D_h}",    "")
      c.add_param("W_O",                  "R^{D×D}",   "")
      c.add_param("b_O",                  "R^{D}",     "")
      c.step_loop("h ← 1, …, H", "per head")
      c.step_bind("q^h", "x · W_Q^h + b_Q^h",     "q^h ∈ R^{T×D_h}")
      c.step_bind("k^h", "x · W_K^h + b_K^h",     "k^h ∈ R^{T×D_h}")
      c.step_bind("v^h", "x · W_V^h + b_V^h",     "v^h ∈ R^{T×D_h}")
      c.step_bind("S^h", "q^h · (k^h)^⊤ / √D_h", "S^h ∈ R^{T×T}")
      c.step_update("S^h", "CausalMask(S^h)",   "",                  "j>i ↦ −∞")
      c.step_bind("A^h", "softmax_rows(S^h)",    "A^h ∈ R^{T×T}")
      c.step_bind("o^h", "A^h · v^h",            "o^h ∈ R^{T×D_h}")
      c.step_loop_close
      c.step_bind("y", "concat(o^1, …, o^H) · W_O + b_O", "y ∈ R^{T×D}")
      c.step_return("y")
      c
    end

    def algorithm_card; algorithm.render_pseudocode; end
  end

  # =========================================================================
  # Toy::FFN
  #   GPT-2 MLP: y = up · W2 + b2 where up = gelu(x · W1 + b1).
  #   `act` selects the activation: :gelu_new (tanh approx, HF default)
  #   or :gelu_exact (erf-based).
  # =========================================================================
  class FFN
    attr_accessor :w1, :w2, :b1, :b2, :d_model, :d_ff, :act

    def initialize(d_model, d_ff, act_sym)
      @d_model = d_model
      @d_ff    = d_ff
      @act     = act_sym                  # :gelu_new
      @w1 = Mat.new(d_model, d_ff)
      @w2 = Mat.new(d_ff,    d_model)
      @b1 = Array.new(d_ff,   0.0)
      @b2 = Array.new(d_model, 0.0)
    end

    # x: [T, D] → [T, D]
    def forward(x)
      pre = x.matmul(@w1)                 # [T, Df]
      Toy.add_bias!(pre, @b1)
      hidden = Toy.gelu_new(pre)          # [T, Df]
      out = hidden.matmul(@w2)            # [T, D]
      Toy.add_bias!(out, @b2)
      out
    end

    def summary
      "FFN(d=" + @d_model.to_s + ", hidden=" + @d_ff.to_s + ", act=" + @act.to_s + ")"
    end
    def param_count
      @d_model * @d_ff + @d_ff +     # W1 + b1
        @d_ff * @d_model + @d_model  # W2 + b2
    end

    def algorithm
      c = Toy::Card.new("FFN.forward(x)", "GPT-2-style MLP")
      c.add_input("x",  "R^{T×D}", "")
      c.add_output("y", "R^{T×D}", "")
      c.add_hyper("D",          @d_model.to_s)
      c.add_hyper("D_f",        @d_ff.to_s)
      c.add_hyper("activation", @act.to_s)
      c.add_param("W_1", "R^{D×D_f}", "")
      c.add_param("b_1", "R^{D_f}",   "")
      c.add_param("W_2", "R^{D_f×D}", "")
      c.add_param("b_2", "R^{D}",     "")
      c.step_bind("h", "gelu(x · W_1 + b_1)", "h ∈ R^{T×D_f}")
      c.step_bind("y", "h · W_2 + b_2",       "y ∈ R^{T×D}")
      c.step_return("y")
      c
    end

    def algorithm_card; algorithm.render_pseudocode; end
  end

  # =========================================================================
  # Toy::RMSNorm — root-mean-square LayerNorm. No mean subtraction, no
  # beta. Standard in llama-family models.
  #
  #   y = x / sqrt(mean(x^2) + eps) * gamma   (row-wise)
  # =========================================================================
  class RMSNorm
    attr_accessor :gamma, :d, :eps

    # eps defaults to 1.0e-5 (matches Llama / SmolLM2). Override via
    # `rms.eps = ...` after construction — seeding the ivar with a Float
    # literal pins Spinel's type inference.
    def initialize(d)
      @d     = d
      @eps   = RMS_EPS_DEFAULT
      @gamma = Array.new(d, 1.0)
    end

    # x: [T, D] → [T, D]
    def forward(x)
      t   = x.nrows
      d   = @d
      out = Mat.new(t, d)
      i = 0
      while i < t
        sumsq = 0.0
        j = 0
        while j < d
          v = x.flat[i * d + j]
          sumsq = sumsq + v * v
          j += 1
        end
        inv = 1.0 / Math.sqrt(sumsq / d + @eps)
        j = 0
        while j < d
          out.flat[i * d + j] = x.flat[i * d + j] * inv * @gamma[j]
          j += 1
        end
        i += 1
      end
      out
    end

    def summary;     "RMSNorm(d=" + @d.to_s + ", eps=" + @eps.to_s + ")"; end
    def param_count; @d; end   # gamma only

    def algorithm_card
      "RMSNorm(x; γ, ε) := x / √(mean(x²) + ε) ⊙ γ"
    end
  end

  # =========================================================================
  # Toy::SwiGLU — gated FFN used by Llama / SmolLM2 / Qwen2 / Phi.
  #
  #   gate(x) = x · W_gate       up(x) = x · W_up
  #   y       = (silu(gate(x)) * up(x)) · W_down
  #
  # Three linear layers, no bias by default (llama convention). Element-
  # wise multiply between silu(gate) and up before the down projection.
  # =========================================================================
  class SwiGLU
    attr_accessor :w_gate, :w_up, :w_down, :d_model, :d_ff

    def initialize(d_model, d_ff)
      @d_model = d_model
      @d_ff    = d_ff
      @w_gate  = Mat.new(d_model, d_ff)
      @w_up    = Mat.new(d_model, d_ff)
      @w_down  = Mat.new(d_ff,    d_model)
    end

    # x: [T, D] → [T, D]
    def forward(x)
      gate = x.matmul(@w_gate)              # [T, Df]
      up   = x.matmul(@w_up)                # [T, Df]
      Toy.silu!(gate)                       # [T, Df]
      Toy.hadamard!(gate, up)               # [T, Df]  (gate := gate * up)
      gate.matmul(@w_down)                  # [T, D]
    end

    def summary
      "SwiGLU(d=" + @d_model.to_s + ", d_ff=" + @d_ff.to_s + ")"
    end
    def param_count
      # 3 × (d_model × d_ff) — no biases (llama convention).
      3 * @d_model * @d_ff
    end

    def algorithm
      c = Toy::Card.new("SwiGLU.forward(x)", "Llama-family MLP")
      c.add_input("x",  "R^{T×D}", "")
      c.add_output("y", "R^{T×D}", "")
      c.add_hyper("D",   @d_model.to_s)
      c.add_hyper("D_f", @d_ff.to_s)
      c.add_param("W_gate, W_up", "R^{D×D_f}", "")
      c.add_param("W_down",       "R^{D_f×D}", "no biases — Llama convention")
      c.step_bind("g", "x · W_gate",    "g ∈ R^{T×D_f}")
      c.step_bind("u", "x · W_up",      "u ∈ R^{T×D_f}")
      c.step_bind("h", "silu(g) ⊙ u",   "h ∈ R^{T×D_f}")
      c.step_bind("y", "h · W_down",    "y ∈ R^{T×D}")
      c.step_return("y")
      c
    end

    def algorithm_card; algorithm.render_pseudocode; end
  end

  # =========================================================================
  # Toy::RoPE — rotary position embedding. Llama / SmolLM2 / Qwen2 form
  # ("rotate_half"): split head_dim into two halves and rotate them
  # against each other by an angle that scales with position.
  #
  # Precomputes cos/sin tables at construction. Rotates in-place.
  #
  #   for freq k in [0, Dh/2):
  #     theta_k = theta_base^(-2k/Dh)
  #     angle   = pos * theta_k
  #     x[k],       x[k+Dh/2]  →
  #       x[k]      * cos(angle) - x[k+Dh/2] * sin(angle),
  #       x[k+Dh/2] * cos(angle) + x[k]      * sin(angle)
  # =========================================================================
  class RoPE
    attr_accessor :cos_tbl, :sin_tbl, :d_head, :max_seq

    # theta_base typical values: 10000 (Llama-1/2/TinyLlama), 100000
    # (SmolLM2), 1000000 (Qwen2 long-context). Renamed from `base` to
    # dodge Spinel's local-name collapse with int offsets named `base`.
    def initialize(d_head, max_seq, theta_base)
      @d_head  = d_head
      @max_seq = max_seq
      half     = d_head / 2
      n        = max_seq * half
      @cos_tbl = Array.new(n, 1.0)
      @sin_tbl = Array.new(n, 0.0)
      log_b    = Math.log(theta_base.to_f)
      inv_dh   = 1.0 / d_head.to_f
      p = 0
      while p < max_seq
        k = 0
        while k < half
          theta = Math.exp(-2.0 * k.to_f * inv_dh * log_b)
          angle = p.to_f * theta
          @cos_tbl[p * half + k] = Math.cos(angle)
          @sin_tbl[p * half + k] = Math.sin(angle)
          k += 1
        end
        p += 1
      end
    end

    # x: [T, Dh] rotated in-place. Row t corresponds to absolute
    # position (pos_start + t).
    def rotate!(x, pos_start)
      t    = x.nrows
      dh   = @d_head
      half = dh / 2
      i = 0
      while i < t
        p   = pos_start + i
        row = i * dh
        ck = 0
        while ck < half
          co = @cos_tbl[p * half + ck]
          si = @sin_tbl[p * half + ck]
          xa = x.flat[row + ck]
          xb = x.flat[row + half + ck]
          x.flat[row + ck]        = xa * co - xb * si
          x.flat[row + half + ck] = xb * co + xa * si
          ck += 1
        end
        i += 1
      end
    end

    def summary
      "RoPE(d_head=" + @d_head.to_s + ", max_seq=" + @max_seq.to_s + ")"
    end
    def param_count; 0; end   # cos/sin tables are precomputed, not learned

    def algorithm
      c = Toy::Card.new("RoPE.rotate!(x, p_start)", "rotate_half / NEOX form")
      c.add_input("x",       "R^{T×D_h}", "one head's Q or K")
      c.add_input("p_start", "ℕ",         "absolute position of row 0")
      c.add_hyper("D_h",    @d_head.to_s)
      c.add_hyper("θ_base", "(cos/sin tables precomputed)")
      c.step_loop("t ← 0, …, T-1, k ← 0, …, D_h/2 − 1", "")
      c.step_bind("p", "p_start + t", "")
      c.step_bind("c, s",
                  "cos(p · θ_base^{−2k/D_h}), sin(p · θ_base^{−2k/D_h})", "")
      c.step_update("(x[t,k], x[t,k+D_h/2])",
                    "(x[t,k]·c − x[t,k+D_h/2]·s, x[t,k+D_h/2]·c + x[t,k]·s)",
                    "", "")
      c.step_loop_close
      c
    end

    def algorithm_card; algorithm.render_pseudocode; end
  end

  # =========================================================================
  # Toy::GQAttention — grouped-query causal self-attention.
  #
  # n_heads query heads share n_kv key/value heads (n_heads / n_kv
  # queries per KV head). Used by SmolLM2 (9/3), TinyLlama (32/4),
  # Qwen2.5-0.5B (14/2). When n_heads == n_kv this degenerates to
  # standard MHA.
  #
  # RoPE is applied to Q and K *before* the dot product. The two-arg
  # forward `(x, pos_start)` is needed because RoPE depends on absolute
  # position.
  #
  # Q/K/V biases are pre-allocated to zeros but only applied when
  # `has_qkv_bias` is true. SmolLM2 / Llama / TinyLlama: false (Llama
  # convention). Qwen2 / Qwen2.5: true (Q/K/V have learned biases; O
  # does not). The loader flips the flag when it finds attn_q.bias /
  # attn_k.bias / attn_v.bias in the GGUF — pre-allocation keeps the
  # ivar types stable for Spinel (no reassign-after-construct).
  # =========================================================================
  class GQAttention
    attr_accessor :w_q, :w_k, :w_v, :w_o,
                  :b_q, :b_k, :b_v, :has_qkv_bias,
                  :n_heads, :n_kv, :d_model, :d_head,
                  :group_size, :inv_sqrt, :rope

    def initialize(d_model, n_heads, n_kv, rope_obj)
      @d_model    = d_model
      @n_heads    = n_heads
      @n_kv       = n_kv
      @d_head     = d_model / n_heads
      @group_size = n_heads / n_kv
      @inv_sqrt   = 1.0 / Math.sqrt(@d_head)
      @rope       = rope_obj

      @w_q = [Mat.new(d_model, @d_head)]
      @b_q = [Array.new(@d_head, 0.0)]   # per Q head (zeros until enabled)
      hq = 1
      while hq < n_heads
        @w_q.push(Mat.new(d_model, @d_head))
        @b_q.push(Array.new(@d_head, 0.0))
        hq += 1
      end
      @w_k = [Mat.new(d_model, @d_head)]
      @w_v = [Mat.new(d_model, @d_head)]
      @b_k = [Array.new(@d_head, 0.0)]   # per KV head
      @b_v = [Array.new(@d_head, 0.0)]
      hkv = 1
      while hkv < n_kv
        @w_k.push(Mat.new(d_model, @d_head))
        @w_v.push(Mat.new(d_model, @d_head))
        @b_k.push(Array.new(@d_head, 0.0))
        @b_v.push(Array.new(@d_head, 0.0))
        hkv += 1
      end
      @w_o = Mat.new(d_model, d_model)
      @has_qkv_bias = false               # flipped by the loader for Qwen2.x
    end

    # Called by the GGUF loader when attn_q.bias / attn_k.bias /
    # attn_v.bias are present. Biases are already allocated; this just
    # flips the flag.
    def enable_qkv_bias!
      @has_qkv_bias = true
    end

    # x: [T, D] → [T, D].  pos_start: absolute position of row 0 of x.
    def forward(x, pos_start)
      # 1) project + rotate K, V once per KV head (n_kv times).
      k0 = x.matmul(@w_k[0])                 # [T, Dh]
      if @has_qkv_bias
        Toy.add_bias!(k0, @b_k[0])
      end
      @rope.rotate!(k0, pos_start)
      v0 = x.matmul(@w_v[0])                 # [T, Dh]  (V is not rotated)
      if @has_qkv_bias
        Toy.add_bias!(v0, @b_v[0])
      end
      ks = [k0]
      vs = [v0]
      hkv = 1
      while hkv < @n_kv
        k_h = x.matmul(@w_k[hkv])
        if @has_qkv_bias
          Toy.add_bias!(k_h, @b_k[hkv])
        end
        @rope.rotate!(k_h, pos_start)
        v_h = x.matmul(@w_v[hkv])
        if @has_qkv_bias
          Toy.add_bias!(v_h, @b_v[hkv])
        end
        ks.push(k_h)
        vs.push(v_h)
        hkv += 1
      end

      # 2) per query head: project Q, rotate, attend with the
      # corresponding (shared) K, V.
      head0  = attend(x, 0, ks[0], vs[0], pos_start)
      heads  = [head0]
      hq = 1
      while hq < @n_heads
        grp = hq / @group_size
        heads.push(attend(x, hq, ks[grp], vs[grp], pos_start))
        hq += 1
      end

      concat = Toy.hstack_heads(heads, @n_heads, @d_head, @d_model)   # [T, D]
      concat.matmul(@w_o)                                              # [T, D]
    end

    # One query-head attention.  x: [T, D] → [T, Dh].
    def attend(x, hq, k_h, v_h, pos_start)
      q_h = x.matmul(@w_q[hq])                 # [T, Dh]
      if @has_qkv_bias
        Toy.add_bias!(q_h, @b_q[hq])
      end
      @rope.rotate!(q_h, pos_start)
      scores = q_h.matmul_t(k_h)               # [T, T]
      scores.scale!(@inv_sqrt)
      Toy.causal_mask!(scores)
      Toy.softmax_rows!(scores)
      scores.matmul(v_h)                       # [T, Dh]
    end

    def summary
      "GQAttention(d=" + @d_model.to_s +
        ", n_q=" + @n_heads.to_s + ", n_kv=" + @n_kv.to_s +
        ", d_head=" + @d_head.to_s + ", group=" + @group_size.to_s + ")"
    end
    def param_count
      # Q: n_heads × (d_model × d_head). K/V: n_kv × (d_model × d_head). O: d_model²
      # Plus Q/K/V biases when enabled (Qwen2.x).
      n = @n_heads * @d_model * @d_head +
          2 * @n_kv * @d_model * @d_head +
          @d_model * @d_model
      if @has_qkv_bias
        n = n + @n_heads * @d_head + 2 * @n_kv * @d_head
      end
      n
    end

    def algorithm
      c = Toy::Card.new("GQAttention.forward(x, p_start)", "grouped-query + RoPE")
      c.add_input("x",       "R^{T×D}", "")
      c.add_input("p_start", "ℕ",       "")
      c.add_output("y",      "R^{T×D}", "")
      c.add_hyper("D",       @d_model.to_s)
      c.add_hyper("H",       @n_heads.to_s)
      c.add_hyper("H_kv",    @n_kv.to_s)
      c.add_hyper("g",       @group_size.to_s)
      c.add_hyper("D_h",     @d_head.to_s)
      c.add_param("W_Q^h",  "R^{D×D_h}", "per query head (h=1..H)")
      c.add_param("W_K^j, W_V^j", "R^{D×D_h}",
                  "per KV head (j=1..H_kv); shared across g Q heads")
      if @has_qkv_bias
        c.add_param("b_Q^h",        "R^{D_h}", "per query head")
        c.add_param("b_K^j, b_V^j", "R^{D_h}", "per KV head — Qwen2.x convention")
        c.add_param("W_O", "R^{D×D}", "no output bias")
      else
        c.add_param("W_O", "R^{D×D}", "no biases — Llama convention")
      end
      c.step_loop("j ← 1, …, H_kv", "KV computed once per group")
      if @has_qkv_bias
        c.step_bind("k^j", "RoPE(x · W_K^j + b_K^j, p_start)", "k^j ∈ R^{T×D_h}")
        c.step_update("v^j", "x · W_V^j + b_V^j",              "v^j ∈ R^{T×D_h}", "V not rotated")
      else
        c.step_bind("k^j", "RoPE(x · W_K^j, p_start)",         "k^j ∈ R^{T×D_h}")
        c.step_update("v^j", "x · W_V^j",                      "v^j ∈ R^{T×D_h}", "V not rotated")
      end
      c.step_loop_close
      c.step_loop("h ← 1, …, H", "per query head")
      c.step_bind("j",   "⌊(h−1) / g⌋ + 1", "")
      if @has_qkv_bias
        c.step_bind("q^h", "RoPE(x · W_Q^h + b_Q^h, p_start)", "q^h ∈ R^{T×D_h}")
      else
        c.step_bind("q^h", "RoPE(x · W_Q^h, p_start)",         "q^h ∈ R^{T×D_h}")
      end
      c.step_bind("S^h", "CausalMask(q^h · (k^j)^⊤ / √D_h)", "S^h ∈ R^{T×T}")
      c.step_bind("o^h", "softmax_rows(S^h) · v^j",          "o^h ∈ R^{T×D_h}")
      c.step_loop_close
      c.step_bind("y", "concat(o^1, …, o^H) · W_O", "y ∈ R^{T×D}")
      c.step_return("y")
      c
    end

    def algorithm_card; algorithm.render_pseudocode; end
  end

  # =========================================================================
  # Free-standing helpers. These operate on Mat in place where possible
  # so they read like verbs: `causal_mask!`, `add_bias!`, `softmax_rows!`.
  # =========================================================================

  # SiLU activation, in-place. silu(x) = x / (1 + exp(-x)).
  def self.silu!(m)
    n = m.nrows * m.ncols
    i = 0
    while i < n
      v = m.flat[i]
      m.flat[i] = v / (1.0 + Math.exp(-v))
      i += 1
    end
  end

  # Print a labelled shape line and return the Mat unchanged. Useful
  # to drop in the middle of a forward pass:
  #
  #   x = Toy.tap("after attn", @attn.forward(@ln1.forward(x)))
  #
  # (Implemented with separate puts/print calls to dodge a Spinel quirk
  # where chained String + Mat#shape concat fails to compile.)
  def self.tap(label, x)
    print label
    print ": Mat"
    puts x.shape
    x
  end

  # Like `tap` but with min/max/mean stats — for "is this drifting?"
  # sanity checks during inference.
  def self.tap_info(label, x)
    print label
    print ": "
    puts x.info
    x
  end

  # Pretty-format a parameter count: 49,152 → "49.2K"; 1_233_000 → "1.2M".
  def self.fmt_count(n)
    if n >= 1_000_000_000
      (n.to_f / 1_000_000_000.0).round(2).to_s + "B"
    elsif n >= 1_000_000
      (n.to_f / 1_000_000.0).round(2).to_s + "M"
    elsif n >= 1_000
      (n.to_f / 1_000.0).round(1).to_s + "K"
    else
      n.to_s
    end
  end

  # Elementwise multiply, into `dst` (dst := dst * src). Both have
  # identical shape. Param names avoid `a` / `b` to dodge a Spinel
  # collapse with TinyNN.matmul(a, b) and friends.
  def self.hadamard!(dst, src)
    n = dst.nrows * dst.ncols
    i = 0
    while i < n
      dst.flat[i] = dst.flat[i] * src.flat[i]
      i += 1
    end
  end

  # x[i, j] += b[j], in-place.
  def self.add_bias!(x, b)
    t = x.nrows
    d = x.ncols
    i = 0
    while i < t
      j = 0
      while j < d
        x.flat[i * d + j] = x.flat[i * d + j] + b[j]
        j += 1
      end
      i += 1
    end
  end

  # scores[i, j] = -inf  for j > i. In-place.
  def self.causal_mask!(scores)
    t = scores.nrows
    n = scores.ncols
    i = 0
    while i < t
      j = i + 1
      while j < n
        scores.flat[i * n + j] = NEG_INF_SCORE
        j += 1
      end
      i += 1
    end
  end

  # Row-wise softmax, in-place, numerically stable (max-shift).
  def self.softmax_rows!(m)
    t = m.nrows
    n = m.ncols
    i = 0
    while i < t
      base = i * n
      mx = m.flat[base]
      j = 1
      while j < n
        v = m.flat[base + j]
        if v > mx
          mx = v
        end
        j += 1
      end
      sum = 0.0
      j = 0
      while j < n
        e = Math.exp(m.flat[base + j] - mx)
        m.flat[base + j] = e
        sum = sum + e
        j += 1
      end
      j = 0
      while j < n
        m.flat[base + j] = m.flat[base + j] / sum
        j += 1
      end
      i += 1
    end
  end

  # GeLU (tanh approximation — HF's gelu_new). Returns a fresh Mat.
  # See lib/transformer.rb for GELU_C / GELU_K / GELU_DK constants
  # used here.
  def self.gelu_new(x)
    n   = x.nrows * x.ncols
    out = Mat.new(x.nrows, x.ncols)
    i = 0
    while i < n
      v = x.flat[i]
      u = GELU_C * (v + GELU_K * v * v * v)
      out.flat[i] = 0.5 * v * (1.0 + Math.tanh(u))
      i += 1
    end
    out
  end

  # Pack n_heads × [T, Dh] back into a single [T, D] matrix where
  # head h occupies columns [h*Dh, (h+1)*Dh).
  def self.hstack_heads(per_head, n_heads, d_head, d_model)
    t   = per_head[0].nrows
    out = Mat.new(t, d_model)
    h = 0
    while h < n_heads
      head = per_head[h]
      base = h * d_head
      i = 0
      while i < t
        j = 0
        while j < d_head
          out.flat[i * d_model + (base + j)] = head.flat[i * d_head + j]
          j += 1
        end
        i += 1
      end
      h += 1
    end
    out
  end
end
