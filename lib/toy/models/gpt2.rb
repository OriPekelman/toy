# GPT2LM: inference-only transformer LM matching HF's GPT-2 architecture.
#
# Separate from TransformerLM because TransformerLM carries training
# scaffolding (Gradients, AdamState, layer caches, FFI caches) and uses
# RMSNorm without biases. GPT-2 needs LayerNorm with beta and bias terms
# on every Linear projection — a clean inference-only class is simpler
# than retrofitting all that into TransformerLM.
#
# Architecture (matches transformers.GPT2Model):
#   - learned token + absolute position embeddings, additive
#   - tied input/output embedding (logits = x_final · token_embedᵀ)
#   - per-block: pre-LN, MHA causal, residual, pre-LN, GeLU FFN, residual
#   - bias terms on every Linear: q/k/v/o/ff1/ff2
#   - LayerNorm has gamma and beta
#   - GeLU: tanh approximation (gelu_new in HF)
#
# Native forward only. FFI persistent-graph version lives separately
# (see plan: FullForwardFFICache extension).

require_relative "transformer"   # for Mat

# One transformer block's parameters. Per-head Q/K/V/biases match the
# project's existing Block layout so a future FFI cache can mirror what
# FullForwardFFICache already does.
class GPT2Block
  attr_accessor :ln1_gamma, :ln1_beta, :ln2_gamma, :ln2_beta,
                :w_q, :w_k, :w_v, :w_o, :w_ff1, :w_ff2,
                :b_q, :b_k, :b_v, :b_o, :b_ff1, :b_ff2

  def initialize(d_model, d_head, d_ff, n_heads)
    @ln1_gamma = Array.new(d_model, 1.0)
    @ln1_beta  = Array.new(d_model, 0.0)
    @ln2_gamma = Array.new(d_model, 1.0)
    @ln2_beta  = Array.new(d_model, 0.0)

    # Per-head literal-seed pattern (Array<Mat>, Array<Array<Float>>).
    @w_q = [Mat.new(d_model, d_head)]
    @w_k = [Mat.new(d_model, d_head)]
    @w_v = [Mat.new(d_model, d_head)]
    @b_q = [Array.new(d_head, 0.0)]
    @b_k = [Array.new(d_head, 0.0)]
    @b_v = [Array.new(d_head, 0.0)]
    h = 1
    while h < n_heads
      @w_q.push(Mat.new(d_model, d_head))
      @w_k.push(Mat.new(d_model, d_head))
      @w_v.push(Mat.new(d_model, d_head))
      @b_q.push(Array.new(d_head, 0.0))
      @b_k.push(Array.new(d_head, 0.0))
      @b_v.push(Array.new(d_head, 0.0))
      h += 1
    end

    @w_o   = Mat.new(d_model, d_model)
    @w_ff1 = Mat.new(d_model, d_ff)
    @w_ff2 = Mat.new(d_ff,    d_model)
    @b_o   = Array.new(d_model, 0.0)
    @b_ff1 = Array.new(d_ff,   0.0)
    @b_ff2 = Array.new(d_model, 0.0)
  end
end

class GPT2LM
  attr_accessor :vocab_size, :d_model, :d_ff, :n_heads, :d_head,
                :n_layers, :context_length, :ln_eps,
                :token_embed, :pos_embed,
                :ln_f_gamma, :ln_f_beta, :gpt2_blocks

  def initialize(vocab_size, d_model, d_ff, n_heads, n_layers, context_length)
    @vocab_size     = vocab_size
    @d_model        = d_model
    @d_ff           = d_ff
    @n_heads        = n_heads
    @d_head         = d_model / n_heads
    @n_layers       = n_layers
    @context_length = context_length
    @ln_eps         = RMS_EPS_DEFAULT  # LayerNorm epsilon; same value

    @token_embed = Mat.new(vocab_size, d_model)
    @pos_embed   = Mat.new(context_length, d_model)
    @ln_f_gamma  = Array.new(d_model, 1.0)
    @ln_f_beta   = Array.new(d_model, 0.0)

    @gpt2_blocks = [GPT2Block.new(d_model, @d_head, d_ff, n_heads)]
    li = 1
    while li < n_layers
      @gpt2_blocks.push(GPT2Block.new(d_model, @d_head, d_ff, n_heads))
      li += 1
    end
  end

  # x[i, :] = token_embed[token_ids[i]] + pos_embed[start_pos + i]
  def embed(token_ids, start_pos)
    t = token_ids.length
    out = Mat.new(t, @d_model)
    i = 0
    while i < t
      tid = token_ids[i]
      j = 0
      while j < @d_model
        out.flat[i * @d_model + j] =
          @token_embed.flat[tid * @d_model + j] +
          @pos_embed.flat[(start_pos + i) * @d_model + j]
        j += 1
      end
      i += 1
    end
    out
  end

  # LayerNorm: y_j = (x_j - mean) / sqrt(var + eps) * gamma_j + beta_j
  # per row. New Mat returned (caller may need x unchanged for residual).
  def layer_norm(x, gamma, beta)
    d = gamma.length
    t = x.nrows
    out = Mat.new(t, d)
    i = 0
    while i < t
      sum = 0.0
      j = 0
      while j < d
        sum = sum + x.flat[i * d + j]
        j += 1
      end
      mean = sum / d
      sumsq = 0.0
      j = 0
      while j < d
        v = x.flat[i * d + j] - mean
        sumsq = sumsq + v * v
        j += 1
      end
      inv = 1.0 / Math.sqrt(sumsq / d + @ln_eps)
      j = 0
      while j < d
        n = (x.flat[i * d + j] - mean) * inv
        out.flat[i * d + j] = n * gamma[j] + beta[j]
        j += 1
      end
      i += 1
    end
    out
  end

  # Broadcast-add a length-d row bias to every row of x, in-place.
  def add_bias!(x, bias)
    d = x.ncols
    t = x.nrows
    i = 0
    while i < t
      j = 0
      while j < d
        x.flat[i * d + j] = x.flat[i * d + j] + bias[j]
        j += 1
      end
      i += 1
    end
  end

  # Row-wise softmax with numerical-stability max-shift, in place on m.
  def softmax_rows!(m)
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

  # Causal mask: for each row i, set scores[i, j] = -1e30 for j > i.
  def apply_causal_mask!(scores)
    t = scores.nrows
    n = scores.ncols
    i = 0
    while i < t
      j = i + 1
      while j < n
        scores.flat[i * n + j] = -1.0e30
        j += 1
      end
      i += 1
    end
  end

  # n_heads × (T × d_head) → (T × d_model), packing each head's columns
  # back into the contiguous block of width d_head at offset h*d_head.
  def hstack_heads(per_head)
    t = per_head[0].nrows
    out = Mat.new(t, @d_model)
    h = 0
    while h < @n_heads
      head = per_head[h]
      base = h * @d_head
      i = 0
      while i < t
        j = 0
        while j < @d_head
          out.flat[i * @d_model + (base + j)] = head.flat[i * @d_head + j]
          j += 1
        end
        i += 1
      end
      h += 1
    end
    out
  end

  def self_attention_head(h_in, block, head_idx, inv_sqrt)
    q = h_in.matmul(block.w_q[head_idx])
    add_bias!(q, block.b_q[head_idx])
    k = h_in.matmul(block.w_k[head_idx])
    add_bias!(k, block.b_k[head_idx])
    v = h_in.matmul(block.w_v[head_idx])
    add_bias!(v, block.b_v[head_idx])

    scores = q.matmul_t(k)
    scores.scale!(inv_sqrt)
    apply_causal_mask!(scores)
    softmax_rows!(scores)
    scores.matmul(v)
  end

  def self_attention(h_in, block)
    inv_sqrt = 1.0 / Math.sqrt(@d_head)

    head0 = self_attention_head(h_in, block, 0, inv_sqrt)
    per_head = [head0]
    hi = 1
    while hi < @n_heads
      per_head.push(self_attention_head(h_in, block, hi, inv_sqrt))
      hi += 1
    end

    concat = hstack_heads(per_head)
    proj = concat.matmul(block.w_o)
    add_bias!(proj, block.b_o)
    proj
  end

  # FFN: gelu_new(h · W_ff1 + b_ff1) · W_ff2 + b_ff2.
  # GeLU is the tanh approximation; constants GELU_C / GELU_K live in
  # lib/transformer.rb.
  def feed_forward(h, block)
    pre = h.matmul(block.w_ff1)
    add_bias!(pre, block.b_ff1)
    hidden = Mat.new(pre.nrows, pre.ncols)
    n = pre.nrows * pre.ncols
    i = 0
    while i < n
      x = pre.flat[i]
      u = GELU_C * (x + GELU_K * x * x * x)
      hidden.flat[i] = 0.5 * x * (1.0 + Math.tanh(u))
      i += 1
    end
    out = hidden.matmul(block.w_ff2)
    add_bias!(out, block.b_ff2)
    out
  end

  # One transformer block: pre-LN → MHA → residual → pre-LN → FFN → residual.
  # x is mutated in place via add!; returned for chaining.
  def transformer_block(x, block)
    h_norm  = layer_norm(x, block.ln1_gamma, block.ln1_beta)
    attn    = self_attention(h_norm, block)
    x.add!(attn)

    h_norm2 = layer_norm(x, block.ln2_gamma, block.ln2_beta)
    ff      = feed_forward(h_norm2, block)
    x.add!(ff)

    x
  end

  # embed -> N blocks -> final LN -> tied unembed -> logits[T, vocab]
  def forward(token_ids, start_pos)
    x = embed(token_ids, start_pos)

    li = 0
    while li < @n_layers
      x = transformer_block(x, @gpt2_blocks[li])
      li += 1
    end

    x_final = layer_norm(x, @ln_f_gamma, @ln_f_beta)
    # logits = x_final · token_embedᵀ  (tied output embedding)
    x_final.matmul_t(@token_embed)
  end
end
