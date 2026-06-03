# Spinel-compatible decoder-only Transformer LM.
#
# Same architecture as transformer.rb (multi-head attention, RMSNorm,
# residuals, FFN, cross-entropy, Adam, KV cache) but with storage layouts
# that Spinel's type inference handles cleanly:
#
#   • Mat is a class with flat 1D Float storage. Spinel cannot currently
#     infer "Array of FloatArray" from `Array.new(n) { Array.new(m, 0.0) }`,
#     so all 2D matrices are flat under the hood.
#   • Block / cache / KV cache structures are classes, not hashes —
#     polymorphic hashes with mixed value types confuse Spinel's codegen.
#   • Arrays of objects are built with `[seed] + push` instead of
#     `Array.new(n) { obj }` (Spinel takes an IntArray-only fast path).
#   • Field names: nrows/ncols, not rows/cols (the latter triggers a
#     name-collision in Spinel's iterative type inference).
#
# Requires the Spinel patch to infer_ivar_init_type that propagates the
# fill-type from `Array.new(n, 0.0)` into the containing class's struct.

# The FFN's two matmuls go through TinyNN (ggml-CPU FFI) when this is
# true.  Off by default to keep the toy zero-dep; flip on to use the
# bridge and accelerate at real-LLM scale (see tinynn/README.md).
USE_FFI_MATMUL = false

# GeLU tanh-approximation constants (the GPT-2 formula; identical to
# torch.nn.functional.gelu(..., approximate='tanh'). Defined here so
# the forward, backward, and any per-tensor variant agree byte-for-byte.
#
#   gelu(x) = 0.5 * x * (1 + tanh( GELU_C * (x + GELU_K * x^3) ))
#
GELU_C  = 0.7978845608028654   # sqrt(2/π)
GELU_K  = 0.044715             # cubic coefficient
GELU_DK = 0.134145             # 3 * GELU_K  (derivative of cubic term)

# RMSNorm / LayerNorm default epsilon. Matches Llama / SmolLM2 /
# GPT-2 conventions. Individual instances can override via their
# own @eps ivar; this is the fallback used by the row-level helpers.
RMS_EPS_DEFAULT = 1.0e-5

# Numerical floor for probabilities going into log() in cross_entropy.
# 1e-12 is safely above the F32 / F64 underflow threshold and
# matches PyTorch's "label smoothing" default clip.
LOG_PROB_FLOOR = 1.0e-12

# Causal-mask sentinel: attention scores set to this become ~0 after
# softmax (Math.exp(-1e30) underflows cleanly to 0.0). Avoid -Infinity
# because (Float::INFINITY - Float::INFINITY) is NaN if downstream
# code rescales or subtracts max.
NEG_INF_SCORE = -1.0e30
# tinynn is always required so FFNFFICache is defined (it lives in
# lib/tinynn.rb). The require itself doesn't run any FFI code; only
# feed_forward_ffi's USE_FFI_MATMUL-gated branch does. With
# USE_FFI_MATMUL=false the FFI methods are dead code that Spinel's
# DCE drops, but the library libs still get linked.

# tinynn is always required so FFNFFICache is defined (it lives in
# lib/tinynn.rb). Set USE_FFI_MATMUL=true to enable FFI dispatch.
require_relative "../../tinynn"

# One-shot diagnostic: when MAT_SHAPES_ON=true, every matmul prints
# its shape triple on stdout (post-process with sort | uniq -c). Off
# by default; the const must exist for Spinel to resolve the read.
MAT_SHAPES_ON = (ENV["MAT_SHAPES"] || "") == "1"

# ============================================================================
#   Mat: 2D float matrix, flat-storage. Indexed as flat[i * ncols + j].
# ============================================================================
class Mat
  attr_accessor :nrows, :ncols, :flat

  def initialize(nrows, ncols)
    @nrows = nrows
    @ncols = ncols
    @flat  = Array.new(nrows * ncols, 0.0)
  end

  def fill_random(scale)
    # Spinel's `rand` (no args) is the C `rand()` returning a large int,
    # not Ruby's [0.0, 1.0) float. Use `rand(N)` which behaves the same
    # in both: an integer in [0, N).
    n = @nrows * @ncols
    i = 0
    while i < n
      @flat[i] = (rand(2000).to_f / 1000.0 - 1.0) * scale
      i += 1
    end
  end

  def fill_zero
    n = @nrows * @ncols
    i = 0
    while i < n
      @flat[i] = 0.0
      i += 1
    end
  end

  # (m × n) · (n × p) → (m × p) — using local accumulator (faster than
  # repeated indexed writes since each += would re-load the receiver).
  def matmul(other)
    _trace = TinyNN.tnn_trace_begin("Mat.matmul")
    m = @nrows
    n = @ncols
    p = other.ncols
    if MAT_SHAPES_ON; puts "MAT_SHAPE matmul " + m.to_s + " " + n.to_s + " " + p.to_s; end
    out = Mat.new(m, p)
    i = 0
    while i < m
      j = 0
      while j < p
        s = 0.0
        k = 0
        while k < n
          s = s + @flat[i * n + k] * other.flat[k * p + j]
          k += 1
        end
        out.flat[i * p + j] = s
        j += 1
      end
      i += 1
    end
    TinyNN.tnn_trace_end("Mat.matmul", _trace)
    out
  end

  # self · otherᵀ where other has the same column count as self.
  def matmul_t(other)
    _trace = TinyNN.tnn_trace_begin("Mat.matmul_t")
    m = @nrows
    n = @ncols
    p = other.nrows                # other is (p × n) so otherᵀ is (n × p)
    if MAT_SHAPES_ON; puts "MAT_SHAPE matmul_t " + m.to_s + " " + n.to_s + " " + p.to_s; end
    out = Mat.new(m, p)
    i = 0
    while i < m
      j = 0
      while j < p
        s = 0.0
        k = 0
        while k < n
          s = s + @flat[i * n + k] * other.flat[j * n + k]
          k += 1
        end
        out.flat[i * p + j] = s
        j += 1
      end
      i += 1
    end
    TinyNN.tnn_trace_end("Mat.matmul_t", _trace)
    out
  end

  # selfᵀ · other where self is (n × m) and other is (n × p) → (m × p)
  def t_matmul(other)
    _trace = TinyNN.tnn_trace_begin("Mat.t_matmul")
    n = @nrows
    m = @ncols
    p = other.ncols
    if MAT_SHAPES_ON; puts "MAT_SHAPE t_matmul " + n.to_s + " " + m.to_s + " " + p.to_s; end
    out = Mat.new(m, p)
    i = 0
    while i < m
      j = 0
      while j < p
        s = 0.0
        k = 0
        while k < n
          s = s + @flat[k * m + i] * other.flat[k * p + j]
          k += 1
        end
        out.flat[i * p + j] = s
        j += 1
      end
      i += 1
    end
    TinyNN.tnn_trace_end("Mat.t_matmul", _trace)
    out
  end

  def transpose
    out = Mat.new(@ncols, @nrows)
    i = 0
    while i < @nrows
      j = 0
      while j < @ncols
        out.flat[j * @nrows + i] = @flat[i * @ncols + j]
        j += 1
      end
      i += 1
    end
    out
  end

  # Element-wise sum. Renamed from `add` to dodge a Spinel
  # polymorphic-dispatch with Tep::Router#add (different arity, but
  # Spinel's arg-type-narrowing in 4024216 doesn't catch the arity
  # mismatch). add! (mutating, suffix-banged) is unaffected since
  # method NAME (not signature) is what collides.
  def plus(other)
    _trace = TinyNN.tnn_trace_begin("Mat.plus")
    out = Mat.new(@nrows, @ncols)
    n = @nrows * @ncols
    i = 0
    while i < n
      out.flat[i] = @flat[i] + other.flat[i]
      i += 1
    end
    TinyNN.tnn_trace_end("Mat.plus", _trace)
    out
  end

  def add!(other)
    _trace = TinyNN.tnn_trace_begin("Mat.add!")
    n = @nrows * @ncols
    i = 0
    while i < n
      @flat[i] += other.flat[i]
      i += 1
    end
    TinyNN.tnn_trace_end("Mat.add!", _trace)
    self
  end

  def scale!(s)
    _trace = TinyNN.tnn_trace_begin("Mat.scale!")
    n = @nrows * @ncols
    i = 0
    while i < n
      @flat[i] = @flat[i] * s
      i += 1
    end
    TinyNN.tnn_trace_end("Mat.scale!", _trace)
    self
  end

  def get(i, j)
    @flat[i * @ncols + j]
  end

  def set_at(i, j, v)
    @flat[i * @ncols + j] = v
  end

  # --- introspection helpers -----------------------------------------------
  # Goal: any reader can stop in the middle of a forward pass, type
  # `puts x.shape` or `puts x.info`, and see what they're holding.

  # "[5, 768]" — concise shape string for prints and logs.
  def shape
    "[" + @nrows.to_s + ", " + @ncols.to_s + "]"
  end

  # "Mat[5, 768] min=-2.34 max=1.97 mean=0.012" — shape + a few summary
  # statistics. Useful for "is my activation drifting?" sanity checks.
  def info
    n = @nrows * @ncols
    if n == 0
      return "Mat[" + @nrows.to_s + ", " + @ncols.to_s + "] (empty)"
    end
    mn = @flat[0]
    mx = @flat[0]
    sum = 0.0
    i = 0
    while i < n
      v = @flat[i]
      if v < mn
        mn = v
      end
      if v > mx
        mx = v
      end
      sum = sum + v
      i += 1
    end
    mean = sum / n.to_f
    "Mat[" + @nrows.to_s + ", " + @ncols.to_s +
      "] min=" + mn.to_s + " max=" + mx.to_s + " mean=" + mean.to_s
  end
end

# ============================================================================
#   Block: a transformer block's parameters.
#
#   norm1_gamma, norm2_gamma : Array of Float (length d_model)
#   w_q, w_k, w_v            : Array of Mat (one per head, each d_model × d_head)
#   w_o, w_ff1, w_ff2        : Mat
# ============================================================================
class Block
  attr_accessor :norm1_gamma, :norm2_gamma,
                :w_q, :w_k, :w_v, :w_o, :w_ff1, :w_ff2

  # Zero-initializes everything. Call .fill_random_all(scale) for params,
  # leave as-is for gradients / Adam moments.
  def initialize(d_model, d_head, d_ff, n_heads)
    @norm1_gamma = Array.new(d_model, 1.0)
    @norm2_gamma = Array.new(d_model, 1.0)

    # Per-head matrices: literal-seed pattern so Spinel types as PtrArray of Mat.
    @w_q = [Mat.new(d_model, d_head)]
    @w_k = [Mat.new(d_model, d_head)]
    @w_v = [Mat.new(d_model, d_head)]
    h = 1
    while h < n_heads
      @w_q.push(Mat.new(d_model, d_head))
      @w_k.push(Mat.new(d_model, d_head))
      @w_v.push(Mat.new(d_model, d_head))
      h += 1
    end

    @w_o   = Mat.new(d_model, d_model)
    @w_ff1 = Mat.new(d_model, d_ff)
    @w_ff2 = Mat.new(d_ff,    d_model)
  end

  def fill_random_all(scale)
    h = 0
    while h < @w_q.length
      @w_q[h].fill_random(scale)
      @w_k[h].fill_random(scale)
      @w_v[h].fill_random(scale)
      h += 1
    end
    @w_o.fill_random(scale)
    @w_ff1.fill_random(scale)
    @w_ff2.fill_random(scale)
  end

  def fill_zero
    n = @norm1_gamma.length
    i = 0
    while i < n
      @norm1_gamma[i] = 0.0
      @norm2_gamma[i] = 0.0
      i += 1
    end
    h = 0
    while h < @w_q.length
      @w_q[h].fill_zero
      @w_k[h].fill_zero
      @w_v[h].fill_zero
      h += 1
    end
    @w_o.fill_zero
    @w_ff1.fill_zero
    @w_ff2.fill_zero
  end
end

# ============================================================================
#   Caches — per-sublayer activations needed by backward.
# ============================================================================

class FFCache
  attr_accessor :pre, :hidden
  def initialize(pre, hidden)
    @pre = pre
    @hidden = hidden
  end
end

class HeadCache
  attr_accessor :q, :k, :v, :attn, :head_out
  def initialize(q, k, v, attn, head_out)
    @q = q
    @k = k
    @v = v
    @attn = attn
    @head_out = head_out
  end
end

class AttnCache
  attr_accessor :per_head, :concat
  def initialize(per_head, concat)
    @per_head = per_head
    @concat = concat
  end
end

class LayerCache
  attr_accessor :rms1, :h_norm1, :attn_cache, :x_attn,
                :rms2, :h_norm2, :ff_cache, :x_out
  def initialize
    @rms1 = nil
    @h_norm1 = nil
    @attn_cache = nil
    @x_attn = nil
    @rms2 = nil
    @h_norm2 = nil
    @ff_cache = nil
    @x_out = nil
  end
end

# Full forward cache used by backward.
class ForwardCache
  attr_accessor :token_ids, :x_embed, :layers,
                :x_block_out, :x_final, :rms_final, :logits
  def initialize
    @token_ids = nil
    @x_embed = nil
    @layers = nil
    @x_block_out = nil
    @x_final = nil
    @rms_final = nil
    @logits = nil
  end
end

# Output of cross_entropy_grad: dL/dlogits matrix + scalar loss.
class LossResult
  attr_accessor :dlogits, :loss
  def initialize(dlogits, loss); @dlogits = dlogits; @loss = loss; end
end

# Gradients for the whole model. Structurally a mirror of TransformerLM's
# parameters: same Block-shaped per-layer grads (we reuse ZeroBlock).
class Gradients
  # Tied embeddings: lm_head shares storage with token_embed (lm_head[d,v] ==
  # token_embed[v,d]), so there's a single token_embed gradient that
  # accumulates contributions from BOTH the input embedding lookup and
  # the output (unembed) projection.
  attr_accessor :token_embed, :pos_embed,
                :norm_final_gamma, :blocks, :loss

  def initialize(vocab_size, d_model, d_ff, n_heads, d_head, n_layers, context_length)
    @token_embed = Mat.new(vocab_size, d_model)
    @pos_embed   = Mat.new(context_length, d_model)
    @norm_final_gamma = Array.new(d_model, 0.0)
    @blocks = [Block.new(d_model, d_head, d_ff, n_heads)]
    li = 1
    while li < n_layers
      @blocks.push(Block.new(d_model, d_head, d_ff, n_heads))
      li += 1
    end
    @loss = 0.0
  end

  def fill_zero
    @token_embed.fill_zero
    @pos_embed.fill_zero
    n = @norm_final_gamma.length
    i = 0
    while i < n
      @norm_final_gamma[i] = 0.0
      i += 1
    end
    bi = 0
    while bi < @blocks.length
      @blocks[bi].fill_zero
      bi += 1
    end
    @loss = 0.0
  end
end

# Adam optimizer state: per-parameter first and second moments (m, v),
# plus the accumulating bias-correction products bc1=β1ᵗ, bc2=β2ᵗ. We
# maintain bc1/bc2 as running products (one multiply per step) instead
# of computing β**t each step (one pow() call) — pure perf choice;
# Spinel handles `Float ** Int` cleanly.
#
# m and v are Gradients-shaped (same per-parameter structure as the
# accumulator) so we can reuse Gradients#fill_zero to initialize them.
class AdamState
  attr_accessor :m, :v, :bc1, :bc2

  def initialize(vocab_size, d_model, d_ff, n_heads, d_head, n_layers, context_length)
    @m = Gradients.new(vocab_size, d_model, d_ff, n_heads, d_head, n_layers, context_length)
    @v = Gradients.new(vocab_size, d_model, d_ff, n_heads, d_head, n_layers, context_length)
    @m.fill_zero
    @v.fill_zero
    @bc1 = 1.0   # β1^0
    @bc2 = 1.0   # β2^0
  end
end

# Result wrappers — Spinel doesn't reliably handle [Mat, OtherType] tuple
# destructuring (the tuple gets typed as PolyArray and member access boxes
# the values). One small class per return shape keeps things explicit.
class NormResult
  attr_accessor :y, :rms
  def initialize(y, rms); @y = y; @rms = rms; end
end

class AttnResult
  attr_accessor :proj, :cache
  def initialize(proj, cache); @proj = proj; @cache = cache; end
end

class FFResult
  attr_accessor :out, :cache
  def initialize(o, c); @out = o; @cache = c; end
end

class BlockResult
  attr_accessor :x_out, :cache
  def initialize(x_out, cache); @x_out = x_out; @cache = cache; end
end

# ============================================================================
#   TransformerLM
# ============================================================================
class TransformerLM
  attr_accessor :vocab_size, :d_model, :d_ff, :n_heads, :d_head,
                :n_layers, :context_length,
                :token_embed, :pos_embed,
                :norm_final_gamma, :blocks, :vocabulary

  def initialize(vocab_size, d_model, d_ff, n_heads, n_layers, context_length)
    @vocab_size     = vocab_size
    @d_model        = d_model
    @d_ff           = d_ff
    @n_heads        = n_heads
    @d_head         = d_model / n_heads
    @n_layers       = n_layers
    @context_length = context_length

    s = 1.0 / Math.sqrt(d_model)

    @token_embed = Mat.new(vocab_size, d_model)
    @token_embed.fill_random(s)

    @pos_embed = Mat.new(context_length, d_model)
    @pos_embed.fill_random(s)

    # Tied embeddings: lm_head is @token_embed used in transposed form
    # at unembed time (logits = x_final · token_embedᵀ). No separate
    # @lm_head matrix; the unembed gradient accumulates into the same
    # token_embed grad slot as the input-side embedding lookup.

    @norm_final_gamma = Array.new(d_model, 1.0)

    # Vocabulary: seeded with one placeholder so Spinel infers it as
    # an array of strings; callers should set the real vocab after construction.
    @vocabulary = ["?"]

    # Inline Block.new in the literal — Spinel's scan_ivars runs before
    # local-variable types are inferred, so storing through a temp would
    # mistype @blocks's element class.
    @blocks = [Block.new(d_model, @d_head, d_ff, n_heads)]
    @blocks[0].fill_random_all(s)
    li = 1
    while li < n_layers
      @blocks.push(Block.new(d_model, @d_head, d_ff, n_heads))
      @blocks[li].fill_random_all(s)
      li += 1
    end

    # Pre-allocate layer caches so the array's element type is fixed at
    # construction time. Forward populates fields on these existing objects.
    @layer_caches = [LayerCache.new]
    li = 1
    while li < n_layers
      @layer_caches.push(LayerCache.new)
      li += 1
    end

    # Per-block persistent FFI caches for feed_forward. Lazily realized
    # on first call so we don't need to decide T (sequence length) at
    # model-construction time. With USE_FFI_MATMUL=false they sit
    # unused; the cost is one cheap object alloc per block.
    @ffn_ffi_caches = [FFNFFICache.new]
    li = 1
    while li < n_layers
      @ffn_ffi_caches.push(FFNFFICache.new)
      li += 1
    end
  end

  attr_accessor :ffn_ffi_caches

  attr_accessor :layer_caches

  # ----- Forward -----

  # x[i] = token_embed[token_ids[i]] + pos_embed[start_pos + i]
  def embed(token_ids, start_pos)
    t = token_ids.length
    out = Mat.new(t, @d_model)
    i = 0
    while i < t
      tok_id = token_ids[i]
      j = 0
      while j < @d_model
        out.flat[i * @d_model + j] =
          @token_embed.flat[tok_id * @d_model + j] +
          @pos_embed.flat[(start_pos + i) * @d_model + j]
        j += 1
      end
      i += 1
    end
    out
  end

  # RMSNorm: y_j = gamma_j * x_j / sqrt(mean(x²) + eps),  per row.
  # Returns a NormResult holding the normed Mat and the per-row rms.
  def rms_norm(x, gamma)
    eps = RMS_EPS_DEFAULT
    d = gamma.length
    t = x.nrows
    rms = Array.new(t, 0.0)
    out = Mat.new(t, d)

    i = 0
    while i < t
      sumsq = 0.0
      j = 0
      while j < d
        v = x.flat[i * d + j]
        sumsq += v * v
        j += 1
      end
      r = Math.sqrt(sumsq / d + eps)
      rms[i] = r
      j = 0
      while j < d
        out.flat[i * d + j] = x.flat[i * d + j] * gamma[j] / r
        j += 1
      end
      i += 1
    end

    NormResult.new(out, rms)
  end

  # Row-wise softmax with numerical-stability max-shift, in place on `m`.
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
        sum += e
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

  # Causal mask: for each row i, set scores[i, j] = -1e30 for j > query_offset + i.
  def apply_causal_mask!(scores, query_offset)
    t = scores.nrows
    n = scores.ncols
    i = 0
    while i < t
      first_masked = query_offset + i + 1
      j = first_masked
      while j < n
        scores.flat[i * n + j] = NEG_INF_SCORE
        j += 1
      end
      i += 1
    end
  end

  # Concatenate per-head outputs side by side: n_heads × (T × d_head) → (T × d_model)
  def hstack_heads(per_head)
    t = per_head[0].head_out.nrows
    out = Mat.new(t, @d_model)
    h = 0
    while h < @n_heads
      head = per_head[h].head_out
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

  # Multi-head self-attention. Returns AttnResult.
  def self_attention(h_in, block)
    # Force h_in's type inference via an early Mat-typed access.
    t_seq = h_in.nrows
    inv_sqrt = 1.0 / Math.sqrt(@d_head)

    # Build per-head caches with the seed-then-push pattern.
    head0 = self_attention_head(h_in, block, 0, inv_sqrt)
    per_head = [head0]
    hi = 1
    while hi < @n_heads
      per_head.push(self_attention_head(h_in, block, hi, inv_sqrt))
      hi += 1
    end

    concat = hstack_heads(per_head)
    proj   = concat.matmul(block.w_o)

    AttnResult.new(proj, AttnCache.new(per_head, concat))
  end

  def self_attention_head(h_in, block, head_idx, inv_sqrt)
    q = h_in.matmul(block.w_q[head_idx])
    k = h_in.matmul(block.w_k[head_idx])
    v = h_in.matmul(block.w_v[head_idx])

    # scores = (Q · Kᵀ) / sqrt(d_head)
    scores = q.matmul_t(k)
    scores.scale!(inv_sqrt)
    apply_causal_mask!(scores, 0)

    softmax_rows!(scores)
    head_out = scores.matmul(v)

    HeadCache.new(q, k, v, scores, head_out)
  end

  # FFN: gelu(h · W_ff1) · W_ff2.  Returns (out_mat, FFCache).
  # GeLU uses the tanh approximation: 0.5 x (1 + tanh(c (x + 0.044715 x³))),
  # c = √(2/π).
  def feed_forward(h, block)
    pre = h.matmul(block.w_ff1)
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
    FFResult.new(out, FFCache.new(pre, hidden))
  end

  # Persistent-session FFI variant of feed_forward. Single ggml session
  # runs the chain `mul_mat(w1_t, h) -> gelu -> mul_mat(w2_t, hidden)`
  # in one dispatch; activations live in ggml memory between matmul1
  # and matmul2 (no host round-trip for GeLU).
  #
  # Operand-order trick: with matmul1 = mul_mat(w1_t, h), the result's
  # ne0 is d_ff -- which equals matmul2's k -- so the chain composes
  # without an intermediate transpose. All three result tensors then
  # read back as a straight row-major memcpy.
  def feed_forward_ffi(h, block, ffi_cache)
    t_seq = h.nrows
    d_model = h.ncols
    d_ff = block.w_ff1.ncols

    if !ffi_cache.realized
      ffi_cache.realize_for(t_seq, d_model, d_ff)
    end

    TinyNN.upload_row_major(ffi_cache.sess, ffi_cache.t_h, h)
    TinyNN.stage_transposed_and_upload(ffi_cache.sess, ffi_cache.t_w1_t, block.w_ff1)
    TinyNN.stage_transposed_and_upload(ffi_cache.sess, ffi_cache.t_w2_t, block.w_ff2)
    TinyNN.tnn_compute(ffi_cache.sess)
    pre    = TinyNN.download_row_major(ffi_cache.sess, ffi_cache.t_pre,    t_seq, d_ff)
    hidden = TinyNN.download_row_major(ffi_cache.sess, ffi_cache.t_hidden, t_seq, d_ff)
    out    = TinyNN.download_row_major(ffi_cache.sess, ffi_cache.t_out,    t_seq, d_model)

    FFResult.new(out, FFCache.new(pre, hidden))
  end

  # NOTE: forward via FullForwardFFICache is available as a free
  # function on a separate cache object — see tinynn/full_forward_bench
  # for the usage pattern. We avoid making it a TransformerLM instance
  # method because Spinel's iterative type inference, when forced to
  # process the cache's `t_seq`/`d_model`/`d_ff` ivars together with
  # FFNFFICache's same-named ivars and TransformerLM's own d_model,
  # collapses Mat.@nrows/@ncols to a polymorphic (mrb_float / sp_RbVal)
  # type and pollutes Mat throughout the program (sp_TinyNN_cls_scale
  # then fails to compile its `(int)mat->iv_nrows` cast). The standalone
  # FullForwardFFICache + caller-side weight upload sidesteps this.

  # Full forward pass. Writes intermediates into @layer_caches and @cache,
  # which are pre-allocated so their types are unambiguous to Spinel.
  # Returns the logits Mat (T × vocab_size).
  def forward(token_ids)
    cache = ForwardCache.new
    cache.token_ids = token_ids

    x = embed(token_ids, 0)
    cache.x_embed = x

    x_cur = x
    li = 0
    while li < @n_layers
      transformer_block_into(x_cur, @blocks[li], @layer_caches[li], @ffn_ffi_caches[li])
      x_cur = @layer_caches[li].x_out
      li += 1
    end
    cache.layers = @layer_caches
    cache.x_block_out = x_cur

    nr = rms_norm(x_cur, @norm_final_gamma)
    cache.x_final   = nr.y
    cache.rms_final = nr.rms

    # Tied unembed: logits[t,v] = Σ_d x_final[t,d] · token_embed[v,d]
    cache.logits = nr.y.matmul_t(@token_embed)
    @cache = cache
    cache.logits
  end

  attr_accessor :cache

  # Same as transformer_block but writes into a pre-existing LayerCache.
  # ffi_cache is the persistent-session FFNFFICache for this block;
  # used only when USE_FFI_MATMUL is true.
  def transformer_block_into(x, block, cache, ffi_cache)
    nr1 = rms_norm(x, block.norm1_gamma)
    h1  = nr1.y
    cache.h_norm1 = h1
    cache.rms1    = nr1.rms

    sa = self_attention(h1, block)
    cache.attn_cache = sa.cache
    x_attn = x.plus(sa.proj)
    cache.x_attn = x_attn

    nr2 = rms_norm(x_attn, block.norm2_gamma)
    h2  = nr2.y
    cache.h_norm2 = h2
    cache.rms2    = nr2.rms

    if USE_FFI_MATMUL
      ff = feed_forward_ffi(h2, block, ffi_cache)
    else
      ff = feed_forward(h2, block)
    end
    cache.ff_cache = ff.cache
    x_out = x_attn.plus(ff.out)
    cache.x_out = x_out
  end

  # Cross-entropy on next-token prediction. dL/dlogits = softmax(logits) - one_hot(target).
  # Loss is averaged over the (T-1) prediction positions.
  def cross_entropy_grad(logits, token_ids)
    n_pred = token_ids.length - 1
    dlogits = Mat.new(logits.nrows, logits.ncols)
    total_loss = 0.0
    if n_pred <= 0
      return LossResult.new(dlogits, 0.0)
    end
    inv_n = 1.0 / n_pred
    v = logits.ncols

    i = 0
    while i < n_pred
      base = i * v
      mx = logits.flat[base]
      j = 1
      while j < v
        val = logits.flat[base + j]
        if val > mx
          mx = val
        end
        j += 1
      end
      sum = 0.0
      j = 0
      while j < v
        e = Math.exp(logits.flat[base + j] - mx)
        sum += e
        j += 1
      end
      target = token_ids[i + 1]
      target_logit = logits.flat[base + target]
      pt = Math.exp(target_logit - mx) / sum
      if pt < LOG_PROB_FLOOR
        pt = LOG_PROB_FLOOR
      end
      total_loss -= Math.log(pt)

      j = 0
      while j < v
        p = Math.exp(logits.flat[base + j] - mx) / sum
        dlogits.flat[base + j] = p * inv_n
        j += 1
      end
      ti = base + target
      dlogits.flat[ti] = dlogits.flat[ti] - inv_n

      i += 1
    end

    LossResult.new(dlogits, total_loss / n_pred)
  end

  # ----- Backward -----
  #
  # Each backward helper writes its parameter-gradients into a target
  # ZeroBlock (or a Mat passed in) and returns the gradient w.r.t. its
  # input as a single Mat. Avoiding tuple returns keeps Spinel's type
  # inference happy.

  # RMSNorm backward.
  #   For y = gamma * x / r,  with r = sqrt(mean(x²) + eps):
  #     dL/dx_k    = (dy_k * gamma_k - x_k * coef) / r,
  #         coef = (Σ_j dy_j * gamma_j * x_j) / (d * r²)
  #     dL/dgamma_j (summed over rows) += dy_j * x_j / r
  # `rms` is the FloatArray of per-row r values cached from the forward
  # pass — saves recomputing sumsq.
  def rms_norm_backward(x, gamma, rms, dy, target_dgamma)
    d = gamma.length
    t_seq = x.nrows
    dx = Mat.new(t_seq, d)

    i = 0
    while i < t_seq
      r = rms[i]

      inner = 0.0
      j = 0
      while j < d
        inner += dy.flat[i * d + j] * gamma[j] * x.flat[i * d + j]
        j += 1
      end
      coef = inner / (d * r * r)

      j = 0
      while j < d
        dx.flat[i * d + j] =
          (dy.flat[i * d + j] * gamma[j] - x.flat[i * d + j] * coef) / r
        target_dgamma[j] = target_dgamma[j] +
                           dy.flat[i * d + j] * x.flat[i * d + j] / r
        j += 1
      end
      i += 1
    end

    dx
  end

  # Row-wise softmax backward (for attention).
  #   d_scores[i,j] = attn[i,j] * (d_attn[i,j] - Σk attn[i,k]·d_attn[i,k])
  def softmax_rows_backward(softmax_out, d_softmax)
    t_seq = softmax_out.nrows
    n = softmax_out.ncols
    out = Mat.new(t_seq, n)
    i = 0
    while i < t_seq
      base = i * n
      s = 0.0
      j = 0
      while j < n
        s += softmax_out.flat[base + j] * d_softmax.flat[base + j]
        j += 1
      end
      j = 0
      while j < n
        out.flat[base + j] = softmax_out.flat[base + j] *
                              (d_softmax.flat[base + j] - s)
        j += 1
      end
      i += 1
    end
    out
  end

  # Split a (T × d_model) matrix back into n_heads × (T × d_head) heads.
  def hsplit_heads(d_concat)
    t_seq = d_concat.nrows
    out = [Mat.new(t_seq, @d_head)]
    h = 1
    while h < @n_heads
      out.push(Mat.new(t_seq, @d_head))
      h += 1
    end
    h = 0
    while h < @n_heads
      base = h * @d_head
      m = out[h]
      i = 0
      while i < t_seq
        j = 0
        while j < @d_head
          m.flat[i * @d_head + j] = d_concat.flat[i * @d_model + (base + j)]
          j += 1
        end
        i += 1
      end
      h += 1
    end
    out
  end

  # FFN backward. Writes w_ff1, w_ff2 grads into target_block. Returns d_h.
  def feed_forward_backward(d_ff_out, h, ff_cache, block, target_block)
    t_seq = h.nrows           # type hint: h is a Mat
    d_w_ff2  = ff_cache.hidden.t_matmul(d_ff_out)
    d_hidden = d_ff_out.matmul_t(block.w_ff2)

    # GeLU' (tanh approximation; see top-of-file GELU_* constants):
    #   gelu(x)  = 0.5 x (1 + t),                t = tanh(u),  u = C (x + K x³)
    #   gelu'(x) = 0.5 (1 + t) + 0.5 x (1 - t²) · C (1 + DK x²)
    # where C = sqrt(2/π), K = 0.044715, DK = 3 K = 0.134145.
    d_pre = Mat.new(d_hidden.nrows, d_hidden.ncols)
    n = d_hidden.nrows * d_hidden.ncols
    i = 0
    while i < n
      x     = ff_cache.pre.flat[i]
      u     = GELU_C * (x + GELU_K * x * x * x)
      t     = Math.tanh(u)
      du_dx = GELU_C * (1.0 + GELU_DK * x * x)
      deriv = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du_dx
      d_pre.flat[i] = d_hidden.flat[i] * deriv
      i += 1
    end

    d_w_ff1 = h.t_matmul(d_pre)
    d_h     = d_pre.matmul_t(block.w_ff1)

    target_block.w_ff1 = d_w_ff1
    target_block.w_ff2 = d_w_ff2
    d_h
  end

  # Self-attention backward. Writes per-head w_q/k/v + w_o grads into
  # target_block. Returns d_h_in.
  def self_attention_backward(d_proj, h_in, attn_cache, block, target_block)
    t_seq = h_in.nrows        # type hint
    inv_sqrt = 1.0 / Math.sqrt(@d_head)

    # proj = concat · w_o
    d_w_o = attn_cache.concat.t_matmul(d_proj)
    d_concat = d_proj.matmul_t(block.w_o)
    target_block.w_o = d_w_o

    d_outs = self.hsplit_heads(d_concat)

    # Build per-head Q/K/V grads (Mat per head). Seed-then-push for typing.
    d_w_q_heads = [Mat.new(@d_model, @d_head)]
    d_w_k_heads = [Mat.new(@d_model, @d_head)]
    d_w_v_heads = [Mat.new(@d_model, @d_head)]
    h = 1
    while h < @n_heads
      d_w_q_heads.push(Mat.new(@d_model, @d_head))
      d_w_k_heads.push(Mat.new(@d_model, @d_head))
      d_w_v_heads.push(Mat.new(@d_model, @d_head))
      h += 1
    end

    d_h_in = Mat.new(t_seq, @d_model)

    h = 0
    while h < @n_heads
      head = attn_cache.per_head[h]
      d_out_h = d_outs[h]

      # out = attn · V
      d_attn = d_out_h.matmul_t(head.v)
      d_v    = head.attn.t_matmul(d_out_h)

      # softmax row-wise (masked entries had attn = 0 so contribute nothing)
      d_scores = self.softmax_rows_backward(head.attn, d_attn)
      d_scores.scale!(inv_sqrt)

      # scores = Q · Kᵀ
      d_q = d_scores.matmul(head.k)
      d_k = d_scores.transpose.matmul(head.q)

      d_w_q_heads[h] = h_in.t_matmul(d_q)
      d_w_k_heads[h] = h_in.t_matmul(d_k)
      d_w_v_heads[h] = h_in.t_matmul(d_v)

      d_h_in.add!(d_q.matmul_t(block.w_q[h]))
      d_h_in.add!(d_k.matmul_t(block.w_k[h]))
      d_h_in.add!(d_v.matmul_t(block.w_v[h]))

      h += 1
    end

    target_block.w_q = d_w_q_heads
    target_block.w_k = d_w_k_heads
    target_block.w_v = d_w_v_heads
    d_h_in
  end

  # Backward through one block. Writes grads into target_block_grads.
  # Returns d_x_in (Mat).
  def transformer_block_backward(dx_out, x_in, block, layer_cache, target_block_grads)
    # x_in is only passed as an arg below — never accessed directly. Spinel's
    # body-usage parameter inference needs at least one method call to type
    # the param. `.nrows` is a Mat-only method, so this anchors x_in's type.
    _x_t = x_in.nrows

    # FFN sublayer residual: x_out = x_attn + ff_out → grad flows to both branches.
    d_h_norm2 = self.feed_forward_backward(dx_out, layer_cache.h_norm2,
                                           layer_cache.ff_cache, block, target_block_grads)
    d_x_attn_via_norm = self.rms_norm_backward(layer_cache.x_attn, block.norm2_gamma,
                                               layer_cache.rms2, d_h_norm2,
                                               target_block_grads.norm2_gamma)
    d_x_attn = dx_out.plus(d_x_attn_via_norm)

    # Attention sublayer residual: x_attn = x_in + attn_proj.
    d_h_norm1 = self.self_attention_backward(d_x_attn, layer_cache.h_norm1,
                                             layer_cache.attn_cache, block, target_block_grads)
    d_x_in_via_norm = self.rms_norm_backward(x_in, block.norm1_gamma,
                                             layer_cache.rms1, d_h_norm1,
                                             target_block_grads.norm1_gamma)

    d_x_attn.plus(d_x_in_via_norm)
  end

  # ----- Optimization -----
  #
  # Two optimizers live side-by-side. Plain SGD (apply_gradients_sgd) is
  # what the train_minimal smoke test uses — a few dozen steps to prove
  # forward/backward/update compile and converge, with no extra state.
  # Adam (apply_gradients_adam, below) is what the TinyStories run uses,
  # walking the same parameter inventory but with parallel m/v moment
  # shadows held in AdamState.

  def apply_gradients_sgd(grads, lr)
    self.sgd_step_mat(@token_embed, grads.token_embed, lr)
    self.sgd_step_mat(@pos_embed,   grads.pos_embed,   lr)
    self.sgd_step_vec(@norm_final_gamma, grads.norm_final_gamma, lr)

    li = 0
    while li < @n_layers
      self.sgd_step_block(@blocks[li], grads.blocks[li], lr)
      li += 1
    end
  end

  def sgd_step_mat(p, g, lr)
    n = p.flat.length
    i = 0
    while i < n
      p.flat[i] -= lr * g.flat[i]
      i += 1
    end
  end

  def sgd_step_vec(p, g, lr)
    n = p.length
    i = 0
    while i < n
      p[i] -= lr * g[i]
      i += 1
    end
  end

  def sgd_step_block(p_block, g_block, lr)
    self.sgd_step_vec(p_block.norm1_gamma, g_block.norm1_gamma, lr)
    self.sgd_step_vec(p_block.norm2_gamma, g_block.norm2_gamma, lr)
    self.sgd_step_mat(p_block.w_o,   g_block.w_o,   lr)
    self.sgd_step_mat(p_block.w_ff1, g_block.w_ff1, lr)
    self.sgd_step_mat(p_block.w_ff2, g_block.w_ff2, lr)

    h = 0
    while h < @n_heads
      self.sgd_step_mat(p_block.w_q[h], g_block.w_q[h], lr)
      self.sgd_step_mat(p_block.w_k[h], g_block.w_k[h], lr)
      self.sgd_step_mat(p_block.w_v[h], g_block.w_v[h], lr)
      h += 1
    end
  end

  # Adam: per-coordinate adaptive learning rate driven by running
  # estimates of the gradient mean (m) and squared mean (v).
  #
  #   m  ← β1·m + (1−β1)·g           v  ← β2·v + (1−β2)·g²
  #   m̂ = m / (1 − β1ᵗ)              v̂ = v / (1 − β2ᵗ)
  #   p -= lr · m̂ / (√v̂ + ε)
  #
  # bc1 / bc2 are kept as running products in AdamState (one multiply
  # per step) rather than recomputing β**t (one pow() per step).
  def apply_gradients_adam(grads, state, lr, beta1, beta2, eps)
    state.bc1 = state.bc1 * beta1
    state.bc2 = state.bc2 * beta2
    omc1 = 1.0 - state.bc1
    omc2 = 1.0 - state.bc2

    self.adam_step_mat(@token_embed, grads.token_embed,
                       state.m.token_embed, state.v.token_embed,
                       lr, beta1, beta2, eps, omc1, omc2)
    self.adam_step_mat(@pos_embed, grads.pos_embed,
                       state.m.pos_embed, state.v.pos_embed,
                       lr, beta1, beta2, eps, omc1, omc2)
    self.adam_step_vec(@norm_final_gamma, grads.norm_final_gamma,
                       state.m.norm_final_gamma, state.v.norm_final_gamma,
                       lr, beta1, beta2, eps, omc1, omc2)

    li = 0
    while li < @n_layers
      self.adam_step_block(@blocks[li], grads.blocks[li],
                           state.m.blocks[li], state.v.blocks[li],
                           lr, beta1, beta2, eps, omc1, omc2)
      li += 1
    end
  end

  def adam_step_mat(p, g, m, v, lr, b1, b2, eps, omc1, omc2)
    one_minus_b1 = 1.0 - b1
    one_minus_b2 = 1.0 - b2
    n = p.flat.length
    i = 0
    while i < n
      gi = g.flat[i]
      new_m = b1 * m.flat[i] + one_minus_b1 * gi
      new_v = b2 * v.flat[i] + one_minus_b2 * gi * gi
      m.flat[i] = new_m
      v.flat[i] = new_v
      m_hat = new_m / omc1
      v_hat = new_v / omc2
      p.flat[i] -= lr * m_hat / (Math.sqrt(v_hat) + eps)
      i += 1
    end
  end

  def adam_step_vec(p, g, m, v, lr, b1, b2, eps, omc1, omc2)
    one_minus_b1 = 1.0 - b1
    one_minus_b2 = 1.0 - b2
    n = p.length
    i = 0
    while i < n
      gi = g[i]
      new_m = b1 * m[i] + one_minus_b1 * gi
      new_v = b2 * v[i] + one_minus_b2 * gi * gi
      m[i] = new_m
      v[i] = new_v
      m_hat = new_m / omc1
      v_hat = new_v / omc2
      p[i] -= lr * m_hat / (Math.sqrt(v_hat) + eps)
      i += 1
    end
  end

  def adam_step_block(p_block, g_block, m_block, v_block, lr, b1, b2, eps, omc1, omc2)
    self.adam_step_vec(p_block.norm1_gamma, g_block.norm1_gamma,
                       m_block.norm1_gamma, v_block.norm1_gamma,
                       lr, b1, b2, eps, omc1, omc2)
    self.adam_step_vec(p_block.norm2_gamma, g_block.norm2_gamma,
                       m_block.norm2_gamma, v_block.norm2_gamma,
                       lr, b1, b2, eps, omc1, omc2)
    self.adam_step_mat(p_block.w_o,   g_block.w_o,
                       m_block.w_o,   v_block.w_o,
                       lr, b1, b2, eps, omc1, omc2)
    self.adam_step_mat(p_block.w_ff1, g_block.w_ff1,
                       m_block.w_ff1, v_block.w_ff1,
                       lr, b1, b2, eps, omc1, omc2)
    self.adam_step_mat(p_block.w_ff2, g_block.w_ff2,
                       m_block.w_ff2, v_block.w_ff2,
                       lr, b1, b2, eps, omc1, omc2)

    h = 0
    while h < @n_heads
      self.adam_step_mat(p_block.w_q[h], g_block.w_q[h],
                         m_block.w_q[h], v_block.w_q[h],
                         lr, b1, b2, eps, omc1, omc2)
      self.adam_step_mat(p_block.w_k[h], g_block.w_k[h],
                         m_block.w_k[h], v_block.w_k[h],
                         lr, b1, b2, eps, omc1, omc2)
      self.adam_step_mat(p_block.w_v[h], g_block.w_v[h],
                         m_block.w_v[h], v_block.w_v[h],
                         lr, b1, b2, eps, omc1, omc2)
      h += 1
    end
  end

  # No `train_step` here: Spinel compiles every class method whether or
  # not it has callers. With no callers in the current program its
  # IntArray param defaults to `mrb_int`, and the body's
  # `forward(seq_ids)` then fails to type-check. Each driver inlines the
  # forward / backward / optimizer-step sequence at its top level, which
  # is short and makes the per-step cost obvious.
  # Block i's input is the previous block's output, or the embedded input
  # for block 0.
  def x_in_for_layer(li)
    li == 0 ? @cache.x_embed : @cache.layers[li - 1].x_out
  end

  # Embedding backward: each row of dx routes to its token's embedding row
  # and to position i's positional embedding row. Repeats accumulate.
  def embed_backward(token_ids, dx, target_grads)
    t_seq = token_ids.length
    i = 0
    while i < t_seq
      # Spinel-pin: `.to_i` forces Int when this method is dead code
      # (qwen25_kv doesn't call it). Without a caller to constrain
      # `token_ids`, Spinel boxes `tok_id` as RbVal and breaks the
      # int-context use below. The explicit cast is a no-op at
      # runtime when token_ids[i] is already Int.
      tok_id = token_ids[i].to_i
      j = 0
      while j < @d_model
        pi = i * @d_model + j
        target_grads.token_embed.flat[tok_id * @d_model + j] += dx.flat[pi]
        target_grads.pos_embed.flat[pi]                      += dx.flat[pi]
        j += 1
      end
      i += 1
    end
  end

  # Full backward pass. Fills `target_grads` with this example's gradients
  # and the loss. Caller is responsible for calling forward(token_ids) first.
  def backward(input_ids, target_grads)
    n_pred = input_ids.length - 1
    if n_pred <= 0
      target_grads.loss = 0.0
      return
    end

    loss_res = self.cross_entropy_grad(@cache.logits, input_ids)
    target_grads.loss = loss_res.loss

    # Tied unembed: logits = x_final · token_embedᵀ.
    #   d_token_embed[v,d] += Σ_t dlogits[t,v] · x_final[t,d]
    #     ⇒ dlogits.t_matmul(x_final)   (vocab × d_model)
    #   d_x_final[t,d]      = Σ_v dlogits[t,v] · token_embed[v,d]
    #     ⇒ dlogits.matmul(token_embed) (T × d_model)
    # The unembed gradient is added directly into target_grads.token_embed
    # — embed_backward later adds the input-side row contributions on top.
    d_te_unembed = loss_res.dlogits.t_matmul(@cache.x_final)
    target_grads.token_embed.add!(d_te_unembed)
    dx_final = loss_res.dlogits.matmul(@token_embed)

    # Final RMSNorm. Use `self.` so Spinel's call-site parameter inference
    # picks up the typed args (only fires for explicit-receiver calls).
    dx = self.rms_norm_backward(@cache.x_block_out, @norm_final_gamma,
                                @cache.rms_final, dx_final,
                                target_grads.norm_final_gamma)

    # Each block in reverse.
    li = @n_layers - 1
    while li >= 0
      dx = self.transformer_block_backward(dx, self.x_in_for_layer(li),
                                           @blocks[li], @cache.layers[li],
                                           target_grads.blocks[li])
      li -= 1
    end

    self.embed_backward(input_ids, dx, target_grads)
  end

  # One transformer block (pre-norm). Returns BlockResult.
  # Locals are explicit so Spinel can type-trace argument types into the
  # called methods (passing `nr1.y` directly through doesn't propagate).
  def transformer_block(x, block)
    cache = LayerCache.new

    nr1 = rms_norm(x, block.norm1_gamma)
    h1  = nr1.y
    cache.h_norm1 = h1
    cache.rms1    = nr1.rms

    sa = self_attention(h1, block)
    cache.attn_cache = sa.cache
    x_attn = x.plus(sa.proj)
    cache.x_attn = x_attn

    nr2 = rms_norm(x_attn, block.norm2_gamma)
    h2  = nr2.y
    cache.h_norm2 = h2
    cache.rms2    = nr2.rms

    ff = feed_forward(h2, block)
    cache.ff_cache = ff.cache
    x_out = x_attn.plus(ff.out)
    cache.x_out = x_out

    BlockResult.new(x_out, cache)
  end

  # =====================================================================
  #  GENERATION — autoregressive sampling from a starting token-id list.
  #
  #  Tokenizing a prompt string would drag the French tokenizer (which
  #  uses unicode_normalize and complex regex) into the Spinel-compiled
  #  binary; instead we let the caller pre-tokenize and pass IDs.
  # =====================================================================
  def generate_from_ids(start_ids, max_tokens, temperature)
    # Anchor start_ids as an IntArray for Spinel param-type inference.
    n_start = start_ids.length
    # Copy start_ids into a fresh IntArray we'll grow.
    tokens = [start_ids[0]]
    i = 1
    while i < n_start
      tokens.push(start_ids[i])
      i += 1
    end

    step = 0
    while step < max_tokens
      ctx_len = tokens.length
      if ctx_len > @context_length
        ctx_len = @context_length
      end
      # Build the trailing-window context.
      ctx = [tokens[tokens.length - ctx_len]]
      j = 1
      while j < ctx_len
        ctx.push(tokens[tokens.length - ctx_len + j])
        j += 1
      end

      logits = self.forward(ctx)
      next_id = self.sample_logits_row(logits, ctx_len - 1, temperature)
      tokens.push(next_id)
      step += 1
    end
    tokens
  end

  # Sample a token ID from row `row` of `logits` (T × vocab_size flat).
  # temperature <= 0 → argmax, else softmax with temperature + cumulative
  # sample. `rand(N).to_f / N` gives a uniform [0,1) under both Spinel
  # (where bare `rand` returns C's int rand) and CRuby.
  def sample_logits_row(logits, row, temperature)
    v = logits.ncols
    base = row * v
    if temperature <= 0.0
      best_id  = 0
      best_val = logits.flat[base]
      j = 1
      while j < v
        val = logits.flat[base + j]
        if val > best_val
          best_val = val
          best_id  = j
        end
        j += 1
      end
      return best_id
    end

    inv_t = 1.0 / temperature

    # Stable-softmax: subtract the max before exp.
    mx = logits.flat[base]
    j = 1
    while j < v
      val = logits.flat[base + j]
      if val > mx
        mx = val
      end
      j += 1
    end

    sum = 0.0
    j = 0
    while j < v
      sum = sum + Math.exp((logits.flat[base + j] - mx) * inv_t)
      j += 1
    end

    r   = (rand(1_000_000).to_f / 1_000_000.0) * sum
    cum = 0.0
    j = 0
    while j < v
      cum = cum + Math.exp((logits.flat[base + j] - mx) * inv_t)
      if r < cum
        return j
      end
      j += 1
    end
    v - 1
  end
end


# Spinel anchor block. Pin `token_ids` as Array<Int> at module scope so
# TransformerLM#embed_backward's `tok = token_ids[i]` types as Int (not
# boxed RbVal). Spinel's post-6b2ae3b body-usage inference otherwise
# emits sp_poly_mul(boxed_tok, ...) and fails to compile the resulting C.
# The collapse is by local-variable NAME, so the anchor must literally
# bind `token_ids` (not `_ai_ids` or anything else).
if false
  token_ids = [0]
  _ai_dx    = Mat.new(1, 1)
  _ai_lm    = TransformerLM.new(1, 1, 1, 1, 1, 1)
  _ai_lm.embed_backward(token_ids, _ai_dx, _ai_lm)
end
