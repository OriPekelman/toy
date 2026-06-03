# Sampler — composable transforms applied to logits before pick.
#
# All transforms take a `Mat` (1 × vocab) of logits + a context object
# and return a `Mat` (1 × vocab) — possibly mutating in place. Final
# pick is `argmax_or_multinomial` (deterministic at T=0; sampled
# otherwise).
#
# Standard pipeline:
#   logits = repetition_penalty(logits, ctx, cfg.rep_penalty)
#   logits = temperature(logits, cfg.temperature)
#   logits = top_k(logits, cfg.top_k)
#   logits = top_p(logits, cfg.top_p)
#   pick   = argmax_or_multinomial(logits, cfg)
#
# Context (`SamplerContext`) carries generated IDs (for rep penalty)
# + RNG seed. Config (`SamplerConfig`) carries per-call hyperparams.

require_relative "../../transformer"   # for Mat

class SamplerConfig
  attr_accessor :temperature, :top_k, :top_p, :rep_penalty, :seed, :max_tokens

  def initialize
    @temperature = 0.0       # 0 = greedy argmax
    @top_k       = 0         # 0 = disabled
    @top_p       = 1.0       # 1.0 = disabled
    @rep_penalty = 1.0       # 1.0 = disabled
    @seed        = 42
    @max_tokens  = 64
  end
end

class SamplerContext
  attr_accessor :generated_ids, :prompt_ids, :rng_state

  def initialize(prompt_ids, seed)
    @generated_ids = [0]
    @generated_ids.pop
    @prompt_ids = prompt_ids
    @rng_state  = seed
  end

  # xorshift32 — Spinel-safe deterministic RNG. Don't use Kernel#rand;
  # we want reproducibility across Spinel builds.
  def next_u32
    x = @rng_state
    x = x ^ ((x << 13) & 0xFFFFFFFF)
    x = x ^ (x >> 17)
    x = x ^ ((x << 5) & 0xFFFFFFFF)
    @rng_state = x & 0xFFFFFFFF
    @rng_state
  end

  # Float in [0, 1).
  def next_unit
    (next_u32.to_f) / 4294967296.0
  end
end

module Sampler
  # In-place divide by temperature. T=0 means "do nothing here, let
  # argmax_or_multinomial fall through to argmax."
  def self.temperature(logits, t)
    if t <= 0.0 || t == 1.0
      return logits
    end
    inv = 1.0 / t
    n = logits.ncols
    j = 0
    while j < n
      logits.flat[j] = logits.flat[j] * inv
      j = j + 1
    end
    logits
  end

  # Subtract `rep_penalty` from logits of any token already in the
  # generated context. Default 1.0 = disabled. The HF convention
  # DIVIDES positive logits and MULTIPLIES negative ones; we use the
  # simpler subtract-on-positive variant to keep it Spinel-friendly.
  # For most fine-tunes a value of 1.05–1.2 is reasonable.
  def self.repetition_penalty(logits, ctx, p)
    if p <= 1.0
      return logits
    end
    seen = ctx.generated_ids
    i = 0
    while i < seen.length
      tid = seen[i]
      if tid >= 0 && tid < logits.ncols
        v = logits.flat[tid]
        if v > 0.0
          logits.flat[tid] = v / p
        else
          logits.flat[tid] = v * p
        end
      end
      i = i + 1
    end
    logits
  end

  # Keep top-k logits; mask the rest with -INFINITY (= -1e30 here so
  # softmax never sees -Inf). k=0 disables.
  def self.top_k(logits, k)
    if k <= 0 || k >= logits.ncols
      return logits
    end
    n = logits.ncols
    # Find the k-th largest by k passes of argmax. O(k*n); fine for
    # k ≤ ~100 at vocab=150K. For bigger k a real partial-sort would
    # be better; current sizes don't need it.
    kept = [0]
    kept.pop
    snapshot = [0.0]
    snapshot.pop
    j = 0
    while j < n
      snapshot.push(logits.flat[j])
      j = j + 1
    end
    pass = 0
    while pass < k
      best_i = -1
      best_v = NEG_INF_SCORE
      j = 0
      while j < n
        v = snapshot[j]
        if v > best_v
          best_v = v
          best_i = j
        end
        j = j + 1
      end
      if best_i < 0
        # already all masked
        return logits
      end
      kept.push(best_i)
      snapshot[best_i] = NEG_INF_SCORE
      pass = pass + 1
    end
    # Build keep-set as a flag array
    keep = [false]
    keep.pop
    j = 0
    while j < n
      keep.push(false)
      j = j + 1
    end
    j = 0
    while j < kept.length
      keep[kept[j]] = true
      j = j + 1
    end
    j = 0
    while j < n
      if !keep[j]
        logits.flat[j] = NEG_INF_SCORE
      end
      j = j + 1
    end
    logits
  end

  # Top-p / nucleus: softmax → cumulative sort → keep smallest set
  # whose probability mass ≥ p. p>=1 disables.
  def self.top_p(logits, p)
    if p >= 1.0 || p <= 0.0
      return logits
    end
    n = logits.ncols
    # softmax in place into a copy
    probs = [0.0]
    probs.pop
    max_v = NEG_INF_SCORE
    j = 0
    while j < n
      v = logits.flat[j]
      if v > max_v
        max_v = v
      end
      j = j + 1
    end
    sum = 0.0
    j = 0
    while j < n
      e = Math.exp(logits.flat[j] - max_v)
      probs.push(e)
      sum = sum + e
      j = j + 1
    end
    inv_sum = 1.0 / sum
    j = 0
    while j < n
      probs[j] = probs[j] * inv_sum
      j = j + 1
    end
    # Sort indices by descending prob (selection-sort with mark; O(n^2)).
    # Acceptable because vocab ≤ 200K and we typically prune via top_k
    # first; a partial-sort would help if top_p were used solo at full
    # vocab.
    order = [0]
    order.pop
    taken = [false]
    taken.pop
    j = 0
    while j < n
      taken.push(false)
      j = j + 1
    end
    cum = 0.0
    pass = 0
    while pass < n
      best_i = -1
      best_v = -1.0
      j = 0
      while j < n
        if !taken[j] && probs[j] > best_v
          best_v = probs[j]
          best_i = j
        end
        j = j + 1
      end
      if best_i < 0
        break
      end
      taken[best_i] = true
      cum = cum + best_v
      order.push(best_i)
      if cum >= p
        break
      end
      pass = pass + 1
    end
    # Mask anything NOT in `order`.
    keep = [false]
    keep.pop
    j = 0
    while j < n
      keep.push(false)
      j = j + 1
    end
    j = 0
    while j < order.length
      keep[order[j]] = true
      j = j + 1
    end
    j = 0
    while j < n
      if !keep[j]
        logits.flat[j] = NEG_INF_SCORE
      end
      j = j + 1
    end
    logits
  end

  # Final pick. If cfg.temperature <= 0, return argmax (greedy).
  # Otherwise softmax + multinomial draw using ctx's RNG.
  def self.pick(logits, cfg, ctx)
    if cfg.temperature <= 0.0
      return Sampler.argmax(logits)
    end
    Sampler.multinomial(logits, ctx)
  end

  def self.argmax(logits)
    n = logits.ncols
    best_i = 0
    best_v = logits.flat[0]
    j = 1
    while j < n
      v = logits.flat[j]
      if v > best_v
        best_v = v
        best_i = j
      end
      j = j + 1
    end
    best_i
  end

  def self.multinomial(logits, ctx)
    n = logits.ncols
    # softmax
    max_v = NEG_INF_SCORE
    j = 0
    while j < n
      v = logits.flat[j]
      if v > max_v
        max_v = v
      end
      j = j + 1
    end
    sum = 0.0
    j = 0
    while j < n
      sum = sum + Math.exp(logits.flat[j] - max_v)
      j = j + 1
    end
    target = ctx.next_unit * sum
    cum = 0.0
    j = 0
    while j < n
      cum = cum + Math.exp(logits.flat[j] - max_v)
      if cum >= target
        return j
      end
      j = j + 1
    end
    n - 1
  end
end
