# toy#decode-logprobs (#151) — logprob + top-K helpers over a logits Mat.
#
# OpenAI's /v1/chat/completions with `logprobs=true` returns, for each
# generated token: (a) the token's own logprob, (b) the top-K alternatives
# with their logprobs. Tep's future /v1/chat handler consumes these
# helpers; toy provides the math.
#
# Numerically stable log_softmax (max-shift) and a manual partial-sort
# top-K (Spinel-friendly — avoids sort_by-with-block-on-float, which
# would otherwise be a codegen landmine).
#
# Cost: log_softmax is O(vocab). Top-K is O(vocab × k) — fine for
# k=5 and vocab=128k (~640k ops). Both done in Ruby; we don't push these
# into ggml because they're called once per decode step, not on the
# hot path.

module ToyLogProbs
  # Numerically stable log_softmax over a [1, vocab] Mat. Returns a
  # fresh Mat of the same shape.
  def self.log_softmax(logits)
    vocab = logits.ncols
    m = logits.flat[0]
    j = 1
    while j < vocab
      if logits.flat[j] > m
        m = logits.flat[j]
      end
      j = j + 1
    end
    s = 0.0
    j = 0
    while j < vocab
      s = s + Math.exp(logits.flat[j] - m)
      j = j + 1
    end
    lse = m + Math.log(s)
    out = Mat.new(1, vocab)
    j = 0
    while j < vocab
      out.flat[j] = logits.flat[j] - lse
      j = j + 1
    end
    out
  end

  # Top-K parallel arrays: token_ids (Array<Int>) and logprobs
  # (Array<Float>) of length k, sorted by logprob descending. Returns
  # [ids_array, vals_array] so callers index by position.
  #
  # Two-array shape (rather than Array<Array<mixed>>) is deliberate:
  # Spinel's poly inference can't handle heterogeneous inner arrays
  # cleanly and would emit a runtime poly_array push, which can fault
  # at startup-class-init for this module.
  #
  # Manual partial-sort scan: O(vocab × k); k=5 and vocab=128k = ~640k.
  def self.top_k(logprobs, k)
    vocab = logprobs.ncols
    ids  = Array.new(k, 0)
    vals = Array.new(k, -1.0e30)
    j = 0
    while j < vocab
      v = logprobs.flat[j]
      pos = k
      while pos > 0 && vals[pos - 1] < v
        pos = pos - 1
      end
      if pos < k
        p = k - 1
        while p > pos
          vals[p] = vals[p - 1]
          ids[p]  = ids[p - 1]
          p = p - 1
        end
        vals[pos] = v
        ids[pos]  = j
      end
      j = j + 1
    end
    [ids, vals]
  end

  # Logprob of a specific token from a logprobs Mat.
  def self.token_logprob(logprobs, token_id)
    logprobs.flat[token_id]
  end
end
