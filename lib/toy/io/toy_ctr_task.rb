# lib/toy/io/toy_ctr_task.rb — the synthetic CTR task behind the
# toy#154 (DFA-arch T1) recommender lane. Deterministic from a seed, no
# data files.
#
# WHY SYNTHETIC AND NOT CRITEO. tao#18 settled it: "Criteo SUBSET or the
# synthetic generator. Do not pull full Criteo." A generator keeps the
# lane reproducible across machines with no data plumbing, and — more
# importantly — lets the task's DIFFICULTY be a designed property rather
# than a property of whichever 1% sample someone downloaded.
#
# ── THE SHAPE IS AN FM, AND THAT IS THE POINT ──
#
# Labels come from a fixed random Factorization-Machine teacher:
#
#   logit = b0
#         + sum_f  a_f · t_f[x_f]                  (per-field linear)
#         + sum_(f,g) in P  < t_f[x_f], t_g[x_g] > (PAIRWISE CROSSES)
#         + w_num · numeric
#   y ~ Bernoulli(sigmoid(logit))
#
# The pairwise dot products are the feature CROSSES that CTR models
# exist to capture and that no linear model over the raw features can
# represent. That matters for this lane's control arm: with a purely
# linear teacher, "trainable embeddings + a FROZEN random tower + a
# trained head" would already solve the task, the frozen control would
# tie, and the mandatory frozen-beat half of the success bar could
# never be met by anything — the failure mode toy#152 hit head-on with
# gaussian blobs (see [[control-arm-must-be-able-to-lose]]). The
# crosses are what make the tower load-bearing.
#
# Labels are SAMPLED (Bernoulli), not thresholded, because that is what
# a CTR label is: the Bayes-optimal AUC is < 1 and the metric has a
# real ceiling. A deterministic label would make AUC=1 reachable and
# flatter every arm equally.
#
# NO libm IN THE LABEL PATH beyond the sigmoid: the teacher is dot
# products and one exp, so labels are stable across platforms. The
# input draw uses the same 31-bit LCG as every other random-init in the
# tree.
#
# Spinel hygiene: plain class, no default args, no Struct, while loops,
# typed-empty array seeds, no #{} interpolation.

class CtrTask
  attr_accessor :ct_fields, :ct_card, :ct_numeric, :ct_k,
                :ct_lin, :ct_emb, :ct_pairs_a, :ct_pairs_b,
                :ct_wnum, :ct_b0, :ct_s, :ct_lin_scale

  # `lin_scale` weights the ADDITIVE (per-field linear + numeric) part
  # of the teacher against the pairwise crosses. It is the dial that
  # decides whether this task can tell the arms apart at all, and it
  # was MEASURED, not guessed:
  #
  #   at lin_scale 1.0 / 12 pairs the additive part dominates the logit
  #   variance, embeddings + head alone capture nearly all of it, and
  #   the three arms land within 0.003 AUC with the FROZEN control
  #   ahead (.7951 frozen / .7931 dfa / .7918 chain) — an unfalsifiable
  #   bar, the toy#152-with-blobs failure mode exactly.
  #
  # Turning the additive part down makes the CROSSES carry the signal,
  # and crosses are what a tower is for.
  def initialize(n_fields, cardinality, n_numeric, k, n_pairs, task_seed,
                 base_rate, lin_scale)
    @ct_fields  = n_fields
    @ct_card    = cardinality
    @ct_numeric = n_numeric
    @ct_k       = k
    @ct_lin_scale = lin_scale
    @ct_s       = [0]
    @ct_s[0]    = lcg_seed_state(task_seed)

    # Per-field latent table: t_f[v] in R^k, laid out flat as
    # [(f * cardinality + v) * k + j].
    @ct_emb = [0.0]; @ct_emb.pop
    sc = 1.0 / Math.sqrt(k.to_f)
    i = 0
    while i < n_fields * cardinality * k
      @ct_emb.push(gauss * sc)
      i = i + 1
    end
    # Per-field linear weights over the same latents: a_f in R^k.
    @ct_lin = [0.0]; @ct_lin.pop
    li = 0
    while li < n_fields * k
      @ct_lin.push(gauss * sc)
      li = li + 1
    end
    # The interacting field PAIRS. Drawn once; a pair may repeat, which
    # only strengthens that cross.
    @ct_pairs_a = [0]; @ct_pairs_a.pop
    @ct_pairs_b = [0]; @ct_pairs_b.pop
    pi = 0
    while pi < n_pairs
      fa = next_int(n_fields)
      fb = next_int(n_fields)
      if fa == fb
        fb = (fb + 1) % n_fields
      end
      @ct_pairs_a.push(fa)
      @ct_pairs_b.push(fb)
      pi = pi + 1
    end
    @ct_wnum = [0.0]; @ct_wnum.pop
    ni = 0
    while ni < n_numeric
      @ct_wnum.push(gauss * 0.5)
      ni = ni + 1
    end
    # Intercept: solve nothing, just shift so the mean rate lands near
    # `base_rate`. logit(base_rate) is the right offset for a
    # zero-mean teacher, which the draws above are by construction.
    r = base_rate
    if r < 0.01; r = 0.01; end
    if r > 0.99; r = 0.99; end
    @ct_b0 = Math.log(r / (1.0 - r))
  end

  def reset_stream!(seed)
    @ct_s[0] = lcg_seed_state(seed)
    nil
  end

  # Fill one batch. `idx` is a flat Int array of length n_fields*n
  # (field-major: idx[f * n + i]), `nums` a Mat(n, n_numeric)
  # (sample-major, row-major), `labels` an Int array of length n. All
  # THREE are mutated in place — the caller allocates once and reuses.
  def fill_batch!(n, idx, nums, labels)
    i = 0
    while i < n
      f = 0
      while f < @ct_fields
        idx[f * n + i] = next_int(@ct_card)
        f = f + 1
      end
      nb = i * @ct_numeric
      j = 0
      while j < @ct_numeric
        nums.flat[nb + j] = gauss
        j = j + 1
      end
      labels[i] = sample_label(n, i, idx, nums)
      i = i + 1
    end
    nil
  end

  # The FM teacher + a Bernoulli draw.
  def sample_label(n, i, idx, nums)
    z = @ct_b0
    # per-field linear part
    f = 0
    while f < @ct_fields
      base = (f * @ct_card + idx[f * n + i]) * @ct_k
      lbase = f * @ct_k
      j = 0
      while j < @ct_k
        z = z + @ct_lin_scale * @ct_lin[lbase + j] * @ct_emb[base + j]
        j = j + 1
      end
      f = f + 1
    end
    # pairwise crosses — the part a linear model cannot represent
    p = 0
    while p < @ct_pairs_a.length
      fa = @ct_pairs_a[p]
      fb = @ct_pairs_b[p]
      ba = (fa * @ct_card + idx[fa * n + i]) * @ct_k
      bb = (fb * @ct_card + idx[fb * n + i]) * @ct_k
      j2 = 0
      while j2 < @ct_k
        z = z + @ct_emb[ba + j2] * @ct_emb[bb + j2]
        j2 = j2 + 1
      end
      p = p + 1
    end
    # numeric part
    nb = i * @ct_numeric
    k2 = 0
    while k2 < @ct_numeric
      z = z + @ct_lin_scale * @ct_wnum[k2] * nums.flat[nb + k2]
      k2 = k2 + 1
    end
    pr = 1.0 / (1.0 + Math.exp(0.0 - z))
    next_u < pr ? 1 : 0
  end

  # --- deterministic stream (the tree-wide 31-bit LCG; toy#114) ---

  def lcg_seed_state(seed)
    s = ((seed + 104729) * 2654435761) % 2147483647
    if s <= 0
      s = seed + 104729
    end
    w = 0
    while w < 8
      s = (s * 1103515245 + 12345) & 0x7FFFFFFF
      w = w + 1
    end
    s
  end

  def next_u
    s = @ct_s[0]
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    @ct_s[0] = s
    (s.to_f + 1.0) / 2147483648.0
  end

  def next_int(n)
    s = @ct_s[0]
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    @ct_s[0] = s
    (s >> 8) % n
  end

  def gauss
    u1 = next_u
    u2 = next_u
    if u1 < 1.0e-12; u1 = 1.0e-12; end
    Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
  end
end
