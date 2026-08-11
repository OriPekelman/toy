# lib/toy/io/toy_gtx_task.rb — the synthetic knowledge graph behind the
# toy#160 (DFA-arch T4) graph-transformer lane. Deterministic from a
# seed, no data files.
#
# ── WHAT THE LANE MUST MEASURE, AND WHAT THAT DEMANDS OF THE TASK ──
#
# toy#160 asks ONE question: is ATTENTION DFA-hostile, or was the
# transformer-LM negative only ever about the ~50k output dimension?
# Answering it needs a task where
#
#   (a) the output dim is SMALL by construction (TY*TY = 16 relation
#       classes by default; GLM's head is 17), so the lane cannot
#       re-enter the regime F13/F18 already explained, and
#   (b) attention is not merely APPLIED but has to be LEARNED.
#
# (b) is the part a graph task gets wrong silently. A mask built from an
# adjacency already aggregates each node's neighbourhood for free, so a
# FROZEN random transformer under a structural mask is a perfectly good
# neighbourhood averager — and a lane whose control cannot lose measures
# nothing (see [[control-arm-must-be-able-to-lose]]). So averaging is
# made PROVABLY USELESS here:
#
#   Every entity's neighbourhood holds EXACTLY ONE attribute of EACH
#   type. Mean-pooling a neighbourhood therefore yields the same mixture
#   whatever the entity's type is — zero signal, by construction and not
#   by luck.
#
# What identifies the right neighbour is a KEY MATCH — but a match that
# is invisible to an UNTRAINED reader, which is the part that took a
# measurement to get right. The entity carries a random key in the first
# half of its feature channels; the matching attribute carries that key
# PERMUTED by a fixed, task-wide permutation. THAT attribute's value
# channels hold the entity's type; the other TY-1 hold the other types
# under fresh, unrelated keys.
#
# The permutation is load-bearing and its absence is a measured failure,
# not a theoretical one. The first build had the attribute repeat the
# key VERBATIM, and a FROZEN random transformer then scored 1.000 —
# because a random projection approximately preserves inner products
# (Johnson-Lindenstrauss), so random Q/K already score the verbatim
# match above the distractors and the retrieval needed no learning at
# all. Under a permutation, x_e . x_match is no larger than
# x_e . x_distractor, so a random query aligns with nothing; only a
# learned W_q^T W_k that implements the inverse permutation retrieves
# anything. That is what makes attention LEARNED here rather than merely
# APPLIED, and it is what lets the frozen control lose.
#
# The label of a pair (i, j) is ty(i) * TY + ty(j), so BOTH endpoints
# must be resolved through the graph before the head can be right.
#
# ── THE TOPOLOGY IS FIXED; THE CONTENT IS FRESH EVERY STEP ──
#
# What is learnable here has to be the RULE (the permutation, and how a
# retrieved value maps to a type), never a particular entity. So each
# step redraws every entity's type and key and every attribute's
# contents, over the SAME graph shape. That is the third thing this task
# needed a measurement to get right: with a fixed graph and a
# pair-level split, BP plateaued at .246 while memorising 36 training
# entities, and with a fixed graph and an ENTITY-level split it stayed
# at .246 because the shortcut was cheaper than the rule and nothing
# forced the rule. Fresh content every step leaves nothing to memorise:
# an entity index means nothing across steps, so the only thing that
# can be learned is the retrieval itself.
#
# The permutation and the per-type value directions are the RULE and
# stay fixed for the whole run. Everything else is redrawn.
#
# `--task local` is the DEGENERATE control (this lane's `blobs`/`mean`):
# there the entity's own value channels carry its type, so no retrieval
# and no attention are needed. It ships so degeneracy is a MEASURED fact
# rather than an assumption.
#
# ── LAYOUT ──
#
# Nodes are entities FIRST, then that entity's attributes, contiguously:
#   [0, E)                             entities (the only pair endpoints)
#   E + e * degree + k                 attribute k of entity e
# Features are column-major per node: x[node * d + j] — the [d, N]
# column order the engine uploads directly. Channels [0, d/2) are the
# KEY half, [d/2, d) the VALUE half.
#
# The adjacency is symmetric with self-loops, emitted as an ADDITIVE
# attention mask (0.0 where attention is allowed, MASK_NEG where not) —
# a large negative rather than -inf, because -inf in a softmax whose row
# is fully masked yields NaN, and a NaN forward is a silent wrong result
# rather than a loud one.
#
# Spinel hygiene: plain class, no default args, no Struct, while loops,
# typed-empty array seeds, no #{} interpolation.

class GtxTask
  KIND_RELATIONAL = 0
  KIND_LOCAL      = 1

  # Softmax-safe "closed" entry: exp(-30) is ~9e-14, dead in f32 without
  # being -inf, so a fully-masked row degrades to uniform, not to NaN.
  MASK_NEG = -30.0

  attr_accessor :gt_kind, :gt_d, :gt_dk, :gt_entities, :gt_types,
                :gt_degree, :gt_nodes, :gt_classes, :gt_noise,
                :gt_type_of, :gt_mu, :gt_key, :gt_perm, :gt_s

  # `degree` is forced to `n_types` by the caller: one attribute of each
  # type is what makes mean-pooling provably uninformative.
  def initialize(kind, d, n_entities, n_types, degree, task_seed, noise)
    @gt_kind     = kind
    @gt_d        = d
    @gt_dk       = d / 2
    @gt_entities = n_entities
    @gt_types    = n_types
    @gt_degree   = degree
    @gt_noise    = noise
    @gt_nodes    = n_entities + n_entities * degree
    @gt_classes  = n_types * n_types
    @gt_s        = [0]
    @gt_s[0]     = lcg_seed_state(task_seed)

    # One VALUE direction per type, over the value half only.
    @gt_mu = [0.0]; @gt_mu.pop
    i = 0
    while i < @gt_types * (@gt_d - @gt_dk)
      @gt_mu.push(0.0)
      i = i + 1
    end
    t = 0
    while t < @gt_types
      j = 0
      while j < @gt_d - @gt_dk
        @gt_mu[t * (@gt_d - @gt_dk) + j] = gauss * 2.0
        j = j + 1
      end
      t = t + 1
    end

    # The task-wide key permutation: the matching attribute carries the
    # entity's key through THIS map, so no untrained reader can spot the
    # match by inner product alone (see the header).
    @gt_perm = [0]; @gt_perm.pop
    pi = 0
    while pi < @gt_dk
      @gt_perm.push(pi)
      pi = pi + 1
    end
    pj = @gt_dk - 1
    while pj > 0
      pr = lcg_next(pj + 1)
      ptmp = @gt_perm[pj]
      @gt_perm[pj] = @gt_perm[pr]
      @gt_perm[pr] = ptmp
      pj = pj - 1
    end

    # Per-instance state, redrawn by resample! — allocated here so the
    # arrays exist before the first call.
    @gt_type_of = [0]; @gt_type_of.pop
    @gt_key     = [0.0]; @gt_key.pop
    e = 0
    while e < @gt_entities
      @gt_type_of.push(0)
      k = 0
      while k < @gt_dk
        @gt_key.push(0.0)
        k = k + 1
      end
      e = e + 1
    end
  end

  # Redraw every entity's type and key. The RULE (the permutation, the
  # per-type value directions) is untouched — that is what the model is
  # supposed to learn, and it is the only thing that persists.
  def resample!
    e = 0
    while e < @gt_entities
      @gt_type_of[e] = lcg_next(@gt_types)
      k = 0
      while k < @gt_dk
        @gt_key[e * @gt_dk + k] = gauss * 2.0
        k = k + 1
      end
      e = e + 1
    end
    nil
  end

  # Node features, [d, N] column-major.
  def fill_features!(buf)
    e = 0
    while e < @gt_entities
      # ENTITY: key half = its retrieval key; value half = noise under
      # RELATIONAL (its type is not readable from itself), or its type
      # under LOCAL, which is exactly what makes LOCAL degenerate.
      kj = 0
      while kj < @gt_dk
        buf[e * @gt_d + kj] = @gt_key[e * @gt_dk + kj] + gauss * @gt_noise
        kj = kj + 1
      end
      vj = @gt_dk
      while vj < @gt_d
        if @gt_kind == KIND_LOCAL
          buf[e * @gt_d + vj] =
            @gt_mu[@gt_type_of[e] * (@gt_d - @gt_dk) + (vj - @gt_dk)] + gauss * @gt_noise
        else
          buf[e * @gt_d + vj] = gauss
        end
        vj = vj + 1
      end
      e = e + 1
    end

    # ATTRIBUTES: one per type, in a shuffled slot order so the match is
    # never at a fixed position. The MATCHING one repeats the entity's
    # key and carries the entity's type; the others carry the remaining
    # types under fresh, non-matching keys.
    e2 = 0
    while e2 < @gt_entities
      slots = shuffled_types
      k = 0
      while k < @gt_degree
        node = @gt_entities + e2 * @gt_degree + k
        ty   = slots[k]
        is_match = ty == @gt_type_of[e2]
        kj = 0
        while kj < @gt_dk
          if is_match
            buf[node * @gt_d + kj] =
              @gt_key[e2 * @gt_dk + @gt_perm[kj]] + gauss * @gt_noise
          else
            buf[node * @gt_d + kj] = gauss * 2.0
          end
          kj = kj + 1
        end
        vj = @gt_dk
        while vj < @gt_d
          buf[node * @gt_d + vj] =
            @gt_mu[ty * (@gt_d - @gt_dk) + (vj - @gt_dk)] + gauss * @gt_noise
          vj = vj + 1
        end
        k = k + 1
      end
      e2 = e2 + 1
    end
    nil
  end

  # The ADDITIVE attention mask, [N, N]. Symmetric, with self-loops.
  # An entity sees only its own attributes; an attribute sees only its
  # own entity. Nothing else is reachable, so the neighbourhood IS the
  # evidence and there is no back channel around it.
  def fill_mask!(mask)
    total = @gt_nodes * @gt_nodes
    i = 0
    while i < total
      mask[i] = MASK_NEG
      i = i + 1
    end
    s = 0
    while s < @gt_nodes
      mask[s * @gt_nodes + s] = 0.0
      s = s + 1
    end
    e = 0
    while e < @gt_entities
      k = 0
      while k < @gt_degree
        a = @gt_entities + e * @gt_degree + k
        mask[e * @gt_nodes + a] = 0.0
        mask[a * @gt_nodes + e] = 0.0
        k = k + 1
      end
      e = e + 1
    end
    nil
  end

  # A batch of labelled PAIRS over the CURRENT instance. No entity-level
  # split is needed once the content is redrawn every step: an entity
  # index carries no information across steps, so there is nothing to
  # memorise and nothing to leak. Held-out evaluation is a set of
  # separately-drawn INSTANCES, the same discipline every other lane
  # uses for its held-out batches.
  #
  # Endpoints are drawn DISTINCT: a self-pair has label ty*TY+ty,
  # learnable from "the indices are equal" rather than from the graph.
  def fill_pairs!(n_pairs, idx_a, idx_b, labels)
    p = 0
    while p < n_pairs
      a = lcg_next(@gt_entities)
      b = lcg_next(@gt_entities)
      while b == a
        b = lcg_next(@gt_entities)
      end
      idx_a[p] = a
      idx_b[p] = b
      labels[p] = @gt_type_of[a] * @gt_types + @gt_type_of[b]
      p = p + 1
    end
    nil
  end

  def reset_stream!(seed)
    @gt_s[0] = lcg_seed_state(seed)
    nil
  end

  # --- internals ---

  # 0..TY-1 in a seeded order (Fisher-Yates).
  def shuffled_types
    a = [0]; a.pop
    i = 0
    while i < @gt_types
      a.push(i)
      i = i + 1
    end
    j = @gt_types - 1
    while j > 0
      r = lcg_next(j + 1)
      tmp  = a[j]
      a[j] = a[r]
      a[r] = tmp
      j = j - 1
    end
    a
  end

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

  def lcg_u01
    s = @gt_s[0]
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    @gt_s[0] = s
    (s.to_f + 1.0) / 2147483648.0
  end

  def lcg_next(n)
    (lcg_u01 * n.to_f).to_i % n
  end

  def gauss
    u1 = lcg_u01
    u2 = lcg_u01
    if u1 < 1.0e-12
      u1 = 1.0e-12
    end
    Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
  end
end
