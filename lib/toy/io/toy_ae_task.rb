# lib/toy/io/toy_ae_task.rb — the byte-level TEXT stream behind the
# toy#165 (capstone P1a) latent-autoencoder lane.
#
# ── WHY THIS LANE READS REAL TEXT AND NOT A SEEDED SYNTHETIC ──
#
# Every other cross-architecture lane (mlp/gnn/ssm/lstm/gtx/diff) makes
# its own task from a seed, and that is right when the task IS the
# instrument. Here it would be wrong, and quietly so.
#
# P1a asks where a d-dim per-token latent stops being decodable UNDER
# NOISE. That margin is set by how tightly the per-position head has to
# pack codepoints into d dimensions, so it is governed by the number of
# distinct symbols that actually occur and by how the mass is spread over
# them. A synthetic byte stream exercises a small, low-entropy slice of
# the 256 values, which leaves those codepoints far apart in latent space
# and INFLATES the margin at exactly the small d where the verdict is
# decided. A `go` obtained that way would not transfer to text.
#
# So the corpus is fetched (prep/fetch_text.rb), pinned by SHA, and named
# in provenance. toy#153 set the precedent: the synthetic graph could not
# carry the mandatory bar, so the bar was stated on real Cora.
#
# ── THE ALPHABET IS THE SECOND AXIS, NOT A DETAIL (tao#22) ──
#
# To first order the nearest-neighbour spacing in a d-dim packing of N
# codepoints goes as N^(1/d). At N=27 the packing problem at d=4 is about
# as hard as N=256 at d=8 — i.e. the alphabet alone can manufacture a
# `go`. So the lane ships three pinned corpora at measured N = 27 / 65 /
# 201 and reports the alphabet size it OBSERVED, both over the pack and
# over the held-out windows it actually scored. The margin curve is then
# read against the number of regions that really had to be separated.
#
# The head stays 256-wide on every corpus. Remapping the observed bytes
# onto a compact 0..N-1 range would shrink the head with N and confound
# the alphabet axis with head capacity; leaving it fixed means only the
# DATA changes across the axis. Classes that never occur simply never get
# mass, which is a fact the reported alphabet size already states.
#
# ── PACK FORMAT (written by prep/fetch_text.rb) ──
#   <prefix>.meta.i32   [n_tokens, alphabet_size]
#   <prefix>.tok.i32    n_tokens byte ids, 0..255, in corpus order
#
# Windows are drawn uniformly over the WHOLE pack rather than from a
# prefix. That matters for the multilingual corpus, whose members are
# concatenated in sorted name order: any prefix of it is a handful of
# languages, so a prefix-only reader would silently measure a much
# smaller alphabet than the pack advertises.
#
# Spinel hygiene: plain class, no default args, no Struct, while loops,
# typed-empty array seeds, no #{} interpolation.

class AeTask
  # The head is byte-wide on every corpus; see the header on why this is
  # not derived from the observed alphabet.
  VOCAB = 256

  attr_accessor :at_context, :at_n_tokens, :at_alphabet, :at_tokens,
                :at_counts, :at_floor_p, :at_floor_id, :at_entropy, :at_s

  def initialize(context, task_seed)
    @at_context  = context
    @at_n_tokens = 0
    @at_alphabet = 0
    @at_tokens   = [0]; @at_tokens.pop
    @at_counts   = [0]; @at_counts.pop
    @at_floor_p  = 0.0
    @at_floor_id = 0
    @at_entropy  = 0.0
    @at_s        = [0]
    @at_s[0]     = lcg_seed_state(task_seed)
  end

  # Returns 0 on success, non-zero after printing why. Never returns a
  # partially-loaded pack: a short read here would show up downstream as
  # a corpus of zero bytes, which trains perfectly and means nothing.
  def load_pack!(prefix)
    meta = Array.new(2, 0)
    got = TinyNN.tnn_read_i32_file(prefix + ".meta.i32", 0, 2, meta)
    if got != 2
      puts "ae: could not read " + prefix + ".meta.i32 (rc=" + got.to_s + ")" +
           " — run: ruby prep/fetch_text.rb --all"
      return 1
    end
    n     = meta[0]
    alpha = meta[1]
    if n < 2 || alpha < 2
      puts "ae: pack meta is degenerate: n_tokens=" + n.to_s +
           " alphabet=" + alpha.to_s
      return 1
    end
    if n <= @at_context
      puts "ae: corpus is shorter than one window: n_tokens=" + n.to_s +
           " context=" + @at_context.to_s
      return 1
    end

    @at_tokens = Array.new(n, 0)
    gt = TinyNN.tnn_read_i32_file(prefix + ".tok.i32", 0, n, @at_tokens)
    if gt != n
      puts "ae: " + prefix + ".tok.i32 short: got " + gt.to_s +
           " want " + n.to_s
      return 1
    end
    @at_n_tokens = n

    @at_counts = Array.new(VOCAB, 0)
    i = 0
    while i < n
      tk = @at_tokens[i]
      if tk < 0 || tk >= VOCAB
        puts "ae: token " + tk.to_s + " at index " + i.to_s +
             " is outside 0.." + (VOCAB - 1).to_s +
             " — the pack is not byte ids"
        return 1
      end
      @at_counts[tk] = @at_counts[tk] + 1
      i = i + 1
    end

    distinct = 0
    best_id  = 0
    best_c   = -1
    ent      = 0.0
    c = 0
    while c < VOCAB
      k = @at_counts[c]
      if k > 0
        distinct = distinct + 1
        p = k.to_f / n.to_f
        ent = ent - p * Math.log(p) / Math.log(2.0)
      end
      if k > best_c
        best_c  = k
        best_id = c
      end
      c = c + 1
    end
    # The pack says what it contains; we counted what it contains. If
    # those disagree the pack was written by something other than the
    # current prep/fetch_text.rb, and every alphabet-axis reading off it
    # would be mislabelled.
    if distinct != alpha
      puts "ae: pack meta claims alphabet=" + alpha.to_s +
           " but the tokens contain " + distinct.to_s +
           " distinct values — regenerate with prep/fetch_text.rb"
      return 1
    end
    @at_alphabet = distinct
    @at_floor_id = best_id
    @at_floor_p  = best_c.to_f / n.to_f
    @at_entropy  = ent
    0
  end

  # A uniformly-drawn window start inside [lo, hi] INCLUSIVE.
  #
  # The caller passes disjoint train and val spans. That is deliberate
  # and it is stronger than the seeded-stream separation the synthetic
  # lanes use: those redraw their content every step, so there is nothing
  # to memorise, but a fixed corpus is memorisable and two windows drawn
  # from one span can OVERLAP. An overlapping val window would let a
  # memorised encoder inflate clean reconstruction, which is the one
  # number the noise-margin curve is normalised against.
  def next_start_in(lo, hi)
    span = hi - lo
    lo + lcg_next(span + 1)
  end

  def fill_window!(buf, start)
    i = 0
    while i < @at_context
      buf[i] = @at_tokens[start + i]
      i = i + 1
    end
    nil
  end

  def reset_stream!(seed)
    @at_s[0] = lcg_seed_state(seed)
    nil
  end

  # --- internals (the same LCG every lane in this program uses) ---

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
    s = @at_s[0]
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    @at_s[0] = s
    (s.to_f + 1.0) / 2147483648.0
  end

  def lcg_next(n)
    (lcg_u01 * n.to_f).to_i % n
  end
end
