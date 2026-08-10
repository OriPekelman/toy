# lib/toy/io/toy_mlp_task.rb — the synthetic classification task behind
# the toy#152 (DFA-arch T0) MLP anchor. Deterministic from a seed, no
# data files, no fixture to pin, reproducible across machines.
#
# ── WHY THERE ARE TWO TASKS, AND WHY `teacher` IS THE DEFAULT ──
#
# toy#152 asks for "synthetic gaussian blobs". Isotropic gaussian blobs
# are LINEARLY SEPARABLE: a linear head on the raw input already gets
# ~100%, so the FROZEN control (hidden layers at init, head trains)
# scores as well as anything else. The success bar tao#19 made
# MANDATORY is
#
#     positive = all-DFA within the gap of all-BP
#                AND provably beating the frozen control
#
# and on blobs the second half can NEVER be met by anything — not
# because DFA failed, but because the task is trivially easy. That is
# precisely the trap the bar exists to catch ("near-BP cannot
# distinguish 'DFA learned' from 'this task is trivially easy'"), so
# shipping blobs as the anchor's only task would build a gate that is
# vacuous by construction.
#
# The default is therefore `teacher`: gaussian INPUTS whose labels come
# from a fixed random 2-layer ReLU teacher network (argmax of the
# teacher's logits). That is Refinetti et al.'s own "Align, then
# memorise" setup — the paper toy#152 cites for the output-dim lens —
# it keeps every property the ticket asked for (synthetic, seeded, no
# plumbing), and it makes the hidden layers actually load-bearing, so
# the frozen control is a real control.
#
# `blobs` is still selectable (--task blobs) for the literal reading,
# and its degeneracy is a MEASURED fact rather than an assumption: run
# the three arms on it and the frozen arm ties.
#
# NO libm IN THE LABEL PATH: the teacher uses ReLU and plain arithmetic,
# so labels are bit-identical everywhere. The input draw uses the same
# Box-Muller-from-a-31-bit-LCG stream as every other random-init in the
# tree (llama/vit engines), which the existing byte-exact fixtures
# already depend on.
#
# Spinel hygiene: plain class, no default args, no Struct, while loops,
# typed-empty array seeds, no #{} interpolation.

class MlpTask
  KIND_TEACHER = 0
  KIND_BLOBS   = 1

  attr_accessor :mt_kind, :mt_d_in, :mt_classes, :mt_d_teach,
                :mt_noise, :mt_t1, :mt_t2, :mt_mu, :mt_s

  def initialize(kind, d_in, n_classes, d_teach, task_seed, noise)
    @mt_kind    = kind
    @mt_d_in    = d_in
    @mt_classes = n_classes
    @mt_d_teach = d_teach
    @mt_noise   = noise
    @mt_s       = [0]
    @mt_s[0]    = lcg_seed_state(task_seed)
    @mt_t1 = [0.0]; @mt_t1.pop
    @mt_t2 = [0.0]; @mt_t2.pop
    @mt_mu = [0.0]; @mt_mu.pop
    if kind == KIND_BLOBS
      # One mean per class, drawn once. Spread 1.5 keeps the clusters
      # separated relative to the unit-variance sample noise.
      i = 0
      while i < n_classes * d_in
        @mt_mu.push(gauss * 1.5)
        i = i + 1
      end
    else
      # Teacher: [d_teach, d_in] then [n_classes, d_teach], both
      # fan-in-scaled so the teacher logits stay O(1) at any width.
      sc1 = 1.0 / Math.sqrt(d_in.to_f)
      i = 0
      while i < d_teach * d_in
        @mt_t1.push(gauss * sc1)
        i = i + 1
      end
      sc2 = 1.0 / Math.sqrt(d_teach.to_f)
      j = 0
      while j < n_classes * d_teach
        @mt_t2.push(gauss * sc2)
        j = j + 1
      end
    end
  end

  # Re-seed the SAMPLING stream without touching the teacher/means, so
  # the same task can be sampled from a chosen point.
  #
  # NOT the train/val split mechanism: two different seeds are two
  # OFFSETS INTO THE SAME LCG CYCLE, so a "val" stream can land inside
  # the span a "train" stream later walks — silent contamination. The
  # runner instead materialises the val set from the HEAD of one stream
  # and trains from what follows (see train_mlp.rb).
  def reset_stream!(seed)
    @mt_s[0] = lcg_seed_state(seed)
    nil
  end

  # Fill one batch. `xs` is a Mat(n, d_in) (row-major: sample-major),
  # `labels` an Int array of length n. Both are MUTATED in place — the
  # caller allocates once and reuses (the eval_ce trick: a fresh Mat
  # per step is what OOM-killed toy#149).
  def fill_batch!(n, xs, labels)
    i = 0
    while i < n
      base = i * @mt_d_in
      if @mt_kind == KIND_BLOBS
        c = next_int(@mt_classes)
        k = 0
        while k < @mt_d_in
          xs.flat[base + k] = @mt_mu[c * @mt_d_in + k] + @mt_noise * gauss
          k = k + 1
        end
        labels[i] = c
      else
        k = 0
        while k < @mt_d_in
          xs.flat[base + k] = gauss
          k = k + 1
        end
        labels[i] = teacher_label(xs, base)
      end
      i = i + 1
    end
    nil
  end

  # argmax of T2 · relu(T1 · x). Pure arithmetic — no libm, so labels
  # are bit-identical on every platform.
  def teacher_label(xs, base)
    h = [0.0]; h.pop
    j = 0
    while j < @mt_d_teach
      acc = 0.0
      k = 0
      while k < @mt_d_in
        acc = acc + @mt_t1[j * @mt_d_in + k] * xs.flat[base + k]
        k = k + 1
      end
      h.push(acc > 0.0 ? acc : 0.0)
      j = j + 1
    end
    best   = 0
    best_v = 0.0
    c = 0
    while c < @mt_classes
      acc = 0.0
      j2 = 0
      while j2 < @mt_d_teach
        acc = acc + @mt_t2[c * @mt_d_teach + j2] * h[j2]
        j2 = j2 + 1
      end
      if c == 0 || acc > best_v
        best   = c
        best_v = acc
      end
      c = c + 1
    end
    best
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
    s = @mt_s[0]
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    @mt_s[0] = s
    (s.to_f + 1.0) / 2147483648.0
  end

  def next_int(n)
    s = @mt_s[0]
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    @mt_s[0] = s
    (s >> 8) % n
  end

  def gauss
    u1 = next_u
    u2 = next_u
    if u1 < 1.0e-12; u1 = 1.0e-12; end
    Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
  end
end
