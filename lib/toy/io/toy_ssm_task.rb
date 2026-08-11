# lib/toy/io/toy_ssm_task.rb — the synthetic sequence tasks behind the
# toy#155 (DFA-arch T2) selective-scan lane. Deterministic from a seed,
# no data files.
#
# ── THE DEFAULT TASK IS A DELAYED CUE, AND EVERY PART OF IT IS LOAD-BEARING ──
#
# A sequence of T vectors. Exactly ONE of them — at a random position in
# the first quarter — is a CUE: a class mean plus noise, flagged by a
# dedicated marker channel. Every other position is pure noise. The
# label is the cue's class, and the model reads out at the LAST
# timestep. So the task demands, separately:
#
#   MEMORY     — the cue is ~3T/4 steps before the readout, so whatever
#                carries it has to survive that far. This is what makes
#                the lane a credit-assignment-through-time question at
#                all, and it is why the readout is the last step and NOT
#                a mean-pool: mean-pooling lets a memoryless model
#                average the cue straight into the answer, which would
#                delete the entire point of the lane.
#   SELECTION  — T-1 of the T positions are noise. A recurrence with an
#                INPUT-INDEPENDENT decay has to integrate them all and
#                gets swamped; one that can shut its state gate except
#                at the marker does not. This is Mamba's own motivating
#                argument, and it is what makes `--selection lti` a real
#                control rather than a formality.
#
# `mean` is the DEGENERATE control (this lane's `blobs`, see
# [[control-arm-must-be-able-to-lose]]): the label comes from the MEAN
# of the whole sequence, so no memory and no selection are needed — a
# frozen random recurrence plus a trained head already integrates. It
# ships so that degeneracy is a MEASURED fact rather than an assumption.
#
# Layout: features are laid out step-major, `x[(t * batch + b) * d + j]`,
# which is exactly the [d, T*batch] column order the engine's per-step
# views slice. The last feature channel (index d-1) is the MARKER.
#
# Spinel hygiene: plain class, no-arg ctor, no default args, no Struct,
# while loops, typed-empty array seeds, no #{} interpolation.

class SsmTask
  KIND_CUE  = 0
  KIND_MEAN = 1

  attr_accessor :st_kind, :st_d, :st_t, :st_classes, :st_cue_span,
                :st_mu, :st_s, :st_noise

  # `d` counts the MARKER channel, so the class means live in d-1 dims.
  def initialize(kind, d, t_len, n_classes, cue_span, task_seed, noise)
    @st_kind    = kind
    @st_d       = d
    @st_t       = t_len
    @st_classes = n_classes
    @st_cue_span = cue_span
    @st_noise   = noise
    @st_s       = [0]
    @st_s[0]    = lcg_seed_state(task_seed)
    # One mean per class over the NON-marker channels. Scale 2.0 keeps a
    # single cue vector well clear of unit-variance noise — the task is
    # meant to be hard because of the DELAY and the DISTRACTORS, not
    # because the cue itself is hard to read.
    @st_mu = [0.0]; @st_mu.pop
    i = 0
    while i < n_classes * (d - 1)
      @st_mu.push(gauss * 2.0)
      i = i + 1
    end
  end

  def reset_stream!(seed)
    @st_s[0] = lcg_seed_state(seed)
    nil
  end

  # Fill one batch of `n` sequences. `x` is a flat Float array of length
  # t * n * d (step-major, see above) and `labels` an Int array of
  # length n; BOTH are mutated in place — the caller allocates once.
  def fill_batch!(n, x, labels)
    i = 0
    while i < t_total(n)
      x[i] = 0.0
      i = i + 1
    end
    b = 0
    while b < n
      if @st_kind == KIND_MEAN
        labels[b] = fill_mean_seq!(n, b, x)
      else
        labels[b] = fill_cue_seq!(n, b, x)
      end
      b = b + 1
    end
    nil
  end

  def t_total(n)
    @st_t * n * @st_d
  end

  # The delayed-cue sequence: noise everywhere, one marked cue early.
  def fill_cue_seq!(n, b, x)
    c = next_int(@st_classes)
    # The cue position is drawn from the first `cue_span` steps, so the
    # DELAY between cue and readout is at least t - cue_span.
    span = @st_cue_span
    if span > @st_t; span = @st_t; end
    if span < 1; span = 1; end
    pos = next_int(span)
    t = 0
    while t < @st_t
      base = (t * n + b) * @st_d
      j = 0
      while j < @st_d - 1
        x[base + j] = gauss * @st_noise
        j = j + 1
      end
      x[base + @st_d - 1] = 0.0
      t = t + 1
    end
    cbase = (pos * n + b) * @st_d
    mbase = c * (@st_d - 1)
    k = 0
    while k < @st_d - 1
      x[cbase + k] = @st_mu[mbase + k] + gauss * @st_noise
      k = k + 1
    end
    # The MARKER: a dedicated channel that is 1.0 only at the cue. A
    # selective recurrence can key its state gate off this; an
    # input-independent one cannot use it at all.
    x[cbase + @st_d - 1] = 1.0
    c
  end

  # The DEGENERATE control: the class mean is spread over EVERY step, so
  # the label falls out of an average and neither memory nor selection
  # is needed. Documented so its degeneracy is measured, not assumed.
  def fill_mean_seq!(n, b, x)
    c = next_int(@st_classes)
    mbase = c * (@st_d - 1)
    t = 0
    while t < @st_t
      base = (t * n + b) * @st_d
      j = 0
      while j < @st_d - 1
        x[base + j] = @st_mu[mbase + j] / Math.sqrt(@st_t.to_f) + gauss * @st_noise
        j = j + 1
      end
      x[base + @st_d - 1] = 0.0
      t = t + 1
    end
    c
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
    s = @st_s[0]
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    @st_s[0] = s
    (s.to_f + 1.0) / 2147483648.0
  end

  def next_int(n)
    s = @st_s[0]
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    @st_s[0] = s
    (s >> 8) % n
  end

  def gauss
    u1 = next_u
    u2 = next_u
    if u1 < 1.0e-12; u1 = 1.0e-12; end
    Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
  end
end
