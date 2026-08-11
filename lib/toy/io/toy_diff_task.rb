# lib/toy/io/toy_diff_task.rb — the low-dimensional latent distribution
# and the DDPM noise schedule behind the toy#156 (DFA-arch T2) diffusion
# lane. Deterministic from a seed, no data files.
#
# ── WHY THE LATENT IS LOW-DIMENSIONAL, AND WHY THAT IS THE TICKET ──
#
# toy#156 states a HARD CONSTRAINT: the eps-prediction target has the
# INPUT's dimensionality, so pixel diffusion (~2e5) would defeat DFA
# exactly the way vocab 50257 defeats our LM lanes. The ticket therefore
# scopes to latent / tabular / time-series. This generator IS that
# scope made concrete: `latent_dim` defaults to 16, so the DFA feedback
# matrices are [16, d_hidden] and the lane sits squarely in the regime
# toy#152 measured a positive in.
#
# ── THE DISTRIBUTION IS A GAUSSIAN MIXTURE, AND THE MODES ARE THE POINT ──
#
# A single Gaussian would be a vacuous generative target: the mean and
# covariance of the marginal are already right at initialisation, so a
# FROZEN denoiser scores well and the mandatory frozen-beat half of the
# success bar could never be met ([[control-arm-must-be-able-to-lose]],
# now on its fourth firing). A mixture of `n_modes` well-separated
# Gaussians cannot be faked that way: reproducing it requires the
# denoiser to steer samples TOWARD a mode, which is a learned,
# input-dependent map.
#
# `single` ships as the DEGENERATE control so that fact stays measured
# rather than assumed — this lane's `blobs`.
#
# ── THE SCHEDULE ──
#
# Linear betas (Ho et al. 2020) over `n_steps`, and the cumulative
# alpha-bars precomputed once. Both the forward corruption
#     x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps
# and the ancestral sampler read them, so a single table keeps training
# and sampling from drifting apart — which, if it happened, would look
# exactly like a credit-assignment result.
#
# Spinel hygiene: plain class, no-arg ctor, no default args, no Struct,
# while loops, typed-empty array seeds, no #{} interpolation.

class DiffTask
  KIND_MIXTURE = 0
  KIND_SINGLE  = 1

  attr_accessor :dt_kind, :dt_dim, :dt_modes, :dt_spread, :dt_scale,
                :dt_mu, :dt_s,
                :dt_beta, :dt_alpha, :dt_abar, :dt_steps

  def initialize(kind, latent_dim, n_modes, spread, scale, n_steps,
                 beta_lo, beta_hi, task_seed)
    @dt_kind   = kind
    @dt_dim    = latent_dim
    @dt_modes  = kind == KIND_SINGLE ? 1 : n_modes
    @dt_spread = spread
    @dt_scale  = scale
    @dt_steps  = n_steps
    @dt_s      = [0]
    @dt_s[0]   = lcg_seed_state(task_seed)

    # Mode centres, drawn once and then FIXED. `spread` is measured in
    # units of the per-mode standard deviation, so the separation is a
    # designed property rather than an accident of the draw.
    @dt_mu = [0.0]; @dt_mu.pop
    i = 0
    while i < @dt_modes * latent_dim
      @dt_mu.push(gauss * spread)
      i = i + 1
    end
    if kind == KIND_SINGLE
      j = 0
      while j < latent_dim
        @dt_mu[j] = 0.0
        j = j + 1
      end
    end

    # Linear beta schedule + cumulative alpha-bars.
    @dt_beta  = Array.new(n_steps, 0.0)
    @dt_alpha = Array.new(n_steps, 0.0)
    @dt_abar  = Array.new(n_steps, 0.0)
    run = 1.0
    k = 0
    while k < n_steps
      b = beta_lo + (beta_hi - beta_lo) * (k.to_f / (n_steps - 1).to_f)
      @dt_beta[k]  = b
      @dt_alpha[k] = 1.0 - b
      run = run * (1.0 - b)
      @dt_abar[k]  = run
      k = k + 1
    end
  end

  def reset_stream!(seed)
    @dt_s[0] = lcg_seed_state(seed)
    nil
  end

  # Draw `n` clean latents into `x0` (flat, sample-major: sample i at
  # i * latent_dim). Mutated in place.
  def sample_x0!(n, x0)
    i = 0
    while i < n
      m = @dt_modes > 1 ? next_int(@dt_modes) : 0
      base = i * @dt_dim
      mb   = m * @dt_dim
      j = 0
      while j < @dt_dim
        x0[base + j] = @dt_mu[mb + j] + gauss * @dt_scale
        j = j + 1
      end
      i = i + 1
    end
    nil
  end

  # One training batch: draw x0, draw a timestep per sample, draw eps,
  # and write the corrupted x_t. `ts` receives the drawn timesteps —
  # the caller needs them to build the time features.
  def corrupt_batch!(n, x0, xt, eps, ts)
    sample_x0!(n, x0)
    i = 0
    while i < n
      t = next_int(@dt_steps)
      ts[i] = t
      sa = Math.sqrt(@dt_abar[t])
      sb = Math.sqrt(1.0 - @dt_abar[t])
      base = i * @dt_dim
      j = 0
      while j < @dt_dim
        e = gauss
        eps[base + j] = e
        xt[base + j]  = sa * x0[base + j] + sb * e
        j = j + 1
      end
      i = i + 1
    end
    nil
  end

  # Fourier time features for timestep `t`, written at `out[off..]`.
  # `n_feat` must be even+1-shaped as produced here: one linear term
  # plus (n_feat-1)/2 sin/cos pairs. A raw scalar t is a poor
  # conditioning signal for an MLP; this is the standard fix, kept
  # deliberately small because the lane's output dim is the axis under
  # test and the conditioning is not.
  def time_features(t, n_feat, out, off)
    tau = t.to_f / (@dt_steps - 1).to_f
    out[off] = tau
    k = 1
    f = 1.0
    while k + 1 < n_feat
      out[off + k]     = Math.sin(2.0 * Math::PI * f * tau)
      out[off + k + 1] = Math.cos(2.0 * Math::PI * f * tau)
      f = f * 2.0
      k = k + 2
    end
    if k < n_feat
      out[off + k] = tau * tau
    end
    nil
  end

  # ---- deterministic stream (the tree-wide 31-bit LCG; toy#114) ----

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
    s = @dt_s[0]
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    @dt_s[0] = s
    (s.to_f + 1.0) / 2147483648.0
  end

  def next_int(n)
    s = @dt_s[0]
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    @dt_s[0] = s
    (s >> 8) % n
  end

  def gauss
    u1 = next_u
    u2 = next_u
    if u1 < 1.0e-12; u1 = 1.0e-12; end
    Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
  end
end
