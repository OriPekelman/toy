# E2.4 / GH#14 — learning-rate schedules.
#
# Cosine decay with optional linear warmup. AdamW's hp[0] (lr) is
# the only knob the training graph needs; callers compute the schedule
# value at the current step and upload it before each compute_backward.
#
# Reference: Loshchilov & Hutter 2017 (SGDR / cosine); used as-is in
# Touvron et al. (Llama 2, 3), Qwen, SmolLM2 training recipes.

module ToyLR
  # Cosine schedule from lr_max → lr_min over n_steps, with an
  # optional linear warmup over the first `warmup_steps`. The warmup
  # ramps from lr_max/warmup_steps up to lr_max linearly, then
  # cosine-decays to lr_min by step `n_steps`.
  #
  # step is 0-indexed.
  def self.cosine(step, n_steps, lr_max, lr_min, warmup_steps)
    if step < warmup_steps
      # Linear warmup: 0 → lr_max over warmup_steps.
      return lr_max * (step + 1).to_f / warmup_steps.to_f
    end
    # Cosine decay over (step - warmup_steps) / (n_steps - warmup_steps).
    denom = (n_steps - warmup_steps).to_f
    if denom <= 0.0
      return lr_min
    end
    progress = (step - warmup_steps).to_f / denom
    if progress > 1.0
      progress = 1.0
    end
    lr_min + 0.5 * (lr_max - lr_min) * (1.0 + Math.cos(Math::PI * progress))
  end

  # Constant — for sanity/baseline runs.
  def self.constant(step, lr)
    step
    lr
  end

  # toy#158 (F15) — RAdam's RECTIFICATION TERM (Liu et al. 2019,
  # arXiv:1908.03265), as an LR multiplier. LightOn's working macro-DFA
  # recipe is RAdam-class at lr 5e-5, and our transformer negatives
  # were all plain AdamW at 1e-3, so the optimizer is one of the three
  # things F15 has to control for.
  #
  #   rho_inf = 2/(1-b2) - 1
  #   rho_t   = rho_inf - 2t·b2^t/(1-b2^t)
  #   r_t     = sqrt( ((rho_t-4)(rho_t-2)rho_inf) /
  #                   ((rho_inf-4)(rho_inf-2)rho_t) )     when rho_t > 4
  #
  # WHY THIS IS ONLY AN LR MULTIPLIER HERE, and where that is not
  # RAdam: r_t is a per-STEP SCALAR, so in the rectified regime
  # (rho_t > 4) "AdamW with lr·r_t" IS RAdam exactly — the adaptive
  # step, bias correction and moments are identical, and our
  # opt_step_adamw already does its own bias correction. In the
  # UN-rectified early regime Liu et al. take a NON-ADAPTIVE momentum
  # step (lr·m̂); we return 0.0 instead, i.e. we accumulate the moments
  # and take no step. That is a real deviation, and it is bounded: at
  # b2=0.999 rho_t crosses 4 at t≈5, so it costs the first ~4 steps of
  # a run measured in thousands. Expressing the momentum step properly
  # would mean a second optimizer step in the graph with per-step hp
  # switching (the franken-moe muon idiom) — worth it only if a result
  # ever turns on those 4 steps. Stated here rather than discovered
  # later from a curve that does not match a reference implementation.
  #
  # `step` is 1-INDEXED (t in the paper).
  def self.radam_rect(step, beta2)
    rho_inf = 2.0 / (1.0 - beta2) - 1.0
    b2t     = beta2 ** step.to_f
    denom   = 1.0 - b2t
    if denom <= 0.0
      return 0.0
    end
    rho_t = rho_inf - 2.0 * step.to_f * b2t / denom
    if rho_t <= 4.0
      return 0.0
    end
    num = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf
    den = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t
    if den <= 0.0
      return 0.0
    end
    Math.sqrt(num / den)
  end
end
