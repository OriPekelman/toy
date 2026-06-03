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
end
