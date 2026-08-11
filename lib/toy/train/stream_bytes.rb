# lib/toy/train/stream_bytes.rb — toy#159: the ANALYTIC activation-memory
# instrument for the recurrent lanes, reported NEXT TO the measured
# `graph: bytes=`, never instead of it.
#
# ── WHY AN ANALYTIC NUMBER AT ALL ──
#
# toy#155 and toy#157 both state their success target as "DFA matches BP
# at k-times-less activation memory for sequence length L", and both had
# to report the opposite: the per-step cut's realized graph is 31% bigger
# in nodes (toy#155) and 17% bigger in bytes (toy#157). That is not a bug
# and no amount of better measuring fixes it — in a graph autodiff every
# forward tensor is materialised whatever the credit rule, so a harness
# that BUILDS the whole unrolled graph cannot exhibit a streaming win.
#
# What this module computes is the other half of the sentence: what a
# STREAMING implementation would have to hold, derived from each cell's
# own shapes rather than by walking a graph. tao#21 lists it as the
# cheapest of the three ways to make the claim decidable, and it is the
# only one that needs no second execution engine.
#
# ── THE FINDING THE NUMBERS ENCODE ──
#
# The per-step cut's memory win is NOT a property of the credit rule
# alone. The surrogate's error e = softmax(logits) - labels is known only
# after the readout at t = T, and the update for step t needs step t's
# own activations. So there are exactly two ways to spend it:
#
#   (a) KEEP every step's activations until e arrives  -> O(T), which is
#       what this harness does, and why the measured graph is BIGGER.
#   (b) REPLAY the forward once e is known, applying each step's update
#       as you pass  -> O(1) in T, at ~2x forward compute.
#
# (b) is legal precisely BECAUSE no gradient crosses a timestep: the
# steps are independent, so they can be revisited in any order. Under
# BPTT (b) is not available at all — the backward must traverse time in
# order, so each step's activations must be live when its gradient
# arrives. **What the cut buys is the LEGALITY of the replay, not a
# smaller graph.** Reporting only (a) reads as "DFA costs more memory";
# reporting only (b) reads as a free win. Both, with the compute price
# attached, is the honest statement.
#
# BPTT's own best counter-move is quoted too, or the comparison is
# rigged: sqrt-T gradient checkpointing keeps ~2*sqrt(T) segments live
# for ~1.5x forward compute. The cut is compared against BOTH.
#
# ── WHAT IS AND IS NOT COUNTED ──
#
# Counted: the per-step tensors a backward needs, per layer. Gate/branch
# PRE-activations are excluded where the derivative is computable from
# the output (sigmoid/tanh/silu), which is what every real kernel does —
# counting them would inflate both arms and flatter the ratio.
# Not counted: weights, optimizer moments, and gradient buffers. They are
# identical across arms and independent of T, so they add a constant to
# both sides and only dilute the comparison the ticket asks for.
#
# Spinel hygiene: def self. on a plain module, no Struct, no #{},
# while loops, integer maths (bytes are exact; no float rounding drift).

module Toy; module Train; module StreamBytes
  F32 = 4

  # ---- the two lanes' per-step live sets ----

  # LSTM (lib/toy/llm/engine/lstm_engine.rb): per layer, per step, the
  # four gate post-activations i/f/o/g, the cell state c_t, tanh(c_t),
  # and h_t — SEVEN [hidden, batch] tensors.
  LSTM_PER_STEP = 7

  # Selective scan (lib/toy/llm/engine/ssm_engine.rb), --selection
  # selective: u, uc, z, dt_pre, dt, a, bb, h, c, y_pregate, g, y — TWELVE
  # [d_inner, batch] tensors — plus o and the residual sum, TWO
  # [d_model, batch] ones.
  SSM_SEL_PER_STEP_INNER = 12
  SSM_PER_STEP_MODEL     = 2
  # Under --selection lti the dt/C/gate branches are weights, not
  # per-step tensors: u, uc, bb, h, y remain. The count shrinks; the O(.)
  # structure does not, which is the point.
  SSM_LTI_PER_STEP_INNER = 5

  def self.bytes_2d(rows, cols)
    rows * cols * F32
  end

  def self.lstm_step_live(hidden, batch, n_layers)
    LSTM_PER_STEP * bytes_2d(hidden, batch) * n_layers
  end

  # The LSTM carries h and c across steps; a replay holds them and
  # nothing else. O(1) in T by construction.
  def self.lstm_carry(hidden, batch, n_layers)
    2 * bytes_2d(hidden, batch) * n_layers
  end

  def self.ssm_step_live(d_inner, d_model, batch, n_layers, selective)
    per_inner = selective ? SSM_SEL_PER_STEP_INNER : SSM_LTI_PER_STEP_INNER
    (per_inner * bytes_2d(d_inner, batch) +
     SSM_PER_STEP_MODEL * bytes_2d(d_model, batch)) * n_layers
  end

  # The SSM carries the state h AND the causal conv's K-1 previous
  # inputs. The conv window is O(K) — fixed by the kernel width, NOT by
  # T — so it belongs in the carry rather than making the arm O(T).
  def self.ssm_carry(d_inner, batch, n_layers, conv_k)
    win = conv_k - 1
    if win < 0
      win = 0
    end
    (1 + win) * bytes_2d(d_inner, batch) * n_layers
  end

  # Logits, labels and the detached error, all [classes, batch].
  # Charged to every arm equally.
  def self.head(classes, batch)
    3 * bytes_2d(classes, batch)
  end

  # ---- the three regimes, given a lane's per-step figure ----

  # (a) BPTT — every step's activations stay live for the ordered
  # backward. Linear in T. This is ALSO what the per-step cut costs if it
  # does not replay, which is exactly what this harness builds.
  def self.bptt(step_live, input_b, head_b, t_len)
    t_len * (step_live + input_b) + head_b
  end

  # (b) the per-step cut WITH replay — one step live at a time, plus the
  # carry and the head. CONSTANT in T: t_len is deliberately not a
  # parameter, so this cannot silently acquire a T term.
  def self.cut_replay(step_live, input_b, carry_b, head_b)
    step_live + input_b + carry_b + head_b
  end

  # BPTT under sqrt-T gradient checkpointing: ~2*sqrt(T) segments live
  # (the checkpoints, plus the segment being recomputed) at ~1.5x forward
  # compute. Quoted so the cut is measured against BP's best, not BP's
  # worst.
  def self.bptt_sqrt_checkpointed(step_live, input_b, carry_b, head_b, t_len)
    seg = isqrt(t_len)
    if seg < 1
      seg = 1
    end
    2 * seg * (step_live + input_b) + carry_b + head_b
  end

  # Integer sqrt, rounded UP: a partial segment still has to be held.
  def self.isqrt(n)
    r = 0
    while (r + 1) * (r + 1) <= n
      r = r + 1
    end
    if r * r < n
      r = r + 1
    end
    r
  end

  # num/den as a 2-decimal string, by INTEGER maths. A float here would
  # put a full-precision literal on byte-gated stdout, and a curve that
  # differs in the 15th digit across platforms is the cross-platform gate
  # hazard this repo already has a rule about.
  def self.ratio_str(num, den)
    if den <= 0
      return "inf"
    end
    hundredths = (num * 100) / den
    whole = hundredths / 100
    frac  = hundredths - whole * 100
    tail  = frac < 10 ? "0" + frac.to_s : frac.to_s
    whole.to_s + "." + tail
  end

  # The stdout line both lanes print, immediately after `graph:`. Stable,
  # integer-valued, and it names its own units: these are ANALYTIC bytes
  # for an implementation toy does not have, next to the MEASURED graph
  # bytes for the one it does.
  def self.line(bptt_b, sqrt_b, cut_b)
    "stream: bptt=" + bptt_b.to_s +
      " sqrt_t=" + sqrt_b.to_s +
      " cut=" + cut_b.to_s +
      " cut_vs_bptt=" + ratio_str(bptt_b, cut_b) + "x" +
      " cut_vs_sqrt_t=" + ratio_str(sqrt_b, cut_b) + "x" +
      " replay=2x_fwd"
  end
end; end; end
