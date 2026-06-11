# lib/toy/llm/adamw.rb — Toy::AdamW: NAMED AdamW hyper-parameters.
#
# A pure-Ruby value object that NAMES the 7 slots of the anonymous
# `m_hp` Mat(1,7) that every training recipe hands to the FFI
# optimizer step. It builds the SAME Mat the runners used to build by
# hand, slot-for-slot, byte-identically. NO TinyNN / FFI calls — it
# only touches Mat (already loaded transitively by every caller via
# require_relative "../../toy"), so this file adds NO require and is
# SHARED (not mirrored) across the CPU / CUDA / Metal runners.
#
# Spinel hygiene (mirrors RopeScaling / SmolLM2Config in
# lib/toy_smollm2.rb): a PLAIN class (NO Struct.new), initialize takes
# NO args and sets every default in the BODY (NO kwargs, NO default
# args — landmine #4 widens RbVal across the compiled program), named
# attr_accessor, NO #{} interpolation.
#
# ── LOUD FINDING: slots 5/6 mean DIFFERENT things per recipe family ──
# The SAME m_hp slot index feeds DIFFERENT quantities into the
# underlying FFI optimizer-step graph, depending on which training
# graph family consumes it:
#
#   from-scratch / warm-start / vit graphs read
#       flat[5] = beta1 (0.9), flat[6] = beta2 (0.95)
#   as CONSTANT betas — literally a SECOND copy of the betas already in
#   slots 1/2. Bias correction (if any) happens INSIDE those graphs.
#
#   the lora graph reads
#       flat[5] = 1/(1 - beta1^t), flat[6] = 1/(1 - beta2^t)
#   as PER-STEP BIAS-CORRECTION DENOMINATORS (and beta2 = 0.999, NOT
#   0.95). Bias correction is computed Ruby-side and passed IN.
#
# This is a GENUINE inconsistency in how the two FFI training-graph
# families interpret slots 5/6 — NOT a bug we get to "fix" here.
# train.rb:306-307 warns that using lora-style hp in the from-scratch
# graph BREAKS the byte gate. So `beta2` is a REAL field with no single
# correct default, and `bias_correct` selects the slot-5/6 convention.
# DO NOT unify the two conventions: that would change behaviour and
# break byte-exactness. (Worth an upstream issue: the two graphs should
# agree on what slots 5/6 carry. Flagged, not silently smoothed over.)

module Toy
  class AdamW
    # EXPLICIT MODE (toy#64 item 4) — names which slot-5/6 convention
    # this hp object feeds (the loud finding above). 0 = unset: hp()
    # FAILS LOUD until the caller declares a mode, because feeding the
    # wrong convention into a training graph silently changes numerics
    # (train.rb's gate breaks with lora-style hp). Use the factories:
    #   Toy::AdamW.for_from_scratch  — slots 5/6 = constant betas
    #                                  (from-scratch / warm-start / vit
    #                                  graph family; bias_correct=false)
    #   Toy::AdamW.for_lora          — slots 5/6 = 1/(1-beta^t) per-step
    #                                  bias correction, beta2=0.999
    #                                  (lora-family graphs — ALSO the
    #                                  gpt2 + full-finetune graphs,
    #                                  which read the lora convention;
    #                                  bias_correct=true)
    # or set `mode` explicitly after .new. The mode names the HP-SLOT
    # CONVENTION (graph family), not the recipe name.
    MODE_UNSET        = 0
    MODE_FROM_SCRATCH = 1
    MODE_LORA         = 2

    attr_accessor :lr, :beta1, :beta2, :eps, :weight_decay,
                  :bias_correct, :mode

    # FROM-SCRATCH defaults — the most common / blessed case. All set
    # in the body (NO kwargs / default args — Spinel landmine #4).
    # Callers that differ (lora) set the differing fields AFTER .new via
    # the accessors (or use the .for_from_scratch / .for_lora factories,
    # which also set the REQUIRED mode). warm / vit set adamw.lr per
    # step. mode starts UNSET: hp() raises until it is declared.
    def initialize
      @lr           = 0.001
      @beta1        = 0.9
      @beta2        = 0.95
      @eps          = 1.0e-8
      @weight_decay = 0.0
      @bias_correct = false
      @mode         = MODE_UNSET
    end

    # The from-scratch/warm/vit-family factory: defaults + mode set.
    # Slots 5/6 = constant betas (bias_correct=false).
    def self.for_from_scratch
      a = Toy::AdamW.new
      a.mode = MODE_FROM_SCRATCH
      a
    end

    # The lora-family factory (ALSO the gpt2 + full-finetune graphs):
    # beta2=0.999 + per-step bias correction (slots 5/6 = 1/(1-beta^t)).
    def self.for_lora
      a = Toy::AdamW.new
      a.beta2        = 0.999
      a.bias_correct = true
      a.mode = MODE_LORA
      a
    end

    # Build the Mat(1,7) the recipe hands to step!. Byte-identical to
    # the hand-built m_hp the runners used to fill inline.
    #
    # `step` is ignored when bias_correct == false (from-scratch / warm
    # / vit pass slots 5/6 = constant betas, so hp(step) is step-agnostic
    # in that mode). When bias_correct == true (lora) it is the CALLER's
    # 1-indexed step (>= 1); the `** step.to_f` reproduces
    # train_lora.rb:174-175 / smoke_recipe_lora.rb:88-89 VERBATIM.
    #
    # FAILS LOUD (toy#64 item 4) when mode is unset, or when mode and
    # bias_correct disagree — both are silent-numerics traps (the
    # slot-5/6 dual meaning above).
    def hp(step)
      if @mode == MODE_UNSET
        raise "Toy::AdamW#hp: mode unset — declare which slot-5/6 " +
              "convention this hp feeds (use Toy::AdamW.for_from_scratch " +
              "or Toy::AdamW.for_lora, or set adamw.mode)"
      end
      if @mode == MODE_FROM_SCRATCH && @bias_correct
        raise "Toy::AdamW#hp: mode=from_scratch but bias_correct=true — " +
              "the from-scratch/warm/vit graphs read slots 5/6 as " +
              "CONSTANT BETAS; lora-style hp breaks their byte gates"
      end
      if @mode == MODE_LORA && !@bias_correct
        raise "Toy::AdamW#hp: mode=lora but bias_correct=false — the " +
              "lora-family graphs read slots 5/6 as per-step " +
              "1/(1-beta^t) bias-correction denominators"
      end
      m = Mat.new(1, 7)            # Mat.new zero-fills @flat (transformer.rb:74)
      m.flat[0] = @lr
      m.flat[1] = @beta1
      m.flat[2] = @beta2
      m.flat[3] = @eps
      m.flat[4] = @weight_decay
      # SLOT 5/6 DUAL MEANING (see the loud finding at the top of this
      # file): bias_correct selects which of the two FFI conventions we
      # feed. Do NOT unify — the from-scratch/warm/vit graphs read these
      # as constant betas; the lora graph reads them as 1/(1-beta^t).
      if @bias_correct
        m.flat[5] = 1.0 / (1.0 - (@beta1 ** step.to_f))
        m.flat[6] = 1.0 / (1.0 - (@beta2 ** step.to_f))
      else
        m.flat[5] = @beta1
        m.flat[6] = @beta2
      end
      m
    end

    # MODE-AWARE per-step hp builder (toy#73 item 5): takes the CALLER's
    # 0-INDEXED loop step (the convention every recipe loop already uses
    # for its `step == 0` is_first branch) and applies the mode's step
    # convention internally:
    #
    #   MODE_FROM_SCRATCH — step-agnostic (slots 5/6 = constant betas);
    #                       hp_for_step(k) == hp(k).
    #   MODE_LORA         — the lora-family graphs want a 1-INDEXED t in
    #                       1/(1-beta^t); hp_for_step(k) == hp(k + 1).
    #
    # This removes the lora paths' off-by-one ceremony (a 1-indexed loop
    # carried solely to feed hp(step)). Byte-identical to the hp calls it
    # replaces. The #64 mode guard STAYS: both arms delegate to hp, so
    # mode-unset / mode-vs-bias_correct mismatch still fail loud.
    def hp_for_step(step)
      if @mode == MODE_LORA
        return hp(step + 1)
      end
      hp(step)
    end
  end
end
