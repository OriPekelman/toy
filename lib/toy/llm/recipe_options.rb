# lib/toy/llm/recipe_options.rb — Toy::LLM::RecipeOptions: NAMED
# realize-time options for the L4 recipes (toy#64 item 1).
#
# Replaces the 8-positional `realize!` wall with named accessors:
#
#   opts = Toy::LLM::RecipeOptions.new
#   opts.seed   = 42
#   opts.untied = true
#   recipe.realize!(cfg, opts)
#
# A pure-Ruby value object: NO TinyNN / FFI calls, NO requires of its
# own — it is SHARED (not mirrored) across the CPU / CUDA / Metal
# recipe variants, exactly like Toy::AdamW / Toy::Labels.
#
# Spinel hygiene (mirrors AdamW / SmolLM2Config): a PLAIN class (NO
# Struct.new — landmine #16 / matz/spinel#1043), initialize takes NO
# args and sets every default in the BODY (NO kwargs, NO default args —
# landmine #4 widens RbVal program-wide), named attr_accessor with
# PLAIN names (SmolLM2Config proves unprefixed accessor names are fine;
# every member is a monomorphic Int/Bool/Float so no type-isolation
# prefix is needed), NO #{} interpolation.
#
# Defaults are the CANONICAL smoke/gate values (the compute-surface
# fixture: realize!(cfg, t_seq, 1, 0, false, false, 0, 1.0)). t_seq has
# NO sane universal default — 0 forces the caller to set it (the FFI
# realize would build a zero-length sequence graph otherwise; callers
# always know their context length).
#
# Field map onto the engine realize calls:
#   t_seq        — sequence length T of the training graph (ctx window)
#   t_batch      — micro-batch B (B>1 allocs the block-causal mask)
#   weight_dtype — 0 = f32, 1 = f16 (GH#9 mixed precision)
#   untied       — separate lm_head (MANDATORY when cfg.donor_d_in > 0)
#   qkv_bias     — per-projection attention bias tensors (Qwen-style)
#   seed         — random-init / adapter-init seed
#   init_scale   — random-init scale multiplier
#
# ── LOUD WARNING: feed these accessors MONOMORPHIC values only ──
# Assigning a POLY-typed value (e.g. a reader from a default-arg-ctor
# class like SmolLM2Flags — landmine #4) silently poisons this object's
# ivar types and SHIFTS TRAINING NUMERICS with NO analyzer warning
# (verified on spinel a699cf9: smoke_recipe_lora's loss curve moved at
# step 2 when flags.untied flowed in unpinned). Pin at the boundary:
#   opts.untied = flags.untied ? true : false
#
# Per-recipe consumption (each recipe reads ONLY what its engine path
# takes — same pattern, different subsets):
#   FromScratch#realize!(cfg, opts)        — all 7
#   WarmStart#realize_scratch!(cfg, opts)  — all 7
#   LoRA#realize!(gguf, cfg, rank, opts)   — t_seq, untied, qkv_bias,
#                                            seed, init_scale (the mmap
#                                            path has no t_batch /
#                                            weight_dtype knob)
#   VitTiny#realize!(cfg, opts)            — seed, init_scale only

module Toy; module LLM
  class RecipeOptions
    attr_accessor :t_seq, :t_batch, :weight_dtype, :untied,
                  :qkv_bias, :seed, :init_scale,
                  # toy#109 P2 — FrankenModel credit assignment. Read ONLY by
                  # Recipes::FrankenFromScratch; the classic recipes ignore
                  # them. credit_assignment: per-layer INT modes (0=chain,
                  # 1=dfa); EMPTY = all-chain (the parity config). dfa_b_*:
                  # the fixed-feedback-matrix axes (Toy::Train::DfaB codes) —
                  # dist 0=gaussian/1=uniform/2=rademacher, scale
                  # 0=inv_sqrt_fan/1=glorot/2=fixed(sigma).
                  # modes: 0=chain 1=dfa 2=mix(dfa_mix_alpha)
                  # 3=mask-dfa(|g_bp|>tau) 4=mask-bp(|g_dfa|>tau)
                  :credit_assignment, :dfa_b_seed, :dfa_b_dist,
                  :dfa_b_scale, :dfa_b_sigma, :dfa_mix_alpha, :dfa_mask_tau,
                  # GDN reintegration: 0-based indices of layers built as
                  # Gated-DeltaNet blocks (empty = all-attention, the
                  # byte-gated default). Read by FromScratch + Franken.
                  :gdn_layers

    def initialize
      @t_seq        = 0
      @t_batch      = 1
      @weight_dtype = 0
      @untied       = false
      @qkv_bias     = false
      @seed         = 0
      @init_scale   = 1.0
      @credit_assignment = [0]; @credit_assignment.pop  # typed-empty IntArray
      @dfa_b_seed   = 0
      @dfa_b_dist   = 0
      @dfa_b_scale  = 0
      @dfa_b_sigma  = 0.0
      @dfa_mix_alpha = 0.5
      @dfa_mask_tau  = 0.0
      @gdn_layers = [0]; @gdn_layers.pop   # typed-empty IntArray
    end
  end
end; end
