# lib/toy/llm/recipes/franken_from_scratch.rb — toy#109 P2: the
# FrankenModel from-scratch recipe. Identical to Recipes::FromScratch
# except realize! builds the training step through
# build_training_step_franken, honoring the RecipeOptions credit-
# assignment policy (per-layer 0=chain / 1=dfa) and the dfa_b_* axes.
# With an EMPTY policy the emitted graph must be IDENTICAL to
# FromScratch's — prep/smokes/smoke_franken_parity.rb pins that
# byte-for-byte, per step, in one process.
#
# Same Spinel hygiene contract as from_scratch.rb: plain class, no-arg
# ctor, ff_-prefixed members for type isolation, no tinynn require here.

require_relative "../recipe_options"
require_relative "../../train/dfa_b"

module Toy; module LLM; module Recipes
  class FrankenFromScratch
    attr_accessor :ff_cache, :ff_t_loss, :ff_t_labels, :ff_t_hp, :ff_step_index

    def initialize
      @ff_cache      = Toy::LLM::Engine::LlamaSeqEngine.new
      @ff_t_loss     = nil
      @ff_t_labels   = nil
      @ff_t_hp       = nil
      @ff_step_index = 0
    end

    def realize!(cfg, opts)
      # toy#129 item 2: a no-shadow run must tell the engine BEFORE
      # realize so alloc skips the dfa qkv param flags (the engine
      # fail-louds on a half-applied state).
      if opts.no_shadow == 1
        @ff_cache.franken_no_shadow_init(opts.credit_assignment)
      end
      @ff_cache.realize_for_random_init(cfg, opts.t_seq, opts.t_batch,
                                        opts.weight_dtype, opts.untied,
                                        opts.qkv_bias, opts.seed,
                                        opts.init_scale)
      result       = @ff_cache.build_training_step_franken(
                       opts.credit_assignment, opts.dfa_b_seed,
                       opts.dfa_b_dist, opts.dfa_b_scale, opts.dfa_b_sigma,
                       opts.dfa_mix_alpha, opts.dfa_mask_tau,
                       opts.no_shadow)
      @ff_t_loss   = result[0]
      @ff_t_labels = result[1]
      @ff_t_hp     = result[2]
      nil
    end

    # VERBATIM step order from FromScratch#step! (which itself is the
    # smoke_projection_lens op order) — the parity smoke depends on it.
    def step!(seq_ids, positions, m_labels, m_hp, is_first)
      s = @ff_cache.sess
      if is_first
        TinyNN.tnn_graph_reset(s)
      else
        TinyNN.tnn_graph_reset_grads_only(s)
      end
      TinyNN.upload_int_array(s, @ff_cache.t_seq_token_ids, seq_ids)
      TinyNN.upload_int_array(s, @ff_cache.t_seq_positions, positions)
      TinyNN.upload_row_major(s, @ff_t_labels, m_labels)
      TinyNN.upload_row_major(s, @ff_t_hp,     m_hp)
      TinyNN.tnn_compute_backward(s)
      loss_mat = TinyNN.download_row_major(s, @ff_t_loss, 1, 1)
      loss_mat.flat[0]
    end
  end
end; end; end
