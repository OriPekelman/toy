# lib/toy/llm/recipes/from_scratch.rb — L4 recipe: the from-scratch
# random-init training plan. This is the MINIMAL viable surface that
# encapsulates the EXISTING random_init training loop (the inlined loop
# in prep/smokes/smoke_projection_lens.rb:97-112 and the fuller driver in
# examples/legacy/06_train_from_scratch.rb): realize a random-init forward+
# backward+AdamW graph on a Toy::LLM::Engine::LlamaSeqEngine, then drive one
# training step at a time.
#
# Greenfield NOTE: unlike the L1-L3 lifts, the L4 Recipe classes did not
# exist — only the README sketch (lib/toy/llm/recipes/README.md). This
# pass builds ONLY FromScratch. LoRA / WarmStart / Curriculum and the
# full DataSpecs/Evals/Stage/Trainer taxonomy from the sketch are
# DEFERRED — they are not built speculatively.
#
# NO separate Trainer/Stage class is introduced. Rationale: the FFI
# cache's build_training_step ALREADY bakes AdamW into the ggml backward
# graph via opt_step_adamw nodes (llama_seq_forward_ffi.rb:1480+). There
# is no Ruby optimizer to wrap, so a Toy::Trainers::AdamW would be empty
# speculation; a Stage value object is speculative for a single-stage
# recipe. The flat realize!/step! surface is the minimum that
# encapsulates the existing loop. The README sketch's each_stage/Stage/
# Trainer/DataSpec/Eval shape is deferred to the multi-stage recipes.
#
# Spinel hygiene: NEVER Struct.new (landmine #16 / matz/spinel#1043) —
# this is a hand-written plain class with an explicit no-arg ctor; NO
# default-arg ctor (default-arg poisoning, landmine #4). Member names are
# uniquely fs_-prefixed for type-isolation. No Card/step_bind/FFI :str
# args at class load (step_bind :str landmine 2026-05-28). The
# experiment-specific Mat/label/hp/positions construction stays in the
# FIXTURE (lib-vs-example scope), never here.
#
# This file does NOT `require_relative "tinynn"`: the loading module
# (lib/llama_seq_forward_ffi.rb, required by the fixture before this
# file) already loads the correct backend's TinyNN, exactly like
# llama_arch.rb:33-38. This recipe inlines TinyNN.* calls in step! so it
# is backend-coupled; the CUDA mirror (from_scratch_cuda.rb) is DEFERRED
# alongside the GPU deferral — this pass ships CPU-only.

require_relative "../recipe_options"

module Toy; module LLM; module Recipes
  # The from-scratch random-init training recipe. Encapsulates the
  # existing loop: realize! builds the random-init forward+CE+backward+
  # AdamW graph on a Toy::LLM::Engine::LlamaSeqEngine (random-init realize
  # self-enables full_finetune + train_embeddings, so no extra enable_*
  # call is needed), then step! drives one training step. The caller
  # (fixture) owns the experiment config and the per-step input Mats.
  class FromScratch
    attr_accessor :fs_cache, :fs_t_loss, :fs_t_labels, :fs_t_hp, :fs_step_index

    def initialize
      @fs_cache      = Toy::LLM::Engine::LlamaSeqEngine.new
      @fs_t_loss     = nil
      @fs_t_labels   = nil
      @fs_t_hp       = nil
      @fs_step_index = 0
    end

    # Realize the random-init graph. Delegates VERBATIM to the cache:
    # realize_for_random_init (which self-enables @ft_train_embeddings_enabled
    # + @seq_full_finetune_enabled) then build_training_step (forward + CE
    # + backward + opt_step_adamw baked into the ggml graph). Stashes the
    # returned [t_loss, t_labels, t_hp] triple. `opts` is a
    # Toy::LLM::RecipeOptions (toy#64 item 1) carrying the former 7
    # trailing positional args (t_seq, t_batch, weight_dtype, untied,
    # qkv_bias, seed, init_scale) — unpacked here in the engine's exact
    # positional order, so the realize is byte-identical. Returns nil.
    def realize!(cfg, opts)
      @fs_cache.realize_for_random_init(cfg, opts.t_seq, opts.t_batch,
                                        opts.weight_dtype, opts.untied,
                                        opts.qkv_bias, opts.seed,
                                        opts.init_scale)
      result       = @fs_cache.build_training_step
      @fs_t_loss   = result[0]
      @fs_t_labels = result[1]
      @fs_t_hp     = result[2]
      nil
    end

    # ONE training step. Op order is VERBATIM from
    # smoke_projection_lens.rb:97-112: graph_reset on the first step else
    # reset_grads_only; the four uploads in order
    # (token_ids/positions/labels/hp); compute_backward;
    # download_row_major(t_loss, 1, 1). is_first selects the reset; the
    # @fs_step_index accessor is carried for callers that want it but is
    # NOT used for the reset decision, so the caller stays in full control
    # of the step==0 branch (matches the gate's step==0 branch exactly).
    # Returns the loss Float. Per-step input Mats are built by the caller.
    def step!(seq_ids, positions, m_labels, m_hp, is_first)
      s = @fs_cache.sess
      if is_first
        TinyNN.tnn_graph_reset(s)
      else
        TinyNN.tnn_graph_reset_grads_only(s)
      end
      TinyNN.upload_int_array(s, @fs_cache.t_seq_token_ids, seq_ids)
      TinyNN.upload_int_array(s, @fs_cache.t_seq_positions, positions)
      TinyNN.upload_row_major(s, @fs_t_labels, m_labels)
      TinyNN.upload_row_major(s, @fs_t_hp,     m_hp)
      TinyNN.tnn_compute_backward(s)
      loss_mat = TinyNN.download_row_major(s, @fs_t_loss, 1, 1)
      loss_mat.flat[0]
    end
  end
end; end; end
