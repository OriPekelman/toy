# lib/toy/llm/recipes/from_scratch_cuda.rb — CUDA twin of
# lib/toy/llm/recipes/from_scratch.rb. Hand-written (NOT mechanically
# mirrored): the CPU recipe inlines backend-coupled TinyNN.* calls in
# step!, so it cannot be substituted by the generic gen_cuda_mirror.rb
# rewriter — it is kept ABSENT from MIRRORABLE and maintained by hand.
#
# This recipe drives the random-init forward+CE+backward+AdamW graph on a
# Toy::LLM::Engine::LlamaSeqEngineCuda (the CUDA cache) and computes via TinyNNCuda.
# Numerics / op-order are IDENTICAL to the CPU recipe; only the cache type
# (line: @fs_cache) and the compute module prefix (TinyNN. -> TinyNNCuda.)
# differ.
#
# This file does NOT `require_relative "tinynn_cuda"`: the loading monolith
# (lib/llama_seq_forward_ffi_cuda.rb, required by the CUDA runner before this
# file) already loads TinyNNCuda, exactly like the CPU recipe relies on
# llama_seq_forward_ffi.rb loading TinyNN. The CUDA monolith transitively
# requires transformer -> tinynn too, so the CPU TinyNN is also defined
# (the runner uses it only for the checkpoint write/fuse seam); compute here
# is CUDA-only.
#
# Spinel hygiene: NEVER Struct.new (landmine #16 / matz/spinel#1043) — plain
# hand-written class, no-arg ctor, no default-arg poisoning. fs_-prefixed
# member names for type-isolation. Single-type binary: TinyNNCuda is the only
# compute module referenced here.

require_relative "../recipe_options"

module Toy; module LLM; module Recipes
  # The from-scratch random-init training recipe, CUDA backend. Encapsulates
  # the existing loop: realize! builds the random-init forward+CE+backward+
  # AdamW graph on a Toy::LLM::Engine::LlamaSeqEngineCuda (random-init realize
  # self-enables full_finetune + train_embeddings, so no extra enable_*
  # call is needed), then step! drives one training step via TinyNNCuda.
  # The caller (runner) owns the experiment config and the per-step input Mats.
  class FromScratchCuda
    attr_accessor :fs_cache, :fs_t_loss, :fs_t_labels, :fs_t_hp, :fs_step_index

    def initialize
      @fs_cache      = Toy::LLM::Engine::LlamaSeqEngineCuda.new
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

    # ONE training step. Op order is VERBATIM from the CPU recipe
    # (from_scratch.rb:83-97): graph_reset on the first step else
    # reset_grads_only; the four uploads in order
    # (token_ids/positions/labels/hp); compute_backward;
    # download_row_major(t_loss, 1, 1). is_first selects the reset; the
    # @fs_step_index accessor is carried for callers that want it but is
    # NOT used for the reset decision, so the caller stays in full control
    # of the step==0 branch. Returns the loss Float. Per-step input Mats are
    # built by the caller.
    def step!(seq_ids, positions, m_labels, m_hp, is_first)
      s = @fs_cache.sess
      if is_first
        TinyNNCuda.tnn_graph_reset(s)
      else
        TinyNNCuda.tnn_graph_reset_grads_only(s)
      end
      TinyNNCuda.upload_int_array(s, @fs_cache.t_seq_token_ids, seq_ids)
      TinyNNCuda.upload_int_array(s, @fs_cache.t_seq_positions, positions)
      TinyNNCuda.upload_row_major(s, @fs_t_labels, m_labels)
      TinyNNCuda.upload_row_major(s, @fs_t_hp,     m_hp)
      TinyNNCuda.tnn_compute_backward(s)
      loss_mat = TinyNNCuda.download_row_major(s, @fs_t_loss, 1, 1)
      loss_mat.flat[0]
    end
  end
end; end; end
