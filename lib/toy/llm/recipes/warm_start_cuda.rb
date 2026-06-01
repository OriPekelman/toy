# lib/toy/llm/recipes/warm_start_cuda.rb — CUDA twin of
# lib/toy/llm/recipes/warm_start.rb. Hand-written (NOT mechanically
# mirrored): the CPU recipe inlines backend-coupled TinyNN.* calls in
# step!/realize_warm!, so it cannot be substituted by the generic
# gen_cuda_mirror.rb rewriter — it is kept ABSENT from MIRRORABLE and
# maintained by hand. `make verify-mirrors` stays green.
#
# This recipe drives the random-init-plus-(optional)donor warm-start
# forward+CE+backward+AdamW graph on a LlamaSeqForwardFFICacheCuda (the
# CUDA cache) and computes via TinyNNCuda. Numerics / op-order are
# IDENTICAL to the CPU recipe (warm_start.rb); only the cache type (the
# ctor line: @ws_cache) and the compute module prefix (TinyNN. ->
# TinyNNCuda.) differ. The realize/build SPLIT (realize_scratch! /
# realize_warm! / build!) is preserved verbatim — see warm_start.rb for
# the rationale (donor upload must land BETWEEN realize and build).
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
# hand-written class, no-arg ctor, no default-arg poisoning. ws_-prefixed
# member names for type-isolation. Single-type binary: TinyNNCuda is the only
# compute module referenced here.

module Toy; module LLM; module Recipes
  # The warm-start training recipe, CUDA backend. realize_scratch! builds
  # the random-init forward+CE+backward+AdamW graph on a
  # LlamaSeqForwardFFICacheCuda (random-init realize self-enables
  # full_finetune + train_embeddings) and OPENS the warm window;
  # realize_warm! (optional) uploads an already-read donor embedding into
  # the realize'd embed table BEFORE the graph is baked; build! CLOSES the
  # window by baking forward+CE+backward+opt_step_adamw into the ggml graph.
  # step! then drives one training step via TinyNNCuda. The caller (runner)
  # owns the experiment config, the donor/PCA GGUF read, the corpus stream,
  # the LR schedule, and the per-step input Mats.
  class WarmStartCuda
    attr_accessor :ws_cache, :ws_t_loss, :ws_t_labels, :ws_t_hp, :ws_step_index

    def initialize
      @ws_cache      = LlamaSeqForwardFFICacheCuda.new
      @ws_t_loss     = nil
      @ws_t_labels   = nil
      @ws_t_hp       = nil
      @ws_step_index = 0
    end

    # Realize the random-init graph and OPEN the warm window. Delegates
    # VERBATIM to the cache: realize_for_random_init. Positional args match
    # realize_for_random_init exactly (cfg, t_seq, t_batch, weight_dtype,
    # untied, qkv_bias, seed, init_scale). Does NOT bake the graph — that is
    # build!'s job. Returns nil.
    def realize_scratch!(cfg, t_seq, t_batch, weight_dtype, untied, qkv_bias, seed, init_scale)
      @ws_cache.realize_for_random_init(cfg, t_seq, t_batch, weight_dtype,
                                        untied, qkv_bias, seed, init_scale)
      nil
    end

    # OPTIONAL: upload an already-read donor embedding into the realize'd
    # token_embed table. Mechanism ONLY. Must be called AFTER realize_scratch!
    # and BEFORE build!. INIT=scratch skips this method entirely. Returns nil.
    def realize_warm!(donor_buf_flat, n_floats)
      TinyNNCuda.tnn_upload_from_float_array(@ws_cache.sess,
                                             @ws_cache.t_seq_token_embed,
                                             donor_buf_flat, n_floats)
      nil
    end

    # CLOSE the warm window: bake forward + CE + backward + opt_step_adamw
    # into the ggml graph. Delegates VERBATIM to build_training_step and
    # stashes the returned [t_loss, t_labels, t_hp] triple. Returns nil.
    def build!
      result       = @ws_cache.build_training_step
      @ws_t_loss   = result[0]
      @ws_t_labels = result[1]
      @ws_t_hp     = result[2]
      nil
    end

    # ONE training step. Op order is VERBATIM from the CPU recipe
    # (warm_start.rb:131-145): graph_reset on the first step else
    # reset_grads_only; the four uploads in order (token_ids/positions/
    # labels/hp); compute_backward; download_row_major(t_loss, 1, 1).
    # is_first selects the reset. The per-step LR enters ONLY via the caller
    # mutating m_hp.flat[0] before this call. Returns the loss Float.
    def step!(seq_ids, positions, m_labels, m_hp, is_first)
      s = @ws_cache.sess
      if is_first
        TinyNNCuda.tnn_graph_reset(s)
      else
        TinyNNCuda.tnn_graph_reset_grads_only(s)
      end
      TinyNNCuda.upload_int_array(s, @ws_cache.t_seq_token_ids, seq_ids)
      TinyNNCuda.upload_int_array(s, @ws_cache.t_seq_positions, positions)
      TinyNNCuda.upload_row_major(s, @ws_t_labels, m_labels)
      TinyNNCuda.upload_row_major(s, @ws_t_hp,     m_hp)
      TinyNNCuda.tnn_compute_backward(s)
      loss_mat = TinyNNCuda.download_row_major(s, @ws_t_loss, 1, 1)
      loss_mat.flat[0]
    end
  end
end; end; end
