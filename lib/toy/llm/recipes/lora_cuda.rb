# lib/toy/llm/recipes/lora_cuda.rb — CUDA twin of
# lib/toy/llm/recipes/lora.rb. Hand-written (NOT mechanically mirrored):
# the CPU recipe inlines backend-coupled TinyNN.* calls in step!, so it
# cannot be substituted by the generic gen_cuda_mirror.rb rewriter — it is
# kept ABSENT from MIRRORABLE and maintained by hand. `make verify-mirrors`
# stays green.
#
# This recipe drives the frozen-base + LoRA-Q-adapter forward+CE+backward+
# AdamW graph on a LlamaSeqForwardFFICacheCuda (the CUDA cache) and computes
# via TinyNNCuda. Numerics / op-order are IDENTICAL to the CPU recipe
# (lora.rb); only the cache type (the ctor line: @lora_cache) and the
# compute module prefix in step! (TinyNN. -> TinyNNCuda.) differ.
#
# This file does NOT `require_relative "tinynn_cuda"`: the loading monolith
# (lib/llama_seq_forward_ffi_cuda.rb, required by the CUDA runner before this
# file) already loads TinyNNCuda, exactly like the CPU recipe relies on
# llama_seq_forward_ffi.rb loading TinyNN. The CUDA monolith transitively
# requires transformer -> tinynn too, so the CPU TinyNN is also defined and
# used by the runner ONLY for the checkpoint write seam (ToyDriftGrad.params
# downloads trainable params via CPU TinyNN); compute here is CUDA-only.
#
# Spinel hygiene: NEVER Struct.new (landmine #16 / matz/spinel#1043) — plain
# hand-written class, no-arg ctor, no default-arg poisoning. lora_-prefixed
# member names for type-isolation. Single-type binary: TinyNNCuda is the only
# compute module referenced here.

module Toy; module LLM; module Recipes
  # The LoRA fine-tune recipe, CUDA backend. realize! builds the frozen-base +
  # LoRA-Q-adapter forward+CE+backward+AdamW graph on a
  # LlamaSeqForwardFFICacheCuda (base weights mmap'd from the GGUF, only the
  # rank-r adapters + Adam moments are trainable), then step! drives one
  # training step via TinyNNCuda. The caller (runner) owns the loaded GGUF
  # handle, the experiment config, and the per-step input Mats.
  class LoRACuda
    attr_accessor :lora_cache, :lora_t_loss, :lora_t_labels, :lora_t_hp, :lora_step_index

    def initialize
      @lora_cache      = LlamaSeqForwardFFICacheCuda.new
      @lora_t_loss     = nil
      @lora_t_labels   = nil
      @lora_t_hp       = nil
      @lora_step_index = 0
    end

    # Realize the LoRA graph. Delegates VERBATIM to the cache in the
    # reference's order: enable_lora_q!(rank) + enable_lora_q_adamw! (set the
    # two flags BEFORE realize), then realize_for_mmap (mmap the frozen base
    # in place), then the seeded upload_lora_q_init!(seed, init_scale), then
    # build_training_step. Stashes the returned [t_loss, t_labels, t_hp]
    # triple. Positional args + order match lora.rb:65-75 exactly. Returns nil.
    def realize!(gguf_handle, cfg, t_seq, untied, qkv_bias, rank, seed, init_scale)
      @lora_cache.enable_lora_q!(rank)
      @lora_cache.enable_lora_q_adamw!
      @lora_cache.realize_for_mmap(gguf_handle, cfg, t_seq, untied, qkv_bias)
      @lora_cache.upload_lora_q_init!(seed, init_scale)
      result         = @lora_cache.build_training_step
      @lora_t_loss   = result[0]
      @lora_t_labels = result[1]
      @lora_t_hp     = result[2]
      nil
    end

    # ONE training step. Op order is VERBATIM from the CPU recipe
    # (lora.rb:87-101): graph_reset on the first step else reset_grads_only;
    # the four uploads in order (token_ids/positions/labels/hp);
    # compute_backward; download_row_major(t_loss, 1, 1). is_first selects the
    # reset. Returns the loss Float. Per-step input Mats (including the
    # bias-corrected hp row) are built by the caller.
    def step!(seq_ids, positions, m_labels, m_hp, is_first)
      s = @lora_cache.sess
      if is_first
        TinyNNCuda.tnn_graph_reset(s)
      else
        TinyNNCuda.tnn_graph_reset_grads_only(s)
      end
      TinyNNCuda.upload_int_array(s, @lora_cache.t_seq_token_ids, seq_ids)
      TinyNNCuda.upload_int_array(s, @lora_cache.t_seq_positions, positions)
      TinyNNCuda.upload_row_major(s, @lora_t_labels, m_labels)
      TinyNNCuda.upload_row_major(s, @lora_t_hp,     m_hp)
      TinyNNCuda.tnn_compute_backward(s)
      loss_mat = TinyNNCuda.download_row_major(s, @lora_t_loss, 1, 1)
      loss_mat.flat[0]
    end
  end
end; end; end
