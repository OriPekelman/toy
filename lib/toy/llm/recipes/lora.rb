# lib/toy/llm/recipes/lora.rb — L4 recipe: the LoRA fine-tune plan.
# SIBLING of FromScratch (from_scratch.rb). Same L4 shape: a minimal
# hand-written class with realize!/step!, uniquely lora_-prefixed
# members, NO Struct.new, NO speculative Trainer/Stage (AdamW is baked
# into the ggml backward graph by build_training_step, exactly as in
# FromScratch).
#
# The difference from FromScratch is realize!, NOT step!:
#   FromScratch.realize! → realize_for_random_init + build_training_step
#   LoRA.realize!        → enable_lora_q! + enable_lora_q_adamw! +
#                          realize_for_mmap (frozen GGUF base, mmap'd in
#                          place) + upload_lora_q_init! (seeded adapter
#                          init) + build_training_step.
# The train loop (step!) is LITERALLY IDENTICAL to FromScratch's:
# graph_reset on the first step else reset_grads_only, the four uploads
# in order (token_ids/positions/labels/hp), compute_backward,
# download-loss. Per the L4 design note in from_scratch.rb, a sibling
# class duplicating the ~8-line step! is acceptable and simpler than a
# shared module; we do NOT over-abstract. Op order is VERBATIM from the
# frozen reference examples/03_finetune_lora.rb:179-191.
#
# This recipe just CALLS realize_for_mmap — it does not refactor it — so
# no realize-bulk coverage rules apply.
#
# Spinel hygiene: NEVER Struct.new (landmine #16 / matz/spinel#1043) —
# hand-written plain class, explicit no-arg ctor, NO default-arg ctor
# (landmine #4). Members uniquely lora_-prefixed for type-isolation. No
# Card/step_bind/FFI :str args at class load. Experiment config (GGUF
# path, RANK, tokens, labels, hp, LR, seed/scale) stays in the FIXTURE
# (examples/smoke_recipe_lora.rb) per lib-vs-example scope, never here.
#
# Like FromScratch, this file does NOT `require_relative "tinynn"`: the
# loading module (lib/llama_seq_forward_ffi.rb, required by the fixture
# before this file) already loads the correct backend's TinyNN. step!
# inlines TinyNN.* calls so it is backend-coupled; the CUDA mirror is
# DEFERRED alongside the GPU deferral — this pass ships CPU-only.

module Toy; module LLM; module Recipes
  # The LoRA fine-tune recipe. realize! builds the frozen-base +
  # LoRA-Q-adapter forward+CE+backward+AdamW graph on a
  # Toy::LLM::Engine::LlamaSeqEngine (base weights mmap'd from the GGUF, only the
  # rank-r adapters + Adam moments are trainable), then step! drives one
  # training step. The caller (fixture) owns the loaded GGUF handle, the
  # experiment config, and the per-step input Mats.
  class LoRA
    attr_accessor :lora_cache, :lora_t_loss, :lora_t_labels, :lora_t_hp, :lora_step_index

    def initialize
      @lora_cache      = Toy::LLM::Engine::LlamaSeqEngine.new
      @lora_t_loss     = nil
      @lora_t_labels   = nil
      @lora_t_hp       = nil
      @lora_step_index = 0
    end

    # Realize the LoRA graph. Delegates VERBATIM to the cache in the
    # reference's order (03_finetune_lora.rb:67-76): enable_lora_q!(rank)
    # + enable_lora_q_adamw! (set the two flags BEFORE realize), then
    # realize_for_mmap (mmap the frozen base in place), then the seeded
    # upload_lora_q_init!(seed, init_scale) (deterministic adapter init),
    # then build_training_step (forward + CE + backward + opt_step_adamw
    # on the adapters baked into the ggml graph). Stashes the returned
    # [t_loss, t_labels, t_hp] triple. Positional args match the
    # reference call sites exactly. Returns nil.
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

    # ONE training step. Op order is VERBATIM from
    # 03_finetune_lora.rb:179-191 (and LITERALLY IDENTICAL to
    # FromScratch#step!): graph_reset on the first step else
    # reset_grads_only; the four uploads in order
    # (token_ids/positions/labels/hp); compute_backward;
    # download_row_major(t_loss, 1, 1). is_first selects the reset; the
    # @lora_step_index accessor is carried for callers that want it but is
    # NOT used for the reset decision, so the caller stays in full control
    # of the step==first branch. Returns the loss Float. Per-step input
    # Mats (including the bias-corrected hp row) are built by the caller.
    def step!(seq_ids, positions, m_labels, m_hp, is_first)
      s = @lora_cache.sess
      if is_first
        TinyNN.tnn_graph_reset(s)
      else
        TinyNN.tnn_graph_reset_grads_only(s)
      end
      TinyNN.upload_int_array(s, @lora_cache.t_seq_token_ids, seq_ids)
      TinyNN.upload_int_array(s, @lora_cache.t_seq_positions, positions)
      TinyNN.upload_row_major(s, @lora_t_labels, m_labels)
      TinyNN.upload_row_major(s, @lora_t_hp,     m_hp)
      TinyNN.tnn_compute_backward(s)
      loss_mat = TinyNN.download_row_major(s, @lora_t_loss, 1, 1)
      loss_mat.flat[0]
    end
  end
end; end; end
