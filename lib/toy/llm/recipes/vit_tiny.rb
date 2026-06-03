# lib/toy/llm/recipes/vit_tiny.rb — L4 recipe: ViT-Tiny from-scratch
# random-init image-classifier training plan. Encapsulates the EXISTING
# ViT training loop (examples/07_train_vit_tiny.rb at INIT=scratch): realize
# a random-init forward+CE+backward+AdamW graph on a Toy::LLM::Engine::ViTTinyEngine,
# then drive one training step at a time.
#
# Mirrors the CLASS SHAPE of from_scratch.rb (hand-written plain class, NO
# Struct.new — landmine #16 / matz/spinel#1043; NO default-arg ctor —
# landmine #4; uniquely vt_-prefixed ivars for type-isolation), but adapted
# to the ViT cache's SINGLE-return build (build_training_step returns ONE
# t_loss ptr, not the [t_loss, t_labels, t_hp] triple the llama cache
# returns — t_labels/t_hp/t_cls_idx are cache accessors here).
#
# NO require_relative: the runner (lib/toy/run/train_vit.rb) requires
# vit_tiny_engine.rb first, which pulls TinyNN + transformer + toy_vit
# + toy_smollm2 — same bare-header convention as from_scratch.rb. This recipe
# inlines TinyNN.* in step! so it is backend-coupled; CPU-only this slice (no
# CUDA twin — vit train is absent from MIRRORABLE).

module Toy; module LLM; module Recipes
  # The ViT-Tiny from-scratch random-init training recipe. realize! builds
  # the random-init forward+CE+backward+AdamW graph (realize_for_random_init
  # self-seeds every PARAM + Adam moment via Box-Muller xorshift64 — no donor
  # GGUF), then step! drives one training step. The caller (runner) owns the
  # ViT config + the per-step input Mats (image/labels/hp/cls_idx).
  class VitTiny
    attr_accessor :vt_cache, :vt_t_loss

    def initialize
      @vt_cache  = Toy::LLM::Engine::ViTTinyEngine.new
      @vt_t_loss = nil
    end

    # Realize the random-init graph. Delegates VERBATIM to the cache:
    # realize_for_random_init (3-arg: cfg, seed, init_scale — self-seeds via
    # Box-Muller, NO donor) then build_training_step (forward + CE + backward
    # + opt_step_adamw baked into the ggml graph; returns a SINGLE t_loss
    # ptr). init_scale=1.0 per 07_train_vit_tiny.rb:80. Returns nil.
    def realize!(cfg, seed, init_scale)
      @vt_cache.realize_for_random_init(cfg, seed, init_scale)
      @vt_t_loss = @vt_cache.build_training_step
      nil
    end

    # ONE training step. Op order is VERBATIM from
    # examples/07_train_vit_tiny.rb:269-281: graph_reset on the first step
    # else reset_grads_only; the four uploads in order
    # (image/cls_idx/labels/hp); compute_backward; download_row_major(t_loss,
    # 1, 1). is_first selects the reset, so the caller stays in full control
    # of the step==0 branch (matches the gate's step==0 branch exactly).
    # Reads t_image/t_cls_idx/t_labels/t_hp from the CACHE accessors (NOT a
    # stashed triple). Returns the loss Float. Per-step input Mats are built
    # by the caller (runner).
    def step!(m_image, cls_idx, m_labels, m_hp, is_first)
      s = @vt_cache.sess
      if is_first
        TinyNN.tnn_graph_reset(s)
      else
        TinyNN.tnn_graph_reset_grads_only(s)
      end
      TinyNN.upload_row_major(s, @vt_cache.t_image,   m_image)
      TinyNN.upload_int_array(s, @vt_cache.t_cls_idx, cls_idx)
      TinyNN.upload_row_major(s, @vt_cache.t_labels,  m_labels)
      TinyNN.upload_row_major(s, @vt_cache.t_hp,      m_hp)
      TinyNN.tnn_compute_backward(s)
      loss_mat = TinyNN.download_row_major(s, @vt_t_loss, 1, 1)
      loss_mat.flat[0]
    end
  end
end; end; end
