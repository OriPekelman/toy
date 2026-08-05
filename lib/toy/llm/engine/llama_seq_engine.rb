# lib/toy/llm/engine/llama_seq_engine.rb — sequence-mode forward graph for the
# llama-family architecture (RMSNorm + SwiGLU + RoPE + GQA).
#
# Sibling of lib/toy_smollm2_ffi_kv.rb (KV-cache decode for single
# positions). This class builds ONE forward graph that processes T
# token positions in a single compute — the prerequisite for real SFT
# (F1.2 step 6c/6d) where masked CE across all T positions is the loss.
#
# Step 1 (this file): forward only, no LoRA, no training. Parity gate
# is T=1 vs SmolLM2KVFFICache#decode_step at pos=0; full-T parity smoke
# (T>1) follows in step 2.
#
# Spinel hygiene — see docs/design/m3-seq-forward-2026-05-21.md, risk #6:
# the M1 full-forward integration was blocked by ivar-type unification
# across cache classes (t_seq / d_model / d_ff fields colliding). To
# sidestep it cleanly, every dimension/parameter ivar here uses a
# `@seq_*` prefix. Verbose but type-isolated.

require_relative "../../models/transformer"
require_relative "../../../toy"
require_relative "../../models/toy_smollm2"
require_relative "../../ffi/tinynn"
require_relative "../primitives/rms_norm"
require_relative "../../train/dfa_b"
require_relative "../primitives/rope"
require_relative "../primitives/swiglu"
require_relative "../primitives/situ_glu"
require_relative "../primitives/gqa"
require_relative "../blocks/transformer_block"
require_relative "../primitives/gdn"
require_relative "../blocks/gdn_block"
require_relative "../primitives/kda"
require_relative "../primitives/muon"
require_relative "../blocks/kda_block"
require_relative "../primitives/mla"
require_relative "../blocks/mla_block"
require_relative "../archs/layer_spec"
require_relative "../archs/llama_arch"

module Toy; module LLM; module Engine
class LlamaSeqEngine
  attr_accessor :sess,
                # P2.5 — the five arch-level persistent handles
                # (t_seq_token_embed, t_seq_final_norm_gamma, t_seq_output,
                # t_seq_w_proj) and the blocks array now LIVE on @seq_arch
                # (Toy::LLM::Archs::LlamaArch). Cache delegators below
                # preserve the public accessor surface so the realize
                # paths, external PCA-init (fcache.t_seq_w_proj=), and the
                # examples (fcache.t_seq_*) keep working by name.
                :seq_arch,
                :seq_has_untied_output, :seq_has_qkv_bias,
                :seq_t, :seq_b, :seq_d_model, :seq_d_ff, :seq_n_heads, :seq_n_kv,
                :seq_d_head, :seq_group_size, :seq_n_layers, :seq_vocab_size,
                :seq_rope_base, :seq_rope_scaling, :t_seq_rope_freq_factors,
                :seq_rms_eps, :seq_realized, :seq_weight_dtype,
                :t_seq_token_ids, :t_seq_positions, :t_seq_attn_mask,
                :t_seq_x_embed, :t_seq_x_final, :t_seq_logits,
                :seq_gguf_handle_keepalive,
                # M3 step 3 LoRA flags. Both must be set BEFORE
                # realize_for_mmap. enable_lora_q!(r) allocates the
                # adapter A/B pair per Q head; enable_lora_q_adamw!
                # additionally allocates persistent Adam m/v alongside
                # them (mirrors F1.2 step 6b).
                :seq_lora_q_enabled, :seq_lora_q_rank,
                :seq_lora_q_adamw_enabled,
                # F3 full fine-tune. When set BEFORE
                # realize_for_full_finetune, every per-block weight
                # tensor is allocated writable F32 in ctx_w with paired
                # Adam m/v and marked set_param; build_training_step
                # then emits opt_step on each.
                :seq_full_finetune_enabled,
                # Per-cache (not per-block) parallel arrays for the
                # global trainable weights — token_embed, final-norm
                # gamma, optional untied output projection. Populated
                # in realize_for_full_finetune; empty otherwise.
                :ft_globals_weights, :ft_globals_m, :ft_globals_v,
                # Opt-in: include the embedding / final-norm /
                # untied-output tensors in full FT. Default off — the
                # embed tensor on Qwen-class models is huge (V=152K ×
                # d=1536 = 233M params F32) and brings the full
                # forward+backward+opt-step memory closer to the GPU
                # limit. Works correctly on CUDA after vendor-patches
                # 0006 chunked the get_rows_back kernel launch
                # (previously hit CUDA's gridDim.y = 65535 cap).
                :ft_train_embeddings_enabled

  # P2.5 — delegators forwarding the arch-owned handle accessors to
  # @seq_arch (Toy::LLM::Archs::LlamaArch). These preserve the cache's
  # former public attr_accessor surface (the realize paths assign via
  # self.t_seq_token_embed=, external PCA-init writes fcache.t_seq_w_proj=,
  # examples read fcache.t_seq_*). Single source of truth: the arch.
  # toy#138 K3b — AttnRes handles live on the arch (it owns the forward).
  def t_seq_attnres_ones;       @seq_arch.t_seq_attnres_ones;       end
  def seq_attnres_q;            @seq_arch.seq_attnres_q;            end
  def t_seq_token_embed;        @seq_arch.t_seq_token_embed;        end
  def t_seq_token_embed=(v);    @seq_arch.t_seq_token_embed = v;    end
  def t_seq_final_norm_gamma;     @seq_arch.t_seq_final_norm_gamma;     end
  def t_seq_final_norm_gamma=(v); @seq_arch.t_seq_final_norm_gamma = v; end
  def t_seq_output;             @seq_arch.t_seq_output;             end
  def t_seq_output=(v);         @seq_arch.t_seq_output = v;         end
  def t_seq_w_proj;             @seq_arch.t_seq_w_proj;             end
  def t_seq_w_proj=(v);         @seq_arch.t_seq_w_proj = v;         end
  # E2.3 — projection-lens donor width (0 disables the lens). Plain
  # ivar (NOT in the attr_accessor list); the GGUF-fold writer reads it
  # to know the donor->d_model contraction dimension.
  def seq_donor_d_in;           @seq_donor_d_in;                    end
  def seq_blocks_ffi;           @seq_arch.seq_blocks_ffi;           end
  def seq_blocks_ffi=(v);       @seq_arch.seq_blocks_ffi = v;       end

  def initialize
    # P2.5 — the arch owns the arch-level persistent handles + the
    # blocks array (seeded with one block in the arch ctor, matching the
    # former cache seed). Constructed first so the delegators are live.
    @seq_arch       = Toy::LLM::Archs::LlamaArch.new
    @seq_realized   = false
    @seq_t          = 0
    @seq_b          = 1
    # GH#9 — mixed-precision compute. 0 = F32 (current behaviour;
    # bit-identical to pre-GH#9). 1 = F16, 30 = BF16. When != 0,
    # weight matmuls inside build_seq_block / build_seq_qhead route
    # through mp_matmul which casts the F32 master to the chosen
    # dtype inline in the forward graph. F32 master is kept (required
    # by opt_step_adamw); the cast result lives in transient scratch.
    @seq_weight_dtype = 0
    @seq_d_model    = 0
    @seq_d_ff       = 0
    @seq_n_heads    = 0
    @seq_n_kv       = 0
    @seq_d_head     = 0
    @seq_group_size = 0
    @seq_n_layers   = 0
    @seq_gdn_layer_indices = [0];     @seq_gdn_layer_indices.pop
    @seq_is_gdn            = [false]; @seq_is_gdn.pop
    @seq_is_kda            = [false]; @seq_is_kda.pop
    @seq_kda_layer_indices = [0]; @seq_kda_layer_indices.pop
    @seq_is_mla            = [false]; @seq_is_mla.pop
    @seq_mla_layer_indices = [0]; @seq_mla_layer_indices.pop
    @seq_vocab_size = 0
    @seq_rope_base            = 10000.0
    @seq_rope_scaling         = Toy::RopeScaling.none
    # Seed a concrete Cfg from the same defaults so the ivar always
    # holds a real Toy::LLM::Primitives::RoPE::Cfg (never nil/RbVal).
    # Rebuilt per realize path once the true dims are known.
    @seq_rope_cfg             = Toy::LLM::Primitives::RoPE::Cfg.new(
                                  @seq_d_head, @seq_rope_base,
                                  @seq_rope_scaling.freq_scale,
                                  @seq_rope_scaling.ext_factor,
                                  @seq_rope_scaling.attn_factor,
                                  @seq_rope_scaling.beta_fast,
                                  @seq_rope_scaling.beta_slow)
    @t_seq_rope_freq_factors  = TinyNN.tnn_null_ptr
    @seq_rms_eps    = 1.0e-5
    @sess                  = TinyNN.tnn_null_ptr
    # P2.5 — token_embed / final_norm_gamma / output / w_proj and the
    # blocks array are seeded on @seq_arch (see arch ctor); the cache
    # reaches them via the delegators above.
    @seq_has_untied_output = false
    @seq_has_qkv_bias      = false
    @seq_gguf_handle_keepalive = TinyNN.tnn_null_ptr
    @t_seq_token_ids = TinyNN.tnn_null_ptr
    @t_seq_positions = TinyNN.tnn_null_ptr
    # GH#7 — batched-training block-causal attention mask. Allocated
    # only when @seq_b > 1 (realize_for_random_init with t_batch > 1);
    # otherwise stays NULL and build_seq_qhead falls back to the
    # diag_mask_inf + softmax path (bit-identical to today at B=1).
    @t_seq_attn_mask = TinyNN.tnn_null_ptr
    @t_seq_x_embed   = TinyNN.tnn_null_ptr
    @t_seq_x_final   = TinyNN.tnn_null_ptr
    @t_seq_logits    = TinyNN.tnn_null_ptr
    @seq_lora_q_enabled       = false
    @seq_lora_q_rank          = 0
    @seq_lora_q_adamw_enabled = false
    @seq_full_finetune_enabled = false
    # toy#129 item 2: no-shadow policy recorded BEFORE realize (empty =
    # shadow-shaped legacy — every weight is an autodiff param). Lives
    # in initialize — build_gdn_flags!/realize must NOT reset it (the
    # setter runs before realize).
    @seq_franken_noshadow_policy = [0]; @seq_franken_noshadow_policy.pop
    @seq_nope_flag = 0
    @seq_kda_conv = 1        # toy#137 K2c: ShortConv on KDA q/k/v (K3 eq 2)
    # K-series M2: the KV latent width r. 0 = 'derive from the head
    # width' (r = inner/2, K3's ratio), set explicitly by --mla-rank.
    @seq_mla_rank    = 0
    # K-series M10 (MTP). 0 = off, byte-null. lambda weights the second
    # CE root in the backward total; the REPORTED step loss stays the
    # main CE so curves remain comparable across --mtp.
    @seq_mtp        = 0
    @seq_mtp_lambda = 0.3
    @t_seq_mtp_labels = TinyNN.tnn_null_ptr
    @t_seq_mtp_loss   = TinyNN.tnn_null_ptr
    @seq_mla_kv_norm = 1     # DeepSeek's kv_a_norm; 0 only for the null
    @seq_mla_gate    = 1     # K3's full-rank output gate
    @seq_attnres = 0         # toy#138 K3b: Attention Residuals (K3 §2.2)
    @seq_optimizer = 0       # toy#139/K5: 0 adamw | 1 muon
    @t_seq_hp_sgd = TinyNN.tnn_null_ptr
    @ft_globals_weights = [TinyNN.tnn_null_ptr]; @ft_globals_weights.pop
    @ft_globals_m       = [TinyNN.tnn_null_ptr]; @ft_globals_m.pop
    @ft_globals_v       = [TinyNN.tnn_null_ptr]; @ft_globals_v.pop
    @ft_train_embeddings_enabled = false
    # E2.3 (towards GH#14) — projection-lens path. donor_d_in is read
    # from cfg in realize_for_random_init; t_seq_w_proj is the
    # trainable [donor_d_in, d_model] linear inserted after the embed
    # get_rows when donor_d_in > 0.
    @seq_donor_d_in   = 0
    # P2.5 — t_seq_w_proj is seeded on @seq_arch (arch ctor); cache
    # reaches it via the t_seq_w_proj delegator.
  end

  # F3 — additionally train the embedding / final-norm gamma / untied
  # output. Opt-in: the embed tensor on Qwen-class vocab is large and
  # makes the memory budget noticeably tighter, but the math itself
  # works correctly post vendor-patches/0006 (chunked get_rows_back).
  # GDN reintegration (docs/roadmap/gdn-hybrid-engine-reintegration.md,
  # applied 2026-07-25 post-thaw). Call BEFORE realize_for_random_init,
  # once per GDN layer index — INT ARG ONLY: the array-parameter form
  # (set_gdn_layers!(indices)) arrived EMPTY at realize (the #688
  # array-param landmine, the doc's trigger #3, re-confirmed on
  # b96280b3). No indices added (the default) = all-attention,
  # byte-identical to before.
  def add_gdn_layer!(idx)
    @seq_gdn_layer_indices.push(idx)
  end

  # toy#137 K2b: one KDA layer index per call, INT ARG ONLY (the #688
  # array-param landmine that bit GDN's set_gdn_layers!). No indices =
  # all-attention, byte-identical to before.
  def add_kda_layer!(idx)
    @seq_kda_layer_indices.push(idx)
  end

  # K-series M2: one Gated-MLA layer index per call. INT ARG ONLY, same
  # #688 array-param landmine as the KDA/GDN setters. No indices = no
  # MLA layers, byte-identical to before.
  def add_mla_layer!(idx)
    @seq_mla_layer_indices.push(idx)
  end

  # K-series M2: the KV latent rank r. Call BEFORE realize. 0 keeps the
  # derived default (inner/2).
  # K-series M10: enable the MTP module. Call BEFORE realize.
  def mtp_init(v)
    @seq_mtp = v
    @seq_arch.seq_mtp_on = v
    0
  end

  def mtp_lambda_init(v)
    @seq_mtp_lambda = v
    0
  end

  # K-series M10: upload the MTP inputs for this step. Called BEFORE
  # recipe.step! — deliberately a separate setter rather than extra
  # step! parameters, so no recipe call site changes. Same graph, so the
  # uploads are still in place when step! computes.
  def mtp_upload!(next_ids, m_labels2)
    if @seq_mtp == 0
      return 0
    end
    TinyNN.upload_int_array(@sess, @seq_arch.t_seq_mtp_tok, next_ids)
    TinyNN.upload_row_major(@sess, @t_seq_mtp_labels, m_labels2)
    0
  end

  def mtp_loss_value
    if @seq_mtp == 0
      return 0.0
    end
    m = TinyNN.download_row_major(@sess, @t_seq_mtp_loss, 1, 1)
    m.flat[0]
  end

  def seq_mtp_flag
    @seq_mtp
  end

  # Sum of squares over EVERY MTP-owned weight — the input projection
  # plus the block's own tensors. Emitted in run_start and run_end so a
  # with/without comparison PROVES the module trains.
  #
  # This is the K4b/experts_sig lesson and it is not optional here: the
  # MTP block is not in seq_blocks_ffi, so it needs its own optimizer
  # arm, and if that arm were missing the module would still build,
  # still emit a second loss, and still produce a perfectly healthy
  # training curve — while never updating a single MTP weight. The curve
  # cannot see that. This signature can.
  #
  # It must also move under --mtp-lambda 0: lambda scales the second
  # root's contribution to the BACKBONE, not the gradient reaching the
  # MTP weights from their own loss.
  def mtp_sig
    if @seq_mtp == 0
      return 0.0
    end
    acc = mtp_sumsq(@seq_arch.t_seq_mtp_proj)
    mtb = @seq_arch.seq_mtp_block
    wi = 0
    while wi < mtb.ft_weights.length
      acc = acc + mtp_sumsq(mtb.ft_weights[wi])
      wi = wi + 1
    end
    acc
  end

  def mtp_sumsq(t)
    n = TinyNN.tnn_tensor_nelements(t)
    m = TinyNN.download_row_major(@sess, t, 1, n)
    acc = 0.0
    k = 0
    while k < n
      acc = acc + m.flat[k] * m.flat[k]
      k = k + 1
    end
    acc
  end

  # Read by the recipe's step!: the MTP inputs (next-token ids and the
  # t+2 one-hot) only exist when the module is on.
  def t_seq_mtp_tok_ref
    @seq_arch.t_seq_mtp_tok
  end

  def t_seq_mtp_labels_ref
    @t_seq_mtp_labels
  end

  def t_seq_mtp_loss_ref
    @t_seq_mtp_loss
  end

  def mla_rank_init(v)
    @seq_mla_rank = v
    0
  end

  # K-series M2: 0 drops DeepSeek's kv_a_norm / K3's output gate. These
  # exist so the reduction nulls are expressible (see Primitives::MLA);
  # both default to 1, the faithful form.
  def mla_kv_norm_init(v)
    @seq_mla_kv_norm = v
    0
  end

  def mla_gate_init(v)
    @seq_mla_gate = v
    0
  end

  # toy#137 K2c: 0 disables the ShortConv on KDA q/k/v. Call BEFORE
  # realize. Default 1 = the faithful K3 form (identity-inited, so a
  # conv-on run is a step-1 forward no-op vs conv-off).
  def kda_conv_init(v)
    @seq_kda_conv = v
    0
  end

  # toy#138 K3b: 1 enables Attention Residuals (each layer's input is a
  # learned softmax mixture over the embedding + every preceding
  # layer's function output, replacing residual accumulation). Call
  # BEFORE realize. 0 = the byte-gated plain-residual path.
  def attnres_init(v)
    @seq_attnres = v
    0
  end

  # toy#139 / K-series K5 (PER-HEAD Muon). 1 = Muon on the 2D HIDDEN
  # matrices, AdamW on norms/embeddings/head (the Jordan recipe). Call
  # BEFORE realize.
  #
  # WHY THIS IS *PER-HEAD* MUON FOR FREE: K3 refines Muon by
  # partitioning the attention projections along the head dimension and
  # orthogonalizing each head's block separately ("full-matrix
  # orthogonalization treats all heads as a single coupled block").
  # toy's random-init llama layout ALREADY stores q/k/v as one tensor
  # PER HEAD (t_seq_w_q[h] et al, the GH#17 per-head decomposition), so
  # orthogonalizing each ft_weights entry IS per-head orthogonalization
  # — the refinement is structural here, not extra code.
  def optimizer_init(v)
    @seq_optimizer = v
    0
  end

  # ONE parameter's update, routed by optimizer and param class. 2D =
  # a hidden matrix (per-head q/k/v, o, gate/up/down) -> Muon; 1D
  # (norms, scalars) -> AdamW. Globals never come here.
  def emit_opt_step(t, grad, t_m, t_v, t_hp)
    to = TinyNN.tnn_null_ptr
    nel = TinyNN.tnn_tensor_nelements(t)
    ne0 = TinyNN.tnn_tensor_ne0(t)
    ne1 = nel / ne0
    if @seq_optimizer == 1 && ne1 > 1
      step = Toy::LLM::Primitives::Muon.update(@sess, grad, t_m, ne0, ne1)
      to = TinyNN.tnn_opt_step_sgd(@sess, t, step, @t_seq_hp_sgd)
    else
      to = TinyNN.tnn_opt_step_adamw(@sess, t, grad, t_m, t_v, t_hp)
    end
    TinyNN.tnn_extend_backward_graph(@sess, to)
    0
  end

  def build_gdn_flags!
    @seq_is_gdn = [false]; @seq_is_gdn.pop
    li = 0
    while li < @seq_n_layers
      @seq_is_gdn.push(false)
      li = li + 1
    end
    k = 0
    while k < @seq_gdn_layer_indices.length
      idx = @seq_gdn_layer_indices[k]
      if idx >= 0 && idx < @seq_n_layers
        @seq_is_gdn[idx] = true
      end
      k = k + 1
    end
    # toy#137 K2b: the KDA flags ride the same pass. A layer claimed by
    # BOTH lists fails loud rather than silently letting one kind win.
    @seq_is_kda = [false]; @seq_is_kda.pop
    li2 = 0
    while li2 < @seq_n_layers
      @seq_is_kda.push(false)
      li2 = li2 + 1
    end
    k2 = 0
    while k2 < @seq_kda_layer_indices.length
      idx2 = @seq_kda_layer_indices[k2]
      if idx2 >= 0 && idx2 < @seq_n_layers
        if @seq_is_gdn[idx2]
          raise "layer " + idx2.to_s + " claimed by BOTH gdn_layers and kda_layers"
        end
        @seq_is_kda[idx2] = true
        @seq_arch.set_kda_layer!(idx2)
      end
      k2 = k2 + 1
    end
    # K-series M2: same pass for MLA. A layer claimed by more than one
    # kind fails loud rather than letting the last writer win.
    @seq_is_mla = [false]; @seq_is_mla.pop
    li3 = 0
    while li3 < @seq_n_layers
      @seq_is_mla.push(false)
      li3 = li3 + 1
    end
    k3 = 0
    while k3 < @seq_mla_layer_indices.length
      idx3 = @seq_mla_layer_indices[k3]
      if idx3 >= 0 && idx3 < @seq_n_layers
        if @seq_is_gdn[idx3]
          raise "layer " + idx3.to_s + " claimed by BOTH gdn_layers and mla_layers"
        end
        if @seq_is_kda[idx3]
          raise "layer " + idx3.to_s + " claimed by BOTH kda_layers and mla_layers"
        end
        @seq_is_mla[idx3] = true
        @seq_arch.set_mla_layer!(idx3)
      end
      k3 = k3 + 1
    end
    # If NO layer is KIND_ATTENTION, nothing in the graph consumes the
    # positions tensor — it gets no backing buffer, and the recipe's
    # step! upload would write to unallocated storage. That is exactly
    # the trap --nope hit in toy#136; a k3 stack (all KDA + MLA) walks
    # into it without anyone passing --nope, and so would an all-GDN
    # stack. Raise the same guard flag from the layer kinds.
    #
    # ONLY EVER RAISES IT: an explicit seq_nope_init(1) must not be
    # undone here. build_gdn_flags! runs during realize, after the
    # setters — resetting engine ivars in this method is the wipe
    # landmine that already cost one debugging session.
    any_attn = false
    la = 0
    while la < @seq_n_layers
      if !@seq_is_gdn[la] && !@seq_is_kda[la] && !@seq_is_mla[la]
        any_attn = true
      end
      la = la + 1
    end
    if !any_attn
      @seq_nope_flag = 1
    end
  end

  def seq_gdn_blocks_ffi_ref
    @seq_arch.seq_gdn_blocks_ffi
  end

  def seq_kda_blocks_ffi_ref
    @seq_arch.seq_kda_blocks_ffi
  end

  def seq_mla_blocks_ffi_ref
    @seq_arch.seq_mla_blocks_ffi
  end

  # The effective KV latent rank: explicit --mla-rank, else K3's ratio
  # (half the attention inner width).
  def seq_mla_rank_eff
    inner = @seq_n_heads * @seq_d_head
    if @seq_mla_rank > 0
      return @seq_mla_rank
    end
    inner / 2
  end

  def enable_full_finetune_embeddings!
    @ft_train_embeddings_enabled = true
  end

  # F3 — turn on full fine-tune. Every per-block weight tensor will
  # be allocated as writable F32 in ctx_w (instead of mmap'd from the
  # GGUF), paired with persistent Adam m/v, and marked trainable.
  # Mutually exclusive with enable_lora_q!. Call BEFORE
  # realize_for_full_finetune.
  def enable_full_finetune!
    @seq_full_finetune_enabled = true
  end

  # M3 step 3 — turn on LoRA on the Q projection. Adapter A is (r, d_model),
  # B is (d_head, r). Standard init: A small Gaussian, B zero → adapter is
  # a no-op at step 0. Call BEFORE realize_for_mmap. Mirrors
  # SmolLM2KVFFICache#enable_lora_q!.
  def enable_lora_q!(r)
    @seq_lora_q_enabled = true
    @seq_lora_q_rank    = r
  end

  # M3 step 3 — allocate persistent AdamW moments next to each LoRA pair
  # (parallel to F1.2 step 6b on SmolLM2KVFFICache). Required to keep
  # optimizer state alive across reset_for_rebuild / multi-step training.
  def enable_lora_q_adamw!
    @seq_lora_q_adamw_enabled = true
  end

  # P2.6 — shared config-prologue helper. Writes the @seq_* shape/RoPE
  # ivars that every realize_for_* path needs before allocating tensors.
  # Pure ivar writes reading only cfg.*; no FFI, no graph state. Each
  # realize path keeps its own `@seq_t = t_seq` (and any path-local
  # extras) at the call site and then calls this. Byte-identical to the
  # block that previously lived inline in all four realize_for_* methods.
  def apply_seq_cfg!(cfg)
    @seq_d_model    = cfg.d_model
    @seq_d_ff       = cfg.d_ff
    @seq_n_heads    = cfg.n_heads
    @seq_n_kv       = cfg.n_kv
    @seq_d_head     = cfg.head_dim
    @seq_group_size = cfg.n_heads / cfg.n_kv
    @seq_n_layers   = cfg.n_layers
    @seq_vocab_size = cfg.vocab
    @seq_rope_base    = cfg.rope_base
    @seq_rope_scaling = cfg.rope_scaling
    @seq_rope_cfg     = Toy::LLM::Primitives::RoPE::Cfg.new(
                          @seq_d_head, @seq_rope_base,
                          @seq_rope_scaling.freq_scale,
                          @seq_rope_scaling.ext_factor,
                          @seq_rope_scaling.attn_factor,
                          @seq_rope_scaling.beta_fast,
                          @seq_rope_scaling.beta_slow)
    @seq_rms_eps    = cfg.rms_eps
  end

  # F4 alternative realize for CUDA + Q8 base. Allocates every weight
  # tensor in the standard ggml ctx_w (NOT the BYO mmap region), then
  # verbatim-copies the GGUF bytes in. Buys correctness on CUDA at the
  # cost of holding the weights twice transiently (mmap + ctx_w during
  # load; ctx_w only after). Required because the BYO-pointer cuda
  # buffer's quantized padding zeroing (cudaMemset past tensor data)
  # would otherwise crash on Q8 tensors with `ne0 % 512 != 0`.
  #
  # Use this realize when (a) the GGUF is Q8 AND (b) the backend is
  # CUDA. CPU + Q8 stays on realize_for_mmap (no padding issue).
  def realize_for_q8_copy(gguf_handle, cfg, t_seq, untied, qkv_bias)
    @seq_t          = t_seq
    apply_seq_cfg!(cfg)

    @seq_gguf_handle_keepalive = gguf_handle
    @sess                  = TinyNN.tnn_session_new(0)

    # llama3 / LongRoPE: allocate the freq_factors tensor in ctx_w
    # before finalize_weights. Values uploaded post-finalize.
    if @seq_rope_scaling.kind == :llama3
      @t_seq_rope_freq_factors = TinyNN.tnn_rope_freq_factors_alloc(@sess, cfg.head_dim)
    else
      @t_seq_rope_freq_factors = TinyNN.tnn_null_ptr
    end
    @seq_has_untied_output = untied
    @seq_has_qkv_bias      = qkv_bias

    # Read source tensor types so we can allocate ctx_w tensors of the
    # MATCHING type (verbatim copy requires source/target types match).
    eidx = TinyNN.tnn_gguf_find_index(gguf_handle, "token_embd.weight")
    etyp = TinyNN.tnn_gguf_tensor_type(gguf_handle, eidx)
    self.t_seq_token_embed = TinyNN.tnn_input_2d_persistent_typed(@sess,
                           @seq_vocab_size, @seq_d_model, etyp)
    self.t_seq_final_norm_gamma = TinyNN.tnn_input_1d_f32_persistent(@sess, @seq_d_model)
    if untied
      oidx = TinyNN.tnn_gguf_find_index(gguf_handle, "output.weight")
      otyp = TinyNN.tnn_gguf_tensor_type(gguf_handle, oidx)
      self.t_seq_output = TinyNN.tnn_input_2d_persistent_typed(@sess,
                        @seq_vocab_size, @seq_d_model, otyp)
    end

    # P2.6 Step 2 — seeding loop moved onto the arch (LlamaArch#seed_blocks!).
    @seq_arch.seed_blocks!(@seq_n_layers)

    # P2.7 pass-3 — the per-block ALLOC-typed loop body moved onto
    # TransformerBlock#alloc_q8_typed_from_gguf! (verbatim; called ONLY
    # from here). Mirrors load_from_gguf_mmap!'s arg-passing exactly: every
    # dim/flag arrives as an arg, NO ivar reads off the block. The q8 path
    # never names LoRA tensors, so the moved body is :str-free (#16-clean)
    # — no lora_name_q! back-calls (unlike load_from_gguf_mmap!).
    li = 0
    while li < @seq_n_layers
      blk = self.seq_blocks_ffi[li]
      blk.alloc_q8_typed_from_gguf!(@sess, gguf_handle, li,
                                    @seq_n_heads, @seq_n_kv, @seq_d_head, @seq_d_model,
                                    @seq_d_ff, @seq_vocab_size, @seq_lora_q_enabled,
                                    @seq_lora_q_rank, @seq_lora_q_adamw_enabled, qkv_bias)
      li = li + 1
    end

    if @seq_lora_q_enabled
      li2 = 0
      while li2 < @seq_n_layers
        blk2 = self.seq_blocks_ffi[li2]
        hq_p = 0
        while hq_p < @seq_n_heads
          TinyNN.tnn_set_param(blk2.t_seq_w_lora_a_q[hq_p])
          TinyNN.tnn_set_param(blk2.t_seq_w_lora_b_q[hq_p])
          hq_p = hq_p + 1
        end
        li2 = li2 + 1
      end
    end

    finalize_weights_and_upload_constants!

    # Load all weight bytes from the GGUF into the now-allocated
    # backend buffers. Verbatim copy keeps Q8 as Q8.
    TinyNN.tnn_gguf_copy_verbatim_to_persistent(gguf_handle, eidx, @sess, self.t_seq_token_embed)
    fnidx = TinyNN.tnn_gguf_find_index(gguf_handle, "output_norm.weight")
    TinyNN.tnn_gguf_copy_1d_to_persistent(gguf_handle, fnidx, @sess, self.t_seq_final_norm_gamma)
    if untied
      oidx2 = TinyNN.tnn_gguf_find_index(gguf_handle, "output.weight")
      TinyNN.tnn_gguf_copy_verbatim_to_persistent(gguf_handle, oidx2, @sess, self.t_seq_output)
    end

    # P2.7 pass-3 Step 2 — the per-block VERBATIM-COPY loop body moved onto
    # TransformerBlock#copy_q8_bytes_from_gguf! (verbatim; called ONLY from
    # here). The copy phase fills the backend buffers allocated by the
    # alloc_q8_typed_from_gguf! pass; the block reads its OWN t_seq_* handles
    # and writes nothing on itself. NO ivar reads off the cache — every dim
    # (n_heads, n_kv, d_head) and the qkv_bias flag arrive as ARGS. All the
    # moved primitives are tnn_gguf_copy_* / tnn_gguf_find_index — the same
    # :str-at-runtime pattern alloc_q8_typed_from_gguf! already uses, never
    # block class-load scope (#16). The GLOBALS verbatim-copy above (token
    # embed / final norm / untied output) STAYS on the cache — those touch
    # cache-level t_seq_* handles, not the block.
    li_l = 0
    while li_l < @seq_n_layers
      blk = self.seq_blocks_ffi[li_l]
      blk.copy_q8_bytes_from_gguf!(@sess, gguf_handle, li_l,
                                   @seq_n_heads, @seq_n_kv, @seq_d_head, qkv_bias)
      li_l = li_l + 1
    end

    if @seq_lora_q_adamw_enabled
      za = Mat.new(@seq_lora_q_rank, @seq_d_model)
      zb = Mat.new(@seq_d_head,      @seq_lora_q_rank)
      i = 0
      while i < @seq_lora_q_rank * @seq_d_model; za.flat[i] = 0.0; i = i + 1; end
      j = 0
      while j < @seq_d_head * @seq_lora_q_rank; zb.flat[j] = 0.0; j = j + 1; end
      li_z = 0
      while li_z < @seq_n_layers
        blk_z = self.seq_blocks_ffi[li_z]
        hqz = 0
        while hqz < @seq_n_heads
          TinyNN.upload_row_major(@sess, blk_z.t_seq_w_lora_a_q_m[hqz], za)
          TinyNN.upload_row_major(@sess, blk_z.t_seq_w_lora_a_q_v[hqz], za)
          TinyNN.upload_row_major(@sess, blk_z.t_seq_w_lora_b_q_m[hqz], zb)
          TinyNN.upload_row_major(@sess, blk_z.t_seq_w_lora_b_q_v[hqz], zb)
          hqz = hqz + 1
        end
        li_z = li_z + 1
      end
    end

    build_and_realize!
  end

  # Allocate persistent weights mmap'd from `gguf_handle` (caller is
  # responsible for keeping the handle alive — we keepalive it via
  # @seq_gguf_handle_keepalive), compute inputs, and the full forward
  # graph for T = `t_seq` positions. Fixed T; rebuild for a different T.
  #
  # Weight layout matches SmolLM2KVFFICache#realize_for_mmap exactly
  # (same byte offsets + per-head split), so a sharded GGUF can be
  # loaded by either class.
  def realize_for_mmap(gguf_handle, cfg, t_seq, untied, qkv_bias)
    @seq_t          = t_seq
    apply_seq_cfg!(cfg)

    @seq_gguf_handle_keepalive = gguf_handle
    @sess                  = TinyNN.tnn_session_new(0)

    # llama3 / LongRoPE: allocate the freq_factors tensor in ctx_w
    # before finalize_weights. Values uploaded post-finalize.
    if @seq_rope_scaling.kind == :llama3
      @t_seq_rope_freq_factors = TinyNN.tnn_rope_freq_factors_alloc(@sess, cfg.head_dim)
    else
      @t_seq_rope_freq_factors = TinyNN.tnn_null_ptr
    end
    @seq_has_untied_output = untied
    @seq_has_qkv_bias      = qkv_bias

    map_base = TinyNN.tnn_gguf_mmap_base(gguf_handle)
    map_size = TinyNN.tnn_gguf_mmap_size(gguf_handle)
    TinyNN.tnn_session_attach_weight_mmap(@sess, map_base, map_size)

    # Embeddings + final norm + optional untied LM head.
    # P2.6 pass-2 Step 1 — the three arch-owned global mmap allocs moved
    # onto LlamaArch#load_globals_from_gguf_mmap! (verbatim; called ONLY
    # from here). Mirrors the seed_blocks! / alloc_trainable_f32_weights!
    # extraction precedents.
    @seq_arch.load_globals_from_gguf_mmap!(@sess, gguf_handle,
                                           @seq_vocab_size, @seq_d_model, untied)

    # P2.6 Step 2 — seeding loop moved onto the arch (LlamaArch#seed_blocks!).
    @seq_arch.seed_blocks!(@seq_n_layers)

    # P2.7 — the per-block alloc-from-mmap-offsets loop body moved onto
    # TransformerBlock#load_from_gguf_mmap! (verbatim; called ONLY from
    # here). Mirrors the alloc_trainable_f32_weights! / seed_blocks! /
    # load_globals_from_gguf_mmap! extraction precedents. head_nbytes and
    # the LoRA :str tnn_tensor_set_name naming stay on THIS cache and are
    # back-called through the passed `self` ref (lora_name_q! /
    # lora_name_q_adam! issue the :str FFI at this runtime scope, never in
    # block class-load scope — landmine #16).
    li = 0
    while li < @seq_n_layers
      blk = self.seq_blocks_ffi[li]
      blk.load_from_gguf_mmap!(@sess, self, gguf_handle, li,
                               @seq_n_heads, @seq_n_kv, @seq_d_head, @seq_d_model,
                               @seq_d_ff, @seq_lora_q_enabled, @seq_lora_q_rank,
                               @seq_lora_q_adamw_enabled, qkv_bias)
      li = li + 1
    end

    # Mark LoRA tensors as trainable BEFORE finalize. set_param flips
    # the PARAM flag so build_backward walks them when emitting grad nodes.
    if @seq_lora_q_enabled
      li2 = 0
      while li2 < @seq_n_layers
        blk2 = self.seq_blocks_ffi[li2]
        hq_p = 0
        while hq_p < @seq_n_heads
          TinyNN.tnn_set_param(blk2.t_seq_w_lora_a_q[hq_p])
          TinyNN.tnn_set_param(blk2.t_seq_w_lora_b_q[hq_p])
          hq_p = hq_p + 1
        end
        li2 = li2 + 1
      end
    end

    finalize_weights_and_upload_constants!

    # Zero-init persistent AdamW moments. Same contract as F1.2 step 6b
    # on SmolLM2KVFFICache — m and v start at 0 per the AdamW update rule.
    if @seq_lora_q_adamw_enabled
      za = Mat.new(@seq_lora_q_rank, @seq_d_model)
      zb = Mat.new(@seq_d_head,      @seq_lora_q_rank)
      i = 0
      while i < @seq_lora_q_rank * @seq_d_model; za.flat[i] = 0.0; i = i + 1; end
      j = 0
      while j < @seq_d_head * @seq_lora_q_rank; zb.flat[j] = 0.0; j = j + 1; end
      li_z = 0
      while li_z < @seq_n_layers
        blk_z = self.seq_blocks_ffi[li_z]
        hqz = 0
        while hqz < @seq_n_heads
          TinyNN.upload_row_major(@sess, blk_z.t_seq_w_lora_a_q_m[hqz], za)
          TinyNN.upload_row_major(@sess, blk_z.t_seq_w_lora_a_q_v[hqz], za)
          TinyNN.upload_row_major(@sess, blk_z.t_seq_w_lora_b_q_m[hqz], zb)
          TinyNN.upload_row_major(@sess, blk_z.t_seq_w_lora_b_q_v[hqz], zb)
          hqz = hqz + 1
        end
        li_z = li_z + 1
      end
    end

    build_and_realize!
  end

  # F3 — full fine-tune realize path. Parallel to realize_for_mmap
  # but every per-block weight is allocated writable F32 in ctx_w
  # (no mmap), set_param-marked, paired with Adam m/v, and loaded
  # from the GGUF post-finalize via the dequantize-friendly
  # tnn_gguf_copy_* primitives. The embedding tensor + final_norm
  # gamma stay mmap'd (read-only) — the MVP doesn't train them.
  def realize_for_full_finetune(gguf_handle, cfg, t_seq, untied, qkv_bias)
    @seq_t          = t_seq
    apply_seq_cfg!(cfg)

    @seq_gguf_handle_keepalive = gguf_handle
    @sess                  = TinyNN.tnn_session_new(0)

    # llama3 / LongRoPE: allocate the freq_factors tensor in ctx_w
    # before finalize_weights. Values uploaded post-finalize.
    if @seq_rope_scaling.kind == :llama3
      @t_seq_rope_freq_factors = TinyNN.tnn_rope_freq_factors_alloc(@sess, cfg.head_dim)
    else
      @t_seq_rope_freq_factors = TinyNN.tnn_null_ptr
    end
    @seq_has_untied_output = untied
    @seq_has_qkv_bias      = qkv_bias

    # Token embed + final-norm gamma + (untied) output: trainable
    # only when opt-in (ft_train_embeddings_enabled). Otherwise
    # they stay mmap'd / read-only (still need a mmap attach for
    # this branch).
    if @ft_train_embeddings_enabled
      self.t_seq_token_embed = TinyNN.tnn_input_2d_f32_persistent(@sess,
                             @seq_vocab_size, @seq_d_model)
      ft_add_global_2d(self.t_seq_token_embed, @seq_vocab_size, @seq_d_model)
      ft_name_last_global("token_embd.weight")

      self.t_seq_final_norm_gamma = TinyNN.tnn_input_1d_f32_persistent(@sess, @seq_d_model)
      ft_add_global_1d(self.t_seq_final_norm_gamma)
      ft_name_last_global("output_norm.weight")

      if untied
        self.t_seq_output = TinyNN.tnn_input_2d_f32_persistent(@sess,
                          @seq_vocab_size, @seq_d_model)
        ft_add_global_2d(self.t_seq_output, @seq_vocab_size, @seq_d_model)
        ft_name_last_global("output.weight")
      end
    else
      map_base = TinyNN.tnn_gguf_mmap_base(gguf_handle)
      map_size = TinyNN.tnn_gguf_mmap_size(gguf_handle)
      TinyNN.tnn_session_attach_weight_mmap(@sess, map_base, map_size)

      eidx = TinyNN.tnn_gguf_find_index(gguf_handle, "token_embd.weight")
      eoff = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, eidx)
      etyp = TinyNN.tnn_gguf_tensor_type(gguf_handle, eidx)
      self.t_seq_token_embed = TinyNN.tnn_input_2d_persistent_mmap(@sess,
                             @seq_vocab_size, @seq_d_model, etyp, eoff)

      fnidx = TinyNN.tnn_gguf_find_index(gguf_handle, "output_norm.weight")
      fnoff = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, fnidx)
      self.t_seq_final_norm_gamma = TinyNN.tnn_input_1d_persistent_mmap(@sess,
                                  @seq_d_model, 0, fnoff)

      if untied
        oidx = TinyNN.tnn_gguf_find_index(gguf_handle, "output.weight")
        ooff = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, oidx)
        otyp = TinyNN.tnn_gguf_tensor_type(gguf_handle, oidx)
        self.t_seq_output = TinyNN.tnn_input_2d_persistent_mmap(@sess,
                          @seq_vocab_size, @seq_d_model, otyp, ooff)
      end
    end

    # P2.6 Step 2 — seeding loop moved onto the arch (LlamaArch#seed_blocks!).
    @seq_arch.seed_blocks!(@seq_n_layers)

    # P2-finish — per-block FT alloc lifted onto the block (verbatim:
    # TransformerBlock#alloc_full_finetune_f32_weights!), mirroring how
    # realize_for_random_init drives alloc_trainable_f32_weights!. The block
    # owns its self.t_seq_* handles + the per-block set_param loop; the cache
    # passes @sess + dims + qkv_bias and the ft_add_*/ft_name_last recorders
    # are back-called. Gated byte-exact by prep/full_finetune_gate.rb.
    li = 0
    while li < @seq_n_layers
      blk = self.seq_blocks_ffi[li]
      prefix = "blk." + li.to_s + "."
      blk.alloc_full_finetune_f32_weights!(@sess, self, prefix,
                                           @seq_d_model, @seq_d_ff, @seq_d_head,
                                           @seq_n_heads, @seq_n_kv, qkv_bias)
      li = li + 1
    end

    # Globals are trainable too only when embeddings are opt-in.
    if @ft_train_embeddings_enabled
      gi = 0
      while gi < @ft_globals_weights.length
        TinyNN.tnn_set_param(@ft_globals_weights[gi])
        gi = gi + 1
      end
    end

    finalize_weights_and_upload_constants!

    # Post-finalize: load every writable weight from the GGUF.
    if @ft_train_embeddings_enabled
      ft_load_globals(gguf_handle, untied)
    end
    ft_load_from_gguf(gguf_handle, qkv_bias)
    ft_zero_init_adam(qkv_bias)
    if @ft_train_embeddings_enabled
      ft_zero_init_adam_globals
    end

    build_and_realize!
  end

  # P2-α: from-scratch training entry. Allocates the same persistent
  # tensor layout as realize_for_full_finetune (embeddings + per-block
  # weights, all trainable F32 in ctx_w), then random-initialises
  # every weight via Ruby-side Gaussian upload — no GGUF needed.
  #
  # Force-enables `@ft_train_embeddings_enabled` so the existing
  # full-FT machinery allocates persistent F32 embeddings instead
  # of the mmap branch. Caller doesn't need to call
  # enable_full_finetune_embeddings! first.
  #
  # Currently Llama-arch only (RMSNorm + GQA + RoPE + SwiGLU). Other
  # architectures (GPT-2 LN, MHA + biases) need a separate trainer
  # cache class; deferred until we actually need GPT-2 from-scratch.
  def realize_for_random_init(cfg, t_seq, t_batch, weight_dtype, untied, qkv_bias, seed, init_scale)
    @ft_train_embeddings_enabled = true   # forces persistent-F32 alloc of embeddings
    @seq_full_finetune_enabled   = true   # build_training_step gates on this


    @seq_t          = t_seq
    # GH#7 — micro-batching. B=1 keeps the codepath bit-identical to
    # the pre-GH#7 single-sequence training. B>1 lays out tokens as a
    # flat [T*B] vector with a block-causal attention mask uploaded
    # post-finalize and applied via soft_max_ext.
    @seq_b          = t_batch
    # GH#9 — mixed-precision compute. 0 = F32 (bit-identical to
    # pre-GH#9). 1 = F16, 30 = BF16. See mp_matmul + ivar comment in
    # initialize for the master-copy details.
    @seq_weight_dtype = weight_dtype
    apply_seq_cfg!(cfg)

    @sess                  = TinyNN.tnn_session_new(0)
    # GH#17 — per-head decomposition makes node count scale as
    # O(n_layers × n_heads). The default 65536 cap overflows on
    # 24L × 16-head Qwen-shape at backward-expand. Empirically a
    # 24L × 16-head model needs ~450k nodes for forward + backward +
    # AdamW, so we budget ~1000 nodes per (layer × head) cell + floor.
    cap = cfg.n_layers * cfg.n_heads * 1000 + 65536
    TinyNN.tnn_session_set_graph_capacity(@sess, cap)
    @seq_has_untied_output = untied
    @seq_has_qkv_bias      = qkv_bias
    @seq_donor_d_in        = cfg.donor_d_in   # E2.3 — 0 disables projection lens

    if @seq_rope_scaling.kind == :llama3
      @t_seq_rope_freq_factors = TinyNN.tnn_rope_freq_factors_alloc(@sess, cfg.head_dim)
    else
      @t_seq_rope_freq_factors = TinyNN.tnn_null_ptr
    end

    # Globals — trainable persistent F32 (+ E2.3 projection-lens branch when
    # @seq_donor_d_in > 0). P2-finish: the alloc lifted onto the arch
    # (LlamaArch#alloc_globals_trainable_f32!), which already owns these handles
    # — verbatim, same order, byte-identical. @ft_globals_* recorders + the
    # frozen-embed namer are back-called through `self`.
    @seq_arch.alloc_globals_trainable_f32!(@sess, self, @seq_vocab_size,
                                           @seq_d_model, @seq_donor_d_in, untied)
    # toy#138 K3b: the AttnRes pseudo-queries register as GLOBALS, so
    # the existing param-marking + opt walker pick them up with no new
    # arm. Must precede seed_blocks!/finalize (persistent inputs are
    # alloc-before-finalize — the toy#133 lesson).
    if @seq_attnres == 1
      @seq_arch.alloc_attnres!(@sess, self, @seq_n_layers, @seq_d_model)
    end
    # toy#139: Muon applies through ggml's SGD step ([alpha, wd]).
    # PERSISTENT — a compute-context input has no buffer unless the
    # graph reaches it, and the per-step upload would then abort.
    if @seq_optimizer == 1
      @t_seq_hp_sgd = TinyNN.tnn_input_1d_f32_persistent(@sess, 2)
    end

    # Per-block weights — identical structure to realize_for_full_finetune.
    # P2.6 Step 2 — the block-array seeding loop now lives on the arch
    # (LlamaArch#seed_blocks!), which already owns @seq_blocks_ffi.
    @seq_arch.seed_blocks!(@seq_n_layers)

    # GDN reintegration: mark layer kinds (INT-arg calls only — the
    # #688 array-param landmine shape stays banned).
    build_gdn_flags!
    marked = 0
    kk = 0
    while kk < @seq_gdn_layer_indices.length
      gidx = @seq_gdn_layer_indices[kk]
      if gidx >= 0 && gidx < @seq_n_layers
        @seq_arch.set_gdn_layer!(gidx)
        marked = marked + 1
      end
      kk = kk + 1
    end
    if @seq_gdn_layer_indices.length > 0
      puts "gdn: requested=" + @seq_gdn_layer_indices.length.to_s +
           " marked=" + marked.to_s + " of " + @seq_n_layers.to_s + " layers"
    end

    # P2.6 Step 4 — the per-block F32 ALLOC loop body now lives on the
    # block (TransformerBlock#alloc_trainable_f32_weights!), which already
    # OWNS these self.t_seq_* handles at forward time. The block takes
    # @sess + the seq dims + the name prefix as ARGS (no ivar reads on the
    # block) and calls the cache's ft_add_1d / ft_add_2d / ft_name_last
    # recorders BACK through the passed `self` reference — those stay on
    # the cache (they read @sess and issue tnn_tensor_set_name :str at
    # runtime; never migrate into block class-load scope). w_o keeps its
    # random_init shape ne=[d_model, n_heads*d_head] inside the block
    # method (not unified with full_finetune's [d_model,d_model]).
    li = 0
    while li < @seq_n_layers
      if @seq_is_gdn[li]
        gblk = self.seq_gdn_blocks_ffi_ref[li]
        gblk.alloc_trainable_f32_weights!(@sess, @seq_d_model, @seq_d_head,
                                          @seq_n_heads)
        gblk.set_params!
      elsif @seq_is_kda[li]
        kblk = self.seq_kda_blocks_ffi_ref[li]
        kblk.alloc_trainable_f32_weights!(@sess, @seq_d_model, @seq_d_head,
                                          @seq_n_heads, @seq_kda_conv)
        kblk.set_params!
      elsif @seq_is_mla[li]
        mblk = self.seq_mla_blocks_ffi_ref[li]
        mblk.alloc_trainable_f32_weights!(@sess, @seq_d_model, @seq_d_head,
                                          @seq_n_heads, self.seq_mla_rank_eff,
                                          @seq_mla_kv_norm, @seq_mla_gate)
        mblk.set_params!
      else
        blk = self.seq_blocks_ffi[li]
        prefix = "blk." + li.to_s + "."
        blk.alloc_trainable_f32_weights!(@sess, self, prefix,
                                         @seq_d_model, @seq_d_ff, @seq_d_head,
                                         @seq_n_heads, @seq_n_kv)
        # toy#129 item 2: param marking (moved out of the block alloc).
        # Shadow mode (empty noshadow policy) marks every weight —
        # byte-identical to the old in-block loop. No-shadow: dfa qkv
        # weights stay OUT of the param set (late-param at build time,
        # the MoE-top1 idiom) so build_backward never expands them.
        ns_mode = 0
        if li < @seq_franken_noshadow_policy.length
          ns_mode = @seq_franken_noshadow_policy[li]
        end
        wp = 0
        while wp < blk.ft_weights.length
          skip_qkv = ns_mode > 0 && wp >= 2 && wp < 2 + @seq_n_heads + 2 * @seq_n_kv
          if !skip_qkv
            TinyNN.tnn_set_param(blk.ft_weights[wp])
          end
          wp = wp + 1
        end
      end
      li = li + 1
    end

    # K-series M10: the MTP module's own weights — the [d, 2d] input
    # projection and ONE block mirroring a backbone layer. Allocated
    # AFTER the layer loop so no existing weight's registration order
    # moves, which is what keeps --mtp off byte-null.
    if @seq_mtp == 1
      @seq_arch.t_seq_mtp_proj = TinyNN.tnn_input_2d_f32_persistent(@sess, @seq_d_model, 2 * @seq_d_model)
      TinyNN.tnn_set_param(@seq_arch.t_seq_mtp_proj)
      ft_add_global_2d(@seq_arch.t_seq_mtp_proj, @seq_d_model, 2 * @seq_d_model)
      @seq_arch.t_seq_mtp_tok = TinyNN.tnn_input_1d_i32(@sess, @seq_t * @seq_b)
      mtb = @seq_arch.seq_mtp_block
      mtb.alloc_trainable_f32_weights!(@sess, self, "mtp.",
                                       @seq_d_model, @seq_d_ff, @seq_d_head,
                                       @seq_n_heads, @seq_n_kv)
      wm = 0
      while wm < mtb.ft_weights.length
        TinyNN.tnn_set_param(mtb.ft_weights[wm])
        wm = wm + 1
      end
    end

    # Mark globals as params too (gated on @ft_train_embeddings_enabled).
    gi = 0
    while gi < @ft_globals_weights.length
      TinyNN.tnn_set_param(@ft_globals_weights[gi])
      gi = gi + 1
    end

    finalize_weights_and_upload_constants!

    # Random-init every weight + zero biases + ones gammas.
    upload_random_init!(seed, init_scale, qkv_bias, untied)
    ft_zero_init_adam(qkv_bias)
    ft_zero_init_adam_globals

    build_and_realize!
  end

  # GH#7 — build + upload the block-causal attention mask for B>1.
  # Layout: scores from matmul(K[d_head, T*B], Q[d_head, T*B]) have
  # ne=[T*B, T*B] where ne0 indexes keys and ne1 indexes queries
  # (ggml column-major: flat[ne0_idx + ne1_idx * T*B]). For query
  # position i1 = b_q*T + p_q and key position i0 = b_k*T + p_k:
  #   mask = 0.0  iff b_k == b_q AND p_k <= p_q   (intra-batch causal)
  #   mask = NEG  otherwise                      (cross-batch + future)
  # NEG = -1.0e30 so exp(NEG) == 0.0 in f32 (avoids Float::INFINITY,
  # which would also work but is one less Spinel codegen variable).
  def upload_block_causal_mask!
    tb = @seq_t * @seq_b
    neg = -1.0e30
    mask_arr = [0.0]; mask_arr.pop
    i1 = 0
    while i1 < tb
      b_q = i1 / @seq_t
      p_q = i1 % @seq_t
      i0 = 0
      while i0 < tb
        b_k = i0 / @seq_t
        p_k = i0 % @seq_t
        if b_k == b_q && p_k <= p_q
          mask_arr.push(0.0)
        else
          mask_arr.push(neg)
        end
        i0 = i0 + 1
      end
      i1 = i1 + 1
    end
    TinyNN.tnn_upload_from_float_array(@sess, @t_seq_attn_mask, mask_arr, mask_arr.length)
  end

  # Fill every persistent weight tensor with N(0, std) values.
  # Norm gammas → 1.0, biases (if present) → 0.0, matmul weights →
  # N(0, init_scale/sqrt(fan_in)). Token embedding uses GPT-2-style
  # N(0, 0.02). All values computed in Ruby, uploaded in bulk via
  # tnn_upload_from_float_array.
  def upload_random_init!(seed, init_scale, qkv_bias, untied)
    state = lcg_seed_state(seed)

    # Token embed: width depends on projection lens.
    # When donor_d_in > 0, embed is [vocab, donor_d_in] (caller may
    # overwrite with real donor values after realize); the trainable
    # projection W_proj [donor_d_in, d_model] also gets a Gaussian init.
    embed_cols = @seq_donor_d_in > 0 ? @seq_donor_d_in : @seq_d_model
    upload_gaussian(self.t_seq_token_embed, @seq_vocab_size * embed_cols, 0.02, state)
    if @seq_donor_d_in > 0
      upload_gaussian(self.t_seq_w_proj, @seq_donor_d_in * @seq_d_model,
                       1.0 / Math.sqrt(@seq_donor_d_in.to_f), state)
    end
    upload_constant(self.t_seq_final_norm_gamma, @seq_d_model, 1.0)
    if @seq_attnres == 1
      # ones-gamma for the score kernel's RMSNorm (constant, no params
      # of its own); pseudo-queries ZERO — a zero query gives equal
      # scores, i.e. the mixture starts as the plain MEAN over sources,
      # a neutral and fully-specified starting point that the softmax
      # then learns away from.
      upload_constant(self.t_seq_attnres_ones, @seq_d_model, 1.0)
      aq = 0
      while aq < self.seq_attnres_q.length
        upload_constant(self.seq_attnres_q[aq], @seq_d_model, 0.0)
        aq = aq + 1
      end
    end
    if untied
      upload_gaussian(self.t_seq_output, @seq_vocab_size * @seq_d_model, 0.02, state)
    end

    inv_sqrt_d   = init_scale / Math.sqrt(@seq_d_model.to_f)
    inv_sqrt_dff = init_scale / Math.sqrt(@seq_d_ff.to_f)

    li = 0
    while li < @seq_n_layers
      if @seq_is_gdn[li]
        gblk = self.seq_gdn_blocks_ffi_ref[li]
        inner = @seq_n_heads * @seq_d_head
        upload_constant(gblk.t_rn_gamma, @seq_d_model, 1.0)
        upload_gaussian(gblk.t_w_q, inner * @seq_d_model, inv_sqrt_d, state)
        upload_gaussian(gblk.t_w_k, inner * @seq_d_model, inv_sqrt_d, state)
        upload_gaussian(gblk.t_w_v, inner * @seq_d_model, inv_sqrt_d, state)
        upload_gaussian(gblk.t_w_z, inner * @seq_d_model, inv_sqrt_d, state)
        upload_gaussian(gblk.t_w_a, @seq_n_heads * @seq_d_model, inv_sqrt_d, state)
        upload_gaussian(gblk.t_w_b, @seq_n_heads * @seq_d_model, inv_sqrt_d, state)
        # Small-noise init for the decay pair (NOT the doc's planned 0.0:
        # empirically the exact-zero pair NaNs at T=32 on the smoke shape,
        # while the hybrid runner's small-noise init trains — mirror it).
        upload_gaussian(gblk.t_a_log,   @seq_n_heads, 0.25, state)
        upload_gaussian(gblk.t_dt_bias, @seq_n_heads, 0.25, state)
        upload_constant(gblk.t_go_gamma, inner, 1.0)
        upload_gaussian(gblk.t_w_o, @seq_d_model * inner, inv_sqrt_d, state)
        li = li + 1
        next
      end
      if @seq_is_kda[li]
        # toy#137 K2b. Init notes: b_alpha ZERO and a_log ZERO put the
        # decay at g = -5*sigmoid(z) with z from the low-rank projection
        # — i.e. alpha starts spread through (e^-5, 1) rather than pinned,
        # which is what K3's A_h=0 initialization intends (their b_alpha
        # follows Kimi Linear's per-head schedule; zero is the neutral
        # toy-scale choice and keeps the stream's own noise as the only
        # asymmetry). The decay pair is NOT small-noise-inited like GDN's:
        # GDN needed that because its unbounded -e^A*softplus path NaN'd
        # from exact zeros; the bounded sigmoid cannot.
        kblk = self.seq_kda_blocks_ffi_ref[li]
        inner_k = @seq_n_heads * @seq_d_head
        rank_k  = @seq_d_head
        upload_constant(kblk.t_rn_gamma, @seq_d_model, 1.0)
        upload_gaussian(kblk.t_w_q, inner_k * @seq_d_model, inv_sqrt_d, state)
        upload_gaussian(kblk.t_w_k, inner_k * @seq_d_model, inv_sqrt_d, state)
        upload_gaussian(kblk.t_w_v, inner_k * @seq_d_model, inv_sqrt_d, state)
        upload_gaussian(kblk.t_w_g, inner_k * @seq_d_model, inv_sqrt_d, state)
        upload_gaussian(kblk.t_w_b, @seq_n_heads * @seq_d_model, inv_sqrt_d, state)
        upload_gaussian(kblk.t_w_ad, rank_k * @seq_d_model, inv_sqrt_d, state)
        upload_gaussian(kblk.t_w_au, inner_k * rank_k,
                        init_scale / Math.sqrt(rank_k.to_f), state)
        upload_constant(kblk.t_b_alpha, inner_k, 0.0)
        upload_constant(kblk.t_a_log,   @seq_n_heads, 0.0)
        upload_constant(kblk.t_go_gamma, inner_k, 1.0)
        upload_gaussian(kblk.t_w_o, @seq_d_model * inner_k, inv_sqrt_d, state)
        if @seq_kda_conv == 1
          # IDENTITY: tap 0 = 1.0, taps 1..3 = 0.0 (an exact forward
          # no-op at step 1; the taps train away from it).
          upload_constant(kblk.t_cq0, inner_k, 1.0)
          upload_constant(kblk.t_cq1, inner_k, 0.0)
          upload_constant(kblk.t_cq2, inner_k, 0.0)
          upload_constant(kblk.t_cq3, inner_k, 0.0)
          upload_constant(kblk.t_ck0, inner_k, 1.0)
          upload_constant(kblk.t_ck1, inner_k, 0.0)
          upload_constant(kblk.t_ck2, inner_k, 0.0)
          upload_constant(kblk.t_ck3, inner_k, 0.0)
          upload_constant(kblk.t_cv0, inner_k, 1.0)
          upload_constant(kblk.t_cv1, inner_k, 0.0)
          upload_constant(kblk.t_cv2, inner_k, 0.0)
          upload_constant(kblk.t_cv3, inner_k, 0.0)
        end
        li = li + 1
        next
      end
      if @seq_is_mla[li]
        # K-series M2. The latent up-projections are scaled by 1/sqrt(r)
        # rather than 1/sqrt(d_model): their FAN-IN is the latent width,
        # so reusing inv_sqrt_d would shrink K/V by sqrt(r/d_model) and
        # start the attention logits near-degenerate. Same reasoning as
        # KDA's t_w_au above.
        mblk = self.seq_mla_blocks_ffi_ref[li]
        inner_m = @seq_n_heads * @seq_d_head
        rank_m  = self.seq_mla_rank_eff
        inv_sqrt_r = init_scale / Math.sqrt(rank_m.to_f)
        upload_constant(mblk.t_rn_gamma, @seq_d_model, 1.0)
        upload_gaussian(mblk.t_w_kv_a, rank_m * @seq_d_model, inv_sqrt_d, state)
        hm = 0
        while hm < @seq_n_heads
          upload_gaussian(mblk.t_w_q[hm],   @seq_d_head * @seq_d_model, inv_sqrt_d, state)
          upload_gaussian(mblk.t_w_k_b[hm], @seq_d_head * rank_m, inv_sqrt_r, state)
          upload_gaussian(mblk.t_w_v_b[hm], @seq_d_head * rank_m, inv_sqrt_r, state)
          hm = hm + 1
        end
        if @seq_mla_kv_norm == 1
          upload_constant(mblk.t_kv_gamma, rank_m, 1.0)
        end
        if @seq_mla_gate == 1
          upload_gaussian(mblk.t_w_g, inner_m * @seq_d_model, inv_sqrt_d, state)
          upload_constant(mblk.t_go_gamma, inner_m, 1.0)
        end
        upload_gaussian(mblk.t_w_o, @seq_d_model * inner_m, inv_sqrt_d, state)
        li = li + 1
        next
      end
      blk = self.seq_blocks_ffi[li]
      upload_constant(blk.t_seq_rn1_gamma, @seq_d_model, 1.0)
      upload_constant(blk.t_seq_rn2_gamma, @seq_d_model, 1.0)

      hq = 0
      while hq < @seq_n_heads
        upload_gaussian(blk.t_seq_w_q[hq], @seq_d_head * @seq_d_model, inv_sqrt_d, state)
        hq = hq + 1
      end
      hkv = 0
      while hkv < @seq_n_kv
        upload_gaussian(blk.t_seq_w_k[hkv], @seq_d_head * @seq_d_model, inv_sqrt_d, state)
        upload_gaussian(blk.t_seq_w_v[hkv], @seq_d_head * @seq_d_model, inv_sqrt_d, state)
        hkv = hkv + 1
      end

      upload_gaussian(blk.t_seq_w_o,    @seq_d_model * @seq_n_heads * @seq_d_head, inv_sqrt_d, state)
      upload_gaussian(blk.t_seq_w_gate, @seq_d_ff    * @seq_d_model, inv_sqrt_d, state)
      upload_gaussian(blk.t_seq_w_up,   @seq_d_ff    * @seq_d_model, inv_sqrt_d, state)
      upload_gaussian(blk.t_seq_w_down, @seq_d_model * @seq_d_ff,    inv_sqrt_dff, state)
      li = li + 1
    end

    # K-series M10: the MTP module draws LAST from the same stream, so
    # every pre-existing weight keeps the draws it had — --mtp off and
    # --mtp on share an identical backbone init, which is what makes the
    # gate's "same init, MTP only adds a head" comparison meaningful.
    if @seq_mtp == 1
      mtb = @seq_arch.seq_mtp_block
      upload_gaussian(@seq_arch.t_seq_mtp_proj, @seq_d_model * 2 * @seq_d_model, inv_sqrt_d, state)
      upload_constant(mtb.t_seq_rn1_gamma, @seq_d_model, 1.0)
      upload_constant(mtb.t_seq_rn2_gamma, @seq_d_model, 1.0)
      hqm = 0
      while hqm < @seq_n_heads
        upload_gaussian(mtb.t_seq_w_q[hqm], @seq_d_head * @seq_d_model, inv_sqrt_d, state)
        hqm = hqm + 1
      end
      hkm = 0
      while hkm < @seq_n_kv
        upload_gaussian(mtb.t_seq_w_k[hkm], @seq_d_head * @seq_d_model, inv_sqrt_d, state)
        upload_gaussian(mtb.t_seq_w_v[hkm], @seq_d_head * @seq_d_model, inv_sqrt_d, state)
        hkm = hkm + 1
      end
      upload_gaussian(mtb.t_seq_w_o,    @seq_d_model * @seq_n_heads * @seq_d_head, inv_sqrt_d, state)
      upload_gaussian(mtb.t_seq_w_gate, @seq_d_ff    * @seq_d_model, inv_sqrt_d, state)
      upload_gaussian(mtb.t_seq_w_up,   @seq_d_ff    * @seq_d_model, inv_sqrt_d, state)
      upload_gaussian(mtb.t_seq_w_down, @seq_d_model * @seq_d_ff,    inv_sqrt_dff, state)
    end
  end

  # toy#114 — the init stream is a 31-bit LCG (the lora/gpt2 idiom, proven
  # byte-consistent CRuby<->Spinel<->CUDA by the mri/lora differential
  # gates). The previous 64-bit xorshift stream read NEGATIVE under Spinel
  # past 2^63 (matz/spinel#3371: signed-int64 wrap where CRuby promotes to
  # bignum) and silently diverged from CRuby -- with all LCG state below
  # 2^31, no value ever approaches the wrap. Seed enters through a
  # multiply-fold + 8 warmup draws: nonzero and well-mixed for EVERY seed
  # including 0 (the raw [seed] xorshift state had a zero fixed point that
  # froze +-37sigma inits into the historical baselines).
  # Constant-fill upload (gammas to 1.0, biases/decays to 0.0).
  # toy#137 K2c: IDENTITY init for a [inner, KSIZE] short-conv weight —
  # tap 0 = 1.0, later taps 0.0 (ne=[inner,K] so tap i occupies
  # indices [i*inner, (i+1)*inner)). Makes the conv an exact forward
  # no-op at step 1; the taps train away from it.
  def upload_conv_identity(t, inner, ksize)
    vals = [0.0]; vals.pop
    i = 0
    while i < inner * ksize
      if i < inner
        vals.push(1.0)
      else
        vals.push(0.0)
      end
      i = i + 1
    end
    TinyNN.tnn_upload_from_float_array(@sess, t, vals, inner * ksize)
  end

  def upload_constant(tensor, n, v)
    buf = [0.0]; buf.pop
    i = 0
    while i < n
      buf.push(v)
      i = i + 1
    end
    TinyNN.tnn_upload_from_float_array(@sess, tensor, buf, n)
  end

  # Box-Muller gaussian upload from the LCG uniform stream. state is a
  # one-element Array<Integer> so the mutable PRNG state survives across
  # calls without class variables. Emits exactly `n` gaussian F32 values.
  def upload_gaussian(tensor, n, std, state)
    buf = [0.0]; buf.pop
    pair = 0
    saved = 0.0
    i = 0
    while i < n
      if pair == 0
        u1 = lcg_uniform!(state)
        u2 = lcg_uniform!(state)
        if u1 < 1.0e-300; u1 = 1.0e-300; end
        r = Math.sqrt(-2.0 * Math.log(u1))
        theta = 2.0 * Math::PI * u2
        z0 = r * Math.cos(theta) * std
        z1 = r * Math.sin(theta) * std
        buf.push(z0)
        saved = z1
        pair = 1
      else
        buf.push(saved)
        pair = 0
      end
      i = i + 1
    end
    TinyNN.tnn_upload_from_float_array(@sess, tensor, buf, n)
  end

  def lcg_uniform!(state)
    s = state[0]
    s = (s * 1103515245 + 12345) & 0x7FFFFFFF
    state[0] = s
    (s.to_f + 1.0) / 2147483648.0
  end

  def lcg_seed_state(seed)
    s = ((seed + 104729) * 2654435761) % 2147483647
    if s <= 0
      s = seed + 104729
    end
    st = [s]
    w = 0
    while w < 8
      lcg_uniform!(st)
      w = w + 1
    end
    st
  end

  # Append (weight, m, v) to the block's parallel arrays. Allocates
  # Adam m and v of the same shape as `weight` as a side effect.
  def ft_add_2d(blk, weight, rows, cols)
    blk.ft_weights.push(weight)
    blk.ft_m.push(TinyNN.tnn_input_2d_f32_persistent(@sess, rows, cols))
    blk.ft_v.push(TinyNN.tnn_input_2d_f32_persistent(@sess, rows, cols))
  end

  def ft_add_1d(blk, weight)
    n = TinyNN.tnn_tensor_nelements(weight)
    blk.ft_weights.push(weight)
    blk.ft_m.push(TinyNN.tnn_input_1d_f32_persistent(@sess, n))
    blk.ft_v.push(TinyNN.tnn_input_1d_f32_persistent(@sess, n))
  end

  # Name the most-recently-pushed (weight, m, v) triple in a block.
  # Used right after ft_add_2d / ft_add_1d so drift/grad event consumers
  # see llama.cpp-convention names like "blk.0.attn_norm.weight" instead
  # of ggml's auto-generated "node_N". toy#semantic-tensor-names.
  def ft_name_last(blk, name)
    last = blk.ft_weights.length - 1
    TinyNN.tnn_tensor_set_name(blk.ft_weights[last], name)
    TinyNN.tnn_tensor_set_name(blk.ft_m[last],       name + ".m")
    TinyNN.tnn_tensor_set_name(blk.ft_v[last],       name + ".v")
  end

  def ft_name_last_global(name)
    last = @ft_globals_weights.length - 1
    TinyNN.tnn_tensor_set_name(@ft_globals_weights[last], name)
    TinyNN.tnn_tensor_set_name(@ft_globals_m[last],       name + ".m")
    TinyNN.tnn_tensor_set_name(@ft_globals_v[last],       name + ".v")
  end

  # Name a single FROZEN global (e.g. the projection-lens donor embed, which is
  # NOT pushed to @ft_globals so ft_name_last_global cannot reach it). Kept on
  # the engine so this tnn_tensor_set_name(:str) FFI stays on the cache realize
  # runtime path — same discipline as ft_name_last / lora_name_q!. Back-called
  # by LlamaArch#alloc_globals_trainable_f32!.
  def name_global!(t, name)
    TinyNN.tnn_tensor_set_name(t, name)
  end

  # P2.7 — LoRA-Q tensor naming callbacks for the extracted block-side
  # mmap loader (TransformerBlock#load_from_gguf_mmap!). The :str
  # tnn_tensor_set_name FFI calls MUST stay on the cache realize RUNTIME
  # path — never migrate into block class-load scope (step_bind / :str
  # landmine #16). The block assembles the runtime name string and hands
  # it here, exactly as it hands ft_name_last its assembled name. Verbatim
  # lift of the former realize_for_mmap loop lines 567-570 / 597-604.
  def lora_name_q!(t_a, t_b, head_prefix)
    TinyNN.tnn_tensor_set_name(t_a, head_prefix + ".lora_a.weight")
    TinyNN.tnn_tensor_set_name(t_b, head_prefix + ".lora_b.weight")
  end

  def lora_name_q_adam!(t_a_m, t_a_v, t_b_m, t_b_v, head_prefix)
    TinyNN.tnn_tensor_set_name(t_a_m, head_prefix + ".lora_a.m")
    TinyNN.tnn_tensor_set_name(t_a_v, head_prefix + ".lora_a.v")
    TinyNN.tnn_tensor_set_name(t_b_m, head_prefix + ".lora_b.m")
    TinyNN.tnn_tensor_set_name(t_b_v, head_prefix + ".lora_b.v")
  end

  # Same shape as ft_add_2d / ft_add_1d but writes to the cache-level
  # globals arrays (token_embed, final-norm, untied output).
  def ft_add_global_2d(weight, rows, cols)
    @ft_globals_weights.push(weight)
    @ft_globals_m.push(TinyNN.tnn_input_2d_f32_persistent(@sess, rows, cols))
    @ft_globals_v.push(TinyNN.tnn_input_2d_f32_persistent(@sess, rows, cols))
  end

  def ft_add_global_1d(weight)
    n = TinyNN.tnn_tensor_nelements(weight)
    @ft_globals_weights.push(weight)
    @ft_globals_m.push(TinyNN.tnn_input_1d_f32_persistent(@sess, n))
    @ft_globals_v.push(TinyNN.tnn_input_1d_f32_persistent(@sess, n))
  end

  # Load token_embed + final-norm + (untied) output from the GGUF
  # into their now-allocated backend buffers.
  def ft_load_globals(gguf, untied)
    eidx = TinyNN.tnn_gguf_find_index(gguf, "token_embd.weight")
    TinyNN.tnn_gguf_copy_to_persistent(gguf, eidx, @sess, self.t_seq_token_embed)
    fnidx = TinyNN.tnn_gguf_find_index(gguf, "output_norm.weight")
    TinyNN.tnn_gguf_copy_1d_to_persistent(gguf, fnidx, @sess, self.t_seq_final_norm_gamma)
    if untied
      oidx = TinyNN.tnn_gguf_find_index(gguf, "output.weight")
      TinyNN.tnn_gguf_copy_to_persistent(gguf, oidx, @sess, self.t_seq_output)
    end
  end

  def ft_zero_init_adam_globals
    gi = 0
    while gi < @ft_globals_weights.length
      TinyNN.tnn_zero_tensor(@sess, @ft_globals_m[gi])
      TinyNN.tnn_zero_tensor(@sess, @ft_globals_v[gi])
      gi = gi + 1
    end
  end

  # Pull bytes from the GGUF into each writable weight. Uses the
  # existing C-side dequantize-and-copy primitives so a Q8 source
  # transparently becomes F32 in the target tensor.
  def ft_load_from_gguf(gguf, qkv_bias)
    li = 0
    while li < @seq_n_layers
      blk = self.seq_blocks_ffi[li]
      prefix = "blk." + li.to_s

      rn1_idx = TinyNN.tnn_gguf_find_index(gguf, prefix + ".attn_norm.weight")
      rn2_idx = TinyNN.tnn_gguf_find_index(gguf, prefix + ".ffn_norm.weight")
      TinyNN.tnn_gguf_copy_1d_to_persistent(gguf, rn1_idx, @sess, blk.t_seq_rn1_gamma)
      TinyNN.tnn_gguf_copy_1d_to_persistent(gguf, rn2_idx, @sess, blk.t_seq_rn2_gamma)

      q_idx = TinyNN.tnn_gguf_find_index(gguf, prefix + ".attn_q.weight")
      hq = 0
      while hq < @seq_n_heads
        TinyNN.tnn_gguf_copy_head_slice_to_persistent_native(gguf, q_idx, @sess,
          blk.t_seq_w_q[hq], hq, @seq_n_heads, @seq_d_model, @seq_d_head)
        hq = hq + 1
      end

      k_idx = TinyNN.tnn_gguf_find_index(gguf, prefix + ".attn_k.weight")
      v_idx = TinyNN.tnn_gguf_find_index(gguf, prefix + ".attn_v.weight")
      hkv = 0
      while hkv < @seq_n_kv
        TinyNN.tnn_gguf_copy_head_slice_to_persistent_native(gguf, k_idx, @sess,
          blk.t_seq_w_k[hkv], hkv, @seq_n_kv, @seq_d_model, @seq_d_head)
        TinyNN.tnn_gguf_copy_head_slice_to_persistent_native(gguf, v_idx, @sess,
          blk.t_seq_w_v[hkv], hkv, @seq_n_kv, @seq_d_model, @seq_d_head)
        hkv = hkv + 1
      end

      if qkv_bias
        # qbias / kbias / vbias are 1-D head-sliced. We don't have a
        # dedicated head-slice loader for them; fall through and use
        # tnn_gguf_copy_head_bias_slice_to_persistent.
        qb_idx = TinyNN.tnn_gguf_find_index(gguf, prefix + ".attn_q.bias")
        kb_idx = TinyNN.tnn_gguf_find_index(gguf, prefix + ".attn_k.bias")
        vb_idx = TinyNN.tnn_gguf_find_index(gguf, prefix + ".attn_v.bias")
        hbq = 0
        while hbq < @seq_n_heads
          TinyNN.tnn_gguf_copy_head_bias_slice_to_persistent(gguf, qb_idx, @sess,
            blk.t_seq_b_q[hbq], hbq, @seq_d_head)
          hbq = hbq + 1
        end
        hbkv = 0
        while hbkv < @seq_n_kv
          TinyNN.tnn_gguf_copy_head_bias_slice_to_persistent(gguf, kb_idx, @sess,
            blk.t_seq_b_k[hbkv], hbkv, @seq_d_head)
          TinyNN.tnn_gguf_copy_head_bias_slice_to_persistent(gguf, vb_idx, @sess,
            blk.t_seq_b_v[hbkv], hbkv, @seq_d_head)
          hbkv = hbkv + 1
        end
      end

      o_idx    = TinyNN.tnn_gguf_find_index(gguf, prefix + ".attn_output.weight")
      gate_idx = TinyNN.tnn_gguf_find_index(gguf, prefix + ".ffn_gate.weight")
      up_idx   = TinyNN.tnn_gguf_find_index(gguf, prefix + ".ffn_up.weight")
      down_idx = TinyNN.tnn_gguf_find_index(gguf, prefix + ".ffn_down.weight")
      TinyNN.tnn_gguf_copy_to_persistent(gguf, o_idx,    @sess, blk.t_seq_w_o)
      TinyNN.tnn_gguf_copy_to_persistent(gguf, gate_idx, @sess, blk.t_seq_w_gate)
      TinyNN.tnn_gguf_copy_to_persistent(gguf, up_idx,   @sess, blk.t_seq_w_up)
      TinyNN.tnn_gguf_copy_to_persistent(gguf, down_idx, @sess, blk.t_seq_w_down)

      li = li + 1
    end
  end

  # Zero-init the Adam moments. m and v both start at 0 per the
  # AdamW step-0 contract. Uses the backend-side memset primitive
  # (tnn_zero_tensor) so a 1 GB Adam state doesn't materialize a
  # Mat-of-zeros in Ruby first.
  def ft_zero_init_adam(qkv_bias)
    li = 0
    while li < @seq_n_layers
      if @seq_is_kda[li]
        kblk = self.seq_kda_blocks_ffi_ref[li]
        kblk.zero_state!(@sess)
        ki2 = 0
        while ki2 < kblk.ft_weights.length
          TinyNN.tnn_zero_tensor(@sess, kblk.ft_m[ki2])
          TinyNN.tnn_zero_tensor(@sess, kblk.ft_v[ki2])
          ki2 = ki2 + 1
        end
        li = li + 1
        next
      end
      if @seq_is_mla[li]
        # No zero_state!: MLA is stateless (softmax attention over the
        # window), unlike the KDA/GDN recurrences which carry an S_0.
        mblk = self.seq_mla_blocks_ffi_ref[li]
        mi2 = 0
        while mi2 < mblk.ft_weights.length
          TinyNN.tnn_zero_tensor(@sess, mblk.ft_m[mi2])
          TinyNN.tnn_zero_tensor(@sess, mblk.ft_v[mi2])
          mi2 = mi2 + 1
        end
        li = li + 1
        next
      end
      if @seq_is_gdn[li]
        gblk = self.seq_gdn_blocks_ffi_ref[li]
        gblk.zero_state!(@sess)
        gi2 = 0
        while gi2 < gblk.ft_weights.length
          TinyNN.tnn_zero_tensor(@sess, gblk.ft_m[gi2])
          TinyNN.tnn_zero_tensor(@sess, gblk.ft_v[gi2])
          gi2 = gi2 + 1
        end
        li = li + 1
        next
      end
      blk = self.seq_blocks_ffi[li]
      i = 0
      while i < blk.ft_weights.length
        TinyNN.tnn_zero_tensor(@sess, blk.ft_m[i])
        TinyNN.tnn_zero_tensor(@sess, blk.ft_v[i])
        i = i + 1
      end
      li = li + 1
    end

    # K-series M10: the MTP block is NOT in seq_blocks_ffi, so the loop
    # above never reaches it. Its Adam moments would be allocated and
    # left UNZEROED — m/v starting from whatever the buffer held, which
    # corrupts the very first updates and does so silently. (The [d,2d]
    # projection is fine: ft_add_global_2d put it in the globals list,
    # which the globals loop zeroes.)
    if @seq_mtp == 1
      mtz = @seq_arch.seq_mtp_block
      zi = 0
      while zi < mtz.ft_weights.length
        TinyNN.tnn_zero_tensor(@sess, mtz.ft_m[zi])
        TinyNN.tnn_zero_tensor(@sess, mtz.ft_v[zi])
        zi = zi + 1
      end
    end
  end

  # P2.6 — finalize the backend weight buffers and upload the
  # per-model constants that depend on the buffers existing. This is
  # the identical head-of-tail shared by all four realize_for_* paths:
  #   1. allocate the B>1 block-causal mask in ctx_w (NULL at B=1),
  #   2. tnn_finalize_weights,
  #   3. upload the llama3 RoPE freq_factors (no-op unless :llama3),
  #   4. upload the B>1 block-causal mask values.
  # Stays a CACHE method: the finalize FFI sequencing is session-scoped.
  # Gate-covered end-to-end by smoke_projection_lens (B=1, non-llama3):
  # the two inner branches are dead under the gate but relocate verbatim.
  def finalize_weights_and_upload_constants!
    # GH#7 — block-causal attention mask for B>1. At B=1 the mask stays
    # NULL and build_seq_qhead uses diag_mask_inf + softmax. Allocated
    # in ctx_w as f32 persistent so it survives reset_for_rebuild.
    if @seq_b > 1
      tb_alloc = @seq_t * @seq_b
      @t_seq_attn_mask = TinyNN.tnn_input_2d_f32_persistent(@sess, tb_alloc, tb_alloc)
    end

    # #1449: pre-create the token-id index leaf in ctx_w (galloc-external) before
    # finalize, so galloc can't free its compute-arena slot and reuse it for the
    # loss output (-> backward get_rows reads loss bits -> wild index -> OOB
    # abort, layout-flaky on the C backend). build_forward reuses this handle;
    # it's re-uploaded each step. Positions stay in the compute ctx (the loss
    # aliases off=0 = token_ids, not positions). See tnn_input_1d_i32_persistent.
    @t_seq_token_ids = TinyNN.tnn_input_1d_i32_persistent(@sess, @seq_t * @seq_b)

    TinyNN.tnn_finalize_weights(@sess)

    # Upload llama3-style RoPE freq_factors once the backend buffer
    # exists. Per-model constant; never re-uploaded.
    if @seq_rope_scaling.kind == :llama3
      ff = Toy::RopeScaling.compute_llama3_freq_factors(
        @seq_d_head, @seq_rope_base,
        @seq_rope_scaling.orig_max_pos, @seq_rope_scaling.factor,
        @seq_rope_scaling.low_freq_factor, @seq_rope_scaling.high_freq_factor)
      TinyNN.tnn_upload_from_float_array(@sess, @t_seq_rope_freq_factors, ff, ff.length)
    end

    if @seq_b > 1
      upload_block_causal_mask!
    end
  end

  # P2.6 — the identical tail-of-tail shared by all four realize_for_*
  # paths: build the forward graph in the current ctx, realize it, and
  # flip @seq_realized. Stays a CACHE method (build_forward_in_current_ctx
  # is the cache->arch wrapper; tnn_realize is session-scoped).
  # Gate-covered by smoke_projection_lens via realize_for_random_init.
  def build_and_realize!
    build_forward_in_current_ctx
    TinyNN.tnn_realize(@sess, @t_seq_logits)
    @seq_realized = true
  end

  # Build the forward graph in the CURRENT compute context. Used both
  # from realize_for_mmap (first realize) and after tnn_reset_for_rebuild
  # (e.g. when switching from inference to training, which needs the
  # forward + loss + backward + opt_step all in one rebuilt ctx).
  # Stores the per-graph tensor handles back on `self`.
  # P2.5 — thin wrapper around Toy::LLM::Archs::LlamaArch#build_forward.
  # Allocates the per-graph INPUT handles (token_ids, positions) — which
  # stay CACHE-owned graph I/O, read by forward() and the uploaders —
  # then hands the realize-set rope_cfg / donor_d_in onto the arch and
  # calls the lifted orchestration. The three per-graph OUTPUT handles
  # come back in a LlamaArchForwardOut and are spread onto the cache's
  # own ivars so every downstream reader (@t_seq_logits accessor,
  # build_training_step CE-loss consumer, examples/06 fcache.t_seq_logits)
  # is untouched.
  def build_forward_in_current_ctx
    # GH#7 — at B=1, @seq_t * @seq_b == @seq_t (legacy behaviour).
    # At B>1, the layout is flat [T*B]: per-batch positions cycle
    # 0..T-1 (the caller-built positions array is responsible for
    # that ordering); RoPE applies per-batch positional encoding
    # because rope_ext reads positions[k] for each ne[2] slot.
    tb = @seq_t * @seq_b
    # #1449: token_ids is pre-created persistent (ctx_w) in finalize and survives
    # reset_for_rebuild — reuse it. Only allocate here as a fallback for any path
    # that builds the forward without the finalize pre-creation (then it lands in
    # the compute ctx, the legacy behaviour).
    @t_seq_token_ids = TinyNN.tnn_input_1d_i32(@sess, tb) if @t_seq_token_ids == TinyNN.tnn_null_ptr
    @t_seq_positions = TinyNN.tnn_input_1d_i32_ctx(@sess, tb)

    # The arch reads seq_rope_cfg / seq_donor_d_in off itself; the cache
    # rebuilds rope_cfg and sets donor_d_in in each realize prologue, so
    # mirror the realize-set values onto the arch right before the call.
    @seq_arch.seq_rope_cfg   = @seq_rope_cfg
    @seq_arch.seq_donor_d_in = @seq_donor_d_in
    # K-series M10: same reason — mtp_init runs BEFORE realize, and the
    # arch's own initialize resets seq_mtp_on to 0, so the flag has to be
    # re-mirrored here or the module is silently never built (which is
    # exactly how it presented: a null projection into mul_mat).
    @seq_arch.seq_mtp_on     = @seq_mtp
    # M10 (b): lambda now shapes the GRAPH (grad_scale nodes in the arch),
    # not just a scalar on the loss root, so it has to be mirrored here
    # for the same reason seq_mtp_on does — otherwise every run would
    # build at the arch default of 1.0 and --mtp-lambda would be inert.
    @seq_arch.seq_mtp_lambda = @seq_mtp_lambda

    out = @seq_arch.build_forward(
      @sess, @t_seq_token_ids, @t_seq_positions, @t_seq_rope_freq_factors,
      @t_seq_attn_mask, @seq_rms_eps, @seq_d_head, @seq_n_kv, @seq_n_heads,
      @seq_group_size, @seq_has_qkv_bias, @seq_weight_dtype,
      @seq_lora_q_enabled, @seq_t, @seq_b, @seq_n_layers,
      @seq_has_untied_output)

    @t_seq_x_embed = out.t_seq_x_embed
    @t_seq_x_final = out.t_seq_x_final
    @t_seq_logits  = out.t_seq_logits
  end

  # M3 step 3 — rebuild the session graph as forward + CE loss + backward
  # + AdamW opt_step over every LoRA pair. After this, callers upload
  # token IDs + positions + labels (one-hot vocab×T) + hp vector and
  # call tnn_compute_backward to get one training step over the whole
  # T-position sequence.
  #
  # Returns the (loss_tensor, labels_tensor, hp_tensor) triple.
  def build_training_step
    if !@seq_full_finetune_enabled && (!@seq_lora_q_enabled || !@seq_lora_q_adamw_enabled)
      puts "build_training_step: requires enable_lora_q! AND enable_lora_q_adamw!  (or enable_full_finetune!)"
      return nil
    end
    TinyNN.tnn_reset_for_rebuild(@sess)
    build_forward_in_current_ctx

    # Label tensor: same shape as logits, ggml ne=[vocab, T*B]. Our
    # wrapper takes (rows, cols) and emits ggml(cols, rows), so pass
    # (T*B, vocab) here to get ne=[vocab, T*B]. One-hot per ne1-column
    # (i.e. per (batch, position) slot). At B=1, identical to legacy.
    t_labels = TinyNN.tnn_input_2d_f32(@sess, @seq_t * @seq_b, @seq_vocab_size)
    # Hyper-params vector for AdamW: alpha, beta1, beta2, eps, wd, beta1h, beta2h.
    t_hp = TinyNN.tnn_input_1d_f32(@sess, 7)

    # CE loss over all T columns. ggml_cross_entropy_loss returns the
    # mean over columns — masking is a follow-up (would zero specific
    # columns in labels before this op).
    t_loss = TinyNN.tnn_cross_entropy_loss(@sess, @t_seq_logits, t_labels)
    TinyNN.tnn_set_output(t_loss)

    # K-series M10: the SECOND CE root, against t+2. Backward runs on
    # total = main + lambda * mtp, but the tensor the caller reads back
    # as "the loss" stays the MAIN CE — so a --mtp curve is directly
    # comparable to one without it, instead of being offset by a term
    # that measures a different prediction. Patched IDENTICALLY into
    # build_training_step and build_training_step_franken; those two
    # must emit the same graph under an empty policy (the parity gate).
    if @seq_mtp == 1
      @t_seq_mtp_labels = TinyNN.tnn_input_2d_f32(@sess, @seq_t * @seq_b, @seq_vocab_size)
      @t_seq_mtp_loss = TinyNN.tnn_cross_entropy_loss(@sess, @seq_arch.t_seq_mtp_logits, @t_seq_mtp_labels)
      TinyNN.tnn_set_output(@t_seq_mtp_loss)
      # M10 (b): the second root is UNSCALED. lambda is applied inside
      # the arch as a gradient scale on every quantity the MTP branch
      # borrows from the backbone, so MTP-private weights train at full
      # strength while lambda controls only the coupling. Scaling HERE
      # would multiply the MTP-private gradient too, which is exactly
      # what made lambda=0 freeze the module outright.
      t_total = TinyNN.tnn_add(@sess, t_loss, @t_seq_mtp_loss)
      TinyNN.tnn_set_output(t_total)
      TinyNN.tnn_set_loss(t_total)
      TinyNN.tnn_add_to_graph(@sess, t_loss)
      TinyNN.tnn_build_forward_only(@sess, t_total)
    else
      TinyNN.tnn_set_loss(t_loss)
      TinyNN.tnn_build_forward_only(@sess, t_loss)
    end
    TinyNN.tnn_build_backward(@sess)

    if @seq_full_finetune_enabled
      # F3 — emit opt_step_adamw for every recorded (weight, m, v)
      # triple. The arrays are populated in realize_for_full_finetune.
      li = 0
      while li < @seq_n_layers
        if @seq_is_kda[li]
          kblk = self.seq_kda_blocks_ffi_ref[li]
          wk = 0
          while wk < kblk.ft_weights.length
            twk = kblk.ft_weights[wk]
            tgk = TinyNN.tnn_tensor_grad(@sess, twk)
            tok = TinyNN.tnn_opt_step_adamw(@sess, twk, tgk,
                                             kblk.ft_m[wk], kblk.ft_v[wk], t_hp)
            TinyNN.tnn_extend_backward_graph(@sess, tok)
            wk = wk + 1
          end
          li = li + 1
          next
        end
        if @seq_is_mla[li]
          # AdamW directly, like the KDA/GDN arms above: --optimizer
          # muon reaches KIND_ATTENTION layers only (emit_opt_step is
          # wired on the franken attention path). Pre-existing scope,
          # stated rather than silently widened.
          mblk = self.seq_mla_blocks_ffi_ref[li]
          wm = 0
          while wm < mblk.ft_weights.length
            twm = mblk.ft_weights[wm]
            tgm = TinyNN.tnn_tensor_grad(@sess, twm)
            tom = TinyNN.tnn_opt_step_adamw(@sess, twm, tgm,
                                             mblk.ft_m[wm], mblk.ft_v[wm], t_hp)
            TinyNN.tnn_extend_backward_graph(@sess, tom)
            wm = wm + 1
          end
          li = li + 1
          next
        end
        if @seq_is_gdn[li]
          gblk = self.seq_gdn_blocks_ffi_ref[li]
          wg = 0
          while wg < gblk.ft_weights.length
            twg = gblk.ft_weights[wg]
            tgg = TinyNN.tnn_tensor_grad(@sess, twg)
            tog = TinyNN.tnn_opt_step_adamw(@sess, twg, tgg,
                                             gblk.ft_m[wg], gblk.ft_v[wg], t_hp)
            TinyNN.tnn_extend_backward_graph(@sess, tog)
            wg = wg + 1
          end
          li = li + 1
          next
        end
        blk = self.seq_blocks_ffi[li]
        wi = 0
        while wi < blk.ft_weights.length
          tw = blk.ft_weights[wi]
          tg = TinyNN.tnn_tensor_grad(@sess, tw)
          to = TinyNN.tnn_opt_step_adamw(@sess, tw, tg,
                                          blk.ft_m[wi], blk.ft_v[wi], t_hp)
          TinyNN.tnn_extend_backward_graph(@sess, to)
          wi = wi + 1
        end
        li = li + 1
      end
      # K-series M10: the MTP block's own weights. The [d,2d] projection
      # rides the globals list below (ft_add_global_2d registered it);
      # the block's tensors are its own and need this arm, or the module
      # would build, produce a loss, and never actually train.
      if @seq_mtp == 1
        mtb = @seq_arch.seq_mtp_block
        wmo = 0
        while wmo < mtb.ft_weights.length
          twm = mtb.ft_weights[wmo]
          tgm = TinyNN.tnn_tensor_grad(@sess, twm)
          tom = TinyNN.tnn_opt_step_adamw(@sess, twm, tgm, mtb.ft_m[wmo], mtb.ft_v[wmo], t_hp)
          TinyNN.tnn_extend_backward_graph(@sess, tom)
          wmo = wmo + 1
        end
      end
      # Globals (token_embed, final-norm, optional untied output).
      gi = 0
      while gi < @ft_globals_weights.length
        tw = @ft_globals_weights[gi]
        tg = TinyNN.tnn_tensor_grad(@sess, tw)
        to = TinyNN.tnn_opt_step_adamw(@sess, tw, tg,
                                        @ft_globals_m[gi], @ft_globals_v[gi], t_hp)
        TinyNN.tnn_extend_backward_graph(@sess, to)
        gi = gi + 1
      end
    else
      # LoRA-only training (M3 step 3). One opt_step_adamw per LoRA-A
      # and per LoRA-B tensor; thread each through extend_backward_graph
      # so sched sees the writes.
      li = 0
      while li < @seq_n_layers
        blk = self.seq_blocks_ffi[li]
        hq = 0
        while hq < @seq_n_heads
          t_a       = blk.t_seq_w_lora_a_q[hq]
          t_b       = blk.t_seq_w_lora_b_q[hq]
          t_grad_a  = TinyNN.tnn_tensor_grad(@sess, t_a)
          t_grad_b  = TinyNN.tnn_tensor_grad(@sess, t_b)
          t_opt_a   = TinyNN.tnn_opt_step_adamw(@sess, t_a, t_grad_a,
                                                  blk.t_seq_w_lora_a_q_m[hq],
                                                  blk.t_seq_w_lora_a_q_v[hq], t_hp)
          t_opt_b   = TinyNN.tnn_opt_step_adamw(@sess, t_b, t_grad_b,
                                                  blk.t_seq_w_lora_b_q_m[hq],
                                                  blk.t_seq_w_lora_b_q_v[hq], t_hp)
          TinyNN.tnn_extend_backward_graph(@sess, t_opt_a)
          TinyNN.tnn_extend_backward_graph(@sess, t_opt_b)
          hq = hq + 1
        end
        li = li + 1
      end
    end

    # Pin every node in graph_b before sched-alloc — workaround for the
    # ggml-cpu sched-aliasing bug on long backward chains (documented in
    # project_cpu_cuda_lora_train_divergence_2026_05_21). Memory cost
    # grows roughly with node count; fine for SmolLM2-135M at T<=64.
    TinyNN.tnn_pin_all_graph_b_nodes(@sess)
    TinyNN.tnn_realize_backward(@sess)
    [t_loss, t_labels, t_hp]
  end

  # toy#109 P2/#112 — franken telemetry handles, recorded by
  # build_training_step_franken for align-event consumers (parallel
  # arrays: dfa grad node, chain grad-acc, layer index, ft weight index).
  # toy#117 — re-upload every B matrix (deterministic regeneration from
  # the recorded seeds). B leaves were once-uploaded graph inputs and the
  # sched scratch-reused their buffers: B[0]'s norm read 3.2e5 after ONE
  # compute and hit Infinity by step 5, feeding the exponential DFA-grad
  # explosion in the mask arms (the labels contract — inputs live only if
  # re-uploaded per step — applies to EVERY graph-input leaf). Call at
  # the top of each training step, before compute.
  # toy#129 item 2: record the credit-assignment policy BEFORE
  # realize_for_random_init so the alloc-time param marking can skip
  # the dfa qkv weights (no chain grad-acc, no backward expansion).
  # Call ONLY for a no-shadow run; not calling = shadow-shaped legacy.
  # toy#136 (K1): activation + positional-encoding axes. Call BEFORE
  # build (the seq_rope_cfg pattern — the arch field feeds the shared
  # block ctx). 0/0 = the byte-gated defaults.
  def seq_act_init(v)
    @seq_arch.seq_act = v
    0
  end

  def seq_nope_init(v)
    @seq_arch.seq_nope = v
    @seq_nope_flag = v
    0
  end

  # toy#136: read by the recipe's step! — under NoPE the positions
  # tensor never enters the graph, so it has NO backing buffer and an
  # upload would hit unallocated storage (found the hard way).
  def seq_nope_flag
    @seq_nope_flag
  end

  def franken_no_shadow_init(policy)
    @seq_franken_noshadow_policy = [0]; @seq_franken_noshadow_policy.pop
    pi = 0
    while pi < policy.length
      @seq_franken_noshadow_policy.push(policy[pi])
      pi = pi + 1
    end
    0
  end

  def franken_refresh_b!
    bi = 0
    while bi < @franken_b_handles.length
      nb = @seq_d_head * @seq_vocab_size
      sig = Toy::Train::DfaB.sigma_for(@franken_b_scale_rec, @seq_vocab_size,
                                       @seq_d_head, @franken_b_sigma_rec)
      TinyNN.tnn_upload_from_float_array(@sess, @franken_b_handles[bi],
        Toy::Train::DfaB.fill(nb, @franken_b_seeds_rec[bi], @franken_b_dist_rec, sig), nb)
      bi = bi + 1
    end
    0
  end

  attr_accessor :franken_align_grads, :franken_align_accs,
                :franken_align_lis, :franken_align_wis,
                :franken_mask_tensors, :franken_mask_lis, :franken_mask_wis,
                :franken_b_handles

  # toy#109 P2 — the FrankenModel training step: build_training_step's
  # full-finetune arm with a per-layer CREDIT-ASSIGNMENT policy. policy
  # is a flat IntArray (0 = :chain, 1 = :dfa); layers beyond its length
  # are chain. With an EMPTY/all-zero policy this method must emit the
  # IDENTICAL graph to build_training_step (the parity gate pins it —
  # every op below is verbatim from the original except inside the
  # dfa-classified branch).
  #
  # :dfa scope (v1, per-matmul identity-boundary — design doc §4c): the
  # ATTENTION Q/K/V per-head weights only, whose shared input tap
  # (blk.tap_attn_norm, the GH#15 CKA tap) the forward already records.
  # o/FFN/norms/globals stay :chain in every policy. Each dfa weight
  # gets a fixed random B (ne=[vocab, d_head]) as a GRAPH-INPUT tensor
  # uploaded ONCE post-realize (read-only leaf, same persistence class
  # as labels between uploads — no realize-path changes needed);
  # δ = B·e with e = (softmax(logits) − labels)/(T·B); grad = tapᵀ·δ
  # appended as pure forward ops (P0 mechanics) and fed to
  # opt_step_adamw in place of the autodiff grad-acc. The autodiff
  # param set is UNCHANGED (shadow-shaped build; the cut primitive is a
  # later phase) — chain grads for dfa weights exist but are unused.
  #
  # b_dist/b_scale/b_sigma are Toy::Train::DfaB codes; b_seed feeds
  # per-weight streams (seed + li*1000 + wi) so re-policying one layer
  # never reshuffles another's B. qkv_bias + dfa is unsupported (fail
  # loud); dfa on a non-attention (GDN) layer fails loud.
  def build_training_step_franken(policy, b_seed, b_dist, b_scale, b_sigma,
                                  mix_alpha, mask_tau, no_shadow)
    if !@seq_full_finetune_enabled
      puts "build_training_step_franken: requires realize_for_random_init (full-finetune arm only)"
      return nil
    end
    # toy#129 item 2: the no-shadow contract is a TWO-phase agreement —
    # alloc skipped the dfa qkv param flags (franken_no_shadow_init
    # before realize) and this build late-params them. A half-applied
    # state silently rebuilds the shadow, so both directions fail loud.
    if no_shadow == 1 && @seq_franken_noshadow_policy.length == 0
      raise "build_training_step_franken: no_shadow=1 but franken_no_shadow_init was not called before realize (alloc marked every param — the shadow would silently persist)"
    end
    if no_shadow == 0 && @seq_franken_noshadow_policy.length > 0
      raise "build_training_step_franken: shadow build requested but the alloc ran no-shadow (dfa qkv weights lack param flags)"
    end
    TinyNN.tnn_reset_for_rebuild(@sess)
    build_forward_in_current_ctx

    t_labels = TinyNN.tnn_input_2d_f32(@sess, @seq_t * @seq_b, @seq_vocab_size)
    t_hp = TinyNN.tnn_input_1d_f32(@sess, 7)

    t_loss = TinyNN.tnn_cross_entropy_loss(@sess, @t_seq_logits, t_labels)
    TinyNN.tnn_set_output(t_loss)

    # K-series M10: the SECOND CE root, against t+2. Backward runs on
    # total = main + lambda * mtp, but the tensor the caller reads back
    # as "the loss" stays the MAIN CE — so a --mtp curve is directly
    # comparable to one without it, instead of being offset by a term
    # that measures a different prediction. Patched IDENTICALLY into
    # build_training_step and build_training_step_franken; those two
    # must emit the same graph under an empty policy (the parity gate).
    if @seq_mtp == 1
      @t_seq_mtp_labels = TinyNN.tnn_input_2d_f32(@sess, @seq_t * @seq_b, @seq_vocab_size)
      @t_seq_mtp_loss = TinyNN.tnn_cross_entropy_loss(@sess, @seq_arch.t_seq_mtp_logits, @t_seq_mtp_labels)
      TinyNN.tnn_set_output(@t_seq_mtp_loss)
      # M10 (b): the second root is UNSCALED. lambda is applied inside
      # the arch as a gradient scale on every quantity the MTP branch
      # borrows from the backbone, so MTP-private weights train at full
      # strength while lambda controls only the coupling. Scaling HERE
      # would multiply the MTP-private gradient too, which is exactly
      # what made lambda=0 freeze the module outright.
      t_total = TinyNN.tnn_add(@sess, t_loss, @t_seq_mtp_loss)
      TinyNN.tnn_set_output(t_total)
      TinyNN.tnn_set_loss(t_total)
      TinyNN.tnn_add_to_graph(@sess, t_loss)
      TinyNN.tnn_build_forward_only(@sess, t_total)
    else
      TinyNN.tnn_set_loss(t_loss)
      TinyNN.tnn_build_forward_only(@sess, t_loss)
    end
    TinyNN.tnn_build_backward(@sess)

    any_dfa = false
    pi = 0
    while pi < policy.length
      if policy[pi] > 0
        any_dfa = true
      end
      pi = pi + 1
    end

    t_e = TinyNN.tnn_null_ptr
    if any_dfa
      if @seq_has_qkv_bias
        raise "build_training_step_franken: qkv_bias + :dfa unsupported (v1)"
      end
      t_p = TinyNN.tnn_softmax(@sess, @t_seq_logits)
      t_e = TinyNN.tnn_scale(@sess, TinyNN.tnn_sub(@sess, t_p, t_labels),
                             1.0 / (@seq_t * @seq_b).to_f)
    end

    b_handles = [TinyNN.tnn_null_ptr]; b_handles.pop
    b_seeds   = [0]; b_seeds.pop
    @franken_b_handles = b_handles
    @franken_b_seeds_rec = b_seeds
    @franken_b_dist_rec  = b_dist
    @franken_b_scale_rec = b_scale
    @franken_b_sigma_rec = b_sigma
    @franken_align_grads = [TinyNN.tnn_null_ptr]; @franken_align_grads.pop
    @franken_align_accs  = [TinyNN.tnn_null_ptr]; @franken_align_accs.pop
    @franken_align_lis   = [0]; @franken_align_lis.pop
    @franken_align_wis   = [0]; @franken_align_wis.pop
    @franken_mask_tensors = [TinyNN.tnn_null_ptr]; @franken_mask_tensors.pop
    @franken_mask_lis     = [0]; @franken_mask_lis.pop
    @franken_mask_wis     = [0]; @franken_mask_wis.pop

    tb = @seq_t * @seq_b
    li = 0
    while li < @seq_n_layers
      blk = self.seq_blocks_ffi[li]
      mode = 0
      if li < policy.length
        mode = policy[li]
      end
      if mode > 0 && self.seq_arch.seq_layer_kinds[li] != Toy::LLM::Archs::LayerSpec::KIND_ATTENTION
        raise "build_training_step_franken: :dfa on non-attention layer " + li.to_s
      end
      if @seq_is_kda[li]
        kblk = self.seq_kda_blocks_ffi_ref[li]
        wk = 0
        while wk < kblk.ft_weights.length
          twk = kblk.ft_weights[wk]
          tgk = TinyNN.tnn_tensor_grad(@sess, twk)
          tok = TinyNN.tnn_opt_step_adamw(@sess, twk, tgk,
                                           kblk.ft_m[wk], kblk.ft_v[wk], t_hp)
          TinyNN.tnn_extend_backward_graph(@sess, tok)
          wk = wk + 1
        end
        li = li + 1
        next
      end
      if @seq_is_mla[li]
        mblk = self.seq_mla_blocks_ffi_ref[li]
        wm = 0
        while wm < mblk.ft_weights.length
          twm = mblk.ft_weights[wm]
          tgm = TinyNN.tnn_tensor_grad(@sess, twm)
          tom = TinyNN.tnn_opt_step_adamw(@sess, twm, tgm,
                                           mblk.ft_m[wm], mblk.ft_v[wm], t_hp)
          TinyNN.tnn_extend_backward_graph(@sess, tom)
          wm = wm + 1
        end
        li = li + 1
        next
      end
      if @seq_is_gdn[li]
        gblk = self.seq_gdn_blocks_ffi_ref[li]
        wg = 0
        while wg < gblk.ft_weights.length
          twg = gblk.ft_weights[wg]
          tgg = TinyNN.tnn_tensor_grad(@sess, twg)
          tog = TinyNN.tnn_opt_step_adamw(@sess, twg, tgg,
                                           gblk.ft_m[wg], gblk.ft_v[wg], t_hp)
          TinyNN.tnn_extend_backward_graph(@sess, tog)
          wg = wg + 1
        end
        li = li + 1
        next
      end
      nh  = @seq_n_heads
      nkv = @seq_n_kv
      wi = 0
      while wi < blk.ft_weights.length
        tw = blk.ft_weights[wi]
        # ft layout (alloc_trainable_f32_weights!): [rn1, rn2,
        # q×n_heads, (k,v)×n_kv interleaved, o, gate, up, down].
        is_qkv = wi >= 2 && wi < 2 + nh + 2 * nkv
        if mode > 0 && is_qkv
          if no_shadow == 1 && mode > 1
            raise "build_training_step_franken: no_shadow supports modes 0/1 only (mode " + mode.to_s + " reads the chain grad-acc)"
          end
          t_b = TinyNN.tnn_input_2d_f32(@sess, @seq_d_head, @seq_vocab_size)
          TinyNN.tnn_set_output(t_b)
          t_delta = TinyNN.tnn_matmul(@sess, t_b, t_e)
          t_tap_t = TinyNN.tnn_cont_2d(@sess,
                      TinyNN.tnn_transpose(@sess, blk.tap_attn_norm), tb, @seq_d_model)
          t_del_t = TinyNN.tnn_cont_2d(@sess,
                      TinyNN.tnn_transpose(@sess, t_delta), tb, @seq_d_head)
          t_gd = TinyNN.tnn_matmul(@sess, t_tap_t, t_del_t)
          t_g = t_gd
          # P3 combiners (design §4b): mix(alpha) blends the chain
          # grad-acc; mask gates one signal by the other's magnitude
          # via the smooth near-hard gate 0.5(1+tanh(1e4(|g|-tau))) —
          # tau=-1 saturates to exactly 1.0 (== pure source signal),
          # huge tau to exactly 0.0.
          if mode == 2
            t_acc = TinyNN.tnn_tensor_grad(@sess, tw)
            t_g = TinyNN.tnn_add(@sess,
                    TinyNN.tnn_scale(@sess, t_acc, mix_alpha),
                    TinyNN.tnn_scale(@sess, t_gd, 1.0 - mix_alpha))
          elsif mode == 3
            t_acc = TinyNN.tnn_tensor_grad(@sess, tw)
            t_ab  = TinyNN.tnn_sqrt(@sess, TinyNN.tnn_mul(@sess, t_acc, t_acc))
            t_y   = TinyNN.tnn_scale_bias(@sess, t_ab, 1.0, 0.0 - mask_tau)
            t_mk  = TinyNN.tnn_scale_bias(@sess,
                      TinyNN.tnn_tanh(@sess, TinyNN.tnn_scale(@sess, t_y, 10000.0)), 0.5, 0.5)
            t_g = TinyNN.tnn_mul(@sess, t_gd, t_mk)
            TinyNN.tnn_set_output(t_mk)
            @franken_mask_tensors.push(t_mk)
            @franken_mask_lis.push(li)
            @franken_mask_wis.push(wi)
          elsif mode == 4
            t_acc = TinyNN.tnn_tensor_grad(@sess, tw)
            t_ab  = TinyNN.tnn_sqrt(@sess, TinyNN.tnn_mul(@sess, t_gd, t_gd))
            t_y   = TinyNN.tnn_scale_bias(@sess, t_ab, 1.0, 0.0 - mask_tau)
            t_mk  = TinyNN.tnn_scale_bias(@sess,
                      TinyNN.tnn_tanh(@sess, TinyNN.tnn_scale(@sess, t_y, 10000.0)), 0.5, 0.5)
            t_g = TinyNN.tnn_mul(@sess, t_acc, t_mk)
            TinyNN.tnn_set_output(t_mk)
            @franken_mask_tensors.push(t_mk)
            @franken_mask_lis.push(li)
            @franken_mask_wis.push(wi)
          end
          TinyNN.tnn_set_output(t_g)
          if no_shadow == 1
            # late-param (the MoE-top1 idiom): the flag lands AFTER
            # build_backward so the walker never expanded this weight;
            # opt_step_adamw's PARAM assert is satisfied.
            TinyNN.tnn_set_param(tw)
          end
          emit_opt_step(tw, t_g, blk.ft_m[wi], blk.ft_v[wi], t_hp)
          b_handles.push(t_b)
          b_seeds.push(b_seed + li * 1000 + wi)
          if no_shadow == 0
            @franken_align_grads.push(t_g)
            # PIN the shadow acc: for pure-:dfa weights it has NO consumer
            # (the opt reads t_g), so sched aliases its slot and the align
            # download reads zeros (P0 pin-read-backs lesson, engine form).
            t_acc_rec = TinyNN.tnn_tensor_grad(@sess, tw)
            TinyNN.tnn_set_output(t_acc_rec)
            @franken_align_accs.push(t_acc_rec)
            @franken_align_lis.push(li)
            @franken_align_wis.push(wi)
          end
        else
          tg = TinyNN.tnn_tensor_grad(@sess, tw)
          emit_opt_step(tw, tg, blk.ft_m[wi], blk.ft_v[wi], t_hp)
        end
        wi = wi + 1
      end
      li = li + 1
    end
    gi = 0
    while gi < @ft_globals_weights.length
      tw = @ft_globals_weights[gi]
      tg = TinyNN.tnn_tensor_grad(@sess, tw)
      to = TinyNN.tnn_opt_step_adamw(@sess, tw, tg,
                                      @ft_globals_m[gi], @ft_globals_v[gi], t_hp)
      TinyNN.tnn_extend_backward_graph(@sess, to)
      gi = gi + 1
    end

    # Loud wiring summary (never-mask: an empty dfa count under a
    # non-empty policy is a bug, not a quiet fallback).
    puts "franken: policy_len=" + policy.length.to_s +
         " dfa_wired=" + b_handles.length.to_s +
         " shadow=" + (no_shadow == 1 ? "off" : "on")

    TinyNN.tnn_pin_all_graph_b_nodes(@sess)
    TinyNN.tnn_realize_backward(@sess)

    # B uploads: buffers exist only after sched alloc. Read-only leaves —
    # uploaded once, stable across computes (labels-class persistence).
    bi = 0
    while bi < b_handles.length
      nb = @seq_d_head * @seq_vocab_size
      sig = Toy::Train::DfaB.sigma_for(b_scale, @seq_vocab_size, @seq_d_head, b_sigma)
      TinyNN.tnn_upload_from_float_array(@sess, b_handles[bi],
        Toy::Train::DfaB.fill(nb, b_seeds[bi], b_dist, sig), nb)
      bi = bi + 1
    end
    [t_loss, t_labels, t_hp]
  end

  # GGUF type → bytes-per-row stride for per-head slicing. Mirrors the
  # SmolLM2KVFFICache helper of the same name. F32=0, Q8_0=8.
  def head_nbytes(ggml_type, d_head, d_model)
    if ggml_type == 0
      d_head * d_model * 4
    elsif ggml_type == 8
      d_head * (d_model / 32) * 34
    else
      0
    end
  end

  # Run one forward pass. `ids` and `positions` are length-T Int arrays.
  # Returns the t_seq_logits handle; caller downloads via download_row_major
  # against (vocab, T) shape.
  def forward(ids, positions)
    TinyNN.upload_int_array(@sess, @t_seq_token_ids, ids)
    TinyNN.upload_int_array(@sess, @t_seq_positions, positions)
    TinyNN.tnn_compute(@sess)
    @t_seq_logits
  end

  # Seed LoRA-A with a small Gaussian and LoRA-B with zero — the
  # standard init makes the adapter a no-op at step 0 (forward output
  # equals the base model). Mirror of SmolLM2KVFFICache#upload_lora_q_init!.
  def upload_lora_q_init!(seed, init_scale)
    if !@seq_lora_q_enabled; return; end
    s = seed
    m_a = Mat.new(@seq_lora_q_rank, @seq_d_model)
    m_b = Mat.new(@seq_d_head, @seq_lora_q_rank)
    i_b = 0
    while i_b < @seq_d_head * @seq_lora_q_rank
      m_b.flat[i_b] = 0.0
      i_b = i_b + 1
    end
    li = 0
    while li < @seq_n_layers
      blk = self.seq_blocks_ffi[li]
      hq = 0
      while hq < @seq_n_heads
        ii = 0
        while ii < @seq_lora_q_rank * @seq_d_model
          s = (s * 1103515245 + 12345) & 0x7FFFFFFF
          u1 = (s.to_f + 1.0) / 2147483648.0
          s = (s * 1103515245 + 12345) & 0x7FFFFFFF
          u2 = (s.to_f + 1.0) / 2147483648.0
          m_a.flat[ii] = init_scale * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
          ii = ii + 1
        end
        TinyNN.upload_row_major(@sess, blk.t_seq_w_lora_a_q[hq], m_a)
        TinyNN.upload_row_major(@sess, blk.t_seq_w_lora_b_q[hq], m_b)
        hq = hq + 1
      end
      li = li + 1
    end
  end
end
end; end; end
