# lib/toy/llm/archs/llama_arch.rb — L3 arch: the llama/qwen-family
# sequence-mode forward orchestration (token-embed get_rows → optional
# projection-lens matmul → stacked L2 transformer blocks → final RMSNorm
# → tied/untied logits matmul).
#
# Extracted from lib/llama_seq_forward_ffi.rb (P2.5). This is the
# MINIMAL faithful lift of Toy::LLM::Engine::LlamaSeqEngine#build_forward_in_current_ctx:
# the orchestration body is moved VERBATIM (op order unchanged →
# bit-identical graph). The arch now OWNS the arch-level persistent
# tensor handles (token_embed, final_norm_gamma, output, w_proj, the
# blocks array); the cache's four realize paths still ALLOCATE and
# ASSIGN them through cache delegators (exactly the L2 pattern: the
# block owns its weights, the cache realize paths allocate+assign them).
#
# DIVERGENCE from archs/README sketch: build_forward takes NO cfg and
# does NO per_layer/with_hyper/build_initial_state/learned-position-
# embedding. Those are forward-looking sketch only. Reality:
# positions/RoPE live on the L2 block via the shared read-only
# TransformerBlockCtx; one ctx is shared across all blocks; the LM head
# is tied/untied via @seq_has_untied_output; an optional E2.3
# projection-lens matmul sits between embed and blocks. This is the
# minimal faithful lift of Toy::LLM::Engine::LlamaSeqEngine#build_forward_in_current_ctx
# (P2.5, orchestration only).
#
# Spinel hygiene: NEVER Struct.new (landmine #16 / matz/spinel#1043) —
# LlamaArchForwardOut is a hand-written plain class with an explicit
# positional ctor; LlamaArch#initialize takes NO args and has NO
# default-arg ctor (default-arg poisoning, landmine #4). No
# Card/step_bind/FFI :str args at class load (step_bind :str landmine
# 2026-05-28). Member names keep the VERBATIM `@t_seq_*` / `@seq_*`
# prefixes from the cache for type-isolation and so cache-side
# realize/train/decode walkers keep working by accessor name.
#
# This file does NOT `require_relative "tinynn"`: the loading module
# (lib/llama_seq_forward_ffi.rb) already loads the correct backend's
# TinyNN before requiring this arch, exactly like the L1 primitives and
# the L2 block. The mirror generator picks the backend via the
# monolith's require rewrite.

module Toy; module LLM; module Archs
  # Carries the three per-graph OUTPUT handles back to the cache caller,
  # which spreads them onto its OWN ivars so every existing downstream
  # reader (@t_seq_logits accessor, build_training_step CE-loss consumer,
  # examples/06 fcache.t_seq_logits) is untouched.
  #
  # Hand-written plain class with a positional ctor — NEVER Struct.new
  # (landmine #16 / matz/spinel#1043): a Struct's synthesized accessors
  # would unify across modules and mis-compile unrelated callers. The
  # member names reuse the cache's `t_seq_*` prefixes (type-isolated, no
  # collision, same rationale as TransformerBlockCtx). Carries values,
  # no behavior.
  class LlamaArchForwardOut
    attr_accessor :t_seq_x_embed, :t_seq_x_final, :t_seq_logits

    def initialize(t_seq_x_embed, t_seq_x_final, t_seq_logits)
      @t_seq_x_embed = t_seq_x_embed
      @t_seq_x_final = t_seq_x_final
      @t_seq_logits  = t_seq_logits
    end
  end

  # The llama-family sequence-mode arch. Owns the arch-level persistent
  # handles (the cache realize paths allocate+assign them via cache
  # delegators). Field names are UNCHANGED from the former cache ivars so
  # the cache-side realize / train / decode / tap walkers keep working by
  # accessor name.
  class LlamaArch
    attr_accessor :t_seq_token_embed, :t_seq_final_norm_gamma, :t_seq_output,
                  :t_seq_w_proj, :seq_blocks_ffi,
                  # Phase 3 — per-layer descriptor array, parallel to
                  # seq_blocks_ffi (same length == n_layers).
                  :seq_layer_specs,
                  # Phase 5 — the dispatch key is a plain INT array (one kind per
                  # layer), NOT LayerSpec.kind reads: constructing/mutating
                  # LayerSpec objects on a realize path trips a Spinel codegen
                  # miscompile (corrupts the token-id finalize). Mutating a plain
                  # int array element is proven-safe. build_forward dispatches on
                  # this; LayerSpec stays the descriptor type/constants home.
                  :seq_layer_kinds,
                  # Phase 5 — parallel GDN-block array (same length; entry is a
                  # GDNBlock at KIND_GDN positions, null elsewhere). The KIND_GDN
                  # dispatch arm calls into THIS array — a concrete typed call,
                  # so the seam stays monomorphic per call site.
                  :seq_gdn_blocks_ffi,
                  # toy#137 K2b: the parallel KDA array — same
                  # flat-int-kind dispatch discipline as GDN (one
                  # concrete typed block class per arm).
                  :seq_kda_blocks_ffi,
                  # Orchestration-gating carriers — bare cache ivars with
                  # no accessor before P2.5. The lens-branch guard reads
                  # seq_donor_d_in; the shared ctx reads seq_rope_cfg.
                  # The cache wrapper sets both from the realize-set
                  # values before build_forward runs.
                  :seq_donor_d_in, :seq_rope_cfg,
                  # toy#136 (K1): activation + positional-encoding axes —
                  # the engine overwrites these BEFORE build_forward
                  # (exactly the seq_rope_cfg pattern). 0 = the
                  # byte-gated defaults (swiglu, rope).
                  :seq_act, :seq_nope,
                  # toy#138 K3b (AttnRes, K3 §2.2): per-layer learnable
                  # pseudo-queries + the ones-gamma the score kernel's
                  # RMSNorm uses. Empty array = OFF (plain residual
                  # accumulation, byte-null).
                  :seq_attnres_q, :t_seq_attnres_ones, :seq_attnres_on

    def initialize
      @t_seq_token_embed      = TinyNN.tnn_null_ptr
      @t_seq_final_norm_gamma = TinyNN.tnn_null_ptr
      @t_seq_output           = TinyNN.tnn_null_ptr
      @t_seq_w_proj           = TinyNN.tnn_null_ptr
      # Seed with one block — matches the former cache init (L112).
      @seq_blocks_ffi         = [Toy::LLM::Blocks::TransformerBlock.new]
      # Phase 3 — parallel seed: one attention spec for the seed block.
      @seq_layer_specs        = [Toy::LLM::Archs::LayerSpec.new(Toy::LLM::Archs::LayerSpec::KIND_ATTENTION)]
      # Phase 5 — parallel int dispatch keys (KIND_ATTENTION for the seed).
      @seq_layer_kinds        = [Toy::LLM::Archs::LayerSpec::KIND_ATTENTION]
      # Phase 5 — parallel GDN-block slots. Seeded with GDNBlock placeholders so
      # the array is MONOMORPHIC (all GDNBlock) — the seam's KIND_GDN call site
      # never sees a mixed null/object array (Spinel poly-array landmine). At
      # KIND_ATTENTION layers the placeholder is simply never invoked.
      @seq_gdn_blocks_ffi     = [Toy::LLM::Blocks::GDNBlock.new]
      @seq_kda_blocks_ffi     = [Toy::LLM::Blocks::KDABlock.new]
      @seq_donor_d_in         = 0
      # The cache overwrites seq_rope_cfg with the real RoPE::Cfg before
      # build_forward runs (each realize prologue rebuilds it).
      @seq_rope_cfg           = TinyNN.tnn_null_ptr
      @seq_act                = 0
      @seq_nope               = 0
      @seq_attnres_q          = [TinyNN.tnn_null_ptr]; @seq_attnres_q.pop
      @t_seq_attnres_ones     = TinyNN.tnn_null_ptr
      @seq_attnres_on         = 0
    end

    # Reset @seq_blocks_ffi and fill it with exactly n_layers fresh
    # TransformerBlocks. The four cache realize paths each ran this
    # identical loop verbatim (P2.6 Step 2) — seed one block, then push
    # n_layers-1 more — so the SHAPE here matches the former cache loop
    # byte-for-byte (length == n_layers; first element is a fresh block,
    # exactly like the cache's `[TransformerBlock.new]` seed). The arch
    # already OWNS this array (ctor seeds it with one block at L83) and
    # already constructs TransformerBlock.new there, so no new class /
    # Struct / FFI :str at class load. Each realize path now calls this
    # via the cache's seq_blocks_ffi delegator chain (self.seq_arch).
    # Phase 5 hybrid — rebuild the per-layer spec array from a per-layer GDN
    # bool flag, using the LayerSpec CTOR (never the .kind= setter: mutating
    # LayerSpec.kind elsewhere while build_forward reads it trips a Spinel
    # codegen miscompile that corrupts the token-id finalize). Called after
    # seed_blocks!, before alloc.
    # Mark ONE layer as GDN. Takes an INT index (never an array param — a
    # function-parameter array trips the Spinel #688 type-lock landmine, which
    # here manifests as a token-id-finalize codegen miscompile). Mutates the
    # plain int dispatch array element (proven-safe).
    # toy#138 K3b: ONE AttnRes mixture (K3 eq 9).
    #   s_j   = qᵀ RMSNorm(v_j)            -> [1, T]   per source
    #   α     = softmax_j(s_j)             -> [n, T]   (over the SOURCE axis)
    #   out   = Σ_j α_j ⊙ v_j              -> [d, T]
    # The RMSNorm is the paper's magnitude guard (a layer with large
    # outputs must not dominate); it uses a constant ones-gamma, so it
    # adds no parameters.
    #
    # PLUMBING NOTE (the K2c lesson): the per-source weight rows are
    # STRIDED VIEWS of the softmax output, and a strided view used as a
    # mul operand poisons the backward (grad path through GGML_OP_VIEW
    # hits a scale on non-contiguous storage — ggml.c:3392). Every row
    # is therefore cont'd before it multiplies anything.
    def attnres_mix(sess, srcs, q, ones_gamma, eps)
      n = srcs.length
      if n == 1
        return srcs[0]   # softmax over one source is exactly 1.0
      end
      fbytes = 4
      scores = TinyNN.tnn_null_ptr
      j = 0
      while j < n
        nj = TinyNN.tnn_rms_norm(sess, srcs[j], ones_gamma, eps)
        sj = TinyNN.tnn_matmul(sess, q, nj)        # [d,1]ᵀ·[d,T] -> [1,T]
        if j == 0
          scores = sj
        else
          scores = TinyNN.tnn_concat(sess, scores, sj, 0)   # -> [n, T]
        end
        j = j + 1
      end
      alpha = TinyNN.tnn_softmax(sess, scores)     # over ne0 = the source axis
      out = TinyNN.tnn_null_ptr
      j = 0
      while j < n
        row = TinyNN.tnn_cont_2d(sess,
                TinyNN.tnn_view_2d(sess, alpha, 1, seq_t_of(srcs[0]),
                                   n * fbytes, j * fbytes),
                1, seq_t_of(srcs[0]))
        term = TinyNN.tnn_mul(sess, srcs[j], row)  # [d,T] * [1,T] broadcast
        if j == 0
          out = term
        else
          out = TinyNN.tnn_add(sess, out, term)
        end
        j = j + 1
      end
      out
    end

    # Token count of a [d, T] activation (ne1).
    def seq_t_of(t)
      TinyNN.tnn_tensor_nelements(t) / TinyNN.tnn_tensor_ne0(t)
    end

    def set_gdn_layer!(idx)
      @seq_layer_kinds[idx] = Toy::LLM::Archs::LayerSpec::KIND_GDN
    end

    # toy#138 K3b: allocate the AttnRes pseudo-queries — ONE [d,1] per
    # layer (layer l mixes sources 0..l) plus ONE for the final
    # aggregation, and the ones-gamma the score RMSNorm reads. The
    # queries are registered as GLOBALS (via the cache's recorder) so
    # the existing opt walker trains them with no new arm.
    def alloc_attnres!(sess, cache, n_layers, d_model)
      @seq_attnres_on = 1
      @t_seq_attnres_ones = TinyNN.tnn_input_1d_f32_persistent(sess, d_model)
      li_a = 0
      while li_a < n_layers + 1
        q = TinyNN.tnn_input_2d_f32_persistent(sess, 1, d_model)   # ne=[d,1]
        @seq_attnres_q.push(q)
        cache.ft_add_global_2d(q, 1, d_model)
        li_a = li_a + 1
      end
      0
    end

    def set_kda_layer!(idx)
      @seq_layer_kinds[idx] = Toy::LLM::Archs::LayerSpec::KIND_KDA
    end

    def seed_blocks!(n_layers)
      @seq_blocks_ffi = [Toy::LLM::Blocks::TransformerBlock.new]
      # Phase 3 — seed the parallel spec array in lockstep. Every layer is
      # KIND_ATTENTION for now (the homogeneous-Llama refactor gate); Phase 5
      # overwrites individual entries with KIND_GDN for Dragon's pattern.
      @seq_layer_specs = [Toy::LLM::Archs::LayerSpec.new(Toy::LLM::Archs::LayerSpec::KIND_ATTENTION)]
      @seq_gdn_blocks_ffi = [Toy::LLM::Blocks::GDNBlock.new]
      @seq_kda_blocks_ffi = [Toy::LLM::Blocks::KDABlock.new]
      @seq_layer_kinds = [Toy::LLM::Archs::LayerSpec::KIND_ATTENTION]
      li_init = 1
      while li_init < n_layers
        @seq_blocks_ffi.push(Toy::LLM::Blocks::TransformerBlock.new)
        @seq_layer_specs.push(Toy::LLM::Archs::LayerSpec.new(Toy::LLM::Archs::LayerSpec::KIND_ATTENTION))
        @seq_gdn_blocks_ffi.push(Toy::LLM::Blocks::GDNBlock.new)
        @seq_kda_blocks_ffi.push(Toy::LLM::Blocks::KDABlock.new)
        @seq_layer_kinds.push(Toy::LLM::Archs::LayerSpec::KIND_ATTENTION)
        li_init = li_init + 1
      end
    end

    # Allocate the three arch-owned PERSISTENT global tensors from the
    # mmap'd GGUF: token_embd.weight (2d, native type), output_norm.weight
    # (1d f32), and — when untied — output.weight (2d, native type). The
    # cache's realize_for_mmap formerly ran this block inline (P2.6 pass-2
    # Step 1); it is moved VERBATIM here (same FFI primitives, same
    # find_index/file_offset/type LITERAL string lookups at runtime, same
    # untied conditional which the GGUF round-trip gate exercises true).
    # The arch already OWNS these accessors (L68), so no new class / Struct
    # / FFI :str at class load. Called ONLY from realize_for_mmap — the
    # random_init globals and full_finetune's else-branch globals are
    # structurally different and are NOT routed through this helper.
    # Mirrors the seed_blocks! / alloc_trainable_f32_weights! precedents.
    def load_globals_from_gguf_mmap!(sess, gguf_handle, vocab, d_model, untied)
      eidx = TinyNN.tnn_gguf_find_index(gguf_handle, "token_embd.weight")
      eoff = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, eidx)
      etyp = TinyNN.tnn_gguf_tensor_type(gguf_handle, eidx)
      @t_seq_token_embed = TinyNN.tnn_input_2d_persistent_mmap(sess,
                             vocab, d_model, etyp, eoff)

      fnidx = TinyNN.tnn_gguf_find_index(gguf_handle, "output_norm.weight")
      fnoff = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, fnidx)
      @t_seq_final_norm_gamma = TinyNN.tnn_input_1d_persistent_mmap(sess,
                                  d_model, 0, fnoff)

      if untied
        oidx = TinyNN.tnn_gguf_find_index(gguf_handle, "output.weight")
        ooff = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, oidx)
        otyp = TinyNN.tnn_gguf_tensor_type(gguf_handle, oidx)
        @t_seq_output = TinyNN.tnn_input_2d_persistent_mmap(sess,
                          vocab, d_model, otyp, ooff)
      end
    end

    # P2-finish — the RANDOM-INIT (+ projection-lens) trainable-F32 GLOBAL alloc,
    # lifted VERBATIM from Toy::LLM::Engine::LlamaSeqEngine#realize_for_random_init
    # (alloc + ft_add_global / ft_name_last_global ORDER unchanged → bit-identical
    # graph; gated by train_gate from-scratch + smoke_projection_lens). The arch
    # already OWNS these handles (the load_globals_from_gguf_mmap! precedent); the
    # engine's @ft_globals_* recorders + the frozen-embed :str namer are back-called
    # through `cache` (the tnn_tensor_set_name :str FFI stays on the cache realize
    # runtime path — same discipline as ft_name_last / lora_name_q!). donor_d_in>0 =
    # projection lens (frozen donor-width embed + trainable lens.proj); 0 = standard.
    def alloc_globals_trainable_f32!(sess, cache, vocab, d_model, donor_d_in, untied)
      if donor_d_in > 0
        self.t_seq_token_embed = TinyNN.tnn_input_2d_f32_persistent(sess, vocab, donor_d_in)
        cache.name_global!(self.t_seq_token_embed, "token_embd.weight")
        self.t_seq_w_proj = TinyNN.tnn_input_2d_f32_persistent(sess, d_model, donor_d_in)
        cache.ft_add_global_2d(self.t_seq_w_proj, d_model, donor_d_in)
        cache.ft_name_last_global("lens.proj.weight")
      else
        self.t_seq_token_embed = TinyNN.tnn_input_2d_f32_persistent(sess, vocab, d_model)
        cache.ft_add_global_2d(self.t_seq_token_embed, vocab, d_model)
        cache.ft_name_last_global("token_embd.weight")
      end

      self.t_seq_final_norm_gamma = TinyNN.tnn_input_1d_f32_persistent(sess, d_model)
      cache.ft_add_global_1d(self.t_seq_final_norm_gamma)
      cache.ft_name_last_global("output_norm.weight")

      if untied
        self.t_seq_output = TinyNN.tnn_input_2d_f32_persistent(sess, vocab, d_model)
        cache.ft_add_global_2d(self.t_seq_output, vocab, d_model)
        cache.ft_name_last_global("output.weight")
      end
    end

    # SEQ-MODE forward orchestration. The per-graph INPUT handles
    # (token_ids, positions) are ALLOCATED BY THE CACHE before this call
    # (cache-owned graph I/O, read by forward() and the uploaders) and
    # passed in; ditto t_rope_freq_factors and t_attn_mask. The arch
    # builds: get_rows(token_embed, token_ids) → x_embed (tap), optional
    # projection-lens matmul(w_proj, x_embed) when seq_donor_d_in>0 (tap),
    # the shared TransformerBlockCtx built ONCE, the block-stacking loop,
    # final RMSNorm (tap), tied/untied logits matmul (tap). Returns the
    # three per-graph output handles in a LlamaArchForwardOut.
    def build_forward(sess, t_token_ids, t_positions, t_rope_freq_factors,
                      t_attn_mask, seq_eps, seq_d_head, seq_n_kv, seq_n_heads,
                      seq_group_size, seq_has_qkv_bias, seq_weight_dtype,
                      seq_lora_q_enabled, seq_t, seq_b, seq_n_layers,
                      seq_has_untied_output)
      eps   = seq_eps
      scale = 1.0 / Math.sqrt(seq_d_head.to_f)

      # Per-forward block context: the 14 config/handle values the block
      # body reads. Positional class (no keyword_init) — matches the
      # TransformerBlockCtx member order exactly. Built once before the
      # block-stacking loop; shared (read-only) across all blocks.
      ctx = Toy::LLM::Blocks::TransformerBlockCtx.new(
        scale, eps, seq_n_kv, seq_n_heads, seq_group_size,
        seq_has_qkv_bias, seq_weight_dtype, seq_lora_q_enabled,
        t_positions, t_rope_freq_factors, self.seq_rope_cfg,
        seq_t, seq_b, t_attn_mask, self.seq_act, self.seq_nope)

      # toy#138 K3b (AttnRes): sources for the depth-attention. src[0]
      # is the embedding (post-lens); src[i+1] is layer i's FUNCTION
      # output f_i(h_i) — recovered as (block_out − block_in), because
      # toy's blocks return the residual sum x+f(x) while K3 eq 8 keys
      # and values on f_i itself. Plain-residual mode leaves this array
      # empty and never touches it.
      ar_srcs = [TinyNN.tnn_null_ptr]; ar_srcs.pop

      x_embed = TinyNN.tnn_get_rows(sess, self.t_seq_token_embed, t_token_ids)
      TinyNN.tnn_set_output(x_embed)

      # E2.3 — projection lens. ggml matmul(W, x) with W=[donor_d_in, d_model]
      # and x=[donor_d_in, T] gives [d_model, T] (contraction on ne[0]).
      if self.seq_donor_d_in > 0
        t_proj = TinyNN.tnn_matmul(sess, self.t_seq_w_proj, x_embed)
        TinyNN.tnn_set_output(t_proj)
        t_cur = t_proj
      else
        t_cur = x_embed
      end
      if self.seq_attnres_on == 1
        ar_srcs.push(t_cur)
      end
      li_g = 0
      while li_g < seq_n_layers
        # Phase 3 — per-layer descriptor dispatch. The branch compares a FLAT
        # INT (spec.kind) and each arm calls a CONCRETE typed block method, so
        # every .build_forward call site stays monomorphic (one receiver
        # class). KIND_ATTENTION is the only arm wired today; KIND_GDN gets its
        # own arm + its own typed block array in Phase 5. Unknown kinds fail
        # loud rather than silently building the wrong graph (never-mask rule).
        # AttnRes: this layer's INPUT is the learned mixture over every
        # preceding source instead of the accumulated residual.
        ar_in = t_cur
        if self.seq_attnres_on == 1
          ar_in = attnres_mix(sess, ar_srcs, self.seq_attnres_q[li_g],
                              self.t_seq_attnres_ones, eps)
          t_cur = ar_in
        end
        spec_kind = self.seq_layer_kinds[li_g]
        if spec_kind == Toy::LLM::Archs::LayerSpec::KIND_ATTENTION
          t_cur = self.seq_blocks_ffi[li_g].build_forward(sess, t_cur, ctx)
        elsif spec_kind == Toy::LLM::Archs::LayerSpec::KIND_GDN
          # Concrete typed call into the parallel GDN array — the GDN block reads
          # its own dims (set at alloc); seq_t/eps come from the shared ctx.
          t_cur = self.seq_gdn_blocks_ffi[li_g].build_forward(sess, t_cur, seq_t, eps)
        elsif spec_kind == Toy::LLM::Archs::LayerSpec::KIND_KDA
          # toy#137 K2b: concrete typed call into the parallel KDA array.
          t_cur = self.seq_kda_blocks_ffi[li_g].build_forward(sess, t_cur, seq_t, eps)
        else
          raise "LlamaArch#build_forward: unsupported layer kind #{spec_kind} at layer #{li_g}"
        end
        if self.seq_attnres_on == 1
          # f_l = (x + f_l(x)) − x, the layer's own contribution.
          ar_srcs.push(TinyNN.tnn_sub(sess, t_cur, ar_in))
        end
        li_g = li_g + 1
      end

      if self.seq_attnres_on == 1
        # "The final output layer then aggregates all block
        # representations" (K3 §2.2) — one more mixture, own query.
        t_cur = attnres_mix(sess, ar_srcs, self.seq_attnres_q[seq_n_layers],
                            self.t_seq_attnres_ones, eps)
      end
      x_final = Toy::LLM::Primitives::RMSNorm.build(sess, t_cur, self.t_seq_final_norm_gamma, eps)
      TinyNN.tnn_set_output(x_final)

      if seq_has_untied_output
        logits = TinyNN.tnn_matmul(sess, self.t_seq_output, x_final)
      else
        logits = TinyNN.tnn_matmul(sess, self.t_seq_token_embed, x_final)
      end
      TinyNN.tnn_set_output(logits)

      Toy::LLM::Archs::LlamaArchForwardOut.new(x_embed, x_final, logits)
    end
  end
end; end; end
