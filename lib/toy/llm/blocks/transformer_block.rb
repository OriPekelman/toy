# lib/toy/llm/blocks/transformer_block.rb — L2 block: one Llama/Qwen-family transformer block
# (RMSNorm + GQA attention with RoPE + SwiGLU FFN), seq-mode forward.
#
# Extracted from lib/llama_seq_forward_ffi.rb (P2.4). This is the
# minimal faithful lift of the former LlamaSeqBlockFFI class +
# build_seq_block / build_seq_qhead / mp_matmul: the forward body is
# moved VERBATIM (op order unchanged → bit-identical output) with only
# mechanical rewrites — the per-forward context the body previously
# read off the cache as @ivars is now passed IN via a positional ctx
# object (TransformerBlockCtx), and the block owns its own weight
# handles (former blk.* are now self.*).
#
# DIVERGENCE from the L2 README sketch: the README shows a forward-
# looking build_forward(sess, x, state, cfg) -> [out, state_out] with a
# KV-cache "state" threaded per block. That incremental KV-decode does
# NOT exist in seq mode — full-sequence forward threads NO per-block KV
# state (KV-cache decode is the separate lib/toy_smollm2_ffi_kv.rb
# path, out of scope here). We adapt to build_forward(sess, t_x, ctx)
# -> t_resid (single handle). We also keep the owned-weight field names
# (t_seq_*) verbatim rather than renaming to the sketch's short names,
# so the cache-side realize / train / tap walkers keep working by
# accessor name with no parity risk.
#
# Spinel hygiene: TransformerBlockCtx is a plain class with an explicit
# positional initialize (NO kwargs, NO default args — default-arg
# poisoning, landmine #4); TransformerBlock#initialize takes NO args and
# has NO default-arg ctor; no Card /
# step_bind / FFI :str args at class load (step_bind :str landmine
# 2026-05-28 — ft_name_last's tnn_tensor_set_name :str stays on the
# cache realize runtime path, not here). The IntArray ptr-array params
# (t_k_per_kv, t_vt_per_kv) keep their trailing positional slots so
# Spinel's locked IntArray param typing (#688) does not shift.
#
# This file does NOT `require_relative "tinynn"`: the loading module
# (lib/llama_seq_forward_ffi.rb) already loads the correct backend's
# TinyNN before requiring this block, exactly like the L1 primitives.
# The mirror generator picks the backend via the monolith's require
# rewrite.

module Toy; module LLM; module Blocks
  # Per-forward read-only context: the 14 config/handle values the body
  # previously read off the cache as @ivars, packed fresh each forward.
  #
  # Plain class with an explicit POSITIONAL initialize (NO default args,
  # NO kwargs) — the same Spinel-safe pattern as RoPE::Cfg. We do NOT use
  # Struct.new here, and the member names keep the original cache's
  # `@seq_*` / `@t_seq_*` prefixes VERBATIM. Both choices are forced by
  # Spinel's whole-program type inference, which unifies an accessor's
  # return type across every class exposing the same accessor name:
  #   - a Struct's auto-generated `#n_heads` collided with
  #     SmolLM2Config#n_heads / CausalSelfAttention#n_heads, breaking an
  #     unrelated call (Toy::SmolLM2#algorithm's `@cfg.n_heads.to_s`);
  #   - short names `#t` / `#b` collided with Toy::Linear#b (a weight
  #     tensor / FloatArray), so `ctx.b`/`ctx.t` were inferred as
  #     FloatArray* and poisoned RoPE.apply_2d / GQA.attention's integer
  #     batch args.
  # The `@seq_*`-prefixed names were chosen on the cache precisely for
  # this type-isolation (see lib/llama_seq_forward_ffi.rb header), so
  # reusing them here keeps every accessor's return type local and
  # concrete. Carries values, no behavior.
  class TransformerBlockCtx
    attr_accessor :seq_scale, :seq_eps, :seq_n_kv, :seq_n_heads, :seq_group_size,
                  :seq_has_qkv_bias, :seq_weight_dtype, :seq_lora_q_enabled,
                  :t_seq_positions, :t_seq_rope_freq_factors, :seq_rope_cfg,
                  :seq_t, :seq_b, :t_seq_attn_mask

    def initialize(seq_scale, seq_eps, seq_n_kv, seq_n_heads, seq_group_size,
                   seq_has_qkv_bias, seq_weight_dtype, seq_lora_q_enabled,
                   t_seq_positions, t_seq_rope_freq_factors, seq_rope_cfg,
                   seq_t, seq_b, t_seq_attn_mask)
      @seq_scale               = seq_scale
      @seq_eps                 = seq_eps
      @seq_n_kv                = seq_n_kv
      @seq_n_heads             = seq_n_heads
      @seq_group_size          = seq_group_size
      @seq_has_qkv_bias        = seq_has_qkv_bias
      @seq_weight_dtype        = seq_weight_dtype
      @seq_lora_q_enabled      = seq_lora_q_enabled
      @t_seq_positions         = t_seq_positions
      @t_seq_rope_freq_factors = t_seq_rope_freq_factors
      @seq_rope_cfg            = seq_rope_cfg
      @seq_t                   = seq_t
      @seq_b                   = seq_b
      @t_seq_attn_mask         = t_seq_attn_mask
    end
  end

  # Per-block tensor handles. Field names are UNCHANGED from the former
  # LlamaSeqBlockFFI so the cache-side realize / train / tap walkers
  # keep working by accessor name.
  class TransformerBlock
    attr_accessor :t_seq_rn1_gamma, :t_seq_rn2_gamma,
                  :t_seq_w_q,  :t_seq_w_k,  :t_seq_w_v,  :t_seq_w_o,
                  :t_seq_b_q,  :t_seq_b_k,  :t_seq_b_v,
                  :t_seq_w_gate, :t_seq_w_up, :t_seq_w_down,
                  # M3 step 3: LoRA-Q adapter pair + persistent Adam moments.
                  :t_seq_w_lora_a_q, :t_seq_w_lora_b_q,
                  :t_seq_w_lora_a_q_m, :t_seq_w_lora_a_q_v,
                  :t_seq_w_lora_b_q_m, :t_seq_w_lora_b_q_v,
                  # F3 full fine-tune (CPU + CUDA via gen_cuda_mirror).
                  # Parallel arrays of (weight, m, v) triples — one entry
                  # per trainable weight tensor in this block.
                  :ft_weights, :ft_m, :ft_v,
                  # GH#15 — per-block activation tap targets for CKA.
                  :tap_attn_norm, :tap_ffn_out, :tap_resid_post

    def initialize
      @t_seq_rn1_gamma = TinyNN.tnn_null_ptr
      @t_seq_rn2_gamma = TinyNN.tnn_null_ptr
      @t_seq_w_q = [TinyNN.tnn_null_ptr]
      @t_seq_w_k = [TinyNN.tnn_null_ptr]
      @t_seq_w_v = [TinyNN.tnn_null_ptr]
      @t_seq_b_q = [TinyNN.tnn_null_ptr]
      @t_seq_b_k = [TinyNN.tnn_null_ptr]
      @t_seq_b_v = [TinyNN.tnn_null_ptr]
      @t_seq_w_o    = TinyNN.tnn_null_ptr
      @t_seq_w_gate = TinyNN.tnn_null_ptr
      @t_seq_w_up   = TinyNN.tnn_null_ptr
      @t_seq_w_down = TinyNN.tnn_null_ptr
      @t_seq_w_lora_a_q   = [TinyNN.tnn_null_ptr]
      @t_seq_w_lora_b_q   = [TinyNN.tnn_null_ptr]
      @t_seq_w_lora_a_q_m = [TinyNN.tnn_null_ptr]
      @t_seq_w_lora_a_q_v = [TinyNN.tnn_null_ptr]
      @t_seq_w_lora_b_q_m = [TinyNN.tnn_null_ptr]
      @t_seq_w_lora_b_q_v = [TinyNN.tnn_null_ptr]
      @ft_weights = [TinyNN.tnn_null_ptr]; @ft_weights.pop
      @ft_m       = [TinyNN.tnn_null_ptr]; @ft_m.pop
      @ft_v       = [TinyNN.tnn_null_ptr]; @ft_v.pop
      @tap_attn_norm  = TinyNN.tnn_null_ptr
      @tap_ffn_out    = TinyNN.tnn_null_ptr
      @tap_resid_post = TinyNN.tnn_null_ptr
    end

    # One transformer block. SEQ-MODE forward: no `state` input, no
    # `state_out` return (KV decode is the separate toy_smollm2_ffi_kv.rb
    # path). The per-forward context (scale, eps, dims, positions, rope
    # cfg, mask, ...) arrives in `ctx`; the block owns its weight handles
    # as self.t_seq_*. Single tensor return (t_resid).
    #
    #   h1   = RMSNorm(x)
    #   per KV head kv_h:
    #     k_pre = w_k[kv_h] @ h1  (+ b_k[kv_h])         ne=[d_head, T]
    #     k     = RoPE(k_pre, positions)
    #     v     = w_v[kv_h] @ h1  (+ b_v[kv_h])         ne=[d_head, T]
    #     v_t   = transpose(v)                          ne=[T, d_head]
    #   per Q head q_h (kv_h = q_h / group_size):
    #     q_pre = w_q[q_h] @ h1  (+ b_q[q_h])           ne=[d_head, T]
    #     q     = RoPE(q_pre, positions)
    #     scores = k[kv_h] @ q                          ne=[T_keys, T_queries]
    #     scaled = scores / sqrt(d_head)
    #     masked = diag_mask_inf(scaled, 0)              causal triangle
    #     attn   = softmax(masked)                       ne=[T, T]
    #     head_h = v_t[kv_h] @ attn                     ne=[d_head, T]
    #   concat heads along ne0 → ne=[d_model, T]
    #   x_attn = x + (w_o @ concat)
    #   h2     = RMSNorm(x_attn)
    #   ff     = w_down @ (silu(w_gate @ h2) * (w_up @ h2))
    #   x_out  = x_attn + ff
    def build_forward(sess, t_x, ctx)
      t_h = Toy::LLM::Primitives::RMSNorm.build(sess, t_x, self.t_seq_rn1_gamma, ctx.seq_eps)
      # GH#15 — tap the post-attn-norm activation. set_output keeps it
      # alive across graph computation so the host can download it.
      self.tap_attn_norm = t_h
      TinyNN.tnn_set_output(t_h)

      # K, V over all KV heads. Pre-compute v_t per head so the per-Q-head
      # attention loop can index it (avoids n_heads × transpose).
      # See the mirror for the Spinel landmine (issue #688 partial fix;
      # the function-parameter type for build_qhead was already locked in
      # as IntArray before the local-var ptr-push promotion runs).
      # Re-verified 2026-05-26: bare `[]` still fires the warning.
      t_k_per_kv  = [TinyNN.tnn_null_ptr]; t_k_per_kv.pop
      t_vt_per_kv = [TinyNN.tnn_null_ptr]; t_vt_per_kv.pop
      hkv = 0
      while hkv < ctx.seq_n_kv
        t_k_raw = mp_matmul(sess, ctx.seq_weight_dtype, self.t_seq_w_k[hkv], t_h)
        if ctx.seq_has_qkv_bias
          t_k_pre = TinyNN.tnn_add(sess, t_k_raw, self.t_seq_b_k[hkv])
        else
          t_k_pre = t_k_raw
        end
        # ggml_rope_ext requires a->ne[2] == positions->ne[0]. Our K is
        # ne=[d_head, T*B] (ne[2]=1); reshape to ne=[d_head, 1, T*B] so
        # ne[2]==T*B, then reshape back after rope. Reshape is metadata-
        # only (no copy) on contiguous tensors. At T=1, B=1 this is a
        # no-op (1 == 1).
        t_k = Toy::LLM::Primitives::RoPE.apply_2d(
                sess, t_k_pre, ctx.t_seq_positions,
                ctx.t_seq_rope_freq_factors, ctx.seq_rope_cfg, ctx.seq_t, ctx.seq_b)
        t_k_per_kv.push(t_k)

        t_v_raw = mp_matmul(sess, ctx.seq_weight_dtype, self.t_seq_w_v[hkv], t_h)
        if ctx.seq_has_qkv_bias
          t_v = TinyNN.tnn_add(sess, t_v_raw, self.t_seq_b_v[hkv])
        else
          t_v = t_v_raw
        end
        # head_out = v_t @ attn. v has ne=[d_head, T]; transpose to
        # ne=[T, d_head] so the second matmul's contraction lines up.
        t_v_t = TinyNN.tnn_transpose(sess, t_v)
        t_vt_per_kv.push(t_v_t)
        hkv = hkv + 1
      end

      # Per-Q-head attention. GQA: each Q head reads from kv_h = q_h / group_size.
      t_head_out0 = build_qhead(sess, ctx, t_h, 0, t_k_per_kv, t_vt_per_kv)
      t_head_outs = [t_head_out0]
      hq = 1
      while hq < ctx.seq_n_heads
        t_head_outs.push(build_qhead(sess, ctx, t_h, hq, t_k_per_kv, t_vt_per_kv))
        hq = hq + 1
      end

      t_concat = t_head_outs[0]
      hq2 = 1
      while hq2 < ctx.seq_n_heads
        t_concat = TinyNN.tnn_concat(sess, t_concat, t_head_outs[hq2], 0)
        hq2 = hq2 + 1
      end

      t_out_proj = mp_matmul(sess, ctx.seq_weight_dtype, self.t_seq_w_o, t_concat)
      t_x_attn   = TinyNN.tnn_add(sess, t_x, t_out_proj)

      # SwiGLU FFN.
      t_h2    = Toy::LLM::Primitives::RMSNorm.build(sess, t_x_attn, self.t_seq_rn2_gamma, ctx.seq_eps)
      t_gate  = mp_matmul(sess, ctx.seq_weight_dtype, self.t_seq_w_gate, t_h2)
      t_up    = mp_matmul(sess, ctx.seq_weight_dtype, self.t_seq_w_up,   t_h2)
      t_gated = Toy::LLM::Primitives::SwiGLU.gate(sess, t_gate, t_up)
      t_dn    = mp_matmul(sess, ctx.seq_weight_dtype, self.t_seq_w_down, t_gated)
      # GH#15 — tap the FFN output (pre-residual). set_output to pin.
      self.tap_ffn_out = t_dn
      TinyNN.tnn_set_output(t_dn)

      t_resid = TinyNN.tnn_add(sess, t_x_attn, t_dn)
      # GH#15 — tap the residual-stream value AFTER this block. Stable,
      # matched-across-runs region name: resid_post_block.
      self.tap_resid_post = t_resid
      TinyNN.tnn_set_output(t_resid)
      t_resid
    end

    # P2.6 Step 4 — allocate this block's trainable persistent-F32 weight
    # tensors for the random_init realize path. Moved VERBATIM from the
    # per-block ALLOC loop body in
    # Toy::LLM::Engine::LlamaSeqEngine#realize_for_random_init (op order unchanged
    # → bit-identical graph): the block now OWNS the alloc + assignment of
    # its self.t_seq_* handles, exactly as it already owns them at forward
    # time. NO ivar reads off the cache — every value (sess, the seq dims,
    # the name prefix) arrives as an ARG.
    #
    # The ft_add_1d / ft_add_2d / ft_name_last RECORDING primitives STAY
    # on the cache and are called BACK through the passed `cache`
    # reference: they read the cache's @sess to allocate the Adam m/v
    # moments and (ft_name_last) issue tnn_tensor_set_name with a :str
    # name at RUNTIME. That :str call MUST remain on the cache's realize
    # runtime path — never migrated into block class-load scope (step_bind
    # :str landmine 2026-05-28). They push to THIS block's
    # ft_weights/ft_m/ft_v arrays (passed in as `self`/`blk`).
    #
    # CRITICAL: w_o is allocated ne=[d_model, n_heads*d_head] — VERBATIM
    # from random_init (NOT [d_model, d_model]; the two differ under
    # latent GQA where n_heads*d_head != d_model, a divergence the smoke
    # gate cannot catch). Do NOT unify with realize_for_full_finetune's
    # w_o alloc. random_init allocates NO qkv biases (the qkv_bias arg is
    # honoured only by the uploader / Adam-zero paths), so there is no
    # bias branch here.
    #
    # Closes with the per-block set_param loop (former L1082-1086) so the
    # freshly-recorded ft_weights become graph params, same scope as the
    # alloc.
    def alloc_trainable_f32_weights!(sess, cache, prefix,
                                     seq_d_model, seq_d_ff, seq_d_head,
                                     seq_n_heads, seq_n_kv)
      self.t_seq_rn1_gamma = TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_model)
      self.t_seq_rn2_gamma = TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_model)
      cache.ft_add_1d(self, self.t_seq_rn1_gamma)
      cache.ft_name_last(self, prefix + "attn_norm.weight")
      cache.ft_add_1d(self, self.t_seq_rn2_gamma)
      cache.ft_name_last(self, prefix + "ffn_norm.weight")

      self.t_seq_w_q = [TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_head, seq_d_model)]
      hq = 1
      while hq < seq_n_heads
        self.t_seq_w_q.push(TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_head, seq_d_model))
        hq = hq + 1
      end
      hq2 = 0
      while hq2 < seq_n_heads
        cache.ft_add_2d(self, self.t_seq_w_q[hq2], seq_d_head, seq_d_model)
        cache.ft_name_last(self, prefix + "attn_q.head_" + hq2.to_s + ".weight")
        hq2 = hq2 + 1
      end

      self.t_seq_w_k = [TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_head, seq_d_model)]
      self.t_seq_w_v = [TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_head, seq_d_model)]
      hkv = 1
      while hkv < seq_n_kv
        self.t_seq_w_k.push(TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_head, seq_d_model))
        self.t_seq_w_v.push(TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_head, seq_d_model))
        hkv = hkv + 1
      end
      hkv2 = 0
      while hkv2 < seq_n_kv
        cache.ft_add_2d(self, self.t_seq_w_k[hkv2], seq_d_head, seq_d_model)
        cache.ft_name_last(self, prefix + "attn_k.head_" + hkv2.to_s + ".weight")
        cache.ft_add_2d(self, self.t_seq_w_v[hkv2], seq_d_head, seq_d_model)
        cache.ft_name_last(self, prefix + "attn_v.head_" + hkv2.to_s + ".weight")
        hkv2 = hkv2 + 1
      end

      self.t_seq_w_o    = TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_model, seq_n_heads * seq_d_head)
      self.t_seq_w_gate = TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_ff,    seq_d_model)
      self.t_seq_w_up   = TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_ff,    seq_d_model)
      self.t_seq_w_down = TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_model, seq_d_ff)
      cache.ft_add_2d(self, self.t_seq_w_o,    seq_d_model, seq_n_heads * seq_d_head)
      cache.ft_name_last(self, prefix + "attn_output.weight")
      cache.ft_add_2d(self, self.t_seq_w_gate, seq_d_ff,    seq_d_model)
      cache.ft_name_last(self, prefix + "ffn_gate.weight")
      cache.ft_add_2d(self, self.t_seq_w_up,   seq_d_ff,    seq_d_model)
      cache.ft_name_last(self, prefix + "ffn_up.weight")
      cache.ft_add_2d(self, self.t_seq_w_down, seq_d_model, seq_d_ff)
      cache.ft_name_last(self, prefix + "ffn_down.weight")

      # toy#129 item 2: param MARKING moved to the engine call sites --
      # the engine knows the credit-assignment policy and (no-shadow
      # mode) skips the flag on dfa qkv weights so build_backward never
      # expands their chain grads. Flag-set order is numerics-neutral
      # (all flags land before finalize_weights); shadow mode marks
      # everything, exactly as the old in-block loop did.
    end

    # P2-finish — full fine-tune per-block alloc. Lifted VERBATIM from
    # Toy::LLM::Engine::LlamaSeqEngine#realize_for_full_finetune's per-block loop
    # (op order unchanged → bit-identical graph, gated by prep/full_finetune_gate.rb).
    # The block OWNS the alloc + assignment of its self.t_seq_* handles, exactly
    # as alloc_trainable_f32_weights! does. NO ivar reads off the cache — sess,
    # the seq dims and qkv_bias arrive as ARGS; cache.ft_add_* / cache.ft_name_last
    # are back-called (the :str naming stays on the cache realize runtime path —
    # step_bind / :str landmine).
    #
    # TWO deliberate divergences from alloc_trainable_f32_weights! (why this is a
    # SEPARATE method, NOT a reuse):
    #   - w_o is HARD-SQUARE ne=[d_model, d_model] (full_finetune loads a real
    #     GGUF whose attn_output.weight is square) — NOT random_init's divergent
    #     [d_model, n_heads*d_head].
    #   - qkv biases ARE allocated when qkv_bias (alloc_trainable has none).
    def alloc_full_finetune_f32_weights!(sess, cache, prefix,
                                         seq_d_model, seq_d_ff, seq_d_head,
                                         seq_n_heads, seq_n_kv, qkv_bias)
      self.t_seq_rn1_gamma = TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_model)
      self.t_seq_rn2_gamma = TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_model)
      cache.ft_add_1d(self, self.t_seq_rn1_gamma)
      cache.ft_name_last(self, prefix + "attn_norm.weight")
      cache.ft_add_1d(self, self.t_seq_rn2_gamma)
      cache.ft_name_last(self, prefix + "ffn_norm.weight")

      self.t_seq_w_q = [TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_head, seq_d_model)]
      hq = 1
      while hq < seq_n_heads
        self.t_seq_w_q.push(TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_head, seq_d_model))
        hq = hq + 1
      end
      hq2 = 0
      while hq2 < seq_n_heads
        cache.ft_add_2d(self, self.t_seq_w_q[hq2], seq_d_head, seq_d_model)
        cache.ft_name_last(self, prefix + "attn_q.head_" + hq2.to_s + ".weight")
        hq2 = hq2 + 1
      end

      self.t_seq_w_k = [TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_head, seq_d_model)]
      self.t_seq_w_v = [TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_head, seq_d_model)]
      hkv = 1
      while hkv < seq_n_kv
        self.t_seq_w_k.push(TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_head, seq_d_model))
        self.t_seq_w_v.push(TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_head, seq_d_model))
        hkv = hkv + 1
      end
      hkv2 = 0
      while hkv2 < seq_n_kv
        cache.ft_add_2d(self, self.t_seq_w_k[hkv2], seq_d_head, seq_d_model)
        cache.ft_name_last(self, prefix + "attn_k.head_" + hkv2.to_s + ".weight")
        cache.ft_add_2d(self, self.t_seq_w_v[hkv2], seq_d_head, seq_d_model)
        cache.ft_name_last(self, prefix + "attn_v.head_" + hkv2.to_s + ".weight")
        hkv2 = hkv2 + 1
      end

      if qkv_bias
        self.t_seq_b_q = [TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_head)]
        hbq = 1
        while hbq < seq_n_heads
          self.t_seq_b_q.push(TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_head))
          hbq = hbq + 1
        end
        self.t_seq_b_k = [TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_head)]
        self.t_seq_b_v = [TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_head)]
        hbkv = 1
        while hbkv < seq_n_kv
          self.t_seq_b_k.push(TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_head))
          self.t_seq_b_v.push(TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_head))
          hbkv = hbkv + 1
        end
        hbq2 = 0
        while hbq2 < seq_n_heads
          cache.ft_add_1d(self, self.t_seq_b_q[hbq2])
          cache.ft_name_last(self, prefix + "attn_q.head_" + hbq2.to_s + ".bias")
          hbq2 = hbq2 + 1
        end
        hbkv2 = 0
        while hbkv2 < seq_n_kv
          cache.ft_add_1d(self, self.t_seq_b_k[hbkv2])
          cache.ft_name_last(self, prefix + "attn_k.head_" + hbkv2.to_s + ".bias")
          cache.ft_add_1d(self, self.t_seq_b_v[hbkv2])
          cache.ft_name_last(self, prefix + "attn_v.head_" + hbkv2.to_s + ".bias")
          hbkv2 = hbkv2 + 1
        end
      end

      self.t_seq_w_o    = TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_model, seq_d_model)
      self.t_seq_w_gate = TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_ff,    seq_d_model)
      self.t_seq_w_up   = TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_ff,    seq_d_model)
      self.t_seq_w_down = TinyNN.tnn_input_2d_f32_persistent(sess, seq_d_model, seq_d_ff)
      cache.ft_add_2d(self, self.t_seq_w_o,    seq_d_model, seq_d_model)
      cache.ft_name_last(self, prefix + "attn_output.weight")
      cache.ft_add_2d(self, self.t_seq_w_gate, seq_d_ff,    seq_d_model)
      cache.ft_name_last(self, prefix + "ffn_gate.weight")
      cache.ft_add_2d(self, self.t_seq_w_up,   seq_d_ff,    seq_d_model)
      cache.ft_name_last(self, prefix + "ffn_up.weight")
      cache.ft_add_2d(self, self.t_seq_w_down, seq_d_model, seq_d_ff)
      cache.ft_name_last(self, prefix + "ffn_down.weight")

      wi = 0
      while wi < self.ft_weights.length
        TinyNN.tnn_set_param(self.ft_weights[wi])
        wi = wi + 1
      end
    end

    # P2.7 — load this block's PERSISTENT weight handles from the mmap'd
    # GGUF for the realize_for_mmap path. Moved VERBATIM from the per-block
    # ALLOC-from-offsets loop body in Toy::LLM::Engine::LlamaSeqEngine#realize_for_mmap
    # (op order unchanged → bit-identical graph): the block now OWNS the
    # alloc + assignment of its self.t_seq_* handles, exactly as it already
    # owns them at forward time and in alloc_trainable_f32_weights!. NO ivar
    # reads off the cache — every value (sess, the seq dims, the lora flags,
    # qkv_bias, the gguf handle, and the layer index `li`) arrives as an ARG.
    #
    # CRITICAL constraints (all per the alloc_trainable_f32_weights! /
    # load_globals_from_gguf_mmap! precedents):
    #   - head_nbytes STAYS on the cache and is back-called through the
    #     passed `cache` ref (same pattern alloc_trainable_f32_weights! uses
    #     for cache.ft_add_* / cache.ft_name_last).
    #   - The LoRA tnn_tensor_set_name(:str) naming is NOT issued here in
    #     block class-load scope (step_bind / :str landmine #16). The block
    #     assembles the runtime name string and hands it to the cache via
    #     cache.lora_name_q! / cache.lora_name_q_adam! — the actual :str FFI
    #     call lives on the cache realize runtime path, exactly as
    #     ft_name_last stays on the cache.
    #   - w_o is allocated hard-square ne=[d_model, d_model] — VERBATIM from
    #     the former line 668. Do NOT unify with alloc_trainable_f32_weights!'s
    #     divergent [d_model, n_heads*d_head]; the gguf round-trip PINS
    #     n_heads*d_head == d_model so this branch is divergence-blind.
    #   - The set_param marking loop, finalize, Adam zero-init and
    #     build_and_realize! STAY on the cache realize method as head/tail.
    def load_from_gguf_mmap!(sess, cache, gguf_handle, li,
                             seq_n_heads, seq_n_kv, seq_d_head, seq_d_model,
                             seq_d_ff, seq_lora_q_enabled, seq_lora_q_rank,
                             seq_lora_q_adamw_enabled, qkv_bias)
      prefix = "blk." + li.to_s

      rn1_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_norm.weight")
      rn2_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_norm.weight")
      self.t_seq_rn1_gamma = TinyNN.tnn_input_1d_persistent_mmap(sess,
                              seq_d_model, 0,
                              TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, rn1_idx))
      self.t_seq_rn2_gamma = TinyNN.tnn_input_1d_persistent_mmap(sess,
                              seq_d_model, 0,
                              TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, rn2_idx))

      # Q heads — per-head [d_head, d_model] tensor, n_heads of them.
      q_idx      = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_q.weight")
      q_off_base = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, q_idx)
      q_type     = TinyNN.tnn_gguf_tensor_type(gguf_handle, q_idx)
      q_stride   = cache.head_nbytes(q_type, seq_d_head, seq_d_model)
      self.t_seq_w_q = [TinyNN.tnn_input_2d_persistent_mmap(sess,
                         seq_d_head, seq_d_model, q_type, q_off_base)]
      hq = 1
      while hq < seq_n_heads
        self.t_seq_w_q.push(TinyNN.tnn_input_2d_persistent_mmap(sess,
                             seq_d_head, seq_d_model, q_type,
                             q_off_base + hq * q_stride))
        hq = hq + 1
      end

      # M3 step 3 — LoRA-Q adapter pair per Q head. Trainable F32 in
      # ctx_w (mirrors SmolLM2KVFFICache). Optional persistent Adam m/v.
      # Names ride the llama.cpp convention extended for the per-head /
      # adapter axes: blk.N.attn_q.head_H.lora_{a,b}.weight (+ .m / .v).
      lora_prefix = "blk." + li.to_s + ".attn_q.head_"
      if seq_lora_q_enabled
        self.t_seq_w_lora_a_q = [TinyNN.tnn_input_2d_f32_persistent(sess,
                                  seq_lora_q_rank, seq_d_model)]
        self.t_seq_w_lora_b_q = [TinyNN.tnn_input_2d_f32_persistent(sess,
                                  seq_d_head, seq_lora_q_rank)]
        hql = 1
        while hql < seq_n_heads
          self.t_seq_w_lora_a_q.push(TinyNN.tnn_input_2d_f32_persistent(sess,
                                      seq_lora_q_rank, seq_d_model))
          self.t_seq_w_lora_b_q.push(TinyNN.tnn_input_2d_f32_persistent(sess,
                                      seq_d_head, seq_lora_q_rank))
          hql = hql + 1
        end
        hqn = 0
        while hqn < seq_n_heads
          cache.lora_name_q!(self.t_seq_w_lora_a_q[hqn],
                             self.t_seq_w_lora_b_q[hqn],
                             lora_prefix + hqn.to_s)
          hqn = hqn + 1
        end

        if seq_lora_q_adamw_enabled
          self.t_seq_w_lora_a_q_m = [TinyNN.tnn_input_2d_f32_persistent(sess,
                                      seq_lora_q_rank, seq_d_model)]
          self.t_seq_w_lora_a_q_v = [TinyNN.tnn_input_2d_f32_persistent(sess,
                                      seq_lora_q_rank, seq_d_model)]
          self.t_seq_w_lora_b_q_m = [TinyNN.tnn_input_2d_f32_persistent(sess,
                                      seq_d_head, seq_lora_q_rank)]
          self.t_seq_w_lora_b_q_v = [TinyNN.tnn_input_2d_f32_persistent(sess,
                                      seq_d_head, seq_lora_q_rank)]
          hqm = 1
          while hqm < seq_n_heads
            self.t_seq_w_lora_a_q_m.push(TinyNN.tnn_input_2d_f32_persistent(sess,
                                          seq_lora_q_rank, seq_d_model))
            self.t_seq_w_lora_a_q_v.push(TinyNN.tnn_input_2d_f32_persistent(sess,
                                          seq_lora_q_rank, seq_d_model))
            self.t_seq_w_lora_b_q_m.push(TinyNN.tnn_input_2d_f32_persistent(sess,
                                          seq_d_head, seq_lora_q_rank))
            self.t_seq_w_lora_b_q_v.push(TinyNN.tnn_input_2d_f32_persistent(sess,
                                          seq_d_head, seq_lora_q_rank))
            hqm = hqm + 1
          end
          hmn = 0
          while hmn < seq_n_heads
            cache.lora_name_q_adam!(self.t_seq_w_lora_a_q_m[hmn],
                                    self.t_seq_w_lora_a_q_v[hmn],
                                    self.t_seq_w_lora_b_q_m[hmn],
                                    self.t_seq_w_lora_b_q_v[hmn],
                                    lora_prefix + hmn.to_s)
            hmn = hmn + 1
          end
        end
      end

      # K, V heads — per-KV-head [d_head, d_model].
      k_idx      = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_k.weight")
      v_idx      = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_v.weight")
      k_off_base = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, k_idx)
      v_off_base = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, v_idx)
      k_type     = TinyNN.tnn_gguf_tensor_type(gguf_handle, k_idx)
      v_type     = TinyNN.tnn_gguf_tensor_type(gguf_handle, v_idx)
      k_stride   = cache.head_nbytes(k_type, seq_d_head, seq_d_model)
      v_stride   = cache.head_nbytes(v_type, seq_d_head, seq_d_model)
      self.t_seq_w_k = [TinyNN.tnn_input_2d_persistent_mmap(sess,
                         seq_d_head, seq_d_model, k_type, k_off_base)]
      self.t_seq_w_v = [TinyNN.tnn_input_2d_persistent_mmap(sess,
                         seq_d_head, seq_d_model, v_type, v_off_base)]
      hkv = 1
      while hkv < seq_n_kv
        self.t_seq_w_k.push(TinyNN.tnn_input_2d_persistent_mmap(sess,
                             seq_d_head, seq_d_model, k_type,
                             k_off_base + hkv * k_stride))
        self.t_seq_w_v.push(TinyNN.tnn_input_2d_persistent_mmap(sess,
                             seq_d_head, seq_d_model, v_type,
                             v_off_base + hkv * v_stride))
        hkv = hkv + 1
      end

      # Optional Q/K/V biases (Qwen2.x). 1D [d_head] per head, contiguous.
      if qkv_bias
        qb_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_q.bias")
        kb_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_k.bias")
        vb_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_v.bias")
        qb_off = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, qb_idx)
        kb_off = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, kb_idx)
        vb_off = TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, vb_idx)
        bias_stride = seq_d_head * 4

        self.t_seq_b_q = [TinyNN.tnn_input_1d_persistent_mmap(sess, seq_d_head, 0, qb_off)]
        hq = 1
        while hq < seq_n_heads
          self.t_seq_b_q.push(TinyNN.tnn_input_1d_persistent_mmap(sess, seq_d_head, 0,
                               qb_off + hq * bias_stride))
          hq = hq + 1
        end
        self.t_seq_b_k = [TinyNN.tnn_input_1d_persistent_mmap(sess, seq_d_head, 0, kb_off)]
        self.t_seq_b_v = [TinyNN.tnn_input_1d_persistent_mmap(sess, seq_d_head, 0, vb_off)]
        hkv = 1
        while hkv < seq_n_kv
          self.t_seq_b_k.push(TinyNN.tnn_input_1d_persistent_mmap(sess, seq_d_head, 0,
                               kb_off + hkv * bias_stride))
          self.t_seq_b_v.push(TinyNN.tnn_input_1d_persistent_mmap(sess, seq_d_head, 0,
                               vb_off + hkv * bias_stride))
          hkv = hkv + 1
        end
      end

      # O, FFN — full 2D weights, no per-head split.
      o_idx    = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_output.weight")
      gate_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_gate.weight")
      up_idx   = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_up.weight")
      down_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_down.weight")
      self.t_seq_w_o    = TinyNN.tnn_input_2d_persistent_mmap(sess, seq_d_model, seq_d_model,
                           TinyNN.tnn_gguf_tensor_type(gguf_handle, o_idx),
                           TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, o_idx))
      self.t_seq_w_gate = TinyNN.tnn_input_2d_persistent_mmap(sess, seq_d_ff, seq_d_model,
                           TinyNN.tnn_gguf_tensor_type(gguf_handle, gate_idx),
                           TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, gate_idx))
      self.t_seq_w_up   = TinyNN.tnn_input_2d_persistent_mmap(sess, seq_d_ff, seq_d_model,
                           TinyNN.tnn_gguf_tensor_type(gguf_handle, up_idx),
                           TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, up_idx))
      self.t_seq_w_down = TinyNN.tnn_input_2d_persistent_mmap(sess, seq_d_model, seq_d_ff,
                           TinyNN.tnn_gguf_tensor_type(gguf_handle, down_idx),
                           TinyNN.tnn_gguf_tensor_file_offset(gguf_handle, down_idx))
    end

    # P2.7 pass-3 — allocate this block's PERSISTENT weight handles for the
    # realize_for_q8_copy path (Q8-stays-Q8 verbatim copy on CUDA). Moved
    # VERBATIM from the per-block ALLOC-typed loop body in
    # Toy::LLM::Engine::LlamaSeqEngine#realize_for_q8_copy (op order unchanged →
    # bit-identical graph): the block now OWNS the alloc + assignment of its
    # self.t_seq_* handles, exactly as in load_from_gguf_mmap! /
    # alloc_trainable_f32_weights!. NO ivar reads off the cache — every value
    # (sess, the seq dims, the lora flags, qkv_bias, the gguf handle, and the
    # layer index `li`) arrives as an ARG. Mirrors load_from_gguf_mmap!'s
    # arg-passing exactly.
    #
    # CRITICAL constraints:
    #   - Allocates ctx_w tensors of the on-disk gguf type via
    #     tnn_input_2d_persistent_typed (verbatim copy requires source/target
    #     types match). rn1/rn2 gammas + qkv_bias + LoRA/Adam are F32.
    #   - This path never names LoRA tensors (the q8 loop body issues NO
    #     tnn_tensor_set_name), so the moved body is :str-free and
    #     Spinel-#16-clean — no cache.lora_name_q! back-calls here.
    #   - w_o is allocated hard-square ne=[d_model, d_model] — VERBATIM from
    #     the former cache line 318. Do NOT unify with the divergent shape;
    #     the gguf round-trip PINS n_heads*d_head == d_model.
    #   - seq_vocab_size is accepted (positional parity with the cache's
    #     intent / the mmap precedent) but UNUSED here — the block allocates
    #     no global tensors.
    #   - The set_param marking loop, finalize, verbatim-copy phase, Adam
    #     zero-init and build_and_realize! STAY on the cache realize method.
    def alloc_q8_typed_from_gguf!(sess, gguf_handle, li,
                                  seq_n_heads, seq_n_kv, seq_d_head, seq_d_model,
                                  seq_d_ff, seq_vocab_size, seq_lora_q_enabled,
                                  seq_lora_q_rank, seq_lora_q_adamw_enabled, qkv_bias)
      prefix = "blk." + li.to_s

      self.t_seq_rn1_gamma = TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_model)
      self.t_seq_rn2_gamma = TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_model)

      q_idx  = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_q.weight")
      q_type = TinyNN.tnn_gguf_tensor_type(gguf_handle, q_idx)
      self.t_seq_w_q = [TinyNN.tnn_input_2d_persistent_typed(sess, seq_d_head, seq_d_model, q_type)]
      hq = 1
      while hq < seq_n_heads
        self.t_seq_w_q.push(TinyNN.tnn_input_2d_persistent_typed(sess, seq_d_head, seq_d_model, q_type))
        hq = hq + 1
      end

      k_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_k.weight")
      v_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_v.weight")
      k_type = TinyNN.tnn_gguf_tensor_type(gguf_handle, k_idx)
      v_type = TinyNN.tnn_gguf_tensor_type(gguf_handle, v_idx)
      self.t_seq_w_k = [TinyNN.tnn_input_2d_persistent_typed(sess, seq_d_head, seq_d_model, k_type)]
      self.t_seq_w_v = [TinyNN.tnn_input_2d_persistent_typed(sess, seq_d_head, seq_d_model, v_type)]
      hkv = 1
      while hkv < seq_n_kv
        self.t_seq_w_k.push(TinyNN.tnn_input_2d_persistent_typed(sess, seq_d_head, seq_d_model, k_type))
        self.t_seq_w_v.push(TinyNN.tnn_input_2d_persistent_typed(sess, seq_d_head, seq_d_model, v_type))
        hkv = hkv + 1
      end

      if qkv_bias
        self.t_seq_b_q = [TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_head)]
        hbq = 1
        while hbq < seq_n_heads
          self.t_seq_b_q.push(TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_head))
          hbq = hbq + 1
        end
        self.t_seq_b_k = [TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_head)]
        self.t_seq_b_v = [TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_head)]
        hbkv = 1
        while hbkv < seq_n_kv
          self.t_seq_b_k.push(TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_head))
          self.t_seq_b_v.push(TinyNN.tnn_input_1d_f32_persistent(sess, seq_d_head))
          hbkv = hbkv + 1
        end
      end

      o_idx    = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_output.weight")
      gate_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_gate.weight")
      up_idx   = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_up.weight")
      down_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_down.weight")
      self.t_seq_w_o    = TinyNN.tnn_input_2d_persistent_typed(sess, seq_d_model, seq_d_model,
                           TinyNN.tnn_gguf_tensor_type(gguf_handle, o_idx))
      self.t_seq_w_gate = TinyNN.tnn_input_2d_persistent_typed(sess, seq_d_ff, seq_d_model,
                           TinyNN.tnn_gguf_tensor_type(gguf_handle, gate_idx))
      self.t_seq_w_up   = TinyNN.tnn_input_2d_persistent_typed(sess, seq_d_ff, seq_d_model,
                           TinyNN.tnn_gguf_tensor_type(gguf_handle, up_idx))
      self.t_seq_w_down = TinyNN.tnn_input_2d_persistent_typed(sess, seq_d_model, seq_d_ff,
                           TinyNN.tnn_gguf_tensor_type(gguf_handle, down_idx))

      # LoRA + Adam allocations (same as realize_for_mmap path).
      if seq_lora_q_enabled
        self.t_seq_w_lora_a_q = [TinyNN.tnn_input_2d_f32_persistent(sess,
                                  seq_lora_q_rank, seq_d_model)]
        self.t_seq_w_lora_b_q = [TinyNN.tnn_input_2d_f32_persistent(sess,
                                  seq_d_head, seq_lora_q_rank)]
        hql = 1
        while hql < seq_n_heads
          self.t_seq_w_lora_a_q.push(TinyNN.tnn_input_2d_f32_persistent(sess,
                                      seq_lora_q_rank, seq_d_model))
          self.t_seq_w_lora_b_q.push(TinyNN.tnn_input_2d_f32_persistent(sess,
                                      seq_d_head, seq_lora_q_rank))
          hql = hql + 1
        end
        if seq_lora_q_adamw_enabled
          self.t_seq_w_lora_a_q_m = [TinyNN.tnn_input_2d_f32_persistent(sess,
                                      seq_lora_q_rank, seq_d_model)]
          self.t_seq_w_lora_a_q_v = [TinyNN.tnn_input_2d_f32_persistent(sess,
                                      seq_lora_q_rank, seq_d_model)]
          self.t_seq_w_lora_b_q_m = [TinyNN.tnn_input_2d_f32_persistent(sess,
                                      seq_d_head, seq_lora_q_rank)]
          self.t_seq_w_lora_b_q_v = [TinyNN.tnn_input_2d_f32_persistent(sess,
                                      seq_d_head, seq_lora_q_rank)]
          hqm = 1
          while hqm < seq_n_heads
            self.t_seq_w_lora_a_q_m.push(TinyNN.tnn_input_2d_f32_persistent(sess,
                                          seq_lora_q_rank, seq_d_model))
            self.t_seq_w_lora_a_q_v.push(TinyNN.tnn_input_2d_f32_persistent(sess,
                                          seq_lora_q_rank, seq_d_model))
            self.t_seq_w_lora_b_q_m.push(TinyNN.tnn_input_2d_f32_persistent(sess,
                                          seq_d_head, seq_lora_q_rank))
            self.t_seq_w_lora_b_q_v.push(TinyNN.tnn_input_2d_f32_persistent(sess,
                                          seq_d_head, seq_lora_q_rank))
            hqm = hqm + 1
          end
        end
      end
    end

    # P2.7 pass-3 Step 2 — fill this block's PERSISTENT backend buffers from
    # the GGUF for the realize_for_q8_copy path (Q8-stays-Q8 verbatim copy).
    # Moved VERBATIM from the per-block VERBATIM-COPY loop body in
    # Toy::LLM::Engine::LlamaSeqEngine#realize_for_q8_copy (op order unchanged →
    # bit-identical weights): this is the COPY phase that follows the
    # alloc_q8_typed_from_gguf! ALLOC phase. The block READS its own
    # self.t_seq_* handles (allocated by alloc_q8_typed_from_gguf!) and
    # writes NOTHING on itself — the FFI copy primitives fill the backend
    # buffers by handle. NO ivar reads off the cache — every value (sess,
    # the seq dims, qkv_bias, the gguf handle, the layer index `li`) arrives
    # as an ARG, mirroring alloc_q8_typed_from_gguf! exactly.
    #
    # CRITICAL constraints:
    #   - Per-head slice args are byte-VERBATIM: w_q[hq] takes (hq, n_heads),
    #     w_k/w_v[hkv] take (hkv, n_kv), the qkv biases take (h, d_head). A
    #     swapped index produces deterministic-but-WRONG logits that the
    #     2x-forward byte-identity gate cannot catch — so arg fidelity is the
    #     load-bearing constraint, not behavior the gate observes.
    #   - All primitives are tnn_gguf_copy_* / tnn_gguf_find_index. The
    #     find_index :str arg is issued at RUNTIME (same as
    #     alloc_q8_typed_from_gguf!), never block class-load scope (#16); this
    #     path names NO LoRA tensors, so there are no cache back-calls.
    #   - The GLOBALS verbatim-copy (token embed / final norm / untied output)
    #     STAYS on the cache realize method — those touch cache-level handles.
    def copy_q8_bytes_from_gguf!(sess, gguf_handle, li,
                                 seq_n_heads, seq_n_kv, seq_d_head, qkv_bias)
      prefix = "blk." + li.to_s
      rn1_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_norm.weight")
      rn2_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_norm.weight")
      TinyNN.tnn_gguf_copy_1d_to_persistent(gguf_handle, rn1_idx, sess, self.t_seq_rn1_gamma)
      TinyNN.tnn_gguf_copy_1d_to_persistent(gguf_handle, rn2_idx, sess, self.t_seq_rn2_gamma)

      q_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_q.weight")
      hq = 0
      while hq < seq_n_heads
        TinyNN.tnn_gguf_copy_verbatim_head_slice_to_persistent(gguf_handle, q_idx, sess,
          self.t_seq_w_q[hq], hq, seq_n_heads)
        hq = hq + 1
      end
      k_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_k.weight")
      v_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_v.weight")
      hkv = 0
      while hkv < seq_n_kv
        TinyNN.tnn_gguf_copy_verbatim_head_slice_to_persistent(gguf_handle, k_idx, sess,
          self.t_seq_w_k[hkv], hkv, seq_n_kv)
        TinyNN.tnn_gguf_copy_verbatim_head_slice_to_persistent(gguf_handle, v_idx, sess,
          self.t_seq_w_v[hkv], hkv, seq_n_kv)
        hkv = hkv + 1
      end

      if qkv_bias
        qb_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_q.bias")
        kb_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_k.bias")
        vb_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_v.bias")
        hbq = 0
        while hbq < seq_n_heads
          TinyNN.tnn_gguf_copy_head_bias_slice_to_persistent(gguf_handle, qb_idx, sess,
            self.t_seq_b_q[hbq], hbq, seq_d_head)
          hbq = hbq + 1
        end
        hbkv = 0
        while hbkv < seq_n_kv
          TinyNN.tnn_gguf_copy_head_bias_slice_to_persistent(gguf_handle, kb_idx, sess,
            self.t_seq_b_k[hbkv], hbkv, seq_d_head)
          TinyNN.tnn_gguf_copy_head_bias_slice_to_persistent(gguf_handle, vb_idx, sess,
            self.t_seq_b_v[hbkv], hbkv, seq_d_head)
          hbkv = hbkv + 1
        end
      end

      o_idx    = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".attn_output.weight")
      gate_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_gate.weight")
      up_idx   = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_up.weight")
      down_idx = TinyNN.tnn_gguf_find_index(gguf_handle, prefix + ".ffn_down.weight")
      TinyNN.tnn_gguf_copy_verbatim_to_persistent(gguf_handle, o_idx,    sess, self.t_seq_w_o)
      TinyNN.tnn_gguf_copy_verbatim_to_persistent(gguf_handle, gate_idx, sess, self.t_seq_w_gate)
      TinyNN.tnn_gguf_copy_verbatim_to_persistent(gguf_handle, up_idx,   sess, self.t_seq_w_up)
      TinyNN.tnn_gguf_copy_verbatim_to_persistent(gguf_handle, down_idx, sess, self.t_seq_w_down)
    end

    private

    def build_qhead(sess, ctx, t_h, hq, t_k_per_kv, t_vt_per_kv)
      hkv = hq / ctx.seq_group_size
      t_q_raw = mp_matmul(sess, ctx.seq_weight_dtype, self.t_seq_w_q[hq], t_h)
      # M3 step 3 — LoRA splice on Q (same as decode_step). With B
      # zero-initialized this is a no-op at step 0. LoRA matmuls use
      # TinyNN.tnn_matmul DIRECTLY (F32, no cast) — NOT mp_matmul.
      if ctx.seq_lora_q_enabled
        t_lora_a   = TinyNN.tnn_matmul(sess, self.t_seq_w_lora_a_q[hq], t_h)
        t_lora_b_a = TinyNN.tnn_matmul(sess, self.t_seq_w_lora_b_q[hq], t_lora_a)
        t_q_raw    = TinyNN.tnn_add(sess, t_q_raw, t_lora_b_a)
      end
      if ctx.seq_has_qkv_bias
        t_q_pre = TinyNN.tnn_add(sess, t_q_raw, self.t_seq_b_q[hq])
      else
        t_q_pre = t_q_raw
      end
      # Same rope-shape lift as the K path; see comment in build_forward.
      t_q = Toy::LLM::Primitives::RoPE.apply_2d(
              sess, t_q_pre, ctx.t_seq_positions,
              ctx.t_seq_rope_freq_factors, ctx.seq_rope_cfg, ctx.seq_t, ctx.seq_b)

      # Attention math (scores -> scaled+masked softmax -> weighted V).
      # GQA KV-head selection (hkv) stays on the block; the primitive
      # gets the already-selected K/Vt handles. At B=1 the mask handle
      # (ctx.t_seq_attn_mask) is NULL and is never read by the B=1 branch.
      # head ne=[d_head, T*B].
      Toy::LLM::Primitives::GQA.attention(
        sess, t_k_per_kv[hkv], t_q, t_vt_per_kv[hkv],
        ctx.t_seq_attn_mask, ctx.seq_scale, ctx.seq_b)
    end

    # GH#9 — mixed-precision matmul helper. When weight_dtype != 0,
    # casts the F32 master weight to the chosen dtype before the matmul
    # so the math hits the lower-precision tensor-core path. At
    # weight_dtype == 0 this is `tnn_matmul` verbatim — bit-identical to
    # pre-GH#9 graphs. Takes sess + weight_dtype as args (no ivar reads).
    def mp_matmul(sess, weight_dtype, w, x)
      if weight_dtype != 0
        wc = TinyNN.tnn_cast(sess, w, weight_dtype)
        return TinyNN.tnn_matmul(sess, wc, x)
      end
      TinyNN.tnn_matmul(sess, w, x)
    end
  end
end; end; end
