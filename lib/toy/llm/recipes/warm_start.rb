# lib/toy/llm/recipes/warm_start.rb — L4 recipe: the warm-start
# random-init-plus-donor training plan. SIBLING of FromScratch
# (from_scratch.rb) and LoRA (lora.rb). Same L4 shape: a minimal
# hand-written class with explicit realize methods + step!, uniquely
# ws_-prefixed members, NO Struct.new (#16), NO speculative
# Trainer/Stage/each_stage/DataSpec/Eval (AdamW is baked into the ggml
# backward graph by build_training_step, exactly as in FromScratch/LoRA).
#
# THE ONE STRUCTURAL DIFFERENCE from FromScratch: realize and build are
# SPLIT into separate public methods (realize_scratch! / realize_warm! /
# build!) so the fixture can upload the donor embedding (and/or the PCA
# lens) BETWEEN realize and build — mirroring the frozen reference
# examples/legacy/09_warm_start_train.rb: realize (L138) → upload donor
# (L145-184) / PCA lens (L188-229) → build_training_step (L231).
# FromScratch FUSES realize_for_random_init + build_training_step into
# one realize! because it has nothing to upload between; WarmStart must
# not, or the donor upload lands before the random-init weights exist
# (or after the graph is baked) and trains through the wrong values.
#
# These TWO explicit realize methods ARE the "realize_scratch!/
# realize_warm! NOT each_stage/Stage object" instruction — there is no
# Stage value object and no each_stage driver. realize_warm! is OPTIONAL:
# INIT=scratch (09's default) skips it entirely and trains from the
# random init, which is what reproduces 09's scratch loss curve.
#
# Since toy#73 item 4 realize_warm! OWNS the donor plumbing (the GGUF
# open / width dim-check / find_index / read / upload / free that every
# consumer hand-rolled, ~25 lines each): realize_warm!(path, cfg) does
# the whole read and fails NAMED+LOUD on every mismatch; the class
# method donor_embed_width(path) is the pre-realize half (cfg.donor_d_in
# must carry the donor width BEFORE realize_scratch! sizes the lens).
# The raw upload mechanism survives as upload_donor!(buf, n) for
# already-read buffers. Still caller-side (lib-vs-example scope): the
# PCA-lens W_proj read+upload (09 L188-229) through @ws_cache.sess +
# the t_seq_w_proj delegator, the cosine LR schedule, and the corpus
# streaming; the per-step LR enters ONLY via the caller mutating
# m_hp.flat[0] before step! — there is deliberately NO lr param on
# step! (would diverge from the siblings and move schedule logic into
# the recipe).
#
# Spinel hygiene: NEVER Struct.new (landmine #16 / matz/spinel#1043) —
# hand-written plain class, explicit no-arg ctor, NO default-arg ctor
# (landmine #4). Members uniquely ws_-prefixed for type-isolation. No
# Card/step_bind/FFI :str args at class load — the lone FFI :str-touching
# op (tnn_upload_from_float_array, via :float_array) lives inside a
# method body (step_bind :str landmine 2026-05-28).
#
# Like FromScratch/LoRA, this file does NOT `require_relative "tinynn"`:
# the loading module (lib/llama_seq_forward_ffi.rb, required by the
# fixture before this file) already loads the correct backend's TinyNN.
# step! inlines TinyNN.* calls so it is backend-coupled; the CUDA mirror
# is DEFERRED alongside the GPU deferral — this pass ships CPU-only.

require_relative "../recipe_options"

module Toy; module LLM; module Recipes
  # The warm-start training recipe. realize_scratch! builds the
  # random-init forward+CE+backward+AdamW graph on a
  # Toy::LLM::Engine::LlamaSeqEngine (random-init realize self-enables
  # full_finetune + train_embeddings, so no extra enable_* call is
  # needed) and OPENS the warm window; realize_warm! (optional) uploads
  # an already-read donor embedding into the realize'd embed table BEFORE
  # the graph is baked; build! CLOSES the window by baking
  # forward+CE+backward+opt_step_adamw into the ggml graph. step! then
  # drives one training step. The caller (fixture) owns the experiment
  # config, the donor/PCA GGUF read, the corpus stream, the LR schedule,
  # and the per-step input Mats.
  class WarmStart
    attr_accessor :ws_cache, :ws_t_loss, :ws_t_labels, :ws_t_hp, :ws_step_index

    def initialize
      @ws_cache      = Toy::LLM::Engine::LlamaSeqEngine.new
      @ws_t_loss     = nil
      @ws_t_labels   = nil
      @ws_t_hp       = nil
      @ws_step_index = 0
    end

    # Realize the random-init graph and OPEN the warm window. Delegates
    # VERBATIM to the cache: realize_for_random_init (which self-enables
    # @ft_train_embeddings_enabled + @seq_full_finetune_enabled).
    # `opts` is a Toy::LLM::RecipeOptions (toy#64 item 1) carrying the
    # former 7 trailing positional args, unpacked here in the engine's
    # exact positional order (identical to FromScratch#realize! / 09
    # L138), so the realize is byte-identical. Does NOT bake the graph —
    # that is build!'s job, leaving the window open for an optional
    # realize_warm! upload in between. Returns nil.
    def realize_scratch!(cfg, opts)
      @ws_cache.realize_for_random_init(cfg, opts.t_seq, opts.t_batch,
                                        opts.weight_dtype, opts.untied,
                                        opts.qkv_bias, opts.seed,
                                        opts.init_scale)
      nil
    end

    # Read the donor's embedding width (llama.embedding_length) from a
    # GGUF path — the value the caller must put in cfg.donor_d_in
    # BEFORE realize_scratch! (the projection lens is sized
    # donor_d_in x d_model at realize time, so the recipe cannot learn
    # it later). FAILS LOUD on a missing/corrupt donor or a
    # non-llama-family GGUF. (toy#73 item 4 — the read half of the
    # donor plumbing realize_warm! owns.)
    def self.donor_embed_width(donor_gguf_path)
      if !File.exist?(donor_gguf_path)
        raise "WarmStart.donor_embed_width: donor GGUF not found: " +
              donor_gguf_path
      end
      ggh = TinyNN.tnn_gguf_load(donor_gguf_path)
      if ggh == nil || ggh == TinyNN.tnn_null_ptr
        raise "WarmStart.donor_embed_width: failed to open " +
              donor_gguf_path + " (not a GGUF?)"
      end
      donor_d = TinyNN.tnn_gguf_get_u32(ggh, "llama.embedding_length")
      TinyNN.tnn_gguf_free(ggh)
      if donor_d <= 0
        raise "WarmStart.donor_embed_width: donor has no " +
              "llama.embedding_length key — not llama-family? (" +
              donor_gguf_path + ")"
      end
      donor_d
    end

    # OPTIONAL: warm the realize'd embed table from a donor GGUF. Owns
    # the WHOLE donor read (toy#73 item 4 — was ~25 lines of bare GGUF
    # plumbing in every consumer): open, re-read llama.embedding_length
    # and DIM-CHECK it against cfg.donor_d_in (the width the lens was
    # realized at — a mismatch would silently upload garbage through
    # the wrong stride), find token_embd.weight, read the first
    # cfg.vocab rows, upload through upload_donor!, free. Every failure
    # raises NAMED + LOUD (which tensor, expected vs got, which path).
    # Must be called AFTER realize_scratch! (the tensor exists) and
    # BEFORE build! (else we train through the random init).
    # INIT=scratch flows skip this method entirely. Returns nil.
    def realize_warm!(donor_gguf_path, cfg)
      if !File.exist?(donor_gguf_path)
        raise "WarmStart#realize_warm!: donor GGUF not found: " +
              donor_gguf_path
      end
      ggh = TinyNN.tnn_gguf_load(donor_gguf_path)
      if ggh == nil || ggh == TinyNN.tnn_null_ptr
        raise "WarmStart#realize_warm!: failed to open " +
              donor_gguf_path + " (not a GGUF?)"
      end
      donor_d = TinyNN.tnn_gguf_get_u32(ggh, "llama.embedding_length")
      if donor_d <= 0
        TinyNN.tnn_gguf_free(ggh)
        raise "WarmStart#realize_warm!: donor has no " +
              "llama.embedding_length key — not llama-family? (" +
              donor_gguf_path + ")"
      end
      if donor_d != cfg.donor_d_in
        TinyNN.tnn_gguf_free(ggh)
        raise "WarmStart#realize_warm!: token_embd.weight width " +
              "mismatch: expected donor_d_in=" + cfg.donor_d_in.to_s +
              " (the width realize_scratch! sized the lens at) but " +
              "donor llama.embedding_length=" + donor_d.to_s + " (" +
              donor_gguf_path + ")"
      end
      te_idx = TinyNN.tnn_gguf_find_index(ggh, "token_embd.weight")
      if te_idx < 0
        TinyNN.tnn_gguf_free(ggh)
        raise "WarmStart#realize_warm!: donor has no " +
              "token_embd.weight tensor (" + donor_gguf_path + ")"
      end
      n_floats = cfg.vocab * donor_d
      te_buf = Mat.new(1, n_floats)
      rc = TinyNN.tnn_gguf_read_f32_to_doubles(ggh, te_idx,
                                               te_buf.flat, n_floats)
      if rc != 0
        TinyNN.tnn_gguf_free(ggh)
        raise "WarmStart#realize_warm!: token_embd.weight read failed " +
              "rc=" + rc.to_s + " — wanted " + n_floats.to_s +
              " floats (vocab " + cfg.vocab.to_s + " x donor_d " +
              donor_d.to_s + ") from " + donor_gguf_path
      end
      upload_donor!(te_buf.flat, n_floats)
      TinyNN.tnn_gguf_free(ggh)
      nil
    end

    # The raw upload MECHANISM realize_warm! rides (and the seam for
    # already-read buffers — e.g. the legacy PCA-lens flow): one
    # tnn_upload_from_float_array into the realize'd token_embed table
    # (mirrors 09 L180). Same window rules as realize_warm!. The PCA
    # lens W_proj upload (09 L188-229) stays caller-side through
    # @ws_cache.sess + the t_seq_w_proj delegator. Returns nil.
    def upload_donor!(donor_buf_flat, n_floats)
      TinyNN.tnn_upload_from_float_array(@ws_cache.sess,
                                         @ws_cache.t_seq_token_embed,
                                         donor_buf_flat, n_floats)
      nil
    end

    # CLOSE the warm window: bake forward + CE + backward +
    # opt_step_adamw into the ggml graph (no Ruby Trainer/optimizer —
    # same rationale as FromScratch). Delegates VERBATIM to
    # build_training_step and stashes the returned [t_loss, t_labels,
    # t_hp] triple. Returns nil.
    def build!
      result       = @ws_cache.build_training_step
      @ws_t_loss   = result[0]
      @ws_t_labels = result[1]
      @ws_t_hp     = result[2]
      nil
    end

    # ONE training step. Op order is COPIED VERBATIM from
    # FromScratch#step! (from_scratch.rb:83-97) and LITERALLY IDENTICAL
    # to LoRA#step!: graph_reset on the first step else reset_grads_only;
    # the four uploads in order (token_ids/positions/labels/hp);
    # compute_backward; download_row_major(t_loss, 1, 1). is_first selects
    # the reset; the @ws_step_index accessor is carried for callers that
    # want it but is NOT used for the reset decision, so the caller stays
    # in full control of the step==0 branch. The per-step LR enters ONLY
    # via the caller mutating m_hp.flat[0] before this call — there is
    # deliberately NO lr param here (matches the siblings; keeps schedule
    # logic in the fixture). Returns the loss Float. Per-step input Mats
    # are built by the caller.
    def step!(seq_ids, positions, m_labels, m_hp, is_first)
      s = @ws_cache.sess
      if is_first
        TinyNN.tnn_graph_reset(s)
      else
        TinyNN.tnn_graph_reset_grads_only(s)
      end
      TinyNN.upload_int_array(s, @ws_cache.t_seq_token_ids, seq_ids)
      TinyNN.upload_int_array(s, @ws_cache.t_seq_positions, positions)
      TinyNN.upload_row_major(s, @ws_t_labels, m_labels)
      TinyNN.upload_row_major(s, @ws_t_hp,     m_hp)
      TinyNN.tnn_compute_backward(s)
      loss_mat = TinyNN.download_row_major(s, @ws_t_loss, 1, 1)
      loss_mat.flat[0]
    end
  end
end; end; end
