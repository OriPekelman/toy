# tao#gguf-checkpoint-writer — GGUF snapshot writer for training runs.
#
# Writes $TAO_RUN_DIR/weights/step_N.gguf on a schedule (CHECKPOINT_EVERY=N),
# plus a final snapshot at run_end. Maintains $TAO_RUN_DIR/weights/latest
# as a symlink to the most recent file.
#
# Format: thin wrap around ggml's gguf writer (tinynn/tinynn_ggml.c).
# Caller supplies the model arch + hyperparams (which become GGUF metadata
# KV pairs) and the session whose PARAM tensors should be serialised.
#
# Loadable-by-inference status: the GGUF we write is structurally correct
# (parseable via gguf_init_from_file, all metadata + tensor data present),
# but toy's inference loader (lib/toy_smollm2_loader.rb) expects llama.cpp
# tensor naming convention (`blk.N.attn_q.weight`, etc.) and per-LAYER
# fused tensors — neither of which matches toy's per-head training graph.
# Bridging that is tracked as toy#gguf-checkpoint-reload (filed separately).
#
# Spinel notes:
#   - Naming uses simple String.concat; no #{} interpolation.
#   - Plist (Array<:ptr>) flows from the caller; we don't construct
#     Array<:ptr> inside the module to avoid recurring landmine #1.

module ToyGGUFWriter
  # Name each PARAM tensor with a stable "param_N" convention, where N
  # is the param's position in the graph walk. This is opaque to
  # llama.cpp but deterministic and self-describing across writes (a
  # given seed → same graph order → same name assignment), so a
  # toy-side reloader can match by position.
  def self.name_params(plist)
    i = 0
    while i < plist.length
      n = "param_" + i.to_s
      TinyNN.tnn_tensor_set_name(plist[i], n)
      i = i + 1
    end
  end

  # Write a checkpoint. `cfg` carries the model hyperparams; `plist`
  # is the param-ordered tensor pointer array (from ToyDriftGrad.params
  # or ToyDescribeFlow's index builder); `path` is the destination
  # GGUF (caller manages directory creation + naming convention).
  # Returns 0 on success, negative on failure.
  def self.write(cfg, plist, path, run_id, step)
    ctx = TinyNN.tnn_gguf_w_init
    if ctx == nil || ctx == TinyNN.tnn_null_ptr
      return -1
    end

    # Standard arch metadata — enough for downstream tooling to know
    # what shape was trained. We use "llama" so future Tao tooling
    # that sniffs general.architecture has something familiar.
    TinyNN.tnn_gguf_w_set_str(ctx, "general.architecture",  "llama")
    TinyNN.tnn_gguf_w_set_str(ctx, "general.name",          "toy-from-scratch")
    TinyNN.tnn_gguf_w_set_str(ctx, "general.run_id",        run_id)
    TinyNN.tnn_gguf_w_set_u32(ctx, "general.step",          step)
    TinyNN.tnn_gguf_w_set_u32(ctx, "llama.vocab_size",      cfg.vocab)
    TinyNN.tnn_gguf_w_set_u32(ctx, "llama.embedding_length",     cfg.d_model)
    TinyNN.tnn_gguf_w_set_u32(ctx, "llama.feed_forward_length",  cfg.d_ff)
    TinyNN.tnn_gguf_w_set_u32(ctx, "llama.block_count",     cfg.n_layers)
    TinyNN.tnn_gguf_w_set_u32(ctx, "llama.attention.head_count",     cfg.n_heads)
    TinyNN.tnn_gguf_w_set_u32(ctx, "llama.attention.head_count_kv",  cfg.n_kv)
    TinyNN.tnn_gguf_w_set_u32(ctx, "llama.context_length",  cfg.ctx)
    TinyNN.tnn_gguf_w_set_f32(ctx, "llama.attention.layer_norm_rms_epsilon", cfg.eps)
    TinyNN.tnn_gguf_w_set_f32(ctx, "llama.rope.freq_base",  cfg.rope_base)
    # Provenance — the toy-side checkpoint format version.
    TinyNN.tnn_gguf_w_set_str(ctx, "toy.checkpoint_format", "toy-from-scratch/v1")
    TinyNN.tnn_gguf_w_set_u32(ctx, "toy.n_params_written",  plist.length)

    # Name each param and add it.
    name_params(plist)
    i = 0
    while i < plist.length
      TinyNN.tnn_gguf_w_add_tensor(ctx, plist[i])
      i = i + 1
    end

    rc = TinyNN.tnn_gguf_w_finalize(ctx, path)
    TinyNN.tnn_gguf_w_free(ctx)
    rc
  end

  # Convenience: ensure $TAO_RUN_DIR/weights/ exists, then write the
  # checkpoint + update the `latest` symlink. `weights_dir` is the
  # full path (e.g. "/tmp/runs/abc/weights"). Returns 0 on success.
  def self.write_step(cfg, plist, weights_dir, run_id, step)
    TinyNN.tnn_filesystem_mkdir(weights_dir)
    fname = "step_" + step.to_s + ".gguf"
    fpath = weights_dir + "/" + fname
    rc = write(cfg, plist, fpath, run_id, step)
    if rc == 0
      # latest → step_N.gguf (relative target so it works under symlinked
      # weights dirs and across rsync moves).
      lpath = weights_dir + "/latest"
      TinyNN.tnn_filesystem_symlink(fname, lpath)
    end
    rc
  end
end
