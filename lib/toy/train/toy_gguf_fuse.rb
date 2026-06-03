# P2.6 — head-fusing GGUF writer helper. ToyGGUFFuser converts a
# random_init Toy::LLM::Engine::LlamaSeqEngine (whose attention weights are
# named PER-HEAD: "blk.N.attn_q.head_H.weight", each a contiguous
# [d_head, d_model] F32 tensor) into the FUSED llama.cpp naming
# ("blk.N.attn_q.weight", a single [n_heads*d_head, d_model] tensor)
# that realize_for_mmap expects.
#
# Why this is the identity layout (NOT a reorder):
#   Each per-head tensor is allocated tnn_input_2d_f32_persistent(sess,
#   rows=d_head, cols=d_model): a fully-contiguous ggml tensor ne0=d_model,
#   ne1=d_head, i.e. d_head*d_model contiguous f32 in storage-element order.
#   On reload, realize_for_mmap reads head h at q_off_base +
#   h*head_nbytes(F32) where head_nbytes == d_head*d_model*4, and rebuilds a
#   view ne=[d_model,d_head] at that address. So the fused tensor on disk
#   must be head-0's d_head*d_model f32 block, then head-1's, ... — which is
#   exactly a single contiguous tensor ne0=d_model, ne1=n_heads*d_head
#   (Ruby rows=n_heads*d_head, cols=d_model). No transpose, no reorder.
#
# Lossless f32 round-trip: tnn_download_to_f64_array does dst[i] =
# (double)f32_storage[i] (exact f32->f64 widening); tnn_upload_from_float_array
# does scratch[i] = (float)data[i] (f64->f32 narrowing of an exactly-widened
# f32 returns the identical f32 bits). Both walk the LINEAR data buffer in
# storage-element order, so no transpose is introduced by the round-trip.
#
# F32-ONLY: this helper serialises the F32 params the random_init path
# produces. Q8 (head_nbytes type-8 branch) needs quantize-on-write the
# writer lacks and is explicitly out of scope.
#
# Spinel notes:
#   - No Struct.new (landmine #16); positional methods, no default args.
#   - The returned plist is built by pushing :ptr handles onto an array
#     seeded `[TinyNN.tnn_null_ptr]; pop` — the same pattern ToyDriftGrad
#     uses; Spinel infers sp_*_ptr_array. We do NOT construct an Array<:ptr>
#     literal inside the module (landmine #1).
#   - tnn_tensor_set_name (:str) is only ever called at runtime against a
#     passed session's finalized tensor, never at class-load scope
#     (project_step_bind_landmine_2026_05_28).
#   - Uniquely-prefixed locals (tgf_*) to dodge type-inference collisions.
module ToyGGUFFuser
  # Allocate every FUSED-name tensor in `write_sess`, finalize the write
  # session, then copy the F32 values across from `src_cache` (head-major
  # concat for attention weights, verbatim for everything else). Returns
  # the param-ordered Array<:ptr> of FUSED tensors living in `write_sess`,
  # ready to hand to ToyGGUFWriter.write.
  #
  # Args (no default args — Spinel):
  #   src_cache  : a realized Toy::LLM::Engine::LlamaSeqEngine (random_init, F32).
  #   write_sess : a fresh TinyNN.tnn_session_new(0); MUST stay alive
  #                until ToyGGUFWriter.write finalizes (gguf_add_tensor
  #                reads host data ptrs at finalize time).
  #   untied     : true => emit "output.weight"; false => tied.
  #
  # NOTE: src_cache.sess must ALSO stay alive across the whole call (we
  # download from it after write_sess is finalized). Both sessions are
  # held by the caller; we only read handles here.
  def self.build_fused_into_write_session(src_cache, write_sess, untied)
    tgf_d_model  = src_cache.seq_d_model
    tgf_d_ff     = src_cache.seq_d_ff
    tgf_d_head   = src_cache.seq_d_head
    tgf_n_heads  = src_cache.seq_n_heads
    tgf_n_kv     = src_cache.seq_n_kv
    tgf_vocab    = src_cache.seq_vocab_size
    tgf_layers   = src_cache.seq_n_layers

    # --- Phase 1: ALLOCATE fused tensors in write_sess (pre-finalize) ---
    # Arch-level globals first (mirrors realize_for_random_init order).
    tgf_w_embed = TinyNN.tnn_input_2d_f32_persistent(write_sess, tgf_vocab, tgf_d_model)
    tgf_w_fnorm = TinyNN.tnn_input_1d_f32_persistent(write_sess, tgf_d_model)
    tgf_w_out   = TinyNN.tnn_null_ptr
    if untied
      tgf_w_out = TinyNN.tnn_input_2d_f32_persistent(write_sess, tgf_vocab, tgf_d_model)
    end

    # Per-block fused tensors. Q is [n_heads*d_head, d_model]; K/V are
    # [n_kv*d_head, d_model]; o/gate/up/down keep their full 2D shapes.
    tgf_blk_rn1  = [TinyNN.tnn_null_ptr]; tgf_blk_rn1.pop
    tgf_blk_rn2  = [TinyNN.tnn_null_ptr]; tgf_blk_rn2.pop
    tgf_blk_q    = [TinyNN.tnn_null_ptr]; tgf_blk_q.pop
    tgf_blk_k    = [TinyNN.tnn_null_ptr]; tgf_blk_k.pop
    tgf_blk_v    = [TinyNN.tnn_null_ptr]; tgf_blk_v.pop
    tgf_blk_o    = [TinyNN.tnn_null_ptr]; tgf_blk_o.pop
    tgf_blk_gate = [TinyNN.tnn_null_ptr]; tgf_blk_gate.pop
    tgf_blk_up   = [TinyNN.tnn_null_ptr]; tgf_blk_up.pop
    tgf_blk_down = [TinyNN.tnn_null_ptr]; tgf_blk_down.pop

    tgf_li = 0
    while tgf_li < tgf_layers
      tgf_blk_rn1.push(TinyNN.tnn_input_1d_f32_persistent(write_sess, tgf_d_model))
      tgf_blk_rn2.push(TinyNN.tnn_input_1d_f32_persistent(write_sess, tgf_d_model))
      tgf_blk_q.push(TinyNN.tnn_input_2d_f32_persistent(write_sess,
                       tgf_n_heads * tgf_d_head, tgf_d_model))
      tgf_blk_k.push(TinyNN.tnn_input_2d_f32_persistent(write_sess,
                       tgf_n_kv * tgf_d_head, tgf_d_model))
      tgf_blk_v.push(TinyNN.tnn_input_2d_f32_persistent(write_sess,
                       tgf_n_kv * tgf_d_head, tgf_d_model))
      tgf_blk_o.push(TinyNN.tnn_input_2d_f32_persistent(write_sess,
                       tgf_d_model, tgf_d_model))
      tgf_blk_gate.push(TinyNN.tnn_input_2d_f32_persistent(write_sess,
                          tgf_d_ff, tgf_d_model))
      tgf_blk_up.push(TinyNN.tnn_input_2d_f32_persistent(write_sess,
                        tgf_d_ff, tgf_d_model))
      tgf_blk_down.push(TinyNN.tnn_input_2d_f32_persistent(write_sess,
                          tgf_d_model, tgf_d_ff))
      tgf_li = tgf_li + 1
    end

    TinyNN.tnn_finalize_weights(write_sess)

    # --- Phase 2: COPY values across + set FUSED names ---
    # Globals — verbatim element-for-element (same shape both sides).
    copy_verbatim(src_cache.sess, src_cache.t_seq_token_embed,
                  write_sess, tgf_w_embed, tgf_vocab * tgf_d_model)
    TinyNN.tnn_tensor_set_name(tgf_w_embed, "token_embd.weight")

    copy_verbatim(src_cache.sess, src_cache.t_seq_final_norm_gamma,
                  write_sess, tgf_w_fnorm, tgf_d_model)
    TinyNN.tnn_tensor_set_name(tgf_w_fnorm, "output_norm.weight")

    if untied
      copy_verbatim(src_cache.sess, src_cache.t_seq_output,
                    write_sess, tgf_w_out, tgf_vocab * tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_w_out, "output.weight")
    end

    tgf_li2 = 0
    while tgf_li2 < tgf_layers
      tgf_src_blk = src_cache.seq_blocks_ffi[tgf_li2]
      tgf_prefix  = "blk." + tgf_li2.to_s + "."

      copy_verbatim(src_cache.sess, tgf_src_blk.t_seq_rn1_gamma,
                    write_sess, tgf_blk_rn1[tgf_li2], tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_blk_rn1[tgf_li2], tgf_prefix + "attn_norm.weight")

      copy_verbatim(src_cache.sess, tgf_src_blk.t_seq_rn2_gamma,
                    write_sess, tgf_blk_rn2[tgf_li2], tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_blk_rn2[tgf_li2], tgf_prefix + "ffn_norm.weight")

      # Head-major concat: head h's d_head*d_model block lands at element
      # offset h*d_head*d_model == byte offset h*head_nbytes(F32) — exactly
      # the slice offset realize_for_mmap re-reads.
      copy_heads_concat(src_cache.sess, tgf_src_blk.t_seq_w_q, tgf_n_heads,
                        write_sess, tgf_blk_q[tgf_li2], tgf_d_head, tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_blk_q[tgf_li2], tgf_prefix + "attn_q.weight")

      copy_heads_concat(src_cache.sess, tgf_src_blk.t_seq_w_k, tgf_n_kv,
                        write_sess, tgf_blk_k[tgf_li2], tgf_d_head, tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_blk_k[tgf_li2], tgf_prefix + "attn_k.weight")

      copy_heads_concat(src_cache.sess, tgf_src_blk.t_seq_w_v, tgf_n_kv,
                        write_sess, tgf_blk_v[tgf_li2], tgf_d_head, tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_blk_v[tgf_li2], tgf_prefix + "attn_v.weight")

      copy_verbatim(src_cache.sess, tgf_src_blk.t_seq_w_o,
                    write_sess, tgf_blk_o[tgf_li2], tgf_d_model * tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_blk_o[tgf_li2], tgf_prefix + "attn_output.weight")

      copy_verbatim(src_cache.sess, tgf_src_blk.t_seq_w_gate,
                    write_sess, tgf_blk_gate[tgf_li2], tgf_d_ff * tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_blk_gate[tgf_li2], tgf_prefix + "ffn_gate.weight")

      copy_verbatim(src_cache.sess, tgf_src_blk.t_seq_w_up,
                    write_sess, tgf_blk_up[tgf_li2], tgf_d_ff * tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_blk_up[tgf_li2], tgf_prefix + "ffn_up.weight")

      copy_verbatim(src_cache.sess, tgf_src_blk.t_seq_w_down,
                    write_sess, tgf_blk_down[tgf_li2], tgf_d_model * tgf_d_ff)
      TinyNN.tnn_tensor_set_name(tgf_blk_down[tgf_li2], tgf_prefix + "ffn_down.weight")

      tgf_li2 = tgf_li2 + 1
    end

    # --- Phase 3: build the param-ordered plist (push, never literal) ---
    tgf_plist = [TinyNN.tnn_null_ptr]; tgf_plist.pop
    tgf_plist.push(tgf_w_embed)
    tgf_plist.push(tgf_w_fnorm)
    if untied
      tgf_plist.push(tgf_w_out)
    end
    tgf_li3 = 0
    while tgf_li3 < tgf_layers
      tgf_plist.push(tgf_blk_rn1[tgf_li3])
      tgf_plist.push(tgf_blk_rn2[tgf_li3])
      tgf_plist.push(tgf_blk_q[tgf_li3])
      tgf_plist.push(tgf_blk_k[tgf_li3])
      tgf_plist.push(tgf_blk_v[tgf_li3])
      tgf_plist.push(tgf_blk_o[tgf_li3])
      tgf_plist.push(tgf_blk_gate[tgf_li3])
      tgf_plist.push(tgf_blk_up[tgf_li3])
      tgf_plist.push(tgf_blk_down[tgf_li3])
      tgf_li3 = tgf_li3 + 1
    end
    tgf_plist
  end

  # P4 — projection-lens variant of build_fused_into_write_session, for
  # the from-scratch / warm-start RANDOM-INIT recipes that train under a
  # projection lens (cfg.donor_d_in > 0). In that recipe the on-session
  # token_embed is a FROZEN donor table [vocab, donor_d_in] and the
  # TRAINABLE lens.proj.weight [donor_d_in, d_model] sits between
  # get_rows and the first block (matmul(W_proj, embed) → d_model). The
  # plain fuser would emit a [vocab, donor_d_in] embed + a lens.proj
  # tensor that realize_for_mmap does not know how to load.
  #
  # This method FOLDS the lens into the embedding at write time so the
  # checkpoint is a STANDARD fused-llama GGUF (token_embd.weight is the
  # already-projected [vocab, d_model] table, NO lens.proj). The fold is
  # mathematically EXACT and matches the train-forward lens:
  #   ggml matmul(W_proj, x) with W_proj ne=[donor, d_model] and
  #   x=embed_donor ne=[donor, T] gives out[r,t] = sum_c W_proj[c,r]*embed[c,t]
  #   (contraction on ne[0]=donor). Per-row v:
  #     embed_eff[v, r] = sum_c embed_donor[v, c] * W_proj[c, r]
  #   In ggml storage order (ne0 = inner contiguous):
  #     embed_donor element [v*donor + c]   (ne0=donor, ne1=vocab)
  #     W_proj      element [r*donor + c]   (ne0=donor, ne1=d_model)
  #     embed_eff   element [v*d_model + r] (ne0=d_model, ne1=vocab)
  #
  # Everything ELSE (per-block fused attention + FFN + norms + untied
  # output) is byte-identical to build_fused_into_write_session — only
  # the embed copy is replaced by the fold, and lens.proj is dropped.
  #
  # Args (no default args — Spinel):
  #   src_cache  : a realized Toy::LLM::Engine::LlamaSeqEngine, donor_d_in > 0, F32.
  #   write_sess : fresh TinyNN.tnn_session_new(0); MUST stay alive until
  #                ToyGGUFWriter.write finalizes.
  #   untied     : true => emit "output.weight" (required when donor>0).
  def self.build_lens_folded_into_write_session(src_cache, write_sess, untied)
    tgf_d_model  = src_cache.seq_d_model
    tgf_d_ff     = src_cache.seq_d_ff
    tgf_d_head   = src_cache.seq_d_head
    tgf_n_heads  = src_cache.seq_n_heads
    tgf_n_kv     = src_cache.seq_n_kv
    tgf_vocab    = src_cache.seq_vocab_size
    tgf_layers   = src_cache.seq_n_layers
    tgf_donor    = src_cache.seq_donor_d_in

    # --- Fold the lens into an effective [vocab, d_model] embedding ---
    # Download the donor table (ne0=donor, ne1=vocab) and the lens
    # (ne0=donor, ne1=d_model), both f32->f64 (exact), linear storage.
    tgf_embed_n = tgf_vocab * tgf_donor
    tgf_proj_n  = tgf_d_model * tgf_donor
    tgf_embed_donor = Mat.new(1, tgf_embed_n)
    tgf_proj        = Mat.new(1, tgf_proj_n)
    TinyNN.tnn_download_to_f64_array(src_cache.sess, src_cache.t_seq_token_embed,
                                     tgf_embed_donor.flat, tgf_embed_n)
    TinyNN.tnn_download_to_f64_array(src_cache.sess, src_cache.t_seq_w_proj,
                                     tgf_proj.flat, tgf_proj_n)

    # embed_eff[v*d_model + r] = sum_c donor[v*donor+c] * proj[r*donor+c]
    tgf_eff_n  = tgf_vocab * tgf_d_model
    tgf_embed_eff = Mat.new(1, tgf_eff_n)
    tgf_v = 0
    while tgf_v < tgf_vocab
      tgf_vbase = tgf_v * tgf_donor
      tgf_obase = tgf_v * tgf_d_model
      tgf_r = 0
      while tgf_r < tgf_d_model
        tgf_rbase = tgf_r * tgf_donor
        tgf_acc = 0.0
        tgf_c = 0
        while tgf_c < tgf_donor
          tgf_acc = tgf_acc + tgf_embed_donor.flat[tgf_vbase + tgf_c] *
                              tgf_proj.flat[tgf_rbase + tgf_c]
          tgf_c = tgf_c + 1
        end
        tgf_embed_eff.flat[tgf_obase + tgf_r] = tgf_acc
        tgf_r = tgf_r + 1
      end
      tgf_v = tgf_v + 1
    end

    # --- Phase 1: ALLOCATE fused tensors in write_sess (pre-finalize) ---
    # token_embd is now the STANDARD [vocab, d_model] table — NO lens.
    tgf_w_embed = TinyNN.tnn_input_2d_f32_persistent(write_sess, tgf_vocab, tgf_d_model)
    tgf_w_fnorm = TinyNN.tnn_input_1d_f32_persistent(write_sess, tgf_d_model)
    tgf_w_out   = TinyNN.tnn_null_ptr
    if untied
      tgf_w_out = TinyNN.tnn_input_2d_f32_persistent(write_sess, tgf_vocab, tgf_d_model)
    end

    tgf_blk_rn1  = [TinyNN.tnn_null_ptr]; tgf_blk_rn1.pop
    tgf_blk_rn2  = [TinyNN.tnn_null_ptr]; tgf_blk_rn2.pop
    tgf_blk_q    = [TinyNN.tnn_null_ptr]; tgf_blk_q.pop
    tgf_blk_k    = [TinyNN.tnn_null_ptr]; tgf_blk_k.pop
    tgf_blk_v    = [TinyNN.tnn_null_ptr]; tgf_blk_v.pop
    tgf_blk_o    = [TinyNN.tnn_null_ptr]; tgf_blk_o.pop
    tgf_blk_gate = [TinyNN.tnn_null_ptr]; tgf_blk_gate.pop
    tgf_blk_up   = [TinyNN.tnn_null_ptr]; tgf_blk_up.pop
    tgf_blk_down = [TinyNN.tnn_null_ptr]; tgf_blk_down.pop

    tgf_li = 0
    while tgf_li < tgf_layers
      tgf_blk_rn1.push(TinyNN.tnn_input_1d_f32_persistent(write_sess, tgf_d_model))
      tgf_blk_rn2.push(TinyNN.tnn_input_1d_f32_persistent(write_sess, tgf_d_model))
      tgf_blk_q.push(TinyNN.tnn_input_2d_f32_persistent(write_sess,
                       tgf_n_heads * tgf_d_head, tgf_d_model))
      tgf_blk_k.push(TinyNN.tnn_input_2d_f32_persistent(write_sess,
                       tgf_n_kv * tgf_d_head, tgf_d_model))
      tgf_blk_v.push(TinyNN.tnn_input_2d_f32_persistent(write_sess,
                       tgf_n_kv * tgf_d_head, tgf_d_model))
      tgf_blk_o.push(TinyNN.tnn_input_2d_f32_persistent(write_sess,
                       tgf_d_model, tgf_d_model))
      tgf_blk_gate.push(TinyNN.tnn_input_2d_f32_persistent(write_sess,
                          tgf_d_ff, tgf_d_model))
      tgf_blk_up.push(TinyNN.tnn_input_2d_f32_persistent(write_sess,
                        tgf_d_ff, tgf_d_model))
      tgf_blk_down.push(TinyNN.tnn_input_2d_f32_persistent(write_sess,
                          tgf_d_model, tgf_d_ff))
      tgf_li = tgf_li + 1
    end

    TinyNN.tnn_finalize_weights(write_sess)

    # --- Phase 2: COPY values across + set FUSED names ---
    # token_embd is the FOLDED embed_eff (upload directly, NOT verbatim).
    TinyNN.tnn_upload_from_float_array(write_sess, tgf_w_embed,
                                       tgf_embed_eff.flat, tgf_eff_n)
    TinyNN.tnn_tensor_set_name(tgf_w_embed, "token_embd.weight")

    copy_verbatim(src_cache.sess, src_cache.t_seq_final_norm_gamma,
                  write_sess, tgf_w_fnorm, tgf_d_model)
    TinyNN.tnn_tensor_set_name(tgf_w_fnorm, "output_norm.weight")

    if untied
      copy_verbatim(src_cache.sess, src_cache.t_seq_output,
                    write_sess, tgf_w_out, tgf_vocab * tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_w_out, "output.weight")
    end

    tgf_li2 = 0
    while tgf_li2 < tgf_layers
      tgf_src_blk = src_cache.seq_blocks_ffi[tgf_li2]
      tgf_prefix  = "blk." + tgf_li2.to_s + "."

      copy_verbatim(src_cache.sess, tgf_src_blk.t_seq_rn1_gamma,
                    write_sess, tgf_blk_rn1[tgf_li2], tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_blk_rn1[tgf_li2], tgf_prefix + "attn_norm.weight")

      copy_verbatim(src_cache.sess, tgf_src_blk.t_seq_rn2_gamma,
                    write_sess, tgf_blk_rn2[tgf_li2], tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_blk_rn2[tgf_li2], tgf_prefix + "ffn_norm.weight")

      copy_heads_concat(src_cache.sess, tgf_src_blk.t_seq_w_q, tgf_n_heads,
                        write_sess, tgf_blk_q[tgf_li2], tgf_d_head, tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_blk_q[tgf_li2], tgf_prefix + "attn_q.weight")

      copy_heads_concat(src_cache.sess, tgf_src_blk.t_seq_w_k, tgf_n_kv,
                        write_sess, tgf_blk_k[tgf_li2], tgf_d_head, tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_blk_k[tgf_li2], tgf_prefix + "attn_k.weight")

      copy_heads_concat(src_cache.sess, tgf_src_blk.t_seq_w_v, tgf_n_kv,
                        write_sess, tgf_blk_v[tgf_li2], tgf_d_head, tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_blk_v[tgf_li2], tgf_prefix + "attn_v.weight")

      copy_verbatim(src_cache.sess, tgf_src_blk.t_seq_w_o,
                    write_sess, tgf_blk_o[tgf_li2], tgf_d_model * tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_blk_o[tgf_li2], tgf_prefix + "attn_output.weight")

      copy_verbatim(src_cache.sess, tgf_src_blk.t_seq_w_gate,
                    write_sess, tgf_blk_gate[tgf_li2], tgf_d_ff * tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_blk_gate[tgf_li2], tgf_prefix + "ffn_gate.weight")

      copy_verbatim(src_cache.sess, tgf_src_blk.t_seq_w_up,
                    write_sess, tgf_blk_up[tgf_li2], tgf_d_ff * tgf_d_model)
      TinyNN.tnn_tensor_set_name(tgf_blk_up[tgf_li2], tgf_prefix + "ffn_up.weight")

      copy_verbatim(src_cache.sess, tgf_src_blk.t_seq_w_down,
                    write_sess, tgf_blk_down[tgf_li2], tgf_d_model * tgf_d_ff)
      TinyNN.tnn_tensor_set_name(tgf_blk_down[tgf_li2], tgf_prefix + "ffn_down.weight")

      tgf_li2 = tgf_li2 + 1
    end

    # --- Phase 3: build the param-ordered plist (push, never literal) ---
    tgf_plist = [TinyNN.tnn_null_ptr]; tgf_plist.pop
    tgf_plist.push(tgf_w_embed)
    tgf_plist.push(tgf_w_fnorm)
    if untied
      tgf_plist.push(tgf_w_out)
    end
    tgf_li3 = 0
    while tgf_li3 < tgf_layers
      tgf_plist.push(tgf_blk_rn1[tgf_li3])
      tgf_plist.push(tgf_blk_rn2[tgf_li3])
      tgf_plist.push(tgf_blk_q[tgf_li3])
      tgf_plist.push(tgf_blk_k[tgf_li3])
      tgf_plist.push(tgf_blk_v[tgf_li3])
      tgf_plist.push(tgf_blk_o[tgf_li3])
      tgf_plist.push(tgf_blk_gate[tgf_li3])
      tgf_plist.push(tgf_blk_up[tgf_li3])
      tgf_plist.push(tgf_blk_down[tgf_li3])
      tgf_li3 = tgf_li3 + 1
    end
    tgf_plist
  end

  # Download `n` f32 elements from src tensor (f32->f64), upload them
  # into dst (f64->f32). Both walk linear storage order, so this is an
  # exact element-for-element copy when src and dst have the same total
  # element count.
  def self.copy_verbatim(src_sess, src_t, dst_sess, dst_t, n)
    tgf_buf = Mat.new(1, n)
    TinyNN.tnn_download_to_f64_array(src_sess, src_t, tgf_buf.flat, n)
    TinyNN.tnn_upload_from_float_array(dst_sess, dst_t, tgf_buf.flat, n)
  end

  # Concatenate `n_heads` per-head [d_head, d_model] tensors (head order
  # 0..n_heads-1) into one linear buffer, then upload into the fused dst
  # tensor [n_heads*d_head, d_model]. head h's d_head*d_model block lands
  # at element offset h*d_head*d_model.
  def self.copy_heads_concat(src_sess, src_head_arr, n_heads, dst_sess, dst_t, d_head, d_model)
    tgf_per   = d_head * d_model
    tgf_total = n_heads * tgf_per
    tgf_buf   = Mat.new(1, tgf_total)
    tgf_tmp   = Mat.new(1, tgf_per)
    tgf_h = 0
    while tgf_h < n_heads
      TinyNN.tnn_download_to_f64_array(src_sess, src_head_arr[tgf_h], tgf_tmp.flat, tgf_per)
      tgf_base = tgf_h * tgf_per
      tgf_e = 0
      while tgf_e < tgf_per
        tgf_buf.flat[tgf_base + tgf_e] = tgf_tmp.flat[tgf_e]
        tgf_e = tgf_e + 1
      end
      tgf_h = tgf_h + 1
    end
    TinyNN.tnn_upload_from_float_array(dst_sess, dst_t, tgf_buf.flat, tgf_total)
  end
end
