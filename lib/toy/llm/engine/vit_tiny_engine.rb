# lib/toy/llm/engine/vit_tiny_engine.rb — ViT-Tiny forward + training graph
# (towards GH#13 / E1.3).
#
# Single image (N=1) per training step for now; batching is the
# transformer-side GH#7 work and lands separately.
#
# Forward pipeline:
#   image [W, H, IC, 1]
#     → patch_embed (conv2d+permute+cont_2d)        [d_model, N_patches]
#     → prepend cls token                            [d_model, N_patches+1]
#     → add learned pos_embed                        [d_model, T] where T=N_patches+1
#     → 12 ViT blocks (LayerNorm + MSA + MLP-GELU)   [d_model, T]
#     → final LayerNorm                              [d_model, T]
#     → take cls token (column 0)                    [d_model]
#     → matmul w_head                                [num_classes]
#
# Training step adds CE loss vs one-hot target_class + backward +
# AdamW opt_step on every PARAM. Reuses GH#17's session graph
# capacity sizing (n_layers × n_heads * 1000 + 65536).
#
# Spinel landmines respected: no Array<Array<mixed>> seeds, all
# helper arrays seed-and-pop with concrete element types.

require_relative "../../../transformer"
require_relative "../../../tinynn"
require_relative "../../../toy_vit"
require_relative "../../../toy_smollm2"   # for Toy::RopeScaling stub (unused, but consistent)

module Toy; module LLM; module Engine
class ViTTinyBlockFFI
  # NOTE on norm choice: ggml has no backward for GGML_OP_NORM
  # (LayerNorm); only GGML_OP_RMS_NORM has a registered backward.
  # We use RMSNorm in the training path. This is a documented ViT
  # variant (e.g. timm's `vit_giant_patch14_clip_224` uses LayerScale +
  # variants; LayerNorm/RMSNorm both train; LayerNorm is the timm
  # default but not load-bearing for the recipe). For E1 acceptance
  # (loss decreases on a memorisation smoke) this swap is fine. The
  # follow-up to ship a vendored LayerNorm backward — and the
  # timm-loader compatibility that goes with it — is its own issue.
  attr_accessor :t_ln1_gamma, :t_ln2_gamma,
                :t_w_q, :t_w_k, :t_w_v, :t_w_o,
                :t_w_up, :t_w_down,
                :ft_weights, :ft_m, :ft_v

  def initialize
    @t_ln1_gamma = TinyNN.tnn_null_ptr
    @t_ln2_gamma = TinyNN.tnn_null_ptr
    @t_w_q = [TinyNN.tnn_null_ptr]
    @t_w_k = [TinyNN.tnn_null_ptr]
    @t_w_v = [TinyNN.tnn_null_ptr]
    @t_w_o    = TinyNN.tnn_null_ptr
    @t_w_up   = TinyNN.tnn_null_ptr
    @t_w_down = TinyNN.tnn_null_ptr
    @ft_weights = [TinyNN.tnn_null_ptr]; @ft_weights.pop
    @ft_m       = [TinyNN.tnn_null_ptr]; @ft_m.pop
    @ft_v       = [TinyNN.tnn_null_ptr]; @ft_v.pop
  end
end

class ViTTinyEngine
  attr_accessor :sess,
                :cfg, :n_patches, :seq_t,
                :t_patch_kernel, :t_cls_token, :t_pos_embed,
                :t_final_ln_gamma,
                :t_w_head, :t_image, :t_logits,
                :ft_globals_weights, :ft_globals_m, :ft_globals_v,
                :blocks, :realized

  def initialize
    @sess = TinyNN.tnn_null_ptr
    @ft_globals_weights = [TinyNN.tnn_null_ptr]; @ft_globals_weights.pop
    @ft_globals_m       = [TinyNN.tnn_null_ptr]; @ft_globals_m.pop
    @ft_globals_v       = [TinyNN.tnn_null_ptr]; @ft_globals_v.pop
    @blocks   = [ViTTinyBlockFFI.new]; @blocks.pop
    @realized = false
  end

  # Allocate every PARAM + Adam moments. Random-init at the end via
  # Box-Muller; the caller can overwrite specific tensors (e.g.
  # patch_embed.proj.weight from a timm donor) before the first step.
  def realize_for_random_init(cfg, seed, init_scale)
    @cfg = cfg
    # T = number of "tokens" entering the transformer = N_patches + 1 (cls).
    @n_patches = (cfg.image_size / cfg.patch_size) * (cfg.image_size / cfg.patch_size)
    @seq_t     = @n_patches + 1

    @sess = TinyNN.tnn_session_new(0)
    # Per-head decomposition pushes node count up; budget from cfg.
    cap = cfg.n_layers * cfg.n_heads * 1000 + 65536
    TinyNN.tnn_session_set_graph_capacity(@sess, cap)

    # Globals: patch_embed weight, cls_token, pos_embed, final norm, head.
    #
    # patch_embed is implemented as a linear projection over the
    # flattened patch axis (mathematically equivalent to a Conv2d with
    # stride=patch, no overlap, no padding). We don't use ggml_conv_2d
    # in the training path because its internal implementation ends in
    # a `ggml_cont(ggml_permute(...))` and ggml's auto-backward
    # requires gradients into `cont` to be contiguous — the assertion
    # fires at `tnn_build_backward` time. The flat-linear form has a
    # clean matmul backward.
    #
    # Caller responsibility: pre-flatten each patch into
    # [IC*P*P, N_patches] before uploading to t_image. Done host-side
    # (cheap; see examples/smoke_vit_tiny.rb).
    @patch_flat_dim = cfg.num_channels * cfg.patch_size * cfg.patch_size
    @t_patch_kernel = TinyNN.tnn_input_2d_f32_persistent(@sess,
                        cfg.d_model, @patch_flat_dim)
    ft_add_global_2d(@t_patch_kernel, cfg.d_model, @patch_flat_dim)
    ft_name_last_global("patch_embed.proj.weight")

    @t_cls_token = TinyNN.tnn_input_2d_f32_persistent(@sess, 1, cfg.d_model)
    ft_add_global_2d(@t_cls_token, 1, cfg.d_model)
    ft_name_last_global("cls_token")

    @t_pos_embed = TinyNN.tnn_input_2d_f32_persistent(@sess, @seq_t, cfg.d_model)
    ft_add_global_2d(@t_pos_embed, @seq_t, cfg.d_model)
    ft_name_last_global("pos_embed")

    @t_final_ln_gamma = TinyNN.tnn_input_1d_f32_persistent(@sess, cfg.d_model)
    ft_add_global_1d(@t_final_ln_gamma)
    ft_name_last_global("norm.weight")

    @t_w_head = TinyNN.tnn_input_2d_f32_persistent(@sess, cfg.num_classes, cfg.d_model)
    ft_add_global_2d(@t_w_head, cfg.num_classes, cfg.d_model)
    ft_name_last_global("head.weight")

    # Per-block weights.
    li_init = 0
    while li_init < cfg.n_layers
      @blocks.push(ViTTinyBlockFFI.new)
      li_init = li_init + 1
    end

    li = 0
    while li < cfg.n_layers
      blk = @blocks[li]
      prefix = "blk." + li.to_s + "."

      blk.t_ln1_gamma = TinyNN.tnn_input_1d_f32_persistent(@sess, cfg.d_model)
      blk.t_ln2_gamma = TinyNN.tnn_input_1d_f32_persistent(@sess, cfg.d_model)
      ft_add_1d(blk, blk.t_ln1_gamma); ft_name_last(blk, prefix + "ln1.weight")
      ft_add_1d(blk, blk.t_ln2_gamma); ft_name_last(blk, prefix + "ln2.weight")

      # Per-head Q/K/V. n_kv = n_heads (no GQA in ViT-Tiny).
      blk.t_w_q = [TinyNN.tnn_input_2d_f32_persistent(@sess, cfg.d_head, cfg.d_model)]
      blk.t_w_k = [TinyNN.tnn_input_2d_f32_persistent(@sess, cfg.d_head, cfg.d_model)]
      blk.t_w_v = [TinyNN.tnn_input_2d_f32_persistent(@sess, cfg.d_head, cfg.d_model)]
      h = 1
      while h < cfg.n_heads
        blk.t_w_q.push(TinyNN.tnn_input_2d_f32_persistent(@sess, cfg.d_head, cfg.d_model))
        blk.t_w_k.push(TinyNN.tnn_input_2d_f32_persistent(@sess, cfg.d_head, cfg.d_model))
        blk.t_w_v.push(TinyNN.tnn_input_2d_f32_persistent(@sess, cfg.d_head, cfg.d_model))
        h = h + 1
      end
      h2 = 0
      while h2 < cfg.n_heads
        ft_add_2d(blk, blk.t_w_q[h2], cfg.d_head, cfg.d_model)
        ft_name_last(blk, prefix + "attn_q.head_" + h2.to_s + ".weight")
        ft_add_2d(blk, blk.t_w_k[h2], cfg.d_head, cfg.d_model)
        ft_name_last(blk, prefix + "attn_k.head_" + h2.to_s + ".weight")
        ft_add_2d(blk, blk.t_w_v[h2], cfg.d_head, cfg.d_model)
        ft_name_last(blk, prefix + "attn_v.head_" + h2.to_s + ".weight")
        h2 = h2 + 1
      end

      blk.t_w_o    = TinyNN.tnn_input_2d_f32_persistent(@sess, cfg.d_model, cfg.d_model)
      blk.t_w_up   = TinyNN.tnn_input_2d_f32_persistent(@sess, cfg.d_ff,    cfg.d_model)
      blk.t_w_down = TinyNN.tnn_input_2d_f32_persistent(@sess, cfg.d_model, cfg.d_ff)
      ft_add_2d(blk, blk.t_w_o,    cfg.d_model, cfg.d_model)
      ft_name_last(blk, prefix + "attn_output.weight")
      ft_add_2d(blk, blk.t_w_up,   cfg.d_ff,    cfg.d_model)
      ft_name_last(blk, prefix + "mlp_up.weight")
      ft_add_2d(blk, blk.t_w_down, cfg.d_model, cfg.d_ff)
      ft_name_last(blk, prefix + "mlp_down.weight")

      # Mark every block weight as PARAM.
      wi = 0
      while wi < blk.ft_weights.length
        TinyNN.tnn_set_param(blk.ft_weights[wi])
        wi = wi + 1
      end
      li = li + 1
    end

    # Mark globals as PARAM.
    gi = 0
    while gi < @ft_globals_weights.length
      TinyNN.tnn_set_param(@ft_globals_weights[gi])
      gi = gi + 1
    end

    # Image input (graph input, NOT a PARAM): pre-flattened patch
    # matrix shape [IC*P*P, N_patches]. Caller does the host-side
    # patch extraction; see examples/smoke_vit_tiny.rb.
    @t_image = TinyNN.tnn_input_2d_f32_persistent(@sess,
                  @n_patches, @patch_flat_dim)

    TinyNN.tnn_finalize_weights(@sess)

    # Random-init every PARAM tensor.
    upload_random_init!(seed, init_scale)
    @realized = true
  end

  def build_forward_in_current_ctx
    cfg = @cfg
    # patch_embed as a flat linear: t_image ne=[IC*P*P, N_patches],
    # W_patch ne=[IC*P*P, d_model] → matmul gives ne=[d_model, N_patches].
    t_patches = TinyNN.tnn_matmul(@sess, @t_patch_kernel, @t_image)
    # Prepend cls token via concat along the sequence (ne[1]) axis.
    # cls token ne=[d_model, 1]; t_patches ne=[d_model, N_patches].
    # concat axis=1 (the "sequence" axis): ne=[d_model, 1 + N_patches]
    t_seq = TinyNN.tnn_concat(@sess, @t_cls_token, t_patches, 1)
    # Add learned pos embed [d_model, T]. ggml broadcasts add OK if
    # shapes match exactly — pos_embed and t_seq are both [d_model, T].
    t_x = TinyNN.tnn_add(@sess, t_seq, @t_pos_embed)
    TinyNN.tnn_set_output(t_x)

    scale = 1.0 / Math.sqrt(cfg.d_head.to_f)
    li = 0
    while li < cfg.n_layers
      t_x = build_vit_block(t_x, @blocks[li], scale)
      li = li + 1
    end

    # Final norm — RMSNorm (see ViTTinyBlockFFI comment).
    t_final = TinyNN.tnn_rms_norm(@sess, t_x, @t_final_ln_gamma, cfg.ln_eps)
    TinyNN.tnn_set_output(t_final)

    # Take cls token: view(t_final, ne[d_model, 1] @ offset 0).
    # Easier: use ggml_view_2d via a slice helper. Approximation —
    # use get_rows with idx=[0] to pick column 0.
    idx_buf = [0]
    t_idx = TinyNN.tnn_input_1d_i32(@sess, 1)
    # Mark t_idx as the cls-index input; uploaded as [0] in each step.
    @t_cls_idx = t_idx
    t_cls_vec = TinyNN.tnn_get_rows(@sess, t_final, t_idx)
    # Head matmul: w_head [num_classes, d_model] · cls [d_model, 1] → [num_classes, 1]
    @t_logits = TinyNN.tnn_matmul(@sess, @t_w_head, t_cls_vec)
    TinyNN.tnn_set_output(@t_logits)
  end

  attr_accessor :t_cls_idx, :t_labels, :t_hp

  def build_training_step
    TinyNN.tnn_reset_for_rebuild(@sess)
    build_forward_in_current_ctx

    # Labels [num_classes, 1] one-hot for the single image.
    @t_labels = TinyNN.tnn_input_2d_f32(@sess, 1, @cfg.num_classes)
    @t_hp     = TinyNN.tnn_input_1d_f32(@sess, 7)

    t_loss = TinyNN.tnn_cross_entropy_loss(@sess, @t_logits, @t_labels)
    TinyNN.tnn_set_output(t_loss)
    TinyNN.tnn_set_loss(t_loss)

    TinyNN.tnn_build_forward_only(@sess, t_loss)
    TinyNN.tnn_build_backward(@sess)

    # AdamW opt_step on every PARAM (block + global).
    li = 0
    while li < @cfg.n_layers
      blk = @blocks[li]
      wi = 0
      while wi < blk.ft_weights.length
        tw = blk.ft_weights[wi]
        tg = TinyNN.tnn_tensor_grad(@sess, tw)
        to = TinyNN.tnn_opt_step_adamw(@sess, tw, tg, blk.ft_m[wi], blk.ft_v[wi], @t_hp)
        TinyNN.tnn_extend_backward_graph(@sess, to)
        wi = wi + 1
      end
      li = li + 1
    end
    gi = 0
    while gi < @ft_globals_weights.length
      tw = @ft_globals_weights[gi]
      tg = TinyNN.tnn_tensor_grad(@sess, tw)
      to = TinyNN.tnn_opt_step_adamw(@sess, tw, tg, @ft_globals_m[gi], @ft_globals_v[gi], @t_hp)
      TinyNN.tnn_extend_backward_graph(@sess, to)
      gi = gi + 1
    end

    TinyNN.tnn_pin_all_graph_b_nodes(@sess)
    TinyNN.tnn_realize_backward(@sess)
    t_loss
  end

  def build_vit_block(t_x, blk, scale)
    cfg = @cfg
    # Block-1: RMSNorm → MSA → residual (LayerNorm has no ggml backward).
    t_h = TinyNN.tnn_rms_norm(@sess, t_x, blk.t_ln1_gamma, cfg.ln_eps)

    # Per-head Q, K, V over the [d_model, T] activation.
    # For ViT-Tiny: no RoPE, no causal mask — pure scaled dot-product.
    t_head_outs = [TinyNN.tnn_null_ptr]; t_head_outs.pop
    h = 0
    while h < cfg.n_heads
      t_q = TinyNN.tnn_matmul(@sess, blk.t_w_q[h], t_h)   # [d_head, T]
      t_k = TinyNN.tnn_matmul(@sess, blk.t_w_k[h], t_h)   # [d_head, T]
      t_v = TinyNN.tnn_matmul(@sess, blk.t_w_v[h], t_h)   # [d_head, T]
      # scores = Q^T @ K, shape [T, T]. Then softmax then attn = V @ softmax^T (per ggml conv).
      # ggml matmul(K, Q) → ne=[T, T] (K.ne[0]=d_head contracts with Q.ne[0]).
      t_scores = TinyNN.tnn_matmul(@sess, t_k, t_q)
      t_scaled = TinyNN.tnn_scale(@sess, t_scores, scale)
      t_attn   = TinyNN.tnn_softmax(@sess, t_scaled)
      # V @ attn: V ne=[d_head, T], attn ne=[T, T]. Need [d_head, T].
      # matmul expects ne[0] contraction: V.ne[0]=d_head vs attn.ne[0]=T → mismatch.
      # Transpose V to [T, d_head], then matmul(V_t, attn): V_t.ne[0]=T vs attn.ne[0]=T ✓
      # but result ne=[d_head_new=V.ne[1]=T, T] — wrong.
      # Use the same pattern as the LLM cache: build_seq_qhead does V_t = transpose(V),
      # then matmul(V_t, attn) gives ne=[d_head, T].
      t_v_t = TinyNN.tnn_transpose(@sess, t_v)
      t_head_out = TinyNN.tnn_matmul(@sess, t_v_t, t_attn)
      t_head_outs.push(t_head_out)
      h = h + 1
    end

    # Concat heads along dim 0 (d_head axis): n_heads × [d_head, T] → [d_model, T].
    t_concat = t_head_outs[0]
    hc = 1
    while hc < cfg.n_heads
      t_concat = TinyNN.tnn_concat(@sess, t_concat, t_head_outs[hc], 0)
      hc = hc + 1
    end
    t_out_proj = TinyNN.tnn_matmul(@sess, blk.t_w_o, t_concat)
    t_x_attn   = TinyNN.tnn_add(@sess, t_x, t_out_proj)

    # Block-2: RMSNorm → MLP(SiLU) → residual.
    # ggml has no GELU backward (only SILU among activations), so we
    # use SiLU here for the smoke; algorithmically this matches a
    # documented ViT variant. timm-loader compat (which assumes GELU)
    # is its own piece — see vendored-backward follow-up issue.
    t_h2  = TinyNN.tnn_rms_norm(@sess, t_x_attn, blk.t_ln2_gamma, cfg.ln_eps)
    t_up  = TinyNN.tnn_matmul(@sess, blk.t_w_up, t_h2)
    t_act = TinyNN.tnn_silu(@sess, t_up)
    t_dn  = TinyNN.tnn_matmul(@sess, blk.t_w_down, t_act)
    TinyNN.tnn_add(@sess, t_x_attn, t_dn)
  end

  # --- bookkeeping helpers (parallel to Toy::LLM::Engine::LlamaSeqEngine) ---

  def ft_add_global_2d(weight, rows, cols)
    m = TinyNN.tnn_input_2d_f32_persistent(@sess, rows, cols)
    v = TinyNN.tnn_input_2d_f32_persistent(@sess, rows, cols)
    @ft_globals_weights.push(weight)
    @ft_globals_m.push(m)
    @ft_globals_v.push(v)
  end

  def ft_add_global_1d(weight)
    n = TinyNN.tnn_tensor_ne0(weight)
    m = TinyNN.tnn_input_1d_f32_persistent(@sess, n)
    v = TinyNN.tnn_input_1d_f32_persistent(@sess, n)
    @ft_globals_weights.push(weight)
    @ft_globals_m.push(m)
    @ft_globals_v.push(v)
  end

  def ft_add_2d(blk, weight, rows, cols)
    m = TinyNN.tnn_input_2d_f32_persistent(@sess, rows, cols)
    v = TinyNN.tnn_input_2d_f32_persistent(@sess, rows, cols)
    blk.ft_weights.push(weight); blk.ft_m.push(m); blk.ft_v.push(v)
  end

  def ft_add_1d(blk, weight)
    n = TinyNN.tnn_tensor_ne0(weight)
    m = TinyNN.tnn_input_1d_f32_persistent(@sess, n)
    v = TinyNN.tnn_input_1d_f32_persistent(@sess, n)
    blk.ft_weights.push(weight); blk.ft_m.push(m); blk.ft_v.push(v)
  end

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

  def upload_random_init!(seed, init_scale)
    state = [seed]
    cfg   = @cfg
    inv_d = init_scale / Math.sqrt(cfg.d_model.to_f)
    # patch_embed.proj.weight (linear over flat patches): Gaussian.
    upload_gaussian(@t_patch_kernel, @patch_flat_dim * cfg.d_model, 0.02, state)
    upload_constant(@t_cls_token, cfg.d_model, 0.0)
    upload_gaussian(@t_pos_embed, @seq_t * cfg.d_model, 0.02, state)
    upload_constant(@t_final_ln_gamma, cfg.d_model, 1.0)
    upload_gaussian(@t_w_head, cfg.num_classes * cfg.d_model, 0.02, state)

    li = 0
    while li < cfg.n_layers
      blk = @blocks[li]
      upload_constant(blk.t_ln1_gamma, cfg.d_model, 1.0)
      upload_constant(blk.t_ln2_gamma, cfg.d_model, 1.0)
      h = 0
      while h < cfg.n_heads
        upload_gaussian(blk.t_w_q[h], cfg.d_head * cfg.d_model, inv_d, state)
        upload_gaussian(blk.t_w_k[h], cfg.d_head * cfg.d_model, inv_d, state)
        upload_gaussian(blk.t_w_v[h], cfg.d_head * cfg.d_model, inv_d, state)
        h = h + 1
      end
      upload_gaussian(blk.t_w_o,    cfg.d_model * cfg.d_model, inv_d, state)
      upload_gaussian(blk.t_w_up,   cfg.d_ff    * cfg.d_model, inv_d, state)
      upload_gaussian(blk.t_w_down, cfg.d_model * cfg.d_ff,
                       init_scale / Math.sqrt(cfg.d_ff.to_f), state)
      # Zero-init Adam moments (m, v).
      wi = 0
      while wi < blk.ft_m.length
        zero_tensor(blk.ft_m[wi])
        zero_tensor(blk.ft_v[wi])
        wi = wi + 1
      end
      li = li + 1
    end
    gi = 0
    while gi < @ft_globals_m.length
      zero_tensor(@ft_globals_m[gi])
      zero_tensor(@ft_globals_v[gi])
      gi = gi + 1
    end
  end

  def upload_gaussian(tensor, n, std, state)
    buf = Array.new(n, 0.0)
    i = 0
    while i < n
      # Box-Muller from a xorshift64 stream (state[0] mutated in place).
      s = state[0]
      s = s ^ (s << 13);  s = s & 0xFFFFFFFFFFFFFFFF
      s = s ^ (s >> 7);   s = s & 0xFFFFFFFFFFFFFFFF
      s = s ^ (s << 17);  s = s & 0xFFFFFFFFFFFFFFFF
      state[0] = s
      u1 = (s & 0xFFFFFFFF).to_f / 4294967296.0
      s = state[0]
      s = s ^ (s << 13);  s = s & 0xFFFFFFFFFFFFFFFF
      s = s ^ (s >> 7);   s = s & 0xFFFFFFFFFFFFFFFF
      s = s ^ (s << 17);  s = s & 0xFFFFFFFFFFFFFFFF
      state[0] = s
      u2 = (s & 0xFFFFFFFF).to_f / 4294967296.0
      if u1 < 1.0e-12; u1 = 1.0e-12; end
      z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math::PI * u2)
      buf[i] = z * std
      i = i + 1
    end
    TinyNN.tnn_upload_from_float_array(@sess, tensor, buf, n)
  end

  def upload_constant(tensor, n, val)
    buf = Array.new(n, val)
    TinyNN.tnn_upload_from_float_array(@sess, tensor, buf, n)
  end

  def zero_tensor(tensor)
    n = TinyNN.tnn_tensor_nelements(tensor)
    buf = Array.new(n, 0.0)
    TinyNN.tnn_upload_from_float_array(@sess, tensor, buf, n)
  end
end
end; end; end
