# Parity check for FullForwardFFICache with one full transformer block
# (pre-norm -> multi-head attention -> residual -> pre-norm -> FFN ->
# residual) plus final norm + tied unembed.
#
# Builds a TransformerLM(n_layers=1, n_heads=2, d_head=4) with
# deterministic weights and compares its forward() to the FFI graph's
# logits.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"

vocab_size = 10
d_model    = 16
d_ff       = 32
n_heads    = 4
n_layers   = 2
context_length = 3

model = TransformerLM.new(vocab_size, d_model, d_ff, n_heads, n_layers, context_length)

# Deterministic fill (use the project's existing fill_random with a
# fixed seed). srand is portable across Spinel and CRuby.
srand(42)
model.token_embed.fill_random(0.1)
model.pos_embed.fill_random(0.05)
i = 0
while i < d_model
  model.norm_final_gamma[i] = 1.0 + i.to_f * 0.01
  i = i + 1
end
li = 0
while li < n_layers
  b = model.blocks[li]
  i = 0
  while i < d_model
    b.norm1_gamma[i] = 1.0 + i.to_f * 0.02
    b.norm2_gamma[i] = 1.0 + i.to_f * 0.03
    i = i + 1
  end
  b.fill_random_all(0.1)
  li = li + 1
end

# Native forward.
ids = [2, 0, 4]
nat_logits = model.forward(ids)

# === FFI ===
cache = FullForwardFFICache.new
cache.realize_for(context_length, d_model, d_ff, n_heads, n_layers, vocab_size)

TinyNN.upload_row_major(cache.sess, cache.t_token_embed, model.token_embed)
# Upload only the first context_length rows of pos_embed.
pos_slice = Mat.new(context_length, d_model)
i = 0
while i < context_length * d_model
  pos_slice.flat[i] = model.pos_embed.flat[i]
  i = i + 1
end
TinyNN.upload_row_major(cache.sess, cache.t_pos_slice, pos_slice)

# 1D float upload for gammas: tnn_upload_from_float_array accepts any
# tensor + a float_array + count.
TinyNN.tnn_upload_from_float_array(cache.sess, cache.t_final_norm_gamma,
                                    model.norm_final_gamma, d_model)

li = 0
while li < n_layers
  blk_n = model.blocks[li]
  blk_f = cache.blocks_ffi[li]
  TinyNN.tnn_upload_from_float_array(cache.sess, blk_f.t_norm1_gamma, blk_n.norm1_gamma, d_model)
  TinyNN.tnn_upload_from_float_array(cache.sess, blk_f.t_norm2_gamma, blk_n.norm2_gamma, d_model)
  h = 0
  while h < n_heads
    TinyNN.stage_transposed_and_upload(cache.sess, blk_f.t_w_q[h], blk_n.w_q[h])
    TinyNN.stage_transposed_and_upload(cache.sess, blk_f.t_w_k[h], blk_n.w_k[h])
    TinyNN.stage_transposed_and_upload(cache.sess, blk_f.t_w_v[h], blk_n.w_v[h])
    h = h + 1
  end
  TinyNN.stage_transposed_and_upload(cache.sess, blk_f.t_w_o,   blk_n.w_o)
  TinyNN.stage_transposed_and_upload(cache.sess, blk_f.t_w_ff1, blk_n.w_ff1)
  TinyNN.stage_transposed_and_upload(cache.sess, blk_f.t_w_ff2, blk_n.w_ff2)
  li = li + 1
end

TinyNN.upload_int_array(cache.sess, cache.t_token_ids, ids)
TinyNN.tnn_compute(cache.sess)

ffi_logits = TinyNN.download_row_major(cache.sess, cache.t_logits, context_length, vocab_size)

# Compare.
max_d = 0.0
n = context_length * vocab_size
i = 0
while i < n
  d = nat_logits.flat[i] - ffi_logits.flat[i]
  if d < 0
    d = -d
  end
  if d > max_d
    max_d = d
  end
  i = i + 1
end

puts "n_layers=" + n_layers.to_s + " n_heads=" + n_heads.to_s
puts "nat_logits[0,0..2]=" + nat_logits.flat[0].to_s + " " + nat_logits.flat[1].to_s + " " + nat_logits.flat[2].to_s
puts "ffi_logits[0,0..2]=" + ffi_logits.flat[0].to_s + " " + ffi_logits.flat[1].to_s + " " + ffi_logits.flat[2].to_s
puts "max_abs_diff=" + max_d.to_s
ok = max_d < 1.0e-3   # tolerant for ggml's f16-LUT gelu inside FFN
puts "M1.2 BLOCK OK: " + ok.to_s
TinyNN.tnn_session_free(cache.sess)
