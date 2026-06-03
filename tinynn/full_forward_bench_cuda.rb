# Wallclock bench: native TransformerLM.forward vs FullForwardFFICacheCuda
# at a "more than toy" shape (vocab=64, d_model=64, d_ff=128, n_heads=4,
# n_layers=2, T=32). 20 iterations each, deterministic weights.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn_cuda"

VOCAB    = 4096
D_MODEL  = 384
D_FF     = 1024
N_HEADS  = 6
N_LAYERS = 6
T_SEQ    = 128
CONTEXT  = 128
ITERS    = 10

srand(7)
model = TransformerLM.new(VOCAB, D_MODEL, D_FF, N_HEADS, N_LAYERS, CONTEXT)
model.token_embed.fill_random(0.05)
model.pos_embed.fill_random(0.03)
i = 0
while i < D_MODEL
  model.norm_final_gamma[i] = 1.0
  i = i + 1
end
li = 0
while li < N_LAYERS
  b = model.blocks[li]
  i = 0
  while i < D_MODEL
    b.norm1_gamma[i] = 1.0
    b.norm2_gamma[i] = 1.0
    i = i + 1
  end
  b.fill_random_all(0.05)
  li = li + 1
end

ids = Array.new(T_SEQ, 0)
i = 0
while i < T_SEQ
  ids[i] = i % VOCAB
  i = i + 1
end

# Warm: one native call (sets up caches).
_warm = model.forward(ids)

# ----- native -----
t0 = Time.now
i = 0
while i < ITERS
  _ = model.forward(ids)
  i = i + 1
end
t1 = Time.now
nat_ms = (t1 - t0) * 1000.0

# ----- FFI: realize once, then loop forward -----
cache = FullForwardFFICacheCuda.new
cache.realize_for(T_SEQ, D_MODEL, D_FF, N_HEADS, N_LAYERS, VOCAB)

TinyNNCuda.upload_row_major(cache.sess, cache.t_token_embed, model.token_embed)
pos_slice = Mat.new(T_SEQ, D_MODEL)
i = 0
while i < T_SEQ * D_MODEL
  pos_slice.flat[i] = model.pos_embed.flat[i]
  i = i + 1
end
TinyNNCuda.upload_row_major(cache.sess, cache.t_pos_slice, pos_slice)
TinyNNCuda.tnn_upload_from_float_array(cache.sess, cache.t_final_norm_gamma,
                                    model.norm_final_gamma, D_MODEL)
li = 0
while li < N_LAYERS
  blk_n = model.blocks[li]
  blk_f = cache.blocks_ffi[li]
  TinyNNCuda.tnn_upload_from_float_array(cache.sess, blk_f.t_norm1_gamma, blk_n.norm1_gamma, D_MODEL)
  TinyNNCuda.tnn_upload_from_float_array(cache.sess, blk_f.t_norm2_gamma, blk_n.norm2_gamma, D_MODEL)
  h = 0
  while h < N_HEADS
    TinyNNCuda.stage_transposed_and_upload(cache.sess, blk_f.t_w_q[h], blk_n.w_q[h])
    TinyNNCuda.stage_transposed_and_upload(cache.sess, blk_f.t_w_k[h], blk_n.w_k[h])
    TinyNNCuda.stage_transposed_and_upload(cache.sess, blk_f.t_w_v[h], blk_n.w_v[h])
    h = h + 1
  end
  TinyNNCuda.stage_transposed_and_upload(cache.sess, blk_f.t_w_o,   blk_n.w_o)
  TinyNNCuda.stage_transposed_and_upload(cache.sess, blk_f.t_w_ff1, blk_n.w_ff1)
  TinyNNCuda.stage_transposed_and_upload(cache.sess, blk_f.t_w_ff2, blk_n.w_ff2)
  li = li + 1
end

# Warm one FFI call.
TinyNNCuda.upload_int_array(cache.sess, cache.t_token_ids, ids)
TinyNNCuda.tnn_compute(cache.sess)
_ = TinyNNCuda.download_row_major(cache.sess, cache.t_logits, T_SEQ, VOCAB)

t0 = Time.now
i = 0
while i < ITERS
  TinyNNCuda.upload_int_array(cache.sess, cache.t_token_ids, ids)
  TinyNNCuda.tnn_compute(cache.sess)
  _ = TinyNNCuda.download_row_major(cache.sess, cache.t_logits, T_SEQ, VOCAB)
  i = i + 1
end
t1 = Time.now
ffi_ms = (t1 - t0) * 1000.0

puts "Shape: vocab=" + VOCAB.to_s + " d_model=" + D_MODEL.to_s + " d_ff=" + D_FF.to_s +
     " n_heads=" + N_HEADS.to_s + " n_layers=" + N_LAYERS.to_s + " T=" + T_SEQ.to_s
puts "Native forward  " + ITERS.to_s + " iters: " + nat_ms.to_s + " ms  (" + (nat_ms / ITERS.to_f).to_s + " ms/iter)"
puts "FFI    forward  " + ITERS.to_s + " iters: " + ffi_ms.to_s + " ms  (" + (ffi_ms / ITERS.to_f).to_s + " ms/iter)"
speedup = nat_ms / ffi_ms
puts "Speedup: " + speedup.to_s + "x"
TinyNNCuda.tnn_session_free(cache.sess)
