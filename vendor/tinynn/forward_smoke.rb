# Forward-only smoke: build a real TransformerLM, run model.forward, print
# logits. If this works at Spinel master (where train_minimal SIGBUSes in
# backward), it gives us a verification platform for S2 *now* — swap a
# Mat#matmul in feed_forward and compare logits with vs. without FFI.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

# Same dimensions train_minimal uses: vocab=7, d_model=16, d_ff=32,
# n_heads=2, n_layers=2, context_length=8. (d_head = d_model/n_heads).
model = TransformerLM.new(7, 16, 32, 2, 2, 8)
puts "built model: vocab=" + model.vocab_size.to_s + " d_model=" + model.d_model.to_s + " blocks=" + model.blocks.length.to_s

token_ids = [0, 1, 2]
logits = model.forward(token_ids)
puts "logits shape=" + logits.nrows.to_s + "x" + logits.ncols.to_s
puts "logits[0,0]=" + logits.flat[0].to_s
puts "logits[0,1]=" + logits.flat[1].to_s

# --- A/B against TinyNN.matmul on the real FFN inputs ---
# After forward, model.layer_caches[0].h_norm2 holds the FFN input
# (post-RMSNorm activations), and model.blocks[0].w_ff1 is the first
# projection weight matrix. This is the exact pair that the real
# `feed_forward(h, block)` computes as `h.matmul(block.w_ff1)`.
h     = model.layer_caches[0].h_norm2
w_ff1 = model.blocks[0].w_ff1
puts "h shape=" + h.nrows.to_s + "x" + h.ncols.to_s + " w_ff1 shape=" + w_ff1.nrows.to_s + "x" + w_ff1.ncols.to_s

native = h.matmul(w_ff1)
ffi    = TinyNN.matmul(h, w_ff1)

ok = true
max_d = 0.0
n = native.nrows * native.ncols
i = 0
while i < n
  d = native.flat[i] - ffi.flat[i]
  if d < 0
    d = -d
  end
  if d > max_d
    max_d = d
  end
  if d > 1.0e-3
    ok = false
  end
  i = i + 1
end
puts "FFN first matmul real-shapes A/B: native[0]=" + native.flat[0].to_s + " ffi[0]=" + ffi.flat[0].to_s
puts "max-abs-diff=" + max_d.to_s + " match=" + ok.to_s
