# A/B parity for a chained 3-op FFN-shaped pipeline:
#   result = gelu(h * w1) * w2
# Realistic shape of what an S4 FFN port will do: build matmul --- gelu ---
# matmul as one ggml graph rather than three FFI calls.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

def gelu_native(x)
  c = 0.7978845608028654
  out = Mat.new(x.nrows, x.ncols)
  n = x.nrows * x.ncols
  i = 0
  while i < n
    v = x.flat[i]
    u = c * (v + 0.044715 * v * v * v)
    out.flat[i] = 0.5 * v * (1.0 + Math.tanh(u))
    i = i + 1
  end
  out
end

# Tiny FFN dimensions: T=3, d_model=4, d_ff=6.
h  = Mat.new(3, 4)
w1 = Mat.new(4, 6)
w2 = Mat.new(6, 4)

# Fill h with i*0.1 - 0.5, etc.
n_h = h.nrows * h.ncols
i = 0
while i < n_h
  h.flat[i] = i.to_f * 0.1 - 0.5
  i = i + 1
end

n_w1 = w1.nrows * w1.ncols
i = 0
while i < n_w1
  w1.flat[i] = i.to_f * 0.05 - 0.3
  i = i + 1
end

n_w2 = w2.nrows * w2.ncols
i = 0
while i < n_w2
  w2.flat[i] = i.to_f * 0.07 - 0.4
  i = i + 1
end

pre_n    = h.matmul(w1)
hidden_n = gelu_native(pre_n)
native   = hidden_n.matmul(w2)

ffi = TinyNN.ffn_pipeline(h, w1, w2)

ok = true
n = native.nrows * native.ncols
i = 0
while i < n
  diff = native.flat[i] - ffi.flat[i]
  if diff < 0
    diff = -diff
  end
  if diff > 1.0e-3
    ok = false
  end
  i = i + 1
end

puts "native[0,0]=" + native.flat[0].to_s + " ffi[0,0]=" + ffi.flat[0].to_s
puts "native[last]=" + native.flat[n - 1].to_s + " ffi[last]=" + ffi.flat[n - 1].to_s
puts "checked " + n.to_s + " cells against tolerance 1e-3"
puts "match: " + ok.to_s
