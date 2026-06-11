# A/B parity for element-wise GeLU (tanh approx).
# Native reference matches lib/transformer.rb's feed_forward GeLU:
#   c = sqrt(2/π);  u = c * (x + 0.044715 * x^3);  y = 0.5 * x * (1 + tanh(u))

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

def gelu_native(x)
  c = 0.7978845608028654   # sqrt(2/pi)
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

# Spread of values: zero, small positive, large positive, negative, large negative.
a = Mat.new(1, 5)
a.flat[0] =  0.0
a.flat[1] =  0.5
a.flat[2] =  3.0
a.flat[3] = -0.5
a.flat[4] = -3.0

native = gelu_native(a)
ffi    = TinyNN.gelu(a)

i = 0
ok = true
while i < 5
  diff = native.flat[i] - ffi.flat[i]
  if diff < 0
    diff = -diff
  end
  puts "[" + i.to_s + "] x=" + a.flat[i].to_s + " native=" + native.flat[i].to_s + " ffi=" + ffi.flat[i].to_s
  # GeLU goes through a tanh, so allow a touch more slack than add/matmul.
  if diff > 1.0e-3
    ok = false
  end
  i = i + 1
end
puts "match: " + ok.to_s
