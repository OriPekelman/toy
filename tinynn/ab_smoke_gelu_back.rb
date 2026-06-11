# A/B parity for gelu_back: dx = dh * d/dx GeLU(x) (tanh approx).
#
# Native reference derived by chain rule:
#   u  = c * (x + k * x^3),  c = sqrt(2/pi), k = 0.044715
#   y  = 0.5 * x * (1 + tanh(u))
#   dy/dx = 0.5 (1 + tanh u) + 0.5 x (1 - tanh^2 u) * c * (1 + 3 k x^2)
#   dx = dh * dy/dx

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

def gelu_back_native(x, dh)
  c = 0.7978845608028654
  k = 0.044715
  out = Mat.new(x.nrows, x.ncols)
  n = x.nrows * x.ncols
  i = 0
  while i < n
    xi = x.flat[i]
    di = dh.flat[i]
    xi2 = xi * xi
    u = c * (xi + k * xi * xi2)
    tu = Math.tanh(u)
    sech2 = 1.0 - tu * tu
    dudx = c * (1.0 + 3.0 * k * xi2)
    dgelu = 0.5 * (1.0 + tu) + 0.5 * xi * sech2 * dudx
    out.flat[i] = di * dgelu
    i = i + 1
  end
  out
end

# (2, 4): zero, small positive, large positive, small negative, large negative.
xtest = Mat.new(2, 4)
xtest.flat[0] = 0.0
xtest.flat[1] = 0.5
xtest.flat[2] = 2.5
xtest.flat[3] = -0.3
xtest.flat[4] = -1.2
xtest.flat[5] = 4.0
xtest.flat[6] = 0.01
xtest.flat[7] = -3.5

dhtest = Mat.new(2, 4)
dhtest.flat[0] = 1.0
dhtest.flat[1] = -0.5
dhtest.flat[2] = 2.0
dhtest.flat[3] = -1.0
dhtest.flat[4] = 0.3
dhtest.flat[5] = 0.7
dhtest.flat[6] = -2.0
dhtest.flat[7] = 0.5

nat = gelu_back_native(xtest, dhtest)
ffi = TinyNN.gelu_back(xtest, dhtest)

ok = true
max_d = 0.0
i = 0
while i < 8
  d = nat.flat[i] - ffi.flat[i]
  if d < 0
    d = -d
  end
  if d > max_d
    max_d = d
  end
  if d > 1.0e-3
    ok = false
  end
  puts "x=" + xtest.flat[i].to_s + " dh=" + dhtest.flat[i].to_s + " native=" + nat.flat[i].to_s + " ffi=" + ffi.flat[i].to_s
  i = i + 1
end
puts "max-abs-diff=" + max_d.to_s + " match=" + ok.to_s
