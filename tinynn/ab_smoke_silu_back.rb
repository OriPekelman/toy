# A/B parity for silu_back: dx = dy * d/dx SiLU(x).
#
# Reference derived by chain rule:
#   y       = x * sigmoid(x)
#   dy/dx   = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
#           = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
#   dx      = dy * dy/dx

require_relative "../lib/transformer"
require_relative "../lib/tinynn"

def silu_back_native(x, dy)
  out = Mat.new(x.nrows, x.ncols)
  n = x.nrows * x.ncols
  i = 0
  while i < n
    xi = x.flat[i]
    di = dy.flat[i]
    s  = 1.0 / (1.0 + Math.exp(-xi))
    dsilu = s * (1.0 + xi * (1.0 - s))
    out.flat[i] = di * dsilu
    i = i + 1
  end
  out
end

# (2, 4): zero, small +, large +, small -, large -.
xtest = Mat.new(2, 4)
xtest.flat[0] = 0.0
xtest.flat[1] = 0.5
xtest.flat[2] = 2.5
xtest.flat[3] = -0.3
xtest.flat[4] = -1.2
xtest.flat[5] = 4.0
xtest.flat[6] = 0.01
xtest.flat[7] = -3.5

dytest = Mat.new(2, 4)
dytest.flat[0] = 1.0
dytest.flat[1] = -0.5
dytest.flat[2] = 2.0
dytest.flat[3] = -1.0
dytest.flat[4] = 0.3
dytest.flat[5] = 0.7
dytest.flat[6] = -2.0
dytest.flat[7] = 0.5

nat = silu_back_native(xtest, dytest)
ffi = TinyNN.silu_back(xtest, dytest)

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
  puts "x=" + xtest.flat[i].to_s + " dy=" + dytest.flat[i].to_s +
       " native=" + nat.flat[i].to_s + " ffi=" + ffi.flat[i].to_s
  i = i + 1
end
puts "silu_back: max-abs-diff=" + max_d.to_s + " match=" + ok.to_s
