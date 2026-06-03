# A/B parity for SGD step: param_new = param - lr * grad.
# Native baseline matches the inner loop of TransformerLM#apply_gradients_sgd:
#   @flat[i] -= lr * grad.flat[i]

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"

def sgd_native(param, grad, lr)
  out = Mat.new(param.nrows, param.ncols)
  n = param.nrows * param.ncols
  i = 0
  while i < n
    out.flat[i] = param.flat[i] - lr * grad.flat[i]
    i = i + 1
  end
  out
end

# Toy shapes: (4, 3) params and grads.
param = Mat.new(4, 3)
grad  = Mat.new(4, 3)
i = 0
while i < 12
  param.flat[i] = i.to_f * 0.5 - 2.0
  grad.flat[i]  = (i.to_f - 6.0) * 0.1
  i = i + 1
end

lr = 0.05

nat = sgd_native(param, grad, lr)
ffi = TinyNN.sgd_step(param, grad, lr)

ok = true
n = 12
i = 0
while i < n
  d = nat.flat[i] - ffi.flat[i]
  if d < 0
    d = -d
  end
  if d > 1.0e-5
    ok = false
  end
  i = i + 1
end
puts "param[0]=" + param.flat[0].to_s + " grad[0]=" + grad.flat[0].to_s + " lr=" + lr.to_s
puts "native[0]=" + nat.flat[0].to_s + " ffi[0]=" + ffi.flat[0].to_s
puts "sgd_step: match=" + ok.to_s
