# A/B parity for Adam step. Reference matches lib/transformer.rb's
# adam_step_mat inner loop:
#   m_new = b1*m + (1-b1)*g
#   v_new = b2*v + (1-b2)*g*g
#   m_hat = m_new / omc1
#   v_hat = v_new / omc2
#   p -= lr * m_hat / (sqrt(v_hat) + eps)

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"

def adam_native(param, grad, m, v, lr, b1, b2, eps, omc1, omc2)
  np = Mat.new(param.nrows, param.ncols)
  nm = Mat.new(param.nrows, param.ncols)
  nv = Mat.new(param.nrows, param.ncols)
  one_minus_b1 = 1.0 - b1
  one_minus_b2 = 1.0 - b2
  n = param.nrows * param.ncols
  i = 0
  while i < n
    gi = grad.flat[i]
    new_m = b1 * m.flat[i] + one_minus_b1 * gi
    new_v = b2 * v.flat[i] + one_minus_b2 * gi * gi
    nm.flat[i] = new_m
    nv.flat[i] = new_v
    m_hat = new_m / omc1
    v_hat = new_v / omc2
    np.flat[i] = param.flat[i] - lr * m_hat / (Math.sqrt(v_hat) + eps)
    i = i + 1
  end
  [np, nm, nv]
end

# Toy params: (3, 4).
param = Mat.new(3, 4)
grad  = Mat.new(3, 4)
mst   = Mat.new(3, 4)
vst   = Mat.new(3, 4)
i = 0
while i < 12
  param.flat[i] = i.to_f * 0.1 - 0.5
  grad.flat[i]  = (i.to_f - 6.0) * 0.05
  mst.flat[i]   = i.to_f * 0.02
  vst.flat[i]   = i.to_f * 0.01
  i = i + 1
end

lr   = 0.001
b1   = 0.9
b2   = 0.999
eps  = 1.0e-8
omc1 = 1.0 - 0.9
omc2 = 1.0 - 0.999

ref = adam_native(param, grad, mst, vst, lr, b1, b2, eps, omc1, omc2)
nat_p = ref[0]
nat_m = ref[1]
nat_v = ref[2]

ffi_res = TinyNN.adam_step(param, grad, mst, vst, lr, b1, b2, eps, omc1, omc2)
ffi_p = ffi_res.param
ffi_m = ffi_res.mom_m
ffi_v = ffi_res.mom_v

def cmp(name, n, ffi, tol)
  ok = true
  max_d = 0.0
  i = 0
  total = n.nrows * n.ncols
  while i < total
    d = n.flat[i] - ffi.flat[i]
    if d < 0
      d = -d
    end
    if d > max_d
      max_d = d
    end
    if d > tol
      ok = false
    end
    i = i + 1
  end
  puts name + ": max-abs-diff=" + max_d.to_s + " match=" + ok.to_s
end

cmp("param", nat_p, ffi_p, 1.0e-4)
cmp("m",     nat_m, ffi_m, 1.0e-5)
cmp("v",     nat_v, ffi_v, 1.0e-5)
