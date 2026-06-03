# A/B parity for cross_entropy_grad:
#   dlogits[i, v] = (softmax(logits)[i, v] - one_hot(targets[i])[v]) / n_pred
#
# Native reference computes softmax + subtract one_hot + divide inline.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"

def softmax_native(x)
  out = Mat.new(x.nrows, x.ncols)
  r = 0
  while r < x.nrows
    nc = x.ncols
    m = x.flat[r * nc]
    c = 1
    while c < nc
      v = x.flat[r * nc + c]
      if v > m
        m = v
      end
      c = c + 1
    end
    s = 0.0
    c = 0
    while c < nc
      e = Math.exp(x.flat[r * nc + c] - m)
      out.flat[r * nc + c] = e
      s = s + e
      c = c + 1
    end
    c = 0
    while c < nc
      out.flat[r * nc + c] = out.flat[r * nc + c] / s
      c = c + 1
    end
    r = r + 1
  end
  out
end

def cross_entropy_grad_native(logits, targets, n_pred)
  sm = softmax_native(logits)
  out = Mat.new(logits.nrows, logits.ncols)
  inv_n = 1.0 / n_pred.to_f
  i = 0
  while i < n_pred
    v = 0
    while v < logits.ncols
      term = sm.flat[i * logits.ncols + v]
      if v == targets[i]
        term = term - 1.0
      end
      out.flat[i * logits.ncols + v] = term * inv_n
      v = v + 1
    end
    i = i + 1
  end
  out
end

# (3, 5) logits, 3 predictions targeting (4, 0, 2).
logitsm = Mat.new(3, 5)
i = 0
while i < 15
  logitsm.flat[i] = (i.to_f * 0.3 - 2.0)
  i = i + 1
end
targets = [4, 0, 2]
n_pred = 3

nat = cross_entropy_grad_native(logitsm, targets, n_pred)
ffi = TinyNN.cross_entropy_grad(logitsm, targets, n_pred)

ok = true
max_d = 0.0
i = 0
while i < 15
  d = nat.flat[i] - ffi.flat[i]
  if d < 0
    d = -d
  end
  if d > max_d
    max_d = d
  end
  if d > 1.0e-4
    ok = false
  end
  i = i + 1
end
puts "row 0: native=" + nat.flat[0].to_s + " ffi=" + ffi.flat[0].to_s
puts "row 0 target slot: native=" + nat.flat[4].to_s + " ffi=" + ffi.flat[4].to_s
puts "row 2 last: native=" + nat.flat[14].to_s + " ffi=" + ffi.flat[14].to_s
puts "max-abs-diff=" + max_d.to_s + " match=" + ok.to_s
