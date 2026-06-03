# A/B parity for RMSNorm with gamma scale.
# Native reference (matches lib/transformer.rb):
#   for each row r:
#     rms = sqrt( mean(x[r,*]²) + eps )
#     y[r,c] = x[r,c] / rms * gamma[c]

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"

def rms_norm_native(x, gamma, eps)
  out = Mat.new(x.nrows, x.ncols)
  r = 0
  while r < x.nrows
    sum_sq = 0.0
    c = 0
    while c < x.ncols
      v = x.flat[r * x.ncols + c]
      sum_sq = sum_sq + v * v
      c = c + 1
    end
    mean_sq = sum_sq / x.ncols.to_f
    rms = Math.sqrt(mean_sq + eps)
    c = 0
    while c < x.ncols
      out.flat[r * x.ncols + c] = x.flat[r * x.ncols + c] / rms * gamma[c]
      c = c + 1
    end
    r = r + 1
  end
  out
end

# 2 rows, 4 features per row — covers both row-direction normalization
# and per-feature gamma scaling.
x = Mat.new(2, 4)
x.flat[0] =  1.0;  x.flat[1] =  2.0;  x.flat[2] = -1.0;  x.flat[3] =  0.5
x.flat[4] = -0.5;  x.flat[5] =  3.0;  x.flat[6] =  0.0;  x.flat[7] =  1.5

gamma = [1.0, 0.5, 2.0, 1.5]
eps = 0.00001

native = rms_norm_native(x, gamma, eps)
ffi    = TinyNN.rms_norm(x, gamma, eps)

i = 0
ok = true
while i < 8
  diff = native.flat[i] - ffi.flat[i]
  if diff < 0
    diff = -diff
  end
  puts "[" + i.to_s + "] native=" + native.flat[i].to_s + " ffi=" + ffi.flat[i].to_s
  if diff > 1.0e-3
    ok = false
  end
  i = i + 1
end
puts "match: " + ok.to_s
