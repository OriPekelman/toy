# A/B parity for per-row softmax.
# Native reference matches lib/transformer.rb's softmax_rows!:
#   for each row: subtract row max, exp, divide by row sum.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

def softmax_native(x)
  out = Mat.new(x.nrows, x.ncols)
  r = 0
  while r < x.nrows
    # row max for numerical stability
    m = x.flat[r * x.ncols]
    c = 1
    while c < x.ncols
      v = x.flat[r * x.ncols + c]
      if v > m
        m = v
      end
      c = c + 1
    end
    # exp(x - m), accumulate sum
    s = 0.0
    c = 0
    while c < x.ncols
      e = Math.exp(x.flat[r * x.ncols + c] - m)
      out.flat[r * x.ncols + c] = e
      s = s + e
      c = c + 1
    end
    # normalize
    c = 0
    while c < x.ncols
      out.flat[r * x.ncols + c] = out.flat[r * x.ncols + c] / s
      c = c + 1
    end
    r = r + 1
  end
  out
end

# 2 rows, 4 cols each.
x = Mat.new(2, 4)
x.flat[0] = 1.0;  x.flat[1] = 2.0;  x.flat[2] = 3.0;  x.flat[3] = 4.0
x.flat[4] = 0.5;  x.flat[5] = 0.5;  x.flat[6] = 0.5;  x.flat[7] = 0.5

native = softmax_native(x)
ffi    = TinyNN.softmax(x)

ok = true
i = 0
while i < 8
  diff = native.flat[i] - ffi.flat[i]
  if diff < 0
    diff = -diff
  end
  puts "[" + i.to_s + "] native=" + native.flat[i].to_s + " ffi=" + ffi.flat[i].to_s
  if diff > 1.0e-5
    ok = false
  end
  i = i + 1
end

# Per-row sum should be ~1.0.
r = 0
while r < 2
  s = 0.0
  c = 0
  while c < 4
    s = s + ffi.flat[r * 4 + c]
    c = c + 1
  end
  puts "row " + r.to_s + " sum=" + s.to_s
  r = r + 1
end

puts "match: " + ok.to_s
