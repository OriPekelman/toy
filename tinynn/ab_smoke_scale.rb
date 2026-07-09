# A/B parity for elementwise scale: a * s as a new Mat.
# Project's Mat#scale! is in-place; we use a `dup-and-scale` pattern
# native side to keep semantics out-of-place for comparison.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/ffi/tinynn"

def scale_native(a, s)
  out = Mat.new(a.nrows, a.ncols)
  n = a.nrows * a.ncols
  i = 0
  while i < n
    out.flat[i] = a.flat[i] * s
    i = i + 1
  end
  out
end

a = Mat.new(2, 3)
a.flat[0] = 1.0;   a.flat[1] = 2.5;   a.flat[2] = -3.0
a.flat[3] = 0.0;   a.flat[4] = 100.0; a.flat[5] = -0.5

s = 0.5

native = scale_native(a, s)
ffi    = TinyNN.scale(a, s)

ok = true
i = 0
while i < 6
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

puts "match: " + ok.to_s
