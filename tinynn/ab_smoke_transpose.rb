# A/B parity for transpose: Mat#transpose vs TinyNN.transpose.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"

# 3x2 -> 2x3
a = Mat.new(3, 2)
a.flat[0] = 1.0;  a.flat[1] = 2.0
a.flat[2] = 3.0;  a.flat[3] = 4.0
a.flat[4] = 5.0;  a.flat[5] = 6.0

native = a.transpose
ffi    = TinyNN.transpose(a)

puts "native shape: " + native.nrows.to_s + "x" + native.ncols.to_s
puts "ffi    shape: " + ffi.nrows.to_s + "x" + ffi.ncols.to_s

ok = true
if native.nrows != ffi.nrows || native.ncols != ffi.ncols
  ok = false
end

n = native.nrows * native.ncols
i = 0
while i < n
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
