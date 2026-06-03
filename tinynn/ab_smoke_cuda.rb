# CUDA variant of tinynn/ab_smoke.rb. Runs the same 2x3 @ 3x2 matmul
# through native Mat#matmul AND TinyNNCuda.matmul (ggml-CUDA backend on
# the local NVIDIA device). Expected: identical values, match=true.
#
# Run via `make ab-smoke-cuda`.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn_cuda"

a = Mat.new(2, 3)
a.flat[0] = 1.0;  a.flat[1] = 2.0;  a.flat[2] = 3.0
a.flat[3] = 4.0;  a.flat[4] = 5.0;  a.flat[5] = 6.0

b = Mat.new(3, 2)
b.flat[0] =  7.0;  b.flat[1] =  8.0
b.flat[2] =  9.0;  b.flat[3] = 10.0
b.flat[4] = 11.0;  b.flat[5] = 12.0

native = a.matmul(b)
ffi    = TinyNNCuda.matmul(a, b)

puts "native [0,0]=" + native.flat[0].to_s + " [0,1]=" + native.flat[1].to_s
puts "native [1,0]=" + native.flat[2].to_s + " [1,1]=" + native.flat[3].to_s
puts "ffi    [0,0]=" + ffi.flat[0].to_s    + " [0,1]=" + ffi.flat[1].to_s
puts "ffi    [1,0]=" + ffi.flat[2].to_s    + " [1,1]=" + ffi.flat[3].to_s

ok = true
i = 0
while i < 4
  diff = native.flat[i] - ffi.flat[i]
  if diff < 0
    diff = -diff
  end
  if diff > 1.0e-4
    ok = false
  end
  i = i + 1
end
puts ""
puts "match: " + ok.to_s
