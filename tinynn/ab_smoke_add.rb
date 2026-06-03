# A/B parity for element-wise add: Mat#add vs TinyNN.add.
# Run via `make ab-smoke-add`.

require_relative "../lib/toy/models/transformer"
require_relative "../lib/tinynn"

a = Mat.new(2, 3)
a.flat[0] = 1.5;  a.flat[1] = 2.5;  a.flat[2] = -3.0
a.flat[3] = 0.0;  a.flat[4] = 4.0;  a.flat[5] = 100.0

b = Mat.new(2, 3)
b.flat[0] = -0.5; b.flat[1] = 7.5;  b.flat[2] =  3.0
b.flat[3] =  0.0; b.flat[4] = 1.0;  b.flat[5] = -50.0

native = a.plus(b)
ffi    = TinyNN.add(a, b)

i = 0
ok = true
while i < 6
  diff = native.flat[i] - ffi.flat[i]
  if diff < 0
    diff = -diff
  end
  puts "[" + i.to_s + "] native=" + native.flat[i].to_s + " ffi=" + ffi.flat[i].to_s
  if diff > 1.0e-4
    ok = false
  end
  i = i + 1
end
puts "match: " + ok.to_s
