# A/B parity for element-wise SiLU. silu(x) = x * sigmoid(x).
# Native reference matches Toy.silu! (lib/toy.rb).

require_relative "../lib/toy"

# Spread of values: zero, small positive, large positive, negative, large negative.
a = Mat.new(1, 5)
a.flat[0] =  0.0
a.flat[1] =  0.5
a.flat[2] =  3.0
a.flat[3] = -0.5
a.flat[4] = -3.0

# Native reference: copy then silu! in place.
native = Mat.new(1, 5)
i = 0
while i < 5
  native.flat[i] = a.flat[i]
  i = i + 1
end
Toy.silu!(native)

ffi = TinyNN.silu(a)

i = 0
ok = true
while i < 5
  diff = native.flat[i] - ffi.flat[i]
  if diff < 0
    diff = -diff
  end
  puts "[" + i.to_s + "] x=" + a.flat[i].to_s +
       " native=" + native.flat[i].to_s + " ffi=" + ffi.flat[i].to_s
  if diff > 1.0e-5
    ok = false
  end
  i = i + 1
end
puts "match: " + ok.to_s
