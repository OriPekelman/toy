# A/B parity for element-wise multiply: c = a * b.
# Native reference matches Toy.hadamard! (lib/toy.rb).

require_relative "../lib/toy"

# Two small mats with hand-set values for predictable parity.
mu_a = Mat.new(2, 3)
mu_a.flat[0]=1.0; mu_a.flat[1]=2.0; mu_a.flat[2]=3.0
mu_a.flat[3]=4.0; mu_a.flat[4]=5.0; mu_a.flat[5]=6.0
mu_b = Mat.new(2, 3)
mu_b.flat[0]=0.5; mu_b.flat[1]=2.0; mu_b.flat[2]=0.0
mu_b.flat[3]=-1.0; mu_b.flat[4]=0.25; mu_b.flat[5]=10.0

# Native reference.
native = Mat.new(2, 3)
i = 0
while i < 6
  native.flat[i] = mu_a.flat[i]
  i = i + 1
end
Toy.hadamard!(native, mu_b)

ffi = TinyNN.mul(mu_a, mu_b)

i = 0
ok = true
while i < 6
  diff = native.flat[i] - ffi.flat[i]
  if diff < 0
    diff = -diff
  end
  puts "[" + i.to_s + "] " + mu_a.flat[i].to_s + " * " + mu_b.flat[i].to_s +
       " native=" + native.flat[i].to_s + " ffi=" + ffi.flat[i].to_s
  if diff > 1.0e-6
    ok = false
  end
  i = i + 1
end
puts "match: " + ok.to_s
