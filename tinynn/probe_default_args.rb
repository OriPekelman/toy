# Probe landmine #4 — `def f(x = nil)` widens x to RbVal across the
# whole compiled program, poisoning every downstream call.

def lookup(key = nil)
  if key == nil
    return "(none)"
  end
  "found:" + key
end

puts lookup("alpha")
puts lookup
puts lookup("beta")

# Pin the return type for the caller
r1 = lookup("gamma")
puts "r1 length=" + r1.length.to_s
