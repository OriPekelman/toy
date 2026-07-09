# Probe — Spinel landmine #16: Struct.new accessor names participate in
# cross-module, name-based type unification, so a Struct member collides
# with an unrelated same-named method/field and mis-compiles distant code.
#
# Hit 2026-05-29 while extracting an L2 TransformerBlock: a per-forward
# context built as `Struct.new(:scale, :eps, :n_kv, :n_heads, :t, :b,
# :attn_mask)` could not be C-compiled — the Struct's generated `#b`
# reader unified with `Toy::Linear#b` (a Float-array weight), poisoning
# the *integer* `batch` argument at an unrelated call site.
#
# This minimal repro isolates that: `Weight#b` returns a Float array;
# a `Struct.new(:t, :b)` member `b` is meant to be an Integer used as a
# loop bound. Expectation under correct semantics: both compile and run,
# `ctx.b` is the Integer 5. Observed: the name-keyed merge of `#b`
# breaks one of the two sites.

class Weight
  def initialize
    @v = [1.0, 2.0, 3.0]   # Float array
  end

  # Same accessor *name* as the Struct member below.
  def b
    @v
  end
end

Ctx = Struct.new(:t, :b)   # member :b collides with Weight#b

# Plain integer arithmetic; `batch` must stay an Integer.
def sum_to(batch)
  r = batch % 7    # modulo: hard C error if `batch` is mis-typed as a pointer
  i = 0
  s = 0
  while i < batch
    s = s + i
    i = i + 1
  end
  s + r
end

w = Weight.new
puts "weight.b.length = " + w.b.length.to_s   # Float-array use of #b

ctx = Ctx.new(3, 5)                            # ctx.b is Integer 5
puts "ctx.b = " + ctx.b.to_s                   # Integer use of #b
puts "sum_to(ctx.b) = " + sum_to(ctx.b).to_s   # Integer arg — collision target

# Stronger: index an Array<Int> with ctx.b. If #b was unified to the
# Float-array pointer type, this emits an invalid C array subscript.
table = [10, 20, 30, 40, 50, 60]
puts "table[ctx.b] = " + table[ctx.b].to_s

puts "PROBE-PASS"
