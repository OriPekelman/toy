## Summary

When two unrelated classes define a same-named accessor that returns different element types, Spinel's whole-program type inference unifies the field's return type to `sp_RbVal`, breaking *every* call site of that field across the program — even sites where the receiver type is statically obvious.

This is the *field-name* counterpart to the local-variable collapse documented in #538 (local-var name unification). Same mechanism, different binding form.

## Minimal reproduction

```ruby
class HouseDog
  attr_accessor :friends                 # → Array<Dog>
  def initialize
    @friends = [Dog.new]
  end
end

class HouseCat
  attr_accessor :friends                 # → Array<Cat>
  def initialize
    @friends = [Cat.new]
  end
end

dog = HouseDog.new
cat = HouseCat.new
puts dog.friends[0].bark    # cannot resolve call to 'bark' on sp_RbVal
puts cat.friends[0].meow    # cannot resolve call to 'meow' on sp_RbVal
```

CRuby: prints `woof` and `meow`. Spinel: both call sites fail because `friends` collapses across the two classes.

## Where it bit me

Building a sugar-API rewrite of a GPT-2 model. The existing `TransformerLM` (training-time model) has `@blocks : Array<Block>`. The new `Toy::GPT2` (inference-time sugar) had `@blocks : Array<Toy::GPT2Block>` (different element class). Both classes coexist in the same compiled program. Every `model.blocks[i]` call became `sp_RbVal`, and every method called on a block (`.attn`, `.ln1`, `.ffn`) failed.

Workaround: rename the new field. I called it `Toy::GPT2#stack`. The receiver-type-obvious rename fixed everything.

## Suggested fix

Field/method lookups are receiver-typed: `model.blocks` should resolve in the context of `model`'s static type. The current behavior — unifying the *result* type across all classes that expose the same accessor name — discards the receiver-type context that's already in scope.

If a global rename isn't feasible, even a warning at compile time ("attribute `blocks` defined with two different return types in `HouseDog` and `HouseCat`; result will be `sp_RbVal`") would save users from chasing the failure six method-calls deep into call-site code.

## Why this is annoying

Field names like `blocks`, `layers`, `cache`, `state`, `config` are *natural* in transformer / ML / app code. The first instinct is to use the obvious name; the cost of the collapse is invisible until something deep downstream loses its type.

## Environment

- Spinel: master @ a9dabfa
- OS: macOS 26 (Apple Silicon)
- Related: local-var name collapse (file 05 in this directory)
