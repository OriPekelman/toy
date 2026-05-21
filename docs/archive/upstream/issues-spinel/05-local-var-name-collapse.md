## Summary

When the same local-variable (or parameter) name is bound to two different types at two unrelated sites in a Spinel-compiled program, the type of *both* sites collapses to `sp_RbVal`, breaking type inference at both. This is the local-binding counterpart to issue #04 (field-name collapse).

This is the single most common surprise I've hit in ~2000 lines of Spinel-compiled transformer code. The symptom is far from the cause: typically a method 200 lines away starts emitting "cannot resolve call to X on sp_RbVal" warnings because somewhere else in the program a local with the same name was assigned a different type.

## Minimal reproduction

```ruby
class Dog
  def bark; "woof"; end
end

class Cat
  def meow; "meow"; end
end

def visit_dog
  pet = Dog.new
  puts pet.bark         # cannot resolve call to 'bark' on sp_RbVal
end

def visit_cat
  pet = Cat.new
  puts pet.meow         # cannot resolve call to 'meow' on sp_RbVal
end

visit_dog
visit_cat
```

CRuby: prints `woof` then `meow`. Spinel: both fail because the *name* `pet` unifies across the two methods.

The same collapse happens for *formal parameters* of methods with the same param name across the program:

```ruby
def feed(pet)
  pet.eat               # collapses if `pet` is used as two types elsewhere
end
```

## Where it bit me, repeatedly

Real cases from the same project this week:

1. `cache` — a `GPT2FullForwardFFICache` in one method, a `GPT2KVFFICache` in another. Workaround: renamed to `fwd_cache` and `kv_cache`.
2. `tensor` — used as a method parameter in two different `upload_*` helpers with different element types. Workaround: renamed one to `dl_handle`.
3. `blk` — a top-level `GPT2Block` in `load_gpt2`, a `Toy::GPT2Block` in `load_toy_gpt2` (same file). Workaround: `tblk` in the second site.
4. `rms` — a `Toy::RMSNorm` local in a smoke test, an `Array<Float>` parameter in `TransformerLM#rms_norm_backward`. Workaround: renamed to `rnorm`. Symptom was an arithmetic-vs-sp_RbVal error in the *backward* method, 700 lines away from the smoke.

Each time the symptom was several files away from the cause. None of these are intuitive to debug from the error message alone — the user has to *know* about this collapse mechanism.

## Suggested fix

Local-variable scope is method-local. Method-parameter scope is method-local. Unifying types across method boundaries by name is a design choice; with method-local scope it's not needed.

If a full per-method scoping change is too invasive: emit a compile-time warning that points at the collision (`"local 'pet' has incompatible types Dog (in visit_dog) and Cat (in visit_cat); inferred type sp_RbVal at both sites"`). Most of the debugging time is spent *finding* the collision; the fix is trivial once located.

## Why this matters

The collapse forces a global naming convention on every project: pick unique names for every local across the entire codebase. That's not Ruby. It's also the dominant source of "Spinel compiled but the output is wrong / segfaults" surprises in our experience.

## Environment

- Spinel: master @ a9dabfa
- OS: macOS 26 (Apple Silicon)
- Related: #04 (field-name collapse, the same-but-for-accessors)
