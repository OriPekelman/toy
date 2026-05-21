## Summary

For `Hash<String, Int>` reads, a stored value of `0` is indistinguishable from a missing key — both look like `nil` to a `!= nil` check, but the same value also prints as `0` via `inspect`. This makes the standard "look it up, check for nil, otherwise use the value" idiom impossible to use safely when the value range includes 0.

Related to (but distinct from) PR #491 (`use ruby truthiness for scalar conditions`) — that one fixed scalar truthiness in `if`/`while` etc, but the same conflation persists for `!= nil` on int-typed hash retrievals.

## Minimal reproduction

```ruby
h = {}
h["x"] = 0    # store real 0
h["y"] = 5    # store real 5

v_x       = h["x"]   # expected 0
v_y       = h["y"]   # expected 5
v_missing = h["z"]   # expected nil

puts "v_x.inspect=#{v_x.inspect}   v_x != nil → #{(v_x != nil).inspect}   v_x == nil → #{(v_x == nil).inspect}"
puts "v_y.inspect=#{v_y.inspect}   v_y != nil → #{(v_y != nil).inspect}   v_y == nil → #{(v_y == nil).inspect}"
puts "v_missing.inspect=#{v_missing.inspect}   v_missing != nil → #{(v_missing != nil).inspect}"
```

## Expected behaviour (CRuby)

```
v_x.inspect=0   v_x != nil → true   v_x == nil → false
v_y.inspect=5   v_y != nil → true   v_y == nil → false
v_missing.inspect=nil   v_missing != nil → false
```

## Actual behaviour (Spinel `a9dabfa`)

```
v_x.inspect=0   v_x != nil → false   v_x == nil → true
v_y.inspect=5   v_y != nil → true    v_y == nil → false
v_missing.inspect=0   v_missing != nil → false
```

`v_x.inspect` says `0` but `v_x == nil` is `true`. `v_missing` also looks like `0`. The two cases collapse onto each other.

## Why it bites in practice

A GPT-2 BPE merges table stores integer ranks 0..49999, keyed by token-pair string. Rank 0 is GPT-2's highest-priority merge (`"Ġ t"` — leading space + word-starting consonant). With the natural code:

```ruby
rank = merges[key]
if rank != nil && rank < best_rank
  ...
end
```

the rank-0 case is silently skipped — every space-prefixed word tokenizes wrong. Visible to the user as "different IDs from HF transformers on identical input"; no warning, no error.

Workaround: store `rank + 1` and treat 0 as missing. Works but is a foot-gun for anyone touching the code later.

## Suggested

Either:
1. Make `int == nil` follow standard Ruby semantics (false for non-nil ints, regardless of value); OR
2. Use an explicit sentinel for "missing" that is *not* 0 (e.g. `LLONG_MIN`) and have `inspect` print it as `nil`; OR
3. Document the constraint and offer a `Hash#has_key?` / `Hash#fetch` path that's reliable.

## Environment

- Spinel: master @ a9dabfa
- OS: macOS 26 (Apple Silicon)
