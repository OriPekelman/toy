## Summary

`String#index` returns **-1** in Spinel when the pattern is not found, while CRuby returns **nil**. The natural Ruby idiom `if pos = s.index(needle)` therefore fires for "not found" (because `-1` is truthy), and `if pos.nil?` never fires. Combined with #521 (`0 != nil` returning `false`), every `(pos != nil)` check on a `String#index` result is broken: at position 0 it's wrong because of #521, at "not found" it's wrong because of the -1 sentinel.

## Minimal reproduction

```ruby
s = "hello world"
puts "s.index('world')  = " + s.index("world").inspect    # CRuby: 6           Spinel: 6
puts "s.index('xyz')    = " + s.index("xyz").inspect      # CRuby: nil         Spinel: -1
puts "s.index('hello')  = " + s.index("hello").inspect    # CRuby: 0           Spinel: 0

pos = s.index("xyz")
puts "  pos != nil      = " + (pos != nil).inspect        # CRuby: false       Spinel: true
puts "  pos.nil?        = " + pos.nil?.inspect            # CRuby: true        Spinel: false

pos = s.index("hello")
puts "  pos != nil      = " + (pos != nil).inspect        # CRuby: true        Spinel: false (#521)
puts "  pos.nil?        = " + pos.nil?.inspect            # CRuby: false       Spinel: true
```

Either bug alone would force users into workarounds; the combination makes `String#index` essentially unusable through a `nil` lens — every check is wrong in at least one of the three buckets (-1 / 0 / positive).

## Suggested fix

Match CRuby semantics for `String#index`: return `nil` (an actual nil value, not 0 and not -1) when the pattern is absent. That + the #521 fix for `Int 0` would restore the natural idiom.

If keeping Int-only returns is important for performance, an explicit `Hash#has_key?`-style companion (`String#index?`, `String#has?`) that returns a bool would also work.

## Why it bit me

Walking a JSON body to find every `"content":"..."` occurrence — the body parser of an OpenAI-compatible chat-completions endpoint. The natural loop:

```ruby
while true
  pos = body.index('"content"', i)
  break if pos.nil?
  ...
  i = pos + needle_len
end
```

with Spinel ends up either:
- at not-found: `pos = -1`, the `nil?` check is false → infinite loop using `i + needle_len`, eventually reads past the end and Bus-errors.
- at first match (pos=0): `pos.nil?` is true → break, parser returns empty.

I rewrote the parser as a manual byte-level scan with no `String#index`.

## Environment

- Spinel: master @ a9dabfa
- OS: macOS 26 (Apple Silicon)
- Related: #521 (Int 0 == nil)
