## Summary

`Array#pop` on a nested integer array (`Array<Array<Int>>`) is not handled correctly. Spinel emits a codegen warning, continues compilation, and replaces the `pop` call with `0`. The runtime call is a silent no-op — the array still contains the original element(s).

Same pattern as issue #156 (`Array#transpose` on nested int arrays), but for `pop`.

## Minimal reproduction

```ruby
a = [[1, 2]]
puts "before pop: #{a.inspect}"
a.pop
puts "after  pop: #{a.inspect}"
a.push([3, 4])
puts "after push: #{a.inspect}"
```

## Expected behaviour (CRuby)

```
before pop: [[1, 2]]
after  pop: []
after push: [[3, 4]]
```

## Actual behaviour (Spinel `a9dabfa`)

```
$ spinel repro.rb -o /tmp/repro
warning: in (top level): cannot resolve call to 'pop' on int_array_ptr_array (emitting 0)

$ /tmp/repro
before pop: [[1, 2]]
after  pop: [[1, 2]]    # pop was silently a no-op
after push: [[1, 2], [3, 4]]
```

## Why it bites in practice

The "seed-and-pop" idiom — `arr = [first_element]; arr.pop` — is the standard way to pin Spinel's type inference for an otherwise-empty array of known element type. It works for `Array<String>`, `Array<Mat>`, etc., but silently fails on `Array<Array<Int>>` due to this bug. Downstream, the seed stays in and gets processed alongside real data, producing visibly-wrong output. In my case (a BPE tokenizer accumulator), every encode call leaked a phantom `\0` byte at position 0.

Workaround: inline whatever needed the accumulator and avoid `Array<Array<Int>>` entirely.

## Environment

- Spinel: master @ a9dabfa
- OS: macOS 26 (Apple Silicon)
