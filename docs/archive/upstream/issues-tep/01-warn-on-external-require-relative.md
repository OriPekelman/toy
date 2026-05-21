## Summary

`bin/tep build` silently drops external `require_relative` lines (anything not pointing at `tep/lib`). Apps that use Tep but also `require_relative` their own project files end up with "uninitialized constant" warnings *from spinel*, dozens of lines after the actual cause, with no hint that the require itself was ignored.

A one-line warning at translation time would point users straight at the cause.

## Reproduction

```ruby
# myapp.rb
require_relative "../my_project/lib/widgets"

get '/' do
  Widgets.greeting
end
```

```
$ bin/tep build myapp.rb -o /tmp/myapp
warning: in TepRoute_0#handle: uninitialized constant 'Widgets' (emitting 0)
warning: in TepRoute_0#handle: cannot resolve call to 'greeting' on int (emitting 0)
...
```

Nothing in the output mentions that `require_relative "../my_project/lib/widgets"` was dropped. Users hunt for the "uninitialized constant" trail through ~3000 lines of inlined Tep before noticing.

## Proposed fix

In `bin/tep`, where the translator currently does

```ruby
when "require", "require_relative" then return  # ignore
```

emit a warning when the require path is *outside* the Tep tree:

```ruby
when "require", "require_relative"
  if name == "require_relative" && node.arguments && node.arguments.arguments.size == 1
    arg = node.arguments.arguments.first
    if arg.is_a?(Prism::StringNode)
      path = arg.unescaped
      # tep's own libs ('tep', 'tep/...') are inlined elsewhere — ignore those.
      unless path == "tep" || path.start_with?("tep/")
        warnings << "external `require_relative #{path.inspect}` is ignored " \
                    "(tep build only inlines tep/lib); inline the file " \
                    "manually or use a build wrapper"
      end
    end
  end
  return
```

That makes the failure mode self-explanatory at the same call site, in the same log run, without changing semantics.

## Stretch

Two follow-up options if external-require support is in scope:

1. **Recursively inline** non-Tep `require_relative` chains the same way Tep inlines its own libs. Would need the user to opt in (`--inline-external`) since it can pull in arbitrary code.
2. **Pass-through** `require_relative` to spinel. The comment on the existing drop says "Spinel's require_relative is finicky about long paths and absolute paths — both fail silently"; that's still true, but for short relative paths in the same project it works in practice (the `toy_ruby_neural_network` project's pre-bin/tep handlers used to do this).

## Why this came up

Building an OpenAI-compatible chat completions endpoint on top of a real-model inference stack. The handlers need `require_relative "../lib/gpt2_ffi_kv"`, `"../lib/bpe"`, etc. The silent drop made the 50-warning spinel output the only signal anything was wrong; I worked it out by reading the translated file directly. Wrote a wrapper script (`prep/build_tep_app.sh`) that concatenates the project libs onto the source before invoking `bin/tep build`.

## Environment

- Tep: master @ aa8ae63
- Spinel: master @ a9dabfa
