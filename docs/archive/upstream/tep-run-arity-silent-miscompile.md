# Draft: Tep issue — 3-arg Tep.run! call silently miscompiles to a
# corrupted server (handler bodies render as `h1= ok`)
#
# To post at: https://github.com/OriPekelman/tep/issues/new

## Summary

When a user app calls the legacy 3-arg form `Tep.run!(port, workers,
quiet)`, Spinel compiles the binary with a warning (`cannot resolve
call to 'run!' on int (emitting 0)`) and the resulting server runs
but **every handler's response body is replaced with the 6-byte
string `h1= ok`**. The handler's status code is correct (200 OK for
matched routes; 404 for unmatched), the route table matches normally,
the handler's `res.headers["Content-Type"]` setter is ignored —
Content-Type comes back as `text/html; charset=utf-8` even when the
handler set `text/plain`.

The trigger is Tep's `def self.run!(port, workers, quiet, scheduled)`
signature requiring a 4th `scheduled` argument that older apps don't
pass. Passing the 4th arg explicitly (`Tep.run!(port, 1, false,
false)`) builds a binary that behaves correctly — same handler, same
route, same Spinel.

## Reproducer

Repo: https://github.com/OriPekelman/toy_ruby_neural_network @
`39af647` (or any recent ref — `tep_demo/hello_api.rb` and every
`tep_demo/openai_api_*` ship the 3-arg form).

```bash
git clone https://github.com/OriPekelman/toy_ruby_neural_network
cd toy_ruby_neural_network
make setup-ggml
make tep_demo/hello
./tep_demo/hello &
curl -v http://localhost:4567/
# → 200 OK, Content-Type: text/html; charset=utf-8, body: "h1= ok"
```

Expected (handler returns `"hello from tep + spinel\n"` with
Content-Type `text/plain`):

```
< HTTP/1.1 200 OK
< Content-Type: text/plain
< Content-Length: 24
< 
hello from tep + spinel
```

Actual:

```
< HTTP/1.1 200 OK
< Content-Type: text/html; charset=utf-8
< Content-Length: 6
< 
h1= ok
```

## Minimal-diff fix that restores correctness

Change the run! call site from

```ruby
Tep.run!(4567, 1, false)
```

to

```ruby
Tep.run!(4567, 1, false, false)
```

Confirmed against the same Tep source + Spinel `f5bc710`. Hex-checked
the response body — handler's `"hello from tep + spinel\n"` (24 bytes)
comes through unmangled, `text/plain` header is preserved, etc.

## Where this likely sits

Tep's `lib/tep.rb` `def self.run!(port, workers, quiet, scheduled)`
was extended to take `scheduled` (probably alongside the
Server::Scheduled / per-fiber GC root fix in `c7d247b`). The change
broke source-level compatibility with every user app that calls the
3-arg form.

Spinel's behaviour on the arity mismatch is to:
- Emit a warning: `cannot resolve call to 'run!' on int (emitting 0)`.
- Produce a buildable binary.
- The binary's server returns a fixed 6-byte body for every matched
  route, with `text/html; charset=utf-8` regardless of handler
  intent.

The 6-byte body `h1= ok` is suspicious — it's not a substring of the
404 page or any obvious literal. Possibly it's the result of Spinel
emitting `0` (per the warning) where a `bool` was expected, which
flows through some Server-side conditional and ends up rendering a
default that I haven't traced.

The "correct" caller behaviour now is the 4-arg form. The questions
for Tep:

1. **Restore source-level compatibility.** Could `run!` accept either
   3 or 4 args (default `scheduled = false`)? Spinel doesn't support
   default args in the C signature directly, but a wrapper /
   overload pattern might. Or: provide `Tep.run_simple(port,
   workers, quiet)` as the documented 3-arg entry point.
2. **Fail loudly on arity mismatch.** The current behaviour (build a
   silently-corrupted server) is the worst case — users see "compiles
   clean (modulo a warning), server starts, returns 200" and assume
   it works. Either Tep or Spinel should refuse to build the binary
   instead of producing a broken one. (May be a Spinel-side fix
   request, depending on what's tractable.)
3. **Document the change in README + examples.** The Tep upstream
   `examples/hello.rb` should use the 4-arg form so anyone copying
   it gets a working server.

## Affected demos in this project (for migration)

All of these need updating to the 4-arg form once Tep's compat story
is settled:

```
tep_demo/hello_api.rb               Tep.run!(4567, 1, false)
tep_demo/inference_api.rb           Tep.run!(4567, 1, false)
tep_demo/openai_api_smollm2.rb      Tep.run!(__port, __workers, __quiet)
tep_demo/openai_api_qwen25_1.5b.rb  Tep.run!(__port, __workers, __quiet)
tep_demo/openai_api_qwen25_1.5b_q8.rb           (same)
tep_demo/openai_api_qwen25_3b.rb                (same)
tep_demo/openai_api_qwen25_3b_q8.rb             (same)
tep_demo/openai_api_qwen25_7b.rb                (same)
tep_demo/openai_api_qwen25_7b_q8.rb             (same)
```

`examples/04_serve_http.rb` already uses the 4-arg form.

## What I tested

- 3-arg call site (`tep_demo/hello`): broken — `h1= ok` body.
- 4-arg call site (verbatim same Ruby source, just adding `, false`):
  works — handler's body comes through unmodified, Content-Type
  honoured.
- Tested across both GET and POST routes (`tep_demo/openai_api_*`
  show the same behaviour on `/v1/completions`).
- Confirmed against Spinel `f5bc710` (latest as of 2026-05-21).

Happy to add more bisect detail or run further experiments. Let me
know if a smaller standalone repro outside the toy_ruby_neural_network
tree would help.
