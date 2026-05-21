# Draft: Tep #13 not fully fixed by #14 — 3-arg run! still mangles
# handler output
#
# To post at: https://github.com/OriPekelman/tep/issues/new

Reopening / followup to #13. Tep#14 added a default `scheduled =
false` parameter to `Tep.run!`, but the 3-arg call form still
silently miscompiles via Spinel — handlers still return `h1= ok`
instead of their actual body. The explicit 4-arg form works
correctly with the same Tep + same Spinel.

## Confirmation matrix

Same Tep `b07e548` (current main, post-Tep#14), same Spinel `f5bc710`,
same hello_api.rb except for the `run!` invocation:

```ruby
# Both forms:
class HelloHandler < Tep::Handler
  def handle(req, res)
    res.headers["Content-Type"] = "text/plain"
    "hello from tep + spinel\n"
  end
end
Tep.get "/", HelloHandler.new
```

| Invocation | Result |
|---|---|
| `Tep.run!(4567, 1, false)`           (3-arg, relies on default) | 200 OK, Content-Type `text/html`, body `h1= ok` (6 bytes) — wrong |
| `Tep.run!(4567, 1, false, false)`    (explicit 4-arg)            | 200 OK, Content-Type `text/plain`, body `hello from tep + spinel\n` — correct |

So Tep#14's `scheduled = false` default isn't actually firing. Likely
this is a Spinel default-argument codegen issue (Spinel emits the
warning "cannot resolve call to 'run!' on int (emitting 0)" then
produces a binary where the missing arg gets some non-default value).

## Bigger context — likely a Spinel issue surface

The literal "h1= ok" output is 6 bytes; the smallest plausible body
the server could send is `send_simple`'s `<h1>200 OK</h1>...` which
is way longer. So something is corrupting the response body in a
very specific way that's worth bisecting through Spinel.

Three options that route around it from Tep's side:

1. **Ditch default args entirely.** Provide TWO public entry points:
   ```ruby
   Tep.run_simple(port, workers, quiet)             # 3-arg, prefork
   Tep.run_scheduled(port, workers, quiet)          # 3-arg, scheduled
   ```
   `Tep.run!(port, workers, quiet, scheduled)` becomes a private
   4-arg form callers don't reach directly. This avoids the
   default-arg codepath entirely.

2. **Spinel-side fix.** Push the default-arg codegen bug upstream
   into matz/spinel. Slower; touches another project.

3. **Status quo + loud warning.** Detect 3-arg miscompile at
   Tep.run! entry and abort with a message pointing at the migration.
   Requires runtime detection of how many args got bound, which
   Spinel may not give us.

I'd vote option 1 — explicit run_simple / run_scheduled is honest
API design even setting aside the Spinel bug. Defaults invite this
class of "looks fine, silently wrong" surprises.

## Reproducer

Same as before:

```
git clone https://github.com/OriPekelman/toy_ruby_neural_network
cd toy_ruby_neural_network
./prep/sync_tep.rb                  # syncs current upstream Tep
make tep_demo/hello
./tep_demo/hello &
curl http://localhost:4567/         # → h1= ok  (wrong)
```

To verify the 4-arg form fixes it, edit `tep_demo/hello_api.rb`:

```diff
-Tep.run!(4567, 1, false)
+Tep.run!(4567, 1, false, false)
```

Rebuild. Now `curl` returns `hello from tep + spinel`.

## Friends in the ecosystem

Same 3-arg pattern is in every `tep_demo/openai_api_*` of the toy_ruby_neural_network repo (10 binaries). Each is broken
identically until the run_simple migration lands OR Spinel grows the
default-arg fix.

## A bigger Tep ask while we're here

Lazy battery loading. Today `require_relative "tep"` eagerly pulls
in `tep/pg`, `tep/sqlite`, `tep/net (sphttp)`, `tep/websocket`,
`tep/llm`, etc. Each carries a C-side dep (libpq, libsqlite3, etc).
For projects that only want the HTTP core, the eager require turns
"compile Tep" from "I have ruby + cc" into "I also need libpq-dev
installed because Spinel can't even analyze pg.rb without it"
(see the `@TEP_PG_CFLAGS@` placeholder situation).

Proposed: split `tep.rb` into the core (handler / server / router /
request / response / json / app / filter / streamer / parser /
session / version) and the batteries (pg / sqlite / net / websocket
/ llm / http / job / scheduler). The core requires only itself. Apps
that want a battery do `require_relative "tep/pg"` (or
`Tep.battery :pg` if there's a discovery layer).

That makes the "I just want an HTTP server" case work without
libpq, without sqlite, without websocket. Matches the framework
spirit: pay for what you use.

Happy to draft the PR if you'd like — let me know which option (1, 2,
or 3 above + lazy batteries) to pursue first.
