# Where does the `@TEP_*@` placeholder substitution belong?

**Date:** 2026-05-27. **Status:** design doc; nothing committed yet.
**Context:** companion to `spinelgems-tep-adoption-2026-05-27.md`. The
spinelgems Phase 1 migration shipped — `prep/sync_tep.rb` got replaced
by `prep/post_vendor_tep.rb`. The new script is smaller but it still
hardcodes the same four placeholders and the same pkg-config recipe.
That hardcoding is the smell.

## The actual problem

Tep ships three FFI shims as part of its `lib/`:
- `tep/net.rb`     binds `sphttp` (a C HTTP server) — links `sphttp.o`
- `tep/sqlite.rb`  binds SQLite via `tep_sqlite.o` + system `-lsqlite3`
- `tep/pg.rb`      binds Postgres via `tep_pg.o` + system pkg-config(`libpq`)

Each file says, at module top level:
```ruby
ffi_cflags "@TEP_SOMETHING@"
```

The placeholder is a build-time directive — Spinel's `ffi_cflags`
takes a **string literal**, not a Ruby expression. The comment in
`tep/net.rb` is unambiguous: *"Spinel doesn't support `__dir__` or
`ENV.fetch` in top-level `ffi_cflags`, so build-time substitution is
the cleanest portable shape."*

So we can't avoid substitution. The design question is **where the
substitution lives** and **where the values come from**.

## The placeholders fall into three categories

| Category | Examples | Source of truth |
| --- | --- | --- |
| **A. Self-referential paths** | `@TEP_SPHTTP_O@`, the `.o` part of `@TEP_SQLITE_O@`, `@TEP_PG_O@` | Tep itself — they're paths to `.o` files Tep's own `make` produces inside `lib/tep/` |
| **B. System library specs**   | The `-lsqlite3` part of `@TEP_SQLITE_O@`, `@TEP_PG_CFLAGS@`     | Consumer environment — pkg-config / brew / apt |
| **C. Opt-outs / disable**     | `TEP_PG_CFLAGS=-DNO_PG TEP_PG_LIBS=-lc` escape hatch            | Consumer policy ("I don't use Pg") |

The first category is *Tep's structural concern* (it knows its own
layout). The second is *consumer environment* (pkg-config etc). The
third is *consumer policy*. Today's `prep/post_vendor_tep.rb` does
all three by hand-rolling a single dict — that's the smell.

## Three approaches, ordered by ambition

### Approach 1: tep ships an FFI manifest, post-vendor reads it (recommended)

The cleanest fix: **Tep declares its own FFI shape**, post-vendor
reads the declaration + applies category-B (pkg-config) and category-C
(env opt-outs) generically. A toy-side hardcoded dict goes away.

Manifest lives at `tep/lib/tep/ffi_manifest.rb` (Spinel-friendly Ruby):

```ruby
module Tep
  # Declarative manifest of Tep's FFI shims. Read by post-vendor
  # tools to substitute @TEP_*@ placeholders at build time.
  # Each entry says: which file holds the placeholder(s), what
  # local .o to point at, which system libs are needed (pkg-config
  # names), and whether the module is optional (so a consumer can
  # TEP_DISABLE=pg-and-skip the whole battery).
  FFI_MANIFEST = [
    {
      module: "sphttp",
      file: "tep/net.rb",
      subs: { "@TEP_SPHTTP_O@" => :obj },
      obj: "tep/sphttp.o",
      pkg_config: nil,
      optional: false,
    },
    {
      module: "sqlite",
      file: "tep/sqlite.rb",
      subs: { "@TEP_SQLITE_O@" => :obj_plus_libs },
      obj: "tep/tep_sqlite.o",
      pkg_config: "sqlite3",
      pkg_config_fallback: "-lsqlite3",
      optional: true,
    },
    {
      module: "pg",
      file: "tep/pg.rb",
      subs: {
        "@TEP_PG_O@"      => :obj,
        "@TEP_PG_CFLAGS@" => :cflags_plus_libs,
      },
      obj: "tep/tep_pg.o",
      pkg_config: "libpq",
      pkg_config_fallback: nil,        # bail loud
      optional: true,
    },
  ].freeze
end
```

The post-vendor tool becomes a 50-line generic reader. Concretely:

```ruby
# In toy's prep/post_vendor_tep.rb (or upstreamed to spinelgems):
require_relative "../vendor/spinel/tep/lib/tep/ffi_manifest"

Tep::FFI_MANIFEST.each do |entry|
  # Skip optional modules the consumer opted out of.
  next if entry[:optional] && ENV["TEP_DISABLE"]&.split(",")&.include?(entry[:module])
  # Resolve every placeholder per entry[:subs], apply to entry[:file].
  resolve_and_substitute(entry)
end
```

**Benefits**:
- Tep is the **single source of truth** for its own FFI shape. Adding
  a new FFI shim (e.g. `tep/redis`) is one manifest entry; toy
  doesn't change.
- The "I don't use Pg" opt-out becomes structured
  (`TEP_DISABLE=pg`) instead of "set the right env-var pair to make
  the placeholder substitute a fake C flag." The runtime-failure
  pattern dies.
- spinelgems gets a reusable "FFI post-vendor" hook — toy is no
  longer alone in needing it, and any other consumer of Tep
  (Roundhouse, future Tao) benefits.

**Costs**:
- Two-side change: needs a small commit on Tep, and a rewrite of
  `post_vendor_tep.rb` on toy.
- The "what if Tep manifest version doesn't match the consumer's
  reader" question is mild — manifests are read at vendor time, so
  the consumer's reader is in lockstep with whatever Tep version
  was vendored (`vendor/spinel/tep/lib/tep/ffi_manifest.rb`).
- We're choosing not to upstream this *into* `spinel-compat` itself
  yet. It can live in the consumer's prep/ until a second consumer
  appears that wants it. Premature framework-ization is its own
  smell.

### Approach 2: tep ships pre-baked absolute paths via `gem build`

A weaker version: Tep's `gemspec` runs a hook at `gem build` time
that walks `lib/tep/*.rb` and substitutes category-A (self-references)
based on the gem's intended install location. Consumers only deal
with category-B (system libs).

Doesn't actually work for our flow: spinelgems' `vendor` copies
the gem's `lib/` verbatim, but the absolute-path substitutions
would assume an install location, and tep isn't gem-installed
under the spinelgems convention — it's path-vendored. So the
pre-baked paths would be wrong.

### Approach 3: Spinel grows expression evaluation for `ffi_cflags`

If Spinel's compiler could fold `File.expand_path("sphttp.o",
__dir__)` to a string literal at compile time, the placeholders
disappear entirely. tep/net.rb becomes:

```ruby
ffi_cflags File.expand_path("sphttp.o", __dir__)
```

…and the C link command knows where to find it from any vendored
location.

**This is the long-term right answer.** It pushes the work to where
it logically belongs (compile-time constant folding), removes a
whole category of build-time templating across every Spinel
project that ever ships a C-ext, and is a single Spinel-side change
benefiting everyone.

**But**: it's a Spinel feature request, not a toy/tep concern. Worth
filing upstream as `spinel#NNN: const-fold __dir__ + File.expand_path
in top-level ffi_cflags`. Until then, the build-time substitution
shim is the workaround.

## Recommendation

**Ship Approach 1 now; file Approach 3 as the long-term cleanup.**

Approach 1 is small (~50 LOC manifest + 50 LOC reader, replacing
~80 LOC of hardcoded substitution), is clearly correct, and
addresses the actual user complaint ("we still hardcode the same 4
placeholders"). It doesn't preempt Approach 3 — when Spinel grows
constant folding, the manifest just gets thinner.

## What about namespacing + shared types?

Two distinct adjacent questions:

**Namespacing**: Tep's FFI modules live at top level (`Sock`,
`Sqlite`, `Pg`) — not under `Tep::Sock` etc. The comment in
`tep/net.rb` says: *"All FFI plumbing lives at the top level so
spinel's name resolver finds it from anywhere in the Tep tree
(nested modules confuse it)."* This is a Spinel-side limit, not a
design choice. Once Spinel's name resolver handles nested modules
robustly, renaming `Sock` → `Tep::Sock` (etc.) is mechanical and
removes the only real toy↔tep namespace collision we've hit so far
(`Mat#add` vs `Tep::Router#add`, addressed at the time by renaming
`Mat#add` → `Mat#plus`).

**Shared types**: toy and tep share basically nothing structural.
The only contact surface is what crosses the Tep B7 OpenAI-server
boundary — the `Backend` interface (token IDs in/out, optional
streaming), and the `events.jsonl` schema (frozen `toy/v1` at
[`docs/events-schema.md`](../events-schema.md)). Both are
ad-hoc-but-documented contracts that work fine. A shared-types gem
(`toy_tep_protocol`) would formalise them but adds a third place
for `bundle lock`-style coordination — not worth it until we
actually have a third consumer that benefits.

## Open questions

- Does spinel-compat want to grow a generic "C-ext post-vendor
  hook" surface (so toy's `post_vendor_tep.rb` becomes a config
  file, not a Ruby script)? Probably yes eventually; not now.
- Should the manifest live in `lib/tep/ffi_manifest.rb` (loaded
  the same way as other Tep modules) or in a sibling file like
  `tep_ffi.toml`? Ruby is convenient (no extra parser) and Tep is
  Spinel-compatible so we can read it from a normal Ruby script;
  but a TOML/JSON shape would be safer for non-Ruby consumers if
  any appear. Defaulting to Ruby for now.
- What's the precedence between manifest defaults and consumer
  env overrides? Today: env wins. Likely correct.
