# spinelgems adoption: replacing prep/sync_tep.rb with the Gemfile convention

**Date:** 2026-05-27. **Status:** plan + drafts; blocked on tep#95.
**Context:** the sibling [spinelgems](https://github.com/OriPekelman/spinelgems)
project ships an experimental Gemfile convention for Spinel-compiled
projects. Its [adoption guide](https://github.com/OriPekelman/spinelgems/blob/main/docs/adoption.md)
calls out toy as the flagship consumer — the rsync-and-substitute
hack in `prep/sync_tep.rb` + `tep_demo/_tep_lib/` is literally the
pain it was designed to retire.

## Current state — the rsync hack

`prep/sync_tep.rb` (the existing approach):
1. `rsync ~/sites/tep/lib/  →  tep_demo/_tep_lib/`
2. Substitute `@TEP_SPHTTP_O@` / `@TEP_SQLITE_O@` / `@TEP_PG_O@`
   placeholders that point at pre-built `.o` files in tep's checkout
3. `tep_demo/*.rb` does `require_relative "_tep_lib/tep"`

Pain:
- Manual: re-run after every tep checkout update.
- Duplicates tep's source into our tree; the rsync side IS tep
  HEAD-pinned but easy to drift.
- The placeholder substitution is a separate concern hand-folded in.

## Phase 1 — what the new convention looks like

The dogfood doc's recipe, applied here:

```ruby
# toy_ruby_neural_network/Gemfile
source "https://rubygems.org"
ruby "3.2.3", engine: "spinel", engine_version: "0.0.0"

# Sibling — path: for the gx10 dev loop, git: once we want
# reproducibility across machines.
gem "tep", path: "../tep"
```

Then:

```sh
bundle lock                                 # Gemfile.lock
spinel-compat check Gemfile.lock            # gate (passes prism is dev-only)
spinel-compat vendor                        # vendor/spinel/tep/lib/… + deps.rb
# tep_demo/*.rb:  require_relative "../vendor/spinel/deps"
# (replaces:      require_relative "_tep_lib/tep")
```

The `.o`-placeholder substitution stays as a thin post-vendor step
on the toy side until spinel-compat has a first-class C-ext story.

## Blockers (tep-side; we don't commit to tep directly)

Filed as [tep#95](https://github.com/OriPekelman/tep/issues/95) —
two issues in the one gemspec:

1. `s.add_runtime_dependency "prism", "~> 1.0"` should be
   `add_development_dependency`. prism is build-time only (used by
   `bin/tep`'s translator); Spinel-compiled binaries don't need it.
   As a runtime dep, it pulls a native C-ext gem into every
   consumer's lockfile and trips the spinelgems gate.
2. `s.required_ruby_version = ">= 3.4.0"` (rationale: "Prism is
   bundled with Ruby 3.3+") is moot once #1 lands. The Spinel
   binaries don't run Ruby. Dropping to `>= 3.2.0` lets gx10 (Ruby
   3.2.3) bundle-lock without spinning up a newer interpreter.

## Local diagnostics done today

- `spinel-compat probe tep --dir ~/sites/tep` returns `~ risky` —
  static analysis flags `define_method`, `class_eval`, `send`,
  `binding` (known Tep features; the probe is appropriately
  cautious). The `verified` rung (CRuby-vs-Spinel differential)
  is the trust level worth landing once Phase 1 is wired.
- gx10 has Ruby 3.2.3; Tep's current gemspec rejects it at
  `bundle lock` time before any of the spinelgems machinery can
  run. (Hence tep#95's second-bullet ask.)
- gx10 bundler installed at `~/.local/share/gem/ruby/3.2.0/bin/bundle`
  via `gem install --user-install bundler`. Add it to PATH (or use
  the full path) before running the spinelgems flow.

## Sequencing

```
tep#95 closes (gemspec fix)
       │
       ▼
Phase 1a:  bundle lock  →  spinel-compat check  →  spinel-compat vendor
       │
       ▼
Phase 1b:  one tep_demo app switches its require to vendor/spinel/deps
           (diff side-by-side vs the rsync path; bench equality)
       │
       ▼
Phase 1c:  retire prep/sync_tep.rb (keep the .o placeholder helper
           as a thin post-vendor hook)
       │
       ▼
Phase 2:   spinel-compat verify tep --smoke <smoke.rb>   →  tep earns
           the `verified` rung; first real ledger entry; spinelgems.org
           catalog gets content.
```

## What we ship without unblocking tep#95

This doc + a draft `Gemfile.toy.draft` (committed but not picked up
by bundler — `.draft` suffix is intentional). When tep#95 lands,
rename to `Gemfile`, `bundle lock`, run the rest of Phase 1. Nothing
in the toy code changes until then; the existing `prep/sync_tep.rb`
keeps working.
