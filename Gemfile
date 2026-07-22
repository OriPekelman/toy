source "https://rubygems.org"
ruby "3.2.3", engine: "spinel", engine_version: "0.0.0"

# Tep is consumed as a RELEASED gem via the spinelgems convention
# (bundler-spinel). No hand-rolled vendoring or @TEP_*@ substitution:
# `bundle lock` resolves tep, `spinel-compat vendor` copies tep's lib/ into
# vendor/spinel/tep/ AND natively compiles+wires its C extensions (driven by
# tep's shipped spinel-ext.json), then a Spinel entrypoint does
# `require_relative "vendor/spinel/deps"`.
#
# Pinned to the published gem (toy#31; was the git:main pre-publish stand-in).
# tep 0.11.x ships artifact-free with prism as a DEV dep + spinel-ext.json at
# the gem root, so it resolves + vendors cleanly. `~> 0.11.6` (toy#103): PG is
# opt-in (tep#216, serve compiles without libpq), boot-safe seeds (tep#223),
# spinel_kit as a declared runtime dep (tep#220 — one shared copy, clears the
# SpinelKit::Json double-bundle key-blanking, tep#213).
gem "tep", "~> 0.11.6"

# SpinelKit — the shared Spinel stdlib-surface gem (toy#44). 0.3.0 retired the
# JSON codec (toy absorbed it as Toy::Json, lib/toy/io/json_builder.rb); toy
# consumes spinel_kit/git only, and git.rb is byte-identical 0.2.0 ↔ 0.3.0
# (pure lib/→root move).
#
# HELD at 0.2.x ON THIS PATH ONLY: 0.3.0 is a spin package (require_paths=".",
# no lib/) and `spinel-compat vendor` assumes lib/ — it vendors only sig/*.rbs
# (spinelgems issue filed). The spin path declares `~> 0.3` in spin.toml; the
# two resolve to identical sources for everything toy requires. Re-align to
# `~> 0.3` when spinel-compat honors require_paths.
gem "spinel_kit", "~> 0.2.0"
