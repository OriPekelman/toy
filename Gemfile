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
# the gem root, so it resolves + vendors cleanly. `~> 0.11.2` = the current
# release (0.11.2) — allows 0.11.x patches, holds the minor.
gem "tep", "~> 0.11.2"
