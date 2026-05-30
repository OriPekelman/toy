source "https://rubygems.org"
ruby "3.2.3", engine: "spinel", engine_version: "0.0.0"

# Tep is consumed as a gem from its `main` branch via the spinelgems
# convention (bundler-spinel). No hand-rolled vendoring or @TEP_*@
# substitution: `bundle lock` resolves tep, `spinel-compat vendor`
# copies tep's lib/ into vendor/spinel/tep/ AND natively compiles+wires
# its C extensions (driven by tep's shipped spinel-ext.json), then a
# Spinel entrypoint does `require_relative "vendor/spinel/deps"`.
#
# `git:` (not path:) for reproducible cross-machine locks. tep#95
# (prism→dev-dep, ruby>=3.2.0) and tep#98 (spinel-ext.json supersedes
# the ffi_manifest #97 design) are landed on tep main, so this resolves
# + vendors cleanly with zero toy-side tricks.
gem "tep", git: "https://github.com/OriPekelman/tep.git", branch: "main"
