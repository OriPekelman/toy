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

# SpinelKit — the shared Spinel stdlib-surface gem (toy#44): JSON
# builder/encoder/decoder + .git/HEAD provenance, consolidated out of toy's
# (and tep's) hand-rolled Toy::Json / Toy::Git shims. Pure Ruby, spinel-ext.json
# is [] (no native ext), so it vendors as plain lib/ copies under
# vendor/spinel/spinel_kit/. Consumed as the RELEASED RubyGems gem (like tep);
# `~> 0.1` allows 0.1.x patches, holds the minor.
gem "spinel_kit", "~> 0.1"
