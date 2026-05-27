# Draft Gemfile for the spinelgems adoption — blocked on tep#95.
# Rename to "Gemfile" once tep.gemspec moves prism to a dev-dep and
# drops required_ruby_version to ">= 3.2.0". See
# docs/roadmap/spinelgems-tep-adoption-2026-05-27.md for the plan.

source "https://rubygems.org"
ruby "3.2.3", engine: "spinel", engine_version: "0.0.0"

# Sibling. path: for the gx10 dev loop; switch to git: once we want
# reproducible locks across machines (post-tep#95).
gem "tep", path: "../tep"
