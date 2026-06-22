# Toy gem version. Lifted out of lib/toy.rb so the gemspec can read
# it without dragging in `require_relative "tinynn"` (tinynn uses
# Spinel-only `ffi_lib` / `ffi_cflags` directives that don't load
# under CRuby).
module Toy
  # Single source of truth: gemspec + `toy --version` + `toy --manifest`
  # all read this; README/CHANGELOG/git tag display it as v0.9.0.
  # v0.8.0 (2026-06-12) was the first PUBLISHED version (RubyGems);
  # v0.9.0 adds the Dragon / Gated-DeltaNet trainable hybrid arc.
  # Pre-1.0: not API-stable.
  VERSION = "0.9.0".freeze
end
