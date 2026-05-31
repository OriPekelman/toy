# Toy gem version. Lifted out of lib/toy.rb so the gemspec can read
# it without dragging in `require_relative "tinynn"` (tinynn uses
# Spinel-only `ffi_lib` / `ffi_cflags` directives that don't load
# under CRuby).
module Toy
  # Gem-canonical prerelease form of "v0.7.0-pre-alpha" (RubyGems renders
  # dots, not dashes). Single source of truth: gemspec + `toy --version`
  # + `toy --manifest` all read this; README/CHANGELOG display it as
  # v0.7.0-pre-alpha.
  VERSION = "0.7.0.pre.alpha".freeze
end
