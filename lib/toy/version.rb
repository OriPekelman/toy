# Toy gem version. Lifted out of lib/toy.rb so the gemspec can read
# it without dragging in `require_relative "tinynn"` (tinynn uses
# Spinel-only `ffi_lib` / `ffi_cflags` directives that don't load
# under CRuby).
module Toy
  VERSION = "0.1.0".freeze
end
