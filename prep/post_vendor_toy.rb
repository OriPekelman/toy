#!/usr/bin/env ruby
# post_vendor_toy.rb — downstream-consumer hook that rewrites the
# vendored tinynn{,_cuda,_metal}.rb's ffi_cflags from toy's relative
# `-L.` / `-Ltinynn` / `-Lvendor/...` paths into absolute paths
# anchored at TOY_SRC.
#
# This file is shipped *as a template* — consumers (e.g.
# `tao_transfer`) copy it into their own `prep/` and adjust the
# VENDORED constant. The substitution logic itself is generic.
#
# Symmetric to prep/post_vendor_tep.rb. Same shape, different
# manifest. Once the spinelgems `vendor` grows a generic "C-ext
# post-vendor hook," this becomes a config file rather than a
# Ruby script — see OriPekelman/spinelgems#3 for the trajectory.
#
# Run from a consumer project (e.g. tao_transfer):
#   ./prep/post_vendor_toy.rb                   # default TOY_SRC=~/sites/toy
#   TOY_SRC=/path ./prep/post_vendor_toy.rb
#
# Expectations:
#   - vendor/spinel/toy/lib/ exists (i.e. spinel-compat vendor ran)
#   - TOY_SRC/tinynn/libtinynn_ggml.a (and the per-backend ggml .a
#     archives) exist — i.e. `make tinynn/libtinynn_ggml.a` (or
#     `make setup-ggml`) was run in TOY_SRC

require "fileutils"

TOY_SRC  = ENV.fetch("TOY_SRC", File.expand_path("~/sites/toy_ruby_neural_network"))
VENDORED = File.expand_path("../vendor/spinel/toy/lib", __dir__)

def die(msg)
  $stderr.puts "[post_vendor_toy] FAIL: #{msg}"
  exit 1
end

die "TOY_SRC=#{TOY_SRC} doesn't exist" unless Dir.exist?(TOY_SRC)
die "vendor/spinel/toy/lib not found — run `spinel-compat vendor` first" unless Dir.exist?(VENDORED)

manifest_path = File.join(VENDORED, "toy", "ffi_manifest.rb")
die "no lib/toy/ffi_manifest.rb in vendored tree — vendored toy predates toy#19" unless File.exist?(manifest_path)
load(manifest_path)
die "ffi_manifest.rb didn't define Toy::FFIManifest" unless defined?(Toy::FFIManifest)

# Verify the prebuilt static archive exists for each backend the
# consumer wants. We only check CPU by default — CUDA/Metal need
# their own setup-* recipes upstream; the consumer can disable a
# backend by setting TOY_DISABLE=cuda,metal etc.
disabled = (ENV["TOY_DISABLE"] || "").split(",").map(&:strip).reject(&:empty?)
die "TOY_SRC/tinynn/libtinynn_ggml.a not found (run `make tinynn/libtinynn_ggml.a` in TOY_SRC)" \
  unless File.exist?(File.join(TOY_SRC, "tinynn/libtinynn_ggml.a"))

touched = []
Toy::FFIManifest::BACKENDS.each do |backend_key, cfg|
  next if disabled.include?(backend_key.to_s)
  target = File.join(VENDORED, cfg[:ffi_file])
  next unless File.exist?(target)   # not every backend's mirror is generated yet

  current = Toy::FFIManifest::CURRENT_FFI_CFLAGS.fetch(backend_key)
  absolute = Toy::FFIManifest.absolute_cflags(backend_key, TOY_SRC)
  src = File.read(target)
  out = src.gsub("ffi_cflags \"#{current}\"", "ffi_cflags \"#{absolute}\"")
  if out == src
    $stderr.puts "[post_vendor_toy] WARN: no ffi_cflags match in #{cfg[:ffi_file]} — manifest's CURRENT_FFI_CFLAGS may be stale"
  else
    File.write(target, out)
    touched << cfg[:ffi_file]
  end
end

puts "[post_vendor_toy] rewrote ffi_cflags in #{touched.length} backend file(s):"
touched.each { |f| puts "  #{f}" }
puts "  TOY_SRC=#{TOY_SRC}"
puts "  TOY_DISABLE=#{disabled.join(',')}" unless disabled.empty?
