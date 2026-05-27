#!/usr/bin/env ruby
# post_vendor_tep.rb — thin post-vendor step for the spinelgems flow.
#
# `spinel-compat vendor` places tep's lib/ into vendor/spinel/tep/lib/
# verbatim, including `@TEP_*@` placeholders that bin/tep normally
# rewrites at build time. This script does that rewrite.
#
# Reads `vendor/spinel/tep/lib/tep/ffi_manifest.rb`
# (`Tep::FFIManifest::ENTRIES`) as the single source of truth for
# tep's FFI shape — same manifest bin/tep reads on the tep side.
# See OriPekelman/tep#97 for the design. Manifest is BUILD-TIME
# CRuby (lib/tep.rb never requires it under Spinel).
#
# Resolution policy (consumer-side, env-first to honor caller wishes):
#   1. ENV[entry's obj_env / libs_env / cflags_env]      — overrides win
#   2. pkg-config <entry[:pkg_config]>                   — distro-managed libs
#   3. spec[:libs_default] OR entry[:pkg_config_fallback] — last resort
#
# Opt-outs (TEP_DISABLE=pg,sqlite,...): each optional entry's
# placeholders are substituted with `-DNO_<MOD> -lc`. The file still
# compiles (Spinel rejects empty ffi_cflags); calls into the disabled
# battery fail at runtime.
#
# Run:
#   ./prep/post_vendor_tep.rb                  # default TEP_SRC=~/sites/tep
#   TEP_SRC=/path ./prep/post_vendor_tep.rb
#   TEP_DISABLE=pg ./prep/post_vendor_tep.rb   # skip pg battery

require "fileutils"

TEP_SRC  = ENV.fetch("TEP_SRC", File.expand_path("~/sites/tep"))
VENDORED = File.expand_path("../vendor/spinel/tep/lib", __dir__)
TEP_LIB  = File.join(TEP_SRC, "lib")

def die(msg)
  $stderr.puts "[post_vendor_tep] FAIL: #{msg}"
  exit 1
end

die "TEP_SRC=#{TEP_SRC} doesn't exist" unless Dir.exist?(TEP_SRC)
die "vendor/spinel/tep/lib not found — run `spinel-compat vendor` first" unless Dir.exist?(VENDORED)

manifest_path = File.join(VENDORED, "tep", "ffi_manifest.rb")
die "no ffi_manifest.rb in vendored tree — vendored tep predates tep#97" unless File.exist?(manifest_path)

# CRuby `load`. The manifest is build-time-only — Spinel never sees it.
load(manifest_path)
die "ffi_manifest.rb didn't define Tep::FFIManifest::ENTRIES" unless defined?(Tep::FFIManifest::ENTRIES)

# macOS Homebrew keeps libpq / sqlite as keg-only; surface their pkg-configs.
if RUBY_PLATFORM.include?("darwin")
  extra_pc = []
  %w[libpq sqlite].each do |formula|
    prefix = `brew --prefix #{formula} 2>/dev/null`.strip
    next if prefix.empty?
    pc = "#{prefix}/lib/pkgconfig"
    extra_pc << pc if File.directory?(pc)
  end
  unless extra_pc.empty?
    ENV["PKG_CONFIG_PATH"] = (extra_pc + [ENV["PKG_CONFIG_PATH"].to_s]).reject(&:empty?).join(":")
  end
end

def pkgconfig(pkg, flavor)
  return nil if pkg.nil?
  out = `pkg-config --#{flavor} #{pkg} 2>/dev/null`.strip
  out.empty? ? nil : out
end

# Resolve one placeholder's value per its spec.
def resolve_placeholder(entry, spec)
  default_obj = File.join(TEP_LIB, entry[:obj])
  obj = spec[:obj_env] ? ENV.fetch(spec[:obj_env].to_s, default_obj) : default_obj
  pkg = entry[:pkg_config]

  case spec[:kind]
  when :obj
    File.exist?(obj) or die "missing #{obj} (run `make` in the tep checkout)"
    obj
  when :obj_plus_libs
    File.exist?(obj) or die "missing #{obj} (run `make` in the tep checkout)"
    libs = ENV[spec[:libs_env].to_s] ||
           pkgconfig(pkg, "libs") ||
           entry[:pkg_config_fallback] ||
           ""
    "#{obj} #{libs}".strip
  when :cflags_plus_libs
    cflags = ENV[spec[:cflags_env].to_s] ||
             pkgconfig(pkg, "cflags") ||
             ""
    libs   = ENV[spec[:libs_env].to_s] ||
             pkgconfig(pkg, "libs") ||
             spec[:libs_default] ||
             entry[:pkg_config_fallback]
    if libs.nil?
      die "#{entry[:module]}: pkg-config #{pkg} not found, no fallback. " \
          "Install the dev package OR set #{spec[:cflags_env]}/#{spec[:libs_env]} OR TEP_DISABLE=#{entry[:module]}."
    end
    "#{cflags} #{libs}".strip
  else
    die "unknown placeholder kind: #{spec[:kind].inspect}"
  end
end

# Disabled placebo. Non-empty literal so Spinel accepts it; -DNO_<MOD>
# makes it visible in the binary that this battery isn't really linked.
def disabled_value(mod_upper)
  "-DNO_#{mod_upper} -lc"
end

disabled = Tep::FFIManifest.respond_to?(:disabled) ? Tep::FFIManifest.disabled : []
touched_files  = []
disabled_files = []
processed_targets = []

Tep::FFIManifest::ENTRIES.each do |entry|
  target_file = File.join(VENDORED, entry[:file])
  die "manifest references #{entry[:file]} which doesn't exist in #{VENDORED}" unless File.exist?(target_file)
  processed_targets << File.expand_path(target_file)

  src = File.read(target_file)
  out = src.dup

  if entry[:optional] && disabled.include?(entry[:module])
    placebo = disabled_value(entry[:module].upcase)
    entry[:placeholders].each_key { |ph| out.gsub!(ph, placebo) }
    disabled_files << entry[:file]
  else
    entry[:placeholders].each do |placeholder, spec|
      out.gsub!(placeholder, resolve_placeholder(entry, spec))
    end
  end

  if out != src
    File.write(target_file, out)
    touched_files << entry[:file]
  end
end

# Sanity: every manifest-referenced target file has zero `@TEP_*@`
# placeholders left. We deliberately ONLY check the manifest's target
# files — the manifest itself contains `@TEP_*@` *as data* (the
# placeholder names are keys in the entries' :placeholders hashes),
# so a blanket grep across all .rb files would false-positive on
# ffi_manifest.rb.
leftovers = []
processed_targets.each do |f|
  File.foreach(f).with_index do |line, i|
    leftovers << "#{f}:#{i + 1}: #{line.strip}" if line.include?("@TEP_")
  end
end
unless leftovers.empty?
  $stderr.puts "[post_vendor_tep] FAIL: unresolved placeholders after substitution:"
  leftovers.first(10).each { |l| $stderr.puts "  #{l}" }
  exit 1
end

puts "[post_vendor_tep] substituted #{touched_files.length} file(s) under #{VENDORED}"
puts "  TEP_SRC=#{TEP_SRC}"
puts "  TEP_DISABLE=#{disabled.join(',')}"  unless disabled.empty?
puts "  disabled (placebo subs only): #{disabled_files.join(', ')}" unless disabled_files.empty?
