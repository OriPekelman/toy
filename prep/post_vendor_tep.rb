#!/usr/bin/env ruby
# post_vendor_tep.rb — thin post-vendor step for the spinelgems flow.
#
# `spinel-compat vendor` places tep's lib/ into vendor/spinel/tep/lib/
# verbatim, including the `@TEP_*@` placeholders that upstream's
# bin/tep normally rewrites at build time. This script handles that
# rewrite — and ONLY that rewrite — pointing at tep's pre-built .o
# files in-place (no copy).
#
# This is the spinelgems-side counterpart of what prep/sync_tep.rb
# does (and the only remaining concern after rsync is replaced by
# `spinel-compat vendor`). See
# docs/roadmap/spinelgems-tep-adoption-2026-05-27.md and
# https://github.com/OriPekelman/spinelgems/blob/main/docs/adoption.md.
#
# Run:
#   uv run prep/preprocess_images.py   # (unrelated, just an example uv)
#   # — or just:
#   ./prep/post_vendor_tep.rb            # defaults to ~/sites/tep
#   TEP_SRC=/path ./prep/post_vendor_tep.rb
#
# Expectations:
#   - vendor/spinel/tep/lib/ exists (i.e. `spinel-compat vendor` ran)
#   - TEP_SRC/lib/tep/sphttp.o etc. exist (i.e. `make` was run in tep)
#
# Bails loud if anything's missing — never silently degrade.

require "fileutils"

TEP_SRC = ENV.fetch("TEP_SRC", File.expand_path("~/sites/tep"))
VENDORED = File.expand_path("../vendor/spinel/tep/lib", __dir__)

def die(msg)
  $stderr.puts "[post_vendor_tep] FAIL: #{msg}"
  exit 1
end

die "TEP_SRC=#{TEP_SRC} doesn't exist"           unless Dir.exist?(TEP_SRC)
die "vendor/spinel/tep/lib not found — run `spinel-compat vendor` first" unless Dir.exist?(VENDORED)
die "TEP_SRC has no lib/tep/sphttp.o (run `make` in the tep checkout)" unless File.exist?("#{TEP_SRC}/lib/tep/sphttp.o")
die "TEP_SRC has no lib/tep/tep_sqlite.o" unless File.exist?("#{TEP_SRC}/lib/tep/tep_sqlite.o")
die "TEP_SRC has no lib/tep/tep_pg.o"     unless File.exist?("#{TEP_SRC}/lib/tep/tep_pg.o")

# Resolve system libs for sqlite and postgres via pkg-config — same
# recipe as prep/sync_tep.rb (the version of truth for the
# placeholder values). macOS Homebrew workaround included.
if RUBY_PLATFORM.include?("darwin")
  extra_pc = []
  ["libpq", "sqlite"].each do |formula|
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
  out = `pkg-config --#{flavor} #{pkg} 2>/dev/null`.strip
  out.empty? ? nil : out
end

sqlite_cflags = ENV["TEP_SQLITE_CFLAGS"] || pkgconfig("sqlite3", "cflags") || ""
sqlite_libs   = ENV["TEP_SQLITE_LIBS"]   || pkgconfig("sqlite3", "libs")   || "-lsqlite3"
pg_cflags     = ENV["TEP_PG_CFLAGS"]     || pkgconfig("libpq", "cflags")
pg_libs       = ENV["TEP_PG_LIBS"]       || pkgconfig("libpq", "libs")

if pg_cflags.nil? || pg_libs.nil?
  $stderr.puts "[post_vendor_tep] libpq not found via pkg-config."
  $stderr.puts ""
  $stderr.puts "  Tep::PG needs libpq's headers + library at Spinel-compile time."
  $stderr.puts "  Spinel rejects empty ffi_cflags. Two ways forward:"
  $stderr.puts ""
  $stderr.puts "    1. Install the PG dev package:"
  $stderr.puts "         apt install libpq-dev    # Debian/Ubuntu"
  $stderr.puts "         brew install libpq       # macOS"
  $stderr.puts "         dnf install libpq-devel  # Fedora/RHEL"
  $stderr.puts ""
  $stderr.puts "    2. If you don't actually use Tep::PG, opt out at sub time:"
  $stderr.puts "         TEP_PG_CFLAGS=-DNO_PG TEP_PG_LIBS=-lc ./prep/post_vendor_tep.rb"
  $stderr.puts "       Tep::PG then fails at runtime if any code calls into it."
  exit 1
end

subs = {
  "@TEP_SPHTTP_O@"  => "#{TEP_SRC}/lib/tep/sphttp.o",
  "@TEP_SQLITE_O@"  => "#{TEP_SRC}/lib/tep/tep_sqlite.o #{sqlite_libs}",
  "@TEP_PG_O@"      => "#{TEP_SRC}/lib/tep/tep_pg.o",
  "@TEP_PG_CFLAGS@" => "#{pg_cflags} #{pg_libs}".strip,
}

touched = 0
Dir.glob("#{VENDORED}/**/*.rb").each do |f|
  src = File.read(f)
  out = src.dup
  subs.each { |k, v| out.gsub!(k, v) }
  if out != src
    File.write(f, out)
    touched += 1
  end
end

# Sanity check — make sure no @TEP_*@ placeholders survive.
leftovers = []
Dir.glob("#{VENDORED}/**/*.rb").each do |f|
  File.foreach(f).with_index do |line, i|
    leftovers << "#{f}:#{i + 1}: #{line.strip}" if line.include?("@TEP_")
  end
end
unless leftovers.empty?
  $stderr.puts "[post_vendor_tep] FAIL: unresolved placeholders after substitution:"
  leftovers.first(10).each { |l| $stderr.puts "  #{l}" }
  exit 1
end

puts "[post_vendor_tep] substituted #{touched} file(s) under #{VENDORED}"
puts "  TEP_SRC=#{TEP_SRC}"
