#!/usr/bin/env ruby
# sync_tep.rb — keep `tep_demo/_tep_lib/` in sync with the upstream
# Tep sibling project. Tep is co-developed with this repo; we want to
# track HEAD continuously (no backwards-compat concerns).
#
# What this does:
#   1. rsync upstream Tep's lib/ → our tep_demo/_tep_lib/.
#   2. Substitute the @TEP_*@ placeholders that upstream's bin/tep
#      normally handles at build time. We point at upstream's
#      pre-built .o files in-place — no copying, so a Tep rebuild
#      doesn't need a re-sync.
#   3. Bail loud (non-zero exit + message) if anything we depend on
#      is missing — never silently degrade.
#
# Run:
#   ./prep/sync_tep.rb              # defaults to ~/sites/tep
#   TEP_SRC=/path ./prep/sync_tep.rb
#
# After sync, the Makefile's existing `spinel demo.rb -o demo` rules
# work as-is. We do NOT use upstream's `bin/tep build` because:
#   - It requires the Prism gem (not always installed on a deploy box).
#   - It also runs a Sinatra-style AST translator we don't need (our
#     demos use Tep::Handler classes directly).
#   - The placeholder substitution is the only thing we want from it.

require "fileutils"
require "pathname"

TEP_SRC = ENV.fetch("TEP_SRC", File.expand_path("~/sites/tep"))
TEP_DST = File.expand_path("../tep_demo/_tep_lib", __dir__)

def die(msg)
  $stderr.puts "[sync_tep] FAIL: #{msg}"
  exit 1
end

die "TEP_SRC=#{TEP_SRC} doesn't exist"      unless Dir.exist?(TEP_SRC)
die "TEP_SRC has no lib/tep.rb"             unless File.exist?("#{TEP_SRC}/lib/tep.rb")
die "TEP_SRC has no lib/tep/sphttp.o (run `make` in the tep checkout)" unless File.exist?("#{TEP_SRC}/lib/tep/sphttp.o")
die "TEP_SRC has no lib/tep/tep_sqlite.o"   unless File.exist?("#{TEP_SRC}/lib/tep/tep_sqlite.o")
die "TEP_SRC has no lib/tep/tep_pg.o"       unless File.exist?("#{TEP_SRC}/lib/tep/tep_pg.o")

# Tep's optional batteries link against system libpq / libsqlite3. We
# resolve their cflags + libs once and substitute. If a system lib is
# missing, bail loudly — users can rerun after `apt install libpq-dev`
# or equivalent, OR set TEP_PG_CFLAGS / TEP_SQLITE_CFLAGS by hand.
def pkgconfig(name, kind)
  flags = `pkg-config --#{kind} #{name} 2>/dev/null`.strip
  flags.empty? ? nil : flags
end

pg_cflags    = ENV["TEP_PG_CFLAGS"]    || pkgconfig("libpq", "cflags")
pg_libs      = ENV["TEP_PG_LIBS"]      || pkgconfig("libpq", "libs")
sqlite_cflags= ENV["TEP_SQLITE_CFLAGS"]|| pkgconfig("sqlite3", "cflags") || ""
sqlite_libs  = ENV["TEP_SQLITE_LIBS"]  || pkgconfig("sqlite3", "libs")   || "-lsqlite3"

if pg_cflags.nil? || pg_libs.nil?
  $stderr.puts "[sync_tep] libpq not found via pkg-config."
  $stderr.puts ""
  $stderr.puts "  Tep::PG is a sibling-tracked battery; any Tep app that loads"
  $stderr.puts "  the namespace (via require_relative 'tep' which pulls in"
  $stderr.puts "  tep/pg.rb) needs libpq's headers + library available at"
  $stderr.puts "  Spinel-compile time. Spinel rejects empty ffi_cflags."
  $stderr.puts ""
  $stderr.puts "  Two ways forward:"
  $stderr.puts ""
  $stderr.puts "    1. Install the PG dev package:"
  $stderr.puts "         apt install libpq-dev    # Debian/Ubuntu"
  $stderr.puts "         brew install libpq       # macOS"
  $stderr.puts "         dnf install libpq-devel  # Fedora/RHEL"
  $stderr.puts ""
  $stderr.puts "    2. If you genuinely don't want PG, override:"
  $stderr.puts "         TEP_PG_CFLAGS=-DNO_PG TEP_PG_LIBS=-lc ./prep/sync_tep.rb"
  $stderr.puts "       The placeholder gets a no-op cflags; Tep::PG will fail"
  $stderr.puts "       at runtime if any code actually calls into it."
  exit 1
end

# rsync — preserve everything, delete extras so we mirror upstream
# exactly. -av is enough; --delete keeps our copy from drifting.
FileUtils.mkdir_p(TEP_DST)
system("rsync", "-a", "--delete", "#{TEP_SRC}/lib/", "#{TEP_DST}/") or
  die "rsync failed"

# Substitute placeholders in-place. The upstream files contain
# strings like `ffi_cflags "@TEP_SPHTTP_O@"` that Spinel can't parse;
# after substitution they become absolute paths Spinel happily
# accepts.
subs = {
  "@TEP_SPHTTP_O@"  => "#{TEP_SRC}/lib/tep/sphttp.o",
  "@TEP_SQLITE_O@"  => "#{TEP_SRC}/lib/tep/tep_sqlite.o #{sqlite_libs}",
  "@TEP_PG_O@"      => "#{TEP_SRC}/lib/tep/tep_pg.o",
  "@TEP_PG_CFLAGS@" => "#{pg_cflags} #{pg_libs}".strip,
}

Dir.glob("#{TEP_DST}/**/*.rb").each do |f|
  src = File.read(f)
  out = src.dup
  subs.each { |k, v| out.gsub!(k, v) }
  File.write(f, out) if out != src
end

# Sanity check — make sure no @TEP_*@ placeholders survive.
leftovers = []
Dir.glob("#{TEP_DST}/**/*.rb").each do |f|
  File.foreach(f).with_index do |line, i|
    leftovers << "#{f}:#{i + 1}: #{line.strip}" if line.include?("@TEP_")
  end
end
unless leftovers.empty?
  $stderr.puts "[sync_tep] FAIL: unresolved placeholders after substitution:"
  leftovers.first(10).each { |l| $stderr.puts "  #{l}" }
  exit 1
end

# Report what we synced.
upstream_rev = `cd #{TEP_SRC} && git rev-parse --short HEAD 2>/dev/null`.strip
upstream_rev = "(no git)" if upstream_rev.empty?
n_files = Dir.glob("#{TEP_DST}/**/*.rb").size
puts "[sync_tep] OK — synced #{n_files} files from #{TEP_SRC} @ #{upstream_rev}"
puts "[sync_tep]      PG:     #{pg_cflags.empty? ? '(disabled — libpq missing)' : 'enabled'}"
puts "[sync_tep]      SQLite: #{sqlite_libs}"
