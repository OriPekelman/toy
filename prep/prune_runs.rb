#!/usr/bin/env ruby
# prep/prune_runs.rb — keep runs/ small, repeatably.
#
# WHY THIS EXISTS. runs/ reached 15,842 bundles / 2.6 GB before the
# 2026-08-24 cleanup, and gate-run-log scans the WHOLE tree, so the battery
# paid for every one of them. A one-time manual prune does not hold: the
# battery itself writes bundles, measured at ~49 per `make gates-framework`
# run and more for a full sweep. Left alone, runs/ climbs back to five
# figures and the cost comes with it. So the prune is a target, not an act.
#
# WHAT IT KEEPS. The KEEP most recent bundles PER ARCH PREFIX that are
# actually usable — parses as toy/v1, non-empty, carries a final loss.
# Per-prefix rather than globally most-recent because examples/
# 06_runlog_compare.rb and gate-run-log want a spread of architectures, and
# a global cut would keep 24 bundles of whichever lane ran last.
#
# Validity is not cosmetic: run_log_gate asserts logs.ALL? parse as toy/v1,
# so ONE malformed bundle fails the leg. A killed run leaves a zero-byte
# events.jsonl and has broken that gate before — this is also how such a
# bundle gets swept out.
#
# DEFAULT IS A DRY RUN. Deleting is irreversible (runs/ is gitignored and
# untracked, so there is no `git checkout` back). Pass --apply to act, and
# --archive DIR to move rather than delete.
#
#   ruby prep/prune_runs.rb                      # report only
#   ruby prep/prune_runs.rb --apply              # delete all but KEEP/prefix
#   ruby prep/prune_runs.rb --apply --archive /srv/data/scratch/toy-runs
#   KEEP=5 ruby prep/prune_runs.rb --apply
require "json"
require "fileutils"

ROOT    = File.expand_path("../runs", __dir__)
KEEP    = (ENV["KEEP"] || "3").to_i
APPLY   = ARGV.include?("--apply")
ai      = ARGV.index("--archive")
ARCHIVE = ai ? ARGV[ai + 1] : nil

abort "prune_runs: no runs/ at #{ROOT}" unless Dir.exist?(ROOT)
abort "prune_runs: --archive needs a directory" if ai && (ARCHIVE.nil? || ARCHIVE.start_with?("--"))
abort "prune_runs: KEEP must be >= 1 (got #{KEEP})" if KEEP < 1

# A bundle is keepable only if a reader could actually use it.
def usable?(dir)
  f = File.join(dir, "events.jsonl")
  return false unless File.file?(f) && File.size(f) > 0
  return false unless (JSON.parse(File.open(f, &:readline))["schema"] rescue nil) == "toy/v1"
  # ...and it must have got far enough to have a loss. `run_end` or any
  # step line carrying one; an un-stepped run has nothing to compare.
  File.foreach(f).any? do |l|
    o = (JSON.parse(l) rescue nil)
    o && (o["kind"] == "run_end" || o["loss"].is_a?(Float))
  end
rescue StandardError
  false
end

all = Dir.children(ROOT).select { |d| File.directory?(File.join(ROOT, d)) }
groups = all.group_by { |d| d.sub(/-\d{8}-\d+\z/, "").sub(/-\d+\z/, "") }

keep, drop, unusable = [], [], []
per_prefix = {}   # prefix -> kept count, recorded HERE rather than re-derived
groups.each do |prefix, ds|
  ok = ds.sort.reverse.select { |d| usable?(File.join(ROOT, d)) }
  unusable.concat(ds - ok)
  kept = ok.first(KEEP)
  # Recorded from the grouping itself, NOT re-counted later by matching the
  # prefix against bundle names. `keep.count { |d| d.start_with?(p) }` reads
  # like the same thing and is not: "diff" prefixes "difflm" and "moe"
  # prefixes "moe-cuda-gate", so that form reported `diff 1 -> 2` (keeping
  # more bundles than the group holds) and `moe 157 -> 4` (more than KEEP).
  # The selection was always right; only the report lied, which is the worse
  # failure of the two — nobody re-checks a number that looks plausible.
  per_prefix[prefix] = kept.size
  keep.concat(kept)
  drop.concat(ok.drop(KEEP))
end
drop.concat(unusable)

puts "runs/ at #{ROOT}"
puts "  bundles      : #{all.size}"
puts "  keeping      : #{keep.size}  (#{KEEP}/prefix across #{groups.size} prefixes)"
puts "  dropping     : #{drop.size}  (of which #{unusable.size} unusable: empty, unparseable, or never stepped)"
groups.keys.sort.each do |p|
  puts format("    %-16s %5d -> %d", p, groups[p].size, per_prefix[p])
end

if keep.empty? && !all.empty?
  abort "prune_runs: REFUSING — every bundle looks unusable, which is far more " \
        "likely a bug in this script than a real state. Nothing was touched."
end

unless APPLY
  puts "\n  DRY RUN. Nothing changed. Re-run with --apply to act" \
       "#{ARCHIVE ? " (moving to #{ARCHIVE})" : " (deleting)"}."
  exit 0
end

if ARCHIVE
  FileUtils.mkdir_p(ARCHIVE)
  drop.each { |d| FileUtils.mv(File.join(ROOT, d), File.join(ARCHIVE, d), force: true) }
  puts "\n  moved #{drop.size} bundles to #{ARCHIVE}"
else
  drop.each { |d| FileUtils.rm_rf(File.join(ROOT, d)) }
  puts "\n  deleted #{drop.size} bundles"
end
puts "  runs/ now: #{Dir.children(ROOT).size} bundles"
