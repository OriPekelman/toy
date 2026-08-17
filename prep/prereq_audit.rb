#!/usr/bin/env ruby
# prep/prereq_audit.rb — toy#171: do the libexec/* Makefile prerequisite
# lists still match what the runners actually require?
#
# WHY THIS EXISTS. The prerequisite lists are hand-maintained, and they
# duplicate information that already lives in each runner's
# `require_relative` graph. Adding a require and forgetting the Makefile
# line is a one-line omission with NO feedback: nothing fails, `make`
# reports "is up to date", and the stale binary is then exercised by its
# own gate — which passes, while testing the previous code.
#
# That is not hypothetical. It was hit twice in two days on toy#170:
# toy-train-gtx and toy-train-ssm both required lib/toy/io/toy_ae_task.rb
# without declaring it, so a change to that very file rebuilt neither.
# The systemic audit then found it across most of the tree, including
# toy-train-franken missing lib/toy/train/dfa_b.rb — the DFA program's
# own dense lane not rebuilding when the feedback matrix changes.
#
# LOWER BOUND, NOT AN ORACLE. This resolves `require_relative` textually.
# It does not follow plain `require`, dynamic requires, or anything
# computed at runtime, so a clean report means "no drift THIS FINDS".
#
# Exit 0 = no drift. Exit 1 = drift, listed. `--fix` rewrites the
# Makefile in place, which is how the initial 26-target backlog was
# cleared; day to day the gate should simply fail and a human adds the
# line next to the require they just wrote.
require "set"

ROOT = File.expand_path("..", __dir__)
MAKEFILE = File.join(ROOT, "Makefile")
FIX = ARGV.include?("--fix")

src = File.read(MAKEFILE)

# A target header runs from `libexec/name:` to the first line not ending
# in a backslash. Captured whole so --fix can rewrite it as a unit.
# The separator after the colon is CAPTURED, not skipped: --fix rebuilds
# the header by concatenation, so dropping the original whitespace makes
# the substitution miss and the fix abort (which it did, safely).
HEADER = %r{^(libexec/[a-z0-9._-]+):([ \t]*)((?:[^\n]*\\\n)*[^\n]*\n)}

def closure(path, seen = {})
  return seen if seen[path] || !File.file?(path)
  seen[path] = true
  File.read(path).scan(/^\s*require_relative\s+["']([^"']+)["']/) do |(rel)|
    p2 = File.expand_path(rel, File.dirname(path))
    p2 += ".rb" unless p2.end_with?(".rb")
    closure(p2, seen)
  end
  seen
end

drift = []
examined = 0
src.scan(HEADER) do |name, sep, body|
  deps = body.gsub("\\\n", " ").split(/\s+/).reject(&:empty?)
  declared = deps.take_while { |d| d != "|" }
  runner = declared.find { |d| d.start_with?("lib/toy/run/") && d.end_with?(".rb") }
  next unless runner && File.file?(File.join(ROOT, runner))
  examined += 1
  need = closure(File.join(ROOT, runner)).keys
           .map { |p| p.sub(ROOT + "/", "") }
           .select { |p| p.start_with?("lib/") } - [runner]
  missing = need.reject { |p| declared.include?(p) }.sort
  drift << [name, runner, missing, sep, body] unless missing.empty?
end

# A checker that silently stops checking is worse than no checker. Adding
# a capture group to HEADER once shifted the block parameters, so every
# target was skipped and this reported a clean PASS while examining
# nothing. The floor is asserted so that failure can never be quiet.
MIN_EXAMINED = 20
if examined < MIN_EXAMINED
  abort "prereq_audit: only #{examined} target(s) had a resolvable runner " \
        "(expected >= #{MIN_EXAMINED}). The Makefile layout or HEADER regex " \
        "changed and this audit is no longer looking at anything — a PASS " \
        "here would be meaningless."
end

if drift.empty?
  puts "GATE PASS [prereq]: every libexec/* target declares the full require_relative closure of its runner (#{src.scan(HEADER).size} targets examined)"
  exit 0
end

if FIX
  out = src.dup
  drift.each do |name, runner, missing, sep, body|
    # APPEND, never prepend. The recipe is `$(SPINEL) $< -o $@`, and `$<`
    # is the FIRST prerequisite — so putting the missing files at the
    # front silently repoints the build at lib/toy.rb instead of the
    # runner. That is not a broken build that announces itself; it is a
    # binary compiled from the wrong source. An earlier version of this
    # script did exactly that, and it is why the check below exists.
    #
    # Insertion goes before the order-only `|` if there is one, so the
    # `| libexec` tail keeps its meaning.
    old = "#{name}:#{sep}#{body}"
    add = missing.join(" ")
    fixed = if body.include?(" | ")
              "#{name}:#{sep}#{body.sub(/ \| /, " \\\n\t\t#{add} | ")}"
            else
              "#{name}:#{sep}#{body.sub(/\n\z/, " \\\n\t\t#{add}\n")}"
            end
    out.sub!(old, fixed) or abort "prereq_audit: could not rewrite #{name}"
  end

  # Self-check BEFORE writing: every rewritten target must still name its
  # runner as the first prerequisite, or $< is wrong and the build would
  # compile something else entirely.
  out.scan(HEADER) do |nm, _s, bd|
    deps = bd.gsub("\\\n", " ").split(/\s+/).reject(&:empty?)
    first_dep = deps.first
    next unless first_dep&.start_with?("lib/")
    if !first_dep.start_with?("lib/toy/run/")
      abort "prereq_audit: refusing to write — #{nm} would take `#{first_dep}` as its " \
            "FIRST prerequisite, so $< no longer names the runner and the target would " \
            "compile the wrong source. Makefile left untouched."
    end
  end

  File.write(MAKEFILE, out)
  puts "fixed #{drift.size} target(s); re-run without --fix to verify"
  exit 0
end

puts "GATE FAIL [prereq]: #{drift.size} target(s) do not declare files their runner requires."
puts "A change to an undeclared file rebuilds NOTHING — `make` says 'up to date' and the"
puts "gate then passes against a stale binary. Add the file next to the require, or run:"
puts "  ruby prep/prereq_audit.rb --fix"
puts
drift.each do |name, runner, missing, _sep, _body|
  puts "#{name}   (#{runner})"
  missing.each { |m| puts "    MISSING  #{m}" }
end
exit 1
