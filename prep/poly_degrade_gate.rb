#!/usr/bin/env ruby
# prep/poly_degrade_gate.rb — guard against silent Spinel poly-degradation in the
# numerical paths (issue #32).
#
# THE FAILURE MODE. Spinel does whole-program type inference and, when it can't
# resolve a call's receiver to a concrete type, it emits a `cannot resolve call
# to X on poly … (emitting 0)` warning and compiles a literal 0 in place. The
# binary builds clean and exits 0, but the numerical result is silently wrong —
# `compiled != correct`. We hit this TWICE during the events/AdamW cleanup (an
# unused require that name-collided a hot param; a missing require that emit-0'd
# AdamW). Both would have been caught here.
#
# WHAT THIS GATE DOES. Compiles each canonical compute entrypoint with `spinel`,
# scans the build warnings for the `(emitting 0)` resolve signature, and compares
# the set to a frozen baseline (prep/fixtures/poly_degrade_baseline.txt). The
# baseline is the set of KNOWN-BENIGN emit-0s (dead pure-Ruby teaching paths:
# Toy_Embedding#lookup, TransformerLM#embed_backward/#cross_entropy_grad; the
# nil-sampler accessors in ToyLM#generate). A NEW warning = a regression: a
# refactor just silently emit-0'd a path. Fail loud.
#
#   ruby prep/poly_degrade_gate.rb            # compare to baseline (CI)
#   ruby prep/poly_degrade_gate.rb --record   # (re)record the baseline
#
# Self-contained: uses the project's own `spinel` (the same compile the runners
# already do), no external spinel-dev tooling required. The analyze/codegen pass
# that emits these warnings runs before linking, so the scan is what matters.

require "open3"
require "fileutils"

ROOT     = File.expand_path("..", __dir__)
BASELINE = File.join(ROOT, "prep", "fixtures", "poly_degrade_baseline.txt")
RECORD   = ARGV.include?("--record")

# Resolve the spinel binary the same way the Makefile does.
SPINEL = ENV["SPINEL_BIN"] ||
         File.join(ENV["SPINEL_DIR"] || File.join(Dir.home, "sites", "spinel"), "spinel")

# Canonical numerical compute entrypoints (the hand-written CPU runners — the
# paths where a silent emit-0 corrupts training/eval/inference output). The
# CUDA/Metal twins share the same Ruby methods, so the CPU compile covers them.
ENTRYPOINTS = [
  "lib/toy/run/train.rb",
  "lib/toy/run/train_gpt2.rb",
  "lib/toy/run/infer.rb",
  "lib/toy/run/eval.rb",
  "lib/toy/run/eval_lmc.rb",
]

unless File.executable?(SPINEL)
  warn "poly-degrade-gate: spinel not found at #{SPINEL} (set SPINEL_BIN/SPINEL_DIR)"
  exit 2
end

# Compile one entrypoint and return the sorted-unique set of emit-0 resolve
# warnings, each prefixed with the entrypoint so a regression names its file.
def scan(entry)
  out_bin = "/tmp/poly_degrade_#{File.basename(entry, '.rb')}_#{Process.pid}"
  stdout_err, _status = Open3.capture2e(SPINEL, entry, "-o", out_bin, chdir: ROOT)
  FileUtils.rm_f(out_bin)
  found = []
  stdout_err.each_line do |line|
    # in <Class>#<method>: cannot resolve call to '<x>' on <type> … (emitting 0)
    m = line.match(/in (\S+#\S+): cannot resolve call to ('[^']+' on \w+).*\(emitting 0\)/)
    found << "#{entry} :: #{m[1]}: #{m[2]} (emitting 0)" if m
  end
  found.uniq
end

current = []
ENTRYPOINTS.each do |e|
  unless File.file?(File.join(ROOT, e))
    warn "poly-degrade-gate: missing entrypoint #{e}"
    exit 2
  end
  $stderr.print "  scanning #{e} … "
  s = scan(e)
  $stderr.puts "#{s.length} emit-0 warning(s)"
  current.concat(s)
end
current = current.sort.uniq

if RECORD
  File.write(BASELINE, current.join("\n") + "\n")
  puts "RECORDED [poly-degrade]: #{BASELINE} (#{current.length} known-benign emit-0 warnings)"
  exit 0
end

unless File.file?(BASELINE)
  warn "poly-degrade-gate: no baseline at #{BASELINE} — run with --record first"
  exit 2
end
baseline = File.read(BASELINE).split("\n").reject(&:empty?).sort.uniq

added   = current - baseline
removed = baseline - current

if added.empty? && removed.empty?
  puts "GATE PASS [poly-degrade]: no new silent emit-0 in the numerical paths " \
       "(#{baseline.length} known-benign, unchanged)."
  exit 0
end

if !added.empty?
  puts "GATE FAIL [poly-degrade]: NEW silent emit-0 warning(s) — a numerical path " \
       "just degraded to poly and compiles a literal 0 (compiled != correct):"
  added.each { |a| puts "  + #{a}" }
  puts "  Fix the type degradation (a missing/colliding require, an unconstrained"
  puts "  param — see feedback_spinel_type_inference_landmines), or if genuinely"
  puts "  benign, re-record with: ruby prep/poly_degrade_gate.rb --record"
end
if !removed.empty?
  puts "NOTE [poly-degrade]: #{removed.length} baseline warning(s) GONE (improvement)."
  puts "  Re-record to tighten the baseline: ruby prep/poly_degrade_gate.rb --record"
  removed.each { |r| puts "  - #{r}" }
end
# Additions are regressions → fail. Removals alone (improvements) → also fail so
# the baseline gets tightened, matching the byte-exact gate philosophy.
exit 1
