#!/usr/bin/env ruby
# frozen_string_literal: true
# prep/difflm_report.rb — the toy#166 (capstone P1b) GENERATION METRICS.
#
# Reads the byte streams a difflm run emitted (`gen.bytes`, `real.bytes`)
# and computes the distribution-match half of the metric:
#
#   * n-gram JS divergence (orders 2 and 3) between generated and real
#   * distinct-3gram ratio for both — the DEGENERACY guard
#
# WHY THIS IS NOT IN THE RUNNER. It was, first. A dense count table of
# 2500 cells reproducibly drove the Spinel-compiled process to 117 GB RSS
# and SIGKILL — a codegen problem, not a real memory need. Moving it out
# is better regardless: the metric is pure post-processing over a byte
# array (no model, no FFI, no session), so out here it is independently
# checkable, re-scorable without re-running a cell, and reviewable in a
# language where a JS divergence is four obvious lines. The runner keeps
# only what genuinely needs the judge MODEL, which is bits/byte.
#
# WHY BOTH METRICS AND NOT JUST NLL. Reference bits/byte alone REWARDS
# DEGENERACY — a model that emits "eeeeee" forever scores beautifully
# under any judge. The n-gram JS is the distribution guard, and the
# distinct-3gram ratio makes the failure legible when it happens.
#
#   ruby prep/difflm_report.rb <run-dir> [<run-dir> ...]
#
# Plain MRI (NOT Spinel-compiled) — same status as every other prep/
# analysis script.

def read_bytes(path)
  return nil unless File.file?(path)
  File.readlines(path).map { |l| l.to_i }
end

# Bytes are remapped to a compact id space so the count table is over the
# symbols that actually occur (~65 for shakespeare) and not over 256.
def counts(seq, order, map, m)
  h = Hash.new(0)
  tot = 0
  (0..(seq.length - order)).each do |i|
    idx = 0
    order.times { |q| idx = idx * m + map[seq[i + q]] }
    h[idx] += 1
    tot += 1
  end
  [h, tot]
end

def js_divergence(a, b, order)
  map = {}
  (a + b).each { |x| map[x] ||= map.size }
  m = map.size
  ha, ta = counts(a, order, map, m)
  hb, tb = counts(b, order, map, m)
  return nil if ta.zero? || tb.zero?
  js = 0.0
  (ha.keys | hb.keys).each do |k|
    pa = ha[k].to_f / ta
    pb = hb[k].to_f / tb
    mid = 0.5 * (pa + pb)
    js += 0.5 * pa * Math.log2(pa / mid) if pa > 0
    js += 0.5 * pb * Math.log2(pb / mid) if pb > 0
  end
  js
end

def distinct_ratio(seq, order)
  return nil if seq.length < order
  map = {}
  seq.each { |x| map[x] ||= map.size }
  h, tot = counts(seq, order, map, map.size)
  return nil if tot.zero?
  h.size.to_f / tot
end

dirs = ARGV
abort "difflm_report: pass one or more run dirs" if dirs.empty?

rows = []
dirs.each do |d|
  gen  = read_bytes(File.join(d, "gen.bytes"))
  real = read_bytes(File.join(d, "real.bytes"))
  unless gen && real
    warn "difflm_report: #{d} has no gen.bytes/real.bytes"
    next
  end
  rows << {
    dir: d,
    arm: File.basename(d),
    n: gen.length,
    js2: js_divergence(gen, real, 2),
    js3: js_divergence(gen, real, 3),
    d3_gen: distinct_ratio(gen, 3),
    d3_real: distinct_ratio(real, 3),
  }
end

fmt = ->(v) { v.nil? ? "   n/a" : format("%6.4f", v) }
puts format("%-20s %7s %7s %7s %9s %9s", "arm", "n", "js2", "js3", "distinct3", "(real)")
rows.each do |r|
  puts format("%-20s %7d %7s %7s %9s %9s", r[:arm], r[:n], fmt.(r[:js2]),
              fmt.(r[:js3]), fmt.(r[:d3_gen]), fmt.(r[:d3_real]))
end
