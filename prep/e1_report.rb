#!/usr/bin/env ruby
# prep/e1_report.rb — toy#172/E1: read the B-conditioning instrument off
# the rank ladder.
#
# THE VERDICT IS READ ACROSS ARMS, NOT JUST ACROSS RUNGS. If DFA's
# feedback conditioning were what C_E measures, the dfa arm would separate
# from bp and frozen at the same width. If all three agree, C_E's
# conditioning is a property of the TASK at that width and the instrument
# — whatever it shows — is not measuring the credit rule.
#
# Every number is printed beside its sample count and ceiling verdict.
# rank(C_E) <= n, so a reading near n is capped rather than measured, and
# E1's `rank` branch must never be read off a capped cell.
require "json"

DIR   = ENV["DIR"] || "/srv/data/scratch/e1"
RUNGS = (ENV["RUNGS"] || "ae_shak_a65 ae_shak_a192 ae_shak_a380 ae_shak_a508 ae_shak_a1008 ae_shak_a2504").split
ARMS  = %w[dfa-layer bp-body frozen-body]

def mean(a) = a.sum / a.size.to_f
def sd(a)
  return 0.0 if a.size < 2
  m = mean(a)
  Math.sqrt(a.sum { |x| (x - m)**2 } / (a.size - 1).to_f)
end

rows = {}
RUNGS.each do |r|
  ARMS.each do |arm|
    Dir[File.join(DIR, "#{arm}_#{r}_v*_n*_s*.txt")].sort.each do |f|
      body = File.read(f)
      bc = body.lines.find { |l| l.start_with?("bcond: ") }
      next unless bc
      h = {}
      bc.scan(/(\w+)=([0-9.eE+-]+|[A-Za-z-]+)/) { |k, v| h[k] = v }
      alpha = body[/alphabet=(\d+)/, 1].to_i
      (rows[[r, arm]] ||= []) << {
        rank: alpha, n: h["n"].to_i, v: h["v"].to_i,
        sr: h["stable_rank"].to_f, pr: h["participation_ratio"].to_f,
        lm: h["lambda_max"].to_f, tr: h["trace"].to_f, capped: h["capped"],
        bpb: body[/bpb=([0-9.]+)/, 1].to_f
      }
    end
  end
end

abort "e1_report: no cells under #{DIR}" if rows.empty?

puts "E1 — B-conditioning instrument across the P6 rank ladder"
puts "dir: #{DIR}"
puts
puts "  rank    arm            n     v      stable_rank        participation_ratio   capped"
RUNGS.each do |r|
  ARMS.each do |arm|
    c = rows[[r, arm]]
    next unless c
    caps = c.map { |x| x[:capped] }.uniq.join("/")
    puts "  %5d   %-13s %5d %5d   %7.2f +- %-6.2f   %8.2f +- %-7.2f  %s" %
         [c.first[:rank], arm, c.first[:n], c.first[:v],
          mean(c.map { |x| x[:sr] }), sd(c.map { |x| x[:sr] }),
          mean(c.map { |x| x[:pr] }), sd(c.map { |x| x[:pr] }), caps]
  end
end

puts
puts "== DOES THE INSTRUMENT SEPARATE THE ARMS? =="
puts "   (if dfa does not separate from bp/frozen, C_E's conditioning is a"
puts "    property of the task at that width, NOT of the credit rule)"
RUNGS.each do |r|
  d = rows[[r, "dfa-layer"]]
  b = rows[[r, "bp-body"]]
  f = rows[[r, "frozen-body"]]
  next unless d && b && f
  ds = mean(d.map { |x| x[:sr] })
  bs = mean(b.map { |x| x[:sr] })
  fs = mean(f.map { |x| x[:sr] })
  spread = [ds, bs, fs].max - [ds, bs, fs].min
  # Against the within-arm seed noise, so "same" means indistinguishable
  # rather than merely close.
  noise = [sd(d.map { |x| x[:sr] }), sd(b.map { |x| x[:sr] }), sd(f.map { |x| x[:sr] })].max
  verdict = spread <= 2.0 * (noise <= 0 ? 1e-9 : noise) ? "INDISTINGUISHABLE" : "separated"
  puts "  rank %5d: dfa %.2f  bp %.2f  frozen %.2f   spread %.3f vs seed-noise %.3f -> %s" %
       [d.first[:rank], ds, bs, fs, spread, noise, verdict]
end

puts
puts "== THE DFA ARM vs WIDTH (E1's own question) =="
prev = nil
RUNGS.each do |r|
  d = rows[[r, "dfa-layer"]]
  next unless d
  sr = mean(d.map { |x| x[:sr] })
  v  = d.first[:rank]
  frac = sr / v.to_f
  delta = prev ? " (%+.2f vs previous rung)" % (sr - prev) : ""
  cap = d.map { |x| x[:capped] }.uniq.join("/")
  puts "  rank %5d: stable_rank %7.2f  = %.3f of V  n=%d [%s]%s" %
       [v, sr, frac, d.first[:n], cap, delta]
  prev = sr
end

puts
puts "== READING THE VERDICT =="
puts "  `rank`       — effective rank COLLAPSES with width AND is not sample-capped,"
puts "                 and the dfa arm separates from the controls."
puts "  `anisotropy` — effective rank is far below V but does not collapse; the"
puts "                 covariance is ill-conditioned rather than rank-deficient."
puts "  `neither`    — the arms are indistinguishable, i.e. C_E is a task property"
puts "                 and this instrument does not see the credit rule at all."
puts
puts "  NOTE: a rung flagged `below-V` is measured at n < V by design (matched-n"
puts "  across rungs keeps the ceiling CONSTANT rather than letting it move with"
puts "  width). It is only unreadable if the statistic approaches n, which is what"
puts "  `SAMPLE-CAPPED` marks."
