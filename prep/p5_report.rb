#!/usr/bin/env ruby
# prep/p5_report.rb — read the P5 cells and report the alphabet axis.
#
# PRIMARY STATISTIC = ABSOLUTE BITS RECOVERED (frozen - dfa).
#
# The recovery FRACTION (frozen-dfa)/(frozen-bp) was the headline through
# P3/P4 and it could not separate alphabet 27 from 65 even at n=8
# (Welch t=1.86, Mann-Whitney U=15). Its denominator is only 0.3-0.9 bits
# and is itself noisy, so dividing by it costs more resolution than the
# difficulty-normalisation buys. The absolute scale separated the same
# cells at t=4.50. Both are printed; the absolute one is the claim.
#
# The fraction is still reported because it does control for task
# difficulty and the absolute scale does not — on the INFLATION arms
# difficulty is held fixed by construction, so there the two should
# agree, and a disagreement is a red flag about the control itself.
require "json"

ARMS = ["bp-body", "frozen-body", "dfa-layer", "dfa-step"]

# MODE=alphabet — the remap sweep: effective rank moves, B pinned at 256.
# MODE=width    — the head-width sweep: B moves, effective rank pinned.
# The two together are what decompose the output-dim law; neither does it
# alone, which is why one script reads both and prints the same statistic.
MODE = ENV["MODE"] || "alphabet"
DIRS = if !ARGV.empty?
         ARGV
       elsif MODE == "width"
         ["/srv/data/scratch/p5hw"]
       else
         ["/srv/data/scratch/p5", "/srv/data/scratch/p4"]
       end
# The x-axis value is read from each run's OWN provenance line, never
# from the filename and never assumed.
SERIES = MODE == "width" ?
  ["v65", "v128", "v256", "v512", "v1024"] :
  # ae_shak_a65 and ae_shakespeare are the SAME TASK on the same text,
  # differing only in whether the ids are dense — so they are the control
  # for the remap pipeline itself. If they disagree, the repacking moved
  # the result and none of the other three points mean what they claim.
  ["ae_shak_a27", "ae_shak_a65", "ae_shakespeare", "ae_shak_a129", "ae_shak_a192"]

def cells_for(arm, key)
  out = {}
  DIRS.each do |d|
    next unless File.directory?(d)
    Dir[File.join(d, "#{arm}_#{key}_lr*_s*.txt")].each do |f|
      m = File.basename(f).match(/_lr([0-9.e-]+)_s(\d+)\.txt\z/)
      next unless m
      body = File.read(f)
      bpb = body[/bpb=([0-9.]+)/, 1]
      next unless bpb
      x = MODE == "width" ? body[/ b_dim=(\d+)/, 1] : body[/alphabet=(\d+)/, 1]
      # A cell that is not the arm its name claims is not a data point.
      bad = []
      if MODE == "width"
        w = key.sub("v", "")
        # BOTH, separately: a head that narrows while B stays 256 wide
        # measures the opposite axis and still produces plausible bpb.
        bad << "head width" unless body.include?("vocab=#{w} ")
        bad << "B width" unless body.include?("b_dim=#{w} ")
      else
        bad << "corpus" unless body.include?("pack=data/#{key} ")
      end
      bad << "frozen" if arm == "frozen-body" && !body.include?("frozen=4 ")
      bad << "dfa_wired" if arm.start_with?("dfa-") && !body.include?("dfa_wired=4 ")
      out[[m[1], m[2].to_i]] = { bpb: bpb.to_f, alpha: x&.to_i, bad: bad, file: f }
    end
  end
  out
end

def mean(a) = a.sum / a.size.to_f
def sd(a)
  return 0.0 if a.size < 2
  m = mean(a)
  Math.sqrt(a.sum { |x| (x - m)**2 } / (a.size - 1).to_f)
end
def median(a)
  s = a.sort
  s.size.odd? ? s[s.size / 2] : (s[s.size / 2 - 1] + s[s.size / 2]) / 2.0
end
# Welch's t. Reported as a magnitude only — with n=5-8 per group the df
# correction and any p-value attached to it are decorative, and this arc
# has already been burned once by over-reading a 3-seed number.
def welch_t(a, b)
  va = sd(a)**2 / a.size
  vb = sd(b)**2 / b.size
  return 0.0 if va + vb <= 0
  (mean(a) - mean(b)) / Math.sqrt(va + vb)
end

report = {}
SERIES.each do |corpus|
  arms = {}
  ok = true
  ARMS.each do |arm|
    cells = cells_for(arm, corpus)
    if cells.empty?
      ok = false
      break
    end
    lrs = cells.keys.map(&:first).uniq
    # Best LR = argmin over SEED 0 only. Picking it per-seed would be
    # selecting the minimum of several noisy draws and would bias every
    # arm downward by a different amount.
    s0 = lrs.map { |lr| [lr, cells[[lr, 0]]] }.reject { |_, c| c.nil? }
    next_best = s0.min_by { |_, c| c[:bpb] }
    unless next_best
      ok = false
      break
    end
    best_lr = next_best[0]
    seeds = cells.select { |(lr, _), _| lr == best_lr }
                 .map { |(_, s), c| [s, c] }.to_h
    arms[arm] = { lr: best_lr, seeds: seeds,
                  grid: lrs.sort_by(&:to_f),
                  alpha: seeds.values.map { |c| c[:alpha] }.compact.first }
  end
  next unless ok
  report[corpus] = arms
end

if report.empty?
  abort "p5_report: no complete corpora found under #{DIRS.join(', ')}"
end

puts MODE == "width" ?
  "P5 — NOMINAL HEAD WIDTH (one corpus: ae_shak_a65, effective rank fixed at 65), depth 4" :
  "P5 — controlled alphabet remap (one corpus: shakespeare), depth 4, nominal head 256"
puts "dirs: #{DIRS.join(', ')}"
puts

summary = {}
report.each do |corpus, arms|
  alpha = arms.values.map { |a| a[:alpha] }.compact.first
  # Only seeds where EVERY arm ran are usable — a per-seed difference
  # built from two different seeds is not a difference.
  common = arms.values.map { |a| a[:seeds].keys }.reduce(:&).sort
  bad = arms.flat_map { |name, a| a[:seeds].values.flat_map { |c| c[:bad].map { |b| "#{name}:#{b}" } } }.uniq
  puts "== #{corpus}  #{MODE == "width" ? "b_dim" : "alphabet"}=#{alpha}  seeds=#{common.inspect}"
  puts "   best LR: " + arms.map { |n, a| "#{n}=#{a[:lr]}" }.join(" ")
  arms.each do |n, a|
    if a[:lr] == a[:grid].first || a[:lr] == a[:grid].last
      puts "   EDGE OPTIMUM: #{n} best lr=#{a[:lr]} at grid boundary #{a[:grid].inspect} — not bracketed"
    end
  end
  puts "   INTEGRITY FAILURES: #{bad.join(', ')}" unless bad.empty?

  # Which DFA cut carries the corpus is decided ONCE, by median over the
  # common seeds — never per seed, which would be taking the min of two
  # noisy draws and would inflate every recovery number.
  dl = common.map { |s| arms["dfa-layer"][:seeds][s][:bpb] }
  ds = common.map { |s| arms["dfa-step"][:seeds][s][:bpb] }
  cut = median(dl) <= median(ds) ? "dfa-layer" : "dfa-step"
  bp = common.map { |s| arms["bp-body"][:seeds][s][:bpb] }
  fz = common.map { |s| arms["frozen-body"][:seeds][s][:bpb] }
  dfa = cut == "dfa-layer" ? dl : ds

  # The pre-registered gate: if bp does not beat frozen, the body is not
  # worth anything on this cell and every credit number derived from it
  # is uninterpretable. Checked PER SEED, not on the median.
  gate = common.each_index.map { |i| bp[i] < fz[i] }
  puts "   GATE bp<frozen: #{gate.count(true)}/#{gate.size}" +
       (gate.all? ? "" : "  ** FAILED — drop this point **")

  abs  = common.each_index.map { |i| fz[i] - dfa[i] }
  frac = common.each_index.map { |i| (fz[i] - dfa[i]) / (fz[i] - bp[i]) }
  puts "   carried cut: #{cut}  (layer median %.3f vs step median %.3f)" % [median(dl), median(ds)]
  puts "   bpb    bp %.3f  dfa %.3f  frozen %.3f   (medians)" % [median(bp), median(dfa), median(fz)]
  puts "   body worth (frozen-bp): mean %.3f sd %.3f" % [mean(fz.each_index.map { |i| fz[i] - bp[i] }),
                                                          sd(fz.each_index.map { |i| fz[i] - bp[i] })]
  puts "   ABS bits recovered:     mean %+.3f sd %.3f  min %+.3f max %+.3f" %
       [mean(abs), sd(abs), abs.min, abs.max]
  puts "   fraction:               mean %+.1f%% sd %.1f  min %+.1f%% max %+.1f%%" %
       [mean(frac) * 100, sd(frac) * 100, frac.min * 100, frac.max * 100]
  puts "   per-seed abs:  " + abs.map { |v| "%+.3f" % v }.join(" ")
  puts "   per-seed frac: " + frac.map { |v| "%+.1f%%" % (v * 100) }.join(" ")
  puts
  summary[corpus] = { alpha: alpha, abs: abs, frac: frac, cut: cut,
                      bp: bp, fz: fz, dfa: dfa, seeds: common,
                      gate: gate.all?, bad: bad }
end

puts "== pairwise separation on the PRIMARY statistic (absolute bits recovered) =="
keys = summary.keys
keys.combination(2).each do |a, b|
  ta = welch_t(summary[a][:abs], summary[b][:abs])
  # The fraction's |t| is printed beside it because the two statistics
  # answer to different confounds: the absolute scale is blind to task
  # difficulty, the fraction divides by a small noisy denominator. A
  # difference that only shows up on one of them is a claim about that
  # statistic, not about the axis — say which, and say it out loud.
  tf = welch_t(summary[a][:frac], summary[b][:frac])
  puts "  %-14s (a=%d, %+.3f)  vs  %-14s (a=%d, %+.3f)   d=%+.3f  Welch |t|=%.2f  [frac |t|=%.2f]" %
       [a, summary[a][:alpha], mean(summary[a][:abs]),
        b, summary[b][:alpha], mean(summary[b][:abs]),
        mean(summary[a][:abs]) - mean(summary[b][:abs]), ta.abs, tf.abs]
end
puts
if MODE != "width"
puts "== the inflation control's own check =="
puts "  On ae_shak_a129 / a192 the added entropy is log2(m) on EVERY arm, so"
puts "  `body worth` (frozen-bp) should be UNCHANGED from ae_shakespeare in"
puts "  theory. If it moved, the inflation is not the control it claims to be"
puts "  and no alphabet reading off these arms is safe."
["ae_shakespeare", "ae_shak_a129", "ae_shak_a192"].each do |c|
  next unless summary[c]
  w = summary[c][:seeds].each_index.map { |i| summary[c][:fz][i] - summary[c][:bp][i] }
  puts "  %-14s body worth mean %.3f sd %.3f" % [c, mean(w), sd(w)]
end

end

dest = MODE == "width" ? "/tmp/p5hw_summary.json" : "/tmp/p5_summary.json"
File.write(dest, JSON.pretty_generate(summary))
puts
puts "raw summary -> #{dest}"
