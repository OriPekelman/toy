#!/usr/bin/env ruby
# prep/e1_ndfa_report.rb — toy#172/E1 Phase 1.2: read the nDFA arms off
# the P6 rank ladder beside the unpreconditioned ones.
#
# PRIMARY STATISTIC = PER-ARM EXCESS OVER log2(m), the same one P6's
# verdict is stated on (prep/p5_report.rb MODE=rank). Each rung's
# inflation adds b = H(rung) - H(base) bits of pure noise to the label,
# so the part of a bpb rise that is NOT b is the part the credit rule is
# responsible for. Paired BY SEED against the b=0 rung, so a seed's own
# draw cancels rather than adding variance to a difference of two noisy
# quantities.
#
# Recovery (frozen - dfa)/(frozen - bp) is printed SECOND and stays
# second: its denominator is a small difference of two noisy numbers and
# toy#170 measured it flipping sign between seeds on a 0.5-bit gap.
#
# ── WHAT THE bp AND frozen COLUMNS CAN AND CANNOT SAY HERE ──
#
# E1's discriminator is "nDFA moves the DFA arm and leaves bp/frozen
# unmoved". On this implementation bp and frozen are unmoved BY
# CONSTRUCTION: nDFA folds into B, B exists only where a `dfa` block
# taps, and the runner REFUSES the flag on any other policy. So their
# columns are the reference the dfa excess is measured against, and the
# "did it move all three" branch is answered structurally rather than
# measured. That is stated in the output, every time, because a
# tautology printed in a results table reads exactly like evidence.
require "json"

P6   = ENV["BASE"] || "/srv/data/scratch/p6"
ND   = ENV["DIR"]  || "/srv/data/scratch/e1ndfa"
HEAD = ENV["HEAD"] || "4096"
RUNGS = (ENV["RUNGS"] || "ae_shak_a65 ae_shak_a508 ae_shak_a1008 ae_shak_a2504").split
LR_BP  = (ENV["LR_BP"]  || "0.0003 0.001 0.003 0.01").split
LR_DFA = (ENV["LR_DFA"] || "0.00001 0.00003 0.0001 0.0003 0.001 0.003").split
# The cadence, the sample count and the gain rule are part of the arm's
# IDENTITY, not decoration: two cells at the same lambda and different
# cadences are different experiments. Filtered here rather than merged,
# because merging them is silent and produces a table that looks fine.
EVERY = ENV["EVERY"] || "500"
MSAMP = ENV["M"]     || "256"
GAIN  = ENV["GAIN"]  || "preserve"

def mean(a) = a.empty? ? nil : a.sum / a.size.to_f
def sd(a)
  return 0.0 if a.size < 2
  m = mean(a)
  Math.sqrt(a.sum { |x| (x - m)**2 } / (a.size - 1).to_f)
end
def bpb_of(f)
  return nil unless File.file?(f)
  v = File.read(f)[/bpb=([0-9.]+)/, 1]
  v&.to_f
end

# The per-rung best LR, re-derived exactly as prep/p6_ladder.sh derived
# it — off seed 0, over that arm's own grid. Never a single LR across
# rungs: the DFA optimum slid 3e-4 -> 1e-3 and then CRASHED to 3e-5.
def best_lr(dir, arm, rung, grid, head)
  best = nil
  bv = nil
  grid.each do |lr|
    v = bpb_of(File.join(dir, "#{arm}_#{rung}_v#{head}_lr#{lr}_s0.txt"))
    next unless v
    if bv.nil? || v < bv
      bv = v
      best = lr
    end
  end
  [best, bv]
end

# ---- gather ----
base_rung = RUNGS.first
ent = {}
RUNGS.each do |r|
  jf = File.join("data", "#{r}.json")
  abort "e1_ndfa_report: missing #{jf}" unless File.file?(jf)
  ent[r] = JSON.parse(File.read(jf))["entropy"]
end

off = {}    # off[[rung, arm]] = {seed => bpb}
lrs = {}
RUNGS.each do |r|
  { "bp-body" => LR_BP, "frozen-body" => LR_BP, "dfa-layer" => LR_DFA }.each do |arm, grid|
    lr, = best_lr(P6, arm, r, grid, HEAD)
    next unless lr
    lrs[[r, arm]] = lr
    h = {}
    Dir[File.join(P6, "#{arm}_#{r}_v#{HEAD}_lr#{lr}_s*.txt")].each do |f|
      s = File.basename(f)[/_s(\d+)\.txt\z/, 1].to_i
      v = bpb_of(f)
      h[s] = v if v
    end
    off[[r, arm]] = h
  end
end

nd = {}     # nd[[rung, lambda]] = {seed => bpb}
ndmeta = {} # provenance, one sample per (rung, lambda)
brk = {}    # brk[[rung, lambda]] = {lr => {seed => bpb}} — the LR bracket
Dir[File.join(ND, "dfa-ndfa_*.txt")].sort.each do |f|
  b = File.basename(f)
  m = b.match(/\Adfa-ndfa_(.+)_v#{HEAD}_lam([0-9.e+-]+)_e(\d+)_m(\d+)_(\w+)_lr([0-9.e-]+)_s(\d+)\.txt\z/)
  next unless m
  rung, lam, every, msz, gain, lr, seed = m[1], m[2], m[3], m[4], m[5], m[6], m[7].to_i
  next unless RUNGS.include?(rung)
  next unless every == EVERY && msz == MSAMP && gain == GAIN
  v = bpb_of(f)
  next unless v
  # Every cell joins the LR BRACKET table; only cells at the rung's
  # plain-DFA optimum join the sweep. Mixing the two would compare an arm
  # against another arm's LR, which is the one thing this program
  # refuses to do — but throwing the bracket away would hide the check
  # that makes the comparison legitimate, so it gets its own table.
  ((brk[[rung, lam]] ||= {})[lr] ||= {})[seed] = v
  next unless lrs[[rung, "dfa-layer"]] == lr
  body = File.read(f)
  (nd[[rung, lam]] ||= {})[seed] = v
  ndmeta[[rung, lam]] ||= {
    every: every, m: msz, gain: gain, lr: lr,
    line: body.lines.find { |l| l.start_with?("ndfa: ") }.to_s.chomp
  }
end

abort "e1_ndfa_report: no nDFA cells under #{ND}" if nd.empty?
LAMS = nd.keys.map { |(_, l)| l }.uniq.sort_by { |l| -l.to_f }

puts "E1 Phase 1.2 — nDFA across the P6 rank ladder"
puts "  off arms: #{P6} (reused; GTX_NDFA=0 is byte-identical to the pre-flag runner)"
puts "  nDFA arms: #{ND}  [every=#{EVERY} m=#{MSAMP} gain=#{GAIN}]"
puts

puts "== PER-RUNG LR (P6's own per-arm optimum, seed 0) =="
RUNGS.each do |r|
  puts "  %-16s bp=%-8s frozen=%-8s dfa=%-8s" %
       [r, lrs[[r, "bp-body"]], lrs[[r, "frozen-body"]], lrs[[r, "dfa-layer"]]]
end
puts

# ---- the primary table ----
puts "== PER-ARM EXCESS over the irreducible log2(m)  [PRIMARY] =="
puts "   paired by seed against #{base_rung}; lower is better; bp/frozen are the"
puts "   REFERENCE arms and cannot carry nDFA (there is no B on them)"
puts
hdr = "   %-16s %5s %6s  %-15s %-15s %-15s" % ["rung", "rank", "b", "bp", "frozen", "dfa (off)"]
LAMS.each { |l| hdr += " %-15s" % "dfa lam=#{l}" }
puts hdr

rows = {}
RUNGS.each do |r|
  b = ent[r] - ent[base_rung]
  rank = File.file?("data/#{r}.json") ? JSON.parse(File.read("data/#{r}.json"))["distinct"] : 0
  cells = []
  series = [["bp-body", off[[r, "bp-body"]]], ["frozen-body", off[[r, "frozen-body"]]],
            ["dfa-layer", off[[r, "dfa-layer"]]]]
  LAMS.each { |l| series << ["dfa-lam#{l}", nd[[r, l]]] }
  series.each do |name, h|
    basearm = name.start_with?("dfa-lam") ? off[[base_rung, "dfa-layer"]] : off[[base_rung, name]]
    if h.nil? || basearm.nil?
      cells << "%-15s" % "-"
      next
    end
    # The nDFA arms' own b=0 reference is the nDFA cell at the base rung
    # when one exists — an arm must be paired against ITSELF, or the
    # excess absorbs whatever nDFA did at the base rung.
    if name.start_with?("dfa-lam")
      lam = name.sub("dfa-lam", "")
      basearm = nd[[base_rung, lam]] || basearm
    end
    common = (h.keys & basearm.keys).sort
    if common.empty?
      cells << "%-15s" % "-"
      next
    end
    ex = common.map { |s| (h[s] - basearm[s]) - b }
    rows[[r, name]] = ex
    cells << "%-15s" % ("%+.3f+-%.3f(%d)" % [mean(ex), sd(ex), common.size])
  end
  puts "   %-16s %5d %6.3f  %s" % [r, rank, b, cells.join(" ")]
end
puts "   (the #{base_rung} row is the reference — 0 by construction, shown for orientation)"
puts

# ---- the raw paired delta, which the excess table CANNOT show ----
#
# The excess statistic is a rung-to-rung difference against the b=0 rung,
# and each arm is paired against ITSELF there — so whatever nDFA does at
# the base rung is subtracted out of every row including its own, and the
# base row reads +0.000 for every arm by construction. That is correct for
# the P6-comparable statistic and completely blind to an effect at rank
# 65. This table is the direct readout: same seed, same LR, same rung,
# nDFA on minus nDFA off.
puts "== RAW PAIRED bpb DELTA (nDFA - off), same seed / rung / LR =="
puts "   NEGATIVE = nDFA helps. This is the only table that can see the"
puts "   base rung, where the excess statistic is 0 by construction."
puts "   %-16s %-15s %s" % ["rung", "lambda", "delta bpb"]
RUNGS.each do |r|
  o = off[[r, "dfa-layer"]]
  next if o.nil?
  LAMS.each do |l|
    a = nd[[r, l]]
    next if a.nil?
    common = (a.keys & o.keys).sort
    next if common.empty?
    d = common.map { |s| a[s] - o[s] }
    puts "   %-16s %-15s %+.4f +- %.4f  (n=%d)" % [r, l, mean(d), sd(d), common.size]
  end
end
puts

# ---- the discriminator ----
puts "== THE DISCRIMINATOR: does nDFA move the DFA arm? =="
puts "   d = excess(dfa, nDFA) - excess(dfa, off), paired by seed. NEGATIVE = nDFA helps."
puts "   bp and frozen are omitted because nDFA CANNOT be applied to them:"
puts "   it folds into B, and B exists only where a `dfa` block taps. The runner"
puts "   REFUSES the flag on any other policy (prep/gtx_gate.rb leg 16), so"
puts "   'bp and frozen unmoved' here is a structural fact, not a measurement."
puts
puts
puts "   THE EXCESS DELTA IS DECOMPOSED, and it has to be. The excess is a"
puts "   difference against the b=0 rung, so"
puts "       delta_excess(rung) = delta_bpb(rung) - delta_bpb(base)."
puts "   If nDFA damages the BASE rung and leaves a wide rung alone, the"
puts "   excess at the wide rung IMPROVES by exactly the damage done at the"
puts "   base — a number that reads as a rescue and is the reference getting"
puts "   worse. Both terms are printed; `base-driven` flags the rows where"
puts "   the base term is the larger of the two."
puts
puts "   %-16s %-9s %-19s %-19s %-19s %s" %
     ["rung", "lambda", "delta excess", "  = delta bpb(rung)", "  - delta bpb(base)", "reading"]
verdicts = []
RUNGS.each do |r|
  next if r == base_rung
  LAMS.each do |l|
    a = nd[[r, l]]
    o = off[[r, "dfa-layer"]]
    nb = nd[[base_rung, l]] || off[[base_rung, "dfa-layer"]]
    ob = off[[base_rung, "dfa-layer"]]
    next if a.nil? || o.nil? || nb.nil? || ob.nil?
    common = (a.keys & o.keys & nb.keys & ob.keys).sort
    next if common.empty?
    d     = common.map { |s| (a[s] - nb[s]) - (o[s] - ob[s]) }
    d_r   = common.map { |s| a[s] - o[s] }
    d_b   = common.map { |s| nb[s] - ob[s] }
    verdicts << [r, l, mean(d), sd(d), common.size, mean(d_r), mean(d_b)]
    tag = if mean(d_r).abs < 1.0e-9 && mean(d_b).abs < 1.0e-9
            "identity (control)"
          elsif mean(d_b).abs > mean(d_r).abs
            "BASE-DRIVEN"
          elsif mean(d_r) < 0.0
            "rung improved"
          else
            "rung worsened"
          end
    puts "   %-16s %-9s %+.4f +- %-8.4f %+.4f%-13s %+.4f%-13s %s" %
         [r, l, mean(d), sd(d), mean(d_r), "", mean(d_b), "", tag]
  end
end
puts

# ---- recovery, second ----
puts "== RECOVERY (frozen - dfa) / (frozen - bp)  [SECONDARY] =="
puts "   a ratio of two noisy differences; toy#170 saw it flip sign between seeds"
puts "   on a 0.5-bit denominator. Read the excess table above first."
puts "   %-16s %-15s %8s %8s" % ["rung", "arm", "abs bits", "fraction"]
RUNGS.each do |r|
  bp = off[[r, "bp-body"]]
  fz = off[[r, "frozen-body"]]
  next if bp.nil? || fz.nil?
  series = [["dfa (off)", off[[r, "dfa-layer"]]]]
  LAMS.each { |l| series << ["dfa lam=#{l}", nd[[r, l]]] }
  series.each do |name, h|
    next if h.nil?
    common = (h.keys & bp.keys & fz.keys).sort
    next if common.empty?
    abs  = common.map { |s| fz[s] - h[s] }
    frac = common.map { |s| (fz[s] - bp[s]).abs < 1e-9 ? 0.0 : (fz[s] - h[s]) / (fz[s] - bp[s]) }
    puts "   %-16s %-15s %+8.3f %8.3f   (n=%d)" % [r, name, mean(abs), mean(frac), common.size]
  end
end
puts

# ---- the per-arm LR check, which the sweep tables depend on ----
#
# "An arm measured at ANOTHER arm's cell is not a negative" is this
# program's most expensive rule (toy#160 nearly published "attention is
# DFA-hostile" from BP's LR). nDFA is a new arm and it gets the same
# treatment: where a bracket exists it is printed in full, with nDFA's
# own optimum beside the plain arm's, so a harmful reading can be
# checked against the possibility that it is only an LR mismatch.
if brk.any? { |_, h| h.size > 1 }
  puts "== nDFA's OWN LR BRACKET (seed 0 unless noted) =="
  puts "   %-16s %-9s %-11s %-11s %-11s %s" %
       ["rung", "lambda", "nDFA best", "nDFA bpb", "plain best", "plain bpb"]
  brk.keys.sort.each do |k|
    h = brk[k]
    next if h.size < 2
    rung, lam = k
    blr, bv = h.map { |lr, sh| [lr, sh[0]] }.reject { |_, v| v.nil? }.min_by { |_, v| v }
    plr = lrs[[rung, "dfa-layer"]]
    pv  = bpb_of(File.join(P6, "dfa-layer_#{rung}_v#{HEAD}_lr#{plr}_s0.txt"))
    next if blr.nil? || pv.nil?
    flag = blr == plr ? "" : "   <- OPTIMUM MOVED"
    puts "   %-16s %-9s %-11s %-11.4f %-11s %-11.4f%s" % [rung, lam, blr, bv, plr, pv, flag]
  end
  puts
  # ...and the comparison that actually respects the rule: EACH ARM AT
  # ITS OWN CELL, over every seed run at those two LRs. This is the row
  # to quote. The shared-LR table above it is the same measurement with
  # nDFA held at the plain arm's rate, which flatters neither arm
  # consistently but is not the fair form.
  puts "== PAIRED DELTA WITH EACH ARM AT ITS OWN BRACKETED LR [the fair form] =="
  puts "   %-16s %-9s %-10s %-10s %s" % ["rung", "lambda", "nDFA lr", "plain lr", "delta bpb"]
  brk.keys.sort.each do |k|
    h = brk[k]
    next if h.size < 2
    rung, lam = k
    blr, = h.map { |lr, sh| [lr, sh[0]] }.reject { |_, v| v.nil? }.min_by { |_, v| v }
    plr = lrs[[rung, "dfa-layer"]]
    next if blr.nil? || plr.nil?
    a = h[blr]
    o = off[[rung, "dfa-layer"]]
    next if a.nil? || o.nil?
    common = (a.keys & o.keys).sort
    next if common.empty?
    d = common.map { |s| a[s] - o[s] }
    puts "   %-16s %-9s %-10s %-10s %+.4f +- %.4f  (n=%d)" %
         [rung, lam, blr, plr, mean(d), sd(d), common.size]
  end
  puts
end

# ---- provenance, always ----
puts "== nDFA PROVENANCE (one cell per rung x lambda) =="
ndmeta.keys.sort.each { |k| puts "   #{k[0]} lam=#{k[1]}: #{ndmeta[k][:line]}" }
puts

# ---- the verdict, spelled out ----
puts "== VERDICT =="
if verdicts.empty?
  puts "   not enough paired cells yet"
else
  best = verdicts.min_by { |v| v[2] }
  worst = verdicts.max_by { |v| v[2] }
  moved = verdicts.select { |v| v[2].abs > 2.0 * (v[3] / Math.sqrt(v[4])) }
  based = verdicts.select { |v| v[6].abs > v[5].abs }
  puts "   best  delta_excess: %s lam=%s  %+.4f +- %.4f" % [best[0], best[1], best[2], best[3]]
  puts "   worst delta_excess: %s lam=%s  %+.4f +- %.4f" % [worst[0], worst[1], worst[2], worst[3]]
  puts "   cells whose delta_excess exceeds 2 SEM: #{moved.size} of #{verdicts.size}"
  puts "   cells whose delta_excess is BASE-DRIVEN: #{based.size} of #{verdicts.size}"
  puts
  # The raw table is what decides, not the excess table. A negative excess
  # that comes from the base rung getting worse is not a rescue, and it is
  # the exact shape of the confound this arc keeps finding (P4's udhr
  # language confound, Phase 1.1's sampling ceiling).
  raw = []
  RUNGS.each do |r|
    o = off[[r, "dfa-layer"]]
    next if o.nil?
    LAMS.each do |l|
      next if l.to_f > 1.0e6   # the identity control is not a measurement
      a = nd[[r, l]]
      next if a.nil?
      common = (a.keys & o.keys).sort
      next if common.empty?
      raw << [r, l, mean(common.map { |s| a[s] - o[s] })]
    end
  end
  helped = raw.count { |x| x[2] < -0.01 }
  hurt   = raw.count { |x| x[2] > 0.01 }
  puts "   ON RAW bpb (the table that cannot be base-driven): nDFA HELPED in"
  puts "   #{helped} of #{raw.size} rung x lambda cells and HURT in #{hurt}."
  puts
  puts "   E1 branches: `full rescue` needs the dfa arm's ABSOLUTE bpb driven"
  puts "   toward bp's; `partial` needs a consistent raw improvement outside"
  puts "   noise; `inert` is |raw delta| inside noise everywhere. A negative"
  puts "   excess with a zero raw delta is NONE of those — it is the base"
  puts "   rung getting worse, and it must be reported as that."
end
