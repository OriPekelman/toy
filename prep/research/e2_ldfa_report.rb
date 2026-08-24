#!/usr/bin/env ruby
# prep/research/e2_ldfa_report.rb — toy#172/E2: read the LDFA arms off the P6 rank
# ladder. Fixed-wide vs fixed-lowrank vs adaptive-lowrank, at each rung
# and each r.
#
# ── THE HYPOTHESIS IS THE FIXED-VS-ADAPTIVE CONTRAST ──
#
# LDFA predicts fixed low-rank still degrades (spurious fixed points) and
# adaptive low-rank flattens. So the row that decides is
# `dfa-oja-r minus dfa-fix-r at the same rung and the same r`, with the
# full-width arm printed beside it — a contrast between two low-rank arms
# that both beat or both lose to the wide baseline says something quite
# different from one that crosses it.
#
# Unlike E1 Phase 1.2's discriminator, this one carries evidential weight:
# BOTH arms modify B, both collect the same error samples on the same
# steps, and the runner gates that at eta=0 they are byte-identical. The
# only difference is the Oja update.
#
# ── PRIMARY IS RAW / ABSOLUTE, NOT EXCESS ──
#
# Excess-over-log2(m) is printed because it is P6's statistic and the
# ladder's shape is stated on it, but it is a difference against the b=0
# rung and an intervention that damages the REFERENCE rung reads as an
# improvement everywhere above. That confound has now appeared three
# times in this arc (P4's udhr language confound, Phase 1.1's sampling
# ceiling, Phase 1.2's nDFA base damage), so every excess row here is
# DECOMPOSED into its rung term and its base term and tagged BASE-DRIVEN
# when the base term is the larger.
#
# ── rank_eff IS NOT r ──
#
# rank(Q.P) <= min(dout, r) and dout = d_model = 128 on this fixture, so
# r = 256 does NOT give a rank-256 feedback path — it gives the SAME
# matrix rank as full width with the row space confined to P's span. The
# provenance table prints rank_eff beside r and the verdict reads the
# ladder accordingly.
require "json"

DIR   = ENV["DIR"] || "/srv/data/scratch/e2ldfa"
HEAD  = ENV["HEAD"] || "4096"
RUNGS = (ENV["RUNGS"] || "ae_shak_a65 ae_shak_a508 ae_shak_a1008 ae_shak_a2504").split
RANKS = (ENV["RANKS"] || "16 64 256").split
# The bp/frozen grid runs LOWER than P6's because it had to: at the top
# rung the frozen control's optimum landed on 3e-4, the boundary of the
# original {3e-4 .. 1e-2} grid, and an optimum on an edge is not an
# optimum. Extending down to 1e-5 put it back in the interior (1e-4
# 10.814 / 3e-4 10.557 / 1e-3 10.581) at the same value — but the frozen
# control is what RECOVERY is measured against, so leaving it unbracketed
# would have put an unmeasured arm in the denominator of the headline.
LR_BP  = (ENV["LR_BP"]  || "0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01").split
LR_DFA = (ENV["LR_DFA"] || "0.00001 0.00003 0.0001 0.0003 0.001 0.003").split

def mean(a) = a.empty? ? nil : a.sum / a.size.to_f
def sd(a)
  return 0.0 if a.size < 2
  m = mean(a)
  Math.sqrt(a.sum { |x| (x - m)**2 } / (a.size - 1).to_f)
end
def sem(a) = a.size < 2 ? 0.0 : sd(a) / Math.sqrt(a.size)
def bpb_of(f)
  return nil unless File.file?(f)
  File.read(f)[/bpb=([0-9.]+)/, 1]&.to_f
end
def ldfa_line(f)
  return nil unless File.file?(f)
  File.read(f).lines.find { |l| l.start_with?("ldfa: ") }&.chomp
end
def fld(line, key)
  line && line[/#{Regexp.escape(key)}=([0-9.eE+-]+)/, 1]
end

ARMS = ["bp-body", "frozen-body", "dfa-wide"] +
       RANKS.flat_map { |r| ["dfa-fix-#{r}", "dfa-oja-#{r}"] }
DFA_ARMS = ARMS.reject { |a| a == "bp-body" || a == "frozen-body" }

def path(arm, rung, lr, seed) = File.join(DIR, "#{arm}_#{rung}_v#{HEAD}_lr#{lr}_s#{seed}.txt")

# ---- the per-arm, per-rung LR bracket ----
#
# "An arm measured at ANOTHER arm's cell is not a negative" is this
# program's most expensive rule, and E2 adds seven new arms to a lane
# where the DFA optimum already slid 3e-4 -> 1e-3 and then CRASHED to
# 3e-5 across the ladder. Every arm is bracketed at every rung, and an
# optimum sitting on a grid boundary is FLAGGED rather than used quietly.
best = {}   # best[[rung, arm]] = [lr, bpb, edge?]
RUNGS.each do |r|
  ARMS.each do |a|
    grid = (a == "bp-body" || a == "frozen-body") ? LR_BP : LR_DFA
    cand = grid.map { |lr| [lr, bpb_of(path(a, r, lr, 0))] }.reject { |_, v| v.nil? }
    next if cand.empty?
    lr, v = cand.min_by { |_, x| x }
    best[[r, a]] = [lr, v, lr == grid.first || lr == grid.last]
  end
end

# ---- gather the seeds at each arm's OWN cell ----
vals = {}   # vals[[rung, arm]] = {seed => bpb}
prov = {}   # prov[[rung, arm]] = the ldfa: line
RUNGS.each do |r|
  ARMS.each do |a|
    b = best[[r, a]]
    next unless b
    h = {}
    Dir[File.join(DIR, "#{a}_#{r}_v#{HEAD}_lr#{b[0]}_s*.txt")].each do |f|
      s = File.basename(f)[/_s(\d+)\.txt\z/, 1].to_i
      v = bpb_of(f)
      h[s] = v if v
    end
    vals[[r, a]] = h unless h.empty?
    prov[[r, a]] = ldfa_line(path(a, r, b[0], 0))
  end
end
abort "e2_ldfa_report: no cells under #{DIR}" if vals.empty?

ent = {}
RUNGS.each do |r|
  jf = File.join("data", "#{r}.json")
  abort "e2_ldfa_report: missing #{jf}" unless File.file?(jf)
  ent[r] = JSON.parse(File.read(jf))
end
base_rung = RUNGS.first

puts "E2 — LDFA (adaptive low-rank feedback) across the P6 rank ladder"
puts "  cells: #{DIR}   head #{HEAD}   ranks #{RANKS.join(', ')}"
puts "  device: the tao#24 CUDA twin throughout. P6's cells are CPU and are"
puts "  NOT quoted here — dfa-fixed-wide, bp and frozen are re-measured on"
puts "  this device, per the E-series one-device rule."
puts

# ---- 0. the integrity checks, first, because the rest depends on them ----
puts "== INTEGRITY: SCALE, RANK AND CONVERGENCE =="
puts "   The scale match is what makes any of this mean anything: a rank-r Q.P"
puts "   has a different ||B||_F from the full-width B it replaces, so an"
puts "   unnormalised arm would make 'low rank hurts' indistinguishable from"
puts "   'the updates got smaller'. p_energy against p_energy_rand = r/V is the"
puts "   convergence check on Oja: an adaptation that ran but learned nothing"
puts "   sits at the random baseline."
puts
puts "   %-16s %-12s %5s %8s %14s %10s %10s %7s" %
     ["rung", "arm", "r", "rank_eff", "scale_ratio", "p_energy", "rand r/V", "x rand"]
scale_bad = 0
RUNGS.each do |r|
  DFA_ARMS.each do |a|
    next if a == "dfa-wide"
    l = prov[[r, a]]
    next unless l
    sr = fld(l, "scale_ratio").to_f
    pe = fld(l, "p_energy").to_f
    pr = fld(l, "p_energy_rand").to_f
    scale_bad += 1 unless (sr - 1.0).abs < 1e-9
    puts "   %-16s %-12s %5s %8s %14.12f %10.5f %10.5f %7.1f%s" %
         [r, a, fld(l, "rank"), fld(l, "rank_eff"), sr, pe, pr,
          pr > 0 ? pe / pr : 0.0,
          (sr - 1.0).abs < 1e-9 ? "" : "   <- SCALE MISMATCH"]
  end
end
puts
puts scale_bad.zero? ?
  "   ALL cells match ||B_full||_F to 1e-9: the arms differ in RANK and not in SCALE." :
  "   #{scale_bad} cells FAIL the scale match — those rows are confounded by a gain."
puts

# ---- 1. the LR bracket ----
puts "== PER-ARM, PER-RUNG LR (each arm at its OWN bracketed optimum, seed 0) =="
puts "   %-16s %s" % ["rung", ARMS.map { |a| "%-14s" % a }.join]
RUNGS.each do |r|
  cells = ARMS.map do |a|
    b = best[[r, a]]
    b ? ("%-14s" % (b[0] + (b[2] ? "*" : ""))) : ("%-14s" % "-")
  end
  puts "   %-16s %s" % [r, cells.join]
end
edges = best.select { |_, v| v[2] }
puts "   * = optimum on a GRID BOUNDARY. An optimum on an edge is not an optimum;"
puts "     #{edges.size} of #{best.size} cells are flagged." if edges.any?
puts "   no cell sits on a grid boundary." if edges.empty?
puts

# ---- 2. raw bpb, the primary ----
puts "== RAW HELD-OUT bpb, each arm at its own cell  [PRIMARY] =="
puts "   lower is better. This is the table that cannot be base-driven."
puts "   %-16s %s" % ["rung", ARMS.map { |a| "%-16s" % a }.join]
RUNGS.each do |r|
  cells = ARMS.map do |a|
    h = vals[[r, a]]
    next ("%-16s" % "-") unless h
    "%-16s" % ("%.3f+-%.3f(%d)" % [mean(h.values), sd(h.values), h.size])
  end
  puts "   %-16s %s" % [r, cells.join]
end
puts

# ---- 3. THE CONTRAST ----
puts "== THE HYPOTHESIS: adaptive MINUS fixed at the same r  [THE CONTRAST] =="
puts "   paired by seed, each arm at its OWN bracketed LR. NEGATIVE = the"
puts "   adaptation helps. The wide baseline is printed beside it because two"
puts "   low-rank arms that both lose to full width say something different"
puts "   from two that straddle it."
puts
puts "   %-16s %4s %-20s %-16s %-16s %-16s" %
     ["rung", "r", "oja - fix", "fix - wide", "oja - wide", "reading"]
contrast = []
RUNGS.each do |r|
  w = vals[[r, "dfa-wide"]]
  RANKS.each do |rk|
    f = vals[[r, "dfa-fix-#{rk}"]]
    o = vals[[r, "dfa-oja-#{rk}"]]
    next if f.nil? || o.nil?
    cs = (f.keys & o.keys).sort
    next if cs.empty?
    d = cs.map { |s| o[s] - f[s] }
    fw = w ? (f.keys & w.keys).sort.map { |s| f[s] - w[s] } : []
    ow = w ? (o.keys & w.keys).sort.map { |s| o[s] - w[s] } : []
    sig = mean(d).abs > 2.0 * sem(d)
    tag = if !sig                      then "inside 2 SEM"
          elsif mean(d) < 0.0 && !ow.empty? && mean(ow) < 0.0 then "ADAPTIVE HELPS, beats wide"
          elsif mean(d) < 0.0          then "adaptive helps, still under wide"
          else                              "adaptive HURTS"
          end
    contrast << [r, rk, mean(d), sem(d), cs.size, fw.empty? ? nil : mean(fw),
                 ow.empty? ? nil : mean(ow), tag]
    puts "   %-16s %4s %+.4f+-%-11.4f %-16s %-16s %s" %
         [r, rk, mean(d), sd(d),
          fw.empty? ? "-" : ("%+.4f" % mean(fw)),
          ow.empty? ? "-" : ("%+.4f" % mean(ow)), tag]
  end
end
puts

# ---- 4. excess, with the decomposition ----
puts "== PER-ARM EXCESS over the irreducible log2(m)  [P6's statistic, SECONDARY] =="
puts "   paired by seed against #{base_rung}, EACH ARM AGAINST ITSELF there."
puts "   READ THE DECOMPOSITION, NOT THE HEADLINE. excess(rung) = bpb(rung) -"
puts "   bpb(base) - b, so an intervention that damages the BASE rung reads as"
puts "   an improvement at every rung above it. That confound has now appeared"
puts "   three times in this arc; every row below is decomposed and tagged."
puts
puts "   %-16s %5s %6s  %s" %
     ["rung", "rank", "b", ARMS.map { |a| "%-16s" % a }.join]
excess = {}
RUNGS.each do |r|
  b = ent[r]["entropy"] - ent[base_rung]["entropy"]
  cells = ARMS.map do |a|
    h  = vals[[r, a]]
    hb = vals[[base_rung, a]]
    next ("%-16s" % "-") if h.nil? || hb.nil?
    cs = (h.keys & hb.keys).sort
    next ("%-16s" % "-") if cs.empty?
    ex = cs.map { |s| (h[s] - hb[s]) - b }
    excess[[r, a]] = ex
    "%-16s" % ("%+.3f+-%.3f(%d)" % [mean(ex), sd(ex), cs.size])
  end
  puts "   %-16s %5d %6.3f  %s" % [r, ent[r]["distinct"], b, cells.join]
end
puts "   (the #{base_rung} row is the reference — 0 by construction, shown for orientation)"
puts
puts "== EXCESS DECOMPOSED: is any negative-excess row base-driven? =="
puts "   for each low-rank arm, against the WIDE arm:"
puts "     delta_excess = [bpb(arm,rung) - bpb(wide,rung)] - [bpb(arm,base) - bpb(wide,base)]"
puts "   %-16s %-12s %-20s %-14s %-14s %s" %
     ["rung", "arm", "delta excess", "rung term", "base term", "reading"]
based = 0
RUNGS.each do |r|
  next if r == base_rung
  wr = vals[[r, "dfa-wide"]]
  wb = vals[[base_rung, "dfa-wide"]]
  next if wr.nil? || wb.nil?
  DFA_ARMS.each do |a|
    next if a == "dfa-wide"
    hr = vals[[r, a]]
    hb = vals[[base_rung, a]]
    next if hr.nil? || hb.nil?
    cs = (hr.keys & hb.keys & wr.keys & wb.keys).sort
    next if cs.empty?
    d_r = cs.map { |s| hr[s] - wr[s] }
    d_b = cs.map { |s| hb[s] - wb[s] }
    d   = cs.map { |s| (hr[s] - wr[s]) - (hb[s] - wb[s]) }
    bd  = mean(d_b).abs > mean(d_r).abs
    based += 1 if bd && mean(d) < 0.0
    tag = bd ? "BASE-DRIVEN" : (mean(d) < 0.0 ? "rung improved" : "rung worsened")
    puts "   %-16s %-12s %+.4f+-%-12.4f %+.4f%-8s %+.4f%-8s %s" %
         [r, a, mean(d), sd(d), mean(d_r), "", mean(d_b), "", tag]
  end
end
puts
puts based.zero? ? "   no negative-excess row is base-driven." :
                   "   #{based} negative-excess rows are BASE-DRIVEN and must not be read as improvements."
puts

# ---- 5. recovery, secondary ----
puts "== RECOVERY vs the frozen control  [SECONDARY] =="
puts "   ABSOLUTE bits recovered (frozen - arm) is the primary of this pair;"
puts "   the FRACTION divides by (frozen - bp), a small difference of two noisy"
puts "   numbers that toy#170 measured flipping SIGN between seeds. A POSITIVE"
puts "   absolute means the arm beats a frozen body; NEGATIVE means it loses to"
puts "   one, which is what P6 found at wide output."
puts "   %-16s %-12s %10s %10s %6s" % ["rung", "arm", "abs bits", "fraction", "n"]
recov = {}
RUNGS.each do |r|
  bp = vals[[r, "bp-body"]]
  fz = vals[[r, "frozen-body"]]
  next if bp.nil? || fz.nil?
  DFA_ARMS.each do |a|
    h = vals[[r, a]]
    next if h.nil?
    cs = (h.keys & bp.keys & fz.keys).sort
    next if cs.empty?
    abs  = cs.map { |s| fz[s] - h[s] }
    frac = cs.map { |s| (fz[s] - bp[s]).abs < 1e-9 ? 0.0 : (fz[s] - h[s]) / (fz[s] - bp[s]) }
    recov[[r, a]] = abs
    puts "   %-16s %-12s %+10.3f %10.3f %6d" % [r, a, mean(abs), mean(frac), cs.size]
  end
end
puts

# ---- 6. provenance ----
puts "== LDFA PROVENANCE (one cell per rung x arm) =="
prov.keys.sort.each { |k| puts "   #{k[0]} #{k[1]}: #{prov[k]}" if prov[k] }
puts

# ---- 7. verdict ----
puts "== VERDICT =="
if contrast.empty?
  puts "   not enough paired cells yet"
else
  helped = contrast.count { |c| c[2] < 0 && c[2].abs > 2 * c[3] }
  hurt   = contrast.count { |c| c[2] > 0 && c[2].abs > 2 * c[3] }
  flat   = contrast.size - helped - hurt
  puts "   THE CONTRAST (oja - fix, outside 2 SEM): adaptive HELPED in #{helped} of"
  puts "   #{contrast.size} rung x r cells, HURT in #{hurt}, and was inside noise in #{flat}."
  puts
  # The flip criterion is stated on RECOVERY at the WIDE rungs, which is
  # what E2 was gated on: does adaptive low-rank keep recovery POSITIVE
  # where fixed-wide and fixed-lowrank go negative?
  wide_rungs = RUNGS.drop(1)
  puts "   THE FLIP CRITERION — recovery (abs bits vs frozen) at the WIDE rungs:"
  puts "   %-16s %-12s %10s" % ["rung", "arm", "abs bits"]
  flip = false
  wide_rungs.each do |r|
    w = recov[[r, "dfa-wide"]]
    next if w.nil?
    RANKS.each do |rk|
      o = recov[[r, "dfa-oja-#{rk}"]]
      f = recov[[r, "dfa-fix-#{rk}"]]
      next if o.nil? || f.nil?
      flip = true if mean(w) < 0 && mean(f) < 0 && mean(o) > 0
    end
  end
  RUNGS.each do |r|
    DFA_ARMS.each do |a|
      next unless recov[[r, a]]
      puts "   %-16s %-12s %+10.3f" % [r, a, mean(recov[[r, a]])]
    end
  end
  puts
  puts flip ?
    "   VERDICT: flip — adaptive low-rank holds recovery POSITIVE at a wide" \
    "\n   output where BOTH the full-width and the fixed low-rank arms go" \
    "\n   negative. That is the result E2 was built to look for." :
    "   VERDICT: no-flip — there is no rung x r where the wide and fixed-lowrank" \
    "\n   arms lose to a frozen body and the adaptive one does not. Whatever the" \
    "\n   adaptation buys (see THE CONTRAST above), it does not carry the arm" \
    "\n   across the frozen control at wide output, so the wide-output limit is" \
    "\n   deeper than feedback rank."
end
