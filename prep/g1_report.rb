#!/usr/bin/env ruby
# prep/g1_report.rb — toy#173 (G1): read the AST-node-type arms.
#
# THE THREE READS TAO ASKED FOR, in order:
#   1. the structural node-type dimension (the head width) WITH the raw
#      leaf vocabulary beside it — the leaf count is the quantity that
#      decides `no-go-dim`, so it is never reported alone;
#   2. dfa-structure vs bp vs frozen, per-arm LR, with sd — does F17's
#      Cora result (dfa .736 > bp .672) transfer from a homophilous
#      citation graph to a heterophilous code tree;
#   3. leaf separability — handled by prep/g1_leafsep.sh, not here.
#
# The gate on every comparison is `bp > frozen`, checked PER SEED. On a
# tree that is not free: structure is informative and a frozen random GNN
# is already a neighbourhood aggregator, so a control that cannot lose
# would make every credit number uninterpretable (the trap that took
# three builds to clear on the gtx relational task).
require "json"

DIR  = ENV["DIR"] || "/srv/data/scratch/g1"
PACK = ENV["PACK"] || "data/ast_code"
ARMS = %w[bp dfa-structure dfa-direct frozen]

def mean(a) = a.sum / a.size.to_f
def sd(a)
  return 0.0 if a.size < 2
  m = mean(a)
  Math.sqrt(a.sum { |x| (x - m)**2 } / (a.size - 1).to_f)
end
def welch_t(a, b)
  va = sd(a)**2 / a.size
  vb = sd(b)**2 / b.size
  return 0.0 if va + vb <= 0
  (mean(a) - mean(b)) / Math.sqrt(va + vb)
end

rows = {}
ARMS.each do |arm|
  Dir[File.join(DIR, "#{arm}_lr*_s*.txt")].each do |f|
    m = File.basename(f).match(/_lr([0-9.]+)_s(\d+)\.txt\z/)
    next unless m
    acc = File.read(f)[/acc=([0-9.]+)(?!.*acc=)/m, 1]
    acc ||= File.read(f).scan(/acc=([0-9.]+)/).flatten.last
    next unless acc
    (rows[arm] ||= {})[[m[1], m[2].to_i]] = acc.to_f
  end
end
abort "g1_report: no cells under #{DIR}" if rows.empty?

# Best LR from SEED 0 only. Picking it per-seed would select the max of
# several noisy draws and bias every arm upward by a different amount.
best = {}
ARMS.each do |arm|
  next unless rows[arm]
  lrs = rows[arm].keys.map(&:first).uniq
  s0 = lrs.map { |lr| [lr, rows[arm][[lr, 0]]] }.reject { |_, v| v.nil? }
  next if s0.empty?
  b = s0.max_by { |_, v| v }
  best[arm] = { lr: b[0], grid: lrs.sort_by(&:to_f),
                seeds: rows[arm].select { |(lr, _), _| lr == b[0] }
                                .map { |(_, s), v| [s, v] }.to_h }
end

meta = File.file?(PACK + ".json") ? JSON.parse(File.read(PACK + ".json")) : {}
puts "G1 — masked AST-node-type prediction on the GNN lane (F17 engine, unchanged)"
puts
if meta.any?
  st = meta["structural_node_types"]
  rl = meta["raw_leaf_vocab_total"]
  puts "== 1. THE DIMENSION (the headline), with the leaf vocab beside it =="
  puts "   structural node types (head width) : #{st}"
  puts "   raw leaf vocabulary                : #{rl}   #{meta['raw_leaf_vocab'].inspect}"
  puts "   ratio raw_leaf / structural        : #{(rl / st.to_f).round(1)}x"
  puts "   corpus: #{meta['n_functions']} functions, #{meta['n_nodes']} nodes, #{meta['n_edges']} edges"
  puts "   provenance: python #{meta['python']}, #{meta['modules_found']} pinned modules, sha #{meta['sources_sha256'][0, 12]}"
  puts "   NOTE: a head width under 256 is the `go` condition on DIMENSION only."
  puts "   The leaf vocabulary is the open-vocab risk and is reported with it, always."
  puts
end

puts "== 2. THE ARMS (held-out node-type accuracy) =="
puts "   arm             lr        n   mean acc   sd       seeds"
ARMS.each do |arm|
  b = best[arm]
  next unless b
  v = b[:seeds].keys.sort.map { |s| b[:seeds][s] }
  edge = (b[:lr] == b[:grid].first || b[:lr] == b[:grid].last) ? "  EDGE-OPTIMUM" : ""
  puts "   %-15s %-8s %2d   %.4f     %.4f   %s%s" %
       [arm, b[:lr], v.size, mean(v), sd(v), b[:seeds].keys.sort.inspect, edge]
end

bp = best["bp"] && best["bp"][:seeds]
fz = best["frozen"] && best["frozen"][:seeds]
if bp && fz
  common = (bp.keys & fz.keys).sort
  wins = common.count { |s| bp[s] > fz[s] }
  puts
  puts "== THE PRECONDITION: can the control lose? =="
  PRECOND_OK = (wins == common.size)
  puts "   bp > frozen on #{wins}/#{common.size} seeds" +
       (PRECOND_OK ? "" : "   ** FAILS — the cell does not discriminate and no credit number below is interpretable **")
  puts "   margin: %.4f (bp %.4f vs frozen %.4f), chance = %.4f" %
       [mean(common.map { |s| bp[s] }) - mean(common.map { |s| fz[s] }),
        mean(common.map { |s| bp[s] }), mean(common.map { |s| fz[s] }),
        meta["structural_node_types"] ? 1.0 / meta["structural_node_types"] : 0.0]
end

puts
puts "== 3. DOES F17'S GRAPH RESULT TRANSFER? =="
puts "   On Cora (homophilous): dfa-structure .736 BEAT bp .672."
puts "   Here the graph is a heterophilous code tree."
%w[dfa-structure dfa-direct].each do |arm|
  next unless best[arm] && best["bp"] && best["frozen"]
  d = best[arm][:seeds]
  common = (d.keys & bp.keys & fz.keys).sort
  next if common.empty?
  dv = common.map { |s| d[s] }
  bv = common.map { |s| bp[s] }
  fv = common.map { |s| fz[s] }
  t = welch_t(dv, bv)
  # A transfer verdict is MEANINGLESS when the control ties bp: an arm
  # "beating bp" on a cell where a FROZEN body also matches bp is not
  # evidence about credit assignment, it is evidence the body is not
  # doing the work. The script printed "BEATS bp (transfers)" at
  # |t|=2.85 on exactly such a cell — a false finding one line below its
  # own failing precondition. Suppressed structurally rather than left
  # to the reader.
  verdict = if !PRECOND_OK then "UNINTERPRETABLE (precondition failed — frozen ties bp)"
            elsif mean(dv) > mean(bv) && t.abs >= 2.0 then "BEATS bp (transfers)"
            elsif t.abs < 2.0 then "indistinguishable from bp"
            else "LOSES to bp (does not transfer)"
            end
  puts "   %-15s %.4f vs bp %.4f (Welch |t|=%.2f) — %s; vs frozen %+.4f" %
       [arm, mean(dv), mean(bv), t.abs, verdict, mean(dv) - mean(fv)]
end
