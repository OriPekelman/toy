#!/usr/bin/env ruby
# prep/franken_moe_gate.rb — toy#109 P2b gates for the Franken-MoE arm
# (libexec/toy-train-franken-moe). CRuby harness; the runner is Spinel.
#
#   1. TWIN-PARITY: FRANKEN_MOE=chain — lanes byte-identical per step.
#   2. DFA-EXPERTS TRAINS: 60 steps — lane B CE < 20% of start (the
#      BP-router + DFA-experts demonstration: expert updates are pure
#      forward ops; no autodiff through expert internals is required),
#      and differs from lane A (policy has effect).
#   3. ROUTER HEALTH: g0_mean stays in (0.05, 0.95) every step (soft
#      router doesn't collapse while experts learn by DFA).
#   4. ALIGNMENT WELL-FORMED: 4 lines/step, cos finite in [-1, 1].
#   5. BYTE-REPRO: two identical invocations, identical stdout.
#   6. TOP1 + AUX (FRANKEN_MOE_AUX load-balancing): no-aux control
#      collapses to one expert (the documented DFA-router finding);
#      aux=0.05 restores a per-expert utilization floor (temporal
#      alternation at this shape), trains, byte-repro, auxbal lines
#      well-formed.

ROOT   = File.expand_path("..", __dir__)
RUNNER = File.join(ROOT, "libexec", "toy-train-franken-moe")

require "open3"

def run_moe(mode, steps)
  env = { "FRANKEN_MOE" => mode, "STEPS" => steps.to_s, "FRANKEN_SEED" => "1234" }
  out, st = Open3.capture2e(env, RUNNER, chdir: ROOT)
  abort "franken_moe_gate: runner exited #{st.exitstatus} (mode=#{mode}):\n#{out.lines.last(10).join}" unless st.success?
  out
end

unless File.executable?(RUNNER)
  build_out, build_st = Open3.capture2e("make", "-C", ROOT, "libexec/toy-train-franken-moe")
  unless build_st.success? && File.executable?(RUNNER)
    warn "franken_moe_gate: build failed:\n#{build_out.lines.last(15).join}"
    exit 2
  end
end

failures = []

# ---- 1. twin parity ----
out = run_moe("chain", 8)
steps_seen = 0
out.each_line do |line|
  next unless line.start_with?("step ")
  steps_seen += 1
  m = line.match(/lane_a=(\S+) lane_b=(\S+)/)
  failures << "twin-parity diverged: #{line.strip}" if m.nil? || m[1] != m[2]
end
failures << "twin-parity: no step lines" if steps_seen == 0
puts failures.empty? ? "  ok: twin-parity — #{steps_seen} steps byte-identical (chain)" : "  FAIL: twin-parity"

# ---- 2-4. dfa-experts ----
out = run_moe("dfa-experts", 60)
first_b = nil; last_b = nil; first_a = nil; last_a = nil
align_count = 0; align_bad = 0; router_bad = 0; router_count = 0
out.each_line do |line|
  if (m = line.match(/^step \d+: lane_a=(\S+) lane_b=(\S+)/))
    a = Float(m[1]); b = Float(m[2])
    first_a ||= a; first_b ||= b
    last_a = a; last_b = b
  elsif (m = line.match(/^align step=\d+ w=\S+ cos=(\S+)/))
    align_count += 1
    c = Float(m[1]) rescue nil
    align_bad += 1 if c.nil? || c.nan? || c.abs > 1.0001
  elsif (m = line.match(/^router step=\d+ g0_mean=(\S+)/))
    router_count += 1
    g = Float(m[1]) rescue nil
    router_bad += 1 if g.nil? || g.nan? || g < 0.05 || g > 0.95
  end
end
if first_b.nil?
  failures << "dfa-experts: no step lines"
else
  failures << "dfa-experts: lane_b #{first_b} -> #{last_b} (want < #{(first_b * 0.2).round(3)})" unless last_b < first_b * 0.2
  failures << "bp-sanity: lane_a #{first_a} -> #{last_a}" unless last_a < first_a * 0.1
  failures << "dfa-experts: NaN" if last_b != last_b
  failures << "dfa-experts: identical to chain (no effect)" if last_b == last_a
end
failures << "alignment: #{align_count} lines (want 240)" unless align_count == 240
failures << "alignment: #{align_bad} malformed" unless align_bad == 0
failures << "router: #{router_bad}/#{router_count} unhealthy g0_mean" unless router_bad == 0 && router_count == 60
puts failures.empty? ? "  ok: dfa-experts — lane_b CE #{first_b&.round(3)} -> #{last_b&.round(4)} (BP lane -> #{last_a&.round(4)}); router healthy" : "  FAIL: dfa-experts leg"

# ---- top1 hard-routing legs (toy#109 hard-routed MoE) ----
out = run_moe("dfa-experts", 40)  # dense reference for comparison
t1a = Open3.capture2e({ "FRANKEN_MOE" => "dfa-experts", "FRANKEN_MOE_ROUTING" => "top1",
                        "STEPS" => "40", "FRANKEN_SEED" => "1234" }, RUNNER, chdir: ROOT)
abort "franken_moe_gate: top1 run failed:\n#{t1a[0].lines.last(8).join}" unless t1a[1].success?
t1b = Open3.capture2e({ "FRANKEN_MOE" => "dfa-experts", "FRANKEN_MOE_ROUTING" => "top1",
                        "STEPS" => "40", "FRANKEN_SEED" => "1234" }, RUNNER, chdir: ROOT)
failures << "top1: byte-repro failed" unless t1a[0] == t1b[0]
t1_losses = t1a[0].lines.select { |l| l.start_with?("step ") }
                  .map { |l| l[/lane_b=(\S+)/, 1].to_f }
failures << "top1: NaN" if t1_losses.any?(&:nan?)
failures << "top1: did not train (#{t1_losses.first} -> #{t1_losses.last})" unless t1_losses.last < t1_losses.first - 0.1
routes = t1a[0].lines.select { |l| l.start_with?("route ") }
               .map { |l| l[/e0_share=(\S+)/, 1].to_f }
failures << "top1: no route telemetry" if routes.empty?
failures << "top1: malformed shares" unless routes.all? { |r| r >= 0.0 && r <= 1.0 }
puts failures.empty? ? "  ok: top1 — trains (#{t1_losses.first.round(3)} -> #{t1_losses.last.round(3)}), deterministic, #{routes.length} route lines (final e0_share=#{routes.last})" : "  FAIL: top1 leg"

# ---- top1 load-balancing aux legs (the router-collapse fix) ----
# Control contrast: WITHOUT aux the DFA-fed router starves one expert
# permanently (routes pinned to a single expert over the tail). With
# FRANKEN_MOE_AUX the starvation breaks: at this rig's shape the 4
# tokens' router logits move together, so balance is TEMPORAL (the
# router alternates which expert serves the whole batch) — the
# contract is a utilization floor per expert, not per-batch mixing.
tail_routes = routes.last(15)
failures << "aux-control: no-aux router did not collapse (tail #{tail_routes.uniq.inspect}) — the aux rationale needs re-examining" unless tail_routes.uniq.length == 1 && (tail_routes.first == 0.0 || tail_routes.first == 1.0)
aux_env = { "FRANKEN_MOE" => "dfa-experts", "FRANKEN_MOE_ROUTING" => "top1",
            "FRANKEN_MOE_AUX" => "0.05", "STEPS" => "120", "FRANKEN_SEED" => "1234" }
x1 = Open3.capture2e(aux_env, RUNNER, chdir: ROOT)
abort "franken_moe_gate: aux run failed:\n#{x1[0].lines.last(8).join}" unless x1[1].success?
x2 = Open3.capture2e(aux_env, RUNNER, chdir: ROOT)
failures << "aux: byte-repro failed" unless x1[0] == x2[0]
ax_losses = x1[0].lines.select { |l| l.start_with?("step ") }
                 .map { |l| l[/lane_b=(\S+)/, 1].to_f }
failures << "aux: NaN" if ax_losses.any?(&:nan?)
failures << "aux: did not train (#{ax_losses.first} -> #{ax_losses.last})" unless ax_losses.last < ax_losses.first - 0.1
ax_routes = x1[0].lines.select { |l| l.start_with?("route ") }
                 .map { |l| l[/e0_share=(\S+)/, 1].to_f }
tail = ax_routes.drop(20)
e0_maj = tail.count { |r| r > 0.5 }
e1_maj = tail.count { |r| r < 0.5 }
floor = (tail.length * 0.2).floor
failures << "aux: expert starvation persists (e0-majority #{e0_maj}, e1-majority #{e1_maj} of #{tail.length}; floor #{floor})" unless e0_maj >= floor && e1_maj >= floor
auxbals = x1[0].lines.select { |l| l.start_with?("auxbal ") }
                .map { |l| l[/loss=(\S+)/, 1].to_f }
failures << "aux: #{auxbals.length} auxbal lines (want 120)" unless auxbals.length == 120
failures << "aux: malformed auxbal values" unless auxbals.all? { |v| v.finite? && v >= 0.0 }
puts failures.empty? ? "  ok: aux — no-aux control collapses; aux=0.05 breaks starvation (e0/e1 step-majorities #{e0_maj}/#{e1_maj} of #{tail.length}), trains (#{ax_losses.first.round(3)} -> #{ax_losses.last.round(3)}), deterministic, 120 auxbal lines" : "  FAIL: aux leg"

# ---- 5. byte-repro ----
r1 = run_moe("dfa-experts", 10)
r2 = run_moe("dfa-experts", 10)
failures << "byte-repro: outputs differ" unless r1 == r2
puts r1 == r2 ? "  ok: byte-repro — two dfa-experts runs identical" : "  FAIL: byte-repro"

if failures.empty?
  puts "GATE PASS [franken-moe]: parity + dfa-experts + top1-hard-routing + aux-load-balancing + router-health + alignment + byte-repro (toy#109 P2b + hard-routed + aux legs)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [franken-moe]: #{f}" }
  exit 1
end
