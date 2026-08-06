#!/usr/bin/env ruby
# prep/franken_gate.rb — toy#109 P1 gates for the FrankenModel twin-lane
# runner (libexec/toy-train-franken). CRuby-only harness; the runner is the
# Spinel binary.
#
#   1. TWIN-PARITY (null hypothesis): FRANKEN_POLICY=chain,chain — lane B
#      must equal lane A byte-for-byte on every step line.
#   2. DFA-DECREASES: FRANKEN_POLICY=dfa,dfa, 60 steps — lane B CE must
#      drop below half its start (and lane A must still behave as BP).
#   3. ALIGNMENT WELL-FORMED: 8 align lines per step (2 layers × q/k/v/o),
#      every cos finite and within [-1, 1]. (No positivity assertion —
#      whether matmul-granularity DFA aligns is the EXPERIMENT, not a gate.)
#   4. BYTE-REPRO: two identical invocations produce identical stdout.
#
#   ruby prep/franken_gate.rb   # exit 0 all pass, 1 fail, 2 setup

ROOT   = File.expand_path("..", __dir__)
RUNNER = File.join(ROOT, "libexec", "toy-train-franken")

require "open3"

def run_franken(policy, steps)
  env = { "FRANKEN_POLICY" => policy, "STEPS" => steps.to_s, "FRANKEN_SEED" => "1234" }
  out, st = Open3.capture2e(env, RUNNER, chdir: ROOT)
  abort "franken_gate: runner exited #{st.exitstatus} (policy=#{policy}):\n#{out.lines.last(10).join}" unless st.success?
  out
end

unless File.executable?(RUNNER)
  build_out, build_st = Open3.capture2e("make", "-C", ROOT, "libexec/toy-train-franken")
  unless build_st.success? && File.executable?(RUNNER)
    warn "franken_gate: build failed:\n#{build_out.lines.last(15).join}"
    exit 2
  end
end

failures = []
# LEG BOOKKEEPING: every leg records the failure count at its START in
# `n0` and summarises with `failures.length == n0`, so each leg reports
# on ITS OWN assertions. Legs used to summarise with `failures.empty?`,
# which made every later leg print FAIL once ANY earlier leg had failed
# — misleading exactly when you are debugging. `n0` is seeded at top
# level so re-assignments inside blocks mutate the outer local.
n0 = 0
n0 = failures.length

# ---- 1. twin parity ----
out = run_franken("chain,chain", 8)
steps_seen = 0
out.each_line do |line|
  next unless line.start_with?("step ")
  steps_seen += 1
  m = line.match(/lane_a=(\S+) lane_b=(\S+)/)
  if m.nil? || m[1] != m[2]
    failures << "twin-parity: lanes diverged: #{line.strip}"
    break
  end
end
failures << "twin-parity: no step lines" if steps_seen == 0
puts steps_seen > 0 && failures.empty? ? "  ok: twin-parity — #{steps_seen} steps byte-identical (chain,chain)" : "  FAIL: twin-parity"

# ---- 2 + 3. dfa decreases + alignment well-formed ----
out = run_franken("dfa,dfa", 60)
first_b = nil; last_b = nil; first_a = nil; last_a = nil
align_count = 0; align_bad = 0
out.each_line do |line|
  if (m = line.match(/^step \d+: lane_a=(\S+) lane_b=(\S+)/))
    a = Float(m[1]); b = Float(m[2])
    first_a ||= a; first_b ||= b
    last_a = a; last_b = b
  elsif (m = line.match(/^align step=\d+ li=\d+ w=[qkvo] cos=(\S+)/))
    align_count += 1
    c = Float(m[1]) rescue nil
    align_bad += 1 if c.nil? || c.nan? || c.abs > 1.0001
  end
end
if first_b.nil?
  failures << "dfa: no step lines"
else
  failures << "dfa-decreases: lane_b #{first_b} -> #{last_b}" unless last_b < first_b * 0.5
  failures << "bp-sanity: lane_a #{first_a} -> #{last_a}" unless last_a < first_a * 0.2
  failures << "dfa: NaN" if last_b != last_b
end
expected_aligns = 60 * 8
failures << "alignment: #{align_count} lines (want #{expected_aligns})" unless align_count == expected_aligns
failures << "alignment: #{align_bad} malformed cos values" unless align_bad == 0
puts failures.length == n0 ? "  ok: dfa,dfa — lane_b CE #{first_b&.round(3)} -> #{last_b&.round(3)}; #{align_count} well-formed align lines" : "  FAIL: dfa leg"

# ---- P3 combiner legs ----
# mix(1.0) must byte-equal pure chain (lane parity), the strong null.
out = run_franken("mix:1.0,mix:1.0", 6)
mix_null_ok = true
out.each_line do |line|
  next unless line.start_with?("step ")
  m = line.match(/lane_a=(\S+) lane_b=(\S+)/)
  mix_null_ok = false if m.nil? || m[1] != m[2]
end
failures << "mix(1.0) != chain" unless mix_null_ok
puts mix_null_ok ? "  ok: mix(1.0) byte-equals chain" : "  FAIL: mix(1.0)"

# maskdfa(-1) must byte-equal pure dfa (saturated mask == exactly 1.0).
ref = run_franken("dfa,dfa", 6).lines.select { |l| l.start_with?("step ") }
msk = run_franken("maskdfa:-1,maskdfa:-1", 6)
msk_steps = msk.lines.select { |l| l.start_with?("step ") }
mask_null_ok = (ref == msk_steps)
dens_bad = msk.lines.count { |l| l.start_with?("mask ") && !l.include?("density=1.0") }
failures << "maskdfa(-1) != dfa" unless mask_null_ok
failures << "maskdfa(-1): #{dens_bad} non-saturated densities" unless dens_bad == 0
puts mask_null_ok && dens_bad == 0 ? "  ok: maskdfa(-1) byte-equals dfa, densities saturated" : "  FAIL: mask null"

# mid-range: mix(0.5) and maskbp train; maskbp densities strictly in (0,1).
out = run_franken("mix:0.5,mix:0.5", 40)
mixc = out.lines.select { |l| l.start_with?("step ") }
mfirst = Float(mixc.first.match(/lane_b=(\S+)/)[1]); mlast = Float(mixc.last.match(/lane_b=(\S+)/)[1])
failures << "mix(0.5): #{mfirst} -> #{mlast}" unless mlast < mfirst * 0.2
out = run_franken("maskbp:0.0005,maskbp:0.0005", 40)
mbc = out.lines.select { |l| l.start_with?("step ") }
bfirst = Float(mbc.first.match(/lane_b=(\S+)/)[1]); blast = Float(mbc.last.match(/lane_b=(\S+)/)[1])
failures << "maskbp: #{bfirst} -> #{blast}" unless blast < bfirst * 0.2
dvals = out.lines.select { |l| l.start_with?("mask ") }.map { |l| Float(l.match(/density=(\S+)/)[1]) }
failures << "maskbp: no density lines" if dvals.empty?
failures << "maskbp: densities not strictly interior" unless dvals.all? { |d| d > 0.0 && d < 1.0 }
puts "  ok: mix(0.5) -> #{mlast.round(3)}; maskbp -> #{blast.round(4)} (densities interior: #{dvals.empty? ? 'n/a' : dvals.minmax.map { |x| x.round(3) }.join('..')})" if failures.empty?

# ---- 4. byte-repro ----
r1 = run_franken("dfa,dfa", 10)
r2 = run_franken("dfa,dfa", 10)
if r1 == r2
  puts "  ok: byte-repro — two dfa runs identical"
else
  failures << "byte-repro: outputs differ"
end

if failures.empty?
  puts "GATE PASS [franken]: twin-parity + dfa + mix/mask combiners + alignment + byte-repro (toy#109 P1+P3)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [franken]: #{f}" }
  exit 1
end
