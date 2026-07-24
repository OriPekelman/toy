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
puts failures.empty? ? "  ok: dfa,dfa — lane_b CE #{first_b&.round(3)} -> #{last_b&.round(3)}; #{align_count} well-formed align lines" : "  FAIL: dfa leg"

# ---- 4. byte-repro ----
r1 = run_franken("dfa,dfa", 10)
r2 = run_franken("dfa,dfa", 10)
if r1 == r2
  puts "  ok: byte-repro — two dfa runs identical"
else
  failures << "byte-repro: outputs differ"
end

if failures.empty?
  puts "GATE PASS [franken]: twin-parity + dfa-decreases + alignment + byte-repro (toy#109 P1)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [franken]: #{f}" }
  exit 1
end
