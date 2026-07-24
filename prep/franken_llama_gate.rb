#!/usr/bin/env ruby
# prep/franken_llama_gate.rb — toy#112 gates for the spec-callable franken
# runner (libexec/toy-train-franken-llama, `toy train franken`).
#
#   1. F0 BYTE-PARITY: empty policy, STEPS=5 SEED=0 — the stdout loss
#      curve must byte-equal prep/fixtures/train_baseline.txt (the same
#      fixture `toy train from-scratch` is gated on): the franken runner
#      with no policy IS the from-scratch trainer.
#   2. BUNDLE STRUCTURE (policy run, TAO_RUN_DIR): valid JSONL; run_start
#      first with schema toy/v1 + a `franken` object carrying the exact
#      policy/b axes; one step event per step; align events == 12 ×
#      steps when FRANKEN_ALIGN=1 (2 layers × [4q+4k+4v] on the dfa
#      layer... exactly the dfa-wired count) with finite cos in [-1,1]
#      and non-negative norms; run_end last; weights/step_<N>.gguf.
#   3. DFA EFFECT: the chain,dfa curve differs from the F0 curve.
#   4. BYTE-REPRO: two identical policy runs — identical stdout.

ROOT   = File.expand_path("..", __dir__)
RUNNER = File.join(ROOT, "libexec", "toy-train-franken-llama")
FIXTURE = File.join(ROOT, "prep", "fixtures", "train_baseline.txt")

require "open3"
require "json"
require "tmpdir"
require "fileutils"

def run_franken_llama(extra_env, run_dir)
  env = { "STEPS" => "5", "SEED" => "0" }.merge(extra_env)
  if run_dir
    FileUtils.mkdir_p(File.join(run_dir, "weights"))
    env = env.merge("TAO_RUN_DIR" => run_dir, "TOY_RUN_ID" => "franken-gate")
  end
  out, st = Open3.capture2e(env, RUNNER, chdir: ROOT)
  abort "franken_llama_gate: runner exited #{st.exitstatus}:\n#{out.lines.last(10).join}" unless st.success?
  out
end

unless File.executable?(RUNNER)
  build_out, build_st = Open3.capture2e("make", "-C", ROOT, "libexec/toy-train-franken-llama")
  unless build_st.success? && File.executable?(RUNNER)
    warn "franken_llama_gate: build failed:\n#{build_out.lines.last(15).join}"
    exit 2
  end
end

failures = []

# ---- 1. F0 byte-parity ----
f0_out = run_franken_llama({}, nil)
f0_curve = f0_out.lines.select { |l| l.start_with?("step ") }
expect = File.readlines(FIXTURE).reject { |l| l.start_with?("#") || l.strip.empty? }
if f0_curve == expect
  puts "  ok: F0 — empty-policy curve byte-equals train_baseline.txt (#{f0_curve.length} steps)"
else
  failures << "F0: curve != train_baseline.txt\ngot:  #{f0_curve.join}want: #{expect.join}"
end

# ---- 2 + 3. policy run: bundle structure + dfa effect ----
Dir.mktmpdir("franken_llama_gate") do |dir|
  pol_out = run_franken_llama({ "FRANKEN_POLICY" => "chain,dfa",
                                "FRANKEN_B_SEED" => "42",
                                "FRANKEN_ALIGN"  => "1" }, dir)
  pol_curve = pol_out.lines.select { |l| l.start_with?("step ") }
  failures << "dfa-effect: policy curve identical to F0" if pol_curve == f0_curve

  ev_path = File.join(dir, "events.jsonl")
  if File.file?(ev_path)
    events = File.readlines(ev_path).map { |l| JSON.parse(l) }
    failures << "bundle: first event not run_start" unless events.first && events.first["kind"] == "run_start"
    failures << "bundle: last event not run_end" unless events.last && events.last["kind"] == "run_end"
    rs = events.first || {}
    failures << "bundle: schema != toy/v1" unless rs["schema"] == "toy/v1"
    fr = rs["franken"]
    if fr.nil?
      failures << "bundle: run_start has no franken object"
    else
      failures << "bundle: franken.policy != [0,1] (got #{fr['policy'].inspect})" unless fr["policy"] == [0, 1]
      failures << "bundle: franken.b_seed != 42" unless fr["b_seed"] == 42
    end
    steps_ev = events.count { |e| e["kind"] == "step" }
    failures << "bundle: #{steps_ev} step events (want 5)" unless steps_ev == 5
    aligns = events.select { |e| e["kind"] == "align" }
    failures << "bundle: #{aligns.length} align events (want 60 = 12 dfa weights x 5 steps)" unless aligns.length == 60
    bad = aligns.count do |e|
      c = e["cos"]
      !c.is_a?(Numeric) || c.to_f.nan? || c.to_f.abs > 1.0001 ||
        !e["dfa_norm"].is_a?(Numeric) || e["dfa_norm"] < 0 ||
        !e["bp_norm"].is_a?(Numeric) || e["bp_norm"] < 0
    end
    failures << "bundle: #{bad} malformed align events" unless bad == 0
    # the engine-scale telemetry sanity: dfa signal is alive
    live = aligns.count { |e| e["dfa_norm"] > 0 }
    failures << "bundle: dfa_norm not positive anywhere" unless live == aligns.length
  else
    failures << "bundle: no events.jsonl"
  end
  ckpt = File.join(dir, "weights", "step_5.gguf")
  failures << "bundle: missing checkpoint #{ckpt}" unless File.file?(ckpt)
  puts "  ok: bundle — run_start(franken provenance) + 5 steps + 60 align + run_end + checkpoint; dfa curve differs" if failures.empty?
end

# ---- 4. byte-repro ----
r1 = run_franken_llama({ "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42" }, nil)
r2 = run_franken_llama({ "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42" }, nil)
failures << "byte-repro: outputs differ" unless r1 == r2
puts "  ok: byte-repro — two policy runs identical" if r1 == r2

if failures.empty?
  puts "GATE PASS [franken-llama]: F0 byte-parity + bundle/provenance/align + dfa-effect + byte-repro (toy#112)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [franken-llama]: #{f}" }
  exit 1
end
