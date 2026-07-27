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
  # toy#112 gap closed: flow.json (the toy#25 self-describing bundle)
  fj = File.join(dir, "flow.json")
  if File.file?(fj)
    flow = JSON.parse(File.read(fj)) rescue nil
    if flow.nil? || flow["format"] != "toy/v1" || !flow["nodes"].is_a?(Array) || flow["nodes"].empty?
      failures << "bundle: flow.json invalid (format/nodes)"
    end
  else
    failures << "bundle: no flow.json"
  end
  puts "  ok: bundle — run_start(franken provenance) + 5 steps + 60 align + run_end + checkpoint; dfa curve differs" if failures.empty?
end

# ---- seed!=0 parity (toy#113): franken empty-policy must equal
# toy-train AT THE SAME NONZERO SEED, compared DYNAMICALLY (no fixture:
# seed!=0 curves are toy-version-scoped; only seed=0 is frozen). This
# tripwires the whole-program numeric-stream divergence class — at
# f7cea71 the two units compiled DIFFERENT xorshift/Box-Muller streams
# from identical source (seed=0 masked it).
FS_RUNNER = File.join(ROOT, "libexec", "toy-train")
fs_out, fs_st = Open3.capture2e({ "STEPS" => "8", "SEED" => "1" }, FS_RUNNER, chdir: ROOT)
fr_out = run_franken_llama({ "STEPS" => "8", "SEED" => "1" }, nil)
fs_curve = fs_out.lines.select { |l| l.start_with?("step ") }
fr_curve = fr_out.lines.select { |l| l.start_with?("step ") }
if fs_st.success? && fs_curve.length == 8 && fs_curve == fr_curve
  puts "  ok: seed=1 parity — franken empty-policy byte-equals toy-train (8 steps, dynamic)"
else
  failures << "seed1-parity: curves differ or toy-train failed\nfs: #{fs_curve.join}fr: #{fr_curve.join}"
end

# ---- TOY_RUN_ID passthrough (toy#115): the CLI's controlled env must
# forward a caller-supplied run id into run_start.run_id (the tao#flow
# contract) instead of substituting the internal counter.
Dir.mktmpdir("franken_rid_gate") do |dir|
  _, st = Open3.capture2e({ "TOY_RUN_ID" => "gate/rid/check" },
                          File.join(ROOT, "bin", "toy"), "train", "franken",
                          "--steps", "1", "--seed", "0", "--out", dir, chdir: ROOT)
  rid = st.success? ? (JSON.parse(File.readlines(File.join(dir, "events.jsonl")).first)["run_id"] rescue nil) : nil
  if rid == "gate/rid/check"
    puts "  ok: TOY_RUN_ID passthrough (CLI controlled env honors caller id)"
  else
    failures << "run-id: TOY_RUN_ID not honored (got #{rid.inspect})"
  end
end

# ---- llama-shape maskbp byte-null (toy#117): maskbp:-1 (gate saturated
# to exactly 1.0) must byte-equal toy-train — the leg that would have
# caught the B-buffer scratch-reuse explosion (rig shape provably
# doesn't cover it; B leaves are per-step re-uploads now).
fs8 = Open3.capture2e({ "STEPS" => "8", "SEED" => "0" }, FS_RUNNER, chdir: ROOT)[0]
mb8 = run_franken_llama({ "FRANKEN_POLICY" => "maskbp:-1,maskbp:-1",
                          "FRANKEN_B_SEED" => "42", "STEPS" => "8" }, nil)
fs8c = fs8.lines.select { |l| l.start_with?("step ") }
mb8c = mb8.lines.select { |l| l.start_with?("step ") }
if fs8c.length == 8 && fs8c == mb8c
  puts "  ok: maskbp(-1) byte-equals toy-train at the llama shape (toy#117 pin)"
else
  failures << "maskbp-null: curves differ\nfs: #{fs8c.join}mb: #{mb8c.join}"
end

# ---- toy#122: F6 long-horizon surface — corpus stream + align thinning ----
# corpus arm: deterministic, actually streams (differs from the fixed-seq
# feed), and leaves the byte-gated default path untouched (leg 1 pins it).
c1 = run_franken_llama({ "STEPS" => "8", "CORPUS" => "data/ts_seqs.bin" }, nil)
c2 = run_franken_llama({ "STEPS" => "8", "CORPUS" => "data/ts_seqs.bin" }, nil)
c_curve = c1.lines.select { |l| l.start_with?("step ") }
failures << "corpus: not deterministic" unless c1 == c2
failures << "corpus: only #{c_curve.length} steps" unless c_curve.length == 8
d_curve = run_franken_llama({ "STEPS" => "8" }, nil).lines.select { |l| l.start_with?("step ") }
failures << "corpus: curve identical to fixed-seq feed (stream not live)" if c_curve == d_curve
puts failures.empty? ? "  ok: --corpus streams (deterministic, differs from fixed-seq; default feed untouched)" : "  FAIL: corpus leg"

# align-every thinning: N=3 over 6 steps -> emissions at steps 1,4 ->
# 12 weights x 2 = 24 align events; N=1 == legacy per-step (72).
Dir.mktmpdir("franken_ae_gate") do |dir|
  run_franken_llama({ "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42",
                      "FRANKEN_ALIGN" => "1", "FRANKEN_ALIGN_EVERY" => "3",
                      "STEPS" => "6" }, dir)
  evs = File.readlines(File.join(dir, "events.jsonl")).map { |l| JSON.parse(l) }
  aligns = evs.select { |e| e["kind"] == "align" }
  steps_seen = aligns.map { |e| e["step"] }.uniq.sort
  failures << "align-every: #{aligns.length} events (want 24)" unless aligns.length == 24
  failures << "align-every: wrong steps #{steps_seen.inspect} (want [1,4])" unless steps_seen == [1, 4]
  failures << "align-every: step events thinned too (#{evs.count { |e| e['kind'] == 'step' }})" unless evs.count { |e| e["kind"] == "step" } == 6
end
puts failures.empty? ? "  ok: --align-every thins align emissions (24 @ N=3/6 steps; step events untouched)" : "  FAIL: align-every leg"

# ---- toy#124: shape presets, byte-pinned per preset ----
%w[wide deep].each do |sh|
  fx = File.join(ROOT, "prep", "fixtures", "franken_#{sh}_baseline.txt")
  exp = File.readlines(fx).reject { |l| l.start_with?("#") || l.strip.empty? }.map(&:chomp)
  got = run_franken_llama({ "FRANKEN_SHAPE" => sh, "STEPS" => "5" }, nil)
        .lines.select { |l| l.start_with?("step ") }.map(&:chomp)
  if got == exp
    puts "  ok: --shape #{sh} byte-equals its recorded baseline (5 steps)"
  else
    failures << "shape-#{sh}: curve != #{File.basename(fx)}\ngot:  #{got.join(' | ')}\nwant: #{exp.join(' | ')}"
  end
  s1 = run_franken_llama({ "FRANKEN_SHAPE" => sh, "STEPS" => "5", "SEED" => "1" }, nil)
  s2 = run_franken_llama({ "FRANKEN_SHAPE" => sh, "STEPS" => "5", "SEED" => "1" }, nil)
  failures << "shape-#{sh}: seed=1 not deterministic" unless s1 == s2
end

# ---- 4. byte-repro ----
r1 = run_franken_llama({ "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42" }, nil)
r2 = run_franken_llama({ "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42" }, nil)
failures << "byte-repro: outputs differ" unless r1 == r2
puts "  ok: byte-repro — two policy runs identical" if r1 == r2

if failures.empty?
  puts "GATE PASS [franken-llama]: F0 byte-parity + seed!=0 parity + bundle/provenance/align + dfa-effect + corpus/align-every (toy#122) + shape presets (toy#124) + byte-repro (toy#112/#113)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [franken-llama]: #{f}" }
  exit 1
end
