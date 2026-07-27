#!/usr/bin/env ruby
# prep/franken_llama_cuda_gate.rb — toy#109 CUDA franken leg gates.
#
#   1. EMPTY-POLICY PARITY (dynamic, seeds 0 and 1): franken-cuda must
#      byte-equal toy-train-cuda — the same contract the CPU leg pins,
#      on this backend's own curves (CUDA is never compared to CPU).
#   2. DFA ARM: trains (decreases, finite), differs from empty-policy,
#      byte-deterministic run-to-run (the CUDA determinism contract).
#   3. BUNDLE: TAO_RUN_DIR events with franken provenance + align
#      events + the checkpoint (the CPU write-seam inside the CUDA unit).

ROOT   = File.expand_path("..", __dir__)
FRK    = File.join(ROOT, "libexec", "toy-train-franken-llama-cuda")
FS     = File.join(ROOT, "libexec", "toy-train-cuda")

require "open3"
require "json"
require "tmpdir"
require "fileutils"

def run_bin(bin, env, run_dir = nil)
  env = { "STEPS" => "5", "SEED" => "0" }.merge(env)
  if run_dir
    FileUtils.mkdir_p(File.join(run_dir, "weights"))
    env = env.merge("TAO_RUN_DIR" => run_dir, "TOY_RUN_ID" => "franken-cuda-gate")
  end
  out, st = Open3.capture2e(env, bin, chdir: ROOT)
  abort "franken_llama_cuda_gate: #{File.basename(bin)} exited #{st.exitstatus}:\n#{out.lines.last(8).join}" unless st.success?
  out
end

def curve(out)
  out.lines.select { |l| l.start_with?("step ") }
end

failures = []

# ---- 1. empty-policy parity, both seeds ----
[0, 1].each do |seed|
  fs = curve(run_bin(FS,  { "SEED" => seed.to_s }))
  fr = curve(run_bin(FRK, { "SEED" => seed.to_s }))
  if fs.length == 5 && fs == fr
    puts "  ok: empty-policy byte-equals toy-train-cuda (seed=#{seed})"
  else
    failures << "parity seed=#{seed}: curves differ\nfs: #{fs.join}fr: #{fr.join}"
  end
end

# ---- 2. dfa arm ----
d1 = run_bin(FRK, { "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42", "STEPS" => "8" })
d2 = run_bin(FRK, { "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42", "STEPS" => "8" })
dc = curve(d1)
losses = dc.map { |l| l[/loss=(\S+)/, 1].to_f }
failures << "dfa: NaN/short curve" unless losses.length == 8 && losses.none?(&:nan?)
failures << "dfa: did not decrease (#{losses.first} -> #{losses.last})" unless losses.last < losses.first - 0.05
failures << "dfa: byte-repro failed (CUDA determinism)" unless d1 == d2
ec = curve(run_bin(FRK, { "STEPS" => "8" }))
failures << "dfa: identical to empty policy" if dc == ec
puts failures.empty? ? "  ok: dfa arm trains (#{losses.first.round(3)} -> #{losses.last.round(3)}), deterministic, differs from chain" : "  FAIL: dfa arm"

# ---- 3. bundle ----
Dir.mktmpdir("franken_cuda_gate") do |dir|
  run_bin(FRK, { "FRANKEN_POLICY" => "chain,dfa", "FRANKEN_B_SEED" => "42",
                 "FRANKEN_ALIGN" => "1" }, dir)
  ev_path = File.join(dir, "events.jsonl")
  if File.file?(ev_path)
    events = File.readlines(ev_path).map { |l| JSON.parse(l) }
    rs = events.first || {}
    failures << "bundle: no franken provenance" unless rs["kind"] == "run_start" && rs["franken"].is_a?(Hash)
    failures << "bundle: wrong policy" unless rs.dig("franken", "policy") == [0, 1]
    aligns = events.count { |e| e["kind"] == "align" }
    failures << "bundle: #{aligns} align events (want 60)" unless aligns == 60
    failures << "bundle: no run_end" unless events.last["kind"] == "run_end"
  else
    failures << "bundle: no events.jsonl"
  end
  failures << "bundle: missing checkpoint" unless File.file?(File.join(dir, "weights", "step_5.gguf"))
  fj = File.join(dir, "flow.json")
  flow = File.file?(fj) ? (JSON.parse(File.read(fj)) rescue nil) : nil
  failures << "bundle: flow.json missing/invalid" if flow.nil? || flow["format"] != "toy/v1" || !flow["nodes"].is_a?(Array) || flow["nodes"].empty?
  puts failures.empty? ? "  ok: bundle — provenance + 60 align events + run_end + checkpoint (CPU write-seam)" : "  FAIL: bundle"
end

# ---- toy#124/#122: shape + corpus on the CUDA twin (determinism-scoped;
# CUDA is never compared to CPU) ----
w1 = run_bin(FRK, { "FRANKEN_SHAPE" => "wide", "CORPUS" => "data/ts_seqs.bin", "STEPS" => "6" })
w2 = run_bin(FRK, { "FRANKEN_SHAPE" => "wide", "CORPUS" => "data/ts_seqs.bin", "STEPS" => "6" })
failures << "shape-wide: cuda not deterministic" unless w1 == w2
wl = curve(w1).map { |l| l[/loss=(\S+)/, 1].to_f }
failures << "shape-wide: NaN/short" unless wl.length == 6 && wl.none?(&:nan?)
puts failures.empty? ? "  ok: --shape wide + --corpus on CUDA — deterministic, trains" : "  FAIL: cuda shape leg"

# ---- toy#126: lr/warmup on the CUDA twin (determinism-scoped) ----
l1 = run_bin(FRK, { "FRANKEN_LR" => "0.01", "FRANKEN_WARMUP" => "3", "STEPS" => "6" })
l2 = run_bin(FRK, { "FRANKEN_LR" => "0.01", "FRANKEN_WARMUP" => "3", "STEPS" => "6" })
failures << "lr/warmup: cuda not deterministic" unless l1 == l2
d6 = run_bin(FRK, { "STEPS" => "6" })
failures << "lr/warmup: curve identical to default (knob dead on cuda)" if curve(l1) == curve(d6)
puts failures.empty? ? "  ok: --lr/--warmup on CUDA — deterministic, moves the curve" : "  FAIL: cuda lr leg"

if failures.empty?
  puts "GATE PASS [franken-llama-cuda]: parity(seed 0+1) + dfa-arm + determinism + bundle + shape/corpus + lr/warmup (toy#109/#122/#124/#126 CUDA leg)"
  exit 0
else
  failures.each { |f| warn "GATE FAIL [franken-llama-cuda]: #{f}" }
  exit 1
end
