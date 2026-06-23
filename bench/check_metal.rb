# bench/check_metal.rb — Metal (Apple GPU) perf leg. macOS-only.
#
# THE GAP this fills: toy has a Metal numeric *parity* gate (prep/metal_gate.rb)
# but no Metal *perf* bench — no baselines, no regression signal. This is the
# lightweight first leg: it times the ALREADY-BUILT metal/cpu infer runners
# (libexec/toy-infer-metal, libexec/toy-infer) and reports on-device decode
# throughput, plus the Metal-vs-CPU ratio on the same machine.
#
# METHOD — N-differencing (why it's honest): a single `toy infer` process pays
# a large fixed cost (process spawn + GGUF mmap + graph realize/compile) that
# dwarfs the per-token decode for a 135M model. We time the SAME runner at two
# token counts (N_LO, N_HI) and take (t_hi - t_lo)/(N_HI - N_LO): the fixed cost
# cancels, leaving a clean steady-state decode ms/token. best-of-REPS run to
# damp scheduler noise. (Cross-check: the CPU number this yields matches
# bench/check.rb's infer_step_ms to ~1%.)
#
# Usage:
#   ruby bench/check_metal.rb            # run + compare vs baselines_metal.csv
#   ruby bench/check_metal.rb --update   # run + (re)write the Mac baseline
#   ruby bench/check_metal.rb --report   # run + print, no gate
#
# Baseline file: bench/baselines_metal.csv (same schema as baselines.csv). The
# baseline is Mac-pinned (Apple GPUs differ per chip) — it is an A/B anchor for
# THIS machine, not a cross-machine truth, exactly like the metal_gate float
# baseline note. Commit the baseline from a known-good machine; treat a cross-
# machine compare as informational.
#
# SCOPE / next legs (deliberately NOT built here — see issue #104 part C):
#   * train wallclock: libexec/toy-train-metal runs but emits no timing; a real
#     train leg wants BENCH-line instrumentation INSIDE the runner (per-step ms),
#     not process-level wallclock (startup dominates a 5-step from-scratch run).
#   * per-op Metal timing already exists via tinynn's P6 sched eval callback
#     (Chrome-Trace) — wire that for a flame view when chasing a specific op.
#   * a heavier model (qwen2.5-1.5B) would show Metal WINNING; 135M decode is
#     dispatch-bound so CPU leads. Add when a larger native gguf is on the Mac.

require "csv"

ROOT = File.expand_path("../..", __FILE__)
BASELINES_PATH = File.expand_path("../baselines_metal.csv", __FILE__)

if RUBY_PLATFORM !~ /darwin/
  puts "SKIP [check_metal]: Metal is macOS-only (platform #{RUBY_PLATFORM})."
  exit 0
end

GGUF   = ENV["GGUF"]   || "data/smollm2-135m-f32.gguf"
N_LO   = (ENV["N_LO"]  || "8").to_i
N_HI   = (ENV["N_HI"]  || "72").to_i
REPS   = (ENV["REPS"]  || "3").to_i
PROMPT_IDS = ENV["PROMPT_IDS"] || "6403 1980 253 655 28"

INFER_METAL = File.join(ROOT, "libexec", "toy-infer-metal")
INFER_CPU   = File.join(ROOT, "libexec", "toy-infer")

# best-of-REPS wallclock (ms) of one runner at N tokens.
def time_ms(bin, n)
  best = nil
  REPS.times do
    t0 = Time.now
    system({ "GGUF" => GGUF, "PROMPT_IDS" => PROMPT_IDS, "N_NEW" => n.to_s },
           bin, out: File::NULL, err: File::NULL)
    dt = (Time.now - t0) * 1000.0
    best = dt if best.nil? || dt < best
  end
  best
end

# steady-state decode ms/token via N-differencing.
def decode_ms_per_tok(bin)
  lo = time_ms(bin, N_LO)
  hi = time_ms(bin, N_HI)
  (hi - lo) / (N_HI - N_LO).to_f
end

[["toy-infer-metal", INFER_METAL], ["toy-infer", INFER_CPU]].each do |name, bin|
  unless File.executable?(bin)
    warn "check_metal: missing #{name} (run `make libexec/#{name}`#{name.include?("metal") ? " — needs `make setup-ggml-metal` first" : ""})"
    exit 2
  end
end
unless File.file?(File.join(ROOT, GGUF))
  warn "check_metal: GGUF not found: #{GGUF}"
  exit 2
end

puts "== toy metal perf bench (N-diff #{N_LO}->#{N_HI}, best-of-#{REPS}, #{File.basename(GGUF)}) =="
metal_ms = decode_ms_per_tok(INFER_METAL)
cpu_ms   = decode_ms_per_tok(INFER_CPU)
speedup  = cpu_ms / metal_ms   # >1 => metal faster; <1 => cpu faster

observed = {
  "infer_metal_decode_ms_per_tok" => metal_ms,
  "infer_cpu_decode_ms_per_tok"   => cpu_ms,
  "metal_vs_cpu_decode_speedup"   => speedup,
}

# --- baseline I/O (mirror bench/check.rb) ---------------------------------
def infer_dir(m);  m.include?("speedup") ? "higher" : "lower"; end
def infer_unit(m); m.include?("speedup") ? "x" : "ms"; end

update_mode = ARGV.include?("--update")
report_only = ARGV.include?("--report")

baselines = {}
if File.exist?(BASELINES_PATH) && !update_mode
  CSV.foreach(BASELINES_PATH, headers: true) do |r|
    baselines[r["metric"]] = { value: r["value"].to_f, unit: r["unit"],
                               direction: r["direction"], tolerance_pct: r["tolerance_pct"].to_f }
  end
end

if update_mode
  rows = observed.map do |metric, value|
    e = baselines[metric] || {}
    { "metric" => metric, "value" => sprintf("%.6f", value),
      "unit" => e[:unit] || infer_unit(metric),
      "direction" => e[:direction] || infer_dir(metric),
      "tolerance_pct" => (e[:tolerance_pct] || 20.0).to_s }
  end
  CSV.open(BASELINES_PATH, "w", write_headers: true,
           headers: %w[metric value unit direction tolerance_pct]) do |csv|
    rows.each { |r| csv << r.values_at(*csv.headers) }
  end
  puts "wrote #{BASELINES_PATH} (#{rows.length} rows)"
  exit 0
end

def evaluate(metric, observed, baseline)
  return [sprintf("  %-34s = %10.4f  (no baseline — run --update)", metric, observed), true] if baseline.nil?
  b = baseline[:value]
  rel = b.zero? ? 0.0 : 100.0 * (observed - b) / b
  bad = baseline[:direction] == "lower" ? rel > baseline[:tolerance_pct] : -rel > baseline[:tolerance_pct]
  [sprintf("  %-34s = %10.4f %-2s (baseline %.4f, %+6.2f %%, tol ±%.1f %%)  [%s]",
           metric, observed, baseline[:unit], b, rel, baseline[:tolerance_pct], bad ? "REGRESS" : "ok"), !bad]
end

puts "results:"
regress = 0
observed.each do |metric, value|
  line, ok = evaluate(metric, value, baselines[metric])
  puts line
  regress += 1 unless ok
end

if report_only
  puts "(report mode — not gating)"; exit 0
end
if regress > 0
  puts "\nFAIL: #{regress} metric(s) regressed past tolerance"; exit 1
end
puts "\nok — metal perf within baselines"
exit 0
