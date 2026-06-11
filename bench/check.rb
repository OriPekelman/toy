# bench/check.rb — run each bench binary, parse the BENCH lines,
# compare against bench/baselines.csv. Print a one-screen report and
# exit non-zero on any regression past its per-metric tolerance.
#
# Usage:
#   ruby bench/check.rb                  # run + compare to baselines
#   ruby bench/check.rb --update         # run + write new baselines
#   ruby bench/check.rb --report         # run + just print, no compare
#
# Baseline file is intentionally CSV (one row per metric) so it
# diffs cleanly in git. Each row: metric,value,unit,direction,
# tolerance_pct. `direction` is `lower` (lower is better — latency)
# or `higher` (higher is better — throughput). `tolerance_pct` is
# what fraction worse than baseline counts as a regression (10 means
# ±10% noise is acceptable; >10% in the bad direction fails).

require "csv"

BASELINES_PATH = File.expand_path("../baselines.csv", __FILE__)
ROOT = File.expand_path("../..", __FILE__)

# Each bench has a Ruby source and a built binary; the binary path
# is relative to ROOT. Env vars apply only to that bench's invocation.
BENCHES = [
  # B=1 — legacy single-sequence LoRA step. Emits both the b1-suffixed
  # metric AND the unsuffixed legacy name (baseline-row compatibility).
  { binary: "bench/build/lora_step",   env: { "STEPS" => "8", "BATCH" => "1" }, source: "bench/lora_step.rb" },
  # B=4 — micro-batched LoRA step (GH#7). Tracks how step time scales
  # with effective batch size; reports `lora_step_b4_ms` only.
  { binary: "bench/build/lora_step",   env: { "STEPS" => "8", "BATCH" => "4" }, source: "bench/lora_step.rb" },
  { binary: "bench/build/inference",   env: { "N_NEW" => "32" }, source: "bench/inference.rb" },
  { binary: "bench/build/tokenizer",   env: {},                  source: "bench/tokenizer.rb" },
]

def infer_direction(metric)
  return "higher" if metric.include?("_per_sec") || metric.include?("toks_per")
  "lower"
end

def infer_unit(metric)
  return "tok/s" if metric.include?("toks_per_sec")
  return "us/tok" if metric.include?("us_per_tok")
  return "ms" if metric.end_with?("_ms")
  ""
end

# Detect mode.
update_mode = ARGV.include?("--update")
report_only = ARGV.include?("--report")

# Load baselines (or initialize empty if --update). Schema:
#   metric, value, unit, direction (lower|higher), tolerance_pct
baselines = {}
if File.exist?(BASELINES_PATH) && !update_mode
  CSV.foreach(BASELINES_PATH, headers: true) do |row|
    baselines[row["metric"]] = {
      value: row["value"].to_f,
      unit: row["unit"],
      direction: row["direction"],
      tolerance_pct: row["tolerance_pct"].to_f,
    }
  end
end

# Parse "BENCH <metric> <value>" lines from a process output.
def parse_bench(output)
  out = {}
  output.each_line do |line|
    line = line.chomp
    next unless line.start_with?("BENCH ")
    cols = line.split(" ", 3)
    out[cols[1]] = cols[2].to_f if cols.length == 3
  end
  out
end

# Build a bench binary from its source. Returns true on success.
def build_bench(source, binary)
  src_path = File.join(ROOT, source)
  bin_path = File.join(ROOT, binary)
  unless File.exist?(src_path)
    warn "missing bench source: #{source}"; return false
  end
  return true if File.exist?(bin_path) && File.mtime(bin_path) > File.mtime(src_path)
  # bench/build/ is gitignored, so a fresh checkout has no such dir — the
  # linker's `-o bench/build/<bin>` then fails with errno=2. Create it first
  # so `make bench` works on a clean clone (e.g. Mac), not just where a prior
  # run already made the dir.
  # Honor SPINEL_DIR (the Makefile's pin knob) so a pinned-rev bench run
  # actually builds with the pinned compiler instead of ~/sites/spinel.
  spinel = File.join(ENV["SPINEL_DIR"] || File.expand_path("~/sites/spinel"), "spinel")
  cmd = "cd #{ROOT} && mkdir -p #{File.dirname(binary)} && #{spinel} #{source} -o #{binary} 2>&1"
  out = `#{cmd}`
  unless $?.success?
    warn "build failed: #{source}\n#{out}"; return false
  end
  true
end

# Run a bench, capture stdout, parse.
def run_bench(binary, env)
  bin_path = File.join(ROOT, binary)
  unless File.exist?(bin_path)
    return nil, "binary not found: #{binary}"
  end
  prefix = env.map { |k, v| "#{k}=#{v}" }.join(" ")
  cmd = "cd #{ROOT} && #{prefix} #{binary} 2>&1"
  out = `#{cmd}`
  # A model-gated bench prints "SKIP: …" + exits 0 when its (gitignored) model
  # is absent — treat that as a skip, not a metric or a failure.
  if out =~ /^SKIP:/
    return :skip, out[/^SKIP:.*/]
  end
  unless $?.success?
    return nil, "run failed (exit #{$?.exitstatus})\n#{out}"
  end
  return parse_bench(out), nil
end

# Compare and format. Returns [line, ok?].
def evaluate(metric, observed, baseline)
  if baseline.nil?
    return [sprintf("  %-40s = %12.4f  (no baseline)", metric, observed), true]
  end
  b = baseline[:value]
  delta = observed - b
  rel = b.zero? ? 0.0 : 100.0 * delta / b
  tol = baseline[:tolerance_pct]
  bad = baseline[:direction] == "lower" ? rel > tol : -rel > tol
  marker = bad ? "REGRESS" : "ok"
  line = sprintf("  %-40s = %12.4f %s  (baseline %.4f, %+6.2f %%, tol ±%.1f %%)  [%s]",
                 metric, observed, baseline[:unit], b, rel, tol, marker)
  [line, !bad]
end

# --- Drive ---

puts "running benches..."
all_observed = {}
BENCHES.each do |b|
  unless build_bench(b[:source], b[:binary])
    puts "FAIL: cannot build #{b[:source]}"; exit 2
  end
  observed, err = run_bench(b[:binary], b[:env])
  if observed == :skip
    puts "skip: #{b[:source]} (#{err})"; next
  end
  if err
    puts "FAIL: #{b[:source]}\n#{err}"; exit 2
  end
  all_observed.merge!(observed)
end

if update_mode
  rows = all_observed.map do |metric, value|
    existing = baselines[metric] || {}
    {
      "metric" => metric,
      "value" => sprintf("%.6f", value),
      "unit" => existing[:unit] || infer_unit(metric),
      "direction" => existing[:direction] || infer_direction(metric),
      "tolerance_pct" => (existing[:tolerance_pct] || 15.0).to_s,
    }
  end
  CSV.open(BASELINES_PATH, "w", write_headers: true,
           headers: %w[metric value unit direction tolerance_pct]) do |csv|
    rows.each { |r| csv << r.values_at(*csv.headers) }
  end
  puts "wrote #{BASELINES_PATH} with #{rows.length} rows"
  exit 0
end

regress = 0
puts "results:"
all_observed.each do |metric, value|
  line, ok = evaluate(metric, value, baselines[metric])
  puts line
  regress += 1 unless ok
end

if report_only
  puts "(report mode — not gating on regressions)"
  exit 0
end

if regress > 0
  puts ""
  puts "FAIL: #{regress} regression(s) past tolerance"
  exit 1
end

puts ""
puts "ok"
exit 0
