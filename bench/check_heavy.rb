# bench/check_heavy.rb — toy-only "heavy" bench orchestrator. Runs the
# two ambitious workloads on the actual demo binaries (no PyTorch, no
# docker — fast iteration loop for choosing between optimization
# strategies). Compares absolute toy-side numbers to a saved baseline
# so a candidate optimization can be A/B-compared run-over-run.
#
# Workloads:
#   1. LoRA training step, Qwen2.5-1.5B, seq_len=256
#        - exercises the backward graph + AdamW at non-toy shape
#        - metrics: step_ms_{mean,p95,stddev}
#   2. KV-cache greedy decode, Qwen2.5-7B-Q8, KV_Q8+FLASH on,
#      prefill=512, n_new=128
#        - exercises mmap + Q8 dequant + the opt-in perf knobs
#        - metrics: realize_ms, prefill_ms, decode_ms_{mean,p95},
#          decode_toks_per_sec
#
# Usage:
#   ruby bench/check_heavy.rb                # run + compare vs baselines
#   ruby bench/check_heavy.rb --update       # run + rewrite baselines
#   ruby bench/check_heavy.rb --report       # run + print, no gate
#
# Baseline file: bench/baselines_heavy.csv (same schema as baselines.csv).
# Tolerance defaults to 15% but is per-row in the CSV — bump tolerances
# for noisier metrics (stddev) and tighten them for stable ones (mean).

require "csv"

ROOT = File.expand_path("../..", __FILE__)
BASELINES_PATH = File.expand_path("../baselines_heavy.csv", __FILE__)

# Two workloads. Each is an existing binary in demos/ invoked with
# heavy env. BENCH_TAG controls which BENCH lines they emit; same tag
# becomes the metric-name prefix in baselines_heavy.csv so the
# orchestrator and the binary stay in lock-step.
WORKLOADS = [
  {
    name: "heavy_train_lora_1p5b",
    binary: "demos/seq_train_bench_cuda",
    env: {
      "BENCH_TAG" => "heavy_train_lora_1p5b",
      "MODE"      => "lora",
      "STEPS"     => "8",
      "SEQ_LEN"   => "256",
      "GGUF"      => ENV["HEAVY_TRAIN_GGUF"] || "data/qwen25-1.5b-native.gguf",
    },
  },
  {
    name: "heavy_infer_7b_q8",
    binary: "demos/qwen25_bench_cuda",
    env: {
      "BENCH_TAG"  => "heavy_infer_7b_q8",
      "KV_Q8"      => "1",
      "FLASH_ATTN" => "1",
      "MAX_T"      => "2048",
      "PREFILL_T"  => "512",
      "N_NEW"      => "128",
      "N_WARMUP"   => "8",
      "GGUF"       => ENV["HEAVY_INFER_GGUF"] || "data/qwen25-7b-native-q8.gguf",
    },
  },
]

def infer_direction(metric)
  return "higher" if metric.include?("_per_sec") || metric.include?("toks_per")
  "lower"
end

def infer_unit(metric)
  return "tok/s" if metric.include?("toks_per_sec")
  return "ms"    if metric.end_with?("_ms") || metric.include?("_ms_")
  ""
end

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

def run_workload(w)
  bin = File.join(ROOT, w[:binary])
  unless File.exist?(bin)
    return nil, "binary not found: #{w[:binary]} (run `make #{w[:binary]}`)"
  end
  unless File.exist?(File.join(ROOT, w[:env]["GGUF"]))
    return nil, "GGUF not found: #{w[:env]["GGUF"]} (set HEAVY_TRAIN_GGUF / HEAVY_INFER_GGUF)"
  end
  prefix = w[:env].map { |k, v| "#{k}=#{v}" }.join(" ")
  cmd = "cd #{ROOT} && #{prefix} #{w[:binary]} 2>&1"
  out = `#{cmd}`
  unless $?.success?
    return nil, "#{w[:name]} run failed (exit #{$?.exitstatus})\n#{out}"
  end
  parsed = parse_bench(out)
  if parsed.empty?
    return nil, "#{w[:name]} produced no BENCH lines\n#{out}"
  end
  return parsed, nil
end

update_mode = ARGV.include?("--update")
report_only = ARGV.include?("--report")

baselines = {}
if File.exist?(BASELINES_PATH) && !update_mode
  CSV.foreach(BASELINES_PATH, headers: true) do |r|
    baselines[r["metric"]] = {
      value: r["value"].to_f, unit: r["unit"],
      direction: r["direction"], tolerance_pct: r["tolerance_pct"].to_f,
    }
  end
end

puts "== toy heavy bench =="
all_observed = {}
WORKLOADS.each do |w|
  puts "  -> #{w[:name]} (#{w[:binary]}, GGUF=#{w[:env]["GGUF"]})"
  observed, err = run_workload(w)
  if err
    puts "FAIL: #{err}"; exit 2
  end
  all_observed.merge!(observed)
end

def evaluate(metric, observed, baseline)
  return [sprintf("  %-44s = %12.4f  (no baseline — run --update)", metric, observed), true] if baseline.nil?
  b = baseline[:value]
  rel = b.zero? ? 0.0 : 100.0 * (observed - b) / b
  bad = baseline[:direction] == "lower" ? rel > baseline[:tolerance_pct] : -rel > baseline[:tolerance_pct]
  marker = bad ? "REGRESS" : "ok"
  line = sprintf("  %-44s = %12.4f %s  (baseline %.4f, %+6.2f %%, tol ±%.1f %%)  [%s]",
                 metric, observed, baseline[:unit], b, rel, baseline[:tolerance_pct], marker)
  [line, !bad]
end

if update_mode
  rows = all_observed.map do |metric, value|
    existing = baselines[metric] || {}
    {
      "metric" => metric,
      "value"  => sprintf("%.6f", value),
      "unit"   => existing[:unit] || infer_unit(metric),
      "direction" => existing[:direction] || infer_direction(metric),
      "tolerance_pct" => (existing[:tolerance_pct] || 15.0).to_s,
    }
  end
  CSV.open(BASELINES_PATH, "w", write_headers: true,
           headers: %w[metric value unit direction tolerance_pct]) do |csv|
    rows.each { |r| csv << r.values_at(*csv.headers) }
  end
  puts "wrote #{BASELINES_PATH} (#{rows.length} rows)"
  exit 0
end

puts "\n== results =="
regress = 0
all_observed.each do |metric, value|
  line, ok = evaluate(metric, value, baselines[metric])
  puts line
  regress += 1 unless ok
end

if report_only
  puts "\n(report mode — not gating)"; exit 0
end
if regress > 0
  puts "\nFAIL: #{regress} metric(s) regressed past tolerance"; exit 1
end
puts "\nok — heavy bench within baselines"
exit 0
