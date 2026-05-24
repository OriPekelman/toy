# bench/check_vs_pytorch.rb — routine toy-vs-PyTorch comparison on a
# single GPU. PyTorch is the "old-stable" yardstick; this records how
# far off toy is and gates that *ratio* (not absolute ms, which is
# machine-dependent) so design changes that quietly widen the gap fail.
#
# Runs ON gx10 (where both live): toy CUDA benches run natively; the
# PyTorch reference runs in the dev-pytorch container. Same model on
# both sides (SmolLM2-135M dims) for two workloads — a full-FT training
# step and KV-cache decode.
#
# Usage (on gx10, from the toy checkout):
#   ruby bench/check_vs_pytorch.rb            # run + gate vs budget
#   ruby bench/check_vs_pytorch.rb --update   # re-record the budget
#   ruby bench/check_vs_pytorch.rb --report   # run + print, no gate
#
# Knobs (env):
#   PT_CMD  — how to invoke the PyTorch reference (default: docker run
#             the dev-pytorch image). The one environment-specific bit.
#   GGUF    — model for the toy side (default smollm2-135m-native.gguf).
#
# Ratios are "toy / pytorch" (train: ms; infer: tok/s flipped to pt/toy)
# so >1 means toy is slower; lower is better. Budget rows live in
# bench/baselines_vs_pytorch.csv with the same schema as baselines.csv.

require "csv"

ROOT = File.expand_path("../..", __FILE__)
BUDGET_PATH = File.expand_path("../baselines_vs_pytorch.csv", __FILE__)
GGUF = ENV["GGUF"] || "data/smollm2-135m-native.gguf"

# The one environment-specific knob: how to reach a torch+CUDA python.
PT_CMD = ENV["PT_CMD"] || (
  'docker run --rm --gpus all --ipc=host -v "$PWD":/w -w /w ' \
  'gx10/dev-pytorch:latest python3 bench/ref_pytorch.py --workload both --device cuda'
)

# Each engine command emits BENCH lines we parse. The toy demos print
# human-readable lines, so we give a regex to lift the number out.
def sh(cmd)
  `cd #{ROOT} && #{cmd} 2>&1`
end

def grab(output, regex)
  m = output.match(regex)
  m ? m[1].to_f : nil
end

def parse_bench(output, metric)
  output.each_line do |line|
    next unless line.start_with?("BENCH #{metric} ")
    return line.split(" ", 3)[2].to_f
  end
  nil
end

update = ARGV.include?("--update")
report = ARGV.include?("--report")

puts "== PyTorch reference (yardstick) =="
pt_out = sh(PT_CMD)
pt_train = parse_bench(pt_out, "pt_train_ms")
pt_toks  = parse_bench(pt_out, "pt_infer_toks_per_sec")
puts "  pt_train_ms            = #{pt_train.inspect}"
puts "  pt_infer_toks_per_sec  = #{pt_toks.inspect}"
warn "  (PyTorch reference produced no parseable BENCH lines)\n#{pt_out}" if pt_train.nil? && pt_toks.nil?

puts "== toy CUDA (native) =="
train_out = sh("MODE=ft GGUF=#{GGUF} ./demos/seq_train_bench_cuda")
toy_train = grab(train_out, /mean=([\d.]+)\s*ms/)
infer_out = sh("GGUF=#{GGUF} PREFILL_T=32 N_NEW=8 ./demos/qwen25_bench_cuda")
toy_toks  = grab(infer_out, /throughput:\s*([\d.]+)\s*tok\/s/)
puts "  toy_train_ms           = #{toy_train.inspect}"
puts "  toy_infer_toks_per_sec = #{toy_toks.inspect}"

# Ratios (lower is better; >1 = toy slower than PyTorch).
ratios = {}
ratios["ratio_train_toy_over_pt"] = toy_train / pt_train if toy_train && pt_train && pt_train > 0
ratios["ratio_infer_toy_over_pt"] = pt_toks / toy_toks   if toy_toks && pt_toks && toy_toks > 0

if ratios.empty?
  puts "\nFAIL: no ratios computed — a sub-bench did not produce a number (see above)."
  exit 2
end

if update
  CSV.open(BUDGET_PATH, "w", write_headers: true,
           headers: %w[metric value unit direction tolerance_pct]) do |csv|
    ratios.each { |m, v| csv << [m, sprintf("%.4f", v), "x", "lower", "25.0"] }
  end
  puts "\nwrote #{BUDGET_PATH} (#{ratios.length} ratios)"
  exit 0
end

budget = {}
if File.exist?(BUDGET_PATH)
  CSV.foreach(BUDGET_PATH, headers: true) do |r|
    budget[r["metric"]] = { value: r["value"].to_f, tol: r["tolerance_pct"].to_f }
  end
end

puts "\n== ratios (toy / PyTorch — lower is better) =="
regress = 0
ratios.each do |metric, v|
  b = budget[metric]
  if b.nil?
    puts sprintf("  %-26s = %6.3f×  (no budget — run --update)", metric, v)
    next
  end
  rel = b[:value].zero? ? 0.0 : 100.0 * (v - b[:value]) / b[:value]
  bad = rel > b[:tol]
  regress += 1 if bad
  puts sprintf("  %-26s = %6.3f×  (budget %.3f×, %+5.1f %%, tol +%.0f %%)  [%s]",
               metric, v, b[:value], rel, b[:tol], bad ? "REGRESS" : "ok")
end

if report
  puts "\n(report mode — not gating)"; exit 0
end
if regress > 0
  puts "\nFAIL: #{regress} ratio(s) widened past tolerance vs PyTorch"; exit 1
end
puts "\nok — toy stays within budget of PyTorch"
exit 0
