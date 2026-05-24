# bench/check_vs_pytorch.rb — routine toy-vs-PyTorch comparison on a
# single GPU. PyTorch is the "old-stable" yardstick; this records how
# far off toy is and gates that *ratio* (not absolute ms, which is
# machine-dependent) so design changes that quietly widen the gap fail.
#
# Two modes, sharing the same machinery:
#   default — light (SmolLM2-135M shape, ~minute). Run on every commit.
#   --heavy — ambitious (Qwen2.5-1.5B LoRA train, Qwen2.5-7B-Q8 decode
#             with KV_Q8+FLASH). ~3-5 min. Run weekly / on release.
#
# Both modes:
#   - Live on gx10 (where both run).
#   - Toy: native CUDA demos.
#   - PyTorch: dev-pytorch docker image.
#   - Gate the *ratio* (toy/PT for train ms; PT/toy for infer tok/s),
#     so >1 always means "toy slower than PT, lower is better".
#
# Usage (on gx10, from the toy checkout):
#   ruby bench/check_vs_pytorch.rb               # light, run + gate
#   ruby bench/check_vs_pytorch.rb --update      # light, re-record budget
#   ruby bench/check_vs_pytorch.rb --report      # light, print, no gate
#   ruby bench/check_vs_pytorch.rb --heavy       # heavy, run + gate
#   ruby bench/check_vs_pytorch.rb --heavy --update
#   ruby bench/check_vs_pytorch.rb --heavy --report
#
# Knobs (env):
#   PT_CMD  — how to invoke the PyTorch reference (default: docker run
#             the dev-pytorch image). The one environment-specific bit.
#             Override the FULL command line; the script appends args.
#   GGUF    — light: model for the toy side (default smollm2-135m-native).
#   HEAVY_TRAIN_GGUF — heavy: toy LoRA-train model (default qwen25-1.5b-native).
#   HEAVY_INFER_GGUF — heavy: toy decode model (default qwen25-7b-native-q8).

require "csv"

ROOT = File.expand_path("../..", __FILE__)
heavy   = ARGV.include?("--heavy")
update  = ARGV.include?("--update")
report  = ARGV.include?("--report")

BUDGET_PATH = File.expand_path(
  heavy ? "../baselines_vs_pytorch_heavy.csv" : "../baselines_vs_pytorch.csv",
  __FILE__)

# The one environment-specific knob: how to reach a torch+CUDA python.
# We append args (workload, arch, lengths) per call.
PT_BASE = ENV["PT_CMD"] || (
  'docker run --rm --gpus all --ipc=host -v "$PWD":/w -w /w ' \
  'gx10/dev-pytorch:latest python3 bench/ref_pytorch.py')

def parse_bench(output, metric)
  output.each_line do |line|
    next unless line.start_with?("BENCH #{metric} ")
    return line.split(" ", 3)[2].to_f
  end
  nil
end

def grab(output, regex)
  m = output.match(regex); m ? m[1].to_f : nil
end

# Run a PT and toy half. Returns the four numbers + a printable label.
def run_light(pt_base)
  gguf = ENV["GGUF"] || "data/smollm2-135m-native.gguf"
  puts "== PyTorch reference (light: smollm2_135m) =="
  pt_out = `cd #{ROOT} && #{pt_base} --workload both --device cuda 2>&1`
  pt_train = parse_bench(pt_out, "pt_train_ms")
  pt_toks  = parse_bench(pt_out, "pt_infer_toks_per_sec")
  puts "  pt_train_ms            = #{pt_train.inspect}"
  puts "  pt_infer_toks_per_sec  = #{pt_toks.inspect}"
  warn "  (PT produced no parseable BENCH lines)\n#{pt_out}" if pt_train.nil? && pt_toks.nil?

  puts "== toy CUDA (native, light) =="
  train_out = `cd #{ROOT} && MODE=ft GGUF=#{gguf} ./demos/seq_train_bench_cuda 2>&1`
  toy_train = grab(train_out, /mean=([\d.]+)\s*ms/)
  infer_out = `cd #{ROOT} && GGUF=#{gguf} PREFILL_T=32 N_NEW=8 ./demos/qwen25_bench_cuda 2>&1`
  toy_toks  = grab(infer_out, /throughput:\s*([\d.]+)\s*tok\/s/)
  puts "  toy_train_ms           = #{toy_train.inspect}"
  puts "  toy_infer_toks_per_sec = #{toy_toks.inspect}"

  ratios = {}
  ratios["ratio_train_toy_over_pt"] = toy_train / pt_train if toy_train && pt_train && pt_train > 0
  ratios["ratio_infer_toy_over_pt"] = pt_toks / toy_toks   if toy_toks && pt_toks && toy_toks > 0
  ratios
end

def run_heavy(pt_base)
  train_gguf = ENV["HEAVY_TRAIN_GGUF"] || "data/qwen25-1.5b-native.gguf"
  infer_gguf = ENV["HEAVY_INFER_GGUF"] || "data/qwen25-7b-native-q8.gguf"

  puts "== PyTorch reference (heavy: qwen25_1p5b LoRA train, seq=256) =="
  pt_train_cmd = "#{pt_base} --workload train --device cuda " \
                 "--arch qwen25_1p5b --lora --train_t 256 --steps 8 --warmup 2 " \
                 "--metric_prefix pt_heavy_train_lora_1p5b_"
  pt_train_out = `cd #{ROOT} && #{pt_train_cmd} 2>&1`
  pt_lora = parse_bench(pt_train_out, "pt_heavy_train_lora_1p5b_train_ms")
  puts "  pt_heavy_train_lora_1p5b_train_ms = #{pt_lora.inspect}"
  warn "  (PT heavy train no BENCH line)\n#{pt_train_out}" if pt_lora.nil?

  puts "== PyTorch reference (heavy: qwen25_7b decode, prefill=512 n_new=128) =="
  pt_infer_cmd = "#{pt_base} --workload infer --device cuda " \
                 "--arch qwen25_7b --prompt_len 512 --n_new 128 --warmup 8 " \
                 "--metric_prefix pt_heavy_infer_7b_"
  pt_infer_out = `cd #{ROOT} && #{pt_infer_cmd} 2>&1`
  pt_7b_toks = parse_bench(pt_infer_out, "pt_heavy_infer_7b_infer_toks_per_sec")
  puts "  pt_heavy_infer_7b_toks_per_sec = #{pt_7b_toks.inspect}"
  warn "  (PT heavy infer no BENCH line)\n#{pt_infer_out}" if pt_7b_toks.nil?

  puts "== toy CUDA (native, heavy) =="
  train_env = "BENCH_TAG=heavy_train_lora_1p5b MODE=lora STEPS=8 SEQ_LEN=256 GGUF=#{train_gguf}"
  train_out = `cd #{ROOT} && #{train_env} ./demos/seq_train_bench_cuda 2>&1`
  toy_lora = parse_bench(train_out, "heavy_train_lora_1p5b_step_ms_mean")
  puts "  toy heavy_train_lora_1p5b_step_ms_mean = #{toy_lora.inspect}"

  infer_env = "BENCH_TAG=heavy_infer_7b_q8 KV_Q8=1 FLASH_ATTN=1 " \
              "MAX_T=2048 PREFILL_T=512 N_NEW=128 N_WARMUP=8 GGUF=#{infer_gguf}"
  infer_out = `cd #{ROOT} && #{infer_env} ./demos/qwen25_bench_cuda 2>&1`
  toy_7b_toks = parse_bench(infer_out, "heavy_infer_7b_q8_decode_toks_per_sec")
  puts "  toy heavy_infer_7b_q8_decode_toks_per_sec = #{toy_7b_toks.inspect}"

  ratios = {}
  ratios["ratio_heavy_train_lora_1p5b_toy_over_pt"] = toy_lora / pt_lora if toy_lora && pt_lora && pt_lora > 0
  ratios["ratio_heavy_infer_7b_q8_toy_over_pt"] = pt_7b_toks / toy_7b_toks if toy_7b_toks && pt_7b_toks && toy_7b_toks > 0
  ratios
end

ratios = heavy ? run_heavy(PT_BASE) : run_light(PT_BASE)

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
    puts sprintf("  %-40s = %6.3f×  (no budget — run --update)", metric, v)
    next
  end
  rel = b[:value].zero? ? 0.0 : 100.0 * (v - b[:value]) / b[:value]
  bad = rel > b[:tol]
  regress += 1 if bad
  puts sprintf("  %-40s = %6.3f×  (budget %.3f×, %+5.1f %%, tol +%.0f %%)  [%s]",
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
