#!/usr/bin/env ruby
# bench/aggregate_trace.rb — read a Chrome Trace JSON emitted by tnn_trace
# (open via TRACE=path; per-op events via TRACE_OPS=1) and produce a
# per-op attribution table: which ops dominate, how many times each is
# called, mean duration per call.
#
# Usage:
#   ruby bench/aggregate_trace.rb <trace.json>
#   ruby bench/aggregate_trace.rb <trace.json> --top=15
#   ruby bench/aggregate_trace.rb <trace.json> --json     # machine-readable
#
# What it does:
#   - Reads all "X" (complete) events from the JSON array.
#   - Groups events by `name`. Anything that looks like a top-level span
#     ("step", "Mat.matmul", etc.) is treated separately from ggml-op
#     names. The op-vs-span split is heuristic but works in practice:
#     ggml op names are short lowercase identifiers (matmul, rms_norm,
#     get_rows, …) while Ruby spans contain `.`.
#   - Sorts ops by total duration descending and emits a table.
#   - Reports total ops time, total span time, and the "unaccounted"
#     gap (host orchestration, sync, etc.) so the trade-off is visible.

require "json"
require "set"

unless ARGV.length >= 1
  warn "usage: ruby bench/aggregate_trace.rb <trace.json> [--top=N] [--json]"
  exit 2
end

path = ARGV[0]
top_n = (ARGV.find { |a| a.start_with?("--top=") }&.split("=", 2)&.last || "20").to_i
emit_json = ARGV.include?("--json")

events = JSON.parse(File.read(path))
unless events.is_a?(Array)
  warn "trace is not an array — got #{events.class}"
  exit 2
end

# A Ruby-level span ("step", "Mat.matmul", "Mat.plus", …) is anything
# that contains a "." or matches a small set of known span names. The
# rest are ggml-op events captured via TRACE_OPS=1.
SPAN_BAREWORDS = Set["step", "forward", "backward", "optimizer"]
def span?(name) = name.include?(".") || SPAN_BAREWORDS.include?(name)

ops   = Hash.new { |h, k| h[k] = { total_us: 0.0, count: 0, max_us: 0.0 } }
spans = Hash.new { |h, k| h[k] = { total_us: 0.0, count: 0, max_us: 0.0 } }

events.each do |e|
  next unless e["ph"] == "X"
  name = e["name"]
  dur  = e["dur"].to_f
  bucket = span?(name) ? spans : ops
  bucket[name][:total_us] += dur
  bucket[name][:count]    += 1
  bucket[name][:max_us]    = dur if dur > bucket[name][:max_us]
end

total_ops_us   = ops.values.sum { |v| v[:total_us] }
step_span      = spans["step"]
total_step_us  = step_span ? step_span[:total_us] : 0.0
step_count     = step_span ? step_span[:count] : 0

if emit_json
  out = {
    "trace_path" => path,
    "n_events"   => events.length,
    "step" => {
      "count_in_trace" => step_count,
      "total_us"       => total_step_us,
      "mean_us"        => step_count > 0 ? total_step_us / step_count : 0,
    },
    "ops" => ops.sort_by { |_, v| -v[:total_us] }.map { |name, v|
      { "name" => name, "count" => v[:count], "total_us" => v[:total_us],
        "mean_us" => v[:total_us] / v[:count], "max_us" => v[:max_us] }
    },
    "spans" => spans.sort_by { |_, v| -v[:total_us] }.map { |name, v|
      { "name" => name, "count" => v[:count], "total_us" => v[:total_us] }
    },
    "totals_us" => {
      "all_ops" => total_ops_us,
      "all_steps" => total_step_us,
      "unaccounted_us" => total_step_us - total_ops_us,
    },
  }
  puts JSON.pretty_generate(out)
  exit 0
end

puts "trace: #{path}"
puts "  events:    #{events.length}"
puts "  step span: #{step_count} steps, total = #{(total_step_us / 1000).round(2)} ms" \
     ", mean = #{step_count > 0 ? (total_step_us / step_count / 1000).round(2) : 0} ms"
puts "  per-op sum: #{(total_ops_us / 1000).round(2)} ms across #{ops.values.sum { |v| v[:count] }} events"
if step_count > 0 && total_ops_us > 0
  unaccounted = total_step_us - total_ops_us
  pct = 100.0 * unaccounted / total_step_us
  puts sprintf("  unaccounted: %.2f ms (%+.1f%% of step time) — host orchestration, syncs, overlap",
               unaccounted / 1000, pct)
end

if !ops.empty?
  puts
  puts "top #{top_n} ops by total time (out of #{ops.size} unique op names):"
  puts sprintf("  %-26s %8s %12s %10s %10s %8s",
               "op", "count", "total ms", "mean us", "max us", "%step")
  ops.sort_by { |_, v| -v[:total_us] }.first(top_n).each do |name, v|
    pct = total_step_us > 0 ? 100.0 * v[:total_us] / total_step_us : 0.0
    puts sprintf("  %-26s %8d %12.3f %10.2f %10.2f %7.2f%%",
                 name, v[:count], v[:total_us] / 1000, v[:total_us] / v[:count],
                 v[:max_us], pct)
  end
end

if !spans.empty?
  puts
  puts "spans (Ruby-level instrumented blocks):"
  spans.sort_by { |_, v| -v[:total_us] }.each do |name, v|
    puts sprintf("  %-30s count=%d total=%.2f ms mean=%.2f ms",
                 name, v[:count], v[:total_us] / 1000, v[:total_us] / v[:count] / 1000)
  end
end
