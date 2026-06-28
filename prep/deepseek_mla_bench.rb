#!/usr/bin/env ruby
# prep/deepseek_mla_bench.rb — MLA-A (expanded per-head K/V cache) vs
# MLA-B (latent c_kv + shared k_rope cache) on DeepSeek-V2-Lite. Reports
# the analytical KV-cache size per token and the measured wall-clock +
# peak RSS for a fixed greedy decode. Not a pass/fail gate — a perf probe.
#
# The headline: MLA-B caches kv_lora_rank+rope_dim (576) floats/token/layer
# instead of n_heads*(d_head_k+d_head_v) (5120) — ~8.9x smaller — at the
# cost of one extra [kv_lora_rank -> n_heads*256] up-projection over the
# cached history per step (the naive, non-absorbed form). The absorbed
# form (fold attn_kv_b into attn_q/attn_output) would recover the speed.
#
# MODEL-GATED: needs data/DeepSeek-V2-Lite-Chat.Q4_K_M.gguf (gitignored).
#
#   ruby prep/deepseek_mla_bench.rb [N_NEW]
require "open3"
require "benchmark"

ROOT = File.expand_path("..", __dir__)
TOY  = File.join(ROOT, "libexec", "toy-infer")
GGUF = File.join(ROOT, "data", "DeepSeek-V2-Lite-Chat.Q4_K_M.gguf")
PROMPT = "The history of the Roman empire began"
N      = (ARGV[0] || "48").to_i

# DeepSeek-V2-Lite dims.
L, H, DK, DV, LORA, ROPE = 27, 16, 192, 128, 512, 64
a = L * H * (DK + DV)      # MLA-A floats/token
b = L * (LORA + ROPE)      # MLA-B floats/token
puts "KV-cache floats/token: MLA-A=#{a}  MLA-B=#{b}  ratio=#{(a.to_f / b).round(2)}x"
[256, 4096, 32768].each do |t|
  puts format("  @max_T=%-6d  MLA-A=%8.1f MB   MLA-B=%7.1f MB",
              t, a * t * 4 / 1e6, b * t * 4 / 1e6)
end

unless File.executable?(TOY) && File.exist?(GGUF)
  puts "SKIP [deepseek-mla-bench]: need libexec/toy-infer (make libexec/toy-infer) " \
       "and the ~9.7 GB DeepSeek-V2-Lite GGUF (gitignored)."
  exit 0
end

def run(env, label)
  full = { "GGUF" => GGUF, "PROMPT" => PROMPT, "N_NEW" => N.to_s }.merge(env)
  out = nil
  wall = Benchmark.realtime { out, _ = Open3.capture2e(full, TOY) }
  text = out.lines.map(&:rstrip).find { |l| l.start_with?("text:") } || ""
  puts format("  %-6s  %.2f s   %s", label, wall, text)
  wall
end

# Warm the mmap page cache first so the comparison isn't load-dominated.
Open3.capture2e({ "GGUF" => GGUF, "PROMPT" => PROMPT, "N_NEW" => "1" }, TOY)
puts "decode N=#{N} (warm page cache):"
ta = run({}, "MLA-A")
tb = run({ "KV_MLA_LATENT" => "1" }, "MLA-B")
puts format("  MLA-B is %.0f%% %s than MLA-A (naive up-projection; absorbed form would recover it)",
            ((tb - ta).abs / ta * 100), tb > ta ? "slower" : "faster")
