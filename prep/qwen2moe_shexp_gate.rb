#!/usr/bin/env ruby
# prep/qwen2moe_shexp_gate.rb — regression guard for SHARED EXPERTS (MoE
# Phase 2, roadmap "modern-LLM primitives" #1 / P2).
#
# WHAT IT GUARDS. Qwen1.5-MoE-A2.7B (qwen2moe arch) is the clean shared-expert
# target — standard attention (not MLA), 60 routed experts top-4 PLUS an
# always-on shared expert with a Qwen-style sigmoid gate. It exercises:
#   1. shared-expert load — blk.N.ffn_{gate,up,down}_shexp + ffn_gate_inp_shexp,
#      detected by tensor presence (the metadata omits expert_shared_count);
#      sized from gate_shexp ne[1] (sh_ff=5632);
#   2. the shared SwiGLU added to the routed output, scaled by sigmoid(h·gate);
#   3. the per-arch norm_topk default — qwen2moe is norm_topk_prob=FALSE (unlike
#      qwen3moe); applying the renorm here degrades the decode
#      ("opposite of hot is the same as the opposite of the other").
#
# Greedy decode is deterministic; we assert a template-robust factual
# completion ("The opposite of hot is" → contains "cold"). Structural, not
# byte-exact.
#
# MODEL-GATED: needs data/Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf — a ~9 GB
# gitignored dev artifact. Absent → SKIP loudly (exit 0).
#
#   ruby prep/qwen2moe_shexp_gate.rb
require "open3"

ROOT = File.expand_path("..", __dir__)
TOY  = File.join(ROOT, "bin", "toy")
GGUF = File.join(ROOT, "data", "Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf")

PROMPT = "The opposite of hot is"
N      = 6
EXPECT = "cold"

unless File.executable?(TOY)
  warn "qwen2moe-shexp-gate: bin/toy not executable: #{TOY}"
  exit 2
end

unless File.exist?(GGUF)
  puts "SKIP [qwen2moe-shexp]: model absent (#{GGUF})."
  puts "  This gate needs the ~9 GB Qwen1.5-MoE-A2.7B-Chat Q4_K_M dev GGUF (gitignored)."
  puts "  Pull it: toy fetch RichardErkhov/Qwen_-_Qwen1.5-MoE-A2.7B-Chat-gguf Qwen1.5-MoE-A2.7B-Chat.Q4_K_M.gguf"
  exit 0
end

out, st = Open3.capture2e(TOY, "infer", GGUF,
                          "--prompt", PROMPT, "--n", N.to_s, chdir: ROOT)
unless st.success?
  puts "GATE FAIL [qwen2moe-shexp]: `toy infer` exited #{st.exitstatus}"
  puts out
  exit 1
end

text = out.lines.map(&:rstrip).reject(&:empty?).last.to_s
puts "[qwen2moe-shexp] decode: #{text}"

unless text.downcase.include?(EXPECT.downcase)
  puts "GATE FAIL [qwen2moe-shexp]: expected the decode to contain #{EXPECT.inspect}"
  puts "  (shared-expert load/forward or the qwen2moe norm_topk default regressed)"
  exit 1
end

puts "GATE PASS [qwen2moe-shexp]: coherent shared-expert decode (Qwen1.5-MoE)."
exit 0
