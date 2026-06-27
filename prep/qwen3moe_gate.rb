#!/usr/bin/env ruby
# prep/qwen3moe_gate.rb — regression guard for Qwen3-MoE (qwen3moe arch)
# inference: the routing-variant work in Phase 1 (roadmap "modern-LLM
# primitives" #1).
#
# WHAT IT GUARDS. Qwen3-30B-A3B exercises three things the older softmax-only
# MoE path didn't:
#   1. arch-prefix loading — qwen3moe.* metadata keys (GGUFLoad.detect_arch_prefix);
#   2. separate expert intermediate size — expert_ff (768) ≠ dense d_ff (6144),
#      read from the gate_exps tensor's ne[1] in realize_for_mmap (else the 3D
#      expert mmap alloc aborts 8× oversize);
#   3. norm_topk_prob — the K selected expert weights renormalized to sum 1
#      (GGUFLoad.moe_norm_topk? → build_moe_ffn); without it the decode stutters
#      ("…Germany is is Berlin").
#
# Greedy decode is deterministic, so we assert the model recalls a fact through
# the MoE FFN ("The capital of France is" → contains "Paris") AND the
# continuation isn't degenerate. Structural, not byte-exact (K-quant noise drifts
# across ggml versions).
#
# MODEL-GATED: needs data/Qwen3-30B-A3B-Q4_K_M.gguf — an ~18 GB gitignored dev
# artifact (unsloth/Qwen3-30B-A3B-GGUF). Absent → SKIP loudly (exit 0).
#
#   ruby prep/qwen3moe_gate.rb
require "open3"

ROOT = File.expand_path("..", __dir__)
TOY  = File.join(ROOT, "bin", "toy")
GGUF = File.join(ROOT, "data", "Qwen3-30B-A3B-Q4_K_M.gguf")

PROMPT  = "The capital of France is"
N       = 8
EXPECT  = "Paris"   # factual recall through router → top-8 experts → renorm

unless File.executable?(TOY)
  warn "qwen3moe-gate: bin/toy not executable: #{TOY}"
  exit 2
end

unless File.exist?(GGUF)
  puts "SKIP [qwen3moe]: model absent (#{GGUF})."
  puts "  This gate needs the ~18 GB Qwen3-30B-A3B Q4_K_M dev GGUF (gitignored)."
  puts "  Pull it: toy fetch unsloth/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf"
  exit 0
end

out, st = Open3.capture2e(TOY, "infer", GGUF,
                          "--prompt", PROMPT, "--n", N.to_s, chdir: ROOT)
unless st.success?
  puts "GATE FAIL [qwen3moe]: `toy infer` exited #{st.exitstatus}"
  puts out
  exit 1
end

# Last non-build line is the generated text (build chatter precedes it).
text = out.lines.map(&:rstrip).reject(&:empty?).last.to_s
puts "[qwen3moe] decode: #{text}"

unless text.downcase.include?(EXPECT.downcase)
  puts "GATE FAIL [qwen3moe]: expected the decode to contain #{EXPECT.inspect}"
  puts "  (Qwen3-MoE routing/expert-ff/renorm regressed — got: #{text.inspect})"
  exit 1
end

puts "GATE PASS [qwen3moe]: coherent Qwen3-MoE decode (arch-prefix + expert-ff + norm_topk)."
exit 0
