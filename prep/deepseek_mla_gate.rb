#!/usr/bin/env ruby
# prep/deepseek_mla_gate.rb — regression guard for DeepSeek-V2 MLA-A
# (deepseek2 arch) inference: the Multi-head Latent Attention milestone
# (docs/roadmap/deepseek-mla-arch.md).
#
# WHAT IT GUARDS. DeepSeek-V2-Lite exercises several things no prior toy
# model did, all on the correctness-first MLA-A path:
#   1. deepseek2 arch-prefix load (kv_lora_rank / key_length=192 /
#      value_length=128 / rope.dimension_count=64 dims);
#   2. MLA projection — attn_q split into nope/rope, attn_kv_a_mqa → latent
#      c_kv + shared decoupled-RoPE key, attn_kv_a_norm, attn_kv_b up-proj;
#   3. ASYMMETRIC per-head K(192)/V(128) cache;
#   4. decoupled YaRN RoPE on the 64-dim pe slice (tnn_rope_ext_yarn with
#      n_ctx_orig) + the mscale-adjusted softmax scale;
#   5. per-layer dense/MoE dispatch (leading_dense_block_count=1: layer 0
#      dense, layers 1-26 MoE with 64 experts top-6 + fused shared expert);
#   6. add_bos_token honored on the byte-level BPE path (without the BOS
#      attention sink the model degenerates to "is is is").
#
# Greedy decode is deterministic, so we assert the model recalls a fact
# through the full MLA + MoE stack ("The capital of France is" → contains
# "Paris"). Structural, not byte-exact (K-quant noise drifts across ggml
# versions). Matches llama.cpp's reference continuation " Paris."
#
# MODEL-GATED: needs data/DeepSeek-V2-Lite-Chat.Q4_K_M.gguf — a ~9.7 GB
# gitignored dev artifact (mradermacher/DeepSeek-V2-Lite-Chat-GGUF).
# Absent → SKIP loudly (exit 0).
#
#   ruby prep/deepseek_mla_gate.rb
require "open3"

ROOT = File.expand_path("..", __dir__)
TOY  = File.join(ROOT, "bin", "toy")
GGUF = File.join(ROOT, "data", "DeepSeek-V2-Lite-Chat.Q4_K_M.gguf")

PROMPT = "The capital of France is"
N      = 6
EXPECT = "Paris"   # factual recall through MLA attention + routed/shared MoE

unless File.executable?(TOY)
  warn "deepseek-mla-gate: bin/toy not executable: #{TOY}"
  exit 2
end

unless File.exist?(GGUF)
  puts "SKIP [deepseek-mla]: model absent (#{GGUF})."
  puts "  This gate needs the ~9.7 GB DeepSeek-V2-Lite-Chat Q4_K_M dev GGUF (gitignored)."
  puts "  Pull it: toy fetch mradermacher/DeepSeek-V2-Lite-Chat-GGUF DeepSeek-V2-Lite-Chat.Q4_K_M.gguf"
  exit 0
end

out, st = Open3.capture2e(TOY, "infer", GGUF,
                          "--prompt", PROMPT, "--n", N.to_s, chdir: ROOT)
unless st.success?
  puts "GATE FAIL [deepseek-mla]: `toy infer` exited #{st.exitstatus}"
  puts out
  exit 1
end

# Last non-build line is the generated text (build chatter precedes it).
text = out.lines.map(&:rstrip).reject(&:empty?).last.to_s
puts "[deepseek-mla] decode: #{text}"

unless text.downcase.include?(EXPECT.downcase)
  puts "GATE FAIL [deepseek-mla]: expected the decode to contain #{EXPECT.inspect}"
  puts "  (MLA projection / asymmetric KV / decoupled YaRN RoPE / per-layer MoE"
  puts "   / add_bos regressed — got: #{text.inspect})"
  exit 1
end

puts "GATE PASS [deepseek-mla]: coherent DeepSeek-V2 MLA decode " +
     "(latent-attention + decoupled YaRN RoPE + per-layer MoE)."
exit 0
