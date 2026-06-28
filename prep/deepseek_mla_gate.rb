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

# Run a decode under a given env (MLA-A = expanded cache; MLA-B = latent
# cache via KV_MLA_LATENT). Returns the generated-text line.
def decode(toy, gguf, prompt, n, root, env)
  out, st = Open3.capture2e(env, toy, "infer", gguf,
                            "--prompt", prompt, "--n", n.to_s, chdir: root)
  unless st.success?
    puts "GATE FAIL [deepseek-mla]: `toy infer` exited #{st.exitstatus}"
    puts out
    exit 1
  end
  out.lines.map(&:rstrip).reject(&:empty?).last.to_s
end

# MLA-A (expanded per-head K/V cache — the default path).
text_a = decode(TOY, GGUF, PROMPT, N, ROOT, {})
puts "[deepseek-mla] MLA-A decode: #{text_a}"
unless text_a.downcase.include?(EXPECT.downcase)
  puts "GATE FAIL [deepseek-mla]: expected MLA-A decode to contain #{EXPECT.inspect}"
  puts "  (MLA projection / asymmetric KV / decoupled YaRN RoPE / per-layer MoE"
  puts "   / add_bos regressed — got: #{text_a.inspect})"
  exit 1
end

# MLA-B (latent cache — the memory win). Must recall the same fact AND
# match MLA-A's greedy decode exactly (same math, different cache layout).
text_b = decode(TOY, GGUF, PROMPT, N, ROOT, { "KV_MLA_LATENT" => "1" })
puts "[deepseek-mla] MLA-B decode: #{text_b}"
unless text_b.downcase.include?(EXPECT.downcase)
  puts "GATE FAIL [deepseek-mla]: expected MLA-B (latent cache) decode to contain #{EXPECT.inspect}"
  puts "  (latent-cache up-projection regressed — got: #{text_b.inspect})"
  exit 1
end
unless text_a == text_b
  puts "GATE FAIL [deepseek-mla]: MLA-A and MLA-B greedy decodes diverged"
  puts "  A: #{text_a.inspect}"
  puts "  B: #{text_b.inspect}"
  exit 1
end

puts "GATE PASS [deepseek-mla]: coherent DeepSeek-V2 MLA decode, MLA-A == MLA-B " +
     "(latent-attention + decoupled YaRN RoPE + per-layer MoE; expanded & latent cache)."
exit 0
