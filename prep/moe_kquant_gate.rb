#!/usr/bin/env ruby
# prep/moe_kquant_gate.rb — OLMoE-1B-7B Q4_K_M greedy decode against
# llama.cpp's ids. Guards two bugs that both gave plausible wrong text:
#
#   1. K-quant attention head-stride collapse (the bug long misfiled as
#      ggml#1506): head_nbytes() returned 0 for K-quant attention weights, so
#      every attention head read head 0's weight slice. Signature: first
#      token 20065, then `20065 20065 20065 42859 …`. See
#      docs/notes/mul_mat_id_quants.md.
#   2. Per-head QK-norm (toy#213): OLMoE's attn_q_norm / attn_k_norm are an
#      RMSNorm over the WHOLE Q / K vector (d_model wide) before the head
#      split. Toy normalised each head's d_head slice on its own. Signature:
#      the first token still matches (36062, "Paris"), then `8 753 1051` and
#      a run of newlines.
#
# WHAT IT ASSERTS: the 10 greedy ids for "The capital of France is" equal
# llama.cpp's (`ghcr.io/ggml-org/llama.cpp:full-cuda`, 2026-11-13, same file):
# "Paris. The capital of Spain is Madrid." A K-quant file on another machine
# could in principle flip a near-tie; if this fails only at a late index,
# check the top-2 margin there before calling it a regression (toy#212).
#
# MODEL-GATED: needs data/OLMoE-1b-7b-0924-Instruct-Q4_K_M.gguf — a ~4 GB
# gitignored dev artifact (Meshwa/OLMoE-1b-7b-0924-Instruct-gguf). When absent
# the gate SKIPs loudly (exit 0) rather than failing a fixture-only checkout.
#
#   ruby prep/moe_kquant_gate.rb [--device cuda]
require "open3"

ROOT   = File.expand_path("..", __dir__)
TOY    = File.join(ROOT, "bin", "toy")
GGUF   = File.join(ROOT, "data", "OLMoE-1b-7b-0924-Instruct-Q4_K_M.gguf")
DEVICE = (i = ARGV.index("--device")) ? ARGV[i + 1].to_s : "cpu"
LABEL  = DEVICE == "cuda" ? "moe-kquant-cuda" : "moe-kquant"

PROMPT_IDS  = "510 38479 1171 33639 261"
N           = 10
DEGEN_TOKEN = 20065   # head-collapse signature (bug 1)
LLAMA_CPP   = [36062, 15, 510, 38479, 1171, 49348, 261, 19008, 6992, 15].freeze
PER_HEAD_QK = [36062, 8, 753].freeze   # per-head QK-norm signature (bug 2)

unless %w[cpu cuda].include?(DEVICE)
  warn "moe-kquant-gate: --device must be cpu or cuda"
  exit 2
end
unless File.executable?(TOY)
  warn "moe-kquant-gate: bin/toy not executable: #{TOY}"
  exit 2
end

unless File.exist?(GGUF)
  puts "SKIP [#{LABEL}]: model absent (#{GGUF})."
  puts "  This gate needs the ~4 GB OLMoE-1B-7B Q4_K_M dev GGUF (gitignored)."
  puts "  Pull it from Meshwa/OLMoE-1b-7b-0924-Instruct-gguf to run the gate."
  exit 0
end

out, st = Open3.capture2e(TOY, "infer", GGUF, "--device", DEVICE,
                          "--prompt-ids", PROMPT_IDS, "--n", N.to_s, chdir: ROOT)
unless st.success?
  puts "GATE FAIL [#{LABEL}]: `toy infer` exited #{st.exitstatus}"
  puts out
  exit 1
end

line = out.lines.find { |l| l.start_with?("ids:") }
unless line
  puts "GATE FAIL [#{LABEL}]: no `ids:` line in output"
  puts out
  exit 1
end

all_ids = line.sub("ids:", "").split.map(&:to_i)
prompt  = PROMPT_IDS.split.map(&:to_i)
cont    = all_ids[prompt.length..] || []

agree = 0
agree += 1 while agree < [cont.length, N].min && cont[agree] == LLAMA_CPP[agree]

puts "[#{LABEL}] continuation: #{cont.join(' ')}"
puts "[#{LABEL}] llama.cpp   : #{LLAMA_CPP.join(' ')}"

if cont == LLAMA_CPP
  puts "GATE PASS [#{LABEL}]: OLMoE Q4_K_M decodes llama.cpp's #{N} ids on #{DEVICE}."
  exit 0
end

why =
  if cont.first == DEGEN_TOKEN
    "attention head collapse regressed (head_nbytes returning a zero stride for K-quants)"
  elsif cont.first(3) == PER_HEAD_QK
    "QK-norm is being applied per head again instead of over the whole Q/K vector (toy#213)"
  else
    "ids differ from llama.cpp at index #{agree} (got #{cont[agree].inspect}, expected #{LLAMA_CPP[agree]})"
  end
puts "GATE FAIL [#{LABEL}]: #{why}; agrees on the first #{agree} of #{N}."
exit 1
