#!/usr/bin/env ruby
# prep/moe_kquant_gate.rb — regression guard for the K-quant MoE attention
# head-stride collapse (the bug long misfiled as ggml#1506).
#
# THE BUG: head_nbytes() returned 0 for K-quant attention weights, so the
# per-head mmap slice offset `off_base + hq*0` made every attention head read
# head 0's weight slice. On OLMoE-1B-7B Q4_K_M (forced down realize_for_mmap as
# a MoE model) that collapsed multi-head attention into degenerate, repeating
# output: `… 261 | 20065 20065 20065 20065 20065 42859 42859 42859 42859 42859`.
# After the fix the same greedy decode is varied/coherent. See
# docs/notes/mul_mat_id_quants.md and lib/toy_smollm2_ffi_kv.rb#head_nbytes.
#
# This is a STRUCTURAL gate, not a byte-exact one: it asserts the continuation
# is non-degenerate (enough distinct tokens, no long single-token run) rather
# than pinning exact ids — so it guards "attention didn't collapse" without
# false-alarming on benign K-quant quant-noise drift across ggml versions.
#
# MODEL-GATED: needs data/OLMoE-1b-7b-0924-Instruct-Q4_K_M.gguf — a ~4 GB
# gitignored dev artifact (Meshwa/OLMoE-1b-7b-0924-Instruct-gguf). When absent
# the gate SKIPs loudly (exit 0) rather than failing a fixture-only checkout.
#
#   ruby prep/moe_kquant_gate.rb
require "open3"

ROOT = File.expand_path("..", __dir__)
TOY  = File.join(ROOT, "bin", "toy")
GGUF = File.join(ROOT, "data", "OLMoE-1b-7b-0924-Instruct-Q4_K_M.gguf")

# The exact prompt + the degenerate token recorded pre-fix (token 20065 = "Dub",
# repeated). Greedy decode, no sampler → deterministic.
PROMPT_IDS  = "510 38479 1171 33639 261"
N           = 10
DEGEN_TOKEN = 20065
MIN_DISTINCT = 5   # coherent gave 10/10 distinct; degenerate gave 2/10
MAX_RUN      = 3   # degenerate had runs of 5; coherent has no long single-token run

unless File.executable?(TOY)
  warn "moe-kquant-gate: bin/toy not executable: #{TOY}"
  exit 2
end

unless File.exist?(GGUF)
  puts "SKIP [moe-kquant]: model absent (#{GGUF})."
  puts "  This gate needs the ~4 GB OLMoE-1B-7B Q4_K_M dev GGUF (gitignored)."
  puts "  Pull it from Meshwa/OLMoE-1b-7b-0924-Instruct-gguf to run the gate."
  exit 0
end

out, st = Open3.capture2e(TOY, "infer", GGUF,
                          "--prompt-ids", PROMPT_IDS, "--n", N.to_s, chdir: ROOT)
unless st.success?
  puts "GATE FAIL [moe-kquant]: `toy infer` exited #{st.exitstatus}"
  puts out
  exit 1
end

line = out.lines.find { |l| l.start_with?("ids:") }
unless line
  puts "GATE FAIL [moe-kquant]: no `ids:` line in output"
  puts out
  exit 1
end

all_ids = line.sub("ids:", "").split.map(&:to_i)
prompt  = PROMPT_IDS.split.map(&:to_i)
cont    = all_ids[prompt.length..] || []

if cont.length < N
  puts "GATE FAIL [moe-kquant]: expected #{N} continuation ids, got #{cont.length}"
  puts "  ids: #{all_ids.join(' ')}"
  exit 1
end

# longest run of a single repeated token in the continuation
max_run = 1
run = 1
cont.each_cons(2) { |a, b| run = (a == b ? run + 1 : 1); max_run = [max_run, run].max }
distinct = cont.uniq.length

degenerate = (cont.first == DEGEN_TOKEN) || (distinct < MIN_DISTINCT) || (max_run > MAX_RUN)

puts "[moe-kquant] continuation: #{cont.join(' ')}"
puts "[moe-kquant] distinct=#{distinct}/#{cont.length}  max_run=#{max_run}  first=#{cont.first}"

if degenerate
  puts "GATE FAIL [moe-kquant]: degenerate output — attention head collapse regressed."
  puts "  (pre-fix signature: token #{DEGEN_TOKEN} repeated; head_nbytes returned 0 for K-quants)"
  exit 1
end

puts "GATE PASS [moe-kquant]: coherent K-quant MoE decode (no head-stride collapse)."
exit 0
