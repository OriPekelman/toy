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
# WHAT IT ASSERTS (re-anchored 2026-11-13, toy#206): the FIRST generated
# token. The prompt is "The capital of France is"; llama.cpp's greedy first
# token on this file is 36062 ("Paris"), and so is Toy's. With the collapse
# put back (head_nbytes forced to a zero stride) it is 20065, followed by
# `20065 20065 20065 42859 …`.
#
# The gate used to assert "enough distinct tokens, no long run" over 10
# tokens. That passed only because `generate` fed the last prompt token twice
# (fixed in toy#206), which produced varied but wrong text
# (`2320 658 15 11368 …`). The corrected decode is `36062 8 753 1051` and then
# a run of newlines, where llama.cpp continues `15 510 38479 1171 49348 …`
# (". The capital of Spain is Madrid."). So Toy and llama.cpp agree on the
# first token of this model and DIFFER from the second: an open accuracy gap
# on OLMoE (the per-head QK-norm approximation is the first suspect), printed
# below on every run and not asserted on.
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
FIRST_TOKEN = 36062   # "Paris" — llama.cpp's greedy first token for this prompt
LLAMA_CPP   = [36062, 15, 510, 38479, 1171, 49348, 261, 19008, 6992, 15].freeze

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

degenerate = (cont.first == DEGEN_TOKEN) || (cont.first != FIRST_TOKEN)
agree = 0
agree += 1 while agree < cont.length && cont[agree] == LLAMA_CPP[agree]

puts "[moe-kquant] continuation: #{cont.join(' ')}"
puts "[moe-kquant] distinct=#{distinct}/#{cont.length}  max_run=#{max_run}  first=#{cont.first}"

puts "[moe-kquant] llama.cpp    : #{LLAMA_CPP.join(' ')}  (agrees on the first #{agree} of #{cont.length}; open gap, see header)"

if degenerate
  puts "GATE FAIL [moe-kquant]: first token #{cont.first}, expected #{FIRST_TOKEN} — attention head collapse regressed."
  puts "  (pre-fix signature: token #{DEGEN_TOKEN} repeated; head_nbytes returned 0 for K-quants)"
  exit 1
end

puts "GATE PASS [moe-kquant]: K-quant MoE decode gives llama.cpp's first token (no head-stride collapse)."
exit 0
