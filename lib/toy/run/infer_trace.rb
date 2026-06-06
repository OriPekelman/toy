# lib/toy/run/infer_trace.rb — ggml#1506 localization runner. Loads a GGUF,
# enables the cache trace, and runs ONE decode_step per PROMPT_IDS token so the
# per-tap min/max/|mean|/nan dump prints for every layer. Diff the Q4_K_M dump
# against the q8_0 dump to find the first op where the K-quant run diverges.
#
#   make libexec/toy-infer-trace
#   GGUF=data/OLMoE-...-Q4_K_M.gguf PROMPT_IDS=261 libexec/toy-infer-trace
require_relative "../models/arch"
require_relative "../models/transformer_lm"

GGUF = ENV["GGUF"] || "data/smollm2-135m-f32.gguf"
PROMPT_IDS = ENV["PROMPT_IDS"] || "261"

arch = Arch.from_gguf(GGUF)
if arch == nil
  puts "infer_trace: could not load " + GGUF
  exit 1
end
puts arch.summary

lm = ToyLM.new(arch, :cpu)
lm.load(GGUF)
lm.kv_cpu.enable_trace!

parts = PROMPT_IDS.split(" ")
pos = 0
j = 0
while j < parts.length
  tokstr = parts[j]
  if tokstr.length > 0
    tid = tokstr.to_i
    puts "=== decode_step(token=" + tid.to_s + ", pos=" + pos.to_s + ") ==="
    lm.decode_step(tid, pos)
    pos = pos + 1
  end
  j = j + 1
end
