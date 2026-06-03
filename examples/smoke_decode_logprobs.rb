# toy#decode-logprobs (#151) — smoke for decode_step_with_logprobs.
# Drives a tiny prefill, then a single decode step, and prints the top-5
# (token_id, logprob) pairs. Tep's future /v1/chat/completions with
# `logprobs=true` consumes the same building block.
#
#   make examples/smoke_decode_logprobs
#   GGUF=data/llama-3.2-1b-native.gguf ./examples/smoke_decode_logprobs

require_relative "../lib/toy/models/arch"
require_relative "../lib/transformer_lm"

GGUF = ENV["GGUF"] || "data/llama-3.2-1b-native.gguf"
TOP_K = (ENV["TOP_K"] || "5").to_i

STDOUT.sync = true
puts "smoke_decode_logprobs start"
puts "Arch.from_gguf …"
arch = Arch.from_gguf(GGUF)
if arch == nil
  puts "smoke_decode_logprobs: could not load " + GGUF
  exit 1
end
puts arch.summary

puts "ToyLM.new + load …"
lm = ToyLM.new(arch, :cpu)
lm.load(GGUF)
puts "loaded"

# 3-token prefill, then decode once at pos=3.
ids = [1, 100, 200]
i = 0
while i < ids.length
  puts "decode_step " + ids[i].to_s + " @ " + i.to_s
  lm.decode_step(ids[i], i)
  i = i + 1
end
puts "prefill done"

# Now ask for logprobs at the next position.
result   = lm.decode_step_with_logprobs(0, ids.length, TOP_K)
logits   = result[0]
logprobs = result[1]
top_ids  = result[2]
top_vals = result[3]

puts "top-" + TOP_K.to_s + " (id, logprob):"
k = 0
while k < top_ids.length
  puts "  [" + top_ids[k].to_s + ", " + top_vals[k].to_s + "]"
  k = k + 1
end

# Sanity: argmax logit == top_ids[0] (logsumexp is monotonic).
argmax = 0
mv     = logits.flat[0]
j      = 1
while j < logits.ncols
  if logits.flat[j] > mv
    mv = logits.flat[j]
    argmax = j
  end
  j = j + 1
end
if argmax == top_ids[0]
  puts "argmax(logits) == top-1 id OK"
else
  puts "WARN: argmax(logits)=" + argmax.to_s + " but top-1 id=" + top_ids[0].to_s
end
puts "decode_step_with_logprobs OK"
