# examples/05_eval_logprobs.rb — what does the model think comes next?
#
# WHAT YOU'LL SEE: a prompt is prefilled through the KV-cache engine,
# then at the next position the model's full distribution is computed
# and the top-K candidates printed as log-probabilities:
#
#   logprob: 198 -2.5708273628290854
#   logprob: 19 -2.7158488014277182
#   ...
#
# (id 198 is "\n" for SmolLM2 — after "1 100 200" a newline is its best
# guess.) Logprobs are the building block for perplexity, calibration
# curves, and API logprobs=true responses.
#
# HOW LONG: ~3 s (CPU). Needs one llama-family GGUF (`toy list`).
#
#   make example_05
#   GGUF=data/smollm2-135m-f32.gguf ./examples/example_05_eval_logprobs
#
# WHAT TO TWEAK (env, no recompile):
#   TOP_K=10               more candidates
#   PROMPT_IDS="1 2 3 4"   a different prefill (raw token ids)
#
# THE API — same ToyLM as 04, two decode calls:
#   lm.decode_step(id, pos)                      — prefill, one position
#   lm.decode_step_with_logprobs(id, pos, k)     — logits → log_softmax
#                                                  → top-K (deterministic
#                                                  CPU math; this is what
#                                                  `toy eval` gates)
# The CLI form of this example is `toy eval <model.gguf>`.

require_relative "../lib/toy/models/arch"
require_relative "../lib/toy/models/transformer_lm"

GGUF  = ENV["GGUF"]  || "data/smollm2-135m-f32.gguf"
TOP_K = (ENV["TOP_K"] || "5").to_i
PROMPT_IDS = ENV["PROMPT_IDS"] || "1 100 200"

# Fail LOUD before compute (spinel-dev#17: silent File.read of a
# missing path returns "").
if !File.exist?(GGUF)
  puts "05_eval_logprobs: model GGUF not found: " + GGUF
  puts "  point GGUF= at any llama-family GGUF (`toy list` shows local caches)."
  exit 1
end

arch = Arch.load_or_fail(GGUF, "05_eval_logprobs")
puts arch.summary

lm = ToyLM.new(arch, :cpu)
lm.load(GGUF)

# Parse + validate the prefill ids.
parts = PROMPT_IDS.split(" ")
ids = [0]; ids.pop
j = 0
while j < parts.length
  if parts[j].length > 0
    idv = parts[j].to_i
    if idv < 0 || idv >= arch.vocab_size
      puts "05_eval_logprobs: PROMPT_IDS token " + parts[j] +
           " out of range [0," + arch.vocab_size.to_s + ")"
      exit 1
    end
    ids.push(idv)
  end
  j = j + 1
end
if ids.length == 0
  puts "05_eval_logprobs: PROMPT_IDS parsed to no token ids"
  exit 1
end

# Prefill: feed each prompt id at its position (order is load-bearing —
# the KV cache is built up step by step).
i = 0
while i < ids.length
  lm.decode_step(ids[i], i)
  i = i + 1
end
print "prefill:"
i = 0
while i < ids.length
  print " " + ids[i].to_s
  i = i + 1
end
puts "  (" + ids.length.to_s + " positions)"

# One more decode, returning the top-K (id, logprob) pairs at the next
# position. log_softmax is max-shifted for stability; ties break
# first-seen — fully deterministic.
result   = lm.decode_step_with_logprobs(0, ids.length, TOP_K)
top_ids  = result[2]
top_vals = result[3]

k = 0
while k < top_ids.length
  puts "logprob: " + top_ids[k].to_s + " " + top_vals[k].to_s
  k = k + 1
end
