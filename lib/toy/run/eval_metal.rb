# lib/toy/run/eval_metal.rb — Spinel-compiled Metal evaluation COMPUTE runner.
#
# Metal sibling of lib/toy/run/eval.rb (libexec/toy-eval). The CRuby CLI shell
# (lib/toy/core/cli/eval.rb) selects this binary when invoked with
# `--device metal` on macOS: it builds libexec/toy-eval-metal and shells out
# with the SAME controlled ENV (GGUF/TOP_K). DEVICE is irrelevant here — this
# binary IS the metal path.
#
# HAND-WRITTEN, NOT mechanically mirrored: ToyLMMetal's ctor arity is 1 and it
# has NO decode_step_with_logprobs (unlike ToyLM). So this file is DELIBERATELY
# ABSENT from MIRRORABLE in prep/gen_cuda_mirror.rb, exactly like the CPU runner.
#
# OPTION (B): we inline ToyLogProbs here (decode_step → log_softmax → top_k)
# rather than add decode_step_with_logprobs to ToyLMMetal. This (a) avoids
# touching the Metal class, and (b) keeps this a SINGLE-TYPE file (only
# ToyLMMetal referenced) so the Spinel polymorphic-lm-var miscompile
# (landmine #16) cannot fire. transformer_lm_metal.rb does NOT pull
# toy_logprobs, so we require it explicitly. ToyLogProbs is backend-agnostic CPU.
#
# This reproduces ToyLM#decode_step_with_logprobs (transformer_lm.rb:182-190)
# EXACTLY: decode_step at the frozen pos → ToyLogProbs.log_softmax →
# ToyLogProbs.top_k → the same "logprob: <id> <val>" line shape.
#
# DETERMINISM: pure Metal F32 forward + ToyLogProbs.log_softmax (max-shift
# stable) + manual top-K (strict-< first-seen tie-break). No sampler/seed.
# prep/metal_gate.rb gates the top-K id ORDERING metal-vs-cpu.
# RUNTIME-UNVERIFIED on gx10 (Linux, no Apple frameworks) — gate on the Mac.

require_relative "../models/arch"
require_relative "../models/transformer_lm_metal"
require_relative "../dev/toy_logprobs"

GGUF  = ENV["GGUF"] || "data/smollm2-135m-f32.gguf"
TOP_K = (ENV["TOP_K"] || "5").to_i

arch = Arch.from_gguf(GGUF)
if arch == nil
  puts "toy-eval: could not load " + GGUF +
       " — set GGUF= to a valid file (see `toy list`)."
  exit 1
end

lm = ToyLMMetal.new(arch)
lm.load(GGUF)

# Frozen eval point: EXACT same prefill order as the CPU runner. KV-cache state
# is op-order load-bearing — the sequential decode_step calls must match.
ids = [1, 100, 200]
i = 0
while i < ids.length
  lm.decode_step(ids[i], i)
  i = i + 1
end

logits   = lm.decode_step(0, ids.length)
logprobs = ToyLogProbs.log_softmax(logits)
pair     = ToyLogProbs.top_k(logprobs, TOP_K)
top_ids  = pair[0]
top_vals = pair[1]

k = 0
while k < top_ids.length
  puts "logprob: " + top_ids[k].to_s + " " + top_vals[k].to_s
  k = k + 1
end
