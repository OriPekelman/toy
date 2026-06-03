# lib/toy/run/eval.rb — Spinel-compiled evaluation COMPUTE runner.
#
# This is the lib-side home of `toy eval`'s compute. The CRuby CLI shell
# (lib/toy/core/cli/eval.rb) cannot compute in-process — every ffi_lib-
# bearing lib crashes under MRI — so it locates the toy root, builds this
# runner (`make libexec/toy-eval`), and shells out to it via Open3 with a
# controlled ENV. This is the SAME PATTERN infer/train follow.
#
# SLICE 1 SCOPE: single-model per-token logprobs. The cleanest deterministic
# eval — one model, no checkpoint provenance, no sampler/seed. CE/perplexity
# over a --data corpus and the two-checkpoint LMC eval are LATER slices.
#
# CONTRACT (read from ENV only — lib-vs-example scope, no experiment config
# baked in):
#   GGUF   — path to the model (required; the CLI always passes it)
#   TOP_K  — number of top logprobs to report (default 5)
#
# The prefill IDs [1, 100, 200] and the decode position (= ids.length) are
# HARDCODED constants here, NOT flags — they are the frozen eval point, the
# analog of infer.rb's hardcoded fallback IDs and train.rb's hardcoded SHAPE
# (lib-vs-example scope). Changing them silently alters the gated logprobs.
#
# Backend: CPU only. decode_step / decode_step_with_logprobs are CPU-only
# (transformer_lm.rb prints + returns on :cuda); CUDA/Metal eval lives in a
# later slice with a different runner. Hence this file is deliberately ABSENT
# from MIRRORABLE in prep/gen_cuda_mirror.rb (exactly like infer/train).
#
# DETERMINISM: decode_step_with_logprobs is a pure CPU f32 forward +
# ToyLogProbs.log_softmax (max-shift stable) + manual partial top-K sort
# (strict-< first-seen tie-break). No sampler, no rand, no seed. The output
# is byte-for-byte reproducible — this is what prep/eval_gate.rb gates
# against a recorded baseline.
#
# OUTPUT (byte-exact prefix lines the CLI parses): one flat line per rank,
# in returned descending-logprob order:
#   "logprob: <id> <logprob>"
# All human chatter is omitted; the gated lines are ONLY the "logprob: " ones.

require_relative "../models/arch"
require_relative "../models/transformer_lm"

GGUF  = ENV["GGUF"] || "data/smollm2-135m-f32.gguf"
TOP_K = (ENV["TOP_K"] || "5").to_i

arch = Arch.from_gguf(GGUF)
if arch == nil
  puts "toy-eval: could not load " + GGUF +
       " — set GGUF= to a valid file (see `toy list`)."
  exit 1
end

lm = ToyLM.new(arch, :cpu)
lm.load(GGUF)

# Frozen eval point: a 3-token prefill, then logprobs at the next position.
# Op-order is load-bearing — the KV-cache state depends on sequential
# decode_step calls in this exact order.
ids = [1, 100, 200]
i = 0
while i < ids.length
  lm.decode_step(ids[i], i)
  i = i + 1
end

result   = lm.decode_step_with_logprobs(0, ids.length, TOP_K)
top_ids  = result[2]
top_vals = result[3]

k = 0
while k < top_ids.length
  puts "logprob: " + top_ids[k].to_s + " " + top_vals[k].to_s
  k = k + 1
end
