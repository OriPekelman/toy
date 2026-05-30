# lib/toy/run/infer.rb — Spinel-compiled inference COMPUTE runner.
#
# This is the lib-side home of `toy infer`'s compute. The CRuby CLI shell
# (lib/toy/core/cli/infer.rb) cannot compute in-process — every ffi_lib-
# bearing lib crashes under MRI — so it locates the toy root, builds this
# runner (`make libexec/toy-infer`), and shells out to it via Open3 with a
# controlled ENV. This is the PATTERN train/eval/serve will follow.
#
# CONTRACT (read from ENV only — lib-vs-example scope, no experiment config
# baked in):
#   GGUF   — path to the model (required; the CLI always passes it)
#   PROMPT — input text (used only when the GGUF embeds a tokenizer)
#   N_NEW  — number of tokens to generate (default 16)
#
# Backend: CPU only. CUDA/Metal inference lives in separate hand-written
# classes (ToyLMCuda / ToyLMMetal) with different ctor arity, so this runner
# is NOT mechanically mirrorable; a --device runner is a later slice. Hence
# this file is deliberately ABSENT from MIRRORABLE in prep/gen_cuda_mirror.rb.
#
# DETERMINISM: generate(..) with no sampler_config → Sampler.argmax (greedy,
# first-max-wins, no rand/temperature/seed). The output is byte-for-byte
# reproducible — this is what prep/infer_gate.rb gates against a recorded
# baseline.
#
# OUTPUT (byte-exact prefixes the CLI parses):
#   "text: <decoded>"  when the GGUF embeds a tokenizer
#   "ids: <id> <id>…"  otherwise (raw IDs, space-prefixed)
#
# DELIBERATELY NOT inherited from the retired examples/01_inference.rb: its
# ModelIndex auto-select fallback (a silent fallback the never-mask rule
# forbids). This runner requires an explicit GGUF and fails loud on a bad one.

require_relative "../../arch"
require_relative "../../transformer_lm"
require_relative "../../tokenizer"

GGUF  = ENV["GGUF"] || "data/smollm2-135m-f32.gguf"
PROMPT = ENV["PROMPT"] || "Once upon a time"
N_NEW  = (ENV["N_NEW"] || "16").to_i

arch = Arch.from_gguf(GGUF)
if arch == nil
  puts "toy-infer: could not load " + GGUF +
       " — set GGUF= to a valid file (see `toy list`)."
  exit 1
end
puts arch.summary

lm = ToyLM.new(arch, :cpu)
lm.load(GGUF)

# Try to load an embedded tokenizer. If absent (the converter was run
# without --with-tokenizer), fall back to the hardcoded SmolLM2 prompt
# IDs and emit raw IDs instead of decoded text.
tok = Tokenizer.from_gguf(GGUF)

if tok.present
  in_ids = tok.encode(PROMPT)
  puts "prompt: " + PROMPT.inspect + " → " + in_ids.length.to_s + " tokens"
  out_ids = lm.generate(in_ids, N_NEW)
  # The model returns prompt + continuation; decode the whole thing.
  puts "text: " + tok.decode(out_ids)
else
  puts "no tokenizer in GGUF; re-convert with --with-tokenizer for text I/O"
  ids = lm.generate([6403, 1980, 253, 655, 28], N_NEW)
  print "ids:"
  k = 0
  while k < ids.length
    print " " + ids[k].to_s
    k = k + 1
  end
  puts ""
end
