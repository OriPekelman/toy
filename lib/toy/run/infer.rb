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
require_relative "../io/tokenizer"

GGUF  = ENV["GGUF"] || "data/smollm2-135m-f32.gguf"
PROMPT = ENV["PROMPT"] || "Once upon a time"
N_NEW  = (ENV["N_NEW"] || "16").to_i
# PROMPT_IDS — whitespace-separated NUMERIC token ids. Used for
# tokenizer-less models (e.g. from-scratch checkpoints, vocab=627, no
# embedded tokenizer): a string prompt cannot be tokenized, so the
# caller passes raw ids. Empty when unset.
PROMPT_IDS = ENV["PROMPT_IDS"] || ""

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

# all_digits? — true iff `s` is non-empty and every char is 0..9. Explicit
# char scan (NOT exception-based Integer(s)) per the Spinel landmines.
def all_digits?(s)
  return false if s.length == 0
  i = 0
  while i < s.length
    c = s[i]
    if c < "0" || c > "9"
      return false
    end
    i = i + 1
  end
  true
end

if PROMPT_IDS.length > 0
  # NUMERIC-ids path (tokenizer-less or explicitly id-driven). Parse +
  # validate EVERY id is an integer in [0, vocab). Fail loud on any bad
  # id (never silently mistokenize / clamp).
  parts = PROMPT_IDS.split(" ")
  parsed = [0]; parsed.pop
  j = 0
  while j < parts.length
    tokstr = parts[j]
    if tokstr.length > 0
      if all_digits?(tokstr) == false
        puts "toy-infer: PROMPT_IDS contains invalid token '" + tokstr +
             "' (must be integer in [0," + arch.vocab_size.to_s + "))"
        exit 1
      end
      idv = tokstr.to_i
      if idv < 0 || idv >= arch.vocab_size
        puts "toy-infer: PROMPT_IDS contains invalid token '" + tokstr +
             "' (must be integer in [0," + arch.vocab_size.to_s + "))"
        exit 1
      end
      parsed.push(idv)
    end
    j = j + 1
  end
  if parsed.length == 0
    puts "toy-infer: PROMPT_IDS parsed to no token ids"
    exit 1
  end
  out_ids = lm.generate(parsed, N_NEW)
  print "ids:"
  k = 0
  while k < out_ids.length
    print " " + out_ids[k].to_s
    k = k + 1
  end
  puts ""
elsif tok.present
  in_ids = tok.encode(PROMPT)
  puts "prompt: " + PROMPT.inspect + " → " + in_ids.length.to_s + " tokens"
  out_ids = lm.generate(in_ids, N_NEW)
  # The model returns prompt + continuation; decode the whole thing.
  puts "text: " + tok.decode(out_ids)
else
  # Tokenizer-less model AND no PROMPT_IDS: a string prompt cannot be
  # tokenized. Fail loud (never the old silent hardcoded-id fallback).
  puts "toy-infer: model has no embedded tokenizer; a string prompt cannot " +
       "be tokenized. Pass numeric token IDs via --prompt-ids (PROMPT_IDS=...) " +
       "or re-convert with --with-tokenizer."
  exit 1
end
