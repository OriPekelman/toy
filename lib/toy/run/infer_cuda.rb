# lib/toy/run/infer_cuda.rb — Spinel-compiled CUDA inference COMPUTE runner.
#
# CUDA sibling of lib/toy/run/infer.rb (libexec/toy-infer). The CRuby CLI shell
# (lib/toy/core/cli/infer.rb) selects this binary when invoked with
# `--device cuda`: it builds libexec/toy-infer-cuda and shells out with the
# SAME controlled ENV (GGUF/PROMPT/N_NEW). DEVICE is irrelevant in here — this
# binary IS the cuda path.
#
# HAND-WRITTEN, NOT mechanically mirrored: ToyLMCuda's ctor arity is 1
# (ToyLMCuda.new(arch)), unlike ToyLM.new(arch, :cpu), so gen_cuda_mirror.rb
# cannot derive this file from the CPU runner. It is therefore DELIBERATELY
# ABSENT from MIRRORABLE in prep/gen_cuda_mirror.rb — exactly like the CPU
# runner is, for the inverse reason.
#
# SINGLE-TYPE FILE: only ToyLMCuda is referenced (no ToyLM), so the Spinel
# polymorphic-lm-var miscompile (landmine #16) cannot fire. Each device has its
# own self-contained runner; the CLI does the dispatch by picking the binary.
#
# OUTPUT: byte-identical prefixes to the CPU runner ("text: …" / "ids: …") so
# the CLI's scan_line and prep/infer_gate.rb parse both branches identically.
# Hand-built output only — no #{} interpolation in any emitted line.
#
# DETERMINISM: generate(..) with no sampler_config → Sampler.argmax (greedy,
# first-max-wins, no rand/seed). The CUDA F32 forward is expected to match the
# CPU runner's generated token IDs; prep/infer_gate.rb (TOY_GATE_CUDA=1) gates
# that discrete invariant cuda-vs-cpu in the same run.

require_relative "../models/arch"
require_relative "../models/transformer_lm_cuda"
require_relative "../io/tokenizer"

GGUF  = ENV["GGUF"] || "data/smollm2-135m-f32.gguf"
PROMPT = ENV["PROMPT"] || "Once upon a time"
N_NEW  = (ENV["N_NEW"] || "16").to_i
# PROMPT_IDS — whitespace-separated NUMERIC token ids, for tokenizer-less
# models (parity with the CPU runner). Empty when unset.
PROMPT_IDS = ENV["PROMPT_IDS"] || ""

# all_digits? — true iff `s` is non-empty and every char is 0..9 (explicit
# char scan, not exception-based Integer(s), per the Spinel landmines).
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

arch = Arch.from_gguf(GGUF)
if arch == nil
  puts "toy-infer: could not load " + GGUF +
       " — set GGUF= to a valid file (see `toy list`)."
  exit 1
end
puts arch.summary

lm = ToyLMCuda.new(arch)
lm.load(GGUF)

tok = Tokenizer.from_gguf(GGUF)

if PROMPT_IDS.length > 0
  # NUMERIC-ids path (tokenizer-less or explicitly id-driven). Parse +
  # validate EVERY id is an integer in [0, vocab). Fail loud on any bad id
  # (never silently mistokenize / clamp) — parity with the CPU runner.
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
  puts "text: " + tok.decode(out_ids)
else
  # Tokenizer-less AND no PROMPT_IDS: fail loud (never the old silent
  # hardcoded-id fallback). Parity with the CPU runner.
  puts "toy-infer: model has no embedded tokenizer; a string prompt cannot " +
       "be tokenized. Pass numeric token IDs via --prompt-ids (PROMPT_IDS=...) " +
       "or re-convert with --with-tokenizer."
  exit 1
end
