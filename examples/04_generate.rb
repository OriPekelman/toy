# examples/04_generate.rb — load a GGUF, decode tokens, print text.
#
# WHAT YOU'LL SEE: a real model loaded from a GGUF file (arch + shape
# read from its header), the prompt run through the KV-cache decode
# engine one position at a time, and the greedy continuation printed —
# as text when the GGUF embeds its tokenizer, as raw token ids when not.
#
# HOW LONG: ~5 s for 16 tokens of SmolLM2-135M (CPU). One model file
# needed — `toy list` shows what's cached, or fetch one:
#   toy fetch bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf
#
#   make example_04
#   GGUF=data/smollm2-135m-f32.gguf ./examples/example_04_generate
#
# WHAT TO TWEAK (env, no recompile):
#   PROMPT="The capital of France is"    your prompt (tokenizer models)
#   PROMPT_IDS="1 100 200"               raw ids (tokenizer-less models)
#   N_NEW=64                             generate more tokens
#
# THE API — three objects:
#   Arch.load_or_fail(gguf, who)  — read + validate the GGUF header
#   ToyLM.new(arch, :cpu)         — the model wrapper; .load mmaps weights
#   lm.generate(ids, n)           — greedy KV-cache decode (argmax; the
#                                   deterministic path `toy infer` gates)
# Tokenizer.from_gguf reads the embedded BPE when present. The CLI form
# of this example is `toy infer <model.gguf>`.

require_relative "../lib/toy/models/arch"
require_relative "../lib/toy/models/transformer_lm"
require_relative "../lib/toy/io/tokenizer"

GGUF   = ENV["GGUF"]   || "data/smollm2-135m-f32.gguf"
PROMPT = ENV["PROMPT"] || "Once upon a time"
N_NEW  = (ENV["N_NEW"] || "16").to_i
PROMPT_IDS = ENV["PROMPT_IDS"] || ""

# Fail LOUD before compute (spinel-dev#17: silent File.read of a
# missing path returns "").
if !File.exist?(GGUF)
  puts "04_generate: model GGUF not found: " + GGUF
  puts "  point GGUF= at any llama-family GGUF. `toy list` shows local caches;"
  puts "  toy fetch bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf"
  exit 1
end

arch = Arch.load_or_fail(GGUF, "04_generate")
puts arch.summary

lm = ToyLM.new(arch, :cpu)
lm.load(GGUF)

tok = Tokenizer.from_gguf(GGUF)

if PROMPT_IDS.length > 0
  # Raw-ids path (tokenizer-less GGUFs, e.g. your own 01 checkpoints).
  parts = PROMPT_IDS.split(" ")
  in_ids = [0]; in_ids.pop
  j = 0
  while j < parts.length
    if parts[j].length > 0
      idv = parts[j].to_i
      if idv < 0 || idv >= arch.vocab_size
        puts "04_generate: PROMPT_IDS token " + parts[j] +
             " out of range [0," + arch.vocab_size.to_s + ")"
        exit 1
      end
      in_ids.push(idv)
    end
    j = j + 1
  end
  out_ids = lm.generate(in_ids, N_NEW)
  print "ids:"
  k = 0
  while k < out_ids.length
    print " " + out_ids[k].to_s
    k = k + 1
  end
  puts ""
elsif tok.present
  in_ids = tok.encode(PROMPT)
  puts "prompt: " + PROMPT.inspect + " -> " + in_ids.length.to_s + " tokens"
  out_ids = lm.generate(in_ids, N_NEW)   # greedy; returns prompt + new
  puts "text: " + tok.decode(out_ids)
else
  puts "04_generate: " + GGUF + " has no embedded tokenizer, so a string"
  puts "  prompt cannot be encoded. Pass raw ids: PROMPT_IDS=\"1 100 200\""
  exit 1
end
