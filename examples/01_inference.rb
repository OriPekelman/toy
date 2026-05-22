# Run an open-weight model end-to-end: text in, text out. When the GGUF
# embeds a tokenizer (use prep/convert_smollm2_to_gguf.py
# --with-tokenizer), we encode PROMPT, generate N_NEW tokens, and decode
# back. Without an embedded tokenizer we fall back to the canonical
# five-ID prompt and print raw IDs.
#
#   make example_inference
#   PROMPT="Once upon a time" ./examples/example_inference
#   GGUF=data/llama-3.2-1b-tok.gguf ./examples/example_inference
#   N_NEW=32 ./examples/example_inference
#
# On CUDA, change `:cpu` → `:cuda` and build with
# `make example_inference_cuda` (the only difference is the FFI
# backend module).

require_relative "../lib/arch"
require_relative "../lib/transformer_lm"
require_relative "../lib/tokenizer"

GGUF   = ENV["GGUF"]   || "data/smollm2-135m-f32.gguf"
PROMPT = ENV["PROMPT"] || "Once upon a time"
N_NEW  = (ENV["N_NEW"] || "16").to_i

arch = Arch.from_gguf(GGUF)
if arch == nil
  puts "example_inference: could not load " + GGUF +
       " — set GGUF= to a valid file (see examples/example_list_models)."
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
