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
require_relative "../lib/model_index"

DEFAULT_GGUF = "data/smollm2-135m-f32.gguf"
GGUF_ENV     = ENV["GGUF"]
GGUF         = GGUF_ENV || DEFAULT_GGUF
PROMPT       = ENV["PROMPT"] || "Once upon a time"
N_NEW        = (ENV["N_NEW"] || "16").to_i

# Resolve the GGUF: if the user passed GGUF= explicitly we honor it; if
# the default is missing on disk we fall back to whatever ModelIndex
# finds in the local caches (HF / Ollama / LM Studio / data / models).
# Keeps a fresh clone from face-planting with "no such file".
gguf = GGUF
if GGUF_ENV == nil && TinyNN.tnn_file_size(gguf) == 0
  entries = ModelIndex.scan_sources(ModelIndex.default_sources)
  if entries.length > 0
    gguf = entries[0].path
    puts "[example_inference] no GGUF= set and " + DEFAULT_GGUF + " missing."
    puts "[example_inference] auto-selected: " + entries[0].name +
         " (" + entries[0].source + ")"
    puts "[example_inference]   " + gguf
    # Heuristic warning: Instruct/Chat models with a raw completion prompt
    # produce incoherent text. Cheap check — name-based.
    lname = gguf
    is_instruct = lname.include?("instruct") || lname.include?("Instruct") ||
                  lname.include?("INSTRUCT") || lname.include?("-it-") ||
                  lname.include?("-chat") || lname.include?("Chat")
    if is_instruct
      puts "[example_inference] NOTE: this looks like an instruction-tuned model."
      puts "[example_inference]   Raw completion prompts may produce incoherent"
      puts "[example_inference]   text. Pass GGUF= to a base checkpoint, or"
      puts "[example_inference]   PROMPT= with the model's chat template."
    end
  end
end
arch = Arch.from_gguf(gguf)
if arch == nil
  puts "example_inference: could not load " + gguf +
       " — set GGUF= to a valid file (see examples/example_list_models),"
  puts "  or run `make hello` for a guided first-run."
  exit 1
end
puts arch.summary

lm = ToyLM.new(arch, :cpu)
lm.load(gguf)

# Try to load an embedded tokenizer. If absent (the converter was run
# without --with-tokenizer), fall back to the hardcoded SmolLM2 prompt
# IDs and emit raw IDs instead of decoded text.
tok = Tokenizer.from_gguf(gguf)

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
