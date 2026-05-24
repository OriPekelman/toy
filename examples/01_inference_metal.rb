# examples/01_inference_metal.rb — Metal mirror of examples/01_inference.rb.
#
# Runs an open-weight LM end-to-end on Apple Silicon's Metal backend.
# Same shape as the CPU example, but every compute graph runs through
# ggml-metal kernels on the GPU instead of ggml-cpu on the CPU. With
# an embedded tokenizer (use --with-tokenizer at GGUF-conversion time)
# we encode PROMPT, generate N_NEW tokens, decode back. Without it we
# fall back to a fixed five-ID prompt + raw IDs.
#
#   make setup-ggml-metal           # one-time
#   make example_inference_metal
#   ./examples/example_inference_metal
#   PROMPT="Once upon a time" ./examples/example_inference_metal
#   GGUF=data/llama-3.2-1b-tok.gguf N_NEW=32 ./examples/example_inference_metal
#
# First run on a given binary triggers a one-time ~15s metallib JIT
# compile (the ggml-metal shaders embedded into the static archive get
# compiled by the Metal driver at first device load). Subsequent runs
# read the cached compiled kernels.
#
# Exits via TinyNNMetal.tnn_force_exit (= _exit) to skip cxa_finalize
# — ggml-metal's static device vector destructor asserts on a non-
# empty residency set, which the cached-engine pattern guarantees we
# have. The OS reclaims everything cleanly; see tinynn_backend_metal.m
# for the full rationale.

require_relative "../lib/arch"
require_relative "../lib/tinynn_metal"
require_relative "../lib/transformer_lm_metal"
require_relative "../lib/tokenizer"

GGUF   = ENV["GGUF"]   || "data/smollm2-135m-f32.gguf"
PROMPT = ENV["PROMPT"] || "Once upon a time"
N_NEW  = (ENV["N_NEW"] || "16").to_i

arch = Arch.from_gguf(GGUF)
if arch == nil
  puts "example_inference_metal: could not load " + GGUF +
       " — set GGUF= to a valid file (see examples/example_list_models)."
  TinyNNMetal.tnn_force_exit(1)
end
puts arch.summary

lm = ToyLMMetal.new(arch)
lm.load(GGUF)

# Try to load an embedded tokenizer. If absent, fall back to hardcoded
# SmolLM2 prompt IDs and emit raw IDs.
tok = Tokenizer.from_gguf(GGUF)

if tok.present
  in_ids = tok.encode(PROMPT)
  puts "prompt: " + PROMPT.inspect + " → " + in_ids.length.to_s + " tokens"
  out_ids = lm.generate(in_ids, N_NEW)
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

TinyNNMetal.tnn_force_exit(0)
