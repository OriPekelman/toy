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

DEFAULT_GGUF = "data/smollm2-135m-f32.gguf"
GGUF_ENV     = ENV["GGUF"]
GGUF         = GGUF_ENV || DEFAULT_GGUF
PROMPT       = ENV["PROMPT"] || "Once upon a time"
N_NEW        = (ENV["N_NEW"] || "16").to_i

# Auto-fallback: if the default GGUF is missing and the user didn't pass
# GGUF=, scan the standard caches via TinyNNMetal.tnn_list_ggufs (the
# Metal binary loads tinynn_metal not tinynn, so we can't `require`
# model_index.rb here without dragging in the CPU module).
gguf = GGUF
if GGUF_ENV == nil && TinyNNMetal.tnn_file_size(gguf) == 0
  home = ENV["HOME"] || "/"
  # Seed-then-pop to pin Spinel's type inference to String[].
  search = [""]; search.pop
  env = ENV["TOY_MODEL_DIR"]
  if env != nil && env.length > 0; search.push(env); end
  search.push("./data")
  search.push("./models")
  search.push(home + "/.cache/huggingface/hub")
  search.push(home + "/.ollama/models")
  search.push(home + "/.lmstudio/models")
  search.push(home + "/models")
  found = ""
  si = 0
  while si < search.length && found.length == 0
    blob = TinyNNMetal.tnn_list_ggufs(search[si])
    if blob != nil && blob.length > 0
      lines = blob.split("\n")
      li = 0
      while li < lines.length
        ln = lines[li]
        if ln.length > 0
          # Only accept if Arch loads — skip non-llama-family GGUFs.
          test_arch = Arch.from_gguf(ln)
          if test_arch != nil && test_arch.vocab_size > 0 &&
             test_arch.d_model > 0 && test_arch.n_layers > 0
            found = ln
            li = lines.length
          end
        end
        li = li + 1
      end
    end
    si = si + 1
  end
  if found.length > 0
    gguf = found
    puts "[example_inference_metal] no GGUF= set and " + DEFAULT_GGUF + " missing."
    puts "[example_inference_metal] auto-selected: " + gguf
    is_instruct = gguf.include?("instruct") || gguf.include?("Instruct") ||
                  gguf.include?("INSTRUCT") || gguf.include?("-it-") ||
                  gguf.include?("-chat") || gguf.include?("Chat")
    if is_instruct
      puts "[example_inference_metal] NOTE: this looks like an instruction-tuned"
      puts "[example_inference_metal]   model. Raw completion prompts may produce"
      puts "[example_inference_metal]   incoherent text. Pass GGUF= to a base"
      puts "[example_inference_metal]   checkpoint, or PROMPT= with the model's"
      puts "[example_inference_metal]   chat template."
    end
  end
end
arch = Arch.from_gguf(gguf)
if arch == nil
  puts "example_inference_metal: could not load " + gguf +
       " — set GGUF= to a valid file (see `toy list`),"
  puts "  or fetch one with `toy fetch <hf-repo> <file.gguf>`."
  TinyNNMetal.tnn_force_exit(1)
end
puts arch.summary

lm = ToyLMMetal.new(arch)
lm.load(gguf)

# Try to load an embedded tokenizer. If absent, fall back to hardcoded
# SmolLM2 prompt IDs and emit raw IDs.
tok = Tokenizer.from_gguf(gguf)

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
