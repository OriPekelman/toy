# Serve a model over HTTP. POST a JSON body with `prompt` (array of
# token IDs) and `n` (number of tokens to generate); get back the
# generated IDs.
#
#   make example_serve
#   ./examples/example_serve &
#   curl -s localhost:4567/generate \
#     -H 'Content-Type: application/json' \
#     -d '{"prompt":[12092,4845,253,1429],"n":16}'
#   #  → {"ids":[198,198,504,808,2775,288,...]}
#
# The single binary embeds the model loader + decode loop + HTTP
# server. Drop it on a box, run, serve.
#
# Token-IDs in / token-IDs out by design — text↔ID work belongs
# client-side (or wire in lib/tokenizer.rb if you need it
# server-side). Same choice tep_demo/openai_api_smollm2 makes.
#
# Current status (2026-05-22): segfaults at startup on Spinel 8ff3a4e.
# Tep itself is fine (`tep_demo/hello` works end-to-end on the same
# Spinel). The crash is a Spinel codegen bug — `CFG_VOCAB = cfg.vocab`
# at top level hoists the const init above the `cfg = ...` assignment,
# so it reads from a zero-init struct and the call chain SIGSEGVs
# through FFI. Filed as matz/spinel#647.
#
# Workaround if you want to test the serve shape today: replace the
# `CFG_VOCAB = cfg.vocab` line with `cfg_vocab = cfg.vocab` (a local),
# then reference `cfg_vocab` in the handler. The example as written
# preserves the constant form intentionally as a regression check.

require_relative "../tep_demo/_tep_lib/tep"
require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv"

GGUF = ENV["GGUF"] || "data/smollm2-135m-native.gguf"
PORT = (ENV["PORT"] || "4567").to_i

if !File.exist?(GGUF)
  puts "example_serve: cannot find " + GGUF
  puts ""
  puts "Serving uses the mmap KV-cache path, which requires the"
  puts "GGUF to be in native HF layout (toy.ggml_native=true)."
  puts ""
  puts "Build one from the HuggingFace checkpoint:"
  puts ""
  puts "  ./prep/convert_smollm2_to_gguf.py --ggml-native \\"
  puts "      --out " + GGUF
  puts ""
  puts "Then re-run this example."
  exit 1
end

cfg   = SmolLM2ConfigLoader.read(GGUF)
flags = GGUFLoad.detect_smollm2_flags(GGUF)
gguf  = TinyNN.tnn_gguf_load(GGUF)
KV    = SmolLM2KVFFICache.new
KV.realize_for_mmap(gguf, cfg, 512, flags.untied, flags.qkv_bias)
CFG_VOCAB = cfg.vocab
puts "loaded " + GGUF + " — vocab=" + CFG_VOCAB.to_s

def generate(prompt_ids, n_new, vocab)
  pos = 0
  while pos < prompt_ids.length
    SmolLM2KV.decode_step(KV, prompt_ids[pos], pos)
    pos = pos + 1
  end
  out = [0]; out.pop
  step = 0
  while step < n_new
    last_id = step == 0 ? prompt_ids[prompt_ids.length - 1] : out[out.length - 1]
    logits = SmolLM2KV.decode_step(KV, last_id, prompt_ids.length + step)
    best = 0; best_v = logits.flat[0]; j = 1
    while j < vocab
      if logits.flat[j] > best_v; best_v = logits.flat[j]; best = j; end
      j = j + 1
    end
    out.push(best)
    step = step + 1
  end
  out
end

class GenerateHandler < Tep::Handler
  def handle(req, res)
    res.headers["Content-Type"] = "application/json"
    body = req.body
    prompt_ids = Tep::Json.get_int_array(body, "prompt")
    if prompt_ids.length == 0
      res.set_status(400)
      return "{\"error\":\"prompt must be a non-empty int array\"}\n"
    end
    n_new = 16
    if Tep::Json.has_key?(body, "n"); n_new = Tep::Json.get_int(body, "n"); end
    if n_new <= 0; n_new = 16; end
    if n_new > 256; n_new = 256; end
    ids = generate(prompt_ids, n_new, CFG_VOCAB)
    "{\"ids\":" + Tep::Json.from_int_array(ids) + "}\n"
  end
end

Tep.post "/generate", GenerateHandler.new
puts "serving on :" + PORT.to_s + " — POST /generate"
Tep.run!(PORT, 1, false)
