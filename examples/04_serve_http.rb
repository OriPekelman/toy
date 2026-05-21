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
# Current status (2026-05-21): the Tep HTTP layer has a regression
# on Spinel f5bc710 where handler bodies render as a static
# placeholder instead of the value the handler returned. The
# `tep_demo/openai_api_smollm2` and `tep_demo/hello` binaries are
# also affected. This example will build cleanly but the response
# body will be wrong until the upstream Tep+Spinel compatibility is
# restored. The SHAPE here (how to wire the model into Tep) is
# preserved as the reference. Track via the Tep repo.

require_relative "../tep_demo/_tep_lib/tep"
require_relative "../lib/toy"
require_relative "../lib/toy_smollm2"
require_relative "../lib/toy_smollm2_loader"
require_relative "../lib/toy_smollm2_ffi_kv"

GGUF = ENV["GGUF"] || "data/smollm2-135m-native.gguf"
PORT = (ENV["PORT"] || "4567").to_i

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
