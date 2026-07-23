# lib/toy/serve/openai/handlers.rb -- the app-mounted Tep::Handler
# subclasses: Health (/health) + Index (/).
#
# The four /v1 endpoints moved to tep's OpenAI battery (toy#30):
# Server.use(ToyBackend) + serve! mounts models/completions/chat/
# embeddings — see toy_backend.rb. The hand-rolled Models/Completions/
# ChatCompletions handlers (originally lifted from
# tep_demo/openai_api_llama.rb in the P4 serve move) and the separate
# embeddings_handler.rb are retired; choices[0].ids now comes from
# Completion#token_ids (the tep#209 contract).
#
# Tep is consumed purely as transport. Spinel hygiene (#16): String-concat
# JSON, no #{} interpolation, no Struct.new.

class HealthHandler < Tep::Handler
  def handle(req, res)
    res.headers["Content-Type"] = "text/plain"
    if STATE.ready
      "ok\n"
    else
      res.set_status(503)
      "loading\n"
    end
  end
end

class IndexHandler < Tep::Handler
  def handle(req, res)
    res.headers["Content-Type"] = "text/html; charset=utf-8"
    "<!doctype html><html><head><title>toy openai-compat (IDs only)</title></head>" +
    "<body><h1>toy openai-compat API (ID-only)</h1>" +
    "<p>Model: <code>" + STATE.model_name + "</code> via direct GGUF&rarr;FFI loader.</p>" +
    "<p>Tokenize client-side; this server speaks integer token IDs only. " +
    "Run <code>prep/qwen25_tokens.py encode \"...\"</code> to get IDs.</p>" +
    "<p>Endpoints:</p><ul>" +
    "<li><code>POST /v1/completions</code> &mdash; body <code>{\"prompt\":[int,...],\"max_tokens\":N}</code></li>" +
    "<li><code>POST /v1/chat/completions</code> &mdash; 501 (tokenizer required)</li>" +
    "<li><code>POST /v1/embeddings</code> &mdash; body <code>{\"input\":[int,...]}</code> &rarr; mean-pooled vector</li>" +
    "<li><code>GET /v1/models</code></li>" +
    "<li><code>GET /health</code></li>" +
    "</ul></body></html>"
  end
end
