# lib/toy/serve/openai/handlers.rb -- the app-mounted Tep::Handler
# subclasses that sit ALONGSIDE tep's OpenAI battery: Health (/health)
# and Index (/).
#
# toy#30: the OpenAI endpoints (Models, Completions, ChatCompletions,
# Embeddings) are now served by tep's Tep::Llm::OpenAI battery via
# ToyBackend + Server.serve! (lib/toy/serve/openai/backend.rb); their
# hand-rolled handlers were removed here. Health + Index stay toy-owned
# because they're app concerns, not part of the OpenAI Backend contract.
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
    "<li><code>POST /v1/completions</code> &mdash; body <code>{\"prompt\":[int,...],\"max_tokens\":N}</code> &rarr; <code>choices[0].ids</code></li>" +
    "<li><code>POST /v1/chat/completions</code> &mdash; 501 (tokenizer required)</li>" +
    "<li><code>POST /v1/embeddings</code> &mdash; body <code>{\"input\":[int,...]}</code> &rarr; mean-pooled vector</li>" +
    "<li><code>GET /v1/models</code></li>" +
    "<li><code>GET /health</code></li>" +
    "</ul></body></html>"
  end
end
