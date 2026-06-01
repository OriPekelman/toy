# lib/toy/serve/openai/handlers.rb -- Tep::Handler subclasses for the
# OpenAI-compatible endpoints.
#
# MOVED from tep_demo/openai_api_llama.rb:268-365 (P4 toy serve).
# Handlers: Health (/health), Models (/v1/models), Completions
# (/v1/completions -- choices[0].ids is the gate field), ChatCompletions
# (/v1/chat/completions -> 501), Index (/). The /v1/embeddings handler
# lives in embeddings_handler.rb.
#
# FIX during the move: the pre-existing HTML bug in the index page
# (openai_api_llama.rb:359-360) -- an orphaned String literal with no
# leading `+` followed by a stray trailing `++`. It is not gated either
# way (the gate hits /v1/completions, not /), so fixing it during the
# lift is safe and removes a latent syntax landmine.
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

class ModelsHandler < Tep::Handler
  def handle(req, res)
    res.headers["Content-Type"] = "application/json"
    "{\"object\":\"list\",\"data\":[{" +
      Tep::Json.encode_pair_str("id", STATE.model_name) + "," +
      Tep::Json.encode_pair_str("object", "model") + "," +
      Tep::Json.encode_pair_int("created", api_now_unix) + "," +
      Tep::Json.encode_pair_str("owned_by", "toy") +
    "}]}\n"
  end
end

class CompletionsHandler < Tep::Handler
  def handle(req, res)
    res.headers["Content-Type"] = "application/json"
    body = req.body

    # Accept the prompt as a JSON int array. OpenAI spec allows
    # `prompt: <int-array>` for pre-tokenized input; we require it.
    prompt_ids = ApiJson.get_int_array(body, "prompt")
    if prompt_ids.length == 0
      prompt_ids = ApiJson.get_int_array(body, "prompt_ids")
    end

    if prompt_ids.length == 0
      res.set_status(400)
      return "{\"error\":{\"message\":\"prompt must be a non-empty int array " +
             "(this server speaks IDs only; tokenize client-side)\"," +
             "\"type\":\"invalid_request_error\"}}\n"
    end

    n_new = 16
    if Tep::Json.has_key?(body, "max_tokens")
      n_new = Tep::Json.get_int(body, "max_tokens")
    end
    if n_new <= 0; n_new = 16; end
    if n_new > 256; n_new = 256; end

    # Measure generation latency across the single compute call (cheap: two
    # tnn_events_now_seconds reads regardless of whether events are active).
    t_start = TinyNN.tnn_events_now_seconds
    new_ids = api_generate_ids(prompt_ids, n_new)
    t_end   = TinyNN.tnn_events_now_seconds
    prompt_len = prompt_ids.length
    completion_len = new_ids.length
    latency_us = ((t_end - t_start) * 1.0e6).to_i

    # ONE eval/serve/request event per completion (toy/v1). FILE-only side
    # effect: the response String below is byte-UNCHANGED. Guarded by the
    # process-global C events state (tnn_events_active==1 only when the runner
    # opened events.jsonl, i.e. TAO_RUN_DIR was set), so the handler needs no
    # EVENTS-path knowledge. NO emit on the 400 empty-prompt early-return.
    if TinyNN.tnn_events_active == 1
      STATE.req_seq = STATE.req_seq + 1
      req_id = "req-" + STATE.req_seq.to_s
      ev  = "{\"kind\":\"eval\",\"phase\":\"serve\""
      ev = ev + ",\"t\":"    + TinyNN.tnn_events_now_seconds.to_s
      ev = ev + ",\"name\":\"request\""
      ev = ev + ",\"extra\":{"
      ev = ev +   "\"model\":\"" + STATE.model_name + "\""
      ev = ev +   ",\"prompt_tokens\":"     + prompt_len.to_s
      ev = ev +   ",\"completion_tokens\":" + completion_len.to_s
      ev = ev +   ",\"latency_us\":"        + latency_us.to_s
      ev = ev +   ",\"sampling\":{\"max_tokens\":" + n_new.to_s + "}"
      ev = ev +   ",\"request_id\":\"" + req_id + "\""
      ev = ev + "}}"
      TinyNN.tnn_events_emit(ev)
    end

    "{" +
      Tep::Json.encode_pair_str("id", api_gen_id("cmpl")) + "," +
      Tep::Json.encode_pair_str("object", "text_completion") + "," +
      Tep::Json.encode_pair_int("created", api_now_unix) + "," +
      Tep::Json.encode_pair_str("model", STATE.model_name) + "," +
      "\"choices\":[{\"index\":0," +
        Tep::Json.encode_pair_str("text", "") + "," +
        "\"ids\":" + Tep::Json.from_int_array(new_ids) + "," +
        "\"finish_reason\":\"length\"}]," +
      "\"usage\":{" +
        Tep::Json.encode_pair_int("prompt_tokens", prompt_len) + "," +
        Tep::Json.encode_pair_int("completion_tokens", completion_len) + "," +
        Tep::Json.encode_pair_int("total_tokens", prompt_len + completion_len) +
      "}}\n"
  end
end

class ChatCompletionsHandler < Tep::Handler
  def handle(req, res)
    res.headers["Content-Type"] = "application/json"
    res.set_status(501)
    "{\"error\":{\"message\":\"chat/completions requires a tokenizer; " +
    "this server speaks IDs only. Use POST /v1/completions with " +
    "prompt as an int array.\",\"type\":\"not_implemented\"}}\n"
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
