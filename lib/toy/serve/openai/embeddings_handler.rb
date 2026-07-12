# lib/toy/serve/openai/embeddings_handler.rb -- OpenAI-shape
# /v1/embeddings handler.
#
# MOVED from tep_demo/embeddings_handler.rb (P4 toy serve). Decodes the input
# token-id array with Toy::Json.get_int_array (toy#44; the former local
# ApiJson shim is retired — SpinelKit's decoder replaces it). Constructed with
# (STATE, MODEL_NAME) so the handler doesn't need cross-file constant resolution.
#
# Contract:
# - Body: {"input": [int, int, ...], "model": "..."} -- IDs only,
#   tokenize client-side (matches the server's overall "IDs only"
#   policy).
# - Lookup: per-token embedding via tnn_embed_lookup_to_doubles
#   (dequantize-aware; works on f32 + Q8 + Q4 tables).
# - Pooling: mean over the input tokens -> single d_model vector.
# - Response shape: OpenAI /v1/embeddings v1.
#
# Spinel notes:
# - State is passed in at construction so the handler doesn't need
#   cross-file constant resolution (STATE is per-server).
# - The pooling loop uses pure while loops + Array<Float> buffers
#   seeded with pop-to-empty (the [0.0]; arr.pop type-pin pattern).

class EmbeddingsHandler < Tep::Handler
  def initialize(state, model_name)
    @state      = state
    @model_name = model_name
  end

  def handle(req, res)
    res.headers["Content-Type"] = "application/json"
    body = req.body

    ids = Toy::Json.get_int_array(body, "input")
    if ids.length == 0
      res.set_status(400)
      return "{\"error\":{\"message\":\"input must be a non-empty int array " +
             "(this server speaks IDs only; tokenize client-side)\"," +
             "\"type\":\"invalid_request_error\"}}\n"
    end

    d_model = @state.cfg.d_model
    sess    = @state.kv.sess
    t_embed = @state.kv.t_token_embed

    # Mean-pool buffer: sum across tokens, then divide.
    sum_buf = [0.0]; sum_buf.pop
    j0 = 0
    while j0 < d_model
      sum_buf.push(0.0)
      j0 = j0 + 1
    end
    tok_buf = [0.0]; tok_buf.pop
    j1 = 0
    while j1 < d_model
      tok_buf.push(0.0)
      j1 = j1 + 1
    end

    i = 0
    while i < ids.length
      rc = TinyNN.tnn_embed_lookup_to_doubles(sess, t_embed, ids[i], tok_buf, d_model)
      if rc != 0
        # Per the "never mask, fail loud" rule, surface as an error
        # response rather than silently returning a partial vector.
        res.set_status(500)
        return "{\"error\":{\"message\":\"embed_lookup rc=" + rc.to_s +
               " at token index " + i.to_s + " (id=" + ids[i].to_s +
               ")\",\"type\":\"server_error\"}}\n"
      end
      k = 0
      while k < d_model
        sum_buf[k] = sum_buf[k] + tok_buf[k]
        k = k + 1
      end
      i = i + 1
    end

    inv_n = 1.0 / ids.length.to_f
    out = "{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":["
    kk = 0
    while kk < d_model
      if kk > 0; out = out + ","; end
      out = out + (sum_buf[kk] * inv_n).to_s
      kk = kk + 1
    end
    out = out + "]}],\"model\":\"" + @model_name +
          "\",\"usage\":{\"prompt_tokens\":" + ids.length.to_s +
          ",\"total_tokens\":" + ids.length.to_s + "}}\n"
    out
  end
end
