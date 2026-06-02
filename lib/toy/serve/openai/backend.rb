# lib/toy/serve/openai/backend.rb -- ToyBackend: toy's adapter onto
# tep's Tep::Llm::OpenAI::Backend (toy#30 convergence).
#
# Replaces the hand-rolled ModelsHandler / CompletionsHandler /
# ChatCompletionsHandler / EmbeddingsHandler: tep's battery now owns the
# OpenAI wire shapes, streaming, and the /v1/* routes (Server.use +
# Server.serve!), so toy supplies ONLY the model-specific surface here.
#
# IDs-only contract preserved: generate_from_tokens returns the generated
# token IDs in Completion#token_ids (tep emits them as choices[0].ids,
# needs tep >= 0.11.x's token-ID completion support); text stays "" and
# finish_reason is "length" (greedy, fixed n_new). The per-request
# eval/serve/request event (toy/v1) is emitted HERE -- moved verbatim from
# the old CompletionsHandler -- via TinyNN.tnn_events_emit, so toy keeps
# its C-side events pipeline. tep's own Events emitter is left
# unconfigured (Server.serve! with no path => disabled, zero overhead).
#
# Spinel hygiene (#16): String-concat JSON, no #{} interpolation, longhand
# while loops, collision-free names. Reads STATE + api_generate_ids from
# server.rb (same compilation unit after bin/tep inlining).

class ToyBackend < Tep::Llm::OpenAI::Backend
  # /v1/models lists the single loaded model under owned_by "toy".
  def list_models
    [STATE.model_name]
  end

  def model_owner
    "toy"
  end

  def device_kind
    "cpu"
  end

  # Token-level greedy generation. Returns a Completion carrying the
  # generated IDs (token_ids) -- tep's CompletionsHandler emits them as
  # choices[0].ids. Emits the per-request toy/v1 serving event as a
  # side effect (gated by tnn_events_active; FILE-only, no response bytes).
  def generate_from_tokens(model, token_ids, sampling)
    n_new = sampling.max_tokens
    if n_new <= 0; n_new = 16; end
    if n_new > 256; n_new = 256; end

    t_start = TinyNN.tnn_events_now_seconds
    new_ids = api_generate_ids(token_ids, n_new)
    t_end   = TinyNN.tnn_events_now_seconds

    prompt_len     = token_ids.length
    completion_len = new_ids.length
    latency_us     = ((t_end - t_start) * 1.0e6).to_i

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

    c = Tep::Llm::OpenAI::Completion.new
    c.token_ids         = new_ids
    c.prompt_tokens     = prompt_len
    c.completion_tokens = completion_len
    c.finish_reason     = "length"
    c
  end

  def supports_embeddings?
    true
  end

  # Mean-pooled embedding over the input token IDs (moved verbatim from
  # the old EmbeddingsHandler). Returns Array[Float] of length d_model;
  # tep's EmbeddingsHandler formats the OpenAI envelope around it.
  #
  # NOTE (toy#30 delta): the old handler returned HTTP 500 on a non-zero
  # tnn_embed_lookup_to_doubles rc ("never mask, fail loud"). The Backend
  # contract returns a vector, not an HTTP status, so a lookup failure now
  # surfaces as a zero-contribution token rather than a 500. rc is rare
  # (valid-table + in-range id); revisit if it needs loud failure.
  def generate_embeddings(model, token_ids)
    d_model = STATE.cfg.d_model
    sess    = STATE.kv.sess
    t_embed = STATE.kv.t_token_embed

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
    while i < token_ids.length
      TinyNN.tnn_embed_lookup_to_doubles(sess, t_embed, token_ids[i], tok_buf, d_model)
      k = 0
      while k < d_model
        sum_buf[k] = sum_buf[k] + tok_buf[k]
        k = k + 1
      end
      i = i + 1
    end

    inv_n = 1.0 / token_ids.length.to_f
    m = 0
    while m < d_model
      sum_buf[m] = sum_buf[m] * inv_n
      m = m + 1
    end
    sum_buf
  end
end
