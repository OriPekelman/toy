# lib/toy/serve/openai/toy_backend.rb -- toy's Tep::Llm::OpenAI::Backend
# (toy#30: the hand-rolled /v1 handlers collapse onto tep's battery).
#
# IDs-only backend: token_ids carries the generated IDs and text stays ""
# (tep's CompletionsHandler emits choices[0].ids when token_ids is
# non-empty — the tep#209 contract). Constructed with (state, model_name)
# so no cross-file constant resolution (same pattern as the retired
# EmbeddingsHandler). Generation + id minting reuse the top-level helpers
# in server.rb (api_generate_ids / api_gen_id) — the greedy argmax path
# the serve gate's recorded baseline pins.
#
# The per-request toy/v1 event moves here from the retired handler,
# byte-identical (kind:eval phase:serve name:request, req-N request_id
# sequence). It rides the C emitter (tnn_events_*), NOT tep's
# Tep::Events — serve.rb calls Server.serve!("") so tep's own event
# stream stays disabled (its run_start lacks phase/schema and its
# latency is second-resolution; serve_events_gate pins ours).
#
# Empty prompt: tep's handler has no error channel to 400 through (tep
# issue filed), so an empty token_ids returns an EMPTY completion
# (deterministic, no junk generation) instead of the old handler's 400.
#
# Spinel hygiene (#16): String-concat JSON, no #{} interpolation; typed
# empty-array seeds; pure while loops in the pooling path.

class ToyBackend < Tep::Llm::OpenAI::Backend
  def initialize(state, model_name)
    @state      = state
    @model_name = model_name
  end

  def list_models
    out = [""]
    out.delete_at(0)
    out.push(@model_name)
    out
  end

  def model_owner
    "toy"
  end

  def device_kind
    "cpu"
  end

  def supports_embeddings?
    true
  end

  def generate_from_tokens(model, token_ids, sampling)
    comp = Tep::Llm::OpenAI::Completion.new
    comp.id = api_gen_id("cmpl")
    if token_ids.length == 0
      return comp
    end

    # max_tokens: same defaults/clamps as the retired handler (0 or
    # absent => 16; ceiling 256).
    n_new = sampling.max_tokens
    if n_new <= 0; n_new = 16; end
    if n_new > 256; n_new = 256; end

    t_start = TinyNN.tnn_events_now_seconds
    new_ids = api_generate_ids(token_ids, n_new)
    t_end   = TinyNN.tnn_events_now_seconds
    latency_us = ((t_end - t_start) * 1.0e6).to_i

    # ONE eval/serve/request event per completion (toy/v1). FILE-only
    # side effect; response bytes unaffected. Guarded by the process-
    # global C events state. Byte-identical to the retired handler's.
    if TinyNN.tnn_events_active == 1
      @state.req_seq = @state.req_seq + 1
      req_id = "req-" + @state.req_seq.to_s
      ev  = "{\"kind\":\"eval\",\"phase\":\"serve\""
      ev = ev + ",\"t\":"    + TinyNN.tnn_events_now_seconds.to_s
      ev = ev + ",\"name\":\"request\""
      ev = ev + ",\"extra\":{"
      ev = ev +   "\"model\":\"" + @model_name + "\""
      ev = ev +   ",\"prompt_tokens\":"     + token_ids.length.to_s
      ev = ev +   ",\"completion_tokens\":" + new_ids.length.to_s
      ev = ev +   ",\"latency_us\":"        + latency_us.to_s
      ev = ev +   ",\"sampling\":{\"max_tokens\":" + n_new.to_s + "}"
      ev = ev +   ",\"request_id\":\"" + req_id + "\""
      ev = ev + "}}"
      TinyNN.tnn_events_emit(ev)
    end

    comp.token_ids         = new_ids
    comp.prompt_tokens     = token_ids.length
    comp.completion_tokens = new_ids.length
    comp.finish_reason     = "length"
    comp
  end

  # Mean-pooled d_model vector over the input token embeddings
  # (dequantize-aware lookup; moved verbatim from the retired
  # EmbeddingsHandler). A lookup failure returns an EMPTY array —
  # tep's EmbeddingsHandler has no error channel either; empty is
  # loud downstream (zero-length embedding) rather than a silently
  # partial vector.
  def generate_embeddings(model, token_ids)
    out = [0.0]
    out.delete_at(0)
    if token_ids.length == 0
      return out
    end

    d_model = @state.cfg.d_model
    sess    = @state.kv.sess
    t_embed = @state.kv.t_token_embed

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
      rc = TinyNN.tnn_embed_lookup_to_doubles(sess, t_embed, token_ids[i], tok_buf, d_model)
      if rc != 0
        # never-mask: warn loud with context; empty result (not partial).
        STDERR.puts "[toy_backend] embed_lookup rc=" + rc.to_s +
                    " at token index " + i.to_s + " (id=" + token_ids[i].to_s + ")"
        return out
      end
      k = 0
      while k < d_model
        sum_buf[k] = sum_buf[k] + tok_buf[k]
        k = k + 1
      end
      i = i + 1
    end

    inv_n = 1.0 / token_ids.length.to_f
    kk = 0
    while kk < d_model
      out.push(sum_buf[kk] * inv_n)
      kk = kk + 1
    end
    out
  end
end
