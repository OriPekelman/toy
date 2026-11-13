# lib/toy/serve/openai/toy_backend.rb -- toy's Tep::Llm::OpenAI::Backend
# (toy#30: the hand-rolled /v1 handlers collapse onto tep's battery).
#
# token_ids carries the generated IDs (tep's CompletionsHandler emits
# choices[0].ids when token_ids is non-empty — the tep#209 contract). With
# a GGUF that embeds a tokenizer the backend is also a TEXT backend
# (toy#209): `text` is decoded, generation stops at EOS, and chat is on.
# Without one it is ids-only, as before. Constructed with (state, model_name)
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

  # toy#209: chat needs the GGUF's embedded tokenizer (template text in,
  # decoded text out). Without one /v1/chat/completions stays 501.
  def supports_chat?
    @state.tok.present
  end

  # /v1/completions. The prompt is token ids (tep's handler reads `prompt`
  # as an int array only). The reply always carries choices[0].ids; when the
  # GGUF embeds a tokenizer it also carries the decoded `text`, and
  # generation stops at an end-of-generation token (finish_reason "stop").
  def generate_from_tokens(model, token_ids, sampling)
    comp = Tep::Llm::OpenAI::Completion.new
    comp.id = api_gen_id("cmpl")
    if token_ids.length == 0
      return comp
    end
    # max_tokens: 0 or absent => 16 (unchanged). The ceiling is the
    # context, applied in run_generation (it was a literal 256).
    n_new = sampling.max_tokens
    if n_new <= 0; n_new = 16; end
    run_generation(comp, token_ids, n_new)
    comp
  end

  # /v1/chat/completions (toy#209): messages -> chat template -> ids ->
  # greedy decode to an end-of-turn token -> text.
  def chat_completion(req)
    comp = Tep::Llm::OpenAI::Completion.new
    comp.id = api_gen_id("chatcmpl")
    body = req.raw_body
    token_ids = chat_prompt_ids(body)
    if token_ids.length == 0
      return comp
    end
    run_generation(comp, token_ids, chat_max_tokens(body, token_ids.length))
    # A chat reply is text; the ids stay out of the chat envelope.
    comp.token_ids = [0]
    comp.token_ids.delete_at(0)
    comp
  end

  # Streaming chat: one delta per generated token, cut at UTF-8 character
  # boundaries (a byte-level BPE token can end mid-character).
  def chat_completion_stream(req, sink)
    body = req.raw_body
    token_ids = chat_prompt_ids(body)
    if token_ids.length == 0
      return 0
    end
    n_new = fit_context(token_ids.length, chat_max_tokens(body, token_ids.length))
    if n_new <= 0
      return 0
    end
    t_start = TinyNN.tnn_events_now_seconds
    out_ids = [0]
    out_ids.pop
    emitted = 0
    id = api_first_id(token_ids)
    step = 0
    while step < n_new
      if api_stop_id?(id)
        break
      end
      out_ids.push(id)
      text = @state.tok.decode(out_ids)
      safe = api_utf8_safe_len(text)
      if safe > emitted
        sink.emit_token(text.byteslice(emitted, safe - emitted))
        emitted = safe
      end
      step = step + 1
      if step < n_new
        id = api_next_id(id, token_ids.length + out_ids.length - 1)
      end
    end
    t_end = TinyNN.tnn_events_now_seconds
    emit_request_event(token_ids.length, out_ids.length,
                       ((t_end - t_start) * 1.0e6).to_i, n_new)
    out_ids.length
  end

  # ---- shared pieces ----

  # Messages -> template text -> ids. Empty when the body has no messages.
  def chat_prompt_ids(body)
    msgs  = Tep::Llm::OpenAI.parse_messages(body)
    pairs = [["", ""]]
    pairs.delete_at(0)
    i = 0
    while i < msgs.length
      pairs.push([msgs[i].role, msgs[i].content])
      i = i + 1
    end
    if pairs.length == 0
      empty = [0]
      empty.pop
      return empty
    end
    @state.tok.encode(ToyChatTemplate.apply(@state.chat_family, pairs, true))
  end

  # Chat max_tokens: `max_completion_tokens` or `max_tokens` when given,
  # otherwise whatever the context has left after the prompt.
  def chat_max_tokens(body, n_prompt)
    n_new = 0
    if SpinelKit::Json.has_key?(body, "max_completion_tokens")
      n_new = SpinelKit::Json.get_int(body, "max_completion_tokens")
    elsif SpinelKit::Json.has_key?(body, "max_tokens")
      n_new = SpinelKit::Json.get_int(body, "max_tokens")
    end
    if n_new <= 0
      n_new = MAX_T - n_prompt
    end
    n_new
  end

  # toy#207: the request must fit the allocated context. Returns how many
  # tokens may be generated: `n_new` cut to what fits, or 0 when the prompt
  # alone does not fit (logged; nothing is generated, and never a truncated
  # prompt decoded as if it were the whole one).
  def fit_context(n_prompt, n_new)
    if n_prompt >= MAX_T
      puts "[openai_api_llama] request refused: prompt " + n_prompt.to_s +
           " tokens does not fit context " + MAX_T.to_s + " (start with --ctx N)"
      return 0
    end
    if n_prompt + n_new > MAX_T
      return MAX_T - n_prompt
    end
    n_new
  end

  # Generate into `comp`: ids, text (when a tokenizer is embedded), usage,
  # finish_reason, and the per-request event.
  def run_generation(comp, token_ids, n_new_req)
    n_new = fit_context(token_ids.length, n_new_req)
    if n_new <= 0
      # tep's backend API has no error channel (Completion carries only
      # finish_reason), so the reason goes there and to the log.
      comp.prompt_tokens = token_ids.length
      comp.finish_reason = "context_length_exceeded"
      return 0
    end

    t_start = TinyNN.tnn_events_now_seconds
    new_ids = api_generate_ids(token_ids, n_new)
    t_end   = TinyNN.tnn_events_now_seconds
    emit_request_event(token_ids.length, new_ids.length,
                       ((t_end - t_start) * 1.0e6).to_i, n_new)

    comp.token_ids         = new_ids
    comp.prompt_tokens     = token_ids.length
    comp.completion_tokens = new_ids.length
    comp.finish_reason     = @state.finish
    if @state.tok.present
      comp.text = @state.tok.decode(new_ids)
    end
    0
  end

  # ONE eval/serve/request event per completion (toy/v1). FILE-only side
  # effect; response bytes unaffected. Guarded by the process-global C
  # events state. Byte-identical to the retired handler's.
  def emit_request_event(n_prompt, n_completion, latency_us, n_new)
    if TinyNN.tnn_events_active == 1
      @state.req_seq = @state.req_seq + 1
      req_id = "req-" + @state.req_seq.to_s
      ev  = "{\"kind\":\"eval\",\"phase\":\"serve\""
      ev = ev + ",\"t\":"    + TinyNN.tnn_events_now_seconds.to_s
      ev = ev + ",\"name\":\"request\""
      ev = ev + ",\"extra\":{"
      ev = ev +   "\"model\":\"" + @model_name + "\""
      ev = ev +   ",\"prompt_tokens\":"     + n_prompt.to_s
      ev = ev +   ",\"completion_tokens\":" + n_completion.to_s
      ev = ev +   ",\"latency_us\":"        + latency_us.to_s
      ev = ev +   ",\"sampling\":{\"max_tokens\":" + n_new.to_s + "}"
      ev = ev +   ",\"request_id\":\"" + req_id + "\""
      ev = ev + "}}"
      TinyNN.tnn_events_emit(ev)
    end
    0
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
