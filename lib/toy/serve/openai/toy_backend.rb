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
# in server.rb (the slot engine: api_slot_wait / api_slot_run, and
# api_gen_id) — the greedy argmax path the serve gate's recorded baseline
# pins. toy#217: a request holds a SLOT while it decodes; the decode steps
# are shared by every busy slot.
#
# The per-request toy/v1 event moves here from the retired handler,
# byte-identical (kind:eval phase:serve name:request, req-N request_id
# sequence). It rides the C emitter (tnn_events_*), NOT tep's
# Tep::Events — serve.rb calls Server.serve!("") so tep's own event
# stream stays disabled (its run_start lacks phase/schema and its
# latency is second-resolution; serve_events_gate pins ours).
#
# toy#216: the non-streamed /v1/completions and /v1/chat/completions are
# answered by ToyCompletionsHandler / ToyChatHandler at the end of this
# file (serve_text / serve_chat), which add logprobs, stop strings, a
# forced answer prefix and HTTP error replies. generate_from_tokens and
# chat_completion remain as the tep Backend interface but no route reaches
# them; streamed requests still go through tep's handlers to the *_stream
# methods here.
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
    begin
      run_generation(comp, token_ids, n_new)
    rescue => e
      request_failed(comp, e.message)
    end
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
    begin
      run_generation(comp, token_ids, chat_max_tokens(body, token_ids.length))
    rescue => e
      request_failed(comp, e.message)
    end
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
    emitted = 0
    n_out   = 0
    cached  = 0
    shared  = 0
    # toy#216: stop strings. Text that could still turn out to be the start
    # of a stop string is held back (`hold` bytes) until more arrives.
    stops = api_json_str_list(body, "stop")
    hold = 0
    si = 0
    while si < stops.length
      if stops[si].bytesize - 1 > hold
        hold = stops[si].bytesize - 1
      end
      si = si + 1
    end
    slot_i = -1
    begin
      if SERVE_TEST_FAULT_AT > 0 && token_ids.length == SERVE_TEST_FAULT_AT
        raise "injected test fault (SERVE_TEST_FAULT_AT)"
      end
      slot_i = api_slot_wait(token_ids, n_new)
      slot = SLOTS[slot_i]
      slot.stop_strs = stops
      cached = slot.cached
      seen = 0
      while true
        api_slot_turn(slot)
        if slot.err.length > 0
          raise slot.err
        end
        if slot.out_ids.length > seen || slot.done
          # Send what is new, up to a stop string and up to the last whole
          # UTF-8 character; the last `hold` bytes wait unless this is the end.
          seen = slot.out_ids.length
          text = @state.tok.decode(slot.out_ids)
          lim = text.bytesize
          if slot.stop_cut >= 0
            lim = slot.stop_cut
          elsif !slot.done
            lim = lim - hold
          end
          if lim < 0
            lim = 0
          end
          safe = api_utf8_safe_len(text.byteslice(0, lim))
          if safe > emitted
            sink.emit_token(text.byteslice(emitted, safe - emitted))
            emitted = safe
          end
        end
        if slot.done
          break
        end
        api_yield
      end
      n_out  = slot.out_ids.length
      shared = slot.shared_steps
    rescue => e
      # toy#214: the stream ends with what was sent so far.
      puts "[openai_api_llama] request failed: " + e.message
    end
    if slot_i >= 0
      api_slot_release(SLOTS[slot_i])
    end
    t_end = TinyNN.tnn_events_now_seconds
    emit_request_event(token_ids.length, n_out,
                       ((t_end - t_start) * 1.0e6).to_i, n_new, cached, slot_i, shared)
    n_out
  end

  # ---- shared pieces ----

  # toy#214: an exception inside one request must not end the process (a
  # packed artifact has no supervisor). tep's backend API has no error
  # channel, so the reply is an empty completion with finish_reason
  # "error", and the reason goes to the log. The KV bookkeeping stays
  # valid: a slot only lists rows that were written (api_pump).
  def request_failed(comp, msg)
    puts "[openai_api_llama] request failed: " + msg
    comp.text = ""
    comp.token_ids = [0]
    comp.token_ids.delete_at(0)
    comp.completion_tokens = 0
    comp.finish_reason = "error"
    0
  end

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
    # toy#216: a LAST message with role "assistant" is a partial answer to
    # continue (a forced answer prefix). The template is rendered for the
    # messages before it, the assistant turn is opened, and the partial
    # text follows with no end-of-turn marker; the reply is what comes next.
    last = pairs[pairs.length - 1]
    if last[0] == "assistant"
      pairs.pop
      return @state.tok.encode(ToyChatTemplate.apply(@state.chat_family, pairs, true) + last[1])
    end
    @state.tok.encode(ToyChatTemplate.apply(@state.chat_family, pairs, true))
  end

  # ---- toy#216: toy's own /v1/completions and /v1/chat/completions ----
  #
  # The two non-streamed endpoints are answered by ToyChatHandler /
  # ToyCompletionsHandler (below) instead of tep's handlers, because the
  # reply needs fields tep's envelope has no place for (logprobs, the
  # backend's finish_reason on chat) and real error responses.

  # POST /v1/chat/completions, not streamed.
  def serve_chat(body)
    rep = ToyReply.new
    rep.id = api_gen_id("chatcmpl")
    token_ids = chat_prompt_ids(body)
    if token_ids.length == 0
      return rep.fail(400, "invalid_request_error", "messages must be a non-empty array")
    end
    serve_generate(rep, body, token_ids, chat_max_tokens(body, token_ids.length))
    rep
  end

  # POST /v1/completions, not streamed. `prompt` is an array of token ids
  # or (toy#216, with an embedded tokenizer) a text, in which special
  # tokens written out ("<|im_start|>") are their single ids.
  def serve_text(body)
    rep = ToyReply.new
    rep.id = api_gen_id("cmpl")
    token_ids = [0]
    token_ids.pop
    pos = SpinelKit::Json.find_value_start(body, "prompt")
    if pos >= 0 && pos < body.length && body[pos] == "\""
      if !@state.tok.present
        return rep.fail(400, "invalid_request_error",
                        "a text prompt needs a GGUF with an embedded tokenizer; send token ids")
      end
      token_ids = @state.tok.encode(SpinelKit::Json.parse_str_value(body, pos))
    else
      token_ids = SpinelKit::Json.get_int_array(body, "prompt")
    end
    if token_ids.length == 0
      return rep.fail(400, "invalid_request_error",
                      "prompt must be a non-empty text or array of token ids")
    end
    n_new = SpinelKit::Json.get_int(body, "max_tokens")
    if n_new <= 0; n_new = 16; end
    serve_generate(rep, body, token_ids, n_new)
    rep.with_ids = true
    rep
  end

  # How many alternatives the request wants log-probabilities for: -1 when
  # it asks for none. Chat form: "logprobs": true with "top_logprobs": n.
  # Completions form: "logprobs": n.
  def lp_request(body)
    pos = SpinelKit::Json.find_value_start(body, "logprobs")
    if pos < 0 || pos >= body.length
      return -1
    end
    n = 0
    c = body[pos]
    if c == "t"
      n = 0
    elsif c >= "0" && c <= "9"
      n = SpinelKit::Json.parse_int_value(body, pos)
    else
      return -1
    end
    if SpinelKit::Json.has_key?(body, "top_logprobs")
      n = SpinelKit::Json.get_int(body, "top_logprobs")
    end
    if n < 0; n = 0; end
    if n > 20; n = 20; end
    n
  end

  # Generate into `rep` with the request's options: stop strings and
  # log-probabilities (toy#216), the context check (toy#207), and one
  # request event. A Ruby exception becomes a 500 reply (toy#214).
  def serve_generate(rep, body, token_ids, n_new_req)
    rep.prompt_tokens = token_ids.length
    stops = api_json_str_list(body, "stop")
    top   = lp_request(body)
    if (stops.length > 0 || top >= 0) && !@state.tok.present
      return rep.fail(400, "invalid_request_error",
                      "stop strings and logprobs need a GGUF with an embedded tokenizer")
    end
    n_new = fit_context(token_ids.length, n_new_req)
    if n_new <= 0
      return rep.fail(400, "context_length_exceeded",
                      "the prompt is " + token_ids.length.to_s + " tokens and the context is " +
                      MAX_T.to_s + " (start the server with --ctx N)")
    end
    slot_i = -1
    begin
      if SERVE_TEST_FAULT_AT > 0 && token_ids.length == SERVE_TEST_FAULT_AT
        raise "injected test fault (SERVE_TEST_FAULT_AT)"
      end
      t_start = TinyNN.tnn_events_now_seconds
      slot_i = api_slot_wait(token_ids, n_new)
      slot = SLOTS[slot_i]
      slot.stop_strs = stops
      slot.lp_top    = top
      api_slot_run(slot)
      if slot.err.length > 0
        raise slot.err
      end
      new_ids = slot.out_ids
      t_end   = TinyNN.tnn_events_now_seconds
      emit_request_event(token_ids.length, new_ids.length,
                         ((t_end - t_start) * 1.0e6).to_i, n_new, slot.cached, slot_i, slot.shared_steps)
      rep.completion_tokens = new_ids.length
      rep.finish = slot.finish
      if @state.tok.present
        text = @state.tok.decode(new_ids)
        if slot.stop_cut >= 0
          # The reply ends where the stop string began. Ids that reach
          # past that point are dropped from `ids` and from logprobs;
          # completion_tokens still counts every token generated.
          text = text.byteslice(0, slot.stop_cut)
          while new_ids.length > 0 && @state.tok.decode(new_ids).bytesize > slot.stop_cut
            new_ids.pop
          end
        end
        rep.text = text
        if top >= 0
          rep.lp_json = lp_json(slot, new_ids, top)
        end
      end
      rep.ids = new_ids
    rescue => e
      puts "[openai_api_llama] request failed: " + e.message
      rep.fail(500, "server_error", e.message)
    end
    if slot_i >= 0
      api_slot_release(SLOTS[slot_i])
    end
    0
  end

  # One token as a logprobs entry body: "token", "id", "logprob", "bytes".
  # "token" is "" when the token's bytes are not a whole number of UTF-8
  # characters on their own; "bytes" always carries them.
  def lp_fields(id, lp)
    piece = @state.tok.decode([id])
    bytes = "["
    i = 0
    while i < piece.bytesize
      if i > 0; bytes = bytes + ","; end
      bytes = bytes + piece.getbyte(i).to_s
      i = i + 1
    end
    bytes = bytes + "]"
    shown = piece
    if !api_utf8_valid?(piece)
      shown = ""
    end
    SpinelKit::Json.encode_pair_str("token", shown) + "," +
      SpinelKit::Json.encode_pair_int("id", id) + "," +
      "\"logprob\":" + lp.to_s + "," +
      "\"bytes\":" + bytes
  end

  # OpenAI's chat `logprobs` object for the reply's ids, from what the
  # decode steps recorded in the slot.
  def lp_json(slot, ids, top)
    out = "{\"content\":["
    i = 0
    while i < ids.length && i < slot.lp_tok.length
      if i > 0; out = out + ","; end
      out = out + "{" + lp_fields(ids[i], slot.lp_tok[i]) + ",\"top_logprobs\":["
      k = 0
      n = 0
      while k < top
        aid = slot.lp_alt_ids[i * top + k]
        if aid >= 0
          if n > 0; out = out + ","; end
          out = out + "{" + lp_fields(aid, slot.lp_alt_lps[i * top + k]) + "}"
          n = n + 1
        end
        k = k + 1
      end
      out = out + "]}"
      i = i + 1
    end
    out + "]}"
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

    if SERVE_TEST_FAULT_AT > 0 && token_ids.length == SERVE_TEST_FAULT_AT
      raise "injected test fault (SERVE_TEST_FAULT_AT)"
    end
    t_start = TinyNN.tnn_events_now_seconds
    slot = SLOTS[api_slot_wait(token_ids, n_new)]
    api_slot_run(slot)
    api_slot_release(slot)
    if slot.err.length > 0
      raise slot.err
    end
    new_ids = slot.out_ids
    t_end   = TinyNN.tnn_events_now_seconds
    emit_request_event(token_ids.length, new_ids.length,
                       ((t_end - t_start) * 1.0e6).to_i, n_new, slot.cached, slot.idx, slot.shared_steps)

    comp.token_ids         = new_ids
    comp.prompt_tokens     = token_ids.length
    comp.completion_tokens = new_ids.length
    comp.finish_reason     = slot.finish
    if @state.tok.present
      comp.text = @state.tok.decode(new_ids)
    end
    0
  end

  # ONE eval/serve/request event per completion (toy/v1). FILE-only side
  # effect; response bytes unaffected. Guarded by the process-global C
  # events state. Byte-identical to the retired handler's.
  # toy#217: `slot` is the slot that served it and `shared_steps` the decode
  # steps it shared with at least one other request.
  def emit_request_event(n_prompt, n_completion, latency_us, n_new, cached, slot, shared_steps)
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
      ev = ev +   ",\"cached_tokens\":"     + cached.to_s
      ev = ev +   ",\"slot\":"              + slot.to_s
      ev = ev +   ",\"shared_steps\":"      + shared_steps.to_s
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

# What one non-streamed request produced (toy#216). `status` 200 carries a
# completion; anything else carries an error message and type.
class ToyReply
  attr_accessor :status, :err_type, :err_msg, :id, :text, :ids, :with_ids
  attr_accessor :prompt_tokens, :completion_tokens, :finish, :lp_json

  def initialize
    @status   = 200
    @err_type = ""
    @err_msg  = ""
    @id       = ""
    @text     = ""
    @ids      = [0]
    @ids.pop
    @with_ids = false
    @prompt_tokens     = 0
    @completion_tokens = 0
    @finish   = "stop"
    @lp_json  = ""
  end

  def fail(status, type, msg)
    @status   = status
    @err_type = type
    @err_msg  = msg
    self
  end

  def error_json
    "{\"error\":{" +
      SpinelKit::Json.encode_pair_str("message", @err_msg) + "," +
      SpinelKit::Json.encode_pair_str("type", @err_type) +
    "}}"
  end

  def usage_json
    "\"usage\":{" +
      SpinelKit::Json.encode_pair_int("prompt_tokens", @prompt_tokens) + "," +
      SpinelKit::Json.encode_pair_int("completion_tokens", @completion_tokens) + "," +
      SpinelKit::Json.encode_pair_int("total_tokens", @prompt_tokens + @completion_tokens) +
    "}"
  end

  # `"logprobs":{...},` when the request asked for them, "" otherwise.
  def lp_frag
    if @lp_json.length == 0
      return ""
    end
    "\"logprobs\":" + @lp_json + ","
  end
end

def api_wants_stream?(body)
  Tep.str_find(body, "\"stream\":true", 0) >= 0 || Tep.str_find(body, "\"stream\": true", 0) >= 0
end

# POST /v1/chat/completions. Streamed requests go to tep's handler (which
# drives ToyBackend#chat_completion_stream); the rest is answered here.
class ToyChatHandler < Tep::Handler
  def initialize(backend)
    @backend = backend
    @stream  = Tep::Llm::OpenAI::ChatCompletionsHandler.new
  end

  def handle(req, res)
    body = req.raw_body
    if !@backend.supports_chat? || api_wants_stream?(body)
      return @stream.handle(req, res)
    end
    res.headers["Content-Type"] = "application/json"
    rep = @backend.serve_chat(body)
    if rep.status != 200
      res.set_status(rep.status)
      return rep.error_json
    end
    "{" +
      SpinelKit::Json.encode_pair_str("id", rep.id) + "," +
      SpinelKit::Json.encode_pair_str("object", "chat.completion") + "," +
      SpinelKit::Json.encode_pair_int("created", Time.now.to_i) + "," +
      SpinelKit::Json.encode_pair_str("model", SpinelKit::Json.get_str(body, "model")) + "," +
      "\"choices\":[{" +
        SpinelKit::Json.encode_pair_int("index", 0) + "," +
        "\"message\":{" +
          SpinelKit::Json.encode_pair_str("role", "assistant") + "," +
          SpinelKit::Json.encode_pair_str("content", rep.text) +
        "}," +
        rep.lp_frag +
        SpinelKit::Json.encode_pair_str("finish_reason", rep.finish) +
      "}]," +
      rep.usage_json +
    "}"
  end
end

# POST /v1/completions. Streamed requests go to tep's handler.
class ToyCompletionsHandler < Tep::Handler
  def initialize(backend)
    @backend = backend
    @stream  = Tep::Llm::OpenAI::CompletionsHandler.new
  end

  def handle(req, res)
    body = req.raw_body
    if api_wants_stream?(body)
      return @stream.handle(req, res)
    end
    res.headers["Content-Type"] = "application/json"
    rep = @backend.serve_text(body)
    if rep.status != 200
      res.set_status(rep.status)
      return rep.error_json
    end
    ids_frag = ""
    if rep.with_ids && rep.ids.length > 0
      ids_frag = "\"ids\":" + SpinelKit::Json.from_int_array(rep.ids) + ","
    end
    "{" +
      SpinelKit::Json.encode_pair_str("id", rep.id) + "," +
      SpinelKit::Json.encode_pair_str("object", "text_completion") + "," +
      SpinelKit::Json.encode_pair_int("created", Time.now.to_i) + "," +
      SpinelKit::Json.encode_pair_str("model", SpinelKit::Json.get_str(body, "model")) + "," +
      "\"choices\":[{" +
        SpinelKit::Json.encode_pair_int("index", 0) + "," +
        SpinelKit::Json.encode_pair_str("text", rep.text) + "," +
        ids_frag +
        rep.lp_frag +
        SpinelKit::Json.encode_pair_str("finish_reason", rep.finish) +
      "}]," +
      rep.usage_json +
    "}"
  end
end
