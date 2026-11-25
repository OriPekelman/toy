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
# them. toy#219: streamed requests are answered here too (ToyStreamer,
# serve_stream), so they carry logprobs, the finish_reason and the label.
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

  # ---- toy#219: streamed replies, written by toy ----
  #
  # A streamed request is checked before the stream opens (stream_check),
  # so a bad one is an HTTP error like its non-streamed form. After that
  # ToyStreamer#pump calls serve_stream, which writes every frame.

  # The checks serve_generate makes before it generates. Returns the number
  # of tokens to generate; on a refusal `rep` carries the error.
  def stream_check(rep, body, token_ids, n_new_req, chat)
    fmt_check(rep, body, chat)
    if rep.status != 200
      return 0
    end
    if !@state.tok.present
      rep.fail(400, "invalid_request_error",
               "a streamed reply needs a GGUF with an embedded tokenizer")
      return 0
    end
    n_new = fit_context(token_ids.length, n_new_req)
    if n_new <= 0
      rep.fail(400, "context_length_exceeded",
               "the prompt is " + token_ids.length.to_s + " tokens and the context is " +
               MAX_T.to_s + " (start the server with --ctx N)")
      return 0
    end
    n_new
  end

  # One SSE frame. `inner` is the body of choices[0]; `tail` is "" or more
  # top-level fields, each preceded by a comma.
  def stream_frame(id, model, chat, inner, tail)
    obj = "text_completion"
    if chat
      obj = "chat.completion.chunk"
    end
    "data: {" +
      SpinelKit::Json.encode_pair_str("id", id) + "," +
      SpinelKit::Json.encode_pair_str("object", obj) + "," +
      SpinelKit::Json.encode_pair_int("created", Time.now.to_i) + "," +
      SpinelKit::Json.encode_pair_str("model", model) + "," +
      "\"choices\":[{\"index\":0," + inner + "}]" + tail +
    "}\n\n"
  end

  # Generate and write the reply as SSE frames: text cut at UTF-8 character
  # boundaries (a byte-level BPE token can end mid-character) and held back
  # while it could still be the start of a stop string. A frame carries the
  # log-probabilities (when asked for) of the tokens whose text it completes,
  # so the entries of all frames are those of the non-streamed reply. The
  # last frame has the finish_reason, the usage and the "toy" label (toy#221).
  def serve_stream(out, body, token_ids, n_new, chat, id)
    model = SpinelKit::Json.get_str(body, "model")
    scale = ad_request(body)
    stops = api_json_str_list(body, "stop")
    top   = lp_request(body)
    hold = 0
    si = 0
    while si < stops.length
      if stops[si].bytesize - 1 > hold
        hold = stops[si].bytesize - 1
      end
      si = si + 1
    end
    if chat
      out.write(stream_frame(id, model, chat,
                             "\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null", ""))
    end
    t_start = TinyNN.tnn_events_now_seconds
    emitted = 0
    n_out   = 0
    cached  = 0
    shared  = 0
    finish  = "stop"
    failed  = ""
    # ends[i]: byte length of the text of the first i+1 generated tokens.
    ends = [0]
    ends.pop
    tok_sent = 0
    slot_i = -1
    begin
      if SERVE_TEST_FAULT_AT > 0 && token_ids.length == SERVE_TEST_FAULT_AT
        raise "injected test fault (SERVE_TEST_FAULT_AT)"
      end
      slot_i = api_slot_wait(token_ids, n_new, scale)
      slot = SLOTS[slot_i]
      slot.stop_strs = stops
      slot.lp_top    = top
      fmt_apply(slot, body, chat)
      cached = slot.cached
      while true
        api_slot_turn(slot)
        if slot.err.length > 0
          raise slot.err
        end
        if slot.out_ids.length > ends.length || slot.done
          text = @state.tok.decode(slot.out_ids)
          while ends.length < slot.out_ids.length
            if ends.length + 1 == slot.out_ids.length
              ends.push(text.bytesize)
            else
              part = [0]
              part.pop
              pi = 0
              while pi <= ends.length
                part.push(slot.out_ids[pi])
                pi = pi + 1
              end
              ends.push(@state.tok.decode(part).bytesize)
            end
          end
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
          # Tokens whose text is now wholly sent. At the end without a stop
          # string that is every token, as in the non-streamed reply.
          upto = tok_sent
          while upto < ends.length && ends[upto] <= safe
            upto = upto + 1
          end
          if slot.done && slot.stop_cut < 0
            upto = ends.length
          end
          if safe > emitted || upto > tok_sent
            piece = ""
            if safe > emitted
              piece = text.byteslice(emitted, safe - emitted)
            end
            inner = ""
            if chat
              inner = "\"delta\":{" + SpinelKit::Json.encode_pair_str("content", piece) + "},"
            else
              inner = SpinelKit::Json.encode_pair_str("text", piece) + ",\"ids\":["
              ti = tok_sent
              while ti < upto
                if ti > tok_sent; inner = inner + ","; end
                inner = inner + slot.out_ids[ti].to_s
                ti = ti + 1
              end
              inner = inner + "],"
            end
            if top >= 0
              inner = inner + "\"logprobs\":{\"content\":["
              ti = tok_sent
              while ti < upto && ti < slot.lp_tok.length
                if ti > tok_sent; inner = inner + ","; end
                inner = inner + lp_entry(slot, slot.out_ids[ti], ti, top)
                ti = ti + 1
              end
              inner = inner + "]},"
            end
            out.write(stream_frame(id, model, chat, inner + "\"finish_reason\":null", ""))
            emitted  = safe
            tok_sent = upto
          end
        end
        if slot.done
          break
        end
        api_yield
      end
      n_out  = slot.out_ids.length
      shared = slot.shared_steps
      finish = slot.finish
    rescue => e
      puts "[openai_api_llama] request failed: " + e.message
      failed = e.message
    end
    if slot_i >= 0
      api_slot_release(SLOTS[slot_i])
    end
    t_end = TinyNN.tnn_events_now_seconds
    emit_request_event(token_ids.length, n_out,
                       ((t_end - t_start) * 1.0e6).to_i, n_new, cached, slot_i, shared, scale)
    if failed.length > 0
      # toy#214: the stream ends with what was sent so far, then says why.
      out.write("data: {\"error\":{" +
                SpinelKit::Json.encode_pair_str("message", failed) + "," +
                SpinelKit::Json.encode_pair_str("type", "server_error") + "}}\n\n")
      return 0
    end
    last = "\"text\":\"\","
    if chat
      last = "\"delta\":{},"
    end
    usage = "\"usage\":{" +
      SpinelKit::Json.encode_pair_int("prompt_tokens", token_ids.length) + "," +
      SpinelKit::Json.encode_pair_int("completion_tokens", n_out) + "," +
      SpinelKit::Json.encode_pair_int("total_tokens", token_ids.length + n_out) +
    "}"
    out.write(stream_frame(id, model, chat,
                           last + SpinelKit::Json.encode_pair_str("finish_reason", finish),
                           "," + usage + "," + prov_json(scale)))
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
    serve_generate(rep, body, token_ids, chat_max_tokens(body, token_ids.length), true)
    rep
  end

  # POST /v1/completions, not streamed. `prompt` is an array of token ids
  # or (toy#216, with an embedded tokenizer) a text, in which special
  # tokens written out ("<|im_start|>") are their single ids.
  def serve_text(body)
    rep = ToyReply.new
    rep.id = api_gen_id("cmpl")
    token_ids = text_prompt_ids(rep, body)
    if rep.status != 200
      return rep
    end
    serve_generate(rep, body, token_ids, text_max_tokens(body), false)
    rep.with_ids = true
    rep
  end

  # The `prompt` of a /v1/completions request as ids. On a refusal `rep`
  # carries the error.
  def text_prompt_ids(rep, body)
    token_ids = [0]
    token_ids.pop
    pos = SpinelKit::Json.find_value_start(body, "prompt")
    if pos >= 0 && pos < body.length && body[pos] == "\""
      if !@state.tok.present
        rep.fail(400, "invalid_request_error",
                 "a text prompt needs a GGUF with an embedded tokenizer; send token ids")
        return token_ids
      end
      token_ids = @state.tok.encode(SpinelKit::Json.parse_str_value(body, pos))
    else
      token_ids = SpinelKit::Json.get_int_array(body, "prompt")
    end
    if token_ids.length == 0
      rep.fail(400, "invalid_request_error",
               "prompt must be a non-empty text or array of token ids")
    end
    token_ids
  end

  def text_max_tokens(body)
    n_new = SpinelKit::Json.get_int(body, "max_tokens")
    if n_new <= 0; n_new = 16; end
    n_new
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

  # toy#219: what `response_format` asks for: 0 free text (absent, null or
  # {"type":"text"}), 1 a JSON object ({"type":"json_object"}), -1 anything
  # else (a schema is not enforced, so it is refused rather than ignored).
  def fmt_request(body)
    pos = SpinelKit::Json.find_value_start(body, "response_format")
    if pos < 0 || pos >= body.length || body[pos] == "n"
      return 0
    end
    v = ""
    i = pos
    open = true
    while open && i < body.length && v.length < 40
      c = body[i]
      if c != " " && c != "\n" && c != "\t" && c != "\r"
        v = v + c
      end
      if c == "}"
        open = false
      end
      i = i + 1
    end
    if v == "{\"type\":\"json_object\"}"
      return 1
    end
    if v == "{\"type\":\"text\"}"
      return 0
    end
    -1
  end

  # The JSON state a request's reply starts from: empty, or, for a chat
  # request whose last message is a partial assistant answer (toy#216), that
  # answer, so the reply continues the object it began.
  def fmt_state(body, chat)
    js = JsonPrefix.new
    if chat
      msgs = Tep::Llm::OpenAI.parse_messages(body)
      if msgs.length > 0 && msgs[msgs.length - 1].role == "assistant"
        if !js.feed(msgs[msgs.length - 1].content)
          js.st = -1
        end
      end
    end
    js
  end

  # Refuses (on `rep`) a response_format this server cannot enforce.
  def fmt_check(rep, body, chat)
    f = fmt_request(body)
    if f == 0
      return 0
    end
    if f < 0
      rep.fail(400, "invalid_request_error",
               "response_format: only {\"type\":\"json_object\"} and {\"type\":\"text\"} are supported")
      return 0
    end
    if !@state.tok.present
      rep.fail(400, "invalid_request_error",
               "response_format needs a GGUF with an embedded tokenizer")
      return 0
    end
    js = fmt_state(body, chat)
    if js.st == -1 || js.complete
      rep.fail(400, "invalid_request_error",
               "response_format: the partial assistant message is not the start of a JSON object")
      return 0
    end
    1
  end

  # Tells the slot to keep its reply a JSON object when the request asks.
  def fmt_apply(slot, body, chat)
    if fmt_request(body) == 1
      slot.fmt  = 1
      slot.json = fmt_state(body, chat)
    end
    0
  end

  # toy#218: the adapter scale a request asks for: "lora_scale" when given,
  # otherwise 1 (the adapter as loaded). Always 0 without an adapter.
  def ad_request(body)
    if @state.kv.adapter_n == 0
      return 0.0
    end
    if body.length > 0 && SpinelKit::Json.has_key?(body, "lora_scale")
      return SpinelKit::Json.get_float(body, "lora_scale")
    end
    1.0
  end

  # toy#221: the weights a reply came from: the model file, the adapter
  # file ("" without one) and the scale the adapter was applied at.
  def prov_json(scale)
    "\"toy\":{" +
      SpinelKit::Json.encode_pair_str("model_sha256", MODEL_SHA256) + "," +
      SpinelKit::Json.encode_pair_str("adapter_sha256", ADAPTER_SHA256) + "," +
      "\"lora_scale\":" + scale.to_s +
    "}"
  end

  # POST /v1/hidden (toy#218): the residual stream at the last token of a
  # prompt. The prompt is `messages` (chat template applied, as in chat) or
  # `prompt` (text or token ids, as in completions). `layers` lists what to
  # return: 0 is the embedding output, i the stream after block i (the
  # numbering of HF `output_hidden_states`); default every block, 1..L.
  # `position` cuts the prompt after that token index first (negative
  # counts from the end; default -1, the whole prompt). Nothing is
  # generated. Returns the JSON reply in rep.text.
  def serve_hidden(body)
    rep = ToyReply.new
    if !@state.kv.fused
      return rep.fail(501, "not_supported",
                      "hidden states need the fused decode graph, which this model or configuration does not use")
    end
    token_ids = [0]
    token_ids.pop
    if SpinelKit::Json.has_key?(body, "messages")
      if !@state.tok.present
        return rep.fail(400, "invalid_request_error",
                        "messages need a GGUF with an embedded tokenizer; send prompt token ids")
      end
      token_ids = chat_prompt_ids(body)
    else
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
    end
    if token_ids.length == 0
      return rep.fail(400, "invalid_request_error",
                      "give messages, or prompt as a non-empty text or array of token ids")
    end
    at = token_ids.length - 1
    if SpinelKit::Json.has_key?(body, "position")
      at = SpinelKit::Json.get_int(body, "position")
      if at < 0
        at = token_ids.length + at
      end
      if at < 0 || at >= token_ids.length
        return rep.fail(400, "invalid_request_error",
                        "position is outside the prompt's " + token_ids.length.to_s + " tokens")
      end
      while token_ids.length > at + 1
        token_ids.pop
      end
    end
    n_layers = @state.cfg.n_layers
    layers = [0]
    layers.pop
    if SpinelKit::Json.has_key?(body, "layers")
      asked = SpinelKit::Json.get_int_array(body, "layers")
      # Ascending and distinct, which is the order the reply uses.
      lay = 0
      while lay <= n_layers
        i = 0
        hit = false
        while i < asked.length
          if asked[i] == lay
            hit = true
          end
          i = i + 1
        end
        if hit
          layers.push(lay)
        end
        lay = lay + 1
      end
      i = 0
      while i < asked.length
        if asked[i] < 0 || asked[i] > n_layers
          return rep.fail(400, "invalid_request_error",
                          "layers are 0 (embeddings) to " + n_layers.to_s + " (after the last block)")
        end
        i = i + 1
      end
      if layers.length == 0
        return rep.fail(400, "invalid_request_error", "layers must be a non-empty array of integers")
      end
    else
      lay = 1
      while lay <= n_layers
        layers.push(lay)
        lay = lay + 1
      end
    end
    if fit_context(token_ids.length, 1) <= 0
      return rep.fail(400, "context_length_exceeded",
                      "the prompt is " + token_ids.length.to_s + " tokens and the context is " +
                      MAX_T.to_s + " (start the server with --ctx N)")
    end
    scale = ad_request(body)
    slot_i = -1
    begin
      t_start = TinyNN.tnn_events_now_seconds
      slot_i = api_slot_wait(token_ids, 1, scale)
      slot = SLOTS[slot_i]
      slot.tap_layers = layers
      api_slot_run(slot)
      if slot.err.length > 0
        raise slot.err
      end
      d = @state.cfg.d_model
      if slot.hidden.length != layers.length * d
        raise "hidden-state read returned " + slot.hidden.length.to_s + " values for " +
              layers.length.to_s + " layers of " + d.to_s
      end
      t_end = TinyNN.tnn_events_now_seconds
      emit_request_event(token_ids.length, 0, ((t_end - t_start) * 1.0e6).to_i, 0,
                         slot.cached, slot_i, slot.shared_steps, scale)
      out = "{" + SpinelKit::Json.encode_pair_str("object", "hidden_states") + "," +
            SpinelKit::Json.encode_pair_str("model", SpinelKit::Json.get_str(body, "model")) + "," +
            "\"layers\":" + SpinelKit::Json.from_int_array(layers) + "," +
            SpinelKit::Json.encode_pair_int("position", at) + "," +
            SpinelKit::Json.encode_pair_int("prompt_tokens", token_ids.length) + "," +
            SpinelKit::Json.encode_pair_int("d_model", d) + "," +
            "\"lora_scale\":" + scale.to_s + "," +
            prov_json(scale) + "," +
            "\"data\":["
      li = 0
      while li < layers.length
        # One layer's row is built on its own: appending 3-4 thousand
        # numbers to the whole reply would copy it each time.
        row = "["
        x = 0
        while x < d
          if x > 0
            row = row + ","
          end
          row = row + slot.hidden[li * d + x].to_s
          x = x + 1
        end
        if li > 0
          out = out + ","
        end
        out = out + row + "]"
        li = li + 1
      end
      rep.text = out + "]}"
      rep.prompt_tokens = token_ids.length
    rescue => e
      puts "[openai_api_llama] request failed: " + e.message
      rep.fail(500, "server_error", e.message)
    end
    if slot_i >= 0
      api_slot_release(SLOTS[slot_i])
    end
    rep
  end

  # Generate into `rep` with the request's options: stop strings and
  # log-probabilities (toy#216), the context check (toy#207), and one
  # request event. A Ruby exception becomes a 500 reply (toy#214).
  def serve_generate(rep, body, token_ids, n_new_req, chat)
    rep.prompt_tokens = token_ids.length
    scale = ad_request(body)
    rep.prov = prov_json(scale)
    stops = api_json_str_list(body, "stop")
    top   = lp_request(body)
    if (stops.length > 0 || top >= 0) && !@state.tok.present
      return rep.fail(400, "invalid_request_error",
                      "stop strings and logprobs need a GGUF with an embedded tokenizer")
    end
    fmt_check(rep, body, chat)
    if rep.status != 200
      return rep
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
      slot_i = api_slot_wait(token_ids, n_new, ad_request(body))
      slot = SLOTS[slot_i]
      slot.stop_strs = stops
      slot.lp_top    = top
      fmt_apply(slot, body, chat)
      api_slot_run(slot)
      if slot.err.length > 0
        raise slot.err
      end
      new_ids = slot.out_ids
      t_end   = TinyNN.tnn_events_now_seconds
      emit_request_event(token_ids.length, new_ids.length,
                         ((t_end - t_start) * 1.0e6).to_i, n_new, slot.cached, slot_i, slot.shared_steps, scale)
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
      out = out + lp_entry(slot, ids[i], i, top)
      i = i + 1
    end
    out + "]}"
  end

  # The entry of generated token `i`: itself and its `top` alternatives.
  def lp_entry(slot, id, i, top)
    out = "{" + lp_fields(id, slot.lp_tok[i]) + ",\"top_logprobs\":["
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
    slot = SLOTS[api_slot_wait(token_ids, n_new, ad_request(""))]
    api_slot_run(slot)
    api_slot_release(slot)
    if slot.err.length > 0
      raise slot.err
    end
    new_ids = slot.out_ids
    t_end   = TinyNN.tnn_events_now_seconds
    emit_request_event(token_ids.length, new_ids.length,
                       ((t_end - t_start) * 1.0e6).to_i, n_new, slot.cached, slot.idx, slot.shared_steps, ad_request(""))

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
  def emit_request_event(n_prompt, n_completion, latency_us, n_new, cached, slot, shared_steps, scale)
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
      ev = ev +   ",\"lora_scale\":"        + scale.to_s
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
  attr_accessor :prompt_tokens, :completion_tokens, :finish, :lp_json, :prov

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
    @prov     = ""
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

# One streamed reply (toy#219). tep calls pump with the open stream.
class ToyStreamer < Tep::Streamer
  attr_accessor :body, :token_ids, :n_new, :chat, :id

  def initialize(backend)
    @backend   = backend
    @body      = ""
    @token_ids = [0]
    @token_ids.pop
    @n_new     = 0
    @chat      = false
    @id        = ""
  end

  def pump(out)
    @backend.serve_stream(out, @body, @token_ids, @n_new, @chat, @id)
    out.write("data: [DONE]\n\n")
    0
  end
end

# Open a stream for a request whose prompt is `token_ids`, or answer with
# the HTTP error `rep` carries.
def api_stream_reply(backend, res, rep, body, token_ids, n_new_req, chat)
  n_new = 0
  if rep.status == 200
    n_new = backend.stream_check(rep, body, token_ids, n_new_req, chat)
  end
  if rep.status != 200
    res.headers["Content-Type"] = "application/json"
    res.set_status(rep.status)
    return rep.error_json
  end
  res.headers["Content-Type"]  = "text/event-stream"
  res.headers["Cache-Control"] = "no-cache"
  st = ToyStreamer.new(backend)
  st.body      = body
  st.token_ids = token_ids
  st.n_new     = n_new
  st.chat      = chat
  st.id        = rep.id
  res.start_stream(st)
  ""
end

# POST /v1/chat/completions. Without a tokenizer tep's handler answers 501.
class ToyChatHandler < Tep::Handler
  def initialize(backend)
    @backend = backend
    @no_chat = Tep::Llm::OpenAI::ChatCompletionsHandler.new
  end

  def handle(req, res)
    body = req.raw_body
    if !@backend.supports_chat?
      return @no_chat.handle(req, res)
    end
    if api_wants_stream?(body)
      rep = ToyReply.new
      rep.id = api_gen_id("chatcmpl")
      token_ids = @backend.chat_prompt_ids(body)
      if token_ids.length == 0
        rep.fail(400, "invalid_request_error", "messages must be a non-empty array")
      end
      return api_stream_reply(@backend, res, rep, body, token_ids,
                              @backend.chat_max_tokens(body, token_ids.length), true)
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
      rep.usage_json + "," +
      rep.prov +
    "}"
  end
end

# POST /v1/completions.
class ToyCompletionsHandler < Tep::Handler
  def initialize(backend)
    @backend = backend
  end

  def handle(req, res)
    body = req.raw_body
    if api_wants_stream?(body)
      rep = ToyReply.new
      rep.id = api_gen_id("cmpl")
      token_ids = @backend.text_prompt_ids(rep, body)
      return api_stream_reply(@backend, res, rep, body, token_ids,
                              @backend.text_max_tokens(body), false)
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
      rep.usage_json + "," +
      rep.prov +
    "}"
  end
end

# POST /v1/hidden (toy#218): see ToyBackend#serve_hidden.
class ToyHiddenHandler < Tep::Handler
  def initialize(backend)
    @backend = backend
  end

  def handle(req, res)
    res.headers["Content-Type"] = "application/json"
    rep = @backend.serve_hidden(req.raw_body)
    if rep.status != 200
      res.set_status(rep.status)
      return rep.error_json
    end
    rep.text
  end
end
