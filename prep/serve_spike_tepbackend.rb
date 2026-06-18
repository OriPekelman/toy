# prep/serve_spike_tepbackend.rb -- PHASE 1 SPIKE for toy#30.
#
# NOT a shipped runner. Verifies, at the toy-v0.8.0 union Spinel pin,
# whether the tep OpenAI Backend battery (Tep::Llm::OpenAI::Server.use +
# serve! + the battery's ModelsHandler/CompletionsHandler/EmbeddingsHandler)
# produces CORRECT JSON when a Backend subclass authored in TOY's
# compilation unit dispatches through APP.openai_backend.
#
# DECISION GATE: if the battery JSON is byte-correct (no blank keys, no
# pointer-ints) -> proceed to the collapse. If "Bug B" (Tep::Json.escape
# poly-degradation blanking keys/values) reproduces -> STOP, stay on toy's
# hand-rolled handlers. See toy#30 / tep#205.
#
# Loads toy's FULL serve surface (model load via server.rb's STATE +
# api_generate_ids + the embeddings lookup) so sched_fibers stays concrete
# (a minimal surface degrades it -> matz/spinel#1369 boot crash).

require_relative "../lib/toy/llm/engine/llama_kv_engine"
require_relative "../vendor/spinel/spinel_kit/lib/spinel_kit/json_builder"
require_relative "../lib/toy/io/toy_events"
require_relative "../lib/toy/io/loaders/toy_smollm2_loader"
require_relative "../vendor/spinel/deps"
# server.rb defines STATE (model loaded at require time), MODEL_NAME,
# api_generate_ids, api_now_unix. Reused verbatim -- the spike does not
# duplicate the loader/decode logic.
require_relative "../lib/toy/serve/openai/server"

# ToyBackend authored HERE (toy's compilation unit), subclassing the
# vendored tep Backend. Overrides the three model-specific surfaces.
class ToyBackend < Tep::Llm::OpenAI::Backend
  def list_models
    out = [""]
    out.delete_at(0)
    out.push(STATE.model_name)
    out
  end

  def device_kind
    "cpu"
  end

  # token-level greedy generation -> Tep::Llm::OpenAI::Completion.
  # Reuses server.rb's api_generate_ids (deterministic argmax). The tep
  # CompletionsHandler emits `text` (not toy's `ids`); for the SPIKE we
  # only care that the JSON ENVELOPE encodes correctly, so text carries
  # a deterministic string of the decoded ids.
  def generate_from_tokens(model, token_ids, sampling)
    n_new = sampling.max_tokens
    if n_new <= 0
      n_new = 16
    end
    if n_new > 256
      n_new = 256
    end
    new_ids = api_generate_ids(token_ids, n_new)
    comp = Tep::Llm::OpenAI::Completion.new
    txt = ""
    i = 0
    while i < new_ids.length
      if i > 0
        txt = txt + ","
      end
      txt = txt + new_ids[i].to_s
      i = i + 1
    end
    comp.text              = txt
    comp.prompt_tokens     = token_ids.length
    comp.completion_tokens = new_ids.length
    comp
  end

  def supports_embeddings?
    true
  end

  # mean-pooled embedding -> Array[Float] of length d_model.
  # The lookup + pooling moves here from EmbeddingsHandler#handle.
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
      rc = TinyNN.tnn_embed_lookup_to_doubles(sess, t_embed, token_ids[i], tok_buf, d_model)
      k = 0
      while k < d_model
        sum_buf[k] = sum_buf[k] + tok_buf[k]
        k = k + 1
      end
      i = i + 1
    end

    inv_n = 1.0 / token_ids.length.to_f
    out = [0.0]; out.pop
    kk = 0
    while kk < d_model
      out.push(sum_buf[kk] * inv_n)
      kk = kk + 1
    end
    out
  end
end

SERVE_PORT = (ENV["PORT"] || "4567").to_i

# Pin api_gen_id's `prefix` param to String so co-loading tep's battery
# doesn't widen the (otherwise-uncalled) helper to poly -> char* (the
# 2026-06-08 spike artifact; server.rb:108). A String-literal call keeps
# the param concrete. The real collapse keeps a String call site too.
_pin_gen_id = api_gen_id("cmpl")
puts "[spike] api_gen_id pin: " + _pin_gen_id

# The collapse shape: register the backend, mount the battery routes.
Tep::Llm::OpenAI::Server.use(ToyBackend.new)
Tep::Llm::OpenAI::Server.serve!("")

puts "[spike] tep battery mounted; serving on " + SERVE_PORT.to_s
Tep.run!(SERVE_PORT, 1, false)
