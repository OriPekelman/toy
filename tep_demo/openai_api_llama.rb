# tep_demo/openai_api_llama.rb -- OpenAI-compatible API backed by
# Toy::SmolLM2KVFFICache (llama-family architecture: SmolLM2, Qwen2.5,
# TinyLlama, Llama-3.x with appropriate weights).
#
# Companion to tep_demo/openai_api.rb (which serves GPT-2 via lib/bpe.rb).
# *This* binary takes a different design choice: **token IDs in, token
# IDs out**, no server-side tokenizer. SmolLM2/Qwen vocab is HF
# tokenizer-flavoured (SentencePiece / tiktoken-BPE) and shipping a
# Ruby port of those would either pull in deps or duplicate gigabytes
# of vocab tables; the cleaner story for v1 is to keep tokenization
# client-side (`prep/qwen25_tokens.py encode "..."` → integer IDs)
# and have the server speak only in IDs.
#
# Endpoints (OpenAI-compatible where it makes sense):
#
#   GET  /v1/models                  list available models
#   POST /v1/embeddings             expects `input: [int, int, ...]`; returns
#                                    mean-pooled embedding (OpenAI shape)
#   POST /v1/completions             expects `prompt: [int, int, ...]`
#                                    returns choices[0].text = ""
#                                    plus a non-standard
#                                    choices[0].ids = [int...]
#                                    field with the generated tokens.
#                                    The OpenAI spec already allows
#                                    `prompt` to be an int array, so
#                                    well-behaved clients can talk to
#                                    this without code changes.
#   POST /v1/chat/completions        returns 501 (chat templating
#                                    requires a tokenizer; out of
#                                    scope for the ID-only path).
#   GET  /health                     liveness probe.
#   GET  /                           HTML landing page.
#
# Build + run (from project root):
#   make tep_demo/openai_api_llama
#   ./tep_demo/openai_api_llama -p 4567 -w 1                # SmolLM2-135M default
#   MODEL_PATH=data/qwen25-1.5b-native-q8.gguf \
#     ./tep_demo/openai_api_llama -p 4567 -w 1              # Qwen-1.5B Q8
#
# Smoke test:
#   # tokenise on the client side:
#   IDS=$(prep/qwen25_tokens.py encode "Hello, my name is" 2>&1 | tail -1)
#   curl -X POST http://127.0.0.1:4567/v1/completions \
#        -H 'Content-Type: application/json' \
#        -d "{\"model\":\"qwen25-1.5b\",\"prompt\":[$(echo $IDS | tr ' ' ',')],\"max_tokens\":16}"
#   # decode the response ids client-side.
#
# Model selection (GH#188 — consolidated from 7 near-duplicate
# servers in v0.6.0-pre-alpha). Env knobs at boot:
#
#   MODEL_PATH   path to any llama-family GGUF (default
#                data/smollm2-135m-native.gguf)
#   MODEL_NAME   user-facing label in /v1/models + completion
#                responses. Defaults to GGUF basename minus .gguf.
#   MAX_T        max context for the KV cache (default 256)
#
# Previously each (model, quantization) pair had its own
# tep_demo/openai_api_<name>.rb because Spinel module-constant
# inference was sketchy on env-driven values. After landmines
# #11/#15 retired and the codegen settled, env-driven constants
# work; one binary serves them all.
#
# Built directly with `$(SPINEL)` (no `tep build` translation step) —
# this file uses the explicit `Tep::Handler` form, same pattern as
# tep_demo/inference_api.rb.

require_relative "../lib/toy_smollm2_ffi_kv"
require_relative "../lib/toy_smollm2_loader"
require_relative "../vendor/spinel/deps"
require_relative "embeddings_handler"

# Local helper: parse a JSON value at `start_pos` as an int array.
# Returns Array<Int> (empty on absent / non-array / non-int-element).
# Lives here rather than in `Tep::Json` because `_tep_lib/` is a
# regenerable snapshot of upstream Tep; durable additions belong in
# the Tep gem itself.
module ApiJson
  def self.get_int_array(s, key)
    out_ia = [0]
    out_ia.pop
    pos_ia = Tep::Json.find_value_start(s, key)
    if pos_ia < 0
      return out_ia
    end
    pos_ia = Tep::Json.skip_ws(s, pos_ia)
    if pos_ia >= s.length || s[pos_ia] != "["
      return out_ia
    end
    pos_ia += 1
    while pos_ia < s.length
      pos_ia = Tep::Json.skip_ws(s, pos_ia)
      if pos_ia >= s.length
        return out_ia
      end
      if s[pos_ia] == "]"
        return out_ia
      end
      neg_ia = false
      if s[pos_ia] == "-"
        neg_ia = true
        pos_ia += 1
      end
      acc_ia = 0
      saw_digit_ia = false
      while pos_ia < s.length
        ch_ia = s[pos_ia]
        if ch_ia >= "0" && ch_ia <= "9"
          acc_ia = acc_ia * 10 + (ch_ia.bytes[0] - "0".bytes[0])
          saw_digit_ia = true
          pos_ia += 1
        else
          break
        end
      end
      if !saw_digit_ia
        empty_ia = [0]
        empty_ia.pop
        return empty_ia
      end
      if neg_ia
        out_ia.push(0 - acc_ia)
      else
        out_ia.push(acc_ia)
      end
      pos_ia = Tep::Json.skip_ws(s, pos_ia)
      if pos_ia < s.length && s[pos_ia] == ","
        pos_ia += 1
      elsif pos_ia < s.length && s[pos_ia] == "]"
        return out_ia
      end
    end
    out_ia
  end
end

# GH#188 — model selection via env. Consolidated from 7 near-duplicate
# servers (openai_api_smollm2.rb + openai_api_qwen25_{0.5,1.5,3,7}b{,_q8}.rb).
# Defaults to SmolLM2-135M for a cheap-to-test smoke; override for any
# llama-family GGUF (SmolLM2, Qwen2.5, TinyLlama, Llama-3.x, …).
#
#   MODEL_PATH=data/qwen25-1.5b-native-q8.gguf MODEL_NAME=qwen25-1.5b-q8 \
#     ./tep_demo/openai_api_llama -p 4567
#
# If MODEL_NAME isn't set, it defaults to the basename of MODEL_PATH
# minus the ".gguf" suffix — close enough for the /v1/models response.
GGUF_PATH = ENV["MODEL_PATH"] || "data/smollm2-135m-native.gguf"
MODEL_NAME_ENV = ENV["MODEL_NAME"] || ""
# Derive a default model name from the GGUF basename when not given.
# basename: strip everything before the last "/"; then strip ".gguf".
_mn_last_slash = GGUF_PATH.rindex("/")
_mn_basename   = _mn_last_slash == nil ? GGUF_PATH : GGUF_PATH[_mn_last_slash + 1..-1]
_mn_dot_gguf   = _mn_basename.rindex(".gguf")
_mn_default    = _mn_dot_gguf == nil ? _mn_basename : _mn_basename[0...(_mn_dot_gguf)]
MODEL_NAME = MODEL_NAME_ENV.length > 0 ? MODEL_NAME_ENV : _mn_default
MAX_T      = (ENV["MAX_T"] || "256").to_i

# ---- Inference state. Class instance held as a CONSTANT so spinel
#      emits a typed slot for it (same pattern as inference_api.rb /
#      openai_api.rb). ----

class State
  attr_accessor :cfg, :kv, :gguf, :model_name, :ready
  def initialize
    @cfg  = nil
    @kv   = nil
    @gguf = TinyNN.tnn_null_ptr
    @model_name = MODEL_NAME
    @ready      = false
  end
end
STATE = State.new
# GH#188 — explicit re-assignment AFTER STATE.new. State#initialize
# captures `@model_name = MODEL_NAME` but Spinel's ivar typing
# appears to evaluate it before the MODEL_NAME ternary resolves
# (the embeddings handler receives MODEL_NAME via constructor at
# boot and prints it correctly; STATE.model_name accessed at
# request time shows empty). Setting it explicitly here works
# around it cleanly without touching the State class.
STATE.model_name = MODEL_NAME

puts "[openai_api_llama] loading config from " + GGUF_PATH
STATE.cfg = SmolLM2ConfigLoader.read(GGUF_PATH)
puts "[openai_api_llama] vocab=" + STATE.cfg.vocab.to_s +
     " d=" + STATE.cfg.d_model.to_s +
     " L=" + STATE.cfg.n_layers.to_s +
     " n_heads=" + STATE.cfg.n_heads.to_s +
     " n_kv=" + STATE.cfg.n_kv.to_s

flags = GGUFLoad.detect_smollm2_flags(GGUF_PATH)
puts "[openai_api_llama] flags: untied=" + flags.untied.to_s +
     " qkv_bias=" + flags.qkv_bias.to_s

# Auto-dispatch: native GGUFs get BYO-pointer mmap (Phase 2);
# legacy GGUFs fall through to realize_for + load_weights copy.
# Regenerate with `--ggml-native` (and optionally `--quantize q8_0`)
# to unlock the mmap fast path on this binary.
puts "[openai_api_llama] realising (MAX_T=" + MAX_T.to_s + ")..."
STATE.kv = SmolLM2KVFFICache.new
STATE.gguf = STATE.kv.realize_and_load_auto(GGUF_PATH, MAX_T, STATE.cfg, flags)

STATE.ready = true
puts "[openai_api_llama] ready; serving"

# ---- Helpers ----

def api_now_unix
  Time.now.to_i
end

def api_gen_id(prefix)
  t = Time.now
  v = (t.to_i * 1_000_003) ^ ((t.to_f - t.to_i).to_f * 1.0e9).to_i
  prefix + "-" + v.to_s
end

# Greedy generation from a pre-tokenized prompt. KV-cache decode:
# prefill the prompt one step at a time, then sample greedily for
# `n_new` more steps. Returns Array<Int> of the new token IDs (does
# NOT include the prompt).
#
# This routine re-runs the prefill from position 0 every call; the
# cache's t_K / t_V tensors are persistent and get overwritten in
# place. (A future optimisation would be a fast prefix-cache for
# shared prompts.)
def api_generate_ids(prompt_ids, n_new)
  out_ids = [0]
  out_ids.pop

  vocab = STATE.cfg.vocab
  last_logits = Mat.new(1, vocab)
  prefill_pos = 0
  while prefill_pos < prompt_ids.length
    last_logits = SmolLM2KV.decode_step(STATE.kv, prompt_ids[prefill_pos], prefill_pos)
    prefill_pos = prefill_pos + 1
  end

  # First generated token comes from the last prefill step's logits.
  best_idx = 0
  best_val = last_logits.flat[0]
  v_iter = 1
  while v_iter < vocab
    val = last_logits.flat[v_iter]
    if val > best_val; best_val = val; best_idx = v_iter; end
    v_iter = v_iter + 1
  end
  out_ids.push(best_idx)

  step = 1
  while step < n_new
    last_logits = SmolLM2KV.decode_step(STATE.kv, out_ids[out_ids.length - 1],
                                         prompt_ids.length + out_ids.length - 1)
    best_idx = 0
    best_val = last_logits.flat[0]
    v_iter = 1
    while v_iter < vocab
      val = last_logits.flat[v_iter]
      if val > best_val; best_val = val; best_idx = v_iter; end
      v_iter = v_iter + 1
    end
    out_ids.push(best_idx)
    step = step + 1
  end
  out_ids
end

# ---- Handlers ----

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

    new_ids = api_generate_ids(prompt_ids, n_new)
    prompt_len = prompt_ids.length
    completion_len = new_ids.length

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
    "<p>Model: <code>" + STATE.model_name + "</code> via direct GGUF→FFI loader.</p>" +
    "<p>Tokenize client-side; this server speaks integer token IDs only. " +
    "Run <code>prep/qwen25_tokens.py encode \"...\"</code> to get IDs.</p>" +
    "<p>Endpoints:</p><ul>" +
    "<li><code>POST /v1/completions</code> — body <code>{\"prompt\":[int,...],\"max_tokens\":N}</code></li>" +
    "<li><code>POST /v1/chat/completions</code> — 501 (tokenizer required)</li>" 
    "<li><code>POST /v1/embeddings</code> — body <code>{\"input\":[int,...]}</code> → mean-pooled vector</li>" ++
    "<li><code>GET /v1/models</code></li>" +
    "<li><code>GET /health</code></li>" +
    "</ul></body></html>"
  end
end

Tep.get  "/",                     IndexHandler.new
Tep.get  "/health",               HealthHandler.new
Tep.get  "/v1/models",            ModelsHandler.new
Tep.post "/v1/completions",       CompletionsHandler.new
Tep.post "/v1/chat/completions",  ChatCompletionsHandler.new
Tep.post "/v1/embeddings",        EmbeddingsHandler.new(STATE, MODEL_NAME)

# CLI: -p PORT -w WORKERS -q (quiet)
__port = 4567
__workers = 1
__quiet = false
__i = 0
while __i < ARGV.length
  __a = ARGV[__i]
  if __a == "-p" && __i + 1 < ARGV.length
    __port = ARGV[__i + 1].to_i
    __i += 2
  elsif __a == "-w" && __i + 1 < ARGV.length
    __workers = ARGV[__i + 1].to_i
    __i += 2
  elsif __a == "-q"
    __quiet = true
    __i += 1
  else
    __i += 1
  end
end
Tep.run!(__port, __workers, __quiet)
