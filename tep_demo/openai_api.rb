# tep_demo/openai_api.rb -- OpenAI-compatible chat completions API
# backed by GPT2KV.
#
# Endpoints (subset of OpenAI's surface, enough that openai-python,
# curl, the `openai` CLI, langchain's openai backend etc. just work):
#
#   GET  /v1/models                  list available models
#   POST /v1/chat/completions        non-streaming chat completion
#   POST /v1/completions             non-streaming legacy completion
#   GET  /health                     liveness probe (non-OpenAI)
#
# Streaming (`stream: true` SSE) is a follow-up; non-streaming first.
#
# GPT-2 isn't fine-tuned for chat, so the input "messages" array is
# folded into a single prompt by concatenating each message's
# `content` field. The standard system / user / assistant role tags
# get prepended (no special template — bare text, plus newlines).
# Sampling is greedy (top-1 argmax); `temperature` / `top_p` ignored.
#
# Build + run (from project root):
#   ~/sites/tep/bin/tep build tep_demo/openai_api.rb -o tep_demo/openai_api
#   ./tep_demo/openai_api -p 4567 -w 1
#
# Smoke test (in another shell):
#   curl http://127.0.0.1:4567/v1/models
#   curl -X POST http://127.0.0.1:4567/v1/chat/completions \
#        -H 'Content-Type: application/json' \
#        -d '{"model":"gpt2","messages":[{"role":"user","content":"Hello, my name is"}],"max_tokens":15}'
#
# Or via the openai CLI:
#   OPENAI_API_BASE=http://127.0.0.1:4567/v1 openai api chat_completions.create \
#     -m gpt2 -g user "Hello, my name is" --max-tokens 15

require_relative "../lib/toy/models/transformer"
require_relative "../lib/toy/models/gpt2"
require_relative "../lib/gpt2_ffi_kv"
require_relative "../lib/toy/io/gguf_load"
require_relative "../lib/toy/io/bpe"

set :workers, 1   # one worker; FFI session isn't thread-safe.

# ---- Inference state. Loaded once at top level (same pattern as
#      the older tep_demo/inference_api.rb -- a class instance held
#      as a constant so spinel emits a typed slot for it). ----

class State
  attr_accessor :cfg, :model, :kv, :bpe, :model_name, :ready
  def initialize
    @cfg = nil; @model = nil; @kv = nil; @bpe = nil
    @model_name = "gpt2"
    @ready = false
  end
end
STATE = State.new

GGUF_PATH = "data/gpt2-f32.gguf"
BPE_DIR   = "data"
MAX_T     = 1024

puts "[openai_api] loading config from " + GGUF_PATH
STATE.cfg = GPT2ConfigLoader.read(GGUF_PATH)
puts "[openai_api] vocab=" + STATE.cfg.vocab_size.to_s +
     " d=" + STATE.cfg.d_model.to_s +
     " layers=" + STATE.cfg.n_layers.to_s

puts "[openai_api] constructing native GPT2LM..."
STATE.model = GPT2LM.new(STATE.cfg.vocab_size, STATE.cfg.d_model,
                          STATE.cfg.d_ff, STATE.cfg.n_heads,
                          STATE.cfg.n_layers, STATE.cfg.context_length)
GGUFLoad.load_gpt2(STATE.model, GGUF_PATH)

puts "[openai_api] realising KV cache (MAX_T=" + MAX_T.to_s + ")..."
STATE.kv = GPT2KVFFICache.new
STATE.kv.realize_for(MAX_T, STATE.cfg.d_model, STATE.cfg.d_ff,
                      STATE.cfg.n_heads, STATE.cfg.n_layers,
                      STATE.cfg.vocab_size, STATE.cfg.context_length)
GPT2KV.upload_from(STATE.kv, STATE.model)

puts "[openai_api] loading BPE tables..."
STATE.bpe = GPT2BPE.load(BPE_DIR)

STATE.ready = true
puts "[openai_api] ready; serving"

# ---- Helpers ----

# Extract the *concatenated* content of every `"content":"..."` in the
# request body. Tep::Json doesn't traverse nested arrays/objects, so
# we walk manually. The byte scan also handles JSON-escape decoding
# (`\"`, `\\`, `\n`, `\t`) inline.
def extract_messages_text(body)
  out = ""
  n = body.bytesize
  needle_bytes = [0x22, 0x63, 0x6f, 0x6e, 0x74, 0x65, 0x6e, 0x74, 0x22]   # "content"
  nl = 9
  i = 0
  count = 0
  while i + nl <= n
    # Try to match needle at position i.
    matched = true
    k = 0
    while k < nl
      if body.getbyte(i + k) != needle_bytes[k]
        matched = false
        break
      end
      k = k + 1
    end
    if !matched
      i = i + 1
    else
      # Move past needle + optional space + ':' + optional space + '"'
      j = i + nl
      while j < n && (body.getbyte(j) == 0x20 || body.getbyte(j) == 0x09)
        j = j + 1
      end
      if j >= n || body.getbyte(j) != 0x3a   # ':' expected
        i = i + nl
      else
        j = j + 1   # past ':'
        while j < n && (body.getbyte(j) == 0x20 || body.getbyte(j) == 0x09)
          j = j + 1
        end
        if j >= n || body.getbyte(j) != 0x22   # opening '"' expected
          i = i + nl
        else
          j = j + 1   # past opening "
          start = j
          # Collect until unescaped '"'.
          piece = ""
          while j < n
            b = body.getbyte(j)
            if b == 0x5c && j + 1 < n
              nb = body.getbyte(j + 1)
              if nb == 0x22
                piece = piece + "\""; j = j + 2
              elsif nb == 0x5c
                piece = piece + "\\"; j = j + 2
              elsif nb == 0x6e
                piece = piece + "\n"; j = j + 2
              elsif nb == 0x74
                piece = piece + "\t"; j = j + 2
              else
                # passthrough single char
                piece = piece + nb.chr
                j = j + 2
              end
            elsif b == 0x22
              break
            else
              piece = piece + b.chr
              j = j + 1
            end
          end
          if count > 0
            out = out + "\n"
          end
          out = out + piece
          count = count + 1
          i = j + 1
        end
      end
    end
  end
  out
end

def now_unix
  # Tep doesn't ship a date helper; epoch seconds is fine.
  Time.now.to_i
end

def gen_id(prefix)
  # 16 hex chars from Time.now's microsecond fraction + pid xor. Not
  # crypto-grade, just unique-ish for response ids.
  t = Time.now
  v = (t.to_i * 1_000_003) ^ ((t.to_f - t.to_i).to_f * 1.0e9).to_i
  prefix + "-" + v.to_s
end

# Greedy generation. Returns Array<Int> of new token IDs (length = n_new
# or fewer if a stop token were ever hit -- we don't define one).
def generate_ids(prompt_ids, n_new)
  out = [0]
  out.pop
  # Prefill the prompt.
  last_logits = Mat.new(1, STATE.cfg.vocab_size)
  pos = 0
  while pos < prompt_ids.length
    last_logits = GPT2KV.decode_step(STATE.kv, prompt_ids[pos], pos)
    pos = pos + 1
  end

  # First generated token comes from the last prefill step's logits.
  vocab = STATE.cfg.vocab_size
  best   = 0
  best_v = last_logits.flat[0]
  v = 1
  while v < vocab
    val = last_logits.flat[v]
    if val > best_v; best_v = val; best = v; end
    v = v + 1
  end
  out.push(best)

  step = 1
  while step < n_new
    last_logits = GPT2KV.decode_step(STATE.kv, out[out.length - 1],
                                     prompt_ids.length + out.length - 1)
    best   = 0
    best_v = last_logits.flat[0]
    v = 1
    while v < vocab
      val = last_logits.flat[v]
      if val > best_v; best_v = val; best = v; end
      v = v + 1
    end
    out.push(best)
    step = step + 1
  end
  out
end

# Serialize an Array<Int> of token IDs as a JSON array literal.
def json_int_array(arr)
  s = "["
  i = 0
  while i < arr.length
    s = s + arr[i].to_s
    if i < arr.length - 1; s = s + ","; end
    i = i + 1
  end
  s + "]"
end

# ---- Handlers ----

get '/health' do
  res.headers["Content-Type"] = "text/plain"
  if STATE.ready
    "ok\n"
  else
    res.set_status(503); "loading\n"
  end
end

get '/v1/models' do
  res.headers["Content-Type"] = "application/json"
  "{\"object\":\"list\",\"data\":[{" +
    Tep::Json.encode_pair_str("id", STATE.model_name) + "," +
    Tep::Json.encode_pair_str("object", "model") + "," +
    Tep::Json.encode_pair_int("created", now_unix) + "," +
    Tep::Json.encode_pair_str("owned_by", "toy") +
  "}]}\n"
end

# Result of one inference call. Dedicated class instead of an
# [text, p, c] tuple — Spinel mistypes mixed-type arrays as
# polymorphic, which then poisons every encode_pair_* call site
# downstream.
class InferenceResult
  attr_accessor :text, :prompt_tokens, :completion_tokens
  def initialize
    @text = ""
    @prompt_tokens = 0
    @completion_tokens = 0
  end
end

def run_inference(prompt_text, n_new)
  result = InferenceResult.new
  prompt_ids = GPT2BPE.encode(prompt_text, STATE.bpe)
  if n_new <= 0; n_new = 16; end
  if n_new > 256; n_new = 256; end
  new_ids = generate_ids(prompt_ids, n_new)
  result.text = GPT2BPE.decode(new_ids, STATE.bpe)
  result.prompt_tokens = prompt_ids.length
  result.completion_tokens = new_ids.length
  result
end

post '/v1/chat/completions' do
  res.headers["Content-Type"] = "application/json"
  body = req.body
  prompt_text = extract_messages_text(body)
  if prompt_text.length == 0
    res.set_status(400)
    "{\"error\":{\"message\":\"messages[].content not found\",\"type\":\"invalid_request_error\"}}\n"
  else
    n_new = 16
    if Tep::Json.has_key?(body, "max_tokens")
      n_new = Tep::Json.get_int(body, "max_tokens")
    end
    result = run_inference(prompt_text, n_new)
    "{" +
      Tep::Json.encode_pair_str("id", gen_id("chatcmpl")) + "," +
      Tep::Json.encode_pair_str("object", "chat.completion") + "," +
      Tep::Json.encode_pair_int("created", now_unix) + "," +
      Tep::Json.encode_pair_str("model", STATE.model_name) + "," +
      "\"choices\":[{\"index\":0,\"message\":{" +
        Tep::Json.encode_pair_str("role", "assistant") + "," +
        Tep::Json.encode_pair_str("content", result.text) +
      "},\"finish_reason\":\"length\"}]," +
      "\"usage\":{" +
        Tep::Json.encode_pair_int("prompt_tokens", result.prompt_tokens) + "," +
        Tep::Json.encode_pair_int("completion_tokens", result.completion_tokens) + "," +
        Tep::Json.encode_pair_int("total_tokens", result.prompt_tokens + result.completion_tokens) +
      "}}\n"
  end
end

post '/v1/completions' do
  body = req.body
  res.headers["Content-Type"] = "application/json"

  prompt_text = Tep::Json.get_str(body, "prompt")
  if prompt_text.length == 0
    res.set_status(400)
    "{\"error\":{\"message\":\"prompt is required\",\"type\":\"invalid_request_error\"}}\n"
  else
    n_new = 16
    if Tep::Json.has_key?(body, "max_tokens")
      n_new = Tep::Json.get_int(body, "max_tokens")
    end
    result = run_inference(prompt_text, n_new)
    "{" +
      Tep::Json.encode_pair_str("id", gen_id("cmpl")) + "," +
      Tep::Json.encode_pair_str("object", "text_completion") + "," +
      Tep::Json.encode_pair_int("created", now_unix) + "," +
      Tep::Json.encode_pair_str("model", STATE.model_name) + "," +
      "\"choices\":[{\"index\":0," +
        Tep::Json.encode_pair_str("text", result.text) + "," +
        "\"finish_reason\":\"length\"}]," +
      "\"usage\":{" +
        Tep::Json.encode_pair_int("prompt_tokens", result.prompt_tokens) + "," +
        Tep::Json.encode_pair_int("completion_tokens", result.completion_tokens) + "," +
        Tep::Json.encode_pair_int("total_tokens", result.prompt_tokens + result.completion_tokens) +
      "}}\n"
  end
end

get '/' do
  res.headers["Content-Type"] = "text/html; charset=utf-8"
  "<!doctype html><html><head><title>toy openai-compatible api</title></head>" +
  "<body><h1>toy openai-compatible api</h1>" +
  "<p>Model: " + STATE.model_name + " (gpt-2 124M)</p>" +
  "<p>Endpoints:</p><ul>" +
  "<li><code>GET /v1/models</code></li>" +
  "<li><code>POST /v1/chat/completions</code></li>" +
  "<li><code>POST /v1/completions</code></li>" +
  "<li><code>GET /health</code></li>" +
  "</ul></body></html>"
end
