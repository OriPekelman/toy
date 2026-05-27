# Tep + Spinel HTTP inference API. Loads a tiny TransformerLM, builds a
# FullForwardFFICache for accelerated forward, and exposes:
#
#   GET  /health           -> "ok"
#   GET  /model            -> {"vocab","d_model","d_ff","n_heads","n_layers"}
#   GET  /generate?n=N     -> {"prompt":[...], "ids":[...], "ms": ...}
#
# Build: spinel tep_demo/inference_api.rb -o tep_demo/api
# Run:   ./tep_demo/api -p 4567
# Bench: wrk -t4 -c64 -d10s "http://localhost:4567/generate?n=5"

require_relative "../lib/transformer"
require_relative "../lib/tinynn"
require_relative "../vendor/spinel/deps"

VOCAB    = 16
D_MODEL  = 32
D_FF     = 64
N_HEADS  = 4
N_LAYERS = 2
CONTEXT  = 16
T_SEQ    = CONTEXT

# Spinel-compat wrapper. `$global = TransformerLM.new(...)` emits a
# `static sp_TransformerLM * gv_model = NULL;` declaration BEFORE the
# struct definition, which doesn't compile. Using a singleton class
# instance assigned to a CONSTANT works because class defs are
# emitted before any constant initializer that references them.
class InferenceState
  attr_accessor :model, :fwd_cache, :pos_slice
  def initialize
    @model     = nil
    @fwd_cache = nil
    @pos_slice = nil
  end
end
STATE = InferenceState.new

# === Build the model (random weights -- generation isn't semantically
#     meaningful but the HTTP path is what we're benchmarking) ===
srand(42)
STATE.model = TransformerLM.new(VOCAB, D_MODEL, D_FF, N_HEADS, N_LAYERS, CONTEXT)
STATE.model.token_embed.fill_random(0.1)
STATE.model.pos_embed.fill_random(0.05)
i_init = 0
while i_init < D_MODEL
  STATE.model.norm_final_gamma[i_init] = 1.0
  i_init = i_init + 1
end
li_init = 0
while li_init < N_LAYERS
  b_init = STATE.model.blocks[li_init]
  i_init = 0
  while i_init < D_MODEL
    b_init.norm1_gamma[i_init] = 1.0
    b_init.norm2_gamma[i_init] = 1.0
    i_init = i_init + 1
  end
  b_init.fill_random_all(0.1)
  li_init = li_init + 1
end

# === FFI cache setup ===
STATE.fwd_cache = FullForwardFFICache.new
STATE.fwd_cache.realize_for(T_SEQ, D_MODEL, D_FF, N_HEADS, N_LAYERS, VOCAB)

TinyNN.upload_row_major(STATE.fwd_cache.sess, STATE.fwd_cache.t_token_embed, STATE.model.token_embed)
STATE.pos_slice = Mat.new(T_SEQ, D_MODEL)
i_init = 0
while i_init < T_SEQ * D_MODEL
  STATE.pos_slice.flat[i_init] = STATE.model.pos_embed.flat[i_init]
  i_init = i_init + 1
end
TinyNN.upload_row_major(STATE.fwd_cache.sess, STATE.fwd_cache.t_pos_slice, STATE.pos_slice)
TinyNN.tnn_upload_from_float_array(STATE.fwd_cache.sess, STATE.fwd_cache.t_final_norm_gamma,
                                    STATE.model.norm_final_gamma, D_MODEL)
li_init = 0
while li_init < N_LAYERS
  blk_n_init = STATE.model.blocks[li_init]
  blk_f_init = STATE.fwd_cache.blocks_ffi[li_init]
  TinyNN.tnn_upload_from_float_array(STATE.fwd_cache.sess, blk_f_init.t_norm1_gamma, blk_n_init.norm1_gamma, D_MODEL)
  TinyNN.tnn_upload_from_float_array(STATE.fwd_cache.sess, blk_f_init.t_norm2_gamma, blk_n_init.norm2_gamma, D_MODEL)
  h_init = 0
  while h_init < N_HEADS
    TinyNN.stage_transposed_and_upload(STATE.fwd_cache.sess, blk_f_init.t_w_q[h_init], blk_n_init.w_q[h_init])
    TinyNN.stage_transposed_and_upload(STATE.fwd_cache.sess, blk_f_init.t_w_k[h_init], blk_n_init.w_k[h_init])
    TinyNN.stage_transposed_and_upload(STATE.fwd_cache.sess, blk_f_init.t_w_v[h_init], blk_n_init.w_v[h_init])
    h_init = h_init + 1
  end
  TinyNN.stage_transposed_and_upload(STATE.fwd_cache.sess, blk_f_init.t_w_o,   blk_n_init.w_o)
  TinyNN.stage_transposed_and_upload(STATE.fwd_cache.sess, blk_f_init.t_w_ff1, blk_n_init.w_ff1)
  TinyNN.stage_transposed_and_upload(STATE.fwd_cache.sess, blk_f_init.t_w_ff2, blk_n_init.w_ff2)
  li_init = li_init + 1
end
puts "model ready (random weights)"

# === Generation helpers ===
def gen_pad_to_t(ids, t_seq)
  out = Array.new(t_seq, 0)
  i = 0
  n = ids.length
  if n > t_seq
    n = t_seq
  end
  while i < n
    out[i] = ids[i]
    i = i + 1
  end
  out
end

def gen_argmax_row(logits, row, vocab)
  best   = 0
  best_v = logits.flat[row * vocab]
  v = 1
  while v < vocab
    val = logits.flat[row * vocab + v]
    if val > best_v
      best_v = val
      best   = v
    end
    v = v + 1
  end
  best
end

def gen_via_ffi(cache, prompt, n_new, t_seq, vocab)
  ids = []
  i = 0
  while i < prompt.length
    ids.push(prompt[i])
    i = i + 1
  end
  step = 0
  while step < n_new
    padded = gen_pad_to_t(ids, t_seq)
    TinyNN.upload_int_array(cache.sess, cache.t_token_ids, padded)
    TinyNN.tnn_compute(cache.sess)
    logits = TinyNN.download_row_major(cache.sess, cache.t_logits, t_seq, vocab)
    next_id = gen_argmax_row(logits, ids.length - 1, vocab)
    ids.push(next_id)
    step = step + 1
  end
  ids
end

# === Handlers ===
class HealthHandler < Tep::Handler
  def handle(req, res)
    res.headers["Content-Type"] = "text/plain"
    "ok\n"
  end
end

class ModelHandler < Tep::Handler
  def handle(req, res)
    res.headers["Content-Type"] = "application/json"
    "{\"vocab\":" + VOCAB.to_s +
    ",\"d_model\":" + D_MODEL.to_s +
    ",\"d_ff\":" + D_FF.to_s +
    ",\"n_heads\":" + N_HEADS.to_s +
    ",\"n_layers\":" + N_LAYERS.to_s +
    ",\"context\":" + CONTEXT.to_s + "}\n"
  end
end

class GenerateHandler < Tep::Handler
  def handle(req, res)
    n_str = req.params["n"]
    if n_str == nil || n_str == ""
      n_str = "5"
    end
    n_new = n_str.to_i
    if n_new < 1
      n_new = 1
    end
    if n_new > 64
      n_new = 64
    end

    prompt = [3, 7, 1]
    t0 = Time.now
    ids = gen_via_ffi(STATE.fwd_cache, prompt, n_new, T_SEQ, VOCAB)
    t1 = Time.now
    ms = (t1 - t0) * 1000.0

    res.headers["Content-Type"] = "application/json"
    body = "{\"prompt\":["
    i = 0
    while i < prompt.length
      body = body + prompt[i].to_s
      if i < prompt.length - 1
        body = body + ","
      end
      i = i + 1
    end
    body = body + "],\"ids\":["
    i = 0
    while i < ids.length
      body = body + ids[i].to_s
      if i < ids.length - 1
        body = body + ","
      end
      i = i + 1
    end
    body = body + "],\"ms\":" + ms.to_s + "}\n"
    body
  end
end

class IndexHandler < Tep::Handler
  def handle(req, res)
    res.headers["Content-Type"] = "text/html; charset=utf-8"
    "<!doctype html><html><head><title>Toy LM Inference API</title></head><body>" +
    "<h1>Toy LM Inference API</h1>" +
    "<p>Tep (Sinatra-style) + Spinel (Ruby AOT) + tinynn (ggml FFI). " +
    "Forward via FullForwardFFICache " +
    "(persistent ggml graph; vocab=" + VOCAB.to_s + ", d_model=" + D_MODEL.to_s +
    ", n_layers=" + N_LAYERS.to_s + ", n_heads=" + N_HEADS.to_s + ").</p>" +
    "<ul>" +
    "<li><a href=\"/health\">GET /health</a></li>" +
    "<li><a href=\"/model\">GET /model</a></li>" +
    "<li><a href=\"/generate?n=5\">GET /generate?n=5</a></li>" +
    "</ul></body></html>\n"
  end
end

Tep.get "/",         IndexHandler.new
Tep.get "/health",   HealthHandler.new
Tep.get "/model",    ModelHandler.new
Tep.get "/generate", GenerateHandler.new
Tep.run!(4567, 1, false)
