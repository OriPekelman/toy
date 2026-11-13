# lib/toy/serve/openai/server.rb -- the Server state + boot + greedy
# decode for `toy serve` (OpenAI-compatible, llama-family GGUF).
#
# MOVED from tep_demo/openai_api_llama.rb:136-264 (P4 toy serve). The
# endpoint logic that USED to live in the tep_demo binary now lives in
# lib/toy/serve/openai/; the tep_demo source STAYS until the later
# cleanup pass retires it (serve must be gated first).
#
# Tep is consumed purely as HTTP transport (Tep::Handler / Tep.get/post/
# Tep.run!). JSON encode/decode is Toy::Json now, NOT Tep::Json (toy#44):
# serve's only remaining Tep coupling is the handler/server surface. Nothing
# here patches Tep -- this is build-dep usage, not "Tep re-adaptation" (that =
# Tep consuming Toy, DEFERRED per user directive until Toy is stable).
#
# CONTROLLED ENV contract (set by `toy serve` via lib/toy/run/serve.rb):
#   MODEL_PATH   path to any llama-family GGUF (default
#                data/smollm2-135m-native.gguf)
#   MODEL_NAME   user-facing label in /v1/models + completion responses.
#                Defaults to GGUF basename minus .gguf.
#   MAX_T        max context for the KV cache (default: the GGUF's
#                context_length clamped to [256, 4096]; toy#207)
#   STOP_IDS     extra end-of-generation token ids, comma-separated
#                (`toy serve --stop-ids`; toy#209)
#   PORT         (read in the entrypoint, not here)
#
# Spinel hygiene (#16): no #{} interpolation, no Struct.new, String-concat
# JSON, long-hand loops. KEEP the GH#188 landmine workarounds VERBATIM:
#   - CONSTANT-held STATE (so Spinel emits a typed slot)
#   - explicit STATE.model_name re-assign AFTER STATE.new
#   - rindex-based basename derivation (the rindex landmine workaround)
# Greedy api_generate_ids: pure argmax, first-max-wins, NO rand/seed/
# temperature -- CONFIRMED deterministic (this is what the HTTP gate
# reproduces byte-for-byte).

# GH#188 -- model selection via env. Defaults to SmolLM2-135M for a
# cheap-to-test smoke; override for any llama-family GGUF.
# If MODEL_NAME isn't set, it defaults to the basename of MODEL_PATH
# minus the ".gguf" suffix -- close enough for the /v1/models response.
GGUF_PATH = ENV["MODEL_PATH"] || "data/smollm2-135m-native.gguf"
MODEL_NAME_ENV = ENV["MODEL_NAME"] || ""
# Derive a default model name from the GGUF basename when not given.
# basename: strip everything before the last "/"; then strip ".gguf".
_mn_last_slash = GGUF_PATH.rindex("/")
_mn_basename   = _mn_last_slash == nil ? GGUF_PATH : GGUF_PATH[_mn_last_slash + 1..-1]
_mn_dot_gguf   = _mn_basename.rindex(".gguf")
_mn_default    = _mn_dot_gguf == nil ? _mn_basename : _mn_basename[0...(_mn_dot_gguf)]
MODEL_NAME = MODEL_NAME_ENV.length > 0 ? MODEL_NAME_ENV : _mn_default
# toy#207: MAX_T= (the CLI's --ctx) when given, otherwise the model's own
# context_length clamped to [256, 4096] — it was a literal 256.
MAX_T      = GGUFLoad.resolve_ctx(GGUF_PATH, ENV["MAX_T"] || "", "toy-serve")

# ---- Inference state. Class instance held as a CONSTANT so spinel
#      emits a typed slot for it (same pattern as inference_api.rb /
#      openai_api.rb). ----

class State
  # :req_seq is a monomorphic Integer slot (always Integer) for the
  # per-request monotonic counter the serving events emitter uses as a
  # deterministic-shaped request_id. Held on the already-CONSTANT-held STATE
  # so Spinel emits a typed slot; NO new polymorphic accessor (landmine #16).
  # toy#209: :tok is the GGUF's embedded tokenizer (present == false when
  # the file has none: the server then stays ids-only). :stop_ids are the
  # end-of-generation token ids, :chat_family the template family, and
  # :finish the finish_reason of the LAST generation ("stop" | "length").
  attr_accessor :cfg, :kv, :gguf, :model_name, :ready, :req_seq
  attr_accessor :tok, :stop_ids, :chat_family, :finish
  def initialize
    @cfg  = nil
    @kv   = nil
    @gguf = TinyNN.tnn_null_ptr
    @model_name = MODEL_NAME
    @ready      = false
    @req_seq    = 0
    @tok         = nil
    @stop_ids    = [0]
    @stop_ids.pop
    @chat_family = "chatml"
    @finish      = "length"
  end
end
STATE = State.new

# True when `id` ends generation (toy#209). Never true on a GGUF without
# a tokenizer (no stop ids).
def api_stop_id?(id)
  i = 0
  while i < STATE.stop_ids.length
    if STATE.stop_ids[i] == id
      return true
    end
    i = i + 1
  end
  false
end

# GH#188 -- explicit re-assignment AFTER STATE.new. State#initialize
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

# toy#209: the embedded tokenizer, when the GGUF has one. It gives text
# out (/v1/completions `text`), chat (/v1/chat/completions) and the stop
# ids. Without one the server is ids-only exactly as before: no stop ids,
# so generation runs to max_tokens.
STATE.tok = Tokenizer.from_gguf(GGUF_PATH)
if STATE.tok.present
  # End-of-generation ids: the GGUF's eos plus the end-of-turn specials of
  # the supported chat families, where the vocabulary has them.
  if STATE.tok.eos_id >= 0
    STATE.stop_ids.push(STATE.tok.eos_id)
  end
  _stop_names = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>", "<end_of_turn>"]
  _sn = 0
  while _sn < _stop_names.length
    _sid = STATE.tok.special_id(_stop_names[_sn])
    if _sid >= 0
      _dup = false
      _sj = 0
      while _sj < STATE.stop_ids.length
        if STATE.stop_ids[_sj] == _sid; _dup = true; end
        _sj = _sj + 1
      end
      if !_dup
        STATE.stop_ids.push(_sid)
      end
    end
    _sn = _sn + 1
  end
  # Chat template family, from the special tokens the vocabulary carries.
  if STATE.tok.special_id("<|im_start|>") >= 0
    STATE.chat_family = "chatml"
  elsif STATE.tok.special_id("<|start_header_id|>") >= 0
    STATE.chat_family = "llama3"
  elsif STATE.tok.special_id("<start_of_turn>") >= 0
    STATE.chat_family = "gemma2"
  elsif STATE.tok.spm
    STATE.chat_family = "mistral"
  else
    STATE.chat_family = "chatml"
  end
  puts "[openai_api_llama] tokenizer: " + STATE.tok.vocab_size.to_s + " tokens, " +
       STATE.tok.n_special.to_s + " special, " + STATE.stop_ids.length.to_s +
       " stop id(s), chat template " + STATE.chat_family
else
  puts "[openai_api_llama] no embedded tokenizer: ids-only (no text, no chat, no EOS stop)"
end

# STOP_IDS: extra end-of-generation token ids, comma-separated (`toy serve
# --stop-ids`). For a GGUF whose metadata names no eos, or a workload that
# ends on some other token. Applies with or without a tokenizer.
_extra_stops = (ENV["STOP_IDS"] || "").split(",")
_es = 0
while _es < _extra_stops.length
  _piece = _extra_stops[_es].strip
  if _piece.length > 0
    _eid = _piece.to_i
    if _eid.to_s != _piece || _eid < 0 || _eid >= STATE.cfg.vocab
      puts "[openai_api_llama] STOP_IDS: not a token id in [0," + STATE.cfg.vocab.to_s + "): " + _piece
      exit 1
    end
    if !api_stop_id?(_eid)
      STATE.stop_ids.push(_eid)
    end
  end
  _es = _es + 1
end
if _extra_stops.length > 0
  puts "[openai_api_llama] stop ids: " + STATE.stop_ids.length.to_s + " (with STOP_IDS)"
end

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
def api_argmax(logits, vocab)
  best_idx = 0
  best_val = logits.flat[0]
  v_iter = 1
  while v_iter < vocab
    val = logits.flat[v_iter]
    if val > best_val; best_val = val; best_idx = v_iter; end
    v_iter = v_iter + 1
  end
  best_idx
end

# Prefill the prompt; returns the FIRST generated token id.
def api_first_id(prompt_ids)
  vocab = STATE.cfg.vocab
  last_logits = Mat.new(1, vocab)
  prefill_pos = 0
  while prefill_pos < prompt_ids.length
    last_logits = SmolLM2KV.decode_step(STATE.kv, prompt_ids[prefill_pos], prefill_pos)
    prefill_pos = prefill_pos + 1
  end
  api_argmax(last_logits, vocab)
end

# Feed `prev_id` at position `pos`; returns the next token id.
def api_next_id(prev_id, pos)
  api_argmax(SmolLM2KV.decode_step(STATE.kv, prev_id, pos), STATE.cfg.vocab)
end

# Generates up to `n_new` ids and stops BEFORE a stop id (which is not
# returned). STATE.finish says which happened: "stop" or "length".
def api_generate_ids(prompt_ids, n_new)
  out_ids = [0]
  out_ids.pop
  STATE.finish = "length"
  if n_new <= 0
    return out_ids
  end

  id = api_first_id(prompt_ids)
  step = 0
  while step < n_new
    if api_stop_id?(id)
      STATE.finish = "stop"
      return out_ids
    end
    out_ids.push(id)
    step = step + 1
    if step < n_new
      id = api_next_id(id, prompt_ids.length + out_ids.length - 1)
    end
  end
  out_ids
end

# Byte length of the longest prefix of `s` that does not end inside a
# multi-byte UTF-8 character. A byte-level BPE token can end mid-character,
# so a streamed piece is cut here and the rest waits for the next token.
def api_utf8_safe_len(s)
  n = s.bytesize
  i = n - 1
  back = 0
  while i >= 0 && back < 4
    b = s.getbyte(i)
    if b < 128
      return n
    end
    if b >= 192
      need = 2
      if b >= 240
        need = 4
      elsif b >= 224
        need = 3
      end
      if back + 1 >= need
        return n
      end
      return i
    end
    i = i - 1
    back = back + 1
  end
  n
end
