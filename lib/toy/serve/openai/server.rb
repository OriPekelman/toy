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
#   PREFIX_FILE / PREFIX_CACHE / KV_REUSE   KV prefix (toy#210), below
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
#
# ---- Packed artifact (toy#211) ----
# `toy pack` appends the model, a config, a description and optionally a
# KV prefix file to this runner binary (layout: tinynn/tinynn_gguf.h). When
# the running executable carries a "model" entry the server is a PACK: the
# model, its name, the context, the stop ids and the KV prefix come from
# the pack and NOT from the environment, and the command line is read:
#
#   <artifact>              serve on the packed port (or PORT, or --port)
#   <artifact> --port N
#   <artifact> --describe   print the card and provenance, load nothing
#
# An unpacked runner ignores its command line, as before.
# A cut or damaged pack must not fall back to being a plain runner (it would
# look for a model in the environment or the working directory).
if TinyNN.tnn_pack_count < 0
  puts "[openai_api_llama] this executable carries a damaged pack (truncated or altered file); refusing to start"
  exit 1
end
PACK_ON = TinyNN.tnn_pack_count > 0 && TinyNN.tnn_pack_has("model") == 1

# Reads the pack's command line. Returns the --port value ("" when not
# given); --describe and --help print and exit here, before any model load.
def api_pack_args
  port = ""
  i = 0
  while i < ARGV.length
    a = ARGV[i]
    if a == "--describe"
      puts TinyNN.tnn_pack_text("describe")
      exit 0
    elsif a == "--help" || a == "-h"
      puts "usage: " + TinyNN.tnn_pack_cfg("MODEL_NAME") + " [--port N] | --describe"
      puts "  Serves the packed model over an OpenAI-compatible HTTP API."
      puts "  Port: --port, else PORT, else the packed default."
      exit 0
    elsif a == "--port" && i + 1 < ARGV.length
      port = ARGV[i + 1]
      if port.to_i.to_s != port || port.to_i < 1 || port.to_i > 65535
        puts "not a TCP port: " + port
        exit 2
      end
      i = i + 1
    else
      puts "unknown argument: " + a + " (try --help)"
      exit 2
    end
    i = i + 1
  end
  port
end
PACK_PORT_ARG = PACK_ON ? api_pack_args : ""

# A setting of the serve contract: the pack's config in a pack, otherwise
# the environment `toy serve` controls.
def api_setting(key)
  if PACK_ON
    return TinyNN.tnn_pack_cfg(key)
  end
  ENV[key] || ""
end

# The port to bind, as a string: --port (pack only), then PORT, then the
# packed default, then 4567.
def api_port_setting
  if PACK_PORT_ARG.length > 0
    return PACK_PORT_ARG
  end
  v = ENV["PORT"] || ""
  if v.length > 0
    return v
  end
  if PACK_ON
    v = TinyNN.tnn_pack_cfg("PORT")
    if v.length > 0
      return v
    end
  end
  "4567"
end

GGUF_PATH = PACK_ON ? "@pack:model" : (ENV["MODEL_PATH"] || "data/smollm2-135m-native.gguf")
MODEL_NAME_ENV = api_setting("MODEL_NAME")
# Derive a default model name from the GGUF basename when not given.
# basename: strip everything before the last "/"; then strip ".gguf".
_mn_last_slash = GGUF_PATH.rindex("/")
_mn_basename   = _mn_last_slash == nil ? GGUF_PATH : GGUF_PATH[_mn_last_slash + 1..-1]
_mn_dot_gguf   = _mn_basename.rindex(".gguf")
_mn_default    = _mn_dot_gguf == nil ? _mn_basename : _mn_basename[0...(_mn_dot_gguf)]
MODEL_NAME = MODEL_NAME_ENV.length > 0 ? MODEL_NAME_ENV : _mn_default
# toy#207: MAX_T= (the CLI's --ctx) when given, otherwise the model's own
# context_length clamped to [256, 4096] — it was a literal 256.
MAX_T      = GGUFLoad.resolve_ctx(GGUF_PATH, api_setting("MAX_T"), "toy-serve")
# toy#210 (see the KV prefix block below). KV_REUSE=0 turns off the
# reuse of cached rows between requests (every prompt is evaluated from
# position 0, as before); for measurement and for the gate's reference.
# toy#211: a pack carries the finished KV prefix file as its "kv" entry and
# no prefix text to rebuild it from.
PREFIX_FILE  = PACK_ON ? "" : (ENV["PREFIX_FILE"] || "")
PREFIX_CACHE = PACK_ON ? (TinyNN.tnn_pack_has("kv") == 1 ? "@pack:kv" : "") : (ENV["PREFIX_CACHE"] || "")
KV_REUSE_OFF = (ENV["KV_REUSE"] || "") == "0"

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
  # toy#210: :kv_ids are the token ids whose K/V rows are currently in the
  # cache (row i holds token kv_ids[i]); :cached is how many of them the
  # LAST request reused instead of re-evaluating; :model_fp fingerprints
  # the GGUF for the KV prefix file.
  attr_accessor :kv_ids, :cached, :model_fp
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
    @kv_ids      = [0]
    @kv_ids.pop
    @cached      = 0
    @model_fp    = ""
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

# ---- KV prefix file (toy#210) ----
#
# Layout (tinynn's tnn_kvf_* record file): format tag, model fingerprint,
# backend name, shape ints [n_layers, n_kv, d_head, kv_type_k, kv_type_v], the prefix
# token ids, then per layer and KV tensor the first n rows of K and of V.
#
# The tag names the row layout: "-1" is one tensor per KV head (the
# per-head decode graph), "-2" one tensor per layer holding every head of a
# position (the fused graph, toy#206). A file is only read back by a server
# using the layout it was written with.

KV_FILE_TAG       = "toy-kv-prefix-1"
KV_FILE_TAG_FUSED = "toy-kv-prefix-2"

def api_kv_tag
  if STATE.kv.fused
    return KV_FILE_TAG_FUSED
  end
  KV_FILE_TAG
end

# Saves the first `n` cached positions. Returns "" on success, otherwise
# the reason.
def api_kv_save(path, n)
  kv = STATE.kv
  if kv.is_mla
    return "KV prefix files are not supported for MLA attention models"
  end
  if n <= 0 || n > STATE.kv_ids.length
    return "nothing cached to save"
  end
  if TinyNN.tnn_kvf_open_w(path) != 0
    return "cannot write " + path
  end
  TinyNN.tnn_kvf_put_str(api_kv_tag)
  TinyNN.tnn_kvf_put_str(STATE.model_fp)
  TinyNN.tnn_kvf_put_str(TinyNN.tnn_backend_name(kv.sess))
  shape = [kv.n_layers, kv.n_kv, kv.d_head, kv.kv_type_k, kv.kv_type_v]
  TinyNN.tnn_kvf_put_ints(shape, 5)
  ids = [0]
  ids.pop
  i = 0
  while i < n
    ids.push(STATE.kv_ids[i])
    i = i + 1
  end
  TinyNN.tnn_kvf_put_ints(ids, n)
  li = 0
  while li < kv.n_layers
    blk = kv.kv_blocks_ffi[li]
    h = 0
    while h < kv.kv_tensor_count
      TinyNN.tnn_kvf_put_tensor(blk.t_K[h], n)
      TinyNN.tnn_kvf_put_tensor(blk.t_V[h], n)
      h = h + 1
    end
    li = li + 1
  end
  if TinyNN.tnn_kvf_close != 0
    return "write to " + path + " failed"
  end
  ""
end

# Restores a prefix file into the cache and STATE.kv_ids. Returns "" on
# success, otherwise the reason (the cache is then treated as empty).
def api_kv_restore(path)
  kv = STATE.kv
  if kv.is_mla
    return "KV prefix files are not supported for MLA attention models"
  end
  if TinyNN.tnn_kvf_open_r(path) != 0
    return "not a toy KV prefix file"
  end
  why = api_kv_restore_open(kv)
  TinyNN.tnn_kvf_close
  if why.length > 0
    while STATE.kv_ids.length > 0
      STATE.kv_ids.pop
    end
  end
  why
end

def api_kv_restore_open(kv)
  tag = TinyNN.tnn_kvf_get_str
  if tag != api_kv_tag
    if tag == KV_FILE_TAG || tag == KV_FILE_TAG_FUSED
      return "its cache layout does not match this server (fused vs per-head KV)"
    end
    return "unknown KV prefix file version"
  end
  if TinyNN.tnn_kvf_get_str != STATE.model_fp
    return "it was made from a different model file"
  end
  # K/V rows computed on another backend differ in the low bits, so a
  # reply could differ from one this server computes itself.
  if TinyNN.tnn_kvf_get_str != TinyNN.tnn_backend_name(kv.sess)
    return "it was made on a different device"
  end
  shape = [0, 0, 0, 0, 0]
  if TinyNN.tnn_kvf_get_ints(shape, 5) != 0
    return "damaged header"
  end
  if shape[0] != kv.n_layers || shape[1] != kv.n_kv || shape[2] != kv.d_head ||
     shape[3] != kv.kv_type_k || shape[4] != kv.kv_type_v
    return "its cache layout does not match this server (layers/heads/KV type)"
  end
  n = TinyNN.tnn_kvf_peek_ints
  if n <= 0
    return "damaged header"
  end
  if n >= MAX_T
    return "its " + n.to_s + "-token prefix does not fit context " + MAX_T.to_s
  end
  ids = [0]
  ids.pop
  i = 0
  while i < n
    ids.push(0)
    i = i + 1
  end
  if TinyNN.tnn_kvf_get_ints(ids, n) != 0
    return "damaged token list"
  end
  while STATE.kv_ids.length > 0
    STATE.kv_ids.pop
  end
  li = 0
  while li < kv.n_layers
    blk = kv.kv_blocks_ffi[li]
    h = 0
    while h < kv.kv_tensor_count
      if TinyNN.tnn_kvf_get_tensor(blk.t_K[h], n) != 0 ||
         TinyNN.tnn_kvf_get_tensor(blk.t_V[h], n) != 0
        return "damaged or truncated tensor data"
      end
      h = h + 1
    end
    li = li + 1
  end
  i = 0
  while i < n
    STATE.kv_ids.push(ids[i])
    i = i + 1
  end
  ""
end

# Evaluate ids[from...] into the cache (no sampling); used for the boot
# prefix. Positions below `from` must already be cached.
def api_kv_feed(ids, from)
  if from >= ids.length
    return 0
  end
  SmolLM2KV.decode_batch(STATE.kv, ids, from, ids.length - from, from)
  pos = from
  while pos < ids.length
    STATE.kv_ids.push(ids[pos])
    pos = pos + 1
  end
  0
end

# GH#188 -- explicit re-assignment AFTER STATE.new. State#initialize
# captures `@model_name = MODEL_NAME` but Spinel's ivar typing
# appears to evaluate it before the MODEL_NAME ternary resolves
# (the embeddings handler receives MODEL_NAME via constructor at
# boot and prints it correctly; STATE.model_name accessed at
# request time shows empty). Setting it explicitly here works
# around it cleanly without touching the State class.
STATE.model_name = MODEL_NAME

if PACK_ON
  puts "[openai_api_llama] packed artifact: model, config" +
       (PREFIX_CACHE.length > 0 ? ", KV prefix" : "") + " read from this executable"
end
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
# toy#206: KV_FUSED=0 keeps the per-head decode graph; KV_BATCH=<n> caps
# the prompt tokens one batched step evaluates.
if (ENV["KV_FUSED"] || "") == "0"
  STATE.kv.disable_fused!
end
if (ENV["KV_BATCH"] || "").to_i > 0
  STATE.kv.batch_max = (ENV["KV_BATCH"] || "").to_i
end
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
_extra_stops = api_setting("STOP_IDS").split(",")
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

# ---- KV prefix (toy#210) ----
# PREFIX_FILE  text of the system message every request starts with. It is
#              rendered as the chat template's system turn and evaluated
#              once at boot, so requests only evaluate what follows it.
# PREFIX_CACHE a KV prefix file: restored at boot when it matches this
#              model (and PREFIX_FILE, when given), otherwise rebuilt from
#              PREFIX_FILE and saved.
STATE.model_fp = TinyNN.tnn_file_fingerprint(GGUF_PATH)
_prefix_ids = [0]
_prefix_ids.pop
if PREFIX_FILE.length > 0
  if !STATE.tok.present
    puts "[openai_api_llama] --prefix-file needs a GGUF with an embedded tokenizer"
    exit 1
  end
  if !File.exist?(PREFIX_FILE)
    puts "[openai_api_llama] no such prefix file: " + PREFIX_FILE
    exit 1
  end
  _ptext = File.read(PREFIX_FILE)
  # One trailing newline is the editor's, not the prompt's.
  if _ptext.length > 0 && _ptext[_ptext.length - 1] == "\n"
    _ptext = _ptext[0, _ptext.length - 1]
  end
  _ppairs = [["", ""]]
  _ppairs.delete_at(0)
  _ppairs.push(["system", _ptext])
  _prefix_ids = STATE.tok.encode(ToyChatTemplate.apply(STATE.chat_family, _ppairs, false))
  if _prefix_ids.length == 0 || _prefix_ids.length >= MAX_T
    puts "[openai_api_llama] prefix is " + _prefix_ids.length.to_s +
         " tokens; it must be non-empty and fit context " + MAX_T.to_s
    exit 1
  end
end
_restored = false
if PREFIX_CACHE.length > 0 && (PACK_ON || File.exist?(PREFIX_CACHE))
  _t0 = Time.now.to_f
  _why = api_kv_restore(PREFIX_CACHE)
  if _why.length == 0 && _prefix_ids.length > 0
    # The file must hold exactly the prefix asked for.
    _same = STATE.kv_ids.length == _prefix_ids.length
    _pi = 0
    while _same && _pi < _prefix_ids.length
      if STATE.kv_ids[_pi] != _prefix_ids[_pi]; _same = false; end
      _pi = _pi + 1
    end
    if !_same
      _why = "it holds a different prefix than --prefix-file"
      while STATE.kv_ids.length > 0
        STATE.kv_ids.pop
      end
    end
  end
  if _why.length == 0
    _restored = true
    puts "[openai_api_llama] KV prefix: restored " + STATE.kv_ids.length.to_s + " tokens from " +
         PREFIX_CACHE + " in " + ((Time.now.to_f - _t0) * 1000.0).round(1).to_s + " ms"
  elsif _prefix_ids.length > 0
    puts "[openai_api_llama] KV prefix: " + PREFIX_CACHE + " not used (" + _why + "); rebuilding"
  else
    # Nothing to rebuild it from: refuse rather than serve without the
    # prefix the operator asked for.
    puts "[openai_api_llama] KV prefix: cannot use " + PREFIX_CACHE + " (" + _why +
         (PACK_ON ? "); repack the artifact" : "); pass --prefix-file to rebuild it")
    exit 1
  end
elsif PREFIX_CACHE.length > 0 && _prefix_ids.length == 0
  puts "[openai_api_llama] KV prefix: " + PREFIX_CACHE + " does not exist and there is no --prefix-file to build it from"
  exit 1
end
if !_restored && _prefix_ids.length > 0
  _t0 = Time.now.to_f
  api_kv_feed(_prefix_ids, 0)
  puts "[openai_api_llama] KV prefix: evaluated " + _prefix_ids.length.to_s + " tokens in " +
       ((Time.now.to_f - _t0) * 1000.0).round(1).to_s + " ms"
  if PREFIX_CACHE.length > 0
    _t0 = Time.now.to_f
    _why = api_kv_save(PREFIX_CACHE, _prefix_ids.length)
    if _why.length > 0
      puts "[openai_api_llama] KV prefix: could not save " + PREFIX_CACHE + " (" + _why + ")"
      exit 1
    end
    puts "[openai_api_llama] KV prefix: saved to " + PREFIX_CACHE + " in " +
         ((Time.now.to_f - _t0) * 1000.0).round(1).to_s + " ms"
  end
end

STATE.ready = true

# SERVE_BOOT_ONLY=1: load, build and save the KV prefix, then exit without
# serving. `toy pack` uses it to produce the KV prefix file it embeds.
if (ENV["SERVE_BOOT_ONLY"] || "") == "1"
  puts "[openai_api_llama] boot only: not serving"
  exit 0
end
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
# prefill the prompt (batched, toy#206), then sample greedily for
# `n_new` more steps. Returns Array<Int> of the new token IDs (does
# NOT include the prompt).
#
# The cache's t_K / t_V tensors are persistent; rows a prompt shares with
# what is already cached are kept (toy#210, api_first_id) and the rest are
# overwritten in place.
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
#
# toy#210: the K/V rows of the longest prefix this prompt shares with what
# is already in the cache (the previous request, or a restored prefix file)
# are kept, and evaluation starts after them. The last prompt token is
# always evaluated, since its logits are what the caller needs. The result
# is the same ids as a full prefill: the kept rows are the rows a full
# prefill would write.
def api_first_id(prompt_ids)
  vocab = STATE.cfg.vocab
  lcp = 0
  lim = STATE.kv_ids.length
  if prompt_ids.length < lim
    lim = prompt_ids.length
  end
  while lcp < lim && STATE.kv_ids[lcp] == prompt_ids[lcp]
    lcp = lcp + 1
  end
  start = lcp
  if start > prompt_ids.length - 1
    start = prompt_ids.length - 1
  end
  if KV_REUSE_OFF
    start = 0
  end
  STATE.cached = start
  while STATE.kv_ids.length > start
    STATE.kv_ids.pop
  end
  # toy#206: the uncached part of the prompt goes through batched steps
  # (one token at a time on a per-head cache).
  last_logits = SmolLM2KV.decode_batch(STATE.kv, prompt_ids, start,
                                       prompt_ids.length - start, start)
  prefill_pos = start
  while prefill_pos < prompt_ids.length
    STATE.kv_ids.push(prompt_ids[prefill_pos])
    prefill_pos = prefill_pos + 1
  end
  api_argmax(last_logits, vocab)
end

# Feed `prev_id` at position `pos`; returns the next token id.
def api_next_id(prev_id, pos)
  while STATE.kv_ids.length > pos
    STATE.kv_ids.pop
  end
  logits = SmolLM2KV.decode_step(STATE.kv, prev_id, pos)
  STATE.kv_ids.push(prev_id)
  api_argmax(logits, STATE.cfg.vocab)
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
