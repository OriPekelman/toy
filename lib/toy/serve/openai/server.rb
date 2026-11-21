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
# Greedy decode (api_pump / tnn_logits_pick): pure argmax, first-max-wins,
# NO rand/seed/temperature -- CONFIRMED deterministic (this is what the HTTP gate
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
  port  = ""
  slots = ""
  i = 0
  while i < ARGV.length
    a = ARGV[i]
    if a == "--describe"
      puts TinyNN.tnn_pack_text("describe")
      exit 0
    elsif a == "--help" || a == "-h"
      puts "usage: " + TinyNN.tnn_pack_cfg("MODEL_NAME") + " [--port N] [--slots N] | --describe"
      puts "  Serves the packed model over an OpenAI-compatible HTTP API."
      puts "  Port: --port, else PORT, else the packed default."
      puts "  Requests decoded at the same time: --slots, else SLOTS, else the packed default."
      exit 0
    elsif a == "--port" && i + 1 < ARGV.length
      port = ARGV[i + 1]
      if port.to_i.to_s != port || port.to_i < 1 || port.to_i > 65535
        puts "not a TCP port: " + port
        exit 2
      end
      i = i + 1
    elsif a == "--slots" && i + 1 < ARGV.length
      slots = ARGV[i + 1]
      if slots.to_i.to_s != slots || slots.to_i < 1
        puts "not a number of slots: " + slots
        exit 2
      end
      i = i + 1
    else
      puts "unknown argument: " + a + " (try --help)"
      exit 2
    end
    i = i + 1
  end
  [port, slots]
end
PACK_ARGS      = PACK_ON ? api_pack_args : ["", ""]
PACK_PORT_ARG  = PACK_ARGS[0]
PACK_SLOTS_ARG = PACK_ARGS[1]

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
# toy#217: how many requests are decoded at the same time (`toy serve
# --slots N`; in a pack --slots, then SLOTS, then the packed default).
# 1 serves one request at a time, as before.
SLOTS_MAX = 64
def api_slots_setting
  v = PACK_SLOTS_ARG
  if v.length == 0
    v = ENV["SLOTS"] || ""
  end
  if v.length == 0 && PACK_ON
    v = TinyNN.tnn_pack_cfg("SLOTS")
  end
  if v.length == 0
    return 1
  end
  n = v.to_i
  if n.to_s != v || n < 1 || n > SLOTS_MAX
    puts "[openai_api_llama] SLOTS must be a number from 1 to " + SLOTS_MAX.to_s + ": " + v
    exit 1
  end
  n
end
SERVE_SLOTS = api_slots_setting
# Test hook for the request-failure path (prep/serve_chat_gate.rb): a
# request whose prompt is exactly this many tokens raises inside generation.
# 0 (the default) never matches.
SERVE_TEST_FAULT_AT = (ENV["SERVE_TEST_FAULT_AT"] || "0").to_i

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
  # end-of-generation token ids and :chat_family the template family.
  attr_accessor :cfg, :kv, :gguf, :model_name, :ready, :req_seq
  attr_accessor :tok, :stop_ids, :chat_family
  # toy#210: :kv_ids are the token ids of the SHARED PREFIX: the K/V rows
  # evaluated or restored at boot, cache cells 0 .. kv_ids.length-1, which
  # every request that starts with those ids reads instead of evaluating
  # them. :model_fp fingerprints the GGUF for the KV prefix file.
  attr_accessor :kv_ids, :model_fp
  # toy#217, the cell pool. The K/V tensors hold :n_cells cells; cell c
  # holds position :cell_pos[c] of the sequence :cell_owner[c] (a slot
  # index; -1 = the shared prefix; -2 = free). :cell_hi is one past the
  # highest cell in use, :cell_lo a lower bound on the lowest free one.
  # :step_seq counts decode steps (api_pump).
  attr_accessor :n_cells, :cell_owner, :cell_pos, :cell_hi, :cell_lo, :step_seq
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
    @kv_ids      = [0]
    @kv_ids.pop
    @model_fp    = ""
    @n_cells     = 0
    @cell_owner  = [0]
    @cell_owner.pop
    @cell_pos    = [0]
    @cell_pos.pop
    @cell_hi     = 0
    @cell_lo     = 0
    @step_seq    = 0
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
# toy#217: a full context of cells per slot, plus one more when a boot
# prefix is configured (its cells are shared and never given to a slot, and
# it is shorter than the context).
STATE.kv.kv_cells = MAX_T * SERVE_SLOTS
if PREFIX_FILE.length > 0 || PREFIX_CACHE.length > 0
  STATE.kv.kv_cells = STATE.kv.kv_cells + MAX_T
end
STATE.gguf = STATE.kv.realize_and_load_auto(GGUF_PATH, MAX_T, STATE.cfg, flags)
if SERVE_SLOTS > 1 && !STATE.kv.fused
  puts "[openai_api_llama] --slots " + SERVE_SLOTS.to_s + " needs the fused decode graph, which this " +
       "model or configuration does not use (MoE, MLA, a quantized KV cache, KV_FUSED=0, ...); " +
       "start with one slot"
  exit 1
end

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

# ---- Cell pool and slots (toy#217) ----
# The boot prefix holds cells 0 .. P-1 as the shared sequence (owner -1);
# everything else is free.
STATE.n_cells = STATE.kv.fused ? STATE.kv.kv_cells : MAX_T
_ci = 0
while _ci < STATE.n_cells
  if _ci < STATE.kv_ids.length
    STATE.cell_owner.push(-1)
    STATE.cell_pos.push(_ci)
  else
    STATE.cell_owner.push(-2)
    STATE.cell_pos.push(0)
  end
  _ci = _ci + 1
end
STATE.cell_hi = STATE.kv_ids.length
STATE.cell_lo = STATE.kv_ids.length
if SERVE_SLOTS > 1
  puts "[openai_api_llama] slots: " + SERVE_SLOTS.to_s + " (" + STATE.n_cells.to_s + " cache cells)"
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

# A request field that is one string or an array of strings (OpenAI's
# `stop`) as an array; empty when the key is absent or holds anything else.
def api_json_str_list(body, key)
  out = [""]
  out.pop
  pos = SpinelKit::Json.find_value_start(body, key)
  if pos < 0 || pos >= body.length
    return out
  end
  if body[pos] == "\""
    one = SpinelKit::Json.parse_str_value(body, pos)
    if one.bytesize > 0
      out.push(one)
    end
    return out
  end
  if body[pos] != "["
    return out
  end
  pos = pos + 1
  while pos < body.length
    pos = SpinelKit::Json.skip_ws(body, pos)
    if pos >= body.length || body[pos] == "]"
      return out
    end
    if body[pos] == "\""
      one = SpinelKit::Json.parse_str_value(body, pos)
      if one.bytesize > 0
        out.push(one)
      end
      pos = SpinelKit::Json.skip_str(body, pos)
      if pos < 0
        return out
      end
    elsif body[pos] == ","
      pos = pos + 1
    else
      pos = SpinelKit::Json.skip_value(body, pos)
    end
  end
  out
end

# True when `s` is well-formed UTF-8 (a single BPE token may not be).
def api_utf8_valid?(s)
  n = s.bytesize
  i = 0
  while i < n
    b = s.getbyte(i)
    need = 0
    if b < 128
      need = 0
    elsif b >= 194 && b < 224
      need = 1
    elsif b >= 224 && b < 240
      need = 2
    elsif b >= 240 && b < 245
      need = 3
    else
      return false
    end
    if i + need > n - 1
      return false
    end
    j = 1
    while j <= need
      c = s.getbyte(i + j)
      if c < 128 || c >= 192
        return false
      end
      j = j + 1
    end
    i = i + need + 1
  end
  true
end

# Byte offset of the first occurrence of `needle` in `hay`, or -1.
def api_byte_find(hay, needle)
  n = hay.bytesize
  m = needle.bytesize
  if m == 0 || m > n
    return -1
  end
  first = needle.getbyte(0)
  i = 0
  while i <= n - m
    if hay.getbyte(i) == first
      j = 1
      while j < m && hay.getbyte(i + j) == needle.getbyte(j)
        j = j + 1
      end
      if j == m
        return i
      end
    end
    i = i + 1
  end
  -1
end

# Earliest byte offset in `text` where one of `strs` begins; -1 when none
# is there.
def api_stop_cut(text, strs)
  cut = -1
  i = 0
  while i < strs.length
    at = api_byte_find(text, strs[i])
    if at >= 0 && (cut < 0 || at < cut)
      cut = at
    end
    i = i + 1
  end
  cut
end

# ---- Slots (toy#217) ----
#
# A SLOT is one sequence being decoded: the request it serves and the cache
# cells that hold its K/V rows. Every slot shares one pool of cells (the
# fused cache's K/V tensors); a decode step (api_pump) carries the pending
# tokens of ALL busy slots through one graph, and the visibility mask keeps
# each token to the cells of its own sequence and of the shared prefix. So
# N requests cost one graph evaluation per token position, not N.
#
# A slot keeps its cells after its request ends: the next request that
# starts with the same ids reuses them (toy#210). A request takes the idle
# slot whose cached ids share the longest prefix with its prompt.
#
# Per-head caches (MoE, MLA, quantized KV, ...) have one slot, and there a
# cell is a position: the step is the per-head decode, one token at a time.
class Slot
  # :busy      a request holds the slot
  # :seq_ids   ids whose K/V rows are cached for this slot, by position
  # :shared_n  how many leading positions are the shared prefix's cells
  # :cells     the cell of each later position (cells[i] holds position
  #            shared_n + i)
  attr_accessor :idx, :busy, :seq_ids, :shared_n, :cells
  # The request:
  #   :prompt_ids / :fed   the prompt, and how many of its ids are cached
  #   :n_new      most tokens to generate
  #   :out_ids    generated so far (a stop id is not included)
  #   :next_id    the id to feed in the next step (generation phase)
  #   :phase      0 = evaluating the prompt, 1 = generating, 2 = done
  #   :finish     "stop" | "length"
  #   :cached     prompt positions reused instead of evaluated
  #   :err        why the request failed ("" = it did not)
  #   :seen       the last decode step this request's handler looked at
  #   :shared_steps  decode steps this request shared with another one
  attr_accessor :prompt_ids, :fed, :n_new, :out_ids, :next_id, :phase
  attr_accessor :finish, :cached, :err, :seen, :shared_steps, :step_mark
  # toy#216, per-request options and their results:
  #   :lp_top     -1 = no log-probabilities; n >= 0 = record each picked
  #               token's log-probability and the n most likely tokens
  #   :lp_tok     log-probability of each token picked
  #   :lp_alt_ids / :lp_alt_lps   the n most likely ids / their
  #               log-probabilities, n entries per pick, best first
  #   :stop_strs  strings that end the reply when they appear in its text
  #   :stop_cut   byte offset in the decoded reply where a stop string
  #               began (-1: none matched)
  attr_accessor :lp_top, :lp_tok, :lp_alt_ids, :lp_alt_lps, :stop_strs, :stop_cut
  def initialize(idx)
    @idx      = idx
    @busy     = false
    @seq_ids  = [0]
    @seq_ids.pop
    @shared_n = 0
    @cells    = [0]
    @cells.pop
    @prompt_ids = [0]
    @prompt_ids.pop
    @fed      = 0
    @n_new    = 0
    @out_ids  = [0]
    @out_ids.pop
    @next_id  = 0
    @phase    = 2
    @finish   = "length"
    @cached   = 0
    @err      = ""
    @seen     = 0
    @shared_steps = 0
    @step_mark    = 0
    @lp_top   = -1
    @lp_tok   = [0.0]
    @lp_tok.pop
    @lp_alt_ids = [0]
    @lp_alt_ids.pop
    @lp_alt_lps = [0.0]
    @lp_alt_lps.pop
    @stop_strs = [""]
    @stop_strs.pop
    @stop_cut  = -1
  end

  def done
    @phase == 2
  end
end

SLOTS = [Slot.new(0)]
_sl = 1
while _sl < SERVE_SLOTS
  SLOTS.push(Slot.new(_sl))
  _sl = _sl + 1
end

# Takes the lowest free cell for position `pos` of sequence `owner`.
def api_cell_take(owner, pos)
  c = STATE.cell_lo
  while c < STATE.n_cells && STATE.cell_owner[c] != -2
    c = c + 1
  end
  if c >= STATE.n_cells
    raise "KV cell pool exhausted (" + STATE.n_cells.to_s + " cells)"
  end
  STATE.cell_owner[c] = owner
  STATE.cell_pos[c]   = pos
  STATE.cell_lo = c + 1
  if c + 1 > STATE.cell_hi
    STATE.cell_hi = c + 1
  end
  c
end

def api_cell_free(c)
  STATE.cell_owner[c] = -2
  if c < STATE.cell_lo
    STATE.cell_lo = c
  end
  while STATE.cell_hi > 0 && STATE.cell_owner[STATE.cell_hi - 1] == -2
    STATE.cell_hi = STATE.cell_hi - 1
  end
  0
end

# Forgets the slot's cached positions from `keep` on (their cells go back
# to the pool). Positions below shared_n are the shared prefix's and stay
# where they are; keeping fewer than shared_n of them lowers shared_n.
def api_slot_truncate(slot, keep)
  own_keep = keep - slot.shared_n
  if own_keep < 0
    own_keep = 0
  end
  while slot.cells.length > own_keep
    api_cell_free(slot.cells.pop)
  end
  while slot.seq_ids.length > keep
    slot.seq_ids.pop
  end
  if slot.shared_n > keep
    slot.shared_n = keep
  end
  0
end

# Length of the common prefix of two id lists.
def api_lcp(a, b)
  lim = a.length
  if b.length < lim
    lim = b.length
  end
  n = 0
  while n < lim && a[n] == b[n]
    n = n + 1
  end
  n
end

# Gives a request an idle slot, primed with what the cache already holds of
# its prompt. Returns the slot's index; -1 when every slot is busy.
#
# toy#210: the K/V rows of the longest prefix the prompt shares with the
# slot's cached ids, or with the shared boot prefix, are kept, and
# evaluation starts after them. The last prompt token is always evaluated,
# since its logits are what the request needs. The result is the same ids
# as a full prefill: the kept rows are the rows a full prefill would write.
def api_slot_acquire(prompt_ids, n_new)
  best = -1
  best_lcp = -1
  i = 0
  while i < SLOTS.length
    if !SLOTS[i].busy
      l = api_lcp(SLOTS[i].seq_ids, prompt_ids)
      if l > best_lcp
        best_lcp = l
        best = i
      end
    end
    i = i + 1
  end
  if best < 0
    return -1
  end
  slot = SLOTS[best]
  pre_lcp = api_lcp(STATE.kv_ids, prompt_ids)
  if KV_REUSE_OFF
    best_lcp = 0
    pre_lcp  = 0
  end
  start = best_lcp
  if pre_lcp > best_lcp
    # The boot prefix covers more of this prompt than the slot's own
    # sequence: drop that and read the prefix cells.
    api_slot_truncate(slot, 0)
    start = pre_lcp
  end
  if start > prompt_ids.length - 1
    start = prompt_ids.length - 1
  end
  if pre_lcp > best_lcp
    slot.shared_n = start
    k = 0
    while k < start
      slot.seq_ids.push(prompt_ids[k])
      k = k + 1
    end
  else
    api_slot_truncate(slot, start)
  end
  if !STATE.kv.fused && slot.shared_n < STATE.kv_ids.length
    # Per-head cache: a cell is a position, so a prompt that leaves the
    # boot prefix early overwrites the rest of it, as before toy#217.
    while STATE.kv_ids.length > slot.shared_n
      api_cell_free(STATE.kv_ids.length - 1)
      STATE.kv_ids.pop
    end
  end
  slot.busy       = true
  slot.prompt_ids = prompt_ids
  slot.fed        = start
  slot.cached     = start
  slot.n_new      = n_new
  slot.out_ids    = [0]
  slot.out_ids.pop
  slot.phase      = 0
  slot.finish     = "length"
  slot.err        = ""
  slot.seen       = STATE.step_seq
  slot.shared_steps = 0
  slot.lp_top     = -1
  slot.lp_tok.clear
  slot.lp_alt_ids.clear
  slot.lp_alt_lps.clear
  slot.stop_strs.clear
  slot.stop_cut   = -1
  best
end

# Ends the slot's request. Its cached positions stay for the next one.
def api_slot_release(slot)
  slot.busy  = false
  slot.phase = 2
  0
end

# One generated id for the slot: ends the request at a stop id (not
# returned), at a stop string, or at n_new ids; otherwise the id is fed in
# the next step.
def api_slot_take(slot, id)
  slot.phase = 1
  if api_stop_id?(id)
    slot.finish = "stop"
    slot.phase  = 2
    return 0
  end
  slot.out_ids.push(id)
  # toy#216: a stop string in the text so far ends the reply. The ids that
  # spell it stay in out_ids; the caller cuts the text at stop_cut.
  if slot.stop_strs.length > 0
    cut = api_stop_cut(STATE.tok.decode(slot.out_ids), slot.stop_strs)
    if cut >= 0
      slot.stop_cut = cut
      slot.finish   = "stop"
      slot.phase    = 2
      return 0
    end
  end
  if slot.out_ids.length >= slot.n_new
    slot.finish = "length"
    slot.phase  = 2
    return 0
  end
  slot.next_id = id
  0
end

# One decode step for every busy slot: a generating slot feeds its last id,
# a slot still on its prompt feeds as much of it as the step has room for
# (kv.batch_max tokens in all). Tokens that need logits (the last prompt
# token, every generated one) go last in the step. Does nothing when no
# slot has a token to feed. A failure fails every request in the step.
def api_pump
  kv = STATE.kv
  # Non-output tokens (a_*) then output tokens (b_*).
  a_ids = [0];  a_ids.pop
  a_pos = [0];  a_pos.pop
  a_own = [0];  a_own.pop
  b_ids = [0];  b_ids.pop
  b_pos = [0];  b_pos.pop
  b_own = [0];  b_own.pop
  room = kv.batch_max
  si = 0
  while si < SLOTS.length
    slot = SLOTS[si]
    if slot.busy && slot.phase == 1 && room > 0
      b_ids.push(slot.next_id)
      b_pos.push(slot.seq_ids.length)
      b_own.push(si)
      room = room - 1
    end
    si = si + 1
  end
  si = 0
  while si < SLOTS.length
    slot = SLOTS[si]
    if slot.busy && slot.phase == 0 && room > 0
      take = slot.prompt_ids.length - slot.fed
      if take > room
        take = room
      end
      k = 0
      while k < take
        at = slot.fed + k
        if at == slot.prompt_ids.length - 1
          b_ids.push(slot.prompt_ids[at].to_i)
          b_pos.push(at)
          b_own.push(si)
        else
          a_ids.push(slot.prompt_ids[at].to_i)
          a_pos.push(at)
          a_own.push(si)
        end
        k = k + 1
      end
      room = room - take
    end
    si = si + 1
  end
  n_out = b_ids.length
  n_tok = a_ids.length + n_out
  if n_tok == 0
    return 0
  end
  STATE.step_seq = STATE.step_seq + 1
  ids = a_ids
  pos = a_pos
  own = a_own
  k = 0
  while k < n_out
    ids.push(b_ids[k])
    pos.push(b_pos[k])
    own.push(b_own[k])
    k = k + 1
  end
  begin
    # Enter the step's tokens in the pool and in their slots.
    cells  = [0]; cells.pop
    shared = [0]; shared.pop
    several = false
    k = 1
    while k < n_tok
      if own[k] != own[0]
        several = true
      end
      k = k + 1
    end
    k = 0
    while k < n_tok
      slot = SLOTS[own[k]]
      if several && slot.step_mark != STATE.step_seq
        slot.step_mark    = STATE.step_seq
        slot.shared_steps = slot.shared_steps + 1
      end
      c = api_cell_take(own[k], pos[k])
      slot.cells.push(c)
      slot.seq_ids.push(ids[k])
      if slot.phase == 0
        slot.fed = slot.fed + 1
      end
      cells.push(c)
      shared.push(slot.shared_n)
      k = k + 1
    end
    rows = n_out
    if rows == 0
      rows = 1   # a step needs an output: the last token's logits, unused
    end
    t_logits = TinyNN.tnn_null_ptr
    if kv.fused
      # A single token that sees every cell in use needs no mask.
      one = SLOTS[own[0]]
      use_mask = n_tok > 1 || one.cells.length + one.shared_n != STATE.cell_hi
      t_logits = SmolLM2KV.decode_cells(kv, ids, pos, cells, rows, STATE.cell_hi, use_mask,
                                        STATE.cell_owner, STATE.cell_pos, own, shared)
    else
      k = 0
      while k < n_tok
        if own[k] != 0 || cells[k] != pos[k]
          raise "per-head KV cache: cell " + cells[k].to_s + " is not position " + pos[k].to_s
        end
        t_logits = SmolLM2KV.decode_step_run(kv, ids[k], pos[k])
        k = k + 1
      end
    end
    if n_out > 0
      top = -1
      k = 0
      while k < n_out
        if SLOTS[b_own[k]].lp_top > top
          top = SLOTS[b_own[k]].lp_top
        end
        k = k + 1
      end
      best = [0];    best.pop
      lps  = [0.0];  lps.pop
      alt_ids = [0];   alt_ids.pop
      alt_lps = [0.0]; alt_lps.pop
      k = 0
      while k < n_out
        best.push(0)
        lps.push(0.0)
        j = 0
        while j < top
          alt_ids.push(-1)
          alt_lps.push(0.0)
          j = j + 1
        end
        k = k + 1
      end
      if alt_ids.length == 0
        alt_ids.push(-1)
        alt_lps.push(0.0)
      end
      rc = TinyNN.tnn_logits_pick(kv.sess, t_logits, n_out, STATE.cfg.vocab, top,
                                  best, lps, alt_ids, alt_lps)
      if rc != 0
        raise "logits pick failed rc=" + rc.to_s
      end
      k = 0
      while k < n_out
        slot = SLOTS[b_own[k]]
        if slot.lp_top >= 0
          slot.lp_tok.push(lps[k])
          j = 0
          while j < slot.lp_top
            slot.lp_alt_ids.push(alt_ids[k * top + j])
            slot.lp_alt_lps.push(alt_lps[k * top + j])
            j = j + 1
          end
        end
        api_slot_take(slot, best[k])
        k = k + 1
      end
    end
  rescue => e
    # Which cells the failed step wrote is unknown: every request in it
    # fails, and their slots forget what they had cached.
    puts "[openai_api_llama] decode step failed: " + e.message
    k = 0
    while k < n_tok
      slot = SLOTS[own[k]]
      slot.err   = e.message
      slot.phase = 2
      api_slot_truncate(slot, 0)
      k = k + 1
    end
  end
  n_tok
end

# Hands the processor to the other requests' fibers (the fiber server,
# SLOTS > 1). On the blocking server there is nobody to hand it to.
def api_yield
  if Tep::Scheduler.scheduled_context?
    Tep::Scheduler.pause(0)
  end
  0
end

# One turn of a request waiting on `slot`: when it has already seen the
# latest decode step it runs the next one (for every busy slot), otherwise
# another request's turn ran a step since and this one only catches up.
# With the scheduler resuming waiting requests in rotation that is one step
# per round, seen by each of them.
def api_slot_turn(slot)
  if slot.seen == STATE.step_seq
    api_pump
  end
  slot.seen = STATE.step_seq
  0
end

# Waits for an idle slot and returns its index (requests beyond SERVE_SLOTS
# queue here, keeping the busy slots decoding while they wait).
def api_slot_wait(prompt_ids, n_new)
  si = api_slot_acquire(prompt_ids, n_new)
  seen = STATE.step_seq
  while si < 0
    if seen == STATE.step_seq
      api_pump
    end
    seen = STATE.step_seq
    api_yield
    si = api_slot_acquire(prompt_ids, n_new)
  end
  si
end

# Runs the slot's request to its end.
def api_slot_run(slot)
  while !slot.done
    api_slot_turn(slot)
    if !slot.done
      api_yield
    end
  end
  0
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
