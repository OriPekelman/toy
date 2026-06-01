# lib/toy/serve/openai/server.rb -- the Server state + boot + greedy
# decode for `toy serve` (OpenAI-compatible, llama-family GGUF).
#
# MOVED from tep_demo/openai_api_llama.rb:136-264 (P4 toy serve). The
# endpoint logic that USED to live in the tep_demo binary now lives in
# lib/toy/serve/openai/; the tep_demo source STAYS until the later
# cleanup pass retires it (serve must be gated first).
#
# Tep is consumed purely as transport (Tep::Handler / Tep::Json /
# Tep.get/post/Tep.run!). Nothing here patches Tep -- this is build-dep
# usage, not "Tep re-adaptation" (that = Tep consuming Toy, DEFERRED per
# user directive until Toy is stable).
#
# CONTROLLED ENV contract (set by `toy serve` via lib/toy/run/serve.rb):
#   MODEL_PATH   path to any llama-family GGUF (default
#                data/smollm2-135m-native.gguf)
#   MODEL_NAME   user-facing label in /v1/models + completion responses.
#                Defaults to GGUF basename minus .gguf.
#   MAX_T        max context for the KV cache (default 256)
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
MAX_T      = (ENV["MAX_T"] || "256").to_i

# ---- Inference state. Class instance held as a CONSTANT so spinel
#      emits a typed slot for it (same pattern as inference_api.rb /
#      openai_api.rb). ----

class State
  # :req_seq is a monomorphic Integer slot (always Integer) for the
  # per-request monotonic counter the serving events emitter uses as a
  # deterministic-shaped request_id. Held on the already-CONSTANT-held STATE
  # so Spinel emits a typed slot; NO new polymorphic accessor (landmine #16).
  attr_accessor :cfg, :kv, :gguf, :model_name, :ready, :req_seq
  def initialize
    @cfg  = nil
    @kv   = nil
    @gguf = TinyNN.tnn_null_ptr
    @model_name = MODEL_NAME
    @ready      = false
    @req_seq    = 0
  end
end
STATE = State.new
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
