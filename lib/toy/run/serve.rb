# lib/toy/run/serve.rb -- Spinel-compiled PERSISTENT serve runner.
#
# This is the lib-side home of `toy serve`'s HTTP server. Unlike the
# infer/train/eval runners (which compute once and exit), this runner
# is a PERSISTENT process: it boots the model, registers the OpenAI-
# compatible routes, then calls Tep.run! and blocks forever (until
# SIGINT/SIGTERM). The CRuby CLI shell (lib/toy/core/cli/serve.rb)
# EXECs this binary in the foreground rather than Open3-capturing it
# (capture would block the CLI forever).
#
# CONTROLLED ENV contract (read from ENV only -- `toy serve` sets these):
#   MODEL_PATH   path to the GGUF (default data/smollm2-135m-native.gguf)
#   MODEL_NAME   user-facing label (default = GGUF basename minus .gguf)
#   MAX_T        KV-cache context (default 256)
#   PORT         TCP port to bind (NEW -- replaces the old ARGV -p parse;
#                `toy serve` owns the CLI UX now). Default 4567.
#
# Tep is a BUILD-DEP TRANSPORT here: require "../../../vendor/spinel/deps"
# pulls in Tep via Spinel exactly as tep_demo/openai_api_llama.rb did.
# Nothing patches Tep; Toy does not make Tep consume Toy (that "Tep
# re-adaptation" is DEFERRED until Toy is stable, per user directive).
#
# Backend: CPU only, workers=1. Per-process FFI/KV STATE is not fork-safe
# to assume, and Metal serving was already noted unsupported; multi-worker
# prefork would change Tep's run_end aggregation -- out of scope. NOT in
# MIRRORABLE (prep/gen_cuda_mirror.rb): CPU-only, no CUDA mirror.

require_relative "../../toy_smollm2_ffi_kv"
require_relative "../io/toy_json"
require_relative "../models/toy_smollm2_loader"
require_relative "../../../vendor/spinel/deps"
require_relative "../serve/openai/api_json"
require_relative "../serve/openai/embeddings_handler"
require_relative "../serve/openai/server"
require_relative "../serve/openai/handlers"

# ---- Events sink (toy/v1 serving telemetry; FILE only). -------------------
# TOP-LEVEL constants (NEVER inside a branch — Spinel does not initialize a
# top-level CONSTANT assigned inside a conditional arm at runtime; it reads
# back empty, silently skipping all event writes; landmine, train.rb:82-85).
# TAO_RUN_DIR is set by `toy serve` (lib/toy/core/cli/serve.rb) when it has
# resolved a run id + created runs/<id>/. When empty, serving is events-OFF
# (cheap-when-off: every emit guard short-circuits), exactly like train.
TAO_RUN_DIR = ENV["TAO_RUN_DIR"] || ""
RUN_ID      = ENV["TOY_RUN_ID"] || ""
EVENTS      = TAO_RUN_DIR.length > 0 ? (TAO_RUN_DIR + "/events.jsonl") : ""

# git provenance read pure-Ruby from .git/HEAD (COPIED VERBATIM from
# train.rb:124-151; resolves ref: → 40-char sha; defaults "unknown").
git_sha    = "unknown"
git_branch = "unknown"
if File.exist?(".git/HEAD")
  head = File.read(".git/HEAD")
  if head.length > 0 && head[head.length - 1...head.length] == "\n"
    head = head[0...head.length - 1]
  end
  if head.length > 5 && head[0...5] == "ref: "
    ref_rel = head[5...head.length]
    pp = ref_rel.split("/")
    if pp.length >= 3
      git_branch = pp[pp.length - 1]
    end
    ref_path = ".git/" + ref_rel
    if File.exist?(ref_path)
      sha = File.read(ref_path)
      if sha.length >= 40
        git_sha = sha[0...40]
      end
    end
  else
    if head.length >= 40
      git_sha    = head[0...40]
      git_branch = "HEAD"
    end
  end
end

# PORT hoisted ABOVE the run_start emit so config.port is available. No side
# effects, so the hoist is safe (the Tep.run! call below still binds it).
SERVE_PORT = (ENV["PORT"] || "4567").to_i

# run_start at server boot — AFTER server.rb's require has loaded the model
# (STATE.ready==true, STATE.cfg/STATE.kv.sess populated) and BEFORE Tep.run!.
# Mirrors train.rb:153-188 EXACTLY (hand-built String concat, no #{}), with
# phase:"serve" (NOT "train") and config{max_t,port}.
if EVENTS.length > 0
  rc = TinyNN.tnn_events_open(EVENTS)
  if rc == 0
    rid = RUN_ID.length > 0 ? RUN_ID : "anonymous"
    rs = Toy::Json.new
    rs.j_str("kind", "run_start")
    rs.j_str("schema", "toy/v1")
    rs.j_num("t", TinyNN.tnn_events_now_seconds)
    rs.j_str("started_at", TinyNN.tnn_events_iso8601_now)
    rs.j_str("run_id", rid)
    rs.j_str("phase", "serve")
    host = Toy::Json.new
    host.j_str("name", TinyNN.tnn_provenance_host_name)
    host.j_str("os",   TinyNN.tnn_provenance_host_os)
    host.j_str("arch", TinyNN.tnn_provenance_host_arch)
    rs.j_obj("host", host)
    backend = Toy::Json.new
    backend.j_str("kind", TinyNN.tnn_backend_name(STATE.kv.sess))
    rs.j_obj("backend", backend)
    git = Toy::Json.new
    git.j_str("sha",    git_sha)
    git.j_str("branch", git_branch)
    rs.j_obj("git", git)
    model = Toy::Json.new
    model.j_str("arch", "llama")
    model.j_str("name", STATE.model_name)
    model.j_num("vocab",    STATE.cfg.vocab)
    model.j_num("d_model",  STATE.cfg.d_model)
    model.j_num("n_layers", STATE.cfg.n_layers)
    model.j_num("n_heads",  STATE.cfg.n_heads)
    model.j_num("n_kv",     STATE.cfg.n_kv)
    model.j_num("d_head",   STATE.cfg.head_dim)
    model.j_num("d_ff",     STATE.cfg.d_ff)
    rs.j_obj("model", model)
    config = Toy::Json.new
    config.j_num("max_t", MAX_T)
    config.j_num("port",  SERVE_PORT)
    rs.j_obj("config", config)
    TinyNN.tnn_events_emit(rs.j_dump)
  else
    puts "events_open failed: rc=" + rc.to_s + " (path=" + EVENTS + ")"
  end
end

# Routes -- Tep consumed purely as transport (Tep.get/post + Tep::Handler).
Tep.get  "/",                     IndexHandler.new
Tep.get  "/health",               HealthHandler.new
Tep.get  "/v1/models",            ModelsHandler.new
Tep.post "/v1/completions",       CompletionsHandler.new
Tep.post "/v1/chat/completions",  ChatCompletionsHandler.new
Tep.post "/v1/embeddings",        EmbeddingsHandler.new(STATE, MODEL_NAME)

# PORT from the controlled env (NEW; replaces the ARGV -p/-w/-q parse the
# tep_demo binary used -- `toy serve` owns the UX). CPU-only, workers=1,
# not quiet (the child streams its boot chatter so `toy serve` users see
# progress). Tep.run! blocks until SIGINT/SIGTERM. SERVE_PORT is hoisted
# above the run_start emit (its value feeds config.port).
Tep.run!(SERVE_PORT, 1, false)

# ============================================================================
# LOUD KNOWN GAP — run_end is NOT reliably emitted on shutdown.  (NOT masked.)
# ============================================================================
# The plan ASSUMED Tep.run!(port,1,false) returns cleanly on SIGTERM/SIGINT
# (workers<=1 branch: sp_net's sigaction handler sets a flag, sphttp_accept
# returns -1, worker_loop breaks, Server#run returns, Tep.run! returns here).
# That assumption was FALSIFIED on this build by direct measurement:
#
#   * On SIGTERM the runner exits with status 139 (SIGSEGV) — it CRASHES
#     during the Tep/Spinel shutdown/teardown path and NEVER returns to this
#     line.  Verified reproducible on the CLEAN main binary too (no toy-events
#     code involved), so this is a PRE-EXISTING Tep/Spinel shutdown segfault,
#     not introduced by the events wiring.
#   * Spinel implements NO Kernel#trap / at_exit (sp_runtime has no `trap`),
#     so a Ruby-level signal handler is NOT available as a fallback either.
#
# Consequence: run_start + the per-request eval/serve events ARE emitted
# reliably (and fsync'd line-by-line by the C emitter, so they survive the
# crash). run_end is the IRREDUCIBLE GAP until either (a) the Tep/Spinel
# shutdown segfault is fixed so Tep.run! returns and the block below runs, or
# (b) Spinel grows Kernel#trap. The block below is KEPT (correct + harmless):
# if Tep.run! ever returns cleanly it emits a well-formed run_end; today it is
# simply never reached. The guard makes a never-opened/already-closed session
# a cheap no-op (no double-run_end: Tep's own run_end rides APP.openai_events,
# which serve never configures, so it is disabled).
if EVENTS.length > 0 && TinyNN.tnn_events_active == 1
  re = Toy::Json.new
  re.j_str("kind", "run_end")
  re.j_num("t",         TinyNN.tnn_events_now_seconds)
  re.j_str("ended_at",  TinyNN.tnn_events_iso8601_now)
  re.j_str("reason",    "completed")
  re.j_raw("exit_code", "0")
  TinyNN.tnn_events_emit(re.j_dump)
  TinyNN.tnn_events_close
end
