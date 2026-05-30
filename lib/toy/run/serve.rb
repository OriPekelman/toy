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
require_relative "../../toy_smollm2_loader"
require_relative "../../../vendor/spinel/deps"
require_relative "../serve/openai/api_json"
require_relative "../serve/openai/embeddings_handler"
require_relative "../serve/openai/server"
require_relative "../serve/openai/handlers"

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
# progress). Tep.run! blocks until SIGINT/SIGTERM.
SERVE_PORT = (ENV["PORT"] || "4567").to_i
Tep.run!(SERVE_PORT, 1, false)
