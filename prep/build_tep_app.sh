#!/usr/bin/env bash
# Build a Tep-Spinel HTTP app that needs to pull in project lib/*
# files. `bin/tep build` only inlines tep's own library; external
# `require_relative` lines get silently dropped. We solve that by
# concatenating the lib files (in dependency order, with require
# lines stripped) into a single source file and feeding *that* to
# `bin/tep build`.
#
# Usage:
#   prep/build_tep_app.sh tep_demo/openai_api.rb -o tep_demo/openai_api
#
# Any flags after the first arg are passed through to `bin/tep build`.

set -euo pipefail

APP_SRC="$1"
shift
APP_DIR="$(dirname "$APP_SRC")"
BASE="$(basename "$APP_SRC" .rb)"
COMBINED="${APP_DIR}/.${BASE}.combined.rb"

# Locate Tep checkout — same fallback that bin/tep uses.
TEP_BIN="${TEP_BIN:-${HOME}/sites/tep/bin/tep}"
if [ ! -x "$TEP_BIN" ]; then
  echo "tep CLI not found at $TEP_BIN" >&2
  echo "set TEP_BIN=/path/to/tep/bin/tep" >&2
  exit 2
fi

# Order matters (dependency chain): tinynn → transformer → gpt2 →
# {gpt2_ffi_kv, gguf_load, bpe} → app source. The require_relative
# lines we strip have no effect at runtime once everything's inlined.
LIBS=(
  lib/toy/ffi/tinynn.rb
  lib/transformer.rb
  lib/gpt2.rb
  lib/gpt2_ffi_kv.rb
  lib/gguf_load.rb
  lib/bpe.rb
)

{
  echo "# Combined from ${LIBS[*]} + ${APP_SRC} for tep build"
  for f in "${LIBS[@]}"; do
    echo "# --- inlined: $f ---"
    grep -v '^require_relative' "$f"
    echo ""
  done
  echo "# --- inlined app: $APP_SRC ---"
  grep -v '^require_relative' "$APP_SRC"
} > "$COMBINED"

echo "[build_tep_app] wrote $COMBINED ($(wc -l < "$COMBINED" | tr -d ' ') lines)"
echo "[build_tep_app] invoking $TEP_BIN build"
exec "$TEP_BIN" build "$COMBINED" "$@"
