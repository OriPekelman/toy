#!/usr/bin/env bash
# Fetch a GGUF model from HuggingFace into a local cache that
# model_index will pick up. Optional helper for first-time users
# who don't have a model already in their HF / Ollama / LM Studio
# caches.
#
# Usage:
#   prep/fetch_model.sh <hf-repo> [<file.gguf>]
#
# Examples:
#   prep/fetch_model.sh bartowski/SmolLM2-135M-Instruct-GGUF SmolLM2-135M-Instruct-Q8_0.gguf
#   prep/fetch_model.sh Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q8_0.gguf
#
# Files land in ~/.cache/huggingface/hub/ (the standard HF cache).
# `./examples/example_list_models` will then see them, identify
# their architecture, and report sizes.
#
# After a successful fetch we also drop a relative symlink at
# `data/<file>.gguf` pointing into the HF cache. That makes the
# example default GGUF= path resolve without the user needing to
# remember the full HF cache path. Idempotent: skipped if the
# symlink already points at the right blob.
#
# Defers to `huggingface-cli` if available; falls back to a direct
# curl through https://huggingface.co/<repo>/resolve/main/<file>.
# Both paths populate the same HF cache layout.

set -euo pipefail

REPO="${1:?usage: prep/fetch_model.sh <hf-repo> [<file.gguf>]}"
FILE="${2:-}"

# If file isn't passed, try to list and pick the largest q8_0/native variant.
if [ -z "$FILE" ]; then
  echo "[fetch_model] no file specified; you can pass one as arg 2"
  echo "[fetch_model] common picks per repo (instruction-tuned, q8_0):"
  echo "    bartowski/SmolLM2-135M-Instruct-GGUF  SmolLM2-135M-Instruct-Q8_0.gguf"
  echo "    bartowski/Llama-3.2-1B-Instruct-GGUF  Llama-3.2-1B-Instruct-Q8_0.gguf"
  echo "    Qwen/Qwen2.5-1.5B-Instruct-GGUF       qwen2.5-1.5b-instruct-q8_0.gguf"
  exit 1
fi

if command -v huggingface-cli >/dev/null 2>&1; then
  echo "[fetch_model] huggingface-cli download $REPO $FILE"
  huggingface-cli download "$REPO" "$FILE"
else
  TARGET="${HOME}/.cache/huggingface/hub/models--${REPO//\//--}/snapshots/manual"
  mkdir -p "$TARGET"
  URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"
  echo "[fetch_model] curl -L $URL -> $TARGET/$FILE"
  curl -fL -o "$TARGET/$FILE" "$URL"
fi

# Resolve the cached blob path and drop a relative symlink into data/.
# The HF cache layout is:
#   ~/.cache/huggingface/hub/models--<repo--with--dashes>/
#     snapshots/<sha>/<FILE>   ->   ../../blobs/<hash>
# We point data/<FILE> at the snapshot path (not the blob hash) so the
# filename in `./examples/example_list_models` stays meaningful.
HF_ROOT="${HOME}/.cache/huggingface/hub"
REPO_DIR="${HF_ROOT}/models--${REPO//\//--}"
SNAPSHOT=""
if [ -d "${REPO_DIR}/snapshots" ]; then
  # Pick the snapshot dir that actually contains FILE (most recently
  # touched first). huggingface-cli sometimes creates multiple snapshot
  # dirs across refs; we want the one with our file.
  for snap in "${REPO_DIR}/snapshots/"*/; do
    [ -e "${snap}${FILE}" ] || continue
    SNAPSHOT="${snap}${FILE}"
    break
  done
fi

# Fall back to the curl-target path if the HF-cache snapshot lookup
# didn't find anything (curl branch above writes there directly).
if [ -z "$SNAPSHOT" ] && [ -e "${REPO_DIR}/snapshots/manual/${FILE}" ]; then
  SNAPSHOT="${REPO_DIR}/snapshots/manual/${FILE}"
fi

if [ -n "$SNAPSHOT" ] && [ -e "$SNAPSHOT" ]; then
  mkdir -p data
  LINK="data/${FILE}"
  # Resolve to an absolute path so the symlink survives data/ being
  # entered from anywhere (and rsync/copy won't accidentally inline it).
  ABS=$(cd "$(dirname "$SNAPSHOT")" && pwd)/$(basename "$SNAPSHOT")
  if [ -L "$LINK" ] && [ "$(readlink "$LINK")" = "$ABS" ]; then
    echo "[fetch_model] data/${FILE} already linked"
  else
    rm -f "$LINK"
    ln -s "$ABS" "$LINK"
    echo "[fetch_model] linked data/${FILE} -> $ABS"
  fi
else
  echo "[fetch_model] (no data/ symlink — couldn't resolve snapshot path)"
fi

echo "[fetch_model] done. Verify with ./examples/example_list_models"
