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
# their architecture, and report sizes. Drop the toy binaries on
# the same box and they just work.
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

echo "[fetch_model] done. Verify with ./examples/example_list_models"
