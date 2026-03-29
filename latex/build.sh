#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MAIN_FILE="samplepaper"
TEX_ENGINE="xelatex"
TEX_FLAGS=(-shell-escape -interaction=nonstopmode -halt-on-error)

if ! command -v "$TEX_ENGINE" >/dev/null 2>&1; then
  echo "Error: $TEX_ENGINE was not found in PATH."
  exit 1
fi

echo "Quick build: running $TEX_ENGINE..."
"$TEX_ENGINE" "${TEX_FLAGS[@]}" "${MAIN_FILE}.tex"

echo "Build complete. Output: $SCRIPT_DIR/${MAIN_FILE}.pdf"
