#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MAIN_FILE="samplepaper"
TEX_ENGINE="xelatex"
TEX_FLAGS=(-shell-escape -interaction=nonstopmode -halt-on-error)

# Check for xelatex availability
if ! command -v "$TEX_ENGINE" >/dev/null 2>&1; then
  echo "Error: $TEX_ENGINE was not found in PATH."
  exit 1
fi

echo "--- Starting Full LaTeX Build Cycle ---"

echo "[Pass 1] Running $TEX_ENGINE..."
"$TEX_ENGINE" "${TEX_FLAGS[@]}" "${MAIN_FILE}.tex"

echo "Running bibtex..."
if command -v bibtex >/dev/null 2>&1; then
  bibtex "$MAIN_FILE" || echo "Warning: BibTeX reported errors."
else
  echo "Warning: bibtex not found, citations might not be updated."
fi

echo "[Pass 2] Running $TEX_ENGINE..."
"$TEX_ENGINE" "${TEX_FLAGS[@]}" "${MAIN_FILE}.tex"

echo "[Pass 3] Finalizing with $TEX_ENGINE..."
"$TEX_ENGINE" "${TEX_FLAGS[@]}" "${MAIN_FILE}.tex"

echo "--- Build Complete ---"
echo "Output: $SCRIPT_DIR/${MAIN_FILE}.pdf"

# xelatex -shell-escape -interaction=nonstopmode samplepaper.tex


# giai thich table, thêm số vào cho colume 2 , train model from scratch,  add model metrics from scatch vao csv and ve lai hinh  