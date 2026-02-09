#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to build the LaTeX paper with latexmk
TEXFILE="paper.latex"

if ! command -v latexmk >/dev/null 2>&1; then
  echo "Error: latexmk is not installed."
  echo "On Debian/Ubuntu: sudo apt update && sudo apt install -y latexmk texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended"
  exit 2
fi

echo "Building ${TEXFILE}..."
latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode -file-line-error" -use-make "${TEXFILE}"
echo "Build finished."
