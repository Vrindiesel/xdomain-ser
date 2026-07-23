#!/bin/bash
# scripts/download_models.sh -- ensure the LoRA extractor and ranker
# checkpoints are available locally at models/extractor and models/ranker.
#
# If both directories exist, exit 0 (no download needed).
# Otherwise, pull the missing one(s) from HuggingFace Hub via
# huggingface-cli into the expected location.
#
# Set DOWNLOAD_MODELS_FORCE=1 to re-download even if the directory exists.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EXTRACTOR_DIR="models/extractor"
RANKER_DIR="models/ranker"
EXTRACTOR_REPO="DavanHarrison/xdomain-ser-extractor"
RANKER_REPO="DavanHarrison/xdomain-ser-ranker"

# --- short-circuit if already present and not forcing ---
if [[ -z "${DOWNLOAD_MODELS_FORCE:-}" && -d "$EXTRACTOR_DIR" && -d "$RANKER_DIR" ]]; then
  echo "[download_models] OK: $EXTRACTOR_DIR and $RANKER_DIR present."
  echo "  (set DOWNLOAD_MODELS_FORCE=1 to re-download)"
  exit 0
fi

# --- pick the HF download command (prefer 'hf', fall back to legacy
#     'huggingface-cli' for older huggingface_hub installs).
if command -v hf >/dev/null 2>&1; then
  HF_DL="hf download"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DL="huggingface-cli download"
else
  cat <<EOF
[download_models] No HuggingFace CLI found on PATH ('hf' or 'huggingface-cli').

Install with:
  pip install huggingface_hub
(or run 'pip install -r requirements.txt' from the repo root, which
pulls it in via the core dependencies).
EOF
  exit 1
fi

mkdir -p models

# --- extractor ---
if [[ -n "${DOWNLOAD_MODELS_FORCE:-}" || ! -d "$EXTRACTOR_DIR" ]]; then
  echo "[download_models] fetching $EXTRACTOR_REPO -> $EXTRACTOR_DIR  ($HF_DL)"
  $HF_DL "$EXTRACTOR_REPO" --local-dir "$EXTRACTOR_DIR"
else
  echo "[download_models] $EXTRACTOR_DIR already present, skipping."
fi

# --- ranker ---
if [[ -n "${DOWNLOAD_MODELS_FORCE:-}" || ! -d "$RANKER_DIR" ]]; then
  echo "[download_models] fetching $RANKER_REPO -> $RANKER_DIR  ($HF_DL)"
  $HF_DL "$RANKER_REPO" --local-dir "$RANKER_DIR"
else
  echo "[download_models] $RANKER_DIR already present, skipping."
fi

echo "[download_models] done."
