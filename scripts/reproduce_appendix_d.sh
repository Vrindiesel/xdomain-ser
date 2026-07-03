#!/bin/bash
# scripts/reproduce_appendix_d.sh -- Reproduce Appendix D of the GEM 2026 paper.
#
# Appendix D reports the five-way comparison on the 1000-example
# gold-annotated PERSONAGE set:
#   1. E2E Aligner (rule-based)
#   2. LoRA MySER (extraction + ranking)
#   3. NLI Baseline (RoBERTa-MNLI)
#   4. Score-threshold routing (LoRA/NLI by ranker confidence)
#   5. LR routing (logistic regression on per-example features)
#
# NLI threshold is pinned to 0.5 to match the published run, which used
# precomputed NLI results from the gold three-way comparison (threshold
# 0.5), not personality.py's live-inference default of 0.3. With today's
# 138-template NLI set (the paper's runs used 133), Aligner / LoRA / NLI /
# ScoreRouting rows reproduce exactly; LR-Routing refits on shifted NLI
# probability features and lands within ~1-7 pp of published. See
# REPRODUCE.md for details.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "==================================================================="
echo "  xdomain-ser :: reproduce_appendix_d.sh"
echo "  Reproduces: Appendix D -- 5-way comparison on PERSONAGE gold"
echo "==================================================================="
echo ""

PY=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null || echo MISSING)
if [[ "$PY" != "3.11" ]]; then
  echo "ERROR: Python 3.11 required."
  exit 1
fi

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: No GPU detected."
  exit 1
fi

GOLD="evaluation/gold/gold-annotated.json"
if [[ ! -f "$GOLD" ]]; then
  echo "ERROR: Missing gold-annotated data: $GOLD"
  echo "Stage 10 of the release migration adds the Eval-2 gold data."
  exit 1
fi

OUT="evaluation/results/appendix_d"
mkdir -p "$OUT"

echo "Running 5-way routing comparison on PERSONAGE gold..."
python -m xdomain_ser.routing.personality \
  --input_path "$GOLD" \
  --output_dir "$OUT" \
  --nli_threshold 0.5 \
  --seed 42

echo ""
echo "==================================================================="
echo "5-way comparison tables written to: $OUT/"
echo "==================================================================="
