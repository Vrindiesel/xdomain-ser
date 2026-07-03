#!/bin/bash
# scripts/reproduce_appendix_gold_comparison.sh -- PERSONAGE-gold three-way
# SER comparison (E2E aligner vs LoRA vs NLI) against human-annotated gold
# SER labels. Backs the gold-set analyses in the paper (Table 7 data and
# the appendix breakdowns). Formerly mis-titled reproduce_table6.sh; paper
# Table 6 is reproduced by scripts/reproduce_table6.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "==================================================================="
echo "  xdomain-ser :: reproduce_appendix_gold_comparison.sh"
echo "  Three-way SER comparison on the 1,000-example gold set"
echo "==================================================================="
echo ""

PY=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null || echo MISSING)
if [[ "$PY" != "3.11" ]]; then
  echo "ERROR: Python 3.11 required (found: $PY). Activate the gem-ser conda env."
  exit 1
fi

GOLD="evaluation/gold/gold-annotated.json"
if [[ ! -f "$GOLD" ]]; then
  echo "ERROR: Missing gold-annotated data: $GOLD"
  exit 1
fi

OUT="evaluation/results/appendix_gold_comparison"
mkdir -p "$OUT"

echo "Running the three-way SER comparison (E2E aligner vs LoRA vs NLI)..."
python -m xdomain_ser.eval2.compare \
  --input_path "$GOLD" \
  --output_path "$OUT/comparison.json"

echo ""
echo "Generating breakdown tables..."
python -m xdomain_ser.eval2.tables \
  --input_path "$OUT/comparison.json" \
  --gold_path "$GOLD" \
  --output_dir "$OUT/tables"

echo ""
echo "==================================================================="
echo "Comparison: $OUT/comparison.json"
echo "Tables:     $OUT/tables/method-comparison-*.tsv"
echo "==================================================================="
