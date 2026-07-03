#!/bin/bash
# scripts/reproduce_table7.sh -- Reproduce Table 7 of the GEM 2026 paper.
#
# Table 7 reports SER robustness on PERSONAGE-style stylistic-NLG outputs.
# Its "LLM Pers." and "Seq2Seq" columns are the per-source breakdown of the
# five-way routing comparison on the 250+250-example test split of the
# gold-annotated set (the same run as Appendix D); the "E2E (no style)"
# column comes from Table 6's E2E rows (run scripts/reproduce_table6.sh).
#
# NLI threshold is pinned to 0.5 to match the published run (see the note
# in scripts/reproduce_appendix_d.sh): Aligner / LoRA / NLI / ScoreRouting
# cells reproduce exactly; LR-Routing lands within ~1-7 pp of published
# due to NLI template-set drift (133 -> 138). See REPRODUCE.md.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "==================================================================="
echo "  xdomain-ser :: reproduce_table7.sh"
echo "  Reproduces: Table 7 -- robustness on PERSONAGE stylistic outputs"
echo "==================================================================="
echo ""

PY=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null || echo MISSING)
if [[ "$PY" != "3.11" ]]; then
  echo "ERROR: Python 3.11 required (found: $PY). Activate the gem-ser conda env."
  exit 1
fi

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: No GPU detected (needed for NLI inference)."
  exit 1
fi

GOLD="evaluation/gold/gold-annotated.json"
if [[ ! -f "$GOLD" ]]; then
  echo "ERROR: Missing gold-annotated data: $GOLD"
  exit 1
fi

OUT="evaluation/results/table7"
mkdir -p "$OUT"

echo "Running the five-way routing comparison (per-source = Table 7)..."
python -m xdomain_ser.routing.personality \
  --input_path "$GOLD" \
  --output_dir "$OUT" \
  --nli_threshold 0.5 \
  --seed 42

echo ""
echo "==================================================================="
echo "Table 7 (LLM/Seq2Seq columns): $OUT/per_source_comparison.tsv"
echo "E2E (no style) column:         evaluation/results/table6/table6.tsv"
echo "                               (run scripts/reproduce_table6.sh)"
echo "==================================================================="
