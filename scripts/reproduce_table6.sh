#!/bin/bash
# scripts/reproduce_table6.sh -- Reproduce Table 6 of the GEM 2026 paper.
#
# Table 6 compares the learned methods (from Table 5's per-topic results)
# against the three published rule-based SER tools (E2E script, RNNLG,
# ViGGO) on the six Eval-2 topics where rule tools exist. CPU-only, ~2 min.
#
# Prerequisite: scripts/reproduce_table5.sh (provides the learned per-topic
# rows). See the protocol notes in scripts/make_table6.py and ERRATA.md.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "==================================================================="
echo "  xdomain-ser :: reproduce_table6.sh"
echo "  Reproduces: Table 6 -- vs rule-based SER tools (six topics)"
echo "==================================================================="
echo ""

PY=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null || echo MISSING)
if [[ "$PY" != "3.11" ]]; then
  echo "ERROR: Python 3.11 required (found: $PY). Activate the gem-ser conda env."
  exit 1
fi

SCORES="data/ranking_eval/negatives-v6-test.200.pe5.b10.rexp4.ckpt60.scores.json"
TABLE5="evaluation/results/table5/per_topic_comparison.tsv"
if [[ ! -f "$SCORES" ]]; then
  echo "ERROR: Missing scored eval file: $SCORES"
  exit 1
fi
if [[ ! -f "$TABLE5" ]]; then
  echo "ERROR: Missing $TABLE5"
  echo "Run scripts/reproduce_table5.sh first (provides the learned rows)."
  exit 1
fi

python scripts/make_table6.py \
  --scores_file "$SCORES" \
  --table5_dir "evaluation/results/table5" \
  --output_dir "evaluation/results/table6"

echo ""
echo "==================================================================="
echo "Table written to: evaluation/results/table6/table6.tsv"
echo "Summary + paper deltas: evaluation/results/table6/table6.md"
echo "==================================================================="
