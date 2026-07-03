#!/bin/bash
# scripts/reproduce_phase0.sh -- Reproduce the phase-0 post-publication
# follow-up experiments documented in the phase-0 sections of RELEASE_NOTES.md.
#
# Headline phase-0 result: XGBoost routing with cost-sensitive weighting
# achieves All_acc = 0.9004, a +3.24pp improvement over LR-Routing (0.8681).
#
# Three sub-experiments:
#  * XGBoost routing (Exp 2.3+2.4):  ~5 min on RTX 3090
#  * DeBERTa NLI swap (Exp 4.1):     ~10 min (downloads DeBERTa-v3-large)
#  * Value-normalization features (Exp 2.1+2.2): ~10 min (needs SBERT)
#
# Requires the experimental extra: pip install xdomain-ser[experimental]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "==================================================================="
echo "  xdomain-ser :: reproduce_phase0.sh"
echo "  Reproduces: Phase-0 post-publication follow-ups"
echo "  Headline:   XGBoost-CW All_acc = 0.9004 (+3.24pp over LR-Routing)"
echo "==================================================================="
echo ""

# --- Environment ---
PY=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null || echo MISSING)
if [[ "$PY" != "3.11" ]]; then
  echo "ERROR: Python 3.11 required (found: $PY). Activate the gem-ser conda env."
  exit 1
fi

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: No GPU detected."
  exit 1
fi

if ! python -c "import xgboost" 2>/dev/null; then
  echo "ERROR: xgboost not installed. Run: pip install xdomain-ser[experimental]"
  exit 1
fi

EVAL_FILE="data/ranking_eval/negatives-v6-test.200.pe5.b10.rexp4.ckpt60.scores.json"
if [[ ! -f "$EVAL_FILE" ]]; then
  echo "ERROR: Missing $EVAL_FILE"
  exit 1
fi

OUT="evaluation/results/phase0"
mkdir -p "$OUT"

# --- 1. XGBoost routing ---
echo ""
echo "=== Phase-0 Exp 2.3+2.4: XGBoost routing + cost-sensitive weighting ==="
python -m xdomain_ser.routing.experimental.xgboost_routing \
  --eval_file "$EVAL_FILE" \
  --output_dir "$OUT/xgboost" \
  --seed 42

# --- 2. DeBERTa NLI swap (optional, slower) ---
if [[ "${PHASE0_DEBERTA:-0}" == "1" ]]; then
  echo ""
  echo "=== Phase-0 Exp 4.1: DeBERTa NLI swap ==="
  python -m xdomain_ser.nli.experimental.deberta \
    --eval_file "$EVAL_FILE" \
    --output_dir "$OUT/deberta" \
    --seed 42
else
  echo ""
  echo "[skip] DeBERTa NLI swap (set PHASE0_DEBERTA=1 to enable)"
fi

# --- 3. Value normalization (optional, requires sentence-transformers) ---
if [[ "${PHASE0_VALUENORM:-0}" == "1" ]]; then
  if python -c "import sentence_transformers" 2>/dev/null; then
    echo ""
    echo "=== Phase-0 Exp 2.1+2.2: Value normalization ==="
    python -m xdomain_ser.routing.experimental.value_normalization \
      --eval_file "$EVAL_FILE" \
      --output_dir "$OUT/value_norm" \
      --seed 42
  else
    echo ""
    echo "[skip] Value normalization (pip install sentence-transformers to enable)"
  fi
else
  echo ""
  echo "[skip] Value normalization (set PHASE0_VALUENORM=1 to enable)"
fi

# --- Summary: XGBoost headline ---
echo ""
echo "==================================================================="
HEADLINE=$(python -c "
import json
with open('$OUT/xgboost/best_config.json') as f:
    d = json.load(f)
print(d['test_metrics']['all_acc'])
")
TARGET=0.9004
TOL=0.01

DIFF=$(python -c "print(round(abs($HEADLINE - $TARGET), 4))")
WITHIN=$(python -c "print(abs($HEADLINE - $TARGET) <= $TOL)")

printf "  Reproduced XGBoost-CW All_acc: %s\n" "$HEADLINE"
printf "  Paper target:                  %s (+/- %s)\n" "$TARGET" "$TOL"
printf "  |reproduced - target|:         %s\n" "$DIFF"
echo "==================================================================="

if [[ "$WITHIN" == "True" ]]; then
  echo "PASS: phase-0 headline within tolerance."
  exit 0
else
  echo "FAIL: outside +/- $TOL tolerance."
  exit 1
fi
