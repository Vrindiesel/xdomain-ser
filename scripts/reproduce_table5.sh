#!/bin/bash
# scripts/reproduce_table5.sh -- Reproduce Table 5 of the GEM 2026 paper.
#
# Headline result: LR-Routing All_acc = 0.868 on the held-out test split
# of the rexp4.ckpt60.scores.json evaluation set.
#
# By default the shipped ranker scores in the eval file are used (cached
# model predictions from the published run); NLI inference, the dev/test
# split, the threshold sweep, and the LR fit always run live. Pass
# --rescore to also re-run the LoRA ranker over the cached beam
# candidates (per-text ref=true protocol, fp32 compute, ~75 min on
# RTX 3090) instead of using the shipped scores. See REPRODUCE.md,
# "What each script recomputes".

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RESCORE=0
for arg in "$@"; do
  case "$arg" in
    --rescore) RESCORE=1 ;;
    *) echo "Unknown option: $arg (supported: --rescore)"; exit 1 ;;
  esac
done

echo "==================================================================="
echo "  xdomain-ser :: reproduce_table5.sh"
echo "  Reproduces: Table 5 -- SER agreement with per-example routing"
echo "  Headline:   LR-Routing All_acc = 0.868"
echo "==================================================================="
echo ""

# --- Environment validation ---
PY=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null || echo MISSING)
if [[ "$PY" != "3.11" ]]; then
  echo "ERROR: Python 3.11 required (found: $PY). Activate the gem-ser conda env."
  exit 1
fi

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: No GPU detected. This script needs a CUDA GPU for NLI inference."
  exit 1
fi

# --- Required data ---
EVAL_FILE="data/ranking_eval/negatives-v6-test.200.pe5.b10.rexp4.ckpt60.scores.json"
if [[ ! -f "$EVAL_FILE" ]]; then
  echo "ERROR: Missing eval data: $EVAL_FILE"
  echo "Run the data migration step or provide --eval_file."
  exit 1
fi

# --- Output dir ---
OUT="evaluation/results/table5"
mkdir -p "$OUT"

# --- Optional live re-scoring (--rescore) ---
# Re-runs the LoRA ranker over the cached beam candidates with the
# published per-text ref=true protocol, writing fresh pred_scores in
# place of the shipped ones. Compute dtype is pinned to float32 (the
# published runs' dtype) because candidate scores sit on near-ties:
# 53% of texts have a top-2 score gap < 0.02, and bf16 scoring flips
# ~32% of top-1 selections, moving the headline several pp. The
# +/-0.01 gate below applies to the fp32 path.
if [[ "$RESCORE" == "1" ]]; then
  bash "$SCRIPT_DIR/download_models.sh"
  RESCORED="$OUT/rescored.scores.json"
  echo "Re-scoring 2,288 x 10 candidates with the live ranker (fp32, ~75 min on RTX 3090)..."
  python -m xdomain_ser.ranking.score \
    --checkpoint_dir models/ranker \
    --hint_map_path data/multi_ser_v9/hint_maps_v4.json \
    --eval_path "$EVAL_FILE" \
    --ref_source true \
    --compute_dtype float32 \
    --output_path "$RESCORED"
  EVAL_FILE="$RESCORED"
fi

# --- Run routing pipeline ---
echo "Running routing selector (~3-5 min on RTX 3090)..."
python -m xdomain_ser.routing.selector \
  --eval_file "$EVAL_FILE" \
  --output_dir "$OUT" \
  --seed 42 \
  --batch_size 32

# --- Extract headline number ---
echo ""
echo "==================================================================="
HEADLINE=$(python -c "
import json
with open('$OUT/routing_details.json') as f:
    d = json.load(f)
print(d['summary']['LR-Routing']['all_acc'])
")
TARGET=0.868
TOL=0.01

DIFF=$(python -c "print(round(abs($HEADLINE - $TARGET), 4))")
WITHIN=$(python -c "print(abs($HEADLINE - $TARGET) <= $TOL)")

printf "  Reproduced LR-Routing All_acc: %s\n" "$HEADLINE"
printf "  Paper target:                  %s (+/- %s)\n" "$TARGET" "$TOL"
printf "  |reproduced - target|:         %s\n" "$DIFF"
echo "==================================================================="

if [[ "$WITHIN" == "True" ]]; then
  echo "PASS: within tolerance."
  exit 0
else
  echo "FAIL: outside +/- $TOL tolerance."
  exit 1
fi
