#!/bin/bash
# scripts/reproduce_table4.sh -- Reproduce Table 4 of the GEM 2026 paper.
#
# Table 4 reports the over-generate-and-rank (OGR) effect: LoRA extraction
# with single greedy decoding vs LoRA + beam-10 + ranker scoring.
#
# Pipeline:
#   (a) Greedy single-beam LoRA inference on the test set
#   (b) Beam-10 LoRA inference + ranker scoring
#   (c) SER eval of greedy, top-of-beam, and ranker-selected predictions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "==================================================================="
echo "  xdomain-ser :: reproduce_table4.sh"
echo "  Reproduces: Table 4 -- OGR effect on extraction quality"
echo "==================================================================="
echo ""

# --- Environment ---
PY=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null || echo MISSING)
if [[ "$PY" != "3.11" ]]; then
  echo "ERROR: Python 3.11 required."
  exit 1
fi

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: No GPU detected."
  exit 1
fi

# --- Required data ---
TEST_FILE="data/multi_ser_v9/test.json"
HINTMAP="data/multi_ser_v9/hint_maps_v4.json"
PROMPT_EXAMPLES="data/multi_ser_v9/prompt-examples-dev-repr.json"

for f in "$TEST_FILE" "$HINTMAP" "$PROMPT_EXAMPLES"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: Missing data file: $f"
    echo "Stage 10 of the release migration adds the full multi-domain test data."
    exit 1
  fi
done

bash "$SCRIPT_DIR/download_models.sh"

OUT="evaluation/results/table4"
mkdir -p "$OUT"

# --- (a) Greedy ---
echo ""
echo "=== Greedy single-beam LoRA inference (~30-60 min on RTX 3090) ==="
python -m xdomain_ser.extraction.inference \
  --checkpoint_dir models/extractor \
  --eval_path "$TEST_FILE" \
  --prompt_examples_path "$PROMPT_EXAMPLES" \
  --hint_map_path "$HINTMAP" \
  --num_beams 1 --max_new_tokens 80 --batch_size 16 \
  --output_path "$OUT/greedy_predictions.json"

# --- (b) Beam-10 + ranker ---
echo ""
echo "=== Beam-10 LoRA inference (~60-90 min on RTX 3090) ==="
python -m xdomain_ser.extraction.inference \
  --checkpoint_dir models/extractor \
  --eval_path "$TEST_FILE" \
  --prompt_examples_path "$PROMPT_EXAMPLES" \
  --hint_map_path "$HINTMAP" \
  --num_beams 10 --max_new_tokens 80 --batch_size 4 \
  --output_path "$OUT/beam10_predictions.json"

echo ""
echo "=== Ranker scoring (fp32, ~9 h on RTX 3090 -- the long step) ==="
# fp32 is pinned for the same reason as reproduce_table5.sh --rescore:
# candidate scores sit on near-ties, and bf16 flips ~32% of top-1
# selections, which moves the ranker-selected SER rows away from the
# published values. See REPRODUCE.md, "What each script recomputes".
python -m xdomain_ser.ranking.score \
  --checkpoint_dir models/ranker \
  --eval_path "$OUT/beam10_predictions.json" \
  --hint_map_path "$HINTMAP" \
  --compute_dtype float32 \
  --output_path "$OUT/beam10_scored.json"

# --- (c) Eval ---
echo ""
echo "=== Greedy SER ==="
python -m xdomain_ser.extraction.eval "$OUT/greedy_predictions.json" \
  --save_path "$OUT/greedy_scores.json"

echo ""
echo "=== Beam-10 top-of-beam (no re-ranking) SER ==="
python -m xdomain_ser.extraction.eval "$OUT/beam10_scored.json" \
  --save_path "$OUT/beam10_topbeam_scores.json"

echo ""
echo "=== Beam-10 with ranker re-ranking SER ==="
python -m xdomain_ser.extraction.eval "$OUT/beam10_scored.json" \
  --rank_method scores \
  --save_path "$OUT/beam10_ranked_scores.json"

echo ""
echo "==================================================================="
echo "Results written to: $OUT/"
echo "Compare {greedy,beam10_topbeam,beam10_ranked}_scores.json for OGR effect."
echo "==================================================================="
