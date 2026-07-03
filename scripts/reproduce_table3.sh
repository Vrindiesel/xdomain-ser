#!/bin/bash
# scripts/reproduce_table3.sh -- Reproduce Table 3 of the GEM 2026 paper.
#
# Table 3 compares PBL (GPT-4o, 5-shot) extraction vs LoRA-fine-tuned
# Llama-3.2-3B extraction on the multi-domain test set. Run is two-phase:
#
#   (a) GPT-4o PBL extraction over the test set (uses OpenAI API)
#   (b) LoRA extraction with the released ``extractor`` checkpoint
#
# Then both are scored with ``xdomain_ser.extraction.eval`` and the
# per-topic + overall numbers compared.
#
# Requires:
#   - The PBL optional install:  pip install xdomain-ser[pbl]
#   - An OPENAI_API_KEY env var (or --openai_conf_path JSON)
#   - The multi-domain test data at data/multi_ser_v9/test.json
#     (migrated in Stage 10; flagged in RELEASE_NOTES)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "==================================================================="
echo "  xdomain-ser :: reproduce_table3.sh"
echo "  Reproduces: Table 3 -- PBL vs LoRA extraction"
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

if ! python -c "import openai" 2>/dev/null; then
  echo "ERROR: openai not installed. Run: pip install xdomain-ser[pbl]"
  exit 1
fi

if [[ -z "${OPENAI_API_KEY:-}" && -z "${1:-}" ]]; then
  echo "ERROR: Set OPENAI_API_KEY or pass --openai_conf_path as the first arg."
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

# --- Models ---
bash "$SCRIPT_DIR/download_models.sh"

OUT="evaluation/results/table3"
mkdir -p "$OUT"

# --- (a) PBL extraction via GPT-4o ---
echo ""
echo "=== PBL (GPT-4o) extraction ==="
python -m xdomain_ser.extraction.pbl \
  --input_path "$TEST_FILE" \
  --output_path "$OUT/pbl_predictions.json" \
  --prompt_examples_path "$PROMPT_EXAMPLES" \
  --hint_map_path "$HINTMAP" \
  --num_beams 1 --model "gpt-4o" --seed 232342

# --- (b) LoRA extraction ---
echo ""
echo "=== LoRA extraction ==="
python -m xdomain_ser.extraction.inference \
  --checkpoint_dir models/extractor \
  --eval_path "$TEST_FILE" \
  --prompt_examples_path "$PROMPT_EXAMPLES" \
  --hint_map_path "$HINTMAP" \
  --num_beams 1 --max_new_tokens 80 --batch_size 16 \
  --output_path "$OUT/lora_predictions.json"

# --- Score both ---
echo ""
echo "=== SER + slot-F1 ==="
echo "-- PBL --"
python -m xdomain_ser.extraction.eval "$OUT/pbl_predictions.json" \
  --save_path "$OUT/pbl_scores.json"
echo "-- LoRA --"
python -m xdomain_ser.extraction.eval "$OUT/lora_predictions.json" \
  --save_path "$OUT/lora_scores.json"

echo ""
echo "==================================================================="
echo "Per-topic and overall comparison written to: $OUT/"
echo "Inspect $OUT/{pbl,lora}_scores.json for the headline numbers."
echo "==================================================================="
