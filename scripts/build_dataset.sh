#!/bin/bash
# scripts/build_dataset.sh -- Reconstruct the multi-domain SER dataset (data/multi_ser_v9/).
#
# Rebuilds the training/eval data used in the GEM 2026 paper from the raw public
# corpora, so others can reproduce the exact data examples. Faithful single-shot
# version of the original `preprocess-ser-datasets.sh` execution log.
#
# Pipeline (see xdomain_ser/data_prep/):
#   1. preprocess        raw E2E/ViGGO/RNNLG/Taskmaster  -> unified per-domain JSON
#   2. make_ser_dataset  per-domain JSON                 -> text->MR "ds" + per-topic hint maps
#   3. merge_select      all domains (via files.json)    -> train-200/dev-50/test + example pools
# (dev-repr-120.json is a shipped phase-0 artifact, not rebuilt here -- see Step 3 note.)
#
# PREREQUISITES
#   * The optional dataprep deps:  pip install -r requirements-dataprep.txt
#   * The four raw datasets downloaded into datasets/ (NOT shipped -- see REPRODUCE.md):
#       datasets/e2e-dataset/   datasets/viggo-v1/   datasets/RNNLG/   datasets/Taskmaster/
#   * The per-domain input hint maps shipped in this repo under data/<domain>/hint-map.json.
#
# NOTE: the released v9 data was built with --selection_method random (seed 2323).
# Switch to fracloc below for Facility-Location example selection (needs submodlib).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SELECTION_METHOD="${SELECTION_METHOD:-random}"   # random (released) | fracloc
SEED="${SEED:-2323}"
TRAIN_K="${TRAIN_K:-200}"
DEV_K="${DEV_K:-50}"
DATASETS="${DATASETS:-datasets}"

echo "==================================================================="
echo "  xdomain-ser :: build_dataset.sh"
echo "  Reconstructs: data/multi_ser_v9/ from raw E2E/ViGGO/RNNLG/Taskmaster"
echo "  Selection:    $SELECTION_METHOD   (seed=$SEED, train_k=$TRAIN_K, dev_k=$DEV_K)"
echo "==================================================================="
echo ""

# --- Environment validation ---
PY=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>/dev/null || echo MISSING)
if [[ "$PY" != "3.11" ]]; then
  echo "ERROR: Python 3.11 required (found: $PY). Activate the gem-ser conda env."
  exit 1
fi
if [[ "$SELECTION_METHOD" == "fracloc" ]]; then
  python -c "import submodlib, sentence_transformers" 2>/dev/null || {
    echo "ERROR: --selection_method fracloc needs the dataprep deps: pip install -r requirements-dataprep.txt"
    exit 1; }
fi

# --- Raw dataset presence check ---
for d in e2e-dataset viggo-v1 RNNLG Taskmaster; do
  if [[ ! -d "$DATASETS/$d" ]]; then
    echo "ERROR: missing raw dataset $DATASETS/$d -- see REPRODUCE.md for download instructions."
    exit 1
  fi
done

PP="python -m xdomain_ser.data_prep.preprocess"
MK="python -m xdomain_ser.data_prep.make_ser_dataset"

echo "### Step 1/4: preprocess raw corpora -> unified per-domain JSON ###"

# ViGGO -> data/viggo-v1/v4
for split in train valid test; do
  $PP --input_path "$DATASETS/viggo-v1/viggo-${split}.csv" --output_path data/viggo-v1/v4 --dataset viggo
done

# E2E -> data/e2e-nlg/v1
$PP --input_path "$DATASETS/e2e-dataset/trainset.csv"       --output_path data/e2e-nlg/v1 --dataset e2e_nlg
$PP --input_path "$DATASETS/e2e-dataset/devset.csv"         --output_path data/e2e-nlg/v1 --dataset e2e_nlg
$PP --input_path "$DATASETS/e2e-dataset/testset_w_refs.csv" --output_path data/e2e-nlg/v1 --dataset e2e_nlg

# RNNLG (hotel/laptop/restaurant/tv) -> data/rnnlg/<domain>/v2
for dom in hotel laptop restaurant tv; do
  for split in train test valid; do
    $PP --input_path "$DATASETS/RNNLG/data/original/${dom}/${split}.json" \
        --output_path "data/rnnlg/${dom}/v2" --dataset rnnlg
  done
done

# Taskmaster TM-1 -> data/taskmaster/TM-1-2019/v3
for f in self-dialogs woz-dialogs; do
  $PP --input_path "$DATASETS/Taskmaster/TM-1-2019/${f}.json" \
      --output_path data/taskmaster/TM-1-2019/v3 \
      --partition_file "$DATASETS/Taskmaster/TM-1-2019/train-dev-test/partitions.json" \
      --dataset taskmaster
done

# Taskmaster TM-2 -> data/taskmaster/TM-2-2020/v3
for f in flights food-ordering hotels movies music restaurant-search sports; do
  $PP --input_path "$DATASETS/Taskmaster/TM-2-2020/data/${f}.json" \
      --output_path data/taskmaster/TM-2-2020/v3/ --dataset taskmaster
done

# Taskmaster TM-3 -> data/taskmaster/TM-3-2020/v3
$PP --input_path "$DATASETS/Taskmaster/TM-3-2020/data/data.json" \
    --output_path data/taskmaster/TM-3-2020/v3 \
    --partition_file "$DATASETS/Taskmaster/TM-3-2020/splits/partitions.json" \
    --dataset taskmaster

echo ""
echo "### Step 2/4: make_ser_dataset -> text->MR 'ds' examples + per-topic hint maps ###"

# E2E
$MK data/e2e-nlg/v1/trainset.json       data/e2e-nlg/v1/train-ds.json data/e2e-nlg/hint-map.json e2e_nlg
$MK data/e2e-nlg/v1/devset.json         data/e2e-nlg/v1/valid-ds.json data/e2e-nlg/hint-map.json e2e_nlg
$MK data/e2e-nlg/v1/testset_w_refs.json data/e2e-nlg/v1/test-ds.json  data/e2e-nlg/hint-map.json e2e_nlg

# ViGGO
$MK data/viggo-v1/v4/viggo-train.json data/viggo-v1/v4/train-ds.json data/viggo-v1/hint-map.json viggo
$MK data/viggo-v1/v4/viggo-valid.json data/viggo-v1/v4/valid-ds.json data/viggo-v1/hint-map.json viggo
$MK data/viggo-v1/v4/viggo-test.json  data/viggo-v1/v4/test-ds.json  data/viggo-v1/hint-map.json viggo

# RNNLG -> data/rnnlg/v2/<domain>-<split>-ds.json
for dom in hotel laptop restaurant tv; do
  $MK "data/rnnlg/${dom}/v2/train.json" "data/rnnlg/v2/${dom}-train-ds.json" data/rnnlg/hint-map.json rnnlg
  $MK "data/rnnlg/${dom}/v2/valid.json" "data/rnnlg/v2/${dom}-valid-ds.json" data/rnnlg/hint-map.json rnnlg
  $MK "data/rnnlg/${dom}/v2/test.json"  "data/rnnlg/v2/${dom}-test-ds.json"  data/rnnlg/hint-map.json rnnlg
done

# Taskmaster TM-1 (make_ser_dataset splits by topic internally)
$MK data/taskmaster/TM-1-2019/v3/self-dialogs-train.json data/taskmaster/TM-1-2019/v3/self-dialogs-train.json data/taskmaster/TM-1-2019/hint-map.json tm1
$MK data/taskmaster/TM-1-2019/v3/self-dialogs-test.json  data/taskmaster/TM-1-2019/v3/self-dialogs-test.json  data/taskmaster/TM-1-2019/hint-map.json tm1
$MK data/taskmaster/TM-1-2019/v3/woz-dialogs.json        data/taskmaster/TM-1-2019/v3/woz-dialogs.json        data/taskmaster/TM-1-2019/hint-map.json tm1

# Taskmaster TM-2
for f in flights food-ordering hotels movies music restaurant-search sports; do
  $MK "data/taskmaster/TM-2-2020/v3/${f}.json" "data/taskmaster/TM-2-2020/v3/${f}-ds.json" data/taskmaster/TM-2-2020/hint-map.json tm2
done

# Taskmaster TM-3
$MK data/taskmaster/TM-3-2020/v3/data.json data/taskmaster/TM-3-2020/v3/data.json data/taskmaster/TM-3-2020/hint-map.json tm3

echo ""
echo "### Step 3/3: merge + select -> data/multi_ser_v9/ ###"
mkdir -p data/multi_ser_v9
python -m xdomain_ser.data_prep.merge_select \
    --output_path data/multi_ser_v9/ \
    --files data/multi_ser_v9/files.json \
    --train_k "$TRAIN_K" --dev_k "$DEV_K" \
    --selection_method "$SELECTION_METHOD" --seed "$SEED"

# NOTE: the representative dev set (data/multi_ser_v9/dev-repr-120.json) is a
# SHIPPED phase-0 artifact and is NOT rebuilt here. It was constructed in the
# v5 era against the facloc-selected v5 train-200, so re-running representative_dev
# against this v9 (random) train-200 produces a *different* dev set (cleaner --
# 0 overlap with v9 train-200 -- but not the one the published experiments used).
# To build a fresh representative dev set for a new domain, run:
#   python -m xdomain_ser.data_prep.representative_dev --output_dir <dir> --dev_k 10 --seed 42

echo ""
echo "==================================================================="
echo "  DONE. Reconstructed dataset in data/multi_ser_v9/"
echo "  Compare against the shipped files (train-200.json, test.json, ...)"
echo "  to verify a byte-identical rebuild."
echo "==================================================================="
