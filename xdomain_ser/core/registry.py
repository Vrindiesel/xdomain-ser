"""Central registry for xdomain-ser assets.

Resolves data paths and HuggingFace Hub checkpoint identifiers. Single
source of truth for "where is X" -- no script should hard-code a path
or a HF repo ID.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
EVAL_ROOT = REPO_ROOT / "evaluation"

# Multi-domain SER dataset (v9 canonical)
HINT_MAPS = DATA_ROOT / "multi_ser_v9" / "hint_maps_v4.json"
PROMPT_EXAMPLES = DATA_ROOT / "multi_ser_v9" / "prompt-examples-dev-repr.json"
TRAIN_DATA = DATA_ROOT / "multi_ser_v9" / "train-200.json"
DEV_DATA = DATA_ROOT / "multi_ser_v9" / "dev-repr-120.json"
TEST_DATA = DATA_ROOT / "multi_ser_v9" / "test.json"

# Ranker evaluation set (canonical Eval-2 file; UNSCORED carries the cached
# 10-beam candidates, SCORED adds the published ref=true ranker scores)
RANKER_EVAL_UNSCORED = DATA_ROOT / "ranking_eval" / "negatives-v6-test.200.pe5.b10.json"
RANKER_EVAL_SCORED = DATA_ROOT / "ranking_eval" / "negatives-v6-test.200.pe5.b10.rexp4.ckpt60.scores.json"

# Eval-2 gold-annotated PERSONAGE data
EVAL2_GOLD = EVAL_ROOT / "gold" / "gold-annotated.json"
EVAL2_NEGATIVES = EVAL_ROOT / "gold" / "eval-negatives.json"

# HuggingFace Hub identifiers
HF_EXTRACTOR = "DavanHarrison/xdomain-ser-extractor"
HF_RANKER = "DavanHarrison/xdomain-ser-ranker"
HF_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
HF_NLI_MODEL = "FacebookAI/roberta-large-mnli"
