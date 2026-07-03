# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0.
"""Unit tests for the corrected per-pair routing pipeline (Stage 9)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from xdomain_ser.routing.pair_selector import (
    build_units_deploy,
    evaluate_decisions,
    evaluate_fixed,
    threshold_sweep,
)
from xdomain_ser.routing.persist import (
    load_routers,
    route_with_bundle,
    save_routers,
)
from xdomain_ser.nli.evaluator import compute_metrics

DATA = [{
    "mr": [["a", "1"], ["b", "2"]],
    "pred_mr": ["<LIST>(a: 1); (b: 2)", "<LIST>(a: 1)"],
    "surface_form": "text",
    "hint_map_id": "hm",
    "topic": "t",
    "negatives": [
        {"label": 4, "mr": {"a": ["1"], "b": ["2"]}},   # identity pair
        {"label": 0, "mr": {"a": ["9"]}},               # corrupted pair
    ],
}]

# Pair (0,0): scores pick the full candidate; NLI entails both slots.
# Pair (0,1): scores pick the partial candidate; NLI rejects the slot.
PAIR_SCORES = {(0, 0): [2.0, 1.0], (0, 1): [1.0, 2.0]}
PAIR_NLI = {(0, 0): [("a", "1", 0.9), ("b", "2", 0.9)],
            (0, 1): [("a", "9", 0.05)]}


def _units():
    return build_units_deploy(DATA, PAIR_SCORES, PAIR_NLI, nli_threshold=0.3)


def test_deploy_units_labels_and_results():
    units = _units()
    assert set(units) == {(0, 0), (0, 1)}
    # identity pair: both methods correct -> tie -> LoRA-preferred (1)
    u0 = units[(0, 0)]
    assert (u0["lora_count"], u0["nli_count"], u0["label"]) == (1, 1, 1)
    # corrupted pair: lora misses the inserted-slot count, NLI (subset of
    # the checked MR) can only signal deletions -> both wrong -> tie -> 1
    u1 = units[(0, 1)]
    assert (u1["lora_count"], u1["nli_count"], u1["label"]) == (0, 0, 1)
    # features carry the per-pair conditioning
    assert u1["features"]["n_gold_slots"] == 1
    assert u0["features"]["top_score"] == 2.0


def test_evaluate_decisions_routes_per_unit():
    units = _units()
    keys = sorted(units)
    all_lora, _, _ = evaluate_fixed(units, keys, "lora")
    assert compute_metrics(all_lora)["all_acc"] == 0.5
    # routing both ways must tally each unit's chosen method
    w, _, _ = evaluate_decisions(units, keys, {(0, 0): 1, (0, 1): 0})
    m = compute_metrics(w)
    assert m["count"] == 2 and m["all_acc"] == 0.5


def test_threshold_sweep_separates_units():
    units = _units()
    keys = sorted(units)
    sweep, best = threshold_sweep(units, keys)
    assert len(sweep) > 0 and best is not None
    # at threshold 1.5 both units route to LoRA (top scores 2.0 and 2.0)
    row = next(r for r in sweep if r["threshold"] == 1.5)
    assert (row["n_lora"], row["n_nli"]) == (2, 0)


def test_persist_round_trip(tmp_path):
    rng = np.random.RandomState(0)
    X = rng.rand(40, 3)
    y = (X[:, 0] > 0.5).astype(int)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression().fit(scaler.transform(X), y)

    save_routers(tmp_path, score_threshold=2.95,
                 lr_model=model, lr_scaler=scaler,
                 xgb_model=model, xgb_scaler=scaler,
                 xgb_config={"max_depth": 3}, feature_names=["f0", "f1", "f2"],
                 meta={"protocol": "test"})
    loaded = load_routers(tmp_path)
    assert loaded["score_threshold"] == 2.95
    assert loaded["lr"]["feature_names"] == ["f0", "f1", "f2"]
    feats = {"f0": 0.9, "f1": 0.1, "f2": 0.1}
    assert route_with_bundle(loaded["lr"], feats) == 1
    feats_low = {"f0": 0.1, "f1": 0.1, "f2": 0.1}
    assert route_with_bundle(loaded["lr"], feats_low) == 0
