# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Persist and load fitted routing artifacts.

The routing pipeline historically re-fit the score threshold, the
logistic-regression router, and the XGBoost router in-process on every
run and persisted nothing. This module saves the fitted artifacts (with
their scalers, feature-name order, and provenance metadata) so the
evaluator facade can route without re-fitting:

* ``<dir>/threshold.json`` -- the swept score-routing threshold.
* ``<dir>/lr_router.joblib`` -- ``{"model", "scaler", "feature_names", "meta"}``.
* ``<dir>/xgb_router.joblib`` -- ``{"model", "scaler", "config", "feature_names", "meta"}``.

Feature dicts are converted to vectors in the stored ``feature_names``
order; a router bundle's prediction is 1 = use LoRA, 0 = use NLI.
"""
import json
import os

import joblib
import numpy as np


def _vector(features, feature_names):
    return np.array([[features[f] for f in feature_names]])


def save_routers(out_dir, *, score_threshold, lr_model, lr_scaler,
                 xgb_model, xgb_scaler, xgb_config, feature_names, meta):
    """Write threshold.json, lr_router.joblib, and xgb_router.joblib."""
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "threshold.json"), "w") as f:
        json.dump({"score_threshold": score_threshold, "meta": meta}, f,
                  indent=2)

    joblib.dump({"model": lr_model, "scaler": lr_scaler,
                 "feature_names": list(feature_names), "meta": meta},
                os.path.join(out_dir, "lr_router.joblib"))

    joblib.dump({"model": xgb_model, "scaler": xgb_scaler,
                 "config": xgb_config,
                 "feature_names": list(feature_names), "meta": meta},
                os.path.join(out_dir, "xgb_router.joblib"))

    return [os.path.join(out_dir, n) for n in
            ("threshold.json", "lr_router.joblib", "xgb_router.joblib")]


def load_routers(in_dir):
    """Load the persisted artifacts. Returns a dict with keys
    ``score_threshold``, ``lr``, ``xgb`` (each joblib bundle as saved)."""
    with open(os.path.join(in_dir, "threshold.json")) as f:
        thr = json.load(f)
    return {
        "score_threshold": thr["score_threshold"],
        "threshold_meta": thr.get("meta", {}),
        "lr": joblib.load(os.path.join(in_dir, "lr_router.joblib")),
        "xgb": joblib.load(os.path.join(in_dir, "xgb_router.joblib")),
    }


def route_with_bundle(bundle, features):
    """Predict 1 (LoRA) / 0 (NLI) from a feature dict with a saved
    LR or XGB bundle."""
    x = _vector(features, bundle["feature_names"])
    if bundle.get("scaler") is not None:
        x = bundle["scaler"].transform(x)
    return int(bundle["model"].predict(x)[0])
