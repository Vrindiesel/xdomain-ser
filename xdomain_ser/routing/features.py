# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Per-example routing features for the multi-domain SER pipeline.

Shared by the score-threshold + logistic-regression routing in
``selector.py`` and the phase-0 XGBoost / value-normalisation variants
in ``experimental/``. The personality routing pipeline in
``personality.py`` has a slightly different feature space (pers_X +
source_llm one-hots and pre-computed agreement labels) and keeps its
own local ``extract_features`` -- the per-feature helpers below are
deliberately written to be reusable by either side.

Feature groups (9 base + per-topic one-hots = 22 features on the
canonical multi-domain eval data):

* **Score features (3):** top_score, score_gap, score_spread
* **MR complexity features (3):** n_gold_slots, n_lora_slots, slot_ratio
* **NLI confidence features (3):** mean_nli_prob, min_nli_prob,
  nli_coverage
* **Topic one-hots:** one float per topic in ``all_topics``
"""

import numpy as np

from xdomain_ser.core import ser
from xdomain_ser.mixture.fuse import build_nli_prob_dict
from xdomain_ser.ranking.make_eval_data import select_ranking_preds


def _score_features(scores):
    """top_score / score_gap / score_spread from a list of ranker scores."""
    scores = sorted(scores, reverse=True)
    return {
        "top_score": scores[0],
        "score_gap": scores[0] - scores[1] if len(scores) > 1 else 0.0,
        "score_spread": scores[0] - scores[-1] if len(scores) > 1 else 0.0,
    }


def _complexity_features(n_gold_slots, n_lora_slots):
    """MR complexity features given slot counts."""
    return {
        "n_gold_slots": n_gold_slots,
        "n_lora_slots": n_lora_slots,
        "slot_ratio": n_lora_slots / max(n_gold_slots, 1),
    }


def _nli_features(slot_probs):
    """NLI confidence features from a list of per-slot max entailment probs.

    Missing values default to neutral (0.5).
    """
    if slot_probs:
        return {
            "mean_nli_prob": float(np.mean(slot_probs)),
            "min_nli_prob": float(np.min(slot_probs)),
            "nli_coverage": sum(1 for p in slot_probs if p > 0.5) / len(slot_probs),
        }
    return {
        "mean_nli_prob": 0.5,
        "min_nli_prob": 0.5,
        "nli_coverage": 0.0,
    }


def compute_routing_features(ex, ex_idx, example_nli_results, all_topics):
    """Extract per-example routing features for the multi-domain eval surface.

    Args:
        ex: example dict from the ranker-scored eval JSON (must have
            ``mr``, ``pred_mr``, ``pred_scores``, ``hint_map_id``, ``topic``).
        ex_idx: integer index of this example in the dataset (used to look
            up its precomputed NLI results).
        example_nli_results: dict ``{ex_idx: [(slot, value, prob), ...]}``.
        all_topics: sorted list of all topic names for the one-hot encoding.

    Returns:
        dict ``{feature_name: float}`` -- 22 features on the canonical
        multi-domain eval data (9 base + 13 topic one-hots).
    """
    features = {}

    # --- Ranking score features ---
    features.update(_score_features(ex["pred_scores"]))

    # --- MR complexity features ---
    gold_mr_dict = ser.mr_list_to_dict(ex["mr"])
    lora_mr = select_ranking_preds(ex["pred_mr"], ex["pred_scores"])
    features.update(_complexity_features(len(gold_mr_dict), len(lora_mr)))

    # --- NLI confidence features ---
    nli_results = example_nli_results.get(ex_idx, [])
    nli_prob_dict = build_nli_prob_dict(nli_results)

    slot_probs = []
    for slot in gold_mr_dict:
        probs_for_slot = nli_prob_dict.get(slot, {})
        if probs_for_slot:
            slot_probs.append(max(probs_for_slot.values()))
        else:
            # dontcare / unmapped -- use neutral 0.5
            slot_probs.append(0.5)
    features.update(_nli_features(slot_probs))

    # --- Topic one-hot ---
    topic = ex.get("topic", "unknown")
    for t in all_topics:
        features[f"topic_{t}"] = 1.0 if t == topic else 0.0

    return features


def compute_pair_routing_features(ex, neg, pair_scores, pair_nli_results, all_topics):
    """Per-pair routing features under the corrected (P-deploy) protocol.

    Mirrors :func:`compute_routing_features` with every gold-MR read
    replaced by the pair's modified MR:

    * score features come from hypothesis-conditioned ranker scores
      (``ranking.score --ref_source hypothesis`` records),
    * ``n_gold_slots`` is the modified MR's slot count (the feature name is
      kept for comparability with the published feature tables; semantically
      it is "slots in the reference the system was given"),
    * NLI features iterate the modified MR's slots over per-pair NLI
      results (``nli.evaluator.group_pair_nli_results``).

    Args:
        ex: example dict (needs ``pred_mr``, ``topic``).
        neg: one entry of ``ex["negatives"]`` (needs ``mr``).
        pair_scores: list of k ranker scores for this (example, negative).
        pair_nli_results: list of (slot, value, prob) for this pair.
        all_topics: sorted topic names for the one-hot encoding.

    Returns:
        dict ``{feature_name: float}`` -- same 22-feature space as the
        per-text builder.
    """
    features = {}

    # --- Ranking score features (hypothesis-conditioned) ---
    features.update(_score_features(pair_scores))

    # --- MR complexity features (modified MR is the reference) ---
    hyp_mr_dict = {s: (v if isinstance(v, list) else [v])
                   for s, v in neg["mr"].items()}
    lora_mr = select_ranking_preds(ex["pred_mr"], pair_scores)
    features.update(_complexity_features(len(hyp_mr_dict), len(lora_mr)))

    # --- NLI confidence features (over the modified MR's slots) ---
    nli_prob_dict = build_nli_prob_dict(pair_nli_results)
    slot_probs = []
    for slot in hyp_mr_dict:
        probs_for_slot = nli_prob_dict.get(slot, {})
        if probs_for_slot:
            slot_probs.append(max(probs_for_slot.values()))
        else:
            # dontcare / unmapped -- use neutral 0.5
            slot_probs.append(0.5)
    features.update(_nli_features(slot_probs))

    # --- Topic one-hot ---
    topic = ex.get("topic", "unknown")
    for t in all_topics:
        features[f"topic_{t}"] = 1.0 if t == topic else 0.0

    return features
