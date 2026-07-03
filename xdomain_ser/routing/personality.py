# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Per-example method routing on personality SER comparison data.

Compares five methods against gold-annotated SER on 1000 E2E restaurant
examples (500 LLM + 500 seq2seq):

1. E2E Aligner (rule-based)
2. LoRA MySER (extraction + ranking)
3. NLI Baseline (RoBERTa-MNLI)
4. Score-threshold routing (LoRA/NLI by ranker confidence)
5. LR routing (logistic regression on per-example features)

Note: this script has its OWN local ``extract_features`` because the
personality eval surface differs from the multi-domain routing eval-1
surface in two ways: (a) the gold MR is already a flat dict
(``ex["mr"]``) rather than the list-of-tuples that
``ser.mr_list_to_dict`` converts; (b) the routing features use
personality + source one-hots instead of topic one-hots. The core
9 features (score / complexity / NLI confidence) mirror those in
:mod:`xdomain_ser.routing.features`.
"""
import argparse
import json
import os
from collections import defaultdict, Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from xdomain_ser.core import ser
from xdomain_ser.ranking.make_eval_data import select_ranking_preds

# Note: xdomain_ser.baselines.e2e_aligner is imported lazily inside
# run_aligner() so that this module loads cleanly before Stage 7
# (baselines) is migrated.


# --- Slot name mappings ---

ALIGNER_TO_GOLD_SLOT = {
    "customer rating": "customerRating",
}

LORA_REVERSE_KEY_MAP = {
    "customerRating": "customerRating",
    "family_suitability": "familyFriendly",
    "venue_type": "eatType",
    "cuisine_type": "food",
    "nearby_landmark": "near",
    "area_zone": "area",
}


def mr_dict_to_list(mr_dict):
    """Convert flat MR dict to list-of-tuples for pack_e2e_nlg_mr."""
    result = []
    for slot, val in mr_dict.items():
        if isinstance(val, list):
            result.append([slot] + val)
        else:
            result.append([slot, val])
    return result


def lora_pack_e2e_nlg_mr(d):
    """Pack LoRA-extracted MR to E2E-native slot names."""
    slot_value_pairs = []
    for slot_val in d:
        name = slot_val[0]
        values = slot_val[1:]
        if not isinstance(values, list):
            values = [values]
        orig_name = LORA_REVERSE_KEY_MAP.get(name, name)
        if name == "family_suitability":
            for v in values:
                orig_value = "yes" if v == "family-friendly" else "no"
                slot_value_pairs.append((orig_name, orig_value))
        else:
            slot_value_pairs.append((orig_name, values))
    return slot_value_pairs


def run_aligner(example):
    """Extract MR using surface-form alignment."""
    # Deferred: pulls in the rule-based E2E aligner from Stage 7.
    from xdomain_ser.baselines.e2e_aligner import (
        pack_e2e_nlg_mr as aligner_pack_e2e_nlg_mr,
        extract_mr,
    )

    gold_mr = example["mr"]
    text = example["clean_pred_text"]
    gold_mr_as_list = mr_dict_to_list(gold_mr)
    mr_mapped = aligner_pack_e2e_nlg_mr(gold_mr_as_list)
    extracted = extract_mr(text, mr_mapped)
    extracted_flat = {}
    for k, vals in extracted.items():
        gold_key = ALIGNER_TO_GOLD_SLOT.get(k, k)
        if len(vals) == 1:
            extracted_flat[gold_key] = vals[0]
        else:
            extracted_flat[gold_key] = vals
    return extracted_flat


def run_lora(example):
    """Select best LoRA-extracted MR, convert to E2E slot names."""
    best_pred = select_ranking_preds(example["pred_mr"], example["pred_scores"])
    bpred = []
    for slot, vals in best_pred.items():
        bpred.append([slot] + vals)
    bpred_mapped = lora_pack_e2e_nlg_mr(bpred)
    bpred_mr = {}
    for (k, vals) in bpred_mapped:
        if isinstance(vals, list):
            v = vals[0]
        else:
            v = vals
        bpred_mr[k] = v
    return bpred_mr


SCORE_THRESHOLDS = [1.5, 1.75, 2.0, 2.25, 2.5, 2.6, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95]


def stratified_split_personality(data, seed=42, test_frac=0.5):
    """Split data into dev/test, stratified by personality x source."""
    rng = np.random.RandomState(seed)

    strata = defaultdict(list)
    for i, ex in enumerate(data):
        key = (ex["personality"], ex["source"])
        strata[key].append(i)

    dev_indices = []
    test_indices = []

    for key in sorted(strata.keys()):
        indices = strata[key]
        rng.shuffle(indices)
        split_point = int(len(indices) * test_frac)
        dev_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])

    return sorted(dev_indices), sorted(test_indices)


def run_nli_batch_with_probs(examples, nli_model, batch_size=32):
    """Run NLI inference on all examples, returning per-slot probabilities.

    Uses E2E-native slot names from gold MR directly.
    """
    from xdomain_ser.nli.templates import slot_value_to_template

    all_pairs = []
    pair_index = []

    for ex_idx, ex in enumerate(examples):
        text = ex["clean_pred_text"]
        gold_mr = ex["mr"]

        for slot, value in gold_mr.items():
            if isinstance(value, list):
                for v in value:
                    template = slot_value_to_template(slot, v)
                    if template is None:
                        continue
                    all_pairs.append((text, template))
                    pair_index.append((ex_idx, slot, v))
            else:
                template = slot_value_to_template(slot, value)
                if template is None:
                    continue
                all_pairs.append((text, template))
                pair_index.append((ex_idx, slot, value))

    print(f"  NLI: {len(all_pairs)} total pairs for {len(examples)} examples")

    all_probs = nli_model.batch_entailment(all_pairs, batch_size=batch_size)

    example_nli_probs = defaultdict(list)
    for (ex_idx, slot, value), prob in zip(pair_index, all_probs):
        example_nli_probs[ex_idx].append((slot, value, prob))

    return dict(example_nli_probs)


def recover_nli_mr(gold_mr_dict, nli_results, threshold=0.3):
    """Recover MR from NLI results. Returns flat dict with E2E slot names."""
    recovered = {}
    for slot, value, prob in nli_results:
        if prob > threshold:
            recovered[slot] = value
    return recovered


def compute_ser_agreement(pred_mr_dict, gold_mr_dict, gold_ser):
    """Compute SER of pred_mr vs gold_mr, then compare against gold_ser annotation."""
    computed = ser.compute_ser(pred_mr_dict, gold_mr_dict)

    s_acc = computed["S"] == gold_ser["S"]
    d_acc = computed["D"] == gold_ser["D"]
    i_acc = computed["I"] == gold_ser["I"]
    all_acc = s_acc and d_acc and i_acc
    ser_error = computed["SER"] - gold_ser["SER"]

    return {
        "S_acc": s_acc,
        "D_acc": d_acc,
        "I_acc": i_acc,
        "all_acc": all_acc,
        "ser_error": ser_error,
        "computed_ser": computed,
    }


def extract_features(ex, nli_probs):
    """Extract per-example features for routing (personality variant).

    Differs from ``xdomain_ser.routing.features.compute_routing_features``
    in: (1) gold MR is already a flat dict (``ex["mr"]``), (2) NLI probs
    are passed as a list rather than looked up from a dict, (3) one-hots
    are personality + source instead of topic.
    """
    features = {}

    scores = sorted(ex["pred_scores"], reverse=True)
    features["top_score"] = scores[0]
    features["score_gap"] = scores[0] - scores[1] if len(scores) > 1 else 0.0
    features["score_spread"] = scores[0] - scores[-1] if len(scores) > 1 else 0.0

    gold_mr = ex["mr"]
    lora_mr = run_lora(ex)

    n_gold_slots = len(gold_mr)
    n_lora_slots = len(lora_mr)
    features["n_gold_slots"] = n_gold_slots
    features["n_lora_slots"] = n_lora_slots
    features["slot_ratio"] = n_lora_slots / max(n_gold_slots, 1)

    slot_probs = []
    for slot in gold_mr:
        probs_for_slot = [p for s, v, p in nli_probs if s == slot]
        if probs_for_slot:
            slot_probs.append(max(probs_for_slot))
        else:
            slot_probs.append(0.5)

    if slot_probs:
        features["mean_nli_prob"] = float(np.mean(slot_probs))
        features["min_nli_prob"] = float(np.min(slot_probs))
        features["nli_coverage"] = sum(1 for p in slot_probs if p > 0.5) / len(slot_probs)
    else:
        features["mean_nli_prob"] = 0.5
        features["min_nli_prob"] = 0.5
        features["nli_coverage"] = 0.0

    personalities = ["AGREEABLE", "CONSCIENTIOUSNESS", "DISAGREEABLE",
                     "EXTRAVERT", "UNCONSCIENTIOUSNESS"]
    pers = ex.get("personality", "unknown")
    for p in personalities:
        features[f"pers_{p}"] = 1.0 if pers == p else 0.0

    features["source_llm"] = 1.0 if ex.get("source") == "llm" else 0.0

    return features


def compute_per_example_label(lora_agreement, nli_agreement):
    """Compute routing label: 1 = prefer LoRA, 0 = prefer NLI. Tie -> LoRA."""
    lora_correct = lora_agreement["all_acc"]
    nli_correct = nli_agreement["all_acc"]

    if lora_correct and not nli_correct:
        return 1
    elif nli_correct and not lora_correct:
        return 0
    else:
        return 1


def evaluate_method_on_subset(per_example_results, method_name, indices):
    """Aggregate metrics for a method on a subset of examples."""
    vals = {"S_acc": [], "D_acc": [], "I_acc": [], "all_acc": [], "ser_error": []}

    for idx in indices:
        r = per_example_results[idx][method_name]
        vals["S_acc"].append(r["S_acc"])
        vals["D_acc"].append(r["D_acc"])
        vals["I_acc"].append(r["I_acc"])
        vals["all_acc"].append(r["all_acc"])
        vals["ser_error"].append(r["ser_error"])

    n = len(indices)
    if n == 0:
        return {"S_acc": 0, "D_acc": 0, "I_acc": 0, "all_acc": 0, "SER_MAE": 0, "count": 0}

    return {
        "S_acc": sum(vals["S_acc"]) / n,
        "D_acc": sum(vals["D_acc"]) / n,
        "I_acc": sum(vals["I_acc"]) / n,
        "all_acc": sum(vals["all_acc"]) / n,
        "SER_MAE": sum(abs(e) for e in vals["ser_error"]) / n,
        "count": n,
    }


def evaluate_routing_on_subset(per_example_results, indices, routing_decisions):
    """Evaluate routing on a subset: use lora or nli per routing decision."""
    vals = {"S_acc": [], "D_acc": [], "I_acc": [], "all_acc": [], "ser_error": []}

    for idx in indices:
        method = "lora" if routing_decisions[idx] == 1 else "nli"
        r = per_example_results[idx][method]
        vals["S_acc"].append(r["S_acc"])
        vals["D_acc"].append(r["D_acc"])
        vals["I_acc"].append(r["I_acc"])
        vals["all_acc"].append(r["all_acc"])
        vals["ser_error"].append(r["ser_error"])

    n = len(indices)
    if n == 0:
        return {"S_acc": 0, "D_acc": 0, "I_acc": 0, "all_acc": 0, "SER_MAE": 0, "count": 0}

    return {
        "S_acc": sum(vals["S_acc"]) / n,
        "D_acc": sum(vals["D_acc"]) / n,
        "I_acc": sum(vals["I_acc"]) / n,
        "all_acc": sum(vals["all_acc"]) / n,
        "SER_MAE": sum(abs(e) for e in vals["ser_error"]) / n,
        "count": n,
    }


def score_threshold_sweep(feature_dicts, per_example_results, dev_indices):
    """Sweep top_score thresholds on dev set."""
    sweep_results = []
    best_all_acc = -1
    best_threshold = None

    for threshold in SCORE_THRESHOLDS:
        decisions = {}
        for idx in dev_indices:
            decisions[idx] = 1 if feature_dicts[idx]["top_score"] >= threshold else 0

        m = evaluate_routing_on_subset(per_example_results, dev_indices, decisions)

        n_lora = sum(1 for idx in dev_indices if decisions[idx] == 1)
        n_nli = len(dev_indices) - n_lora

        sweep_results.append({
            "threshold": threshold,
            "n_lora": n_lora,
            "n_nli": n_nli,
            **m,
        })

        if m["all_acc"] > best_all_acc:
            best_all_acc = m["all_acc"]
            best_threshold = threshold

    return sweep_results, best_threshold


def train_logistic_regression(feature_dicts, labels, dev_indices, test_indices,
                              feature_names):
    """Train LR on dev, predict on test."""
    X_dev = np.array([[feature_dicts[i][f] for f in feature_names] for i in dev_indices])
    y_dev = np.array([labels[i] for i in dev_indices])

    X_test = np.array([[feature_dicts[i][f] for f in feature_names] for i in test_indices])

    scaler = StandardScaler()
    X_dev_scaled = scaler.fit_transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_dev_scaled, y_dev)

    dev_acc = model.score(X_dev_scaled, y_dev)

    test_pred_array = model.predict(X_test_scaled)
    test_preds = {idx: int(pred) for idx, pred in zip(test_indices, test_pred_array)}

    coefs = model.coef_[0]
    importance = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)

    return model, scaler, dev_acc, test_preds, importance


def print_comparison(method_metrics, title="COMPARISON"):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)

    header = f"  {'Method':>14} {'S_acc':>8} {'D_acc':>8} {'I_acc':>8} {'All_acc':>8} {'SER_MAE':>8} {'N':>6}"
    print(header)
    print(f"  {'-' * len(header)}")

    for name, m in method_metrics:
        print(f"  {name:>14} {m['S_acc']:>8.4f} {m['D_acc']:>8.4f} {m['I_acc']:>8.4f} "
              f"{m['all_acc']:>8.4f} {m['SER_MAE']:>8.4f} {m['count']:>6}")


def print_breakdown(per_example_results, data, method_metrics_items, indices, group_key, title):
    """Print breakdown by a grouping key (source, personality)."""
    print(f"\n--- {title} ---")

    groups = sorted(set(data[i][group_key] for i in indices))

    for group in groups:
        group_indices = [i for i in indices if data[i][group_key] == group]

        print(f"\n  {group_key}: {group} (N={len(group_indices)})")
        header = f"    {'Method':>14} {'S_acc':>8} {'D_acc':>8} {'I_acc':>8} {'All_acc':>8} {'MAE':>8}"
        print(header)
        print(f"    {'-' * 60}")

        for method_name, decisions_or_name in method_metrics_items:
            if isinstance(decisions_or_name, str):
                m = evaluate_method_on_subset(per_example_results, decisions_or_name, group_indices)
            else:
                m = evaluate_routing_on_subset(per_example_results, group_indices, decisions_or_name)
            print(f"    {method_name:>14} {m['S_acc']:>8.4f} {m['D_acc']:>8.4f} "
                  f"{m['I_acc']:>8.4f} {m['all_acc']:>8.4f} {m['SER_MAE']:>8.4f}")


def save_results(output_dir, data, per_example_results, feature_dicts, labels,
                 dev_indices, test_indices,
                 sweep_results, best_threshold,
                 lr_importance, lr_dev_acc,
                 method_metrics_test, score_routing_test, lr_test_preds):
    """Save all output files."""
    os.makedirs(output_dir, exist_ok=True)

    sweep_path = os.path.join(output_dir, "score_routing_sweep.tsv")
    with open(sweep_path, "w") as f:
        f.write("threshold\tn_lora\tn_nli\tcount\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\n")
        for r in sweep_results:
            f.write(f"{r['threshold']}\t{r['n_lora']}\t{r['n_nli']}\t{r['count']}\t"
                    f"{r['S_acc']:.4f}\t{r['D_acc']:.4f}\t{r['I_acc']:.4f}\t"
                    f"{r['all_acc']:.4f}\t{r['SER_MAE']:.4f}\n")
    print(f"Score routing sweep saved to: {sweep_path}")

    imp_path = os.path.join(output_dir, "lr_feature_importance.tsv")
    with open(imp_path, "w") as f:
        f.write("feature\tcoefficient\n")
        for feat, coef in lr_importance:
            f.write(f"{feat}\t{coef:.4f}\n")
    print(f"LR feature importance saved to: {imp_path}")

    comp_path = os.path.join(output_dir, "comparison_results.tsv")
    with open(comp_path, "w") as f:
        f.write("method\tcount\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\n")
        for name, m in method_metrics_test:
            f.write(f"{name}\t{m['count']}\t{m['S_acc']:.4f}\t{m['D_acc']:.4f}\t"
                    f"{m['I_acc']:.4f}\t{m['all_acc']:.4f}\t{m['SER_MAE']:.4f}\n")
    print(f"Comparison results saved to: {comp_path}")

    source_path = os.path.join(output_dir, "per_source_comparison.tsv")
    with open(source_path, "w") as f:
        f.write("source\tmethod\tcount\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\n")
        methods_eval = [
            ("Aligner", "aligner"), ("LoRA", "lora"), ("NLI", "nli"),
            ("ScoreRouting", score_routing_test), ("LR-Routing", lr_test_preds),
        ]
        for src in ["llm", "seq2seq"]:
            src_indices = [i for i in test_indices if data[i]["source"] == src]
            for method_name, method_ref in methods_eval:
                if isinstance(method_ref, str):
                    m = evaluate_method_on_subset(per_example_results, method_ref, src_indices)
                else:
                    m = evaluate_routing_on_subset(per_example_results, src_indices, method_ref)
                f.write(f"{src}\t{method_name}\t{m['count']}\t{m['S_acc']:.4f}\t"
                        f"{m['D_acc']:.4f}\t{m['I_acc']:.4f}\t{m['all_acc']:.4f}\t"
                        f"{m['SER_MAE']:.4f}\n")
    print(f"Per-source comparison saved to: {source_path}")

    pers_path = os.path.join(output_dir, "per_personality_comparison.tsv")
    with open(pers_path, "w") as f:
        f.write("personality\tmethod\tcount\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\n")
        personalities = sorted(set(data[i]["personality"] for i in test_indices))
        methods_eval = [
            ("Aligner", "aligner"), ("LoRA", "lora"), ("NLI", "nli"),
            ("ScoreRouting", score_routing_test), ("LR-Routing", lr_test_preds),
        ]
        for pers in personalities:
            pers_indices = [i for i in test_indices if data[i]["personality"] == pers]
            for method_name, method_ref in methods_eval:
                if isinstance(method_ref, str):
                    m = evaluate_method_on_subset(per_example_results, method_ref, pers_indices)
                else:
                    m = evaluate_routing_on_subset(per_example_results, pers_indices, method_ref)
                f.write(f"{pers}\t{method_name}\t{m['count']}\t{m['S_acc']:.4f}\t"
                        f"{m['D_acc']:.4f}\t{m['I_acc']:.4f}\t{m['all_acc']:.4f}\t"
                        f"{m['SER_MAE']:.4f}\n")
    print(f"Per-personality comparison saved to: {pers_path}")

    detail_path = os.path.join(output_dir, "routing_details.json")
    routing_details = []
    for idx in test_indices:
        ex = data[idx]
        detail = {
            "ex_idx": idx,
            "id": ex["id"],
            "personality": ex["personality"],
            "source": ex["source"],
            "features": {k: float(v) for k, v in feature_dicts[idx].items()},
            "oracle_label": labels[idx],
            "score_routing_choice": score_routing_test[idx],
            "lr_routing_choice": lr_test_preds[idx],
            "lora_all_acc": per_example_results[idx]["lora"]["all_acc"],
            "nli_all_acc": per_example_results[idx]["nli"]["all_acc"],
            "aligner_all_acc": per_example_results[idx]["aligner"]["all_acc"],
        }
        routing_details.append(detail)

    output_obj = {
        "best_score_threshold": best_threshold,
        "lr_dev_accuracy": lr_dev_acc,
        "summary": {name: m for name, m in method_metrics_test},
        "examples": routing_details,
    }
    with open(detail_path, "w") as f:
        json.dump(output_obj, f, indent=2, default=str)
    print(f"Routing details saved to: {detail_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-example routing on personality SER comparison data")
    parser.add_argument(
        "--input_path", type=str,
        default="evaluation/gold/gold-annotated.json",
        help="Path to gold-annotated examples.")
    parser.add_argument(
        "--precomputed_path", type=str,
        default="evaluation/gold/ser-comparison-results.json",
        help="Path to precomputed 3-method comparison results.")
    parser.add_argument(
        "--output_dir", type=str,
        default="results/personality_routing",
        help="Output directory.")
    parser.add_argument(
        "--nli_threshold", type=float, default=0.3,
        help="NLI entailment threshold (default: 0.3).")
    parser.add_argument(
        "--nli_batch_size", type=int, default=32,
        help="NLI inference batch size.")
    parser.add_argument(
        "--device", default=None,
        help="Torch device for NLI model.")
    parser.add_argument(
        "--skip_nli_inference", action="store_true",
        help="Use precomputed NLI results (no GPU needed, but no NLI prob features).")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for stratified split.")
    args = parser.parse_args()
    print("args:", args)

    print(f"\nLoading gold-annotated data from: {args.input_path}")
    with open(args.input_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")

    print("\nComputing method MRs...")

    example_nli_probs = {}

    if args.skip_nli_inference:
        print(f"Loading precomputed results from: {args.precomputed_path}")
        with open(args.precomputed_path) as f:
            precomputed = json.load(f)

        per_example_results = []
        for idx, (ex, precomp) in enumerate(zip(data, precomputed["per_example"])):
            gold_mr = ex["mr"]
            gold_ser = ex["gold_ser"]

            result = {}
            for method in ["aligner", "lora", "nli"]:
                pred_mr = precomp[method]["pred_mr"]
                result[method] = compute_ser_agreement(pred_mr, gold_mr, gold_ser)

            per_example_results.append(result)
            example_nli_probs[idx] = []

        print("  Using precomputed results (NLI probability features unavailable)")

    else:
        print("Loading NLI model...")
        from xdomain_ser.nli.model import NLIModel
        nli_model = NLIModel(device=args.device)

        print("Running NLI inference...")
        example_nli_probs = run_nli_batch_with_probs(
            data, nli_model, batch_size=args.nli_batch_size)
        del nli_model

        per_example_results = []
        for idx, ex in enumerate(data):
            gold_mr = ex["mr"]
            gold_ser = ex["gold_ser"]

            result = {}

            aligner_mr = run_aligner(ex)
            result["aligner"] = compute_ser_agreement(aligner_mr, gold_mr, gold_ser)

            lora_mr = run_lora(ex)
            result["lora"] = compute_ser_agreement(lora_mr, gold_mr, gold_ser)

            nli_probs = example_nli_probs.get(idx, [])
            nli_mr = recover_nli_mr(gold_mr, nli_probs, threshold=args.nli_threshold)
            result["nli"] = compute_ser_agreement(nli_mr, gold_mr, gold_ser)

            per_example_results.append(result)

    n = len(data)
    for method in ["aligner", "lora", "nli"]:
        acc = sum(r[method]["all_acc"] for r in per_example_results) / n
        mae = sum(abs(r[method]["ser_error"]) for r in per_example_results) / n
        print(f"  {method:>10}: All_acc={acc:.4f}, SER_MAE={mae:.4f}")

    dev_indices, test_indices = stratified_split_personality(data, seed=args.seed)
    print(f"\nDev split: {len(dev_indices)} examples")
    print(f"Test split: {len(test_indices)} examples")

    for split_name, indices in [("Dev", dev_indices), ("Test", test_indices)]:
        src_counts = Counter(data[i]["source"] for i in indices)
        pers_counts = Counter(data[i]["personality"] for i in indices)
        print(f"  {split_name} sources: {dict(sorted(src_counts.items()))}")
        print(f"  {split_name} personalities: {dict(sorted(pers_counts.items()))}")

    print("\nExtracting per-example features and labels...")
    feature_dicts = {}
    labels = {}

    for idx in range(len(data)):
        nli_probs = example_nli_probs.get(idx, [])
        feature_dicts[idx] = extract_features(data[idx], nli_probs)
        labels[idx] = compute_per_example_label(
            per_example_results[idx]["lora"],
            per_example_results[idx]["nli"],
        )

    sample_features = feature_dicts[0]
    feature_names = [f for f in sample_features if not f.startswith("pers_") and f != "source_llm"]
    feature_names += sorted(f for f in sample_features if f.startswith("pers_"))
    feature_names.append("source_llm")

    dev_lora = sum(1 for i in dev_indices if labels[i] == 1)
    dev_nli = len(dev_indices) - dev_lora
    test_lora = sum(1 for i in test_indices if labels[i] == 1)
    test_nli = len(test_indices) - test_lora
    print(f"Dev labels: {dev_lora} LoRA-preferred, {dev_nli} NLI-preferred")
    print(f"Test labels: {test_lora} LoRA-preferred, {test_nli} NLI-preferred")

    print("\n" + "=" * 80)
    print("OPTION A: Score-Threshold Routing (sweep on dev)")
    print("=" * 80)

    sweep_results, best_threshold = score_threshold_sweep(
        feature_dicts, per_example_results, dev_indices)

    print(f"\n  {'threshold':>10} {'n_lora':>7} {'n_nli':>7} {'All_acc':>8} {'SER_MAE':>8}")
    print(f"  {'-' * 45}")
    for r in sweep_results:
        marker = " <-- best" if r["threshold"] == best_threshold else ""
        print(f"  {r['threshold']:>10.2f} {r['n_lora']:>7} {r['n_nli']:>7} "
              f"{r['all_acc']:>8.4f} {r['SER_MAE']:>8.4f}{marker}")

    best_sweep = [r for r in sweep_results if r["threshold"] == best_threshold][0]
    print(f"\nBest threshold: {best_threshold} (All_acc={best_sweep['all_acc']:.4f})")

    score_routing_test = {
        idx: (1 if feature_dicts[idx]["top_score"] >= best_threshold else 0)
        for idx in test_indices
    }

    print("\n" + "=" * 80)
    print("OPTION B: Logistic Regression Routing")
    print("=" * 80)

    model, scaler, lr_dev_acc, lr_test_preds, lr_importance = train_logistic_regression(
        feature_dicts, labels, dev_indices, test_indices, feature_names)

    print(f"\n  Dev accuracy: {lr_dev_acc:.4f}")
    print("\n  Feature importances (top 15):")
    print(f"  {'feature':>20} {'coefficient':>12}")
    print(f"  {'-' * 35}")
    for feat, coef in lr_importance[:15]:
        print(f"  {feat:>20} {coef:>+12.4f}")

    lr_n_lora = sum(1 for idx in test_indices if lr_test_preds[idx] == 1)
    lr_n_nli = len(test_indices) - lr_n_lora
    print(f"\n  LR test routing: {lr_n_lora} LoRA, {lr_n_nli} NLI")

    method_metrics_test = [
        ("Aligner", evaluate_method_on_subset(per_example_results, "aligner", test_indices)),
        ("LoRA", evaluate_method_on_subset(per_example_results, "lora", test_indices)),
        ("NLI", evaluate_method_on_subset(per_example_results, "nli", test_indices)),
        ("ScoreRouting", evaluate_routing_on_subset(per_example_results, test_indices, score_routing_test)),
        ("LR-Routing", evaluate_routing_on_subset(per_example_results, test_indices, lr_test_preds)),
    ]

    print_comparison(method_metrics_test, title="FIVE-WAY COMPARISON (test split)")

    oracle_decisions = {idx: labels[idx] for idx in test_indices}
    oracle_m = evaluate_routing_on_subset(per_example_results, test_indices, oracle_decisions)
    print(f"\n  Oracle routing (test): All_acc={oracle_m['all_acc']:.4f}, "
          f"SER_MAE={oracle_m['SER_MAE']:.4f}")

    methods_eval = [
        ("Aligner", "aligner"), ("LoRA", "lora"), ("NLI", "nli"),
        ("ScoreRouting", score_routing_test), ("LR-Routing", lr_test_preds),
    ]

    print_breakdown(per_example_results, data, methods_eval, test_indices,
                    "source", "BY SOURCE (test split)")
    print_breakdown(per_example_results, data, methods_eval, test_indices,
                    "personality", "BY PERSONALITY (test split)")

    save_results(
        args.output_dir, data, per_example_results, feature_dicts, labels,
        dev_indices, test_indices,
        sweep_results, best_threshold,
        lr_importance, lr_dev_acc,
        method_metrics_test, score_routing_test, lr_test_preds)

    print("\nDone.")


if __name__ == "__main__":
    main()
