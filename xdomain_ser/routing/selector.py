# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Per-example method routing: select LoRA or NLI per example based on confidence.

Two routing strategies:

* Score-threshold -- route by LoRA ranker's top score.
* Logistic regression -- learned classifier on per-example features.

Key insight: instead of fusing at the slot level (which failed because NLI checks
gold values while the keep/remove decision preserves LoRA's potentially different
values), select which method's *complete MR* to use per example.

Feature extraction is factored into :mod:`xdomain_ser.routing.features`; the
``extract_features`` name is re-exported here for backwards compatibility with
phase-0 follow-up modules that imported it from the original
``routing_selector_eval.py``.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from xdomain_ser.core import ser
from xdomain_ser.mixture.threshold_sweep import stratified_split
from xdomain_ser.nli.evaluator import (
    collect_all_nli_pairs,
    compute_metrics,
    recover_mr_from_nli,
    tally_ser,
)
from xdomain_ser.nli.model import NLIModel
from xdomain_ser.ranking.make_eval_data import select_ranking_preds
from xdomain_ser.routing.features import compute_routing_features as extract_features

# Score-threshold sweep grid
SCORE_THRESHOLDS = [1.5, 1.75, 2.0, 2.25, 2.5, 2.6, 2.7, 2.75, 2.8, 2.85, 2.9, 2.95]


def compute_per_example_label(ex, ex_idx, example_nli_results, nli_threshold):
    """
    Compute the routing label for one example.

    For each negative, check whether LoRA or NLI gets all_acc correct.
    label = 1 (prefer LoRA) if lora_correct >= nli_correct, else 0 (prefer NLI).

    Returns:
        label: int (1=LoRA, 0=NLI)
        lora_correct: int
        nli_correct: int
    """
    gold_mr_dict = ser.mr_list_to_dict(ex["mr"])
    lora_mr = select_ranking_preds(ex["pred_mr"], ex["pred_scores"])

    nli_results = example_nli_results.get(ex_idx, [])
    nli_mr = recover_mr_from_nli(gold_mr_dict, nli_results, nli_threshold)

    lora_correct = 0
    nli_correct = 0

    for neg in ex["negatives"]:
        neg_mr = neg["mr"]
        ref_result = ser.compute_ser(gold_mr_dict, neg_mr)

        lora_result = ser.compute_ser(lora_mr, neg_mr)
        nli_result = ser.compute_ser(nli_mr, neg_mr)

        if (ref_result["S"] == lora_result["S"]
                and ref_result["D"] == lora_result["D"]
                and ref_result["I"] == lora_result["I"]):
            lora_correct += 1

        if (ref_result["S"] == nli_result["S"]
                and ref_result["D"] == nli_result["D"]
                and ref_result["I"] == nli_result["I"]):
            nli_correct += 1

    label = 1 if lora_correct >= nli_correct else 0
    return label, lora_correct, nli_correct


def evaluate_routing_on_subset(data, indices, example_nli_results,
                               nli_threshold, routing_decisions):
    """Evaluate routing on a subset: use the method indicated by routing_decisions[ex_idx]
    (1=LoRA, 0=NLI). Returns (working_results, category_results, topic_results)."""
    working = defaultdict(list)
    category = defaultdict(lambda: defaultdict(list))
    topic_res = defaultdict(lambda: defaultdict(list))

    for ex_idx in indices:
        ex = data[ex_idx]
        gold_mr_dict = ser.mr_list_to_dict(ex["mr"])
        topic = ex.get("topic", "unknown")

        use_lora = routing_decisions[ex_idx]
        if use_lora:
            pred_mr = select_ranking_preds(ex["pred_mr"], ex["pred_scores"])
        else:
            nli_results = example_nli_results.get(ex_idx, [])
            pred_mr = recover_mr_from_nli(gold_mr_dict, nli_results, nli_threshold)

        for neg in ex["negatives"]:
            neg_label = neg["label"]
            neg_mr = neg["mr"]
            ref_result = ser.compute_ser(gold_mr_dict, neg_mr)
            pred_result = ser.compute_ser(pred_mr, neg_mr)
            tally_ser(working, category, topic_res,
                      ref_result, pred_result, neg_label, topic)

    return working, category, topic_res


def evaluate_baselines_on_subset(data, indices, example_nli_results, nli_threshold):
    """Evaluate LoRA and NLI baselines on a subset. Returns per-method results."""
    results = {}
    for method in ["lora", "nli"]:
        w = defaultdict(list)
        c = defaultdict(lambda: defaultdict(list))
        t = defaultdict(lambda: defaultdict(list))
        results[method] = (w, c, t)

    for ex_idx in indices:
        ex = data[ex_idx]
        gold_mr_dict = ser.mr_list_to_dict(ex["mr"])
        topic = ex.get("topic", "unknown")

        lora_mr = select_ranking_preds(ex["pred_mr"], ex["pred_scores"])
        nli_results = example_nli_results.get(ex_idx, [])
        nli_mr = recover_mr_from_nli(gold_mr_dict, nli_results, nli_threshold)

        for neg in ex["negatives"]:
            neg_label = neg["label"]
            neg_mr = neg["mr"]
            ref_result = ser.compute_ser(gold_mr_dict, neg_mr)

            for method, pred_mr in [("lora", lora_mr), ("nli", nli_mr)]:
                pred_result = ser.compute_ser(pred_mr, neg_mr)
                w, c, t = results[method]
                tally_ser(w, c, t, ref_result, pred_result, neg_label, topic)

    return results


def score_threshold_sweep(data, dev_indices, example_nli_results,
                          nli_threshold, feature_dicts):
    """Sweep top_score thresholds on dev set to find best routing threshold.

    Returns (sweep_results, best_threshold).
    """
    sweep_results = []
    best_all_acc = -1
    best_threshold = None

    for threshold in SCORE_THRESHOLDS:
        decisions = {}
        for ex_idx in dev_indices:
            decisions[ex_idx] = 1 if feature_dicts[ex_idx]["top_score"] >= threshold else 0

        w, c, t = evaluate_routing_on_subset(
            data, dev_indices, example_nli_results, nli_threshold, decisions
        )
        m = compute_metrics(w)

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
    """Train a logistic regression classifier on dev features, evaluate on test.

    Returns (model, scaler, dev_acc, test_preds, importance).
    """
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
    print(f"\n{'=' * 90}")
    print(title)
    print("=" * 90)

    header = f"  {'Method':>12} {'S_acc':>8} {'D_acc':>8} {'I_acc':>8} {'All_acc':>8} {'SER_MAE':>8} {'N':>6}"
    print(header)
    print(f"  {'-' * len(header)}")

    for name, m in method_metrics:
        print(f"  {name:>12} {m['S_acc']:>8.4f} {m['D_acc']:>8.4f} {m['I_acc']:>8.4f} "
              f"{m['all_acc']:>8.4f} {m['SER_MAE']:>8.4f} {m['count']:>6}")


def print_per_topic(all_topic_results, methods, title="PER-TOPIC COMPARISON"):
    """Print per-topic comparison across methods."""
    print(f"\n{'=' * 90}")
    print(title)
    print("=" * 90)

    first_method = methods[0]
    topics = sorted(all_topic_results[first_method].keys())

    for topic in topics:
        print(f"\n  Topic: {topic}")
        print(f"  {'Method':>12} {'S_acc':>8} {'D_acc':>8} {'I_acc':>8} {'All_acc':>8} {'SER_MAE':>8} {'N':>6}")
        print(f"  {'-' * 62}")
        for method in methods:
            tr = all_topic_results[method]
            if topic in tr:
                m = compute_metrics(tr[topic])
                print(f"  {method:>12} {m['S_acc']:>8.4f} {m['D_acc']:>8.4f} {m['I_acc']:>8.4f} "
                      f"{m['all_acc']:>8.4f} {m['SER_MAE']:>8.4f} {m['count']:>6}")


def save_results(output_dir, sweep_results, best_threshold, lr_importance,
                 method_metrics_test, all_topic_results_test, methods,
                 routing_details, lr_dev_acc):
    """Save all output files."""
    os.makedirs(output_dir, exist_ok=True)

    sweep_path = os.path.join(output_dir, "score_routing_sweep.tsv")
    with open(sweep_path, "w") as f:
        f.write("threshold\tn_lora\tn_nli\tcount\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\tSER_MSE\n")
        for r in sweep_results:
            f.write(f"{r['threshold']}\t{r['n_lora']}\t{r['n_nli']}\t{r['count']}\t"
                    f"{r['S_acc']:.4f}\t{r['D_acc']:.4f}\t{r['I_acc']:.4f}\t"
                    f"{r['all_acc']:.4f}\t{r['SER_MAE']:.4f}\t{r['SER_MSE']:.4f}\n")
    print(f"Score routing sweep saved to: {sweep_path}")

    imp_path = os.path.join(output_dir, "lr_feature_importance.tsv")
    with open(imp_path, "w") as f:
        f.write("feature\tcoefficient\n")
        for feat, coef in lr_importance:
            f.write(f"{feat}\t{coef:.4f}\n")
    print(f"LR feature importance saved to: {imp_path}")

    comp_path = os.path.join(output_dir, "comparison_results.tsv")
    with open(comp_path, "w") as f:
        f.write("method\tcount\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\tSER_MSE\n")
        for name, m in method_metrics_test:
            f.write(f"{name}\t{m['count']}\t{m['S_acc']:.4f}\t{m['D_acc']:.4f}\t"
                    f"{m['I_acc']:.4f}\t{m['all_acc']:.4f}\t{m['SER_MAE']:.4f}\t{m['SER_MSE']:.4f}\n")
    print(f"Comparison results saved to: {comp_path}")

    topic_path = os.path.join(output_dir, "per_topic_comparison.tsv")
    with open(topic_path, "w") as f:
        f.write("topic\tmethod\tcount\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\tSER_MSE\n")
        first_method = methods[0]
        for topic in sorted(all_topic_results_test[first_method].keys()):
            for method in methods:
                tr = all_topic_results_test[method]
                if topic in tr:
                    m = compute_metrics(tr[topic])
                    f.write(f"{topic}\t{method}\t{m['count']}\t{m['S_acc']:.4f}\t"
                            f"{m['D_acc']:.4f}\t{m['I_acc']:.4f}\t{m['all_acc']:.4f}\t"
                            f"{m['SER_MAE']:.4f}\t{m['SER_MSE']:.4f}\n")
    print(f"Per-topic comparison saved to: {topic_path}")

    detail_path = os.path.join(output_dir, "routing_details.json")
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
        description="Per-example method routing: score-threshold and logistic regression"
    )
    parser.add_argument("--eval_file", required=True,
                        help="Path to scored evaluation data JSON.")
    parser.add_argument("--output_dir", default="results/routing")
    parser.add_argument("--nli_threshold", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading eval data from: {args.eval_file}")
    with open(args.eval_file) as f:
        data = json.load(f)

    if args.max_examples:
        data = data[:args.max_examples]

    n_pairs = sum(len(ex["negatives"]) for ex in data)
    print(f"Loaded {len(data)} examples with {n_pairs} evaluation pairs.")

    dev_indices, test_indices = stratified_split(data, seed=args.seed)
    dev_pairs = sum(len(data[i]["negatives"]) for i in dev_indices)
    test_pairs = sum(len(data[i]["negatives"]) for i in test_indices)
    print(f"Dev split: {len(dev_indices)} examples, {dev_pairs} pairs")
    print(f"Test split: {len(test_indices)} examples, {test_pairs} pairs")

    all_topics = sorted(set(ex.get("topic", "unknown") for ex in data))
    for split_name, indices in [("Dev", dev_indices), ("Test", test_indices)]:
        topic_counts = defaultdict(int)
        for i in indices:
            topic_counts[data[i].get("topic", "unknown")] += 1
        print(f"  {split_name} topics: {dict(sorted(topic_counts.items()))}")

    nli_model = NLIModel(device=args.device)

    print("\nCollecting NLI pairs...")
    all_pairs, pair_index = collect_all_nli_pairs(data)
    print(f"Total NLI pairs: {len(all_pairs)}")

    print("Running NLI inference (one-time)...")
    all_probs = nli_model.batch_entailment(all_pairs, batch_size=args.batch_size)

    example_nli_results = defaultdict(list)
    for (ex_idx, slot, value), prob in zip(pair_index, all_probs):
        example_nli_results[ex_idx].append((slot, value, prob))

    del nli_model

    print("\nExtracting per-example features and labels...")
    feature_dicts = {}
    labels = {}
    label_details = {}

    for ex_idx, ex in enumerate(data):
        feature_dicts[ex_idx] = extract_features(
            ex, ex_idx, example_nli_results, all_topics
        )
        label, lora_c, nli_c = compute_per_example_label(
            ex, ex_idx, example_nli_results, args.nli_threshold
        )
        labels[ex_idx] = label
        label_details[ex_idx] = {"lora_correct": lora_c, "nli_correct": nli_c}

    sample_features = feature_dicts[0]
    feature_names = [f for f in sample_features if not f.startswith("topic_")]
    feature_names += sorted(f for f in sample_features if f.startswith("topic_"))

    dev_lora_labels = sum(1 for i in dev_indices if labels[i] == 1)
    dev_nli_labels = len(dev_indices) - dev_lora_labels
    test_lora_labels = sum(1 for i in test_indices if labels[i] == 1)
    test_nli_labels = len(test_indices) - test_lora_labels
    print(f"Dev labels: {dev_lora_labels} LoRA-preferred, {dev_nli_labels} NLI-preferred")
    print(f"Test labels: {test_lora_labels} LoRA-preferred, {test_nli_labels} NLI-preferred")

    print("\n" + "=" * 90)
    print("OPTION A: Score-Threshold Routing (sweep on dev)")
    print("=" * 90)

    sweep_results, best_threshold = score_threshold_sweep(
        data, dev_indices, example_nli_results, args.nli_threshold, feature_dicts
    )

    print(f"\n  {'threshold':>10} {'n_lora':>7} {'n_nli':>7} {'All_acc':>8} {'SER_MAE':>8}")
    print(f"  {'-' * 45}")
    for r in sweep_results:
        marker = " <-- best" if r["threshold"] == best_threshold else ""
        print(f"  {r['threshold']:>10.2f} {r['n_lora']:>7} {r['n_nli']:>7} "
              f"{r['all_acc']:>8.4f} {r['SER_MAE']:>8.4f}{marker}")

    print(f"\nBest threshold: {best_threshold} "
          f"(All_acc={[r for r in sweep_results if r['threshold'] == best_threshold][0]['all_acc']:.4f})")

    score_routing_test = {
        idx: (1 if feature_dicts[idx]["top_score"] >= best_threshold else 0)
        for idx in test_indices
    }

    print("\n" + "=" * 90)
    print("OPTION B: Logistic Regression Routing")
    print("=" * 90)

    model, scaler, lr_dev_acc, lr_test_preds, lr_importance = train_logistic_regression(
        feature_dicts, labels, dev_indices, test_indices, feature_names
    )

    print(f"\n  Dev accuracy: {lr_dev_acc:.4f}")
    print("\n  Feature importances (top 15):")
    print(f"  {'feature':>20} {'coefficient':>12}")
    print(f"  {'-' * 35}")
    for feat, coef in lr_importance[:15]:
        print(f"  {feat:>20} {coef:>+12.4f}")

    lr_n_lora = sum(1 for idx in test_indices if lr_test_preds[idx] == 1)
    lr_n_nli = len(test_indices) - lr_n_lora
    print(f"\n  LR test routing: {lr_n_lora} LoRA, {lr_n_nli} NLI")

    print("\n" + "=" * 90)
    print("FOUR-WAY COMPARISON (test split)")
    print("=" * 90)

    baseline_results = evaluate_baselines_on_subset(
        data, test_indices, example_nli_results, args.nli_threshold
    )

    score_w, score_c, score_t = evaluate_routing_on_subset(
        data, test_indices, example_nli_results, args.nli_threshold,
        score_routing_test
    )

    lr_w, lr_c, lr_t = evaluate_routing_on_subset(
        data, test_indices, example_nli_results, args.nli_threshold,
        lr_test_preds
    )

    lora_w, lora_c, lora_t = baseline_results["lora"]
    nli_w, nli_c, nli_t = baseline_results["nli"]

    method_metrics_test = [
        ("LoRA", compute_metrics(lora_w)),
        ("NLI", compute_metrics(nli_w)),
        ("ScoreRouting", compute_metrics(score_w)),
        ("LR-Routing", compute_metrics(lr_w)),
    ]

    print_comparison(method_metrics_test, title="FOUR-WAY COMPARISON (test split)")

    methods = ["LoRA", "NLI", "ScoreRouting", "LR-Routing"]
    all_topic_results_test = {
        "LoRA": lora_t,
        "NLI": nli_t,
        "ScoreRouting": score_t,
        "LR-Routing": lr_t,
    }
    print_per_topic(all_topic_results_test, methods)

    oracle_decisions = {}
    for ex_idx in test_indices:
        oracle_decisions[ex_idx] = labels[ex_idx]

    oracle_w, oracle_c, oracle_t = evaluate_routing_on_subset(
        data, test_indices, example_nli_results, args.nli_threshold,
        oracle_decisions
    )
    oracle_m = compute_metrics(oracle_w)
    print(f"\n  Oracle routing (test): All_acc={oracle_m['all_acc']:.4f}, "
          f"SER_MAE={oracle_m['SER_MAE']:.4f}")

    routing_details = []
    for ex_idx in test_indices:
        ex = data[ex_idx]
        detail = {
            "ex_idx": ex_idx,
            "topic": ex.get("topic", "unknown"),
            "features": feature_dicts[ex_idx],
            "oracle_label": labels[ex_idx],
            "score_routing_choice": score_routing_test[ex_idx],
            "lr_routing_choice": lr_test_preds[ex_idx],
            "lora_correct": label_details[ex_idx]["lora_correct"],
            "nli_correct": label_details[ex_idx]["nli_correct"],
            "n_negatives": len(ex["negatives"]),
        }
        routing_details.append(detail)

    save_results(
        args.output_dir, sweep_results, best_threshold, lr_importance,
        method_metrics_test, all_topic_results_test, methods,
        routing_details, lr_dev_acc
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
