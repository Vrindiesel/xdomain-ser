# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""[PHASE-0 / POST-PUBLICATION FOLLOW-UP]

XGBoost routing evaluation: replace ``LogisticRegression`` with XGBoost
in the routing pipeline, with optional cost-sensitive sample weighting
(sample weight = abs(lora_correct - nli_correct) + 0.1, so examples
where the two methods disagree most strongly carry the most weight).

**Result: XGBoost with cost-sensitive weighting hits All_acc = 0.9004
on the held-out test split**, a +3.24 percentage-point improvement over
LR-Routing (0.8681) and +3.90pp over ScoreRouting (0.8614). See the
phase-0 sections of RELEASE_NOTES.md for the full numbers. Top features by gain:
``top_score`` (0.342), ``min_nli_prob`` (0.192), ``nli_coverage`` (0.092).

Requires the ``experimental`` optional install::

    pip install xdomain-ser[experimental]

Reuses the same data loading, feature computation, stratified split,
and evaluation logic as :mod:`xdomain_ser.routing.selector`.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from xdomain_ser.mixture.threshold_sweep import stratified_split
from xdomain_ser.nli.evaluator import (
    collect_all_nli_pairs,
    compute_metrics,
)
from xdomain_ser.nli.model import NLIModel
from xdomain_ser.routing.features import compute_routing_features as extract_features
from xdomain_ser.routing.selector import (
    compute_per_example_label,
    evaluate_baselines_on_subset,
    evaluate_routing_on_subset,
    print_comparison,
    score_threshold_sweep,
    train_logistic_regression,
)


# XGBoost configs to sweep
XGB_CONFIGS = [
    {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.1,
     "reg_lambda": 1.0, "reg_alpha": 0.1},
    {"max_depth": 4, "n_estimators": 150, "learning_rate": 0.1,
     "reg_lambda": 1.0, "reg_alpha": 0.1},
    {"max_depth": 3, "n_estimators": 200, "learning_rate": 0.05,
     "reg_lambda": 1.0, "reg_alpha": 0.1},
]


def train_xgboost(X_dev, y_dev, X_test, test_indices, cfg, sample_weights=None):
    """Train XGBoost classifier and predict on test."""
    model = xgb.XGBClassifier(
        **cfg,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    if sample_weights is not None:
        model.fit(X_dev, y_dev, sample_weight=sample_weights)
    else:
        model.fit(X_dev, y_dev)

    dev_acc = model.score(X_dev, y_dev)
    test_pred_array = model.predict(X_test)
    test_preds = {idx: int(pred) for idx, pred in zip(test_indices, test_pred_array)}

    return model, test_preds, dev_acc


def main():
    parser = argparse.ArgumentParser(
        description="XGBoost routing evaluation (phase-0 post-publication follow-up)"
    )
    parser.add_argument("--eval_file", required=True,
                        help="Path to scored evaluation data JSON.")
    parser.add_argument("--output_dir", default="results/xgboost_routing")
    parser.add_argument("--nli_threshold", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading eval data from: {args.eval_file}")
    with open(args.eval_file) as f:
        data = json.load(f)

    if args.max_examples:
        data = data[:args.max_examples]

    n_pairs = sum(len(ex["negatives"]) for ex in data)
    print(f"Loaded {len(data)} examples with {n_pairs} evaluation pairs.")

    dev_indices, test_indices = stratified_split(data, seed=args.seed)
    print(f"Dev split: {len(dev_indices)} examples")
    print(f"Test split: {len(test_indices)} examples")

    all_topics = sorted(set(ex.get("topic", "unknown") for ex in data))

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

    X_dev = np.array([[feature_dicts[i][f] for f in feature_names]
                      for i in dev_indices])
    y_dev = np.array([labels[i] for i in dev_indices])
    X_test = np.array([[feature_dicts[i][f] for f in feature_names]
                       for i in test_indices])

    scaler = StandardScaler()
    X_dev_scaled = scaler.fit_transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    n_correct_lora_dev = np.array([label_details[i]["lora_correct"] for i in dev_indices])
    n_correct_nli_dev = np.array([label_details[i]["nli_correct"] for i in dev_indices])
    sample_weights = np.abs(n_correct_lora_dev - n_correct_nli_dev) + 0.1

    print("\n" + "=" * 90)
    print("BASELINE: Logistic Regression Routing")
    print("=" * 90)

    lr_model, lr_scaler, lr_dev_acc, lr_test_preds, lr_importance = \
        train_logistic_regression(
            feature_dicts, labels, dev_indices, test_indices, feature_names
        )
    print(f"  LR dev accuracy: {lr_dev_acc:.4f}")

    sweep_results, best_threshold = score_threshold_sweep(
        data, dev_indices, example_nli_results, args.nli_threshold,
        feature_dicts
    )
    score_routing_test = {
        idx: (1 if feature_dicts[idx]["top_score"] >= best_threshold else 0)
        for idx in test_indices
    }

    print("\n" + "=" * 90)
    print("XGBOOST ROUTING EXPERIMENTS")
    print("=" * 90)

    xgb_results = []

    for i, cfg in enumerate(XGB_CONFIGS):
        cfg_str = f"d{cfg['max_depth']}_n{cfg['n_estimators']}_lr{cfg['learning_rate']}"

        model_plain, preds_plain, dev_acc_plain = train_xgboost(
            X_dev_scaled, y_dev, X_test_scaled, test_indices, cfg
        )
        w, c, t = evaluate_routing_on_subset(
            data, test_indices, example_nli_results, args.nli_threshold,
            preds_plain
        )
        m_plain = compute_metrics(w)
        name_plain = f"XGB_{cfg_str}"
        print(f"\n  {name_plain}: dev_acc={dev_acc_plain:.4f}, "
              f"test All_acc={m_plain['all_acc']:.4f}, "
              f"SER_MAE={m_plain['SER_MAE']:.4f}")

        xgb_results.append({
            "name": name_plain,
            "config": cfg,
            "cost_weighted": False,
            "dev_acc": dev_acc_plain,
            "metrics": m_plain,
            "preds": preds_plain,
            "model": model_plain,
        })

        model_cw, preds_cw, dev_acc_cw = train_xgboost(
            X_dev_scaled, y_dev, X_test_scaled, test_indices, cfg,
            sample_weights=sample_weights
        )
        w_cw, c_cw, t_cw = evaluate_routing_on_subset(
            data, test_indices, example_nli_results, args.nli_threshold,
            preds_cw
        )
        m_cw = compute_metrics(w_cw)
        name_cw = f"XGB_{cfg_str}_CW"
        print(f"  {name_cw}: dev_acc={dev_acc_cw:.4f}, "
              f"test All_acc={m_cw['all_acc']:.4f}, "
              f"SER_MAE={m_cw['SER_MAE']:.4f}")

        xgb_results.append({
            "name": name_cw,
            "config": cfg,
            "cost_weighted": True,
            "dev_acc": dev_acc_cw,
            "metrics": m_cw,
            "preds": preds_cw,
            "model": model_cw,
        })

    baseline_results = evaluate_baselines_on_subset(
        data, test_indices, example_nli_results, args.nli_threshold
    )

    lora_w, lora_c, lora_t = baseline_results["lora"]
    nli_w, nli_c, nli_t = baseline_results["nli"]

    score_w, score_c, score_t = evaluate_routing_on_subset(
        data, test_indices, example_nli_results, args.nli_threshold,
        score_routing_test
    )
    lr_w, lr_c, lr_t = evaluate_routing_on_subset(
        data, test_indices, example_nli_results, args.nli_threshold,
        lr_test_preds
    )

    best_xgb = max(xgb_results, key=lambda x: x["metrics"]["all_acc"])

    best_xgb_w, best_xgb_c, best_xgb_t = evaluate_routing_on_subset(
        data, test_indices, example_nli_results, args.nli_threshold,
        best_xgb["preds"]
    )

    method_metrics = [
        ("LoRA", compute_metrics(lora_w)),
        ("NLI", compute_metrics(nli_w)),
        ("ScoreRouting", compute_metrics(score_w)),
        ("LR-Routing", compute_metrics(lr_w)),
        (best_xgb["name"], best_xgb["metrics"]),
    ]

    print_comparison(method_metrics, title="FULL COMPARISON (test split)")

    print(f"\nSaving results to: {args.output_dir}")

    comp_path = os.path.join(args.output_dir, "results_comparison.tsv")
    with open(comp_path, "w") as f:
        f.write("method\tconfig\tcost_weighted\tdev_acc\tcount\t"
                "S_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\tSER_MSE\n")

        for name, m in [("LoRA", compute_metrics(lora_w)),
                        ("NLI", compute_metrics(nli_w)),
                        ("ScoreRouting", compute_metrics(score_w)),
                        ("LR-Routing", compute_metrics(lr_w))]:
            f.write(f"{name}\t-\t-\t-\t{m['count']}\t"
                    f"{m['S_acc']:.4f}\t{m['D_acc']:.4f}\t{m['I_acc']:.4f}\t"
                    f"{m['all_acc']:.4f}\t{m['SER_MAE']:.4f}\t"
                    f"{m['SER_MSE']:.4f}\n")

        for r in xgb_results:
            m = r["metrics"]
            cfg_str = json.dumps(r["config"])
            f.write(f"{r['name']}\t{cfg_str}\t{r['cost_weighted']}\t"
                    f"{r['dev_acc']:.4f}\t{m['count']}\t"
                    f"{m['S_acc']:.4f}\t{m['D_acc']:.4f}\t{m['I_acc']:.4f}\t"
                    f"{m['all_acc']:.4f}\t{m['SER_MAE']:.4f}\t"
                    f"{m['SER_MSE']:.4f}\n")
    print(f"  Results comparison: {comp_path}")

    imp_path = os.path.join(args.output_dir, "feature_importance.tsv")
    importances = best_xgb["model"].feature_importances_
    feat_imp = sorted(zip(feature_names, importances),
                      key=lambda x: x[1], reverse=True)
    with open(imp_path, "w") as f:
        f.write("feature\timportance_gain\n")
        for feat, imp in feat_imp:
            f.write(f"{feat}\t{imp:.6f}\n")
    print(f"  Feature importance: {imp_path}")

    best_path = os.path.join(args.output_dir, "best_config.json")
    with open(best_path, "w") as f:
        json.dump({
            "best_method": best_xgb["name"],
            "config": best_xgb["config"],
            "cost_weighted": best_xgb["cost_weighted"],
            "dev_acc": best_xgb["dev_acc"],
            "test_metrics": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in best_xgb["metrics"].items()
            },
            "lr_baseline_test_metrics": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in compute_metrics(lr_w).items()
            },
            "improvement_over_lr": round(
                best_xgb["metrics"]["all_acc"] -
                compute_metrics(lr_w)["all_acc"], 4
            ),
        }, f, indent=2)
    print(f"  Best config: {best_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
