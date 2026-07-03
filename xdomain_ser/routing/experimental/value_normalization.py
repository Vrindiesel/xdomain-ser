# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""[PHASE-0 / POST-PUBLICATION FOLLOW-UP]

Value normalisation evaluation: NLI cross-verification + SBERT soft
matching for routing-feature augmentation.

For each slot where LoRA extraction differs from gold, compute:

1. NLI cross-verification: does the text entail the extracted value?
2. SBERT cosine similarity between extracted and gold slot values

Generates new routing features from these signals
(``mean_cross_nli``, ``min_cross_nli``, ``mean_sbert_sim``,
``min_sbert_sim``, ``n_paraphrase_slots``, ``n_nli_confirmed_slots``)
and re-evaluates routing with the expanded feature set.

Requires ``sentence-transformers`` (not pinned in core; install with
``pip install sentence-transformers``) and the ``experimental`` extra
for the optional XGBoost comparison.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import StandardScaler

from xdomain_ser.core import ser
from xdomain_ser.mixture.threshold_sweep import stratified_split
from xdomain_ser.nli.evaluator import (
    collect_all_nli_pairs,
    compute_metrics,
)
from xdomain_ser.nli.model import NLIModel
from xdomain_ser.nli.templates import slot_value_to_template
from xdomain_ser.ranking.make_eval_data import select_ranking_preds
from xdomain_ser.routing.features import compute_routing_features as extract_features
from xdomain_ser.routing.selector import (
    compute_per_example_label,
    evaluate_baselines_on_subset,
    evaluate_routing_on_subset,
    print_comparison,
    train_logistic_regression,
)


def main():
    parser = argparse.ArgumentParser(
        description="Value normalization: NLI cross-verify + SBERT matching"
    )
    parser.add_argument("--eval_file", required=True,
                        help="Path to scored evaluation data JSON.")
    parser.add_argument("--output_dir", default="results/value_normalization")
    parser.add_argument("--nli_threshold", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--sbert_model", default="all-MiniLM-L6-v2",
                        help="SBERT model for cosine similarity.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading eval data from: {args.eval_file}")
    with open(args.eval_file) as f:
        data = json.load(f)
    if args.max_examples:
        data = data[:args.max_examples]
    print(f"Loaded {len(data)} examples.")

    dev_indices, test_indices = stratified_split(data, seed=args.seed)
    all_topics = sorted(set(ex.get("topic", "unknown") for ex in data))

    nli_model = NLIModel(device=args.device)
    all_pairs, pair_index = collect_all_nli_pairs(data)
    print(f"Running standard NLI inference on {len(all_pairs)} pairs...")
    all_probs = nli_model.batch_entailment(all_pairs, batch_size=args.batch_size)

    example_nli_results = defaultdict(list)
    for (ex_idx, slot, value), prob in zip(pair_index, all_probs):
        example_nli_results[ex_idx].append((slot, value, prob))

    print("\nComputing per-slot LoRA vs Gold comparison...")

    slot_comparisons = []
    per_example_slot_data = {}

    cross_nli_pairs = []
    cross_nli_index = []

    for ex_idx, ex in enumerate(data):
        gold_mr_dict = ser.mr_list_to_dict(ex["mr"])
        lora_mr = select_ranking_preds(ex["pred_mr"], ex["pred_scores"])
        text = ex["surface_form"]
        topic = ex.get("topic", "unknown")

        ex_slots = []
        per_example_slot_data[ex_idx] = ex_slots

        for slot, gold_values in gold_mr_dict.items():
            lora_values = lora_mr.get(slot, [])

            for gv in gold_values:
                if lora_values:
                    lv = lora_values[0]
                    exact_match = (gv.lower().strip() == lv.lower().strip())
                else:
                    lv = None
                    exact_match = False

                slot_data = {
                    "ex_idx": ex_idx,
                    "topic": topic,
                    "slot": slot,
                    "v_gold": gv,
                    "v_extracted": lv,
                    "exact_match": exact_match,
                    "cross_nli_text": None,
                    "cross_nli_value_match": None,
                    "sbert_sim": None,
                }
                ex_slots.append(slot_data)
                slot_comparisons.append(slot_data)

                if lv is not None and not exact_match:
                    template_ext = slot_value_to_template(slot, lv)
                    if template_ext is not None:
                        cross_nli_pairs.append((text, template_ext))
                        cross_nli_index.append(
                            (len(slot_comparisons) - 1, "text_entails_extracted"))

                    template_gold = slot_value_to_template(slot, gv)
                    if template_ext is not None and template_gold is not None:
                        cross_nli_pairs.append((template_ext, template_gold))
                        cross_nli_index.append(
                            (len(slot_comparisons) - 1, "extracted_entails_gold"))

    print(f"Total slot comparisons: {len(slot_comparisons)}")
    print(f"Cross-NLI pairs to evaluate: {len(cross_nli_pairs)}")

    if cross_nli_pairs:
        print("Running NLI cross-verification...")
        cross_probs = nli_model.batch_entailment(
            cross_nli_pairs, batch_size=args.batch_size)

        for (comp_idx, direction), prob in zip(cross_nli_index, cross_probs):
            if direction == "text_entails_extracted":
                slot_comparisons[comp_idx]["cross_nli_text"] = prob
            elif direction == "extracted_entails_gold":
                slot_comparisons[comp_idx]["cross_nli_value_match"] = prob

    del nli_model
    import torch
    torch.cuda.empty_cache()

    print("\nLoading SBERT model for soft matching...")
    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer(args.sbert_model)

    sbert_pairs = []
    sbert_indices = []
    for i, sc in enumerate(slot_comparisons):
        if sc["v_extracted"] is not None and not sc["exact_match"]:
            sbert_pairs.append((sc["v_gold"], sc["v_extracted"]))
            sbert_indices.append(i)

    if sbert_pairs:
        print(f"Computing SBERT similarity for {len(sbert_pairs)} pairs...")
        golds = [p[0] for p in sbert_pairs]
        extracteds = [p[1] for p in sbert_pairs]

        emb_gold = sbert.encode(golds, show_progress_bar=True)
        emb_ext = sbert.encode(extracteds, show_progress_bar=True)

        for idx, (eg, ee) in zip(sbert_indices, zip(emb_gold, emb_ext)):
            cos_sim = float(np.dot(eg, ee) /
                            (np.linalg.norm(eg) * np.linalg.norm(ee)))
            slot_comparisons[idx]["sbert_sim"] = cos_sim

    del sbert
    torch.cuda.empty_cache()

    print("\n" + "=" * 90)
    print("VALUE NORMALIZATION GAP ANALYSIS")
    print("=" * 90)

    n_total = len(slot_comparisons)
    n_exact = sum(1 for sc in slot_comparisons if sc["exact_match"])
    n_missing = sum(1 for sc in slot_comparisons if sc["v_extracted"] is None)
    n_mismatch = n_total - n_exact - n_missing

    n_sbert_para = sum(1 for sc in slot_comparisons
                       if not sc["exact_match"]
                       and sc["sbert_sim"] is not None
                       and sc["sbert_sim"] > 0.85)
    n_nli_confirmed = sum(1 for sc in slot_comparisons
                          if not sc["exact_match"]
                          and sc["cross_nli_text"] is not None
                          and sc["cross_nli_text"] > 0.5)
    n_both_low = sum(1 for sc in slot_comparisons
                     if not sc["exact_match"]
                     and sc["v_extracted"] is not None
                     and (sc["sbert_sim"] is None or sc["sbert_sim"] <= 0.85)
                     and (sc["cross_nli_text"] is None
                          or sc["cross_nli_text"] <= 0.5))

    print(f"  Total slots:         {n_total}")
    print(f"  Exact match:         {n_exact} ({100*n_exact/n_total:.1f}%)")
    print(f"  Missing (deleted):   {n_missing} ({100*n_missing/n_total:.1f}%)")
    print(f"  Mismatch:            {n_mismatch} ({100*n_mismatch/n_total:.1f}%)")
    print("  Of mismatches:")
    if n_mismatch > 0:
        print(f"    SBERT paraphrase (>0.85): {n_sbert_para} "
              f"({100*n_sbert_para/n_mismatch:.1f}%)")
        print(f"    NLI confirmed (>0.5):     {n_nli_confirmed} "
              f"({100*n_nli_confirmed/n_mismatch:.1f}%)")
        print(f"    Both low (genuine error): {n_both_low} "
              f"({100*n_both_low/n_mismatch:.1f}%)")

    domain_stats = defaultdict(lambda: {"total": 0, "exact": 0, "missing": 0,
                                        "mismatch": 0, "sbert_para": 0,
                                        "nli_confirmed": 0})
    for sc in slot_comparisons:
        t = sc["topic"]
        domain_stats[t]["total"] += 1
        if sc["exact_match"]:
            domain_stats[t]["exact"] += 1
        elif sc["v_extracted"] is None:
            domain_stats[t]["missing"] += 1
        else:
            domain_stats[t]["mismatch"] += 1
            if sc["sbert_sim"] is not None and sc["sbert_sim"] > 0.85:
                domain_stats[t]["sbert_para"] += 1
            if (sc["cross_nli_text"] is not None
                    and sc["cross_nli_text"] > 0.5):
                domain_stats[t]["nli_confirmed"] += 1

    print("\nGenerating new routing features...")

    new_features = {}
    for ex_idx in range(len(data)):
        slots = per_example_slot_data.get(ex_idx, [])
        if not slots:
            new_features[ex_idx] = {
                "mean_cross_nli": 0.5,
                "min_cross_nli": 0.5,
                "mean_sbert_sim": 0.5,
                "min_sbert_sim": 0.5,
                "n_paraphrase_slots": 0,
                "n_nli_confirmed_slots": 0,
            }
            continue

        cross_nli_scores = [s["cross_nli_text"] for s in slots
                            if s["cross_nli_text"] is not None]
        sbert_scores = [s["sbert_sim"] for s in slots
                        if s["sbert_sim"] is not None]

        new_features[ex_idx] = {
            "mean_cross_nli": float(np.mean(cross_nli_scores))
                if cross_nli_scores else 0.5,
            "min_cross_nli": float(np.min(cross_nli_scores))
                if cross_nli_scores else 0.5,
            "mean_sbert_sim": float(np.mean(sbert_scores))
                if sbert_scores else 0.5,
            "min_sbert_sim": float(np.min(sbert_scores))
                if sbert_scores else 0.5,
            "n_paraphrase_slots": sum(1 for s in slots
                                      if not s["exact_match"]
                                      and s["sbert_sim"] is not None
                                      and s["sbert_sim"] > 0.85),
            "n_nli_confirmed_slots": sum(1 for s in slots
                                         if s["cross_nli_text"] is not None
                                         and s["cross_nli_text"] > 0.5),
        }

    print("\nRe-running routing with expanded feature set...")

    base_feature_dicts = {}
    labels = {}
    for ex_idx, ex in enumerate(data):
        base_feature_dicts[ex_idx] = extract_features(
            ex, ex_idx, example_nli_results, all_topics
        )
        label, _, _ = compute_per_example_label(
            ex, ex_idx, example_nli_results, args.nli_threshold
        )
        labels[ex_idx] = label

    expanded_feature_dicts = {}
    for ex_idx in range(len(data)):
        expanded_feature_dicts[ex_idx] = {
            **base_feature_dicts[ex_idx],
            **new_features[ex_idx],
        }

    base_sample = base_feature_dicts[0]
    base_feature_names = [f for f in base_sample if not f.startswith("topic_")]
    base_feature_names += sorted(f for f in base_sample if f.startswith("topic_"))

    new_feature_names = list(new_features[0].keys())
    expanded_feature_names = base_feature_names + new_feature_names

    _, _, lr_base_dev_acc, lr_base_preds, _ = train_logistic_regression(
        base_feature_dicts, labels, dev_indices, test_indices, base_feature_names
    )
    lr_base_w, _, _ = evaluate_routing_on_subset(
        data, test_indices, example_nli_results, args.nli_threshold,
        lr_base_preds
    )
    lr_base_m = compute_metrics(lr_base_w)

    _, _, lr_exp_dev_acc, lr_exp_preds, lr_exp_importance = train_logistic_regression(
        expanded_feature_dicts, labels, dev_indices, test_indices,
        expanded_feature_names
    )
    lr_exp_w, _, _ = evaluate_routing_on_subset(
        data, test_indices, example_nli_results, args.nli_threshold,
        lr_exp_preds
    )
    lr_exp_m = compute_metrics(lr_exp_w)

    try:
        import xgboost as xgb

        X_dev_exp = np.array([[expanded_feature_dicts[i][f]
                               for f in expanded_feature_names]
                              for i in dev_indices])
        y_dev = np.array([labels[i] for i in dev_indices])
        X_test_exp = np.array([[expanded_feature_dicts[i][f]
                                for f in expanded_feature_names]
                               for i in test_indices])

        scaler = StandardScaler()
        X_dev_scaled = scaler.fit_transform(X_dev_exp)
        X_test_scaled = scaler.transform(X_test_exp)

        xgb_model = xgb.XGBClassifier(
            max_depth=3, n_estimators=100, learning_rate=0.1,
            reg_lambda=1.0, reg_alpha=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42,
        )
        xgb_model.fit(X_dev_scaled, y_dev)
        xgb_preds_arr = xgb_model.predict(X_test_scaled)
        xgb_preds = {idx: int(p) for idx, p in zip(test_indices, xgb_preds_arr)}
        xgb_w, _, _ = evaluate_routing_on_subset(
            data, test_indices, example_nli_results, args.nli_threshold,
            xgb_preds
        )
        xgb_m = compute_metrics(xgb_w)
        has_xgb = True
    except ImportError:
        has_xgb = False

    baseline_results = evaluate_baselines_on_subset(
        data, test_indices, example_nli_results, args.nli_threshold
    )
    lora_w, _, _ = baseline_results["lora"]
    nli_w, _, _ = baseline_results["nli"]

    method_metrics = [
        ("LoRA", compute_metrics(lora_w)),
        ("NLI", compute_metrics(nli_w)),
        ("LR-Base", lr_base_m),
        ("LR-Expanded", lr_exp_m),
    ]
    if has_xgb:
        method_metrics.append(("XGB-Expanded", xgb_m))

    print_comparison(method_metrics,
                     title="ROUTING WITH NEW FEATURES (test split)")

    print(f"\nSaving results to: {args.output_dir}")

    analysis_path = os.path.join(args.output_dir, "value_norm_analysis.tsv")
    with open(analysis_path, "w") as f:
        f.write("ex_idx\ttopic\tslot\tv_gold\tv_extracted\texact_match\t"
                "sbert_sim\tcross_nli_text\tcross_nli_value_match\n")
        for sc in slot_comparisons:
            sbert_v = f"{sc['sbert_sim']:.4f}" if sc['sbert_sim'] is not None else "N/A"
            cnli_t = f"{sc['cross_nli_text']:.4f}" if sc['cross_nli_text'] is not None else "N/A"
            cnli_v = f"{sc['cross_nli_value_match']:.4f}" if sc['cross_nli_value_match'] is not None else "N/A"
            f.write(f"{sc['ex_idx']}\t{sc['topic']}\t{sc['slot']}\t"
                    f"{sc['v_gold']}\t{sc['v_extracted']}\t"
                    f"{sc['exact_match']}\t{sbert_v}\t{cnli_t}\t{cnli_v}\n")
    print(f"  Value norm analysis: {analysis_path}")

    summary = {
        "total_slots": n_total,
        "exact_match": n_exact,
        "missing": n_missing,
        "mismatch": n_mismatch,
        "sbert_paraphrase": n_sbert_para,
        "nli_confirmed": n_nli_confirmed,
        "both_low": n_both_low,
        "by_domain": {t: dict(s) for t, s in sorted(domain_stats.items())},
    }
    summary_path = os.path.join(args.output_dir, "value_norm_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Value norm summary: {summary_path}")

    feat_path = os.path.join(args.output_dir, "new_routing_features.json")
    with open(feat_path, "w") as f:
        json.dump({str(k): v for k, v in new_features.items()}, f, indent=2)
    print(f"  New routing features: {feat_path}")

    routing_path = os.path.join(args.output_dir,
                                "routing_with_new_features.tsv")
    with open(routing_path, "w") as f:
        f.write("method\tcount\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\tSER_MSE\n")
        for name, m in method_metrics:
            f.write(f"{name}\t{m['count']}\t{m['S_acc']:.4f}\t"
                    f"{m['D_acc']:.4f}\t{m['I_acc']:.4f}\t"
                    f"{m['all_acc']:.4f}\t{m['SER_MAE']:.4f}\t"
                    f"{m['SER_MSE']:.4f}\n")
    print(f"  Routing comparison: {routing_path}")

    imp_path = os.path.join(args.output_dir, "expanded_feature_importance.tsv")
    with open(imp_path, "w") as f:
        f.write("feature\tcoefficient\n")
        for feat, coef in lr_exp_importance:
            f.write(f"{feat}\t{coef:.4f}\n")
    print(f"  Feature importance: {imp_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
