# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Threshold sweep for mixture model fusion parameters.

Runs NLI inference once, splits data into stratified dev/test by topic,
sweeps a grid of ``(t_confirm, t_reject, t_add)`` on dev, then evaluates
the best parameters on the held-out test split. Also exports
``stratified_split`` -- the topic-stratified 50/50 dev/test splitter
used across all of Stage 6 (mixture sweep, routing selector, phase-0
XGBoost / value-normalisation, and the experimental DeBERTa NLI swap).
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

from xdomain_ser.core import ser
from xdomain_ser.mixture.fuse import (
    build_mixture_mr,
    build_nli_prob_dict,
)
from xdomain_ser.nli.evaluator import (
    collect_all_nli_pairs,
    compute_metrics,
    recover_mr_from_nli,
    tally_ser,
)
from xdomain_ser.nli.model import NLIModel
from xdomain_ser.ranking.make_eval_data import select_ranking_preds


# Sweep grid
T_CONFIRM_GRID = [0.2, 0.3, 0.4, 0.5]
T_REJECT_GRID = [0.05, 0.10, 0.15, 0.20, 0.25]
T_ADD_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]

NLI_THRESHOLD = 0.3  # Fixed NLI-only threshold (known best from prior sweep)


def stratified_split(data, seed=42, test_frac=0.5):
    """
    Split data into dev/test, stratified by topic.

    Each topic's examples are shuffled and split 50/50 (controllable via
    ``test_frac``).

    Returns:
        dev_indices: list of int (indices into original data)
        test_indices: list of int
    """
    rng = np.random.RandomState(seed)

    topic_indices = defaultdict(list)
    for i, ex in enumerate(data):
        topic = ex.get("topic", "unknown")
        topic_indices[topic].append(i)

    dev_indices = []
    test_indices = []

    for topic in sorted(topic_indices.keys()):
        indices = topic_indices[topic]
        rng.shuffle(indices)
        split_point = int(len(indices) * test_frac)
        dev_indices.extend(indices[:split_point])
        test_indices.extend(indices[split_point:])

    return sorted(dev_indices), sorted(test_indices)


def stratified_pair_split(data, seed=42, test_frac=0.5):
    """Pair-level view of the canonical example-level split.

    Every (example, negative) pair of one text inherits that text's side
    of :func:`stratified_split`, so no text straddles dev and test (the
    split-integrity guard for the corrected per-pair protocol).

    Returns:
        dev_pairs, test_pairs: lists of (ex_idx, neg_idx).
    """
    dev_idx, test_idx = stratified_split(data, seed=seed, test_frac=test_frac)
    dev_pairs = [(i, j) for i in dev_idx
                 for j in range(len(data[i]["negatives"]))]
    test_pairs = [(i, j) for i in test_idx
                  for j in range(len(data[i]["negatives"]))]
    return dev_pairs, test_pairs


def evaluate_mixture_on_subset(data, indices, example_nli_results,
                               nli_threshold, t_confirm, t_reject, t_add):
    """Evaluate the mixture method on a subset of examples (by index)."""
    working_results = defaultdict(list)
    category_results = defaultdict(lambda: defaultdict(list))
    topic_results = defaultdict(lambda: defaultdict(list))

    for ex_idx in indices:
        ex = data[ex_idx]
        gold_mr_dict = ser.mr_list_to_dict(ex["mr"])
        topic = ex.get("topic", "unknown")

        lora_mr = select_ranking_preds(ex["pred_mr"], ex["pred_scores"])

        nli_results = example_nli_results.get(ex_idx, [])
        nli_prob_dict = build_nli_prob_dict(nli_results)

        mixture_mr = build_mixture_mr(
            lora_mr, gold_mr_dict, nli_prob_dict,
            t_confirm=t_confirm, t_reject=t_reject, t_add=t_add
        )

        for neg in ex["negatives"]:
            neg_label = neg["label"]
            neg_mr = neg["mr"]
            ref_result = ser.compute_ser(gold_mr_dict, neg_mr)
            pred_result = ser.compute_ser(mixture_mr, neg_mr)
            tally_ser(
                working_results, category_results, topic_results,
                ref_result, pred_result, neg_label, topic
            )

    return compute_metrics(working_results)


def evaluate_baselines_on_subset(data, indices, example_nli_results, nli_threshold):
    """Evaluate LoRA and NLI baselines on a subset. Returns (lora_metrics, nli_metrics)."""
    lora_working = defaultdict(list)
    lora_cat = defaultdict(lambda: defaultdict(list))
    lora_topic = defaultdict(lambda: defaultdict(list))

    nli_working = defaultdict(list)
    nli_cat = defaultdict(lambda: defaultdict(list))
    nli_topic = defaultdict(lambda: defaultdict(list))

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

            lora_pred = ser.compute_ser(lora_mr, neg_mr)
            tally_ser(lora_working, lora_cat, lora_topic,
                      ref_result, lora_pred, neg_label, topic)

            nli_pred = ser.compute_ser(nli_mr, neg_mr)
            tally_ser(nli_working, nli_cat, nli_topic,
                      ref_result, nli_pred, neg_label, topic)

    return compute_metrics(lora_working), compute_metrics(nli_working)


def main():
    parser = argparse.ArgumentParser(
        description="Threshold sweep for mixture model fusion parameters"
    )
    parser.add_argument("--eval_file", required=True,
                        help="Path to scored evaluation data JSON.")
    parser.add_argument("--output_dir", default="results/mixture")
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

    print("\nComputing baselines on dev split...")
    dev_lora_m, dev_nli_m = evaluate_baselines_on_subset(
        data, dev_indices, example_nli_results, NLI_THRESHOLD
    )
    print(f"  Dev LoRA:  All_acc={dev_lora_m['all_acc']:.4f}, SER_MAE={dev_lora_m['SER_MAE']:.4f}")
    print(f"  Dev NLI:   All_acc={dev_nli_m['all_acc']:.4f}, SER_MAE={dev_nli_m['SER_MAE']:.4f}")

    total_combos = len(T_CONFIRM_GRID) * len(T_REJECT_GRID) * len(T_ADD_GRID)
    print(f"\nSweeping {total_combos} parameter combinations on dev split...")

    sweep_results = []
    best_all_acc = -1
    best_params = None

    for t_confirm in T_CONFIRM_GRID:
        for t_reject in T_REJECT_GRID:
            for t_add in T_ADD_GRID:
                if t_reject >= t_confirm:
                    continue

                m = evaluate_mixture_on_subset(
                    data, dev_indices, example_nli_results,
                    NLI_THRESHOLD, t_confirm, t_reject, t_add
                )

                sweep_results.append({
                    "t_confirm": t_confirm,
                    "t_reject": t_reject,
                    "t_add": t_add,
                    **m,
                })

                if m["all_acc"] > best_all_acc:
                    best_all_acc = m["all_acc"]
                    best_params = (t_confirm, t_reject, t_add)

    print(f"\nSweep complete. {len(sweep_results)} valid combinations evaluated.")
    print(f"Best dev params: t_confirm={best_params[0]}, t_reject={best_params[1]}, t_add={best_params[2]}")
    print(f"Best dev All_acc: {best_all_acc:.4f}")

    sweep_results.sort(key=lambda x: x["all_acc"], reverse=True)
    print("\nTop 10 parameter combinations (dev):")
    print(f"  {'t_confirm':>10} {'t_reject':>10} {'t_add':>10} {'All_acc':>8} {'SER_MAE':>8} {'S_acc':>8} {'D_acc':>8} {'I_acc':>8}")
    print(f"  {'-' * 76}")
    for r in sweep_results[:10]:
        print(f"  {r['t_confirm']:>10.2f} {r['t_reject']:>10.2f} {r['t_add']:>10.2f} {r['all_acc']:>8.4f} {r['SER_MAE']:>8.4f} {r['S_acc']:>8.4f} {r['D_acc']:>8.4f} {r['I_acc']:>8.4f}")

    print(f"\n{'=' * 80}")
    print("EVALUATING BEST PARAMS ON HELD-OUT TEST SPLIT")
    print(f"{'=' * 80}")

    test_lora_m, test_nli_m = evaluate_baselines_on_subset(
        data, test_indices, example_nli_results, NLI_THRESHOLD
    )
    test_mixture_m = evaluate_mixture_on_subset(
        data, test_indices, example_nli_results,
        NLI_THRESHOLD, best_params[0], best_params[1], best_params[2]
    )

    print(f"\n  {'Method':>10} {'All_acc':>8} {'SER_MAE':>8} {'S_acc':>8} {'D_acc':>8} {'I_acc':>8}")
    print(f"  {'-' * 52}")
    print(f"  {'LoRA':>10} {test_lora_m['all_acc']:>8.4f} {test_lora_m['SER_MAE']:>8.4f} {test_lora_m['S_acc']:>8.4f} {test_lora_m['D_acc']:>8.4f} {test_lora_m['I_acc']:>8.4f}")
    print(f"  {'NLI':>10} {test_nli_m['all_acc']:>8.4f} {test_nli_m['SER_MAE']:>8.4f} {test_nli_m['S_acc']:>8.4f} {test_nli_m['D_acc']:>8.4f} {test_nli_m['I_acc']:>8.4f}")
    print(f"  {'Mixture':>10} {test_mixture_m['all_acc']:>8.4f} {test_mixture_m['SER_MAE']:>8.4f} {test_mixture_m['S_acc']:>8.4f} {test_mixture_m['D_acc']:>8.4f} {test_mixture_m['I_acc']:>8.4f}")

    improvement = test_mixture_m["all_acc"] - max(test_lora_m["all_acc"], test_nli_m["all_acc"])
    print(f"\n  Mixture improvement over best baseline: {improvement:+.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    sweep_path = os.path.join(args.output_dir, "threshold_sweep.tsv")
    with open(sweep_path, "w") as f:
        f.write("t_confirm\tt_reject\tt_add\tcount\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\tSER_MSE\n")
        for r in sweep_results:
            f.write(f"{r['t_confirm']}\t{r['t_reject']}\t{r['t_add']}\t{r['count']}\t{r['S_acc']:.4f}\t{r['D_acc']:.4f}\t{r['I_acc']:.4f}\t{r['all_acc']:.4f}\t{r['SER_MAE']:.4f}\t{r['SER_MSE']:.4f}\n")
    print(f"\nSweep results saved to: {sweep_path}")

    test_path = os.path.join(args.output_dir, "test_split_results.tsv")
    with open(test_path, "w") as f:
        f.write("method\tcount\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\tSER_MSE\n")
        for name, m in [("lora", test_lora_m), ("nli", test_nli_m), ("mixture", test_mixture_m)]:
            f.write(f"{name}\t{m['count']}\t{m['S_acc']:.4f}\t{m['D_acc']:.4f}\t{m['I_acc']:.4f}\t{m['all_acc']:.4f}\t{m['SER_MAE']:.4f}\t{m['SER_MSE']:.4f}\n")
    print(f"Test split results saved to: {test_path}")

    params_path = os.path.join(args.output_dir, "best_params.json")
    with open(params_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "best_params": {
                "t_confirm": best_params[0],
                "t_reject": best_params[1],
                "t_add": best_params[2],
            },
            "dev_metrics": {
                "mixture_all_acc": best_all_acc,
                "lora_all_acc": dev_lora_m["all_acc"],
                "nli_all_acc": dev_nli_m["all_acc"],
            },
            "test_metrics": {
                "lora": {k: round(v, 4) if isinstance(v, float) else v for k, v in test_lora_m.items()},
                "nli": {k: round(v, 4) if isinstance(v, float) else v for k, v in test_nli_m.items()},
                "mixture": {k: round(v, 4) if isinstance(v, float) else v for k, v in test_mixture_m.items()},
            },
        }, f, indent=2)
    print(f"Best params saved to: {params_path}")


if __name__ == "__main__":
    main()
