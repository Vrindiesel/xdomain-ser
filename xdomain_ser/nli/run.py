# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""NLI-based SER evaluation runner (single-threshold and sweep modes).

Default mode runs the NLI evaluation at a single threshold and writes a
per-group / per-label / per-topic TSV. With ``--sweep`` it runs NLI inference
once and re-evaluates the threshold-dependent recovery step across a
configurable threshold grid, picking the best threshold by ``All_acc`` and
writing both a sweep summary TSV and detailed results at the best threshold.

Examples::

    # Single-threshold run
    python -m xdomain_ser.nli.run \\
        --eval_file data/ranking_eval/negatives-v6-test.200.pe5.b10.json \\
        --output_dir results/nli \\
        --threshold 0.5

    # Sweep
    python -m xdomain_ser.nli.run --sweep \\
        --eval_file data/ranking_eval/negatives-v6-test.200.pe5.b10.json \\
        --output_dir results/nli

    # Sanity check on a few examples
    python -m xdomain_ser.nli.run \\
        --eval_file data/ranking_eval/negatives-v6-test.200.pe5.b10.json \\
        --max_examples 5 --threshold 0.3
"""
import argparse
import json
import os
from collections import defaultdict

from xdomain_ser.core import ser
from xdomain_ser.nli.evaluator import (
    collect_all_nli_pairs,
    compute_metrics,
    evaluate_with_nli,
    print_results,
    recover_mr_from_nli,
    results_to_tsv,
    tally_ser,
)
from xdomain_ser.nli.model import NLIModel


DEFAULT_SWEEP_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def _run_single(data, nli_model, threshold, batch_size, output_dir):
    """Single-threshold run -- mirrors the original run_evaluation.py main."""
    print(f"\nRunning NLI evaluation with threshold={threshold}...")
    working_results, category_results, topic_results = evaluate_with_nli(
        data, nli_model, threshold=threshold, batch_size=batch_size,
    )

    print_results(working_results, category_results, topic_results)

    os.makedirs(output_dir, exist_ok=True)
    tsv_lines = results_to_tsv(working_results, category_results, topic_results)
    output_path = os.path.join(output_dir, f"nli_results_t{threshold}.tsv")
    with open(output_path, "w") as f:
        f.write("\n".join(tsv_lines) + "\n")
    print(f"\nResults saved to: {output_path}")


def _run_sweep(data, nli_model, thresholds, batch_size, output_dir):
    """Threshold sweep -- mirrors the original threshold_sweep.py main."""
    print("Collecting NLI pairs...")
    all_pairs, pair_index = collect_all_nli_pairs(data)
    print(f"Total NLI pairs: {len(all_pairs)}")

    print("Running NLI inference (one-time)...")
    all_probs = nli_model.batch_entailment(all_pairs, batch_size=batch_size)

    # Group by example
    example_nli_results = defaultdict(list)
    for (ex_idx, slot, value), prob in zip(pair_index, all_probs):
        example_nli_results[ex_idx].append((slot, value, prob))

    # Evaluate at each threshold
    results_by_threshold = {}

    for threshold in thresholds:
        print(f"\nEvaluating at threshold={threshold}...")
        working_results = defaultdict(list)
        category_results = defaultdict(lambda: defaultdict(list))
        topic_results = defaultdict(lambda: defaultdict(list))

        for ex_idx, ex in enumerate(data):
            gold_mr_dict = ser.mr_list_to_dict(ex["mr"])
            nli_results = example_nli_results.get(ex_idx, [])
            nli_mr = recover_mr_from_nli(gold_mr_dict, nli_results, threshold)
            topic = ex.get("topic", "unknown")

            for neg in ex["negatives"]:
                neg_label = neg["label"]
                neg_mr = neg["mr"]
                ref_result = ser.compute_ser(gold_mr_dict, neg_mr)
                pred_result = ser.compute_ser(nli_mr, neg_mr)
                tally_ser(
                    working_results, category_results, topic_results,
                    ref_result, pred_result, neg_label, topic
                )

        results_by_threshold[threshold] = (
            working_results, category_results, topic_results
        )

    # Print summary comparison
    print("\n" + "=" * 80)
    print("THRESHOLD SWEEP SUMMARY")
    print("=" * 80)
    header = f"{'threshold':>10} {'S_acc':>8} {'D_acc':>8} {'I_acc':>8} {'All_acc':>8} {'SER_MAE':>8}"
    print(header)
    print("-" * len(header))

    best_threshold = None
    best_all_acc = -1

    for t in thresholds:
        working, _, _ = results_by_threshold[t]
        m = compute_metrics(working)
        print(f"{t:>10.1f} {m['S_acc']:>8.4f} {m['D_acc']:>8.4f} {m['I_acc']:>8.4f} {m['all_acc']:>8.4f} {m['SER_MAE']:>8.4f}")

        if m["all_acc"] > best_all_acc:
            best_all_acc = m["all_acc"]
            best_threshold = t

    print(f"\nBest threshold by All_acc: {best_threshold} (All_acc={best_all_acc:.4f})")

    # Save sweep summary
    os.makedirs(output_dir, exist_ok=True)
    sweep_path = os.path.join(output_dir, "threshold_sweep.tsv")
    with open(sweep_path, "w") as f:
        f.write("threshold\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\tSER_MSE\n")
        for t in thresholds:
            working, _, _ = results_by_threshold[t]
            m = compute_metrics(working)
            f.write(f"{t}\t{m['S_acc']:.4f}\t{m['D_acc']:.4f}\t{m['I_acc']:.4f}\t{m['all_acc']:.4f}\t{m['SER_MAE']:.4f}\t{m['SER_MSE']:.4f}\n")
    print(f"Sweep summary saved to: {sweep_path}")

    # Save detailed results for best threshold
    working, cat, topic = results_by_threshold[best_threshold]
    tsv_lines = results_to_tsv(working, cat, topic)
    best_path = os.path.join(output_dir, f"nli_results_best_t{best_threshold}.tsv")
    with open(best_path, "w") as f:
        f.write("\n".join(tsv_lines) + "\n")
    print(f"Best threshold detailed results saved to: {best_path}")

    print(f"\n=== Detailed Results at Best Threshold ({best_threshold}) ===")
    print_results(working, cat, topic)


def main():
    parser = argparse.ArgumentParser(
        description="NLI-based SER evaluation (single threshold or sweep)"
    )
    parser.add_argument(
        "--eval_file", required=True,
        help="Path to evaluation data JSON.",
    )
    parser.add_argument(
        "--output_dir", default="results/nli",
        help="Directory for output TSV files.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Entailment probability threshold (single-run mode only).",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep over multiple thresholds and pick the best by All_acc.",
    )
    parser.add_argument(
        "--sweep_thresholds", type=float, nargs="+",
        default=DEFAULT_SWEEP_THRESHOLDS,
        help="Thresholds to sweep (only used with --sweep).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for NLI inference.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device (default: auto-detect cuda/cpu).",
    )
    parser.add_argument(
        "--max_examples", type=int, default=None,
        help="Limit to first N examples (for debugging).",
    )
    parser.add_argument(
        "--dset", default="all",
        help="Subset filter: 'all' (default), 'e2e', 'rnnlg'.",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading eval data from: {args.eval_file}")
    with open(args.eval_file) as f:
        data = json.load(f)

    if args.dset == "e2e":
        data = [e for e in data if e.get("dataset") == "e2e_nlg"]
    elif args.dset == "rnnlg":
        data = [e for e in data if e.get("topic") in {"laptop", "hotel", "tv", "restaurant"}]

    if args.max_examples:
        data = data[:args.max_examples]

    n_pairs = sum(len(ex["negatives"]) for ex in data)
    print(f"Loaded {len(data)} examples with {n_pairs} evaluation pairs.")

    # Load NLI model
    nli_model = NLIModel(device=args.device)

    if args.sweep:
        _run_sweep(data, nli_model, args.sweep_thresholds, args.batch_size, args.output_dir)
    else:
        _run_single(data, nli_model, args.threshold, args.batch_size, args.output_dir)


if __name__ == "__main__":
    main()
