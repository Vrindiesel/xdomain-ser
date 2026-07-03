# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Mixture model combining NLI slot-level verification with LoRA MR extraction.

Approach: start with LoRA's best-ranked candidate MR, then use NLI per-slot
entailment probabilities to correct it:

* Remove slots LoRA extracted but NLI strongly rejects (catches substitutions)
* Add slots LoRA missed but NLI strongly confirms (catches deletions)
* Keep slots where both agree (preserves LoRA's in-domain strength)

Produces a three-way comparison: LoRA vs NLI vs Mixture. See ``threshold_sweep.py``
for the parameter sweep that picks the production ``(t_confirm, t_reject, t_add)``.
"""
import argparse
import json
import os
from collections import defaultdict


from xdomain_ser.core import ser
from xdomain_ser.nli.evaluator import (
    collect_all_nli_pairs,
    compute_metrics,
    recover_mr_from_nli,
    results_to_tsv,
    tally_ser,
)
from xdomain_ser.nli.model import NLIModel
from xdomain_ser.ranking.make_eval_data import select_ranking_preds


def build_nli_prob_dict(nli_results):
    """
    Convert list of (slot, value, prob) into a lookup dict.

    Returns:
        dict {slot: {value: prob}}
    """
    prob_dict = defaultdict(dict)
    for slot, value, prob in nli_results:
        prob_dict[slot][value] = prob
    return prob_dict


def build_mixture_mr(lora_mr, gold_mr_dict, nli_prob_dict,
                     t_confirm=0.3, t_reject=0.15, t_add=0.5):
    """
    Fuse LoRA extraction with NLI verification at the slot level.

    For each slot in gold_mr:
        - If LoRA has it AND NLI prob >= t_reject: INCLUDE (LoRA's value)
        - If LoRA has it AND NLI prob < t_reject: REMOVE (NLI overrides)
        - If LoRA missing AND NLI prob >= t_add: ADD with gold values
        - If LoRA missing AND NLI prob < t_add: EXCLUDE

    Slots LoRA extracted that are NOT in gold_mr (hallucinations) pass through
    unchanged since NLI can only verify gold slots.

    Args:
        lora_mr: dict {slot: [values]} from LoRA best candidate.
        gold_mr_dict: dict {slot: [values]} from gold MR.
        nli_prob_dict: dict {slot: {value: prob}} from NLI inference.
        t_confirm: NLI threshold to confirm LoRA extraction (unused in
            current logic but reserved for future slot-value-level decisions).
        t_reject: NLI threshold below which to remove LoRA's extraction.
        t_add: NLI threshold above which to add a slot LoRA missed.

    Returns:
        dict {slot: [values]} -- the mixture MR.
    """
    mixture = {}

    for slot, gold_values in gold_mr_dict.items():
        # Get the max NLI probability across values for this slot.
        # For dontcare slots (no NLI template), default to 0.5 (neutral).
        slot_probs = nli_prob_dict.get(slot, {})
        if slot_probs:
            max_prob = max(slot_probs.values())
        else:
            max_prob = 0.5

        if slot in lora_mr:
            if max_prob >= t_reject:
                mixture[slot] = lora_mr[slot]
            # else: NLI strongly rejects -- remove
        else:
            if max_prob >= t_add:
                mixture[slot] = gold_values

    # Pass through LoRA hallucinations (slots not in gold_mr)
    for slot, values in lora_mr.items():
        if slot not in gold_mr_dict:
            mixture[slot] = values

    return mixture


def evaluate_three_methods(data, example_nli_results, nli_threshold=0.3,
                           t_confirm=0.3, t_reject=0.15, t_add=0.5):
    """
    Evaluate LoRA, NLI, and Mixture methods on all examples.

    Returns:
        dict with keys 'lora', 'nli', 'mixture', each containing
        (working_results, category_results, topic_results).
        Also returns per-example detailed results list.
    """
    method_results = {}
    for method in ["lora", "nli", "mixture"]:
        method_results[method] = {
            "working": defaultdict(list),
            "category": defaultdict(lambda: defaultdict(list)),
            "topic": defaultdict(lambda: defaultdict(list)),
        }

    detailed = []

    for ex_idx, ex in enumerate(data):
        gold_mr_dict = ser.mr_list_to_dict(ex["mr"])
        topic = ex.get("topic", "unknown")

        lora_mr = select_ranking_preds(ex["pred_mr"], ex["pred_scores"])

        nli_results = example_nli_results.get(ex_idx, [])
        nli_mr = recover_mr_from_nli(gold_mr_dict, nli_results, nli_threshold)

        nli_prob_dict = build_nli_prob_dict(nli_results)
        mixture_mr = build_mixture_mr(
            lora_mr, gold_mr_dict, nli_prob_dict,
            t_confirm=t_confirm, t_reject=t_reject, t_add=t_add
        )

        ex_detail = {
            "ex_idx": ex_idx,
            "topic": topic,
            "gold_mr": dict(gold_mr_dict),
            "lora_mr": dict(lora_mr),
            "nli_mr": dict(nli_mr),
            "mixture_mr": dict(mixture_mr),
            "negatives": [],
        }

        for neg in ex["negatives"]:
            neg_label = neg["label"]
            neg_mr = neg["mr"]

            ref_result = ser.compute_ser(gold_mr_dict, neg_mr)

            methods_pred = {
                "lora": lora_mr,
                "nli": nli_mr,
                "mixture": mixture_mr,
            }

            neg_detail = {"label": neg_label}
            for method, pred_mr in methods_pred.items():
                pred_result = ser.compute_ser(pred_mr, neg_mr)
                tally_ser(
                    method_results[method]["working"],
                    method_results[method]["category"],
                    method_results[method]["topic"],
                    ref_result, pred_result, neg_label, topic
                )
                neg_detail[f"{method}_all_acc"] = (
                    ref_result["S"] == pred_result["S"]
                    and ref_result["D"] == pred_result["D"]
                    and ref_result["I"] == pred_result["I"]
                )

            ex_detail["negatives"].append(neg_detail)

        detailed.append(ex_detail)

    results = {}
    for method in ["lora", "nli", "mixture"]:
        results[method] = (
            method_results[method]["working"],
            method_results[method]["category"],
            method_results[method]["topic"],
        )

    return results, detailed


def print_comparison(results):
    """Print a formatted three-way comparison table."""
    print("\n" + "=" * 90)
    print("THREE-WAY COMPARISON: LoRA vs NLI vs Mixture")
    print("=" * 90)

    header = f"{'Method':>10} {'S_acc':>8} {'D_acc':>8} {'I_acc':>8} {'All_acc':>8} {'SER_MAE':>8} {'SER_MSE':>8} {'N':>6}"
    print(header)
    print("-" * len(header))

    for method in ["lora", "nli", "mixture"]:
        working, _, _ = results[method]
        m = compute_metrics(working)
        print(f"{method:>10} {m['S_acc']:>8.4f} {m['D_acc']:>8.4f} {m['I_acc']:>8.4f} {m['all_acc']:>8.4f} {m['SER_MAE']:>8.4f} {m['SER_MSE']:>8.4f} {m['count']:>6}")

    print("\n" + "=" * 90)
    print("PER-TOPIC COMPARISON")
    print("=" * 90)

    _, _, topic_results_sample = results["lora"]
    topics = sorted(topic_results_sample.keys())

    for topic in topics:
        print(f"\n  Topic: {topic}")
        print(f"  {'Method':>10} {'S_acc':>8} {'D_acc':>8} {'I_acc':>8} {'All_acc':>8} {'SER_MAE':>8} {'N':>6}")
        print(f"  {'-' * 60}")
        for method in ["lora", "nli", "mixture"]:
            _, _, topic_res = results[method]
            if topic in topic_res:
                m = compute_metrics(topic_res[topic])
                print(f"  {method:>10} {m['S_acc']:>8.4f} {m['D_acc']:>8.4f} {m['I_acc']:>8.4f} {m['all_acc']:>8.4f} {m['SER_MAE']:>8.4f} {m['count']:>6}")


def save_results(results, detailed, output_dir, t_confirm, t_reject, t_add,
                 nli_threshold):
    """Save comparison results, per-topic breakdown, and detailed JSON."""
    os.makedirs(output_dir, exist_ok=True)

    comp_path = os.path.join(output_dir, "comparison_results.tsv")
    with open(comp_path, "w") as f:
        f.write("method\tgroup\tname\tcount\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\tSER_MSE\n")
        for method in ["lora", "nli", "mixture"]:
            working, category, topic = results[method]
            tsv_lines = results_to_tsv(working, category, topic)
            for line in tsv_lines[1:]:
                f.write(f"{method}\t{line}\n")
    print(f"Comparison results saved to: {comp_path}")

    topic_path = os.path.join(output_dir, "per_topic_comparison.tsv")
    with open(topic_path, "w") as f:
        f.write("topic\tmethod\tcount\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\tSER_MSE\n")
        _, _, topic_sample = results["lora"]
        for topic in sorted(topic_sample.keys()):
            for method in ["lora", "nli", "mixture"]:
                _, _, topic_res = results[method]
                if topic in topic_res:
                    m = compute_metrics(topic_res[topic])
                    f.write(f"{topic}\t{method}\t{m['count']}\t{m['S_acc']:.4f}\t{m['D_acc']:.4f}\t{m['I_acc']:.4f}\t{m['all_acc']:.4f}\t{m['SER_MAE']:.4f}\t{m['SER_MSE']:.4f}\n")
    print(f"Per-topic comparison saved to: {topic_path}")

    detail_path = os.path.join(output_dir, "detailed_results.json")
    output_obj = {
        "params": {
            "t_confirm": t_confirm,
            "t_reject": t_reject,
            "t_add": t_add,
            "nli_threshold": nli_threshold,
        },
        "summary": {},
        "examples": detailed,
    }
    for method in ["lora", "nli", "mixture"]:
        working, _, _ = results[method]
        output_obj["summary"][method] = compute_metrics(working)
    with open(detail_path, "w") as f:
        json.dump(output_obj, f, indent=2, default=str)
    print(f"Detailed results saved to: {detail_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Mixture model: NLI slot-level verification of LoRA extraction"
    )
    parser.add_argument("--eval_file", required=True,
                        help="Path to scored evaluation data JSON.")
    parser.add_argument("--output_dir", default="results/mixture")
    parser.add_argument("--t_confirm", type=float, default=0.3,
                        help="NLI threshold to confirm LoRA extraction.")
    parser.add_argument("--t_reject", type=float, default=0.15,
                        help="NLI threshold below which to remove LoRA extraction.")
    parser.add_argument("--t_add", type=float, default=0.5,
                        help="NLI threshold above which to add slot LoRA missed.")
    parser.add_argument("--nli_threshold", type=float, default=0.3,
                        help="NLI-only method threshold.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading eval data from: {args.eval_file}")
    with open(args.eval_file) as f:
        data = json.load(f)

    if args.max_examples:
        data = data[:args.max_examples]

    n_pairs = sum(len(ex["negatives"]) for ex in data)
    print(f"Loaded {len(data)} examples with {n_pairs} evaluation pairs.")

    nli_model = NLIModel(device=args.device)

    print("Collecting NLI pairs...")
    all_pairs, pair_index = collect_all_nli_pairs(data)
    print(f"Total NLI pairs: {len(all_pairs)}")

    print("Running NLI inference (one-time)...")
    all_probs = nli_model.batch_entailment(all_pairs, batch_size=args.batch_size)

    example_nli_results = defaultdict(list)
    for (ex_idx, slot, value), prob in zip(pair_index, all_probs):
        example_nli_results[ex_idx].append((slot, value, prob))

    print("\nEvaluating three methods...")
    print(f"  Mixture params: t_confirm={args.t_confirm}, t_reject={args.t_reject}, t_add={args.t_add}")
    print(f"  NLI threshold: {args.nli_threshold}")

    results, detailed = evaluate_three_methods(
        data, example_nli_results,
        nli_threshold=args.nli_threshold,
        t_confirm=args.t_confirm,
        t_reject=args.t_reject,
        t_add=args.t_add,
    )

    print_comparison(results)
    save_results(
        results, detailed, args.output_dir,
        args.t_confirm, args.t_reject, args.t_add, args.nli_threshold
    )


if __name__ == "__main__":
    main()
