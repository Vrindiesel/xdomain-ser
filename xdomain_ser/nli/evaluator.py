# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""NLI-based semantic accuracy evaluator.

Core logic: for each ``(text, gold_mr)``, run NLI on all gold slot-value
pairs to build a recovered MR dict. Then use ``ser.compute_ser(nli_mr,
neg_mr)`` to compute S/D/I error counts and compare them against the
ground-truth error counts from ``ser.compute_ser(gold_mr, neg_mr)``.

Follows the ``tally_ser()`` + ``print_results()`` pattern from
``xdomain_ser.ranking.make_eval_data``.
"""
from collections import defaultdict

import numpy as np

from xdomain_ser.core import ser
from xdomain_ser.nli.templates import slot_value_to_template


def build_nli_pairs_for_example(text, gold_mr_dict):
    """
    Build all (premise, hypothesis) NLI pairs needed to check whether each
    gold MR slot-value is realized in the text.

    Args:
        text: the generated utterance.
        gold_mr_dict: dict {slot: [values]} from ser.mr_list_to_dict().

    Returns:
        list of (premise, hypothesis, slot, value) tuples.
        Entries where the template is None (dontcare) are excluded.
    """
    pairs = []
    for slot, values in gold_mr_dict.items():
        for value in values:
            template = slot_value_to_template(slot, value)
            if template is None:
                continue
            pairs.append((text, template, slot, value))
    return pairs


def recover_mr_from_nli(gold_mr_dict, nli_results, threshold=0.5):
    """
    Build a recovered MR dict based on NLI entailment results.

    For each slot in the gold MR, if the entailment probability exceeds the
    threshold, the slot-value is considered realized in the text and included
    in the recovered MR.

    Args:
        gold_mr_dict: dict {slot: [values]} -- the gold MR.
        nli_results: list of (slot, value, entailment_prob) tuples.
        threshold: entailment probability threshold.

    Returns:
        dict {slot: [values]} -- the NLI-recovered MR.
    """
    recovered = defaultdict(list)
    for slot, value, prob in nli_results:
        if prob > threshold:
            recovered[slot].append(value)
    return dict(recovered)


def collect_all_nli_pairs(data):
    """
    Collect all NLI (premise, hypothesis) pairs across the entire dataset,
    with index tracking for mapping results back.

    Args:
        data: list of examples from the eval JSON.

    Returns:
        all_pairs: list of (premise, hypothesis) for batch inference.
        pair_index: list of (example_idx, slot, value) for mapping back.
    """
    all_pairs = []
    pair_index = []

    for ex_idx, ex in enumerate(data):
        text = ex["surface_form"]
        gold_mr_dict = ser.mr_list_to_dict(ex["mr"])

        for slot, values in gold_mr_dict.items():
            for value in values:
                template = slot_value_to_template(slot, value)
                if template is None:
                    continue
                all_pairs.append((text, template))
                pair_index.append((ex_idx, slot, value))

    return all_pairs, pair_index


def collect_pair_nli_pairs(data):
    """Per-pair NLI premise/hypothesis collection (corrected protocol).

    Hypothesis templates come from each negative's modified MR slot-values,
    not the example's true MR -- the deployed verifier only sees the MR it
    is checking. Note the structural consequence: with the hypothesis MR as
    the only inventory, NLI cannot detect deletions (text content absent
    from the checked MR); per-error-type tables must surface this.

    Args:
        data: list of examples from the eval JSON (with ``negatives``).

    Returns:
        all_pairs: list of (premise, hypothesis) for batch inference.
        pair_index: list of (ex_idx, neg_idx, slot, value) for mapping back.
    """
    all_pairs = []
    pair_index = []

    for ex_idx, ex in enumerate(data):
        text = ex["surface_form"]
        for neg_idx, neg in enumerate(ex.get("negatives", [])):
            for slot, values in neg["mr"].items():
                if not isinstance(values, list):
                    values = [values]
                for value in values:
                    template = slot_value_to_template(slot, value)
                    if template is None:
                        continue
                    all_pairs.append((text, template))
                    pair_index.append((ex_idx, neg_idx, slot, value))

    return all_pairs, pair_index


def group_pair_nli_results(pair_index, probs):
    """Map batched NLI probs back to per-pair result lists.

    Returns dict ``{(ex_idx, neg_idx): [(slot, value, prob), ...]}`` --
    each list is consumable by :func:`recover_mr_from_nli` unchanged (the
    recovered MR is then the entailed subset of the pair's modified MR).
    """
    results = defaultdict(list)
    for (ex_idx, neg_idx, slot, value), prob in zip(pair_index, probs):
        results[(ex_idx, neg_idx)].append((slot, value, prob))
    return dict(results)


def evaluate_with_nli(data, nli_model, threshold=0.5, batch_size=32):
    """
    Run the full NLI evaluation pipeline.

    For each example:
    1. Use NLI to recover which gold MR slots are realized in the text.
    2. For each negative MR, compute SER(nli_mr, neg_mr) and compare to
       ground-truth SER(gold_mr, neg_mr).

    Args:
        data: list of examples from the eval JSON.
        nli_model: NLIModel instance.
        threshold: entailment probability threshold.
        batch_size: batch size for NLI inference.

    Returns:
        working_results: dict of metric lists (S_acc, D_acc, I_acc, etc.)
        category_results: dict[label] -> dict of metric lists
        topic_results: dict[topic] -> dict of metric lists
    """
    # Step 1: Collect all NLI pairs for batch inference
    print("Collecting NLI pairs...")
    all_pairs, pair_index = collect_all_nli_pairs(data)
    print(f"Total NLI pairs to evaluate: {len(all_pairs)}")

    # Step 2: Run batched NLI inference
    all_probs = nli_model.batch_entailment(all_pairs, batch_size=batch_size)

    # Step 3: Group results by example
    example_nli_results = defaultdict(list)  # ex_idx -> [(slot, value, prob)]
    for (ex_idx, slot, value), prob in zip(pair_index, all_probs):
        example_nli_results[ex_idx].append((slot, value, prob))

    # Step 4: Evaluate each example against its negatives
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

            # Compute SER for gold MR vs negative (ground truth)
            ref_result = ser.compute_ser(gold_mr_dict, neg_mr)
            # Compute SER for NLI-recovered MR vs negative (system)
            pred_result = ser.compute_ser(nli_mr, neg_mr)

            tally_ser(
                working_results, category_results, topic_results,
                ref_result, pred_result, neg_label, topic
            )

    return working_results, category_results, topic_results


def tally_ser(working_results, category_results, topic_results,
              ref_result, pred_result, neg_label, topic):
    """
    Record agreement between ref (gold) and pred (NLI) SER results.

    Follows the pattern from xdomain_ser.ranking.make_eval_data.
    """
    s_acc = ref_result["S"] == pred_result["S"]
    d_acc = ref_result["D"] == pred_result["D"]
    i_acc = ref_result["I"] == pred_result["I"]
    all_acc = s_acc and d_acc and i_acc
    ser_error = ref_result["SER"] - pred_result["SER"]

    for results_dict in [working_results,
                         category_results[neg_label],
                         topic_results[topic]]:
        results_dict["S_acc"].append(s_acc)
        results_dict["D_acc"].append(d_acc)
        results_dict["I_acc"].append(i_acc)
        results_dict["all_acc"].append(all_acc)
        results_dict["ser_error"].append(ser_error)


def compute_metrics(results_dict):
    """
    Compute aggregate metrics from a results dict.

    Returns dict with S_acc, D_acc, I_acc, all_acc, SER_MAE, SER_MSE, count.
    """
    n = len(results_dict["S_acc"])
    if n == 0:
        return {"S_acc": 0, "D_acc": 0, "I_acc": 0, "all_acc": 0,
                "SER_MAE": 0, "SER_MSE": 0, "count": 0}

    return {
        "S_acc": np.mean(results_dict["S_acc"]),
        "D_acc": np.mean(results_dict["D_acc"]),
        "I_acc": np.mean(results_dict["I_acc"]),
        "all_acc": np.mean(results_dict["all_acc"]),
        "SER_MAE": np.mean([abs(e) for e in results_dict["ser_error"]]),
        "SER_MSE": np.mean([e ** 2 for e in results_dict["ser_error"]]),
        "count": n,
    }


def print_results(working_results, category_results, topic_results):
    """Print formatted results matching the existing codebase output style."""
    print("\n=== Overall Results ===")
    metrics = compute_metrics(working_results)
    _print_metrics(metrics)

    print("\n=== Results by Difficulty Label ===")
    for label in sorted(category_results.keys()):
        metrics = compute_metrics(category_results[label])
        print(f"\nLabel {label} (n={metrics['count']}):")
        _print_metrics(metrics)

    print("\n=== Results by Topic ===")
    for topic in sorted(topic_results.keys()):
        metrics = compute_metrics(topic_results[topic])
        print(f"\nTopic: {topic} (n={metrics['count']}):")
        _print_metrics(metrics)


def _print_metrics(metrics):
    """Print a single set of metrics."""
    print(f"  S_acc:    {metrics['S_acc']:.4f}")
    print(f"  D_acc:    {metrics['D_acc']:.4f}")
    print(f"  I_acc:    {metrics['I_acc']:.4f}")
    print(f"  All_acc:  {metrics['all_acc']:.4f}")
    print(f"  SER MAE:  {metrics['SER_MAE']:.4f}")
    print(f"  SER MSE:  {metrics['SER_MSE']:.4f}")


def results_to_tsv(working_results, category_results, topic_results):
    """
    Format results as TSV lines for output.

    Returns list of TSV-formatted lines (with header).
    """
    header = "group\tname\tcount\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\tSER_MSE"
    lines = [header]

    # Overall
    m = compute_metrics(working_results)
    lines.append(f"overall\tall\t{m['count']}\t{m['S_acc']:.4f}\t{m['D_acc']:.4f}\t{m['I_acc']:.4f}\t{m['all_acc']:.4f}\t{m['SER_MAE']:.4f}\t{m['SER_MSE']:.4f}")

    # By label
    for label in sorted(category_results.keys()):
        m = compute_metrics(category_results[label])
        lines.append(f"label\t{label}\t{m['count']}\t{m['S_acc']:.4f}\t{m['D_acc']:.4f}\t{m['I_acc']:.4f}\t{m['all_acc']:.4f}\t{m['SER_MAE']:.4f}\t{m['SER_MSE']:.4f}")

    # By topic
    for topic in sorted(topic_results.keys()):
        m = compute_metrics(topic_results[topic])
        lines.append(f"topic\t{topic}\t{m['count']}\t{m['S_acc']:.4f}\t{m['D_acc']:.4f}\t{m['I_acc']:.4f}\t{m['all_acc']:.4f}\t{m['SER_MAE']:.4f}\t{m['SER_MSE']:.4f}")

    return lines
