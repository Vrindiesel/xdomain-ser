# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Build the MR-ranker training dataset by sampling graded negative MRs.

For each gold (text, MR) example, samples MRs from the same-domain pool
that hit each target grade on the 5-grade SER rubric (or 7-grade slot-F1
rubric) -- this gives the ranker positive examples at every grade rather
than only at perfect / near-perfect matches.

Output: a JSON file alongside the input with per-example ``negatives``
lists keyed by ``label`` and ``mr``, ready to feed
``xdomain_ser.ranking.train_ranker``.
"""
from typing import Any, Dict, Iterable, List, Tuple
from collections import Counter, defaultdict
import argparse
import json
import os
import random

import tqdm

from xdomain_ser.core import ser

"""
Sampling negatives: create hard negatives by:
 - shuffling slot values, dropping 1-2 slots
 - replacing values with close confounders (e.g., rating good<->average, platforms PC<->PlayStation).
"""


DIGITS = ["0", "1", "2", "3", "4"]

def build_score_prompt(text: str, mr_str: str) -> str:
    """Format the legacy 5-grade scoring prompt for one (text, MR) pair."""
    return (
        "You are scoring how well a meaning representation (MR) matches the text.\n"
        "Return only one digit: 0=bad, 1=weak, 2=okay, 3=good, 4=excellent.\n\n"
        f"Text:\n{text}\n\nMR:\n{mr_str}\n\nScore:\n"
    )


def bin_ser_score(s: float) -> int:
    """Bin SER accuracy (1 - SER) in [0, 1] into the 5-grade rubric labels 0-4."""
    if s >= 0.90: return 4
    if s >= 0.75: return 3
    if s >= 0.50: return 2
    if s >= 0.25: return 1
    return 0


def bin_f1_score(s: float) -> int:
    """Bin a slot-F1 score in [0, 1] into the 7-grade rubric labels 0-6."""
    if s >= 0.90: return 6
    if s >= 0.80: return 5
    if s >= 0.70: return 4
    if s >= 0.60: return 3
    if s >= 0.50: return 2
    if s >= 0.25: return 1
    return 0


def fetch_ser_label_mr(ref_mr: Dict[str, Any], targ_labels: Iterable[int],
                       mr_pool: List[Dict[str, Any]],
                       max_tries: int = 1_000) -> List[Tuple[int, Dict[str, Any]]]:
    """Sample negative MRs from ``mr_pool`` hitting each target SER grade.

    Shuffles the pool in place and scans until every target label is found
    once or ``max_tries`` candidates were tried. Returns (label, mr) pairs.
    """
    random.shuffle(mr_pool)
    found_mrs = []
    found_labels = set()
    targ_labels = set(targ_labels)
    for j, mr in enumerate(mr_pool):
        ser_score = ser.compute_ser(mr, ref_mr)
        s = 1. - ser_score["SER"]
        bs = bin_ser_score(s)
        if bs in targ_labels and bs not in found_labels:
            found_mrs.append((bs, mr))
            found_labels.add(bs)
        if len(found_mrs) == len(targ_labels) or j >= max_tries:
            break
    return found_mrs


def fetch_f1_label_mr(ref_mr: Dict[str, Any], targ_labels: Iterable[int],
                      mr_pool: List[Dict[str, Any]],
                      max_tries: int = 1_000) -> List[Tuple[int, Dict[str, Any]]]:
    """Sample negative MRs from ``mr_pool`` hitting each target slot-F1 grade.

    Same contract as ``fetch_ser_label_mr`` with the 7-grade F1 rubric.
    """
    random.shuffle(mr_pool)
    found_mrs = []
    found_labels = set()
    targ_labels = set(targ_labels)
    for j, mr in enumerate(mr_pool):
        f1_score = ser.compute_slot_f1(mr, ref_mr)
        s = f1_score["f1"]
        bs = bin_f1_score(s)
        if bs in targ_labels and bs not in found_labels:
            found_mrs.append((bs, mr))
            found_labels.add(bs)
        if len(found_mrs) == len(targ_labels) or j >= max_tries:
            break
    return found_mrs


def mr_list_2_mr_dict(mr_list: Any) -> Dict[str, List[str]]:
    """Convert a (slot, value) pair list to ``{slot: [values]}``; dicts pass through."""
    mr_dict = mr_list
    if isinstance(mr_list, list):
        mr_dict = defaultdict(list)
        for (a, v) in mr_list:
            mr_dict[a].append(v)
    return mr_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--hintmap_path", type=str, default=None)
    parser.add_argument("--random_seed", type=int, default=232323)
    parser.add_argument("--min_mr_size", type=int, default=2)
    parser.add_argument("--max_tries", type=int, default=1_000)
    parser.add_argument("--method", type=str, default="ser", choices=["ser", "f1"])
    args = parser.parse_args()
    print("args:", args)
    random.seed(args.random_seed)

    with open(args.input_path) as fin:
        data = json.load(fin)
    if isinstance(data, dict):
        all_examples = []
        for topic, examples in data.items():
            all_examples.extend(examples)
    else:
        all_examples = data

    topic_slot_values = defaultdict(lambda: defaultdict(list))
    topic_mrs = defaultdict(list)
    for example in all_examples:
        gold_mr = example["mr"]
        if "slots" in gold_mr:
            gold_mr = gold_mr["slots"]
        gold_mr = mr_list_2_mr_dict(gold_mr)
        topic_mrs[example["hint_map_id"]].append(gold_mr)
        if example.get("pred_mr"):
            for mr in example["pred_mr"]:
                topic_mrs[example["hint_map_id"]].append(ser.extract_attributes_dict(mr))

        for slot_name, slot_values in gold_mr.items():
            if slot_values is None:
                continue
            elif isinstance(slot_values, str):
                slot_values = [slot_values]
            topic_slot_values[example["hint_map_id"]][slot_name].extend(slot_values)

    for topic in topic_slot_values:
        for slot_name, slot_vals in topic_slot_values[topic].items():
            topic_slot_values[topic][slot_name] = Counter(slot_vals)

    if not os.path.exists(args.output_path):
        print(f"Creating {args.output_path} ...")
        os.makedirs(args.output_path)

    fname = os.path.basename(args.input_path)
    fname = f"negatives-{fname}".replace(".json", "")

    with open(os.path.join(args.output_path, f"{fname}-slot-values.json"), "w") as fout:
        json.dump(topic_slot_values, fout, indent=2)


    if args.method == "f1":
        label_set = [0, 1, 2, 3, 4, 5, 6]
    elif args.method == "ser":
        label_set = [0, 1, 2, 3, 4]
        etype_set = ["S", "D", "I"]

    max_tries = args.max_tries
    topic_error_counts = defaultdict(Counter)
    topic_neg_counts = defaultdict(Counter)
    for ex in tqdm.tqdm(all_examples):
        gold_mr = ex["mr"]
        if "slots" in gold_mr:
            gold_mr = gold_mr["slots"]
        gold_mr = mr_list_2_mr_dict(gold_mr)
        if len(gold_mr) < args.min_mr_size:
            continue

        topic = ex["hint_map_id"]
        if args.method == "f1":
            neg_mrs = fetch_f1_label_mr(gold_mr, label_set, topic_mrs[topic], max_tries=max_tries)
        elif args.method == "ser":
            neg_mrs = fetch_ser_label_mr(gold_mr, label_set, topic_mrs[topic], max_tries=max_tries)

        if len(neg_mrs) < 1:
            continue
        for label, neg_mr in neg_mrs:

            topic_error_counts[topic][f"{label}_label"] += 1
            topic_error_counts[topic]["N"] += 1
            if "negatives" not in ex:
                ex["negatives"] = []

            e = {
                "label": label,
                "mr": neg_mr,
            }

            if args.method == "ser":
                ser_vals = ser.compute_ser(neg_mr, gold_mr)
                for etype in etype_set:
                    topic_error_counts[topic][etype] += ser_vals[etype]
                e["ser_vals"] = ser_vals

            ex["negatives"].append(e)
            topic_neg_counts[topic][f"{len(ex['negatives'])}_negatives"] += 1


    with open(os.path.join(args.output_path, fname + ".json"), "w") as fout:
        json.dump(all_examples, fout, indent=2)

    with open(os.path.join(args.output_path, f"{fname}-error-counts.json"), "w") as fout:
        json.dump(topic_error_counts, fout, indent=2)
    with open(os.path.join(args.output_path, f"{fname}-negative-counts.json"), "w") as fout:
        json.dump(topic_neg_counts, fout, indent=2)

    counts = Counter()
    for k, v in topic_error_counts.items():
        counts.update(Counter(v))

    print("Total Counts:")
    for k, v in counts.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()
