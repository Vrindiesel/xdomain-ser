# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Augment gold-annotated examples with negative MRs at labels 0-4.

Reuses logic from :mod:`xdomain_ser.ranking.make_dataset`:
``fetch_ser_label_mr()``, ``bin_ser_score()``, ``ser.compute_ser()``.

Input:  ``evaluation/gold/gold-annotated.json``
Output: ``evaluation/gold/eval-negatives.json``
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict

import tqdm

from xdomain_ser.core import ser


LABEL_SET = [0, 1, 2, 3, 4]


def bin_ser_score(s):
    """Bin SER accuracy (1 - SER) into labels 0-4."""
    if s >= 0.90:
        return 4
    if s >= 0.75:
        return 3
    if s >= 0.50:
        return 2
    if s >= 0.25:
        return 1
    return 0


def fetch_ser_label_mr(ref_mr, targ_labels, mr_pool, max_tries=1000):
    """Find MRs from pool matching target SER label bins.

    Returns [(label, mr), ...] for each target label found.
    """
    random.shuffle(mr_pool)
    found_mrs = []
    found_labels = set()
    targ_labels = set(targ_labels)
    for j, mr in enumerate(mr_pool):
        ser_score = ser.compute_ser(mr, ref_mr)
        s = 1.0 - ser_score["SER"]
        bs = bin_ser_score(s)
        if bs in targ_labels and bs not in found_labels:
            found_mrs.append((bs, mr))
            found_labels.add(bs)
        if len(found_mrs) == len(targ_labels) or j >= max_tries:
            break
    return found_mrs


def build_mr_pool(examples):
    """Build a pool of all gold MRs from the dataset for negative sampling."""
    pool = []
    for ex in examples:
        mr = ex["mr"]
        if isinstance(mr, list):
            mr_dict = defaultdict(list)
            for entry in mr:
                mr_dict[entry[0]].append(entry[1])
            mr = dict(mr_dict)
        pool.append(mr)
    return pool


def main():
    parser = argparse.ArgumentParser(
        description="Create evaluation dataset with augmented negative MRs")
    parser.add_argument("--input_path", type=str,
                        default="evaluation/gold/gold-annotated.json")
    parser.add_argument("--output_path", type=str,
                        default="evaluation/gold/eval-negatives.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_tries", type=int, default=1000)
    args = parser.parse_args()
    print("args:", args)

    random.seed(args.seed)

    with open(args.input_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} gold-annotated examples")

    mr_pool = build_mr_pool(data)
    print(f"MR pool size: {len(mr_pool)}")

    label_counts = Counter()
    examples_with_negs = 0

    for ex in tqdm.tqdm(data, desc="Generating negatives"):
        ref_mr = ex["mr"]
        if isinstance(ref_mr, list):
            ref_mr_dict = defaultdict(list)
            for entry in ref_mr:
                ref_mr_dict[entry[0]].append(entry[1])
            ref_mr = dict(ref_mr_dict)

        neg_mrs = fetch_ser_label_mr(
            ref_mr, LABEL_SET, list(mr_pool), max_tries=args.max_tries)

        negatives = []
        for label, neg_mr in neg_mrs:
            ser_vals = ser.compute_ser(neg_mr, ref_mr)
            negatives.append({
                "label": label,
                "mr": neg_mr,
                "ser_vals": ser_vals,
            })
            label_counts[label] += 1

        ex["negatives"] = negatives
        if negatives:
            examples_with_negs += 1

    print("\nNegative generation stats:")
    print(f"  Examples with negatives: {examples_with_negs}/{len(data)}")
    print(f"  Label distribution: {dict(sorted(label_counts.items()))}")
    print(f"  Total negatives: {sum(label_counts.values())}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
