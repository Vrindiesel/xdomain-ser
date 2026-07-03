# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Create a representative dev set that covers all test topics, excludes
train-200 and test examples, and biases toward larger (harder) MRs.

Sampling strategy per topic:
  1. Collect all available examples (not in train-200 or test)
  2. Sort by MR size descending
  3. Take the top 50% by MR size as the candidate pool
  4. Randomly sample k from that pool

This ensures dev examples are harder than average while still having
variety within the high-complexity band.

Also generates prompt-examples-dev-representative.json by sampling
a separate pool of 5 examples per topic for in-context prompts.

Usage (from the repository root):
    python -m xdomain_ser.data_prep.representative_dev \
        --output_dir data/multi_ser_v9 \
        --dev_k 10 \
        --seed 42
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict


def get_sf(e):
    """Get surface form as string (some examples have list-valued surface_form)."""
    sf = e["surface_form"]
    return sf if isinstance(sf, str) else sf[0]


def mr_size(e):
    """Count the number of slots in an MR (handles both dict formats)."""
    mr = e["mr"]
    if isinstance(mr, dict) and "slots" in mr:
        return len(mr["slots"])
    elif isinstance(mr, dict):
        return len(mr)
    elif isinstance(mr, list):
        return len(mr)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Create a representative dev set covering all test topics"
    )
    parser.add_argument(
        "--files_json",
        default="data/multi_ser_v9/files.json",
        help="Path to files.json config.",
    )
    parser.add_argument(
        "--train_file",
        default="data/multi_ser_v9/train-200.json",
        help="Path to train-200.json (for exclusion).",
    )
    parser.add_argument(
        "--test_file",
        default="data/multi_ser_v9/test.json",
        help="Path to test.json (for exclusion).",
    )
    parser.add_argument(
        "--output_dir",
        default="data/multi_ser_v9",
        help="Output directory.",
    )
    parser.add_argument(
        "--dev_k", type=int, default=10,
        help="Number of dev examples per topic (default: 10).",
    )
    parser.add_argument(
        "--prompt_k", type=int, default=5,
        help="Number of prompt examples per topic (default: 5).",
    )
    parser.add_argument(
        "--hard_percentile", type=float, default=0.5,
        help="Keep top X fraction by MR size as candidate pool (default: 0.5).",
    )
    parser.add_argument(
        "--min_mr_len", type=int, default=2,
        help="Minimum MR slots to include (default: 2).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()
    random.seed(args.seed)

    # ---- Load exclusion sets ----
    print("Loading train and test for exclusion...")
    with open(args.train_file) as f:
        train_data = json.load(f)
    with open(args.test_file) as f:
        test_data = json.load(f)

    exclude_sfs = set()
    for e in train_data:
        exclude_sfs.add(get_sf(e))
    for e in test_data:
        exclude_sfs.add(get_sf(e))
    print("  Excluding %d surface forms (train + test)" % len(exclude_sfs))

    # ---- Determine target topics (those that appear in test) ----
    test_topics = set(e["hint_map_id"] for e in test_data)
    print("  Test topics (%d): %s" % (len(test_topics), sorted(test_topics)))

    # ---- Load source pools ----
    print("\nLoading source data pools...")
    with open(args.files_json) as f:
        files = json.load(f)

    # Collect from both train and dev sources
    topic_pools = defaultdict(list)
    for source_key in ["train", "dev"]:
        for fpath in files.get(source_key, []):
            try:
                with open(fpath) as f:
                    examples = json.load(f)
            except FileNotFoundError:
                # Try with ../ prefix (script runs from script/ dir)
                try:
                    with open("../" + fpath) as f:
                        examples = json.load(f)
                except FileNotFoundError:
                    print("  WARNING: skipping %s (not found)" % fpath)
                    continue
            for e in examples:
                topic_pools[e["hint_map_id"]].append(e)

    # ---- Filter and sample ----
    print("\nFiltering and sampling dev set...")
    dev_examples = []
    prompt_examples = defaultdict(list)
    stats = {}

    print("\n%-42s %6s %6s %6s %6s %6s %s" % (
        "TOPIC", "POOL", "AVAIL", "HARD", "DEV_K", "PR_K", "MR_RANGE"))
    print("-" * 95)

    for topic in sorted(test_topics):
        pool = topic_pools.get(topic, [])

        # Filter: exclude train/test surface forms, enforce min MR length
        available = [e for e in pool
                     if get_sf(e) not in exclude_sfs
                     and mr_size(e) >= args.min_mr_len]

        if not available:
            print("%-42s %6d %6d %6s -- NO AVAILABLE EXAMPLES --" % (
                topic, len(pool), 0, ""))
            stats[topic] = {"pool": len(pool), "available": 0,
                            "selected": 0, "prompt": 0}
            continue

        # Sort by MR size descending, take top percentile as hard pool
        available.sort(key=mr_size, reverse=True)
        cutoff = max(1, int(len(available) * args.hard_percentile))
        hard_pool = available[:cutoff]

        # Sample dev examples from hard pool
        k_dev = min(args.dev_k, len(hard_pool))
        dev_sample = random.sample(hard_pool, k_dev)
        dev_sfs = set(get_sf(e) for e in dev_sample)

        # Sample prompt examples from remaining available (not in dev sample)
        prompt_pool = [e for e in available if get_sf(e) not in dev_sfs]
        k_prompt = min(args.prompt_k, len(prompt_pool))
        if k_prompt > 0:
            prompt_sample = random.sample(prompt_pool, k_prompt)
        else:
            prompt_sample = []

        dev_examples.extend(dev_sample)
        prompt_examples[topic] = prompt_sample

        dev_mr_sizes = [mr_size(e) for e in dev_sample]
        pool_mr_sizes = [mr_size(e) for e in available]

        print("%-42s %6d %6d %6d %6d %6d  dev=%d-%d  pool=%d-%d" % (
            topic, len(pool), len(available), len(hard_pool),
            k_dev, k_prompt,
            min(dev_mr_sizes), max(dev_mr_sizes),
            min(pool_mr_sizes), max(pool_mr_sizes)))

        stats[topic] = {
            "pool": len(pool),
            "available": len(available),
            "hard_pool": len(hard_pool),
            "selected": k_dev,
            "prompt": k_prompt,
            "dev_mr_range": [min(dev_mr_sizes), max(dev_mr_sizes)],
            "pool_mr_range": [min(pool_mr_sizes), max(pool_mr_sizes)],
        }

    # ---- Verify no overlap ----
    dev_sfs_final = set(get_sf(e) for e in dev_examples)
    train_overlap = dev_sfs_final & set(get_sf(e) for e in train_data)
    test_overlap = dev_sfs_final & set(get_sf(e) for e in test_data)
    print("\n=== OVERLAP CHECK ===")
    print("  Dev examples: %d" % len(dev_examples))
    print("  Overlap with train-200: %d" % len(train_overlap))
    print("  Overlap with test: %d" % len(test_overlap))
    assert len(train_overlap) == 0, "Dev overlaps with train!"
    assert len(test_overlap) == 0, "Dev overlaps with test!"

    # ---- Summary ----
    print("\n=== DEV SET SUMMARY ===")
    dev_tc = Counter(e["hint_map_id"] for e in dev_examples)
    for t in sorted(dev_tc):
        print("  %-42s %4d examples" % (t, dev_tc[t]))
    print("  Total: %d examples across %d topics" % (
        len(dev_examples), len(dev_tc)))

    dev_mr_sizes = [mr_size(e) for e in dev_examples]
    print("  MR slots: mean=%.1f  min=%d  max=%d" % (
        sum(dev_mr_sizes) / len(dev_mr_sizes),
        min(dev_mr_sizes), max(dev_mr_sizes)))

    # ---- Save ----
    os.makedirs(args.output_dir, exist_ok=True)

    dev_path = os.path.join(args.output_dir,
                            "dev-repr-%d.json" % len(dev_examples))
    with open(dev_path, "w") as f:
        json.dump(dev_examples, f, indent=2)
    print("\nDev set saved to: %s" % dev_path)

    prompt_path = os.path.join(args.output_dir,
                               "prompt-examples-dev-repr.json")
    with open(prompt_path, "w") as f:
        json.dump(prompt_examples, f, indent=2)
    print("Prompt examples saved to: %s" % prompt_path)

    stats_path = os.path.join(args.output_dir,
                              "dev-repr-stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "dev_k": args.dev_k,
            "hard_percentile": args.hard_percentile,
            "min_mr_len": args.min_mr_len,
            "n_dev_examples": len(dev_examples),
            "n_topics": len(dev_tc),
            "per_topic": stats,
        }, f, indent=2)
    print("Stats saved to: %s" % stats_path)


if __name__ == "__main__":
    main()
