# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Pick the highest-complexity few-shot examples per topic from a pool.

Auxiliary example-selection helper: reads a ``{topic: [examples]}`` JSON, sorts
each topic's examples by MR slot count (descending), and keeps the top ``--num``
as a few-shot prompt pool. Note the released v9 prompt-example files are produced
directly by ``merge_select.py``; this is a standalone alternative for building a
complexity-biased pool when constructing data for a new domain.

(The earlier ``--features``-keyed random-sampling variant, which depended on the
Chapter-4 PERSONAGE example schema, is dropped.)
"""
import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2323)
    args = parser.parse_args()
    print("args:", args)


    with open(args.input_path) as fin:
        data = json.load(fin)

    print("data keys:", data.keys())

    top_n = defaultdict(list)
    for topic, examples in data.items():
        mr_lengths = []
        for ex in examples:
            mr = ex["mr"]
            if mr.get("slots"):
                mr = mr["slots"]
            mr_lengths.append(len(mr))

        mr_examples = [(l, j, ex) for j, (l, ex) in enumerate(zip(mr_lengths, examples))]
        mr_examples.sort(reverse=True)
        top_n[topic] = [ex for l, j, ex in mr_examples[:args.num]]

    with open(args.output_path, "w") as fout:
        json.dump(top_n, fout, indent=2)


if __name__ == "__main__":
    main()
