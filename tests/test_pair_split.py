# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0.
"""Split-integrity guard for the corrected per-pair Eval-2 protocol.

The pair table must inherit the seed-42 example-level stratified split:
all pairs of one text stay on one side, and the published split sizes
(dev 1,143 examples / 4,532 pairs; test 1,145 / 4,510) are preserved.
"""
import json

from xdomain_ser.core import registry
from xdomain_ser.mixture.threshold_sweep import (
    stratified_pair_split,
    stratified_split,
)


def _data():
    with open(registry.RANKER_EVAL_SCORED) as f:
        return json.load(f)


def test_pair_split_matches_published_sizes():
    data = _data()
    dev_idx, test_idx = stratified_split(data, seed=42)
    assert (len(dev_idx), len(test_idx)) == (1143, 1145)
    dev_pairs, test_pairs = stratified_pair_split(data, seed=42)
    assert (len(dev_pairs), len(test_pairs)) == (4532, 4510)


def test_pair_split_group_purity():
    data = _data()
    dev_pairs, test_pairs = stratified_pair_split(data, seed=42)
    dev_ex = {i for i, _ in dev_pairs}
    test_ex = {i for i, _ in test_pairs}
    assert not (dev_ex & test_ex), "a text straddles dev and test"
    total = sum(len(e["negatives"]) for e in data)
    assert len(dev_pairs) + len(test_pairs) == total
    assert len(set(dev_pairs) | set(test_pairs)) == total


def test_pair_split_deterministic():
    data = _data()
    assert stratified_pair_split(data, seed=42) == stratified_pair_split(data, seed=42)
