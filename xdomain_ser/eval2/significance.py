# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Statistical significance tests for SER method comparisons.

McNemar's test (paired binary per-example accuracy) and paired
permutation (continuous SER_MAE) between all method pairs, run on:

* The personality routing details (output of
  :mod:`xdomain_ser.routing.personality`)
* The multi-domain routing details (output of
  :mod:`xdomain_ser.routing.selector`)
"""
import argparse
import json
import os
from itertools import combinations

import numpy as np
from scipy import stats


def mcnemars_test(a_correct, b_correct):
    """McNemar's test for paired binary data.

    Returns dict with test statistic, p-value, and contingency table counts.
    """
    assert len(a_correct) == len(b_correct)

    both_correct = sum(1 for a, b in zip(a_correct, b_correct) if a and b)
    a_only = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    b_only = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)
    both_wrong = sum(1 for a, b in zip(a_correct, b_correct) if not a and not b)

    n_discordant = a_only + b_only

    if n_discordant == 0:
        return {
            "statistic": 0.0, "p_value": 1.0,
            "both_correct": both_correct, "a_only": a_only,
            "b_only": b_only, "both_wrong": both_wrong,
            "n_discordant": 0,
        }

    chi2 = (abs(a_only - b_only) - 1) ** 2 / (a_only + b_only)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    if n_discordant < 25:
        p_exact = stats.binom_test(a_only, n_discordant, 0.5)
    else:
        p_exact = p_value

    return {
        "statistic": chi2,
        "p_value": p_value,
        "p_exact": p_exact,
        "both_correct": both_correct,
        "a_only": a_only,
        "b_only": b_only,
        "both_wrong": both_wrong,
        "n_discordant": n_discordant,
    }


def paired_permutation_test(a_values, b_values, n_permutations=10000, seed=42):
    """Paired permutation test for difference in means.

    Returns p-value for the null hypothesis that mean(A) == mean(B).
    """
    rng = np.random.RandomState(seed)
    a = np.array(a_values, dtype=float)
    b = np.array(b_values, dtype=float)
    n = len(a)

    diffs = a - b
    observed_diff = np.mean(diffs)

    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=n)
        shuffled_diff = np.mean(diffs * signs)
        if abs(shuffled_diff) >= abs(observed_diff):
            count += 1

    p_value = count / n_permutations
    return {
        "observed_diff": float(observed_diff),
        "p_value": p_value,
        "n_permutations": n_permutations,
    }


def significance_symbol(p):
    """Return significance marker for p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def test_personality_data(details_path):
    """Run significance tests on personality routing data."""
    print("\n" + "=" * 80)
    print("SIGNIFICANCE TESTS: PERSONALITY DATA (gold-annotated, N=500 test)")
    print("=" * 80)

    with open(details_path) as f:
        data = json.load(f)

    examples = data["examples"]
    n = len(examples)

    methods = {}
    methods["Aligner"] = [ex["aligner_all_acc"] for ex in examples]
    methods["LoRA"] = [ex["lora_all_acc"] for ex in examples]
    methods["NLI"] = [ex["nli_all_acc"] for ex in examples]

    methods["ScoreRt"] = [
        ex["lora_all_acc"] if ex["score_routing_choice"] == 1 else ex["nli_all_acc"]
        for ex in examples
    ]
    methods["LR-Rt"] = [
        ex["lora_all_acc"] if ex["lr_routing_choice"] == 1 else ex["nli_all_acc"]
        for ex in examples
    ]

    print(f"\n  Method accuracies (N={n}):")
    for name, vals in methods.items():
        acc = sum(vals) / len(vals)
        print(f"    {name:>10}: {acc:.4f}")

    method_names = list(methods.keys())
    print("\n  Pairwise McNemar's tests:")
    print(f"  {'A':>10} vs {'B':>10} | {'A_acc':>6} {'B_acc':>6} | {'disc.':>5} {'a_only':>6} {'b_only':>6} | {'chi2':>7} {'p':>8} {'sig':>5}")
    print(f"  {'-' * 90}")

    results = []
    for a_name, b_name in combinations(method_names, 2):
        a_vals = methods[a_name]
        b_vals = methods[b_name]
        test = mcnemars_test(a_vals, b_vals)
        a_acc = sum(a_vals) / len(a_vals)
        b_acc = sum(b_vals) / len(b_vals)
        p = test.get("p_exact", test["p_value"])
        sig = significance_symbol(p)

        print(f"  {a_name:>10} vs {b_name:>10} | {a_acc:>6.3f} {b_acc:>6.3f} | "
              f"{test['n_discordant']:>5} {test['a_only']:>6} {test['b_only']:>6} | "
              f"{test['statistic']:>7.3f} {p:>8.4f} {sig:>5}")

        results.append({
            "method_a": a_name, "method_b": b_name,
            "a_acc": a_acc, "b_acc": b_acc,
            "p_value": p, "significance": sig,
            **test,
        })

    return results


def test_multidomain_data(details_path):
    """Run significance tests on multi-domain routing data."""
    print("\n" + "=" * 80)
    print("SIGNIFICANCE TESTS: MULTI-DOMAIN DATA (perturbed negatives, test split)")
    print("=" * 80)

    with open(details_path) as f:
        data = json.load(f)

    examples = data["examples"]
    n = len(examples)

    methods_frac = {}
    methods_frac["LoRA"] = [ex["lora_correct"] / ex["n_negatives"] for ex in examples]
    methods_frac["NLI"] = [ex["nli_correct"] / ex["n_negatives"] for ex in examples]
    methods_frac["ScoreRt"] = [
        ex["lora_correct"] / ex["n_negatives"] if ex["score_routing_choice"] == 1
        else ex["nli_correct"] / ex["n_negatives"]
        for ex in examples
    ]
    methods_frac["LR-Rt"] = [
        ex["lora_correct"] / ex["n_negatives"] if ex["lr_routing_choice"] == 1
        else ex["nli_correct"] / ex["n_negatives"]
        for ex in examples
    ]

    methods_binary = {}
    methods_binary["LoRA"] = [ex["lora_correct"] == ex["n_negatives"] for ex in examples]
    methods_binary["NLI"] = [ex["nli_correct"] == ex["n_negatives"] for ex in examples]
    methods_binary["ScoreRt"] = [
        (ex["lora_correct"] == ex["n_negatives"]) if ex["score_routing_choice"] == 1
        else (ex["nli_correct"] == ex["n_negatives"])
        for ex in examples
    ]
    methods_binary["LR-Rt"] = [
        (ex["lora_correct"] == ex["n_negatives"]) if ex["lr_routing_choice"] == 1
        else (ex["nli_correct"] == ex["n_negatives"])
        for ex in examples
    ]

    print(f"\n  Per-example accuracy (all negatives correct, N={n} examples):")
    for name, vals in methods_binary.items():
        acc = sum(vals) / len(vals)
        print(f"    {name:>10}: {acc:.4f}")

    print("\n  Mean fraction correct per example:")
    for name, vals in methods_frac.items():
        m = np.mean(vals)
        print(f"    {name:>10}: {m:.4f}")

    method_names = list(methods_binary.keys())
    print("\n  Pairwise McNemar's tests (per-example all-correct):")
    print(f"  {'A':>10} vs {'B':>10} | {'A_acc':>6} {'B_acc':>6} | {'disc.':>5} {'a_only':>6} {'b_only':>6} | {'chi2':>7} {'p':>8} {'sig':>5}")
    print(f"  {'-' * 90}")

    results = []
    for a_name, b_name in combinations(method_names, 2):
        a_vals = methods_binary[a_name]
        b_vals = methods_binary[b_name]
        test = mcnemars_test(a_vals, b_vals)
        a_acc = sum(a_vals) / len(a_vals)
        b_acc = sum(b_vals) / len(b_vals)
        p = test.get("p_exact", test["p_value"])
        sig = significance_symbol(p)

        print(f"  {a_name:>10} vs {b_name:>10} | {a_acc:>6.3f} {b_acc:>6.3f} | "
              f"{test['n_discordant']:>5} {test['a_only']:>6} {test['b_only']:>6} | "
              f"{test['statistic']:>7.3f} {p:>8.4f} {sig:>5}")

        results.append({
            "method_a": a_name, "method_b": b_name,
            "a_acc": a_acc, "b_acc": b_acc,
            "p_value": p, "significance": sig,
            **test,
        })

    print("\n  Paired permutation tests (mean fraction correct, 10K iterations):")
    print(f"  {'A':>10} vs {'B':>10} | {'diff':>8} | {'p':>8} {'sig':>5}")
    print(f"  {'-' * 55}")

    for a_name, b_name in combinations(method_names, 2):
        a_vals = methods_frac[a_name]
        b_vals = methods_frac[b_name]
        boot = paired_permutation_test(a_vals, b_vals)
        sig = significance_symbol(boot["p_value"])

        print(f"  {a_name:>10} vs {b_name:>10} | {boot['observed_diff']:>+8.4f} | "
              f"{boot['p_value']:>8.4f} {sig:>5}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Statistical significance tests for SER method comparisons")
    parser.add_argument(
        "--personality_path", type=str,
        default="results/personality_routing/routing_details.json",
        help="Routing details JSON from xdomain_ser.routing.personality."
    )
    parser.add_argument(
        "--multidomain_path", type=str,
        default="results/routing/routing_details.json",
        help="Routing details JSON from xdomain_ser.routing.selector."
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="results/significance",
        help="Output directory for significance_tests.json."
    )
    args = parser.parse_args()

    all_results = {}

    if os.path.exists(args.personality_path):
        all_results["personality"] = test_personality_data(args.personality_path)
    else:
        print(f"[skip] personality data not found: {args.personality_path}")

    if os.path.exists(args.multidomain_path):
        all_results["multidomain"] = test_multidomain_data(args.multidomain_path)
    else:
        print(f"[skip] multidomain data not found: {args.multidomain_path}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "significance_tests.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
