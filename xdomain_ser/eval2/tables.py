# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Generate 5 TSV result tables from the SER comparison results JSON.

Produces:

1. ``method-comparison-overall.tsv``        -- Overall accuracy/MAE per method
2. ``method-comparison-per-source.tsv``     -- LLM vs seq2seq breakdown (key)
3. ``method-comparison-per-personality.tsv`` -- 5 personality types
4. ``method-comparison-per-error-type.tsv``  -- S/D/I detection accuracy
5. ``method-comparison-per-slot.tsv``        -- Per E2E slot

Input: ``evaluation/gold/ser-comparison-results.json`` + ``evaluation/gold/gold-annotated.json``
"""
import argparse
import json
import os

from xdomain_ser.core import ser


E2E_SLOTS = [
    "name", "eatType", "food", "priceRange",
    "customerRating", "area", "familyFriendly", "near",
]

PERSONALITIES = [
    "AGREEABLE", "DISAGREEABLE", "EXTRAVERT",
    "CONSCIENTIOUSNESS", "UNCONSCIENTIOUSNESS",
]


def load_results(path):
    """Load the comparison results JSON."""
    with open(path) as f:
        return json.load(f)


def compute_metrics_from_examples(examples, methods, filter_fn=None):
    """Compute aggregate metrics from per-example results, optionally filtered."""
    if filter_fn:
        examples = [ex for ex in examples if filter_fn(ex)]

    metrics = {}
    for method in methods:
        s_accs = []
        d_accs = []
        i_accs = []
        all_accs = []
        ser_errors = []

        for ex in examples:
            if method not in ex:
                continue
            gold_ser = ex["gold_ser"]
            method_ser = ex[method]["ser"]

            s_acc = method_ser["S"] == gold_ser["S"]
            d_acc = method_ser["D"] == gold_ser["D"]
            i_acc = method_ser["I"] == gold_ser["I"]
            all_acc = s_acc and d_acc and i_acc
            ser_err = method_ser["SER"] - gold_ser["SER"]

            s_accs.append(s_acc)
            d_accs.append(d_acc)
            i_accs.append(i_acc)
            all_accs.append(all_acc)
            ser_errors.append(ser_err)

        n = len(s_accs)
        if n == 0:
            metrics[method] = {"S_acc": 0, "D_acc": 0, "I_acc": 0,
                               "all_acc": 0, "SER_MAE": 0, "count": 0}
        else:
            metrics[method] = {
                "S_acc": sum(s_accs) / n,
                "D_acc": sum(d_accs) / n,
                "I_acc": sum(i_accs) / n,
                "all_acc": sum(all_accs) / n,
                "SER_MAE": sum(abs(e) for e in ser_errors) / n,
                "count": n,
            }

    return metrics


def write_tsv(rows, output_path):
    """Write rows (list of lists) as TSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")
    print(f"  Written: {output_path}")


def fmt(v):
    """Format float to 4 decimal places."""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def generate_overall_table(examples, methods, output_dir):
    """Table 1: Overall accuracy/MAE per method."""
    metrics = compute_metrics_from_examples(examples, methods)
    header = ["method", "count", "S_acc", "D_acc", "I_acc", "all_acc", "SER_MAE"]
    rows = [header]
    for m in methods:
        d = metrics[m]
        rows.append([m, d["count"], fmt(d["S_acc"]), fmt(d["D_acc"]),
                     fmt(d["I_acc"]), fmt(d["all_acc"]), fmt(d["SER_MAE"])])
    write_tsv(rows, os.path.join(output_dir, "method-comparison-overall.tsv"))


def generate_per_source_table(examples, methods, output_dir):
    """Table 2: LLM vs seq2seq breakdown (key table for the paper)."""
    header = ["source", "method", "count", "S_acc", "D_acc", "I_acc", "all_acc", "SER_MAE"]
    rows = [header]
    for src in ["llm", "seq2seq"]:
        metrics = compute_metrics_from_examples(
            examples, methods, filter_fn=lambda ex, s=src: ex["source"] == s)
        for m in methods:
            d = metrics[m]
            rows.append([src, m, d["count"], fmt(d["S_acc"]), fmt(d["D_acc"]),
                         fmt(d["I_acc"]), fmt(d["all_acc"]), fmt(d["SER_MAE"])])
    write_tsv(rows, os.path.join(output_dir, "method-comparison-per-source.tsv"))


def generate_per_personality_table(examples, methods, output_dir):
    """Table 3: Per personality type."""
    header = ["personality", "method", "count", "S_acc", "D_acc", "I_acc", "all_acc", "SER_MAE"]
    rows = [header]
    for pers in PERSONALITIES:
        metrics = compute_metrics_from_examples(
            examples, methods, filter_fn=lambda ex, p=pers: ex["personality"] == p)
        for m in methods:
            d = metrics[m]
            rows.append([pers, m, d["count"], fmt(d["S_acc"]), fmt(d["D_acc"]),
                         fmt(d["I_acc"]), fmt(d["all_acc"]), fmt(d["SER_MAE"])])
    write_tsv(rows, os.path.join(output_dir, "method-comparison-per-personality.tsv"))


def generate_per_error_type_table(examples, methods, output_dir):
    """Table 4: S/D/I detection accuracy broken down by error type."""
    header = ["error_type", "method", "n_examples", "detection_rate"]
    rows = [header]

    for error_type, ser_key in [("substitution", "S"), ("deletion", "D"), ("insertion", "I")]:
        has_error = [ex for ex in examples if ex["gold_ser"][ser_key] > 0]

        for m in methods:
            detected = 0
            total = 0
            for ex in has_error:
                if m not in ex:
                    continue
                total += 1
                method_ser = ex[m]["ser"]
                if method_ser[ser_key] == ex["gold_ser"][ser_key]:
                    detected += 1

            rate = detected / total if total > 0 else 0.0
            rows.append([error_type, m, total, fmt(rate)])

    write_tsv(rows, os.path.join(output_dir, "method-comparison-per-error-type.tsv"))


def generate_per_slot_table(examples, methods, output_dir):
    """Table 5: Per E2E slot - extraction accuracy."""
    header = ["slot", "method", "n_present", "correct_rate"]
    rows = [header]

    for slot in E2E_SLOTS:
        for m in methods:
            correct = 0
            total = 0
            for ex in examples:
                if m not in ex:
                    continue
                gold_mr = ex["mr"]
                gold_pred_mr = ex.get("gold_pred_mr", gold_mr)
                method_mr = ex[m].get("pred_mr", {})

                if slot in gold_pred_mr:
                    total += 1
                    gold_val = gold_pred_mr[slot]
                    pred_val = method_mr.get(slot)
                    if pred_val is not None and ser._values_equal(pred_val, gold_val):
                        correct += 1
                elif slot in method_mr:
                    total += 1

            rate = correct / total if total > 0 else 0.0
            rows.append([slot, m, total, fmt(rate)])

    write_tsv(rows, os.path.join(output_dir, "method-comparison-per-slot.tsv"))


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison result tables from SER comparison results")
    parser.add_argument("--input_path", type=str,
                        default="evaluation/gold/ser-comparison-results.json")
    parser.add_argument("--gold_path", type=str,
                        default="evaluation/gold/gold-annotated.json",
                        help="Gold-annotated JSON (needed for per-slot table)")
    parser.add_argument("--output_dir", type=str,
                        default="evaluation/gold/tables")
    args = parser.parse_args()
    print("args:", args)

    data = load_results(args.input_path)
    examples = data["per_example"]
    methods = data["methods"]
    print(f"Loaded {len(examples)} examples, methods: {methods}")

    with open(args.gold_path) as f:
        gold_data = json.load(f)
    gold_by_id = {ex["id"]: ex for ex in gold_data}
    for ex in examples:
        gold_ex = gold_by_id.get(ex["id"], {})
        ex["mr"] = gold_ex.get("mr", {})
        ex["gold_pred_mr"] = gold_ex.get("gold_pred_mr", {})

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nGenerating tables...")
    generate_overall_table(examples, methods, args.output_dir)
    generate_per_source_table(examples, methods, args.output_dir)
    generate_per_personality_table(examples, methods, args.output_dir)
    generate_per_error_type_table(examples, methods, args.output_dir)
    generate_per_slot_table(examples, methods, args.output_dir)

    print(f"\nAll tables written to {args.output_dir}/")


if __name__ == "__main__":
    main()
