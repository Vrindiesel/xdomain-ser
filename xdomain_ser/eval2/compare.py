# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Three-way SER method comparison against gold annotations.

Compares:

* **E2E Aligner** (surface-form keyword/regex matching, from
  :mod:`xdomain_ser.baselines.e2e_aligner`)
* **LoRA MySER** (Llama-3.2-3B fine-tuned MR extraction + ranker)
* **NLI Baseline** (RoBERTa-MNLI per-slot entailment)

For each example, each method extracts an MR from the generated text,
computes SER vs the reference, and is compared against gold-annotated
error counts.

Input:  ``evaluation/gold/gold-annotated.json``
Output: ``evaluation/gold/ser-comparison-results.json``
"""
import argparse
import json
import os
from collections import defaultdict

from xdomain_ser.baselines.e2e_aligner import (
    pack_e2e_nlg_mr as aligner_pack_e2e_nlg_mr,
    extract_mr,
)
from xdomain_ser.core import ser
from xdomain_ser.ranking.make_eval_data import select_ranking_preds


# --- Slot name mappings ---

# Aligner extract_mr returns "customer rating" for customerRating
ALIGNER_TO_GOLD_SLOT = {
    "customer rating": "customerRating",
}

# LoRA pred_mr uses internal names (venue_type, cuisine_type, etc.)
LORA_REVERSE_KEY_MAP = {
    "customerRating": "customerRating",
    "family_suitability": "familyFriendly",
    "venue_type": "eatType",
    "cuisine_type": "food",
    "nearby_landmark": "near",
    "area_zone": "area",
}


def mr_dict_to_list(mr_dict):
    """Convert flat MR dict to list-of-tuples for pack_e2e_nlg_mr."""
    result = []
    for slot, val in mr_dict.items():
        if isinstance(val, list):
            result.append([slot] + val)
        else:
            result.append([slot, val])
    return result


def lora_pack_e2e_nlg_mr(d):
    """Pack LoRA-extracted MR to E2E-native slot names."""
    slot_value_pairs = []
    for slot_val in d:
        name = slot_val[0]
        values = slot_val[1:]
        if not isinstance(values, list):
            values = [values]
        orig_name = LORA_REVERSE_KEY_MAP.get(name, name)
        if name == "family_suitability":
            for v in values:
                orig_value = "yes" if v == "family-friendly" else "no"
                slot_value_pairs.append((orig_name, orig_value))
        else:
            slot_value_pairs.append((orig_name, values))
    return slot_value_pairs


def compute_per_example_errors(pred_mr_dict, gold_mr_dict):
    """Compare pred vs gold MR dicts and return a list of error dicts."""
    errors = []
    pred_keys = set(pred_mr_dict.keys())
    gold_keys = set(gold_mr_dict.keys())

    for k in pred_keys - gold_keys:
        pred_val = pred_mr_dict[k]
        if isinstance(pred_val, list):
            pred_val = pred_val[0] if len(pred_val) == 1 else pred_val
        errors.append({
            "type": "insertion", "slot": k,
            "gold_value": None, "pred_value": pred_val,
        })

    for k in gold_keys:
        gold_val = gold_mr_dict[k]
        if k not in pred_keys:
            errors.append({
                "type": "deletion", "slot": k,
                "gold_value": gold_val, "pred_value": None,
            })
        else:
            pred_val = pred_mr_dict[k]
            if isinstance(pred_val, list):
                pred_val = pred_val[0] if len(pred_val) == 1 else pred_val
            if not ser._values_equal(pred_val, gold_val):
                errors.append({
                    "type": "substitution", "slot": k,
                    "gold_value": gold_val, "pred_value": pred_val,
                })

    return errors


# --- Method A: E2E Aligner ---

def run_aligner(example):
    """Extract MR using surface-form alignment (e2e_aligner.py)."""
    gold_mr = example["mr"]
    text = example["clean_pred_text"]

    gold_mr_as_list = mr_dict_to_list(gold_mr)
    mr_mapped = aligner_pack_e2e_nlg_mr(gold_mr_as_list)
    extracted = extract_mr(text, mr_mapped)

    extracted_flat = {}
    for k, vals in extracted.items():
        gold_key = ALIGNER_TO_GOLD_SLOT.get(k, k)
        if len(vals) == 1:
            extracted_flat[gold_key] = vals[0]
        else:
            extracted_flat[gold_key] = vals

    return extracted_flat


# --- Method B: LoRA MySER ---

def run_lora(example):
    """Select best LoRA-extracted MR using ranker scores."""
    best_pred = select_ranking_preds(example["pred_mr"], example["pred_scores"])

    bpred = []
    for slot, vals in best_pred.items():
        bpred.append([slot] + vals)
    bpred_mapped = lora_pack_e2e_nlg_mr(bpred)

    bpred_mr = {}
    for (k, vals) in bpred_mapped:
        if isinstance(vals, list):
            v = vals[0]
        else:
            v = vals
        bpred_mr[k] = v

    return bpred_mr


# --- Method C: NLI Baseline ---

def run_nli_batch(examples, nli_model, threshold=0.5, batch_size=32):
    """Run NLI on all examples in batch, return list of recovered MR dicts."""
    from xdomain_ser.nli.templates import slot_value_to_template

    all_pairs = []
    pair_index = []

    for ex_idx, ex in enumerate(examples):
        text = ex["clean_pred_text"]
        gold_mr = ex["mr"]

        for slot, value in gold_mr.items():
            if isinstance(value, list):
                for v in value:
                    template = slot_value_to_template(slot, v)
                    if template is None:
                        continue
                    all_pairs.append((text, template))
                    pair_index.append((ex_idx, slot, v))
            else:
                template = slot_value_to_template(slot, value)
                if template is None:
                    continue
                all_pairs.append((text, template))
                pair_index.append((ex_idx, slot, value))

    print(f"  NLI: {len(all_pairs)} total pairs for {len(examples)} examples")

    all_probs = nli_model.batch_entailment(all_pairs, batch_size=batch_size)

    example_results = defaultdict(list)
    for (ex_idx, slot, value), prob in zip(pair_index, all_probs):
        example_results[ex_idx].append((slot, value, prob))

    recovered_mrs = []
    for ex_idx in range(len(examples)):
        nli_results = example_results.get(ex_idx, [])
        recovered = defaultdict(list)
        for slot, value, prob in nli_results:
            if prob > threshold:
                recovered[slot].append(value)

        flat = {}
        for k, vals in recovered.items():
            if len(vals) == 1:
                flat[k] = vals[0]
            else:
                flat[k] = vals
        recovered_mrs.append(flat)

    return recovered_mrs


# --- Comparison Logic ---

def compare_against_gold(method_mr, gold_mr, gold_pred_mr):
    """Compare a method's extracted MR against gold + gold_pred_mr."""
    method_ser = ser.compute_ser(method_mr, gold_mr)
    errors_vs_gold_pred = compute_per_example_errors(method_mr, gold_pred_mr)

    gold_pred_keys = set(gold_pred_mr.keys())
    method_keys = set(method_mr.keys())

    slot_tp = len(gold_pred_keys & method_keys)
    slot_fp = len(method_keys - gold_pred_keys)
    slot_fn = len(gold_pred_keys - method_keys)

    return {
        "method_ser": method_ser,
        "errors_vs_gold_pred": errors_vs_gold_pred,
        "slot_tp": slot_tp,
        "slot_fp": slot_fp,
        "slot_fn": slot_fn,
    }


def evaluate_all(examples, skip_nli=False, nli_batch_size=32, nli_threshold=0.5):
    """Run all methods and compare against gold annotations."""
    nli_mrs = None
    if not skip_nli:
        print("Loading NLI model...")
        from xdomain_ser.nli.model import NLIModel
        nli_model = NLIModel()
        print("Running NLI inference...")
        nli_mrs = run_nli_batch(
            examples, nli_model,
            threshold=nli_threshold, batch_size=nli_batch_size)

    methods = ["aligner", "lora"]
    if not skip_nli:
        methods.append("nli")

    results = {m: defaultdict(list) for m in methods}
    per_example_results = []

    for idx, ex in enumerate(examples):
        gold_mr = ex["mr"]
        gold_pred_mr = ex["gold_pred_mr"]
        gold_ser = ex["gold_ser"]

        example_entry = {
            "id": ex["id"],
            "source": ex["source"],
            "experiment": ex["experiment"],
            "personality": ex["personality"],
            "gold_ser": gold_ser,
        }

        aligner_mr = run_aligner(ex)
        aligner_cmp = compare_against_gold(aligner_mr, gold_mr, gold_pred_mr)
        example_entry["aligner"] = {
            "pred_mr": aligner_mr,
            "ser": aligner_cmp["method_ser"],
            "errors_vs_gold_pred": aligner_cmp["errors_vs_gold_pred"],
        }

        lora_mr = run_lora(ex)
        lora_cmp = compare_against_gold(lora_mr, gold_mr, gold_pred_mr)
        example_entry["lora"] = {
            "pred_mr": lora_mr,
            "ser": lora_cmp["method_ser"],
            "errors_vs_gold_pred": lora_cmp["errors_vs_gold_pred"],
        }

        if nli_mrs is not None:
            nli_mr = nli_mrs[idx]
            nli_cmp = compare_against_gold(nli_mr, gold_mr, gold_pred_mr)
            example_entry["nli"] = {
                "pred_mr": nli_mr,
                "ser": nli_cmp["method_ser"],
                "errors_vs_gold_pred": nli_cmp["errors_vs_gold_pred"],
            }

        per_example_results.append(example_entry)

        for method_name, cmp in [("aligner", aligner_cmp), ("lora", lora_cmp)]:
            _record_metrics(results[method_name], cmp, gold_ser, ex)
        if nli_mrs is not None:
            _record_metrics(results["nli"], nli_cmp, gold_ser, ex)

    return results, per_example_results, methods


def _record_metrics(result_dict, cmp, gold_ser, ex):
    """Record per-example metrics into aggregate result dict."""
    method_ser = cmp["method_ser"]

    s_acc = method_ser["S"] == gold_ser["S"]
    d_acc = method_ser["D"] == gold_ser["D"]
    i_acc = method_ser["I"] == gold_ser["I"]
    all_acc = s_acc and d_acc and i_acc
    ser_error = method_ser["SER"] - gold_ser["SER"]

    result_dict["S_acc"].append(s_acc)
    result_dict["D_acc"].append(d_acc)
    result_dict["I_acc"].append(i_acc)
    result_dict["all_acc"].append(all_acc)
    result_dict["ser_error"].append(ser_error)
    result_dict["source"].append(ex["source"])
    result_dict["personality"].append(ex["personality"])

    for err in cmp["errors_vs_gold_pred"]:
        result_dict[f"slot_error_{err['slot']}"].append(err["type"])
    for slot in ex["mr"]:
        result_dict[f"slot_present_{slot}"].append(True)


def compute_aggregate_metrics(result_dict, indices=None):
    """Compute aggregate metrics from result dict, optionally filtered by indices."""
    if indices is None:
        indices = range(len(result_dict["S_acc"]))

    s_vals = [result_dict["S_acc"][i] for i in indices]
    d_vals = [result_dict["D_acc"][i] for i in indices]
    i_vals = [result_dict["I_acc"][i] for i in indices]
    all_vals = [result_dict["all_acc"][i] for i in indices]
    ser_vals = [result_dict["ser_error"][i] for i in indices]

    n = len(s_vals)
    if n == 0:
        return {"S_acc": 0, "D_acc": 0, "I_acc": 0, "all_acc": 0,
                "SER_MAE": 0, "count": 0}

    return {
        "S_acc": sum(s_vals) / n,
        "D_acc": sum(d_vals) / n,
        "I_acc": sum(i_vals) / n,
        "all_acc": sum(all_vals) / n,
        "SER_MAE": sum(abs(e) for e in ser_vals) / n,
        "count": n,
    }


def print_comparison(results, methods):
    """Print comparison summary."""
    print("\n" + "=" * 60)
    print("OVERALL COMPARISON")
    print("=" * 60)
    header = f"{'Method':<12} {'S_acc':>7} {'D_acc':>7} {'I_acc':>7} {'All_acc':>8} {'MAE':>7} {'N':>6}"
    print(header)
    print("-" * 60)
    for method in methods:
        m = compute_aggregate_metrics(results[method])
        print(f"{method:<12} {m['S_acc']:>7.4f} {m['D_acc']:>7.4f} "
              f"{m['I_acc']:>7.4f} {m['all_acc']:>8.4f} {m['SER_MAE']:>7.4f} {m['count']:>6}")

    for src in ["llm", "seq2seq"]:
        print(f"\n--- Source: {src} ---")
        print(header)
        print("-" * 60)
        for method in methods:
            indices = [i for i, s in enumerate(results[method]["source"]) if s == src]
            m = compute_aggregate_metrics(results[method], indices)
            print(f"{method:<12} {m['S_acc']:>7.4f} {m['D_acc']:>7.4f} "
                  f"{m['I_acc']:>7.4f} {m['all_acc']:>8.4f} {m['SER_MAE']:>7.4f} {m['count']:>6}")

    personalities = ["AGREEABLE", "DISAGREEABLE", "EXTRAVERT",
                     "CONSCIENTIOUSNESS", "UNCONSCIENTIOUSNESS"]
    for pers in personalities:
        print(f"\n--- Personality: {pers} ---")
        print(header)
        print("-" * 60)
        for method in methods:
            indices = [i for i, p in enumerate(results[method]["personality"]) if p == pers]
            m = compute_aggregate_metrics(results[method], indices)
            print(f"{method:<12} {m['S_acc']:>7.4f} {m['D_acc']:>7.4f} "
                  f"{m['I_acc']:>7.4f} {m['all_acc']:>8.4f} {m['SER_MAE']:>7.4f} {m['count']:>6}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate three SER methods against gold annotations")
    parser.add_argument("--input_path", type=str,
                        default="evaluation/gold/gold-annotated.json")
    parser.add_argument("--output_path", type=str,
                        default="evaluation/gold/ser-comparison-results.json")
    parser.add_argument("--skip_nli", action="store_true",
                        help="Skip NLI method (e.g. if no GPU available)")
    parser.add_argument("--nli_batch_size", type=int, default=32)
    parser.add_argument("--nli_threshold", type=float, default=0.5)
    args = parser.parse_args()
    print("args:", args)

    with open(args.input_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} gold-annotated examples")

    results, per_example_results, methods = evaluate_all(
        data,
        skip_nli=args.skip_nli,
        nli_batch_size=args.nli_batch_size,
        nli_threshold=args.nli_threshold,
    )

    print_comparison(results, methods)

    output = {
        "methods": methods,
        "per_example": per_example_results,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved detailed results to {args.output_path}")


if __name__ == "__main__":
    main()
