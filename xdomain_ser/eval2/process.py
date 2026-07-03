# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Parse completed annotations (TSV or text format) into gold-annotated JSON.

Auto-detects format. Parses error notation (``D:slot``, ``S:slot=val``,
``I:slot=val``) into structured error lists, computes gold SER, and
reconstructs ``gold_pred_mr`` (what a perfect extractor would produce).

Input:  annotated TSV/text + the original ``sampled-examples.json``
Output: ``evaluation/gold/gold-annotated.json``
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from copy import deepcopy


def detect_format(path):
    """Auto-detect whether annotation file is TSV or text format."""
    with open(path) as f:
        first_line = f.readline()
    if "\t" in first_line and first_line.startswith("id"):
        return "tsv"
    return "text"


def parse_error_string(error_str):
    """Parse an error details string into a list of structured error dicts.

    Notation:
        D:slot              -> {"type": "deletion", "slot": "slot"}
        S:slot=wrong_value  -> {"type": "substitution", "slot": "slot", "value": "wrong_value"}
        I:slot=hal_value    -> {"type": "insertion", "slot": "slot", "value": "hal_value"}
    """
    if not error_str or error_str.strip() in ("", "______", "N/A", "none", "-"):
        return []

    errors = []
    parts = [p.strip() for p in error_str.split(",") if p.strip()]

    for part in parts:
        m = re.match(r"^([DSI]):(\w+)(?:=(.+))?$", part.strip())
        if not m:
            print(f"  Warning: could not parse error notation: '{part}'")
            continue

        error_type_code = m.group(1)
        slot = m.group(2)
        value = m.group(3)

        type_map = {"D": "deletion", "S": "substitution", "I": "insertion"}
        error = {"type": type_map[error_type_code], "slot": slot}
        if value is not None:
            error["value"] = value.strip()
        errors.append(error)

    return errors


def parse_tsv(path):
    """Parse annotated TSV file. Returns dict of id -> {has_error, errors}."""
    annotations = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ex_id = row["id"]
            has_error_raw = row.get("has_error", "").strip().upper()
            error_details = row.get("error_details", "").strip()

            has_error = has_error_raw == "YES"
            errors = parse_error_string(error_details) if has_error else []
            annotations[ex_id] = {"has_error": has_error, "errors": errors}

    return annotations


def parse_text(path):
    """Parse annotated text file. Returns dict of id -> {has_error, errors}."""
    annotations = {}
    with open(path) as f:
        content = f.read()

    blocks = re.split(r"=== ID: (\S+) ===", content)
    for i in range(1, len(blocks), 2):
        ex_id = blocks[i]
        body = blocks[i + 1] if i + 1 < len(blocks) else ""

        has_error_match = re.search(r"Has error \(YES/NO\):\s*(\S+)", body)
        has_error_raw = has_error_match.group(1) if has_error_match else ""
        has_error = has_error_raw.strip().upper() == "YES"

        error_match = re.search(r"Error details:\s*(.+?)(?:\n|$)", body)
        error_str = error_match.group(1).strip() if error_match else ""
        errors = parse_error_string(error_str) if has_error else []

        annotations[ex_id] = {"has_error": has_error, "errors": errors}

    return annotations


def compute_gold_ser(mr_dict, errors):
    """Compute gold SER from annotations.

    Returns: {"SER": float, "S": int, "D": int, "I": int, "N_ref": int}
    """
    n_ref = len(mr_dict)
    s = sum(1 for e in errors if e["type"] == "substitution")
    d = sum(1 for e in errors if e["type"] == "deletion")
    i = sum(1 for e in errors if e["type"] == "insertion")
    n = n_ref if n_ref > 0 else 1
    ser_val = (s + d + i) / n

    return {"SER": round(ser_val, 6), "S": s, "D": d, "I": i, "N_ref": n_ref}


def reconstruct_gold_pred_mr(mr_dict, errors):
    """Reconstruct what a perfect extractor would produce: the gold MR
    with annotated errors applied (reflecting what's actually in the text).
    """
    pred_mr = deepcopy(mr_dict)

    for error in errors:
        slot = error["slot"]
        etype = error["type"]

        if etype == "deletion":
            pred_mr.pop(slot, None)
        elif etype == "substitution":
            if "value" in error:
                pred_mr[slot] = error["value"]
        elif etype == "insertion":
            if "value" in error:
                pred_mr[slot] = error["value"]

    return pred_mr


def main():
    parser = argparse.ArgumentParser(
        description="Process annotations into gold-annotated JSON")
    parser.add_argument("--annotation_path", type=str, default=None,
                        help="Path to annotated file (TSV or text). "
                             "Auto-detects format. Tries TSV then text if not specified.")
    parser.add_argument("--sampled_path", type=str,
                        default="evaluation/gold/raw/sampled-examples.json",
                        help="Path to original sampled examples JSON")
    parser.add_argument("--output_path", type=str,
                        default="evaluation/gold/gold-annotated.json")
    args = parser.parse_args()
    print("args:", args)

    with open(args.sampled_path) as f:
        sampled = json.load(f)
    print(f"Loaded {len(sampled)} sampled examples")
    sampled_by_id = {ex["id"]: ex for ex in sampled}

    if args.annotation_path:
        ann_path = args.annotation_path
    else:
        tsv_path = "evaluation/gold/raw/annotation-data.tsv"
        txt_path = "evaluation/gold/raw/annotation-data.txt"
        if os.path.exists(tsv_path):
            ann_path = tsv_path
        elif os.path.exists(txt_path):
            ann_path = txt_path
        else:
            print("Error: no annotation file found. Run create_files.py first.")
            sys.exit(1)

    fmt = detect_format(ann_path)
    print(f"Detected format: {fmt} ({ann_path})")

    if fmt == "tsv":
        annotations = parse_tsv(ann_path)
    else:
        annotations = parse_text(ann_path)

    print(f"Parsed {len(annotations)} annotations")

    gold_data = []
    error_count = 0
    for ex in sampled:
        ex_id = ex["id"]
        ann = annotations.get(ex_id, {"has_error": False, "errors": []})

        gold_ser = compute_gold_ser(ex["mr"], ann["errors"])
        gold_pred_mr = reconstruct_gold_pred_mr(ex["mr"], ann["errors"])

        entry = {
            **ex,
            "has_error": ann["has_error"],
            "annotation_errors": ann["errors"],
            "gold_ser": gold_ser,
            "gold_pred_mr": gold_pred_mr,
        }
        gold_data.append(entry)

        if ann["has_error"]:
            error_count += 1

    print("\nAnnotation stats:")
    print(f"  Total examples: {len(gold_data)}")
    print(f"  With errors: {error_count}")
    print(f"  Without errors: {len(gold_data) - error_count}")
    print(f"  Error rate: {error_count / len(gold_data):.2%}")

    error_types = Counter()
    for entry in gold_data:
        for err in entry["annotation_errors"]:
            error_types[err["type"]] += 1
    print(f"  Error types: {dict(error_types)}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(gold_data, f, indent=2)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
