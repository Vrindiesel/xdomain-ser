# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Generate annotation files in both TSV and structured-text formats.

Input:  ``evaluation/gold/raw/sampled-examples.json`` (from
        :mod:`xdomain_ser.eval2.sample`)
Output: ``evaluation/gold/raw/annotation-data.tsv``
        ``evaluation/gold/raw/annotation-data.txt``

Annotators mark each example as YES / NO and list errors using
``D:slot``, ``S:slot=val``, ``I:slot=val`` notation.
"""
import argparse
import csv
import json
import os


def format_mr_readable(mr_dict):
    """Format an MR dict as a readable string: name[value], eatType[restaurant], ..."""
    parts = []
    for slot, value in mr_dict.items():
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        parts.append(f"{slot}[{value}]")
    return ", ".join(parts)


def write_tsv(examples, output_path):
    """Write annotation TSV with header and instructions."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        writer.writerow([
            "id", "source", "experiment", "personality",
            "mr", "pred_text", "has_error", "error_details"
        ])

        for ex in examples:
            mr_str = format_mr_readable(ex["mr"])
            pred_text = ex["clean_pred_text"]
            writer.writerow([
                ex["id"],
                ex["source"],
                ex["experiment"],
                ex["personality"],
                mr_str,
                pred_text,
                "",  # has_error: annotator fills YES/NO
                "",  # error_details: annotator fills D:slot, S:slot=val, I:slot=val
            ])

    print(f"  TSV written: {output_path} ({len(examples)} rows)")


ANNOTATION_INSTRUCTIONS = """\
===============================================================================
SER ANNOTATION INSTRUCTIONS
===============================================================================

For each example, compare the MR (meaning representation) to the generated text.

Mark has_error as:
  YES -- if ANY slot-value pair is incorrect
  NO  -- if ALL slot-value pairs are correctly realized

If YES, list errors in error_details using this notation:
  D:slot           -- Deletion: slot is in the MR but missing from the text
  S:slot=wrong_val -- Substitution: slot mentioned but with wrong value
  I:slot=hal_val   -- Insertion: slot in text but NOT in the MR (hallucinated)

Multiple errors: separate with commas, e.g.: D:area, S:food=Japanese, I:priceRange=cheap

IMPORTANT: Variable placeholders (nameVariable, nearVariable, _area_variable_)
are always considered correct by convention -- do not mark these as errors.

E2E slot types: name, eatType, food, priceRange, customerRating, area,
                familyFriendly, near

===============================================================================
"""


def write_text(examples, output_path):
    """Write structured-text annotation file."""
    with open(output_path, "w") as f:
        f.write(ANNOTATION_INSTRUCTIONS)
        f.write("\n")

        for ex in examples:
            mr_str = format_mr_readable(ex["mr"])
            pred_text = ex["clean_pred_text"]

            f.write(f"=== ID: {ex['id']} ===\n")
            f.write(f"Source: {ex['source']} | Experiment: {ex['experiment']} | "
                    f"Personality: {ex['personality']}\n\n")
            f.write(f"MR: {mr_str}\n\n")
            f.write(f"Generated text:\n{pred_text}\n\n")
            f.write("Has error (YES/NO): ______\n")
            f.write("Error details: ______\n\n")
            f.write("===\n\n")

    print(f"  Text written: {output_path} ({len(examples)} entries)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotation files from sampled examples")
    parser.add_argument("--input_path", type=str,
                        default="evaluation/gold/raw/sampled-examples.json")
    parser.add_argument("--output_dir", type=str,
                        default="evaluation/gold/raw")
    args = parser.parse_args()
    print("args:", args)

    with open(args.input_path) as f:
        examples = json.load(f)
    print(f"Loaded {len(examples)} examples")

    os.makedirs(args.output_dir, exist_ok=True)

    tsv_path = os.path.join(args.output_dir, "annotation-data.tsv")
    txt_path = os.path.join(args.output_dir, "annotation-data.txt")

    write_tsv(examples, tsv_path)
    write_text(examples, txt_path)

    print("\nDone. Annotate either file, then run process.py")


if __name__ == "__main__":
    main()
