# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""[PHASE-0 / POST-PUBLICATION FOLLOW-UP]

Evaluate an LR sweep over fine-tuned extraction models on the representative
dev set (``dev-repr-120``). Computes both standard SER and slot-F1 to avoid
the ceiling effects seen with the original ``dev-50`` set, where all
candidate learning rates appeared near-perfect.

This module is part of the post-publication follow-up experiments and is
not part of the headline paper pipeline. See the phase-0 sections of
RELEASE_NOTES.md for the original write-up.

Usage::

    python -m xdomain_ser.extraction.experimental.lr_sweep \\
        --sweep_dir multi-ser/ \\
        --output_dir results/lr_sweep
"""
import argparse
import json
import os
import glob
import re
from collections import defaultdict

from xdomain_ser.core import ser


def extract_lr_from_dirname(dirname):
    """Extract LR value from directory name like lr_sweep_lr1em5."""
    match = re.search(r'lr_sweep_lr(\d+em?\d+)$', dirname)
    if not match:
        return None
    lr_str = match.group(1).replace('m', '-')
    try:
        return float(lr_str)
    except ValueError:
        return None


def evaluate_predictions(pred_file):
    """
    Evaluate predictions from a multi-ser inference output file.

    Returns dict with slot-F1, SER, per-topic breakdown.
    """
    with open(pred_file) as f:
        examples = json.load(f)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_ser_s = 0
    total_ser_d = 0
    total_ser_i = 0
    total_ser_n = 0
    n_perfect = 0
    n_total = 0

    topic_results = defaultdict(lambda: {
        "tp": 0, "fp": 0, "fn": 0, "n": 0, "perfect": 0,
        "S": 0, "D": 0, "I": 0, "N_ref": 0,
    })

    for e in examples:
        # Get reference MR
        ref_mr = ser.SlotErrorRate.normalize_mr(e["mr"])

        # Get predicted MR (best ranked if multiple)
        pred_raw = e.get("pred_mr", [])
        if isinstance(pred_raw, list):
            if e.get("pred_scores"):
                # Deferred import: select_ranking_preds lives in xdomain_ser.ranking
                # (migrated in Stage 4). Importing here keeps Stage 3 self-contained.
                from xdomain_ser.ranking.make_eval_data import select_ranking_preds
                pred_mr = select_ranking_preds(pred_raw, e["pred_scores"])
            else:
                pred_mr = ser.extract_attributes_dict(pred_raw[0]) if pred_raw else {}
        else:
            pred_mr = ser.extract_attributes_dict(pred_raw)

        topic = e.get("hint_map_id", e.get("topic", "unknown"))

        # Slot-F1
        f1_result = ser.compute_slot_f1(pred_mr, ref_mr)
        tp = f1_result["tp"]
        fp = f1_result["fp"]
        fn = f1_result["fn"]

        total_tp += tp
        total_fp += fp
        total_fn += fn

        topic_results[topic]["tp"] += tp
        topic_results[topic]["fp"] += fp
        topic_results[topic]["fn"] += fn
        topic_results[topic]["n"] += 1

        # SER
        ser_result = ser.compute_ser(pred_mr, ref_mr)
        total_ser_s += ser_result["S"]
        total_ser_d += ser_result["D"]
        total_ser_i += ser_result["I"]
        total_ser_n += ser_result["N_ref"]

        topic_results[topic]["S"] += ser_result["S"]
        topic_results[topic]["D"] += ser_result["D"]
        topic_results[topic]["I"] += ser_result["I"]
        topic_results[topic]["N_ref"] += ser_result["N_ref"]

        # Perfect extraction
        if tp > 0 and fp == 0 and fn == 0:
            n_perfect += 1
            topic_results[topic]["perfect"] += 1

        n_total += 1

    # Compute aggregate metrics
    f1, precision, recall = ser.calc_p_r_f(total_fn, total_fp, total_tp)
    ser_val = ser._compute_ser_val(total_ser_d, total_ser_i, total_ser_s, total_ser_n)

    result = {
        "n_examples": n_total,
        "n_perfect": n_perfect,
        "perfect_pct": round(100 * n_perfect / max(n_total, 1), 2),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "SER": round(ser_val, 6),
        "S": total_ser_s,
        "D": total_ser_d,
        "I": total_ser_i,
        "N_ref": total_ser_n,
        "per_topic": {},
    }

    for topic in sorted(topic_results):
        tr = topic_results[topic]
        t_f1, t_prec, t_rec = ser.calc_p_r_f(tr["fn"], tr["fp"], tr["tp"])
        t_ser = ser._compute_ser_val(tr["D"], tr["I"], tr["S"], tr["N_ref"])
        result["per_topic"][topic] = {
            "n": tr["n"],
            "perfect": tr["perfect"],
            "tp": tr["tp"], "fp": tr["fp"], "fn": tr["fn"],
            "precision": round(t_prec, 4),
            "recall": round(t_rec, 4),
            "f1": round(t_f1, 4),
            "SER": round(t_ser, 4),
        }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LR sweep on representative dev set"
    )
    parser.add_argument(
        "--sweep_dir", default="multi-ser/",
        help="Parent directory with lr_sweep_lr* subdirs.",
    )
    parser.add_argument(
        "--pred_filename", default="dev-repr-120.json",
        help="Prediction filename to look for in checkpoint dirs.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/lr_sweep",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find sweep directories
    pattern = os.path.join(args.sweep_dir, "lr_sweep_lr*")
    sweep_dirs = sorted(glob.glob(pattern))

    results = []

    print("Evaluating LR sweep on %s" % args.pred_filename)
    print("=" * 80)

    for sweep_dir in sweep_dirs:
        dirname = os.path.basename(sweep_dir)
        lr = extract_lr_from_dirname(dirname)
        if lr is None:
            continue

        # Find prediction file (search all checkpoints, not just the last)
        ckpts = sorted(glob.glob(os.path.join(sweep_dir, "checkpoint-*")),
                       key=lambda x: int(x.split("-")[-1]))
        if not ckpts:
            print("  LR=%s: no checkpoints found" % lr)
            continue

        pred_file = None
        last_ckpt = None
        for ckpt in reversed(ckpts):
            candidate = os.path.join(ckpt, args.pred_filename)
            if os.path.exists(candidate):
                pred_file = candidate
                last_ckpt = ckpt
                break
        if pred_file is None:
            print("  LR=%s: no %s in any checkpoint" % (lr, args.pred_filename))
            continue

        print("\n  LR=%s (checkpoint: %s)" % (lr, os.path.basename(last_ckpt)))
        metrics = evaluate_predictions(pred_file)
        metrics["lr"] = lr
        metrics["checkpoint"] = os.path.basename(last_ckpt)
        results.append(metrics)

        print("    F1=%.4f  Prec=%.4f  Rec=%.4f  SER=%.4f  Perfect=%d/%d (%.1f%%)" % (
            metrics["f1"], metrics["precision"], metrics["recall"],
            metrics["SER"], metrics["n_perfect"], metrics["n_examples"],
            metrics["perfect_pct"]))
        print("    TP=%d  FP=%d  FN=%d  S=%d  D=%d  I=%d" % (
            metrics["tp"], metrics["fp"], metrics["fn"],
            metrics["S"], metrics["D"], metrics["I"]))

        # Per-topic summary
        for topic, tm in sorted(metrics["per_topic"].items()):
            print("      %-35s n=%3d F1=%.4f Prec=%.4f Rec=%.4f SER=%.4f" % (
                topic, tm["n"], tm["f1"], tm["precision"], tm["recall"], tm["SER"]))

    if not results:
        print("\nNo results found.")
        return

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("  %-8s %6s %6s %6s %6s %4s %4s %4s %8s" % (
        "LR", "F1", "Prec", "Rec", "SER", "TP", "FP", "FN", "Perfect"))
    print("  " + "-" * 60)
    for r in sorted(results, key=lambda x: x["lr"]):
        print("  %-8s %6.4f %6.4f %6.4f %6.4f %4d %4d %4d %4d/%d" % (
            r["lr"], r["f1"], r["precision"], r["recall"], r["SER"],
            r["tp"], r["fp"], r["fn"], r["n_perfect"], r["n_examples"]))

    # ---- Save ----
    # TSV
    tsv_path = os.path.join(args.output_dir, "lr_sweep_results_repr_dev.tsv")
    with open(tsv_path, "w") as f:
        f.write("lr\tcheckpoint\tn_examples\tf1\tprecision\trecall\tSER\t"
                "tp\tfp\tfn\tS\tD\tI\tN_ref\tn_perfect\tperfect_pct\n")
        for r in sorted(results, key=lambda x: x["lr"]):
            f.write("%.0e\t%s\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.2f\n" % (
                r["lr"], r["checkpoint"], r["n_examples"],
                r["f1"], r["precision"], r["recall"], r["SER"],
                r["tp"], r["fp"], r["fn"],
                r["S"], r["D"], r["I"], r["N_ref"],
                r["n_perfect"], r["perfect_pct"]))
    print("\nResults saved to: %s" % tsv_path)

    # Per-topic TSV
    topic_path = os.path.join(args.output_dir, "lr_sweep_per_topic_repr_dev.tsv")
    with open(topic_path, "w") as f:
        f.write("lr\ttopic\tn\tf1\tprecision\trecall\tSER\ttp\tfp\tfn\n")
        for r in sorted(results, key=lambda x: x["lr"]):
            for topic, tm in sorted(r["per_topic"].items()):
                f.write("%.0e\t%s\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%d\t%d\n" % (
                    r["lr"], topic, tm["n"],
                    tm["f1"], tm["precision"], tm["recall"], tm["SER"],
                    tm["tp"], tm["fp"], tm["fn"]))
    print("Per-topic saved to: %s" % topic_path)

    # JSON summary
    best = max(results, key=lambda x: x["f1"])
    default_results = [r for r in results if abs(r["lr"] - 5e-4) < 1e-6]
    summary = {
        "eval_file": args.pred_filename,
        "n_lr_values": len(results),
        "best_lr": best["lr"],
        "best_f1": best["f1"],
        "best_ser": best["SER"],
        "default_lr": 5e-4,
        "default_f1": default_results[0]["f1"] if default_results else None,
        "default_ser": default_results[0]["SER"] if default_results else None,
        "all_results": [{k: v for k, v in r.items() if k != "per_topic"}
                        for r in sorted(results, key=lambda x: x["lr"])],
    }
    sum_path = os.path.join(args.output_dir, "lr_sweep_summary_repr_dev.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary saved to: %s" % sum_path)


if __name__ == "__main__":
    main()
