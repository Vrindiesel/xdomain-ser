# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""SER-delta evaluation for the MR ranker.

For an input file of examples with ``pred_mr`` (k candidates) and
``negatives`` (gold-graded MRs), compute downstream SER-fidelity metrics
across two ranking modes:

* ``ranking_model`` -- pick the candidate with the highest ranker score
  (production path; uses ``ex['pred_scores']``)
* ``none`` -- pick the first candidate (top of beam search, no re-ranking)

Reports S/D/I-error accuracy (matching the rule-based aligner's S/D/I
counts) plus SER MSE / MAE, both overall and per-category and per-topic.
``select_ranking_preds`` is the public helper used by
``xdomain_ser.extraction.experimental.lr_sweep``.
"""
from collections import defaultdict
import argparse
import json

import numpy as np

from xdomain_ser.core import ser


def select_ranking_preds(pred_mrs, pred_scores, build_attr_dict=True, return_score=False):
    if build_attr_dict:
        pred_mrs = [ser.extract_attributes_dict(pmr) for pmr in pred_mrs]
    pred_mrs = [(score, i, mr) for i, (mr, score) in enumerate(zip(pred_mrs, pred_scores))]
    pred_mrs.sort(reverse=True)

    retval = (pred_mrs[0][2], pred_mrs[0][0]) if return_score else pred_mrs[0][2]
    return retval


def load_pair_scores(path):
    """Load per-pair ranker scores (output of ranking.score --ref_source hypothesis).

    Returns dict ``{(ex_idx, neg_idx): [k floats]}``. Keys are positional:
    they are only valid against the same eval file (in the same order) the
    scores were produced from, and a scores file from a ``--limit`` run
    covers only that prefix of the examples.
    """
    with open(path) as f:
        obj = json.load(f)
    return {(r["ex_idx"], r["neg_idx"]): r["pred_scores"] for r in obj["records"]}


def select_f1_scores(pred_mrs, ref_mr):
    ranked_preds = []
    for j, p in enumerate(pred_mrs):
        pmr = ser.extract_attributes_dict(p)
        f1_score = ser.compute_slot_f1(pmr, ser.SlotErrorRate.normalize_mr(ref_mr))
        ranked_preds.append((f1_score["f1"], j, pmr))
    ranked_preds.sort(reverse=True)
    return ranked_preds[0][2]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--pair_scores", default=None,
                        help="Per-pair scores JSON (ranking.score --ref_source hypothesis); "
                             "adds the 'ranking_model_per_pair' selection mode.")
    args = parser.parse_args()
    print("args:", args)


    with open(args.input_path) as fin:
        data = json.load(fin)

    rank_modes = ["ranking_model", "none"]
    pair_scores = None
    if args.pair_scores:
        pair_scores = load_pair_scores(args.pair_scores)
        rank_modes.append("ranking_model_per_pair")

    for rank_name in rank_modes:

        working_results = defaultdict(list)
        category_results = defaultdict(lambda: defaultdict(list))
        for j, ex in enumerate(data):
            text = ex["surface_form"]
            ref_mr = ser.mr_list_to_dict(ex["mr"])
            negatives = ex["negatives"]
            for neg_idx, neg_example in enumerate(negatives):
                neg_category = neg_example["label"]
                neg_mr = neg_example["mr"]
                if rank_name == "ranking_model":
                    model_mr = select_ranking_preds(ex["pred_mr"], ex["pred_scores"])
                elif rank_name == "ranking_model_per_pair":
                    # Corrected protocol: selection conditioned on the pair's
                    # modified MR (follows the dormant f1_ranking per-pair
                    # precedent below).
                    model_mr = select_ranking_preds(ex["pred_mr"],
                                                    pair_scores[(j, neg_idx)])
                elif rank_name == "f1_ranking":
                    model_mr = select_f1_scores(ex["pred_mr"], neg_mr)
                else:
                    model_mr = ser.extract_attributes_dict(ex["pred_mr"][0])

                tally_ser(category_results, model_mr, neg_category, neg_mr, ref_mr, working_results)

        print(f"\nranking method {rank_name} Results:")
        print_results(category_results, working_results)


    for topic in ["video_games", "laptop", "hotel", "tv", "restaurant", "e2e_nlg"]:
        examples = [e for e in data if e.get("topic") == topic or e.get("dataset") == topic]
        working_results = defaultdict(list)
        category_results = defaultdict(lambda: defaultdict(list))
        for j, ex in enumerate(examples):
            text = ex["surface_form"]
            ref_mr = ser.mr_list_to_dict(ex["mr"])
            negatives = ex["negatives"]
            for neg_example in negatives:
                neg_category = neg_example["label"]
                neg_mr = neg_example["mr"]
                model_mr = select_ranking_preds(ex["pred_mr"], ex["pred_scores"])
                tally_ser(category_results, model_mr, neg_category, neg_mr, ref_mr, working_results)
        print(f"\nTopic {topic} Results:")
        print_results(category_results, working_results)


def print_results(category_results, working_results):

    for name in ["S_acc", "D_acc", "I_acc", "all_acc"]:
        working_results[name] = np.mean(working_results[name])
        print(f"{name}:", working_results[name])

    mse_ser = np.mean([e ** 2 for e in working_results["ser_error"]])
    print("SER mean square error:", mse_ser)
    mabs_ser = np.mean([np.abs(e) for e in working_results["ser_error"]])
    print("SER mean absolute error:", mabs_ser)

    print()
    for category, results in category_results.items():
        print(f"\ncategory {category}:")
        for name in ["S_acc", "D_acc", "I_acc", "all_acc"]:
            n = len(results[name])
            results[name] = np.mean(results[name])
            print(f"{name} ({n}):", results[name])

        mse_ser = np.mean([e ** 2 for e in results["ser_error"]])
        n = len(results["ser_error"])
        print(f"SER mean square error ({n}):", mse_ser)
        mabs_ser = np.mean([np.abs(e) for e in results["ser_error"]])
        print("SER mean absolute error:", mabs_ser)


def tally_ser(category_results, model_mr, neg_category, neg_mr, ref_mr, working_results):
    ref_result = ser.compute_ser(ref_mr, neg_mr)
    pred_result = ser.compute_ser(model_mr, neg_mr)
    working_results["S_acc"].append(ref_result["S"] == pred_result["S"])
    working_results["D_acc"].append(ref_result["D"] == pred_result["D"])
    working_results["I_acc"].append(ref_result["I"] == pred_result["I"])
    all_acc = (ref_result["S"] == pred_result["S"] and ref_result["D"] == pred_result["D"] and
               ref_result["I"] == pred_result["I"])
    working_results["all_acc"].append(all_acc)
    working_results["ser_error"].append(ref_result["SER"] - pred_result["SER"])

    category_results[neg_category]["S_acc"].append(ref_result["S"] == pred_result["S"])
    category_results[neg_category]["D_acc"].append(ref_result["D"] == pred_result["D"])
    category_results[neg_category]["I_acc"].append(ref_result["I"] == pred_result["I"])
    all_acc = (ref_result["S"] == pred_result["S"] and ref_result["D"] == pred_result["D"] and
               ref_result["I"] == pred_result["I"])
    category_results[neg_category]["all_acc"].append(all_acc)

    category_results[neg_category]["ser_error"].append(ref_result["SER"] - pred_result["SER"])


if __name__ == "__main__":
    main()
