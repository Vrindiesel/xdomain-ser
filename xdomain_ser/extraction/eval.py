# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Evaluate MR-extraction predictions against gold MRs.

Computes per-example and per-topic SER plus slot precision / recall / F1.

With ``--rank_method scores`` consumes ranker probability-weighted scores
to pick the best of k beam candidates; with ``--rank_method f1`` re-ranks
by slot-F1 against the gold (oracle, for upper-bound analysis).
"""
import argparse
import json
import os
from collections import defaultdict, Counter


from xdomain_ser.core import ser


def print_example(case, prediction, ref_mr):
    print("-" * 20)
    print("raw output:", case["pred_mr"])
    print("prediction:", prediction)
    reference_str = "; ".join([f"{slot}: {v}" for slot, vals in ref_mr.items() for v in vals])
    pred_str = "; ".join([f"{slot}: {v}" for slot, vals in prediction.items() for v in vals])
    print(f"Input:\n{case['pred_mr']}\nExtracted: {pred_str}\nReference: {reference_str}\nRef Text: {case['text']}\n")

    res = {
        "raw_output": case["pred_mr"],
        "prediction": prediction,
        "input": case["pred_mr"],
        "exctracted": pred_str,
        "reference": reference_str,
        "ref_text": case["text"],
    }

    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="input path")
    parser.add_argument("--save_path", type=str, default=None, help="")
    parser.add_argument("--rank_method", type=str, default=None, help="None, f1, scores")

    #parser.add_argument("output_path", type=str, help="output path")
    args = parser.parse_args()
    print("args:", args)
    wanted_slots = None
    # read inputs and put them in correct format
    with open(args.input_path, "r") as fin:
        examples = []
        topic_counts = []
        for e in json.load(fin):
            if e.get("predictions") and not e.get("pred_mr"):
                e["pred_mr"] = e["predictions"]
            if e.get("pred_mr"):
                pred_mr = e["pred_mr"][0] if isinstance(e["pred_mr"], list) else e["pred_mr"]
                if args.rank_method == "scores" and e.get("pred_scores"):
                    ranked_preds = sorted([(s, j, p) for j, (p, s) in
                                           enumerate(zip(e["pred_mr"], e["pred_scores"]))], reverse=True)
                    pred_mr = ranked_preds[0][2]
                elif args.rank_method == "f1":
                    ranked_preds = []
                    for j, p in enumerate(e["pred_mr"]):
                        pmr = ser.extract_attributes_dict(p)
                        ref_mr = ser.SlotErrorRate.normalize_mr(e["mr"])
                        f1_score = ser.compute_slot_f1(pmr, ref_mr)
                        ranked_preds.append((f1_score["f1"], j, p))
                    ranked_preds.sort(reverse=True)
                    pred_mr = ranked_preds[0][2]
                text = e["text"] if e.get("text") else e["surface_form"]

                examples.append({
                    "mr": e["mr"],
                    "pred_mr": pred_mr,
                    "text": text,
                    "topic": e["hint_map_id"],
                })
                topic_counts.append(e["hint_map_id"])


    # process each instance
    print("examples:", len(examples))
    counts = Counter()
    topic_ser = defaultdict(Counter)
    topic_f1 = defaultdict(Counter)
    topic_text_lengths = defaultdict(list)
    topic_mr_lengths = defaultdict(list)

    ser_counts = Counter()
    slot_f1_counts = Counter()

    for case in examples:
        ref_mr = ser.SlotErrorRate.normalize_mr(case["mr"])
        pred_mr = case["pred_mr"]
        prediction = ser.extract_attributes_dict(pred_mr) #, wanted_attributes=wanted_slots)
        print_example(case, prediction, ref_mr)
        # new metrics
        ser_score = ser.compute_ser(prediction, ref_mr)
        # compute_ser also returns the dict-valued "errors" slot detail;
        # only the integer counts enter the Counter micro-average.
        ser_micro = {k: ser_score[k] for k in ("S", "D", "I", "N_ref")}
        ser_counts.update(ser_micro)
        topic_ser[case["topic"]].update(ser_micro)

        f1_score = ser.compute_slot_f1(prediction, ref_mr)
        slot_f1_counts.update(f1_score)
        topic_f1[case["topic"]].update(f1_score)

        topic_text_lengths[case["topic"]].append(len(case["text"]))
        topic_mr_lengths[case["topic"]].append(len(case["mr"]))

    for topic, count in topic_ser.items():
        ser_score = ser._compute_ser_val(topic_ser[topic]["D"], topic_ser[topic]["I"], topic_ser[topic]["S"], topic_ser[topic]["N_ref"])
        topic_ser[topic]["SER"] = round(ser_score, 6)

    for topic, count in topic_f1.items():
        f1, precision, recall = ser.calc_p_r_f(count["fn"], count["fp"], count["tp"])
        count["precision"] = round(precision, 6)
        count["recall"] = round(recall, 6)
        count["f1"] = round(f1, 6)

    f1, precision, recall = ser.calc_p_r_f(slot_f1_counts["fn"], slot_f1_counts["fp"], slot_f1_counts["tp"])
    ser_score = ser._compute_ser_val(ser_counts["D"], ser_counts["I"], ser_counts["S"], ser_counts["N_ref"])
    errors = {
        "num_examples": len(examples),
        "topics": Counter(topic_counts),
        "slot_f1":{
            "tp": slot_f1_counts["tp"], "fp": slot_f1_counts["fp"], "fn": slot_f1_counts["fn"],
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        },
        "ser_score":{
            "SER": ser_score,
            "D": ser_counts["D"],
            "I": ser_counts["I"],
            "S": ser_counts["S"],
            "N": ser_counts["N_ref"],
        },
        "per_topic_f1": topic_f1,
        "per_topic_ser": topic_ser,
    }


    print(f"Error Counts:\n{counts}")

    if args.save_path is not None:
        if os.path.exists(args.save_path):
            with open(args.save_path, "r") as fin:
                data = json.load(fin)
        else:
            data = {}
        data[args.input_path] = errors
        with open(args.save_path, "w") as fout:
            json.dump(data, fout, indent=2)


if __name__ == "__main__":
    main()
