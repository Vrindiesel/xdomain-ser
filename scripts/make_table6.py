#!/usr/bin/env python
# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Assemble paper Table 6: rule-based SER tools vs learned methods, six topics.

Rule-based rows are computed here by driving the verbatim aligner functions
(`baselines.{e2e,rnnlg,viggo}_aligner`) over the scored Eval-2 file, exactly
replicating each aligner's own `eval_compute_ser()` loop (verified in the
Stage-2 smoke against the verbatim mains' aggregate output). Learned rows are
read from Table 5's `per_topic_comparison.tsv` (run `reproduce_table5.sh`
first).

Protocol notes (recorded during the 2026-06 correction work; see ERRATA):

1. **Aligner conditioning.** As published, every aligner extracts with the
   slot-value inventory of the *true* MR (`ex["mr"]`), once per text -- the
   same oracle conditioning (P-oracle) as the published learned rows. The
   aligner columns are therefore NOT protocol-invariant under the corrected
   per-pair protocol (P-deploy).
2. **Basis.** As published, aligner rows cover ALL pairs of their topic
   (no dev/test split) while learned rows cover the seed-42 test half.
   This script reproduces that, and additionally emits test-half aligner
   rows (marked `paper_row=no`) for a like-for-like comparison.
3. **RNNLG domain quirk.** The published RNNLG rows for hotel/laptop/tv
   were computed with the extractor's default `domain="restaurant"`
   (`eval_rnnlg_ser.py` main loop never passed the topic). Replicated here
   verbatim.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np

from xdomain_ser.baselines import e2e_aligner, rnnlg_aligner, viggo_aligner
from xdomain_ser.mixture.threshold_sweep import stratified_split
from xdomain_ser.ranking.make_eval_data import tally_ser

# (topic key in the eval file, aligner tool)
TOPICS = [
    ("restaurants", "e2e"),
    ("restaurant", "rnnlg"),
    ("hotel", "rnnlg"),
    ("laptop", "rnnlg"),
    ("tv", "rnnlg"),
    ("video_games", "viggo"),
]
TOOL_NAME = {"e2e": "E2E script", "rnnlg": "RNNLG", "viggo": "ViGGO"}
LEARNED_METHODS = ["LoRA", "NLI", "ScoreRouting", "LR-Routing"]

# The learned method shown for each topic in paper Table 6.
PAPER_PICK = {
    "restaurant": "LR-Routing", "hotel": "LR-Routing", "tv": "ScoreRouting",
    "laptop": "NLI", "restaurants": "LR-Routing", "video_games": "LR-Routing",
}

# Published Table 6 values (gem-ser-main.tex:406-421) for delta reporting:
# topic -> {row_name: (S, D, I, All, MAE)}
PAPER_VALUES = {
    "restaurant":  {"LR-Routing": (.989, 1.00, 1.00, .989, .003),
                    "RNNLG": (.972, .876, .957, .827, .055)},
    "hotel":       {"LR-Routing": (.973, .979, .985, .944, .020),
                    "RNNLG": (.902, .658, .823, .504, .193)},
    "tv":          {"ScoreRouting": (.927, .870, .943, .788, .048),
                    "RNNLG": (.764, .538, .838, .336, .174)},
    "laptop":      {"NLI": (.951, .917, .960, .870, .024),
                    "RNNLG": (.873, .676, .881, .544, .117)},
    "restaurants": {"LR-Routing": (.836, .754, .942, .721, .043),
                    "E2E script": (.881, .730, .940, .699, .046)},
    "video_games": {"LR-Routing": (.892, .868, .934, .753, .071),
                    "ViGGO": (.774, .916, .983, .720, .071)},
}


def select_topic(data, topic, tool):
    """Mirror each verbatim main's example filter."""
    if tool == "e2e":
        return [(i, e) for i, e in enumerate(data)
                if e.get("dataset") == "e2e_nlg"]
    return [(i, e) for i, e in enumerate(data) if e.get("topic") == topic]


def eval_aligner(examples, tool):
    """Replicate the verbatim eval_compute_ser() loop of the given aligner."""
    working = defaultdict(list)
    category = defaultdict(lambda: defaultdict(list))

    for ex in examples:
        ref_mr_raw = ex["mr"]
        text = ex["surface_form"]
        if tool == "e2e":
            ref_mapped = e2e_aligner.pack_e2e_nlg_mr(ref_mr_raw)
            extracted = e2e_aligner.extract_mr(text, ref_mapped)
        elif tool == "rnnlg":
            # Default domain on purpose -- protocol note 3 above.
            ref_mapped = rnnlg_aligner.pack_rnnlg_mr(ref_mr_raw, "restaurant")
            extracted = rnnlg_aligner.extract_mr(text, ref_mapped, "restaurant")
        else:
            ref_mapped = viggo_aligner.map_mr(ref_mr_raw)
            *_, extracted = viggo_aligner.extract_mr(text, ref_mapped)

        for neg in ex["negatives"]:
            _nmr = [[slot] + vals for slot, vals in neg["mr"].items()]
            if tool == "e2e":
                neg_mapped = dict(e2e_aligner.pack_e2e_nlg_mr(_nmr))
            elif tool == "rnnlg":
                neg_mapped = dict(rnnlg_aligner.pack_rnnlg_mr(_nmr, "restaurant"))
            else:
                neg_mapped = dict(viggo_aligner.map_mr(_nmr))
            tally_ser(category, extracted, neg["label"], neg_mapped,
                      dict(ref_mapped), working)

    n = len(working["ser_error"])
    if n == 0:
        return {"S_acc": 0.0, "D_acc": 0.0, "I_acc": 0.0, "all_acc": 0.0,
                "SER_MAE": 0.0, "count": 0}
    out = {k: float(np.mean(working[k]))
           for k in ["S_acc", "D_acc", "I_acc", "all_acc"]}
    out["SER_MAE"] = float(np.mean([abs(e) for e in working["ser_error"]]))
    out["count"] = n
    return out


def load_learned_rows(table5_tsv):
    rows = defaultdict(dict)   # topic -> method -> metrics dict
    with open(table5_tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows[r["topic"]][r["method"]] = {
                "S_acc": float(r["S_acc"]), "D_acc": float(r["D_acc"]),
                "I_acc": float(r["I_acc"]), "all_acc": float(r["All_acc"]),
                "SER_MAE": float(r["SER_MAE"]), "count": int(r["count"]),
            }
    return rows


def fmt_row(topic, kind, name, basis, m, paper_row):
    return {
        "topic": topic, "kind": kind, "name": name, "basis": basis,
        "count": m["count"],
        "S_acc": f"{m['S_acc']:.4f}", "D_acc": f"{m['D_acc']:.4f}",
        "I_acc": f"{m['I_acc']:.4f}", "All_acc": f"{m['all_acc']:.4f}",
        "SER_MAE": f"{m['SER_MAE']:.4f}", "paper_row": paper_row,
    }


def delta_line(topic, name, m):
    ref = PAPER_VALUES.get(topic, {}).get(name)
    if ref is None:
        return None
    got = (m["S_acc"], m["D_acc"], m["I_acc"], m["all_acc"], m["SER_MAE"])
    cells = " | ".join(f"{g:.3f} (Δ{abs(g - r):.3f})"
                       for g, r in zip(got, ref))
    return f"| {topic} | {name} | {cells} |"


def main():
    ap = argparse.ArgumentParser(
        description="Assemble paper Table 6 (rule tools vs learned methods)")
    ap.add_argument("--scores_file",
                    default="data/ranking_eval/"
                            "negatives-v6-test.200.pe5.b10.rexp4.ckpt60.scores.json")
    ap.add_argument("--table5_dir", default="evaluation/results/table5")
    ap.add_argument("--output_dir", default="evaluation/results/table6")
    args = ap.parse_args()
    print("args:", args)

    with open(args.scores_file) as f:
        data = json.load(f)
    _, test_indices = stratified_split(data, seed=42)
    test_set = set(test_indices)

    learned = load_learned_rows(
        os.path.join(args.table5_dir, "per_topic_comparison.tsv"))

    out_rows, delta_rows = [], []
    for topic, tool in TOPICS:
        pairs = select_topic(data, topic, tool)
        full = [e for _, e in pairs]
        half = [e for i, e in pairs if i in test_set]
        tool_name = TOOL_NAME[tool]

        m_full = eval_aligner(full, tool)
        print(f"{topic:>14} {tool_name:>10} full: "
              f"All={m_full['all_acc']:.4f} MAE={m_full['SER_MAE']:.4f} "
              f"(n={m_full['count']})")
        out_rows.append(fmt_row(topic, "rule", tool_name, "full", m_full, "yes"))
        d = delta_line(topic, tool_name, m_full)
        if d:
            delta_rows.append(d)

        m_half = eval_aligner(half, tool)
        out_rows.append(fmt_row(topic, "rule", tool_name, "test", m_half, "no"))

        for method in LEARNED_METHODS:
            m = learned.get(topic, {}).get(method)
            if m is None:
                print(f"  WARNING: no table5 row for {topic}/{method}")
                continue
            is_pick = "yes" if PAPER_PICK[topic] == method else "no"
            out_rows.append(fmt_row(topic, "learned", method, "test", m, is_pick))
            if is_pick == "yes":
                d = delta_line(topic, method, m)
                if d:
                    delta_rows.append(d)

    os.makedirs(args.output_dir, exist_ok=True)
    tsv_path = os.path.join(args.output_dir, "table6.tsv")
    with open(tsv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()),
                           delimiter="\t")
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nTable 6 rows written to {tsv_path}")

    md_path = os.path.join(args.output_dir, "table6.md")
    with open(md_path, "w") as f:
        f.write("# Table 6 reproduction — vs rule-based SER tools "
                "(six topics)\n\n")
        f.write("Values vs published (|delta| in parentheses); "
                "paper rows only.\n\n")
        f.write("| topic | row | S | D | I | All | MAE |\n"
                "|---|---|---|---|---|---|---|\n")
        f.write("\n".join(delta_rows) + "\n\n")
        f.write("## Protocol notes\n\n")
        f.write("1. Aligner rows are per-text, true-MR-inventory conditioned "
                "(P-oracle), as published; not protocol-invariant under "
                "P-deploy.\n")
        f.write("2. As published, aligner rows use ALL pairs of the topic; "
                "learned rows use the seed-42 test half. `basis=test` aligner "
                "rows in table6.tsv are the like-for-like extra (not in the "
                "paper).\n")
        f.write("3. RNNLG rows replicate the published default-domain quirk "
                "(all four topics extracted with domain='restaurant').\n")
    print(f"Rendered summary written to {md_path}")


if __name__ == "__main__":
    main()
