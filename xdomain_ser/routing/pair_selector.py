# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Routing under the corrected (P-deploy) protocol, per pair.

A *unit* is one routing decision row:

* ``--protocol deploy`` (default): one unit per (example, negative)
  pair. Ranker scores come from the per-pair scores file
  (``ranking.score --ref_source hypothesis``), NLI results from the
  per-pair NLI file -- every method sees only the pair's modified MR.
  CPU-only (all model outputs are precomputed).
* ``--protocol oracle``: regression mode. One unit per example -- the
  published ``selector.py`` pipeline (per-text features, labels, and
  decisions; live per-text NLI) re-expressed through this module's
  shared evaluation code. The run gates on reproducing the published
  Table-5 numbers exactly (LoRA .7641 / NLI .8051 / ScoreRouting .8614
  at threshold 2.95 / LR-Routing .8681 / Oracle .9073) and exits
  non-zero otherwise.

Both protocols share the label rule (1 = LoRA when
``lora_correct >= nli_correct``), the score-threshold grid, the LR
recipe (via :func:`selector.train_logistic_regression`), and the
phase-0 XGBoost procedure (3 configs x {plain, cost-weighted}, scaled
features, weight = ``|lora_correct - nli_correct| + 0.1``, winner by
test All_acc -- mirrored deliberately, selection criterion recorded).
Evaluation always tallies pairs.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import StandardScaler

from xdomain_ser.core import ser
from xdomain_ser.mixture.threshold_sweep import stratified_split
from xdomain_ser.nli.evaluator import (
    collect_all_nli_pairs,
    compute_metrics,
    recover_mr_from_nli,
    tally_ser,
)
from xdomain_ser.ranking.make_eval_data import load_pair_scores, select_ranking_preds
from xdomain_ser.routing.features import (
    compute_pair_routing_features,
    compute_routing_features,
)
from xdomain_ser.routing.persist import save_routers
from xdomain_ser.routing.selector import (
    SCORE_THRESHOLDS,
    print_comparison,
    save_results,
    train_logistic_regression,
)

# Published Table-5 values the oracle regression must reproduce.
PUBLISHED_ORACLE = {"LoRA": 0.7641, "NLI": 0.8051, "ScoreRouting": 0.8614,
                    "LR-Routing": 0.8681, "Oracle": 0.9073}
PUBLISHED_THRESHOLD = 2.95
PUBLISHED_XGB = 0.9004  # phase-0 follow-up; informational (+/-0.01)


def load_pair_nli(path):
    """Load per-pair NLI results (output of planning/run_stage8_nli_pairs.py).

    Returns dict ``{(ex_idx, neg_idx): [(slot, value, prob), ...]}``;
    pairs with no templatable slot-value are absent (callers default
    to []).
    """
    with open(path) as f:
        obj = json.load(f)
    return {(r["ex_idx"], r["neg_idx"]): [tuple(x) for x in r["results"]]
            for r in obj["records"]}


def _pair_results(ex, gold_mr_dict, lora_mr, nli_mr, negatives):
    """SER results of both methods on the given negatives.

    Returns (pairs, lora_correct, nli_correct) where each pair carries
    the precomputed ref/lora/nli ``compute_ser`` results for tallying.
    """
    topic = ex.get("topic", "unknown")
    pairs, lora_c, nli_c = [], 0, 0
    for neg in negatives:
        ref = ser.compute_ser(gold_mr_dict, neg["mr"])
        lor = ser.compute_ser(lora_mr, neg["mr"])
        nli = ser.compute_ser(nli_mr, neg["mr"])
        l_ok = all(ref[k] == lor[k] for k in ("S", "D", "I"))
        n_ok = all(ref[k] == nli[k] for k in ("S", "D", "I"))
        lora_c += l_ok
        nli_c += n_ok
        pairs.append({"ref": ref, "lora": lor, "nli": nli,
                      "neg_label": neg["label"], "topic": topic})
    return pairs, lora_c, nli_c


def build_units_oracle(data, nli_threshold, batch_size, device):
    """Published per-text protocol: one unit per example (live NLI)."""
    from xdomain_ser.nli.model import NLIModel

    nli_model = NLIModel(device=device)
    all_pairs, pair_index = collect_all_nli_pairs(data)
    print(f"Per-text NLI pairs: {len(all_pairs)}")
    probs = nli_model.batch_entailment(all_pairs, batch_size=batch_size)
    example_nli_results = defaultdict(list)
    for (ex_idx, slot, value), prob in zip(pair_index, probs):
        example_nli_results[ex_idx].append((slot, value, prob))
    del nli_model

    all_topics = sorted(set(ex.get("topic", "unknown") for ex in data))
    units = {}
    for i, ex in enumerate(data):
        gold = ser.mr_list_to_dict(ex["mr"])
        lora_mr = select_ranking_preds(ex["pred_mr"], ex["pred_scores"])
        nli_mr = recover_mr_from_nli(gold, example_nli_results.get(i, []),
                                     nli_threshold)
        feats = compute_routing_features(ex, i, example_nli_results, all_topics)
        pairs, lora_c, nli_c = _pair_results(ex, gold, lora_mr, nli_mr,
                                             ex["negatives"])
        units[i] = {"features": feats, "label": 1 if lora_c >= nli_c else 0,
                    "lora_count": lora_c, "nli_count": nli_c, "pairs": pairs}
    return units


def build_units_deploy(data, pair_scores, pair_nli, nli_threshold):
    """Corrected per-pair protocol: one unit per (example, negative)."""
    all_topics = sorted(set(ex.get("topic", "unknown") for ex in data))
    units = {}
    for i, ex in enumerate(data):
        gold = ser.mr_list_to_dict(ex["mr"])
        for j, neg in enumerate(ex["negatives"]):
            scores = pair_scores[(i, j)]
            nli_res = pair_nli.get((i, j), [])
            hyp_mr_dict = {s: (v if isinstance(v, list) else [v])
                           for s, v in neg["mr"].items()}
            lora_mr = select_ranking_preds(ex["pred_mr"], scores)
            nli_mr = recover_mr_from_nli(hyp_mr_dict, nli_res, nli_threshold)
            feats = compute_pair_routing_features(ex, neg, scores, nli_res,
                                                  all_topics)
            pairs, lora_c, nli_c = _pair_results(ex, gold, lora_mr, nli_mr,
                                                 [neg])
            units[(i, j)] = {"features": feats,
                             "label": 1 if lora_c >= nli_c else 0,
                             "lora_count": lora_c, "nli_count": nli_c,
                             "pairs": pairs}
    return units


def evaluate_decisions(units, keys, decisions):
    """Tally pair-level results using each unit's routed method."""
    working = defaultdict(list)
    category = defaultdict(lambda: defaultdict(list))
    topic_res = defaultdict(lambda: defaultdict(list))
    for k in keys:
        method = "lora" if decisions[k] else "nli"
        for p in units[k]["pairs"]:
            tally_ser(working, category, topic_res,
                      p["ref"], p[method], p["neg_label"], p["topic"])
    return working, category, topic_res


def evaluate_fixed(units, keys, method):
    """Tally pair-level results for a single method on all units."""
    return evaluate_decisions(units, keys, {k: 1 if method == "lora" else 0
                                            for k in keys})


def threshold_sweep(units, dev_keys):
    """Sweep the score-routing threshold on dev units (pair-level all_acc)."""
    sweep_results, best_all, best_t = [], -1.0, None
    for t in SCORE_THRESHOLDS:
        decisions = {k: 1 if units[k]["features"]["top_score"] >= t else 0
                     for k in dev_keys}
        w, _, _ = evaluate_decisions(units, dev_keys, decisions)
        m = compute_metrics(w)
        n_lora = sum(decisions[k] for k in dev_keys)
        sweep_results.append({"threshold": t, "n_lora": n_lora,
                              "n_nli": len(dev_keys) - n_lora, **m})
        if m["all_acc"] > best_all:
            best_all, best_t = m["all_acc"], t
    return sweep_results, best_t


def train_xgb_panel(units, dev_keys, test_keys, feature_names):
    """Phase-0 XGBoost procedure on unit rows: 3 configs x {plain, CW},
    scaled features, winner by test All_acc (the phase-0 criterion)."""
    from xdomain_ser.routing.experimental.xgboost_routing import (
        XGB_CONFIGS,
        train_xgboost,
    )

    X_dev = np.array([[units[k]["features"][f] for f in feature_names]
                      for k in dev_keys])
    y_dev = np.array([units[k]["label"] for k in dev_keys])
    X_test = np.array([[units[k]["features"][f] for f in feature_names]
                       for k in test_keys])
    scaler = StandardScaler()
    X_dev_s = scaler.fit_transform(X_dev)
    X_test_s = scaler.transform(X_test)
    weights = np.array([abs(units[k]["lora_count"] - units[k]["nli_count"]) + 0.1
                        for k in dev_keys])

    panel = []
    for cfg in XGB_CONFIGS:
        cfg_str = f"d{cfg['max_depth']}_n{cfg['n_estimators']}_lr{cfg['learning_rate']}"
        for cw, sw in [(False, None), (True, weights)]:
            model, preds, dev_acc = train_xgboost(
                X_dev_s, y_dev, X_test_s, test_keys, cfg, sample_weights=sw)
            w, _, _ = evaluate_decisions(units, test_keys, preds)
            m = compute_metrics(w)
            panel.append({"name": f"XGB_{cfg_str}{'_CW' if cw else ''}",
                          "config": cfg, "cost_weighted": cw,
                          "dev_acc": dev_acc, "metrics": m, "preds": preds,
                          "model": model})
            print(f"  {panel[-1]['name']}: dev_acc={dev_acc:.4f}, "
                  f"test All_acc={m['all_acc']:.4f}")
    best = max(panel, key=lambda r: r["metrics"]["all_acc"])
    return panel, best, scaler


def write_pair_feature_table(path, units, feature_names):
    """Dump the per-pair feature/label table (the Stage-9 deliverable)."""
    with open(path, "w") as f:
        f.write("ex_idx\tneg_idx\tlabel\tlora_correct\tnli_correct\t"
                + "\t".join(feature_names) + "\n")
        for (i, j) in sorted(units):
            u = units[(i, j)]
            feats = "\t".join(f"{u['features'][n]:.6f}" for n in feature_names)
            f.write(f"{i}\t{j}\t{u['label']}\t{u['lora_count']}\t"
                    f"{u['nli_count']}\t{feats}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Per-pair (P-deploy) routing pipeline with a published "
                    "(P-oracle) regression mode")
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--protocol", choices=["deploy", "oracle"],
                        default="deploy")
    parser.add_argument("--pair_scores", default=None,
                        help="Per-pair ranker scores JSON (deploy mode).")
    parser.add_argument("--pair_nli", default=None,
                        help="Per-pair NLI results JSON (deploy mode).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--persist_dir", default=None,
                        help="Persist fitted routers here (deploy mode).")
    parser.add_argument("--nli_threshold", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="First N examples (smoke runs).")
    args = parser.parse_args()
    print("args:", args)

    with open(args.eval_file) as f:
        data = json.load(f)
    if args.limit:
        data = data[:args.limit]
    print(f"Examples: {len(data)}; pairs: {sum(len(e['negatives']) for e in data)}")

    dev_idx, test_idx = stratified_split(data, seed=args.seed)

    if args.protocol == "deploy":
        if not (args.pair_scores and args.pair_nli):
            parser.error("deploy protocol requires --pair_scores and --pair_nli")
        pair_scores = load_pair_scores(args.pair_scores)
        pair_nli = load_pair_nli(args.pair_nli)
        units = build_units_deploy(data, pair_scores, pair_nli,
                                   args.nli_threshold)
        dev_keys = [(i, j) for i in dev_idx
                    for j in range(len(data[i]["negatives"]))]
        test_keys = [(i, j) for i in test_idx
                     for j in range(len(data[i]["negatives"]))]
    else:
        units = build_units_oracle(data, args.nli_threshold, args.batch_size,
                                   args.device)
        dev_keys, test_keys = list(dev_idx), list(test_idx)

    print(f"Units: {len(units)} ({args.protocol}); "
          f"dev {len(dev_keys)} / test {len(test_keys)}")

    sample = units[dev_keys[0]]["features"]
    feature_names = [f for f in sample if not f.startswith("topic_")]
    feature_names += sorted(f for f in sample if f.startswith("topic_"))

    n_lora_lab = sum(units[k]["label"] for k in dev_keys)
    print(f"Dev labels: {n_lora_lab} LoRA-preferred, "
          f"{len(dev_keys) - n_lora_lab} NLI-preferred")

    # --- Score-threshold routing ---
    sweep_results, best_threshold = threshold_sweep(units, dev_keys)
    print(f"Best score threshold: {best_threshold}")
    score_test = {k: 1 if units[k]["features"]["top_score"] >= best_threshold
                  else 0 for k in test_keys}

    # --- LR routing (published recipe) ---
    feature_dicts = {k: units[k]["features"] for k in units}
    labels = {k: units[k]["label"] for k in units}
    lr_model, lr_scaler, lr_dev_acc, lr_test_preds, lr_importance = \
        train_logistic_regression(feature_dicts, labels, dev_keys, test_keys,
                                  feature_names)
    print(f"LR dev accuracy: {lr_dev_acc:.4f}")

    # --- XGBoost routing (phase-0 procedure) ---
    xgb_panel, xgb_best, xgb_scaler = train_xgb_panel(
        units, dev_keys, test_keys, feature_names)

    # --- Test-split comparison ---
    rows = []
    topic_results = {}
    for name, w_c_t in [
        ("LoRA", evaluate_fixed(units, test_keys, "lora")),
        ("NLI", evaluate_fixed(units, test_keys, "nli")),
        ("ScoreRouting", evaluate_decisions(units, test_keys, score_test)),
        ("LR-Routing", evaluate_decisions(units, test_keys, lr_test_preds)),
        ("XGB-Routing", evaluate_decisions(units, test_keys,
                                           xgb_best["preds"])),
        ("Oracle", evaluate_decisions(units, test_keys,
                                      {k: units[k]["label"]
                                       for k in test_keys})),
    ]:
        w, _, t = w_c_t
        rows.append((name, compute_metrics(w)))
        topic_results[name] = t
    print_comparison(rows, title=f"COMPARISON (test, protocol={args.protocol})")
    print(f"  (XGB-Routing = {xgb_best['name']}, phase-0 selection: "
          f"best test All_acc)")

    methods = ["LoRA", "NLI", "ScoreRouting", "LR-Routing", "XGB-Routing"]
    routing_details = [{
        "key": list(k) if isinstance(k, tuple) else k,
        "topic": units[k]["pairs"][0]["topic"],
        "features": units[k]["features"],
        "oracle_label": units[k]["label"],
        "score_routing_choice": score_test[k],
        "lr_routing_choice": lr_test_preds[k],
        "xgb_routing_choice": xgb_best["preds"][k],
        "lora_correct": units[k]["lora_count"],
        "nli_correct": units[k]["nli_count"],
        "n_pairs": len(units[k]["pairs"]),
    } for k in test_keys]

    save_results(args.output_dir, sweep_results, best_threshold,
                 lr_importance, rows, topic_results, methods,
                 routing_details, lr_dev_acc)

    with open(os.path.join(args.output_dir, "xgb_comparison.tsv"), "w") as f:
        f.write("name\tcost_weighted\tdev_acc\tAll_acc\tSER_MAE\n")
        for r in xgb_panel:
            f.write(f"{r['name']}\t{r['cost_weighted']}\t{r['dev_acc']:.4f}\t"
                    f"{r['metrics']['all_acc']:.4f}\t"
                    f"{r['metrics']['SER_MAE']:.4f}\n")

    if args.protocol == "deploy":
        write_pair_feature_table(
            os.path.join(args.output_dir, "pair_features.tsv"),
            units, feature_names)
        print(f"Pair feature table: "
              f"{os.path.join(args.output_dir, 'pair_features.tsv')}")
        if args.persist_dir:
            meta = {"protocol": "deploy (corrected, per-pair)",
                    "fitted_on": "corrected dev split (seed 42)",
                    "eval_file": args.eval_file,
                    "pair_scores": args.pair_scores,
                    "pair_nli": args.pair_nli,
                    "nli_threshold": args.nli_threshold,
                    "xgb_selection": "best test All_acc (phase-0 procedure)",
                    "xgb_name": xgb_best["name"]}
            paths = save_routers(
                args.persist_dir, score_threshold=best_threshold,
                lr_model=lr_model, lr_scaler=lr_scaler,
                xgb_model=xgb_best["model"], xgb_scaler=xgb_scaler,
                xgb_config=xgb_best["config"], feature_names=feature_names,
                meta=meta)
            print("Persisted routers:", *paths, sep="\n  ")

    if args.protocol == "oracle":
        print("\n" + "=" * 70)
        print("P-ORACLE REGRESSION vs published Table 5")
        print("=" * 70)
        got = {name: m["all_acc"] for name, m in rows}
        failures = []
        for name, want in PUBLISHED_ORACLE.items():
            have = round(got[name], 4)
            ok = have == want
            print(f"  {name:>13}: {have:.4f}  published {want:.4f}  "
                  f"{'OK' if ok else 'MISMATCH'}")
            if not ok:
                failures.append(name)
        thr_ok = best_threshold == PUBLISHED_THRESHOLD
        print(f"  {'threshold':>13}: {best_threshold}  published "
              f"{PUBLISHED_THRESHOLD}  {'OK' if thr_ok else 'MISMATCH'}")
        if not thr_ok:
            failures.append("threshold")
        xgb_acc = got["XGB-Routing"]
        print(f"  {'XGB (info)':>13}: {xgb_acc:.4f}  phase-0 {PUBLISHED_XGB} "
              f"(+/-0.01, test-selected; informational)")
        if failures:
            print(f"REGRESSION FAIL: {failures}")
            raise SystemExit(1)
        print("REGRESSION PASS")


if __name__ == "__main__":
    main()
