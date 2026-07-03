# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""[PHASE-0 / POST-PUBLICATION FOLLOW-UP -- documented negative result]

DeBERTa NLI model swap: replace ``roberta-large-mnli`` with
``cross-encoder/nli-deberta-v3-large``. Re-runs the full NLI evaluation
and routing pipelines side-by-side.

**Result: DeBERTa underperforms RoBERTa for our slot-template hypotheses
on this evaluation surface.** The phase-0 sections of RELEASE_NOTES.md
record the full context. This module ships under
``experimental/`` as a documented negative result rather than a
recommended drop-in NLI replacement.

Auto-detects the entailment label index from ``model.config.id2label``
(DeBERTa cross-encoder checkpoints have different label mappings than
RoBERTa-MNLI).

Stage-6 imports (``mixture``, ``routing``) are deferred into ``main()``
so importing this module does not fail before Stage 6 lands.
"""
import argparse
import json
import os
from collections import defaultdict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

from xdomain_ser.core import ser
from xdomain_ser.nli.evaluator import (
    collect_all_nli_pairs,
    recover_mr_from_nli,
    tally_ser,
    compute_metrics,
)


class FlexibleNLIModel:
    """
    NLI model wrapper that auto-detects the entailment label index
    from model.config.id2label.
    """

    def __init__(self, model_name, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_name = model_name

        print(f"Loading NLI model: {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        id2label = self.model.config.id2label
        print(f"  Label mapping: {id2label}")
        self.entailment_idx = None
        for idx, label in id2label.items():
            if label.lower() in ("entailment", "entail"):
                self.entailment_idx = int(idx)
                break
        if self.entailment_idx is None:
            print("  WARNING: Could not find 'entailment' label, defaulting to index 2")
            self.entailment_idx = 2
        print(f"  Entailment index: {self.entailment_idx}")

    def batch_entailment(self, pairs, batch_size=32, show_progress=True):
        all_probs = []
        n_batches = (len(pairs) + batch_size - 1) // batch_size
        iterator = range(0, len(pairs), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches,
                            desc=f"NLI inference ({self.model_name})")

        for i in iterator:
            batch = pairs[i:i + batch_size]
            premises = [p for p, h in batch]
            hypotheses = [h for p, h in batch]

            inputs = self.tokenizer(
                premises, hypotheses,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, self.entailment_idx]
            all_probs.extend(probs.cpu().tolist())

        return all_probs


def run_nli_evaluation(data, indices, example_nli_results, nli_threshold):
    """
    Evaluate NLI method on a subset using precomputed NLI results.

    Returns (working_results, category_results, topic_results).
    """
    working = defaultdict(list)
    category = defaultdict(lambda: defaultdict(list))
    topic_res = defaultdict(lambda: defaultdict(list))

    for ex_idx in indices:
        ex = data[ex_idx]
        gold_mr_dict = ser.mr_list_to_dict(ex["mr"])
        topic = ex.get("topic", "unknown")
        nli_results = example_nli_results.get(ex_idx, [])
        nli_mr = recover_mr_from_nli(gold_mr_dict, nli_results, nli_threshold)

        for neg in ex["negatives"]:
            neg_label = neg["label"]
            neg_mr = neg["mr"]
            ref_result = ser.compute_ser(gold_mr_dict, neg_mr)
            pred_result = ser.compute_ser(nli_mr, neg_mr)
            tally_ser(working, category, topic_res,
                      ref_result, pred_result, neg_label, topic)

    return working, category, topic_res


# NLI threshold sweep grid
NLI_THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def main():
    # Deferred imports: these modules migrate in Stage 6 (mixture, routing).
    # Importing them at module level would prevent loading this script in
    # any environment that hasn't yet completed Stage 6 of the migration.
    from xdomain_ser.ranking.make_eval_data import select_ranking_preds
    from xdomain_ser.mixture.threshold_sweep import stratified_split  # noqa: F401
    from xdomain_ser.routing.selector import (
        extract_features,
        compute_per_example_label,
        evaluate_routing_on_subset,
        score_threshold_sweep,
        train_logistic_regression,
        print_comparison,
    )

    parser = argparse.ArgumentParser(
        description="DeBERTa NLI model swap (phase-0 negative result)"
    )
    parser.add_argument("--eval_file", required=True,
                        help="Path to ranking-eval JSON with pred_mr / pred_scores / negatives.")
    parser.add_argument("--output_dir", default="results/deberta_nli")
    parser.add_argument(
        "--deberta_model", default="cross-encoder/nli-deberta-v3-large",
        help="DeBERTa NLI model name.",
    )
    parser.add_argument("--nli_threshold", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load data ----
    print(f"Loading eval data from: {args.eval_file}")
    with open(args.eval_file) as f:
        data = json.load(f)
    if args.max_examples:
        data = data[:args.max_examples]
    print(f"Loaded {len(data)} examples.")

    dev_indices, test_indices = stratified_split(data, seed=args.seed)
    all_topics = sorted(set(ex.get("topic", "unknown") for ex in data))
    print(f"Dev: {len(dev_indices)}, Test: {len(test_indices)}")

    # ---- Collect NLI pairs (same for both models) ----
    all_pairs, pair_index = collect_all_nli_pairs(data)
    print(f"Total NLI pairs: {len(all_pairs)}")

    # ---- RoBERTa baseline ----
    print("\n" + "=" * 90)
    print("ROBERTA-LARGE-MNLI (Baseline)")
    print("=" * 90)

    roberta_model = FlexibleNLIModel("roberta-large-mnli", device=args.device)
    roberta_probs = roberta_model.batch_entailment(
        all_pairs, batch_size=args.batch_size)

    roberta_nli_results = defaultdict(list)
    for (ex_idx, slot, value), prob in zip(pair_index, roberta_probs):
        roberta_nli_results[ex_idx].append((slot, value, prob))

    del roberta_model
    torch.cuda.empty_cache()

    # ---- DeBERTa ----
    print("\n" + "=" * 90)
    print(f"DEBERTA ({args.deberta_model})")
    print("=" * 90)

    deberta_model = FlexibleNLIModel(args.deberta_model, device=args.device)
    deberta_probs = deberta_model.batch_entailment(
        all_pairs, batch_size=args.batch_size)

    deberta_nli_results = defaultdict(list)
    for (ex_idx, slot, value), prob in zip(pair_index, deberta_probs):
        deberta_nli_results[ex_idx].append((slot, value, prob))

    del deberta_model
    torch.cuda.empty_cache()

    # ---- Threshold sweep for DeBERTa on dev ----
    print("\nSweeping NLI thresholds on dev for DeBERTa...")
    deberta_sweep = []
    best_deberta_threshold = args.nli_threshold
    best_deberta_acc = -1

    for t in NLI_THRESHOLDS:
        w, c, tr = run_nli_evaluation(
            data, dev_indices, deberta_nli_results, t)
        m = compute_metrics(w)
        deberta_sweep.append({"threshold": t, **m})
        if m["all_acc"] > best_deberta_acc:
            best_deberta_acc = m["all_acc"]
            best_deberta_threshold = t

    print(f"Best DeBERTa threshold on dev: {best_deberta_threshold} "
          f"(All_acc={best_deberta_acc:.4f})")

    # ---- Evaluate both on test ----
    print("\n" + "=" * 90)
    print("NLI COMPARISON (test split)")
    print("=" * 90)

    # RoBERTa NLI on test
    rob_w, rob_c, rob_t = run_nli_evaluation(
        data, test_indices, roberta_nli_results, args.nli_threshold)
    rob_m = compute_metrics(rob_w)

    # DeBERTa NLI on test (best threshold)
    deb_w, deb_c, deb_t = run_nli_evaluation(
        data, test_indices, deberta_nli_results, best_deberta_threshold)
    deb_m = compute_metrics(deb_w)

    # DeBERTa NLI on test (same threshold as RoBERTa for fair comparison)
    deb_same_w, _, _ = run_nli_evaluation(
        data, test_indices, deberta_nli_results, args.nli_threshold)
    deb_same_m = compute_metrics(deb_same_w)

    # LoRA baseline
    lora_w = defaultdict(list)
    lora_c = defaultdict(lambda: defaultdict(list))
    lora_t_res = defaultdict(lambda: defaultdict(list))
    for ex_idx in test_indices:
        ex = data[ex_idx]
        gold_mr_dict = ser.mr_list_to_dict(ex["mr"])
        topic = ex.get("topic", "unknown")
        lora_mr = select_ranking_preds(ex["pred_mr"], ex["pred_scores"])
        for neg in ex["negatives"]:
            ref_result = ser.compute_ser(gold_mr_dict, neg["mr"])
            pred_result = ser.compute_ser(lora_mr, neg["mr"])
            tally_ser(lora_w, lora_c, lora_t_res,
                      ref_result, pred_result, neg["label"], topic)
    lora_m = compute_metrics(lora_w)

    nli_comparison = [
        ("LoRA", lora_m),
        ("RoBERTa-NLI", rob_m),
        (f"DeBERTa-NLI(t={args.nli_threshold})", deb_same_m),
        (f"DeBERTa-NLI(t={best_deberta_threshold})", deb_m),
    ]
    print_comparison(nli_comparison, title="NLI MODEL COMPARISON (test split)")

    # ---- Routing with DeBERTa NLI ----
    print("\n" + "=" * 90)
    print("ROUTING WITH DEBERTA NLI")
    print("=" * 90)

    # Extract features using DeBERTa NLI results
    deberta_feature_dicts = {}
    deberta_labels = {}
    for ex_idx, ex in enumerate(data):
        deberta_feature_dicts[ex_idx] = extract_features(
            ex, ex_idx, deberta_nli_results, all_topics
        )
        label, _, _ = compute_per_example_label(
            ex, ex_idx, deberta_nli_results, best_deberta_threshold
        )
        deberta_labels[ex_idx] = label

    feature_names_sample = deberta_feature_dicts[0]
    feature_names = [f for f in feature_names_sample
                     if not f.startswith("topic_")]
    feature_names += sorted(f for f in feature_names_sample
                            if f.startswith("topic_"))

    # Score-threshold routing with DeBERTa
    deb_sweep, deb_best_thresh = score_threshold_sweep(
        data, dev_indices, deberta_nli_results, best_deberta_threshold,
        deberta_feature_dicts
    )
    deb_score_routing = {
        idx: (1 if deberta_feature_dicts[idx]["top_score"] >= deb_best_thresh else 0)
        for idx in test_indices
    }
    deb_score_w, _, _ = evaluate_routing_on_subset(
        data, test_indices, deberta_nli_results, best_deberta_threshold,
        deb_score_routing
    )
    deb_score_m = compute_metrics(deb_score_w)

    # LR routing with DeBERTa
    _, _, deb_lr_dev_acc, deb_lr_preds, _ = train_logistic_regression(
        deberta_feature_dicts, deberta_labels, dev_indices, test_indices,
        feature_names
    )
    deb_lr_w, _, _ = evaluate_routing_on_subset(
        data, test_indices, deberta_nli_results, best_deberta_threshold,
        deb_lr_preds
    )
    deb_lr_m = compute_metrics(deb_lr_w)

    # Also get RoBERTa routing baselines for comparison
    roberta_feature_dicts = {}
    roberta_labels = {}
    for ex_idx, ex in enumerate(data):
        roberta_feature_dicts[ex_idx] = extract_features(
            ex, ex_idx, roberta_nli_results, all_topics
        )
        label, _, _ = compute_per_example_label(
            ex, ex_idx, roberta_nli_results, args.nli_threshold
        )
        roberta_labels[ex_idx] = label

    rob_sweep, rob_best_thresh = score_threshold_sweep(
        data, dev_indices, roberta_nli_results, args.nli_threshold,
        roberta_feature_dicts
    )
    rob_score_routing = {
        idx: (1 if roberta_feature_dicts[idx]["top_score"] >= rob_best_thresh else 0)
        for idx in test_indices
    }
    rob_score_w, _, _ = evaluate_routing_on_subset(
        data, test_indices, roberta_nli_results, args.nli_threshold,
        rob_score_routing
    )
    rob_score_m = compute_metrics(rob_score_w)

    _, _, rob_lr_dev_acc, rob_lr_preds, _ = train_logistic_regression(
        roberta_feature_dicts, roberta_labels, dev_indices, test_indices,
        feature_names
    )
    rob_lr_w, _, _ = evaluate_routing_on_subset(
        data, test_indices, roberta_nli_results, args.nli_threshold,
        rob_lr_preds
    )
    rob_lr_m = compute_metrics(rob_lr_w)

    routing_comparison = [
        ("LoRA", lora_m),
        ("Rob-ScoreRt", rob_score_m),
        ("Rob-LR", rob_lr_m),
        ("Deb-ScoreRt", deb_score_m),
        ("Deb-LR", deb_lr_m),
    ]
    print_comparison(routing_comparison,
                     title="ROUTING COMPARISON: RoBERTa vs DeBERTa (test)")

    # ---- Per-topic comparison ----
    print("\nPer-topic NLI comparison:")
    rob_topic_metrics = {topic: compute_metrics(rob_t[topic]) for topic in sorted(rob_t.keys())}
    deb_topic_metrics = {topic: compute_metrics(deb_t[topic]) for topic in sorted(deb_t.keys())}

    print(f"  {'Topic':>20} {'Rob_All':>8} {'Deb_All':>8} {'Diff':>8}")
    print(f"  {'-' * 48}")
    for topic in sorted(set(list(rob_topic_metrics.keys()) +
                            list(deb_topic_metrics.keys()))):
        r = rob_topic_metrics.get(topic, {"all_acc": 0})
        d = deb_topic_metrics.get(topic, {"all_acc": 0})
        diff = d["all_acc"] - r["all_acc"]
        print(f"  {topic:>20} {r['all_acc']:>8.4f} {d['all_acc']:>8.4f} {diff:>+8.4f}")

    # ---- Save results ----
    print(f"\nSaving results to: {args.output_dir}")

    # nli_comparison.tsv
    nli_path = os.path.join(args.output_dir, "nli_comparison.tsv")
    with open(nli_path, "w") as f:
        f.write("method\tthreshold\tcount\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\tSER_MSE\n")
        for name, m in nli_comparison:
            t = (args.nli_threshold if "RoBERTa" in name else best_deberta_threshold)
            f.write(f"{name}\t{t}\t{m['count']}\t{m['S_acc']:.4f}\t"
                    f"{m['D_acc']:.4f}\t{m['I_acc']:.4f}\t"
                    f"{m['all_acc']:.4f}\t{m['SER_MAE']:.4f}\t"
                    f"{m['SER_MSE']:.4f}\n")
    print(f"  NLI comparison: {nli_path}")

    # routing_comparison.tsv
    rt_path = os.path.join(args.output_dir, "routing_comparison.tsv")
    with open(rt_path, "w") as f:
        f.write("method\tcount\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\tSER_MSE\n")
        for name, m in routing_comparison:
            f.write(f"{name}\t{m['count']}\t{m['S_acc']:.4f}\t"
                    f"{m['D_acc']:.4f}\t{m['I_acc']:.4f}\t"
                    f"{m['all_acc']:.4f}\t{m['SER_MAE']:.4f}\t"
                    f"{m['SER_MSE']:.4f}\n")
    print(f"  Routing comparison: {rt_path}")

    # per_topic_comparison.tsv
    topic_path = os.path.join(args.output_dir, "per_topic_comparison.tsv")
    with open(topic_path, "w") as f:
        f.write("topic\tmethod\tcount\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\n")
        for topic in sorted(set(list(rob_topic_metrics.keys()) +
                                list(deb_topic_metrics.keys()))):
            for method, tm in [("RoBERTa", rob_topic_metrics),
                               ("DeBERTa", deb_topic_metrics)]:
                if topic in tm:
                    m = tm[topic]
                    f.write(f"{topic}\t{method}\t{m['count']}\t"
                            f"{m['S_acc']:.4f}\t{m['D_acc']:.4f}\t"
                            f"{m['I_acc']:.4f}\t{m['all_acc']:.4f}\t"
                            f"{m['SER_MAE']:.4f}\n")
    print(f"  Per-topic comparison: {topic_path}")

    # deberta_threshold_sweep.tsv
    sweep_path = os.path.join(args.output_dir, "deberta_threshold_sweep.tsv")
    with open(sweep_path, "w") as f:
        f.write("threshold\tcount\tS_acc\tD_acc\tI_acc\tAll_acc\tSER_MAE\n")
        for r in deberta_sweep:
            f.write(f"{r['threshold']}\t{r['count']}\t{r['S_acc']:.4f}\t"
                    f"{r['D_acc']:.4f}\t{r['I_acc']:.4f}\t"
                    f"{r['all_acc']:.4f}\t{r['SER_MAE']:.4f}\n")
    print(f"  DeBERTa threshold sweep: {sweep_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
