# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Stratified sampling of ~1000 examples for manual SER annotation.

Samples 500 LLM outputs (GPT-4o, exp3-exp9) and 500 seq2seq outputs
(PERSONAGE models), balanced at 100 per personality type per source.

Input files: ``outputs/exp{N}/examples-test.outputs.postp.cls.bb.extract.scores.json``
(produced by ``xdomain_ser.eval2.personage_inference`` upstream) and the
SV-NLG personality-only outputs.

Output: ``evaluation/gold/raw/sampled-examples.json``.
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict


PERSONALITIES = [
    "AGREEABLE", "DISAGREEABLE", "EXTRAVERT",
    "CONSCIENTIOUSNESS", "UNCONSCIENTIOUSNESS",
]

LLM_EXPERIMENTS = ["exp3", "exp4", "exp5", "exp6", "exp7", "exp8", "exp9"]
LLM_FILE_PATTERN = "outputs/{exp}/examples-test.outputs.postp.cls.bb.extract.scores.json"

SEQ2SEQ_FILES = {
    "features_guide_model1": "../data/svnlg-outputs-in-paper/personality-only/features_guide_personality_only-model-1.extract.scores.json",
    "features_token_model7": "../data/svnlg-outputs-in-paper/personality-only/features_token_personality_only-model-7.extract.scores.json",
    "token_supervision_model27": "../data/svnlg-outputs-in-paper/personality-only/token_supervision-model-27.extract.scores.json",
}

# Target counts
TARGET_PER_PERSONALITY = 100
LLM_TOTAL = 500
SEQ2SEQ_TOTAL = 500


def load_llm_data():
    """Load all LLM experiment files and index by personality."""
    by_pers_exp = defaultdict(list)
    for exp in LLM_EXPERIMENTS:
        path = LLM_FILE_PATTERN.format(exp=exp)
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        for idx, ex in enumerate(data):
            pers = ex["personality"]
            entry = {
                "source": "llm",
                "experiment": exp,
                "personality": pers,
                "mr": ex["mr"],
                "pred": ex["pred"],
                "clean_pred_text": ex["clean_pred"][0]["clean_text"],
                "ref": ex["ref"],
                "pred_mr": ex["pred_mr"],
                "pred_scores": ex["pred_scores"],
                "orig_idx": idx,
            }
            by_pers_exp[(pers, exp)].append(entry)
    return by_pers_exp


def load_seq2seq_data():
    """Load all seq2seq model files and index by personality."""
    by_pers_model = defaultdict(list)
    for model_name, path in SEQ2SEQ_FILES.items():
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        for idx, ex in enumerate(data):
            pers = ex["personality"]
            entry = {
                "source": "seq2seq",
                "experiment": model_name,
                "personality": pers,
                "mr": ex["mr"],
                "pred": ex["pred"],
                "clean_pred_text": ex["pred"],
                "ref": ex["ref"],
                "pred_mr": ex["pred_mr"],
                "pred_scores": ex["pred_scores"],
                "orig_idx": idx,
            }
            by_pers_model[(pers, model_name)].append(entry)
    return by_pers_model


def stratified_sample_llm(by_pers_exp, rng):
    """Sample 500 LLM examples: 100 per personality, spread across 7 experiments."""
    sampled = []
    for pers in PERSONALITIES:
        available = []
        for exp in LLM_EXPERIMENTS:
            key = (pers, exp)
            if key in by_pers_exp:
                available.append((exp, by_pers_exp[key]))

        if not available:
            print(f"  Warning: no LLM data for {pers}")
            continue

        n_exps = len(available)
        per_exp = TARGET_PER_PERSONALITY // n_exps
        remainder = TARGET_PER_PERSONALITY % n_exps

        pers_sampled = []
        for i, (exp, examples) in enumerate(available):
            n = per_exp + (1 if i < remainder else 0)
            n = min(n, len(examples))
            chosen = rng.sample(examples, n)
            pers_sampled.extend(chosen)

        sampled.extend(pers_sampled)
        print(f"  LLM {pers}: {len(pers_sampled)} sampled")

    return sampled


def stratified_sample_seq2seq(by_pers_model, rng):
    """Sample 500 seq2seq examples: 100 per personality, spread across 3 models."""
    sampled = []
    model_names = list(SEQ2SEQ_FILES.keys())

    for pers in PERSONALITIES:
        available = []
        for model in model_names:
            key = (pers, model)
            if key in by_pers_model:
                available.append((model, by_pers_model[key]))

        if not available:
            print(f"  Warning: no seq2seq data for {pers}")
            continue

        n_models = len(available)
        per_model = TARGET_PER_PERSONALITY // n_models
        remainder = TARGET_PER_PERSONALITY % n_models

        pers_sampled = []
        for i, (model, examples) in enumerate(available):
            n = per_model + (1 if i < remainder else 0)
            n = min(n, len(examples))
            chosen = rng.sample(examples, n)
            pers_sampled.extend(chosen)

        sampled.extend(pers_sampled)
        print(f"  Seq2seq {pers}: {len(pers_sampled)} sampled")

    return sampled


def assign_ids(examples):
    """Assign unique IDs to each example."""
    counters = defaultdict(int)
    for ex in examples:
        src = ex["source"]
        exp = ex["experiment"]
        key = f"{src}_{exp}"
        counters[key] += 1
        idx = counters[key]
        ex["id"] = f"{src[:3]}_{exp}_{idx:03d}"


def main():
    parser = argparse.ArgumentParser(
        description="Stratified sampling for SER annotation")
    parser.add_argument("--output_path", type=str,
                        default="evaluation/gold/raw/sampled-examples.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print("args:", args)

    rng = random.Random(args.seed)

    print("\nLoading LLM data...")
    llm_data = load_llm_data()
    print(f"  {sum(len(v) for v in llm_data.values())} total LLM examples across "
          f"{len(llm_data)} (personality, exp) groups")

    print("\nLoading seq2seq data...")
    seq_data = load_seq2seq_data()
    print(f"  {sum(len(v) for v in seq_data.values())} total seq2seq examples across "
          f"{len(seq_data)} (personality, model) groups")

    print("\nSampling LLM examples...")
    llm_sampled = stratified_sample_llm(llm_data, rng)

    print("\nSampling seq2seq examples...")
    seq_sampled = stratified_sample_seq2seq(seq_data, rng)

    all_sampled = llm_sampled + seq_sampled
    rng.shuffle(all_sampled)
    assign_ids(all_sampled)

    src_counts = Counter(ex["source"] for ex in all_sampled)
    pers_counts = Counter((ex["source"], ex["personality"]) for ex in all_sampled)
    print(f"\nTotal sampled: {len(all_sampled)}")
    print(f"  By source: {dict(src_counts)}")
    for key in sorted(pers_counts):
        print(f"  {key}: {pers_counts[key]}")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(all_sampled, f, indent=2)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
