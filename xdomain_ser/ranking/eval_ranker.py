# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Evaluate the MR ranker with IR metrics on a held-out negatives set.

For each query (gold example) the ranker scores every candidate negative
MR; we then compute MRR, MAP, R-Precision, Precision@k, and NDCG@k against
the gold graded relevance labels (0..6 from the ranking-dataset builder).
Used in Stage-4 reproduction (``scripts/reproduce_table4.sh``).
"""
from collections import defaultdict
from typing import List, Dict, Any, Sequence, Tuple
import argparse
import json
import os
import random

import numpy as np
import torch
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from xdomain_ser.core import data_helper as dh
from xdomain_ser.core.rank_metrics import (
    mean_reciprocal_rank,
    mean_average_precision,
    precision_at_k,
    ndcg_at_k,
    r_precision,
)
from xdomain_ser.ranking import data as rdh


def _digit_token_ids(tokenizer, device) -> torch.Tensor:
    ids = []
    for d in rdh.DIGITS:
        tok = tokenizer.encode(d, add_special_tokens=False)
        if len(tok) != 1:
            raise ValueError(f"Rank label '{d}' is not a single token for this tokenizer.")
        ids.append(tok[0])
    return torch.tensor(ids, device=device, dtype=torch.long)


@torch.inference_mode()
def rank_inference_batched(
    model,
    tokenizer,
    eval_data: Sequence[Dict[str, Any]],
    hint_map_table: Dict[str, str],
    batch_size: int = 32,
    max_length: int = 1024,
) -> List[List[float]]:
    """
    Returns a list of length len(eval_data); each element is a list of predicted scores
    (expected digit in [0, len(DIGITS)-1]) aligned to that example's candidates order.
    Also writes scores into ex['pred_labels'].
    """
    device = model.device
    model.eval()

    # Llama safety: pad token + left padding + mask
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", "right") != "left":
        tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id

    # Flatten all (prompt, owner_idx, candidate_idx)
    prompts: List[str] = []
    owners: List[Tuple[int, int]] = []  # (example_idx, cand_idx)
    for qi, ex in enumerate(eval_data):
        text = ex.get("surface_form") or ex.get("text") or ""
        cands = ex.get("negatives") or ex.get("candidates") or []
        hm_id = ex["hint_map_id"].lower()
        hm = hint_map_table[hm_id]["hint_map"]
        for ci, c in enumerate(cands):
            cmr = dh.make_mr_list(c["mr"])
            _, prompt = rdh.build_prompt(cmr, text, hm)
            prompts.append(prompt + "\n")
            owners.append((qi, ci))

    if not prompts:
        # nothing to score
        for ex in eval_data:
            ex["pred_labels"] = []
        return [[] for _ in eval_data]

    digit_ids = _digit_token_ids(tokenizer, device)
    digit_weights = torch.arange(len(rdh.DIGITS), device=device, dtype=torch.float32)

    flat_scores: List[float] = []
    for start in tqdm(range(0, len(prompts), batch_size), total=(len(prompts)//batch_size)+1):
        chunk = prompts[start:start + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
        ).to(device)
        try:
            out = model(**enc)  # logits: [B, T, V]
        except RuntimeError as e:
            print("\n", type(enc))
            print(enc["input_ids"].size())
            raise e
        # logits at the last non-pad token per row
        lengths = enc["attention_mask"].sum(dim=1) - 1  # [B]
        row_idx = torch.arange(lengths.size(0), device=device)
        last_logits = out.logits[row_idx, lengths, :]  # [B, V]

        sel = last_logits.index_select(dim=1, index=digit_ids)  # [B, n_digits]
        probs = sel.softmax(dim=-1)
        scores = (probs * digit_weights).sum(dim=-1)

        flat_scores.extend(scores.detach().float().cpu().tolist())

    # Group scores back to each example
    per_query_scores: List[List[float]] = [[] for _ in eval_data]
    for (qi, _ci), s in zip(owners, flat_scores):
        per_query_scores[qi].append(float(s))

    # Write back into eval_data for downstream evaluation
    for qi, ex in enumerate(eval_data):
        ex["pred_labels"] = per_query_scores[qi]

    return per_query_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate the MR ranker with IR metrics")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--eval_path", type=str, required=True, help="Path to evaluation data JSON file")
    parser.add_argument("--hint_map_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--min_relevance", type=int, default=4)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=232323)
    parser.add_argument("--per_topic_max", type=int, default=50)

    args = parser.parse_args()
    print("args:", args)
    random.seed(args.seed)

    if args.eval_only:
        with open(args.output_path, "r") as f:
            eval_data = json.load(f)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoPeftModelForCausalLM.from_pretrained(
            args.checkpoint_dir,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model.config.pad_token_id = tokenizer.pad_token_id

        with open(args.hint_map_path, "r") as fin:
            hint_map_table = json.load(fin)

        with open(args.eval_path, "r") as fin:
            eval_data = json.load(fin)

        random.shuffle(eval_data)
        topic_data = defaultdict(list)
        for e in eval_data:
            if len(topic_data[e["topic"]]) < args.per_topic_max:
                topic_data[e["topic"]].append(e)
        eval_data = [e for topic, exs in topic_data.items() for e in exs]


        for ex in tqdm(eval_data):
            if not isinstance(ex.get("negatives"), list):
                continue
            candidates = ex["negatives"]
            hm_id = ex["hint_map_id"].lower()
            hm = hint_map_table[hm_id]["hint_map"]
            ref_mr = ex["mr"]
            if "slots" in ref_mr:
                ref_mr = ref_mr["slots"]
            ref_mr = dh.make_mr_list(ref_mr)
            prompts = []
            for c in candidates:
                cmr = dh.make_mr_list(c["mr"])
                _, prompt = rdh.build_prompt(cmr, ex["surface_form"], hm, ref_mr)
                prompts.append(prompt + "\n")

            scores = rdh.score_candidates(model, tokenizer, prompts)
            ex["pred_labels"] = scores

        print("saving outputs to", args.output_path)
        with open(args.output_path, "w") as fout:
            json.dump(eval_data, fout, indent=2)


    print("evaluating ...")

    # -------- evaluate predictions --------
    all_bin_rankings = []  # list of binary relevance lists in ranked order
    p1_list, p2_list, p3_list, pall_list = [], [], [], []
    rprec_list = []
    ndcg1_list, ndcg2_list, ndcg3_list, ndcg_full_list = [], [], [], []
    for ex in eval_data:
        # gold labels: can be binary or graded (0..6). We'll treat >0 as relevant for binary metrics
        gold_labels = [int(e["label"]) for e in ex["negatives"]]
        pred_probs = ex["pred_labels"]
        pred_scores = []
        for pred_output in pred_probs:
            score = rdh.compute_probs_score(pred_output)
            pred_scores.append(score)

        if not gold_labels or not pred_scores:
            continue
        assert len(gold_labels) == len(pred_scores), "gold/pred length mismatch"
        # sort by predicted score (descending)
        order = np.argsort(pred_scores)[::-1]
        ranked_gold = [gold_labels[i] for i in order]

        # binary relevance for IR-style metrics (nonzero is relevant)
        min_relevance_score = args.min_relevance
        ranked_bin = [1 if g >= min_relevance_score else 0 for g in ranked_gold]
        all_bin_rankings.append(ranked_bin)

        L = len(ranked_gold)
        if L >= 1:  p1_list.append(precision_at_k(ranked_bin, 1))
        if L >= 2:  p2_list.append(precision_at_k(ranked_bin, 2))
        if L >= 3: p3_list.append(precision_at_k(ranked_bin, 3))
        pall_list.append(precision_at_k(ranked_bin, L))

        # R-Precision (precision at last relevant)
        rprec_list.append(r_precision(ranked_bin))

        # NDCG with graded labels where available
        if L >= 1: ndcg1_list.append(ndcg_at_k(ranked_gold, 1))
        if L >= 2: ndcg2_list.append(ndcg_at_k(ranked_gold, 2))
        if L >= 3: ndcg3_list.append(ndcg_at_k(ranked_gold, 3))
        ndcg_full_list.append(ndcg_at_k(ranked_gold, L))


    summary = {
        "MRR": float(mean_reciprocal_rank(all_bin_rankings)) if all_bin_rankings else 0.0,
        "MAP": float(mean_average_precision(all_bin_rankings)) if all_bin_rankings else 0.0,
        "R-Precision": float(np.mean(rprec_list)) if rprec_list else 0.0,
        "P@1": float(np.mean(p1_list)) if p1_list else 0.0,
        "P@1-N": len(p1_list),
        "P@2": float(np.mean(p2_list)) if p2_list else 0.0,
        "P@2-N": len(p2_list),
        "P@3": float(np.mean(p3_list)) if p3_list else 0.0,
        "P@3-N": len(p3_list),
        "P@all": float(np.mean(pall_list)) if pall_list else 0.0,
        "P@all-N": len(pall_list),
        "NDCG@1": float(np.mean(ndcg1_list)) if ndcg1_list else 0.0,
        "NDCG@1-N": len(ndcg1_list),
        "NDCG@2": float(np.mean(ndcg2_list)) if ndcg2_list else 0.0,
        "NDCG@2-N": len(ndcg2_list),
        "NDCG@3": float(np.mean(ndcg3_list)) if ndcg3_list else 0.0,
        "NDCG@3-N": len(ndcg3_list),
        "NDCG@All": float(np.mean(ndcg_full_list)) if ndcg_full_list else 0.0,
        "NDCG@All-N": len(ndcg_full_list),
        "num_queries": len(all_bin_rankings),
        "avg_cands_per_query": float(np.mean([len(r) for r in all_bin_rankings])) if all_bin_rankings else 0.0,
    }

    # print & persist
    print(json.dumps(summary, indent=2))
    if args.output_path:
        metrics_path = os.path.splitext(args.output_path)[0] + ".metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(summary, f, indent=2)
        print("Saved metrics to", metrics_path)


if __name__ == '__main__':
    main()
