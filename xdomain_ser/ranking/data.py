# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Data + prompt helpers for the MR-ranking model.

Defines the 7-grade rubric digits (0=bad ... 6=perfect), the ranker prompt
template (text + gold MR + candidate MR -> single-digit score), the ranker
training-dataset loader with per-label balancing, and the inference-time
``score_candidates`` + ``compute_probs_score`` helpers used by
``xdomain_ser.ranking.score`` and ``xdomain_ser.ranking.eval_ranker``.
"""
from collections import defaultdict, Counter
import json
import random

import torch
from datasets import Dataset

from xdomain_ser.core import data_helper as dh

DIGITS = ["0", "1", "2", "3", "4", "5", "6"]

hm_prompt = """DOMAIN SCHEMA (allowed slots) & description:
{HINT_MAP}
"""

instruction_text = """You are scoring how well a predicted meaning representation (MR) matches the reference text and gold MR.
Return only one digit: 0=bad, 1=poor, 2=weak, 3=mediocre, 4=good, 5=excellent, 6=perfect.

DOMAIN SCHEMA (allowed slots) & description:
{HINT_MAP}

Text:
{TEXT}

Gold Reference:
{REFERENCE}

Predicted MR:
{MR}

Score:"""

def build_prompt(mr_extraction, extraction_text, hint_map, gold_mr, label=None):
    mr_str_extraction = build_mrstring(mr_extraction)
    gold_mr_str = build_mrstring(gold_mr)

    if hint_map is None:
        prompt = instruction_text.format(TEXT=extraction_text, MR=mr_str_extraction)
    else:
        hm_str = "\n".join([f"- {name}:  {desc}" for name, desc in hint_map.items()])
        prompt = instruction_text.format(TEXT=extraction_text, MR=mr_str_extraction, HINT_MAP=hm_str, REFERENCE=gold_mr_str)

    output = "" if label is None else f"\n{label}"
    return output, prompt


def build_mrstring(mr):
    mr_dict = defaultdict(list)
    for (a, v) in mr:
        mr_dict[a].append(v)
    mr_str = mr_string_from_dictlist(mr_dict)
    return mr_str


def mr_string_from_dictlist(mr_dict):
    mr_str = []
    for a, vals in mr_dict.items():
        v = "; ".join(vals)
        mr_str.append(f"({a}: {v})")
    mr_str = "; ".join(mr_str)
    return mr_str


def load_ranking_dataset(data_path, return_dataset=True, per_topic_max=20_000, do_shuffle=True):
    with open(data_path, "r") as fin:
        data = json.load(fin)
    if do_shuffle:
        random.shuffle(data)

    all_examples = defaultdict(lambda: defaultdict(list))
    topic_text_counts = Counter()
    topic_instance_count = []
    for example in data:
        if "negatives" not in example:
            continue

        gold_mr = example["mr"]
        if "slots" in gold_mr:
            gold_mr = gold_mr["slots"]

        topic_text_counts[example["hint_map_id"]] += 1
        for nex in example["negatives"]:
            e = {
                "hint_map_id": example["hint_map_id"],
                "text": example["surface_form"],
                "mr": dh.make_mr_list(nex["mr"]),
                "label": nex["label"],
                "gold_mr": dh.make_mr_list(gold_mr),
            }
            all_examples[example["topic"]][nex["label"]].append(e)
            topic_instance_count.append(example["hint_map_id"])

    balanced_exs = []
    for topic, topic_exs in all_examples.items():
        print("topic:", topic)
        k5 = len(topic_exs.get(5, []))

        for label_val, exs in topic_exs.items():
            k = min(k5 if len(exs) > k5 else len(exs), per_topic_max)
            selected = random.sample(exs, k)
            balanced_exs.extend(selected)
            print(f"\t\tlabel: {label_val},  k: {k} / {len(exs)}")

    print("\nTopic Text Counts")
    for topic, n in topic_text_counts.items():
        print(f" * {topic}: {n}")
    print("\nTopic Instance Count")
    for topic, n in Counter(topic_instance_count).items():
        print(f" * {topic}: {n}")

    dataset = balanced_exs
    if return_dataset:
        dataset = Dataset.from_list(balanced_exs)
    return dataset


@torch.inference_mode()
def score_candidates(model, tokenizer, candidate_prompts, device=None, batch_size=None):
    """Score each candidate prompt as a digit distribution over DIGITS (0..6).

    ``batch_size`` bounds peak GPU memory: with it set, prompts are scored in
    chunks of at most ``batch_size`` and the CUDA cache is released between
    chunks. Default (None) scores all prompts in one batch, preserving the
    original behavior for existing callers.
    """
    device = device or model.device
    digit_ids = torch.tensor([tokenizer.convert_tokens_to_ids(d) for d in DIGITS], device=device)

    if batch_size is None or batch_size >= len(candidate_prompts):
        chunks = [candidate_prompts]
    else:
        chunks = [candidate_prompts[i:i + batch_size]
                  for i in range(0, len(candidate_prompts), batch_size)]

    probs = []
    for chunk in chunks:
        toks = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
        out = model(**toks)
        # next-token logits after the last position of each prompt
        last_logits = out.logits[:, -1, :]  # [B, V]
        sel = last_logits.index_select(dim=1, index=digit_ids)  # [B, len(DIGITS)]
        chunk_probs = sel.softmax(dim=-1)  # per-candidate distribution over digits
        probs.extend(chunk_probs.detach().float().cpu().tolist())
        if batch_size is not None and torch.cuda.is_available():
            del toks, out, last_logits, sel, chunk_probs
            torch.cuda.empty_cache()
    return probs  # higher digits are better


def compute_probs_score(pred_output):
    """Probability-weighted scalar score in [0, 3].

    ``pred_output`` is a 7-vector of probabilities over digits 0..6. We sum
    ``digit * prob`` over digits 3..6 (the "good" half of the rubric),
    yielding a continuous score where higher means a better-ranked MR.
    """
    start_digit = 3
    digits = [0, 1, 2, 3, 4]
    return sum([d * prob for d, prob in zip(digits, pred_output[start_digit:])])
