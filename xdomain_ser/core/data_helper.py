# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Data loading and prompt-formatting helpers for the cross-domain SER pipeline.

Defines the MR list delimiters ``LIST_START`` / ``LIST_END`` used to wrap
``(slot: value); ...`` MR strings, and the dataset loaders / prompt builders
consumed by extraction training, ranker training, and inference.
"""
import random
import json
from collections import defaultdict

import numpy as np
from datasets import Dataset

LIST_START = "<LIST>"
LIST_END = "</LIST>"


def sort_by_key_length(examples, key, reverse=True):
    exs = [(len(ex[key]), random.randint(0, 10 ** 100), ex) for ex in examples]
    exs.sort(reverse=reverse)
    return [ex[2] for ex in exs]


def print_dataset_stats(tokenized_train_dataset):
    lengths = []
    for example in tokenized_train_dataset:
        x = example["attention_mask"]
        x = sum(x)
        lengths.append(x)
    hist, bin_edges = np.histogram(lengths)
    print("Histogram:")
    for i in range(len(hist)):
        bar = "█" * (hist[i] * 50 // max(hist))
        print(f"{int(bin_edges[i]):>3} - {int(bin_edges[i + 1]):<3} | {bar} ({hist[i]})")
    print("Length Statistics:")
    print("Num examples:", len(lengths))
    print("mean:", np.mean(lengths))
    print("max:", np.max(lengths))
    print("min:", np.min(lengths))
    print("median:", np.median(lengths))
    print("std:", np.std(lengths))


def select_train_examples(examples, num_exs=None, method="random"):
    selected_examples = []
    if method == "random":
        prs_prompt_examples = defaultdict(list)
        for ex in examples:
            prs_prompt_examples[ex["personality"]].append(ex)
        for prs, examples in prs_prompt_examples.items():
            examples = [(len(ex["mr"]), random.randint(0, 10**100), ex)  for ex in examples]
            examples.sort(reverse=True)
            longest = [ex[2] for ex in examples[:len(examples) // 4]]
            longest = [(len([v for v in ex["features"].values() if v]),
                        random.randint(0, 10**100), ex)
                        for ex in longest]
            longest.sort(reverse=True)
            longest = [ex[2] for ex in longest[:len(longest) // 4]]
            prs_prompt_examples[prs] = random.sample(longest, k=num_exs)
            for prs, exs in prs_prompt_examples.items():
                selected_examples.extend(exs)
    else:
        raise ValueError("Invalid method")
    return selected_examples


def build_finegrained_prompt_examples_lora(dataset, prompt_config):
    prs_prompt_examples = defaultdict(list)
    for ex in dataset:
        prs_prompt_examples[ex["personality"]].append(ex)
    if prompt_config["num_examples"] > 0:
        for prs, examples in prs_prompt_examples.items():
            examples = [(len(ex["mr"]), random.randint(0, 10**100), ex)  for ex in examples]
            examples.sort(reverse=True)
            longest = [ex[2] for ex in examples[:len(examples) // 4]]
            longest = [(len([v for v in ex["features"].values() if v]),
                        random.randint(0, 10**100), ex)
                        for ex in longest]
            longest.sort(reverse=True)
            longest = [ex[2] for ex in longest[:len(longest) // 4]]
            prs_prompt_examples[prs] = random.sample(longest, k=prompt_config["num_examples"])
    else:
        for prs in prs_prompt_examples:
            prs_prompt_examples[prs] = []
    return prs_prompt_examples


def load_prompt_examples(prompt_examples_path):
    prompt_examples = None
    if prompt_examples_path is not None:
        with open(prompt_examples_path) as fin:
            prompt_examples = json.load(fin)
        if isinstance(prompt_examples, list):
            ped = defaultdict(list)
            for ex in prompt_examples:
                ped[ex["features"]["topic"]].append(ex)
            prompt_examples = ped
    return prompt_examples


def build_prompt_hint_map(mr, input_text, hint_map, prompt_examples, make_output=True):
    hm = hint_map["hint_map"]
    hm_id = hint_map["hint_map_id"]
    instruction_text = "Rewrite the input text content as a list of (attribute: value) pairs. "
    instruction_text += f"Here is a hint map for {hm_id} attributes and values: {hm} \nHere are some examples:\n {prompt_examples}"
    prompt = f"### Instruction:\n{instruction_text}\n\n### Input:\n{input_text}\n\n### Response:\n"
    output = None
    if make_output:
        mr_str = []
        mr_dict = defaultdict(list)
        for (a, v) in mr:
            mr_dict[a].append(v)
        for a, vals in mr_dict.items():
            v = "; ".join(vals)
            mr_str.append(f"({a}: {v})")
        output = "; ".join(mr_str)
        output = f"{LIST_START}{output}{LIST_END}"
    return output, prompt


def make_prompt_example(pe):
    pe_mr = pe["mr"]["slots"] if "slots" in pe["mr"] else pe["mr"]
    mr_str = []
    mr_dict = defaultdict(list)
    for (a, v) in make_mr_list(pe_mr):
        mr_dict[a].append(v)
    for a, vals in mr_dict.items():
        v = "; ".join(vals)
        mr_str.append(f"({a}: {v})")
    mr_str = "; ".join(mr_str)
    text = pe["surface_form"]
    return text, f"{LIST_START}{mr_str}{LIST_END}"


def select_prompt_examples(prompt_examples, ex, num_pe):
    hm_id = ex["hint_map_id"]
    pexs = []
    pe = random.sample(prompt_examples[hm_id], 1)
    while len(pexs) < num_pe:
        pe = pe[0]
        if pe["surface_form"] != ex["surface_form"]:
            pexs.append(pe)
        pe = random.sample(prompt_examples[hm_id], 1)

    pe_list = []
    for pe in pexs:
        if "slots" in pe["mr"]:
            pe_mr = pe["mr"]["slots"]
        else:
            pe_mr = pe["mr"]
        text = pe["surface_form"]
        mr_str = []
        mr_dict = defaultdict(list)
        for (a, v) in make_mr_list(pe_mr):
            mr_dict[a].append(v)
        for a, vals in mr_dict.items():
            v = "; ".join(vals)
            mr_str.append(f"({a}: {v})")
        mr_str = "; ".join(mr_str)
        pe_list.append(f"-EXAMPLE-\nText: {text}\nList: {LIST_START}{mr_str}{LIST_END}")
    pe_list = "\n".join(pe_list)
    return pe_list


def make_mr_list(mr_dict):
    mr_lst = []
    for k, vals in mr_dict.items():
        if vals is None:
            continue
        elif isinstance(vals, str):
            mr_lst.append((k, vals))
        else:
            for v in vals:
                mr_lst.append((k, v))
    return mr_lst


def load_build_dataset2(data_path, args, topic_slots, per_topic_max=None, num_attrs=10, do_shuffle=False,
                        exclude_max_topics=None, do_shortening=False, return_dataset=True, min_mr_size=1,
                        prompt_examples=None, num_pe=5, rseed=232342):
    random.seed(rseed)
    with open(data_path) as fin:
        dataset = json.load(fin)
    if isinstance(dataset, dict):
        dset = []
        for topic, exs in dataset.items():
            dset.extend(exs)
        dataset = dset
    print("loaded dataset:", len(dataset))

    topic_examples = defaultdict(list)
    for ex in dataset:
        topic = ex["hint_map_id"]
        if "slots" in ex["mr"]:
            mr = ex["mr"]["slots"]
        else:
            mr = ex["mr"]
        ex["mr"] = make_mr_list(mr)
        if len(ex["mr"]) >= min_mr_size:
            topic_examples[topic].append(ex)

    dataset = []
    for topic, examples in topic_examples.items():
        print(f"{topic}: {len(examples)} examples")
        hm_id = examples[0]["hint_map_id"]
        if hm_id not in prompt_examples:
            print(f"skipping topic {topic}: no prompt examples available")
            continue
        if do_shuffle:
            random.shuffle(examples)
        if per_topic_max is not None and len(examples) > per_topic_max:
            print(f"randomly sampling {per_topic_max} examples from {topic}")
            examples = random.sample(examples, per_topic_max)
        for e in examples:
            e["prompt_examples"] = select_prompt_examples(prompt_examples, e, num_pe)
        dataset.extend(examples)

    if return_dataset:
        dataset = Dataset.from_list(dataset)
    return dataset


def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} "
        f"|| trainable%: {100 * trainable_params / all_param:.2f}"
    )


def _extract_response(text: str) -> str:
    """Trim everything before '### Response:\\n' and stop at known sentinels if present."""
    marker = "### Response:\n"
    i = text.find(marker)
    if i != -1:
        text = text[i + len(marker):]
    # Optional hard stops (in case your stopping criteria didn't cut them)
    for stop in (LIST_END, "\n\n###"):
        if stop and stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()
