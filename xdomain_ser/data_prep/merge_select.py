# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Merge per-domain ``ds`` files into one multi-domain split and select a subset.

Third and final step of the dataset-construction pipeline (see
``scripts/build_dataset.sh``). Reads a ``files.json`` manifest that lists the
per-domain train/dev/test ``ds`` files, the per-topic hint maps, and the
dev-prompt-example pools produced by ``make_ser_dataset.py``, then for each
split selects ``--train_k`` / ``--dev_k`` examples per topic. Writes the merged
``hint_maps.json``, ``train-<k>.json`` / ``dev-<k>.json`` / ``test.json``,
``topic-examples-{train,dev}.json``, and ``prompt-examples-dev.json`` into
``--output_path`` -- i.e. the contents of ``data/multi_ser_v9/``.

Two selection methods: ``random`` (used to build the released v9 data) and
``fracloc`` (Facility-Location greedy on a combined MiniLM-cosine + slot-Jaccard
kernel, via ``submodlib`` + ``sentence-transformers``). ``select_with_facloc`` is
also imported by ``eval2.personage_inference``. Part of the optional
``[datasets]`` extra. (The earlier random-only ``main2`` variant is dropped.)

Manifest paths are resolved relative to the current working directory (the repo
root), not ``../`` as in the original script.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import numpy as np

# ``sentence-transformers`` and ``submodlib`` are only needed on the
# Facility-Location selection path; they are imported lazily so the module
# loads (and ``--selection_method random``, used to build the released v9 data,
# runs) without the optional dataprep requirements installed.
_FACLOC_HINT = (
    "Facility-Location selection (--selection_method fracloc) requires the "
    "dataset-construction deps. Install them with: pip install -r requirements-dataprep.txt"
)


# ---------------- helpers: normalize + tokenization ----------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def flatten_mr(mr: Dict[str, Any], include_dact: bool = True) -> Dict[str, str]:
    out = {}
    if include_dact and "dact" in mr:
        out["dact"] = _norm(mr["dact"])

    if mr.get("slots"):
        mr = mr["slots"]

    for k, v in mr.items():
        out[_norm(k)] = _norm(v)
    return out

def slot_value_tokens(ex: Dict[str, Any], include_dact: bool = True) -> Set[str]:
    flat = flatten_mr(ex["mr"], include_dact=include_dact)
    return {f"{k}={v}" for k, v in flat.items()}

def jaccard_kernel(sets: List[Set[str]]) -> np.ndarray:
    n = len(sets)
    K = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        si = sets[i]
        for j in range(i, n):
            sj = sets[j]
            u = len(si | sj) or 1
            K[i, j] = K[j, i] = len(si & sj) / u
    return K

def embed_texts(texts: List[str], model_name="sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(_FACLOC_HINT) from e
    st = SentenceTransformer(model_name)
    E = st.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(E, dtype=np.float32)  # rows normalized → cosine = dot

# ---------------- main: submodlib selection ----------------
def select_with_facloc(
    examples: List[Dict[str, Any]],
    k: int,
    alpha_text: float = 0.6,
    alpha_slots: float = 0.6,
    include_dact: bool = True,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Combine a text-similarity kernel (cosine on MiniLM embeddings) and
    a slot-coverage kernel (Jaccard on slot:value tokens), then run
    Facility Location greedy selection for a size-k subset.
    """
    try:
        from submodlib import FacilityLocationFunction
    except ImportError as e:
        raise ImportError(_FACLOC_HINT) from e
    assert 1 <= k <= len(examples)
    if "surface_form" in examples[0]:
        texts = [ex["surface_form"] for ex in examples]
    else:
        texts = [ex["ref"] for ex in examples]
    token_sets = [slot_value_tokens(ex, include_dact=include_dact) for ex in examples]

    # Build kernels
    K_slot = jaccard_kernel(token_sets)                          # [N,N] in [0,1]
    E = embed_texts(texts)                                       # [N,D], L2-normalized
    K_txt = (E @ E.T).astype(np.float64)                         # cosine in [-1,1]
    K_txt = (K_txt + 1.0) / 2.0                                  # → [0,1]

    # Combine (weights are additive)
    K = alpha_text * K_txt + alpha_slots * K_slot                # float64 dense matrix

    # Subset selection with submodlib
    n = len(examples)
    #fl = FacilityLocationFunction(n=n, mode="dense", sijs=K)
    #greedy = fl.maximize(budget=k, optimizer="NaiveGreedy", stopIfZeroGain=False)
    #selected = [i for (i, _gain) in greedy]

    K = np.asarray(K, dtype=np.float64)
    np.fill_diagonal(K, 1.0)

    fl = FacilityLocationFunction(
        n=n,
        mode="dense",
        separate_rep=False,  # <- required when you pass sijs for ground set
        sijs=K
    )
    greedy = fl.maximize(
        budget=k,
        optimizer="NaiveGreedy",  # or "LazyGreedy" for speed
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
    )
    selected = [i for (i, _gain) in greedy]


    # Diagnostics: token coverage
    universe = set().union(*token_sets)
    covered = set().union(*[token_sets[i] for i in selected])
    per_slot = defaultdict(set)
    for tok in covered:
        slot = tok.split("=", 1)[0]
        per_slot[slot].add(tok)

    diag = {
        "N": n,
        "k": k,
        "alpha_text": alpha_text,
        "alpha_slots": alpha_slots,
        "token_universe": len(universe),
        "token_covered": len(covered),
        "coverage_ratio": round(len(covered) / max(1, len(universe)), 4),
        "per_slot_token_counts": {s: len(v) for s, v in per_slot.items()},
        #"selected_indices": selected,
    }
    return selected, diag



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path")
    parser.add_argument("--files", type=str)
    parser.add_argument("--seed", type=int, default=2323)
    parser.add_argument("--train_k", type=int, default=3)
    parser.add_argument("--dev_k", type=int, default=3)
    parser.add_argument("--min_mr_len", type=int, default=2)
    parser.add_argument("--max_n", type=int, default=10_000)
    parser.add_argument("--alpha_text", type=float, default=0.3)
    parser.add_argument("--alpha_slots", type=float, default=0.7)
    parser.add_argument("--selection_method", type=str, default="fracloc", choices=["fracloc", "random"])
    args = parser.parse_args()
    print("args:", args)
    random.seed(args.seed)
    with open(args.files, "r") as fin:
        file_paths = json.load(fin)

    hint_maps = {}
    for hm_path in file_paths["hint_map"]:
        with open(hm_path, "r") as fin:
            hm = json.load(fin)
            hint_maps[hm["hint_map_id"]] = hm

    with open(os.path.join(args.output_path, "hint_maps.json"), "w") as fout:
        json.dump(hint_maps, fout, indent=2)


    for name in ["train", "dev", "test"]:
        this_side_data = []
        topic_data = defaultdict(list)
        print("loading data ...")
        for fpath in file_paths[name]:
            with open(fpath, "r") as fin:
                examples = json.load(fin)
            for e in examples:
                if isinstance(e["surface_form"], list):
                    e["surface_form"] = random.choice(e["surface_form"])
                mr = e["mr"]
                if mr.get("slots"):
                    mr = mr["slots"]
                if len(mr) >= args.min_mr_len:
                    topic_data[e["hint_map_id"]].append(e)


        for topic, examples in topic_data.items():
            if len(examples) >= args.max_n:
                examples = random.sample(examples, args.max_n)

            print("\nselecting examples {} ...".format(topic), len(examples))
            k = args.train_k if name == "train" else args.dev_k
            if name in {"train", "dev"} and len(examples) > k:
                if args.selection_method == "fracloc":
                    selected, info = select_with_facloc(examples, k, alpha_text=args.alpha_text, alpha_slots=args.alpha_slots, include_dact=False)
                    select_data = [examples[i] for i in selected]
                else:
                    assert args.selection_method == "random"
                    select_data = random.sample(examples, k)

            else:
                select_data = examples
            this_side_data.extend(select_data)

        # select prompt example pool
        print("\n\nselecting prompt examples pool")
        if name in {"train", "dev"}:
            if name == "train":
                k = args.train_k * 4
            else: # dev
                k = args.dev_k * 4
            for topic, examples in topic_data.items():
                print(f"processing {topic} ...")
                if len(examples) >= args.max_n:
                    examples = random.sample(examples, args.max_n)
                if len(examples) > k:
                    if args.selection_method == "fracloc":
                        selected, info = select_with_facloc(examples, k, alpha_text=args.alpha_text, alpha_slots=args.alpha_slots, include_dact=False)
                        topic_data[topic] = [examples[i] for i in selected]
                    else:
                        assert args.selection_method == "random"
                        topic_data[topic] = random.sample(examples, k)

            with open(os.path.join(args.output_path, f"topic-examples-{name}.json"), "w") as fout:
                json.dump(topic_data, fout, indent=2)

            k = args.train_k if name == "train" else args.dev_k
            with open(os.path.join(args.output_path, f"{name}-{k}.json"), "w") as fout:
                json.dump(this_side_data, fout, indent=2)
        else:
            with open(os.path.join(args.output_path, f"{name}.json"), "w") as fout:
                json.dump(this_side_data, fout, indent=2)

    # select prompt examples for dev inference
    prompt_examples = defaultdict(list)
    for fpath in file_paths["dev_prompt_examples"]:
        with open(fpath, "r") as fin:
            examples = json.load(fin)
        for e in examples:
            if isinstance(e["surface_form"], list):
                e["surface_form"] = random.choice(e["surface_form"])
            mr = e["mr"]
            if mr.get("slots"):
                mr = mr["slots"]
            if len(mr) >= args.min_mr_len:
                prompt_examples[e["hint_map_id"]].append(e)

    print("selecting dev prompt examples ...")
    k = 5  # number of examples in prompt
    for topic, examples in prompt_examples.items():
        print(f"\nprocessing {topic} ...")
        if len(examples) >= args.max_n:
            examples = random.sample(examples, args.max_n)
        if len(examples) > k:
            if args.selection_method == "fracloc":
                selected, info = select_with_facloc(examples, k, alpha_text=args.alpha_text, alpha_slots=args.alpha_slots, include_dact=False)
                prompt_examples[topic] = [examples[i] for i in selected]
            else:
                assert args.selection_method == "random"
                prompt_examples[topic] = random.sample(examples, k)
    with open(os.path.join(args.output_path, "prompt-examples-dev.json"), "w") as fout:
        json.dump(prompt_examples, fout, indent=2)


if __name__ == "__main__":
    main()
