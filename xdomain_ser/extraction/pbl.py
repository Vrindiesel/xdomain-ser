# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Prompt-based-learning MR extraction via the OpenAI Chat Completions API.

Optional alternative to the LoRA extractor (``inference.py``). Sends few-shot
extraction prompts to a configurable OpenAI model, requesting ``n``
candidates per input with temperature sampling. Concurrent batched dispatch
via ``ThreadPoolExecutor`` + tenacity retries on rate-limit / API errors.

Requires the optional PBL dependencies::

    pip install -r requirements-pbl.txt

API key resolution order:
    1. ``--openai_conf_path`` JSON file (keys ``api_key`` / ``api_key2`` /
       ``OPENAI_API_KEY``)
    2. ``OPENAI_API_KEY`` environment variable
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from tqdm import tqdm

from xdomain_ser.core import data_helper as dh

try:
    from openai import OpenAI, RateLimitError, APIError
    from tenacity import (
        retry, wait_exponential_jitter, stop_after_attempt,
        retry_if_exception_type,
    )
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False
    OpenAI = RateLimitError = APIError = None  # noqa: F811


def _require_openai():
    if not _HAS_OPENAI:
        raise ImportError(
            "xdomain_ser.extraction.pbl requires the optional PBL dependencies. "
            "Install with:  pip install -r requirements-pbl.txt"
        )


def _resolve_api_key(args):
    """Order: --openai_conf_path JSON -> OPENAI_API_KEY env var -> error."""
    if args.openai_conf_path and os.path.exists(args.openai_conf_path):
        with open(args.openai_conf_path) as fin:
            conf = json.load(fin)
        for key in ("api_key", "api_key2", "OPENAI_API_KEY"):
            if conf.get(key):
                return conf[key]
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    raise RuntimeError(
        "No OpenAI API key found. Provide --openai_conf_path "
        "(JSON with 'api_key' / 'api_key2' / 'OPENAI_API_KEY') "
        "or set the OPENAI_API_KEY environment variable."
    )


def _extract_response(text: str) -> str:
    """Trim everything before '### Response:\\n' and stop at known sentinels if present."""
    marker = "### Response:\n"
    i = text.find(marker)
    if i != -1:
        text = text[i + len(marker):]
    # Optional hard stops (in case your stopping criteria didn't cut them)
    for stop in (getattr(dh, "LIST_END", None), "\n\n###"):
        if stop and stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()


def build_extraction_prompt(input_text: str, few_shot_examples: list[tuple[str, str]], domain, hint_map) -> str:
    # few_shot_examples: list of (text, list_str) where list_str already formatted as
    # "<LIST>(slot: val); ...</LIST>"

    header = f"""You are an information extraction model.
Given a single utterance (Text), extract zero or more (slot: value) pairs for the DOMAIN specified below.

RULES
1) Use ONLY slots from the given DOMAIN SCHEMA and keep the exact slot names (including the domain prefix).
2) Copy values verbatim from the Text when possible (keep original casing for names like store names).
3) If a value is relative time or date, keep it as said (e.g., "today", "tomorrow", "morning", "noon").
4) If a slot is not mentioned, DO NOT output it. Never output "None".
5) Separate pairs with semicolons; wrap the whole list inside <LIST> ... </LIST>.
6) Each pair must look like: (slot_name: value)
7) Order pairs by the order they appear in the Text.
8) Output must be exactly one line starting with:  List: <LIST> ... </LIST>
9) If nothing is found, output:  List: <LIST></LIST>

DOMAIN: {domain}

DOMAIN SCHEMA (allowed slots) & description:
{hint_map}
(Use only the ones that appear in the Text.)

FORMAT EXAMPLES (same domain)
"""
    ex_strs = []
    for text, list_str in few_shot_examples:
        ex_strs.append(f"-EXAMPLE-\nText: {text}\nList: {list_str}\n")
    examples_block = "\n".join(ex_strs)

    return f"{header}{examples_block}\n# QUERY (the one to extract now)\nText: {input_text}\nList:"


def _fetch_topn_unwrapped(client, model, prompt: str, n: int = 10,
                          temperature: float = 0.9, seed=None) -> List[str]:
    """
    One API call that returns n completions for a single prompt.
    Uses Chat Completions API because it supports `n`.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        n=n,
        temperature=temperature,
        seed=seed,  # deterministic sampling seed
    )
    return [c.message.content or "" for c in resp.choices]


if _HAS_OPENAI:
    _fetch_topn = retry(
        reraise=True,
        wait=wait_exponential_jitter(initial=1, max=20),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((RateLimitError, APIError)),
    )(_fetch_topn_unwrapped)
else:
    _fetch_topn = _fetch_topn_unwrapped  # call-time error via _require_openai()


def run_batched(client, model,
                prompts: List[str],
                batch_size: int = 10,
                n_per_input: int = 10,
                max_workers: int = 10,
                temperature: float = 0.9,
                seed: int = None) -> List[List[str]]:
    """
    Returns: list of length len(prompts); each element is a list[str] of length n_per_input.
    Processes prompts in batches of `batch_size`, concurrently (max_workers threads).
    """
    all_outputs: List[List[str]] = [None] * len(prompts)  # type: ignore

    for start in tqdm(range(0, len(prompts), batch_size), desc="Batches"):
        end = min(start + batch_size, len(prompts))
        batch_idxs = list(range(start, end))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(_fetch_topn, client, model, prompts[i], n_per_input, temperature, seed): i
                for i in batch_idxs
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    all_outputs[i] = fut.result()
                except Exception as e:
                    # On persistent failure, return an empty list for that prompt
                    all_outputs[i] = []
                    print(f"[WARN] prompt {i} failed: {e}")

    return [outs if outs is not None else [] for outs in all_outputs]


def main():
    parser = argparse.ArgumentParser(description="OpenAI-API MR extraction via prompt-based learning")

    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    parser.add_argument("--prompt_examples_path", type=str, default=None)
    parser.add_argument("--hint_map_path", type=str, default=None)
    parser.add_argument("--openai_conf_path", type=str, default=None,
                        help="JSON config with 'api_key' / 'api_key2' / 'OPENAI_API_KEY'. "
                             "If unset, falls back to the OPENAI_API_KEY env var.")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Number of completions per prompt (n). Despite the name, "
                             "this is sampling-based, not beam search.")
    parser.add_argument("--num_pe", type=int, default=5)
    parser.add_argument("--seed", type=int, default=232342)
    parser.add_argument("--model", type=str, default="gpt-4o")

    args = parser.parse_args()
    print("args:", args)

    _require_openai()
    api_key = _resolve_api_key(args)

    with open(args.hint_map_path, "r") as fin:
        hint_map_table = json.load(fin)
    with open(args.prompt_examples_path, "r") as fin:
        eval_prompt_examples = json.load(fin)

    per_topic_max = None
    eval_dataset = dh.load_build_dataset2(args.input_path, args, None,
                                            per_topic_max=per_topic_max, do_shuffle=False,
                                            prompt_examples=eval_prompt_examples, num_pe=args.num_pe,
                                            return_dataset=False)
    outpath = args.output_path
    print("Saving predictions to", outpath)

    inputs = []
    for j, example in enumerate(tqdm(eval_dataset, desc="Processing ")):
        input_text = example["surface_form"]
        hm_id = example["hint_map_id"].lower()
        hm = hint_map_table[hm_id]["hint_map"]

        hm_str = "\n".join([f"- {name}:  {desc}" for name, desc in hm.items()])
        domain = hint_map_table[hm_id]["domain"]
        pexs = []
        if hm_id not in eval_prompt_examples:
            continue
        for pe in eval_prompt_examples[hm_id]:
            text, mr_str = dh.make_prompt_example(pe)
            pexs.append((text, mr_str))

        prompt = build_extraction_prompt(input_text, pexs, domain, hm_str)
        inputs.append(prompt)

    MODEL = args.model
    N_PER_INPUT = args.num_beams  # number of candidates per prompt
    BATCH_SIZE = 10  # dispatch 10 prompts per batch
    MAX_WORKERS = 10  # parallel calls per batch

    client = OpenAI(api_key=api_key)
    outputs_per_prompt = run_batched(client, MODEL, inputs, batch_size=BATCH_SIZE,
                                     n_per_input=N_PER_INPUT, max_workers=MAX_WORKERS,
                                     temperature=1.0, seed=args.seed)
    print(f"Got {len(outputs_per_prompt)} prompts; first prompt has {len(outputs_per_prompt[0])} candidates")

    for ex, res in zip(eval_dataset, outputs_per_prompt):
        ex["predictions"] = res

    print("Saving predictions to", outpath)
    with open(outpath, "w") as fout:
        json.dump(eval_dataset, fout, indent=2)


if __name__ == "__main__":
    main()
