# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""LoRA MR-extraction inference on PERSONAGE-style M2T outputs.

Specialised variant of :mod:`xdomain_ser.extraction.inference` for the
Eval-2 PERSONAGE evaluation surface:

* Input examples carry a ``personality`` field (one of AGREEABLE /
  DISAGREEABLE / EXTRAVERT / CONSCIENTIOUSNESS / UNCONSCIENTIOUSNESS).
* Few-shot prompt examples are selected per-personality via the
  Facility-Location selector from
  ``xdomain_ser.data_prep.merge_select.select_with_facloc`` (Stage-9-era
  helper -- imported lazily inside ``main_batch()``).
* MR slot names are translated from E2E-native (``eatType``, ``food``,
  ``familyFriendly``, etc.) to LoRA-internal (``venue_type``,
  ``cuisine_type``, ``family_suitability``, etc.) before prompting.

Used as the first step of ``run-my-ser-metric.sh`` -- produces the
``pred_mr`` + ``pred_scores`` fields that :mod:`xdomain_ser.eval2.sample`
consumes downstream.
"""
import argparse
import json
from collections import defaultdict

import torch
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from xdomain_ser.core import data_helper as dh


def _extract_response(text: str) -> str:
    """Trim everything before '### Response:\\n' and stop at known sentinels if present."""
    marker = "### Response:\n"
    i = text.find(marker)
    if i != -1:
        text = text[i + len(marker):]
    for stop in (getattr(dh, "LIST_END", None), "\n\n###"):
        if stop and stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()


def unpack_e2e_nlg_mr(mr_dict):
    """Translate E2E-native slot names to LoRA-internal names.

    Family suitability: returns an enum string (``family-friendly`` /
    ``not-family-friendly``) rather than a boolean. Venue vs cuisine
    conflict, area vs nearby, and absent-slot handling all left to
    downstream alignment.
    """
    KEY_MAP = {
        "customer_rating": "customerRating",
        "customer rating": "customerRating",
        "familyFriendly": "family_suitability",
        "eatType": "venue_type",
        "food": "cuisine_type",
        "near": "nearby_landmark",
        "area": "area_zone",
    }


    slots = defaultdict(list)
    for name, value in mr_dict.items():
        new_value = value
        if name == "familyFriendly":
            new_value = "family-friendly" if value == "yes" else "not-family-friendly"
            new_name = KEY_MAP.get(name, name)
        else:
            new_name = KEY_MAP.get(name, name)

        if new_value:
            if isinstance(new_value, list):
                slots[new_name].extend(new_value)
            else:
                slots[new_name].append(new_value)

    return {"slots": slots}


def preprocess_examples(examples, text_key="pred", lowering=False):
    data = []
    for row in examples:
        mr = unpack_e2e_nlg_mr(row["mr"])
        for k, v in mr["slots"].items():
            if len(v) == 1:
                mr["slots"][k] = v[0]

        text = row[text_key].lower() if lowering else row[text_key]
        data.append({
            "mr": mr,
            "surface_form": text,
            "topic": "restaurants",
            "dataset": "personage_nlg",
            "personality": row["personality"],
        })
    print(f"data size:{len(data)}")
    return data


def build_prompt_example_text(examples):
    pe_list = []
    for pe in examples:
        if "slots" in pe["mr"]:
            pe_mr = pe["mr"]["slots"]
        else:
            pe_mr = pe["mr"]

        if "ref" in pe:
            text = pe["ref"]
        else:
            text = pe["surface_form"]
        mr_str = []
        mr_dict = defaultdict(list)
        for (a, v) in dh.make_mr_list(pe_mr):
            mr_dict[a].append(v)
        for a, vals in mr_dict.items():
            v = "; ".join(vals)
            mr_str.append(f"({a}: {v})")
        mr_str = "; ".join(mr_str)
        pe_list.append(f"-EXAMPLE-\nText: {text}\nList: {dh.LIST_START}{mr_str}{dh.LIST_END}")
    pe_list = "\n".join(pe_list)

    return pe_list


def main_batch():
    # Deferred: select_with_facloc is the Facility-Location few-shot
    # selector. Migration plan v2 §3 puts it under xdomain_ser.data_prep;
    # if data_prep hasn't been migrated yet, surface a clear error here
    # rather than at module import.
    try:
        from xdomain_ser.data_prep.merge_select import select_with_facloc
    except ImportError as e:
        raise ImportError(
            "xdomain_ser.eval2.personage_inference requires "
            "xdomain_ser.data_prep.merge_select.select_with_facloc, "
            "which is part of the data_prep migration step. "
            f"Original error: {e}"
        )

    parser = argparse.ArgumentParser(description="LoRA MR extraction on PERSONAGE M2T outputs")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--eval_path", type=str, required=True, help="Path to evaluation data JSON file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=70)
    parser.add_argument("--max_input_length", type=int, default=800)
    parser.add_argument("--topic_slots_path", type=str, default=None)
    parser.add_argument("--prompt_examples_path", type=str, default=None)
    parser.add_argument("--hint_map_path", type=str, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--num_pe", type=int, default=5)
    parser.add_argument("--output_path", type=str, default=None, help="Path to predictions (json)")
    parser.add_argument("--alpha_text", type=float, default=0.3)
    parser.add_argument("--alpha_slots", type=float, default=0.7)
    parser.add_argument("--load_predictions", type=str, default=None, help="Path to predictions (json)")
    parser.add_argument("--is_pbl", action="store_true", default=False)
    args = parser.parse_args()
    print("args:", args)

    with open(args.hint_map_path, "r") as fin:
        hint_map = json.load(fin)
    if args.is_pbl:
        hint_map = hint_map["hm_e2e_nlg"]

    with open(args.prompt_examples_path, "r") as fin:
        prompt_examples = json.load(fin)

    personality_prompt_examples = defaultdict(list)
    for pex in prompt_examples:
        personality_prompt_examples[pex["personality"]].append(pex)

    for personality, examples in personality_prompt_examples.items():
        print(f"selecting {personality} prompt examples ...")
        selected, info = select_with_facloc(examples, args.num_pe, alpha_text=args.alpha_text,
                                            alpha_slots=args.alpha_slots, include_dact=False)
        pexs = [examples[i] for i in selected]
        pexs = preprocess_examples(pexs, text_key="ref")
        personality_prompt_examples[personality] = build_prompt_example_text(pexs)

    with open(args.eval_path) as fin:
        inference_examples = json.load(fin)

    if args.is_pbl:
        for ex in inference_examples:
            ex["eval_this_pred"] = ex["clean_pred"][0]["text"]
        text_key = "eval_this_pred"
    else:
        text_key = "pred"
    processed_examples = preprocess_examples(inference_examples, text_key=text_key)
    assert len(processed_examples) == len(inference_examples)
    outpath = args.output_path
    print("Saving predictions to", outpath)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.load_predictions is not None:
        with open(args.load_predictions, "r") as fin:
            results = json.load(fin)
    else:
        results = []

    gen_batch_size = getattr(args, "batch_size", 8)
    num_ret = args.num_beams

    save_steps = 10
    for start in tqdm(range(0, len(inference_examples), gen_batch_size), desc="Generating"):
        if len(results) > start:
            continue

        batch = [processed_examples[i] for i in range(start, min(start + gen_batch_size, len(inference_examples)))]

        prompts = []
        metas = []
        for offset, example in enumerate(batch):
            input_text = example["surface_form"]
            _prompt_examples = personality_prompt_examples[example["personality"]]
            _, prompt = dh.build_prompt_hint_map(example["mr"], input_text, hint_map, _prompt_examples,
                                                 make_output=False)
            prompts.append(prompt)
            metas.append((start + offset))

        enc = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                top_p=None,
                temperature=None,
                stop_strings=[dh.LIST_END, "\n\n###"],
                tokenizer=tokenizer,
                num_beams=num_ret,
                num_return_sequences=num_ret,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        in_lengths = enc["attention_mask"].sum(dim=1).tolist()

        for i, ds_index in enumerate(metas):
            preds_i = decoded[i * num_ret:(i + 1) * num_ret]
            cleaned = [_extract_response(p) for p in preds_i]

            ex = dict(inference_examples[ds_index])
            ex["pred_mr"] = cleaned
            ex["pred_input_token_length"] = [int(in_lengths[i])] * num_ret
            ex["prompt"] = prompts[i]
            results.append(ex)

        if len(results) % save_steps == 0:
            with open(outpath, "w") as fout:
                json.dump(results, fout, indent=2)

    print("Saving predictions to", outpath)
    with open(outpath, "w") as fout:
        json.dump(results, fout, indent=2)


if __name__ == "__main__":
    main_batch()
