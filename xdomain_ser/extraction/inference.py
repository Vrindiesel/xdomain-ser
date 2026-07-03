# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Batched MR extraction inference with a LoRA-fine-tuned causal LM.

Loads a PEFT adapter on top of Llama-3.2-3B-Instruct, builds hint-map-
conditioned prompts via ``data_helper.build_prompt_hint_map``, runs batched
beam-search decoding with ``LIST_END`` / ``\\n\\n###`` stop-strings, and
writes per-example predictions to JSON. Used by the over-generate-and-rank
pipeline (typically ``--num_beams 10`` to feed candidates into the ranker).
"""
import argparse
import json
import os

import torch
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from xdomain_ser.core import data_helper as dh


def __extract_response(text: str) -> str:
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


def main():
    parser = argparse.ArgumentParser(description="Run inference with a finetuned LLaMA model")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--eval_path", type=str, required=True, help="Path to evaluation data JSON file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--seed", type=int, default=232342)
    parser.add_argument("--per_topic_max", type=int, default=20_000)
    parser.add_argument("--max_new_tokens", type=int, default=70)
    parser.add_argument("--topic_slots_path", type=str, default=None)
    parser.add_argument("--prompt_examples_path", type=str, default=None)
    parser.add_argument("--hint_map_path", type=str, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--output_path", type=str, default=None, help="Path to predictions (json)")
    parser.add_argument("--load_predictions", type=str, default=None, help="Path to predictions (json)")

    args = parser.parse_args()
    print("args:", args)

    # Load base LLM model and tokenizer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
    )


    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # better for batched generation with KV cache
    model.config.pad_token_id = tokenizer.pad_token_id

    with open(args.hint_map_path, "r") as fin:
        hint_map_table = json.load(fin)

    if args.load_predictions is None:
        with open(args.prompt_examples_path, "r") as fin:
            eval_prompt_examples = json.load(fin)

        per_topic_max = args.per_topic_max if args.per_topic_max > 0 else None
        eval_dataset = dh.load_build_dataset2(args.eval_path, args, None,
                                                per_topic_max=per_topic_max, do_shuffle=False,
                                                prompt_examples=eval_prompt_examples, num_pe=5,
                                              return_dataset=False)
    else:
        with open(args.eval_path, "r") as fin:
            eval_dataset = json.load(fin)

    if args.output_path is None:
        name = os.path.basename(args.eval_path)
        outpath = os.path.join(args.checkpoint_dir, name)
    else:
        outpath = args.output_path

    test_examples_path = outpath.replace(".json", ".sel.json")
    print("Saving selected test examples to", test_examples_path)
    with open(test_examples_path, "w") as fout:
        json.dump(eval_dataset, fout, indent=2)
    print("Saving predictions to", outpath)

    if args.load_predictions is not None:
        with open(args.load_predictions, "r") as fin:
            results = json.load(fin)
    else:
        results = []  # collect updated examples here

    # ---- batched inference ----
    gen_batch_size = getattr(args, "batch_size", 8)  # or set a fixed int
    num_ret = args.num_beams  # you already set num_return_sequences = num_beams

    save_steps = 50
    for start in tqdm(range(0, len(eval_dataset), gen_batch_size), desc="Generating"):
        if len(results) > start:
            continue

        batch = [eval_dataset[i] for i in range(start, min(start + gen_batch_size, len(eval_dataset)))]

        # Build prompts for the whole batch
        prompts = []
        metas = []  # (index_in_dataset, input_len placeholder)
        for offset, example in enumerate(batch):
            input_text = example["surface_form"]
            hm_id = example["hint_map_id"].lower()
            hm = hint_map_table[hm_id]
            _prompt_examples = example.get("prompt_examples")
            _, prompt = dh.build_prompt_hint_map(example["mr"], input_text, hm, _prompt_examples)
            prompts.append(prompt)
            metas.append((start + offset))

        # Tokenize as a batch to get attention_mask
        enc = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1100,
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

        # Decode all sequences at once and regroup per input
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # input lengths (tokens) per row of the batch
        in_lengths = enc["attention_mask"].sum(dim=1).tolist()  # len == batch_size

        # For each input i, take its block of num_ret generations
        for i, ds_index in enumerate(metas):
            preds_i = decoded[i * num_ret:(i + 1) * num_ret]
            cleaned = [dh._extract_response(p) for p in preds_i]

            # Clone the original example and attach predictions
            ex = dict(eval_dataset[ds_index])
            ex["pred_mr"] = cleaned
            # replicate the input length once per returned sequence (keeps your original shape)
            ex["pred_input_token_length"] = [int(in_lengths[i])] * num_ret
            results.append(ex)

        if len(results) % save_steps == 0:
            with open(outpath, "w") as fout:
                json.dump(results, fout, indent=2)

    # 'results' now holds each eval example with fields:
    #   - pred_mr: [num_beams strings]
    #   - pred_input_token_length: [num_beams ints]

    print("Saving predictions to", outpath)
    with open(outpath, "w") as fout:
        json.dump(results, fout, indent=2)


if __name__ == "__main__":
    main()
