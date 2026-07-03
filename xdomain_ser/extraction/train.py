# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""LoRA fine-tuning for the multi-domain MR extraction model.

QLoRA on Llama-3.2-3B-Instruct: 4-bit base, configurable LoRA r/alpha/dropout,
Flash Attention 2, bf16 compute. Drives all extraction-model experiments
documented in the GEM paper.

Authentication: ``huggingface-cli login`` or set ``HF_TOKEN`` before running
(the gated Llama-3 base model requires it).
"""
import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

import torch
import tqdm
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    Trainer, TrainingArguments,
)

from xdomain_ser.core import data_helper as dh
from xdomain_ser.core.eval_helper import compute_custom_metrics_stub


class PostEvalTrainer(Trainer):
    def evaluate(self, *args, **kwargs):
        metrics = super().evaluate(*args, **kwargs)  # <-- runs the normal eval loop
        self._post_eval(metrics)                     # <-- your custom hook
        return metrics

    def _post_eval(self, metrics: dict):
        """
        Runs immediately after the evaluation loop.
        You can use self.model, self.tokenizer, self.eval_dataset, etc.
        """
        # --- EXAMPLE: generate model outputs for the eval set (greedy) ---
        self.model.eval()
        preds_text = []
        gen_kwargs = dict(max_new_tokens=128, do_sample=False, num_beams=1, top_p=None, temperature=None)

        eval_dataloader = self.get_eval_dataloader()
        for batch in tqdm.tqdm(eval_dataloader):
            # If you're on a single GPU this is fine:
            input_ids = batch["input_ids"].to(self.model.device)
            attention_mask = batch["attention_mask"].to(self.model.device)

            with torch.no_grad():
                gen = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs
                )

            # Slice off the prompt to keep only the newly generated tokens
            gen_only = gen[:, input_ids.shape[1]:]
            preds_text.extend(self.tokenizer.batch_decode(gen_only, skip_special_tokens=True))

        # --- TODO: compute your custom metrics here ---
        # If you need golds, pull them from your *raw* eval set (preferred),
        # or store whatever you need in the tokenized dataset columns before training.
        # Example stub:
        custom = compute_custom_metrics_stub(preds_text, getattr(self, "raw_eval_examples", None))

        # Merge with HF metrics, log, and persist
        out = {**metrics, **{f"custom/{k}": v for k, v in custom.items()}}
        self.log(out)
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(os.path.join(self.args.output_dir, "post_eval_metrics.json"), "w") as f:
            json.dump(out, f, indent=2)


def extract_attributes_dict(mr_str):
    #print("mr_str", mr_str)
    if "\n" in mr_str:
        mr_str = mr_str.split("\n")[0]
    slots = [tok.strip() for tok in mr_str.split(",")]
    d = {}
    for slot in slots:
        #print("slot", slot)
        if ":" not in slot:
            break
        a,v = slot.split(":")
        a,v = a.strip(), v.strip()
        #if a not in SLOTS:
        #    break
        if v not in {"None"}:
            d[a] = v

    collapsed = d
    #print("collapsed", collapsed)
    #input(">>>")
    return collapsed


is_first = True
def main():
    global is_first
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--topic_slots_path", type=str, default=None)
    parser.add_argument("--prompt_examples_path", type=str, default=None)
    parser.add_argument("--eval_prompt_examples_path", type=str, default=None)
    parser.add_argument("--hint_map_path", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--per_topic_max", type=int, default=20_000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    #parser.add_argument("learning_data_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--train_epochs", type=int, default=1)
    parser.add_argument("--min_mr_length", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--log_steps", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--bias", type=str, default="none")
    args = parser.parse_args()
    print("args:", args)

    # Load tokenizer and model with 4-bit quantization
    model_name = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token  # LLaMA has no pad token

    if args.eval_prompt_examples_path is not None:
        with open(args.eval_prompt_examples_path, "r") as fin:
            eval_prompt_examples = json.load(fin)

    if args.prompt_examples_path is not None:
        with open(args.prompt_examples_path, "r") as fin:
            prompt_examples = json.load(fin)

    with open(args.hint_map_path, "r") as fin:
        hint_map_table = json.load(fin)

    # Prompt formatting for instruction tuning
    def format_prompt(example):
        global is_first
        input_text = example["surface_form"]
        hm_id = example["hint_map_id"].lower()
        hm = hint_map_table[hm_id]

        _prompt_examples = example["prompt_examples"] if "prompt_examples" in example else None

        output, prompt = dh.build_prompt_hint_map(example["mr"], input_text, hm, _prompt_examples)
        to_tokenize = prompt + output
        if is_first:
            is_first = False
            print(f"\nFull Example:\n{to_tokenize}\n")
        return tokenizer(to_tokenize, truncation=True, max_length=args.max_len)

    per_topic_max = args.per_topic_max
    train_dataset = dh.load_build_dataset2(args.train_path, args, None,
                                            per_topic_max, do_shuffle=True, min_mr_size=args.min_mr_length,
                                            prompt_examples=prompt_examples, num_pe=5)
    tokenized_train_dataset = train_dataset.map(format_prompt, batched=False)
    is_first = True
    eval_dataset = dh.load_build_dataset2(args.eval_path, args, None,
                                            per_topic_max=200, do_shuffle=True,
                                            prompt_examples=eval_prompt_examples, num_pe=5)
    tokenized_eval_dataset = eval_dataset.map(format_prompt, batched=False)
    print("\nTrain Dataset:")
    dh.print_dataset_stats(tokenized_train_dataset)
    print("\nEval Dataset:")
    dh.print_dataset_stats(tokenized_eval_dataset)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias=args.bias,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],  # LLaMA specifics
        task_type=TaskType.CAUSAL_LM,
        use_rslora=True,
        init_lora_weights="eva"
    )

    model = get_peft_model(model, lora_config)

    dh.print_trainable_parameters(model)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.train_epochs,
        logging_dir=args.log_dir,
        logging_steps=args.log_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        bf16=True,
        tf32=True,
        report_to="none",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        label_names=["labels"],
        max_grad_norm=args.max_grad_norm,
    )

    # -------------------------
    # Custom collator: dynamic padding + label masking
    # -------------------------
    @dataclass
    class DataCollatorForCausalLMDynamicPad:
        tokenizer: Any
        pad_to_multiple_of: Optional[int] = 8  # set None to disable

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            batch = self.tokenizer.pad(
                features,
                padding=True,  # longest in the batch
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            # For causal LM, labels are input_ids with pads ignored (-100)
            labels = batch["input_ids"].clone()
            labels[batch["attention_mask"] == 0] = -100
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorForCausalLMDynamicPad(tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    # give your hook access to raw eval examples if you need them
    trainer.raw_eval_examples = eval_dataset  # the *untokenized* dataset you loaded earlier

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
