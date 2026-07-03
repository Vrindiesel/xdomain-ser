# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""LoRA fine-tuning for the MR-ranking model.

QLoRA on Llama-3.2-3B-Instruct: 4-bit base, configurable LoRA r/alpha/dropout,
bf16 compute. Trains the ranker to score candidate MRs against text + gold
on a 7-grade rubric (0=bad ... 6=perfect). Production checkpoint:
``multi-ser-ranking/exp4/checkpoint-60``.

Authentication: ``huggingface-cli login`` or set ``HF_TOKEN`` before running.
"""
from typing import Optional, Dict, List, Any
import argparse
import json
import random
from dataclasses import dataclass

import numpy as np
import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    Trainer, TrainingArguments,
)

from xdomain_ser.core import data_helper as dh
from xdomain_ser.ranking import data as rdh


is_first = True


def main():
    global is_first
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--hint_map_path", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="logs/")
    parser.add_argument("--per_topic_max", type=int, default=20_000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
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
    parser.add_argument("--use_hintmap", default=False, action="store_true")
    parser.add_argument("--bias", type=str, default="none")
    args = parser.parse_args()
    print("args:", args)

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_name = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token  # LLaMA has no pad token


    with open(args.hint_map_path, "r") as fin:
        hint_map_table = json.load(fin)

    print("\nloading train dataset")
    train_dataset = rdh.load_ranking_dataset(args.train_path, per_topic_max=args.per_topic_max)
    print("\nloading eval dataset")
    eval_dataset = rdh.load_ranking_dataset(args.eval_path, per_topic_max=50)

    # Prompt formatting for instruction tuning
    def format_prompt(example):
        global is_first
        hm_id = example["hint_map_id"].lower()
        hm = None
        if args.use_hintmap:
            hm = hint_map_table[hm_id]["hint_map"]
        output, prompt = rdh.build_prompt(example["mr"], example["text"], hm,
                                          example["gold_mr"], example["label"])
        to_tokenize = prompt + output
        if is_first:
            is_first = False
            print(f"\nFull Example:\n{to_tokenize}\n")
        return tokenizer(to_tokenize, truncation=True, max_length=args.max_len)


    tokenized_train_dataset = train_dataset.map(format_prompt, batched=False)
    is_first = True
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
        fp16=True,
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
