# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Score k MR candidates with the trained LoRA ranker.

Reads a JSON file of examples each with ``pred_mr`` (list of k candidates),
loads the LoRA ranker on top of Llama-3.2-3B-Instruct, scores each
candidate via probability-weighted sum over the 7-grade rubric digits
(0..6), and writes scores out. ``--ref_source`` selects what fills the
ranker prompt's Gold Reference field:

* ``true`` (default) -- the example's true MR, once per text; writes the
  input back out with a per-example ``pred_scores`` field. This is the
  published (P-oracle) protocol, byte-compatible with the original runs.
* ``hypothesis`` -- each negative's modified MR, once per (example,
  negative) pair; writes ``{"params": ..., "records": [...]}`` with one
  record per pair keyed by (conversation_id, ex_idx, neg_idx). This is
  the corrected (P-deploy) protocol.
* ``none`` -- blank reference, once per text (ablation); per-text output
  like ``true``.

Beam candidates always come from the input file's ``pred_mr``; this
script never regenerates beams.

Reproducing the published numerics requires ``--compute_dtype float32``
(the published runs' dtype): candidate scores sit on near-ties, so the
faster bf16 default flips a substantial share of top-1 selections and
visibly moves selection-based metrics (see REPRODUCE.md).
"""
import argparse
import json

import torch
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from xdomain_ser.core import data_helper as dh
from xdomain_ser.core import ser
from xdomain_ser.ranking import data as rdh


@torch.inference_mode()
def score_candidates(model, tokenizer, candidate_prompts, device=None):
    """Local convenience wrapper (kept for backwards compatibility).

    Identical to :func:`xdomain_ser.ranking.data.score_candidates`. The
    production path uses ``rdh.score_candidates`` directly; this is here
    so external callers importing from ``score.py`` still resolve.
    """
    device = device or model.device
    toks = tokenizer(candidate_prompts, return_tensors="pt", padding=True).to(device)
    out = model(**toks)
    last_logits = out.logits[:, -1, :]  # [B, V]
    digit_ids = torch.tensor([tokenizer.convert_tokens_to_ids(d) for d in rdh.DIGITS], device=device)
    sel = last_logits.index_select(dim=1, index=digit_ids)
    probs = sel.softmax(dim=-1)
    probs = probs.detach().float().cpu().tolist()
    return probs


def main():
    parser = argparse.ArgumentParser(description="Score k MR candidates with the trained LoRA ranker")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to ranker checkpoint")
    parser.add_argument("--eval_path", type=str, required=True, help="Path to candidates JSON file")
    parser.add_argument("--hint_map_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--ref_source", choices=["true", "hypothesis", "none"], default="true",
                        help="What fills the ranker's Gold Reference field: the example's "
                             "true MR (published per-text protocol), each negative's "
                             "modified MR (per-pair, corrected protocol), or blank "
                             "(per-text ablation).")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Chunk the scoring forward pass (peak-memory bound).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N examples (smoke runs).")
    parser.add_argument("--compute_dtype", choices=["bfloat16", "float32"],
                        default="bfloat16",
                        help="4-bit compute dtype. bfloat16 is the fast default; "
                             "float32 matches the published runs' numerics "
                             "(scores sit on near-ties, so dtype moves "
                             "selection-based metrics).")

    args = parser.parse_args()
    print("args:", args)

    compute_dtype = torch.float32 if args.compute_dtype == "float32" else torch.bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
    )


    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # better for batched generation with KV cache
    model.config.pad_token_id = tokenizer.pad_token_id

    with open(args.hint_map_path, "r") as fin:
        hint_map_table = json.load(fin)

    with open(args.eval_path, "r") as fin:
        eval_data = json.load(fin)
    if args.limit:
        eval_data = eval_data[:args.limit]

    if args.ref_source == "hypothesis":
        # Corrected (P-deploy) protocol: one scoring pass per (example,
        # negative) pair, reference = the pair's modified MR.
        records = []
        for ex_idx, ex in enumerate(tqdm(eval_data)):
            if not isinstance(ex.get("pred_mr"), list):
                continue
            hm = hint_map_table[ex["hint_map_id"].lower()]["hint_map"]
            cands = [dh.make_mr_list(ser.extract_attributes_dict(c))
                     for c in ex["pred_mr"]]
            for neg_idx, neg in enumerate(ex.get("negatives", [])):
                ref_mr = dh.make_mr_list(neg["mr"])
                prompts = [rdh.build_prompt(c, ex["surface_form"], hm, ref_mr)[1] + "\n"
                           for c in cands]
                probs = rdh.score_candidates(model, tokenizer, prompts,
                                             batch_size=args.batch_size)
                records.append({
                    "conversation_id": ex.get("conversation_id"),
                    "ex_idx": ex_idx,
                    "neg_idx": neg_idx,
                    "neg_label": neg.get("label"),
                    "pred_scores": [rdh.compute_probs_score(p) for p in probs],
                })
        output = {
            "params": {"ref_source": "hypothesis",
                       "checkpoint_dir": args.checkpoint_dir,
                       "eval_path": args.eval_path,
                       "dtype": f"4bit-{args.compute_dtype}"},
            "records": records,
        }
    else:
        # Per-text path: 'true' reproduces the published behavior exactly;
        # 'none' blanks the reference field (ablation).
        for ex in tqdm(eval_data):
            if not isinstance(ex.get("pred_mr"), list):
                continue
            candidates = ex["pred_mr"]
            hm_id = ex["hint_map_id"].lower()
            hm = hint_map_table[hm_id]["hint_map"]
            ref_mr = ex["mr"] if args.ref_source == "true" else []
            prompts = []
            for c in candidates:
                c = dh.make_mr_list(ser.extract_attributes_dict(c))
                _, prompt = rdh.build_prompt(c, ex["surface_form"], hm, ref_mr)
                prompts.append(prompt + "\n")

            scores = rdh.score_candidates(model, tokenizer, prompts,
                                          batch_size=args.batch_size)
            ex["pred_scores"] = [rdh.compute_probs_score(pred_output) for pred_output in scores]
        output = eval_data

    print("saving outputs to", args.output_path)
    with open(args.output_path, "w") as fout:
        json.dump(output, fout, indent=2)


if __name__ == '__main__':
    main()
