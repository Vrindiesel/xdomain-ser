# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Top-level CLI for ``xdomain-ser``.

Six subcommands:

* ``score``     -- extract MR from text, compute SER + slot-F1 vs gold MR
* ``extract``   -- extract MR from text only
* ``rank``      -- score k candidate MRs against text+gold with the ranker
* ``nli``       -- NLI-based MR recovery from text + gold slots
* ``route``     -- LoRA extraction + NLI verification + score-threshold routing
* ``reproduce`` -- thin wrapper over ``scripts/reproduce_table_N.sh`` (Stage 9)

Single-example mode is the primary surface (``--text`` + ``--gold-mr``);
each subcommand also accepts ``--input-file`` for batch JSON inputs and
``--limit N`` for quick smoke tests on the first N records.
"""
import json
import random
import subprocess
from pathlib import Path
from typing import Optional

import typer

from xdomain_ser.core import data_helper as dh
from xdomain_ser.core import registry
from xdomain_ser.core import ser

app = typer.Typer(
    name="xdomain-ser",
    help="Cross-domain SER evaluation CLI (GEM @ ACL 2026 companion).",
    no_args_is_help=True,
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Domain aliases + E2E translation tables
# ---------------------------------------------------------------------------

DOMAIN_ALIASES = {
    "e2e": "hm_e2e_nlg",
    "viggo": "hm_viggo",
    "rnnlg_hotel": "hm_rnnlg_hotel",
    "rnnlg_laptop": "hm_rnnlg_laptop",
    "rnnlg_restaurant": "hm_rnnlg_restaurant",
    "rnnlg_tv": "hm_rnnlg_tv",
    "tm1_auto_repair": "hm_tm1_auto_repair_appt",
    "tm1_coffee_ordering": "hm_tm1_coffee_ordering",
    "tm1_movie_tickets": "hm_tm1_movie_tickets",
    "tm1_pizza_ordering": "hm_tm1_pizza_ordering",
    "tm1_restaurant_table": "hm_tm1_restaurant_table",
    "tm1_uber_lyft": "hm_tm1_uber_lyft",
}

# The E2E LoRA extractor produces LoRA-internal slot names; gold MRs from
# users are typically in E2E-native names. Translate the prediction back
# before comparison. (Other domains' LoRA-internal names match the native
# names already, so no translation table is needed.)
E2E_LORA_TO_NATIVE_NAME = {
    "venue_type": "eatType",
    "cuisine_type": "food",
    "area_zone": "area",
    "family_suitability": "familyFriendly",
    "nearby_landmark": "near",
}

E2E_LORA_TO_NATIVE_VALUE = {
    ("family_suitability", "family-friendly"): "yes",
    ("family_suitability", "not-family-friendly"): "no",
}


def _resolve_domain(domain: str) -> str:
    """Resolve a domain alias (e.g. 'e2e') to its hint_map_id (e.g. 'hm_e2e_nlg')."""
    if domain in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[domain]
    if domain.startswith("hm_"):
        return domain
    # Permissive fallback: 'foo' -> 'hm_foo'
    return f"hm_{domain}"


def _translate_lora_mr(pred_mr_dict: dict, domain: str) -> dict:
    """Translate LoRA-internal slot names back to domain-native names.

    Only E2E currently has a non-identity mapping. Returns a fresh dict
    with single-value scalars (not lists) for downstream SER comparison.
    """
    if domain != "e2e":
        # For non-E2E domains, just flatten lists to scalars.
        return {k: (v[0] if isinstance(v, list) and v else v) for k, v in pred_mr_dict.items()}

    out = {}
    for slot, vals in pred_mr_dict.items():
        if not isinstance(vals, list):
            vals = [vals]
        new_slot = E2E_LORA_TO_NATIVE_NAME.get(slot, slot)
        new_vals = []
        for v in vals:
            tv = E2E_LORA_TO_NATIVE_VALUE.get((slot, v), v)
            new_vals.append(tv)
        out[new_slot] = new_vals[0] if len(new_vals) == 1 else new_vals
    return out


def _parse_mr_arg(arg: str) -> dict:
    """Parse a gold-MR CLI argument as either JSON or '<LIST>(slot: val); ...</LIST>'."""
    arg = arg.strip()
    if arg.startswith("{"):
        return json.loads(arg)
    if arg.startswith("<LIST>") or "(" in arg:
        # Strip LIST_END (parser quirk documented in Stage 2 RELEASE_NOTES)
        normalised = arg.replace(dh.LIST_END, "")
        return dict(ser.extract_attributes_dict(normalised))
    raise typer.BadParameter(f"Cannot parse --gold-mr; expected JSON or <LIST>... format. Got: {arg[:60]!r}")


# ---------------------------------------------------------------------------
# Lazy model + asset loaders
# ---------------------------------------------------------------------------

_STATE = {}


def _attn_impl() -> str:
    """Pick the attention backend: Flash Attention 2 if importable, else eager.

    transformers raises a hard ImportError on ``attn_implementation=
    'flash_attention_2'`` when ``flash_attn`` isn't installed; we don't
    want fresh installs to crash on the first inference call, so detect
    once at module load and pass the right value.
    """
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "eager"


_ATTN_IMPL = _attn_impl()


def _resolve_checkpoint_dir(checkpoint_dir: Optional[Path]) -> Path:
    """Default extractor checkpoint location."""
    if checkpoint_dir is not None:
        return checkpoint_dir
    default = registry.REPO_ROOT / "models" / "extractor"
    if not default.exists():
        raise typer.BadParameter(
            f"No extractor checkpoint at {default}. Provide --checkpoint-dir or "
            f"run scripts/download_models.sh."
        )
    return default


def _resolve_ranker_dir(ranker_dir: Optional[Path]) -> Path:
    if ranker_dir is not None:
        return ranker_dir
    default = registry.REPO_ROOT / "models" / "ranker"
    if not default.exists():
        raise typer.BadParameter(
            f"No ranker checkpoint at {default}. Provide --ranker-dir or "
            f"run scripts/download_models.sh."
        )
    return default


def _resolve_hint_map_path(hint_map_path: Optional[Path]) -> Path:
    if hint_map_path is not None:
        return hint_map_path
    if registry.HINT_MAPS.exists():
        return registry.HINT_MAPS
    raise typer.BadParameter(
        f"No hint-map JSON found. Provide --hint-map-path; tried {registry.HINT_MAPS}."
    )


def _resolve_prompt_examples_path(prompt_examples_path: Optional[Path]) -> Path:
    if prompt_examples_path is not None:
        return prompt_examples_path
    if registry.PROMPT_EXAMPLES.exists():
        return registry.PROMPT_EXAMPLES
    raise typer.BadParameter(
        f"No prompt-examples JSON found. Provide --prompt-examples-path; "
        f"tried {registry.PROMPT_EXAMPLES}."
    )


def _load_extractor(checkpoint_dir: Path, hint_map_path: Path, prompt_examples_path: Path):
    """Lazy-load + cache the LoRA extractor model, tokenizer, hint maps, and prompt examples."""
    key = ("extractor", str(checkpoint_dir), str(hint_map_path), str(prompt_examples_path))
    if key in _STATE:
        return _STATE[key]

    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, BitsAndBytesConfig

    typer.echo(f"Loading extractor from {checkpoint_dir} ...", err=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
        attn_implementation=_ATTN_IMPL,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id

    with open(hint_map_path) as f:
        hint_map_table = json.load(f)
    with open(prompt_examples_path) as f:
        prompt_examples_pool = json.load(f)

    _STATE[key] = (model, tokenizer, hint_map_table, prompt_examples_pool)
    return _STATE[key]


def _load_nli(model_name: str = "roberta-large-mnli"):
    key = ("nli", model_name)
    if key in _STATE:
        return _STATE[key]
    from xdomain_ser.nli.model import NLIModel
    _STATE[key] = NLIModel(model_name=model_name)
    return _STATE[key]


def _load_ranker(ranker_dir: Path):
    key = ("ranker", str(ranker_dir))
    if key in _STATE:
        return _STATE[key]

    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, BitsAndBytesConfig

    typer.echo(f"Loading ranker from {ranker_dir} ...", err=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(ranker_dir), torch_dtype=torch.bfloat16, quantization_config=bnb,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(ranker_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.pad_token_id = tokenizer.pad_token_id
    _STATE[key] = (model, tokenizer)
    return _STATE[key]


def _unload_extractor() -> bool:
    """Evict the cached LoRA extractor and release its GPU memory.

    The extractor and ranker are both 4-bit Llama-3.2-3B models; keeping the
    extractor resident while the ranker runs its (float32) scoring forward pass
    can exhaust a 24GB GPU. Call this once extraction is finished and before
    ranking. Returns True if an extractor was evicted.

    NOTE: any caller-held references to the extractor model/tokenizer must be
    dropped (e.g. ``del model, tokenizer``) before calling this, otherwise the
    weights stay alive and the memory is not actually freed.
    """
    import gc

    removed = False
    for key in [k for k in list(_STATE)
                if isinstance(k, tuple) and k and k[0] == "extractor"]:
        entry = _STATE.pop(key)
        del entry
        removed = True
    if removed:
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return removed


# ---------------------------------------------------------------------------
# Single-example inference helpers
# ---------------------------------------------------------------------------

def _format_few_shot_examples(pool: dict, hm_id: str, n: int = 5, seed: int = 42) -> str:
    """Sample n few-shot examples for the requested hint_map_id and format them."""
    rng = random.Random(seed)
    candidates = pool.get(hm_id)
    if not candidates:
        # No examples for this domain -- fall back to empty.
        return ""
    n = min(n, len(candidates))
    sampled = rng.sample(candidates, n)
    formatted = []
    for ex in sampled:
        text, mr_str = dh.make_prompt_example(ex)
        formatted.append(f"-EXAMPLE-\nText: {text}\nList: {mr_str}")
    return "\n".join(formatted)


def _extract_one(text: str, hm_id: str, model, tokenizer, hint_map_table, prompt_examples_pool,
                 max_new_tokens: int = 80, num_beams: int = 1) -> str:
    """Run extraction inference on a single text. Returns the predicted MR string."""
    import torch

    if hm_id not in hint_map_table:
        raise typer.BadParameter(
            f"Hint map id {hm_id!r} not in hint-map JSON. Available: "
            f"{sorted(hint_map_table.keys())[:10]}..."
        )
    hint_map = hint_map_table[hm_id]
    prompt_examples_str = _format_few_shot_examples(prompt_examples_pool, hm_id)
    _, prompt = dh.build_prompt_hint_map(
        mr=[],  # not used when make_output=False
        input_text=text,
        hint_map=hint_map,
        prompt_examples=prompt_examples_str,
        make_output=False,
    )
    enc = tokenizer(prompt, padding=True, truncation=True, max_length=1100,
                    return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=None,
            temperature=None,
            stop_strings=[dh.LIST_END, "\n\n###"],
            tokenizer=tokenizer,
            num_beams=num_beams,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    marker = "### Response:\n"
    i = decoded.find(marker)
    if i != -1:
        decoded = decoded[i + len(marker):]
    for stop in (dh.LIST_END, "\n\n###"):
        if stop in decoded:
            decoded = decoded.split(stop, 1)[0]
    return decoded.strip()

def _extract_n(text: str, hm_id: str, model, tokenizer, hint_map_table, prompt_examples_pool,
                 max_new_tokens: int = 80, num_beams: int = 1) -> str:
    """Run extraction inference on a single text. Returns the predicted MR string."""
    import torch

    if hm_id not in hint_map_table:
        raise typer.BadParameter(
            f"Hint map id {hm_id!r} not in hint-map JSON. Available: "
            f"{sorted(hint_map_table.keys())[:10]}..."
        )
    hint_map = hint_map_table[hm_id]
    prompt_examples_str = _format_few_shot_examples(prompt_examples_pool, hm_id)
    _, prompt = dh.build_prompt_hint_map(
        mr=[],  # not used when make_output=False
        input_text=text,
        hint_map=hint_map,
        prompt_examples=prompt_examples_str,
        make_output=False,
    )
    enc = tokenizer(prompt, padding=True, truncation=True, max_length=1100,
                    return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=None,
            temperature=None,
            stop_strings=[dh.LIST_END, "\n\n###"],
            tokenizer=tokenizer,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Decode all sequences at once and regroup per input
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # input lengths (tokens) per row of the batch
    in_lengths = enc["attention_mask"].sum(dim=1).tolist()  # len == batch_size
    #print("enc.size():", enc["attention_mask"].size())
    #print("in_lengths:", in_lengths)

    #print("decoded:", len(decoded))
    cleaned = [dh._extract_response(p) for p in decoded]

    #marker = "### Response:\n"
    #i = decoded.find(marker)
    #if i != -1:
    #    decoded = decoded[i + len(marker):]
    #for stop in (dh.LIST_END, "\n\n###"):
    #    if stop in decoded:
    #        decoded = decoded.split(stop, 1)[0]
    #return decoded.strip()
    return cleaned


def _ser_report(pred_mr: dict, gold_mr: dict) -> dict:
    """Run SER + slot-F1 on a single (pred, gold) pair."""
    ser_res = ser.compute_ser(pred_mr, gold_mr)
    f1_res = ser.compute_slot_f1(pred_mr, gold_mr)
    return {
        "SER": ser_res["SER"],
        "S": ser_res["S"],
        "D": ser_res["D"],
        "I": ser_res["I"],
        "N_ref": ser_res["N_ref"],
        "slot_f1": f1_res["f1"],
        "slot_precision": f1_res["precision"],
        "slot_recall": f1_res["recall"],
        "tp": f1_res["tp"],
        "fp": f1_res["fp"],
        "fn": f1_res["fn"],
    }



# ---------------------------------------------------------------------------
# Subcommand: score
# ---------------------------------------------------------------------------

@app.command()
def score(
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Input text."),
    gold_mr: Optional[str] = typer.Option(None, "--gold-mr", "-g",
                                          help="Gold MR as JSON object or <LIST>(slot: val); ...</LIST>."),
    domain: Optional[str] = typer.Option(None, "--domain", "-d",
                                         help="Domain alias (e.g. 'e2e', 'viggo', 'rnnlg_hotel') or full 'hm_*' id."),
    input_file: Optional[Path] = typer.Option(None, "--input-file", "-i",
                                              help="Batch JSON: list of {text, mr, hint_map_id}."),
    output_file: Optional[Path] = typer.Option(None, "--output-file", "-o",
                                               help="Write per-example results JSON here."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Process only the first N batch records."),
    checkpoint_dir: Optional[Path] = typer.Option(None, "--checkpoint-dir"),
    hint_map_path: Optional[Path] = typer.Option(None, "--hint-map-path"),
    prompt_examples_path: Optional[Path] = typer.Option(None, "--prompt-examples-path"),
    num_beams: int = typer.Option(1, "--num-beams"),
    max_new_tokens: int = typer.Option(80, "--max-new-tokens"),
):
    """Extract MR from text and compute SER + slot-F1 against gold MR."""
    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir)
    hint_map_path = _resolve_hint_map_path(hint_map_path)
    prompt_examples_path = _resolve_prompt_examples_path(prompt_examples_path)
    model, tokenizer, hint_map_table, pool = _load_extractor(
        checkpoint_dir, hint_map_path, prompt_examples_path
    )

    # Build the list of examples to score
    if input_file is not None:
        with open(input_file) as f:
            examples = json.load(f)
        if limit:
            examples = examples[:limit]
    else:
        if not (text and gold_mr and domain):
            raise typer.BadParameter("Provide --text + --gold-mr + --domain, or --input-file.")
        examples = [{"text": text, "mr": _parse_mr_arg(gold_mr), "hint_map_id": _resolve_domain(domain), "_domain_alias": domain}]

    results = []
    for ex in examples:
        ex_text = ex["text"]
        ex_gold = ex["mr"]
        hm_id = ex.get("hint_map_id") or _resolve_domain(ex.get("_domain_alias") or domain)
        domain_alias = ex.get("_domain_alias", domain or "")

        pred_str = _extract_one(ex_text, hm_id, model, tokenizer, hint_map_table, pool,
                                max_new_tokens=max_new_tokens, num_beams=num_beams)
        pred_dict = dict(ser.extract_attributes_dict(pred_str.replace(dh.LIST_END, "")))
        pred_native = _translate_lora_mr(pred_dict, domain_alias)
        report = _ser_report(pred_native, ex_gold)

        results.append({
            "text": ex_text,
            "gold_mr": ex_gold,
            "pred_mr_lora": pred_dict,
            "pred_mr_native": pred_native,
            "pred_str": pred_str,
            "scores": report,
        })

    if input_file is None:
        # Single-example: pretty-print
        r = results[0]
        typer.echo("")
        typer.echo("=== xdomain-ser score ===")
        typer.echo(f"Text:           {r['text']}")
        typer.echo(f"Domain:         {domain}  ({_resolve_domain(domain)})")
        typer.echo(f"Gold MR:        {json.dumps(r['gold_mr'])}")
        typer.echo(f"Predicted MR:   {json.dumps(r['pred_mr_native'])}")
        typer.echo("")
        s = r["scores"]
        typer.echo(f"  SER:          {s['SER']:.4f}  (S={s['S']}  D={s['D']}  I={s['I']}  N_ref={s['N_ref']})")
        typer.echo(f"  Slot F1:      {s['slot_f1']:.4f}  (precision={s['slot_precision']:.4f}, recall={s['slot_recall']:.4f})")
        typer.echo(f"  TP/FP/FN:     {s['tp']}/{s['fp']}/{s['fn']}")
    else:
        # Batch: summary
        n = len(results)
        avg_ser = sum(r["scores"]["SER"] for r in results) / max(n, 1)
        avg_f1 = sum(r["scores"]["slot_f1"] for r in results) / max(n, 1)
        typer.echo(f"\nProcessed {n} examples. Mean SER={avg_ser:.4f}, Mean slot-F1={avg_f1:.4f}")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        typer.echo(f"Results written to {output_file}")


# ---------------------------------------------------------------------------
# Subcommand: extract
# ---------------------------------------------------------------------------

@app.command()
def extract(
    text: Optional[str] = typer.Option(None, "--text", "-t"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    input_file: Optional[Path] = typer.Option(None, "--input-file", "-i"),
    output_file: Optional[Path] = typer.Option(None, "--output-file", "-o"),
    limit: Optional[int] = typer.Option(None, "--limit"),
    checkpoint_dir: Optional[Path] = typer.Option(None, "--checkpoint-dir"),
    hint_map_path: Optional[Path] = typer.Option(None, "--hint-map-path"),
    prompt_examples_path: Optional[Path] = typer.Option(None, "--prompt-examples-path"),
    num_beams: int = typer.Option(1, "--num-beams"),
    max_new_tokens: int = typer.Option(80, "--max-new-tokens"),
):
    """Extract MR from text using the trained LoRA extractor (no gold comparison)."""
    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir)
    hint_map_path = _resolve_hint_map_path(hint_map_path)
    prompt_examples_path = _resolve_prompt_examples_path(prompt_examples_path)
    model, tokenizer, hint_map_table, pool = _load_extractor(
        checkpoint_dir, hint_map_path, prompt_examples_path
    )

    if input_file is not None:
        with open(input_file) as f:
            examples = json.load(f)
        if limit:
            examples = examples[:limit]
    else:
        if not (text and domain):
            raise typer.BadParameter("Provide --text + --domain or --input-file.")
        examples = [{"text": text, "hint_map_id": _resolve_domain(domain), "_domain_alias": domain}]

    results = []
    for ex in examples:
        hm_id = ex.get("hint_map_id") or _resolve_domain(ex.get("_domain_alias") or domain)
        domain_alias = ex.get("_domain_alias", domain or "")
        pred_str = _extract_one(ex["text"], hm_id, model, tokenizer, hint_map_table, pool,
                                max_new_tokens=max_new_tokens, num_beams=num_beams)
        pred_dict = dict(ser.extract_attributes_dict(pred_str.replace(dh.LIST_END, "")))
        pred_native = _translate_lora_mr(pred_dict, domain_alias)
        results.append({
            "text": ex["text"],
            "pred_mr_lora": pred_dict,
            "pred_mr_native": pred_native,
            "pred_str": pred_str,
        })

    if input_file is None:
        r = results[0]
        typer.echo("")
        typer.echo("=== xdomain-ser extract ===")
        typer.echo(f"Text:           {r['text']}")
        typer.echo(f"Predicted MR:   {json.dumps(r['pred_mr_native'])}")
        typer.echo(f"Raw output:     {r['pred_str']}")
    else:
        typer.echo(f"\nExtracted {len(results)} examples.")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        typer.echo(f"Results written to {output_file}")


# ---------------------------------------------------------------------------
# Subcommand: rank
# ---------------------------------------------------------------------------

def rank_n(mr_extractions, hm, ref_mr, extraction_text, model, tokenizer):
    from xdomain_ser.ranking import data as rdh
    #candidates = ex.get("pred_mr", [])
    if not isinstance(mr_extractions, list):
        mr_extractions = [mr_extractions]
    #hm_id = ex["hint_map_id"]
    #hm = hint_map_table[hm_id]["hint_map"]
    #ref_mr = ex["mr"]
    #text = ex.get("text") or ex.get("surface_form", "")
    print("\nbuilding prompts ...")
    print("ref_mr:", ref_mr)
    ref_mr = dh.make_mr_list(ref_mr)
    prompts = []
    for j, extra in enumerate(mr_extractions):
        print(f"{j}. extra:", extra)
        emr_list = dh.make_mr_list(extra)
        print("emr_list: ", emr_list)
        _, prompt = rdh.build_prompt(emr_list, extraction_text, hm, ref_mr)
        prompts.append(prompt + "\n")
    # Chunk the (float32) scoring forward pass to bound peak GPU memory.
    scores = rdh.score_candidates(model, tokenizer, prompts, batch_size=None)
    pred_scores = [rdh.compute_probs_score(s) for s in scores]
    return pred_scores


@app.command()
def rank(
    input_file: Path = typer.Option(..., "--input-file", "-i",
                                    help="JSON list of {text, mr, hint_map_id, pred_mr: [...]}."),
    output_file: Path = typer.Option(..., "--output-file", "-o"),
    ranker_dir: Optional[Path] = typer.Option(None, "--ranker-dir"),
    hint_map_path: Optional[Path] = typer.Option(None, "--hint-map-path"),
    limit: Optional[int] = typer.Option(None, "--limit"),
):
    """Score k candidate MRs per example using the trained LoRA ranker.

    Input file format mirrors the output of ``xdomain-ser extract`` with
    a list of candidate MR strings under ``pred_mr``. Each example needs
    ``text``, ``mr`` (gold reference for the ranker prompt), and
    ``hint_map_id``.
    """
    from xdomain_ser.ranking import data as rdh
    ranker_dir = _resolve_ranker_dir(ranker_dir)
    hint_map_path = _resolve_hint_map_path(hint_map_path)
    model, tokenizer = _load_ranker(ranker_dir)

    with open(hint_map_path) as f:
        hint_map_table = json.load(f)
    with open(input_file) as f:
        examples = json.load(f)
    if limit:
        examples = examples[:limit]

    from tqdm import tqdm

    for ex in tqdm(examples, desc="Scoring"):
        candidates = ex.get("pred_mr", [])
        if not isinstance(candidates, list):
            candidates = [candidates]
        hm_id = ex["hint_map_id"]
        hm = hint_map_table[hm_id]["hint_map"]
        ref_mr = ex["mr"]
        text = ex.get("text") or ex.get("surface_form", "")
        prompts = []
        for c in candidates:
            c_list = dh.make_mr_list(ser.extract_attributes_dict(c))
            _, prompt = rdh.build_prompt(c_list, text, hm, ref_mr)
            prompts.append(prompt + "\n")
        scores = rdh.score_candidates(model, tokenizer, prompts)
        ex["pred_scores"] = [rdh.compute_probs_score(s) for s in scores]

    with open(output_file, "w") as f:
        json.dump(examples, f, indent=2)
    typer.echo(f"Scored {len(examples)} examples; wrote to {output_file}")


# ---------------------------------------------------------------------------
# Subcommand: nli
# ---------------------------------------------------------------------------

@app.command()
def nli(
    text: Optional[str] = typer.Option(None, "--text", "-t"),
    gold_mr: Optional[str] = typer.Option(None, "--gold-mr", "-g"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d",
                                         help="Used only as a label in the output."),
    threshold: float = typer.Option(0.3, "--threshold"),
    nli_model: str = typer.Option("roberta-large-mnli", "--nli-model"),
    input_file: Optional[Path] = typer.Option(None, "--input-file", "-i"),
    output_file: Optional[Path] = typer.Option(None, "--output-file", "-o"),
    limit: Optional[int] = typer.Option(None, "--limit"),
):
    """NLI-based slot verification: which gold MR slots does the text entail?"""
    from xdomain_ser.nli.evaluator import recover_mr_from_nli
    from xdomain_ser.nli.templates import slot_value_to_template

    model = _load_nli(nli_model)

    if input_file is not None:
        with open(input_file) as f:
            examples = json.load(f)
        if limit:
            examples = examples[:limit]
    else:
        if not (text and gold_mr):
            raise typer.BadParameter("Provide --text + --gold-mr or --input-file.")
        examples = [{"text": text, "mr": _parse_mr_arg(gold_mr), "_domain_alias": domain or ""}]

    results = []
    for ex in examples:
        ex_text = ex["text"]
        gold_mr_d = ex["mr"]

        # Build per-slot hypothesis pairs (skip dontcare-mapped slots)
        pairs = []
        pair_index = []
        for slot, val in gold_mr_d.items():
            vals = val if isinstance(val, list) else [val]
            for v in vals:
                tmpl = slot_value_to_template(slot, str(v))
                if tmpl is None:
                    continue
                pairs.append((ex_text, tmpl))
                pair_index.append((slot, v))

        probs = model.batch_entailment(pairs, batch_size=32, show_progress=False) if pairs else []
        per_slot = [(slot, val, float(prob)) for (slot, val), prob in zip(pair_index, probs)]
        recovered = recover_mr_from_nli(gold_mr_d, per_slot, threshold=threshold)

        results.append({
            "text": ex_text,
            "gold_mr": gold_mr_d,
            "nli_probs": per_slot,
            "recovered_mr": recovered,
            "threshold": threshold,
        })

    if input_file is None:
        r = results[0]
        typer.echo("")
        typer.echo("=== xdomain-ser nli ===")
        typer.echo(f"Text:           {r['text']}")
        typer.echo(f"Gold MR:        {json.dumps(r['gold_mr'])}")
        typer.echo(f"Threshold:      {r['threshold']}")
        typer.echo("")
        typer.echo("Per-slot entailment:")
        for slot, val, prob in r["nli_probs"]:
            mark = "kept " if prob > r["threshold"] else "drop "
            typer.echo(f"  {mark} {slot:>20s} = {val!r:<20s}  prob={prob:.4f}")
        typer.echo("")
        typer.echo(f"Recovered MR: {json.dumps(r['recovered_mr'])}")
    else:
        n = len(results)
        typer.echo(f"\nProcessed {n} examples at threshold={threshold}.")

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        typer.echo(f"Results written to {output_file}")


# ---------------------------------------------------------------------------
# Subcommand: route
# ---------------------------------------------------------------------------

@app.command()
def route(
    text: Optional[str] = typer.Option(None, "--text", "-t"),
    gold_mr: Optional[str] = typer.Option(None, "--gold-mr", "-g"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    score_threshold: float = typer.Option(2.85, "--score-threshold",
                                          help="Ranker top-score above which we route to LoRA."),
    nli_threshold: float = typer.Option(0.3, "--nli-threshold"),
    nli_model: str = typer.Option("roberta-large-mnli", "--nli-model"),
    checkpoint_dir: Optional[Path] = typer.Option(None, "--checkpoint-dir"),
    ranker_dir: Optional[Path] = typer.Option(None, "--ranker-dir"),
    hint_map_path: Optional[Path] = typer.Option(None, "--hint-map-path"),
    prompt_examples_path: Optional[Path] = typer.Option(None, "--prompt-examples-path"),
    num_beams: int = typer.Option(10, "--num-beams",
                                  help="Beam width for LoRA candidate generation."),
    max_new_tokens: int = typer.Option(80, "--max-new-tokens"),
    input_file: Optional[Path] = typer.Option(None, "--input-file", "-i"),
    output_file: Optional[Path] = typer.Option(None, "--output-file", "-o"),
    limit: Optional[int] = typer.Option(None, "--limit"),
):
    """LoRA extraction + NLI verification + score-threshold routing on a single example.

    Routes to LoRA if the ranker's top score is >= --score-threshold; else
    falls back to the NLI-recovered MR.
    """
    from xdomain_ser.nli.evaluator import recover_mr_from_nli
    from xdomain_ser.nli.templates import slot_value_to_template
    from xdomain_ser.ranking import data as rdh

    if input_file is not None:
        raise typer.BadParameter("--input-file not yet implemented for `route`; use single-example mode.")
    if not (text and gold_mr and domain):
        raise typer.BadParameter("Provide --text + --gold-mr + --domain.")

    gold_d = _parse_mr_arg(gold_mr)
    hm_id = _resolve_domain(domain)

    # ---- LoRA path: beam-search candidates + ranker score
    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_dir)
    hint_map_path = _resolve_hint_map_path(hint_map_path)
    prompt_examples_path = _resolve_prompt_examples_path(prompt_examples_path)
    ranker_dir = _resolve_ranker_dir(ranker_dir)

    model, tokenizer, hint_map_table, pool = _load_extractor(
        checkpoint_dir, hint_map_path, prompt_examples_path
    )

    # Generate num_beams candidates
    import torch
    prompt_examples_str = _format_few_shot_examples(pool, hm_id)
    _, prompt = dh.build_prompt_hint_map(
        mr=[], input_text=text, hint_map=hint_map_table[hm_id],
        prompt_examples=prompt_examples_str, make_output=False,
    )
    enc = tokenizer(prompt, padding=True, truncation=True, max_length=1100,
                    return_tensors="pt").to(model.device)
    with torch.no_grad():
        outs = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, top_p=None, temperature=None,
            stop_strings=[dh.LIST_END, "\n\n###"], tokenizer=tokenizer,
            num_beams=num_beams, num_return_sequences=num_beams,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
    candidates = []
    marker = "### Response:\n"
    for d in decoded:
        i = d.find(marker)
        s = d[i + len(marker):] if i != -1 else d
        for stop in (dh.LIST_END, "\n\n###"):
            if stop in s:
                s = s.split(stop, 1)[0]
        candidates.append(s.strip())

    # Score candidates with ranker
    ranker_model, ranker_tokenizer = _load_ranker(ranker_dir)
    hm = hint_map_table[hm_id]["hint_map"]
    ref_mr_list = dh.make_mr_list(gold_d)
    prompts = []
    for c in candidates:
        c_list = dh.make_mr_list(ser.extract_attributes_dict(c))
        _, p = rdh.build_prompt(c_list, text, hm, ref_mr_list)
        prompts.append(p + "\n")
    raw_scores = rdh.score_candidates(ranker_model, ranker_tokenizer, prompts)
    candidate_scores = [rdh.compute_probs_score(s) for s in raw_scores]
    top_idx = max(range(len(candidate_scores)), key=lambda i: candidate_scores[i])
    top_score = candidate_scores[top_idx]
    lora_mr_str = candidates[top_idx]
    lora_mr_dict = dict(ser.extract_attributes_dict(lora_mr_str.replace(dh.LIST_END, "")))
    lora_mr_native = _translate_lora_mr(lora_mr_dict, domain)

    # ---- NLI path: per-slot entailment
    nli = _load_nli(nli_model)
    pairs = []
    pair_index = []
    for slot, val in gold_d.items():
        vals = val if isinstance(val, list) else [val]
        for v in vals:
            tmpl = slot_value_to_template(slot, str(v))
            if tmpl is None:
                continue
            pairs.append((text, tmpl))
            pair_index.append((slot, v))
    probs = nli.batch_entailment(pairs, batch_size=32, show_progress=False) if pairs else []
    per_slot = [(s, v, float(p)) for (s, v), p in zip(pair_index, probs)]
    nli_mr = recover_mr_from_nli(gold_d, per_slot, threshold=nli_threshold)

    # ---- Routing decision
    route_to_lora = top_score >= score_threshold
    chosen_method = "LoRA" if route_to_lora else "NLI"
    chosen_mr = lora_mr_native if route_to_lora else nli_mr

    # Compute SER for each method against gold
    reports = {
        "LoRA": _ser_report(lora_mr_native, gold_d),
        "NLI": _ser_report(nli_mr, gold_d),
    }
    reports["Routed"] = reports[chosen_method]

    # Output
    typer.echo("")
    typer.echo("=== xdomain-ser route ===")
    typer.echo(f"Text:                {text}")
    typer.echo(f"Domain:              {domain}  ({hm_id})")
    typer.echo(f"Gold MR:             {json.dumps(gold_d)}")
    typer.echo("")
    typer.echo(f"LoRA top score:      {top_score:.4f}  (threshold={score_threshold})")
    typer.echo(f"LoRA MR:             {json.dumps(lora_mr_native)}")
    typer.echo(f"NLI MR (thr={nli_threshold}):     {json.dumps(nli_mr)}")
    typer.echo("")
    typer.echo(f"Routed to:           {chosen_method}  ({'LoRA' if route_to_lora else 'NLI'})")
    typer.echo("")
    typer.echo(f"{'Method':>10} {'SER':>8} {'S':>3} {'D':>3} {'I':>3} {'slot_f1':>8}")
    for name in ("LoRA", "NLI", "Routed"):
        r = reports[name]
        typer.echo(f"{name:>10} {r['SER']:>8.4f} {r['S']:>3} {r['D']:>3} {r['I']:>3} {r['slot_f1']:>8.4f}")

    if output_file:
        with open(output_file, "w") as f:
            json.dump({
                "text": text, "domain": domain, "gold_mr": gold_d,
                "lora": {"mr": lora_mr_native, "score": top_score,
                         "candidates": candidates, "candidate_scores": candidate_scores},
                "nli": {"mr": nli_mr, "per_slot": per_slot, "threshold": nli_threshold},
                "route": {"chosen_method": chosen_method, "chosen_mr": chosen_mr,
                          "score_threshold": score_threshold},
                "reports": reports,
            }, f, indent=2)
        typer.echo(f"\nDetails written to {output_file}")


# ---------------------------------------------------------------------------
# Subcommand: reproduce
# ---------------------------------------------------------------------------

@app.command()
def reproduce(
    table: str = typer.Argument(..., help="Table name: 'table3'..'table7', 'appendix_d', 'phase0'."),
    extra_args: Optional[list[str]] = typer.Argument(None, help="Extra args forwarded to the script."),
):
    """Run a paper reproduction script from ``scripts/reproduce_*.sh``.

    Stage 9 of the release migration creates these scripts. Before Stage 9
    lands, this command will error with a 'not found' message.
    """
    script_dir = registry.REPO_ROOT / "scripts"
    candidates = [
        script_dir / f"reproduce_{table}.sh",
        script_dir / f"reproduce_{table.replace('-', '_')}.sh",
    ]
    script_path = next((c for c in candidates if c.exists()), None)
    if script_path is None:
        typer.echo(f"ERROR: no reproduction script found for '{table}'.", err=True)
        typer.echo(f"Looked in: {[str(c) for c in candidates]}", err=True)
        typer.echo("Available scripts:", err=True)
        if script_dir.exists():
            for p in sorted(script_dir.glob("reproduce_*.sh")):
                typer.echo(f"  {p.name}", err=True)
        else:
            typer.echo("  (scripts/ directory does not exist yet -- migrate Stage 9)", err=True)
        raise typer.Exit(code=1)

    cmd = ["bash", str(script_path)] + list(extra_args or [])
    typer.echo(f"Running: {' '.join(cmd)}", err=True)
    result = subprocess.run(cmd, cwd=str(registry.REPO_ROOT))
    raise typer.Exit(code=result.returncode)


if __name__ == "__main__":
    app()
