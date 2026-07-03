# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Robust parsing of decoded MR predictions and aggregate SER/F1 metrics.

Parses strings like ``name: Foo, cuisine: English, price: 20-25`` or JSON
``{"name": "Foo", ...}`` into attribute dicts, then aggregates substitution /
deletion / insertion counts into SER plus per-attribute precision / recall / F1.
Entry point: ``compute_custom_metrics_stub`` -- used by extraction training,
ranker training, and the PBL extractor.
"""
import json
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Union

# ----------------------------
# Parsing & normalization
# ----------------------------

_BOOL_MAP = {
    "yes": "yes", "y": "yes", "true": "yes", "1": "yes",
    "no": "no", "n": "no", "false": "no", "0": "no"
}

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[-1]
    return s.strip()

def _maybe_json(s: str) -> Union[Dict[str, Any], None]:
    s = s.strip()
    # heuristic: looks like JSON if it starts with { and has a colon
    if s.startswith("{") and ":" in s:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None

def _split_top_level_commas(s: str) -> List[str]:
    """
    Split by commas not inside quotes/brackets/parens.
    """
    parts, buf = [], []
    depth_br, depth_par = 0, 0
    in_str, quote = False, None
    i = 0
    while i < len(s):
        ch = s[i]
        if in_str:
            buf.append(ch)
            if ch == quote:
                in_str, quote = False, None
        else:
            if ch in ("'", '"'):
                in_str, quote = True, ch
                buf.append(ch)
            elif ch in "[{(":
                depth_br += (ch in "[{")
                depth_par += (ch == "(")
                buf.append(ch)
            elif ch in "]})":
                depth_br -= (ch in "]}")
                depth_par -= (ch == ")")
                buf.append(ch)
            elif ch == "," and depth_br == 0 and depth_par == 0:
                parts.append("".join(buf).strip())
                buf = []
            else:
                buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf).strip())
    return parts

def _canon_val(x: Any) -> Any:
    """
    Canonicalize atomic values for comparison: lowercase, strip quotes/space.
    Map common boolean synonyms. Keep lists as lists of canonical atoms.
    """
    if isinstance(x, list):
        return sorted({_canon_val(v) for v in x if v is not None})
    if isinstance(x, (int, float)):
        return str(x)
    if x is None:
        return None
    s = str(x).strip().strip('"').strip("'")
    s_l = s.lower()
    if s_l in _BOOL_MAP:
        return _BOOL_MAP[s_l]
    # drop leading/trailing punctuation commonly produced by models
    s_l = s_l.strip(" ;,.")
    return s_l

def _maybe_listify(s: str) -> Union[str, List[str]]:
    """
    Turn simple value strings like 'a and b', 'a, b' into ['a','b'].
    If value looks like a JSON list, parse it; otherwise return original string.
    """
    s = s.strip()
    # JSON list?
    if s.startswith("[") and s.endswith("]"):
        try:
            vals = json.loads(s)
            if isinstance(vals, list):
                return [str(v).strip() for v in vals]
        except Exception:
            pass
    # split on ' and ' or commas when it looks like a multi-value
    if " and " in s or "," in s:
        parts = [p.strip() for p in re.split(r",|\band\b", s) if p.strip()]
        if len(parts) >= 2:
            return parts
    return s

def extract_attributes_dict(mr_str: str) -> Dict[str, Any]:
    """
    Robust parser for decoded predictions that look like:
      name: Foo, cuisine: English, price_range: 20-25
    or JSON:
      {"name":"Foo","cuisine":"English", "tags":["a","b"]}
    Returns a dict of {attr: value}, ignoring values equal to "None".
    """
    if not isinstance(mr_str, str):
        return {}
    s = _strip_code_fences(mr_str)

    # Some generations put references or extra lines; only take the first line
    first_line = s.splitlines()[0].strip()

    # Try JSON first
    obj = _maybe_json(first_line)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, str) and v.strip().lower() == "none":
                continue
            out[str(k).strip()] = v
        return out

    # Fallback: parse "a: b, c: d" with a robust comma splitter
    fields = _split_top_level_commas(first_line)
    out: Dict[str, Any] = {}
    for field in fields:
        if ":" not in field:
            continue
        a, v = field.split(":", 1)
        a, v = a.strip(), v.strip()
        if v.lower() == "none":
            continue
        v = _maybe_listify(v)
        out[a] = v
    return out

# ----------------------------
# Metrics
# ----------------------------

def compare_example(
    pred: Dict[str, Any],
    ref: Dict[str, Any]
) -> Tuple[int, int, int, int, Counter, Dict[str, Counter]]:
    """
    Returns (S, D, I, N_ref, correct_attrvalue_count, attr_error_counters)
    S: substitutions, D: deletions, I: insertions (hallucinations), N_ref: |ref|
    """

    print("pred:", pred)
    print("ref:", ref)

    # Canonicalize both sides
    pred_c = {k: _canon_val(v) for k, v in pred.items()}
    ref_c  = {k: _canon_val(v) for k, v in ref.items()}

    S = D = I = 0
    N_ref = len(ref_c)
    correct = 0
    correct_per_attr = Counter()
    attr_err = defaultdict(Counter)  # err_type -> attr -> count

    # Insertions (hallucinations)
    for a in pred_c:
        if a not in ref_c:
            I += 1
            attr_err["hallucination"][a] += 1

    # Deletions & Substitutions & Corrects
    for a, ref_v in ref_c.items():
        if a not in pred_c:
            D += 1
            attr_err["deletion"][a] += 1
            continue
        pv = pred_c[a]
        # value equality (string) or list set-equality
        if pv == ref_v:
            correct += 1
            correct_per_attr[a] += 1
        else:
            # If both lists, allow orderless compare; if one is str and the other list, try containment
            if isinstance(pv, list) and isinstance(ref_v, list):
                if sorted(pv) == sorted(ref_v):
                    correct += 1
                    correct_per_attr[a] += 1
                    continue
            elif isinstance(pv, list) and isinstance(ref_v, str):
                if ref_v in pv:
                    correct += 1
                    correct_per_attr[a] += 1
                    continue
            elif isinstance(pv, str) and isinstance(ref_v, list):
                if pv in ref_v:
                    correct += 1
                    correct_per_attr[a] += 1
                    continue
            # otherwise substitution
            S += 1
            attr_err["substitution"][a] += 1
            attr_err["substitution_value"][f"{a}::{pv}→{ref_v}"] += 1

    return S, D, I, N_ref, correct_per_attr, attr_err

def aggregate_metrics(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    examples: list of dicts with at least
      - 'pred': decoded string prediction
      - 'mr':   reference dict (attribute -> value or list)
      (optional) 'ref': reference surface form (for error dumps)
    """
    total_S = total_D = total_I = total_N = 0
    correct_per_attr = Counter()
    attr_err_counts = defaultdict(Counter)
    per_attr_tp = Counter()
    per_attr_fp = Counter()  # hallucinations + wrong values count as FP
    per_attr_fn = Counter()  # deletions + wrong values count as FN

    # Optional: collect a few illustrative failures
    samples_sub, samples_del, samples_hall = [], [], []

    for case in examples:
        pred_dict = extract_attributes_dict(case["pred"])
        ref_dict  = case["mr"]

        S, D, I, N, corr_attr, attr_err = compare_example(pred_dict, ref_dict)

        total_S += S; total_D += D; total_I += I; total_N += N
        correct_per_attr.update(corr_attr)
        for k, v in attr_err.items():
            attr_err_counts[k].update(v)

        # per-attribute PR accounting
        for a in ref_dict.keys():
            if a in pred_dict:
                if _canon_val(pred_dict[a]) == _canon_val(ref_dict[a]) \
                   or (isinstance(ref_dict[a], list) and _canon_val(pred_dict[a]) in _canon_val(ref_dict[a])) \
                   or (isinstance(pred_dict[a], list) and _canon_val(ref_dict[a]) in _canon_val(pred_dict[a])):
                    per_attr_tp[a] += 1
                else:
                    per_attr_fp[a] += 1  # wrong value predicted for this attr
                    per_attr_fn[a] += 1  # and also missed the correct value
            else:
                per_attr_fn[a] += 1
        for a in pred_dict.keys():
            if a not in ref_dict:
                per_attr_fp[a] += 1

        # collect a few examples
        if S:
            samples_sub.append(case)
        if D:
            samples_del.append(case)
        if I:
            samples_hall.append(case)

    ser = (total_S + total_D + total_I) / max(1, total_N)

    # per-attribute precision/recall/F1
    per_attr = {}
    for a in set(list(per_attr_tp) + list(per_attr_fp) + list(per_attr_fn)):
        tp, fp, fn = per_attr_tp[a], per_attr_fp[a], per_attr_fn[a]
        prec = tp / max(1, (tp + fp))
        rec  = tp / max(1, (tp + fn))
        f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        per_attr[a] = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    return {
        "SER": round(ser, 6),
        "slots_total": int(total_N),
        "errors": {"S": int(total_S), "D": int(total_D), "I": int(total_I)},
        "correct_per_attr": {k: int(v) for k, v in correct_per_attr.items()},
        "attr_error_counts": {k: dict(v) for k, v in attr_err_counts.items()},
        "per_attribute_scores": per_attr,
        # a few IDs/texts could be added here if your cases include them
        "samples": {
            "substitution": [ {"pred": c["pred"], "mr": c["mr"], "ref": c.get("ref","")} for c in samples_sub[:5] ],
            "deletion":     [ {"pred": c["pred"], "mr": c["mr"], "ref": c.get("ref","")} for c in samples_del[:5] ],
            "hallucination":[ {"pred": c["pred"], "mr": c["mr"], "ref": c.get("ref","")} for c in samples_hall[:5] ],
        }
    }

# ----------------------------
# Convenience wrapper
# ----------------------------

def compute_custom_metrics_stub(pred_texts: List[str],
                                raw_examples: Union[List[Dict], None]) -> Dict[str, Any]:
    """
    pred_texts: decoded generations (list[str]) in the same order as raw_examples
    raw_examples: each item should provide 'mr' (dict) and optional 'ref'
                  If None, returns length stats only.
    """
    if not raw_examples or len(pred_texts) != len(raw_examples):
        avg_len = sum(len(t.split()) for t in pred_texts) / max(1, len(pred_texts))
        return {"avg_generated_words": round(avg_len, 2), "num_predictions": len(pred_texts)}

    examples = []
    for t, raw in zip(pred_texts, raw_examples):
        examples.append({
            "pred": t,
            "mr": raw["mr"],
            "ref": raw.get("surface_form") or raw.get("ref") or ""
        })
    return aggregate_metrics(examples)
