# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Core Slot Error Rate (SER) computation.

Two parallel APIs:

* ``SlotErrorRate`` -- stateful class with per-error-type counts (hallucinations,
  deletions, substitutions, repetitions). Used by older evaluation paths and
  the rule-based baselines.
* ``compute_ser`` / ``compute_slot_f1`` -- stateless functions over predicted /
  reference attribute dicts with value normalisation and list set-equality.
  Used by the LoRA extraction + ranking pipeline.

``extract_attributes_dict`` parses ``<LIST>(slot: value); ...</LIST>`` strings
into the dict form both APIs consume.
"""
from typing import Any, Dict, Iterable, List, Optional, Tuple
import re
from collections import defaultdict, Counter
from string import punctuation

from .data_helper import LIST_START


def strip_punctuation(text: str) -> str:
    """Strip leading and trailing punctuation characters."""
    return text.strip(punctuation)

def mr_list_to_dict(mr: Iterable[List[str]]) -> Dict[str, List[str]]:
    """Convert ``[slot, value, ...]`` entries into ``{slot: [values]}``.

    Values are de-duplicated globally: a value string already seen under
    any slot is not added again under another.
    """
    dict_mr = defaultdict(list)
    # remove duplicate slot/values in the ref MR
    seen_vals = set()
    for entry in mr:
        vals = entry[1:]
        for v in vals:
            #v = strip_punctuation(v)
            if v not in seen_vals:
                dict_mr[entry[0]].append(v)
                seen_vals.add(v)
    return dict_mr

def extract_attributes_dict(
    mr_str: str, wanted_attributes: Optional[Iterable[str]] = None,
) -> Dict[str, List[str]]:
    """Parse a ``(slot: value); ...`` MR string into ``{slot: [values]}``.

    Handles decoded-output framing: anything up to ``>(`` is dropped and a
    legacy ``</List>`` or a following ``<LIST>`` bounds the right side
    (callers strip the all-caps ``</LIST>`` first -- see test_ser.py).
    Multi-values split on ``;``. With ``wanted_attributes``, other slots
    are dropped from the result.
    """
    #mr_str = mr_str.replace(LIST_START, "").replace(LIST_END, "")
    #print("\nmr_str", mr_str)
    if ">(" in mr_str:
        j = mr_str.index(">(")
        mr_str = mr_str[j+len(">("):]

    if "</List>" in mr_str:
        j = mr_str.index("</List>")
        mr_str = mr_str[:j]
    if LIST_START in mr_str:
        j = mr_str.index(LIST_START)
        mr_str = mr_str[:j]

    slot_vals = defaultdict(list)
    for sv in mr_str.split("); ("):
        if ":" not in sv: continue
        slot = sv[:sv.index(":")].strip()
        if slot.startswith("("):
            slot = slot[1:].strip()
        vals = sv[sv.index(":")+1:].strip()
        if ";" in vals:
            vals = vals.split(";")
        else:
            vals = [vals]
        for val in vals:
            val = val.strip()
            if val.endswith(")"):
                val = val[:-1]
            if " ," in val:
                val = val.replace(" ,", ",")
            slot_vals[slot].append(val)

    #print("extracted slot/vals:", slot_vals)
    if wanted_attributes:
        #print("wanted attributes:", wanted_attributes)
        to_delete = {}
        for attr in slot_vals:
            if attr not in wanted_attributes:
                to_delete[attr] = slot_vals[attr]

        for attr in to_delete:
            del slot_vals[attr]
        #print("deleted attributes:", to_delete)
    return slot_vals


class SlotErrorRate(object):
    """Stateful S/D/I/repetition error tallies over a stream of examples.

    ``normalize_mr`` converts pair-list MRs to the dict form the ``calc_*``
    methods take; each ``calc_*`` call updates the running ``err_counts`` /
    ``attr_err_counts`` tallies and prints the errors it finds. Used by the
    rule-based baselines and older evaluation paths.
    """

    def __init__(self, target_slot_names=None, topic_name=None):
        self.target_slot_names = [] if target_slot_names is None else target_slot_names
        self.err_counts = Counter()
        self.attr_err_counts = defaultdict(Counter)
        self.topic_name = topic_name


    @classmethod
    def normalize_mr(cls, mr):

        ref_mr = defaultdict(list)
        # remove duplicate slot/values in the ref MR
        seen_vals = set()
        for entry in mr:
            vals = entry[1:]
            for v in vals:
                #v = strip_punctuation(v)
                if v not in seen_vals:
                    ref_mr[entry[0]].append(v)
                    seen_vals.add(v)
        return ref_mr

    def calc_hallucinations(self, ref_mr, wanted_slots, prediction):
        etype = None
        err_counts = Counter()
        for attr, vals in prediction.items():
            if attr in wanted_slots and attr not in ref_mr:
                # pred attr not in ref MR
                etype = 'hallucination'
                self.err_counts[etype] += 1
                self.attr_err_counts[etype][attr] += 1
                err_counts[etype] += 1
                print(f"Hallucination: {attr}")
        return err_counts

    def calc_deletions(self, ref_mr, prediction):
        res = {}
        etype = None
        err_counts = Counter()
        for ref_attr, ref_vals in ref_mr.items():
            # print(f"ref_attr: {ref_attr}\tref_vals: {ref_vals}")
            if ref_attr in prediction:
                processed_vals = set()
                for val in ref_vals:
                    val = strip_punctuation(val)
                    # print(f"if {val} not in {prediction[ref_attr]}:", val not in prediction[ref_attr])
                    val_in_pred = False
                    for pval in prediction[ref_attr]:
                        val_in_pred = (pval.startswith(val) or val.startswith(pval) or
                                       pval.endswith(val) or val.endswith(pval))
                        if val_in_pred:
                            processed_vals.add(val)
                            break

                    if not val_in_pred and val not in processed_vals:
                        etype = "deletion"
                        self.err_counts[etype] += 1
                        err_counts[etype] += 1
                        self.attr_err_counts[etype][ref_attr] += 1
                        #has_etype[etype] = True
                        print(f"Deletion: {ref_attr}: {val}")
                        res["deletion_attr"] = ref_attr
                        res["deletion_val"] = val
            else:
                # attr not in pred MR
                etype = 'deletion'
                self.err_counts[etype] += 1
                err_counts[etype] += 1
                self.attr_err_counts[etype][ref_attr] += 1
                #has_etype[etype] = True
                print(f"Deletion: {ref_attr}")
                res["deletion_attr"] = ref_attr
        return res, err_counts

    def calc_substitutions(self, ref_mr, prediction):
        etype = None
        err_counts = Counter()
        for ref_attr, ref_vals in ref_mr.items():
            if ref_attr in prediction:
                processed_vals = set()
                for val in ref_vals:
                    val = strip_punctuation(val)
                    val_in_pred = False
                    for pval in prediction[ref_attr]:
                        val_in_pred = (pval.startswith(val) or val.startswith(pval) or
                                       pval.endswith(val) or val.endswith(pval))
                        if val_in_pred:
                            processed_vals.add(val)
                            break

                    if not val_in_pred:
                        etype = 'substitution'
                        self.err_counts[etype] += 1
                        err_counts[etype] += 1
                        self.attr_err_counts[etype][ref_attr] += 1
                        if isinstance(prediction[ref_attr], list):
                            for x in prediction[ref_attr]:
                                self.attr_err_counts["substitution_value"][x] += 1
                        #has_etype[etype] = True
                        print(f"Substitution: {ref_attr}\t{val}\t({prediction[ref_attr]})")
        return err_counts

    def calc_repetitions(self, ref_mr, prediction):
        err_counts = Counter()
        etype = None
        for ref_attr, ref_vals in ref_mr.items():
            #print(f"ref_attr: {ref_attr}\tref_vals: {ref_vals}")
            if ref_attr in prediction:
                if len(ref_vals) < len(prediction[ref_attr]):
                    etype = 'repetition'
                    self.err_counts[etype] += 1
                    err_counts[etype] += 1
                    self.attr_err_counts[etype][ref_attr] += 1
                    #has_etype[etype] = True
                    print(f"Repetition: {ref_attr}\t{prediction[ref_attr]}")
        return err_counts




# --- tiny normalizers ---------------------------------------------------------

_BOOL = {
    "yes":"yes","y":"yes","true":"yes","1":"yes","on":"yes","present":"yes","available":"yes",
    "no":"no","n":"no","false":"no","0":"no","off":"no","absent":"no","unavailable":"no",
    "True": "yes", "False": "no"
}

def _canon_scalar(x: Any) -> str:
    """Normalize a single value to a lowercase string with boolean synonyms mapped."""
    s = str(x).strip().strip('"').strip("'")
    s = s.strip(" ;,.")  # trim common trailing punctuation
    s_low = s.lower()
    if s_low in _BOOL:
        return _BOOL[s_low]
    return s_low

def _maybe_listify(s: str) -> List[str] :
    """Turn 'a, b' or 'a and b' into ['a','b']; keep JSON-like lists if already parsed elsewhere."""
    s = s.strip()
    # crude list detection from plain text
    if "," in s or " and " in s:
        parts = [p.strip() for p in re.split(r",|\band\b", s) if p.strip()]
        if len(parts) >= 2:
            return parts
    return s

def _canon_value(v: Any):
    """Canonicalize lists vs scalars; lists are de-duplicated & sorted (as sets)."""
    if isinstance(v, list):
        items = {_canon_scalar(x) for x in v if x is not None}
        return sorted(items)
    if isinstance(v, str):
        maybe = _maybe_listify(v)
        if isinstance(maybe, list):
            return sorted({_canon_scalar(x) for x in maybe})
        return _canon_scalar(maybe)
    if v is None:
        return None
    return _canon_scalar(v)

def _values_equal(a: Any, b: Any, allow_containment: bool = True) -> bool:
    """
    Equality for slot values:
      - lists compare as set-equality;
      - if one side is a list and the other is a scalar, containment may count as equal
        (set by allow_containment).
    """
    ca, cb = _canon_value(a), _canon_value(b)
    if isinstance(ca, list) and isinstance(cb, list):
        return ca == cb
    if isinstance(ca, list) and isinstance(cb, str):
        return (cb in ca) if allow_containment else False
    if isinstance(ca, str) and isinstance(cb, list):
        return (ca in cb) if allow_containment else False
    return ca == cb

# --- Slot F1 ------------------------------------------------------------------

def compute_slot_f1(
    pred: Dict[str, Any],
    ref: Dict[str, Any],
    allow_containment: bool = True,
) -> Dict[str, float ]:
    """
    Micro-averaged slot precision/recall/F1 for a single example.
    A 'correct' hit requires the slot to exist in both and values to match
    (with list set-equality or optional containment).
    """
    tp = fp = fn = 0

    pred_keys = set(pred.keys())
    ref_keys  = set(ref.keys())

    # true positives
    for k in pred_keys & ref_keys:
        if _values_equal(pred[k], ref[k], allow_containment=allow_containment):
            tp += 1

    # false positives: wrong value or hallucinated slot
    for k in pred_keys:
        if k not in ref_keys:
            fp += 1
        elif not _values_equal(pred[k], ref[k], allow_containment=allow_containment):
            fp += 1

    # false negatives: missing slot or wrong value (missed the correct value)
    for k in ref_keys:
        if k not in pred_keys:
            fn += 1
        elif not _values_equal(pred[k], ref[k], allow_containment=allow_containment):
            fn += 1

    f1, precision, recall = calc_p_r_f(fn, fp, tp)

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def calc_p_r_f(fn: int, fp: int, tp: int) -> Tuple[float, float, float]:
    """Precision/recall/F1 from error counts, returned as ``(f1, precision, recall)``.

    Note the argument order (fn, fp, tp) differs from the return order.
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return f1, precision, recall


# --- SER (Slot Error Rate) ----------------------------------------------------

def compute_ser(
    pred: Dict[str, Any],
    ref: Dict[str, Any],
    allow_containment: bool = True,
) -> Dict[str, float ]:
    """
    SER = (S + D + I) / |ref|
      S: substitutions (slot present in both but wrong value)
      D: deletions (slot missing in pred)
      I: insertions (slot hallucinated in pred)
    """
    S = D = I = 0
    pred_keys = set(pred.keys())
    ref_keys  = set(ref.keys())
    slot_errors = {"S": [] , "D": [] , "I": [] }

    # Insertions
    for k in pred_keys - ref_keys:
        I += 1
        slot_errors["I"].append(f"{k}:{pred[k]}")

    # Deletions & Substitutions
    for k in ref_keys:
        if k not in pred_keys:
            D += 1
            slot_errors["D"].append(k)
        elif not _values_equal(pred[k], ref[k], allow_containment=allow_containment):
            S += 1
            slot_errors["S"].append(f"{k}({ref[k]}!={pred[k]})")

    N = len(ref_keys) or 1  # avoid div by zero
    ser = _compute_ser_val(D, I, S, N)

    return {
        "SER": round(ser, 6),
        "S": S, "D": D, "I": I, "N_ref": len(ref_keys),
        "errors": slot_errors,
    }


def _compute_ser_val(D, I, S, N):
    ser = (S + D + I) / N
    return ser
