# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Rule-based slot aligner / SER evaluator for the E2E NLG Challenge dataset.

Re-implementation of the slot-realisation alignment logic introduced by
Dušek and collaborators for the E2E NLG Challenge -- see Dušek & Kasner
(2020), "Evaluating Semantic Accuracy of Data-to-Text Generation with
Natural Language Inference", and the original E2E NLG Challenge
alignment scripts at https://github.com/tuetschek/e2e-metrics (upstream
license: BSD-2-Clause). The algorithm (keyword matching + negation +
boolean/scalar/categorical slot handlers) is theirs; this module
re-implements it, structured after the ViGGO aligner port in this
package so the three aligners share a common interface
(``pack_*_nlg_mr`` + ``extract_mr`` + ``eval_compute_ser``).

Used as the rule-based baseline in the GEM 2026 SER agreement comparisons
and in the Eval-2 personality SER pipeline (see
``xdomain_ser.routing.personality``).
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from xdomain_ser.ranking.make_eval_data import tally_ser



def pack_e2e_nlg_mr(d):
    """
    Converts unpacked e2e NLG MR dict back to the original e2e MR string format.

    e.g. name[The Vaults], eatType[pub], food[Japanese], priceRange[cheap], customer rating[5 out of 5], area[riverside], familyFriendly[yes], near[Café Adriatic]

    :param d: dict with "slots" key (as produced by unpack_e2e_nlg_mr)
    :return: e2e MR string
    """
    REVERSE_KEY_MAP = {
        "customerRating": "customer rating",
        "family_suitability": "familyFriendly",
        "venue_type": "eatType",
        "cuisine_type": "food",
        "nearby_landmark": "near",
        "area_zone": "area",
    }
    #print("d:", d)
    slot_value_pairs = []

    for slot_val in d:
        name = slot_val[0]
        values = slot_val[1:]
        if not isinstance(values, list):
            values = [values]

        orig_name = REVERSE_KEY_MAP.get(name, name)

        if name == "family_suitability":
            for v in values:
                orig_value = "yes" if v == "family-friendly" else "no"
                slot_value_pairs.append((orig_name, orig_value))
        else:
            slot_value_pairs.append((orig_name, values))

    #print("slot_value_pairs:", slot_value_pairs)
    #input(">>>")

    return slot_value_pairs

# ---------------------------------------------------------------------------
# Simple word tokenizer (avoids NLTK dependency)
# ---------------------------------------------------------------------------
def word_tokenize(text: str) -> List[str]:
    text = re.sub(r"(\w)(n't)\b", r"\1 \2", text)
    text = re.sub(r"(\w)('s)\b", r"\1 \2", text)
    text = re.sub(r"(\w)('re)\b", r"\1 \2", text)
    text = re.sub(r"(\w)('ve)\b", r"\1 \2", text)
    text = re.sub(r"(\w)('ll)\b", r"\1 \2", text)
    text = re.sub(r"(\w)('d)\b", r"\1 \2", text)
    tokens = re.findall(r"\w+(?:[-/]\w+)*|[^\w\s]", text)
    return tokens


# ---------------------------------------------------------------------------
# Utility helpers (from slot_aligner/alignment/utils.py)
# ---------------------------------------------------------------------------
def find_first_in_list(val: str, lst: List[str]) -> Tuple[int, int]:
    idx = -1
    pos = -1
    for i, elem in enumerate(lst):
        if val == elem:
            idx = i
    if idx >= 0:
        punct_cnt = lst[:idx].count('.') + lst[:idx].count(',')
        pos = len(' '.join(lst[:idx])) + 1 - punct_cnt
    return idx, pos


def find_all_in_list(val: str, lst: List[str]) -> Tuple[List[int], List[int]]:
    indexes = []
    positions = []
    for i, elem in enumerate(lst):
        if val == elem:
            indexes.append(i)
            punct_cnt = lst[:i].count('.') + lst[:i].count(',')
            positions.append(len(' '.join(lst[:i])) + 1 - punct_cnt)
    return indexes, positions


# ---------------------------------------------------------------------------
# Alternatives for E2E slot-value matching (from alternatives.json)
# ---------------------------------------------------------------------------
E2E_ALTERNATIVES: Dict[str, Dict[str, list]] = {
    "rating": {
        "excellent": [
            "5 out of", "5 star", "adore", "amazing", "attract", "awesome",
            "best", "fantastic", "favorite", "five", "great", "high", "highly",
            "love", "loved", "loving", "quality", "special", "superb", "top", "unique",
        ],
        "good": [
            "acclaim", "cool", "enjoy", "fun", "like", "liked", "positive", "solid", "well",
        ],
        "average": [
            "3 out of", "3 star", "all right", "alright", "decent", "kinda",
            "kind of", "lukewarm", "mediocre", "meh", "middle", "middling",
            "mixed", "moderate", "ok", "okay", "ordinary", "so so", "three", "unimpress",
        ],
        "poor": [
            "1 out of", "1 star", "avoid", "bad", "badly", "boring", "detest",
            "disappoint", "dislike", "dull", "hate", "hated", "hating",
            "lackluster", "lacking", "loathe", "low", "lowly", "negative",
            "one", "poorly", "underwhelm", "wrong",
        ],
    },
    "pricerange": {
        "cheap": [
            "less than \u00a320", "less than 20", "under \u00a320", "under 20",
            "inexpensive", "affordable", "low", "lower", "budget", "bargain",
        ],
        "moderate": [
            "20 25", "20 to 25", "between 20 and 25", "average", "reasonable",
        ],
        "high": [
            "more than \u00a330", "more than 30", "over \u00a330", "over 30",
            "expensive", "costly", "pricey", "high", "higher",
        ],
        "less than \u00a320": [
            "less than 20", "under \u00a320", "under 20",
            "cheap", "inexpensive", "affordable", "low", "budget",
        ],
        "\u00a320 25": [
            "20 25", "20 to 25", "between 20 and 25", "moderate", "average", "reasonable",
        ],
        "more than \u00a330": [
            "more than 30", "over \u00a330", "over 30",
            "high", "expensive", "costly", "pricey", "high",
        ],
    },
    "area": {
        "city centre": [
            "center", "centre", "downtown",
            ["middle", "city"], ["middle", "town"],
        ],
        "riverside": ["river"],
    },
    "type": {
        "television": ["tv"],
    },
}

CUSTOMERRATING_MAPPING = {
    'slot': 'rating',
    'values': {
        'low': 'poor',
        'average': 'average',
        'high': 'excellent',
        '1 out of 5': 'poor',
        '3 out of 5': 'average',
        '5 out of 5': 'excellent',
    },
}


def get_e2e_slot_value_alternatives(slot: str) -> dict:
    return E2E_ALTERNATIVES.get(slot, {})


# ---------------------------------------------------------------------------
# Keyword matching (from slot_alignment.py _match_keywords_in_text)
# ---------------------------------------------------------------------------
def match_keywords_in_text(keywords, text, ignore_dupes=False):
    pos = -1
    end_pos = 0
    is_duplicated = False

    if isinstance(keywords, str):
        keywords = [keywords]
        fixed_word_order = True
    elif isinstance(keywords, list):
        fixed_word_order = True
    elif isinstance(keywords, tuple):
        fixed_word_order = False
    else:
        raise TypeError('keywords must be str, list, or tuple')

    for word in keywords:
        if re.match(r'\w', word[0]) and re.match(r'\w', word[-1]):
            pattern = re.compile(fr'\b{re.escape(word)}\b')
        else:
            pattern = re.compile(f'{re.escape(word)}')

        start_pos = end_pos if fixed_word_order else 0
        match = pattern.search(text, start_pos)
        if match:
            pos, end_pos = match.span()
            if not ignore_dupes and len(pattern.findall(text)) > 1:
                is_duplicated = True
        else:
            return -1, False

    return pos, is_duplicated


# ---------------------------------------------------------------------------
# Boolean slot alignment (from alignment/boolean_slot.py)
# ---------------------------------------------------------------------------
NEGATION_CUES_PRE = [
    'no', 'not', 'non', 'none', 'neither', 'nor', 'never', "n't", 'cannot',
    'excluded', 'lack', 'lacks', 'lacking', 'unavailable', 'without', 'zero',
    'everything but',
]
NEGATION_CUES_POST = [
    'not', 'nor', 'never', "n't", 'cannot', 'excluded', 'unavailable',
]
CONTRAST_CUES = ['but', 'however', 'although', 'though', 'nevertheless']

BOOLEAN_SLOT_STEMS = {
    'familyfriendly': ['family', 'families', 'kid', 'kids', 'child', 'children'],
}
BOOLEAN_SLOT_ANTONYMS = {
    'familyfriendly': ['adult', 'adults'],
}

NEG_IDX_FALSE_PRE_THRESH = 10
NEG_POS_FALSE_PRE_THRESH = 30
NEG_IDX_TRUE_PRE_THRESH = 5
NEG_POS_TRUE_PRE_THRESH = 15
NEG_IDX_POST_THRESH = 10
NEG_POS_POST_THRESH = 30


def _has_contrast_after_negation(text_segment: str) -> bool:
    for contr in CONTRAST_CUES:
        if contr in text_segment:
            return True
    return False


def _has_contrast_after_negation_tok(text_tok: List[str]) -> bool:
    for contr in CONTRAST_CUES:
        if contr in text_tok:
            return True
    return False


def _find_negation(text, text_tok, idx, pos, expected_true=False, after=False):
    idx_pre_thresh = NEG_IDX_TRUE_PRE_THRESH if expected_true else NEG_IDX_FALSE_PRE_THRESH
    pos_pre_thresh = NEG_POS_TRUE_PRE_THRESH if expected_true else NEG_POS_FALSE_PRE_THRESH

    for negation in NEGATION_CUES_PRE:
        if ' ' in negation:
            neg_pos = text.find(negation)
            if neg_pos >= 0:
                if 0 < (pos - neg_pos - text[neg_pos:pos].count(',')) <= pos_pre_thresh:
                    seg = text[neg_pos + len(negation):pos]
                    return not _has_contrast_after_negation(seg)
        else:
            neg_idxs, _ = find_all_in_list(negation, text_tok)
            for neg_idx in neg_idxs:
                if 0 < (idx - neg_idx - text_tok[neg_idx + 1:idx].count(',')) <= idx_pre_thresh:
                    seg = text_tok[neg_idx + 1:idx]
                    return not _has_contrast_after_negation_tok(seg)

    if after:
        for negation in NEGATION_CUES_POST:
            if ' ' in negation:
                neg_pos = text.find(negation)
                if neg_pos >= 0 and 0 < (neg_pos - pos) < NEG_POS_POST_THRESH:
                    return True
            else:
                neg_idxs, _ = find_all_in_list(negation, text_tok)
                for neg_idx in neg_idxs:
                    if 0 < (neg_idx - idx) < NEG_IDX_POST_THRESH:
                        return True
    return False


def align_boolean_slot(text, text_tok, slot, value, true_val='yes', false_val='no'):
    text = re.sub(r"'", '', text)
    slot_stems = BOOLEAN_SLOT_STEMS.get(slot, [])

    for slot_stem in slot_stems:
        idx, pos = find_first_in_list(slot_stem, text_tok)
        if pos >= 0:
            if value == true_val:
                if not _find_negation(text, text_tok, idx, pos, expected_true=True, after=False):
                    return pos
            else:
                if _find_negation(text, text_tok, idx, pos, expected_true=False, after=True):
                    return pos

    if value == false_val:
        antonyms = BOOLEAN_SLOT_ANTONYMS.get(slot, [])
        for antonym in antonyms:
            if ' ' in antonym:
                pos = text.find(antonym)
            else:
                _, pos = find_first_in_list(antonym, text_tok)
            if pos >= 0:
                return pos

    return -1


# ---------------------------------------------------------------------------
# Scalar slot alignment (from alignment/scalar_slot.py)
# ---------------------------------------------------------------------------
DIST_IDX_THRESH = 10
DIST_POS_THRESH = 30

SCALAR_SLOT_STEMS = {
    'customerrating': ['customer', 'rating', 'ratings', 'rated', 'rate', 'review', 'reviews', 'star', 'stars'],
    'pricerange': ['price', 'pricing', 'cost', 'costs', 'dollars', 'pounds', 'euros', r'\$', '£', '€'],
}


def align_scalar_slot(text, text_tok, slot, value, slot_mapping=None, value_mapping=None, slot_stem_only=False):
    slot_stem_indexes = []
    slot_stem_positions = []
    leftmost_pos = -1

    text_clean = re.sub(r"'", '', text)
    slot_stems = SCALAR_SLOT_STEMS.get(slot, [])

    lookup_slot = slot_mapping if slot_mapping is not None else slot
    alternatives = get_e2e_slot_value_alternatives(lookup_slot)

    for slot_stem in slot_stems:
        if len(slot_stem) == 1 and not slot_stem.isalnum():
            slot_stem_pos = [m.start() for m in re.finditer(slot_stem, text_clean)]
        elif len(slot_stem) > 4 or ' ' in slot_stem:
            slot_stem_pos = [m.start() for m in re.finditer(slot_stem, text_clean)]
        else:
            s_idx, s_pos = find_all_in_list(slot_stem, text_tok)
            if len(s_idx) > 0:
                slot_stem_indexes.extend(s_idx)
            slot_stem_pos = s_pos if s_pos else []

        if len(slot_stem_pos) > 0:
            slot_stem_positions.extend(slot_stem_pos)

    slot_stem_positions.sort()
    slot_stem_indexes.sort()

    if slot_stem_only and len(slot_stem_positions) > 0:
        return slot_stem_positions[0]

    value_alternatives = [value]
    if value_mapping is not None and value in value_mapping:
        value = value_mapping[value]
        value_alternatives.append(value)
    if value in alternatives:
        value_alternatives += alternatives[value]

    for val in value_alternatives:
        if len(val) > 4 or ' ' in val:
            val_positions = [m.start() for m in re.finditer(re.escape(val), text_clean)]
            for pos in val_positions:
                if pos < leftmost_pos or leftmost_pos == -1:
                    leftmost_pos = pos
                if len(slot_stem_positions) > 0:
                    for ssp in slot_stem_positions:
                        if abs(pos - ssp) < DIST_POS_THRESH:
                            return pos
        else:
            val_indexes, val_positions = find_all_in_list(val, text_tok)
            for i, idx in enumerate(val_indexes):
                if val_positions[i] < leftmost_pos or leftmost_pos == -1:
                    leftmost_pos = val_positions[i]
                if len(slot_stem_indexes) > 0:
                    for si in slot_stem_indexes:
                        if abs(idx - si) < DIST_IDX_THRESH:
                            return val_positions[i]

    return leftmost_pos


# ---------------------------------------------------------------------------
# Categorical slot alignment (from alignment/categorical_slots.py)
# ---------------------------------------------------------------------------
def find_value_alternative(text, text_tok, value, alternatives, mode='exact_match', allow_plural=False):
    leftmost_pos = -1

    if mode == 'first_word':
        value_alternatives = [value.split(' ')[0]]
    elif mode == 'any_word':
        value_alternatives = value.split(' ')
    elif mode == 'all_words':
        value_alternatives = [value.split(' ')]
    else:
        value_alternatives = [value]

    if value in alternatives:
        value_alternatives += alternatives[value]

    if allow_plural:
        extras = []
        for va in value_alternatives:
            if isinstance(va, str):
                extras.append(_plural(va))
        value_alternatives += extras

    for value_alt in value_alternatives:
        if not isinstance(value_alt, list):
            value_alt = [value_alt]

        positions = []
        for tok in value_alt:
            if len(tok) > 4 or ' ' in tok:
                pos = text.find(tok)
            else:
                _, pos = find_first_in_list(tok, text_tok)
            positions.append(pos)

        if all(p >= 0 for p in positions):
            leftmost_pos = min(positions)
            break

    return leftmost_pos


def _plural(word):
    if word.endswith('fe'):
        return word[:-2] + 'ves'
    elif word.endswith('f'):
        return word[:-1] + 'ves'
    elif word.endswith('o'):
        return word + 'es'
    elif word.endswith('us'):
        return word[:-2] + 'i'
    elif word.endswith('on'):
        return word[:-2] + 'a'
    elif word.endswith('y'):
        return word[:-1] + 'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh', 'ch']:
        return word + 'es'
    elif word.endswith('an'):
        return word[:-2] + 'en'
    else:
        return word + 's'


def align_categorical_slot(text, text_tok, slot, value, mode='exact_match', allow_plural=False):
    alternatives = get_e2e_slot_value_alternatives(slot)
    pos = find_value_alternative(text, text_tok, value, alternatives, mode=mode, allow_plural=allow_plural)
    return pos


# ---------------------------------------------------------------------------
# Food slot alignment (from alignment/categorical_slots.py foodSlot)
# Uses a static lookup of known cuisine-related words instead of WordNet.
# ---------------------------------------------------------------------------
FOOD_RELATED_WORDS = {
    'chinese', 'english', 'british', 'french', 'italian', 'indian', 'japanese',
    'thai', 'mexican', 'korean', 'vietnamese', 'greek', 'spanish', 'turkish',
    'american', 'mediterranean', 'asian', 'european', 'african',
    'pizza', 'pasta', 'sushi', 'curry', 'noodle', 'noodles', 'rice', 'bread',
    'burger', 'burgers', 'steak', 'seafood', 'fish', 'chicken', 'vegetarian',
    'vegan', 'organic', 'halal', 'kosher',
}


def food_slot(text, text_tok, value):
    """Identify a food/cuisine mention in text. Simplified version of foodSlot that
    avoids WordNet by using a static food-word lookup as fallback."""
    value = value.lower()

    pos = text.find(value)
    if pos >= 0:
        return pos
    elif value == 'english':
        return text.find('british')
    elif value == 'fast food':
        pos = text.find('american style')
        if pos >= 0:
            return pos
        # Also check for "fast food" as two words
        pos = text.find('fast food')
        return pos
    else:
        # Fallback: check if any known food-related word appears
        for token in text_tok:
            if token.lower() in FOOD_RELATED_WORDS:
                return text.find(token)

    return -1


# ---------------------------------------------------------------------------
# Preprocessing (from slot_alignment.py)
# ---------------------------------------------------------------------------
def preprocess_utterance(utt):
    utt = re.sub(r'[-/]', ' ', utt.lower())
    utt = re.sub(r'\s+', ' ', utt)
    utt_tok = [w.strip('.,!?') if len(w) > 1 else w for w in word_tokenize(utt)]
    return utt, utt_tok


def mask_named_entities(mr_as_list, utt):
    """Mask verbatim mentions of name-based slots (name, near) in the utterance."""
    name_slots = {'name', 'near'}

    mr_with_pos = [(s, v, idx) for idx, (s, v) in enumerate(mr_as_list)]
    mr_with_pos.sort(key=lambda x: len(x[1]) if isinstance(x[1], str) else 0, reverse=True)

    value_counts = Counter([val for _, val in mr_as_list if isinstance(val, str)])

    for i, (slot, value, orig_pos) in enumerate(mr_with_pos):
        if slot in name_slots and value and isinstance(value, str):
            pattern = re.compile(fr'\b{re.escape(value)}\b')
            match = pattern.search(utt)
            if match:
                value_counts.subtract([value])
                if value_counts[value] < 1:
                    utt, num_mentions = pattern.subn('_' * len(value), utt)
                else:
                    num_mentions = len(pattern.findall(utt))

                is_dupe = num_mentions > 1 and slot != 'name'
                value_dict = {'text': value, 'pos': match.start(), 'is_dupe': is_dupe}
                mr_with_pos[i] = (slot, value_dict, orig_pos)

    mr_as_list = [(s, v) for s, v, _ in sorted(mr_with_pos, key=lambda x: x[2])]
    return mr_as_list, utt


# ---------------------------------------------------------------------------
# E2E slot name mapping: original E2E names → internal aligner names
# ---------------------------------------------------------------------------
E2E_SLOT_TO_INTERNAL = {
    'familyFriendly': 'familyfriendly',
    'eatType': 'eattype',
    'customer rating': 'customerrating',
    'priceRange': 'pricerange',
    # These stay the same:
    'name': 'name',
    'food': 'food',
    'area': 'area',
    'near': 'near',
}

E2E_INTERNAL_TO_ORIG = {v: k for k, v in E2E_SLOT_TO_INTERNAL.items()}


# ---------------------------------------------------------------------------
# E2E domain find_slot_realization (from the rest_e2e branch)
# ---------------------------------------------------------------------------
def find_e2e_slot_realization(text, text_tok, slot, value, mr, ignore_dupes=False):
    """Find slot-value realization in text for the E2E restaurant domain.

    Args:
        slot: internal aligner slot name (lowercase, no spaces)
        value: preprocessed slot value (lowercase)
    Returns:
        (position, is_duplicate)
    """
    pos = -1
    is_dupe = False

    all_slots = {s for s, _ in mr}

    # Handle special universal values
    if value in ('dontcare', "don't care", ''):
        # For dontcare/empty, just check if slot stem is mentioned
        slot_mentions = {
            'area': ['area', 'location', 'neighborhood', 'place', 'where'],
            'food': ['food', 'cuisine'],
            'eattype': ['eat'],
            'familyfriendly': ['family', 'kid', 'kids', 'child', 'children'],
            'pricerange': ['price'],
            'customerrating': ['customer', 'rating'],
            'name': ['name'],
            'near': ['near'],
        }
        stems = slot_mentions.get(slot, [slot])
        for stem in stems:
            p, d = match_keywords_in_text(stem, text, ignore_dupes=True)
            if p >= 0:
                return p, d
        return -1, False
    elif value == 'none' or value == '?':
        return -1, False

    # E2E restaurant domain-specific slot alignment
    if slot == 'familyfriendly':
        pos = align_boolean_slot(text, text_tok, slot, value)
    elif slot == 'food':
        pos = food_slot(text, text_tok, value)
    elif slot in ('area', 'eattype'):
        pos = align_categorical_slot(text, text_tok, slot, value, mode='exact_match')
    elif slot == 'pricerange':
        pos = align_scalar_slot(text, text_tok, slot, value, slot_stem_only=False)
    elif slot == 'customerrating':
        pos = align_scalar_slot(
            text, text_tok, slot, value,
            slot_mapping=CUSTOMERRATING_MAPPING['slot'],
            value_mapping=CUSTOMERRATING_MAPPING['values'],
            slot_stem_only=False,
        )

    # Fallback: verbatim match
    if pos < 0:
        pos, is_dupe = match_keywords_in_text(value, text, ignore_dupes=ignore_dupes)

    return pos, is_dupe


# ---------------------------------------------------------------------------
# extract_mr: extract slot values from utterance for E2E examples
# ---------------------------------------------------------------------------
def extract_mr(utt: str, mr):
    """Extract slot-value realizations from an utterance given the reference MR.

    Args:
        utt: The surface form / generated utterance.
        mr: List of (orig_slot_name, value) tuples from pack_e2e_nlg_mr.
             Note: value may be a list for non-familyFriendly slots.

    Returns:
        defaultdict(list) mapping original E2E slot names to lists of found values.
    """
    slot_vals_found = defaultdict(list)

    # Build an internal MR representation: list of (internal_slot, value_str) tuples
    # Also keep a parallel list of orig_names for output mapping
    internal_mr = []
    orig_names = []
    for orig_name, value in mr:
        internal_slot = E2E_SLOT_TO_INTERNAL.get(orig_name, orig_name.lower().replace(' ', ''))
        if isinstance(value, list):
            val_str = value[0] if len(value) == 1 else ', '.join(value)
        else:
            val_str = value
        internal_mr.append((internal_slot, val_str))
        orig_names.append(orig_name)

    # Preprocess the utterance
    utt_proc, utt_tok = preprocess_utterance(utt)

    # Mask named entities (name, near)
    internal_mr_masked, utt_proc = mask_named_entities(internal_mr, utt_proc)

    # Re-tokenize after masking
    _, utt_tok = preprocess_utterance(utt_proc)

    # Count slots in MR
    mr_slot_counts = Counter(s for s, _ in internal_mr_masked)

    for i, (slot, value) in enumerate(internal_mr_masked):
        orig_name = orig_names[i]

        if isinstance(value, dict):
            # Masked name-based slot — already found
            pos = value['pos']
            found_value = value['text']
        else:
            # Preprocess the value for matching
            val_lower = re.sub(r'[-/]', ' ', value.lower()).strip(',.?! ')
            val_lower = re.sub(r'\s+', ' ', val_lower)

            pos, is_dupe = find_e2e_slot_realization(
                utt_proc, utt_tok, slot, val_lower, internal_mr_masked,
                ignore_dupes=(mr_slot_counts[slot] > 1)
            )
            found_value = value  # Use the original (unpre-processed) value

        if pos >= 0:
            slot_vals_found[orig_name].append(found_value)

    return slot_vals_found


def eval_compute_ser(examples):
    working_results = defaultdict(list)
    category_results = defaultdict(lambda: defaultdict(list))

    for i, ex in enumerate(examples):
        ref_mr_raw = ex['mr']
        text = ex['surface_form']
        #print("text:", text)
        # Map slot names
        ref_mr_mapped = pack_e2e_nlg_mr(ref_mr_raw)

        for neg_example in ex["negatives"]:
            extracted_mr = extract_mr(text, ref_mr_mapped)
            neg_category = neg_example["label"]
            #print("category:", neg_category)
            _nmr = []
            for slot, vals in neg_example["mr"].items():
                _nmr.append([slot] + vals)
            #print("_nmr:", _nmr)
            neg_mr = pack_e2e_nlg_mr(_nmr)
            neg_mr_mapped_dict = dict(neg_mr)
            #extracted_mr = {k:v for k, v in extracted_mr.items()}
            ref_mr_mapped_dict = dict(ref_mr_mapped)

            #print("extracted_mr:", ref_mr_mapped_dict)
            #print("ref_mr:", ref_mr_mapped)
            #print("neg_mr:", neg_mr)
            tally_ser(category_results, extracted_mr, neg_category, neg_mr_mapped_dict, ref_mr_mapped_dict, working_results)
            #input(">>>")


    print(f"\nResults:")
    for name in ["S_acc", "D_acc", "I_acc", "all_acc"]:
        working_results[name] = np.mean(working_results[name])
        print(f"{name}:", working_results[name])

    mse_ser = np.mean([e ** 2 for e in working_results["ser_error"]])
    print("SER mean square error:", mse_ser)
    mabs_ser = np.mean([np.abs(e) for e in working_results["ser_error"]])
    print("SER mean absolute error:", mabs_ser)

    print()
    for category, results in category_results.items():
        print(f"\ncategory {category}:")
        for name in ["S_acc", "D_acc", "I_acc", "all_acc"]:
            n = len(results[name])
            results[name] = np.mean(results[name])
            print(f"{name} ({n}):", results[name])

        mse_ser = np.mean([e ** 2 for e in results["ser_error"]])
        n = len(results["ser_error"])
        print(f"SER mean square error ({n}):", mse_ser)
        mabs_ser = np.mean([np.abs(e) for e in results["ser_error"]])
        print("SER mean absolute error:", mabs_ser)


    return None


def main():
    parser = argparse.ArgumentParser(
        description='Compute Slot Error Rate (SER) on ViGGO video game NLG outputs.'
    )
    parser.add_argument('input', help='Path to JSON file with examples')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print per-example error details')
    parser.add_argument('--output', '-o', help='Path to save detailed results as JSON')
    args = parser.parse_args()
    print("args", args)

    with open(args.input, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    examples = [e for e in examples if e.get("dataset") == "e2e_nlg"]

    #for e in examples:
    #    if e["topic"] == "video_games":
    #        pass

    print(f'Evaluating SER on {len(examples)} examples...\n')

    eval_compute_ser(examples)


if __name__ == "__main__":
    main()
