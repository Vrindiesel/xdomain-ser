# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Rule-based slot aligner / SER evaluator for the RNNLG datasets.

Re-implementation of the slot-realisation alignment logic from Wen et al.
(2015), "Semantically Conditioned LSTM-based Natural Language Generation
for Spoken Dialogue Systems" -- specifically the ERRScorer evaluator
they shipped with the RNNLG release at
https://github.com/shawnwun/RNNLG. The algorithm (keyword matching with
per-domain slot dictionaries + boolean/scalar/categorical/numeric/list
handlers) is theirs; this Python re-implementation is ours, structured
to match the E2E and ViGGO aligners in this package.

Supports the four RNNLG domains: hotel, laptop, restaurant, tv. Used as
the rule-based baseline for those domains in the GEM 2026 SER agreement
comparisons.
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from xdomain_ser.ranking.make_eval_data import tally_ser



# ===========================================================================
# Simple word tokenizer (avoids NLTK dependency)
# ===========================================================================
def word_tokenize(text: str) -> List[str]:
    text = re.sub(r"(\w)(n't)\b", r"\1 \2", text)
    text = re.sub(r"(\w)('s)\b", r"\1 \2", text)
    text = re.sub(r"(\w)('re)\b", r"\1 \2", text)
    text = re.sub(r"(\w)('ve)\b", r"\1 \2", text)
    text = re.sub(r"(\w)('ll)\b", r"\1 \2", text)
    text = re.sub(r"(\w)('d)\b", r"\1 \2", text)
    tokens = re.findall(r"\w+(?:[-/]\w+)*|[^\w\s]", text)
    return tokens


# ===========================================================================
# Utility helpers (from slot_aligner/alignment/utils.py)
# ===========================================================================
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


# ===========================================================================
# Alternatives for RNNLG slot-value matching
# ===========================================================================
RNNLG_ALTERNATIVES: Dict[str, Dict[str, list]] = {
    # --- shared with E2E / jjuraska aligner ---
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


def get_rnnlg_slot_value_alternatives(slot: str) -> dict:
    return RNNLG_ALTERNATIVES.get(slot, {})


# ===========================================================================
# Keyword matching (from slot_alignment.py _match_keywords_in_text)
# ===========================================================================
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


# ===========================================================================
# Boolean slot alignment (from alignment/boolean_slot.py)
# ===========================================================================
NEGATION_CUES_PRE = [
    'no', 'not', 'non', 'none', 'neither', 'nor', 'never', "n't", 'cannot',
    'excluded', 'lack', 'lacks', 'lacking', 'unavailable', 'without', 'zero',
    'everything but',
]
NEGATION_CUES_POST = [
    'not', 'nor', 'never', "n't", 'cannot', 'excluded', 'unavailable',
]
CONTRAST_CUES = ['but', 'however', 'although', 'though', 'nevertheless']

# Stems for boolean slots across RNNLG domains
BOOLEAN_SLOT_STEMS = {
    'kidsallowed': ['child', 'children', 'kid', 'kids', 'family', 'families'],
    'dogsallowed': ['dog', 'dogs', 'pet', 'pets', 'puppy', 'animal', 'animals'],
    'hasinternet': ['internet', 'wifi', 'wi fi'],
    'acceptscreditcards': ['card', 'cards', 'credit'],
    'isforbusinesscomputing': ['business'],
    'hasusbport': ['usb'],
}

BOOLEAN_SLOT_ANTONYMS = {
    'kidsallowed': ['adult', 'adults'],
    'isforbusinesscomputing': ['personal', 'general', 'home', 'nonbusiness'],
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
        if ' ' in slot_stem:
            stem_pos = text.find(slot_stem)
            if stem_pos >= 0:
                # Use position-based negation check
                if value == true_val:
                    # Check for preceding negation via a simple scan
                    before = text[max(0, stem_pos - NEG_POS_FALSE_PRE_THRESH):stem_pos]
                    has_neg = any(n in before for n in ['not', "n't", 'no ', 'without', 'never'])
                    if not has_neg:
                        return stem_pos
                else:
                    before = text[max(0, stem_pos - NEG_POS_FALSE_PRE_THRESH):stem_pos]
                    has_neg = any(n in before for n in ['not', "n't", 'no ', 'without', 'never'])
                    if has_neg:
                        return stem_pos
        else:
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


# ===========================================================================
# Scalar slot alignment (from alignment/scalar_slot.py)
# ===========================================================================
DIST_IDX_THRESH = 10
DIST_POS_THRESH = 30

SCALAR_SLOT_STEMS = {
    'pricerange': ['price', 'pricing', 'cost', 'costs', 'dollars', 'pounds', 'euros', r'\$', '£', '€'],
    'customerrating': ['customer', 'rating', 'ratings', 'rated', 'rate', 'review', 'reviews', 'star', 'stars'],
    'batteryrating': ['battery'],
    'ecorating': ['eco'],
    'screensizerange': ['screen'],
    'weightrange': ['weight'],
    'driverange': ['drive'],
}


def align_scalar_slot(text, text_tok, slot, value, slot_mapping=None, value_mapping=None, slot_stem_only=False):
    slot_stem_indexes = []
    slot_stem_positions = []
    leftmost_pos = -1

    text_clean = re.sub(r"'", '', text)
    slot_stems = SCALAR_SLOT_STEMS.get(slot, [])

    lookup_slot = slot_mapping if slot_mapping is not None else slot
    alternatives = get_rnnlg_slot_value_alternatives(lookup_slot)

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
            for vpos in val_positions:
                if vpos < leftmost_pos or leftmost_pos == -1:
                    leftmost_pos = vpos
                if len(slot_stem_positions) > 0:
                    for ssp in slot_stem_positions:
                        if abs(vpos - ssp) < DIST_POS_THRESH:
                            return vpos
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


# ===========================================================================
# Categorical slot alignment (from alignment/categorical_slots.py)
# ===========================================================================
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
    alternatives = get_rnnlg_slot_value_alternatives(slot)
    pos = find_value_alternative(text, text_tok, value, alternatives, mode=mode, allow_plural=allow_plural)
    return pos


# ===========================================================================
# Numeric slot alignment (from alignment/numeric_slot.py)
# ===========================================================================
NUMBER_TO_WORD = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
    '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
    '18': 'eighteen', '19': 'nineteen', '20': 'twenty',
    '30': 'thirty', '40': 'forty', '50': 'fifty', '60': 'sixty',
    '70': 'seventy', '80': 'eighty', '90': 'ninety', '100': 'hundred',
}
WORD_TO_NUMBER = {v: k for k, v in NUMBER_TO_WORD.items()}


def align_numeric_slot(text, text_tok, slot, value):
    """Match a simple numeric value (possibly written as a word)."""
    match = re.search(fr'\b{re.escape(value)}\b', text)
    if match:
        return match.start()

    for value_word in value.split():
        for number_map in [NUMBER_TO_WORD, WORD_TO_NUMBER]:
            value_alt = number_map.get(value_word)
            if value_alt:
                match = re.search(fr'\b{re.escape(value_alt)}\b', text)
                if match:
                    return match.start()

    return -1


def align_numeric_slot_with_unit(text, text_tok, slot, value):
    """Match a numeric value that has a unit suffix (e.g. '3.5 hour', '33.7 inch')."""
    value_number = value.split(' ')[0]
    try:
        float(value_number)
    except ValueError:
        return -1

    _, pos = find_first_in_list(value_number, text_tok)
    return pos


# ===========================================================================
# List with conjunctions alignment (from alignment/list_slot.py)
# ===========================================================================
def align_list_with_conjunctions_slot(text, text_tok, slot, value, match_all=True):
    """Match a value composed of multiple items joined by 'and', 'or', ','."""
    separators = {',', 'and', 'or', 'with'}

    value_tok = word_tokenize(value)
    value_items = []
    end_of_prev_item = -1
    leftmost_pos = -1

    # Split the value into items by separators
    for i, tok in enumerate(value_tok):
        if tok in separators and i > end_of_prev_item + 1:
            item = ' '.join(value_tok[end_of_prev_item + 1:i])
            value_items.append(item)
            end_of_prev_item = i

    if end_of_prev_item < len(value_tok) - 1:
        item = ' '.join(value_tok[end_of_prev_item + 1:])
        value_items.append(item)

    # If no separators found, treat the whole value as one item
    if not value_items:
        value_items = [value]

    for item in value_items:
        pos = text.find(item)
        if 0 <= pos < leftmost_pos or leftmost_pos == -1:
            leftmost_pos = pos
        if match_all and pos < 0:
            return -1

    if leftmost_pos < 0:
        return -1

    return leftmost_pos


# ===========================================================================
# Preprocessing
# ===========================================================================
def preprocess_utterance(utt):
    """Lowercase, normalise dashes/slashes/spaces, tokenise."""
    # RNNLG uses "word -s" for plurals; normalise the space around the hyphen
    utt = re.sub(r' -s\b', 's', utt)
    utt = re.sub(r' -', '-', utt)
    utt = re.sub(r'[-/]', ' ', utt.lower())
    utt = re.sub(r'\s+', ' ', utt)
    utt_tok = [w.strip('.,!?') if len(w) > 1 else w for w in word_tokenize(utt)]
    return utt, utt_tok


def mask_named_entities(mr_as_list, utt):
    """Mask verbatim mentions of name-slot values in the utterance."""
    name_slots = {'name', 'near', 'address'}

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


# ===========================================================================
# Slot-mention alternatives (from slot_semantic_dict.txt + detect.pair)
# ===========================================================================
SLOT_MENTION_MAP = {
    'address': ['address'],
    'area': ['area', 'part of the city', 'part of the town', 'part'],
    'count': [],
    'food': ['food'],
    'goodformeal': ['meal'],
    'name': ['name'],
    'near': ['area', 'part of the city', 'part of the town', 'part'],
    'phone': ['phone number', 'number', 'telephone number', 'phone'],
    'postcode': ['postcode', 'post code', 'zipcode', 'zip code'],
    'price': ['price'],
    'pricerange': ['price range', 'pricerange', 'range', 'price'],
    'type': [],
    'battery': ['battery', 'battery life'],
    'batteryrating': ['battery rating', 'battery life', 'battery'],
    'design': ['design'],
    'dimension': ['size', 'dimension', 'display'],
    'drive': ['drive', 'hard drive', 'drive size'],
    'driverange': ['drive range', 'hard drive range', 'drive size', 'hard drive'],
    'family': ['family', 'product line', 'product family', 'line'],
    'memory': ['memory'],
    'platform': ['platform', 'operating system', 'os'],
    'processor': ['processor', 'cpu'],
    'utility': ['utility'],
    'warranty': ['warranty'],
    'weight': ['weight'],
    'weightrange': ['weight range', 'weight'],
    'kidsallowed': ['child', 'children', 'kid', 'kids'],
    'dogsallowed': ['dog', 'dogs', 'puppy'],
    'hasinternet': ['internet', 'wifi', 'internet connection', 'internet service'],
    'acceptscreditcards': ['credit cards', 'credit card', 'cards', 'card'],
    'isforbusinesscomputing': ['business', 'home use', 'business use', 'business computing', 'personal use'],
    'hasusbport': ['usb port', 'usb'],
    'hdmiport': ['hdmi port', 'hdmi'],
    'ecorating': ['eco rating', 'eco'],
    'audio': ['audio'],
    'accessories': ['accessories', 'accessory'],
    'color': ['color', 'colour'],
    'powerconsumption': ['power consumption', 'consumption', 'power'],
    'resolution': ['resolution'],
    'screensize': ['screen size'],
    'screensizerange': ['screen size range'],
}


# ===========================================================================
# Binary slots (which use boolean alignment vs verbatim)
# ===========================================================================
BINARY_SLOTS = {
    'kidsallowed', 'dogsallowed', 'hasinternet',
    'acceptscreditcards', 'isforbusinesscomputing', 'hasusbport',
}

# true_val/false_val differ by domain
BINARY_TRUE_FALSE = {
    # restaurant / hotel use yes/no
    'kidsallowed': ('yes', 'no'),
    'dogsallowed': ('yes', 'no'),
    'hasinternet': ('yes', 'no'),
    'acceptscreditcards': ('yes', 'no'),
    # laptop / tv use true/false
    'isforbusinesscomputing': ('true', 'false'),
    'hasusbport': ('true', 'false'),
}

# Slots using numeric-with-unit alignment (laptop, tv domains)
NUMERIC_WITH_UNIT_SLOTS = {
    'battery', 'dimension', 'drive', 'weight',         # laptop
    'powerconsumption', 'price', 'screensize',          # tv (price with unit)
}

# Slots using list-with-conjunctions alignment (laptop, tv domains)
LIST_CONJUNCTION_SLOTS = {
    'design', 'utility',        # laptop
    'accessories', 'color',     # tv
}


# ===========================================================================
# Domain-specific find_slot_realization
# ===========================================================================
def find_rnnlg_slot_realization(text, text_tok, slot, value, mr, domain,
                                 ignore_dupes=False):
    """Find the realization of a slot-value pair in text for RNNLG domains.

    Returns (position, is_duplicate).
    """
    pos = -1
    is_dupe = False

    # --- Universal special values ---
    if value == 'dontcare' or value == "dont_care" or value == "don't care":
        slot_mentions = SLOT_MENTION_MAP.get(slot, [slot])
        text_clean = re.sub(r"'", '', text.lower())
        for stem in slot_mentions:
            if stem in text_clean:
                for kw in ['any', 'all', 'vary', 'various', 'different', 'regardless',
                            'matter', 'no preference', 'dont care', 'dont mind',
                            'not care', 'not mind', 'do not care']:
                    if kw in text_clean:
                        return 0, False
        return -1, False

    if value == 'none' or value == '':
        slot_mentions = SLOT_MENTION_MAP.get(slot, [slot])
        for stem in slot_mentions:
            if isinstance(stem, str):
                p, d = match_keywords_in_text(stem, text, ignore_dupes=True)
                if p >= 0:
                    return p, d
        return -1, False

    if value == '?':
        # Request act — check for slot mention
        slot_mentions = SLOT_MENTION_MAP.get(slot, [slot])
        for stem in slot_mentions:
            if isinstance(stem, str):
                p, d = match_keywords_in_text(stem, text, ignore_dupes=True)
                if p >= 0:
                    return p, d
        return -1, False

    # --- Binary slots ---
    if slot in BINARY_SLOTS:
        true_val, false_val = BINARY_TRUE_FALSE.get(slot, ('yes', 'no'))
        pos = align_boolean_slot(text, text_tok, slot, value,
                                  true_val=true_val, false_val=false_val)
        if pos >= 0:
            return pos, False

    # --- Domain-specific alignment ---
    if domain in ('restaurant', 'hotel'):
        if slot == 'area':
            pos = align_categorical_slot(text, text_tok, slot, value, mode='exact_match')
        elif slot == 'pricerange':
            pos = align_scalar_slot(text, text_tok, slot, value, slot_stem_only=False)
        elif slot == 'food':
            # Verbatim match for food values in RNNLG (no WordNet needed;
            # food values are specific cuisine names like 'basque', 'thai')
            pos = text.find(value)
        elif slot == 'count':
            pos = align_numeric_slot(text, text_tok, slot, value)
        elif slot == 'type':
            pos = align_categorical_slot(text, text_tok, slot, value,
                                          mode='first_word', allow_plural=True)

    elif domain == 'laptop':
        if slot in NUMERIC_WITH_UNIT_SLOTS:
            pos = align_numeric_slot_with_unit(text, text_tok, slot, value)
        elif slot in LIST_CONJUNCTION_SLOTS:
            pos = align_list_with_conjunctions_slot(text, text_tok, slot, value, match_all=True)
        elif slot == 'isforbusinesscomputing':
            pass  # already handled above in binary slots
        elif slot == 'pricerange':
            pos = align_scalar_slot(text, text_tok, slot, value, slot_stem_only=False)
        elif slot in ('batteryrating', 'weightrange', 'driverange'):
            pos = align_scalar_slot(text, text_tok, slot, value, slot_stem_only=False)
        elif slot == 'count':
            pos = align_numeric_slot(text, text_tok, slot, value)
        elif slot == 'type':
            pos = align_categorical_slot(text, text_tok, slot, value,
                                          mode='first_word', allow_plural=True)

    elif domain == 'tv':
        if slot in NUMERIC_WITH_UNIT_SLOTS:
            pos = align_numeric_slot_with_unit(text, text_tok, slot, value)
        elif slot in LIST_CONJUNCTION_SLOTS:
            pos = align_list_with_conjunctions_slot(text, text_tok, slot, value, match_all=True)
        elif slot == 'hasusbport':
            pass  # already handled above in binary slots
        elif slot == 'pricerange':
            pos = align_scalar_slot(text, text_tok, slot, value, slot_stem_only=False)
        elif slot in ('ecorating', 'screensizerange'):
            pos = align_scalar_slot(text, text_tok, slot, value, slot_stem_only=False)
        elif slot == 'hdmiport':
            pos = align_numeric_slot(text, text_tok, slot, value)
        elif slot == 'count':
            pos = align_numeric_slot(text, text_tok, slot, value)
        elif slot == 'type':
            pos = align_categorical_slot(text, text_tok, slot, value,
                                          mode='first_word', allow_plural=True)

    # --- Fallback: verbatim match ---
    if pos < 0:
        pos, is_dupe = match_keywords_in_text(value, text, ignore_dupes=ignore_dupes)

    return pos, is_dupe


# ===========================================================================
# extract_mr: extract slot values from utterance for RNNLG examples
# ===========================================================================
def extract_mr(utt: str, mr, domain='restaurant'):
    """Extract slot-value realizations from an utterance given the reference MR.

    Args:
        utt: The surface form / generated utterance.
        mr: List of (slot_name, value) tuples from pack_rnnlg_mr.
        domain: One of 'restaurant', 'hotel', 'laptop', 'tv'.

    Returns:
        defaultdict(list) mapping RNNLG slot names to lists of found values.
    """
    slot_vals_found = defaultdict(list)

    # Build internal MR: list of (slot, value_str)
    internal_mr = []
    for slot_name, value in mr:
        val_str = value if value is not None else '?'
        internal_mr.append((slot_name, val_str))

    # Preprocess utterance
    utt_proc, utt_tok = preprocess_utterance(utt)

    # Mask named entities
    internal_mr_masked, utt_proc = mask_named_entities(internal_mr, utt_proc)

    # Re-tokenize after masking
    _, utt_tok = preprocess_utterance(utt_proc)

    # Count slots in MR
    mr_slot_counts = Counter(s for s, _ in internal_mr_masked)

    for i, (slot, value) in enumerate(internal_mr_masked):
        if isinstance(value, dict):
            # Masked name-slot — already found
            slot_vals_found[slot].append(value['text'])
            continue

        # Preprocess the value for matching
        # Normalise RNNLG's "word -s" plural tokens
        val_lower = re.sub(r' -s\b', 's', value)
        val_lower = re.sub(r' -', '-', val_lower)
        val_lower = re.sub(r'[-/]', ' ', val_lower.lower()).strip(',.?! ')
        val_lower = re.sub(r'\s+', ' ', val_lower)

        pos, is_dupe = find_rnnlg_slot_realization(
            utt_proc, utt_tok, slot, val_lower, internal_mr_masked, domain,
            ignore_dupes=(mr_slot_counts[slot] > 1)
        )

        if pos >= 0:
            slot_vals_found[slot].append(value)

    return slot_vals_found





def pack_rnnlg_mr(d, topic):
    """
    Converts an unpacked RNNLG MR dict back to a list of (original_slot_name, original_value) tuples.

    :param d: dict with "dact" and "slots" keys (as produced by unpack_rnnlg_mr)
    :param topic: one of "restaurant", "hotel", "laptop", "tv"
    :return: list of (slot_name, value) tuples using original RNNLG slot names/values
    """

    # Reverse of KEY_MAP: unpacked name (without topic prefix) -> original RNNLG slot name
    # Note: multiple original names may map to the same unpacked name.
    # We pick the canonical form that actually appears in the RNNLG data files.
    REVERSE_KEY_MAP = {
        "child_friendly": "kidsallowed",       # kids-allowed -> child_friendly, but data uses kidsallowed
        "has_internet": "hasinternet",
        "accepts_credit_cards": "acceptscreditcards",
        "pets_allowed": "dogsallowed",          # dogs-allowed/dogs_allowed/dogsallowed -> pets_allowed
        "is_for_business_computing": "isforbusinesscomputing",
        "has_usb_port": "hasusbport",
        "screen_size_range": "screensizerange",
        "eco_rating": "ecorating",
        "price_range": "pricerange",
        "power_consumption": "powerconsumption",
        "screen_size": "screensize",
        "hdmi_port": "hdmiport",
        "weight_range": "weightrange",
        "battery_rating": "batteryrating",
        "drive_range": "driverange",
        "good_for_meal": "goodformeal",
    }

    # Reverse value maps for boolean/binary slots.
    # Maps (unpacked_slot_name, unpacked_value) -> original_value
    REVERSE_VALUE_MAP = {
        # child_friendly (from kids-allowed)
        ("child_friendly", "child_friendly"): "yes",
        ("child_friendly", "not_child_friendly"): "no",
        # has_internet
        ("has_internet", "has_internet"): "yes",
        ("has_internet", "no_internet"): "no",
        # accepts_credit_cards
        ("accepts_credit_cards", "accepts_credit_cards"): "yes",
        ("accepts_credit_cards", "no_credit_cards"): "no",
        # pets_allowed (from dogsallowed)
        ("pets_allowed", "pets_allowed"): "yes",
        ("pets_allowed", "no_pets_allowed"): "no",
        # is_for_business_computing
        ("is_for_business_computing", "business_oriented"): "true",
        ("is_for_business_computing", "not_business_oriented"): "false",
        # has_usb_port
        ("has_usb_port", "has_usb_port"): "true",
        ("has_usb_port", "no_usb_port"): "false",
    }

    prefix = topic + "_"

    slot_value_pairs = []

    for name_values in d:
        prefixed_name, values = name_values[0], name_values[1:]
        if not isinstance(values, list):
            values = [values]

        # Strip the topic prefix
        if prefixed_name.startswith(prefix):
            unpacked_name = prefixed_name[len(prefix):]
        else:
            unpacked_name = prefixed_name

        # Get original RNNLG slot name
        orig_name = REVERSE_KEY_MAP.get(unpacked_name, unpacked_name)

        for v in values:
            if v is None:
                # Slot with no value (e.g., request(area))
                slot_value_pairs.append((orig_name, None))
            elif v == "dontcare":
                slot_value_pairs.append((orig_name, "dontcare"))
            else:
                # Check if this is a boolean slot with transformed values
                orig_value = REVERSE_VALUE_MAP.get((unpacked_name, v), v)
                slot_value_pairs.append((orig_name, orig_value))

    return slot_value_pairs

def eval_compute_ser(examples, domain="restaurant"):
    working_results = defaultdict(list)
    category_results = defaultdict(lambda: defaultdict(list))

    for i, ex in enumerate(examples):
        ref_mr_raw = ex['mr']
        text = ex['surface_form']

        # Map slot names
        ref_mr_mapped = pack_rnnlg_mr(ref_mr_raw, domain)
        #input(">>>")
        for neg_example in ex["negatives"]:
            extracted_mr = extract_mr(text, ref_mr_mapped, domain)
            neg_category = neg_example["label"]
            #print("category:", neg_category)
            #neg_mr = {k:v[0] for k, v in neg_example["mr"].items()}
            #print("neg_mr:", neg_example["mr"])
            _nmr = []
            for slot, vals in neg_example["mr"].items():
                _nmr.append([slot] + vals)
            #print("_nmr:", _nmr)
            neg_mr = pack_rnnlg_mr(_nmr, domain)
            neg_mr_mapped_dict = dict(neg_mr)
            #extracted_mr = {k:v[0] for k, v in extracted_mr.items()}
            ref_mr_mapped_dict = dict(ref_mr_mapped)
            #print("extracted_mr:", extracted_mr)
            #print("ref_mr_mapped_dict:", ref_mr_mapped_dict)
            #print("neg_mr_mapped_dict:", neg_mr_mapped_dict)
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

# ===========================================================================
# CLI
# ===========================================================================

# Map topic field values to RNNLG domain identifiers
TOPIC_TO_DOMAIN = {
    'restaurant': 'restaurant',
    'restaurants': 'restaurant',
    'hotel': 'hotel',
    'hotels': 'hotel',
    'laptop': 'laptop',
    'laptops': 'laptop',
    'tv': 'tv',
    'television': 'tv',
}

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
        data = json.load(f)

    for topic in ["laptop", "hotel", "tv", "restaurant"]:
        print(f'\nEvaluating SER on {topic} examples...\n')
        examples = [e for e in data if e.get("topic") == topic]
        eval_compute_ser(examples)


if __name__ == "__main__":
    main()
