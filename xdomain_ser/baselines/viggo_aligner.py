# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
"""Rule-based slot aligner / SER evaluator for the ViGGO video-game dataset.

Re-implementation of the slot-realisation alignment logic from Juraska's
slug2slug aligner -- see https://github.com/jjuraska/data2text-nlg. The
algorithm (keyword matching + boolean/scalar/categorical/list/year slot
handlers + named-entity masking) is theirs; this Python re-implementation
is ours, structured to match the E2E and RNNLG aligners in this package.

Used as the rule-based baseline for the ViGGO domain in the GEM 2026
SER agreement comparisons.

Usage::

    python -m xdomain_ser.baselines.viggo_aligner input.json
    python -m xdomain_ser.baselines.viggo_aligner input.json --verbose
    python -m xdomain_ser.baselines.viggo_aligner input.json --output results.json

Each input example needs:

* ``mr``: list of ``[slot_name, slot_value]`` pairs
* ``surface_form``: the generated (or reference) utterance
"""
from collections import defaultdict

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple
import numpy as np

from xdomain_ser.ranking.make_eval_data import tally_ser

# ---------------------------------------------------------------------------
# Alternatives for slot‐value matching (from alternatives.json)
# ---------------------------------------------------------------------------
ALTERNATIVES: Dict[str, Dict[str, list]] = {
    "genres": {
        "action adventure": [["action", "adventur"]],
        "adventure": ["adventur"],
        "driving racing": ["driving", "drive", "racing", "race"],
        "fighting": ["fight"],
        "mmorpg": ["massive"],
        "platformer": ["platforming"],
        "real time strategy": ["real time", "rts"],
        "role playing": ["roleplaying", "role play", "rpg", "rpgs"],
        "shooter": ["shoot", "fps"],
        "simulation": ["simulat", "sim"],
        "strategy": ["strateg"],
        "tactical": ["tactic"],
        "trivia board game": ["trivia", "board"],
        "turn based strategy": ["turn based"],
        "vehicular combat": [["vehic", "combat"]],
    },
    "player_perspective": {
        "first person": ["fps"],
        "bird view": ["top down"],
    },
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
    "esrb": {
        "e (for everyone)": ["e rated", "rated e", "e rating", "rating e", "everyone", "all"],
        "e 10+ (for everyone 10 and older)": [
            "e10+", "e 10+", "e 10 plus", "everyone 10",
            "everyone above", "everyone over", "everyone older",
        ],
        "t (for teen)": ["t rated", "rated t", "t rating", "rating t", "teen", "teens", "teenagers"],
        "m (for mature)": ["m rated", "rated m", "m rating", "rating m", "mature", "adult"],
    },
}


# ---------------------------------------------------------------------------
# Slot‐name mapping: user JSON slot names → internal names used by the aligner
# ---------------------------------------------------------------------------
SLOT_NAME_MAP = {
    "review_rating": "rating",
    "multiplayer_mode": "has_multiplayer",
    "perspective": "player_perspective",
}

# Reverse mapping of multiplayer_mode values → has_multiplayer boolean values
MULTIPLAYER_VALUE_MAP = {
    "single-player": "no",
    "multi-player": "yes",
    "multiplayer": "yes",
    "single player": "no",
    "multi player": "yes",
}


# ---------------------------------------------------------------------------
# Simple word tokenizer (avoids NLTK dependency)
# ---------------------------------------------------------------------------
def word_tokenize(text: str) -> List[str]:
    """Simple regex‐based word tokenizer that handles contractions and punctuation."""
    # Split contractions like don't → don, 't  and it's → it, 's
    text = re.sub(r"(\w)(n't)\b", r"\1 \2", text)
    text = re.sub(r"(\w)('s)\b", r"\1 \2", text)
    text = re.sub(r"(\w)('re)\b", r"\1 \2", text)
    text = re.sub(r"(\w)('ve)\b", r"\1 \2", text)
    text = re.sub(r"(\w)('ll)\b", r"\1 \2", text)
    text = re.sub(r"(\w)('d)\b", r"\1 \2", text)
    # Separate punctuation from words
    tokens = re.findall(r"\w+(?:[-/]\w+)*|[^\w\s]", text)
    return tokens


# ---------------------------------------------------------------------------
# Utility helpers (ported from slot_aligner/alignment/utils.py)
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


def get_slot_value_alternatives(slot: str) -> dict:
    return ALTERNATIVES.get(slot, {})


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
    'hasusbport': ['usb'],
    'isforbusinesscomputing': ['business'],
    'has_multiplayer': ['multiplayer', 'friends', 'others'],
    'available_on_steam': ['steam'],
    'has_linux_release': ['linux'],
    'has_mac_release': ['mac'],
}
BOOLEAN_SLOT_ANTONYMS = {
    'familyfriendly': ['adult', 'adults'],
    'isforbusinesscomputing': ['personal', 'general', 'home', 'nonbusiness'],
    'has_multiplayer': ['single player'],
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
    'esrb': ['esrb'],
    'rating': ['rating', 'ratings', 'rated', 'rate', 'review', 'reviews'],
    'customerrating': ['customer', 'rating', 'ratings', 'rated', 'rate', 'review', 'reviews', 'star', 'stars'],
    'pricerange': ['price', 'pricing', 'cost', 'costs', 'dollars', 'pounds', 'euros', r'\$', '£', '€'],
}


def align_scalar_slot(text, text_tok, slot, value, slot_mapping=None, value_mapping=None, slot_stem_only=False):
    slot_stem_indexes = []
    slot_stem_positions = []
    leftmost_pos = -1

    text_clean = re.sub(r"'", '', text)
    slot_stems = SCALAR_SLOT_STEMS.get(slot, [])

    if slot_mapping is not None:
        slot = slot_mapping
    alternatives = get_slot_value_alternatives(slot)

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
    if value_mapping is not None:
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
# List slot alignment (from alignment/list_slot.py)
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


def align_list_slot(text, text_tok, slot, value, match_all=True, mode='exact_match', item_sep=', '):
    leftmost_pos = -1
    alternatives = get_slot_value_alternatives(slot)
    items = [item.strip() for item in value.split(item_sep)]

    for item in items:
        pos = find_value_alternative(text, text_tok, item, alternatives, mode=mode)
        if match_all and pos < 0:
            return -1
        if leftmost_pos < 0 or 0 <= pos < leftmost_pos:
            leftmost_pos = pos

    return leftmost_pos


# ---------------------------------------------------------------------------
# Numeric / year slot alignment (from alignment/numeric_slot.py)
# ---------------------------------------------------------------------------
def align_year_slot(text, text_tok, slot, value):
    try:
        int(value)
    except ValueError:
        return -1

    year_alternatives = [value]
    if len(value) == 4:
        year_alternatives.append("'" + value[-2:])
        year_alternatives.append(value[-2:])

    for val in year_alternatives:
        if len(val) > 2:
            pos = text.find(val)
        else:
            _, pos = find_first_in_list(val, text_tok)
        if pos >= 0:
            return pos

    return -1


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
# Preprocessing (from slot_alignment.py)
# ---------------------------------------------------------------------------
def preprocess_mr(mr_as_list):
    mr_processed = []
    for slot, val in mr_as_list:
        match = re.match(r'<\|(?P<slot_name>.*?)\|>', slot)
        if match:
            slot = match.group('slot_name')
        if slot == 'da':
            continue
        val = re.sub(r'[-/]', ' ', val.lower()).strip(',.?! ')
        val = re.sub(r'\s+', ' ', val)
        mr_processed.append((slot, val))
    return mr_processed


def preprocess_utterance(utt):
    utt = re.sub(r'[-/]', ' ', utt.lower())
    utt = re.sub(r'\s+', ' ', utt)
    utt_tok = [w.strip('.,!?') if len(w) > 1 else w for w in word_tokenize(utt)]
    return utt, utt_tok


def mask_named_entities(mr_as_list, utt, ignore_name_slot_dupes=True):
    name_slots = {'addr', 'depart', 'dest', 'developer', 'name', 'near'}

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

                is_dupe = num_mentions > 1 and (slot != 'name' or not ignore_name_slot_dupes)
                value_dict = {'text': value, 'pos': match.start(), 'is_dupe': is_dupe}
                mr_with_pos[i] = (slot, value_dict, orig_pos)

    mr_as_list = [(s, v) for s, v, _ in sorted(mr_with_pos, key=lambda x: x[2])]
    return mr_as_list, utt


# ---------------------------------------------------------------------------
# Slot‐mention alternatives (from slot_alignment.py get_slot_mention_alternatives)
# ---------------------------------------------------------------------------
SLOT_MENTION_MAP = {
    'available_on_steam': ['steam'],
    'genres': ['genre'],
    'has_linux_release': ['linux'],
    'has_mac_release': ['mac'],
    'has_multiplayer': ['multiplayer', 'friends', 'others'],
    'name': ['name', 'particular', 'specific', 'what', 'which'],
    'platforms': ['platform'],
    'player_perspective': ['perspective'],
    'release_year': ['year'],
}


# ---------------------------------------------------------------------------
# Video game slot realization finder (from find_slot_realization, video_game branch)
# ---------------------------------------------------------------------------
def find_slot_realization(text, text_tok, slot, value, mr, ignore_dupes=False):
    """Find the realization of a slot-value pair in text. Returns (position, is_duplicate)."""
    pos = -1
    is_dupe = False

    import string as string_mod
    slot = slot.rstrip(string_mod.digits)
    all_slots = {s for s, _ in mr}

    # Universal values
    if value == 'dontcare' or value == "don't care":
        # Simplified dontcare matching
        text_clean = re.sub(r"'", '', text.lower())
        text_tok_clean = word_tokenize(text_clean)
        slot_mentions = SLOT_MENTION_MAP.get(slot, [slot])
        for stem in slot_mentions:
            if isinstance(stem, str) and stem in text_tok_clean:
                for kw in ['any', 'all', 'vary', 'various', 'different', 'regardless', 'matter',
                           'no preference', 'dont care', 'dont mind', 'not care', 'not mind']:
                    if kw in text_clean:
                        return 0, False
        return -1, False
    elif value == 'none' or value == '':
        slot_mentions = SLOT_MENTION_MAP.get(slot, [slot])
        for stem in slot_mentions:
            if isinstance(stem, str):
                p, d = match_keywords_in_text(stem, text, ignore_dupes=True)
                if p >= 0:
                    return p, d
        return -1, False
    elif value == '?':
        return -1, False

    # Video game domain-specific slot alignment
    if slot in ['platforms', 'player_perspective']:
        pos = align_list_slot(text, text_tok, slot, value, match_all=True, mode='first_word')
    elif slot == 'genres':
        pos = align_list_slot(text, text_tok, slot, value, match_all=True, mode='exact_match')
    elif slot == 'release_year':
        pos = align_year_slot(text, text_tok, slot, value)
    elif slot in ['esrb', 'rating']:
        pos = align_scalar_slot(text, text_tok, slot, value, slot_stem_only=False)
    elif slot in ['available_on_steam', 'has_linux_release', 'has_mac_release', 'has_multiplayer']:
        pos = align_boolean_slot(text, text_tok, slot, value)

    # Fallback: verbatim match
    if pos < 0:
        pos, is_dupe = match_keywords_in_text(value, text, ignore_dupes=ignore_dupes)

    return pos, is_dupe


# ---------------------------------------------------------------------------
# Duplicate re-evaluation (from slot_alignment.py)
# ---------------------------------------------------------------------------
def reevaluate_duplicate_mentions(is_dupe, slot, value, mr, num_das):
    if not is_dupe:
        return False

    val_text = value['text'] if isinstance(value, dict) else value

    if val_text == '':
        all_slots = {s for s, _ in mr}
        if num_das > 1:
            return False
    else:
        value_counts = Counter([
            v['text'] if isinstance(v, dict) else v for _, v in mr
        ])
        if value_counts[val_text] > 1:
            return False

    return True


# ---------------------------------------------------------------------------
# Core: count slot errors for one example
# ---------------------------------------------------------------------------
def extract_mr(utt: str, mr: List[Tuple[str, str]], verbose: bool = False):
    """Count missing and duplicate slot mentions in an utterance.

    Args:
        utt: The surface form / generated utterance.
        mr: List of (slot_name, slot_value) tuples (already mapped to internal names).

    Returns:
        (num_errors, missing_slots, duplicate_slots, num_content_slots)
    """
    #print("utt:", utt)
    #print("mr:", mr)

    slots_found = Counter()
    slots_with_duplicate_mentions = set()
    slot_vals_found = defaultdict(list)
    # Count dialogue acts
    num_das = sum(1 for s, v in mr if s == 'da')

    # Preprocess
    mr_proc = preprocess_mr(mr)
    utt_proc, utt_tok = preprocess_utterance(utt)
    mr_proc, utt_proc = mask_named_entities(mr_proc, utt_proc, ignore_name_slot_dupes=True)

    # MR slot counts (some datasets have duplicate slot names, e.g. genres appears twice)
    mr_slot_counts = Counter(s for s, _ in mr_proc)

    for slot, value in mr_proc:
        if isinstance(value, dict):
            pos, is_dupe = value['pos'], value['is_dupe']
        else:
            pos, is_dupe = find_slot_realization(
                utt_proc, utt_tok, slot, value, mr_proc,
                ignore_dupes=(mr_slot_counts[slot] > 1)
            )

        is_dupe = reevaluate_duplicate_mentions(is_dupe, slot, value, mr_proc, num_das)

        if pos >= 0:
            slots_found.update([slot])
            if isinstance(value, dict):
                value = value['text']
            slot_vals_found[slot].append(value)
        if is_dupe:
            if verbose:
                print(f'  >> Duplicate: {slot} = {value}')
            slots_with_duplicate_mentions.add(slot)
    #print("slot_vals_found:", slot_vals_found)
    #input(">>>")
    incorrect_slots = mr_slot_counts - slots_found
    num_errors = sum(incorrect_slots.values()) + len(slots_with_duplicate_mentions)
    num_content_slots = len(mr_proc)

    return num_errors, list(incorrect_slots), list(slots_with_duplicate_mentions), num_content_slots, slot_vals_found


# ---------------------------------------------------------------------------
# Map JSON slot names/values to internal representation
# ---------------------------------------------------------------------------
def map_mr(mr_pairs: List[List[str]]) -> List[Tuple[str, str]]:
    """Map slot names and values from the user's JSON format to the internal format."""


    REVERSE_KEY_MAP = {
        "review_rating": "rating",
        "esrb_rating": "esrb",
        "perspective": "player_perspective",
        "released": "release_year",
        "multiplayer_mode": "has_multiplayer",
        "steam_availability": "available_on_steam",
        "pc_os_support": None,  # handled specially below
    }


    mapped = []
    for pair in mr_pairs:
        name = pair[0]
        values = pair[1:]

        if not isinstance(values, list):
            values = [values]

        if name == "steam_availability":
            orig_name = "available_on_steam"
            # on_steam -> yes, not_on_steam -> no
            for v in values:
                orig_value = "yes" if v == "on_steam" else "no"
                mapped.append((orig_name, orig_value))

        elif name == "pc_os_support":
            for v in values:
                if v == "Linux" or v == "not_released_on_Linux":
                    orig_name = "has_linux_release"
                    orig_value = "yes" if v == "Linux" else "no"
                elif v == "macOS" or v == "not_released_on_macOS":
                    orig_name = "has_mac_release"
                    orig_value = "yes" if v == "macOS" else "no"
                else:
                    # e.g. "Windows" or other values — skip, since unpack
                    # doesn't produce these from any original slot
                    continue
                mapped.append((orig_name, orig_value))

        elif name == "multiplayer_mode":
            orig_name = "has_multiplayer"
            for v in values:
                orig_value = "yes" if v == "multiplayer" else "no"
                mapped.append((orig_name, orig_value))

        else:
            orig_name = REVERSE_KEY_MAP.get(name, name)
            # Slots whose values were split from a comma-separated list
            # (e.g. genres, platforms) get re-joined into a single slot
            if isinstance(values, list):
                values = values[0]
            mapped.append((orig_name, values))

    #print(f"  >> Mapped: {mapped}")
    #input(">>>")
    return mapped

#######---------------------------

def pack_viggo_mr(d):
    """
    Converts unpacked viggo MR dict back to the original viggo MR string format.

    e.g. give_opinion(name[SpellForce 3], release_year[2017], developer[Grimlore Games], rating[poor])

    :param d: dict with "dact" and "slots" keys (as produced by unpack_viggo_mr)
    :return: viggo MR string
    """
    REVERSE_KEY_MAP = {
        "review_rating": "rating",
        "esrb_rating": "esrb",
        "perspective": "player_perspective",
        "released": "release_year",
        "multiplayer_mode": "has_multiplayer",
        "steam_availability": "available_on_steam",
        "pc_os_support": None,  # handled specially below
    }

    slots = d["slots"]
    slot_value_pairs = []

    for name, values in slots.items():
        if not isinstance(values, list):
            values = [values]

        if name == "steam_availability":
            orig_name = "available_on_steam"
            # on_steam -> yes, not_on_steam -> no
            for v in values:
                orig_value = "yes" if v == "on_steam" else "no"
                slot_value_pairs.append((orig_name, orig_value))

        elif name == "pc_os_support":
            for v in values:
                if v == "Linux" or v == "not_released_on_Linux":
                    orig_name = "has_linux_release"
                    orig_value = "yes" if v == "Linux" else "no"
                elif v == "macOS" or v == "not_released_on_macOS":
                    orig_name = "has_mac_release"
                    orig_value = "yes" if v == "macOS" else "no"
                else:
                    # e.g. "Windows" or other values — skip, since unpack
                    # doesn't produce these from any original slot
                    continue
                slot_value_pairs.append((orig_name, orig_value))

        elif name == "multiplayer_mode":
            orig_name = "has_multiplayer"
            for v in values:
                orig_value = "yes" if v == "multiplayer" else "no"
                slot_value_pairs.append((orig_name, orig_value))

        else:
            orig_name = REVERSE_KEY_MAP.get(name, name)
            # Slots whose values were split from a comma-separated list
            # (e.g. genres, platforms) get re-joined into a single slot
            slot_value_pairs.append((orig_name, ", ".join(values)))

    attrs_str = ", ".join(f"{n}[{v}]" for n, v in slot_value_pairs)
    return f"{dact}({attrs_str})"


def eval_compute_ser(examples: List[dict], verbose: bool = False) -> dict:
    """Compute slot error rate over a list of examples.

    Returns a dict with overall stats and per-example details.
    """
    total_errors = 0
    total_slots = 0
    examples_with_errors = 0
    per_example_results = []

    working_results = defaultdict(list)
    category_results = defaultdict(lambda: defaultdict(list))

    for i, ex in enumerate(examples):
        ref_mr_raw = ex['mr']
        text = ex['surface_form']
        #print("ref_mr_raw:", ref_mr_raw)

        # Map slot names
        ref_mr_mapped = map_mr(ref_mr_raw)
        #print("ref_mr_mapped:", ref_mr_mapped)
        for neg_example in ex["negatives"]:
            num_errors, missing, duplicates, num_slots, extracted_mr = extract_mr(text, ref_mr_mapped,
                                                                                    verbose=verbose)
            neg_category = neg_example["label"]
            #print("category:", neg_category)
            #neg_mr = {k:v[0] for k, v in neg_example["mr"].items()}
            #print("neg_mr:", neg_example["mr"])
            _nmr = []
            for slot, vals in neg_example["mr"].items():
                _nmr.append([slot] + vals)
            #print("_nmr:", _nmr)
            neg_mr = map_mr(_nmr)
            #extracted_mr = {k:v[0] for k, v in extracted_mr.items()}

            ref_mr_mapped_dict = dict(ref_mr_mapped)
            neg_mr_mapped_dict = dict(neg_mr)
            #print("extracted_mr:", extracted_mr)
            #print("ref_mr:", ref_mr_mapped_dict)
            #print("neg_mr:", neg_mr_mapped_dict)
            #input(">>>")
            tally_ser(category_results, extracted_mr, neg_category, neg_mr_mapped_dict, ref_mr_mapped_dict, working_results)



            total_errors += num_errors
            total_slots += num_slots
            if num_errors > 0:
                examples_with_errors += 1

            result = {
                'index': i,
                'num_errors': num_errors,
                'num_slots': num_slots,
                'missing_slots': missing,
                'duplicate_slots': duplicates,
            }
            per_example_results.append(result)

            if verbose and num_errors > 0:
                print(f'\nExample {i}:')
                print(f'  MR:  {mr_raw}')
                print(f'  Utt: {utt}')
                print(f'  Missing:    {missing}')
                print(f'  Duplicates: {duplicates}')
                print(f'  Errors: {num_errors} / {num_slots}')

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



    slot_level_ser = total_errors / total_slots if total_slots > 0 else 0.0
    utterance_level_ser = examples_with_errors / len(examples) if examples else 0.0
    summary = {
        'slot_level_ser': slot_level_ser,
        'utterance_level_ser': utterance_level_ser,
        'total_errors': total_errors,
        'total_slots': total_slots,
        'examples_with_errors': examples_with_errors,
        'total_examples': len(examples),
        'per_example': per_example_results,
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
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
    examples = [e for e in examples if e["topic"] == "video_games"]

    #for e in examples:
    #    if e["topic"] == "video_games":
    #        pass

    print(f'Evaluating SER on {len(examples)} examples...\n')

    results = eval_compute_ser(examples, verbose=args.verbose)

    print(f'\n{"=" * 50}')
    print(f'Slot-level SER:      {results["slot_level_ser"]:.4f}  ({results["total_errors"]} / {results["total_slots"]})')
    print(f'Utterance-level SER: {results["utterance_level_ser"]:.4f}  ({results["examples_with_errors"]} / {results["total_examples"]})')
    print(f'{"=" * 50}')
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'\nDetailed results saved to: {args.output}')


if __name__ == '__main__':
    main()