# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0.
"""Interface tests for the three rule-based aligner baselines.

Each aligner's ``extract_mr`` runs on a small hand-built example: once
with every slot realised in the utterance, once with one slot missing.
Codifies the Stage-7 migration smoke test (CPU-only, no models).
"""
from xdomain_ser.baselines import e2e_aligner, rnnlg_aligner, viggo_aligner


E2E_MR = [("name", "The Vaults"), ("eatType", "pub"),
          ("food", "Japanese"), ("priceRange", "cheap")]


def test_e2e_extract_all_slots_found():
    found = e2e_aligner.extract_mr("The Vaults is a cheap Japanese pub.", E2E_MR)
    assert set(found) == {"name", "eatType", "food", "priceRange"}


def test_e2e_extract_missing_slot():
    found = e2e_aligner.extract_mr("The Vaults is a cheap pub.", E2E_MR)
    assert "food" not in found
    assert {"name", "eatType", "priceRange"} <= set(found)


RNNLG_MR = [("name", "Red Door Cafe"), ("pricerange", "cheap"), ("area", "nob hill")]


def test_rnnlg_extract_all_slots_found():
    found = rnnlg_aligner.extract_mr(
        "Red Door Cafe is a cheap restaurant in Nob Hill.", RNNLG_MR,
        domain="restaurant")
    assert set(found) == {"name", "pricerange", "area"}


def test_rnnlg_extract_missing_slot():
    found = rnnlg_aligner.extract_mr(
        "Red Door Cafe is a cheap restaurant.", RNNLG_MR, domain="restaurant")
    assert "area" not in found
    assert {"name", "pricerange"} <= set(found)


VIGGO_MR = [("name", "The Witcher 3"), ("release_year", "2015")]


def test_viggo_extract_all_slots_found():
    num_errors, missing, dupes, n_slots, found = viggo_aligner.extract_mr(
        "The Witcher 3 came out in 2015.", VIGGO_MR)
    assert num_errors == 0
    assert missing == []
    assert n_slots == 2


def test_viggo_extract_missing_slot():
    num_errors, missing, dupes, n_slots, found = viggo_aligner.extract_mr(
        "The Witcher 3 is a great game.", VIGGO_MR)
    assert num_errors == 1
    assert missing == ["release_year"]
