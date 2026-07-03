# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0.
"""Stage-2 fixture tests for core SER computation.

These tests exercise the stateless ``compute_ser`` / ``compute_slot_f1`` API
and the ``<LIST>(slot: value); ...</LIST>`` parser. The intent is regression
protection during the rest of the release migration, not full coverage.
"""
from xdomain_ser.core.ser import (
    compute_ser,
    compute_slot_f1,
    extract_attributes_dict,
    _values_equal,
)
from xdomain_ser.core.data_helper import LIST_START, LIST_END


def test_exact_match_zero_ser():
    mr = {"name": "The Vaults", "priceRange": "cheap"}
    result = compute_ser(mr, mr)
    assert result == {
        "SER": 0.0, "S": 0, "D": 0, "I": 0, "N_ref": 2,
        "errors": {"S": [], "D": [], "I": []},
    }


def test_one_substitution():
    gold = {"name": "The Vaults", "priceRange": "cheap"}
    pred = {"name": "The Vaults", "priceRange": "moderate"}
    result = compute_ser(pred, gold)
    assert result["S"] == 1
    assert result["D"] == 0
    assert result["I"] == 0
    assert result["SER"] == 0.5  # 1 error / 2 ref slots


def test_one_deletion():
    gold = {"name": "The Vaults", "priceRange": "cheap"}
    pred = {"name": "The Vaults"}
    result = compute_ser(pred, gold)
    assert result["S"] == 0
    assert result["D"] == 1
    assert result["I"] == 0


def test_one_insertion():
    gold = {"name": "The Vaults"}
    pred = {"name": "The Vaults", "priceRange": "cheap"}
    result = compute_ser(pred, gold)
    assert result["S"] == 0
    assert result["D"] == 0
    assert result["I"] == 1


def test_slot_f1_perfect():
    mr = {"name": "Alimentum", "area": "city centre"}
    result = compute_slot_f1(mr, mr)
    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["f1"] == 1.0


def test_values_equal_boolean_synonyms():
    # SlotErrorRate value normalisation maps boolean synonyms.
    assert _values_equal("yes", "true")
    assert _values_equal("no", "false")
    assert not _values_equal("yes", "no")


def test_extract_attributes_dict_list_delimited():
    # NOTE: the parser strips ``<LIST>(`` from the start, then looks for either
    # ``</List>`` (capitalised; legacy) or another ``<LIST>`` to bound the right
    # side. It does not strip an all-caps ``</LIST>`` -- callers strip that
    # before invocation. We mirror the caller pattern here.
    mr_str = f"{LIST_START}(name: The Vaults); (priceRange: cheap){LIST_END}"
    mr_str = mr_str.replace(LIST_END, "")
    parsed = extract_attributes_dict(mr_str)
    assert parsed["name"] == ["The Vaults"]
    assert parsed["priceRange"] == ["cheap"]


def test_imports_resolve():
    # registry and rank_metrics should be importable as part of the package.
    from xdomain_ser.core import registry  # noqa: F401
    from xdomain_ser.core.rank_metrics import mean_reciprocal_rank
    assert abs(mean_reciprocal_rank([[0, 0, 1], [0, 1, 0], [1, 0, 0]]) - 0.611111) < 1e-3
