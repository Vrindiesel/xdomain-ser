# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0.
"""Tests for core data / prompt helpers (``xdomain_ser.core.data_helper``).

Regression guards for MR-list construction, the
``<LIST>(slot: value); ...</LIST>`` prompt-output formatting (and its round-trip
through the SER parser), and loading the shipped hint-map file.
"""
import json

from xdomain_ser.core.data_helper import (
    LIST_START,
    LIST_END,
    make_mr_list,
    build_prompt_hint_map,
    make_prompt_example,
    sort_by_key_length,
)
from xdomain_ser.core.ser import extract_attributes_dict
from xdomain_ser.core import registry


def test_list_delimiters():
    assert LIST_START == "<LIST>"
    assert LIST_END == "</LIST>"


def test_make_mr_list_str_value():
    assert make_mr_list({"name": "The Vaults"}) == [("name", "The Vaults")]


def test_make_mr_list_multi_value():
    assert make_mr_list({"food": ["Italian", "French"]}) == [
        ("food", "Italian"),
        ("food", "French"),
    ]


def test_make_mr_list_skips_none():
    assert make_mr_list({"a": None, "b": "x"}) == [("b", "x")]


def test_build_prompt_hint_map_output_format():
    mr = [("name", "The Vaults"), ("priceRange", "cheap")]
    hint_map = {"hint_map_id": "hm_e2e_nlg", "hint_map": {"name": "..."}}
    output, prompt = build_prompt_hint_map(mr, "The Vaults is cheap.", hint_map, "")
    assert output == f"{LIST_START}(name: The Vaults); (priceRange: cheap){LIST_END}"
    assert "### Response:" in prompt
    assert "The Vaults is cheap." in prompt
    assert "hm_e2e_nlg" in prompt


def test_build_prompt_hint_map_no_output():
    output, prompt = build_prompt_hint_map(
        [("name", "X")], "txt", {"hint_map_id": "h", "hint_map": {}}, "", make_output=False
    )
    assert output is None
    assert "txt" in prompt


def test_build_prompt_hint_map_round_trips_through_ser_parser():
    mr = [("name", "The Vaults"), ("priceRange", "cheap")]
    output, _ = build_prompt_hint_map(mr, "txt", {"hint_map_id": "h", "hint_map": {}}, "")
    # The SER parser strips the leading ``<LIST>(`` and bounds the right side on
    # ``</List>`` / ``<LIST>``; callers strip the trailing ``</LIST>`` first
    # (mirrors tests/test_ser.py).
    parsed = extract_attributes_dict(output.replace(LIST_END, ""))
    assert parsed["name"] == ["The Vaults"]
    assert parsed["priceRange"] == ["cheap"]


def test_make_prompt_example_flat_mr():
    text, mr_str = make_prompt_example({"mr": {"name": "X"}, "surface_form": "a sentence"})
    assert text == "a sentence"
    assert mr_str == f"{LIST_START}(name: X){LIST_END}"


def test_make_prompt_example_nested_slots():
    text, mr_str = make_prompt_example({"mr": {"slots": {"name": "X"}}, "surface_form": "s"})
    assert mr_str == f"{LIST_START}(name: X){LIST_END}"


def test_sort_by_key_length_descending():
    exs = [{"mr": [1]}, {"mr": [1, 2, 3]}, {"mr": [1, 2]}]
    out = sort_by_key_length(exs, "mr")
    assert [len(e["mr"]) for e in out] == [3, 2, 1]


def test_hint_maps_file_loads():
    path = registry.DATA_ROOT / "multi_ser_v9" / "hint_maps_v4.json"
    with open(path) as fin:
        hm = json.load(fin)
    assert isinstance(hm, dict) and len(hm) > 0
    assert "hm_e2e_nlg" in hm
    entry = hm["hm_e2e_nlg"]
    assert "hint_map_id" in entry and "hint_map" in entry
