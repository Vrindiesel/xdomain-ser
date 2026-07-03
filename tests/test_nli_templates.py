# Copyright 2026 Davan Harrison and Marilyn Walker.
# Licensed under the Apache License, Version 2.0.
"""Tests for the NLI slot-template converter (``xdomain_ser.nli.templates``).

Regression guards for the ``(slot, value) -> hypothesis`` mapping used by the
NLI baseline: template count/coverage, structural invariants, and the behaviour
of ``slot_value_to_template`` for string templates, boolean/enum handlers,
dontcare values, and the unknown-slot fallback. Pure-Python -- no model load.
"""
from xdomain_ser.nli.templates import (
    SLOT_TEMPLATES,
    slot_value_to_template,
    get_all_registered_slots,
    _make_readable_name,
)


def test_template_count():
    # 138 registered slot templates across E2E / ViGGO / RNNLG / Taskmaster.
    # (The module docstring's "133" predates the 5 E2E-native aliases; see the
    # Stage-5 RELEASE_NOTES entry.) Update deliberately if templates are added.
    assert len(SLOT_TEMPLATES) == 138


def test_templates_are_str_or_callable():
    for slot, tmpl in SLOT_TEMPLATES.items():
        assert isinstance(tmpl, str) or callable(tmpl), f"{slot}: {type(tmpl)}"


def test_str_templates_have_value_placeholder():
    for slot, tmpl in SLOT_TEMPLATES.items():
        if isinstance(tmpl, str):
            assert "{value}" in tmpl, f"{slot} missing {{value}}: {tmpl!r}"


def test_str_template_formats():
    assert slot_value_to_template("name", "The Vaults") == "The name is The Vaults."


def test_all_str_templates_format_into_nonempty_hypothesis():
    # Every string template fills its {value} placeholder into a non-empty
    # hypothesis that contains the value. Catches a malformed/empty template.
    for slot, tmpl in SLOT_TEMPLATES.items():
        if isinstance(tmpl, str):
            out = slot_value_to_template(slot, "Xyz")
            assert isinstance(out, str) and out and "Xyz" in out, f"{slot}: {out!r}"


def test_all_callable_templates_return_str_or_none():
    # Boolean / enum handlers must never raise and must return str-or-None
    # across a range of representative values (some, e.g. restaurant_kidsallowed,
    # intentionally return None for values they do not recognise).
    for slot, tmpl in SLOT_TEMPLATES.items():
        if callable(tmpl):
            for v in ("yes", "no", "dontcare", "somevalue"):
                out = slot_value_to_template(slot, v)
                assert out is None or isinstance(out, str), f"{slot}({v!r}): {out!r}"


def test_boolean_template_positive_differs_from_negative():
    pos = slot_value_to_template("family_suitability", "family-friendly")
    neg = slot_value_to_template("family_suitability", "not-family-friendly")
    assert isinstance(pos, str) and isinstance(neg, str)
    assert pos != neg


def test_dontcare_returns_none():
    assert slot_value_to_template("name", "dontcare") is None
    assert slot_value_to_template("name", "dont_care") is None


def test_unknown_slot_uses_readable_fallback():
    assert slot_value_to_template("some_unknown_slot", "x") == "The some unknown slot is x."


def test_make_readable_name_strips_domain_prefixes():
    assert _make_readable_name("auto_repair.date") == "date"
    assert _make_readable_name("hotel_area") == "area"
    assert _make_readable_name("has_internet") == "has internet"


def test_get_all_registered_slots():
    slots = get_all_registered_slots()
    assert isinstance(slots, set)
    assert len(slots) == len(SLOT_TEMPLATES)
    assert "name" in slots
