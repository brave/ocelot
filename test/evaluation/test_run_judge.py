"""Tests for evaluation.run_judge._responses_from_record."""

from __future__ import annotations

import pytest

from evaluation.run_judge import _responses_from_record


def test_responses_from_dict_three_keys() -> None:
    r = _responses_from_record({"A": "one", "B": "two", "C": "three"})
    assert r == [("A", "one"), ("B", "two"), ("C", "three")]


def test_responses_from_dict_wrong_count_raises() -> None:
    with pytest.raises(ValueError, match="exactly 3 keys"):
        _responses_from_record({"A": "1", "B": "2"})


def test_responses_from_list_three_objects() -> None:
    r = _responses_from_record(
        [
            {"name": "m1", "text": "a"},
            {"model": "m2", "response": "b"},
            {"model_name": "m3", "content": "c"},
        ]
    )
    assert r == [("m1", "a"), ("m2", "b"), ("m3", "c")]


def test_responses_from_list_bad_length_raises() -> None:
    with pytest.raises(ValueError, match="length 3"):
        _responses_from_record([{"name": "a", "text": "1"}])


def test_responses_from_list_missing_fields_raises() -> None:
    with pytest.raises(ValueError, match="needs"):
        _responses_from_record(
            [
                {"name": "a", "text": "1"},
                {"name": "b", "text": "2"},
                {"name": "c"},
            ]
        )


def test_responses_invalid_type_raises() -> None:
    with pytest.raises(ValueError, match="object or array"):
        _responses_from_record("nope")
