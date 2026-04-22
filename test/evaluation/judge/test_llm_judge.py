"""Tests for judge output parsing (no LiteLLM calls)."""

from __future__ import annotations

import pytest

from evaluation.judge.llm_judge import _extract_json_object, parse_judge_output


def test_extract_json_object_first_balanced_object() -> None:
    text = 'noise {"a": 1} trailing'
    assert _extract_json_object(text) == {"a": 1}


def test_extract_json_object_nested() -> None:
    text = 'x {"outer": {"inner": 2}} y'
    assert _extract_json_object(text) == {"outer": {"inner": 2}}


def test_extract_json_object_invalid_inner_returns_none() -> None:
    text = '{"broken": }'
    assert _extract_json_object(text) is None


def test_parse_judge_output_from_json_comparative_analysis() -> None:
    raw = """Here is JSON: {"score_a": 4, "score_b": 3.5, "score_c": 2, "comparative_analysis": "B wins"}"""
    (sa, sb, sc), reasoning = parse_judge_output(raw)
    assert sa == 4.0
    assert sb == 3.5
    assert sc == 2.0
    assert reasoning == "B wins"


def test_parse_judge_output_prefers_reasoning_key() -> None:
    raw = '{"score_a": 1, "score_b": 2, "score_c": 3, "reasoning": "ok"}'
    (_, _, _), reasoning = parse_judge_output(raw)
    assert reasoning == "ok"


def test_parse_judge_output_regex_fallback() -> None:
    raw = 'score_a: 1 score_b: 2 score_c: 3'
    (sa, sb, sc), reasoning = parse_judge_output(raw)
    assert sa == 1.0
    assert sb == 2.0
    assert sc == 3.0
    assert raw in reasoning or reasoning == raw


def test_parse_judge_output_raises_when_unparseable() -> None:
    with pytest.raises(ValueError, match="Could not parse judge scores"):
        parse_judge_output("no scores here")
