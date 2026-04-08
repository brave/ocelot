"""Tests for evaluation.judge.judge_prompts.build_comparison_messages."""

from __future__ import annotations

import pytest

from evaluation.judge.judge_prompts import build_comparison_messages


def test_build_comparison_messages_requires_three_responses() -> None:
    with pytest.raises(ValueError, match="expected 3"):
        build_comparison_messages(
            [{"role": "user", "content": "hi"}],
            [("A", "a"), ("B", "b")],
        )


def test_build_comparison_messages_structure() -> None:
    messages = [{"role": "user", "content": "Hello"}]
    responses = [("ModelA", "out a"), ("ModelB", "out b"), ("ModelC", "out c")]
    out = build_comparison_messages(messages, responses)
    assert len(out) == 1
    assert out[0]["role"] == "user"
    content = out[0]["content"]
    assert isinstance(content, list)
    texts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
    joined = "\n".join(texts)
    assert "Hello" in joined
    assert "ModelA Response: <response>" in joined
    assert "out a" in joined
    assert "score_a" in joined


def test_build_comparison_messages_string_content() -> None:
    out = build_comparison_messages(
        [{"role": "user", "content": "plain string"}],
        [("A", "a"), ("B", "b"), ("C", "c")],
    )
    texts = [
        p.get("text", "")
        for p in out[0]["content"]
        if isinstance(p, dict) and p.get("type") == "text"
    ]
    assert any("plain string" in t for t in texts)
