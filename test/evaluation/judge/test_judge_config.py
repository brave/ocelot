"""Tests for evaluation.judge.judge_config.JudgeConfig."""

from __future__ import annotations

import pytest

from evaluation.judge.judge_config import JudgeConfig


def test_litellm_model_bedrock_prefix_preserved() -> None:
    cfg = JudgeConfig(provider="bedrock", model="bedrock/converse/foo.bar")
    assert cfg.litellm_model() == "bedrock/converse/foo.bar"


def test_litellm_model_bedrock_adds_converse_prefix() -> None:
    cfg = JudgeConfig(provider="bedrock", model="anthropic.claude-3-5-sonnet")
    assert cfg.litellm_model() == "bedrock/converse/anthropic.claude-3-5-sonnet"


def test_litellm_model_openai_known_prefix_unchanged() -> None:
    cfg = JudgeConfig(provider="openai", model="azure/gpt-4")
    assert cfg.litellm_model() == "azure/gpt-4"


def test_litellm_model_openai_prepends_openai() -> None:
    cfg = JudgeConfig(provider="openai", model="my-served-model")
    assert cfg.litellm_model() == "openai/my-served-model"


def test_litellm_model_empty_raises() -> None:
    cfg = JudgeConfig(provider="bedrock", model="")
    with pytest.raises(ValueError, match="model is required"):
        cfg.litellm_model()


def test_completion_kwargs_bedrock_region() -> None:
    cfg = JudgeConfig(provider="bedrock", model="m", region="eu-west-1")
    assert cfg.completion_kwargs() == {"aws_region_name": "eu-west-1"}


def test_completion_kwargs_openai_base_and_key() -> None:
    cfg = JudgeConfig(
        provider="openai",
        model="m",
        api_base="http://127.0.0.1:8000/v1/",
        api_key="sk-test",
        temperature=0.2,
    )
    kw = cfg.completion_kwargs()
    assert kw["api_base"] == "http://127.0.0.1:8000/v1"
    assert kw["api_key"] == "sk-test"
    assert kw["temperature"] == 0.2


def test_completion_kwargs_openai_strips_empty_base() -> None:
    cfg = JudgeConfig(provider="openai", model="m", api_base="   ")
    assert "api_base" not in cfg.completion_kwargs()
