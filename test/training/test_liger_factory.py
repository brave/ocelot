from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TRAINING = Path(__file__).resolve().parents[2] / "src" / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from core.config import RunConfig
from modeling.liger_kernels import liger_fused_ce_active, liger_model_kind, use_liger_active


def test_liger_model_kind_qwen3_vl() -> None:
    assert liger_model_kind("Qwen/Qwen3-VL-4B-Instruct") == "qwen3_vl"


def test_liger_model_kind_qwen3_vl_moe() -> None:
    assert liger_model_kind("Qwen/Qwen3-VL-MoE-30B-A3B-Instruct") == "qwen3_vl_moe"


def test_liger_model_kind_unsupported() -> None:
    assert liger_model_kind("Qwen/Qwen2-VL-2B-Instruct") is None


def test_use_liger_active_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCELOT_USE_LIGER", raising=False)
    assert use_liger_active() is False
    monkeypatch.setenv("OCELOT_USE_LIGER", "1")
    assert use_liger_active() is True


def test_liger_fused_ce_active_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCELOT_USE_LIGER", raising=False)
    monkeypatch.delenv("OCELOT_LIGER_CROSS_ENTROPY", raising=False)
    assert liger_fused_ce_active() is False

    monkeypatch.setenv("OCELOT_USE_LIGER", "1")
    assert liger_fused_ce_active() is True

    monkeypatch.setenv("OCELOT_LIGER_CROSS_ENTROPY", "1")
    assert liger_fused_ce_active() is False


def test_liger_fused_ce_active_from_run_config() -> None:
    cfg = RunConfig(
        trainer="sft",
        epochs=1,
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        data_path="",
        output_dir=".",
        resume_from=None,
        deepspeed=None,
        prepared_data_dir=None,
        sft_max_length=1024,
        sft_max_prompt_length=512,
        tokenize_batch_size=1,
        store_vision_dtype="float16",
        vision_max_pixels=None,
        lora_rank=8,
        sft_warmup_epochs=0,
        sft_learning_rate=3e-5,
        pref_learning_rate=5e-6,
        ipo_beta=0.1,
        dpo_beta=0.1,
    )
    assert liger_fused_ce_active(cfg) is False

    cfg_on = RunConfig(**{**cfg.__dict__, "use_liger": True})
    assert liger_fused_ce_active(cfg_on) is True

    cfg_ce = RunConfig(**{**cfg.__dict__, "use_liger": True, "liger_cross_entropy": True})
    assert liger_fused_ce_active(cfg_ce) is False


def test_run_config_use_liger_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCELOT_USE_LIGER", "1")
    monkeypatch.setenv("OCELOT_LIGER_CROSS_ENTROPY", "0")
    cfg = RunConfig.from_argv(["--trainer", "sft", "--epochs", "1", "--data-path", "/tmp/data.json"])
    assert cfg.use_liger is True
    assert cfg.liger_cross_entropy is False


def test_run_config_use_liger_cli_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCELOT_USE_LIGER", "1")
    cfg = RunConfig.from_argv(
        ["--trainer", "sft", "--epochs", "1", "--data-path", "/tmp/data.json", "--no-use-liger"]
    )
    assert cfg.use_liger is False
