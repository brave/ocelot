"""End-to-end SFT then IPO smoke test (opt-in: OCELOT_TRAINING_E2E=1).

Runs one SFT stage, loads the saved PEFT adapter, then runs one IPO (CPO) stage on the same data.

Needs an accelerator: NVIDIA CUDA or Apple Silicon MPS (macOS).

Install: `src/training/requirements-macos.txt` on Mac, or `requirements-linux-cuda.txt` on Linux+GPU.
On Linux+CUDA, optional `flash-attn`: `pip install flash-attn --no-build-isolation` (see `src/training/README.md`).
On Apple Silicon, 4-bit loading is disabled automatically (use full bf16 weights; more RAM).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _e2e_enabled() -> bool:
    return os.environ.get("OCELOT_TRAINING_E2E", "").strip().lower() in {"1", "true", "yes"}


def _find_peft_adapter_dir(sft_dir: Path) -> Path:
    """HF Trainer + PEFT usually writes `checkpoint-*` with `adapter_config.json` under the SFT output dir."""
    ckpts = sorted(sft_dir.glob("checkpoint-*"), key=lambda p: p.name)
    for c in reversed(ckpts):
        if (c / "adapter_config.json").is_file():
            return c
    if (sft_dir / "adapter_config.json").is_file():
        return sft_dir
    names = [p.name for p in sft_dir.iterdir()] if sft_dir.is_dir() else []
    raise AssertionError(
        f"Expected PEFT adapter (adapter_config.json) under {sft_dir}, found entries: {names}"
    )


def test_sft_one_optimizer_step_e2e(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiny_sft_json: Path) -> None:
    if not _e2e_enabled():
        pytest.skip("Set OCELOT_TRAINING_E2E=1 to run GPU end-to-end training (downloads model).")

    torch = pytest.importorskip("torch")
    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if not has_cuda and not has_mps:
        pytest.skip("CUDA or Apple MPS is required for this integration test.")

    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("datasets")
    pytest.importorskip("qwen_vl_utils")
    pytest.importorskip("trl")

    if has_mps:
        monkeypatch.setenv("OCELOT_LOAD_IN_4BIT", "0")

    load_4bit = os.environ.get("OCELOT_LOAD_IN_4BIT", "1").strip().lower() in {"1", "true", "yes"}
    if load_4bit:
        pytest.importorskip("bitsandbytes")

    model_name = os.environ.get("OCELOT_E2E_MODEL_NAME", "Qwen/Qwen3-VL-2B-Instruct").strip()

    # Speed / determinism for a single step
    monkeypatch.setenv("OCELOT_ATTN_IMPLEMENTATION", "sdpa")
    monkeypatch.setenv("GRADIENT_ACCUMULATION_STEPS", "1")
    monkeypatch.setenv("PER_DEVICE_TRAIN_BATCH_SIZE", "1")
    monkeypatch.setenv("PER_DEVICE_EVAL_BATCH_SIZE", "1")
    monkeypatch.setenv("GRADIENT_CHECKPOINTING", "0")
    monkeypatch.setenv("EVAL_STEPS", "0")
    monkeypatch.setenv("TRAIN_IMAGE_DROP_RATIO", "0")
    monkeypatch.setenv("TOKENIZE_BATCH_SIZE", "1")
    monkeypatch.setenv("DATA_SHUFFLE_SEED", "42")

    from core.config import RunConfig
    from methods.ipo import IPOMethod
    from methods.sft import SFTMethod

    out_root = tmp_path / "run"
    shared = dict(
        model_name=model_name,
        data_path=str(tiny_sft_json),
        output_dir=str(out_root),
        deepspeed=None,
        prepared_data_dir=None,
        sft_max_length=1024,
        sft_max_prompt_length=512,
        tokenize_batch_size=1,
        store_vision_dtype="float16",
        vision_max_pixels=None,
        lora_rank=8,
        sft_warmup_epochs=0,
        sft_learning_rate=3e-4,
        pref_learning_rate=5e-6,
        ipo_beta=0.1,
        dpo_beta=0.1,
        enable_qat=False,
    )
    cfg_sft = RunConfig(trainer="sft", epochs=1, resume_from=None, **shared)

    SFTMethod().run(cfg_sft)

    sft_dir = out_root / "sft"
    assert sft_dir.is_dir()
    artifacts = list(sft_dir.rglob("*"))
    assert artifacts, "expected trainer to write at least one file under output_dir/sft"
    assert any(
        p.name == "adapter_config.json" or p.name.startswith("checkpoint-") for p in artifacts if p.is_file()
    ), f"expected PEFT adapter or checkpoint under {sft_dir}, got: {[p.name for p in artifacts if p.is_file()][:20]}"

    adapter_dir = _find_peft_adapter_dir(sft_dir)
    cfg_ipo = RunConfig(trainer="ipo", epochs=1, resume_from=str(adapter_dir), **shared)
    IPOMethod().run(cfg_ipo)

    ipo_dir = out_root / "ipo"
    assert ipo_dir.is_dir()
    assert (out_root / "adapter-ocelot" / "adapter_config.json").is_file(), (
        f"expected IPO stage to write PEFT adapter to {out_root / 'adapter-ocelot'}"
    )
