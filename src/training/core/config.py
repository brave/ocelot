from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class RunConfig:
    # Required user-facing knobs
    trainer: str
    epochs: int

    # Common paths
    model_name: str
    data_path: str
    output_dir: str
    resume_from: str | None  # PEFT adapter checkpoint (e.g. SFT warmup) to load before training
    deepspeed: str | None
    prepared_data_dir: str | None  # Stage 2: load from Parquet (Stage 1 = prepare_data.py)

    # Tokenization / truncation
    sft_max_length: int
    sft_max_prompt_length: int
    tokenize_batch_size: int
    store_vision_dtype: str

    # Vision controls
    vision_max_pixels: int | None

    # LoRA
    lora_rank: int

    # Optional warmup (used by IPO by default)
    sft_warmup_epochs: int

    # Learning rates
    sft_learning_rate: float
    pref_learning_rate: float

    # Betas
    ipo_beta: float
    dpo_beta: float

    # QAT
    enable_qat: bool = False
    qat_target_bits: int = 4
    qat_warmup_steps: int = 120

    @staticmethod
    def _default_deepspeed_path() -> str | None:
        here = Path(__file__).resolve().parents[1]  # src/training
        cand = here / "ds_zero2.json"
        return str(cand) if cand.exists() else None

    @classmethod
    def from_argv(cls, argv: list[str] | None = None) -> "RunConfig":
        p = argparse.ArgumentParser(description="Modular trainer (SFT / IPO / DPO) for Qwen3-VL style data.")
        p.add_argument("--trainer", choices=["sft", "ipo", "dpo"], required=True)
        p.add_argument("--epochs", type=int, required=True, help="Epochs for the selected trainer.")

        p.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-4B-Instruct"))
        p.add_argument(
            "--data-path",
            default=os.environ.get("DATA_PATH", ""),
            required=not (os.environ.get("PREPARED_DATA_DIR") or os.environ.get("DATA_PATH")),
            help="Required unless PREPARED_DATA_DIR is set (Stage 2).",
        )
        p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "./runs/qwen-training"))
        p.add_argument("--resume-from", default=os.environ.get("RESUME_FROM", ""), help="Load PEFT adapter from this path (e.g. SFT checkpoint) before training.")
        p.add_argument("--prepared-data-dir", default=os.environ.get("PREPARED_DATA_DIR", ""), help="Stage 2: load from Parquet (from prepare_data.py)")
        p.add_argument("--deepspeed", default=os.environ.get("DEEPSPEED", cls._default_deepspeed_path()))

        p.add_argument("--sft-max-length", type=int, default=int(os.environ.get("SFT_MAX_LENGTH", "10240")))
        p.add_argument("--sft-max-prompt-length", type=int, default=int(os.environ.get("SFT_MAX_PROMPT_LENGTH", "8192")))
        p.add_argument("--tokenize-batch-size", type=int, default=int(os.environ.get("TOKENIZE_BATCH_SIZE", "32")))
        p.add_argument("--store-vision-dtype", default=os.environ.get("STORE_VISION_DTYPE", "float16"))

        # If <= 0, disable. (Matches train_script behavior.)
        vmp = int(os.environ.get("VISION_MAX_PIXELS", "262144"))
        p.add_argument("--vision-max-pixels", type=int, default=vmp)

        p.add_argument("--lora-rank", type=int, default=int(os.environ.get("LORA_RANK", "64")))

        p.add_argument("--sft-warmup-epochs", type=int, default=int(os.environ.get("SFT_EPOCHS", "0")))

        # Defaults match `train_script.py`:
        # - SFT stage: 1e-5 * 3
        # - preference stage (IPO/DPO): 1e-5 / 2
        p.add_argument(
            "--sft-learning-rate",
            type=float,
            default=float(os.environ.get("SFT_LEARNING_RATE", "3e-5")),
        )
        p.add_argument(
            "--pref-learning-rate",
            type=float,
            default=float(os.environ.get("PREF_LEARNING_RATE", "5e-6")),
        )

        p.add_argument("--ipo-beta", type=float, default=float(os.environ.get("IPO_BETA", "0.1")))
        p.add_argument("--dpo-beta", type=float, default=float(os.environ.get("DPO_BETA", "0.1")))

        p.add_argument("--enable-qat", action=argparse.BooleanOptionalAction, default=_truthy_env("ENABLE_QAT", "0"))
        p.add_argument("--qat-target-bits", type=int, default=int(os.environ.get("QAT_TARGET_BITS", "4")))
        p.add_argument("--qat-warmup-steps", type=int, default=int(os.environ.get("QAT_WARMUP_STEPS", "120")))

        ns = p.parse_args(argv)

        sft_max_length = int(ns.sft_max_length)
        sft_max_prompt_length = min(int(ns.sft_max_prompt_length), sft_max_length)

        vision_max_pixels = None if int(ns.vision_max_pixels) <= 0 else int(ns.vision_max_pixels)

        prepared = (ns.prepared_data_dir or "").strip() or None
        resume_from = (ns.resume_from or "").strip() or None
        return cls(
            trainer=str(ns.trainer),
            epochs=int(ns.epochs),
            model_name=str(ns.model_name),
            data_path=str(ns.data_path),
            output_dir=str(ns.output_dir),
            resume_from=resume_from,
            deepspeed=str(ns.deepspeed) if ns.deepspeed else None,
            prepared_data_dir=prepared,
            sft_max_length=sft_max_length,
            sft_max_prompt_length=sft_max_prompt_length,
            tokenize_batch_size=int(ns.tokenize_batch_size),
            store_vision_dtype=str(ns.store_vision_dtype),
            vision_max_pixels=vision_max_pixels,
            lora_rank=int(ns.lora_rank),
            sft_warmup_epochs=int(ns.sft_warmup_epochs),
            sft_learning_rate=float(ns.sft_learning_rate),
            pref_learning_rate=float(ns.pref_learning_rate),
            ipo_beta=float(ns.ipo_beta),
            dpo_beta=float(ns.dpo_beta),
            enable_qat=bool(ns.enable_qat),
            qat_target_bits=int(ns.qat_target_bits),
            qat_warmup_steps=int(ns.qat_warmup_steps),
        )


