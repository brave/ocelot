from __future__ import annotations

import os

from transformers import TrainingArguments

from core.config import RunConfig


def _grad_accum_steps() -> int:
    return max(1, int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "8")))


def _gradient_checkpointing() -> bool:
    return os.environ.get("GRADIENT_CHECKPOINTING", "1").strip().lower() in {"1", "true", "yes"}


def _per_device_batch_size() -> int:
    return max(1, int(os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "1")))


def _per_device_eval_batch_size() -> int:
    return max(1, int(os.environ.get("PER_DEVICE_EVAL_BATCH_SIZE", os.environ.get("PER_DEVICE_TRAIN_BATCH_SIZE", "1"))))


def _save_steps() -> int:
    return max(1, int(os.environ.get("SAVE_STEPS", "50")))


def _dataloader_pin_memory() -> bool:
    """Pinned memory helps CUDA; MPS/CPU emit a noisy warning if this stays True without CUDA."""
    import torch

    return torch.cuda.is_available()


def build_base_training_args(cfg: RunConfig, *, output_dir: str, epochs: int, learning_rate: float) -> TrainingArguments:
    """
    Shared TrainingArguments defaults used by preference-style trainers (DPO/IPO) in `train_script.py`.
    """
    log_dir = os.path.join(output_dir, "logs")
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", log_dir)
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=_per_device_batch_size(),
        per_device_eval_batch_size=_per_device_eval_batch_size(),
        gradient_accumulation_steps=_grad_accum_steps(),
        learning_rate=float(learning_rate),
        num_train_epochs=int(epochs),
        bf16=True,
        save_strategy="steps",
        save_steps=_save_steps(),
        eval_strategy="steps",
        logging_steps=2,
        eval_steps=50,
        remove_unused_columns=False,
        report_to="none",
        deepspeed=cfg.deepspeed,
        gradient_checkpointing=_gradient_checkpointing(),
        logging_first_step=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=0.5,
        eval_accumulation_steps=1,
        eval_do_concat_batches=False,
        dataloader_pin_memory=_dataloader_pin_memory(),
    )


def build_sft_training_args(cfg: RunConfig, *, output_dir: str, epochs: int, learning_rate: float) -> TrainingArguments:
    """
    Shared TrainingArguments defaults used by the SFT warmup stage in `train_script.py`.
    """
    log_dir = os.path.join(output_dir, "logs")
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", log_dir)
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=_per_device_batch_size(),
        per_device_eval_batch_size=_per_device_eval_batch_size(),
        gradient_accumulation_steps=_grad_accum_steps(),
        learning_rate=float(learning_rate),
        num_train_epochs=int(epochs),
        bf16=True,
        logging_steps=2,
        logging_strategy="steps",
        save_strategy="epoch",
        eval_steps=50,
        remove_unused_columns=False,
        report_to="none",
        deepspeed=cfg.deepspeed,
        gradient_checkpointing=_gradient_checkpointing(),
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=0.5,
        dataloader_pin_memory=_dataloader_pin_memory(),
    )


