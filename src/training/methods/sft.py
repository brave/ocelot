from __future__ import annotations

import os
import gc

import torch
from transformers import Trainer

from callbacks.common import PeriodicEvalCallback, enable_gradient_checkpointing_for_lora, resolve_periodic_eval_steps
from core.config import RunConfig
from core.hf_args import build_sft_training_args
from data.collators import TokenizedSFTCollator
from data.pipeline import load_and_prepare_datasets
from methods.base import TrainingMethod
from modeling.factory import build_model_and_processor


class VisionSFTTrainer(Trainer):
    """
    Plain HF Trainer for SFT, with a vision-safe collator + DeepSpeed-safe scalar loss.
    """

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if isinstance(loss, tuple):
            loss = loss[0]
        if loss.dim() != 0:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)
        return loss.detach()

def run_sft_stage(
    cfg: RunConfig,
    *,
    model,
    processor,
    callbacks: list[object],
    train_dataset,
    eval_dataset,
    epochs: int,
    output_dir: str,
    learning_rate: float,
) -> None:
    collator = TokenizedSFTCollator(vision_dtype=cfg.store_vision_dtype)
    args = build_sft_training_args(cfg, output_dir=output_dir, epochs=epochs, learning_rate=learning_rate)

    trainer = VisionSFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor.tokenizer,
        callbacks=list(callbacks),
    )
    trainer.add_callback(PeriodicEvalCallback(trainer, eval_steps=resolve_periodic_eval_steps(trainer)))
    if getattr(args, "gradient_checkpointing", False):
        enable_gradient_checkpointing_for_lora(model)
    trainer.train()

    # Cleanup mirrors the original script's intent when chaining.
    try:
        trainer.accelerator.free_memory()
    except Exception:
        pass
    try:
        model.zero_grad(set_to_none=True)
    except Exception:
        for p in model.parameters():
            p.grad = None
    del trainer
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


class SFTMethod(TrainingMethod):
    name = "sft"

    def run(self, cfg: RunConfig) -> None:
        bundle = build_model_and_processor(cfg)
        train_ds, val_ds = load_and_prepare_datasets(cfg, processor=bundle.processor)

        out = os.path.join(cfg.output_dir, "sft")
        run_sft_stage(
            cfg,
            model=bundle.model,
            processor=bundle.processor,
            callbacks=bundle.callbacks,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            epochs=cfg.epochs,
            output_dir=out,
            learning_rate=float(cfg.sft_learning_rate),
        )


