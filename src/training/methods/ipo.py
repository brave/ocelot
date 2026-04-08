"""
IPO stage: mirrors train_script.py exactly — VisionSafeCPOTrainer(CPOTrainer), same collator, same config pattern.
"""
from __future__ import annotations

import os

import torch
from  trl.experimental.cpo import CPOConfig, CPOTrainer

from callbacks.common import PeriodicEvalCallback, resolve_periodic_eval_steps
from core.config import RunConfig
from core.hf_args import build_base_training_args, _save_steps
from data.collators import TokenizedDPOCollator
from data.pipeline import load_and_prepare_datasets
from methods.base import TrainingMethod
from methods.sft import run_sft_stage
from methods.trl_vision_safe import prompt_vision_tensors_from_features
from modeling.factory import build_model_and_processor


# Same required columns as train_script.py VisionSafeCPOTrainer
_REQUIRED_COLS = {
    "prompt_input_ids",
    "prompt_attention_mask",
    "chosen_input_ids",
    "chosen_attention_mask",
    "rejected_input_ids",
    "rejected_attention_mask",
}


class VisionSafeCPOTrainer(CPOTrainer):
    """Replica of train_script.py VisionSafeCPOTrainer(CPOTrainer)."""

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        cols = set(getattr(dataset, "column_names", []) or [])
        if _REQUIRED_COLS.issubset(cols):
            return dataset
        return super()._prepare_dataset(dataset, processing_class, args, dataset_name)

    def tokenize_row(self, features, **kwargs):
        if "prompt_input_ids" in features:
            return self.process_row(features, **kwargs)
        return super().tokenize_row(features, **kwargs)

    def process_row(self, features, **kwargs):
        pv_t, grid_t = prompt_vision_tensors_from_features(features)
        return {
            "prompt_input_ids": torch.as_tensor(features["prompt_input_ids"], dtype=torch.int64),
            "prompt_attention_mask": torch.as_tensor(features["prompt_attention_mask"], dtype=torch.int64),
            "prompt_mm_token_type_ids": torch.as_tensor(
                features.get("prompt_mm_token_type_ids", [0] * len(features["prompt_input_ids"])), dtype=torch.int64
            ),
            "prompt_pixel_values": pv_t,
            "prompt_image_grid_thw": grid_t,
            "chosen_input_ids": torch.as_tensor(features["chosen_input_ids"], dtype=torch.int64),
            "chosen_attention_mask": torch.as_tensor(features["chosen_attention_mask"], dtype=torch.int64),
            "chosen_mm_token_type_ids": torch.as_tensor(
                features.get("chosen_mm_token_type_ids", [0] * len(features["chosen_input_ids"])), dtype=torch.int64
            ),
            "rejected_input_ids": torch.as_tensor(features["rejected_input_ids"], dtype=torch.int64),
            "rejected_attention_mask": torch.as_tensor(features["rejected_attention_mask"], dtype=torch.int64),
            "rejected_mm_token_type_ids": torch.as_tensor(
                features.get("rejected_mm_token_type_ids", [0] * len(features["rejected_input_ids"])), dtype=torch.int64
            ),
        }


def _build_cpo_config(cfg: RunConfig, args, processor) -> CPOConfig:
    """Build CPOConfig like train_script.py: base args + IPO params, with fallbacks for TRL API changes."""
    save_steps = _save_steps()
    try:
        cpo_config = CPOConfig(
            **args.to_dict(),
            loss_type="ipo",
            padding_value=processor.tokenizer.pad_token_id,
            cpo_alpha=5,
        )
    except TypeError:
        cpo_config = CPOConfig(**args.to_dict(), loss_type="ipo")
    if hasattr(cpo_config, "beta"):
        setattr(cpo_config, "beta", float(cfg.ipo_beta))
    if hasattr(cpo_config, "loss_type"):
        setattr(cpo_config, "loss_type", "ipo")
    if hasattr(cpo_config, "max_length"):
        setattr(cpo_config, "max_length", int(cfg.sft_max_length))
    if hasattr(cpo_config, "max_prompt_length"):
        setattr(cpo_config, "max_prompt_length", int(cfg.sft_max_prompt_length))
    cpo_config.save_strategy = "steps"
    cpo_config.save_steps = save_steps
    return cpo_config


class IPOMethod(TrainingMethod):
    name = "ipo"

    def run(self, cfg: RunConfig) -> None:
        bundle = build_model_and_processor(cfg)
        train_ds, val_ds = load_and_prepare_datasets(cfg, processor=bundle.processor)

        if int(cfg.sft_warmup_epochs) > 0:
            run_sft_stage(
                cfg,
                model=bundle.model,
                processor=bundle.processor,
                callbacks=bundle.callbacks,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                epochs=int(cfg.sft_warmup_epochs),
                output_dir=os.path.join(cfg.output_dir, "sft"),
                learning_rate=float(cfg.sft_learning_rate),
            )

        collator = TokenizedDPOCollator(vision_dtype=cfg.store_vision_dtype)
        args = build_base_training_args(
            cfg,
            output_dir=os.path.join(cfg.output_dir, "ipo"),
            epochs=cfg.epochs,
            learning_rate=float(cfg.pref_learning_rate),
        )

        cpo_config = _build_cpo_config(cfg, args, bundle.processor)

        trainer = VisionSafeCPOTrainer(
            model=bundle.model,
            args=cpo_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=bundle.processor.tokenizer,
            data_collator=collator,
            callbacks=list(bundle.callbacks),
        )
        trainer.add_callback(PeriodicEvalCallback(trainer, eval_steps=resolve_periodic_eval_steps(trainer)))
        trainer.train()
        bundle.model.save_pretrained(os.path.join(cfg.output_dir, "adapter-ocelot"))
