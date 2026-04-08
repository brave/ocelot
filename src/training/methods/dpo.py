from __future__ import annotations

import contextlib
import os

from trl import DPOConfig, DPOTrainer

from core.config import RunConfig
from core.hf_args import build_base_training_args
from data.collators import TokenizedDPOCollator
from data.pipeline import load_and_prepare_datasets
from methods.base import TrainingMethod
from methods.sft import run_sft_stage
from methods.trl_vision_safe import REQUIRED_TOKEN_COLS, process_row_with_vision
from modeling.factory import build_model_and_processor


class VisionSafeDPOTrainer(DPOTrainer):
    def null_ref_context(self):
        # Keep LoRA ON for the "ref" pass (mirrors train_script.py override).
        return contextlib.nullcontext()

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        # If TRL is precomputing ref log-probs, let it run its normal pipeline.
        if getattr(args, "precompute_ref_log_probs", False):
            return super()._prepare_dataset(dataset, processing_class, args, dataset_name)

        cols = set(getattr(dataset, "column_names", []) or [])
        if REQUIRED_TOKEN_COLS.issubset(cols):
            return dataset
        return super()._prepare_dataset(dataset, processing_class, args, dataset_name)

    def process_row(self, features, **kwargs):
        return process_row_with_vision(features)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Optional length bonus: export DPO_LENGTH_BONUS=0.001
        out = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        if return_outputs:
            loss, outputs = out
        else:
            loss, outputs = out, None

        w_raw = os.environ.get("DPO_LENGTH_BONUS", "0.001").strip()
        w = float(w_raw) if w_raw else 0.0
        if w != 0.0:
            try:
                prompt_len = inputs["prompt_attention_mask"].sum(dim=1)
                chosen_len = inputs["chosen_attention_mask"].sum(dim=1)
                rejected_len = inputs["rejected_attention_mask"].sum(dim=1)
                chosen_comp = (chosen_len - prompt_len).clamp(min=0)
                rejected_comp = (rejected_len - prompt_len).clamp(min=0)
                bonus = (chosen_comp - rejected_comp).float().mean()
                loss = loss - (loss.new_tensor(w) * bonus)
            except Exception:
                pass

        if return_outputs:
            return loss, outputs
        return loss

class DPOMethod(TrainingMethod):
    name = "dpo"

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
            output_dir=os.path.join(cfg.output_dir, "dpo"),
            epochs=cfg.epochs,
            learning_rate=float(cfg.pref_learning_rate),
        )

        dpo_config = DPOConfig(
            beta=float(cfg.dpo_beta),
            max_length=int(cfg.sft_max_length),
            max_prompt_length=int(cfg.sft_max_prompt_length),
            precompute_ref_log_probs=True,
            precompute_ref_batch_size=1,
            **args.to_dict(),
        )

        # --- DPO memory knobs (copied from train_script.py) ---
        # 1) concatenated_forward toggle
        _concat_env = os.environ.get("DPO_CONCATENATED_FORWARD", "0").strip().lower()
        _want_concat = _concat_env in {"1", "true", "yes"}
        if hasattr(dpo_config, "concatenated_forward"):
            setattr(dpo_config, "concatenated_forward", _want_concat)

        # 2) precompute_ref_log_probs override (only if env var is set)
        _precomp_env = os.environ.get("DPO_PRECOMPUTE_REF_LOG_PROBS", "").strip().lower()
        if _precomp_env:
            _want_precomp = _precomp_env in {"1", "true", "yes"}
            if hasattr(dpo_config, "precompute_ref_log_probs"):
                setattr(dpo_config, "precompute_ref_log_probs", _want_precomp)

        # 3) reference_free override (only if env var is set)
        _ref_free_env = os.environ.get("DPO_REFERENCE_FREE", "").strip().lower()
        if _ref_free_env:
            _want_ref_free = _ref_free_env in {"1", "true", "yes"}
            if hasattr(dpo_config, "reference_free"):
                setattr(dpo_config, "reference_free", _want_ref_free)

        trainer = VisionSafeDPOTrainer(
            model=bundle.model,
            args=dpo_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=bundle.processor,
            data_collator=collator,
            callbacks=list(bundle.callbacks),
        )
        trainer.train()
        bundle.model.save_pretrained(os.path.join(cfg.output_dir, "adapter-ocelot"))


