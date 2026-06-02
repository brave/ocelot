from __future__ import annotations

import os

import torch
import torch.nn as nn
from trl.experimental.cpo import CPOConfig, CPOTrainer

from callbacks.common import PeriodicEvalCallback, resolve_periodic_eval_steps
from core.config import RunConfig
from core.hf_args import build_base_training_args, _save_steps
from data.collators import TokenizedDPOCollator
from data.pipeline import load_and_prepare_datasets
from methods.base import TrainingMethod
from methods.logits_to_keep import (
    align_logits_to_labels,
    compute_logits_to_keep_from_labels,
    model_supports_logits_to_keep,
    slice_labels_for_logits_to_keep,
    use_logits_to_keep_enabled,
)
from methods.sft import run_sft_stage
from methods.trl_vision_safe import prompt_vision_tensors_from_features
from modeling.factory import build_model_and_processor


# Same required columns as VisionSafeCPOTrainer
_REQUIRED_COLS = {
    "prompt_input_ids",
    "prompt_attention_mask",
    "chosen_input_ids",
    "chosen_attention_mask",
    "rejected_input_ids",
    "rejected_attention_mask",
}


class VisionSafeCPOTrainer(CPOTrainer):

    def __init__(self, *args, use_logits_to_keep: bool | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_logits_to_keep = use_logits_to_keep_enabled() if use_logits_to_keep is None else use_logits_to_keep

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

    @staticmethod
    def _vision_forward_kwargs(batch: dict) -> dict:
        kwargs: dict = {}
        pv = batch.get("prompt_pixel_values")
        grid = batch.get("prompt_image_grid_thw")
        if isinstance(pv, torch.Tensor) and pv.numel() > 0:
            n = batch["chosen_input_ids"].shape[0]
            if pv.shape[0] == n:
                kwargs["pixel_values"] = torch.cat([pv, pv], dim=0)
            else:
                kwargs["pixel_values"] = pv
        if isinstance(grid, torch.Tensor) and grid.numel() > 0:
            n = batch["chosen_input_ids"].shape[0]
            if grid.shape[0] == n:
                kwargs["image_grid_thw"] = torch.cat([grid, grid], dim=0)
            else:
                kwargs["image_grid_thw"] = grid
        return kwargs

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, list | torch.LongTensor]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        if not self.use_logits_to_keep or not model_supports_logits_to_keep(model):
            return super().concatenated_forward(model, batch)

        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]
        labels = concatenated_batch["concatenated_labels"].clone()

        logits_to_keep = compute_logits_to_keep_from_labels(labels, ignore_index=self.label_pad_token_id)
        if logits_to_keep is None:
            return super().concatenated_forward(model, batch)

        model_kwargs: dict = {"use_cache": False, "logits_to_keep": logits_to_keep}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True
        if self.is_encoder_decoder:
            model_kwargs["decoder_input_ids"] = self._shift_right(labels)
        if "concatenated_mm_token_type_ids" in concatenated_batch:
            model_kwargs["mm_token_type_ids"] = concatenated_batch["concatenated_mm_token_type_ids"]
        model_kwargs.update(self._vision_forward_kwargs(batch))

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        )
        all_logits = align_logits_to_labels(
            outputs.logits, slice_labels_for_logits_to_keep(labels, logits_to_keep)
        )
        labels = slice_labels_for_logits_to_keep(labels, logits_to_keep)

        def cross_entropy_loss(logits, ce_labels):
            if not self.is_encoder_decoder:
                logits = logits[..., :-1, :].contiguous()
                ce_labels = ce_labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_pad_token_id)
            flat_logits = logits.reshape(-1, logits.shape[-1])
            flat_labels = ce_labels.reshape(-1).to(flat_logits.device)
            return loss_fct(flat_logits, flat_labels)

        if self.cpo_alpha == 0:
            nll_loss = torch.tensor(0.0, device=self.accelerator.device)
        else:
            nll_loss = cross_entropy_loss(all_logits[:len_chosen], labels[:len_chosen])

        all_logps = self.get_batch_logps(
            all_logits,
            labels,
            average_log_prob=self.loss_type in ["ipo", "simpo"],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        if self.aux_loss_enabled:
            return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss, outputs.aux_loss)
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)


def _build_cpo_config(cfg: RunConfig, args, processor) -> CPOConfig:
    """Build CPOConfig: base args + IPO params, with fallbacks for TRL API changes."""
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
