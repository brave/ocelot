from __future__ import annotations

import contextlib
import os

import torch
import torch.nn as nn
from trl.experimental.cpo import CPOConfig, CPOTrainer

from callbacks.common import PeriodicEvalCallback, resolve_ipo_eval_steps
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
from methods.trl_vision_safe import (
    concatenated_vision_kwargs,
    dataset_is_pretokenized,
    drop_arrow_unsafe_vision_columns,
    process_row_token_columns,
)
from modeling.factory import build_model_and_processor

_SKIP_CPO_MAP_FNAMES = frozenset({"maybe_extract_prompt", "maybe_apply_chat_template", "tokenize_row"})


def _log_softmax_chunk_size() -> int:
    return max(1, int(os.environ.get("IPO_LOG_SOFTMAX_CHUNK_SIZE", "64")))


def _shift_logits_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_pad_token_id: int,
    is_encoder_decoder: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    gather_labels = labels.masked_fill(~loss_mask, 0)
    return logits, gather_labels, loss_mask


class _ChunkedAverageLogps(torch.autograd.Function):
    """Average selective log-probs; backward runs softmax grad in seq chunks (memory-safe)."""

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        chunk_size: int,
        label_pad_token_id: int,
        is_encoder_decoder: bool,
    ) -> torch.Tensor:
        logits, gather_labels, loss_mask = _shift_logits_labels(
            logits, labels, label_pad_token_id=label_pad_token_id, is_encoder_decoder=is_encoder_decoder
        )
        batch, seq_len, _ = logits.shape
        per_token = logits.new_zeros(batch, seq_len)
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk_logits = logits[:, start:end, :]
            chunk_labels = gather_labels[:, start:end]
            selected = torch.gather(chunk_logits, -1, chunk_labels.unsqueeze(-1)).squeeze(-1)
            per_token[:, start:end] = selected - torch.logsumexp(chunk_logits, dim=-1)
        ctx.save_for_backward(logits, gather_labels, loss_mask)
        ctx.chunk_size = chunk_size
        ctx.is_encoder_decoder = is_encoder_decoder
        denom = loss_mask.sum(-1).clamp(min=1)
        return (per_token * loss_mask).sum(-1) / denom

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits, gather_labels, loss_mask = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        batch, seq_len, vocab = logits.shape
        denom = loss_mask.sum(-1, keepdim=True).clamp(min=1)
        grad_per_token = grad_output.unsqueeze(-1) * loss_mask.float() / denom

        # Single full-size gradient tensor. For decoder models the function input was
        # one position longer (pre-shift), so allocate seq_len+1 and write the shifted
        # region in place — avoids a second (seq x vocab) allocation + copy.
        out_len = seq_len if ctx.is_encoder_decoder else seq_len + 1
        grad_input = logits.new_zeros(batch, out_len, vocab)

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk_grad = grad_per_token[:, start:end]
            if chunk_grad.abs().sum() == 0:
                continue
            chunk_logits = logits[:, start:end]
            chunk_labels = gather_labels[:, start:end]
            probs = torch.softmax(chunk_logits, dim=-1)
            probs.scatter_add_(-1, chunk_labels.unsqueeze(-1), -torch.ones_like(chunk_labels, dtype=probs.dtype).unsqueeze(-1))
            grad_input[:, start:end] = chunk_grad.unsqueeze(-1) * probs

        return grad_input, None, None, None, None


def _fused_linear_enabled() -> bool:
    return os.environ.get("IPO_FUSED_LINEAR", "1").strip().lower() in {"1", "true", "yes"}


def _find_lm_head_weight(model: torch.nn.Module) -> torch.Tensor | None:
    """Locate the lm_head weight through DeepSpeed/PEFT wrappers (None if not found)."""
    for name, mod in model.named_modules():
        if name.endswith("lm_head") and hasattr(mod, "weight") and mod.weight is not None:
            return mod.weight
    get_out = getattr(model, "get_output_embeddings", None)
    if callable(get_out):
        try:
            emb = get_out()
        except Exception:
            emb = None
        if emb is not None and getattr(emb, "weight", None) is not None:
            return emb.weight
    return None


class _FusedLinearPerTokenLogp(torch.autograd.Function):
    """
    Per-token log p(label) = (hidden @ Wᵀ)[label] - logsumexp(hidden @ Wᵀ), computed and
    differentiated one sequence-chunk at a time so the full (seq × vocab) logits tensor is
    never materialized in forward or backward.

    Inputs are already shifted/aligned: hidden[:, t] predicts gather_labels[:, t].
    Returns per-token logp of shape (rows, seq); only ``hidden`` receives gradient.
    """

    @staticmethod
    def forward(ctx, hidden: torch.Tensor, weight: torch.Tensor, gather_labels: torch.Tensor, chunk_size: int):
        rows, seq, _ = hidden.shape
        out = hidden.new_zeros(rows, seq, dtype=torch.float32)
        for start in range(0, seq, chunk_size):
            end = min(start + chunk_size, seq)
            chunk_logits = torch.matmul(hidden[:, start:end], weight.t()).float()
            lse = torch.logsumexp(chunk_logits, dim=-1)
            sel = torch.gather(chunk_logits, -1, gather_labels[:, start:end].unsqueeze(-1)).squeeze(-1)
            out[:, start:end] = sel - lse
        ctx.save_for_backward(hidden, weight, gather_labels)
        ctx.chunk_size = chunk_size
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        hidden, weight, gather_labels = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        seq = hidden.shape[1]
        grad_hidden = torch.zeros_like(hidden)
        wdtype = weight.dtype
        for start in range(0, seq, chunk_size):
            end = min(start + chunk_size, seq)
            g = grad_out[:, start:end]
            if g.abs().sum() == 0:
                continue
            chunk_logits = torch.matmul(hidden[:, start:end], weight.t()).float()
            probs = torch.softmax(chunk_logits, dim=-1)
            # d logp / d logits = onehot - softmax
            probs.scatter_add_(
                -1,
                gather_labels[:, start:end].unsqueeze(-1),
                -torch.ones_like(gather_labels[:, start:end], dtype=probs.dtype).unsqueeze(-1),
            )
            grad_logits = (-g.unsqueeze(-1)) * probs  # -(softmax - onehot) * upstream
            grad_hidden[:, start:end] = torch.matmul(grad_logits.to(wdtype), weight)
        return grad_hidden, None, None, None


@contextlib.contextmanager
def _skip_cpo_dataset_maps():
    import datasets

    orig_map = datasets.Dataset.map

    def _map(self, function=None, *args, **kwargs):
        fn = kwargs.get("function", function)
        if getattr(fn, "__name__", None) in _SKIP_CPO_MAP_FNAMES:
            return self
        return orig_map(self, function, *args, **kwargs)

    datasets.Dataset.map = _map
    try:
        yield
    finally:
        datasets.Dataset.map = orig_map


class VisionSafeCPOTrainer(CPOTrainer):

    def __init__(self, *args, use_logits_to_keep: bool | None = None, **kwargs):
        train_dataset = kwargs.get("train_dataset")
        pretokenized = train_dataset is not None and dataset_is_pretokenized(train_dataset)
        if pretokenized and (os.environ.get("LOCAL_RANK", "0")) == "0":
            print(
                "[ipo] pre-tokenized dataset: skipping CPOTrainer dataset.map "
                "(vision loaded in collator from pixel_values_bytes)",
                flush=True,
            )
        ctx = _skip_cpo_dataset_maps() if pretokenized else contextlib.nullcontext()
        with ctx:
            super().__init__(*args, **kwargs)
        self.use_logits_to_keep = use_logits_to_keep_enabled() if use_logits_to_keep is None else use_logits_to_keep

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        if dataset_is_pretokenized(dataset):
            return dataset
        return super()._prepare_dataset(dataset, processing_class, args, dataset_name)

    def tokenize_row(self, features, **kwargs):
        if "prompt_input_ids" in features:
            return process_row_token_columns(features)
        return super().tokenize_row(features, **kwargs)

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if average_log_prob:
            return _ChunkedAverageLogps.apply(
                logits,
                labels,
                _log_softmax_chunk_size(),
                label_pad_token_id,
                is_encoder_decoder,
            )

        avg = _ChunkedAverageLogps.apply(
            logits, labels, _log_softmax_chunk_size(), label_pad_token_id, is_encoder_decoder
        )
        _, _, loss_mask = _shift_logits_labels(
            logits, labels, label_pad_token_id=label_pad_token_id, is_encoder_decoder=is_encoder_decoder
        )
        return avg * loss_mask.sum(-1)

    def _chosen_nll_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """BC regularizer on chosen (TRL-compatible); chunked CE forward, stays in autograd graph."""
        chunk_size = _log_softmax_chunk_size()
        if not self.is_encoder_decoder:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_pad_token_id, reduction="sum")
        total_loss = logits.new_zeros(())
        total_valid = logits.new_zeros(())
        for start in range(0, logits.shape[1], chunk_size):
            end = min(start + chunk_size, logits.shape[1])
            chunk_logits = logits[:, start:end, :]
            chunk_labels = labels[:, start:end]
            valid = (chunk_labels != self.label_pad_token_id).sum()
            if int(valid.item()) == 0:
                continue
            chunk_ce = loss_fct(
                chunk_logits.reshape(-1, chunk_logits.shape[-1]),
                chunk_labels.reshape(-1),
            )
            total_loss = total_loss + chunk_ce
            total_valid = total_valid + valid.float()
        return total_loss / total_valid.clamp(min=1)

    def _fused_concatenated_forward(
        self, model: torch.nn.Module, batch: dict[str, list | torch.LongTensor]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.Tensor]:
        """
        One model forward for chosen+rejected; log-probs and NLL from a fused linear op so
        the full (seq × vocab) logits tensor is never materialized (matches SFT-style memory).
        """
        weight = _find_lm_head_weight(model)
        if weight is None:
            raise RuntimeError("lm_head weight not found; cannot use fused-linear path")

        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs: dict = {
            "use_cache": False,
            "output_hidden_states": True,
            **concatenated_vision_kwargs(batch),
        }
        if "concatenated_mm_token_type_ids" in concatenated_batch:
            model_kwargs["mm_token_type_ids"] = concatenated_batch["concatenated_mm_token_type_ids"]
        # logits_to_keep=1 makes the model's own lm_head pass trivial; we do our own fused projection.
        if model_supports_logits_to_keep(model):
            model_kwargs["logits_to_keep"] = 1
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("model did not return hidden_states; cannot use fused-linear path")
        hidden = hidden_states[-1]  # (rows, seq, H), grad flows back through transformer

        labels = concatenated_batch["concatenated_labels"]
        if hidden.shape[1] != labels.shape[1]:
            raise RuntimeError(
                f"hidden seq {hidden.shape[1]} != labels seq {labels.shape[1]}; cannot align fused-linear path"
            )

        # Shift: hidden[:, t] predicts labels[:, t+1].
        shift_hidden = hidden[:, :-1, :]
        shift_labels = labels[:, 1:]
        loss_mask = shift_labels != self.label_pad_token_id
        gather_labels = shift_labels.masked_fill(~loss_mask, 0)

        per_token_logp = _FusedLinearPerTokenLogp.apply(
            shift_hidden, weight, gather_labels, _log_softmax_chunk_size()
        )  # (rows, seq-1) fp32

        token_counts = loss_mask.sum(-1).clamp(min=1)
        seq_logp_sum = (per_token_logp * loss_mask).sum(-1)
        if self.loss_type in ["ipo", "simpo"]:
            all_logps = seq_logp_sum / token_counts
        else:
            all_logps = seq_logp_sum

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        if self.cpo_alpha == 0:
            nll_loss = torch.tensor(0.0, device=self.accelerator.device)
        else:
            # NLL = mean negative log-likelihood over chosen answer tokens (TRL BC regularizer).
            chosen_token_logp = per_token_logp[:len_chosen]
            chosen_mask = loss_mask[:len_chosen]
            nll_loss = -(chosen_token_logp * chosen_mask).sum() / chosen_mask.sum().clamp(min=1)

        stub = chosen_logps.new_zeros(())
        if self.aux_loss_enabled:
            return chosen_logps, rejected_logps, stub, stub, nll_loss, outputs.aux_loss
        return chosen_logps, rejected_logps, stub, stub, nll_loss

    def concatenated_forward(
        self, model: torch.nn.Module, batch: dict[str, list | torch.LongTensor]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.Tensor]:
        if _fused_linear_enabled():
            try:
                result = self._fused_concatenated_forward(model, batch)
                if (os.environ.get("LOCAL_RANK", "0")) == "0" and not getattr(self, "_concat_forward_logged", False):
                    print(
                        "[ipo] fused-linear concatenated forward (1× pass, no full logits, "
                        f"chunk_size={_log_softmax_chunk_size()}, cpo_alpha={self.cpo_alpha})",
                        flush=True,
                    )
                    self._concat_forward_logged = True
                return result
            except Exception as exc:
                if (os.environ.get("LOCAL_RANK", "0")) == "0" and not getattr(self, "_fused_fallback_logged", False):
                    print(
                        f"[ipo] fused-linear path unavailable ({exc}); falling back to materialized logits",
                        flush=True,
                    )
                    self._fused_fallback_logged = True
        return self._materialized_concatenated_forward(model, batch)

    def _materialized_concatenated_forward(
        self, model: torch.nn.Module, batch: dict[str, list | torch.LongTensor]
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.Tensor]:
        """
        Fallback: one model forward producing full logits; chunked logps/NLL keep backward
        memory-safe but the (seq × vocab) logits tensor is still materialized in the forward.
        """
        if (os.environ.get("LOCAL_RANK", "0")) == "0" and not getattr(self, "_mat_forward_logged", False):
            print(
                "[ipo] materialized concatenated forward (1× pass, chunked logps/NLL, "
                f"chunk_size={_log_softmax_chunk_size()}, cpo_alpha={self.cpo_alpha})",
                flush=True,
            )
            self._mat_forward_logged = True

        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs: dict = {
            "use_cache": False,
            **concatenated_vision_kwargs(batch),
        }
        if "concatenated_mm_token_type_ids" in concatenated_batch:
            model_kwargs["mm_token_type_ids"] = concatenated_batch["concatenated_mm_token_type_ids"]

        labels = concatenated_batch["concatenated_labels"]
        logits_to_keep = None
        if self.use_logits_to_keep and model_supports_logits_to_keep(model):
            logits_to_keep = compute_logits_to_keep_from_labels(
                labels, ignore_index=self.label_pad_token_id
            )
            if logits_to_keep is not None:
                model_kwargs["logits_to_keep"] = logits_to_keep

        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        outputs = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        )
        all_logits = outputs.logits
        if self.aux_loss_enabled:
            aux_loss = outputs.aux_loss

        if logits_to_keep is not None:
            labels_kept = slice_labels_for_logits_to_keep(labels, logits_to_keep)
            all_logits = align_logits_to_labels(all_logits, labels_kept)
            labels = labels_kept

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]
        chosen_labels = labels[:len_chosen]
        rejected_labels = labels[len_chosen:]

        if self.cpo_alpha == 0:
            nll_loss = torch.tensor(0.0, device=self.accelerator.device)
        else:
            nll_loss = self._chosen_nll_loss(chosen_logits, chosen_labels)

        chosen_logps = self.get_batch_logps(
            chosen_logits,
            chosen_labels,
            average_log_prob=self.loss_type in ["ipo", "simpo"],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        rejected_logps = self.get_batch_logps(
            rejected_logits,
            rejected_labels,
            average_log_prob=self.loss_type in ["ipo", "simpo"],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        stub = chosen_logps.new_zeros(())
        if self.aux_loss_enabled:
            return chosen_logps, rejected_logps, stub, stub, nll_loss, aux_loss
        return chosen_logps, rejected_logps, stub, stub, nll_loss


def _build_cpo_config(cfg: RunConfig, args, processor) -> CPOConfig:
    """Build CPOConfig: base args + IPO params, with fallbacks for TRL API changes."""
    save_steps = _save_steps()
    cpo_alpha = float(os.environ.get("IPO_CPO_ALPHA", "5"))
    try:
        cpo_config = CPOConfig(
            **args.to_dict(),
            loss_type="ipo",
            padding_value=processor.tokenizer.pad_token_id,
            cpo_alpha=cpo_alpha,
        )
    except TypeError:
        cpo_config = CPOConfig(**args.to_dict(), loss_type="ipo")
    if hasattr(cpo_config, "cpo_alpha"):
        setattr(cpo_config, "cpo_alpha", cpo_alpha)
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
        train_ds = drop_arrow_unsafe_vision_columns(train_ds)
        val_ds = drop_arrow_unsafe_vision_columns(val_ds)

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
        # Preference training is slow; avoid HF's built-in step eval (PeriodicEvalCallback is opt-in).
        args.eval_strategy = "no"

        cpo_config = _build_cpo_config(cfg, args, bundle.processor)
        if (os.environ.get("LOCAL_RANK", "0")) == "0":
            print(
                f"[ipo] loss_type=ipo beta={getattr(cpo_config, 'beta', None)} "
                f"(IPO_BETA / --ipo-beta) cpo_alpha={getattr(cpo_config, 'cpo_alpha', None)} "
                f"(IPO_CPO_ALPHA env, BC regularizer weight)",
                flush=True,
            )

        trainer = VisionSafeCPOTrainer(
            model=bundle.model,
            args=cpo_config,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=bundle.processor.tokenizer,
            data_collator=collator,
            callbacks=list(bundle.callbacks),
        )
        ipo_eval_steps = resolve_ipo_eval_steps()
        if ipo_eval_steps > 0:
            if (os.environ.get("LOCAL_RANK", "0")) == "0":
                print(f"[ipo] periodic validation enabled (EVAL_STEPS={ipo_eval_steps})", flush=True)
            trainer.add_callback(PeriodicEvalCallback(trainer, eval_steps=ipo_eval_steps, label="ipo"))
        elif (os.environ.get("LOCAL_RANK", "0")) == "0":
            print("[ipo] validation disabled during training (set EVAL_STEPS>0 to enable)", flush=True)
        trainer.train()
        bundle.model.save_pretrained(os.path.join(cfg.output_dir, "adapter-ocelot"))
