from __future__ import annotations

import inspect
import os

import torch


def use_logits_to_keep_enabled() -> bool:
    return os.environ.get("USE_LOGITS_TO_KEEP", "1").strip().lower() in {"1", "true", "yes"}


def model_supports_logits_to_keep(model) -> bool:
    """Return True if `model.forward` accepts `logits_to_keep`."""
    for candidate in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        if candidate is None:
            continue
        fn = getattr(candidate, "_supports_logits_to_keep", None)
        if callable(fn) and fn():
            return True
        forward = getattr(candidate, "forward", None)
        if forward is not None and "logits_to_keep" in inspect.signature(forward).parameters:
            return True
    return False


def compute_logits_to_keep_from_labels(labels: torch.Tensor, *, ignore_index: int = -100) -> int | None:
    """
    Compute how many trailing positions need lm_head logits.

    Uses the earliest supervised label in the batch (conservative for mixed-length completions).
    """
    loss_mask = labels != ignore_index
    if not loss_mask.any():
        return None
    first_idx = loss_mask.nonzero(as_tuple=True)[1].min()
    return int((labels.shape[1] - first_idx).item() + 1)


def slice_labels_for_logits_to_keep(labels: torch.Tensor, logits_to_keep: int) -> torch.Tensor:
    return labels[:, -logits_to_keep:].contiguous()


def align_logits_to_labels(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Align logits length to labels (Qwen-VL may prepend image-token logits)."""
    if logits.shape[:2] == labels.shape[:2]:
        return logits
    return logits[:, -labels.shape[1] :]
