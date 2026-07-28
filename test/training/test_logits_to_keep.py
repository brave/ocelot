from __future__ import annotations

from __future__ import annotations

import sys
from pathlib import Path

import torch

_TRAINING = Path(__file__).resolve().parents[2] / "src" / "training"
if str(_TRAINING) not in sys.path:
    sys.path.insert(0, str(_TRAINING))

from methods.logits_to_keep import (
    align_logits_to_labels,
    compute_logits_to_keep_from_labels,
    model_supports_logits_to_keep,
    slice_labels_for_logits_to_keep,
)


def test_compute_logits_to_keep_from_labels_prompt_masked() -> None:
    labels = torch.tensor([[-100, -100, -100, 10, 11, 12]])
    assert compute_logits_to_keep_from_labels(labels) == 4


def test_compute_logits_to_keep_all_ignored() -> None:
    labels = torch.full((2, 8), -100)
    assert compute_logits_to_keep_from_labels(labels) is None


def test_slice_labels_for_logits_to_keep() -> None:
    labels = torch.tensor([[1, 2, 3, 4, 5]])
    assert slice_labels_for_logits_to_keep(labels, 2).tolist() == [[4, 5]]


def test_align_logits_to_labels_trims_prefix() -> None:
    logits = torch.zeros(1, 10, 4)
    labels = torch.zeros(1, 6, dtype=torch.long)
    aligned = align_logits_to_labels(logits, labels)
    assert aligned.shape == (1, 6, 4)


class _InnerWithLogitsToKeep:
    def _supports_logits_to_keep(self) -> bool:
        return True

    def forward(self, input_ids, logits_to_keep=None):
        return input_ids


class _DeepSpeedLikeWrapper:
    def __init__(self, module):
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def test_model_supports_logits_to_keep_unwraps_deepspeed_module() -> None:
    inner = _InnerWithLogitsToKeep()
    assert model_supports_logits_to_keep(_DeepSpeedLikeWrapper(inner))
