from __future__ import annotations

import numpy as np
import torch

from data.vision_padding import pad_vision_from_lists


def _pixel_values_for_ex(ex: dict, *, vision_dtype: str):
    """Return per-example pixel_values as list or tensor for pad_vision_from_lists. Handles bytes+shape from Parquet."""
    raw = ex.get("pixel_values_bytes")
    shape = ex.get("pixel_values_shape")
    if raw is not None and shape is not None and len(shape) > 0:
        dtype = np.float16 if vision_dtype in {"fp16", "float16"} else np.float32
        arr = np.frombuffer(raw, dtype=dtype).reshape(tuple(shape))
        return torch.tensor(arr, dtype=torch.float16 if dtype == np.float16 else torch.float32)
    return ex.get("pixel_values")


class TokenizedSFTCollator:
    """Stack-only SFT collator for pre-tokenized, fixed-length inputs."""

    def __init__(self, *, vision_dtype: str = "float16"):
        self.vision_dtype = vision_dtype

    def __call__(self, features):
        xs0 = features[0]["input_ids"]
        if isinstance(xs0, torch.Tensor):
            batch = {
                "input_ids": torch.stack([ex["input_ids"] for ex in features], dim=0),
                "attention_mask": torch.stack([ex["attention_mask"] for ex in features], dim=0),
                "labels": torch.stack([ex["labels"] for ex in features], dim=0),
                "mm_token_type_ids": torch.stack([ex["mm_token_type_ids"] for ex in features], dim=0),
            }
        else:
            batch = {
                "input_ids": torch.as_tensor([ex["input_ids"] for ex in features], dtype=torch.int64),
                "attention_mask": torch.as_tensor([ex["attention_mask"] for ex in features], dtype=torch.int64),
                "labels": torch.as_tensor([ex["labels"] for ex in features], dtype=torch.int64),
                "mm_token_type_ids": torch.as_tensor([ex["mm_token_type_ids"] for ex in features], dtype=torch.int64),
            }

        pv_list = [_pixel_values_for_ex(ex, vision_dtype=self.vision_dtype) for ex in features]
        pv, grid = pad_vision_from_lists(pv_list, [ex.get("image_grid_thw") for ex in features], vision_dtype=self.vision_dtype)
        if pv is not None:
            batch["pixel_values"] = pv
        if grid is not None:
            batch["image_grid_thw"] = grid
        return batch


class TokenizedDPOCollator:
    """Stack-only DPO/IPO collator for pre-tokenized, fixed-length inputs."""

    def __init__(self, *, vision_dtype: str = "float16"):
        self.vision_dtype = vision_dtype

    @staticmethod
    def _make_labels(
        seq_input_ids: torch.Tensor, seq_attention_mask: torch.Tensor, prompt_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        labels = seq_input_ids.clone()
        labels = labels.masked_fill(seq_attention_mask == 0, -100)
        prompt_len = prompt_attention_mask.sum(dim=1).to(torch.long)  # (B,)
        pos = torch.arange(seq_input_ids.size(1), device=seq_input_ids.device).unsqueeze(0)  # (1, L)
        labels = labels.masked_fill(pos < prompt_len.unsqueeze(1), -100)
        return labels

    @staticmethod
    def _stack_or_default(features, key: str, ref_key: str) -> torch.Tensor:
        vals = [ex.get(key) for ex in features]
        if all(v is not None for v in vals):
            v0 = vals[0]
            if isinstance(v0, torch.Tensor):
                return torch.stack(vals, dim=0)
            return torch.as_tensor(vals, dtype=torch.int64)
        # Backward-compat for old prepared datasets: default to zeros.
        ref = [ex[ref_key] for ex in features]
        if isinstance(ref[0], torch.Tensor):
            return torch.zeros_like(torch.stack(ref, dim=0), dtype=torch.int64)
        return torch.zeros_like(torch.as_tensor(ref, dtype=torch.int64))

    def __call__(self, features):
        batch = {
            "prompt_input_ids": torch.stack([ex["prompt_input_ids"] for ex in features], dim=0),
            "prompt_attention_mask": torch.stack([ex["prompt_attention_mask"] for ex in features], dim=0),
            "prompt_mm_token_type_ids": self._stack_or_default(features, "prompt_mm_token_type_ids", "prompt_input_ids"),
            "chosen_input_ids": torch.stack([ex["chosen_input_ids"] for ex in features], dim=0),
            "chosen_attention_mask": torch.stack([ex["chosen_attention_mask"] for ex in features], dim=0),
            "chosen_mm_token_type_ids": self._stack_or_default(features, "chosen_mm_token_type_ids", "chosen_input_ids"),
            "rejected_input_ids": torch.stack([ex["rejected_input_ids"] for ex in features], dim=0),
            "rejected_attention_mask": torch.stack([ex["rejected_attention_mask"] for ex in features], dim=0),
            "rejected_mm_token_type_ids": self._stack_or_default(features, "rejected_mm_token_type_ids", "rejected_input_ids"),
        }

        batch["chosen_labels"] = self._make_labels(
            batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["prompt_attention_mask"]
        )
        batch["rejected_labels"] = self._make_labels(
            batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["prompt_attention_mask"]
        )

        pv_lists = [_pixel_values_for_ex(ex, vision_dtype=self.vision_dtype) for ex in features]
        grid_lists = [ex.get("image_grid_thw") for ex in features]
        pv, grid = pad_vision_from_lists(pv_lists, grid_lists, vision_dtype=self.vision_dtype)
        batch["prompt_pixel_values"] = pv
        batch["prompt_image_grid_thw"] = grid
        return batch


