from __future__ import annotations

import os

import numpy as np
import torch


def _vision_absent(x) -> bool:
    if x is None:
        return True
    if isinstance(x, list) and len(x) == 0:
        return True
    if isinstance(x, torch.Tensor) and x.numel() == 0:
        return True
    return False


def prompt_vision_tensors_from_features(features: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build `prompt_pixel_values` / `prompt_image_grid_thw` for TRL preference trainers.

    **Parquet (prepare_data.py):** text-only rows use **null** for `pixel_values`; vision rows
    often have `pixel_values_bytes` + `pixel_values_shape` (and may also have float lists).

    **JSON → dataset.map(tokenize):** text-only rows use **empty lists** `[]` for `pixel_values`
    / `image_grid_thw` (see `tokenize_all_once_batched` in `data.pipeline`). `[]` is not
    `None`, so `torch.as_tensor([])` runs and HuggingFace `datasets` then fails with
    ArrowInvalid when the same column also holds real nested-list vision rows.

    This helper treats null, `[]`, and zero-element tensors as “no vision”, and decodes
    `pixel_values_bytes` when lists are absent (prepared data).
    """
    empty_pv = torch.empty((0, 1536))
    empty_grid = torch.empty((0, 3), dtype=torch.int64)

    vdt = os.environ.get("STORE_VISION_DTYPE", "float16").strip().lower()
    np_dt = np.float16 if vdt in {"fp16", "float16"} else np.float32
    torch_dt = torch.float16 if np_dt == np.float16 else torch.float32

    raw = features.get("pixel_values_bytes")
    shape = features.get("pixel_values_shape")
    if raw is not None and shape is not None and len(shape) > 0:
        arr = np.frombuffer(memoryview(raw), dtype=np_dt).reshape(tuple(int(x) for x in shape))
        pv_t = torch.tensor(arr, dtype=torch_dt)
        grid = features.get("image_grid_thw")
        if _vision_absent(grid):
            return empty_pv, empty_grid
        return pv_t, torch.as_tensor(grid, dtype=torch.int64)

    pv = features.get("pixel_values")
    grid = features.get("image_grid_thw")
    if _vision_absent(pv):
        return empty_pv, empty_grid
    pv_t = torch.as_tensor(pv)
    if _vision_absent(grid):
        return empty_pv, empty_grid
    return pv_t, torch.as_tensor(grid, dtype=torch.int64)


REQUIRED_TOKEN_COLS = {
    "prompt_input_ids",
    "prompt_attention_mask",
    "chosen_input_ids",
    "chosen_attention_mask",
    "rejected_input_ids",
    "rejected_attention_mask",
}


def process_row_token_columns(features: dict) -> dict:
    """
    Token columns only for TRL map() — vision is attached in the collator (avoids Arrow 2GB overflow).
    """
    return {
        "prompt_input_ids": torch.as_tensor(features["prompt_input_ids"], dtype=torch.int64),
        "prompt_attention_mask": torch.as_tensor(features["prompt_attention_mask"], dtype=torch.int64),
        "prompt_mm_token_type_ids": torch.as_tensor(
            features.get("prompt_mm_token_type_ids", [0] * len(features["prompt_input_ids"])), dtype=torch.int64
        ),
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


def process_row_with_vision(features: dict) -> dict:
    """
    Convert pre-tokenized dataset rows into the structure TRL expects, including optional vision columns.
    """
    out = process_row_token_columns(features)
    pv_t, grid_t = prompt_vision_tensors_from_features(features)
    out["prompt_pixel_values"] = pv_t
    out["prompt_image_grid_thw"] = grid_t
    return out


def dataset_is_pretokenized(dataset) -> bool:
    cols = set(getattr(dataset, "column_names", []) or [])
    return REQUIRED_TOKEN_COLS.issubset(cols)


def dup_vision_for_concatenated(t: torch.Tensor) -> torch.Tensor:
    """
    Duplicate vision tensors for IPO/CPO concatenated forward ([chosen…, rejected…]).

    The collator stacks patches as flat (total_patches, D) and image_grid_thw as (num_images, 3).
    Doubling the full block yields [ex0 imgs, ex1 imgs, …, ex0 imgs, ex1 imgs, …], which matches
    how Qwen-VL maps images onto concatenated text rows.

    Never skip duplication when image_grid_thw.shape[0] == 2 * batch_size — that also occurs when
    one example has two images (batch_size=1), which still must be doubled for chosen+rejected.
    """
    return torch.cat([t, t], dim=0)


def concatenated_vision_kwargs(batch: dict) -> dict:
    """Build model kwargs with pixel_values / image_grid_thw duplicated for concatenated forward."""
    kwargs: dict = {}
    pv = batch.get("prompt_pixel_values")
    grid = batch.get("prompt_image_grid_thw")
    if isinstance(pv, torch.Tensor) and pv.numel() > 0:
        kwargs["pixel_values"] = dup_vision_for_concatenated(pv)
    if isinstance(grid, torch.Tensor) and grid.numel() > 0:
        kwargs["image_grid_thw"] = dup_vision_for_concatenated(grid)
    return kwargs


def drop_arrow_unsafe_vision_columns(dataset):
    """Remove nested-list vision columns; collator uses pixel_values_bytes from Parquet."""
    drop = [c for c in ("pixel_values", "prompt_pixel_values") if c in getattr(dataset, "column_names", [])]
    return dataset.remove_columns(drop) if drop else dataset

