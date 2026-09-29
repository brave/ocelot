from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset

from data.pipeline import (
    _drop_image_samples,
    _filter_tokenized_valid_batch,
    _filter_valid_tokenized_rows,
    _has_image_flags,
    _tokenized_row_is_valid,
    _validation_input_columns,
)


def _row(*, bad: bool = False) -> dict:
    seq_len = 8
    ids = list(range(seq_len))
    row = {
        "input_ids": ids,
        "attention_mask": [1] * seq_len,
        "labels": [-100] * seq_len,
        "mm_token_type_ids": [0] * seq_len,
        "prompt_input_ids": ids[:4],
        "prompt_attention_mask": [1] * 4,
        "prompt_mm_token_type_ids": [0] * 4,
        "chosen_input_ids": ids,
        "chosen_attention_mask": [1] * seq_len,
        "chosen_mm_token_type_ids": [0] * seq_len,
        "rejected_input_ids": ids,
        "rejected_attention_mask": [1] * seq_len,
        "rejected_mm_token_type_ids": [0] * seq_len,
        "image_grid_thw": [[1, 1, 2]],
        "pixel_values_shape": [2, 4],
        "pixel_values_bytes": np.zeros((2, 4), dtype=np.float16).tobytes(),
        "chosen": "ok",
        "rejected": "no",
    }
    if bad:
        row["attention_mask"] = [1] * (seq_len - 1)
    return row


def test_validation_input_columns_excludes_heavy_fields() -> None:
    ds = Dataset.from_list([_row()])
    cols = _validation_input_columns(ds)
    assert "pixel_values_bytes" not in cols
    assert "chosen" not in cols
    assert "pixel_values_shape" in cols


def test_filter_tokenized_valid_batch() -> None:
    batch = Dataset.from_list([_row(), _row(bad=True)]).to_dict()
    keep = _filter_tokenized_valid_batch(batch)
    assert keep == [True, False]


def test_filter_valid_tokenized_rows_drops_bad_rows() -> None:
    ds = Dataset.from_list([_row(), _row(bad=True), _row()])
    filtered, dropped = _filter_valid_tokenized_rows(ds, desc="test")
    assert dropped == 1
    assert len(filtered) == 2
    assert all(_tokenized_row_is_valid(ex) for ex in filtered)


def test_filter_valid_tokenized_rows_can_skip() -> None:
    ds = Dataset.from_list([_row(), _row(bad=True)])
    filtered, dropped = _filter_valid_tokenized_rows(ds, desc="test", skip=True)
    assert dropped == 0
    assert len(filtered) == 2


def test_drop_image_samples_uses_lightweight_columns_only() -> None:
    heavy = b"x" * (1024 * 1024)
    ds = Dataset.from_list(
        [
            {**_row(), "pixel_values_bytes": heavy, "image_grid_thw": [[1, 1, 2]]},
            {**_row(), "pixel_values_bytes": heavy, "image_grid_thw": []},
        ]
    )
    flags = _has_image_flags(ds)
    assert flags == [True, False]
    filtered, n_img, dropped = _drop_image_samples(ds, drop_ratio=1.0, seed=0)
    assert n_img == 1
    assert dropped == 1
    assert len(filtered) == 1
    assert filtered[0]["image_grid_thw"] == []
