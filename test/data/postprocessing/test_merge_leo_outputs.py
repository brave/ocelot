"""Tests for data/postprocessing/merge_leo_outputs.py helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_merge_module(root: Path):
    path = root / "src/data/postprocessing/merge_leo_outputs.py"
    spec = importlib.util.spec_from_file_location("merge_leo_outputs", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def merge(repo_root: Path):
    return _load_merge_module(repo_root)


def test_split_80_10_10_sizes_and_sum(merge) -> None:
    items = [{"i": i} for i in range(10)]
    train, val, test = merge.split_80_10_10(items, seed=0)
    assert len(train) + len(val) + len(test) == 10
    assert len(train) == 8
    assert len(val) == 1
    assert len(test) == 1


def test_split_80_10_10_deterministic_seed(merge) -> None:
    items = [{"i": i} for i in range(20)]
    a = merge.split_80_10_10(items, seed=123)
    b = merge.split_80_10_10(items, seed=123)
    assert a == b


def test_split_80_10_10_empty(merge) -> None:
    assert merge.split_80_10_10([], seed=42) == ([], [], [])


def test_load_records_skips_split_file_and_bad_json(merge, tmp_path: Path) -> None:
    good = tmp_path / "a.json"
    good.write_text(json.dumps({"prompt": []}), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    split = tmp_path / "dataset_split.json"
    split.write_text("{}", encoding="utf-8")

    records = merge.load_records(tmp_path, exclude_name="dataset_split.json")
    assert len(records) == 1
    assert records[0] == {"prompt": []}
