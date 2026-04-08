#!/usr/bin/env python3
"""Combine per-URL JSON artifacts from the data API into a single 80/10/10 split file."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_records(input_dir: Path, exclude_name: str) -> list[dict]:
    records: list[dict] = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name == exclude_name:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"skip {path}: {e}")
            continue
        if isinstance(data, dict):
            records.append(data)
        else:
            print(f"skip {path}: expected object at top level")
    return records


def split_80_10_10(items: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = (n * 80) // 100
    n_val = (n * 10) // 100
    n_test = n - n_train - n_val
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    assert len(train) + len(val) + len(test) == n
    return train, val, test


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "api" / "output",
        help="Directory containing one JSON file per example (API output)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: <input-dir>/dataset_split.json)",
    )
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = p.parse_args()

    input_dir = args.input_dir.resolve()
    out_path = (args.out or (input_dir / "dataset_split.json")).resolve()
    exclude_name = out_path.name

    records = load_records(input_dir, exclude_name=exclude_name)
    if not records:
        print(f"No JSON records found under {input_dir}")
        payload = {"train": [], "validation": [], "test": []}
    else:
        train, val, test = split_80_10_10(records, args.seed)
        payload = {"train": train, "validation": val, "test": test}
        print(
            f"Wrote {len(train)} train, {len(val)} validation, {len(test)} test "
            f"from {len(records)} files -> {out_path}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
