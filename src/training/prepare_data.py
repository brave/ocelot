#!/usr/bin/env python3
"""
Stage 1: Read JSON, preprocess + tokenize in chunks, write Parquet (train.parquet, validation.parquet).
Run once; then Stage 2 (`run.py` with --prepared-data-dir or PREPARED_DATA_DIR) loads shards for low-RAM training.

Usage:
  python src/training/prepare_data.py --data-path /path/to/data.json --output-dir /path/to/prepared
  python src/training/run.py --trainer sft --epochs 1 --prepared-data-dir /path/to/prepared ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    import ijson
except ImportError:
    ijson = None
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

import pyarrow as pa
from transformers import AutoProcessor

# Add parent so "data" and "core" resolve when run as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.parquet_rows import chunk_tokenized_rows_to_arrow
from data.pipeline import (
    _preprocess_example,
    _prompt_within_limit,
    tokenize_all_once_batched_allow_oversize_skip,
)


def _stream_json_array(filepath: str, field: str):
    """Yield items from JSON key `field` (must be an array). Uses ijson if available."""
    if ijson is None:
        with open(filepath) as f:
            data = json.load(f)
        for item in (data.get(field) or []):
            yield item
        return
    with open(filepath, "rb") as f:
        for item in ijson.items(f, f"{field}.item"):
            yield item


def _chunked(iterable, size: int):
    chunk = []
    for x in iterable:
        chunk.append(x)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# Arrow/Parquet limit: single array cannot exceed 2^31-1 bytes
_MAX_ARRAY_BYTES = int(1.9e9)


def main():
    p = argparse.ArgumentParser(description="Stage 1: JSON -> preprocess + tokenize -> Parquet")
    p.add_argument("--data-path", required=True, help="Path to JSON with 'train' and 'validation' keys")
    p.add_argument("--output-dir", required=True, help="Write train.parquet, validation.parquet, config.json here")
    p.add_argument("--model-name", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-4B-Instruct"))
    p.add_argument("--max-length", type=int, default=int(os.environ.get("SFT_MAX_LENGTH", "10240")))
    p.add_argument("--max-prompt-length", type=int, default=int(os.environ.get("SFT_MAX_PROMPT_LENGTH", "8192")))
    p.add_argument("--vision-max-pixels", type=int, default=int(os.environ.get("VISION_MAX_PIXELS", "262144")))
    p.add_argument("--tokenize-batch-size", type=int, default=int(os.environ.get("TOKENIZE_BATCH_SIZE", "32")))
    p.add_argument("--chunk-size", type=int, default=500, help="Examples per chunk (lower = less peak RAM)")
    p.add_argument("--streaming", action="store_true", help="Stream JSON with ijson (avoids full load; requires: pip install ijson)")
    args = p.parse_args()
    if args.streaming and ijson is None:
        p.error("--streaming requires ijson. Install with: pip install ijson")

    vision_max_pixels = None if args.vision_max_pixels <= 0 else args.vision_max_pixels
    store_vision_dtype = os.environ.get("STORE_VISION_DTYPE", "float16").strip().lower()

    print(f"[prepare_data] Loading processor {args.model_name}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_name)
    if getattr(processor, "image_processor", None) is not None and hasattr(processor.image_processor, "max_pixels"):
        try:
            processor.image_processor.max_pixels = min(
                getattr(processor.image_processor, "max_pixels", 1 << 30),
                args.vision_max_pixels if args.vision_max_pixels > 0 else (1 << 30),
            )
        except Exception:
            pass

    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "config.json")

    for split, field in [("train", "train"), ("validation", "validation")]:
        out_parquet = os.path.join(args.output_dir, f"{split}.parquet")
        # Clean old shards/single files so reruns don't mix stale and fresh prepared data.
        stale = [
            f
            for f in os.listdir(args.output_dir)
            if f.endswith(".parquet") and (f == f"{split}.parquet" or f.startswith(f"{split}-"))
        ]
        for f in stale:
            try:
                os.remove(os.path.join(args.output_dir, f))
            except FileNotFoundError:
                pass
        if stale:
            print(f"[prepare_data] Removed {len(stale)} stale {split} parquet file(s)", flush=True)
        tables = []
        total_written = 0
        n_skip = 0
        n_long_prompt = 0
        n_long_seq = 0

        print(f"[prepare_data] Processing {field} -> {out_parquet}", flush=True)
        it = _stream_json_array(args.data_path, field)
        chunk_iter = _chunked(it, args.chunk_size)
        pbar = tqdm(chunk_iter, desc=field, unit=" chunk", leave=True)
        for chunk in pbar:
            # Preprocess
            preprocessed = []
            for ex in chunk:
                out = _preprocess_example(ex, vision_max_pixels=vision_max_pixels)
                if out.get("_skip"):
                    n_skip += 1
                    continue
                preprocessed.append(out)

            if not preprocessed:
                pbar.set_postfix(
                    written=total_written, skip=n_skip, long_prompt=n_long_prompt, long_seq=n_long_seq, refresh=True
                )
                continue

            # Filter long prompts (needs processor)
            filtered = []
            for ex in preprocessed:
                if _prompt_within_limit(ex, processor=processor, max_prompt_length=args.max_prompt_length):
                    filtered.append(ex)
                else:
                    n_long_prompt += 1
            if not filtered:
                pbar.set_postfix(
                    written=total_written, skip=n_skip, long_prompt=n_long_prompt, long_seq=n_long_seq, refresh=True
                )
                continue

            # Batch dict; drop rows where full chosen/rejected tokenized length > max_length
            batch = {
                "text": [ex["text"] for ex in filtered],
                "chosen": [ex["chosen"] for ex in filtered],
                "rejected": [ex["rejected"] for ex in filtered],
                "images": [ex.get("images", []) for ex in filtered],
            }
            tokenized, chosen_kept, rejected_kept, n_drop_seq = tokenize_all_once_batched_allow_oversize_skip(
                batch,
                processor=processor,
                max_length=args.max_length,
                max_prompt_length=args.max_prompt_length,
                store_vision_dtype=store_vision_dtype,
                store_vision_as_bytes=True,
            )
            if n_drop_seq:
                n_long_seq += n_drop_seq
                print(
                    f"[prepare_data] {field}: dropped {n_drop_seq} sample(s) with chosen/rejected sequence length "
                    f"> max_length={args.max_length} (raise SFT_MAX_LENGTH / --max-length to keep more)",
                    flush=True,
                )

            # Convert to list of rows (drop pixel_values list column; we have bytes+shape)
            rows = []
            n = len(tokenized["input_ids"])
            for i in range(n):
                row = {k: tokenized[k][i] for k in tokenized if k not in ("pixel_values",)}
                if tokenized.get("pixel_values") and tokenized["pixel_values"][i] is not None:
                    row["pixel_values"] = tokenized["pixel_values"][i]
                row["chosen"] = chosen_kept[i]
                row["rejected"] = rejected_kept[i]
                rows.append(row)
            if rows:
                # Split into sub-batches so no column exceeds Arrow's ~2GB array limit
                acc = 0
                batch_start = 0
                for j, row in enumerate(rows):
                    b = (row.get("pixel_values_bytes") or b"") if isinstance(row.get("pixel_values_bytes"), bytes) else b""
                    acc += len(b)
                    if acc > _MAX_ARRAY_BYTES:
                        tables.append(chunk_tokenized_rows_to_arrow(rows[batch_start:j]))
                        total_written += j - batch_start
                        batch_start = j
                        acc = len(b)
                tables.append(chunk_tokenized_rows_to_arrow(rows[batch_start:]))
                total_written += len(rows) - batch_start
            pbar.set_postfix(
                written=total_written, skip=n_skip, long_prompt=n_long_prompt, long_seq=n_long_seq, refresh=True
            )

        if not tables:
            print(f"[prepare_data] No data for {field}; writing empty table", flush=True)
            empty = pa.table({
                "input_ids": pa.array([], type=pa.list_(pa.int64())),
                "attention_mask": pa.array([], type=pa.list_(pa.int64())),
                "labels": pa.array([], type=pa.list_(pa.int64())),
                "mm_token_type_ids": pa.array([], type=pa.list_(pa.int64())),
                "prompt_input_ids": pa.array([], type=pa.list_(pa.int64())),
                "prompt_attention_mask": pa.array([], type=pa.list_(pa.int64())),
                "prompt_mm_token_type_ids": pa.array([], type=pa.list_(pa.int64())),
                "chosen_input_ids": pa.array([], type=pa.list_(pa.int64())),
                "chosen_attention_mask": pa.array([], type=pa.list_(pa.int64())),
                "chosen_mm_token_type_ids": pa.array([], type=pa.list_(pa.int64())),
                "rejected_input_ids": pa.array([], type=pa.list_(pa.int64())),
                "rejected_attention_mask": pa.array([], type=pa.list_(pa.int64())),
                "rejected_mm_token_type_ids": pa.array([], type=pa.list_(pa.int64())),
                "pixel_values_bytes": pa.array([], type=pa.large_binary()),
                "pixel_values_shape": pa.array([], type=pa.list_(pa.int64())),
                "image_grid_thw": pa.array([], type=pa.list_(pa.list_(pa.int64()))),
                "chosen": pa.array([], type=pa.string()),
                "rejected": pa.array([], type=pa.string()),
            })
            tables = [empty]
        # Write one Parquet file per chunk; use small row_group_size so no single
        # row group exceeds Arrow/Parquet ~2GB per-array limit (pixel_values_bytes).
        out_dir = os.path.dirname(out_parquet)
        base = os.path.basename(out_parquet).replace(".parquet", "")
        row_group_size = 200  # ~5MB/row -> ~1GB per group, under 2^31 limit
        for i, tbl in enumerate(tables):
            path = os.path.join(out_dir, f"{base}-{i:05d}.parquet")
            with pa.OSFile(path, "wb") as f:
                pa.parquet.write_table(tbl, f, row_group_size=row_group_size)
        print(
            f"[prepare_data] {field}: wrote {total_written} rows in {len(tables)} shard(s) "
            f"(skip={n_skip}, long_prompt={n_long_prompt}, long_seq={n_long_seq})",
            flush=True,
        )

    with open(config_path, "w") as f:
        json.dump({
            "max_length": args.max_length,
            "max_prompt_length": args.max_prompt_length,
            "model_name": args.model_name,
        }, f, indent=2)
    print(f"[prepare_data] Wrote {config_path}. Use PREPARED_DATA_DIR={args.output_dir} for Stage 2.", flush=True)


if __name__ == "__main__":
    main()
