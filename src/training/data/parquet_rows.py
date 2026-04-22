from __future__ import annotations

import pyarrow as pa


def chunk_tokenized_rows_to_arrow(chunk: list[dict]) -> pa.Table:
    """Convert a list of tokenized row dicts (with pixel_values_bytes/shape) to a PyArrow Table."""
    if not chunk:
        return pa.table({})
    keys = list(chunk[0].keys())
    columns: dict[str, pa.Array] = {}
    for k in keys:
        vals = [row[k] for row in chunk]
        if k == "pixel_values_bytes":
            columns[k] = pa.array(vals, type=pa.large_binary())
        elif k == "pixel_values_shape":
            columns[k] = pa.array(vals, type=pa.list_(pa.int64()))
        elif k == "pixel_values" and all(v is None for v in vals):
            continue
        elif k == "pixel_values" and vals[0] is not None:
            columns[k] = pa.array(vals, type=pa.list_(pa.float32()))
        elif k in ("chosen", "rejected"):
            columns[k] = pa.array(vals, type=pa.string())
        elif k == "image_grid_thw":
            columns[k] = pa.array(vals, type=pa.list_(pa.list_(pa.int64())))
        elif isinstance(vals[0], list) and vals[0] and isinstance(vals[0][0], (int, float)):
            columns[k] = pa.array(vals, type=pa.list_(pa.int64() if isinstance(vals[0][0], int) else pa.float32()))
        elif isinstance(vals[0], list) and vals[0] and isinstance(vals[0][0], list):
            columns[k] = pa.array(vals, type=pa.list_(pa.list_(pa.int64())))
        else:
            columns[k] = pa.array(vals)
    return pa.table(columns)
