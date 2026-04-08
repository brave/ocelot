from __future__ import annotations

import torch


def pad_vision_from_lists(pv_list, grid_list, *, vision_dtype: str = "float16"):
    """
    Minimal vision padding helper copied from `train_script.py`.

    Accepts per-example `pixel_values` / `image_grid_thw` stored as Python lists (or None).
    Returns (pixel_values_tensor_or_None, image_grid_thw_tensor_or_None).
    """
    non_empty_pv = [pv for pv in pv_list if pv is not None and len(pv) > 0]
    if not non_empty_pv:
        pixel_values = None
    else:
        pv0 = torch.as_tensor(non_empty_pv[0])
        dtype = torch.float16 if vision_dtype in {"fp16", "float16"} else torch.float32
        if pv0.dim() == 2:
            d = int(pv0.shape[1])
            filled = []
            for pv in pv_list:
                if pv is None or len(pv) == 0:
                    filled.append(torch.zeros((0, d), dtype=dtype))
                else:
                    filled.append(torch.as_tensor(pv, dtype=dtype))
            pixel_values = torch.cat(filled, dim=0)
        elif pv0.dim() == 4:
            c, h, w = int(pv0.shape[1]), int(pv0.shape[2]), int(pv0.shape[3])
            filled = []
            for pv in pv_list:
                if pv is None or len(pv) == 0:
                    filled.append(torch.zeros((0, c, h, w), dtype=dtype))
                else:
                    filled.append(torch.as_tensor(pv, dtype=dtype))
            pixel_values = torch.cat(filled, dim=0)
        else:
            filled = [torch.as_tensor(pv, dtype=dtype) for pv in pv_list if pv is not None]
            pixel_values = torch.cat(filled, dim=0) if filled else None

    non_empty_grid = [g for g in grid_list if g is not None and len(g) > 0]
    if not non_empty_grid:
        image_grid_thw = None
    else:
        filled_g = []
        for g in grid_list:
            if g is None or len(g) == 0:
                filled_g.append(torch.zeros((0, 3), dtype=torch.int64))
            else:
                gt = torch.as_tensor(g, dtype=torch.int64)
                if gt.dim() == 1 and gt.numel() == 3:
                    gt = gt.view(1, 3)
                filled_g.append(gt)
        image_grid_thw = torch.cat(filled_g, dim=0)
        if image_grid_thw.numel() == 0:
            image_grid_thw = None

    return pixel_values, image_grid_thw


