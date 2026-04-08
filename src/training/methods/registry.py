from __future__ import annotations

import importlib
from typing import cast

from methods.base import TrainingMethod

# Keep registry imports light-weight: avoid importing torch/transformers at module import time.
_METHOD_SPECS: dict[str, tuple[str, str]] = {
    "sft": ("methods.sft", "SFTMethod"),
    "ipo": ("methods.ipo", "IPOMethod"),
    "dpo": ("methods.dpo", "DPOMethod"),
}
_CACHE: dict[str, TrainingMethod] = {}


def get_method(name: str) -> TrainingMethod:
    key = (name or "").strip().lower()
    if key not in _METHOD_SPECS:
        raise KeyError(f"Unknown trainer '{name}'. Known: {sorted(_METHOD_SPECS)}")
    if key in _CACHE:
        return _CACHE[key]

    mod_name, cls_name = _METHOD_SPECS[key]
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    inst = cast(TrainingMethod, cls())
    _CACHE[key] = inst
    return inst


