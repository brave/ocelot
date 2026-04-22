from __future__ import annotations

import os
import torch


def patch_torch_linspace_steps() -> None:
    """
    Workaround for torch>=2.9: some Qwen3-VL code paths pass a 0-dim Tensor as `steps` to
    torch.linspace(..., steps=...). torch 2.9+ requires `steps` to be an int.
    """
    if os.environ.get("PATCH_TORCH_LINSPACE_STEPS", "1").strip().lower() not in {"1", "true", "yes"}:
        return

    orig_linspace = torch.linspace

    def _linspace_patched(start, end, steps, *args, **kwargs):
        if isinstance(steps, torch.Tensor):
            if steps.numel() != 1:
                steps = steps.reshape(-1)[0]
            steps = int(steps.item())
        return orig_linspace(start, end, steps, *args, **kwargs)

    torch.linspace = _linspace_patched  # type: ignore[assignment]


