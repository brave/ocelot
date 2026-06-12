from __future__ import annotations

import os

import torch

from core.config import RunConfig


def _truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes"}


def use_liger_active(cfg: RunConfig | None = None) -> bool:
    """True when Liger kernel patching is enabled (``use_liger`` or ``OCELOT_USE_LIGER``)."""
    if cfg is not None:
        return cfg.use_liger
    return _truthy_env("OCELOT_USE_LIGER", "0")


def liger_fused_ce_active(cfg: RunConfig | None = None) -> bool:
    """
    True when Liger fused linear cross-entropy is patched in (``use_liger`` without ``liger_cross_entropy``).

    In this mode SFT must not pass ``logits_to_keep``: Liger slices hidden states internally and its
    fused loss calls ``.view()`` on a non-contiguous slice, which raises at runtime. Fused LCE already
    avoids materializing full-vocab logits, so ``logits_to_keep`` is redundant for SFT anyway.

    When ``cfg`` is omitted, reads ``OCELOT_USE_LIGER`` / ``OCELOT_LIGER_CROSS_ENTROPY``.
    """
    if cfg is not None:
        return cfg.use_liger and not cfg.liger_cross_entropy
    return use_liger_active() and not _truthy_env("OCELOT_LIGER_CROSS_ENTROPY", "0")


def liger_model_kind(model_name: str) -> str | None:
    """Return 'qwen3_vl_moe', 'qwen3_vl', or None if Liger has no patch for this checkpoint."""
    name = model_name.lower()
    if "qwen3-vl-moe" in name or "qwen3_vl_moe" in name:
        return "qwen3_vl_moe"
    if "qwen3-vl" in name or "qwen3_vl" in name:
        return "qwen3_vl"
    return None


def maybe_apply_liger_kernel(cfg: RunConfig) -> None:
    """
    Patch Qwen3-VL modeling with Liger fused kernels before ``from_pretrained``.

    Controlled by ``RunConfig.use_liger`` (CLI ``--use-liger``, env ``OCELOT_USE_LIGER``).
    Defaults to fused linear CE; set ``liger_cross_entropy`` / ``OCELOT_LIGER_CROSS_ENTROPY=1`` for
    ``cross_entropy=True`` instead (mutually exclusive with fused linear CE).
    """
    if not cfg.use_liger:
        return
    if not torch.cuda.is_available():
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print("[model] use_liger=True but CUDA unavailable; skipping Liger kernels", flush=True)
        return

    kind = liger_model_kind(cfg.model_name)
    if kind is None:
        if os.environ.get("LOCAL_RANK", "0") == "0":
            print(f"[model] use_liger=True but no Liger patch for {cfg.model_name!r}; skipping", flush=True)
        return

    use_cross_entropy = cfg.liger_cross_entropy
    fused_linear_cross_entropy = not use_cross_entropy

    if kind == "qwen3_vl_moe":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl_moe

        apply_liger_kernel_to_qwen3_vl_moe(
            cross_entropy=use_cross_entropy,
            fused_linear_cross_entropy=fused_linear_cross_entropy,
        )
    else:
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl

        apply_liger_kernel_to_qwen3_vl(
            cross_entropy=use_cross_entropy,
            fused_linear_cross_entropy=fused_linear_cross_entropy,
        )

    if os.environ.get("LOCAL_RANK", "0") == "0":
        mode = "cross_entropy" if use_cross_entropy else "fused_linear_cross_entropy"
        print(f"[model] applied Liger kernels ({kind}, {mode})", flush=True)
