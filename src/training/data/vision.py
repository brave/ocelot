from __future__ import annotations

import random
from typing import Any

from PIL import Image


def configure_qwen_processor_image_limits(processor: Any, *, vision_max_pixels: int | None) -> None:
    """
    Mirrors the vision-related processor settings in `train_script.py`.
    Safe no-ops if the processor doesn't expose these attributes.
    """
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return

    # These exist for Qwen processors; keep wrapped in try for safety.
    try:
        ip.max_pixels = 1024 * 32 * 32  # Limit to 512 vision tokens
        ip.min_pixels = 4 * 32 * 32
    except Exception:
        pass

    if vision_max_pixels is None:
        return

    if hasattr(ip, "max_pixels"):
        try:
            ip.max_pixels = int(vision_max_pixels)
        except Exception:
            pass


def downscale_pil_random_cap(img: Image.Image) -> Image.Image:
    """
    Matches `train_script.py`: if downscaling is enabled, pick a random pixel cap from a fixed set.
    Note: the cap is NOT derived from `VISION_MAX_PIXELS` in the original script.
    """
    if not isinstance(img, Image.Image):
        return img

    w, h = img.size
    max_pix = random.choice([262144, 571356, 1048576])
    if w * h <= max_pix:
        return img
    scale = (max_pix / float(w * h)) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC", Image.BICUBIC)
    return img.resize((new_w, new_h), resample=resample)


def maybe_downscale_images(image_inputs, *, vision_max_pixels: int | None) -> Any:
    """
    Mirrors `_maybe_downscale_images` from `train_script.py`.
    """
    if vision_max_pixels is None or image_inputs is None:
        return image_inputs
    if isinstance(image_inputs, list):
        return [downscale_pil_random_cap(x) for x in image_inputs]
    if isinstance(image_inputs, Image.Image):
        return downscale_pil_random_cap(image_inputs)
    return image_inputs


